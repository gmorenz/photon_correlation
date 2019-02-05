
#[macro_use] extern crate arrayref;
#[macro_use] extern crate nom;

mod header;

use structopt::StructOpt;
use nom::{IResult, Err};
use std::path::PathBuf;
use std::io::{stdout, Write};
use std::fs::File;
use std::thread;
use std::collections::VecDeque;

#[derive(Clone)]
struct PhotonIter<'a> {
    data: &'a [u8],
    basetime: u64
}

impl <'a> Iterator for PhotonIter<'a> {
    type Item = (u64, bool);
    fn next(&mut self) -> Option<(u64, bool)> {
        const T2WRAPAROUND_V2: u64 = 33554432;

        if self.data.len() == 0 { return None; }

        let (new_data, val) = nom::le_u32(self.data).unwrap();
        self.data = new_data;

        let special: u32 = val >> 31;
        let channel: u32 = (val >> 25) & 0b_11_1111;
        let timetag: u32 = val & 0x01ff_ffff;

        if special == 1 {
            if channel == 0x3F {
                if timetag == 0 {
                    panic!("A comment says this shouldn't happen, even though it's implemented as the following (see code):");
                }
                else {
                    self.basetime += T2WRAPAROUND_V2 * timetag as u64;
                    self.next()
                }
            }
            else if channel >= 1 && channel <= 15 {
                panic!("Unexpected marker");
            }
            else if channel == 0 {
                panic!("Unexpected sync");
            }
            else {
                panic!("Unexpected channel");
            }
        }
        else {
            assert!(channel < 2);
            Some((self.basetime + timetag as u64, channel == 0))
        }
    }
}


impl <'a> PhotonIter<'a> {
    pub fn new(data: &'a[u8]) -> PhotonIter {
        PhotonIter {
            data: data,
            basetime: 0
        }
    }
}


#[derive(Clone)]
struct FastPhotonIter<'a> {
    data: &'a [u32],
    basetime: u64
}

impl <'a> FastPhotonIter<'a> {
    pub fn new(data: &'a[u8]) -> FastPhotonIter {
        assert_eq!(data.len() % 4, 0, "Bad data length in FastPhotonIter");
        assert_eq!(data.as_ptr() as usize % 4, 0, "Bad data allignment in FastPhotonIter");
        // TODO: Assert we are on a little endian machine.

        let data = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u32, data.len() / 4)
        };

        FastPhotonIter {
            data: data,
            basetime: 0
        }
    }
}

impl <'a> Iterator for FastPhotonIter<'a> {
    type Item = (u64, bool);
    fn next(&mut self) -> Option<(u64, bool)> {
        const T2WRAPAROUND_V2: u64 = 33554432;

        if self.data.len() == 0 { return None; }

        // TODO: Check these both get optimized properly (they aren't).
        loop {
            let val = self.data[0];
            self.data = &self.data[1..];

            let special = val >> 31;
            let channel = (val >> 25) & 0b_11_1111;
            let timetag = val & 0x01ff_ffff;

            if special == 1 {
                if channel == 0x3F && timetag != 0 {
                    self.basetime += T2WRAPAROUND_V2 * timetag as u64;
                    continue;
                }
                else {
                    panic!("Something went wrong");
                }
            }
            else {
                return Some((self.basetime + timetag as u64, channel == 0))
            }
        }
    }
}


fn strip_remaining_from_err<T>(val: IResult<&[u8], T>) -> IResult<&[u8], T> {
    fn strip_remaining_from_context(context: nom::Context<&[u8]>) -> nom::Context<&[u8]> {
        fn truncate(i: &[u8], l: usize) -> &[u8] {
            if i.len() < l { i } else { &i[0.. l] }
        }

        match context {
            nom::Context::Code(i, e) => nom::Context::Code(truncate(i, 64), e),
            nom::Context::List(es) => nom::Context::List(es.into_iter().map(|(i, e)| (truncate(i, 64), e)).collect())
        }
    }

    val.map_err(|err| match err {
        Err::Incomplete(x) => Err::Incomplete(x),
        Err::Error(x) => Err::Error(strip_remaining_from_context(x)),
        Err::Failure(x) => Err::Failure(strip_remaining_from_context(x)),
    })
}

#[cfg(target_feature="avx2")]
fn parallel_asymmetric_cross_correlation(bins_forward: &mut [u64], bins_backwards: &mut [u64], bin_step: u64, photons: impl Iterator<Item=(u64, bool)> + Clone + Send, first_photon: u64) {
    fn thread(bins: &mut [u64], bin_step: u64, us: bool, photons: impl Iterator<Item=(u64, bool)>, first_photon: u64) {
        use std::collections::VecDeque;

        let max_time: u64 = bins.len() as u64 * bin_step;
        let mut stored_photons = VecDeque::new();
        // let mut i = 0;

        let mut stored_len: u64 = 0;

        for (photon, channel) in photons {
            if channel == us {
                // if photon >  i {
                //     println!("Passed time {} ({})", i, us);
                //     i += 12_000_000_000;
                // }
                stored_photons.push_back(photon);
                continue;
            }

            while !stored_photons.is_empty() && photon - stored_photons[0] >= max_time {
                stored_photons.pop_front();
            }

            if photon > first_photon + max_time {
                for &old_photon in &stored_photons {
                    stored_len += stored_photons.len() as u64;

                    let delta: u64 = photon - old_photon;
                    let bin: u64 = delta / bin_step;
                    bins[bin as usize] += 1u64;
                }
            }
        }

        println!("stored len: {}", stored_len);
    }

    // Note (Bug): Simultaneous photons counted twice.

    crossbeam_utils::thread::scope(|s| {
        let photons1 = photons.clone();
        s.spawn(|_| {
            // Thread 1:
            thread(bins_forward, bin_step, true, photons1, first_photon);
        });

        // Thread 2:
        thread(bins_backwards, bin_step, false, photons, first_photon);
    }).unwrap();
}

#[inline(always)]
fn asymmetric_cross_correlation(bins_forward: &mut [u64], bins_backwards: &mut [u64], bin_step: u64, photons: impl Iterator<Item=(u64, bool)>, first_photon: u64) {
    use std::collections::VecDeque;
    assert_eq!(bins_forward.len(), bins_backwards.len());
    let max_time = bins_forward.len() as u64 * bin_step;

    let mut stored_photons_1 = VecDeque::new();
    let mut stored_photons_2 = VecDeque::new();

    for (photon, channel) in photons {
        let stored_photons_this;
        let stored_photons_them;
        if channel {
            stored_photons_this = &mut stored_photons_1;
            stored_photons_them = &mut stored_photons_2;
        }
        else {
            stored_photons_this = &mut stored_photons_2;
            stored_photons_them = &mut stored_photons_1;
        }

        stored_photons_this.push_back(photon);
        while !stored_photons_them.is_empty() && photon - stored_photons_them[0] >= max_time {
            stored_photons_them.pop_front();
        }

        if photon > first_photon + max_time {
            let bins = if channel { &mut *bins_forward } else { &mut *bins_backwards };

            for &mut old_photon in stored_photons_them {
                let delta: u64 = photon - old_photon;
                let bin: u64 = delta / bin_step;
                bins[bin as usize] += 1u64;
            }
        }
    }
}

#[inline(always)]
fn symmetric_cross_correlation(bins: &mut [u64], bin_step: u64, photons: impl Iterator<Item=(u64, bool)>, first_photon: u64) {
    use std::collections::VecDeque;
    let max_time = bins.len() as u64 * bin_step;

    let mut stored_photons_1 = VecDeque::new();
    let mut stored_photons_2 = VecDeque::new();

    for (photon, channel) in photons {
        let stored_photons_this;
        let stored_photons_them;
        if channel {
            stored_photons_this = &mut stored_photons_1;
            stored_photons_them = &mut stored_photons_2;
        }
        else {
            stored_photons_this = &mut stored_photons_2;
            stored_photons_them = &mut stored_photons_1;
        }

        stored_photons_this.push_back(photon);
        while !stored_photons_them.is_empty() && photon - stored_photons_them[0] >= max_time {
            stored_photons_them.pop_front();
        }

        if photon > first_photon + max_time {
            for &mut old_photon in stored_photons_them {
                let delta: u64 = photon - old_photon;
                let bin: u64 = delta / bin_step;
                bins[bin as usize] += 1u64;
            }
        }
    }
}

#[inline(always)]
fn approximate_symmetric_cross_correlation(bin_count: u64, bin_step: u64, photons: impl Iterator<Item=(u64, bool)>, first_photon: u64) -> Vec<u64> {
    let mut chan_1_bins = VecDeque::with_capacity(bin_count as usize);
    let mut chan_2_bins = VecDeque::with_capacity(bin_count as usize);

    for _ in 0.. bin_count {
        chan_1_bins.push_back(0u64);
        chan_2_bins.push_back(0u64);
    }

    let mut xcorr_bins = vec![0u64; bin_count as usize];

    let mut end_last_bin = first_photon + bin_step;
    for (photon, channel) in photons {
        let us;
        let them;
        if channel {
            us = &mut chan_1_bins;
            them = &mut chan_2_bins;
        }
        else {
            us = &mut chan_2_bins;
            them = &mut chan_1_bins;
        }

        if photon < end_last_bin {
            // Still in existing bin, just add it.
            *us.back_mut().unwrap() += 1;
            continue;
        }

        // Interesting case, we calculate cross correlate for events in first bins with other events,
        // then we remove the first bin and add a new last bin.

        loop {
            end_last_bin += bin_step;

            let b0 = us.front().unwrap();
            for (i, bin) in them.iter().enumerate() {
                xcorr_bins[i] += b0 * bin;
            }
            let b0 = them.front().unwrap();
            for (i, bin) in us.iter().enumerate() {
                xcorr_bins[i] += b0 * bin;
            }

            us.pop_front();
            them.pop_front();

            if photon >= end_last_bin {
                us.push_back(0);
                them.push_back(0);
            }
            else {
                us.push_back(1);
                them.push_back(0);
                break;
            }
        }
    }

    xcorr_bins[0] /= 2;
    xcorr_bins
}

#[inline(always)]
fn approximate_asymmetric_cross_correlation(bin_count: u64, bin_step: u64, photons: impl Iterator<Item=(u64, bool)>, first_photon: u64) -> (Vec<u64>, Vec<u64>) {
    let mut chan_1_bins = VecDeque::with_capacity(bin_count as usize);
    let mut chan_2_bins = VecDeque::with_capacity(bin_count as usize);

    for _ in 0.. bin_count {
        chan_1_bins.push_back(0u64);
        chan_2_bins.push_back(0u64);
    }

    let mut xcorr_bins_forwards = vec![0u64; bin_count as usize];
    let mut xcorr_bins_backwards = vec![0u64; bin_count as usize];

    let mut end_last_bin = first_photon + bin_step;
    for (photon, channel) in photons {
        let (us, them, us_xcorr, them_xcorr);
        if channel {
            us = &mut chan_1_bins;
            us_xcorr = &mut xcorr_bins_forwards;
            them = &mut chan_2_bins;
            them_xcorr = &mut xcorr_bins_backwards;
        }
        else {
            us = &mut chan_2_bins;
            us_xcorr = &mut xcorr_bins_backwards;
            them = &mut chan_1_bins;
            them_xcorr = &mut xcorr_bins_forwards;
        }

        if photon < end_last_bin {
            // Still in existing bin, just add it.
            *us.back_mut().unwrap() += 1;
            continue;
        }

        // Interesting case, we calculate cross correlate for events in first bins with other events,
        // then we remove the first bin and add a new last bin.  Except we do this in a loop because we
        // might have to do it multiple times.

        loop {
            end_last_bin += bin_step;

            let b0 = us.front().unwrap();
            // print!("\tb0 us: {}\t\t", b0);
            assert_eq!(them.len(), bin_count as usize);
            for (i, bin) in them.iter().enumerate() {
                us_xcorr[i] += b0 * bin;
            }
            let b0 = them.front().unwrap();
            // println!("bo them: {}", b0);
            assert_eq!(us.len(), bin_count as usize);
            for (i, bin) in us.iter().enumerate() {
                them_xcorr[i] += b0 * bin;
            }

            us.pop_front();
            them.pop_front();

            if photon >= end_last_bin {
                us.push_back(0);
                them.push_back(0);
            }
            else {
                us.push_back(1);
                them.push_back(0);
                break;
            }
        }
    }

    assert_eq!(xcorr_bins_backwards[0], xcorr_bins_forwards[0]);
    xcorr_bins_backwards.reverse();
    xcorr_bins_backwards.pop();

    (xcorr_bins_backwards, xcorr_bins_forwards)
}

fn auto_correlation(bins: &mut [u64], bin_step: u64, photons: impl Iterator<Item=u64>) {
    use std::collections::VecDeque;
    let max_time = bins.len() as u64 * bin_step;

    let mut stored_photons = VecDeque::new();

    for photon in photons {
        while !stored_photons.is_empty() && photon - stored_photons[0] >= max_time {
            let old_photon = stored_photons.pop_front().unwrap();
            for &remaining_photon in &stored_photons {
                let delta: u64 = remaining_photon - old_photon;
                let bin: u64 = delta / bin_step;
                bins[bin as usize] += 1;
            }
        }

        stored_photons.push_back(photon);
    }
}
// Thanks /u/DroidLogician for help with this macro
// https://www.reddit.com/r/rust/comments/aft2h3/hey_rustaceans_got_an_easy_question_ask_here_32019/eeaks2l/
macro_rules! tree_of_if {
    // if we've only got one expression (or block) left, just run it.
    ($expr:expr) => {$expr};

    (
        // expressions cannot be followed by blocks in macros
        if let $pat:pat = ($expr:expr) { $($body:stmt ;)* } $($rest:tt)*
    ) => {
        if let $pat = $expr {
            $($body)*
            // we recurse with the remaining patterns in both cases
            tree_of_if!($($rest)*)
        } else {
            tree_of_if!($($rest)*)
        }
    };

    (
        // expressions cannot be followed by blocks in macros
        if ($expr:expr) { $($body:stmt ;)* } $($rest:tt)*
    ) => {
        if $expr {
            $($body ;)*
            // we recurse with the remaining patterns in both cases
            tree_of_if!($($rest)*)
        } else {
            tree_of_if!($($rest)*)
        }
    };
}

fn validate_file(buffer: &[u8]) {
    let mut photon_iter = PhotonIter::new(buffer);
    let mut equal_count: u64 = 0;
    let mut out_of_order: u64 = 0;
    let mut i: u64 = 0;
    loop {
        let photon = photon_iter.next().unwrap();
        let photon2 = if let Some(p) = photon_iter.clone().next() { p } else { break };



        if photon2 < photon {
            println!("Out of order: {}", i);
            out_of_order += 1;
        }
        if photon == photon2 {
            let pdl = photon_iter.data.len();
            let range = if pdl > 50 { 0.. 50 } else { 0.. pdl };
            println!("Equal: {} - {} / {} ({:?})", i, pdl, buffer.len(), &photon_iter.data[range]);
            equal_count += 1;
        }
        i += 1;

        if equal_count + out_of_order > 100 {
            break;
        }
    }

    if out_of_order > 0 || equal_count > 0 {
        panic!("Out of order photons: {}\nEqual Timestamped Photons: {}", out_of_order, equal_count);
    }

    println!("File validated");
}

#[derive(Clone, Debug, StructOpt)]
#[structopt(name = "photon_correlation", about = "Run cross correlation on HydraHarpT2 files")]
struct Opt {
    /// Cross correlate, takes output file as argument ('-' for stdout)
    #[structopt(short = "c", long = "cross-correlate")]
    cross_correlate: Option<PathBuf>,
    /// Auto correlate, takes output file as argument ('-' for stdout)
    #[structopt(short = "a", long = "auto-correlate")]
    auto_correlate: Option<PathBuf>,
    /// Run an approximate cross correlation routine, takes output file as argument ('-' for stdout)
    ///
    /// The output timestamps are the center of each bin. The approximation is that a photon pair
    /// with time difference d, with bins t(n) < d < t(n+1) on each side of it, falls into bin
    /// t(n) with probability (t(n+1) - d)/(t(n+1) - t(n)), otherwise it falls into bin t(n+1).
    ///
    /// The runtime of this method is O(num_photons + "duration file" * num_timestep^2 / cor_time).
    /// On the sample file this is being developed on duration file is ~12 million times num_photons,
    /// so the determining factor is the second expression. As such it is suited for long correlations
    /// with few timesteps.
    #[structopt(long = "approx-cross-correlate")]
    approximate_cross_correlate: Option<PathBuf>,
    /// Print header and number of photons, takes output file as argument ('-' for stdout)
    #[structopt(short = "m", long = "metadata")]
    metadata: Option<PathBuf>,

    /// Input file
    #[structopt(parse(from_os_str))]
    input: PathBuf,

    /// Number of timesteps to cross correlate over.
    #[structopt(short = "n", long = "num_timestep", default_value = "1000",  parse(try_from_str = "parse_usize"))]
    num_timestep: usize,
    /// Length of each timestep.
    #[structopt(short = "T", long = "cor_time", default_value = "10000000", parse(try_from_str = "parse_u64"))]
    cor_time: u64,

    /// Truncate file to this number of photons
    #[structopt(long="num_photons",  parse(try_from_str = "parse_usize"))]
    num_photons: Option<usize>,
    /// Start Time
    #[structopt(short = "s", long="start_time", default_value = "0", parse(try_from_str = "parse_u64"))]
    start_time: u64,
    /// End Time
    #[structopt(short = "e", long="end_time", parse(try_from_str = "parse_u64"))]
    end_time: Option<u64>,

    /// Print progress every n photons.
    #[structopt(long = "progress", parse(try_from_str = "parse_u64"))]
    progress: Option<u64>,

    /// Don't run in parallel
    #[structopt(long="sequential")]
    sequential: bool,
    /// Validate file is like we expect before running (takes a few minutes).
    #[structopt(long="validate-file")]
    validate_file: bool,
    /// Validate correlation outputs by running slower and faster version of code
    /// and make sure they get the same output (slow).
    #[structopt(long="validate-correlation")]
    validate_correlation: bool,
}

fn get_output(path: &PathBuf) -> Box<Write + Send> {
    use std::path::Path;
    if path == Path::new("-") {
        Box::new(stdout())
    }
    else {
        Box::new(File::create(path).expect("Output file doesn't exist"))
    }
}

fn parse_u64(text: &str) -> Result<u64, std::num::ParseIntError> {
    use std::str::FromStr;

    // Try and parse as a bare int
    let r = u64::from_str(text);
    if r.is_ok() { return r };

    // Try and parse as scientific notation
    // TODO: Support '.'
    if let Some(idx) =  text.find('e') {
        let (prefix, suffix) = text.split_at(idx);
        let suffix = &suffix[1..];

        let r_mult = u64::from_str(prefix)?;
        let r_exp = u32::from_str(suffix)?;

        return Ok(r_mult * (10_u64.pow(r_exp)));
    }

    // If not scientific notation, return the original error
    r
}

fn parse_usize(text: &str) -> Result<usize, std::num::ParseIntError> {
    use std::str::FromStr;

    // Try and parse as a bare int
    let r = usize::from_str(text);
    if r.is_ok() { return r };

    // Try and parse as scientific notation
    // TODO: Support '.'
    if let Some(idx) =  text.find('e') {
        let (prefix, suffix) = text.split_at(idx);
        let suffix = &suffix[1..];

        let r_mult = usize::from_str(prefix)?;
        let r_exp = u32::from_str(suffix)?;

        return Ok(r_mult * (10_usize.pow(r_exp)));
    }

    // If not scientific notation, return the original error
    r
}

fn static_ref<T>(v: T) -> &'static T {
    let b = Box::new(v);
    let r = unsafe{ std::mem::transmute::<&T, &'static T>(&b) };
    std::mem::forget(b);
    r
}

fn main() {
    let opt = Opt::from_args();
    let mut thread_handles = vec![];

    let file = std::fs::File::open(&opt.input).unwrap();
    let mmap = unsafe { memmap::MmapOptions::new().map(&file).unwrap() };
    let mmap = static_ref(mmap);
    let (i, header) = strip_remaining_from_err(header::parse(&mmap)).unwrap();

    if opt.validate_file {
        thread_handles.push(
            thread::spawn(move || {
                validate_file(i);
            })
        );
    }

    let num_records: i64 = header[&(b"TTResult_NumberOfRecords\0\0\0\0\0\0\0\0", -1)].int();
    let resolution: f64 = header[&(b"MeasDesc_GlobalResolution\0\0\0\0\0\0\0", -1)].float();

    assert_eq!(resolution, 1e-12, "We expect resolution in picoseconds");

    // println!("Number of records: {}", num_records);
    assert!(num_records > 0);

    let record_type = header[&(b"TTResultFormat_TTTRRecType\0\0\0\0\0\0", -1)];
    assert_eq!(record_type.int(), 0x1010204, "We only parse HydraHarpT2 files for now");

    let mut photons = FastPhotonIter::new(i);

    // Skip photons until the start time.
    while photons.clone().next().expect("No photons after start time").0 < opt.start_time {
        photons.next();
    }

    let (first_photon, _) = photons.clone().next().unwrap();

    let bin_size = opt.cor_time / opt.num_timestep as u64;

    if opt.auto_correlate.is_some() && opt.metadata.is_some() {
        assert_ne!(opt.auto_correlate, opt.metadata, "Output files must differ");
    }
    if opt.cross_correlate.is_some() && opt.metadata.is_some() {
        assert_ne!(opt.cross_correlate, opt.metadata, "Output files must differ");
    }
    if opt.cross_correlate.is_some() && opt.auto_correlate.is_some() {
        assert_ne!(opt.cross_correlate, opt.auto_correlate, "Output files must differ");
    }

    tree_of_if!(
        // TODO: num_photons *really* wants to be a u64, but `.take` expects a usize.
        if let Some(num_photons) = (opt.num_photons) {
            let photons = photons.take(num_photons);
        }

        if let Some(end_time) = (opt.end_time) {
            let photons = photons.take_while(move |p| end_time > p.0);
        }

        if let Some(progress_num) = (opt.progress) {
            let mut count = 0;
            let photons = photons.inspect(move |_| {
                if count % progress_num == 0 {
                    println!("Processed {} photons", count);
                };
                count += 1
            });
        }

        {
            if let Some(ref path) = &opt.metadata {
                let mut meta_output = get_output(&path);

                // Unclear if it would be better to do this as part of other passes. One hopes that running it in parallel
                // ends up preloading photons for other passes (but if it gets too far ahead it will be bad).
                let photons = photons.clone();
                thread_handles.push(
                    thread::spawn(move || {
                        header::write_header(&mut *meta_output, header);
                        let mut count: u64 = 0;
                        let mut last: u64 = 0;
                        for (photon, _) in photons {
                            count += 1;
                            last = photon;
                        }
                        writeln!(meta_output, "Photon count: {}", count).unwrap();
                        writeln!(meta_output, "Last photon: {}", last).unwrap();
                    })
                );
            }

            if let Some(path) = &opt.auto_correlate {
                let mut auto_output = get_output(&path);
                let photons = photons.clone();
                let opt = opt.clone();
                thread_handles.push(
                    thread::spawn(move || {
                        eprintln!("WARNING: Auto correlate probably doesn't do what you want, we ignore channel information right now instead of autocorrelating each channel");
                        let mut bins: Vec<u64> = vec![0; opt.num_timestep];
                        let photons = photons.map(|(time, _channel)| time);
                        auto_correlation(&mut bins, bin_size, photons);

                        let mut bin_start = 0;
                        for &b in &bins {
                            writeln!(auto_output, "{} {}", bin_start, b).unwrap();
                            bin_start += bin_size;
                        }
                    })
                );
            }

            if let Some(path) = &opt.cross_correlate {
                let mut cross_output = get_output(&path);
                let photons = photons.clone();
                let opt = opt.clone();

                thread_handles.push(
                    thread::spawn(move || {

                        let mut bins_forward: Vec<u64> = vec![0; opt.num_timestep];
                        let mut bins_backwards: Vec<u64> = vec![0; opt.num_timestep];

                        if !opt.sequential {
                            parallel_asymmetric_cross_correlation(&mut bins_forward, &mut bins_backwards, bin_size, photons.clone(), first_photon)
                        }
                        else {
                            asymmetric_cross_correlation(&mut bins_forward, &mut bins_backwards, bin_size, photons.clone(), first_photon)
                        }

                        let mut bin_start: i64 = - (bin_size as i64 * opt.num_timestep as i64);
                        for &b in bins_backwards.iter().rev() {
                            writeln!(cross_output, "{} {}", bin_start, b).unwrap();
                            bin_start += bin_size as i64;
                        }

                        assert_eq!(bin_start, 0);

                        for &b in &bins_forward[..] {
                            writeln!(cross_output, "{} {}", bin_start, b).unwrap();
                            bin_start += bin_size as i64;
                        }



                        if opt.validate_correlation {
                            let mut bins: Vec<u64> = vec![0; opt.num_timestep];
                            symmetric_cross_correlation(&mut bins, bin_size, photons.clone(), first_photon);
                            for i in 0.. opt.num_timestep {
                                // Note: This *will* fail if there are two simultaneous photons, because they
                                // are currently double counted. I'll consider that a feature for now.
                                assert_eq!(bins[i], bins_forward[i] + bins_backwards[i]);
                            }
                        }
                    })
                );
            }

            if let Some(path) = &opt.approximate_cross_correlate {
                let mut approx_cross_output = get_output(&path);

                let (bins_backwards, bins_forwards) = approximate_asymmetric_cross_correlation(opt.num_timestep as u64 , bin_size, photons.clone(), first_photon);
                let mut bin_start: i64 = - (bin_size as i64 * bins_backwards.len() as i64);
                for &b in &bins_backwards {
                    writeln!(approx_cross_output, "{} {}", bin_start, b).unwrap();
                    bin_start += bin_size as i64;
                }

                assert_eq!(bin_start, 0);

                for &b in &bins_forwards {
                    writeln!(approx_cross_output, "{} {}", bin_start, b).unwrap();
                    bin_start += bin_size as i64;
                }


                if opt.validate_correlation {
                    let bins = approximate_symmetric_cross_correlation(opt.num_timestep as u64, bin_size, photons.clone(), first_photon);
                    assert_eq!(bins[0], bins_forwards[0]);
                    let back_len = bins_backwards.len();
                    for i in 1.. opt.num_timestep {
                        assert_eq!(bins[i], bins_forwards[i] + bins_backwards[back_len - i]);
                    }
                }
            }
        }
    );

    for handle in thread_handles {
        handle.join().unwrap();
    }
}
