
#[macro_use] extern crate arrayref;
#[macro_use] extern crate nom;
#[macro_use] extern crate structopt;

mod header;

use structopt::StructOpt;
use nom::{IResult, Err};
use std::path::PathBuf;

#[derive(Clone, Copy)]
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

        // TODO: Verify
        let special = val >> 31;
        let channel = (val >> 25) & 0b_11_1111;
        let timetag = val & 0x01ff_ffff;

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


#[derive(Clone, Copy)]
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

fn parallel_asymmetric_cross_correlation(bins_forward: &mut [u64], bins_backwards: &mut [u64], bin_step: u64, photons: impl Iterator<Item=(u64, bool)> + Clone + Send, first_photon: u64) {
    fn thread(bins: &mut [u64], bin_step: u64, us: bool, photons: impl Iterator<Item=(u64, bool)>, first_photon: u64) {
        use std::collections::VecDeque;

        let max_time = bins.len() as u64 * bin_step;
        let mut stored_photons = VecDeque::new();
        // let mut i = 0;

        let mut stored_len = 0;

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
                    stored_len += stored_photons.len();

                    let delta = photon - old_photon;
                    let bin = delta / bin_step;
                    bins[bin as usize] += 1;
                }
            }
        }

        println!("stored len: {}", stored_len);
    }

    // Note: Be careful about what happens with simultaneous photons.

    crossbeam_utils::thread::scope(|s| {
        let photons1 = photons.clone();
        s.spawn(|_| {
            // Thread 1:
            thread(bins_forward, bin_step, true, photons1, first_photon);
        });

        // Thread 2:
        thread(bins_backwards, bin_step, false, photons, first_photon);
    });
}

#[inline(always)]
fn asymmetric_cross_correlation(bins_forward: &mut [u64], bins_backwards: &mut [u64], bin_step: u64, mut photons: impl Iterator<Item=(u64, bool)>, first_photon: u64) {
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
                let delta = photon - old_photon;
                let bin = delta / bin_step;
                bins[bin as usize] += 1;
            }
        }
    }
}

#[inline(always)]
fn symmetric_cross_correlation(bins: &mut [u64], bin_step: u64, mut photons: impl Iterator<Item=(u64, bool)>, first_photon: u64) {
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
                let delta = photon - old_photon;
                let bin = delta / bin_step;
                bins[bin as usize] += 1;
            }
        }
    }
}

fn auto_correlation(bins: &mut [u64], bin_step: u64, mut photons: impl Iterator<Item=u64>) {
    use std::collections::VecDeque;
    let max_time = bins.len() as u64 * bin_step;

    let mut stored_photons = VecDeque::new();

    for photon in photons {
        while !stored_photons.is_empty() && photon - stored_photons[0] >= max_time {
            let old_photon = stored_photons.pop_front().unwrap();
            for &remaining_photon in &stored_photons {
                let delta = remaining_photon - old_photon;
                let bin = delta / bin_step;
                bins[bin as usize] += 1;
            }
        }

        stored_photons.push_back(photon);
    }
}

// // Thanks /u/DroidLogician for help with this macro
// // https://www.reddit.com/r/rust/comments/aft2h3/hey_rustaceans_got_an_easy_question_ask_here_32019/eeaks2l/
// macro_rules! tree_of_if {
//     // if we're out of patterns, call `f` with the iterator we've constructed
//     (($iter:ident => $use_iter:expr)) => {
//         $use_iter
//     };
//     (
//         // we have to define the identifier used for the iterator because of hygiene
//         // defining the identifier of the function is just for reusability
//         ($iter:ident => $use_iter:expr)
//         // expressions cannot be followed by blocks in macros
//         if let $pat:pat = ($expr:expr) $new_iter:block $($rest:tt)*
//     ) => {
//             if let $pat = $expr {
//             let $iter = $new_iter;
//             // we recurse with the remaining patterns in both cases
//                 tree_of_if!(($iter => $use_iter) $($rest)*)
//             } else {
//                 tree_of_if!(($iter => $use_iter) $($rest)*)
//             }
//     };
// }

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

fn run_cross_correlation(num_bins: usize, bin_size: u64, 
        filename: PathBuf, num_photons: Option<usize>, 
        start_time: u64, end_time: Option<u64>,
        parallel: bool, quiet: bool) {
    let file = std::fs::File::open("2018-11-24-ITK5nM-long_000.ptu").unwrap();
    let mmap = unsafe { memmap::MmapOptions::new().map(&file).unwrap() };
    let (i, header) = strip_remaining_from_err(header::parse(&mmap)).unwrap();

    let num_records = header[&(b"TTResult_NumberOfRecords\0\0\0\0\0\0\0\0", -1)].int();
    let resolution = header[&(b"MeasDesc_GlobalResolution\0\0\0\0\0\0\0", -1)].float();

    assert_eq!(resolution, 1e-12, "We expect resolution in picoseconds");

    // println!("Number of records: {}", num_records);
    assert!(num_records > 0);

    let record_type = header[&(b"TTResultFormat_TTTRRecType\0\0\0\0\0\0", -1)];
    assert_eq!(record_type.int(), 0x1010204, "We only parse HydraHarpT2 files for now");

    let mut bins_forward = vec![0; num_bins];
    let mut bins_backwards = vec![0; num_bins];


    let mut photons = FastPhotonIter::new(i);

    // Skip photons until the start time.
    while photons.clone().next().expect("No photons after start time").0 < start_time {
        photons.next();
    }

    let (first_photon, _) = photons.clone().next().unwrap();

    // tree_of_if!(
    //     (photons => asymmetric_cross_correlation(&mut bins_forward, &mut bins_backwards, bin_size, photons, first_photon))

    //     // TODO: num_photons *really* wants to be a u64, but `.take` expects a usize.
    //     if let Some(num_photons) = (num_photons) {
    //         photons.take(num_photons)
    //     }

    //     if let Some(end_time) = (end_time) {
    //         photons.take_while(|p| end_time > p.0)
    //     }
    // );

    tree_of_if!(
        // TODO: num_photons *really* wants to be a u64, but `.take` expects a usize.
        if let Some(num_photons) = (num_photons) {
            let photons = photons.take(num_photons);
        }

        if let Some(end_time) = (end_time) {
            let photons = photons.take_while(|p| end_time > p.0);
        }

        if parallel {
            parallel_asymmetric_cross_correlation(&mut bins_forward, &mut bins_backwards, bin_size, photons, first_photon)
        }
        else {
            asymmetric_cross_correlation(&mut bins_forward, &mut bins_backwards, bin_size, photons, first_photon)
        }
    );

    if !quiet {
        for &b in &bins_backwards[..] {
            println!("{}", b);
        }

        for &b in &bins_forward[..] {
            println!("{}", b);
        }
    }

}

#[derive(Debug, StructOpt)]
#[structopt(name = "photon_correlation", about = "Run cross correlation on HydraHarpT2 files")]
struct Opt {
    /// Number of timesteps to cross correlate over.
    #[structopt(short = "n", long = "num_timestep", default_value = "1000")]
    num_timestep: usize,
    /// Length of each timestep.
    #[structopt(short = "T", long = "cor_time", default_value = "10000000")]
    cor_time: u64,
    /// Input file
    #[structopt(parse(from_os_str))]
    input: PathBuf,
    /// Truncate file to this number of photons
    #[structopt(long="num_photons")]
    num_photons: Option<usize>,
    /// Start Time
    #[structopt(short = "s", long="start_time", default_value = "0")]
    start_time: u64,
    /// End Time
    #[structopt(short = "s", long="end_time")]
    end_time: Option<u64>,
    /// Run in parallel
    #[structopt(short = "p", long="parallel")]
    parallel: bool,
    /// Don't print results
    #[structopt(short = "q", long="quiet")]
    quiet: bool,
}

fn main() {
    let opt = Opt::from_args();

    run_cross_correlation(
        opt.num_timestep,
        opt.cor_time / opt.num_timestep as u64,
        opt.input,
        opt.num_photons,
        opt.start_time,
        opt.end_time,
        opt.parallel,
        opt.quiet,
    );
}
