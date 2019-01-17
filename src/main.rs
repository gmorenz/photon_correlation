
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

#[inline(always)]
fn cross_correlation(bins: &mut [u64], bin_step: u64, mut photons: impl Iterator<Item=(u64, bool)>, first_photon: u64) {
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

fn run_cross_correlation(num_bins: usize, bin_size: u64, filename: PathBuf, num_photons: Option<usize>, start_time: u64, end_time: Option<u64>) {
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

    let mut bins = vec![0; num_bins];


    let mut photons = FastPhotonIter::new(i);
    
    // Skip photons until the start time.
    while photons.clone().next().expect("No photons after start time").0 < start_time {
        photons.next();
    }

    let (first_photon, _) = photons.clone().next().unwrap();


    if let Some(num_photons) = num_photons {
        // TODO: num_photons *really* wants to be a u64, but `.take` expects a usize.
        if let Some(end_time) = end_time {
            cross_correlation(&mut bins, bin_size, photons.take(num_photons).take_while(|p| end_time > p.0), first_photon);
        }
        else {
            cross_correlation(&mut bins, bin_size, photons.take(num_photons), first_photon);
        }
    }
    else {
        if let Some(end_time) = end_time {
            cross_correlation(&mut bins, bin_size, photons.take_while(|p| end_time > p.0), first_photon);
        }
        else {
            cross_correlation(&mut bins, bin_size, photons, first_photon);
        }
    }
    
    for &b in &bins[..] {
        println!("{}", b);
    }

}

#[derive(Debug, StructOpt)]
#[structopt(name = "photon_correlation", about = "Run cross correlation on HydraHarpT2 files")]
struct Opt {
    /// Number of timesteps to cross correlate over.
    #[structopt(short = "n", long = "num_timestep", default_value = "100")]
    num_timestep: usize,
    /// Length of each timestep.
    #[structopt(short = "T", long = "cor_time", default_value = "1000000")]
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
    );
}
