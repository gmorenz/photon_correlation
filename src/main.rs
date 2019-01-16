
#[macro_use] extern crate arrayref;
#[macro_use] extern crate nom;

mod header;

use nom::{IResult, Err};

struct PhotonIter<'a> {
    data: &'a [u8],
    basetime: u64
}

impl <'a> Iterator for PhotonIter<'a> {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
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
            Some(self.basetime + timetag as u64)
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

fn correlation(bins: &mut [u64], bin_step: u64, mut photons: impl Iterator<Item=u64>) {
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

fn main() {
    let file = std::fs::File::open("2018-11-24-ITK5nM-long_000.ptu").unwrap();
    let mmap = unsafe { memmap::MmapOptions::new().map(&file).unwrap() };
    let (i, header) = strip_remaining_from_err(header::parse(&mmap)).unwrap();

    let num_records = header[&(b"TTResult_NumberOfRecords\0\0\0\0\0\0\0\0", -1)].int();
    let resolution = header[&(b"MeasDesc_GlobalResolution\0\0\0\0\0\0\0", -1)].float();

    assert_eq!(resolution, 1e-12, "We expect resolution in picoseconds");

    println!("Number of records: {}", num_records);
    assert!(num_records > 0);

    let record_type = header[&(b"TTResultFormat_TTTRRecType\0\0\0\0\0\0", -1)];
    assert_eq!(record_type.int(), 0x1010204, "We only parse HydraHarpT2 files for now");

    let photons = PhotonIter {
        data: i,
        basetime: 0,
    };

    // let mut i = 0;
    // for photon in photons {
    //     i += 1;
    //     if i > 10000 { break }
    //     println!("{}", photon);
    // }

    // Run a correlation with 1ns sized bins, over 1 microsecond.
    let mut bins = [0; 1000];
    correlation(&mut bins, 1000, photons);
    for &b in &bins[..] {
        println!("{}", b);
    }

    // let (i, ()) = strip_remaining_from_err(read_ht2_records(i, resolution, num_records as u64)).unwrap();
    // assert!(i.len() == 0, "Output remaining after reading records");
    // println!("{:#?}", header);
    // println!("{} {} {} {}", mmap.len(), r.0.len(), r.1.len(), String::from_utf8_lossy(r.1));
}
