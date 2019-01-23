// Unfortunately there is some unreachable code in a macro that gives a warning otherwise,
// and doesn't seem to be able to be disabled by putting this just on that macro.
#![allow(unreachable_code)]

// Used for use in macros that don't accept ::
use nom::{le_i64, le_u64, le_f64};
// Used for writing parsers as functions
use nom::IResult;

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TagValue<'a> {
    Empty,
    Bool(bool),
    Int(i64),
    BitSet(u64),
    Color(u64),
    Float(f64),
    TDateTime(f64),
    /// Python decoding for this is nonsense, panic!
    #[allow(dead_code)]
    Float8Array,
    AnsiString(&'a [u8]),
    WideString(&'a [u8]),
    BinaryBlob(&'a [u8]),
}

impl <'a> TagValue<'a> {
    pub fn int(self) -> i64 {
        if let TagValue::Int(x) = self { x }
        else { panic!("Not an int") }
    }

    pub fn float(self) -> f64 {
        if let TagValue::Float(x) = self { x }
        else { panic!("Not a flat") }
    }
} 

pub type Header<'a> = HashMap<(&'a [u8; 32], i32), TagValue<'a>>;

named!(parse_tag_value<&[u8], TagValue>,
    alt!(
        do_parse!(
            tag!(b"\x08\x00\xFF\xFF") >>
            take!(8) >>
            (TagValue::Empty)
        ) |
        do_parse!(
            tag!(b"\x08\x00\x00\x00") >>
            val: le_i64 >>
            (TagValue::Bool(val != 0))
        ) |
        do_parse!(
            tag!(b"\x08\x00\x00\x10") >>
            val: le_i64 >>
            (TagValue::Int(val))
        ) |
        do_parse!(
            tag!(b"\x08\x00\x00\x11") >>
            val: le_u64 >>
            (TagValue::BitSet(val))
        ) |
        do_parse!(
            tag!(b"\x08\x00\x00\x12") >>
            val: le_u64 >>
            (TagValue::Color(val))
        ) |
        do_parse!(
            tag!(b"\x08\x00\x00\x20") >>
            val: le_f64 >>
            (TagValue::Float(val))
        ) |
        do_parse!(
            tag!(b"\x08\x00\x00\x21") >>
            val: le_f64 >>
            (TagValue::TDateTime(val))
        ) |
        do_parse!(
            tag!(b"\xFF\xFF\x01\x20") >>
            (panic!("FloatArray. Nonsense python parser code. Let's hope it isn't used."))
        ) |
        do_parse!(
            tag!(b"\xFF\xFF\x01\x40") >>
            len: le_i64 >>
            val: take!(len) >>
            (TagValue::AnsiString(val))
        ) |
        do_parse!(
            tag!(b"\xFF\xFF\x02\x40") >>
            len: le_i64 >>
            val: take!(len) >>
            (TagValue::WideString(val)) 
        ) |
        do_parse!(
            tag!(b"\xFF\xFF\xFF\xFF") >>
            len: le_i64 >>
            val: take!(len) >>
            (TagValue::BinaryBlob(val))
        )
    )
);

fn parse_tags(mut i: &[u8]) -> IResult<&[u8], Header> {
    let mut header = Header::new();
    loop {
        let (i_new, tag_ident) = take!(i, 32)?;
        i = i_new;
        let tag_ident = array_ref!(tag_ident, 0, 32);
        
        let (i_new, tag_index) = nom::le_i32(i)?;
        i = i_new;
        
        let (i_new, value) = parse_tag_value(i)?;
        i = i_new;

        // println!("Header {}: {} {} : {:?}", j, String::from_utf8_lossy(tag_ident), tag_index, value);
        header.insert((tag_ident, tag_index), value);

        if tag_ident == b"Header_End\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0" {
            return Ok((i, header));
        }
    }
}

named!(pub parse<&[u8], Header>,
    do_parse!(
        // Magic Number
        tag!(b"PQTTTR\0\0") >>
        // Version
        tag!(b"1.0.00\0\0") >>
        header: parse_tags >>
        (header)
    )
);

pub fn write_header(output: &mut std::io::Write, headers: Header) {
    for (tag, value) in headers {
        let value_string = match value {
            TagValue::AnsiString(x) => String::from_utf8_lossy(x).to_string(),
            _ => format!("{:?}", value),
        };

        let tag_string = match tag {
            (x, -1) => String::from_utf8_lossy(x).to_string(),
            (x, i) => format!("{} ({})", String::from_utf8_lossy(x), i),
        };

        writeln!(output, "{}: {}", tag_string, value_string).unwrap();
    }
}