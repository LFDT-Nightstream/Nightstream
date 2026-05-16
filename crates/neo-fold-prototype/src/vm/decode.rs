//! Owns static decode-field descriptions for a VM.

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodeField {
    pub name: &'static str,
    pub width_bits: u8,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodeSpec<Op> {
    pub opcode_bits: u8,
    pub fields: Vec<DecodeField>,
    pub supported: Vec<Op>,
}
