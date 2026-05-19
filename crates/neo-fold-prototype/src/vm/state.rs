//! Owns static state-shape descriptions for a VM.

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegisterSpec {
    pub name: &'static str,
    pub width_bits: u16,
    pub slots: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateSpec {
    pub registers: Vec<RegisterSpec>,
    pub memory_bytes: usize,
    pub program_counter: &'static str,
}
