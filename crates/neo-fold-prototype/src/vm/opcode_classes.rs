//! Owns static opcode-class descriptions for a VM.

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpcodeClassSpec<Op> {
    pub id: Op,
    pub name: &'static str,
    pub selector_index: usize,
    pub writes_vx: bool,
    pub writes_i: bool,
    pub touches_ram: bool,
}
