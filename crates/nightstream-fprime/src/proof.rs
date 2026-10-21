//! Public assignment returned by package execution.

/// The exact direct-Spartan assignment produced by the package witness IR.
#[derive(Clone, Debug)]
pub struct WitnessAssignment {
    pub(super) private_values: Vec<u64>,
    pub(super) public_values: Vec<u64>,
}

impl WitnessAssignment {
    pub fn private_values(&self) -> &[u64] {
        &self.private_values
    }

    pub fn public_values(&self) -> &[u64] {
        &self.public_values
    }
}
