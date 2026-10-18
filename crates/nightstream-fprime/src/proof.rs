//! Public assignment and proof-run evidence returned by package execution.

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProofRun {
    pub(super) a_nonzeros: usize,
    pub(super) b_nonzeros: usize,
    pub(super) c_nonzeros: usize,
}

impl ProofRun {
    pub fn a_nonzeros(self) -> usize {
        self.a_nonzeros
    }

    pub fn b_nonzeros(self) -> usize {
        self.b_nonzeros
    }

    pub fn c_nonzeros(self) -> usize {
        self.c_nonzeros
    }
}
