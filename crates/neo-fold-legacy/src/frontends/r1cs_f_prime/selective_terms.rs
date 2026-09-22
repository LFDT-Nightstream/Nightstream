//! Construction-time storage for one selectively lowered CCS matrix.

use neo_ccs::{GeometricRowRun, SeededPhi81LinearBlock};
use neo_math::F;

pub(super) struct MatrixTerms {
    pub explicit: Vec<(usize, usize, F)>,
    pub seeded: Vec<SeededPhi81LinearBlock>,
    pub geometric_runs: Vec<GeometricRowRun<F>>,
    pub retain_geometric: bool,
}

impl MatrixTerms {
    pub(super) fn new(retain_geometric: bool) -> Self {
        Self {
            explicit: Vec::new(),
            seeded: Vec::new(),
            geometric_runs: Vec::new(),
            retain_geometric,
        }
    }

    pub(super) fn push(&mut self, term: (usize, usize, F)) {
        self.explicit.push(term);
    }
}
