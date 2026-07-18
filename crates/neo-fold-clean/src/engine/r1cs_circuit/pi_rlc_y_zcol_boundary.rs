//! Narrow semantic boundary for the active PiRLC `y_zcol` projection.
//!
//! Owns: the exact parent, quotient, and transcript-beta columns selected by
//! NIFS orchestration before projection arithmetic is lowered.
//!
//! Does not own: row correctness, PiCCS source authority, parent validity,
//! transcript soundness, costs, or permission to remove constraints.
//!
//! Emits constraints: no.
//!
//! | Leaf | Mathematical role |
//! |---|---|
//! | `parent[limb]` | active coefficients of `dec_wires.parent.y_zcol` |
//! | `quotient[limb]` | bound division-quotient advice for the same identity |
//! | `beta` | transcript-derived evaluation point consumed by both limbs |

/// Columns selected by NIFS orchestration for both active `y_zcol` limbs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiRlcYZcolBoundaryAudit {
    parent_columns: [Vec<usize>; 2],
    quotient_columns: [Vec<usize>; 2],
    beta_columns: [usize; 2],
}

impl PiRlcYZcolBoundaryAudit {
    pub(crate) fn new(
        parent_columns: [Vec<usize>; 2],
        quotient_columns: [Vec<usize>; 2],
        beta_columns: [usize; 2],
    ) -> Self {
        Self {
            parent_columns,
            quotient_columns,
            beta_columns,
        }
    }

    pub fn parent_columns(&self, limb: usize) -> &[usize] {
        &self.parent_columns[limb]
    }

    pub fn quotient_columns(&self, limb: usize) -> &[usize] {
        &self.quotient_columns[limb]
    }

    pub fn beta_columns(&self) -> [usize; 2] {
        self.beta_columns
    }

    pub(crate) fn remap(&self, old_to_new: &[usize]) -> Self {
        let remap = |columns: &[usize]| columns.iter().map(|&column| old_to_new[column]).collect();
        Self::new(
            std::array::from_fn(|limb| remap(&self.parent_columns[limb])),
            std::array::from_fn(|limb| remap(&self.quotient_columns[limb])),
            self.beta_columns.map(|column| old_to_new[column]),
        )
    }
}
