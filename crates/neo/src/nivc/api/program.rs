//! NIVC program definitions: available step types and their specifications

use crate::{F, ivc::StepBindingSpec};
use neo_ccs::CcsStructure;

/// One step specification in an NIVC program (corresponds to one function `F_j` in Definition 11)
#[derive(Clone)]
pub struct NivcStepSpec {
    /// The constraint system (CCS) for this step type (i.e. defines relation F_j)
    pub ccs: CcsStructure<F>,
    /// Binding spec: how to map witness bits to y_step, linking indices, etc.
    pub binding: StepBindingSpec,
}

/// A non-uniform IVC program: a set of available step types `F_0, F_1, …`
#[derive(Clone)]
pub struct NivcProgram {
    /// Vector of step types (lanes) available
    pub steps: Vec<NivcStepSpec>,
}

impl NivcProgram {
    /// Create a new NIVC program from a vector of step specifications
    /// 
    /// # Panics
    /// Panics if any step CCS has fewer than 3 rows (ℓ < 2 requirement after padding)
    pub fn new(steps: Vec<NivcStepSpec>) -> Self {
        // SECURITY: Validate that all step CCS structures meet minimum requirements
        // ℓ = ceil(log2(n)) must be ≥ 2 for the sumcheck protocol
        // n is padded to next power of 2 (max 2), so n=3 → 4 → ℓ=2 is acceptable
        for (lane_idx, spec) in steps.iter().enumerate() {
            if spec.ccs.n < 3 {
                panic!(
                    "CCS validation failed for lane {}: n={} is too small (minimum n=3 required). \
                    The sumcheck challenge length ℓ=ceil(log2(n_padded)) must be ≥ 2 for protocol security. \
                    n is padded to next power-of-2 (minimum 2), so n=3→4→ℓ=2, n=2→2→ℓ=1 (too small). \
                    Please ensure your circuit has at least 3 constraint rows.",
                    lane_idx, spec.ccs.n
                );
            }
        }
        Self { steps } 
    }
    
    /// Number of lanes / step types = ℓ in paper
    pub fn len(&self) -> usize { 
        self.steps.len() 
    }
    
    /// Is program empty? (shouldn't be in valid usage)
    pub fn is_empty(&self) -> bool { 
        self.steps.is_empty() 
    }
}

