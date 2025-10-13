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
    pub fn new(steps: Vec<NivcStepSpec>) -> Self { 
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

