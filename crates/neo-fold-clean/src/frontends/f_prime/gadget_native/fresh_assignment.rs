//! Fresh SuperNeo CCS boundary for one gadget-native encoded assignment.
//!
//! Owns: the single prover-side conversion from a materialized encoded
//! relation to the fresh `(claim, witness)` input consumed by NIFS.
//!
//! Does not own: low-norm validation, assignment packing, Ajtai commitment,
//! source decoding, or CCS satisfaction. Those remain with
//! `CcsInstance::from_low_norm_assignment` and the encoded relation.
//!
//! Emits constraints: no.
//!
//! Authority boundary: the complete encoded assignment is the prover input to
//! this constructor. Its plan-fixed prefix is public; every remaining
//! coordinate is committed in the fresh witness. Verifier authority exists
//! only after the corresponding commitment/proof is verified; this conversion
//! does not call `EncodedGadgetNativeR1cs::is_satisfied` and no digest or
//! decoded source projection substitutes for that verification.
//!
//! | Input | Mathematical obligation | Owner |
//! |---|---|---|
//! | `structure` | relation whose full assignment has length `m` | gadget-native lowering |
//! | `assignment` | low-norm `z = [x, w]`, packed column-major into `Z` | `CcsInstance` |
//! | `plan.public_input_len()` | exact public/private split of `z` | gadget-native plan |

use neo_ajtai::AjtaiSModule;

use crate::paper::params::Params;
use crate::paper::relations::{CcsInstance, RelationError};

use super::EncodedGadgetNativeR1cs;

impl EncodedGadgetNativeR1cs {
    /// Convert the complete encoded assignment into one fresh NIFS input.
    pub fn to_fresh_ccs_instance(&self, params: &Params, log: &AjtaiSModule) -> Result<CcsInstance, RelationError> {
        CcsInstance::from_low_norm_assignment(
            params,
            log,
            &self.structure,
            &self.assignment,
            self.plan.public_input_len(),
        )
    }
}
