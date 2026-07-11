//! Per-step bundle the wasm builder hands to the prover.
//!
//! The R1CS-F' frontend in `neo-fold-clean` takes the raw assignment,
//! bit-decomposes it inside `compile_step`, and constructs the foldable
//! F'-encoded `CcsInstance` internally — neo-wasm does not commit to the
//! assignment itself.

use neo_math::F;

/// One prepared step: an R1CS-satisfying assignment ready to fold through
/// the R1CS-F' chain builder.
#[derive(Clone, Debug)]
pub struct WasmStepBuild {
    pub assignment: Vec<F>,
}
