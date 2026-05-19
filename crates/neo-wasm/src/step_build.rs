//! Per-step bundles the wasm trace builder hands to the prover.
//!
//! Owns the `(label, assignment, extension_data)` triple. The R1CS-F'
//! frontend in `neo-fold-clean` takes the raw assignment, bit-decomposes
//! it inside `compile_step`, and builds the foldable F'-encoded
//! `CcsInstance` internally — neo-wasm does not commit to the assignment
//! itself. Extension data carries the bytecode-fetch and shout-lookup
//! records the future Shout/Twist proving layer will consume; today they
//! are not bound to any subprotocol.

use neo_math::F;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BytecodeFetchRecord {
    pub pc: u16,
    pub opcode: u16,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShoutLookupRecord {
    pub shout_id: u32,
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WasmStepExtensionData {
    pub bytecode_fetch: Option<BytecodeFetchRecord>,
    pub shout_lookup: Option<ShoutLookupRecord>,
}

/// One prepared step: a labeled R1CS-satisfying assignment ready to fold
/// through the R1CS-F' chain builder, plus the extension records the
/// future lookup-proving layer will bind.
#[derive(Clone, Debug)]
pub struct WasmStepBuild {
    pub label: String,
    pub assignment: Vec<F>,
    pub extension_data: WasmStepExtensionData,
}
