//! Exact lowering prerequisites for the outgoing accumulator authority phase.
//!
//! Owns: phase-local source-program validation and compact manifests for
//! constraint families that may later admit a formally justified lowering.
//!
//! Does not own: accumulator semantics, digest authority, or permission to
//! remove any source row.
//!
//! Emits constraints: no.
//!
//! Authority boundary: the source R1CS remains the local implementation
//! arithmetic reference. A child module may describe a candidate lowering only
//! after replaying every owned row and proving that its candidate wires have no
//! unaccounted matrix uses. This does not make the source rows the paper-level
//! semantic authority or independently justify retaining them.
//!
//! | Child path | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
//! |---|---|---|---|---|
//! | `poseidon2_sbox` | Exact `x -> x^7` programs inside the authoritative accumulator hash | no | this module | `Sbox7Compact`, `Sbox7OutputLayout`, and `OutputAuthoritySboxManifest*` |

pub mod poseidon2_sbox;

pub use poseidon2_sbox::{
    audit_output_authority_poseidon2_sboxes, OutputAuthorityPoseidon2SboxCensus,
    OutputAuthorityPoseidon2SboxFamilyLayout, OutputAuthorityPoseidon2SboxManifest,
    OutputAuthorityPoseidon2SboxManifestError, Poseidon2PermutationCall,
};
