//! Π_RLC transcript-derived alphabet sampling in R1CS.
//!
//! Owns: orchestration and stable stage names for deriving fixed `rho` vectors.
//!
//! Does not own: Poseidon2 internals, Π_CCS cursor authority, or low-norm
//! replacement of the emitted rows.
//!
//! Emits constraints: yes, by composing the child leaves below.
//!
//! Authority boundary: the incoming transcript cursor is verifier-owned; this
//! module appends the fixed domains and counters but does not prove the earlier
//! Π_CCS output digest was absorbed.
//!
//! | Child path | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
//! |---|---|---|---|---|
//! | `challenge.transcript` | Bind outputs, replay counters/digests, and decompose lanes | yes | `digest_rounds` | `Transcript.*` |
//! | `challenge.sampler.chunk` | Reject 65535 and map mod 5 into `[-2,2]` | yes | `chunk` | `Sampler.Chunk` |
//! | `challenge.sampler.chunk.accept.packed` | Replace four canonical inverse rows by a nine-row product tree | no | `gadget_native::acceptance` | `Refinement.AggregateAcceptanceRows` |
//! | `nifs.pi_rlc.challenge.sampler.chunk.mod5.packed` | Replace the validated 20-row mod-5 block by three exact packed row families | no | `gadget_native::mod5` | `Sampler.Chunk.Mod5.PackedRows` |
//! | `challenge.sampler.acceptance_bound` | Prove at least 54 accepted chunks among 64 | yes | `acceptance` | `Sampler.Acceptance` |
//! | `challenge.sampler.selection` | Return exactly the first 54 accepted symbols | yes | `selection` | `Sampler.Selection` |
//!
//! Production uses the five-symbol alphabet `[-2, -1, 0, 1, 2]`. For each
//! rho, four Poseidon2 digests yield 64 little-endian 16-bit chunks. The fixed
//! circuit fails closed unless at least 54 chunks are accepted.

mod acceptance;
mod chunk;
mod digest_rounds;
mod selection;
pub mod stage;

pub use stage as pi_rlc_challenge_stage;

use neo_math::ring::D;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::{R1csBuilder, Var};
use crate::engine::r1cs_circuit::transcript::TranscriptGadget;

use acceptance::enforce_enough_accepts;
use digest_rounds::collect_chunks;
use selection::select_first_n_accepts;

pub(super) const MAX_ITER: usize = 4;
pub(super) const CHUNKS_PER_ITER: usize = 16;
pub(super) const TOTAL_CHUNKS: usize = MAX_ITER * CHUNKS_PER_ITER;
pub(super) const MAX_REJECTIONS: usize = TOTAL_CHUNKS - D;
pub(super) const SELECTION_WINDOW: usize = MAX_REJECTIONS + 1;

/// Sample the production length-54 alphabet vector from one transcript state.
///
/// Conditional on at least 54 accepts in the first 64 chunks, the returned
/// witness equals native `draw_alphabet_vector`. The circuit always advances
/// the transcript by four digest iterations and rejects the shortfall event.
pub fn enforce_alphabet_sample_5_d(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    seed: u64,
) -> [Var; D] {
    let chunks = collect_chunks(builder, transcript, seed);
    enforce_enough_accepts(builder, &chunks);
    let selected = select_first_n_accepts(builder, &chunks, D);
    let mut output = [Var::ONE; D];
    output.copy_from_slice(&selected);
    output
}

/// Derive all Π_RLC rho vectors from the authoritative transcript.
pub fn enforce_pi_rlc_rhos_from_transcript(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    count: usize,
) -> Vec<[Var; D]> {
    let mut rhos = Vec::with_capacity(count);
    builder.begin_encoding_stage(pi_rlc_challenge_stage::SAMPLER);
    for i in 0..count {
        builder.begin_encoding_stage(pi_rlc_challenge_stage::RHO_DOMAIN_SEPARATOR);
        transcript.append_fields_raw_const(builder, &[F::ZERO, F::from_u64(i as u64)]);
        rhos.push(enforce_alphabet_sample_5_d(builder, transcript, i as u64));
    }
    rhos
}
