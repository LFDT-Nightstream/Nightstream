//! Fixed transcript rounds and exact candidate extraction for one rho sample.
//!
//! Owns: the eight-round counter schedule, Poseidon2 digest requests, canonical
//! lane decomposition, and exact 16-bit candidate extraction for one rho.
//!
//! Does not own: the incoming cursor, Poseidon2 permutation equations, or
//! chunk arithmetic.
//!
//! Emits constraints: yes, directly and through `chunk`.
//!
//! Authority boundary: the caller supplies the verifier transcript cursor;
//! each lane is recomposed from checked canonical bits before chunk use.
//!
//! | Stage path | Function | Equation | Multiplicity | Emitted rows/formula | Lowered gate | Lean theorem |
//! |---|---|---|---:|---|---|---|
//! | `challenge.sampler.initialize` | `collect_chunks` | prefix count starts at zero | one per rho | one equality | generic R1CS | `rhoDigestTrace` base state |
//! | `challenge.transcript.digest_rounds` | `collect_chunks` | append `[1,seed+iter]`, squeeze digest | eight per rho | transcript/Poseidon2 rows | Poseidon2 | `rhoDigestTrace` |
//! | `challenge.transcript.lane_bit_decomposition` | `collect_chunks` | lane `= sum 2^i bit_i` canonically; candidate `= 65535 - low16` | 32 lanes, 64 candidates per rho | canonical bit rows and complemented candidate expression | generic R1CS | `rhoDigestTrace` |

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder};
use crate::engine::r1cs_circuit::transcript::TranscriptGadget;
use crate::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;

use super::chunk::{process_chunk, ChunkRecord};
use super::{pi_rlc_challenge_stage, MAX_ITER, TOTAL_CHUNKS};

pub(super) fn collect_chunks(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    seed: u64,
) -> Vec<ChunkRecord> {
    builder.begin_encoding_stage(pi_rlc_challenge_stage::SAMPLE_INITIALIZE);
    let mut chunks = Vec::with_capacity(TOTAL_CHUNKS);
    let mut cumulative = builder.alloc(F::ZERO);
    builder.enforce_eq(&Lc::from_var(cumulative), &Lc::zero());

    for iter in 0..MAX_ITER {
        builder.begin_encoding_stage(pi_rlc_challenge_stage::TRANSCRIPT_DIGEST);
        let counter = seed.wrapping_add(iter as u64);
        transcript.append_fields_raw_const(builder, &[F::ONE, F::from_u64(counter)]);
        let digest = transcript.digest_fields(builder);

        for lane in digest {
            builder.begin_encoding_stage(pi_rlc_challenge_stage::LANE_BIT_DECOMPOSITION);
            let bits = decompose_var_to_u64_bits(builder, lane);
            for raw_bits in bits[..32].chunks_exact(16) {
                let chunk = process_chunk(builder, raw_bits, cumulative);
                cumulative = chunk.cumulative;
                chunks.push(chunk);
            }
        }
    }
    debug_assert_eq!(chunks.len(), TOTAL_CHUNKS);
    chunks
}
