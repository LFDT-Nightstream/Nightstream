//! Owns the frozen Phase 0 real-point derivation helpers for RV64IM eval claims.
//!
//! It owns:
//! - deterministic Fiat-Shamir Phase 0 point sampling
//!
//! It does not own:
//! - claim emission from stage artifacts
//! - synthetic selected-opening `logical_index` points
//! - Phase 1 reduction or Phase 2 accumulation

use neo_math::{from_complex, F, K};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

use super::opening_eval_claims::{CommitmentContextId, FamilyEvalSchemaId, OpenedAjtaiObjectId};
use crate::finalize::digest32_as_fields;

pub fn derive_phase0_point(
    opened_object: &OpenedAjtaiObjectId,
    commitment_context: &CommitmentContextId,
    schema: FamilyEvalSchemaId,
    slot: u32,
    binding_digest: [u8; 32],
) -> Vec<K> {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/phase0/point");
    tr.append_fields_raw(&digest32_as_fields(opened_object.digest));
    tr.append_fields_raw(&digest32_as_fields(commitment_context.pp_seed_digest));
    tr.append_fields_raw(&digest32_as_fields(commitment_context.module_shape_digest));
    tr.append_fields_raw(&[F::from_u64(schema.tag()), F::from_u64(slot as u64)]);
    tr.append_fields_raw(&digest32_as_fields(binding_digest));
    (0..opened_object.row_domain_log_size as usize)
        .map(|coord_index| {
            tr.append_fields_raw(&[F::from_u64(coord_index as u64)]);
            let coord = tr.challenge_fields_raw(2);
            from_complex(coord[0], coord[1])
        })
        .collect()
}
