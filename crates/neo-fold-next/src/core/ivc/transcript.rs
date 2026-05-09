//! Owns native SuperNeo IVC transcript snapshots and accumulator handles.

use neo_ajtai::Commitment;
use neo_ccs::CeClaim;
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

use super::types::SuperNeoIvcTranscriptSnapshot;
use crate::finalize::{digest32_as_fields, digest_fields_as_digest32};
use crate::proof::Carry;

pub(super) fn session_transcript() -> Poseidon2Transcript {
    Poseidon2Transcript::new(b"neo.fold.next/session")
}

pub(super) fn accumulator_handle_fields(params: &NeoParams, carry: &Carry) -> [F; 4] {
    digest32_as_fields(accumulator_handle_digest(params, &carry.claims))
}

fn accumulator_handle_digest(params: &NeoParams, claims: &[CeClaim<Commitment, F, K>]) -> [u8; 32] {
    let mut preimage = crate::superneo_circuit::claim::packed_bytes_field_values(
        b"neo.fold.next/direct_ccs/accumulator_phi_dec_parent/v1",
    )
    .into_iter()
    .map(|value| F::from_u64(value.to_canonical_u64()))
    .collect::<Vec<_>>();
    preimage.push(F::from_u64(claims.len() as u64));
    if let Some(first) = claims.first() {
        let parent_len = first.c.data.len();
        preimage.push(F::from_u64(parent_len as u64));
        let base = F::from_u64(params.b as u64);
        let mut powers = Vec::with_capacity(claims.len());
        let mut pow = F::ONE;
        for claim in claims {
            if claim.c.data.len() != parent_len {
                preimage.push(F::from_u64(u64::MAX));
                return digest_fields_as_digest32(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage));
            }
            powers.push(pow);
            pow *= base;
        }
        for lane_idx in 0..parent_len {
            let mut value = F::ZERO;
            for (claim, pow) in claims.iter().zip(powers.iter().copied()) {
                value += claim.c.data[lane_idx] * pow;
            }
            preimage.push(value);
        }
    }
    digest_fields_as_digest32(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage))
}

pub(super) fn transcript_from_snapshot(snapshot: &SuperNeoIvcTranscriptSnapshot) -> Poseidon2Transcript {
    Poseidon2Transcript::from_state_and_absorbed(snapshot.state, snapshot.absorbed)
}

pub(super) fn transcript_snapshot(transcript: &Poseidon2Transcript) -> SuperNeoIvcTranscriptSnapshot {
    SuperNeoIvcTranscriptSnapshot {
        state: transcript.state(),
        absorbed: transcript.absorbed(),
    }
}

pub(super) fn append_chunk_done(transcript: &mut Poseidon2Transcript) {
    transcript.append_message(b"neo.fold.next/chunk_done", &[1]);
}
