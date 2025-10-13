//! Transcript and challenge derivation for IVC
//!
//! This module handles Fiat-Shamir transcript operations for deriving
//! the folding challenge ρ and step digests.

use super::prelude::*;
use p3_symmetric::Permutation;
use neo_ccs::crypto::poseidon2_goldilocks as p2;
use neo_transcript::{Transcript, Poseidon2Transcript};
use neo_transcript::labels as tr_labels;

/// Deterministic Poseidon2 domain-separated hash to derive folding challenge ρ
/// Uses the same Poseidon2 configuration as context_digest_v1 for consistency
#[allow(unused_assignments)]
pub fn rho_from_transcript(prev_acc: &Accumulator, step_digest: [u8; 32], c_step_coords: &[F]) -> (F, [u8; 32]) {
    // Use centralized Merlin-style transcript (Poseidon2 backend)
    #[cfg(feature = "fs-guard")]
    neo_transcript::fs_guard::reset("rho/actual");

    let mut tr = Poseidon2Transcript::new(b"neo/ivc");
    tr.append_fields(tr_labels::STEP, &[F::from_u64(prev_acc.step)]);
    tr.append_message(tr_labels::ACC_DIGEST, &prev_acc.c_z_digest);
    tr.append_fields(b"acc/y", &prev_acc.y_compact);
    tr.append_message(tr_labels::STEP_DIGEST, &step_digest);
    tr.append_fields(tr_labels::COMMIT_COORDS, c_step_coords);
    let rho = tr.challenge_nonzero_field(tr_labels::CHAL_RHO);
    let dig = tr.digest32();

    #[cfg(feature = "fs-guard")]
    {
        use neo_transcript::fs_guard as guard;
        let actual = guard::take();
        guard::reset("rho/spec");
        // SPEC: explicit replay (kept in sync with intended API)
        let mut tr_s = Poseidon2Transcript::new(b"neo/ivc");
        tr_s.append_fields(tr_labels::STEP, &[F::from_u64(prev_acc.step)]);
        tr_s.append_message(tr_labels::ACC_DIGEST, &prev_acc.c_z_digest);
        tr_s.append_fields(b"acc/y", &prev_acc.y_compact);
        tr_s.append_message(tr_labels::STEP_DIGEST, &step_digest);
        tr_s.append_fields(tr_labels::COMMIT_COORDS, c_step_coords);
        let _ = tr_s.challenge_nonzero_field(tr_labels::CHAL_RHO);
        let _ = tr_s.digest32();
        let spec = guard::take();
        if let Some((i, s, a)) = guard::first_mismatch(&spec, &actual) {
            panic!(
                "FS drift in rho_from_transcript at #{}: spec(op={},label={:?},len={}) vs actual(op={},label={:?},len={})",
                i, s.op, s.label, s.len, a.op, a.label, a.len
            );
        }
    }

    (rho, dig)
}

/// Create a digest representing the current step for transcript purposes.
/// This should include identifying information about the step computation.
pub fn create_step_digest(step_data: &[F]) -> [u8; 32] {
    const RATE: usize = p2::RATE;
    
    let poseidon2 = p2::permutation();
    
    let mut st = [Goldilocks::ZERO; p2::WIDTH];
    let mut absorbed = 0;
    
    // Helper macro to avoid borrow checker issues
    macro_rules! absorb_elem {
        ($val:expr) => {
            if absorbed >= RATE {
                st = poseidon2.permute(st);
                absorbed = 0;
            }
            st[absorbed] = $val;
            absorbed += 1;
        };
    }
    
    // Domain separation
    for &byte in b"neo/ivc/step-digest/v1" {
        absorb_elem!(Goldilocks::from_u64(byte as u64));
    }
    
    // Absorb step data
    for &f in step_data {
        absorb_elem!(Goldilocks::from_u64(f.as_canonical_u64()));
    }
    
    // End-of-message marker and final permutation
    if absorbed >= RATE {
        st = poseidon2.permute(st);
        absorbed = 0;
    }
    st[absorbed] = Goldilocks::ONE;
    st = poseidon2.permute(st);
    let mut digest = [0u8; 32];
    for (i, &elem) in st[..4].iter().enumerate() {
        digest[i*8..(i+1)*8].copy_from_slice(&elem.as_canonical_u64().to_le_bytes());
    }
    
    digest
}

/// Build step transcript data for verifier challenge derivation
pub fn build_step_transcript_data(accumulator: &Accumulator, step: u64, step_x: &[F]) -> Vec<F> {
    let mut data = Vec::new();
    data.push(F::from_u64(step));
    data.extend_from_slice(&accumulator.y_compact);
    data.extend_from_slice(step_x);
    data
}

