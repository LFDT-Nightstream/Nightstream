use neo_math::{KExtensions, F, K};
use neo_transcript::{Poseidon2Transcript, Transcript};

pub fn digest_fields(label: &'static [u8], fs: &[F]) -> [u8; 32] {
    let mut h = Poseidon2Transcript::new(b"memory/public_digest");
    h.append_message(b"digest/label", label);
    h.append_message(b"digest/len", &(fs.len() as u64).to_le_bytes());
    h.append_fields(b"digest/fields", fs);
    h.digest32()
}

pub fn bind_batched_claim_sums(
    tr: &mut Poseidon2Transcript,
    prefix: &'static [u8],
    claimed_sums: &[K],
    labels: &[&'static [u8]],
) {
    debug_assert_eq!(claimed_sums.len(), labels.len());
    tr.append_message(prefix, &(claimed_sums.len() as u64).to_le_bytes());
    for (i, (sum, label)) in claimed_sums.iter().zip(labels.iter()).enumerate() {
        tr.append_message(b"addr_batch/label", label);
        tr.append_message(b"addr_batch/idx", &(i as u64).to_le_bytes());
        tr.append_fields(b"addr_batch/claimed_sum", &sum.as_coeffs());
    }
}

pub fn bind_twist_val_eval_claim_sums(tr: &mut Poseidon2Transcript, claims: &[(u8, K)]) {
    tr.append_message(b"twist/val_eval/claimed_sums_len", &(claims.len() as u64).to_le_bytes());
    for (i, (kind, sum)) in claims.iter().enumerate() {
        tr.append_message(b"twist/val_eval/claim_idx", &(i as u64).to_le_bytes());
        tr.append_message(b"twist/val_eval/claim_kind", &[*kind]);
        tr.append_fields(b"twist/val_eval/claimed_sum", &sum.as_coeffs());
    }
}

pub fn bind_batched_dynamic_claims<L: AsRef<[u8]>>(
    tr: &mut Poseidon2Transcript,
    claimed_sums: &[K],
    labels: &[L],
    degree_bounds: &[usize],
    claim_is_dynamic: &[bool],
) {
    debug_assert_eq!(claimed_sums.len(), labels.len());
    debug_assert_eq!(claimed_sums.len(), degree_bounds.len());
    debug_assert_eq!(claimed_sums.len(), claim_is_dynamic.len());

    tr.append_message(b"batched/dynamic_bind/len", &(claimed_sums.len() as u64).to_le_bytes());
    for (idx, (((sum, label), &deg), &dyn_ok)) in claimed_sums
        .iter()
        .zip(labels.iter())
        .zip(degree_bounds.iter())
        .zip(claim_is_dynamic.iter())
        .enumerate()
    {
        tr.append_message(b"batched/dynamic_bind/claim_label", label.as_ref());
        tr.append_message(b"batched/dynamic_bind/claim_idx", &(idx as u64).to_le_bytes());
        tr.append_message(b"batched/dynamic_bind/degree_bound", &(deg as u64).to_le_bytes());
        tr.append_message(b"batched/dynamic_bind/is_dynamic", &[dyn_ok as u8]);
        if !dyn_ok {
            continue;
        }
        tr.append_fields(b"batched/dynamic_bind/claimed_sum", &sum.as_coeffs());
    }
}
