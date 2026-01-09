//! Deterministic RNG seed derivation for the WHIR (P3) closure backends.

#![forbid(unsafe_code)]

use super::ClosureStatementV1;

pub(super) fn derive_seed_v1(
    label: &[u8],
    stmt: &ClosureStatementV1,
    commitment_root_u64: Option<&[u64]>,
) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(b"neo/closure-proof/whir-p3/seed/v1");
    h.update(label);
    h.update(&stmt.context_digest);
    h.update(&stmt.pp_id_digest);
    h.update(&stmt.obligations_digest);
    if let Some(root) = commitment_root_u64 {
        for &x in root {
            h.update(&x.to_le_bytes());
        }
    }
    *h.finalize().as_bytes()
}

pub(super) fn fixed_seed(label: &[u8]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(b"neo/closure-proof/whir-p3/fixed-seed/v1");
    h.update(label);
    *h.finalize().as_bytes()
}

