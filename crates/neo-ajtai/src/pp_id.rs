//! Ajtai public-parameter identity helpers.
//!
//! This defines a stable, domain-separated digest that uniquely identifies the Ajtai PP used for
//! commitments in Neo. It is used by the Spartan bridge public statement (`pp_id_digest`) and by
//! closure-proof backends to ensure "opening correctness" semantics are well-defined.

#![forbid(unsafe_code)]

/// Compute the canonical Ajtai PP identity digest (`pp_id/v1`).
///
/// This MUST match the digest computed by `neo-spartan-bridge` and any closure-proof backends.
pub fn compute_pp_id_digest_v1(d: usize, m: usize, kappa: usize, seed: [u8; 32]) -> [u8; 32] {
    use blake3::Hasher;

    let mut h = Hasher::new();
    h.update(b"neo/spartan-bridge/pp_id/v1");
    h.update(&(d as u64).to_le_bytes());
    h.update(&(m as u64).to_le_bytes());
    h.update(&(kappa as u64).to_le_bytes());
    h.update(&seed);
    *h.finalize().as_bytes()
}

