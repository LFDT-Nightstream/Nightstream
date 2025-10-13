//! Digest computation utilities for NIVC

use crate::F;
use super::super::api::NivcAccumulators;
use neo_ccs::crypto::poseidon2_goldilocks as p2;
use p3_field::{PrimeField64, PrimeCharacteristicRing};

/// Compute a compact digest of all lane digests for transcript binding.
/// Returns 4 field elements (32 bytes) in Goldilocks packed form.
pub fn lanes_root_fields(acc: &NivcAccumulators) -> Vec<F> {
    // Concatenate per‑lane c_digest bytes
    let mut bytes = Vec::with_capacity(acc.lanes.len() * 32 + 16);
    for lane in &acc.lanes {
        bytes.extend_from_slice(&lane.c_digest);
    }
    // Domain separate
    bytes.extend_from_slice(b"neo/nivc/lanes_root/v1");
    let digest = p2::poseidon2_hash_packed_bytes(&bytes);
    digest.into_iter().map(|g| F::from_u64(g.as_canonical_u64())).collect()
}

