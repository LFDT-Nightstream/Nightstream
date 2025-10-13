//! Commitment evolution and verification
//!
//! This module handles the evolution of Ajtai commitment coordinates
//! using the same folding challenge ρ as the y-vector folding.

use super::prelude::*;
use crate::shared::digest::digest_commit_coords;
use subtle::ConstantTimeEq;

/// Evolve commitment coordinates using the same folding as y vectors.
/// 
/// This is critical for end-to-end binding: the commitment must evolve
/// with the same rho used for folding y_prev + rho * y_step = y_next.
/// 
/// # Arguments
/// * `coords_prev` - Previous step's commitment coordinates
/// * `coords_step` - Current step's commitment coordinates  
/// * `rho` - Folding challenge (same as used for y folding)
/// 
/// # Returns
/// * `Ok((coords_next, digest_next))` - Updated coordinates and digest
/// * `Err(String)` - Error if coordinate lengths mismatch
pub(crate) fn evolve_commitment(
    coords_prev: &[F],
    coords_step: &[F],
    rho: F,
) -> Result<(Vec<F>, [u8; 32]), String> {
    if coords_prev.len() != coords_step.len() {
        return Err(format!(
            "commitment coordinate length mismatch: prev={}, step={}", 
            coords_prev.len(), 
            coords_step.len()
        ));
    }
    
    let mut coords_next = coords_prev.to_vec();
    for (o, cs) in coords_next.iter_mut().zip(coords_step.iter()) {
        *o = *o + rho * *cs;
    }

    // Compute digest of evolved coordinates  
    let digest = digest_commit_coords(&coords_next);
    Ok((coords_next, digest))
}

/// Verify the Ajtai commitment evolution equation:
///   c_next == c_prev + rho * c_step
/// Also checks the published digest matches the recomputed digest.
pub(crate) fn verify_commitment_evolution(
    prev_coords: &[F],
    next_coords: &[F],
    published_next_digest: &[u8; 32],
    c_step_coords: &[F],
    rho: F,
) -> bool {
    let (expected_next, expected_digest) = if prev_coords.is_empty() {
        // Base step: c_next should equal c_step (rho is irrelevant because c_prev=0)
        (c_step_coords.to_vec(), digest_commit_coords(c_step_coords))
    } else {
        match evolve_commitment(prev_coords, c_step_coords, rho) {
            Ok(pair) => pair,
            Err(_) => return false,
        }
    };

    let coords_ok = ct_eq_coords(&expected_next, next_coords);
    let digest_ok = ct_eq_bytes(&expected_digest, published_next_digest);
    if !coords_ok || !digest_ok {
        #[cfg(feature = "neo-logs")]
        {
            println!("  commit coords eq: {}", coords_ok);
            println!("  commit digest eq: {}", digest_ok);
            println!("  prev.len={}, step.len={}, next.len={}", prev_coords.len(), c_step_coords.len(), next_coords.len());
            let head = |v: &[F]| v.iter().take(4).map(|f| f.as_canonical_u64()).collect::<Vec<_>>();
            println!("  prev head: {:?}", head(prev_coords));
            println!("  step head: {:?}", head(c_step_coords));
            println!("  next head: {:?}", head(next_coords));
        }
    }
    coords_ok && digest_ok
}

/// Constant-time equality check for byte slices
#[inline]
fn ct_eq_bytes(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() { return false; }
    a.ct_eq(b).unwrap_u8() == 1
}

/// Constant-time equality check for field element coordinates
#[inline]
fn ct_eq_coords(a: &[F], b: &[F]) -> bool {
    if a.len() != b.len() { return false; }
    // Compare as little-endian u64 limbs in constant time
    let mut ok = 1u8;
    for (x, y) in a.iter().zip(b.iter()) {
        let xb = x.as_canonical_u64().to_le_bytes();
        let yb = y.as_canonical_u64().to_le_bytes();
        let eq = (&xb as &[u8]).ct_eq(&yb).unwrap_u8();
        // accumulate mismatches without branches
        ok &= eq;
    }
    ok == 1
}

/// Evolve commitment coordinates using the same folding as y vectors.
/// 
/// This is critical for end-to-end binding: the commitment must evolve
/// with the same rho used for folding y_prev + rho * y_step = y_next.
/// 
/// # Arguments
/// * `coords_prev` - Previous step's commitment coordinates
/// * `coords_step` - Current step's commitment coordinates  
/// * `rho` - Folding challenge (same as used for y folding)
/// 
/// # Returns
/// * `Ok((coords_next, digest_next))` - Updated coordinates and digest
/// * `Err(String)` - Error if coordinate lengths mismatch
pub(crate) fn serialize_accumulator_for_commitment(accumulator: &Accumulator) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut buf = Vec::new();
    buf.push(accumulator.step as u8);
    for coord in &accumulator.c_coords {
        buf.extend_from_slice(&coord.as_canonical_u64().to_le_bytes());
    }
    Ok(buf)
}
