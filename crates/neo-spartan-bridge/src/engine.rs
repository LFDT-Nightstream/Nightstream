//! Experimental Engine configuration for pluggable PCS backends
//!
//! **STATUS**: Parked / Experimental
//!
//! This module contains experimental hooks for Z-polynomial layout strategies.
//! It is NOT currently wired into Spartan2 and should be considered a design sketch.
//!
//! For actual Spartan2 integration, rely on `spartan2::traits::Engine` directly.
//! This module may be refactored or removed once the integration is complete.
//!
//! To use this experimental code, enable the `experimental-engine` feature.

#![cfg(feature = "experimental-engine")]

use ff::PrimeField;
use p3_goldilocks::Goldilocks;
use std::fmt::Debug;

/// Strategy for building the Z polynomial in Spartan.
///
/// Different PCS schemes may require different Z-polynomial layouts.
/// For example, Hash-MLE uses a specific interleaving pattern, while
/// KZG might use a different structure.
///
/// NOTE: This is experimental and not yet wired into Spartan2.
/// For now, rely on Spartan2's existing engine implementations.
pub trait ZPolyLayout<E: BridgeEngine>: Clone + Debug + Send + Sync {
    /// Number of rounds for the y-poly in the sumcheck
    fn num_rounds_y(num_vars: usize) -> usize;

    /// Build the Z polynomial from witness and instance
    fn build_z(witness: &[E::Scalar], instance: &[E::Scalar]) -> Vec<E::Scalar>;

    /// Evaluate Z at a point given evaluation of W and gate bit
    fn eval_z(gate: E::Scalar, eval_w: E::Scalar, eval_x: E::Scalar) -> E::Scalar;
}

/// BridgeEngine: Configuration for Z-polynomial layout (experimental)
///
/// This trait is separate from spartan2::traits::Engine to avoid naming conflicts.
/// It may be merged or refactored once the Z-layout pattern is finalized.
pub trait BridgeEngine: Clone + Copy + Debug + Send + Sync + Sized + Eq + PartialEq {
    /// Base field (may equal Scalar for some curves)
    type Base: PrimeField + Send + Sync;

    /// Scalar field (the field we do R1CS over)
    type Scalar: PrimeField + Send + Sync;

    /// Z-polynomial layout strategy
    type ZLayout: ZPolyLayout<Self>;

    /// Name of this engine (for logging/debugging)
    fn name() -> &'static str;
}

/// Hash-MLE Z-layout strategy for Goldilocks (experimental)
#[derive(Clone, Debug)]
pub struct HashMleZLayout;

impl ZPolyLayout<HashMleEngine> for HashMleZLayout {
    fn num_rounds_y(num_vars: usize) -> usize {
        // Hash-MLE specific: log2(num_vars) + 1
        if num_vars == 0 {
            0
        } else {
            num_vars.next_power_of_two().trailing_zeros() as usize + 1
        }
    }

    fn build_z(witness: &[GoldilocksScalar], instance: &[GoldilocksScalar]) -> Vec<GoldilocksScalar> {
        // Hash-MLE layout: [instance, witness]
        let mut z = Vec::with_capacity(instance.len() + witness.len());
        z.extend_from_slice(instance);
        z.extend_from_slice(witness);
        z
    }

    fn eval_z(gate: GoldilocksScalar, eval_w: GoldilocksScalar, eval_x: GoldilocksScalar) -> GoldilocksScalar {
        // Hash-MLE eval: gate * eval_w + (1 - gate) * eval_x
        gate * eval_w + (GoldilocksScalar::ONE - gate) * eval_x
    }
}

/// Wrapper for Goldilocks to implement PrimeField from ff crate
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GoldilocksScalar(Goldilocks);

impl From<Goldilocks> for GoldilocksScalar {
    fn from(g: Goldilocks) -> Self {
        GoldilocksScalar(g)
    }
}

impl From<GoldilocksScalar> for Goldilocks {
    fn from(g: GoldilocksScalar) -> Self {
        g.0
    }
}

impl From<u64> for GoldilocksScalar {
    fn from(v: u64) -> Self {
        use p3_field::AbstractField;
        GoldilocksScalar(Goldilocks::from_canonical_u64(v))
    }
}

// Note: Full PrimeField implementation would go here, but for now we'll
// use this as a placeholder and rely on the actual Spartan2 integration

/// Hash-MLE engine using Goldilocks field (experimental)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HashMleEngine;

impl BridgeEngine for HashMleEngine {
    type Base = GoldilocksScalar;
    type Scalar = GoldilocksScalar;
    type ZLayout = HashMleZLayout;

    fn name() -> &'static str {
        "HashMLE-Goldilocks"
    }
}
