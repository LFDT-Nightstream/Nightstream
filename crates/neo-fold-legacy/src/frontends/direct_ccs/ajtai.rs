//! Ajtai-module setup helpers for direct-CCS tests/demos.
//!
//! This file is **not** the protocol setup. The audited Ajtai SRS is a
//! verifier-owned global configuration; see `frontends/direct_ccs/mod.rs`'s
//! `preprocess` for the real entry. The helpers here exist so tests and
//! examples don't have to wire `ajtai_setup` themselves every time.
//! **Do not use in production.**

use neo_ajtai::AjtaiSModule;
use neo_math::D;

use crate::paper::params::Params;
use crate::paper::relations::Structure;

/// Derive an `AjtaiSModule` deterministically from `seed`.
///
/// The returned module owns the setup descriptor. It does not use the
/// process-global SRS registry, so equal-shaped tests with different seeds
/// cannot change each other's verifier setup.
pub fn setup_seeded(params: &Params, structure: &Structure, seed: u64) -> AjtaiSModule {
    let cols = structure.m.div_ceil(D);
    let mut seed_bytes = [0u8; 32];
    seed_bytes[..8].copy_from_slice(&seed.to_le_bytes());
    AjtaiSModule::from_seeded(seed_bytes, D, params.kappa() as usize, cols)
        .expect("Ajtai module for direct-CCS structure")
}
