//! Ajtai-module setup helpers for direct-CCS tests/demos.
//!
//! This file is **not** the protocol setup. The audited Ajtai SRS is a
//! verifier-owned global configuration; see `frontends/direct_ccs/mod.rs`'s
//! `preprocess` for the real entry. The helpers here exist so tests and
//! examples don't have to wire `ajtai_setup` themselves every time.
//! **Do not use in production.**

use neo_ajtai::{has_global_pp_for_dims, set_global_pp_seeded, AjtaiSModule};
use neo_math::D;

use crate::paper::params::Params;
use crate::paper::relations::Structure;

/// Derive an `AjtaiSModule` deterministically from `seed`.
///
/// Backed by `neo-ajtai`'s process-global SRS cache: the first call for
/// a given `(D, cols)` shape installs a fresh PP from the seed; subsequent
/// calls for the same shape reuse it. This means seeded helpers across
/// tests share the same SRS as long as the shape matches — which is what
/// you want for stable test artefacts.
pub fn setup_seeded(params: &Params, structure: &Structure, seed: u64) -> AjtaiSModule {
    let cols = structure.m.div_ceil(D);
    if !has_global_pp_for_dims(D, cols) {
        let mut seed_bytes = [0u8; 32];
        seed_bytes[..8].copy_from_slice(&seed.to_le_bytes());
        match set_global_pp_seeded(D, params.kappa() as usize, cols, seed_bytes) {
            Ok(()) => {}
            Err(_err) if has_global_pp_for_dims(D, cols) => {
                // A concurrent test installed the PP first; fine.
            }
            Err(err) => panic!("Ajtai global setup: {err}"),
        }
    }
    AjtaiSModule::from_global_for_dims(D, cols).expect("Ajtai module for direct-CCS structure")
}
