//! Campaign profile v2: the single owner of the frozen classification shape.
//!
//! Owns: the exact parameters, memory profile, ROM, and plan-seed
//! construction that every campaign export, census, witness search, and
//! generated artifact binds to. PROFILE.md documents the freeze; the
//! `campaign_profile_v2_digests_are_frozen` drift gate pins the digests.
//!
//! Does not own: the production regime (open at the protocol level), the
//! paper-B.2 security parameterization, or any removal authority.
//!
//! The shape is the Definition-14 minimal foldable profile: `k_rho = 12` is
//! the smallest value with `(k_rho + 1) * T * (b - 1) < b^k_rho` for this
//! one-fresh-claim profile (13 * 216 * 1 = 2,808 < 4,096), because Pi_DEC
//! always re-enters `k_rho` accumulator limbs into the next Pi_RLC
//! (`engine/optimized.rs`: `let k = pp.k_rho()`). Every extra limb is a
//! full-width committed column block, so the guard minimum is also the
//! committed-column minimum.

use neo_fold_clean::frontends::nebula::f_prime::{NebulaFPrimeConstraintSourceAudit, NebulaFPrimeRelation};
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::paper::params::Params;

use crate::ExportError;

/// Definition-14 minimal foldable `k_rho` for the campaign memory profile.
pub const CAMPAIGN_K_RHO: u32 = 12;

/// Canonical plan seed of the frozen campaign profile (the mirror shape).
pub const CAMPAIGN_PLAN_SEED: [u8; 32] = [0xDA; 32];

/// Canonical preprocessing seed used by campaign captures.
pub const CAMPAIGN_PREPROCESSING_SEED: u64 = 0xDA00_0001;

/// Frozen campaign parameters (PROFILE.md, campaign profile v2).
pub fn campaign_profile_params() -> Params {
    let inner = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        1,
        neo_params::goldilocks_paper_b2::M,
        neo_params::goldilocks_paper_b2::B_BASE,
        CAMPAIGN_K_RHO,
        1,
        neo_params::goldilocks_paper_b2::EXTENSION_DEGREE,
        1,
    )
    .expect("the frozen campaign parameters satisfy every constructor guard");
    Params::test_only_from_neo_params(inner)
}

/// Frozen campaign memory profile and plan for `segment_count` segments.
pub fn campaign_profile_plan(segment_count: u64) -> Result<(Params, NebulaParams, NebulaPlan), ExportError> {
    let params = campaign_profile_params();
    let memory = NebulaParams::new(0, 0, 1, 2, segment_count)
        .map_err(|error| ExportError::new(format!("campaign memory profile: {error:?}")))?;
    let plan = NebulaPlan::new(memory, vec![7], CAMPAIGN_PLAN_SEED, params.kappa() as usize)
        .map_err(|error| ExportError::new(format!("campaign Nebula plan: {error:?}")))?;
    Ok((params, memory, plan))
}

/// Frozen campaign audit (one-segment plan, canonical seed).
pub fn campaign_profile_audit() -> Result<NebulaFPrimeConstraintSourceAudit, ExportError> {
    let (params, _, plan) = campaign_profile_plan(1)?;
    NebulaFPrimeRelation::audit_fixed_point_constraint_sources(&params, &plan)
        .map_err(|error| ExportError::new(format!("campaign audit: {error:?}")))
}
