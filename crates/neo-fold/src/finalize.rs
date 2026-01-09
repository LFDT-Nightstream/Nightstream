//! Obligation finalization (Phase 2 semantics).
//!
//! In the Neo folding pipeline, shard verification (`fold_shard_verify*`) may return deferred
//! `ShardObligations { main, val }`. These are **ME instances** whose commitments/relations have been
//! verified only up to the native verifier boundary. Finalization is the step that closes the
//! remaining soundness gap by asserting that each obligation is **openable** to a bounded witness.
//!
//! ## Closure Contract (consensus-critical)
//!
//! For each obligation `me` in the **canonical order**:
//! - `obligations.main` in vector order, then `obligations.val` in vector order
//!   (this order is exactly `ShardObligations::iter_all()`),
//!
//! a finalizer MUST enforce existence of a witness matrix `Z ∈ F^{d×m}` such that:
//! 1) **Boundedness**: `Z` is in the bounded Ajtai domain required for binding (e.g. `||Z||_∞ < b`).
//! 2) **Commitment opening**: `me.c == Commit(pp, Z)` for the canonical Ajtai public parameters `pp`.
//! 3) **Projection**: `me.X` equals the first `me.m_in` columns of `Z`.
//! 4) **ME consistency**: `me.y` and `me.y_scalars` are consistent with the same `Z` and `me.r`
//!    under the CCS matrices (see `neo_reductions::common::compute_y_from_Z_and_r` for the padded-`y`
//!    optimized-engine semantics).
//!
//! Phase 1 compression (`neo-spartan-bridge`) binds to the resulting obligations list only via
//! digests; Phase 2 adds a succinct proof that these closure checks hold.

use crate::shard_proof_types::ShardObligations;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FinalizeReport {
    pub did_finalize_main: bool,
    pub did_finalize_val: bool,
}

pub trait ObligationFinalizer<Cmt, F, K> {
    type Error;

    /// Finalize all shard obligations.
    ///
    /// Implementations must handle both `obligations.main` and `obligations.val` (Twist `r_val` lane).
    /// Prefer iterating `obligations.iter_all()` unless you intentionally treat lanes differently.
    fn finalize(&mut self, obligations: &ShardObligations<Cmt, F, K>) -> Result<FinalizeReport, Self::Error>;
}

/// Reference (non-succinct) finalizer for tests and oracles.
///
/// This recomputes Ajtai commitments and ME `y/y_scalars` from explicit witness matrices `Z` and
/// checks they match the obligation instances. This is **not** intended for production sizes.
#[derive(Clone)]
pub struct ReferenceFinalizer<L> {
    params: neo_params::NeoParams,
    ccs: std::sync::Arc<neo_ccs::CcsStructure<neo_math::F>>,
    committer: L,
    bus: Option<neo_memory::cpu::BusLayout>,
    ell_d: usize,
    main_wits: Vec<neo_ccs::Mat<neo_math::F>>,
    val_wits: Vec<neo_ccs::Mat<neo_math::F>>,
}

impl<L> ReferenceFinalizer<L>
where
    L: neo_ccs::traits::SModuleHomomorphism<neo_math::F, neo_ajtai::Commitment> + Sync,
{
    pub fn new(
        params: neo_params::NeoParams,
        ccs: std::sync::Arc<neo_ccs::CcsStructure<neo_math::F>>,
        committer: L,
        main_wits: Vec<neo_ccs::Mat<neo_math::F>>,
        val_wits: Vec<neo_ccs::Mat<neo_math::F>>,
    ) -> Result<Self, crate::PiCcsError> {
        Self::new_with_bus(params, ccs, committer, main_wits, val_wits, None)
    }

    pub fn new_with_bus(
        params: neo_params::NeoParams,
        ccs: std::sync::Arc<neo_ccs::CcsStructure<neo_math::F>>,
        committer: L,
        main_wits: Vec<neo_ccs::Mat<neo_math::F>>,
        val_wits: Vec<neo_ccs::Mat<neo_math::F>>,
        bus: Option<neo_memory::cpu::BusLayout>,
    ) -> Result<Self, crate::PiCcsError> {
        let dims =
            neo_reductions::engines::utils::build_dims_and_policy(&params, ccs.as_ref())?;
        Ok(Self {
            params,
            ccs,
            committer,
            bus,
            ell_d: dims.ell_d,
            main_wits,
            val_wits,
        })
    }

    fn check_one(
        &self,
        me: &neo_ccs::MeInstance<neo_ajtai::Commitment, neo_math::F, neo_math::K>,
        Z: &neo_ccs::Mat<neo_math::F>,
        label: &str,
    ) -> Result<(), crate::PiCcsError> {
        if Z.rows() != self.params.d as usize || Z.cols() != self.ccs.m {
            return Err(crate::PiCcsError::InvalidInput(format!(
                "{label}: Z shape mismatch (got {}x{}, want {}x{})",
                Z.rows(),
                Z.cols(),
                self.params.d,
                self.ccs.m
            )));
        }
        if me.m_in > Z.cols() {
            return Err(crate::PiCcsError::InvalidInput(format!(
                "{label}: m_in={} exceeds Z.cols()={}",
                me.m_in,
                Z.cols()
            )));
        }

        // 1) Boundedness: ||Z||_∞ < b.
        neo_ajtai::assert_range_b(Z.as_slice(), self.params.b).map_err(|e| {
            crate::PiCcsError::ProtocolError(format!("{label}: Ajtai range check failed: {e:?}"))
        })?;

        // 2) Commitment opening: c == Commit(pp, Z).
        let c_star = self.committer.commit(Z);
        if c_star != me.c {
            return Err(crate::PiCcsError::ProtocolError(format!(
                "{label}: Ajtai opening mismatch (c != Commit(pp,Z))"
            )));
        }

        // 3) Projection: X equals first m_in columns.
        if me.X.rows() != Z.rows() || me.X.cols() != me.m_in {
            return Err(crate::PiCcsError::InvalidInput(format!(
                "{label}: X shape mismatch (got {}x{}, want {}x{})",
                me.X.rows(),
                me.X.cols(),
                Z.rows(),
                me.m_in
            )));
        }
        for r in 0..Z.rows() {
            for c in 0..me.m_in {
                if me.X[(r, c)] != Z[(r, c)] {
                    return Err(crate::PiCcsError::ProtocolError(format!(
                        "{label}: projection mismatch at (r={r}, c={c})"
                    )));
                }
            }
        }

        // 4) ME consistency: recompute (y, y_scalars) under optimized-engine padded-y semantics.
        let (y_expected, y_scalars_expected) =
            neo_reductions::common::compute_y_from_Z_and_r(self.ccs.as_ref(), Z, me.r.as_slice(), self.ell_d, self.params.b);

        let core_t = self.ccs.t();
        if y_expected.len() != core_t || y_scalars_expected.len() != core_t {
            return Err(crate::PiCcsError::ProtocolError(format!(
                "{label}: internal error: compute_y_from_Z_and_r returned unexpected lengths (y.len()={}, y_scalars.len()={}, core_t={core_t})",
                y_expected.len(),
                y_scalars_expected.len(),
            )));
        }

        if me.y.len() == core_t && me.y_scalars.len() == core_t {
            if me.y != y_expected {
                return Err(crate::PiCcsError::ProtocolError(format!(
                    "{label}: y mismatch"
                )));
            }
            if me.y_scalars != y_scalars_expected {
                return Err(crate::PiCcsError::ProtocolError(format!(
                    "{label}: y_scalars mismatch"
                )));
            }
        } else if me.y.len() > core_t {
            let bus = self.bus.as_ref().ok_or_else(|| {
                crate::PiCcsError::InvalidInput(format!(
                    "{label}: ME instance has bus openings (y.len()={}, core_t={core_t}) but ReferenceFinalizer has no BusLayout",
                    me.y.len(),
                ))
            })?;
            if me.y.len() != core_t + bus.bus_cols || me.y_scalars.len() != core_t + bus.bus_cols {
                return Err(crate::PiCcsError::InvalidInput(format!(
                    "{label}: bus openings length mismatch (y.len()={}, y_scalars.len()={}, expected core_t+bus_cols={})",
                    me.y.len(),
                    me.y_scalars.len(),
                    core_t + bus.bus_cols,
                )));
            }

            let mut expected_me = neo_ccs::MeInstance::<neo_ajtai::Commitment, neo_math::F, neo_math::K> {
                c: me.c.clone(),
                X: me.X.clone(),
                r: me.r.clone(),
                y: y_expected,
                y_scalars: y_scalars_expected,
                m_in: me.m_in,
                fold_digest: [0u8; 32],
                c_step_coords: Vec::new(),
                u_offset: 0,
                u_len: 0,
            };

            crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(
                &self.params,
                bus,
                core_t,
                Z,
                &mut expected_me,
            )?;

            if me.y != expected_me.y {
                return Err(crate::PiCcsError::ProtocolError(format!(
                    "{label}: y mismatch (including bus openings)"
                )));
            }
            if me.y_scalars != expected_me.y_scalars {
                return Err(crate::PiCcsError::ProtocolError(format!(
                    "{label}: y_scalars mismatch (including bus openings)"
                )));
            }
        } else {
            return Err(crate::PiCcsError::InvalidInput(format!(
                "{label}: ME y.len()={} smaller than core_t={core_t}",
                me.y.len()
            )));
        }

        Ok(())
    }
}

impl<L> ObligationFinalizer<neo_ajtai::Commitment, neo_math::F, neo_math::K> for ReferenceFinalizer<L>
where
    L: neo_ccs::traits::SModuleHomomorphism<neo_math::F, neo_ajtai::Commitment> + Sync,
{
    type Error = crate::PiCcsError;

    fn finalize(
        &mut self,
        obligations: &ShardObligations<neo_ajtai::Commitment, neo_math::F, neo_math::K>,
    ) -> Result<FinalizeReport, Self::Error> {
        if self.main_wits.len() != obligations.main.len() {
            return Err(crate::PiCcsError::InvalidInput(format!(
                "ReferenceFinalizer: main witness count mismatch (have {}, need {})",
                self.main_wits.len(),
                obligations.main.len()
            )));
        }
        if self.val_wits.len() != obligations.val.len() {
            return Err(crate::PiCcsError::InvalidInput(format!(
                "ReferenceFinalizer: val witness count mismatch (have {}, need {})",
                self.val_wits.len(),
                obligations.val.len()
            )));
        }

        for (idx, (me, Z)) in obligations.main.iter().zip(self.main_wits.iter()).enumerate() {
            self.check_one(me, Z, &format!("obligations.main[{idx}]"))?;
        }
        for (idx, (me, Z)) in obligations.val.iter().zip(self.val_wits.iter()).enumerate() {
            self.check_one(me, Z, &format!("obligations.val[{idx}]"))?;
        }

        // One-shot guardrail: clear witnesses after use.
        self.main_wits.clear();
        self.val_wits.clear();

        Ok(FinalizeReport {
            did_finalize_main: !obligations.main.is_empty(),
            did_finalize_val: !obligations.val.is_empty(),
        })
    }
}
