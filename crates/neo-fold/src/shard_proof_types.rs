use crate::PiCcsProof;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{matrix::Mat, MeInstance};
use neo_math::{F, K};
use neo_memory::output_check::OutputBindingProof;

pub type TwistProofK = neo_memory::twist::TwistProof<K>;
pub type ShoutProofK = neo_memory::shout::ShoutProof<K>;

/// One fold step’s artifacts (Π_CCS → Π_RLC → Π_DEC).
#[derive(Clone, Debug)]
pub struct FoldStep {
    /// Π_CCS outputs (k ME(b,L) instances)
    pub ccs_out: Vec<MeInstance<Cmt, F, K>>,
    /// Π_CCS proof (engine-agnostic re-export)
    pub ccs_proof: PiCcsProof,
    /// RLC mixing matrices ρ_i ∈ S ⊆ F^{D×D}
    pub rlc_rhos: Vec<Mat<F>>,
    /// The combined parent after RLC: ME(B,L) with B=b^k
    pub rlc_parent: MeInstance<Cmt, F, K>,
    /// DEC children: k ME(b,L) after decomposition of the parent
    pub dec_children: Vec<MeInstance<Cmt, F, K>>,
}

#[derive(Clone, Debug)]
#[must_use]
pub struct ShardObligations<C, FF, KK> {
    pub main: Vec<MeInstance<C, FF, KK>>,
    pub val: Vec<MeInstance<C, FF, KK>>,
}

impl<C, FF, KK> ShardObligations<C, FF, KK> {
    pub fn all_len(&self) -> usize {
        self.main.len() + self.val.len()
    }

    pub fn iter_all(&self) -> impl Iterator<Item = &MeInstance<C, FF, KK>> {
        self.main.iter().chain(self.val.iter())
    }

    pub fn require_all_finalized(
        &self,
        did_finalize_main: bool,
        did_finalize_val: bool,
    ) -> Result<(), crate::PiCcsError> {
        if !self.main.is_empty() && !did_finalize_main {
            return Err(crate::PiCcsError::ProtocolError(
                "finalizer did not process main obligations".into(),
            ));
        }
        if !self.val.is_empty() && !did_finalize_val {
            return Err(crate::PiCcsError::ProtocolError(
                "finalizer did not process val-lane obligations".into(),
            ));
        }
        Ok(())
    }

    pub fn split(self) -> (Vec<MeInstance<C, FF, KK>>, Vec<MeInstance<C, FF, KK>>) {
        (self.main, self.val)
    }
}

#[derive(Clone, Debug)]
#[must_use]
pub struct ShardFoldOutputs<C, FF, KK> {
    pub obligations: ShardObligations<C, FF, KK>,
}

#[derive(Clone, Debug)]
pub struct ShardFoldWitnesses<FF> {
    /// Witnesses for `ShardFoldOutputs::obligations.main` (one per ME instance).
    pub final_main_wits: Vec<Mat<FF>>,
    /// Witnesses for `ShardFoldOutputs::obligations.val` (one per ME instance).
    pub val_lane_wits: Vec<Mat<FF>>,
}

#[derive(Clone, Debug)]
pub enum MemOrLutProof {
    Twist(TwistProofK),
    Shout(ShoutProofK),
}

#[derive(Clone, Debug)]
pub struct MemSidecarProof<C, FF, KK> {
    /// CPU ME claims evaluated at `r_val` (Twist val-eval terminal point).
    ///
    /// In shared-CPU-bus-only mode, Twist reads val-lane openings from these claims
    /// (current step + optional previous step for rollover).
    pub cpu_me_claims_val: Vec<MeInstance<C, FF, KK>>,
    pub proofs: Vec<MemOrLutProof>,
}

/// Proof for the Route A shared-challenge batched sum-check (time/row rounds).
///
/// This batches CCS (row/time rounds) with Twist/Shout time-domain oracles so all
/// protocols share the same transcript-derived `r` (enabling Π_RLC folding).
#[derive(Clone, Debug)]
pub struct BatchedTimeProof {
    /// Claimed sums per participating oracle (in the same order as `round_polys`).
    pub claimed_sums: Vec<K>,
    /// Degree bounds per participating oracle.
    pub degree_bounds: Vec<usize>,
    /// Domain-separation labels per participating oracle.
    pub labels: Vec<&'static [u8]>,
    /// Per-claim sum-check messages: `round_polys[claim][round] = coeffs`.
    pub round_polys: Vec<Vec<Vec<K>>>,
}

/// Proof data for a standalone Π_RLC → Π_DEC lane (no Π_CCS).
#[derive(Clone, Debug)]
pub struct RlcDecProof {
    /// RLC mixing matrices ρ_i ∈ S ⊆ F^{D×D}
    pub rlc_rhos: Vec<Mat<F>>,
    /// The combined parent after RLC: ME(B,L) with B=b^k
    pub rlc_parent: MeInstance<Cmt, F, K>,
    /// DEC children: k ME(b,L) after decomposition of the parent
    pub dec_children: Vec<MeInstance<Cmt, F, K>>,
}

#[derive(Clone, Debug)]
pub struct StepProof {
    pub fold: FoldStep,
    pub mem: MemSidecarProof<Cmt, F, K>,
    pub batched_time: BatchedTimeProof,
    /// Optional second folding lane for Twist val-eval ME claims at `r_val`.
    pub val_fold: Option<RlcDecProof>,
}

#[derive(Clone, Debug)]
pub struct ShardProof {
    pub steps: Vec<StepProof>,
    /// Optional output binding proof (proves final memory matches claimed outputs).
    /// Twist linkage is proven as an extra Route-A batched-time claim on the final step.
    pub output_proof: Option<OutputBindingProof>,
}

impl ShardProof {
    pub fn compute_final_obligations(&self, acc_init: &[MeInstance<Cmt, F, K>]) -> ShardObligations<Cmt, F, K> {
        self.compute_fold_outputs(acc_init).obligations
    }

    /// Returns the final main accumulator only (does not include Twist `r_val` obligations).
    pub fn compute_final_main_children(&self, acc_init: &[MeInstance<Cmt, F, K>]) -> Vec<MeInstance<Cmt, F, K>> {
        self.compute_fold_outputs(acc_init).obligations.main
    }

    /// Legacy alias for CCS-only codepaths: compute the final main accumulator.
    pub fn compute_final_outputs(&self, acc_init: &[MeInstance<Cmt, F, K>]) -> Vec<MeInstance<Cmt, F, K>> {
        self.compute_final_main_children(acc_init)
    }

    #[deprecated(
        note = "Use compute_fold_outputs().obligations (includes val lane) or compute_final_obligations(). \
If you truly want main only, call compute_final_main_children()."
    )]
    pub fn compute_final_children(&self, acc_init: &[MeInstance<Cmt, F, K>]) -> Vec<MeInstance<Cmt, F, K>> {
        let ShardObligations { main, val } = self.compute_fold_outputs(acc_init).obligations;
        main.into_iter().chain(val).collect()
    }

    pub fn compute_fold_outputs(&self, acc_init: &[MeInstance<Cmt, F, K>]) -> ShardFoldOutputs<Cmt, F, K> {
        let main = if self.steps.is_empty() {
            acc_init.to_vec()
        } else {
            self.steps
                .last()
                .expect("non-empty")
                .fold
                .dec_children
                .clone()
        };

        let mut val = Vec::new();
        for step in &self.steps {
            if let Some(val_fold) = &step.val_fold {
                val.extend_from_slice(&val_fold.dec_children);
            }
        }

        ShardFoldOutputs {
            obligations: ShardObligations { main, val },
        }
    }
}
