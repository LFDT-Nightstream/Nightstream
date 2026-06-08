//! Lifecycle wrappers for encoded-F' chains.
//!
//! [`prove_encoded_steps`] is the production entry point for callers
//! that already own an [`EncodedFPrimeStep`] sequence and just
//! need it folded through `lifecycle::prove`.
//!
//! [`FibonacciChainBuilder`] owns the prover-side chain assembly
//! (compile → fold → derive next per-step fold authority → compile
//! recursive → extend) so callers do not have to thread
//! `fold_for_step` manually. Mirrors
//! [`neo_fold_clean::frontends::r1cs_f_prime::R1csChainBuilder`].

use super::compiler::{
    compile_fibonacci_step, start_fibonacci_chain, FibonacciAppStepInput, FibonacciChainState, FibonacciCompiledStep,
    FibonacciCompilerContext, FibonacciFoldForStep,
};
use super::instance::build_instance;
use super::{Error, FibonacciFPrimePreprocessing};
use neo_fold_clean::frontends::f_prime::encoder::EncodedFPrimeStep;
use neo_fold_clean::lifecycle::{Uncompressed, UncompressedAudit};
use neo_fold_clean::paper::construction2::{FoldProof, ProofState};
use neo_fold_clean::paper::digest::digest32_as_fields;
use neo_fold_clean::paper::relations::CcsInstance;

/// Fold a sequence of encoded F' steps through `lifecycle::prove`,
/// one step per batch.
///
/// Each step is converted into a `CcsInstance` via [`build_instance`],
/// which enforces that every step's CCS structure matches the
/// preprocessing's structure (so the chain folds a homogeneous
/// relation).
pub fn prove_encoded_steps(
    prep: &FibonacciFPrimePreprocessing,
    steps: &[EncodedFPrimeStep],
) -> Result<UncompressedAudit, Error> {
    let mut batches: Vec<Vec<CcsInstance>> = Vec::with_capacity(steps.len());
    for step in steps {
        batches.push(vec![build_instance(prep, step)?]);
    }
    Ok(neo_fold_clean::lifecycle::prove(&prep.prep, batches)?)
}

/// Thin prover-side wrapper for one fixed-shape Fibonacci F' chain.
///
/// Owns the otherwise easy-to-mis-thread sequence:
///
/// 1. compile the base Fibonacci step,
/// 2. fold the resulting encoded F' instance through `lifecycle::prove`,
/// 3. derive the next per-step fold authority from the current audit state,
/// 4. compile each recursive step, and
/// 5. extend the audit with the real compiled instance.
///
/// One builder is tied to one [`FibonacciFPrimePreprocessing`] value,
/// and therefore one verifier-owned F' structure (one `pc`).
pub struct FibonacciChainBuilder<'a> {
    prep: &'a FibonacciFPrimePreprocessing,
    ctx: FibonacciCompilerContext,
    audit: Option<UncompressedAudit>,
    latest_instance: Option<CcsInstance>,
}

impl<'a> FibonacciChainBuilder<'a> {
    /// Start a fresh fixed-shape Fibonacci F' chain.
    pub fn new(prep: &'a FibonacciFPrimePreprocessing) -> Result<Self, Error> {
        Ok(Self {
            prep,
            ctx: start_fibonacci_chain(prep)?,
            audit: None,
            latest_instance: None,
        })
    }

    /// Append one Fibonacci app step to the chain.
    ///
    /// The first call compiles the base branch. Later calls derive the
    /// required recursive fold authority from the current audit and feed
    /// it into the compiler before extending the audit with the newly
    /// compiled instance.
    pub fn append_step(&mut self, input: FibonacciAppStepInput) -> Result<FibonacciCompiledStep, Error> {
        if self.audit.is_some() {
            self.prepare_next_fold()?;
        }

        let compiled = compile_fibonacci_step(self.prep, &mut self.ctx, input)?;
        let instance = build_instance(self.prep, &compiled.encoded)?;

        self.audit = Some(match self.audit.take() {
            Some(audit) => neo_fold_clean::lifecycle::extend(&self.prep.prep, audit, vec![instance.clone()])?,
            None => neo_fold_clean::lifecycle::prove(&self.prep.prep, [vec![instance.clone()]])?,
        });
        self.latest_instance = Some(instance);
        Ok(compiled)
    }

    /// Current pre-finalize audit, if at least one step has been appended.
    pub fn audit(&self) -> Option<&UncompressedAudit> {
        self.audit.as_ref()
    }

    /// Current compiler context. Exposed for diagnostics and tests.
    pub fn context(&self) -> &FibonacciCompilerContext {
        &self.ctx
    }

    /// Consume the builder and return the pre-finalize audit.
    pub fn into_audit(self) -> Result<UncompressedAudit, Error> {
        self.audit.ok_or(Error::ChainEmpty)
    }

    /// Finalize while dropping the audit trail.
    ///
    /// This output is suitable for terminal-only
    /// `lifecycle::verify_uncompressed` only for single-chunk F' chains.
    /// Multi-chunk F' chains need [`Self::finish_with_audit`] until the
    /// compressed decider proves the recursive F'/NIFS.V induction.
    pub fn finish(self) -> Result<Uncompressed, Error> {
        let prep = self.prep;
        let audit = self.into_audit()?;
        Ok(neo_fold_clean::lifecycle::finish_uncompressed(&prep.prep, audit)?)
    }

    /// Finalize while keeping the audit trail; useful for diagnostics and
    /// chain-replay tests.
    pub fn finish_with_audit(self) -> Result<UncompressedAudit, Error> {
        let prep = self.prep;
        let audit = self.into_audit()?;
        Ok(neo_fold_clean::lifecycle::finish_uncompressed_with_audit(
            &prep.prep, audit,
        )?)
    }

    fn prepare_next_fold(&mut self) -> Result<(), Error> {
        let audit = self.audit.as_ref().ok_or(Error::ChainEmpty)?;
        let latest_instance = self.latest_instance.as_ref().ok_or(Error::ChainEmpty)?;
        let pre_state = audit.proof.state.clone();

        let (pre_running, latest) = match &pre_state.proof {
            ProofState::Active { running, latest } => (running.clone(), latest.clone()),
            _ => return Err(Error::ChainExpectedActiveState),
        };

        // Derive the per-step NIFS proof for the fold the next F' step
        // must verify. The latest instance is the current lifecycle
        // `latest`, so extending a cloned audit with it produces exactly
        // the fold authority for the next recursive compile without
        // contaminating the real audit. The real audit is extended only
        // after the recursive step has been compiled below.
        let derived = neo_fold_clean::lifecycle::extend(&self.prep.prep, audit.clone(), vec![latest_instance.clone()])?;
        let fold = match &derived.steps.last().expect("extend appended one step").fold {
            FoldProof::Recursive(p) => p.clone(),
            FoldProof::NoFold => return Err(Error::ChainExpectedActiveState),
        };
        let post_running = match &derived.proof.state.proof {
            ProofState::Active { running, .. } => running.clone(),
            _ => return Err(Error::ChainExpectedActiveState),
        };

        self.ctx.chain_state = FibonacciChainState {
            chunk_count: pre_state.chunk_count,
            step_count: pre_state.step_count,
            z_i: digest32_as_fields(pre_state.z_i),
            semantic_state_digest: digest32_as_fields(pre_state.semantic_state_digest),
            acc_digest: digest32_as_fields(pre_state.acc_digest),
            public_trace: digest32_as_fields(pre_state.public_trace),
        };
        self.ctx.fold_for_step = Some(FibonacciFoldForStep {
            pre_running,
            latest,
            proof: fold,
            post_running,
        });

        Ok(())
    }
}
