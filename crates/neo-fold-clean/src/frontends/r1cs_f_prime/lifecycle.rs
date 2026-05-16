//! Lifecycle helpers for R1CS-encoded-F' chains.

use crate::frontends::f_prime_shell::encoder::EncodedFPrimeStep;
use crate::frontends::r1cs_f_prime::compiler::{
    compile_step, start_chain, R1csCompiledStep, R1csCompilerContext, R1csFPrimeStepInput, R1csFoldForStep,
};
use crate::frontends::r1cs_f_prime::instance::build_instance;
use crate::frontends::r1cs_f_prime::{Error, R1csFPrimePreprocessing};
use crate::lifecycle::{Uncompressed, UncompressedAudit};
use crate::paper::construction2::ProofState;
use crate::paper::digest::digest32_as_fields;
use crate::paper::relations::CcsInstance;

/// Fold a sequence of encoded R1CS-F' steps through `lifecycle::prove`,
/// one step per batch.
pub fn prove_encoded_steps(
    prep: &R1csFPrimePreprocessing,
    steps: &[EncodedFPrimeStep],
) -> Result<UncompressedAudit, Error> {
    let mut batches: Vec<Vec<CcsInstance>> = Vec::with_capacity(steps.len());
    for step in steps {
        batches.push(vec![build_instance(prep, step)?]);
    }
    Ok(crate::lifecycle::prove(&prep.prep, batches)?)
}

/// Thin prover-side wrapper for one fixed-shape R1CS-F' chain.
///
/// Owns the otherwise easy-to-mis-thread sequence:
///
/// 1. compile the base R1CS assignment,
/// 2. fold the resulting encoded F' instance through `lifecycle::prove`,
/// 3. derive the next per-step fold authority from the current audit state,
/// 4. compile each recursive assignment, and
/// 5. extend the audit with the real compiled instance.
///
/// It does not support heterogeneous circuits. One builder is tied to
/// one [`R1csFPrimePreprocessing`] value, and therefore one verifier-owned
/// R1CS shape / F' structure.
pub struct R1csChainBuilder<'a> {
    prep: &'a R1csFPrimePreprocessing,
    ctx: R1csCompilerContext,
    audit: Option<UncompressedAudit>,
    latest_instance: Option<CcsInstance>,
}

impl<'a> R1csChainBuilder<'a> {
    /// Start a fresh fixed-shape R1CS-F' chain.
    pub fn new(prep: &'a R1csFPrimePreprocessing) -> Result<Self, Error> {
        Ok(Self {
            prep,
            ctx: start_chain(prep)?,
            audit: None,
            latest_instance: None,
        })
    }

    /// Append one satisfying R1CS assignment to the chain.
    ///
    /// The first call compiles the base branch. Later calls derive the
    /// required recursive fold authority from the current audit and feed
    /// it into the compiler before extending the audit with the newly
    /// compiled instance.
    pub fn append_assignment(&mut self, assignment: Vec<neo_math::F>) -> Result<R1csCompiledStep, Error> {
        self.append_step(R1csFPrimeStepInput { assignment })
    }

    /// Append one explicit R1CS-F' compiler input to the chain.
    pub fn append_step(&mut self, input: R1csFPrimeStepInput) -> Result<R1csCompiledStep, Error> {
        if self.audit.is_some() {
            self.prepare_next_fold()?;
        }

        let compiled = compile_step(self.prep, &mut self.ctx, input)?;
        let instance = build_instance(self.prep, &compiled.encoded)?;

        self.audit = Some(match self.audit.take() {
            Some(audit) => crate::lifecycle::extend(&self.prep.prep, audit, vec![instance.clone()])?,
            None => crate::lifecycle::prove(&self.prep.prep, [vec![instance.clone()]])?,
        });
        self.latest_instance = Some(instance);
        Ok(compiled)
    }

    /// Current pre-finalize audit, if at least one step has been appended.
    pub fn audit(&self) -> Option<&UncompressedAudit> {
        self.audit.as_ref()
    }

    /// Current compiler context. Exposed for diagnostics and tests.
    pub fn context(&self) -> &R1csCompilerContext {
        &self.ctx
    }

    /// Consume the builder and return the pre-finalize audit.
    pub fn into_audit(self) -> Result<UncompressedAudit, Error> {
        self.audit.ok_or(Error::ChainEmpty)
    }

    /// Finalize while dropping the audit trail; output is suitable for
    /// `lifecycle::verify_uncompressed`.
    pub fn finish(self) -> Result<Uncompressed, Error> {
        let prep = self.prep;
        let audit = self.into_audit()?;
        Ok(crate::lifecycle::finish_uncompressed(&prep.prep, audit)?)
    }

    /// Finalize while keeping the audit trail; useful for diagnostics and
    /// chain-replay tests.
    pub fn finish_with_audit(self) -> Result<UncompressedAudit, Error> {
        let prep = self.prep;
        let audit = self.into_audit()?;
        Ok(crate::lifecycle::finish_uncompressed_with_audit(&prep.prep, audit)?)
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
        let derived = crate::lifecycle::extend(&self.prep.prep, audit.clone(), vec![latest_instance.clone()])?;
        let fold = match &derived.steps.last().expect("extend appended one step").fold {
            crate::paper::construction2::FoldProof::Recursive(p) => p.clone(),
            crate::paper::construction2::FoldProof::NoFold => return Err(Error::ChainExpectedActiveState),
        };
        let post_running = match &derived.proof.state.proof {
            ProofState::Active { running, .. } => running.clone(),
            _ => return Err(Error::ChainExpectedActiveState),
        };

        self.ctx.chain_state = crate::frontends::r1cs_f_prime::R1csChainState {
            chunk_count: pre_state.chunk_count,
            step_count: pre_state.step_count,
            z_i: digest32_as_fields(pre_state.z_i),
            acc_digest: digest32_as_fields(pre_state.acc_digest),
            public_trace: digest32_as_fields(pre_state.public_trace),
        };
        self.ctx.fold_for_step = Some(R1csFoldForStep {
            pre_running,
            latest,
            proof: fold,
            post_running,
        });

        Ok(())
    }
}
