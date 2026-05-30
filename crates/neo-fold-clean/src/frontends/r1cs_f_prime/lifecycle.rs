//! Lifecycle helpers for R1CS-encoded-F' chains.

use crate::frontends::f_prime::encoder::EncodedFPrimeStep;
use crate::frontends::r1cs_f_prime::compiler::{
    compile_chunk, compile_step, semantic_state_digests_for_inputs, start_chain, R1csCompiledStep, R1csCompilerContext,
    R1csCompilerError, R1csFPrimeStepInput, R1csFoldForStep,
};
use crate::frontends::r1cs_f_prime::instance::build_instance;
use crate::frontends::r1cs_f_prime::{Error, R1csFPrimePreprocessing};
use crate::lifecycle::{Uncompressed, UncompressedAudit};
use crate::paper::construction2::{LatestInstance, ProofState};
use crate::paper::digest::{digest32_as_fields, digest_fields_as_digest32};
use crate::paper::relations::{CcsClaim, CcsInstance};

/// Fold a sequence of encoded R1CS-F' steps through `lifecycle::prove`,
/// one step per batch.
///
/// **Stateless only.** This entrypoint uses the generic stateless
/// `lifecycle::prove` path, which advances `state.semantic_state_digest`
/// to `new_acc_digest` at every step. That would silently disagree
/// with the encoded image's `state_out.semantic_state_digest_lane`
/// (which encodes `H(state_out_app_vars)`) — the `x_out` chain check
/// would later fail with a confusing diagnostic. Reject up front so a
/// stateful caller is forced toward [`R1csChainBuilder`], which threads
/// the per-step semantic digests correctly.
pub fn prove_encoded_steps(
    prep: &R1csFPrimePreprocessing,
    steps: &[EncodedFPrimeStep],
) -> Result<UncompressedAudit, Error> {
    use crate::paper::construction2::SemanticStateMode;
    if matches!(prep.prep.semantic_state_mode(), SemanticStateMode::Stateful) {
        return Err(Error::ProveEncodedStepsStatefulUnsupported);
    }
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
/// 1. compile a chunk of K satisfying R1CS assignments (all same shape,
///    same prior chain state),
/// 2. build K [`CcsInstance`]s,
/// 3. fold them through `lifecycle::prove`/`extend` as **one** SuperNeo
///    chunk (the new latest is the K-sized batch),
/// 4. derive the next-step fold authority from the current audit state
///    (the next chunk's NIFS proof transcript binds K),
/// 5. compile the next chunk.
///
/// `K=1` callers use [`Self::append_assignment`] / [`Self::append_step`];
/// `K≥1` callers use [`Self::append_assignments`] /
/// [`Self::append_chunk`]. The chain advances **once per chunk**
/// (`chunk_count += 1`, `step_count += K`) regardless of `K`.
///
/// It does not support heterogeneous circuits. One builder is tied to
/// one [`R1csFPrimePreprocessing`] value, and therefore one verifier-owned
/// R1CS shape / F' structure.
pub struct R1csChainBuilder<'a> {
    prep: &'a R1csFPrimePreprocessing,
    ctx: R1csCompilerContext,
    audit: Option<UncompressedAudit>,
    /// Whole previous chunk's instances. Sized K of the most recent
    /// `append_*` call; empty before the first append. The next
    /// `prepare_next_fold` reuses this batch as the fold's `latest`
    /// (mirroring native `state.proof.latest.instances`).
    latest_batch: Vec<CcsInstance>,
    /// The post-fold audit produced inside [`Self::prepare_next_fold`].
    /// It is byte-identical to what `lifecycle::extend` would return for
    /// this chunk **except** its deposited `latest` / `public_batches`
    /// hold the placeholder batch. [`Self::append_chunk`] swaps the real
    /// compiled instances in rather than re-running the (expensive) NIFS
    /// prove — see the comment there for why the swap is exact.
    pending_audit: Option<UncompressedAudit>,
}

impl<'a> R1csChainBuilder<'a> {
    /// Start a fresh fixed-shape R1CS-F' chain.
    pub fn new(prep: &'a R1csFPrimePreprocessing) -> Result<Self, Error> {
        Ok(Self {
            prep,
            ctx: start_chain(prep)?,
            audit: None,
            latest_batch: Vec::new(),
            pending_audit: None,
        })
    }

    /// Append one satisfying R1CS assignment to the chain as a K=1
    /// chunk. Convenience wrapper around [`Self::append_assignments`].
    pub fn append_assignment(&mut self, assignment: Vec<neo_math::F>) -> Result<R1csCompiledStep, Error> {
        let mut chunk = self.append_assignments(vec![assignment])?;
        debug_assert_eq!(chunk.len(), 1);
        Ok(chunk.pop().expect("K=1 append returns 1 step"))
    }

    /// Append one explicit R1CS-F' compiler input to the chain as a K=1
    /// chunk. Convenience wrapper around [`Self::append_chunk`].
    pub fn append_step(&mut self, input: R1csFPrimeStepInput) -> Result<R1csCompiledStep, Error> {
        let mut chunk = self.append_chunk(vec![input])?;
        debug_assert_eq!(chunk.len(), 1);
        Ok(chunk.pop().expect("K=1 append returns 1 step"))
    }

    /// Append K satisfying R1CS assignments as **one** SuperNeo chunk.
    /// All assignments must satisfy the same R1CS shape this chain
    /// committed to; they share the pre-step chain state and produce K
    /// distinct `EncodedFPrimeStep` images (each absorbs its own
    /// `app_public_input` into its `state_x_out`).
    ///
    /// Rejects `K == 0` (SuperNeo requires `K \u{2265} 1`) and the
    /// steady-state RLC bound `(K + k_rho) \u{00B7} T \u{00B7} (b-1) < B`
    /// (propagated from the underlying NIFS at extend / finalize time).
    pub fn append_assignments(&mut self, assignments: Vec<Vec<neo_math::F>>) -> Result<Vec<R1csCompiledStep>, Error> {
        let inputs = assignments
            .into_iter()
            .map(|assignment| R1csFPrimeStepInput { assignment })
            .collect();
        self.append_chunk(inputs)
    }

    /// Append K explicit compiler inputs as one SuperNeo chunk. See
    /// [`Self::append_assignments`] for the semantics.
    pub fn append_chunk(&mut self, inputs: Vec<R1csFPrimeStepInput>) -> Result<Vec<R1csCompiledStep>, Error> {
        if inputs.is_empty() {
            return Err(Error::EmptyChunk);
        }
        let k = inputs.len();
        let is_recursive = self.audit.is_some();
        // Derive the semantic digests once. `compile_chunk` remains the
        // canonical funnel for all callers, but the builder also checks
        // the recursive state link before `prepare_next_fold` so an
        // obviously disconnected chunk cannot stash a pending fold.
        let semantic = semantic_state_digests_for_inputs(self.prep, &inputs)?;

        if is_recursive {
            if let Some(s) = semantic.as_ref() {
                if self.ctx.chain_state.semantic_state_digest != s.input {
                    return Err(Error::Compiler(R1csCompilerError::SemanticStateInputMismatch {
                        expected: self.ctx.chain_state.semantic_state_digest,
                        got: s.input,
                    }));
                }
            }
            #[cfg(feature = "perf-timers")]
            let t_prepare = std::time::Instant::now();
            // Computes the fold proof for the upcoming step AND stashes
            // the post-fold audit in `self.pending_audit` (so the deposit
            // below need not re-run the fold).
            self.prepare_next_fold(k, semantic.map(|s| digest_fields_as_digest32(s.output)))?;
            #[cfg(feature = "perf-timers")]
            eprintln!(
                "[r1cs-chain] prepare_next_fold             {:>7.2}s",
                t_prepare.elapsed().as_secs_f64()
            );
        }

        #[cfg(feature = "perf-timers")]
        let t_compile = std::time::Instant::now();
        // K=1 → keep using the legacy `compile_step` so its error
        // surface (`R1csCompilerError::Unsatisfied(_)`) stays
        // unchanged for single-step callers.
        let compiled = if k == 1 {
            let mut inputs_iter = inputs;
            vec![compile_step(
                self.prep,
                &mut self.ctx,
                inputs_iter.pop().expect("K=1 has one input"),
            )?]
        } else {
            compile_chunk(self.prep, &mut self.ctx, inputs)?
        };
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[r1cs-chain] compile_chunk (K={k})            {:>7.2}s",
            t_compile.elapsed().as_secs_f64()
        );

        #[cfg(feature = "perf-timers")]
        let t_instance = std::time::Instant::now();
        let instances: Vec<CcsInstance> = compiled
            .iter()
            .map(|step| build_instance(self.prep, &step.encoded))
            .collect::<Result<Vec<_>, _>>()?;
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[r1cs-chain] build_instance (K={k})           {:>7.2}s",
            t_instance.elapsed().as_secs_f64()
        );

        #[cfg(feature = "perf-timers")]
        let t_fold = std::time::Instant::now();
        self.audit = Some(if is_recursive {
            // Reuse the fold `prepare_next_fold` already computed instead
            // of re-running NIFS.prove inside `lifecycle::extend`.
            //
            // The stashed `pending_audit` is `extend(prev_audit,
            // placeholder)`. The desired audit is `extend(prev_audit,
            // instances)`. Because the per-step fold consumes only the
            // *previous* chunk's `latest` (not this chunk's deposit) and
            // the chunk digest is shape+count-only (placeholder and
            // `instances` share K and the F' shape), the fold proof,
            // post-fold `running`, advanced counters, boundary digests,
            // and `x_out` are identical between the two. The audits
            // therefore differ in *exactly* the deposited `latest` and
            // `public_batches.last()`. Swap the real instances in.
            let mut audit = self
                .pending_audit
                .take()
                .ok_or(Error::ChainExpectedActiveState)?;
            match &mut audit.proof.state.proof {
                ProofState::Active { latest, .. } => {
                    *latest = LatestInstance::from_instances(instances.clone());
                }
                _ => return Err(Error::ChainExpectedActiveState),
            }
            let real_claims: Vec<CcsClaim> = instances.iter().map(|i| i.claim.clone()).collect();
            *audit
                .public_batches
                .last_mut()
                .ok_or(Error::ChainExpectedActiveState)? = real_claims;
            audit
        } else {
            if let Some(semantic) = semantic {
                crate::lifecycle::prove::prove_one_with_semantic_state(
                    &self.prep.prep,
                    instances.clone(),
                    digest_fields_as_digest32(semantic.input),
                    digest_fields_as_digest32(semantic.output),
                )?
            } else {
                crate::lifecycle::prove(&self.prep.prep, [instances.clone()])?
            }
        });
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[r1cs-chain] deposit (reuse fold / base prove) {:>7.2}s",
            t_fold.elapsed().as_secs_f64()
        );
        self.latest_batch = instances;
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

    /// Derive the next-step fold authority by simulating a K-sized
    /// extend. `next_rows_in_chunk` is the size of the *upcoming*
    /// chunk's batch — it controls the chunk_digest the simulated
    /// extend's NIFS transcript absorbs, which must match what the
    /// upcoming F' R1CS will replay.
    fn prepare_next_fold(
        &mut self,
        next_rows_in_chunk: usize,
        semantic_state_digest_next: Option<[u8; 32]>,
    ) -> Result<(), Error> {
        let audit = self.audit.as_ref().ok_or(Error::ChainEmpty)?;
        if self.latest_batch.is_empty() {
            return Err(Error::ChainEmpty);
        }
        let pre_state = audit.proof.state.clone();

        let (pre_running, latest) = match &pre_state.proof {
            ProofState::Active { running, latest } => (running.clone(), latest.clone()),
            _ => return Err(Error::ChainExpectedActiveState),
        };

        // Derive the fold proof the upcoming F' step will replay. The
        // simulated extend's `next_latest` must match the upcoming
        // batch size so the NIFS transcript absorbs the same
        // `chunk_digest` (K-aware). The actual `x` values inside the
        // placeholder don't matter — `f_prime_chunk_public_digest`
        // reads only the per-claim shape `(d, kappa, m_in)` and the
        // batch length.
        let placeholder = vec![self.latest_batch[0].clone(); next_rows_in_chunk];
        let derived = if let Some(semantic_state_digest_next) = semantic_state_digest_next {
            crate::lifecycle::prove::extend_with_semantic_state(
                &self.prep.prep,
                audit.clone(),
                placeholder,
                semantic_state_digest_next,
            )?
        } else {
            crate::lifecycle::extend(&self.prep.prep, audit.clone(), placeholder)?
        };
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
            semantic_state_digest: digest32_as_fields(pre_state.semantic_state_digest),
            acc_digest: digest32_as_fields(pre_state.acc_digest),
            public_trace: digest32_as_fields(pre_state.public_trace),
        };
        self.ctx.fold_for_step = Some(R1csFoldForStep {
            pre_running,
            latest,
            proof: fold,
            post_running,
        });

        // Keep the post-fold audit so `append_chunk` can deposit the real
        // compiled instances by swapping its placeholder `latest` —
        // avoiding a second, identical NIFS prove inside `extend`.
        self.pending_audit = Some(derived);

        Ok(())
    }
}
