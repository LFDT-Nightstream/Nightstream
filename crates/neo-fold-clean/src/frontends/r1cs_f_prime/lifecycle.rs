//! Prover lifecycle for R1CS-encoded `F'` chains.
//!
//! Owns: compile-to-instance-to-fold orchestration, chunk batching, semantic
//! state threading, and final audit/uncompressed handoff.
//!
//! Does not own: R1CS compilation rules, NIFS internals, decider verification, or
//! constraint emission.
//!
//! Emits constraints: no. It consumes preprocessed structures and compiled
//! encoded steps.
//!
//! Authority boundary: compiler summaries are orchestration data; the retained
//! audit state and NIFS proof verification authorize each fold transition.
//!
//! | Obligation | Local owner | Emits constraints? | Authority source |
//! |---|---|---|---|
//! | Stateless batch | [`prove_encoded_steps`] | no | Preprocessed relation and lifecycle proof |
//! | Stateful append | [`R1csChainBuilder`] append methods | no | Checked prior audit and compiled chunk |
//! | Final handoff | [`R1csChainBuilder::finish_with_audit`] | no | Completed uncompressed audit |

use crate::frontends::f_prime::compiler::FPrimeFoldPostSummary;
use crate::frontends::f_prime::encoder::EncodedFPrimeStep;
use crate::frontends::f_prime::image::{FPrimeImageLayout, StateInDigestTarget, StateOutDigestTarget};
use crate::frontends::f_prime::recursive_plan::{
    build_recursive_step_image_config, state_x_out_digest_mode_for_options,
};
use crate::frontends::r1cs_f_prime::compiler::{
    compile_chunk, compile_step, semantic_state_digests_for_inputs, semantic_state_in_preimage_for_assignment,
    semantic_state_out_preimage_for_assignment, start_chain, R1csCompiledStep, R1csCompilerContext, R1csCompilerError,
    R1csFPrimeStepInput, R1csFoldForStep,
};
use crate::frontends::r1cs_f_prime::instance::build_instance;
use crate::frontends::r1cs_f_prime::{Error, R1csFPrimePreprocessing};
use crate::lifecycle::{Uncompressed, UncompressedAudit};
use crate::paper::construction2::{LatestInstance, ProofState};
use crate::paper::digest::{digest32_as_fields, digest_fields_as_digest32, StateXOutDigestMode};
use crate::paper::nifs::{
    NifsFreshImageOverlayRequest, NifsFreshImageRegion, NifsFreshImageRegionKind, NifsFreshInstancesRequest,
    NifsFreshSemanticStateInOverlay, NifsFreshSemanticStateOutOverlay, NifsFreshStateXOutOverlay, NifsProverAdapter,
};
use crate::paper::relations::{CcsClaim, CcsInstance};
use neo_math::F;

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

    /// Append one assignment while routing recursive NIFS proving through
    /// `adapter`. The base append remains the same F' initialization path;
    /// recursive appends and finalization are where NIFS.P runs.
    pub fn append_assignment_with_nifs_adapter(
        &mut self,
        assignment: Vec<neo_math::F>,
        adapter: &mut dyn NifsProverAdapter,
    ) -> Result<R1csCompiledStep, Error> {
        let mut chunk = self.append_assignments_with_nifs_adapter(vec![assignment], adapter)?;
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

    /// Append one explicit compiler input while routing recursive NIFS
    /// proving through `adapter`.
    pub fn append_step_with_nifs_adapter(
        &mut self,
        input: R1csFPrimeStepInput,
        adapter: &mut dyn NifsProverAdapter,
    ) -> Result<R1csCompiledStep, Error> {
        let mut chunk = self.append_chunk_with_nifs_adapter(vec![input], adapter)?;
        debug_assert_eq!(chunk.len(), 1);
        Ok(chunk.pop().expect("K=1 append returns 1 step"))
    }

    /// Append K satisfying R1CS assignments as **one** SuperNeo chunk.
    /// All assignments must satisfy the same R1CS shape this chain
    /// committed to; they share the pre-step chain state and produce K
    /// distinct `EncodedFPrimeStep` images. Plans that bind app public
    /// input through semantic state are serial today because one chunk
    /// has one outgoing semantic-state digest.
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

    /// Append K assignments while routing recursive NIFS proving through
    /// `adapter`.
    pub fn append_assignments_with_nifs_adapter(
        &mut self,
        assignments: Vec<Vec<neo_math::F>>,
        adapter: &mut dyn NifsProverAdapter,
    ) -> Result<Vec<R1csCompiledStep>, Error> {
        let inputs = assignments
            .into_iter()
            .map(|assignment| R1csFPrimeStepInput { assignment })
            .collect();
        self.append_chunk_with_nifs_adapter(inputs, adapter)
    }

    /// Append K explicit compiler inputs as one SuperNeo chunk. See
    /// [`Self::append_assignments`] for the semantics.
    pub fn append_chunk(&mut self, inputs: Vec<R1csFPrimeStepInput>) -> Result<Vec<R1csCompiledStep>, Error> {
        self.append_chunk_inner(inputs, None)
    }

    /// Append K explicit compiler inputs while routing recursive NIFS
    /// proving through `adapter`.
    pub fn append_chunk_with_nifs_adapter(
        &mut self,
        inputs: Vec<R1csFPrimeStepInput>,
        adapter: &mut dyn NifsProverAdapter,
    ) -> Result<Vec<R1csCompiledStep>, Error> {
        self.append_chunk_inner(inputs, Some(adapter))
    }

    fn append_chunk_inner(
        &mut self,
        inputs: Vec<R1csFPrimeStepInput>,
        mut adapter: Option<&mut dyn NifsProverAdapter>,
    ) -> Result<Vec<R1csCompiledStep>, Error> {
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
                if let Some(input) = s.input {
                    if self.ctx.chain_state.semantic_state_digest != input {
                        return Err(Error::Compiler(R1csCompilerError::SemanticStateInputMismatch {
                            expected: self.ctx.chain_state.semantic_state_digest,
                            got: input,
                        }));
                    }
                }
            }
            #[cfg(feature = "perf-timers")]
            let t_prepare = std::time::Instant::now();
            // Computes the fold proof for the upcoming step AND stashes
            // the post-fold audit in `self.pending_audit` (so the deposit
            // below need not re-run the fold).
            let semantic_state_digest_next = semantic.map(|s| digest_fields_as_digest32(s.output));
            if let Some(adapter) = adapter.as_mut() {
                self.prepare_next_fold(k, semantic_state_digest_next, Some(&mut **adapter))?;
            } else {
                self.prepare_next_fold(k, semantic_state_digest_next, None)?;
            }
            #[cfg(feature = "perf-timers")]
            eprintln!(
                "[r1cs-chain] prepare_next_fold             {:>7.2}s",
                t_prepare.elapsed().as_secs_f64()
            );
        }

        let source_assignments_for_adapter = adapter.is_some().then(|| {
            inputs
                .iter()
                .map(|input| input.assignment.clone())
                .collect::<Vec<_>>()
        });
        let semantic_state_out_preimages_for_adapter = source_assignments_for_adapter
            .as_ref()
            .map(|assignments| {
                assignments
                    .iter()
                    .map(|assignment| semantic_state_out_preimage_for_assignment(self.prep, assignment))
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?
            .and_then(|preimages| preimages.into_iter().collect::<Option<Vec<_>>>());
        let semantic_state_in_preimages_for_adapter = source_assignments_for_adapter
            .as_ref()
            .and_then(|assignments| {
                let preimages = assignments
                    .iter()
                    .map(|assignment| semantic_state_in_preimage_for_assignment(self.prep, assignment))
                    .collect::<Option<Vec<_>>>()?;
                Some(preimages)
            });

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
        let instances: Vec<CcsInstance> = if let Some(adapter) = adapter.as_deref_mut() {
            let assignments = compiled
                .iter()
                .map(|step| step.encoded.image.values.as_slice())
                .collect::<Vec<_>>();
            let source_assignment_refs = source_assignments_for_adapter
                .as_ref()
                .map(|assignments| assignments.iter().map(Vec::as_slice).collect::<Vec<_>>());
            let semantic_state_out_preimage_refs = semantic_state_out_preimages_for_adapter
                .as_ref()
                .map(|preimages| preimages.iter().map(Vec::as_slice).collect::<Vec<_>>());
            let semantic_state_in_preimage_refs = semantic_state_in_preimages_for_adapter
                .as_ref()
                .map(|preimages| preimages.iter().map(Vec::as_slice).collect::<Vec<_>>());
            let semantic_state_out_digest = semantic.map(|s| s.output);
            let layout = FPrimeImageLayout::new(build_recursive_step_image_config(&self.prep.plan));
            let compact_lane_offsets = compact_u64_lane_offsets(&layout);
            let regions = f_prime_image_regions(&layout);
            let image_overlay =
                source_assignment_refs
                    .as_ref()
                    .map(|source_assignments| NifsFreshImageOverlayRequest {
                        app_private_offset: layout.app_private.offset,
                        app_private_var_widths: &self.prep.plan.app_private_var_widths,
                        source_assignments,
                        compact_lane_offsets: &compact_lane_offsets,
                        regions: &regions,
                        semantic_state_in: semantic_state_in_preimage_refs
                            .as_ref()
                            .and_then(|preimages| semantic_state_in_overlay(&layout, self.prep, preimages)),
                        semantic_state_out: semantic_state_out_preimage_refs
                            .as_ref()
                            .and_then(|preimages| {
                                semantic_state_out_digest.and_then(|digest| {
                                    semantic_state_out_overlay(&layout, self.prep, preimages, digest)
                                })
                            }),
                        state_x_out: state_x_out_overlay(&layout, self.prep),
                    });
            match adapter.build_fresh_instances(NifsFreshInstancesRequest {
                pp: &self.prep.prep.params,
                s: self.prep.prep.structure(),
                cache: self.prep.prep.optimized_cache(),
                log: &self.prep.prep.log,
                m_in: compiled[0].encoded.public_input_len(),
                assignments: &assignments,
                image_overlay,
                lane_scheme: None,
            })? {
                Some(instances) => {
                    if instances.len() != compiled.len() {
                        return Err(Error::Nifs(crate::paper::nifs::Error::BackendUnavailable {
                            backend: "adapter",
                            reason: "adapter returned the wrong number of fresh instances",
                        }));
                    }
                    instances
                }
                None => compiled
                    .iter()
                    .map(|step| build_instance(self.prep, &step.encoded))
                    .collect::<Result<Vec<_>, _>>()?,
            }
        } else {
            compiled
                .iter()
                .map(|step| build_instance(self.prep, &step.encoded))
                .collect::<Result<Vec<_>, _>>()?
        };
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
                let initial = semantic
                    .input
                    .map(digest_fields_as_digest32)
                    .unwrap_or_else(|| self.prep.prep.initial_semantic_state_digest());
                crate::lifecycle::prove::prove_one_with_semantic_state(
                    &self.prep.prep,
                    instances.clone(),
                    initial,
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

    /// Finalize while dropping the audit trail.
    ///
    /// This output is suitable for terminal-only
    /// `lifecycle::verify_uncompressed` only for single-chunk F' chains.
    /// Multi-chunk F' chains need [`Self::finish_with_audit`] until the
    /// compressed decider proves the recursive F'/NIFS.V induction.
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

    /// Finalize with an explicit NIFS prover adapter for the terminal
    /// latest-to-running fold.
    pub fn finish_with_audit_and_nifs_adapter(
        self,
        adapter: &mut dyn NifsProverAdapter,
    ) -> Result<UncompressedAudit, Error> {
        let prep = self.prep;
        let audit = self.into_audit()?;
        Ok(crate::lifecycle::finish_uncompressed_with_audit_and_nifs_adapter(
            &prep.prep, adapter, audit,
        )?)
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
        adapter: Option<&mut dyn NifsProverAdapter>,
    ) -> Result<(), Error> {
        if self.latest_batch.is_empty() {
            return Err(Error::ChainEmpty);
        }
        let (pre_chain_state, pre_running_carrier, latest) = {
            let audit = self.audit.as_ref().ok_or(Error::ChainEmpty)?;
            let pre_state = &audit.proof.state;
            let pre_chain_state = crate::frontends::r1cs_f_prime::R1csChainState {
                chunk_count: pre_state.chunk_count,
                step_count: pre_state.step_count,
                z_i: digest32_as_fields(pre_state.z_i),
                semantic_state_digest: digest32_as_fields(pre_state.semantic_state_digest),
                acc_digest: digest32_as_fields(pre_state.acc_digest),
                public_trace: digest32_as_fields(pre_state.public_trace),
            };
            // Keep the carrier intact until we know whether the backend's
            // compile-facing summary is sufficient. Materializing here would
            // force a device-backed running/proof result back through the old
            // host object boundary before the fold even runs.
            let (pre_running_carrier, latest) = match &pre_state.proof {
                ProofState::Active { running, latest } => (running.clone(), latest.clone()),
                _ => return Err(Error::ChainExpectedActiveState),
            };
            (pre_chain_state, pre_running_carrier, latest)
        };

        // Derive the fold proof the upcoming F' step will replay. The
        // simulated extend's `next_latest` must match the upcoming
        // batch size so the NIFS transcript absorbs the same
        // `chunk_digest` (K-aware). The actual `x` values inside the
        // placeholder don't matter — `f_prime_chunk_public_digest`
        // reads only the per-claim shape `(d, kappa, m_in)` and the
        // batch length.
        let placeholder = vec![self.latest_batch[0].clone(); next_rows_in_chunk];
        let fold_needs_native_verify = adapter
            .as_ref()
            .map(|adapter| adapter.requires_recursive_compile_reverify())
            .unwrap_or(true);
        let used_in_place_adapter = adapter.is_some();
        let mut post_summary_override = None;
        match (adapter, semantic_state_digest_next) {
            (Some(adapter), Some(semantic_state_digest_next)) => {
                let audit = self.audit.as_mut().ok_or(Error::ChainEmpty)?;
                let post_summary =
                    crate::lifecycle::prove::extend_in_place_with_semantic_state_and_nifs_adapter_output(
                        &self.prep.prep,
                        adapter,
                        audit,
                        placeholder,
                        semantic_state_digest_next,
                    )?;
                post_summary_override = post_summary.and_then(|summary| summary.f_prime().cloned());
            }
            (Some(adapter), None) => {
                let audit = self.audit.as_mut().ok_or(Error::ChainEmpty)?;
                let post_summary = crate::lifecycle::prove::extend_in_place_with_nifs_adapter_output(
                    &self.prep.prep,
                    adapter,
                    audit,
                    placeholder,
                )?;
                post_summary_override = post_summary.and_then(|summary| summary.f_prime().cloned());
            }
            (None, Some(semantic_state_digest_next)) => {
                let audit = self.audit.as_ref().ok_or(Error::ChainEmpty)?;
                self.pending_audit = Some(crate::lifecycle::prove::extend_with_semantic_state(
                    &self.prep.prep,
                    audit.clone(),
                    placeholder,
                    semantic_state_digest_next,
                )?);
            }
            (None, None) => {
                let audit = self.audit.as_ref().ok_or(Error::ChainEmpty)?;
                self.pending_audit = Some(crate::lifecycle::extend(&self.prep.prep, audit.clone(), placeholder)?);
            }
        }
        self.ctx.chain_state = pre_chain_state;
        if !fold_needs_native_verify {
            if let Some(summary) = post_summary_override {
                self.ctx.fold_for_step = None;
                self.ctx.fold_summary_for_step = Some(summary);
            } else {
                return Err(Error::ChainExpectedActiveState);
            }
        } else {
            let derived = if used_in_place_adapter {
                self.audit.as_ref().ok_or(Error::ChainEmpty)?
            } else {
                self.pending_audit
                    .as_ref()
                    .ok_or(Error::ChainExpectedActiveState)?
            };
            let fold = match &derived.steps.last().expect("extend appended one step").fold {
                crate::paper::construction2::FoldProof::Recursive(p) => p.materialize()?,
                crate::paper::construction2::FoldProof::NoFold => return Err(Error::ChainExpectedActiveState),
            };
            let post_running = match &derived.proof.state.proof {
                ProofState::Active { running, .. } => running.materialize()?.claims_only(),
                _ => return Err(Error::ChainExpectedActiveState),
            };
            let pre_running = pre_running_carrier.materialize()?.claims_only();
            let post_summary = Some(
                FPrimeFoldPostSummary::from_running(
                    &post_running,
                    self.prep.prep.structure(),
                    self.ctx.public_input_len,
                )
                .map_err(R1csCompilerError::from)?,
            );
            self.ctx.fold_for_step = Some(R1csFoldForStep {
                pre_running,
                latest,
                proof: fold,
                post_summary,
                post_running,
            });
            self.ctx.fold_summary_for_step = None;
        }
        self.ctx.fold_for_step_needs_native_verify = fold_needs_native_verify;

        // Keep the post-fold audit so `append_chunk` can deposit the real
        // compiled instances by swapping its placeholder `latest` —
        // avoiding a second, identical NIFS prove inside `extend`.
        if used_in_place_adapter {
            self.pending_audit = self.audit.take();
        }

        Ok(())
    }
}

fn compact_u64_lane_offsets(layout: &FPrimeImageLayout) -> Vec<usize> {
    let mut out = Vec::new();
    push_region_lanes(&mut out, layout.boundary.offset, layout.boundary.bits);
    push_region_lanes(&mut out, layout.state_in.offset, layout.state_in.bits);
    push_region_lanes(&mut out, layout.state_out.offset, layout.state_out.bits);
    push_region_lanes(&mut out, layout.chunk_digest.offset, layout.chunk_digest.bits);
    if layout.is_base.bits > 1 {
        push_region_lanes(&mut out, layout.is_base.offset + 1, layout.is_base.bits - 1);
    }
    push_region_lanes(&mut out, layout.nifs_payloads.offset, layout.nifs_payloads.bits);
    push_region_lanes(&mut out, layout.kmul.offset, layout.kmul.bits);
    push_region_lanes(&mut out, layout.ring_action.offset, layout.ring_action.bits);
    push_region_lanes(&mut out, layout.poseidon.offset, layout.poseidon.bits);
    out.sort_unstable();
    out.dedup();
    out
}

fn f_prime_image_regions(layout: &FPrimeImageLayout) -> Vec<NifsFreshImageRegion> {
    [
        (NifsFreshImageRegionKind::Boundary, layout.boundary),
        (NifsFreshImageRegionKind::StateIn, layout.state_in),
        (NifsFreshImageRegionKind::StateOut, layout.state_out),
        (NifsFreshImageRegionKind::ChunkDigest, layout.chunk_digest),
        (NifsFreshImageRegionKind::AppPrivate, layout.app_private),
        (NifsFreshImageRegionKind::IsBase, layout.is_base),
        (NifsFreshImageRegionKind::NifsPayloads, layout.nifs_payloads),
        (NifsFreshImageRegionKind::Kmul, layout.kmul),
        (NifsFreshImageRegionKind::RingAction, layout.ring_action),
        (NifsFreshImageRegionKind::Poseidon, layout.poseidon),
    ]
    .into_iter()
    .filter_map(|(kind, range)| {
        (range.bits != 0).then_some(NifsFreshImageRegion {
            kind,
            offset: range.offset,
            bits: range.bits,
        })
    })
    .collect()
}

fn semantic_state_out_overlay<'a>(
    layout: &FPrimeImageLayout,
    prep: &'a R1csFPrimePreprocessing,
    preimages: &'a [&'a [F]],
    digest: [F; 4],
) -> Option<NifsFreshSemanticStateOutOverlay<'a>> {
    let state_x_out = prep.plan.state_x_out.as_ref()?;
    let binding = layout
        .config
        .one_shot_digest_to_state_out_bindings
        .iter()
        .find(|binding| binding.state_out_target == StateOutDigestTarget::NewSemanticStateDigest)?;
    let trace_layout = layout.one_shot_poseidon_layouts[binding.one_shot_index];
    if trace_layout.trace_len % 64 != 0 {
        return None;
    }
    Some(NifsFreshSemanticStateOutOverlay {
        trace_splice: layout.one_shot_poseidon_splices[binding.one_shot_index],
        trace_words_per_assignment: trace_layout.trace_len / 64,
        preimages,
        assignment_var_indices: semantic_state_out_assignment_indices(state_x_out),
        digest,
    })
}

fn semantic_state_in_overlay<'a>(
    layout: &FPrimeImageLayout,
    prep: &'a R1csFPrimePreprocessing,
    preimages: &'a [&'a [F]],
) -> Option<NifsFreshSemanticStateInOverlay<'a>> {
    let state_x_out = prep.plan.state_x_out.as_ref()?;
    let binding = layout
        .config
        .one_shot_digest_to_state_in_bindings
        .iter()
        .find(|binding| binding.state_in_target == StateInDigestTarget::SemanticStateDigestIn)?;
    let trace_layout = layout.one_shot_poseidon_layouts[binding.one_shot_index];
    if trace_layout.trace_len % 64 != 0 {
        return None;
    }
    Some(NifsFreshSemanticStateInOverlay {
        trace_splice: layout.one_shot_poseidon_splices[binding.one_shot_index],
        trace_words_per_assignment: trace_layout.trace_len / 64,
        preimages,
        assignment_var_indices: (!state_x_out.semantic_state_in_var_indices.is_empty())
            .then_some(state_x_out.semantic_state_in_var_indices.as_slice()),
    })
}

fn semantic_state_out_assignment_indices(
    state_x_out: &crate::frontends::f_prime::recursive_plan::StateXOutPlanOptions,
) -> Option<&[usize]> {
    if !state_x_out.semantic_state_out_var_indices.is_empty() {
        return Some(state_x_out.semantic_state_out_var_indices.as_slice());
    }
    if !state_x_out.app_public_input_var_indices.is_empty() && state_x_out.app_public_input_bit_var_indices.is_empty() {
        return Some(state_x_out.app_public_input_var_indices.as_slice());
    }
    None
}

fn state_x_out_overlay(
    layout: &FPrimeImageLayout,
    prep: &R1csFPrimePreprocessing,
) -> Option<NifsFreshStateXOutOverlay> {
    let state_x_out = prep.plan.state_x_out.as_ref()?;
    let one_shot_index = layout.one_shot_poseidon_splices.len().checked_sub(1)?;
    let trace_layout = layout.one_shot_poseidon_layouts[one_shot_index];
    if trace_layout.trace_len % 64 != 0 {
        return None;
    }
    Some(NifsFreshStateXOutOverlay {
        image_values_per_assignment: layout.end,
        state_lane_base: layout.state_in.offset,
        trace_splice: layout.one_shot_poseidon_splices[one_shot_index],
        trace_words_per_assignment: trace_layout.trace_len / 64,
        public_x_out_lane_offsets: state_x_out.public_x_out_lane_bit_starts,
        include_semantic_state: matches!(
            state_x_out_digest_mode_for_options(state_x_out),
            StateXOutDigestMode::Stateful
        ),
        pc: state_x_out.pc,
    })
}

fn push_region_lanes(out: &mut Vec<usize>, offset: usize, bits: usize) {
    if bits == 0 {
        return;
    }
    debug_assert_eq!(bits % 64, 0, "compact F' region must be lane-aligned");
    for lane in 0..bits / 64 {
        out.push(offset + lane * 64);
    }
}
