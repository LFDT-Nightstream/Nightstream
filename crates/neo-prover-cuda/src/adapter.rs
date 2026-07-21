//! `CudaNifsProver` — the `NifsProverAdapter` orchestrator.
//!
//! Owns only the fold's data flow: ingest → commit → the SuperNeo
//! reductions (Π_CCS → Π_RLC → Π_DEC) → accumulate → egress. All device
//! state lives in [`crate::session::DeviceSession`]; all computation lives
//! in `crate::commit`, `crate::ingest`, and `crate::reduce`.

use std::sync::{Arc, Mutex};

use cuda_core::{CudaContext, DeviceBuffer};
use neo_ajtai::{AjtaiSModule, Commitment};
use neo_ccs::Mat;
use neo_fold_clean::frontends::f_prime::compiler::{nifs_ce_shape_from_claim, FPrimeFoldPostSummary};
use neo_fold_clean::paper::digest::{self, AccumulatorHandle};
use neo_fold_clean::paper::nifs::{
    DeferredNifsProofMaterializer, Error, NifsFreshInstancesRequest, NifsPostFoldSummary, NifsProof, NifsProofCarrier,
    NifsProverAdapter, NifsProverOutput, NifsProverRequest, NifsRunningCarrier,
};
use neo_fold_clean::paper::params::Params;
use neo_fold_clean::paper::relations::{ajtai_rlc_mixer, mix_adv, recompose_adv, CcsClaim, CeClaim, RlcMixer};
use neo_fold_clean::paper::{pi_ccs, pi_rlc};
use neo_fold_clean::{CcsInstance, CcsWitness, RunningInstance};
use neo_math::{D, F, K};
use neo_reductions::common::ct_from_y_ring_for_ccs_m;
use neo_reductions::optimized_engine::{BackendTranscriptMode, OptimizedStructureCache};
use p3_field::PrimeCharacteristicRing;

use crate::fold_output::{
    device_output_from_carrier, CudaRunningCarrier, DeviceCommitments, DeviceFoldOutput, MixedCommitment,
};
use crate::reduce::ccs::{
    pi_ccs_outputs_digest_field_count, DeviceFeBackend, DeviceFeRowProofLogArchive, DeviceNcBackend,
    DevicePiCcsKSurfaces, DevicePiCcsOutputsDigest, DevicePiCcsPhaseBackend, DevicePiCcsProofLogExporter,
    DevicePublicX, FePhaseWorkspace, NcPhaseWorkspace, PiCcsOutputDigestShell,
};
use crate::reduce::dec::{DecOutputMode, DecParentWitness, DecRecompositionMode};
use crate::reduce::rlc as device_rlc;
use crate::session::{backend_unavailable, CachedDeviceCommitments, CachedRunningPlanes, DeviceSession};

pub struct CudaNifsProver {
    session: DeviceSession,
    fe_phase_mode: FePhaseMode,
    terminal_claims_only: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FePhaseMode {
    RowTrace,
    WholeTrace(WholeFeMode),
    WholeTraceGraph(WholeFeMode),
    WholeTraceGraphBudget { remaining: usize, mode: WholeFeMode },
    WholeTraceGraphRecaptureBudget { remaining: usize, mode: WholeFeMode },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WholeFeMode {
    Fast,
    Parity,
}

impl WholeFeMode {
    fn transcript_mode(self) -> BackendTranscriptMode {
        match self {
            Self::Fast => BackendTranscriptMode::DeviceSnapshot,
            Self::Parity => BackendTranscriptMode::Replay,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FePhaseSettings {
    enabled: bool,
    graph: bool,
    recapture: bool,
    transcript_mode: BackendTranscriptMode,
}

impl FePhaseMode {
    fn next_settings(&mut self) -> FePhaseSettings {
        let disabled = FePhaseSettings {
            enabled: false,
            graph: false,
            recapture: false,
            transcript_mode: BackendTranscriptMode::DeviceSnapshot,
        };
        match self {
            Self::RowTrace => disabled,
            Self::WholeTrace(mode) => FePhaseSettings {
                enabled: true,
                graph: false,
                recapture: false,
                transcript_mode: mode.transcript_mode(),
            },
            Self::WholeTraceGraph(mode) => FePhaseSettings {
                enabled: true,
                graph: true,
                recapture: false,
                transcript_mode: mode.transcript_mode(),
            },
            Self::WholeTraceGraphBudget { remaining, mode } => {
                let enabled = *remaining > 0;
                *remaining = remaining.saturating_sub(1);
                FePhaseSettings {
                    enabled,
                    graph: enabled,
                    recapture: false,
                    transcript_mode: mode.transcript_mode(),
                }
            }
            Self::WholeTraceGraphRecaptureBudget { remaining, mode } => {
                let enabled = *remaining > 0;
                *remaining = remaining.saturating_sub(1);
                FePhaseSettings {
                    enabled,
                    graph: enabled,
                    recapture: enabled,
                    transcript_mode: mode.transcript_mode(),
                }
            }
        }
    }
}

struct CudaDeferredNifsProof {
    state: Mutex<CudaDeferredNifsProofState>,
}

enum CudaDeferredNifsProofState {
    Pending {
        pi_ccs: DeferredCudaPiCcsProof,
        pi_rlc: pi_rlc::Proof,
        pi_dec: neo_fold_clean::paper::pi_dec::Proof,
        fold_output: Arc<DeviceFoldOutput>,
    },
    Ready(NifsProof),
    Failed,
}

struct DeferredCudaPiCcsProof {
    state: DeferredCudaPiCcsProofState,
    output_authority: Option<DevicePiCcsOutputAuthority>,
}

enum DeferredCudaPiCcsProofState {
    Ready(pi_ccs::Proof),
    Rows {
        proof: pi_ccs::DeferredProof,
        fe_rows: DeviceFeRowProofLogArchive,
    },
}

struct DevicePiCcsOutputAuthority {
    commitments: Arc<DeviceCommitments>,
    public_x: DevicePublicX,
    outputs_digest: [F; 4],
}

impl DeferredCudaPiCcsProof {
    fn ready(proof: pi_ccs::Proof, output_authority: Option<DevicePiCcsOutputAuthority>) -> Self {
        Self {
            state: DeferredCudaPiCcsProofState::Ready(proof),
            output_authority,
        }
    }

    fn rows(
        proof: pi_ccs::DeferredProof,
        fe_rows: DeviceFeRowProofLogArchive,
        output_authority: Option<DevicePiCcsOutputAuthority>,
    ) -> Self {
        Self {
            state: DeferredCudaPiCcsProofState::Rows { proof, fe_rows },
            output_authority,
        }
    }

    fn finish(self) -> Result<pi_ccs::Proof, Error> {
        let mut proof = match self.state {
            DeferredCudaPiCcsProofState::Ready(proof) => proof,
            DeferredCudaPiCcsProofState::Rows { proof, fe_rows } => {
                let finished;
                crate::perf_timed!("finalize.proof_export", {
                    let rounds = fe_rows
                        .export_rounds()
                        .map_err(|_| backend_unavailable("export archived device Pi_CCS FE proof log failed"))?;
                    finished = proof.finish_with_fe_rounds(rounds)?;
                });
                finished
            }
        };
        let Some(authority) = self.output_authority else {
            return Ok(proof);
        };
        let commitments = authority.commitments.materialize()?;
        if commitments.len() != proof.outputs.len() {
            return Err(backend_unavailable("device Pi_CCS proof commitment count mismatch"));
        }
        authority
            .public_x
            .materialize_claims(&mut proof.outputs)
            .map_err(|_| backend_unavailable("device Pi_CCS proof public X materialization failed"))?;
        for (output, commitment) in proof.outputs.iter_mut().zip(commitments) {
            output.c = commitment;
        }
        let recomputed = digest::pi_ccs_outputs_digest(&proof.outputs);
        if recomputed != authority.outputs_digest {
            return Err(backend_unavailable("materialized device Pi_CCS output digest mismatch"));
        }
        proof.outputs_digest = recomputed;
        Ok(proof)
    }
}

impl CudaDeferredNifsProof {
    fn new(
        pi_ccs: DeferredCudaPiCcsProof,
        pi_rlc: pi_rlc::Proof,
        pi_dec: neo_fold_clean::paper::pi_dec::Proof,
        fold_output: Arc<DeviceFoldOutput>,
    ) -> Self {
        Self {
            state: Mutex::new(CudaDeferredNifsProofState::Pending {
                pi_ccs,
                pi_rlc,
                pi_dec,
                fold_output,
            }),
        }
    }
}

impl DeferredNifsProofMaterializer for CudaDeferredNifsProof {
    fn materialize(&self) -> Result<NifsProof, Error> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| backend_unavailable("deferred CUDA NIFS proof lock poisoned"))?;
        if let CudaDeferredNifsProofState::Ready(proof) = &*guard {
            return Ok(proof.clone());
        }
        let pending = std::mem::replace(&mut *guard, CudaDeferredNifsProofState::Failed);
        let CudaDeferredNifsProofState::Pending {
            pi_ccs,
            mut pi_rlc,
            mut pi_dec,
            fold_output,
        } = pending
        else {
            return Err(backend_unavailable(
                "deferred CUDA NIFS proof materialization already failed",
            ));
        };
        let finished_pi_ccs = pi_ccs.finish()?;
        pi_rlc.combined = fold_output.materialize_parent_authority()?;
        pi_dec.children = fold_output.materialize_claims()?;
        let proof = NifsProof {
            pi_ccs: finished_pi_ccs,
            pi_rlc,
            pi_dec,
        };
        *guard = CudaDeferredNifsProofState::Ready(proof.clone());
        Ok(proof)
    }
}

struct FinishedPiCcsProof {
    proof: pi_ccs::Proof,
    fe_phase_workspace: Option<FePhaseWorkspace>,
    nc_phase_workspace: Option<NcPhaseWorkspace>,
}

enum PiCcsProofState {
    Ready(pi_ccs::Proof),
    DeferredRows {
        proof: pi_ccs::DeferredProof,
        archive: DeviceFeRowProofLogArchive,
    },
    DeferredPhase {
        proof: pi_ccs::DeferredProof,
        exporter: DevicePiCcsProofLogExporter,
    },
}

impl PiCcsProofState {
    fn outputs(&self) -> &[CeClaim] {
        match self {
            Self::Ready(proof) => &proof.outputs,
            Self::DeferredRows { proof, .. } | Self::DeferredPhase { proof, .. } => proof.outputs(),
        }
    }

    fn outputs_digest(&self) -> Option<[F; 4]> {
        match self {
            Self::Ready(proof) => Some(proof.outputs_digest),
            Self::DeferredRows { .. } | Self::DeferredPhase { .. } => None,
        }
    }

    fn output_count(&self) -> usize {
        match self {
            Self::Ready(proof) => proof.outputs.len(),
            Self::DeferredRows { proof, .. } | Self::DeferredPhase { proof, .. } => proof.output_count(),
        }
    }

    fn claim_shell_metadata(&self) -> Result<device_rlc::ClaimShellMetadata<'_>, Error> {
        match self {
            Self::Ready(proof) => {
                let first = proof
                    .outputs
                    .first()
                    .ok_or_else(|| backend_unavailable("Pi_CCS output metadata missing"))?;
                Ok(device_rlc::ClaimShellMetadata {
                    count: proof.outputs.len(),
                    m_in: first.m_in,
                    r: &first.r,
                    s_col: &first.s_col,
                    has_y_zcol: !first.y_zcol.is_empty(),
                    fold_digest: first.fold_digest,
                })
            }
            Self::DeferredRows { proof, .. } | Self::DeferredPhase { proof, .. } => {
                let shell = proof.output_shell();
                Ok(device_rlc::ClaimShellMetadata {
                    count: shell.count,
                    m_in: shell.m_in,
                    r: &shell.row_chals,
                    s_col: &shell.s_col,
                    has_y_zcol: shell.has_y_zcol,
                    fold_digest: shell.fold_digest,
                })
            }
        }
    }

    fn finish(self) -> Result<FinishedPiCcsProof, Error> {
        match self {
            Self::Ready(proof) => Ok(FinishedPiCcsProof {
                proof,
                fe_phase_workspace: None,
                nc_phase_workspace: None,
            }),
            Self::DeferredRows { proof, archive } => {
                let row_rounds = archive
                    .export_rounds()
                    .map_err(|_| backend_unavailable("export archived device Pi_CCS FE proof log failed"))?;
                Ok(FinishedPiCcsProof {
                    proof: proof.finish_with_fe_rounds(row_rounds)?,
                    fe_phase_workspace: None,
                    nc_phase_workspace: None,
                })
            }
            Self::DeferredPhase { proof, mut exporter } => {
                let proof = proof.finish_with_phase_backend(&mut exporter)?;
                let (fe_phase_workspace, nc_phase_workspace) = exporter.into_workspaces();
                Ok(FinishedPiCcsProof {
                    proof,
                    fe_phase_workspace,
                    nc_phase_workspace,
                })
            }
        }
    }
}

fn resident_commitments_match(cached: &CachedDeviceCommitments, expected: &[Commitment]) -> bool {
    cached.host.len() == expected.len()
        && cached.device.count() == expected.len()
        && cached.device.words().len() == expected.len() * cached.device.words_per_commitment()
        && cached
            .host
            .iter()
            .zip(expected)
            .all(|(cached, expected)| cached == expected)
}

fn compose_resident_commitments(
    device: &crate::device::Device,
    ring: &crate::kernels::ajtai::AjtaiKernelModule,
    fresh_expected: &[Commitment],
    running_count: usize,
    fresh: Option<&CachedDeviceCommitments>,
    running: Option<&DeviceFoldOutput>,
) -> Result<Option<(DeviceBuffer<u64>, usize)>, Error> {
    let fresh = match (fresh_expected.len(), fresh) {
        (0, _) => None,
        (_, Some(cached)) if resident_commitments_match(cached, fresh_expected) => Some(cached.device.as_ref()),
        _ => return Ok(None),
    };
    let running = match (running_count, running) {
        (0, _) => None,
        (_, Some(output)) if output.child_count() == running_count => Some(output.child_commitments().as_ref()),
        _ => return Ok(None),
    };

    let words_per_commitment = fresh
        .or(running)
        .map(DeviceCommitments::words_per_commitment)
        .ok_or_else(|| backend_unavailable("Π_RLC resident commitment inputs missing"))?;
    if words_per_commitment == 0
        || words_per_commitment % D != 0
        || fresh
            .into_iter()
            .chain(running)
            .any(|cached| cached.d() != D || cached.words_per_commitment() != words_per_commitment)
    {
        return Ok(None);
    }

    let pieces: Vec<&DeviceBuffer<u64>> = [
        fresh.map(DeviceCommitments::words),
        running.map(DeviceCommitments::words),
    ]
    .into_iter()
    .flatten()
    .collect();
    let total_words = (fresh_expected.len() + running_count) * words_per_commitment;
    let words = device_rlc::compose_commitment_words_device(device, ring, &pieces, total_words)
        .map_err(|_| backend_unavailable("compose resident Π_RLC commitment inputs failed"))?;
    Ok(Some((words, words_per_commitment / D)))
}

fn authoritative_commitments(
    fresh: &[CcsClaim],
    running: &RunningInstance,
    running_output: Option<&DeviceFoldOutput>,
) -> Result<Vec<Commitment>, Error> {
    let mut commitments = fresh
        .iter()
        .map(|claim| claim.c.clone())
        .collect::<Vec<_>>();
    if let Some(output) = running_output {
        commitments.extend(output.child_commitments().materialize()?);
    } else {
        commitments.extend(running.claims.iter().map(|claim| claim.c.clone()));
    }
    Ok(commitments)
}

impl CudaNifsProver {
    pub fn new() -> Result<Self, Error> {
        let mut session = DeviceSession::new()?;
        session.ensure_kernels_loaded()?;
        Ok(Self::from_session(session))
    }

    /// Create a prover session with its own stream in a caller-owned context.
    ///
    /// This is for independent-chain scheduling: each prover keeps separate
    /// buffers and transcript state, while CUDA can schedule their streams in
    /// one context instead of paying one context per chain.
    pub fn new_on_context(ctx: Arc<CudaContext>) -> Result<Self, Error> {
        let mut session = DeviceSession::new_on_context(ctx)?;
        session.ensure_kernels_loaded()?;
        Ok(Self::from_session(session))
    }

    fn from_session(session: DeviceSession) -> Self {
        Self {
            session,
            fe_phase_mode: FePhaseMode::RowTrace,
            terminal_claims_only: false,
        }
    }

    fn post_fold_summary(&self, running: &RunningInstance) -> Result<NifsPostFoldSummary, Error> {
        let parent = running
            .parent_authority
            .as_ref()
            .ok_or_else(|| backend_unavailable("post-fold running accumulator missing parent authority"))?;
        let handle = AccumulatorHandle::from_running_parts(&running.claims, Some(parent));
        let f_prime = FPrimeFoldPostSummary {
            parent_shape: nifs_ce_shape_from_claim(parent, 0),
            child_count: running.claims.len() as u64,
            acc_digest: handle.digest_fields(),
        };
        Ok(NifsPostFoldSummary::new(Some(handle.digest()), Some(f_prime)))
    }

    fn resident_post_fold_summary(&self, output: &DeviceFoldOutput) -> NifsPostFoldSummary {
        let f_prime = FPrimeFoldPostSummary {
            parent_shape: nifs_ce_shape_from_claim(output.parent_authority(), 0),
            child_count: output.child_count() as u64,
            acc_digest: output.accumulator_digest_fields(),
        };
        NifsPostFoldSummary::new(Some(output.accumulator_digest()), Some(f_prime))
    }

    /// Enable whole-FE graph replay with host transcript parity checks.
    pub fn enable_whole_fe_graph_for_parity(&mut self) {
        self.fe_phase_mode = FePhaseMode::WholeTraceGraph(WholeFeMode::Parity);
    }

    /// Enable whole-FE graph replay with the device transcript snapshot.
    pub fn enable_whole_fe_graph_fast(&mut self) {
        self.fe_phase_mode = FePhaseMode::WholeTraceGraph(WholeFeMode::Fast);
    }

    /// Enable whole-FE device execution without graph capture or replay.
    pub fn enable_whole_fe_trace_for_parity(&mut self) {
        self.fe_phase_mode = FePhaseMode::WholeTrace(WholeFeMode::Parity);
    }

    /// Enable whole-FE device execution without online host replay.
    pub fn enable_whole_fe_trace_fast(&mut self) {
        self.fe_phase_mode = FePhaseMode::WholeTrace(WholeFeMode::Fast);
    }

    /// Keep private terminal planes resident for a later device decider.
    pub fn enable_terminal_claims_only_fast(&mut self) {
        self.terminal_claims_only = true;
    }

    /// Enable parity graph replay for a bounded number of folds.
    pub fn enable_whole_fe_graph_budget_for_parity(&mut self, folds: usize) {
        self.fe_phase_mode = FePhaseMode::WholeTraceGraphBudget {
            remaining: folds,
            mode: WholeFeMode::Parity,
        };
    }

    /// Enable the fast whole-FE graph path for a bounded number of folds.
    pub fn enable_whole_fe_graph_budget_fast(&mut self, folds: usize) {
        self.fe_phase_mode = FePhaseMode::WholeTraceGraphBudget {
            remaining: folds,
            mode: WholeFeMode::Fast,
        };
    }

    /// Recapture a parity graph for each of a bounded number of folds.
    pub fn enable_whole_fe_graph_recapture_budget_for_parity(&mut self, folds: usize) {
        self.fe_phase_mode = FePhaseMode::WholeTraceGraphRecaptureBudget {
            remaining: folds,
            mode: WholeFeMode::Parity,
        };
    }

    /// Enable fast whole-FE graph recapture for a bounded number of folds.
    pub fn enable_whole_fe_graph_recapture_budget_fast(&mut self, folds: usize) {
        self.fe_phase_mode = FePhaseMode::WholeTraceGraphRecaptureBudget {
            remaining: folds,
            mode: WholeFeMode::Fast,
        };
    }

    /// Prepare verifier-owned static data before the online fold loop.
    ///
    /// Callers that know the preprocessing context up front can pay the Ajtai
    /// PP and SuperNeo CSR uploads once during setup. The fold path still
    /// lazily checks these caches, so skipping this method only affects timing,
    /// not correctness.
    pub fn prepare_static(
        &mut self,
        pp: &Params,
        log: &AjtaiSModule,
        cache: &OptimizedStructureCache,
        fresh_count: usize,
    ) -> Result<(), Error> {
        crate::perf_timed!("session.params", {
            self.session.ensure_pp_uploaded(log)?;
        });
        crate::perf_timed!("session.structure", {
            self.session.ensure_structure_uploaded(cache)?;
        });
        let (_, m, t_core) = cache.shape();
        let include_y_zcol = m.div_ceil(D) > 1;
        let claim_counts = [fresh_count, fresh_count + pp.k_rho() as usize];
        for (index, claims) in claim_counts.into_iter().enumerate() {
            if claims == 0 || claim_counts[..index].contains(&claims) {
                continue;
            }
            let field_count = pi_ccs_outputs_digest_field_count(claims, t_core, D.next_power_of_two(), include_y_zcol);
            self.session
                .sis
                .prepare_digest(
                    &self.session.device,
                    neo_fold_clean::paper::reductions::accumulator_sis_circuit::PI_CCS_OUTPUTS_SIS_CONFIG,
                    field_count,
                )
                .map_err(|_| backend_unavailable("prepare Pi_CCS output SIS map failed"))?;
        }
        self.session
            .device
            .sync()
            .map_err(|_| backend_unavailable("synchronize Pi_CCS output SIS maps failed"))?;
        Ok(())
    }
}

impl NifsProverAdapter for CudaNifsProver {
    /// NIFS.P on device: Π_CCS (device sumcheck backends) → Π_RLC (device
    /// rho sampling, witness mix, and CE surfaces) → Π_DEC (device). Transcript
    /// effects and proof bytes are identical to `paper::nifs::prove`.
    fn prove(&mut self, request: NifsProverRequest<'_>) -> Result<NifsProverOutput, Error> {
        let NifsProverRequest {
            tr,
            pp,
            s,
            cache,
            log,
            lanes,
            mix_rhos_commits,
            combine_b_pows,
            fresh,
            running_carrier,
            running,
            cache_output_for_next_step,
            ..
        } = request;
        if neo_fold_clean::paper::construction2::running::uses_pending_accumulator_family(s) {
            return Err(backend_unavailable(
                "CUDA NIFS does not yet carry the authoritative block/lane pending-projection family; use the CPU prover",
            ));
        }
        let running_device_output = device_output_from_carrier(running_carrier);
        let running_accumulator_handle = running_device_output
            .as_ref()
            .map(|output| output.accumulator_digest_fields());
        let running_parent_digest = running_device_output
            .as_ref()
            .map(|output| digest::accumulator_ce_claim_digest(output.parent_authority()));
        crate::perf_timed!("session.params", {
            self.session.ensure_pp_uploaded(log)?;
        });
        crate::perf_timed!("session.kernels", {
            self.session.ensure_kernels_loaded()?;
        });
        let (fresh_claims, fresh_witnesses): (Vec<CcsClaim>, Vec<CcsWitness>) = fresh
            .into_iter()
            .map(|inst| (inst.claim, inst.witness))
            .unzip();

        // One planes buffer per fold, in engine witness order (fresh Zs then
        // running), shared by the Π_CCS Ajtai Y_eval and the Π_RLC mix. When
        // the previous fold's split planes were retained and match `running`,
        // only the fresh planes cross the bus.
        let all_witnesses: Vec<&Mat<F>> = fresh_witnesses
            .iter()
            .map(|w| &w.Z)
            .chain(running.witnesses.iter())
            .collect();
        let z_cols = all_witnesses[0].cols();
        let resident = self
            .session
            .take_cached_running_planes(running_device_output.as_deref(), running, z_cols)?;
        let fresh_input_commitments: Vec<Commitment> = fresh_claims.iter().map(|claim| claim.c.clone()).collect();
        let fresh_commitments = self
            .session
            .take_cached_fresh_commitments(&fresh_input_commitments);
        let fresh_planes = fresh_commitments.as_ref().and_then(|cached| {
            cached.planes.as_ref().filter(|planes| {
                planes.count == fresh_witnesses.len()
                    && planes.plane_len == z_cols * D
                    && planes.words.len() == fresh_witnesses.len() * z_cols * D
            })
        });
        let mut fold_planes = self.session.fold_planes.take();
        crate::perf_timed!("fold.ingest", {
            match (fresh_planes, &resident) {
                (Some(fresh), Some(cached)) => {
                    let kernels = self.session.kernels()?;
                    crate::ingest::compose_resident_fold_planes_into(
                        &self.session.device,
                        kernels.ring(),
                        &fresh.words,
                        &cached.planes,
                        &mut fold_planes,
                    )
                    .map_err(|_| backend_unavailable("resident fold planes composition failed"))?
                }
                (None, Some(cached)) => {
                    let kernels = self.session.kernels()?;
                    crate::ingest::compose_fold_planes_into(
                        &self.session.device,
                        kernels.ring(),
                        &all_witnesses[..fresh_witnesses.len()],
                        &cached.planes,
                        &mut fold_planes,
                    )
                    .map_err(|_| backend_unavailable("fold planes composition failed"))?
                }
                (_, None) => {
                    crate::ingest::upload_witness_planes_into(&self.session.device, &all_witnesses, &mut fold_planes)
                        .map_err(|_| backend_unavailable("witness planes upload failed"))?
                }
            }
        });
        let fold_planes = fold_planes.expect("fold planes staged");

        // 1. Π_CCS — device FE + NC sumcheck rounds, host transcript.
        // The FE backend owns mutable access to the static matrix caches
        // while the CPU engine drives it through trait callbacks; move them
        // in for the call and return them immediately after.
        let bar_matrices = self.session.bar_matrices.take();
        let row_matrices = self.session.row_matrices.take();
        let fe_phase_workspace = self.session.fe_phase_workspace.take();
        let fe_oracle_workspace = self.session.fe_oracle_workspace.take();
        let nc_oracle_workspace = self.session.nc_oracle_workspace.take();
        let mut nc_phase_workspace_in = self.session.nc_phase_workspace.take();
        let fe_ring_scratch = self.session.fe_ring_scratch.take();
        let pi_ccs_proof_state;
        let statics;
        let phase_workspace;
        let oracle_workspace;
        let nc_workspace;
        let nc_phase_workspace;
        let ring_scratch;
        let mut pi_ccs_y_eval_surface;
        let pi_ccs_nc_final_state;
        crate::perf_timed!("fold.superneo.pi_ccs", {
            let kernels = self.session.kernels()?;
            let fe_phase = self.fe_phase_mode.next_settings();
            let use_phase_backend = fe_phase.enabled && all_witnesses.len() > 1;
            if use_phase_backend {
                let mut phase_backend = DevicePiCcsPhaseBackend::new(&self.session.device, kernels);
                phase_backend.set_statics(bar_matrices, row_matrices);
                phase_backend.set_phase_workspace(fe_phase_workspace);
                phase_backend.set_oracle_workspace(fe_oracle_workspace);
                phase_backend.set_nc_oracle_workspace(nc_oracle_workspace);
                phase_backend.set_nc_phase_workspace(nc_phase_workspace_in.take());
                phase_backend.set_ring_scratch(fe_ring_scratch);
                phase_backend.set_witness_planes(&fold_planes, all_witnesses.len());
                phase_backend.set_running_surfaces(
                    running_device_output
                        .as_deref()
                        .map(DeviceFoldOutput::child_surfaces),
                );
                if fe_phase.recapture {
                    phase_backend.enable_whole_fe_trace_recapture_for_parity();
                } else if fe_phase.graph {
                    phase_backend.enable_whole_fe_graph_for_parity();
                } else {
                    phase_backend.enable_whole_fe_trace_for_parity();
                }
                if fe_phase.transcript_mode.replays() {
                    pi_ccs_proof_state =
                        PiCcsProofState::Ready(pi_ccs::prove_from_parts_with_phase_backend_and_transcript_mode(
                            tr,
                            pp,
                            s,
                            cache,
                            log,
                            &fresh_claims,
                            &fresh_witnesses,
                            running,
                            Some(&mut phase_backend),
                            None,
                            None,
                            fe_phase.transcript_mode,
                            running_parent_digest,
                            running_accumulator_handle,
                            None,
                        )?);
                    pi_ccs_y_eval_surface = phase_backend.take_last_y_eval_surface();
                    pi_ccs_nc_final_state = phase_backend.take_last_nc_final_state();
                } else {
                    let deferred_pi_ccs = pi_ccs::defer_from_parts_with_phase_backend_and_transcript_mode(
                        tr,
                        pp,
                        s,
                        cache,
                        log,
                        &fresh_claims,
                        &fresh_witnesses,
                        running,
                        &mut phase_backend,
                        fe_phase.transcript_mode,
                        running_parent_digest,
                        running_accumulator_handle,
                    )?;
                    pi_ccs_y_eval_surface = phase_backend.take_last_y_eval_surface();
                    pi_ccs_nc_final_state = phase_backend.take_last_nc_final_state();
                    pi_ccs_proof_state = match phase_backend.take_proof_log_exporter() {
                        Some(exporter) => PiCcsProofState::DeferredPhase {
                            proof: deferred_pi_ccs,
                            exporter,
                        },
                        None => PiCcsProofState::Ready(deferred_pi_ccs.finish_with_phase_backend(&mut phase_backend)?),
                    };
                }
                statics = phase_backend.take_statics();
                phase_workspace = phase_backend.take_phase_workspace();
                oracle_workspace = phase_backend.take_oracle_workspace();
                nc_workspace = phase_backend.take_nc_oracle_workspace();
                nc_phase_workspace = phase_backend.take_nc_phase_workspace();
                ring_scratch = phase_backend.take_ring_scratch();
            } else {
                let mut fe_backend = DeviceFeBackend::new(&self.session.device, kernels);
                fe_backend.set_statics(bar_matrices, row_matrices);
                fe_backend.set_phase_workspace(fe_phase_workspace);
                fe_backend.set_oracle_workspace(fe_oracle_workspace);
                fe_backend.set_ring_scratch(fe_ring_scratch);
                fe_backend.set_witness_planes(&fold_planes, all_witnesses.len());
                fe_backend.set_running_surfaces(
                    running_device_output
                        .as_deref()
                        .map(DeviceFoldOutput::child_surfaces),
                );
                let mut nc_backend = DeviceNcBackend::new(&self.session.device, kernels);
                nc_backend.set_oracle_workspace(nc_oracle_workspace);
                nc_backend.set_phase_workspace(nc_phase_workspace_in.take());
                nc_backend.set_witness_planes(&fold_planes, all_witnesses.len());
                pi_ccs_proof_state = if cache_output_for_next_step {
                    let proof = pi_ccs::defer_from_parts_with_device_backends_and_transcript_mode(
                        tr,
                        pp,
                        s,
                        cache,
                        log,
                        &fresh_claims,
                        &fresh_witnesses,
                        running,
                        &mut fe_backend,
                        Some(&mut nc_backend),
                        BackendTranscriptMode::DeviceSnapshot,
                        running_parent_digest,
                        running_accumulator_handle,
                    )?;
                    let archive = fe_backend
                        .archive_retained_row_rounds()
                        .map_err(|_| backend_unavailable("archive resident Pi_CCS FE proof log failed"))?
                        .ok_or_else(|| backend_unavailable("resident Pi_CCS FE proof log missing"))?;
                    PiCcsProofState::DeferredRows { proof, archive }
                } else {
                    PiCcsProofState::Ready(pi_ccs::prove_from_parts_with_backends_and_transcript_mode(
                        tr,
                        pp,
                        s,
                        cache,
                        log,
                        &fresh_claims,
                        &fresh_witnesses,
                        running,
                        Some(&mut fe_backend),
                        Some(&mut nc_backend),
                        BackendTranscriptMode::DeviceSnapshot,
                        running_parent_digest,
                        running_accumulator_handle,
                        None,
                    )?)
                };
                pi_ccs_y_eval_surface = fe_backend.take_last_y_eval_surface();
                pi_ccs_nc_final_state = nc_backend.take_last_final_state();
                statics = fe_backend.take_statics();
                phase_workspace = fe_backend.take_phase_workspace();
                oracle_workspace = fe_backend.take_oracle_workspace();
                nc_workspace = nc_backend.take_oracle_workspace();
                nc_phase_workspace = nc_backend.take_phase_workspace();
                ring_scratch = fe_backend.take_ring_scratch();
            }
        });
        (self.session.bar_matrices, self.session.row_matrices) = statics;
        self.session.fe_phase_workspace = phase_workspace;
        self.session.fe_oracle_workspace = oracle_workspace;
        self.session.nc_oracle_workspace = nc_workspace;
        self.session.nc_phase_workspace = nc_phase_workspace;
        self.session.fe_ring_scratch = ring_scratch;

        // 2. Π_RLC — device rho sampling, device witness mix, and
        // device-combined CE surfaces. The CUDA path enters here from its own
        // freshly generated Π_CCS output, so duplicate RLC shape scans stay in
        // verifier/parity gates rather than the online prover path.
        let mut device_rhos;
        let pending_sampling_end;
        let commitment_mix;
        let pi_ccs_output_count = fresh_claims.len() + running.claims.len();
        debug_assert_eq!(pi_ccs_output_count, pi_ccs_proof_state.output_count());
        let metadata = pi_ccs_proof_state.claim_shell_metadata()?;
        let include_y_zcol = metadata.has_y_zcol;
        let combined_m_in = metadata.m_in;
        let kernels = self
            .session
            .kernels
            .as_ref()
            .ok_or_else(|| backend_unavailable("CUDA kernels not loaded"))?;
        let pi_ccs_output_public_x = if running_device_output.is_some() {
            Some(
                DevicePublicX::pack_from_planes(
                    &self.session.device,
                    kernels.rlc(),
                    &fold_planes,
                    pi_ccs_output_count,
                    z_cols * D,
                    combined_m_in,
                )
                .map_err(|_| backend_unavailable("pack resident Pi_CCS public X failed"))?,
            )
        } else {
            None
        };
        let mut pi_ccs_k_surfaces = match pi_ccs_y_eval_surface.as_ref() {
            Some(y_eval) if !include_y_zcol || pi_ccs_nc_final_state.is_some() => Some(
                DevicePiCcsKSurfaces::pack(
                    &self.session.device,
                    kernels,
                    Some(y_eval),
                    if include_y_zcol {
                        pi_ccs_nc_final_state.as_ref()
                    } else {
                        None
                    },
                    D.next_power_of_two(),
                )
                .map_err(|_| backend_unavailable("pack resident Pi_CCS K surfaces failed"))?,
            ),
            _ => None,
        };
        let pi_ccs_output_commitments = if running_device_output.is_some() {
            let (words, kappa) = compose_resident_commitments(
                &self.session.device,
                kernels.ring(),
                &fresh_input_commitments,
                running.claims.len(),
                fresh_commitments.as_ref(),
                running_device_output.as_deref(),
            )?
            .ok_or_else(|| backend_unavailable("resident Pi_CCS output commitments unavailable"))?;
            Some(Arc::new(DeviceCommitments::new(
                Arc::clone(self.session.device.stream()),
                words,
                pi_ccs_output_count,
                D,
                kappa,
            )?))
        } else {
            None
        };
        if pi_ccs_output_commitments.is_some() && pi_ccs_k_surfaces.is_none() {
            return Err(backend_unavailable("resident Pi_CCS output surfaces unavailable"));
        }
        let (device_pi_ccs_outputs_digest, host_pi_ccs_outputs_digest) = match pi_ccs_proof_state.outputs_digest() {
            Some(digest) => (None, Some(digest)),
            None => {
                let surfaces = pi_ccs_k_surfaces
                    .as_ref()
                    .ok_or_else(|| backend_unavailable("deferred Pi_CCS output surfaces unavailable"))?;
                let shells = pi_ccs_proof_state
                    .outputs()
                    .iter()
                    .map(PiCcsOutputDigestShell::from_claim)
                    .collect::<Vec<_>>();
                (
                    Some(
                        DevicePiCcsOutputsDigest::compute_from_shells_with_cache(
                            &self.session.device,
                            kernels,
                            &mut self.session.sis,
                            &shells,
                            surfaces,
                        )
                        .map_err(|_| backend_unavailable("device Pi_CCS output digest failed"))?,
                    ),
                    None,
                )
            }
        };
        // The normal prover replays the complete post-rho projection
        // schedule. TerminalClaimsOnly is a throughput-only contract: the
        // verifier/non-timed parity path recomputes this prover self-check,
        // and neither its digest nor beta is serialized or consumed by DEC.
        let projection_inputs = if self.terminal_claims_only {
            None
        } else {
            Some(crate::projection::materialize_inputs(
                pi_ccs_proof_state.outputs(),
                pi_ccs_output_commitments.as_ref(),
                pi_ccs_output_public_x.as_ref(),
                pi_ccs_k_surfaces.as_ref(),
            )?)
        };
        let reusable_dec_split_planes;
        crate::perf_timed!("fold.superneo.pi_rlc.combine_claims", {
            let sampling_result;
            crate::perf_timed!("fold.superneo.pi_rlc.challenge_rhos", {
                let kernels = self.session.kernels()?;
                crate::perf_timed!("fold.superneo.pi_rlc.challenge_rhos.device", {
                    sampling_result = match device_pi_ccs_outputs_digest.as_ref() {
                        Some(outputs_digest) => {
                            pi_rlc::validate_rho_sampling_count(pp, pi_ccs_output_count)?;
                            device_rlc::sample_rhos_from_device_outputs_digest_deferred(
                                &self.session.device,
                                kernels,
                                pp,
                                tr.snapshot(),
                                outputs_digest.words(),
                                pi_ccs_output_count,
                            )
                            .map_err(|_| backend_unavailable("resident-digest Π_RLC rho sampling failed"))?
                        }
                        None => {
                            let sampling_start = pi_rlc::begin_rho_sampling_from_outputs_digest(
                                tr,
                                pp,
                                pi_ccs_output_count,
                                host_pi_ccs_outputs_digest
                                    .ok_or_else(|| backend_unavailable("host Pi_CCS output digest missing"))?,
                            )?;
                            device_rlc::sample_rhos_device_deferred(
                                &self.session.device,
                                kernels,
                                pp,
                                sampling_start,
                                pi_ccs_output_count,
                            )
                            .map_err(|_| backend_unavailable("device Π_RLC rho sampling failed"))?
                        }
                    };
                });
            });
            (device_rhos, pending_sampling_end) = sampling_result;
            crate::perf_timed!("fold.superneo.pi_rlc.commit_mix", {
                commitment_mix = if std::ptr::fn_addr_eq(mix_rhos_commits, ajtai_rlc_mixer as RlcMixer) {
                    match compose_resident_commitments(
                        &self.session.device,
                        kernels.ring(),
                        &fresh_input_commitments,
                        running.claims.len(),
                        fresh_commitments.as_ref(),
                        running_device_output.as_deref(),
                    )? {
                        Some((commitment_words, kappa)) => MixedCommitment::Pending(
                            device_rlc::enqueue_owned_mix_commitments_device_words(
                                std::sync::Arc::clone(self.session.device.stream()),
                                kernels.ring(),
                                device_rhos.coeffs(),
                                commitment_words,
                                pi_ccs_output_count,
                                kappa,
                            )
                            .map_err(|_| backend_unavailable("resident device Π_RLC commitment mix failed"))?,
                        ),
                        None => {
                            let rho_mats = device_rhos
                                .mats(&self.session.device, pp)
                                .map_err(|_| backend_unavailable("host Π_RLC rho materialization failed"))?;
                            let commitments =
                                authoritative_commitments(&fresh_claims, running, running_device_output.as_deref())?;
                            MixedCommitment::Ready(mix_rhos_commits(&rho_mats, &commitments))
                        }
                    }
                } else {
                    let rho_mats = device_rhos
                        .mats(&self.session.device, pp)
                        .map_err(|_| backend_unavailable("host Π_RLC rho materialization failed"))?;
                    let commitments =
                        authoritative_commitments(&fresh_claims, running, running_device_output.as_deref())?;
                    MixedCommitment::Ready(mix_rhos_commits(&rho_mats, &commitments))
                };
            });
            reusable_dec_split_planes = resident.map(|cached| cached.planes);
            crate::perf_timed!("fold.superneo.pi_rlc.claim_shell", {
                let _metadata = pi_ccs_proof_state.claim_shell_metadata()?;
            });
        });
        let (mut pi_ccs_y_eval_words, mut pi_ccs_dec_forms) = match pi_ccs_y_eval_surface.take() {
            Some(surface) => {
                let (words, forms) = surface.into_parts();
                (Some(words), forms)
            }
            None => (None, None),
        };
        let k_surface_stream = self
            .session
            .device
            .stream()
            .fork()
            .map_err(|_| backend_unavailable("fork Π_RLC K-surface stream failed"))?;
        let pending_k_surfaces;
        crate::perf_timed!("fold.superneo.pi_rlc.output.k_surfaces", {
            let pending_k_surfaces_result = if let Some(pi_ccs_k_surfaces) = pi_ccs_k_surfaces.take() {
                let branch_result;
                crate::perf_timed!("fold.superneo.pi_rlc.output.k_surfaces.device", {
                    branch_result = Some(device_rlc::enqueue_k_output_surfaces_from_device(
                        k_surface_stream,
                        kernels.rlc(),
                        device_rhos.coeffs(),
                        pi_ccs_k_surfaces,
                    ));
                });
                branch_result
            } else {
                let branch_result;
                crate::perf_timed!("fold.superneo.pi_rlc.output.k_surfaces.host_claims", {
                    branch_result = Some(device_rlc::enqueue_k_output_surfaces(
                        k_surface_stream,
                        kernels.rlc(),
                        device_rhos.coeffs(),
                        pi_ccs_proof_state.outputs(),
                        s.t(),
                        include_y_zcol,
                    ));
                });
                branch_result
            };
            pending_k_surfaces = pending_k_surfaces_result
                .ok_or_else(|| backend_unavailable("Π_RLC K-surface branch did not execute"))?
                .map_err(|_| backend_unavailable("enqueue device Π_RLC K-surface combine failed"))?;
        });
        let z_mix;
        crate::perf_timed!("fold.superneo.pi_rlc.mix_witness", {
            z_mix = device_rlc::mix_planes_device_with_rho_coeffs_retained(
                &self.session.device,
                kernels.ring(),
                device_rhos.coeffs(),
                &fold_planes,
                all_witnesses.len(),
                z_cols,
            )
            .map_err(|_| backend_unavailable("device Π_RLC witness mix failed"))?;
        });
        let deferred_parent_commitment = commitment_mix.is_pending();
        let mut combined;
        crate::perf_timed!("fold.superneo.pi_rlc.claim_shell", {
            let metadata = pi_ccs_proof_state.claim_shell_metadata()?;
            let commitment = commitment_mix.claim_shell_commitment(pp.kappa() as usize);
            combined = device_rlc::claim_shell_from_metadata(metadata, device_rhos.count(), commitment)
                .map_err(|_| backend_unavailable("device Π_RLC claim shell failed"))?;
            if pi_ccs_proof_state
                .outputs()
                .iter()
                .any(|claim| claim.adv.is_some())
            {
                let rho_mats = device_rhos
                    .mats(&self.session.device, pp)
                    .map_err(|_| backend_unavailable("Nebula lane rho materialization failed"))?;
                let input_advs = pi_ccs_proof_state
                    .outputs()
                    .iter()
                    .map(|claim| claim.adv.clone())
                    .collect::<Vec<_>>();
                combined.adv = mix_adv(mix_rhos_commits, &rho_mats, &input_advs)
                    .map_err(|_| backend_unavailable("Nebula lane presence mismatch in Π_RLC"))?;
            }
        });
        let x_stream = self
            .session
            .device
            .stream()
            .fork()
            .map_err(|_| backend_unavailable("fork Π_RLC X projection stream failed"))?;
        let pending_x;
        crate::perf_timed!("fold.superneo.pi_rlc.output.X", {
            pending_x = device_rlc::enqueue_project_x_from_mixed_witness(
                x_stream,
                kernels.rlc(),
                z_mix.words(),
                z_cols,
                s.m,
                combined_m_in,
            )
            .map_err(|_| backend_unavailable("device Π_RLC X projection failed"))?;
        });
        crate::perf_timed!("fold.superneo.pi_rlc.output.y_ring", {
            combined.y_ring = vec![vec![K::ZERO; D.next_power_of_two()]; s.t()];
            combined.ct = Vec::new();
        });
        crate::perf_timed!("fold.superneo.pi_rlc.output.y_zcol", {
            if include_y_zcol {
                combined.y_zcol = vec![K::ZERO; D.next_power_of_two()];
            }
        });
        if let Some(workspace) = self.session.fe_phase_workspace.as_mut() {
            if let Some(words) = pi_ccs_y_eval_words.take() {
                workspace.store_y_eval_words(words);
            }
        }

        // 3. Π_DEC — split, per-child eval + commit, self-checks on device.
        // The bar matrices are the same upload Π_CCS used above.
        let dec_output;
        let resident_dec_output = cache_output_for_next_step || self.terminal_claims_only;
        let dec_output_mode = if resident_dec_output {
            DecOutputMode::ResidentOnly
        } else {
            DecOutputMode::Full
        };
        crate::perf_timed!("fold.superneo.pi_dec", {
            let parts = self.session.dec_prover_parts()?;
            dec_output = parts
                .dec
                .prove(
                    parts.device,
                    parts.kernels,
                    parts.ajtai,
                    parts.bar_matrices,
                    pp,
                    s,
                    cache,
                    combine_b_pows,
                    &combined,
                    DecParentWitness::Device(z_mix.words()),
                    reusable_dec_split_planes,
                    pi_ccs_dec_forms.as_ref(),
                    dec_output_mode,
                    if resident_dec_output || deferred_parent_commitment {
                        DecRecompositionMode::DeferYAndXAndCommitment
                    } else {
                        DecRecompositionMode::DeferYAndX
                    },
                )
                .map_err(|_| backend_unavailable("device Π_DEC prove failed"))?;
        });
        if let (Some(workspace), Some(forms)) = (self.session.fe_phase_workspace.as_mut(), pi_ccs_dec_forms.take()) {
            workspace.store_forms(forms);
        }
        let mut children = dec_output.children;
        let mut pi_dec_proof = dec_output.proof;
        let split_planes = dec_output.split;
        let child_commitment_words = dec_output.child_commitment_words;
        let child_surfaces = dec_output.child_surfaces;
        let child_public_x = dec_output.child_public_x;
        let deferred_dec_status = dec_output.deferred_status;
        let device_child_commitments = if resident_dec_output {
            Some(Arc::new(DeviceCommitments::new(
                Arc::clone(self.session.device.stream()),
                child_commitment_words,
                children.claims.len(),
                D,
                pp.kappa() as usize,
            )?))
        } else {
            None
        };
        let (parent_surfaces, parent_public_x, parent_commitment) = if child_surfaces.is_some() {
            let surfaces =
                device_rlc::finish_k_output_surfaces_device(self.session.device.stream(), pending_k_surfaces)
                    .map_err(|_| backend_unavailable("retain device Π_RLC K surfaces failed"))?;
            let public_x = device_rlc::finish_projected_x_device(self.session.device.stream(), pending_x)
                .map_err(|_| backend_unavailable("retain device Π_RLC public X failed"))?;
            let commitment = commitment_mix.finish_device(&self.session.device)?;
            (Some(surfaces), Some(public_x), Some(commitment))
        } else {
            let k_surfaces = device_rlc::finish_k_output_surfaces(self.session.device.stream(), pending_k_surfaces)
                .map_err(|_| backend_unavailable("finish device Π_RLC K-surface combine failed"))?;
            combined.y_ring = k_surfaces.y_ring;
            combined.ct = ct_from_y_ring_for_ccs_m(&combined.y_ring, pp.inner(), s.m);
            if include_y_zcol {
                combined.y_zcol = k_surfaces.y_zcol;
            }
            combined.X = device_rlc::finish_projected_x(self.session.device.stream(), pending_x)
                .map_err(|_| backend_unavailable("finish device Π_RLC X projection failed"))?;
            let child_x = neo_reductions::split_b_matrix_k(&combined.X, pp.k_rho() as usize, pp.b())
                .map_err(|_| backend_unavailable("split deferred Π_DEC parent X failed"))?;
            if child_x.len() != children.claims.len() || child_x.len() != pi_dec_proof.children.len() {
                return Err(backend_unavailable("deferred Π_DEC public X child count mismatch"));
            }
            for ((claim, proof_claim), x) in children
                .claims
                .iter_mut()
                .zip(pi_dec_proof.children.iter_mut())
                .zip(child_x)
            {
                claim.X = x.clone();
                proof_claim.X = x;
            }
            combined.c = commitment_mix.finish(self.session.device.stream())?;
            crate::reduce::dec::verify_y_recomposition(&combined.y_ring, &children.claims, pp.b())
                .map_err(|_| backend_unavailable("deferred device Π_DEC y check failed"))?;
            crate::reduce::dec::verify_x_recomposition(&combined.X, &children.claims, pp.b())
                .map_err(|_| backend_unavailable("deferred device Π_DEC X check failed"))?;
            if deferred_parent_commitment {
                crate::reduce::dec::verify_commitment_recomposition(
                    &combined.c,
                    &children.claims,
                    combine_b_pows,
                    pp.b(),
                )
                .map_err(|_| backend_unavailable("deferred device Π_DEC commitment check failed"))?;
            }
            (None, None, None)
        };
        if combined.adv.is_some() {
            let lane_scheme = lanes.ok_or_else(|| backend_unavailable("Nebula parent requires a lane scheme"))?;
            let child_count = children.claims.len();
            let plane_stride = split_planes.planes().len() / child_count;
            let child_advs;
            crate::perf_timed!("fold.superneo.pi_dec.commit_child_lanes", {
                child_advs = self.session.commit_nebula_child_lanes(
                    lane_scheme,
                    split_planes.planes(),
                    child_count,
                    plane_stride,
                )?;
            });
            if child_advs.len() != child_count || pi_dec_proof.children.len() != child_count {
                return Err(backend_unavailable("Nebula child lane commitment count mismatch"));
            }
            for ((claim, proof_claim), adv) in children
                .claims
                .iter_mut()
                .zip(pi_dec_proof.children.iter_mut())
                .zip(child_advs)
            {
                claim.adv = Some(adv.clone());
                proof_claim.adv = Some(adv);
            }
            let recomposed = recompose_adv(
                combine_b_pows,
                pp.b(),
                &children
                    .claims
                    .iter()
                    .map(|claim| claim.adv.clone())
                    .collect::<Vec<_>>(),
            )
            .map_err(|_| backend_unavailable("Nebula child lane presence mismatch in Π_DEC"))?;
            if recomposed != combined.adv {
                return Err(backend_unavailable("Nebula child lane commitments do not recompose"));
            }
        }
        let projection_combined = crate::projection::materialize_parent(
            &combined,
            parent_surfaces.as_ref(),
            parent_public_x.as_ref(),
            parent_commitment.as_ref(),
        )?;

        // Accumulate: children become the new running instance; the split
        // planes are retained when the caller staged this output as the next
        // fold's running instance (byte-equal to the child witnesses' planes).
        let __egress_planes;
        crate::perf_timed!("fold.egress.retain_planes", {
            __egress_planes = if cache_output_for_next_step && !children.claims.is_empty() {
                let plane_len = split_planes.planes().len() / children.claims.len();
                Some(CachedRunningPlanes {
                    commitments: CachedDeviceCommitments {
                        host: children
                            .claims
                            .iter()
                            .map(|claim| claim.c.clone())
                            .collect(),
                        device: Arc::clone(
                            device_child_commitments
                                .as_ref()
                                .ok_or_else(|| backend_unavailable("resident DEC commitments missing"))?,
                        ),
                        planes: None,
                    },
                    plane_len,
                    planes: split_planes.into_planes(),
                })
            } else {
                None
            };
        });
        self.session.cached_running_planes = __egress_planes;

        let device_fold_output;
        let next_running;
        crate::perf_timed!("fold.accumulate.running", {
            if let Some(surfaces) = child_surfaces {
                device_fold_output = Some(Arc::new(DeviceFoldOutput::new(
                    surfaces,
                    Arc::clone(
                        device_child_commitments
                            .as_ref()
                            .ok_or_else(|| backend_unavailable("device fold-output commitments missing"))?,
                    ),
                    child_public_x.ok_or_else(|| backend_unavailable("device fold-output public X missing"))?,
                    children.claims,
                    projection_combined.clone(),
                    parent_surfaces.ok_or_else(|| backend_unavailable("device parent K surfaces missing"))?,
                    parent_commitment.ok_or_else(|| backend_unavailable("device parent commitment missing"))?,
                    parent_public_x.ok_or_else(|| backend_unavailable("device parent public X missing"))?,
                    deferred_dec_status.ok_or_else(|| backend_unavailable("resident DEC status authority missing"))?,
                )?));
                next_running = None;
            } else {
                device_fold_output = None;
                next_running = Some(RunningInstance::new(
                    children.claims,
                    children.witnesses,
                    Some(combined.clone()),
                    None,
                ));
            }
        });
        let proof_carrier;
        let restored_phase_workspaces;
        crate::perf_timed!("fold.egress.export", {
            let pending_outputs_digest = match device_pi_ccs_outputs_digest.as_ref() {
                Some(outputs_digest) => Some(
                    outputs_digest
                        .enqueue_download(&self.session.device)
                        .map_err(|_| backend_unavailable("resident Pi_CCS output digest export failed"))?,
                ),
                None => None,
            };
            let sampling_end = pending_sampling_end
                .finish(&self.session.device)
                .map_err(|_| backend_unavailable("device Π_RLC transcript restore failed"))?;
            tr.restore_snapshot(sampling_end);
            if let Some(projection_inputs) = projection_inputs.as_deref() {
                crate::perf_timed!("fold.superneo.pi_rlc.projection_binding", {
                    crate::projection::bind_schedule(
                        tr,
                        pp,
                        s,
                        mix_rhos_commits,
                        &mut self.session,
                        &mut device_rhos,
                        projection_inputs,
                        &projection_combined,
                    )?;
                });
            }
            let pi_ccs_outputs_digest = match pending_outputs_digest.as_ref() {
                Some(words) => DevicePiCcsOutputsDigest::decode_download(words.as_slice())
                    .map_err(|_| backend_unavailable("resident Pi_CCS output digest decode failed"))?,
                None => host_pi_ccs_outputs_digest
                    .ok_or_else(|| backend_unavailable("host Pi_CCS output digest missing at egress"))?,
            };
            let pi_ccs_output_authority = match (pi_ccs_output_commitments, pi_ccs_output_public_x) {
                (Some(commitments), Some(public_x)) => Some(DevicePiCcsOutputAuthority {
                    commitments,
                    public_x,
                    outputs_digest: pi_ccs_outputs_digest,
                }),
                (None, None) => None,
                _ => return Err(backend_unavailable("resident Pi_CCS output authority incomplete")),
            };
            let deferred_pi_ccs = match pi_ccs_proof_state {
                PiCcsProofState::DeferredRows { proof, archive } => {
                    restored_phase_workspaces = (None, None);
                    DeferredCudaPiCcsProof::rows(proof, archive, pi_ccs_output_authority)
                }
                state => {
                    let finished = state.finish()?;
                    restored_phase_workspaces = (finished.fe_phase_workspace, finished.nc_phase_workspace);
                    DeferredCudaPiCcsProof::ready(finished.proof, pi_ccs_output_authority)
                }
            };
            if let Some(output) = device_fold_output.as_ref() {
                proof_carrier = NifsProofCarrier::deferred(Arc::new(CudaDeferredNifsProof::new(
                    deferred_pi_ccs,
                    pi_rlc::Proof { combined },
                    pi_dec_proof,
                    Arc::clone(output),
                )));
            } else {
                let pi_ccs = deferred_pi_ccs.finish()?;
                proof_carrier = NifsProofCarrier::materialized(NifsProof {
                    pi_ccs,
                    pi_rlc: pi_rlc::Proof { combined },
                    pi_dec: pi_dec_proof,
                });
            }
        });
        if let Some(workspace) = restored_phase_workspaces.0 {
            let mut workspace = workspace;
            if let Some(words) = pi_ccs_y_eval_words.take() {
                workspace.store_y_eval_words(words);
            }
            self.session.fe_phase_workspace = Some(workspace);
        } else if let Some(workspace) = self.session.fe_phase_workspace.as_mut() {
            if let Some(words) = pi_ccs_y_eval_words.take() {
                workspace.store_y_eval_words(words);
            }
        }
        if let Some(workspace) = restored_phase_workspaces.1 {
            self.session.nc_phase_workspace = Some(workspace);
        }
        self.session.fold_planes = Some(fold_planes);
        let (running_carrier, post_summary) = if let Some(output) = device_fold_output {
            let summary = self.resident_post_fold_summary(&output);
            (
                NifsRunningCarrier::deferred(Arc::new(CudaRunningCarrier::new(output))),
                summary,
            )
        } else {
            let running = next_running.expect("materialized DEC output has a running instance");
            let summary = self.post_fold_summary(&running)?;
            (NifsRunningCarrier::materialized(running), summary)
        };
        let output = NifsProverOutput::deferred(running_carrier, proof_carrier);
        Ok(output.with_post_fold_summary(post_summary))
    }

    /// This adapter produced the fold it would be re-verifying and emits the
    /// same proof into the audit; the recursive-compile NIFS.V replay is a
    /// prover-side sanity check with no verifier-semantics effect (the trait
    /// documents this opt-out), so skip its ~30ms/chunk.
    fn requires_recursive_compile_reverify(&self) -> bool {
        false
    }

    /// GPU fresh-instance build. Must stay field-identical to
    /// `CcsInstance::from_low_norm_assignment`; any input the CPU
    /// constructor would reject falls back to it via `Ok(None)`.
    fn build_fresh_instances(
        &mut self,
        request: NifsFreshInstancesRequest<'_>,
    ) -> Result<Option<Vec<CcsInstance>>, Error> {
        crate::commit::build_fresh_instances(&mut self.session, request)
    }
}
