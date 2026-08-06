//! Native proving and verification for one Construction-2 `F'` step.
//!
//! Owns: the step transcript prefix, base/recursive native control flow, NIFS
//! invocation, state advance, and public-output verification.
//!
//! Does not own: encoded-image construction, R1CS constraints, or NIFS internals.
//!
//! Emits constraints: no.
//!
//! Authority boundary: [`verify`] recomputes transcript-derived checks, validates
//! the NIFS transition, and derives the public output; prover-returned state is
//! never authority by itself.
//!
//! | Obligation | Local owner | Emits constraints? | Authority source |
//! |---|---|---|---|
//! | Transcript context | [`f_prime_step_transcript`] | no | Verifier key and prior state |
//! | Prover transition | [`prove`] and backend variants | no | Native witness plus NIFS prover |
//! | Verifier transition | [`verify`] | no | Replayed transcript and checked NIFS proof |

use neo_ajtai::AjtaiSModule;
use neo_math::F;
use neo_reductions::optimized_engine::OptimizedStructureCache;
use p3_field::PrimeCharacteristicRing;

use crate::engine::transcript::{Poseidon2TranscriptSnapshot, Transcript};
use crate::paper::construction2::{
    self, EncInst, FoldProof, LaneCommitmentMode, LatestInstance, NebulaAdvance, NebulaLane, ProofState,
    RunningInstance, SemanticStateAdvance, SemanticStateMode, State, StepProof, VerifierKey,
};
use crate::paper::digest::digest32_as_fields;
use crate::paper::nifs;
use crate::paper::params::Params;
use crate::paper::relations::{CcsClaim, CcsInstance, DecMixer, LaneScheme, RlcMixer, Structure};

pub use construction2::Error;

/// Canonical transcript label for one F' step.
///
/// Used by `paper::f_prime::{prove, verify}` (native) and must match the
/// `cfg.transcript_label` of [`crate::paper::f_prime::r1cs::FPrimeStepConfig`]
/// when the in-circuit F' R1CS verifies the same step. Both sides initialize
/// their transcript with this label and absorb the state-bound F'-step
/// context below before NIFS.V; if either diverges, Fiat–Shamir challenges
/// disagree at the first absorb and the F' R1CS rejects.
pub const F_PRIME_STEP_TRANSCRIPT_LABEL: &[u8] = b"neo.fold.clean/f_prime/step/v1";

/// Stage reached by the actual public native verifier execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifyStepExecutionStage {
    Entry,
    ChunkDigest,
    Dispatch,
    Nifs,
    Nebula,
    Advance,
    Semantic,
    XOut,
    Complete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifyStepDispatch {
    InitialNoFold,
    InitialRecursive,
    ActiveNoFold,
    ActiveRecursive,
}

#[derive(Clone, Debug)]
pub enum VerifyStepExecutionProofState {
    Initial,
    Active {
        running: RunningInstance,
        latest_claims: Vec<CcsClaim>,
    },
}

/// Verifier-public state input with all prover witnesses removed.
#[derive(Clone, Debug)]
pub struct VerifyStepExecutionState {
    pub chunk_count: u64,
    pub step_count: u64,
    pub z_0: [u8; 32],
    pub z_i: [u8; 32],
    pub pc: u64,
    pub initial_semantic_state_digest: [u8; 32],
    pub semantic_state_digest: [u8; 32],
    pub acc_digest: [u8; 32],
    pub public_trace: [u8; 32],
    pub proof: VerifyStepExecutionProofState,
    pub nebula: Option<NebulaLane>,
}

#[derive(Clone, Debug)]
pub enum VerifyStepExecutionFoldProof {
    NoFold,
    Recursive(nifs::NifsProof),
}

#[derive(Clone, Debug)]
pub struct VerifyStepExecutionProof {
    pub fold: VerifyStepExecutionFoldProof,
    pub nebula_open: Option<[[F; 4]; 3]>,
    pub semantic_state_digest: [u8; 32],
    pub x_out: EncInst,
}

/// Exact verifier-public inputs supplied to one audited native invocation.
#[derive(Clone, Debug)]
pub struct VerifyStepExecutionInput {
    pub state: VerifyStepExecutionState,
    pub next_latest_claims: Vec<CcsClaim>,
    pub proof: VerifyStepExecutionProof,
    pub semantic_mode: SemanticStateMode,
    pub nebula_advance: Option<NebulaAdvance>,
}

/// Result observed at the one actual NIFS verifier call.
#[derive(Clone, Debug)]
pub enum NifsVerifyExecutionOutcome {
    Accepted(RunningInstance),
    Rejected,
}

/// Public-data operations performed by the native verifier, in execution
/// order. No witness matrices are copied into these events.
#[derive(Clone, Debug)]
pub enum VerifyStepExecutionEvent {
    ChunkDigest {
        start_index: u64,
        claims: Vec<CcsClaim>,
        output: [F; 4],
    },
    Dispatch {
        branch: VerifyStepDispatch,
    },
    TranscriptStarted {
        label: &'static [u8],
    },
    TranscriptAppend {
        label: &'static [u8],
        fields: Vec<F>,
    },
    TranscriptPrefix {
        snapshot: Poseidon2TranscriptSnapshot,
    },
    NifsVerify {
        running: RunningInstance,
        fresh_claims: Vec<CcsClaim>,
        proof: nifs::NifsProof,
        outcome: NifsVerifyExecutionOutcome,
    },
    RunningDigest {
        running: RunningInstance,
        relation_columns: usize,
        output: [u8; 32],
    },
    StateAdvanced {
        output: State,
    },
    VerifierDigestRead {
        output: [u8; 32],
    },
    PiCcsHeaderRead {
        output: [F; 4],
    },
    NebulaDigest {
        lane: NebulaLane,
        output: [F; 4],
    },
    StateXOutHash {
        preimage: Vec<F>,
        output_digest: [u8; 32],
        output: EncInst,
    },
}

/// Audit-only execution result. The `result` is the exact typed result
/// returned by the same core control flow that populated `events`.
#[derive(Debug)]
pub struct VerifyStepExecutionReceipt {
    pub input: VerifyStepExecutionInput,
    pub events: Vec<VerifyStepExecutionEvent>,
    pub final_stage: VerifyStepExecutionStage,
    pub result: Result<State, Error>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifyStepExecutionReceiptError {
    DeferredRunningInput,
    DeferredProofInput,
}

fn execution_input(
    state: &State,
    next_latest_claims: &[CcsClaim],
    proof: &StepProof,
    semantic_mode: SemanticStateMode,
    nebula_advance: Option<&NebulaAdvance>,
) -> Result<VerifyStepExecutionInput, VerifyStepExecutionReceiptError> {
    let state_proof = match &state.proof {
        ProofState::Initial => VerifyStepExecutionProofState::Initial,
        ProofState::Active { running, latest } => {
            let running = running
                .as_materialized()
                .ok_or(VerifyStepExecutionReceiptError::DeferredRunningInput)?;
            VerifyStepExecutionProofState::Active {
                running: running.claims_only(),
                latest_claims: latest.claims(),
            }
        }
    };
    let fold = match &proof.fold {
        FoldProof::NoFold => VerifyStepExecutionFoldProof::NoFold,
        FoldProof::Recursive(carrier) => match carrier {
            nifs::NifsProofCarrier::Materialized(proof) => VerifyStepExecutionFoldProof::Recursive(proof.clone()),
            nifs::NifsProofCarrier::Deferred(_) => {
                return Err(VerifyStepExecutionReceiptError::DeferredProofInput);
            }
        },
    };
    Ok(VerifyStepExecutionInput {
        state: VerifyStepExecutionState {
            chunk_count: state.chunk_count,
            step_count: state.step_count,
            z_0: state.z_0,
            z_i: state.z_i,
            pc: state.pc,
            initial_semantic_state_digest: state.initial_semantic_state_digest,
            semantic_state_digest: state.semantic_state_digest,
            acc_digest: state.acc_digest,
            public_trace: state.public_trace,
            proof: state_proof,
            nebula: state.nebula.clone(),
        },
        next_latest_claims: next_latest_claims.to_vec(),
        proof: VerifyStepExecutionProof {
            fold,
            nebula_open: proof.nebula_open.clone(),
            semantic_state_digest: proof.semantic_state_digest,
            x_out: proof.x_out.clone(),
        },
        semantic_mode,
        nebula_advance: nebula_advance.cloned(),
    })
}

trait VerifyStepRecorder: construction2::transition::VerifyTransitionRecorder {
    fn stage(&mut self, _stage: VerifyStepExecutionStage) {}

    fn chunk_digest(&mut self, _start_index: u64, _claims: &[CcsClaim], _output: [F; 4]) {}

    fn dispatch(&mut self, _branch: VerifyStepDispatch) {}

    fn transcript_started(&mut self, _label: &'static [u8]) {}

    fn transcript_append(&mut self, _label: &'static [u8], _fields: &[F]) {}

    fn transcript_prefix(&mut self, _snapshot: Poseidon2TranscriptSnapshot) {}

    fn nifs_verify(
        &mut self,
        _running: &RunningInstance,
        _fresh_claims: &[CcsClaim],
        _proof: &nifs::NifsProof,
        _outcome: Result<&RunningInstance, &nifs::Error>,
    ) {
    }
}

struct NoopVerifyStepRecorder;

impl construction2::transition::VerifyTransitionRecorder for NoopVerifyStepRecorder {}
impl VerifyStepRecorder for NoopVerifyStepRecorder {}

struct ReceiptRecorder {
    events: Vec<VerifyStepExecutionEvent>,
    final_stage: VerifyStepExecutionStage,
}

impl ReceiptRecorder {
    fn new() -> Self {
        Self {
            events: Vec::new(),
            final_stage: VerifyStepExecutionStage::Entry,
        }
    }
}

impl construction2::transition::VerifyTransitionRecorder for ReceiptRecorder {
    fn running_digest(&mut self, running: &RunningInstance, relation_columns: usize, output: [u8; 32]) {
        self.events.push(VerifyStepExecutionEvent::RunningDigest {
            running: running.claims_only(),
            relation_columns,
            output,
        });
    }

    fn state_advanced(&mut self, output: &State) {
        self.events
            .push(VerifyStepExecutionEvent::StateAdvanced { output: output.clone() });
    }

    fn verifier_digest_read(&mut self, output: [u8; 32]) {
        self.events
            .push(VerifyStepExecutionEvent::VerifierDigestRead { output });
    }

    fn pi_ccs_header_read(&mut self, output: [F; 4]) {
        self.events
            .push(VerifyStepExecutionEvent::PiCcsHeaderRead { output });
    }

    fn nebula_digest(&mut self, lane: &NebulaLane, output: [F; 4]) {
        self.events.push(VerifyStepExecutionEvent::NebulaDigest {
            lane: lane.clone(),
            output,
        });
    }

    fn state_x_out_hash(&mut self, preimage: &[F], output_digest: [u8; 32], output: &EncInst) {
        self.events.push(VerifyStepExecutionEvent::StateXOutHash {
            preimage: preimage.to_vec(),
            output_digest,
            output: output.clone(),
        });
    }
}

impl VerifyStepRecorder for ReceiptRecorder {
    fn stage(&mut self, stage: VerifyStepExecutionStage) {
        self.final_stage = stage;
    }

    fn chunk_digest(&mut self, start_index: u64, claims: &[CcsClaim], output: [F; 4]) {
        self.events.push(VerifyStepExecutionEvent::ChunkDigest {
            start_index,
            claims: claims.to_vec(),
            output,
        });
    }

    fn dispatch(&mut self, branch: VerifyStepDispatch) {
        self.events
            .push(VerifyStepExecutionEvent::Dispatch { branch });
    }

    fn transcript_started(&mut self, label: &'static [u8]) {
        self.events
            .push(VerifyStepExecutionEvent::TranscriptStarted { label });
    }

    fn transcript_append(&mut self, label: &'static [u8], fields: &[F]) {
        self.events
            .push(VerifyStepExecutionEvent::TranscriptAppend {
                label,
                fields: fields.to_vec(),
            });
    }

    fn transcript_prefix(&mut self, snapshot: Poseidon2TranscriptSnapshot) {
        self.events
            .push(VerifyStepExecutionEvent::TranscriptPrefix { snapshot });
    }

    fn nifs_verify(
        &mut self,
        running: &RunningInstance,
        fresh_claims: &[CcsClaim],
        proof: &nifs::NifsProof,
        outcome: Result<&RunningInstance, &nifs::Error>,
    ) {
        let outcome = match outcome {
            Ok(output) => NifsVerifyExecutionOutcome::Accepted(output.claims_only()),
            Err(_) => NifsVerifyExecutionOutcome::Rejected,
        };
        self.events.push(VerifyStepExecutionEvent::NifsVerify {
            running: running.claims_only(),
            fresh_claims: fresh_claims.to_vec(),
            proof: proof.clone(),
            outcome,
        });
    }
}

/// Absorb the F'-step context into a transcript.
///
/// Order is fixed and matches `enforce_f_prime_recursive_step_circuit` in
/// `paper::f_prime::r1cs`; do not reorder without updating the in-circuit
/// transcript prefix as well. `structure_digest` is the caller's cached
/// `paper::digest::structure_digest(&prep.structure)` value.
fn append_transcript_fields<R: VerifyStepRecorder>(
    tr: &mut Transcript,
    recorder: &mut R,
    label: &'static [u8],
    fields: &[F],
) {
    tr.append_fields(label, fields);
    recorder.transcript_append(label, fields);
}

fn absorb_f_prime_step_context<R: VerifyStepRecorder>(
    tr: &mut Transcript,
    vk: &VerifierKey,
    _structure_digest: &[F; 4],
    state: &State,
    chunk_digest: [F; 4],
    recorder: &mut R,
) {
    append_transcript_fields(tr, recorder, b"f_prime/vk_fs", &digest32_as_fields(vk.digest()));
    append_transcript_fields(tr, recorder, b"f_prime/pi_ccs_header", &vk.pi_ccs_header_bundle());
    append_transcript_fields(
        tr,
        recorder,
        b"f_prime/chunk_count_in",
        &[F::from_u64(state.chunk_count)],
    );
    append_transcript_fields(tr, recorder, b"f_prime/step_count_in", &[F::from_u64(state.step_count)]);
    append_transcript_fields(tr, recorder, b"f_prime/z_0", &digest32_as_fields(state.z_0));
    append_transcript_fields(tr, recorder, b"f_prime/z_i_in", &digest32_as_fields(state.z_i));
    append_transcript_fields(tr, recorder, b"f_prime/pc", &[F::from_u64(state.pc)]);
    append_transcript_fields(
        tr,
        recorder,
        b"f_prime/semantic_state_in",
        &digest32_as_fields(state.semantic_state_digest),
    );
    append_transcript_fields(
        tr,
        recorder,
        b"f_prime/acc_digest_in",
        &digest32_as_fields(state.acc_digest),
    );
    append_transcript_fields(
        tr,
        recorder,
        b"f_prime/public_trace_in",
        &digest32_as_fields(state.public_trace),
    );
    // Present-only (the carried-lane rules): plain chains keep the pre-Nebula absorb
    // sequence, so the in-circuit transcript prefix stays in parity until
    // the F′ R1CS carries the lane (the F′ lane-transition contract).
    if let Some(lane) = &state.nebula {
        append_transcript_fields(tr, recorder, b"f_prime/nebula_lane_in", &lane.digest());
    }
    append_transcript_fields(tr, recorder, b"f_prime/chunk_digest", &chunk_digest);
}

/// Build a fresh per-step F' transcript, initialized with
/// [`F_PRIME_STEP_TRANSCRIPT_LABEL`] and the F'-step context absorbs.
///
/// `state` is the state **input to this step** (i.e. before `advance_state`
/// runs), so its `z_i`, `public_trace`, etc. match the F' R1CS's `state-in`
/// fields. `chunk_digest` is computed from `next_latest`, the new batch
/// being deposited as `latest` (not the `latest` currently being folded).
/// `structure_digest` is the caller's cached
/// `paper::digest::structure_digest(&prep.structure)`.
pub fn f_prime_step_transcript(
    vk: &VerifierKey,
    structure_digest: &[F; 4],
    state: &State,
    chunk_digest: [F; 4],
) -> Transcript {
    let mut recorder = NoopVerifyStepRecorder;
    f_prime_step_transcript_recorded(vk, structure_digest, state, chunk_digest, &mut recorder)
}

fn f_prime_step_transcript_recorded<R: VerifyStepRecorder>(
    vk: &VerifierKey,
    structure_digest: &[F; 4],
    state: &State,
    chunk_digest: [F; 4],
    recorder: &mut R,
) -> Transcript {
    let mut tr = Transcript::with_label(F_PRIME_STEP_TRANSCRIPT_LABEL);
    recorder.transcript_started(F_PRIME_STEP_TRANSCRIPT_LABEL);
    absorb_f_prime_step_context(&mut tr, vk, structure_digest, state, chunk_digest, recorder);
    tr
}

// ──────────────────────────────────────────────────────────────────────────
// F' prove (native)
// ──────────────────────────────────────────────────────────────────────────

/// One full F' invocation on the prover side.
///
/// Reads `state.proof` to decide base-vs-recursive case:
/// - **Initial** (i = 0): no NIFS.P runs. `next_running` is the fixed
///   Construction-2 default accumulator.
///   `FoldProof::NoFold`.
/// - **Active** (i ≥ 1): NIFS.P folds `state.proof.latest` into
///   `state.proof.running`. `FoldProof::Recursive(π)`.
///
/// The new `next_latest` (the K instances the *next* step will fold) comes
/// from the caller. In strict Construction 2 (PR5+) it would be synthesized
/// internally from the F' encoder; in the direct-CCS interim it's the
/// caller's batch of CcsInstances.
pub fn prove(
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    structure_digest: &[F; 4],
    log: &AjtaiSModule,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    vk: &VerifierKey,
    state: State,
    next_latest: Vec<CcsInstance>,
) -> Result<(State, StepProof), Error> {
    prove_with_semantic_state(
        pp,
        s,
        cache,
        structure_digest,
        log,
        mix_rhos_commits,
        combine_b_pows,
        vk,
        state,
        next_latest,
        SemanticStateAdvance::Stateless,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prove_with_semantic_state(
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    structure_digest: &[F; 4],
    log: &AjtaiSModule,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    vk: &VerifierKey,
    state: State,
    next_latest: Vec<CcsInstance>,
    semantic_advance: SemanticStateAdvance,
    lanes: Option<&LaneScheme>,
    nebula_advance: Option<NebulaAdvance>,
) -> Result<(State, StepProof), Error> {
    let (state, proof, _) = prove_with_nifs_prover_and_semantic_state(
        NifsProverSource::Cpu,
        pp,
        s,
        cache,
        structure_digest,
        log,
        mix_rhos_commits,
        combine_b_pows,
        vk,
        state,
        next_latest,
        semantic_advance,
        lanes,
        nebula_advance,
    )?;
    Ok((state, proof))
}

#[allow(clippy::too_many_arguments)]
pub fn prove_with_adapter_and_semantic_state(
    adapter: &mut dyn nifs::NifsProverAdapter,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    structure_digest: &[F; 4],
    log: &AjtaiSModule,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    vk: &VerifierKey,
    state: State,
    next_latest: Vec<CcsInstance>,
    semantic_advance: SemanticStateAdvance,
    lanes: Option<&LaneScheme>,
    nebula_advance: Option<NebulaAdvance>,
) -> Result<(State, StepProof), Error> {
    let (state, proof, _) = prove_with_adapter_output_and_semantic_state(
        adapter,
        pp,
        s,
        cache,
        structure_digest,
        log,
        mix_rhos_commits,
        combine_b_pows,
        vk,
        state,
        next_latest,
        semantic_advance,
        lanes,
        nebula_advance,
    )?;
    Ok((state, proof))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_with_adapter_output_and_semantic_state(
    adapter: &mut dyn nifs::NifsProverAdapter,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    structure_digest: &[F; 4],
    log: &AjtaiSModule,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    vk: &VerifierKey,
    state: State,
    next_latest: Vec<CcsInstance>,
    semantic_advance: SemanticStateAdvance,
    lanes: Option<&LaneScheme>,
    nebula_advance: Option<NebulaAdvance>,
) -> Result<(State, StepProof, Option<nifs::NifsPostFoldSummary>), Error> {
    prove_with_nifs_prover_and_semantic_state(
        NifsProverSource::Adapter(adapter),
        pp,
        s,
        cache,
        structure_digest,
        log,
        mix_rhos_commits,
        combine_b_pows,
        vk,
        state,
        next_latest,
        semantic_advance,
        lanes,
        nebula_advance,
    )
}

enum NifsProverSource<'a> {
    Cpu,
    Adapter(&'a mut dyn nifs::NifsProverAdapter),
}

fn materialize_running_for_nifs_input(running: &nifs::NifsRunningCarrier) -> Result<RunningInstance, Error> {
    // NIFS.P still accepts a concrete `RunningInstance`, but accelerator
    // backends may satisfy this boundary with a claim shell while keeping
    // prover-private witnesses in backend-owned memory.
    Ok(running.materialize_prover_input()?)
}

#[allow(clippy::too_many_arguments)]
fn prove_with_nifs_prover_and_semantic_state(
    mut nifs_prover: NifsProverSource<'_>,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    structure_digest: &[F; 4],
    log: &AjtaiSModule,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    vk: &VerifierKey,
    state: State,
    next_latest: Vec<CcsInstance>,
    semantic_advance: SemanticStateAdvance,
    lanes: Option<&LaneScheme>,
    nebula_advance: Option<NebulaAdvance>,
) -> Result<(State, StepProof, Option<nifs::NifsPostFoldSummary>), Error> {
    let semantic_mode = match semantic_advance {
        SemanticStateAdvance::Stateless => SemanticStateMode::Stateless,
        SemanticStateAdvance::Stateful(digest) => {
            construction2::validate_digest32("semantic_state_advance", digest)?;
            SemanticStateMode::Stateful
        }
    };
    construction2::enforce_pc_in_range(&state)?;
    construction2::state_base_case_check(&state)?;
    construction2::validate_state_authority(vk, s, &state, semantic_mode)?;
    if next_latest.is_empty() {
        return Err(Error::EmptyStep);
    }

    let fresh_count = next_latest.len() as u64;
    let chunk_digest = construction2::f_prime_chunk_public_digest_for_step(state.step_count, &next_latest);

    // Destructure proof out of state up front so the rest can move the
    // remaining fields freely.
    let State {
        chunk_count,
        step_count,
        z_0,
        z_i,
        pc,
        initial_semantic_state_digest,
        semantic_state_digest,
        acc_digest,
        public_trace,
        proof: prev_proof,
        nebula,
    } = state;

    // F' fold step — branch on the tagged ProofState.
    let (next_running, fold, post_fold_summary) = match prev_proof {
        ProofState::Initial => {
            // HyperNova Construction 2, base case: no NIFS.P runs, but U_1
            // is the fixed vector of default satisfying R_1 instances.
            let m_in = next_latest
                .first()
                .expect("nonempty next_latest was checked above")
                .claim
                .m_in;
            let running = RunningInstance::canonical_zero(
                pp,
                s,
                m_in,
                LaneCommitmentMode::from_nebula(nebula_advance.is_some()),
            )?;
            (nifs::NifsRunningCarrier::materialized(running), FoldProof::NoFold, None)
        }
        ProofState::Active { running, latest } => {
            // Fresh per-step F' transcript: init label + state-in context
            // absorbs that the in-circuit F' R1CS replays bit-for-bit.
            let state_in = State {
                chunk_count,
                step_count,
                z_0,
                z_i,
                pc,
                initial_semantic_state_digest,
                semantic_state_digest,
                acc_digest,
                public_trace,
                proof: ProofState::Initial,
                nebula: nebula.clone(),
            };
            let mut tr = f_prime_step_transcript(vk, structure_digest, &state_in, chunk_digest);
            let running_input = materialize_running_for_nifs_input(&running)?;
            let (next_running, proof_carrier, post_fold_summary) = match &mut nifs_prover {
                NifsProverSource::Cpu => {
                    let (running, proof) = nifs::prove(
                        &mut tr,
                        pp,
                        s,
                        cache,
                        log,
                        lanes,
                        mix_rhos_commits,
                        combine_b_pows,
                        latest.instances,
                        &running_input,
                    )?;
                    (
                        nifs::NifsRunningCarrier::materialized(running),
                        nifs::NifsProofCarrier::materialized(proof),
                        None,
                    )
                }
                NifsProverSource::Adapter(adapter) => {
                    adapter.begin_f_prime_step(nifs::NifsFPrimeStepContext {
                        vk_fs_digest: vk.digest(),
                        structure_digest: *structure_digest,
                        chunk_count,
                        step_count,
                        z_0,
                        z_i,
                        pc,
                        semantic_state_digest,
                        acc_digest,
                        public_trace,
                        chunk_digest,
                    });
                    let output = nifs::prove_with_adapter_output_from_carrier(
                        *adapter,
                        &mut tr,
                        pp,
                        s,
                        cache,
                        log,
                        lanes,
                        mix_rhos_commits,
                        combine_b_pows,
                        latest.instances,
                        &running,
                        &running_input,
                    )?;
                    let (running, proof, post_summary) = output.into_carriers_with_summary();
                    (running, proof, post_summary)
                }
            };
            (
                next_running,
                FoldProof::recursive_carrier(proof_carrier),
                post_fold_summary,
            )
        }
    };
    let post_acc_digest_override = post_fold_summary
        .as_ref()
        .and_then(|summary| summary.acc_digest_override());

    // Build next ProofState: running advances, latest is what the caller just supplied.
    let new_proof = ProofState::active_carrier(next_running, LatestInstance::from_instances(next_latest));

    // F' steps 1, 2, 5 — advance state and compute x_out.
    let prev_state_for_advance = State {
        chunk_count,
        step_count,
        z_0,
        z_i,
        pc,
        initial_semantic_state_digest,
        semantic_state_digest,
        acc_digest,
        public_trace,
        proof: ProofState::Initial, // placeholder; advance_state reads new_proof for the new state
        nebula: nebula.clone(),
    };
    let nebula_open = nebula_advance.as_ref().and_then(|adv| adv.open);
    let next_state = construction2::advance_state_with_acc_digest(
        pp,
        s,
        prev_state_for_advance,
        new_proof,
        fresh_count,
        chunk_digest,
        semantic_advance,
        post_acc_digest_override,
        nebula_advance.map(|adv| adv.lane_out),
    )?;
    let x_out = construction2::compute_x_out(vk, pp, structure_digest, &next_state, semantic_mode);
    let semantic_state_digest = next_state.semantic_state_digest;

    Ok((
        next_state,
        StepProof {
            fold,
            nebula_open,
            semantic_state_digest,
            x_out,
        },
        post_fold_summary,
    ))
}

// ──────────────────────────────────────────────────────────────────────────
// F' verify (native)
// ──────────────────────────────────────────────────────────────────────────

/// One full F' invocation on the verifier side.
///
/// Mirrors `prove`:
/// - Reads `state.proof` to decide base-vs-recursive.
/// - For recursive: replays NIFS.V over `proof.fold` against
///   `proof.folded_claims` and the running claims in `state.proof`.
/// - Advances state, recomputes x_out, asserts it matches `proof.x_out`.
///
/// `semantic_mode` is verifier-owned (set on `Preprocessing` at
/// preprocess time, derived from the frontend plan). For
/// [`SemanticStateMode::Stateless`] chains the F' image's CCS
/// structure has no Poseidon2 binding rows for the semantic lane, so
/// `proof.semantic_state_digest` is **not** authenticated by the
/// in-circuit verifier; this function instead enforces the protocol
/// invariant `proof.semantic_state_digest == new_acc_digest` directly
/// and returns [`Error::StatelessSemanticInvariantViolated`] on
/// mismatch. For [`SemanticStateMode::Stateful`] chains the binding
/// rows are part of the structure, so the digest is authenticated
/// inductively by terminal Π_CCS sumcheck and this function trusts
/// `proof.semantic_state_digest` as the new chain coordinate.
pub fn verify(
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    structure_digest: &[F; 4],
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    vk: &VerifierKey,
    state: State,
    next_latest_claims: &[CcsClaim],
    proof: &StepProof,
    semantic_mode: SemanticStateMode,
    nebula_advance: Option<NebulaAdvance>,
) -> Result<State, Error> {
    let mut recorder = NoopVerifyStepRecorder;
    verify_core(
        pp,
        s,
        cache,
        structure_digest,
        mix_rhos_commits,
        combine_b_pows,
        vk,
        state,
        next_latest_claims,
        proof,
        semantic_mode,
        nebula_advance,
        &mut recorder,
    )
}

/// Execute the exact public native verifier control flow while recording its
/// public operations at their real call sites.
///
/// This is an audit/differential-testing entry point. It does not export
/// witnesses and does not turn any digest into semantic authority.
#[allow(clippy::too_many_arguments)]
pub fn verify_with_execution_receipt(
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    structure_digest: &[F; 4],
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    vk: &VerifierKey,
    state: State,
    next_latest_claims: &[CcsClaim],
    proof: &StepProof,
    semantic_mode: SemanticStateMode,
    nebula_advance: Option<NebulaAdvance>,
) -> Result<VerifyStepExecutionReceipt, VerifyStepExecutionReceiptError> {
    let input = execution_input(
        &state,
        next_latest_claims,
        proof,
        semantic_mode,
        nebula_advance.as_ref(),
    )?;
    let mut recorder = ReceiptRecorder::new();
    let result = verify_core(
        pp,
        s,
        cache,
        structure_digest,
        mix_rhos_commits,
        combine_b_pows,
        vk,
        state,
        next_latest_claims,
        proof,
        semantic_mode,
        nebula_advance,
        &mut recorder,
    );
    Ok(VerifyStepExecutionReceipt {
        input,
        events: recorder.events,
        final_stage: recorder.final_stage,
        result,
    })
}

#[allow(clippy::too_many_arguments)]
fn verify_core<R: VerifyStepRecorder>(
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    structure_digest: &[F; 4],
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    vk: &VerifierKey,
    state: State,
    next_latest_claims: &[CcsClaim],
    proof: &StepProof,
    semantic_mode: SemanticStateMode,
    nebula_advance: Option<NebulaAdvance>,
    recorder: &mut R,
) -> Result<State, Error> {
    recorder.stage(VerifyStepExecutionStage::Entry);
    construction2::enforce_pc_in_range(&state)?;
    construction2::state_base_case_check(&state)?;
    construction2::validate_state_authority(vk, s, &state, semantic_mode)?;
    construction2::validate_digest32("proof.semantic_state_digest", proof.semantic_state_digest)?;
    if next_latest_claims.is_empty() {
        return Err(Error::EmptyStep);
    }

    recorder.stage(VerifyStepExecutionStage::ChunkDigest);
    let fresh_count = next_latest_claims.len() as u64;
    let chunk_digest = construction2::f_prime_chunk_public_digest_from_claims(state.step_count, next_latest_claims);
    recorder.chunk_digest(state.step_count, next_latest_claims, chunk_digest);

    let State {
        chunk_count,
        step_count,
        z_0,
        z_i,
        pc,
        initial_semantic_state_digest,
        semantic_state_digest,
        acc_digest,
        public_trace,
        proof: prev_proof,
        nebula,
    } = state;

    // F' fold-step verifier — branch on (prev_proof, proof.fold).
    recorder.stage(VerifyStepExecutionStage::Dispatch);
    let next_running = match (prev_proof, &proof.fold) {
        (ProofState::Initial, FoldProof::NoFold) => {
            recorder.dispatch(VerifyStepDispatch::InitialNoFold);
            // HyperNova Construction 2, base case: the verifier derives the
            // same fixed default accumulator as the prover. It is not proof
            // advice and no NIFS.V call runs in this branch.
            let m_in = next_latest_claims
                .first()
                .expect("nonempty next_latest_claims was checked above")
                .m_in;
            RunningInstance::canonical_zero(pp, s, m_in, LaneCommitmentMode::from_nebula(nebula_advance.is_some()))?
        }
        (ProofState::Active { running, latest }, FoldProof::Recursive(nifs_proof)) => {
            recorder.dispatch(VerifyStepDispatch::ActiveRecursive);
            recorder.stage(VerifyStepExecutionStage::Nifs);
            let running = running.into_materialized()?;
            // Same fresh per-step F' transcript the prover used.
            let state_in = State {
                chunk_count,
                step_count,
                z_0,
                z_i,
                pc,
                initial_semantic_state_digest,
                semantic_state_digest,
                acc_digest,
                public_trace,
                proof: ProofState::Initial,
                nebula: nebula.clone(),
            };
            let mut tr = f_prime_step_transcript_recorded(vk, structure_digest, &state_in, chunk_digest, recorder);
            recorder.transcript_prefix(tr.snapshot());
            let nifs_proof = nifs_proof.materialize()?;
            let fresh_claims = latest.claims();
            let result = nifs::verify(
                &mut tr,
                pp,
                s,
                cache,
                mix_rhos_commits,
                combine_b_pows,
                &fresh_claims,
                &running,
                &nifs_proof,
            );
            recorder.nifs_verify(&running, &fresh_claims, &nifs_proof, result.as_ref());
            result?
        }
        (ProofState::Initial, FoldProof::Recursive(_)) => {
            recorder.dispatch(VerifyStepDispatch::InitialRecursive);
            return Err(Error::FoldProofVariantMismatch);
        }
        (ProofState::Active { .. }, FoldProof::NoFold) => {
            recorder.dispatch(VerifyStepDispatch::ActiveNoFold);
            return Err(Error::FoldProofVariantMismatch);
        }
    };

    // Build next ProofState (verifier-side: witnesses empty).
    let new_proof = ProofState::active(next_running, latest_from_claims_for_verifier(next_latest_claims));

    // F' steps 1, 2, 5 — advance and compare.
    let prev_state_for_advance = State {
        chunk_count,
        step_count,
        z_0,
        z_i,
        pc,
        initial_semantic_state_digest,
        semantic_state_digest,
        acc_digest,
        public_trace,
        proof: ProofState::Initial, // placeholder; advance reads new_proof
        nebula: nebula.clone(),
    };
    let semantic_advance = match semantic_mode {
        // Stateless plans have no F' image binding rows for the semantic
        // lane. The verifier therefore drives `advance_state` with
        // `Stateless` (which sets `semantic_state_digest = new_acc_digest`)
        // and then explicitly cross-checks the prover's claim against the
        // resulting deterministic value. A mismatch is surfaced with a
        // dedicated error instead of being implicitly caught by the x_out
        // chain check below — the prover should see exactly which
        // invariant they violated.
        SemanticStateMode::Stateless => SemanticStateAdvance::Stateless,
        SemanticStateMode::Stateful => SemanticStateAdvance::Stateful(proof.semantic_state_digest),
    };
    recorder.stage(VerifyStepExecutionStage::Nebula);
    if proof.nebula_open != nebula_advance.as_ref().and_then(|adv| adv.open) {
        return Err(Error::NebulaOpenMismatch);
    }
    recorder.stage(VerifyStepExecutionStage::Advance);
    let next_state = construction2::advance_state_recorded(
        prev_state_for_advance,
        new_proof,
        s,
        fresh_count,
        chunk_digest,
        semantic_advance,
        nebula_advance.map(|adv| adv.lane_out),
        recorder,
    )?;
    recorder.stage(VerifyStepExecutionStage::Semantic);
    if matches!(semantic_mode, SemanticStateMode::Stateless)
        && next_state.semantic_state_digest != proof.semantic_state_digest
    {
        return Err(Error::StatelessSemanticInvariantViolated);
    }
    recorder.stage(VerifyStepExecutionStage::XOut);
    let x_out = construction2::compute_x_out_recorded(vk, structure_digest, &next_state, semantic_mode, recorder);
    if x_out != proof.x_out {
        return Err(Error::XOutMismatch);
    }
    recorder.stage(VerifyStepExecutionStage::Complete);
    Ok(next_state)
}

/// Verifier-side reconstruction of `LatestInstance` — claims only, with
/// shape-only witness placeholders. Verifier-side state never reads the
/// witnesses; they're carried for type uniformity with the prover side.
fn latest_from_claims_for_verifier(claims: &[CcsClaim]) -> LatestInstance {
    LatestInstance::from_instances(
        claims
            .iter()
            .map(|c| CcsInstance {
                claim: c.clone(),
                witness: crate::paper::relations::CcsWitness {
                    w: Vec::new(),
                    Z: neo_ccs::matrix::Mat::zero(0, 0, neo_math::F::default()),
                },
            })
            .collect(),
    )
}

// PR5 will add the in-circuit gadget mirror of `prove` / `verify`. The
// gadget reads Soundness Invariant I-5 — the absorb sequence in
// `paper::digest::state_x_out_digest` — and the `nifs::verify`
// composition above; both must move in lockstep with their R1CS
// counterparts.
