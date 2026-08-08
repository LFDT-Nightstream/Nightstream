//! Deterministic, proof-free receipts for the public native `verify_step` seam.
//!
//! This is deliberately narrower than a Construction-2 conformance claim:
//! `verify_step` checks one local F' transition, but it does not execute an
//! application `Machine.step`, authenticate the previous fresh-link, or run a
//! terminal verifier.  The corpus below records only calls the public Rust
//! function actually makes.

use neo_ccs::{LaneCommitments, Mat};
use neo_fold_clean::paper::construction2::{self, FoldProof, ProofState, SemanticStateMode, State, StepProof};
use neo_fold_clean::paper::digest::AccumulatorHandle;
use neo_fold_clean::paper::nifs::{self, NifsProof};
use neo_fold_clean::paper::relations::{CcsClaim, CeClaim};
use neo_fold_clean::{CcsInstance, Preprocessing, RunningInstance};
use neo_math::{F, K};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64};
use serde::Serialize;

#[path = "native_step_export/lean.rs"]
mod lean;

const SCHEMA: u32 = 2;

pub fn checked_native_step_receipts() -> (String, String) {
    let corpus = build_native_step_corpus();
    assert_eq!(corpus.schema, SCHEMA);
    assert!(corpus.native_step_only);
    assert_eq!(
        corpus
            .cases
            .iter()
            .map(|case| case.name.as_str())
            .collect::<Vec<_>>(),
        [
            "honest_base",
            "honest_recursive",
            "initial_with_recursive_fold",
            "active_with_no_fold",
            "empty_next_latest",
            "semantic_mode_flip",
            "stateless_semantic_digest_mutation",
            "x_out_mutation",
            "nifs_pi_dec_child_mutation",
            "incoming_accumulator_handle_mutation",
            "incoming_stateless_equality_mutation",
        ]
    );
    assert_eq!(
        corpus
            .cases
            .iter()
            .map(|case| &case.outcome)
            .collect::<Vec<_>>(),
        [
            &Outcome::Accepted,
            &Outcome::Accepted,
            &Outcome::Rejected(StableError::FoldProofVariantMismatch),
            &Outcome::Rejected(StableError::FoldProofVariantMismatch),
            &Outcome::Rejected(StableError::EmptyStep),
            &Outcome::Rejected(StableError::XOutMismatch),
            &Outcome::Rejected(StableError::StatelessSemanticInvariantViolated),
            &Outcome::Rejected(StableError::XOutMismatch),
            &Outcome::Rejected(StableError::NifsPiDecVerifyRejected),
            &Outcome::Rejected(StableError::StateAuthorityMismatch),
            &Outcome::Rejected(StableError::StateAuthorityMismatch),
        ]
    );
    let case = |name: &str| {
        corpus
            .cases
            .iter()
            .find(|case| case.name == name)
            .unwrap_or_else(|| panic!("missing native-step receipt {name}"))
    };
    assert_eq!(
        case("empty_next_latest").calls,
        NativeCallTrace {
            execution_order: Vec::new(),
            chunk_digest: None,
            nifs_call: None,
            running_digest: None,
            advanced_state: None,
            verifier_digest_read: None,
            pi_ccs_header_read: None,
            nebula_digest: None,
            computed_x_out: None,
        }
    );
    assert_eq!(case("empty_next_latest").final_stage, ExecutionStage::Entry);
    for name in ["initial_with_recursive_fold", "active_with_no_fold"] {
        let receipt = case(name);
        assert!(receipt.calls.chunk_digest.is_some());
        assert!(receipt.transcript.is_none());
        assert!(receipt.calls.nifs_call.is_none());
        assert!(receipt.calls.advanced_state.is_none());
        assert!(receipt.calls.computed_x_out.is_none());
    }
    let recursive = case("honest_recursive");
    assert!(recursive.transcript.is_some());
    assert!(recursive.calls.nifs_call.is_some());
    assert!(recursive.calls.running_digest.is_some());
    assert!(recursive.calls.advanced_state.is_some());
    assert!(recursive.calls.verifier_digest_read.is_some());
    assert!(recursive.calls.pi_ccs_header_read.is_some());
    assert!(recursive.calls.computed_x_out.is_some());
    assert_eq!(recursive.final_stage, ExecutionStage::Complete);
    let semantic = case("stateless_semantic_digest_mutation");
    assert!(semantic.calls.advanced_state.is_some());
    assert!(semantic.calls.computed_x_out.is_none());
    for name in ["semantic_mode_flip", "x_out_mutation"] {
        let receipt = case(name);
        assert!(receipt.calls.advanced_state.is_some());
        assert!(receipt.calls.computed_x_out.is_some());
    }
    for name in ["nifs_pi_dec_child_mutation"] {
        let receipt = case(name);
        assert!(receipt.transcript.is_some());
        assert!(receipt.calls.nifs_call.is_some());
        assert!(receipt.calls.advanced_state.is_none());
        assert!(receipt.calls.computed_x_out.is_none());
    }
    for name in [
        "incoming_accumulator_handle_mutation",
        "incoming_stateless_equality_mutation",
    ] {
        let receipt = case(name);
        assert_eq!(receipt.final_stage, ExecutionStage::Entry);
        assert!(receipt.calls.execution_order.is_empty());
        assert!(receipt.transcript.is_none());
        assert!(receipt.calls.nifs_call.is_none());
        assert!(receipt.calls.advanced_state.is_none());
        assert!(receipt.calls.computed_x_out.is_none());
    }

    let incoming_stateless = corpus
        .cases
        .iter()
        .find(|case| case.name == "incoming_stateless_equality_mutation")
        .expect("incoming stateless-equality mutation case");
    let state = &corpus.atoms.states[incoming_stateless.input_state as usize];
    assert_ne!(
        state.semantic_state_digest, state.acc_digest,
        "incoming stateless-equality mutation must be explicit in the receipt"
    );

    let first = serde_json::to_string(&corpus).expect("serialize native-step receipt");
    let second = serde_json::to_string(&corpus).expect("serialize native-step receipt twice");
    assert_eq!(first, second, "native-step receipt serialization must be deterministic");
    let json = format!(
        "{}\n",
        serde_json::to_string_pretty(&corpus).expect("serialize deterministic native-step conformance corpus")
    );
    let lean = lean::render(&corpus);
    (json, lean)
}

#[derive(Clone, Debug, Serialize)]
struct NativeStepCorpus {
    schema: u32,
    native_step_only: bool,
    excluded_checks: Vec<&'static str>,
    profile: Profile,
    atoms: Atoms,
    cases: Vec<Receipt>,
}

#[derive(Clone, Debug, Serialize)]
struct Profile {
    name: &'static str,
    params: ParamsProfile,
    relation: RelationProfile,
    semantic_mode: Mode,
    nebula: &'static str,
    transcript_label: String,
    structure_digest: [Felt; 4],
    verifier_key_digest: [u8; 32],
    pi_ccs_header_bundle: [Felt; 4],
    ajtai_pp_digest: [Felt; 4],
}

#[derive(Clone, Debug, Serialize)]
struct ParamsProfile {
    q: u64,
    eta: u32,
    d: u32,
    kappa: u32,
    m: u64,
    b: u32,
    k_rho: u32,
    big_b: u64,
    expansion_t: u32,
    extension_degree: u32,
    lambda: u32,
    max_fresh_count: usize,
}

#[derive(Clone, Debug, Serialize)]
struct RelationProfile {
    rows: usize,
    columns: usize,
    matrix_count: usize,
    public_input_len: Option<usize>,
    fixture_relation: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Stateless,
    Stateful,
}

impl From<SemanticStateMode> for Mode {
    fn from(value: SemanticStateMode) -> Self {
        match value {
            SemanticStateMode::Stateless => Self::Stateless,
            SemanticStateMode::Stateful => Self::Stateful,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
struct Atoms {
    ccs_claims: Vec<CcsClaimAtom>,
    ce_claims: Vec<CeClaimAtom>,
    running: Vec<RunningAtom>,
    latest: Vec<LatestAtom>,
    nifs_proofs: Vec<NifsProofAtom>,
    step_proofs: Vec<StepProofAtom>,
    states: Vec<StateAtom>,
    transcripts: Vec<TranscriptAtom>,
    nifs_calls: Vec<NifsCallAtom>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct Receipt {
    name: String,
    dispatch: Dispatch,
    mode: Mode,
    input_state: u32,
    next_latest: Vec<u32>,
    step_proof: u32,
    transcript: Option<u32>,
    calls: NativeCallTrace,
    final_stage: ExecutionStage,
    recorded_next: Option<u32>,
    outcome: Outcome,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct NativeCallTrace {
    execution_order: Vec<ExecutionEventKind>,
    chunk_digest: Option<ChunkReceipt>,
    nifs_call: Option<u32>,
    running_digest: Option<RunningDigestCall>,
    advanced_state: Option<u32>,
    verifier_digest_read: Option<[u8; 32]>,
    pi_ccs_header_read: Option<[Felt; 4]>,
    nebula_digest: Option<NebulaDigestCall>,
    computed_x_out: Option<ComputedXOut>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ComputedXOut {
    preimage: Vec<Felt>,
    digest: [u8; 32],
    bits_little_endian: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct RunningDigestCall {
    running: u32,
    relation_columns: usize,
    output: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct NebulaDigestCall {
    lane: NebulaLaneAtom,
    output: [Felt; 4],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct NebulaLaneAtom {
    segment_index: u64,
    step_index: u64,
    timestamp: u64,
    gamma: Option<Vec<Ext>>,
    products: Vec<Ext>,
    stack_pointers: Vec<u64>,
    precommitted_digests: Vec<Vec<Felt>>,
    seen_digests: Vec<Vec<Felt>>,
    memory_digest: Vec<Felt>,
}

fn nebula_lane(value: &construction2::NebulaLane) -> NebulaLaneAtom {
    NebulaLaneAtom {
        segment_index: value.seg_idx,
        step_index: value.idx,
        timestamp: value.ts,
        gamma: value.gamma.as_ref().map(|values| exts(values)),
        products: exts(&value.h),
        stack_pointers: value.sp.to_vec(),
        precommitted_digests: value.d_pre.iter().map(|digest| felts(digest)).collect(),
        seen_digests: value.d_seen.iter().map(|digest| felts(digest)).collect(),
        memory_digest: felts(&value.d_mem),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ExecutionEventKind {
    ChunkDigest,
    Dispatch,
    TranscriptStarted,
    TranscriptAppend,
    TranscriptPrefix,
    NifsVerify,
    RunningDigest,
    StateAdvanced,
    VerifierDigestRead,
    PiCcsHeaderRead,
    NebulaDigest,
    StateXOutHash,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ExecutionStage {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Dispatch {
    InitialNoFold,
    InitialRecursive,
    ActiveNoFold,
    ActiveRecursive,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Outcome {
    Accepted,
    Rejected(StableError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum StableError {
    EmptyStep,
    FoldProofVariantMismatch,
    StateAuthorityMismatch,
    StatelessSemanticInvariantViolated,
    XOutMismatch,
    NifsPiDecVerifyRejected,
    NifsPiCcsOutputShapeMismatch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct Felt(u64);

fn felt(value: F) -> Felt {
    Felt(value.as_canonical_u64())
}

fn felts(values: &[F]) -> Vec<Felt> {
    values.iter().copied().map(felt).collect()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct Ext(Vec<Felt>);

fn ext(value: &K) -> Ext {
    Ext(felts(value.as_basis_coefficients_slice()))
}

fn exts(values: &[K]) -> Vec<Ext> {
    values.iter().map(ext).collect()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct CommitmentAtom {
    d: usize,
    kappa: usize,
    column_major: Vec<Felt>,
}

fn commitment(value: &neo_ajtai::Commitment) -> CommitmentAtom {
    CommitmentAtom {
        d: value.d,
        kappa: value.kappa,
        column_major: felts(&value.data),
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct MatrixAtom {
    rows: usize,
    columns: usize,
    row_major: Vec<Felt>,
}

fn matrix(value: &Mat<F>) -> MatrixAtom {
    MatrixAtom {
        rows: value.rows(),
        columns: value.cols(),
        row_major: (0..value.rows())
            .flat_map(|row| (0..value.cols()).map(move |column| felt(value[(row, column)])))
            .collect(),
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct AdvAtom {
    ops: CommitmentAtom,
    is: CommitmentAtom,
    fs: CommitmentAtom,
}

fn adv(value: &LaneCommitments<neo_ajtai::Commitment>) -> AdvAtom {
    AdvAtom {
        ops: commitment(&value.ops),
        is: commitment(&value.is),
        fs: commitment(&value.fs),
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct CcsClaimAtom {
    commitment: CommitmentAtom,
    public_input: Vec<Felt>,
    public_input_len: usize,
    adv: Option<AdvAtom>,
}

impl CcsClaimAtom {
    fn from_claim(value: &CcsClaim) -> Self {
        Self {
            commitment: commitment(&value.c),
            public_input: felts(&value.x),
            public_input_len: value.m_in,
            adv: value.adv.as_ref().map(adv),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct CeClaimAtom {
    commitment: CommitmentAtom,
    public_input_matrix: MatrixAtom,
    row_point: Vec<Ext>,
    ring_evaluations: Vec<Vec<Ext>>,
    constant_terms: Vec<Ext>,
    public_input_len: usize,
    fold_digest: [u8; 32],
    adv: Option<AdvAtom>,
}

impl CeClaimAtom {
    fn from_claim(value: &CeClaim) -> Self {
        Self {
            commitment: commitment(&value.c),
            public_input_matrix: matrix(&value.X),
            row_point: exts(&value.r),
            ring_evaluations: value.y_ring.iter().map(|row| exts(row)).collect(),
            constant_terms: exts(&value.ct),
            public_input_len: value.m_in,
            fold_digest: value.fold_digest,
            adv: value.adv.as_ref().map(adv),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct RunningAtom {
    ordered_children: Vec<u32>,
    parent_authority: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct LatestAtom {
    ordered_claims: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct NifsProofAtom {
    pi_ccs: PiCcsProofAtom,
    pi_rlc_combined: u32,
    pi_dec_children: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct PiCcsProofAtom {
    sumcheck_rounds: Vec<Vec<Ext>>,
    ordered_outputs: Vec<u32>,
    outputs_digest: [Felt; 4],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct StepProofAtom {
    fold: FoldAtom,
    nebula_open: Option<Vec<Vec<Felt>>>,
    semantic_state_digest: [u8; 32],
    x_out_bits_little_endian: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum FoldAtom {
    NoFold,
    Recursive(u32),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct StateAtom {
    chunk_count: u64,
    step_count: u64,
    z_0: [u8; 32],
    z_i: [u8; 32],
    pc: u64,
    initial_semantic_state_digest: [u8; 32],
    semantic_state_digest: [u8; 32],
    acc_digest: [u8; 32],
    public_trace: [u8; 32],
    branch: StateBranch,
    nebula: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum StateBranch {
    Initial,
    Active { running: u32, latest: u32 },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct TranscriptAtom {
    label: String,
    ordered_absorbs: Vec<TranscriptAbsorb>,
    prefix_snapshot: TranscriptSnapshot,
    chunk: ChunkReceipt,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct TranscriptAbsorb {
    label: String,
    fields: Vec<Felt>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct TranscriptSnapshot {
    state: Vec<Felt>,
    absorbed: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ChunkReceipt {
    start_index: u64,
    ordered_claims: Vec<u32>,
    claim_shapes: Vec<ClaimShape>,
    output: [Felt; 4],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ClaimShape {
    commitment_rows: usize,
    commitment_columns: usize,
    public_input_len: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct NifsCallAtom {
    running: u32,
    fresh: Vec<u32>,
    proof: u32,
    outcome: NifsOutcome,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum NifsOutcome {
    Accepted(u32),
    Rejected(StableError),
}

struct CorpusBuilder<'a> {
    prep: &'a Preprocessing,
    atoms: Atoms,
    cases: Vec<Receipt>,
}

impl<'a> CorpusBuilder<'a> {
    fn new(prep: &'a Preprocessing) -> Self {
        Self {
            prep,
            atoms: Atoms::default(),
            cases: Vec::new(),
        }
    }

    fn intern_ccs_claim(&mut self, claim: &CcsClaim) -> u32 {
        intern(&mut self.atoms.ccs_claims, CcsClaimAtom::from_claim(claim))
    }

    fn intern_ce_claim(&mut self, claim: &CeClaim) -> u32 {
        intern(&mut self.atoms.ce_claims, CeClaimAtom::from_claim(claim))
    }

    fn intern_running(&mut self, running: &RunningInstance) -> u32 {
        let ordered_children = running
            .claims
            .iter()
            .map(|claim| self.intern_ce_claim(claim))
            .collect();
        let parent_authority = running
            .parent_authority
            .as_ref()
            .map(|claim| self.intern_ce_claim(claim));
        intern(
            &mut self.atoms.running,
            RunningAtom {
                ordered_children,
                parent_authority,
            },
        )
    }

    fn intern_latest(&mut self, claims: &[CcsClaim]) -> u32 {
        let ordered_claims = claims
            .iter()
            .map(|claim| self.intern_ccs_claim(claim))
            .collect();
        intern(&mut self.atoms.latest, LatestAtom { ordered_claims })
    }

    fn intern_nifs_proof(&mut self, proof: &NifsProof) -> u32 {
        let ordered_outputs = proof
            .pi_ccs
            .outputs
            .iter()
            .map(|claim| self.intern_ce_claim(claim))
            .collect();
        let pi_rlc_combined = self.intern_ce_claim(&proof.pi_rlc.combined);
        let pi_dec_children = proof
            .pi_dec
            .children
            .iter()
            .map(|claim| self.intern_ce_claim(claim))
            .collect();
        let sumcheck = &proof.pi_ccs.sumcheck;
        let pi_ccs = PiCcsProofAtom {
            sumcheck_rounds: sumcheck
                .sumcheck_rounds
                .iter()
                .map(|round| exts(round))
                .collect(),
            ordered_outputs,
            outputs_digest: proof.pi_ccs.outputs_digest.map(felt),
        };
        intern(
            &mut self.atoms.nifs_proofs,
            NifsProofAtom {
                pi_ccs,
                pi_rlc_combined,
                pi_dec_children,
            },
        )
    }

    fn intern_execution_step_proof(&mut self, proof: &construction2::VerifyStepExecutionProof) -> u32 {
        let fold = match &proof.fold {
            construction2::VerifyStepExecutionFoldProof::NoFold => FoldAtom::NoFold,
            construction2::VerifyStepExecutionFoldProof::Recursive(proof) => {
                FoldAtom::Recursive(self.intern_nifs_proof(proof))
            }
        };
        let nebula_open = proof.nebula_open.as_ref().map(|rows| {
            rows.iter()
                .map(|row| row.iter().copied().map(felt).collect())
                .collect()
        });
        intern(
            &mut self.atoms.step_proofs,
            StepProofAtom {
                fold,
                nebula_open,
                semantic_state_digest: proof.semantic_state_digest,
                x_out_bits_little_endian: proof.x_out.bits().to_vec(),
            },
        )
    }

    fn intern_state(&mut self, state: &State) -> u32 {
        assert!(
            state.nebula.is_none(),
            "native-step conformance corpus is intentionally the plain no-Nebula profile"
        );
        let branch = match &state.proof {
            ProofState::Initial => StateBranch::Initial,
            ProofState::Active { running, latest } => {
                let claims = latest.claims();
                let running_id = self.intern_running(running);
                let latest_id = self.intern_latest(&claims);
                StateBranch::Active {
                    running: running_id,
                    latest: latest_id,
                }
            }
        };
        intern(
            &mut self.atoms.states,
            StateAtom {
                chunk_count: state.chunk_count,
                step_count: state.step_count,
                z_0: state.z_0,
                z_i: state.z_i,
                pc: state.pc,
                initial_semantic_state_digest: state.initial_semantic_state_digest,
                semantic_state_digest: state.semantic_state_digest,
                acc_digest: state.acc_digest,
                public_trace: state.public_trace,
                branch,
                nebula: "absent",
            },
        )
    }

    fn intern_execution_state(&mut self, state: &construction2::VerifyStepExecutionState) -> u32 {
        assert!(
            state.nebula.is_none(),
            "native-step conformance corpus is intentionally the plain no-Nebula profile"
        );
        let branch = match &state.proof {
            construction2::VerifyStepExecutionProofState::Initial => StateBranch::Initial,
            construction2::VerifyStepExecutionProofState::Active { running, latest_claims } => {
                let running_id = self.intern_running(running);
                let latest_id = self.intern_latest(latest_claims);
                StateBranch::Active {
                    running: running_id,
                    latest: latest_id,
                }
            }
        };
        intern(
            &mut self.atoms.states,
            StateAtom {
                chunk_count: state.chunk_count,
                step_count: state.step_count,
                z_0: state.z_0,
                z_i: state.z_i,
                pc: state.pc,
                initial_semantic_state_digest: state.initial_semantic_state_digest,
                semantic_state_digest: state.semantic_state_digest,
                acc_digest: state.acc_digest,
                public_trace: state.public_trace,
                branch,
                nebula: "absent",
            },
        )
    }

    fn push_case(
        &mut self,
        name: &str,
        state: State,
        next_latest: Vec<CcsClaim>,
        proof: StepProof,
        mode: SemanticStateMode,
    ) {
        let execution = construction2::verify_step_with_execution_receipt(
            &self.prep.params,
            self.prep.structure(),
            self.prep.optimized_cache(),
            self.prep.structure_digest(),
            self.prep.mix_rhos_commits(),
            self.prep.combine_b_pows(),
            &self.prep.vk,
            state,
            &next_latest,
            &proof,
            mode,
            None,
        );
        let construction2::VerifyStepExecutionReceipt {
            input,
            events,
            final_stage,
            result,
        } = execution;
        let stable_result_error = result
            .as_ref()
            .err()
            .map(|error| stable_step_error(name, error));
        let input_state = self.intern_execution_state(&input.state);
        let ordered_claims = input
            .next_latest_claims
            .iter()
            .map(|claim| self.intern_ccs_claim(claim))
            .collect::<Vec<_>>();
        let step_proof = self.intern_execution_step_proof(&input.proof);

        let mut dispatch = None;
        let mut execution_order = Vec::with_capacity(events.len());
        let mut chunk_digest = None;
        let mut transcript_label = None;
        let mut transcript_absorbs = Vec::new();
        let mut transcript_snapshot = None;
        let mut nifs_call = None;
        let mut running_digest = None;
        let mut advanced_state = None;
        let mut verifier_digest_read = None;
        let mut pi_ccs_header_read = None;
        let mut nebula_digest = None;
        let mut computed_x_out = None;

        for event in events {
            match event {
                construction2::VerifyStepExecutionEvent::ChunkDigest {
                    start_index,
                    claims,
                    output,
                } => {
                    execution_order.push(ExecutionEventKind::ChunkDigest);
                    assert!(chunk_digest.is_none(), "native step emitted two chunk digests");
                    let claim_ids = claims
                        .iter()
                        .map(|claim| self.intern_ccs_claim(claim))
                        .collect();
                    chunk_digest = Some(ChunkReceipt {
                        start_index,
                        ordered_claims: claim_ids,
                        claim_shapes: claims
                            .iter()
                            .map(|claim| ClaimShape {
                                commitment_rows: claim.c.d,
                                commitment_columns: claim.c.kappa,
                                public_input_len: claim.m_in,
                            })
                            .collect(),
                        output: output.map(felt),
                    });
                }
                construction2::VerifyStepExecutionEvent::Dispatch { branch } => {
                    execution_order.push(ExecutionEventKind::Dispatch);
                    assert!(dispatch.is_none(), "native step emitted two dispatches");
                    dispatch = Some(match branch {
                        construction2::VerifyStepDispatch::InitialNoFold => Dispatch::InitialNoFold,
                        construction2::VerifyStepDispatch::InitialRecursive => Dispatch::InitialRecursive,
                        construction2::VerifyStepDispatch::ActiveNoFold => Dispatch::ActiveNoFold,
                        construction2::VerifyStepDispatch::ActiveRecursive => Dispatch::ActiveRecursive,
                    });
                }
                construction2::VerifyStepExecutionEvent::TranscriptStarted { label } => {
                    execution_order.push(ExecutionEventKind::TranscriptStarted);
                    assert!(transcript_label.is_none(), "native step started two transcripts");
                    transcript_label = Some(ascii(label));
                }
                construction2::VerifyStepExecutionEvent::TranscriptAppend { label, fields } => {
                    execution_order.push(ExecutionEventKind::TranscriptAppend);
                    transcript_absorbs.push(TranscriptAbsorb {
                        label: ascii(label),
                        fields: felts(&fields),
                    });
                }
                construction2::VerifyStepExecutionEvent::TranscriptPrefix { snapshot } => {
                    execution_order.push(ExecutionEventKind::TranscriptPrefix);
                    assert!(
                        transcript_snapshot.is_none(),
                        "native step emitted two transcript snapshots"
                    );
                    transcript_snapshot = Some(TranscriptSnapshot {
                        state: snapshot.state().iter().copied().map(felt).collect(),
                        absorbed: snapshot.absorbed(),
                    });
                }
                construction2::VerifyStepExecutionEvent::NifsVerify {
                    running,
                    fresh_claims,
                    proof,
                    outcome,
                } => {
                    execution_order.push(ExecutionEventKind::NifsVerify);
                    assert!(nifs_call.is_none(), "native step emitted two NIFS calls");
                    let running = self.intern_running(&running);
                    let fresh = fresh_claims
                        .iter()
                        .map(|claim| self.intern_ccs_claim(claim))
                        .collect();
                    let proof = self.intern_nifs_proof(&proof);
                    let outcome = match outcome {
                        construction2::NifsVerifyExecutionOutcome::Accepted(output) => {
                            NifsOutcome::Accepted(self.intern_running(&output))
                        }
                        construction2::NifsVerifyExecutionOutcome::Rejected => NifsOutcome::Rejected(
                            stable_result_error.expect("rejected NIFS call must be the final step error"),
                        ),
                    };
                    nifs_call = Some(intern(
                        &mut self.atoms.nifs_calls,
                        NifsCallAtom {
                            running,
                            fresh,
                            proof,
                            outcome,
                        },
                    ));
                }
                construction2::VerifyStepExecutionEvent::RunningDigest {
                    running,
                    relation_columns,
                    output,
                } => {
                    execution_order.push(ExecutionEventKind::RunningDigest);
                    assert!(running_digest.is_none(), "native step emitted two running digests");
                    running_digest = Some(RunningDigestCall {
                        running: self.intern_running(&running),
                        relation_columns,
                        output,
                    });
                }
                construction2::VerifyStepExecutionEvent::StateAdvanced { output } => {
                    execution_order.push(ExecutionEventKind::StateAdvanced);
                    assert!(advanced_state.is_none(), "native step advanced twice");
                    advanced_state = Some(self.intern_state(&output));
                }
                construction2::VerifyStepExecutionEvent::VerifierDigestRead { output } => {
                    execution_order.push(ExecutionEventKind::VerifierDigestRead);
                    assert!(verifier_digest_read.is_none(), "native step read verifier digest twice");
                    verifier_digest_read = Some(output);
                }
                construction2::VerifyStepExecutionEvent::PiCcsHeaderRead { output } => {
                    execution_order.push(ExecutionEventKind::PiCcsHeaderRead);
                    assert!(pi_ccs_header_read.is_none(), "native step read Π_CCS header twice");
                    pi_ccs_header_read = Some(output.map(felt));
                }
                construction2::VerifyStepExecutionEvent::NebulaDigest { lane, output } => {
                    execution_order.push(ExecutionEventKind::NebulaDigest);
                    assert!(nebula_digest.is_none(), "native step hashed two Nebula lanes");
                    nebula_digest = Some(NebulaDigestCall {
                        lane: nebula_lane(&lane),
                        output: output.map(felt),
                    });
                }
                construction2::VerifyStepExecutionEvent::StateXOutHash {
                    preimage,
                    output_digest,
                    output,
                } => {
                    execution_order.push(ExecutionEventKind::StateXOutHash);
                    assert!(computed_x_out.is_none(), "native step hashed two state outputs");
                    computed_x_out = Some(ComputedXOut {
                        preimage: felts(&preimage),
                        digest: output_digest,
                        bits_little_endian: output.bits().to_vec(),
                    });
                }
            }
        }

        let transcript = match (transcript_label, transcript_snapshot) {
            (None, None) => {
                assert!(transcript_absorbs.is_empty());
                None
            }
            (Some(label), Some(prefix_snapshot)) => Some(intern(
                &mut self.atoms.transcripts,
                TranscriptAtom {
                    label,
                    ordered_absorbs: transcript_absorbs,
                    prefix_snapshot,
                    chunk: chunk_digest
                        .clone()
                        .expect("observed transcript must follow observed chunk digest"),
                },
            )),
            _ => panic!("native transcript start/snapshot receipt is incomplete"),
        };
        let (recorded_next, outcome) = match result {
            Ok(state) => (Some(self.intern_state(&state)), Outcome::Accepted),
            Err(error) => (None, Outcome::Rejected(stable_step_error(name, &error))),
        };
        self.cases.push(Receipt {
            name: name.to_owned(),
            dispatch: dispatch.unwrap_or_else(|| classify_execution_dispatch(&input.state, &input.proof)),
            mode: input.semantic_mode.into(),
            input_state,
            next_latest: ordered_claims,
            step_proof,
            transcript,
            calls: NativeCallTrace {
                execution_order,
                chunk_digest,
                nifs_call,
                running_digest,
                advanced_state,
                verifier_digest_read,
                pi_ccs_header_read,
                nebula_digest,
                computed_x_out,
            },
            final_stage: execution_stage(final_stage),
            recorded_next,
            outcome,
        });
    }
}

fn build_native_step_corpus() -> NativeStepCorpus {
    let prep = super::support::toy_preprocessing();
    assert_eq!(
        (prep.structure().n, prep.structure().m, prep.structure().t()),
        (54, 54, 1),
        "native-step receipt profile must remain the one-slot Phi81 zero-polynomial fixture"
    );
    assert_eq!(prep.semantic_state_mode(), SemanticStateMode::Stateless);
    assert!(prep.nebula().is_none());

    let base_audit =
        neo_fold_clean::prove(&prep, Vec::<Vec<CcsInstance>>::new()).expect("construct base native-step fixture");
    let base_input = base_audit.proof.state.clone();
    let after_base = neo_fold_clean::extend(&prep, base_audit, vec![super::support::toy_instance(&prep, 10_001)])
        .expect("construct honest base native step");
    let base_proof = after_base.steps[0].clone();
    let base_next = after_base.public_batches[0].clone();
    let recursive_input = after_base.proof.state.clone();
    let after_recursive = neo_fold_clean::extend(&prep, after_base, vec![super::support::toy_instance(&prep, 10_002)])
        .expect("construct honest recursive native step");
    let recursive_proof = after_recursive.steps[1].clone();
    let recursive_next = after_recursive.public_batches[1].clone();

    let mut builder = CorpusBuilder::new(&prep);
    builder.push_case(
        "honest_base",
        base_input.clone(),
        base_next.clone(),
        base_proof.clone(),
        SemanticStateMode::Stateless,
    );
    builder.push_case(
        "honest_recursive",
        recursive_input.clone(),
        recursive_next.clone(),
        recursive_proof.clone(),
        SemanticStateMode::Stateless,
    );

    let mut initial_recursive = base_proof.clone();
    initial_recursive.fold = recursive_proof.fold.clone();
    builder.push_case(
        "initial_with_recursive_fold",
        base_input.clone(),
        base_next.clone(),
        initial_recursive,
        SemanticStateMode::Stateless,
    );

    let mut active_no_fold = recursive_proof.clone();
    active_no_fold.fold = FoldProof::NoFold;
    builder.push_case(
        "active_with_no_fold",
        recursive_input.clone(),
        recursive_next.clone(),
        active_no_fold,
        SemanticStateMode::Stateless,
    );

    builder.push_case(
        "empty_next_latest",
        base_input.clone(),
        Vec::new(),
        base_proof.clone(),
        SemanticStateMode::Stateless,
    );
    builder.push_case(
        "semantic_mode_flip",
        base_input.clone(),
        base_next.clone(),
        base_proof.clone(),
        SemanticStateMode::Stateful,
    );

    let mut bad_semantic = base_proof.clone();
    bad_semantic.semantic_state_digest[0] ^= 1;
    builder.push_case(
        "stateless_semantic_digest_mutation",
        base_input.clone(),
        base_next.clone(),
        bad_semantic,
        SemanticStateMode::Stateless,
    );

    let mut bad_x_out = base_proof.clone();
    let mut x_out_digest = enc_inst_digest(&bad_x_out);
    x_out_digest[0] ^= 1;
    bad_x_out.x_out = construction2::EncInst::from_digest(x_out_digest);
    builder.push_case(
        "x_out_mutation",
        base_input.clone(),
        base_next.clone(),
        bad_x_out,
        SemanticStateMode::Stateless,
    );

    let mut bad_nifs = recursive_proof.clone();
    let FoldProof::Recursive(nifs_proof) = &mut bad_nifs.fold else {
        panic!("recursive fixture")
    };
    nifs_proof.pi_dec.children[0].c.data[0] += F::ONE;
    builder.push_case(
        "nifs_pi_dec_child_mutation",
        recursive_input.clone(),
        recursive_next.clone(),
        bad_nifs,
        SemanticStateMode::Stateless,
    );

    let mut bad_accumulator_handle = recursive_input.clone();
    bad_accumulator_handle.acc_digest[0] ^= 1;
    let ProofState::Active { running, .. } = &bad_accumulator_handle.proof else {
        panic!("incoming-handle mutation fixture must be active")
    };
    assert_ne!(
        bad_accumulator_handle.acc_digest,
        AccumulatorHandle::from_running_parts(&running.claims, running.parent_authority.as_ref()).digest(),
        "incoming-handle mutation must disagree with the public running claims"
    );
    builder.push_case(
        "incoming_accumulator_handle_mutation",
        bad_accumulator_handle,
        recursive_next.clone(),
        recursive_proof.clone(),
        SemanticStateMode::Stateless,
    );

    let mut bad_stateless_equality = recursive_input;
    bad_stateless_equality.semantic_state_digest[0] ^= 1;
    builder.push_case(
        "incoming_stateless_equality_mutation",
        bad_stateless_equality,
        recursive_next,
        recursive_proof,
        SemanticStateMode::Stateless,
    );

    NativeStepCorpus {
        schema: SCHEMA,
        native_step_only: true,
        excluded_checks: vec![
            "application Machine.step",
            "incoming prior fresh-link",
            "full entry pinning",
            "stateful application authenticity",
            "Nebula transition authenticity",
            "terminal acceptance",
            "private witnesses",
            "R1CS assignment/layout",
        ],
        profile: Profile {
            name: "toy_direct_ccs_identity1_zero_poly_native_step",
            params: ParamsProfile {
                q: prep.params.q(),
                eta: prep.params.eta(),
                d: prep.params.d(),
                kappa: prep.params.kappa(),
                m: prep.params.m(),
                b: prep.params.b(),
                k_rho: prep.params.k_rho(),
                big_b: prep.params.big_b(),
                expansion_t: prep.params.T(),
                extension_degree: prep.params.extension_degree(),
                lambda: prep.params.lambda(),
                max_fresh_count: prep.params.max_fresh_count(),
            },
            relation: RelationProfile {
                rows: prep.structure().n,
                columns: prep.structure().m,
                matrix_count: prep.structure().t(),
                public_input_len: prep.public_input_len,
                fixture_relation: "M_0 = identity(1), f = zero sparse polynomial",
            },
            semantic_mode: prep.semantic_state_mode().into(),
            nebula: "absent",
            transcript_label: ascii(neo_fold_clean::paper::f_prime::native::F_PRIME_STEP_TRANSCRIPT_LABEL),
            structure_digest: prep.structure_digest().map(felt),
            verifier_key_digest: prep.vk.digest(),
            pi_ccs_header_bundle: prep.pi_ccs_header_bundle().map(felt),
            ajtai_pp_digest: prep.ajtai_pp_digest().map(felt),
        },
        atoms: builder.atoms,
        cases: builder.cases,
    }
}

fn classify_execution_dispatch(
    state: &construction2::VerifyStepExecutionState,
    proof: &construction2::VerifyStepExecutionProof,
) -> Dispatch {
    match (&state.proof, &proof.fold) {
        (
            construction2::VerifyStepExecutionProofState::Initial,
            construction2::VerifyStepExecutionFoldProof::NoFold,
        ) => Dispatch::InitialNoFold,
        (
            construction2::VerifyStepExecutionProofState::Initial,
            construction2::VerifyStepExecutionFoldProof::Recursive(_),
        ) => Dispatch::InitialRecursive,
        (
            construction2::VerifyStepExecutionProofState::Active { .. },
            construction2::VerifyStepExecutionFoldProof::NoFold,
        ) => Dispatch::ActiveNoFold,
        (
            construction2::VerifyStepExecutionProofState::Active { .. },
            construction2::VerifyStepExecutionFoldProof::Recursive(_),
        ) => Dispatch::ActiveRecursive,
    }
}

fn execution_stage(stage: construction2::VerifyStepExecutionStage) -> ExecutionStage {
    match stage {
        construction2::VerifyStepExecutionStage::Entry => ExecutionStage::Entry,
        construction2::VerifyStepExecutionStage::ChunkDigest => ExecutionStage::ChunkDigest,
        construction2::VerifyStepExecutionStage::Dispatch => ExecutionStage::Dispatch,
        construction2::VerifyStepExecutionStage::Nifs => ExecutionStage::Nifs,
        construction2::VerifyStepExecutionStage::Nebula => ExecutionStage::Nebula,
        construction2::VerifyStepExecutionStage::Advance => ExecutionStage::Advance,
        construction2::VerifyStepExecutionStage::Semantic => ExecutionStage::Semantic,
        construction2::VerifyStepExecutionStage::XOut => ExecutionStage::XOut,
        construction2::VerifyStepExecutionStage::Complete => ExecutionStage::Complete,
    }
}

fn enc_inst_digest(proof: &StepProof) -> [u8; 32] {
    let bits = proof.x_out.bits();
    let mut digest = [0u8; 32];
    for (index, bit) in bits.iter().copied().enumerate() {
        digest[index / 8] |= bit << (index % 8);
    }
    digest
}

fn stable_step_error(case: &str, error: &construction2::Error) -> StableError {
    match error {
        construction2::Error::EmptyStep => StableError::EmptyStep,
        construction2::Error::FoldProofVariantMismatch => StableError::FoldProofVariantMismatch,
        construction2::Error::StateAuthorityMismatch => StableError::StateAuthorityMismatch,
        construction2::Error::StatelessSemanticInvariantViolated => StableError::StatelessSemanticInvariantViolated,
        construction2::Error::XOutMismatch => StableError::XOutMismatch,
        construction2::Error::Nifs(error) => stable_nifs_error(case, error),
        other => panic!("unexpected native verify_step error in fixed corpus case {case}: {other:?}"),
    }
}

fn stable_nifs_error(case: &str, error: &nifs::Error) -> StableError {
    match error {
        nifs::Error::PiDec(neo_fold_clean::paper::pi_dec::Error::VerifyRejected) => {
            StableError::NifsPiDecVerifyRejected
        }
        nifs::Error::PiCcs(neo_fold_clean::paper::pi_ccs::Error::Engine(
            neo_fold_clean::engine::optimized::Error::Reductions(neo_reductions::error::PiCcsError::InvalidInput(
                message,
            )),
        )) if message == "optimized output does not have the one-joint shape" => {
            StableError::NifsPiCcsOutputShapeMismatch
        }
        other => panic!("unexpected NIFS error in fixed native-step corpus case {case}: {other:?}"),
    }
}

fn ascii(value: &[u8]) -> String {
    String::from_utf8(value.to_vec()).expect("protocol labels are ASCII")
}

fn intern<T: Eq>(atoms: &mut Vec<T>, value: T) -> u32 {
    if let Some(index) = atoms.iter().position(|atom| atom == &value) {
        return u32::try_from(index).expect("native-step atom index fits u32");
    }
    let index = u32::try_from(atoms.len()).expect("native-step atom table fits u32");
    atoms.push(value);
    index
}
