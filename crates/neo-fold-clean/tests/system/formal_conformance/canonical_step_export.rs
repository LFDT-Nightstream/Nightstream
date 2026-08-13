//! Shared-input Rust/Lean differential corpus for one linked F' step.
//!
//! Rust executes the production native verifier on each input.  The generated
//! Lean file receives the equality quotient of exactly the fields observed by
//! the frozen one-slot checker.  NIFS remains an opaque primitive at this
//! boundary: the Rust call outcome is recorded as a receipt, not asserted to
//! be a Lean proof of the primitive.

use neo_ccs::{LaneCommitments, Mat};
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::construction2::{self, FoldProof, ProofState, SemanticStateMode, State, StepProof};
use neo_fold_clean::paper::digest::{
    digest32_as_fields, digest_fields_as_digest32, f_prime_chunk_public_digest, initial_boundary_digest,
    public_trace_seed_digest, state_x_out_digest_with_mode, structure_digest, AccumulatorHandle, StateXOutDigestMode,
};
use neo_fold_clean::paper::f_prime::r1cs::{encode_f_prime_superneo_public_input, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN};
use neo_fold_clean::paper::nifs::NifsProof;
use neo_fold_clean::paper::relations::{CcsClaim, CcsInstance, CeClaim};
use neo_fold_clean::{Preprocessing, RunningInstance};
use neo_math::{F, K};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64};
use serde::Serialize;

#[path = "canonical_step_export/lean.rs"]
mod lean;

const SCHEMA: u32 = 1;

pub fn checked_canonical_step_cases() -> (String, String) {
    let corpus = build_corpus();
    assert_eq!(corpus.schema, SCHEMA);
    assert_eq!(
        corpus
            .cases
            .iter()
            .map(|case| case.name.as_str())
            .collect::<Vec<_>>(),
        [
            "honest_base",
            "honest_recursive",
            "base_initial_state_mutation",
            "initial_with_recursive_fold",
            "active_with_no_fold",
            "recursive_prior_pc_mutation",
            "recursive_prior_public_link_mutation",
            "recursive_nifs_proof_mutation",
            "recursive_x_out_mutation",
        ]
    );
    assert_eq!(
        corpus
            .cases
            .iter()
            .map(|case| case.mapped.rust_accepted)
            .collect::<Vec<_>>(),
        [true, true, false, false, false, false, false, false, false]
    );
    assert!(corpus
        .cases
        .iter()
        .filter(|case| case.mutation == "none")
        .all(|case| case.mapped.rust_accepted));
    assert!(corpus
        .cases
        .iter()
        .filter(|case| case.mutation != "none")
        .all(|case| !case.mapped.rust_accepted));

    let first = serde_json::to_string(&corpus).expect("serialize canonical-step corpus");
    let second = serde_json::to_string(&corpus).expect("serialize canonical-step corpus twice");
    assert_eq!(first, second, "canonical-step corpus must be deterministic");
    let json = format!("{first}\n");
    let lean = lean::render(&corpus);
    (json, lean)
}

#[derive(Clone, Debug, Serialize)]
struct Corpus {
    schema: u32,
    evidence_tier: &'static str,
    scope: &'static str,
    primitive_boundary: &'static str,
    excluded_claims: Vec<&'static str>,
    profile: Profile,
    atoms: Atoms,
    cases: Vec<Case>,
}

#[derive(Clone, Debug, Serialize)]
struct Profile {
    name: &'static str,
    relation_rows: usize,
    relation_columns: usize,
    matrix_count: usize,
    public_input_len: usize,
    semantic_mode: &'static str,
    fresh_count: usize,
    verifier_key_digest: [u8; 32],
    structure_digest: [Felt; 4],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct Felt(u64);

fn felt(value: F) -> Felt {
    Felt(value.as_canonical_u64())
}

fn felts(values: &[F]) -> Vec<Felt> {
    values.iter().copied().map(felt).collect()
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Ext(Vec<Felt>);

fn ext(value: &K) -> Ext {
    Ext(felts(value.as_basis_coefficients_slice()))
}

fn exts(values: &[K]) -> Vec<Ext> {
    values.iter().map(ext).collect()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct CommitmentAtom {
    rows: usize,
    columns: usize,
    column_major: Vec<Felt>,
}

fn commitment(value: &neo_ajtai::Commitment) -> CommitmentAtom {
    CommitmentAtom {
        rows: value.d,
        columns: value.kappa,
        column_major: felts(&value.data),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
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

#[derive(Clone, Debug, Eq, PartialEq)]
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
    /// The exact verifier-visible instance is used only as the equality key
    /// of the finite quotient. Running semantics remain behind NIFS.
    #[serde(skip)]
    equality_key: RunningEqualityKey,
    ordered_child_count: usize,
    parent_authority_present: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RunningEqualityKey {
    ordered_children: Vec<CeClaimAtom>,
    parent_authority: Option<CeClaimAtom>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct BatchAtom {
    ordered_claims: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct DigestAtom {
    bytes: [u8; 32],
    fields: [Felt; 4],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct NifsProofAtom {
    /// The frozen checker treats this sort opaquely.  The complete stable
    /// debug image is used only as the generator-side equality key and is not
    /// emitted into the artifact.
    #[serde(skip)]
    equality_key: String,
    kind: &'static str,
    pi_ccs_rounds: usize,
    pi_ccs_outputs: usize,
    pi_dec_children: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
struct Atoms {
    keys: Vec<[u8; 32]>,
    digests: Vec<DigestAtom>,
    states: Vec<[u8; 32]>,
    ccs_claims: Vec<CcsClaimAtom>,
    witnesses: Vec<BatchAtom>,
    running: Vec<RunningAtom>,
    fresh: Vec<BatchAtom>,
    nifs_proofs: Vec<NifsProofAtom>,
    encoded: Vec<Vec<Felt>>,
}

#[derive(Clone, Debug, Serialize)]
struct Case {
    name: String,
    mutation: &'static str,
    rust_input: RustInput,
    observed: Observed,
    mapped: StepCaseMap,
}

#[derive(Clone, Debug, Serialize)]
struct RustInput {
    state: RustState,
    next_latest: u32,
    fold: RustFold,
    semantic_state_digest: [u8; 32],
    x_out: u32,
    semantic_mode: &'static str,
    nebula: &'static str,
}

#[derive(Clone, Debug, Serialize)]
struct RustState {
    chunk_count: u64,
    step_count: u64,
    z0: u32,
    zi: u32,
    pc: u64,
    initial_semantic_state_digest: [u8; 32],
    semantic_state_digest: [u8; 32],
    accumulator_digest: [u8; 32],
    public_trace: [u8; 32],
    proof: RustProofState,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "branch", rename_all = "snake_case")]
enum RustProofState {
    Initial,
    Active { running: u32, fresh: u32 },
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum RustFold {
    NoFold,
    Recursive { proof: u32 },
}

#[derive(Clone, Debug, Serialize)]
struct Observed {
    event_order: Vec<&'static str>,
    dispatch: Option<&'static str>,
    nifs_outcome: &'static str,
    nifs_output: Option<u32>,
    rust_output: Option<MappedOutput>,
    rust_error: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct MappedOutput {
    z_next: u32,
    running_next: u32,
    pc_next: u64,
    x: u32,
}

#[derive(Clone, Debug, Serialize)]
struct StepCaseMap {
    verifier_key: u32,
    default_running: u32,
    iteration: u64,
    z0: u32,
    zi: u32,
    running: u32,
    fresh: u32,
    prior_pc: u64,
    witness: u32,
    nifs_proof: u32,
    step_receipt: StepReceiptMap,
    trace: StepTraceMap,
    claim: MappedOutput,
    rust_accepted: bool,
}

#[derive(Clone, Debug, Serialize)]
struct StepReceiptMap {
    state: u32,
    witness: u32,
    output: u32,
}

#[derive(Clone, Debug, Serialize)]
struct HashInputMap {
    verifier_key: u32,
    iteration: u64,
    z0: u32,
    current: u32,
    running: u32,
    pc: u64,
}

#[derive(Clone, Debug, Serialize)]
struct HashReceiptMap {
    input: HashInputMap,
    output: u32,
}

#[derive(Clone, Debug, Serialize)]
struct FreshPublicMap {
    input: u32,
    output: u32,
}

#[derive(Clone, Debug, Serialize)]
struct EncodeMap {
    input: u32,
    output: u32,
}

#[derive(Clone, Debug, Serialize)]
struct NifsMap {
    key: u32,
    running: u32,
    fresh: u32,
    proof: u32,
    output: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "branch", rename_all = "snake_case")]
enum StepTraceMap {
    Base {
        next_hash: Option<HashReceiptMap>,
    },
    Recursive {
        prior_hash: Option<HashReceiptMap>,
        fresh_public: Option<FreshPublicMap>,
        encode: Option<EncodeMap>,
        nifs: Option<NifsMap>,
        next_hash: Option<HashReceiptMap>,
    },
}

pub(super) struct Snapshot {
    pub(super) state_in: State,
    pub(super) state_out: State,
    next_latest: Vec<CcsClaim>,
    proof: StepProof,
}

pub(super) struct Fixture {
    pub(super) prep: Preprocessing,
    pub(super) snapshots: Vec<Snapshot>,
}

fn bit_carrier_r1cs() -> R1cs {
    R1cs {
        a: Mat::zero(1, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN, F::ZERO),
        b: Mat::zero(1, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN, F::ZERO),
        c: Mat::zero(1, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN, F::ZERO),
        m_in: F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN,
    }
}

fn base_state(prep: &Preprocessing) -> State {
    let structure = structure_digest(prep.structure());
    let z0 = initial_boundary_digest(&structure, prep.public_input_len);
    let public_trace = public_trace_seed_digest(&structure);
    let empty = AccumulatorHandle::empty().digest();
    State::base(z0, public_trace, empty, empty)
}

fn compute_x_out(prep: &Preprocessing, state: &State) -> [u8; 32] {
    let mode = match prep.semantic_state_mode() {
        SemanticStateMode::Stateless => StateXOutDigestMode::Stateless,
        SemanticStateMode::Stateful => StateXOutDigestMode::Stateful,
    };
    state_x_out_digest_with_mode(
        mode,
        prep.vk.digest(),
        prep.pi_ccs_header_bundle(),
        &structure_digest(prep.structure()),
        state.chunk_count,
        state.step_count,
        state.z_0,
        state.z_i,
        state.pc,
        state.semantic_state_digest,
        state.acc_digest,
        state.public_trace,
        None,
    )
}

fn build_link_instance(prep: &Preprocessing, r1cs: &R1cs, x_out: [u8; 32]) -> CcsInstance {
    let z = encode_f_prime_superneo_public_input(digest32_as_fields(x_out));
    direct_ccs::build_instance(prep, r1cs, &z).expect("linked F' instance")
}

fn peek_next_state(prep: &Preprocessing, state: &State, batch: &[CcsInstance]) -> State {
    construction2::step(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        &prep.log,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &prep.vk,
        state.clone(),
        batch.to_vec(),
    )
    .expect("peek linked step")
    .0
}

pub(super) fn build_fixture() -> Fixture {
    let r1cs = bit_carrier_r1cs();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 42).expect("linked profile preprocessing");
    assert_eq!(prep.semantic_state_mode(), SemanticStateMode::Stateless);
    assert!(prep.nebula().is_none());
    assert_eq!(prep.public_input_len, Some(F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN));

    let placeholder = vec![F::ZERO; prep.structure().m];
    let dummy = || direct_ccs::build_instance(&prep, &r1cs, &placeholder).expect("dummy instance");
    let mut state = base_state(&prep);
    let mut snapshots = Vec::with_capacity(2);
    for _ in 0..2 {
        let predicted = peek_next_state(&prep, &state, &[dummy()]);
        let linked = build_link_instance(&prep, &r1cs, compute_x_out(&prep, &predicted));
        let state_in = state.clone();
        let next_latest = vec![linked.claim.clone()];
        let (state_out, proof) = construction2::step(
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            prep.structure_digest(),
            &prep.log,
            prep.mix_rhos_commits(),
            prep.combine_b_pows(),
            &prep.vk,
            state,
            vec![linked],
        )
        .expect("honest linked step");
        assert_eq!(state_out.z_i, predicted.z_i);
        assert_eq!(state_out.public_trace, predicted.public_trace);
        assert_eq!(state_out.semantic_state_digest, predicted.semantic_state_digest);
        assert_eq!(state_out.acc_digest, predicted.acc_digest);
        snapshots.push(Snapshot {
            state_in,
            state_out: state_out.clone(),
            next_latest,
            proof,
        });
        state = state_out;
    }
    Fixture { prep, snapshots }
}

#[derive(Default)]
struct Builder {
    atoms: Atoms,
    cases: Vec<Case>,
}

fn intern<T: PartialEq>(values: &mut Vec<T>, value: T) -> u32 {
    if let Some(index) = values.iter().position(|found| found == &value) {
        return u32::try_from(index + 1).expect("atom index fits u32");
    }
    values.push(value);
    u32::try_from(values.len()).expect("atom table fits u32")
}

impl Builder {
    fn key(&mut self, value: [u8; 32]) -> u32 {
        intern(&mut self.atoms.keys, value)
    }

    fn digest(&mut self, value: [u8; 32]) -> u32 {
        intern(
            &mut self.atoms.digests,
            DigestAtom {
                bytes: value,
                fields: digest32_as_fields(value).map(felt),
            },
        )
    }

    fn state(&mut self, value: [u8; 32]) -> u32 {
        intern(&mut self.atoms.states, value)
    }

    fn claim(&mut self, value: &CcsClaim) -> u32 {
        intern(&mut self.atoms.ccs_claims, CcsClaimAtom::from_claim(value))
    }

    fn batch(&mut self, values: &[CcsClaim]) -> BatchAtom {
        BatchAtom {
            ordered_claims: values.iter().map(|claim| self.claim(claim)).collect(),
        }
    }

    fn witness(&mut self, values: &[CcsClaim]) -> u32 {
        let value = self.batch(values);
        intern(&mut self.atoms.witnesses, value)
    }

    fn fresh(&mut self, values: &[CcsClaim]) -> u32 {
        let value = self.batch(values);
        intern(&mut self.atoms.fresh, value)
    }

    fn running(&mut self, value: &RunningInstance) -> u32 {
        let ordered_children = value.claims.iter().map(CeClaimAtom::from_claim).collect();
        let parent_authority = value.parent_authority.as_ref().map(CeClaimAtom::from_claim);
        intern(
            &mut self.atoms.running,
            RunningAtom {
                equality_key: RunningEqualityKey {
                    ordered_children,
                    parent_authority,
                },
                ordered_child_count: value.claims.len(),
                parent_authority_present: value.parent_authority.is_some(),
            },
        )
    }

    fn nifs(&mut self, proof: Option<&NifsProof>) -> u32 {
        let atom = match proof {
            None => NifsProofAtom {
                equality_key: "<no recursive NIFS proof>".to_owned(),
                kind: "absent",
                pi_ccs_rounds: 0,
                pi_ccs_outputs: 0,
                pi_dec_children: 0,
            },
            Some(proof) => NifsProofAtom {
                equality_key: format!("{proof:#?}"),
                kind: "materialized_recursive",
                pi_ccs_rounds: proof.pi_ccs.sumcheck.sumcheck_rounds.len(),
                pi_ccs_outputs: proof.pi_ccs.outputs.len(),
                pi_dec_children: proof.pi_dec.children.len(),
            },
        };
        intern(&mut self.atoms.nifs_proofs, atom)
    }

    fn encoded(&mut self, values: &[F]) -> u32 {
        intern(&mut self.atoms.encoded, felts(values))
    }

    fn rust_state(&mut self, state: &State) -> (RustState, u32, u32) {
        let (proof, running, fresh) = match &state.proof {
            ProofState::Initial => {
                let running = self.running(&RunningInstance::default());
                let fresh = self.fresh(&[]);
                (RustProofState::Initial, running, fresh)
            }
            ProofState::Active { running, latest } => {
                let latest = latest.claims();
                let running_id = self.running(running);
                let fresh_id = self.fresh(&latest);
                (
                    RustProofState::Active {
                        running: running_id,
                        fresh: fresh_id,
                    },
                    running_id,
                    fresh_id,
                )
            }
        };
        (
            RustState {
                chunk_count: state.chunk_count,
                step_count: state.step_count,
                z0: self.state(state.z_0),
                zi: self.state(state.z_i),
                pc: state.pc,
                initial_semantic_state_digest: state.initial_semantic_state_digest,
                semantic_state_digest: state.semantic_state_digest,
                accumulator_digest: state.acc_digest,
                public_trace: state.public_trace,
                proof,
            },
            running,
            fresh,
        )
    }

    fn proof_map(&mut self, proof: &StepProof) -> (RustFold, u32) {
        match &proof.fold {
            FoldProof::NoFold => {
                let id = self.nifs(None);
                (RustFold::NoFold, id)
            }
            FoldProof::Recursive(proof) => {
                let id = self.nifs(Some(proof));
                (RustFold::Recursive { proof: id }, id)
            }
        }
    }
}

struct SourceCase {
    name: &'static str,
    mutation: &'static str,
    state: State,
    next_latest: Vec<CcsClaim>,
    proof: StepProof,
    candidate_state: State,
}

struct ExecutionFacts {
    event_order: Vec<&'static str>,
    dispatch: Option<&'static str>,
    nifs: Option<Option<RunningInstance>>,
    state_x_out: Option<[u8; 32]>,
    result: Result<State, construction2::Error>,
}

fn enc_inst_digest(proof: &StepProof) -> [u8; 32] {
    let bits = proof.x_out.bits();
    let mut digest = [0u8; 32];
    for (index, bit) in bits.iter().copied().enumerate() {
        digest[index / 8] |= bit << (index % 8);
    }
    digest
}

fn execution_facts(prep: &Preprocessing, state: State, next_latest: &[CcsClaim], proof: &StepProof) -> ExecutionFacts {
    let receipt = construction2::verify_step_with_execution_receipt(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &prep.vk,
        state,
        next_latest,
        proof,
        SemanticStateMode::Stateless,
        None,
    );
    let mut event_order = Vec::with_capacity(receipt.events.len());
    let mut dispatch = None;
    let mut nifs = None;
    let mut state_x_out = None;
    for event in receipt.events {
        match event {
            construction2::VerifyStepExecutionEvent::ChunkDigest { .. } => {
                event_order.push("chunk_digest");
            }
            construction2::VerifyStepExecutionEvent::Dispatch { branch } => {
                event_order.push("dispatch");
                dispatch = Some(match branch {
                    construction2::VerifyStepDispatch::InitialNoFold => "initial_no_fold",
                    construction2::VerifyStepDispatch::InitialRecursive => "initial_recursive",
                    construction2::VerifyStepDispatch::ActiveNoFold => "active_no_fold",
                    construction2::VerifyStepDispatch::ActiveRecursive => "active_recursive",
                });
            }
            construction2::VerifyStepExecutionEvent::TranscriptStarted { .. } => {
                event_order.push("transcript_started");
            }
            construction2::VerifyStepExecutionEvent::TranscriptAppend { .. } => {
                event_order.push("transcript_append");
            }
            construction2::VerifyStepExecutionEvent::TranscriptPrefix { .. } => {
                event_order.push("transcript_prefix");
            }
            construction2::VerifyStepExecutionEvent::NifsVerify { outcome, .. } => {
                event_order.push("nifs_verify");
                nifs = Some(match outcome {
                    construction2::NifsVerifyExecutionOutcome::Accepted(running) => Some(running),
                    construction2::NifsVerifyExecutionOutcome::Rejected => None,
                });
            }
            construction2::VerifyStepExecutionEvent::RunningDigest { .. } => {
                event_order.push("running_digest");
            }
            construction2::VerifyStepExecutionEvent::StateAdvanced { .. } => {
                event_order.push("state_advanced");
            }
            construction2::VerifyStepExecutionEvent::VerifierDigestRead { .. } => {
                event_order.push("verifier_digest_read");
            }
            construction2::VerifyStepExecutionEvent::PiCcsHeaderRead { .. } => {
                event_order.push("pi_ccs_header_read");
            }
            construction2::VerifyStepExecutionEvent::NebulaDigest { .. } => {
                event_order.push("nebula_digest");
            }
            construction2::VerifyStepExecutionEvent::StateXOutHash { output, .. } => {
                event_order.push("state_x_out_hash");
                let mut digest = [0u8; 32];
                for (index, bit) in output.bits().iter().copied().enumerate() {
                    digest[index / 8] |= bit << (index % 8);
                }
                state_x_out = Some(digest);
            }
        }
    }
    ExecutionFacts {
        event_order,
        dispatch,
        nifs,
        state_x_out,
        result: receipt.result,
    }
}

fn active_parts(state: &State) -> (&RunningInstance, Vec<CcsClaim>) {
    let ProofState::Active { running, latest } = &state.proof else {
        panic!("expected active linked state")
    };
    (running, latest.claims())
}

fn mapped_output(builder: &mut Builder, state: &State, x: [u8; 32]) -> MappedOutput {
    let (running, _) = active_parts(state);
    assert_eq!(state.pc, 1);
    MappedOutput {
        z_next: builder.state(state.z_i),
        running_next: builder.running(running),
        pc_next: state.pc - 1,
        x: builder.digest(x),
    }
}

fn add_case(builder: &mut Builder, prep: &Preprocessing, source: SourceCase) {
    let key = builder.key(prep.vk.digest());
    let m_in = source
        .next_latest
        .first()
        .expect("canonical step carries one fresh instance")
        .m_in;
    let canonical_default = RunningInstance::canonical_zero(
        &prep.params,
        prep.structure(),
        m_in,
        construction2::LaneCommitmentMode::Plain,
    )
    .expect("canonical HyperNova default accumulator");
    let default_running = builder.running(&canonical_default);
    let (rust_state, running, fresh) = builder.rust_state(&source.state);
    let witness = builder.witness(&source.next_latest);
    let (fold, nifs_proof) = builder.proof_map(&source.proof);
    let x_out = enc_inst_digest(&source.proof);
    let rust_input = RustInput {
        state: rust_state,
        next_latest: witness,
        fold,
        semantic_state_digest: source.proof.semantic_state_digest,
        x_out: builder.digest(x_out),
        semantic_mode: "stateless",
        nebula: "absent",
    };

    let chunk = f_prime_chunk_public_digest(source.state.step_count, &source.next_latest);
    let z_next_bytes = digest_fields_as_digest32(chunk);
    assert_eq!(source.candidate_state.z_i, z_next_bytes);
    let z_next = builder.state(z_next_bytes);
    let facts = execution_facts(prep, source.state.clone(), &source.next_latest, &source.proof);
    let rust_accepted = facts.result.is_ok();
    let rust_output = facts
        .result
        .as_ref()
        .ok()
        .map(|state| mapped_output(builder, state, x_out));
    let rust_error = facts
        .result
        .as_ref()
        .err()
        .map(|error| format!("{error:?}"));
    let nifs_output = facts
        .nifs
        .as_ref()
        .and_then(|outcome| outcome.as_ref())
        .map(|running| builder.running(running));
    let nifs_outcome = match facts.nifs {
        None => "not_called",
        Some(None) => "rejected",
        Some(Some(_)) => "accepted",
    };

    let prior_hash_input = HashInputMap {
        verifier_key: key,
        iteration: source.state.chunk_count,
        z0: builder.state(source.state.z_0),
        current: builder.state(source.state.z_i),
        running,
        pc: source.state.pc,
    };
    let trace = match &source.proof.fold {
        FoldProof::NoFold => {
            let next_hash = if source.state.chunk_count == 0 && source.state.z_0 == source.state.z_i {
                let output = facts
                    .state_x_out
                    .expect("valid base branch reaches next hash");
                Some(HashReceiptMap {
                    input: HashInputMap {
                        verifier_key: key,
                        iteration: source.state.chunk_count + 1,
                        z0: builder.state(source.state.z_0),
                        current: z_next,
                        running: default_running,
                        pc: 1,
                    },
                    output: builder.digest(output),
                })
            } else {
                None
            };
            StepTraceMap::Base { next_hash }
        }
        FoldProof::Recursive(_) if source.state.chunk_count == 0 => StepTraceMap::Recursive {
            prior_hash: None,
            fresh_public: None,
            encode: None,
            nifs: None,
            next_hash: None,
        },
        FoldProof::Recursive(_) if source.state.pc != 1 => StepTraceMap::Recursive {
            prior_hash: None,
            fresh_public: None,
            encode: None,
            nifs: None,
            next_hash: None,
        },
        FoldProof::Recursive(_) => {
            let (_, fresh_claims) = active_parts(&source.state);
            assert_eq!(fresh_claims.len(), 1, "fixed profile has one fresh claim");
            let prior_digest = compute_x_out(prep, &source.state);
            let prior_digest_id = builder.digest(prior_digest);
            let actual_public = builder.encoded(&fresh_claims[0].x);
            let expected_public_values = encode_f_prime_superneo_public_input(digest32_as_fields(prior_digest));
            let expected_public = builder.encoded(&expected_public_values);
            let linked = actual_public == expected_public;
            let prior_hash = Some(HashReceiptMap {
                input: prior_hash_input,
                output: prior_digest_id,
            });
            let fresh_public = Some(FreshPublicMap {
                input: fresh,
                output: actual_public,
            });
            let encode = Some(EncodeMap {
                input: prior_digest_id,
                output: expected_public,
            });
            if !linked {
                StepTraceMap::Recursive {
                    prior_hash,
                    fresh_public,
                    encode,
                    nifs: None,
                    next_hash: None,
                }
            } else {
                let observed = facts
                    .nifs
                    .as_ref()
                    .expect("linked recursive branch reaches NIFS");
                let folded = observed.as_ref().map(|running| builder.running(running));
                let nifs = Some(NifsMap {
                    key,
                    running,
                    fresh,
                    proof: nifs_proof,
                    output: folded,
                });
                let next_hash = match folded {
                    None => None,
                    Some(folded) => {
                        let output = facts.state_x_out.expect("accepted NIFS reaches next hash");
                        Some(HashReceiptMap {
                            input: HashInputMap {
                                verifier_key: key,
                                iteration: source.state.chunk_count + 1,
                                z0: builder.state(source.state.z_0),
                                current: z_next,
                                running: folded,
                                pc: 1,
                            },
                            output: builder.digest(output),
                        })
                    }
                };
                StepTraceMap::Recursive {
                    prior_hash,
                    fresh_public,
                    encode,
                    nifs,
                    next_hash,
                }
            }
        }
    };

    let claim = mapped_output(builder, &source.candidate_state, x_out);
    let mapped = StepCaseMap {
        verifier_key: key,
        default_running,
        iteration: source.state.chunk_count,
        z0: builder.state(source.state.z_0),
        zi: builder.state(source.state.z_i),
        running,
        fresh,
        prior_pc: source.state.pc,
        witness,
        nifs_proof,
        step_receipt: StepReceiptMap {
            state: builder.state(source.state.z_i),
            witness,
            output: z_next,
        },
        trace,
        claim,
        rust_accepted,
    };
    if rust_accepted {
        assert_eq!(
            rust_output.as_ref(),
            Some(&mapped.claim),
            "accepted Rust output must equal the mapped canonical claim"
        );
    }
    builder.cases.push(Case {
        name: source.name.to_owned(),
        mutation: source.mutation,
        rust_input,
        observed: Observed {
            event_order: facts.event_order,
            dispatch: facts.dispatch,
            nifs_outcome,
            nifs_output,
            rust_output,
            rust_error,
        },
        mapped,
    });
}

fn build_corpus() -> Corpus {
    let fixture = build_fixture();
    let base = &fixture.snapshots[0];
    let recursive = &fixture.snapshots[1];
    assert!(matches!(base.proof.fold, FoldProof::NoFold));
    assert!(matches!(recursive.proof.fold, FoldProof::Recursive(_)));

    let mut sources = vec![
        SourceCase {
            name: "honest_base",
            mutation: "none",
            state: base.state_in.clone(),
            next_latest: base.next_latest.clone(),
            proof: base.proof.clone(),
            candidate_state: base.state_out.clone(),
        },
        SourceCase {
            name: "honest_recursive",
            mutation: "none",
            state: recursive.state_in.clone(),
            next_latest: recursive.next_latest.clone(),
            proof: recursive.proof.clone(),
            candidate_state: recursive.state_out.clone(),
        },
    ];

    let mut bad_initial_state = base.state_in.clone();
    bad_initial_state.z_i[0] ^= 1;
    sources.push(SourceCase {
        name: "base_initial_state_mutation",
        mutation: "state.z_i[0] ^= 1",
        state: bad_initial_state,
        next_latest: base.next_latest.clone(),
        proof: base.proof.clone(),
        candidate_state: base.state_out.clone(),
    });

    let mut initial_recursive = base.proof.clone();
    initial_recursive.fold = recursive.proof.fold.clone();
    sources.push(SourceCase {
        name: "initial_with_recursive_fold",
        mutation: "proof.fold := recursive",
        state: base.state_in.clone(),
        next_latest: base.next_latest.clone(),
        proof: initial_recursive,
        candidate_state: base.state_out.clone(),
    });

    let mut active_no_fold = recursive.proof.clone();
    active_no_fold.fold = FoldProof::NoFold;
    sources.push(SourceCase {
        name: "active_with_no_fold",
        mutation: "proof.fold := no_fold",
        state: recursive.state_in.clone(),
        next_latest: recursive.next_latest.clone(),
        proof: active_no_fold,
        candidate_state: recursive.state_out.clone(),
    });

    let mut bad_pc = recursive.state_in.clone();
    bad_pc.pc = 2;
    sources.push(SourceCase {
        name: "recursive_prior_pc_mutation",
        mutation: "state.pc := 2",
        state: bad_pc,
        next_latest: recursive.next_latest.clone(),
        proof: recursive.proof.clone(),
        candidate_state: recursive.state_out.clone(),
    });

    let mut bad_link = recursive.state_in.clone();
    let ProofState::Active { latest, .. } = &mut bad_link.proof else {
        panic!("recursive linked state must be active")
    };
    latest.instances[0].claim.x[1] += F::ONE;
    sources.push(SourceCase {
        name: "recursive_prior_public_link_mutation",
        mutation: "state.latest[0].claim.x[1] += 1",
        state: bad_link,
        next_latest: recursive.next_latest.clone(),
        proof: recursive.proof.clone(),
        candidate_state: recursive.state_out.clone(),
    });

    let mut bad_nifs = recursive.proof.clone();
    let FoldProof::Recursive(nifs) = &mut bad_nifs.fold else {
        panic!("recursive proof")
    };
    nifs.pi_dec.children[0].c.data[0] += F::ONE;
    sources.push(SourceCase {
        name: "recursive_nifs_proof_mutation",
        mutation: "proof.nifs.pi_dec.children[0].commitment[0] += 1",
        state: recursive.state_in.clone(),
        next_latest: recursive.next_latest.clone(),
        proof: bad_nifs,
        candidate_state: recursive.state_out.clone(),
    });

    let mut bad_x_out = recursive.proof.clone();
    let mut x_out = enc_inst_digest(&bad_x_out);
    x_out[0] ^= 1;
    bad_x_out.x_out = construction2::EncInst::from_digest(x_out);
    sources.push(SourceCase {
        name: "recursive_x_out_mutation",
        mutation: "proof.x_out.bytes[0] ^= 1",
        state: recursive.state_in.clone(),
        next_latest: recursive.next_latest.clone(),
        proof: bad_x_out,
        candidate_state: recursive.state_out.clone(),
    });

    let mut builder = Builder::default();
    for source in sources {
        add_case(&mut builder, &fixture.prep, source);
    }
    let profile = Profile {
        name: "linked_bit_carrier_one_slot_stateless",
        relation_rows: fixture.prep.structure().n,
        relation_columns: fixture.prep.structure().m,
        matrix_count: fixture.prep.structure().t(),
        public_input_len: fixture
            .prep
            .public_input_len
            .expect("linked profile has public inputs"),
        semantic_mode: "stateless",
        fresh_count: 1,
        verifier_key_digest: fixture.prep.vk.digest(),
        structure_digest: fixture.prep.structure_digest().map(felt),
    };
    Corpus {
        schema: SCHEMA,
        evidence_tier: "bounded Rust-conformant differential",
        scope: "one slot, one fresh instance, stateless linked bit-carrier profile",
        primitive_boundary: "NIFS/hash outcomes are Rust receipts; this corpus does not prove primitive refinement",
        excluded_claims: vec![
            "general Rust acceptance refinement",
            "NIFS primitive correctness",
            "Poseidon2 or Ajtai internals",
            "terminal verifier conformance",
            "R1CS soundness or assignment completeness",
            "typed IR or row/column certification",
        ],
        profile,
        atoms: builder.atoms,
        cases: builder.cases,
    }
}
