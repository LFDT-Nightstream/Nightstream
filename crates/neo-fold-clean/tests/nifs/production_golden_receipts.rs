//! Deterministic production-path golden receipt for the three NIFS reductions.
//!
//! The fixture uses one real R1CS and the normal `NIFS.P -> NIFS.V` path.
//! Rust regenerates every verifier-visible value. Lean checks the committed
//! receipt, while this test checks that production still emits the same bytes.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use neo_ccs::Mat;
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::lifecycle::Preprocessing;
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::digest::{self, AccumulatorHandle};
use neo_fold_clean::paper::nifs::NifsProof;
use neo_fold_clean::paper::relations::{CcsClaim, CeClaim};
use neo_fold_clean::paper::{nifs, pi_ccs, pi_rlc};
use neo_math::{KExtensions, D, F, K};
use neo_reductions::{verify_and_export_pi_ccs_receipt, PiCcsExecutionReceipt};
use neo_transcript::{Poseidon2Transcript, Transcript as NeoTranscript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const FIXTURE_SEED: u64 = 0x4e53_474f_4c44_0001;
const GOLDEN_BIN: &str = "tests/data/nifs_production_golden_v1.bin";
const GENERATED_LEAN_DIR: &str =
    "../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/NifsProductionGolden/Generated";
const GENERATED_NAMESPACE: &str = "Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden.Generated";

struct GoldenRun {
    prep: Preprocessing,
    relation: R1cs,
    assignment: Vec<F>,
    fresh_claims: Vec<CcsClaim>,
    running: RunningInstance,
    proof: NifsProof,
    rhos: Vec<neo_reductions::common::RotRho>,
    receipt: PiCcsExecutionReceipt,
    pi_ccs_post_digest: neo_fold_clean::engine::transcript::Poseidon2TranscriptSnapshot,
    rho_start: neo_fold_clean::engine::transcript::Poseidon2TranscriptSnapshot,
}

#[derive(Clone)]
struct PoseidonPermutationTrace {
    states: Vec<[F; 8]>,
}

struct TranscriptWitness {
    permutations: Vec<PoseidonPermutationTrace>,
    pi_ccs_permutation_count: usize,
    rho_start_permutation_count: usize,
}

#[derive(Clone, Copy)]
struct PoseidonTraceConstants {
    initial: [[F; 8]; 4],
    terminal: [[F; 8]; 4],
    internal: [F; 22],
    diagonal: [F; 8],
}

fn poseidon_trace_constants() -> PoseidonTraceConstants {
    let constants = neo_ccs::crypto::poseidon2_goldilocks::round_constants();
    let initial = std::array::from_fn(|round| constants.initial[round].map(F::from_u64));
    let terminal = std::array::from_fn(|round| constants.terminal[round].map(F::from_u64));
    let internal = std::array::from_fn(|round| F::from_u64(constants.internal[round]));
    PoseidonTraceConstants {
        initial,
        terminal,
        internal,
        diagonal: constants.diag.map(F::from_u64),
    }
}

fn poseidon_sbox7(value: F) -> F {
    let square = value * value;
    let fourth = square * square;
    fourth * square * value
}

fn apply_mat4(values: &mut [F; 4]) {
    let [x0, x1, x2, x3] = *values;
    let t01 = x0 + x1;
    let t23 = x2 + x3;
    let t0123 = t01 + t23;
    let t01123 = t0123 + x1;
    let t01233 = t0123 + x3;
    values[3] = t01233 + F::from_u64(2) * x0;
    values[1] = t01123 + F::from_u64(2) * x2;
    values[0] = t01123 + t01;
    values[2] = t01233 + t23;
}

fn external_linear(input: [F; 8]) -> [F; 8] {
    let mut low = [input[0], input[1], input[2], input[3]];
    let mut high = [input[4], input[5], input[6], input[7]];
    apply_mat4(&mut low);
    apply_mat4(&mut high);
    let sums = [low[0] + high[0], low[1] + high[1], low[2] + high[2], low[3] + high[3]];
    std::array::from_fn(|lane| {
        let block = if lane < 4 { low[lane] } else { high[lane - 4] };
        block + sums[lane % 4]
    })
}

fn internal_linear(input: [F; 8], diagonal: [F; 8]) -> [F; 8] {
    let sum = input
        .iter()
        .copied()
        .fold(F::ZERO, |total, value| total + value);
    std::array::from_fn(|lane| sum + diagonal[lane] * input[lane])
}

fn poseidon_permutation_trace(input: [F; 8]) -> PoseidonPermutationTrace {
    let constants = poseidon_trace_constants();
    let mut states = Vec::with_capacity(31);
    let mut state = external_linear(input);
    states.push(state);
    for constants in constants.initial {
        state = external_linear(std::array::from_fn(|lane| {
            poseidon_sbox7(state[lane] + constants[lane])
        }));
        states.push(state);
    }
    for constant in constants.internal {
        state[0] = poseidon_sbox7(state[0] + constant);
        state = internal_linear(state, constants.diagonal);
        states.push(state);
    }
    for constants in constants.terminal {
        state = external_linear(std::array::from_fn(|lane| {
            poseidon_sbox7(state[lane] + constants[lane])
        }));
        states.push(state);
    }
    assert_eq!(states.len(), 31);
    assert_eq!(
        *states.last().expect("Poseidon trace has a final state"),
        neo_ccs::crypto::poseidon2_goldilocks::permute_state(input)
    );
    PoseidonPermutationTrace { states }
}

struct WitnessedTranscript {
    state: [F; 8],
    absorbed: usize,
    permutations: Vec<PoseidonPermutationTrace>,
}

impl WitnessedTranscript {
    fn from_snapshot(state: [u64; 8], absorbed: usize) -> Self {
        Self {
            state: state.map(F::from_u64),
            absorbed,
            permutations: Vec::new(),
        }
    }

    fn permute(&mut self) {
        let trace = poseidon_permutation_trace(self.state);
        self.state = *trace
            .states
            .last()
            .expect("Poseidon trace has a final state");
        self.absorbed = 0;
        self.permutations.push(trace);
    }

    fn absorb_field(&mut self, value: F) {
        if self.absorbed >= 4 {
            self.permute();
        }
        self.state[self.absorbed] = value;
        self.absorbed += 1;
    }

    fn absorb_fields(&mut self, values: impl IntoIterator<Item = F>) {
        for value in values {
            self.absorb_field(value);
        }
    }

    fn append_fields_raw(&mut self, values: &[F]) {
        self.absorb_field(F::from_u64(values.len() as u64));
        self.absorb_fields(values.iter().copied());
    }

    fn absorb_packed_bytes_with_len(&mut self, bytes: &[u8]) {
        self.absorb_field(F::from_u64(bytes.len() as u64));
        for chunk in bytes.chunks(7) {
            let mut limb = [0_u8; 8];
            limb[..chunk.len()].copy_from_slice(chunk);
            self.absorb_field(F::from_u64(u64::from_le_bytes(limb)));
        }
    }

    fn append_fields(&mut self, label: &[u8], values: &[F]) {
        self.absorb_field(F::from_u64(2));
        self.absorb_packed_bytes_with_len(label);
        self.absorb_field(F::from_u64(values.len() as u64));
        self.absorb_fields(values.iter().copied());
    }

    fn challenge_fields_raw(&mut self) -> [F; 2] {
        self.absorb_field(F::ONE);
        self.permute();
        [self.state[0], self.state[1]]
    }

    fn digest32(&mut self) -> [u8; 32] {
        self.absorb_field(F::ONE);
        self.permute();
        let mut digest = [0_u8; 32];
        for (lane, value) in self.state[..4].iter().enumerate() {
            digest[lane * 8..(lane + 1) * 8].copy_from_slice(&value.as_canonical_u64().to_le_bytes());
        }
        self.absorb_field(F::from_u64(0x104));
        self.absorb_field(F::from_u64(32));
        digest
    }

    fn assert_snapshot(&self, snapshot: &neo_fold_clean::engine::transcript::Poseidon2TranscriptSnapshot) {
        assert_eq!(self.state, snapshot.state());
        assert_eq!(self.absorbed, snapshot.absorbed());
    }
}

fn snapshot(transcript: &Poseidon2Transcript) -> neo_fold_clean::engine::transcript::Poseidon2TranscriptSnapshot {
    neo_fold_clean::engine::transcript::Poseidon2TranscriptSnapshot::from_state_and_absorbed(
        transcript.state(),
        transcript.absorbed(),
    )
}

fn replay_transcript_witness(run: &GoldenRun) -> TranscriptWitness {
    let statement = &run.receipt.statement;
    let mut transcript = WitnessedTranscript::from_snapshot(statement.transcript_state, statement.transcript_absorbed);
    let fields = |values: &[u64]| values.iter().copied().map(F::from_u64).collect::<Vec<_>>();

    transcript.absorb_fields(fields(&statement.public_fields));
    transcript.absorb_fields(fields(&statement.pi_ccs_statement_fields));
    for index in 0..6 {
        transcript.absorb_fields([F::from_u64(42), F::from_u64(index)]);
        let _ = transcript.challenge_fields_raw();
    }
    transcript.absorb_field(F::from_u64(43));
    let _ = transcript.challenge_fields_raw();
    for (round, coefficients) in run.proof.pi_ccs.sumcheck.sumcheck_rounds.iter().enumerate() {
        transcript.absorb_fields([
            F::from_u64(45),
            F::from_u64(round as u64),
            F::from_u64(coefficients.len() as u64),
        ]);
        for coefficient in coefficients {
            transcript.absorb_fields(coefficient.as_coeffs());
        }
        transcript.absorb_fields([F::from_u64(46), F::from_u64(round as u64)]);
        let _ = transcript.challenge_fields_raw();
    }
    let pi_ccs_permutation_count = transcript.permutations.len();

    let fold_digest = transcript.digest32();
    transcript.assert_snapshot(&run.pi_ccs_post_digest);
    assert!(run
        .proof
        .pi_ccs
        .outputs
        .iter()
        .all(|output| output.fold_digest == fold_digest));
    transcript.append_fields(b"pi_rlc/input_claims_digest", &run.proof.pi_ccs.outputs_digest);
    transcript.assert_snapshot(&run.rho_start);
    let rho_start_permutation_count = transcript.permutations.len();

    transcript.append_fields_raw(&[F::ZERO, F::ZERO]);
    let mut sampled_symbols = Vec::with_capacity(64);
    for counter in 0..neo_params::goldilocks_paper_b2::PI_RLC_SAMPLER_DIGEST_ROUNDS {
        transcript.append_fields_raw(&[F::ONE, F::from_usize(counter)]);
        let digest = transcript.digest32();
        for lane in digest.chunks_exact(8) {
            let value = u64::from_le_bytes(lane.try_into().expect("digest lane is eight bytes"));
            for offset in [0, 16] {
                let raw = ((value >> offset) & 0xffff) as u16;
                let candidate = (!raw) as u64;
                if candidate < 65535 {
                    sampled_symbols.push(candidate % 5);
                }
            }
        }
    }
    assert!(sampled_symbols.len() >= D);
    sampled_symbols.truncate(D);
    let alphabet = [-F::from_u64(2), -F::ONE, F::ZERO, F::ONE, F::from_u64(2)];
    for (coefficient, symbol) in sampled_symbols.into_iter().enumerate() {
        assert_eq!(run.rhos[0].as_mat()[(coefficient, 0)], alphabet[symbol as usize]);
    }

    TranscriptWitness {
        permutations: transcript.permutations,
        pi_ccs_permutation_count,
        rho_start_permutation_count,
    }
}

fn production_fixture_relation() -> R1cs {
    let mut a = Mat::zero(1, D, F::ZERO);
    a[(0, 1)] = F::ONE;
    let mut b = Mat::zero(1, D, F::ZERO);
    b[(0, 0)] = F::ONE;
    let mut c = Mat::zero(1, D, F::ZERO);
    c[(0, 1)] = F::ONE;
    R1cs { a, b, c, m_in: D }
}

fn run_production_fixture() -> GoldenRun {
    let relation = production_fixture_relation();
    let prep = direct_ccs::preprocess_seeded(&relation, FIXTURE_SEED).expect("production-profile preprocessing");
    let mut assignment = vec![F::ZERO; prep.structure().m];
    assignment[0] = F::ONE;
    assignment[1] = F::ONE;
    let fresh = direct_ccs::build_instance(&prep, &relation, &assignment).expect("real R1CS instance");
    let fresh_claims = vec![fresh.claim.clone()];
    let running = RunningInstance::default();

    let mut prover_transcript = Transcript::session();
    let (next_running, proof) = nifs::prove(
        &mut prover_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh],
        &running,
    )
    .expect("production NIFS prover");

    let mut verifier_transcript = Transcript::session();
    let verified = nifs::verify(
        &mut verifier_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &fresh_claims,
        &running,
        &proof,
    )
    .expect("production NIFS verifier");
    assert_eq!(verified.claims, next_running.claims);

    let mut phase_transcript = Transcript::session();
    let pi_ccs_outputs = pi_ccs::verify(
        &mut phase_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &fresh_claims,
        &running,
        &proof.pi_ccs,
    )
    .expect("production PiCCS verifier");
    let (rhos, rho_start) =
        pi_rlc::derive_rhos_for_inputs_with_sampling_start(&mut phase_transcript, &prep.params, &pi_ccs_outputs)
            .expect("production PiRLC sampler");

    let public_instance_digest = digest::pi_ccs_instance_digest_parent_authority(
        &fresh_claims,
        running.claims.len(),
        running.parent_authority.as_ref(),
    );
    let mut receipt_transcript = Poseidon2Transcript::new(b"neo.fold.clean/session/v1");
    let receipt = verify_and_export_pi_ccs_receipt(
        &mut receipt_transcript,
        prep.params.inner(),
        prep.structure(),
        &fresh_claims,
        &running.claims,
        &proof.pi_ccs.outputs,
        &proof.pi_ccs.sumcheck,
        prep.optimized_cache(),
        public_instance_digest,
        AccumulatorHandle::empty().digest_fields(),
    )
    .expect("accepted production PiCCS receipt");

    assert_eq!(receipt.proof.proof_bytes, proof.pi_ccs.sumcheck.canonical_bytes());
    assert_eq!(proof.pi_ccs.outputs, pi_ccs_outputs);
    let pi_ccs_post_digest = snapshot(&receipt_transcript);

    GoldenRun {
        prep,
        relation,
        assignment,
        fresh_claims,
        running,
        proof,
        rhos,
        receipt,
        pi_ccs_post_digest,
        rho_start,
    }
}

fn lean_nat_list<I>(values: I, values_per_line: usize) -> String
where
    I: IntoIterator,
    I::Item: ToString,
{
    let values = values
        .into_iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return "[]".to_owned();
    }
    let lines = values
        .chunks(values_per_line)
        .map(|chunk| chunk.join(", "))
        .collect::<Vec<_>>();
    if lines.len() == 1 {
        format!("[{}]", lines[0])
    } else {
        format!("[{}]", lines.join(",\n    "))
    }
}

fn lean_raw_k_list(values: impl IntoIterator<Item = (u64, u64)>) -> String {
    let values = values
        .into_iter()
        .map(|(low, high)| format!("{{ low := {low}, high := {high} }}"))
        .collect::<Vec<_>>();
    if values.is_empty() {
        return "[]".to_owned();
    }
    let lines = values
        .chunks(3)
        .map(|chunk| chunk.join(", "))
        .collect::<Vec<_>>();
    format!("[{}]", lines.join(",\n    "))
}

fn k_limbs(value: K) -> (u64, u64) {
    value.to_limbs_u64()
}

fn field_values(values: impl IntoIterator<Item = F>) -> Vec<u64> {
    values
        .into_iter()
        .map(|value| value.as_canonical_u64())
        .collect()
}

fn raw_claim(claim: &CeClaim) -> String {
    let public_input =
        (0..claim.X.rows()).flat_map(|row| (0..claim.X.cols()).map(move |column| claim.X[(row, column)]));
    let evaluations = claim
        .y_ring
        .iter()
        .flat_map(|row| row.iter().copied())
        .map(k_limbs);
    let fold_digest = digest::digest32_as_fields(claim.fold_digest);
    format!(
        "  commitment :=\n    {{ degree := {}\n      verifierRows := {}\n      data := {} }}\n  publicRows := {}\n  publicColumns := {}\n  publicInput := {}\n  point := {}\n  evaluations := {}\n  constantTerms := {}\n  publicWidth := {}\n  foldDigest := {}\n  advPresent := {}",
        claim.c.d,
        claim.c.kappa,
        lean_nat_list(field_values(claim.c.data.iter().copied()), 8),
        claim.X.rows(),
        claim.X.cols(),
        lean_nat_list(field_values(public_input), 8),
        lean_raw_k_list(claim.r.iter().copied().map(k_limbs)),
        lean_raw_k_list(evaluations),
        lean_raw_k_list(claim.ct.iter().copied().map(k_limbs)),
        claim.m_in,
        lean_nat_list(field_values(fold_digest), 4),
        claim.adv.is_some(),
    )
}

fn generated_header(module_purpose: &str) -> String {
    format!(
        "import Nightstream.Implementation.Rust.NifsProductionGolden.Receipt\n\n\
/-!\nGENERATED FILE - do not edit by hand.\n\n\
{module_purpose}\n\
Regenerated by `cargo test -p neo-fold-clean --release --test nifs_production_golden_receipts`.\n\
-/\n\n\
namespace {GENERATED_NAMESPACE}\n\n\
open Nightstream.Implementation.Rust.NifsProductionGolden\n\
open Nightstream.Implementation.Rust.PiCcsExecution\n\n"
    )
}

fn render_pi_ccs(run: &GoldenRun) -> String {
    let mut relation_matrices = Vec::with_capacity(3 * D);
    for matrix in [&run.relation.a, &run.relation.b, &run.relation.c] {
        for row in 0..matrix.rows() {
            for column in 0..matrix.cols() {
                relation_matrices.push(matrix[(row, column)].as_canonical_u64());
            }
        }
    }
    let statement = &run.receipt.statement;
    let proof = &run.receipt.proof;
    let mut out = generated_header("Exact PiCCS statement, proof, and PiRLC transcript handoff.");
    writeln!(
        out,
        "def relationId : List Nat :=\n  {}\n\ndef relationMatrices : List Nat :=\n  {}\n\ndef fixtureAssignment : List Nat :=\n  {}\n",
        lean_nat_list(statement.relation_id, 4),
        lean_nat_list(relation_matrices, 8),
        lean_nat_list(field_values(run.assignment.iter().copied()), 8),
    )
    .unwrap();
    writeln!(
        out,
        "def piCcsStatement : PiCcsCanonicalStatement :=\n  {{ relationId := relationId\n    transcriptState := {}\n    transcriptAbsorbed := {}\n    publicFields := {}\n    piCcsStatementFields := {}\n    priorPoint := {}\n    claimedCoefficients := {} }}\n",
        lean_nat_list(statement.transcript_state, 8),
        statement.transcript_absorbed,
        lean_nat_list(statement.public_fields.iter().copied(), 8),
        lean_nat_list(statement.pi_ccs_statement_fields.iter().copied(), 8),
        lean_raw_k_list(statement.prior_point.iter().map(|value| (value.low, value.high))),
        lean_raw_k_list(statement.claimed_coefficients.iter().map(|value| (value.low, value.high))),
    )
    .unwrap();
    writeln!(
        out,
        "def piCcsProof : PiCcsExecutionProof :=\n  {{ proofBytes := {}\n    fullOutput := {} }}\n",
        lean_nat_list(proof.proof_bytes.iter().copied(), 16),
        lean_raw_k_list(
            proof
                .full_output
                .iter()
                .map(|value| (value.low, value.high))
        ),
    )
    .unwrap();
    writeln!(
        out,
        "def piCcsOutputsDigest : List Nat :=\n  {}\n\ndef rhoStart : RawTranscriptSnapshot :=\n  {{ lanes := {}\n    absorbed := {} }}\n\ndef canonicalNifsProofByteCount : Nat := {}\n\nend {GENERATED_NAMESPACE}",
        lean_nat_list(field_values(run.proof.pi_ccs.outputs_digest), 4),
        lean_nat_list(field_values(run.rho_start.state()), 8),
        run.rho_start.absorbed(),
        run.proof.canonical_bytes().len(),
    )
    .unwrap();
    out
}

fn render_claim_definition(name: &str, claim: &CeClaim, purpose: &str) -> String {
    let mut out = generated_header(purpose);
    writeln!(
        out,
        "def {name} : RawClaim where\n{}\n\nend {GENERATED_NAMESPACE}",
        raw_claim(claim)
    )
    .unwrap();
    out
}

fn render_claim_list_definition(name: &str, claims: &[CeClaim], purpose: &str) -> String {
    let mut out = generated_header(purpose);
    let claim_names = claims
        .iter()
        .enumerate()
        .map(|(index, claim)| {
            let claim_name = format!("{name}Claim{index}");
            writeln!(out, "private def {claim_name} : RawClaim where\n{}\n", raw_claim(claim)).unwrap();
            claim_name
        })
        .collect::<Vec<_>>();
    writeln!(
        out,
        "def {name} : List RawClaim :=\n  [{}]\n\nend {GENERATED_NAMESPACE}",
        claim_names.join(", ")
    )
    .unwrap();
    out
}

fn render_poseidon_trace_shard(name: &str, traces: &[PoseidonPermutationTrace]) -> String {
    let mut out = generated_header("Exact internal states for canonical Poseidon2 permutations.");
    out.push_str("set_option maxRecDepth 10000\n\n");
    let rendered = traces
        .iter()
        .map(|trace| {
            let states = trace
                .states
                .iter()
                .map(|state| lean_nat_list(field_values(state.iter().copied()), 8))
                .collect::<Vec<_>>()
                .join(",\n      ");
            format!("{{ states := [{states}] }}")
        })
        .collect::<Vec<_>>()
        .join(",\n   ");
    writeln!(
        out,
        "def {name} : List RawPermutationTrace :=\n  [{rendered}]\n\nend {GENERATED_NAMESPACE}"
    )
    .unwrap();
    out
}

fn render_poseidon_traces(witness: &TranscriptWitness, shard_count: usize) -> String {
    let mut out = String::new();
    for shard in 0..shard_count {
        writeln!(out, "import {GENERATED_NAMESPACE}.PoseidonTraces{shard}").unwrap();
    }
    let shards = (0..shard_count)
        .map(|shard| format!("poseidonPermutationTraces{shard}"))
        .collect::<Vec<_>>()
        .join(" ++\n    ");
    writeln!(
        out,
        "\n/-! GENERATED FILE - assembled Poseidon2 permutation witnesses. -/\n\n\
namespace {GENERATED_NAMESPACE}\n\n\
open Nightstream.Implementation.Rust.NifsProductionGolden\n\n\
def poseidonPermutationTraces : List RawPermutationTrace :=\n  {shards}\n\n\
def piCcsPermutationCount : Nat := {}\n\n\
def rhoStartPermutationCount : Nat := {}\n\n\
end {GENERATED_NAMESPACE}",
        witness.pi_ccs_permutation_count, witness.rho_start_permutation_count,
    )
    .unwrap();
    out
}

fn render_receipt() -> String {
    let mut out = String::new();
    out.push_str("import Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden.Generated.PiCcs\n");
    out.push_str("import Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden.Generated.PoseidonTraces\n");
    out.push_str("import Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden.Generated.PiRlcInput\n");
    out.push_str("import Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden.Generated.PiRlcCombined\n");
    for shard in 0..7 {
        writeln!(
            out,
            "import Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden.Generated.PiDecChildren{shard}"
        )
        .unwrap();
    }
    writeln!(
        out,
        "\n/-! GENERATED FILE - assembled deterministic production NIFS receipt. -/\n\n\
namespace {GENERATED_NAMESPACE}\n\n\
open Nightstream.Implementation.Rust.NifsProductionGolden\n\n\
def piDecChildren : List RawClaim :=\n  piDecChildren0 ++ piDecChildren1 ++ piDecChildren2 ++ piDecChildren3 ++\n    piDecChildren4 ++ piDecChildren5 ++ piDecChildren6\n\n\
def receipt : ProductionReceipt :=\n  {{ relationId := relationId\n    relationMatrices := relationMatrices\n    fixtureAssignment := fixtureAssignment\n    piCcsStatement := piCcsStatement\n    piCcsProof := piCcsProof\n    poseidonPermutationTraces := poseidonPermutationTraces\n    piCcsPermutationCount := piCcsPermutationCount\n    rhoStartPermutationCount := rhoStartPermutationCount\n    piCcsOutputsDigest := piCcsOutputsDigest\n    rhoStart := rhoStart\n    piRlcInputs := piRlcInputs\n    piRlcCombined := piRlcCombined\n    piDecChildren := piDecChildren\n    canonicalNifsProofByteCount := canonicalNifsProofByteCount }}\n\n\
end {GENERATED_NAMESPACE}"
    )
    .unwrap();
    out
}

fn generated_lean_files(run: &GoldenRun, witness: &TranscriptWitness) -> Vec<(String, String)> {
    let mut files = vec![
        ("PiCcs.lean".to_owned(), render_pi_ccs(run)),
        (
            "PiRlcInput.lean".to_owned(),
            render_claim_list_definition("piRlcInputs", &run.proof.pi_ccs.outputs, "Exact PiRLC input claim."),
        ),
        (
            "PiRlcCombined.lean".to_owned(),
            render_claim_definition(
                "piRlcCombined",
                &run.proof.pi_rlc.combined,
                "Exact PiRLC combined claim.",
            ),
        ),
    ];
    for (shard, children) in run.proof.pi_dec.children.chunks(2).enumerate() {
        files.push((
            format!("PiDecChildren{shard}.lean"),
            render_claim_list_definition(
                &format!("piDecChildren{shard}"),
                children,
                "Exact ordered PiDEC child-claim shard.",
            ),
        ));
    }
    let traces_per_shard = 8;
    for (shard, traces) in witness.permutations.chunks(traces_per_shard).enumerate() {
        let definition = format!("poseidonPermutationTraces{shard}");
        files.push((
            format!("PoseidonTraces{shard}.lean"),
            render_poseidon_trace_shard(&definition, traces),
        ));
    }
    let trace_shard_count = witness.permutations.len().div_ceil(traces_per_shard);
    files.push((
        "PoseidonTraces.lean".to_owned(),
        render_poseidon_traces(witness, trace_shard_count),
    ));
    files.push(("Receipt.lean".to_owned(), render_receipt()));
    files
}

fn manifest_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn compare_or_write_expected(path: &Path, generated: &[u8], drifted: &mut Vec<PathBuf>) {
    if fs::read(path).ok().as_deref() == Some(generated) {
        return;
    }
    let expected = PathBuf::from(format!("{}.expected", path.display()));
    fs::create_dir_all(expected.parent().expect("golden path has a parent")).expect("create golden parent");
    fs::write(&expected, generated).expect("write generated golden candidate");
    drifted.push(expected);
}

fn assert_rejected(run: &GoldenRun, proof: &NifsProof, label: &str) {
    let mut transcript = Transcript::session();
    assert!(
        nifs::verify(
            &mut transcript,
            &run.prep.params,
            run.prep.structure(),
            run.prep.optimized_cache(),
            run.prep.mix_rhos_commits(),
            run.prep.combine_b_pows(),
            &run.fresh_claims,
            &run.running,
            proof,
        )
        .is_err(),
        "production verifier accepted mutated {label} data"
    );
}

#[test]
fn production_nifs_golden_receipt_matches_and_rejects_phase_mutations() {
    let run = run_production_fixture();
    let transcript_witness = replay_transcript_witness(&run);

    let mut changed_pi_ccs = run.proof.clone();
    changed_pi_ccs.pi_ccs.sumcheck.sumcheck_rounds[0][0] += K::ONE;
    assert_rejected(&run, &changed_pi_ccs, "PiCCS");

    let mut changed_outputs_digest = run.proof.clone();
    changed_outputs_digest.pi_ccs.outputs_digest[0] += F::ONE;
    assert_rejected(&run, &changed_outputs_digest, "PiCCS outputs digest");

    let mut changed_fold_digest = run.proof.clone();
    changed_fold_digest.pi_ccs.outputs[0].fold_digest[0] ^= 1;
    assert_rejected(&run, &changed_fold_digest, "PiCCS fold digest");

    let mut changed_pi_rlc = run.proof.clone();
    changed_pi_rlc.pi_rlc.combined.X[(0, 0)] += F::ONE;
    assert_rejected(&run, &changed_pi_rlc, "PiRLC");

    let mut changed_pi_dec = run.proof.clone();
    changed_pi_dec.pi_dec.children[0].X[(0, 0)] += F::ONE;
    assert_rejected(&run, &changed_pi_dec, "PiDEC");

    let mut drifted = Vec::new();
    compare_or_write_expected(&manifest_path(GOLDEN_BIN), &run.proof.canonical_bytes(), &mut drifted);
    for (name, rendered) in generated_lean_files(&run, &transcript_witness) {
        compare_or_write_expected(
            &manifest_path(&format!("{GENERATED_LEAN_DIR}/{name}")),
            rendered.as_bytes(),
            &mut drifted,
        );
    }
    assert!(
        drifted.is_empty(),
        "production NIFS golden data drifted; inspect and deliberately promote every .expected file: {drifted:?}"
    );
}

#[test]
fn production_fixture_uses_the_selected_small_shape() {
    let run = run_production_fixture();
    assert_eq!(run.prep.structure().n, 1);
    assert_eq!(run.prep.structure().m, D);
    assert_eq!(run.proof.pi_ccs.sumcheck.sumcheck_rounds.len(), 6);
    assert!(run
        .proof
        .pi_ccs
        .sumcheck
        .sumcheck_rounds
        .iter()
        .all(|round| round.len() == 5));
    assert_eq!(run.proof.pi_ccs.outputs.len(), 1);
    assert_eq!(run.proof.pi_dec.children.len(), 14);
}
