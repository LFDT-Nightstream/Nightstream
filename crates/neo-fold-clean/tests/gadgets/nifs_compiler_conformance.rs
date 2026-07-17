//! Exact compiler artifact for the claimed-chain core shared by both Π_CCS
//! SumChecks.  Full-history call-site maps are added below this isolated
//! semantic anchor; the generated rows, not audit metadata, remain authority.

#[path = "checked_program_artifact_support.rs"]
#[allow(dead_code)]
mod checked_program_artifact_support;

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use checked_program_artifact_support::{canonicalize_program, lean_instructions, normalize_range};
use neo_ccs::Mat;
use neo_fold_clean::engine::decider::synthesize_statement_r1cs;
use neo_fold_clean::engine::r1cs_circuit::field_ext::KVar;
use neo_fold_clean::engine::r1cs_circuit::{enforce_sumcheck_round, R1csBuilder};
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::construction2::{self, State};
use neo_fold_clean::paper::digest::{
    digest32_as_fields, initial_boundary_digest, public_trace_seed_digest, state_x_out_digest_with_mode,
    structure_digest, AccumulatorHandle, StateXOutDigestMode,
};
use neo_fold_clean::paper::f_prime::r1cs::{encode_f_prime_public_input, F_PRIME_PUBLIC_INPUT_LEN};
use neo_fold_clean::CcsInstance;
use neo_math::{KExtensions, F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const ARTIFACT_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/Sumcheck/SumcheckRoundArtifact.lean";
const CALLS_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeFullHistory/FPrimeFullHistorySumcheckArtifact.lean";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn kval(c0: u64, c1: u64) -> K {
    K::from_coeffs([F::from_u64(c0), F::from_u64(c1)])
}

fn alloc_k(builder: &mut R1csBuilder, value: K) -> KVar {
    let [c0, c1] = value.as_coeffs();
    KVar::alloc(builder, c0, c1)
}

fn bit_carrier_r1cs() -> R1cs {
    R1cs {
        a: Mat::zero(1, F_PRIME_PUBLIC_INPUT_LEN, F::ZERO),
        b: Mat::zero(1, F_PRIME_PUBLIC_INPUT_LEN, F::ZERO),
        c: Mat::zero(1, F_PRIME_PUBLIC_INPUT_LEN, F::ZERO),
        m_in: F_PRIME_PUBLIC_INPUT_LEN,
    }
}

fn state_x_out(prep: &neo_fold_clean::Preprocessing, state: &State) -> [F; 4] {
    digest32_as_fields(state_x_out_digest_with_mode(
        StateXOutDigestMode::Stateless,
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
    ))
}

fn base_state(prep: &neo_fold_clean::Preprocessing) -> State {
    let structure = structure_digest(prep.structure());
    let z_0 = initial_boundary_digest(&structure, prep.public_input_len);
    let public_trace = public_trace_seed_digest(&structure);
    let acc_digest = AccumulatorHandle::empty().digest();
    State::base(z_0, public_trace, acc_digest, acc_digest)
}

fn build_link_instance(prep: &neo_fold_clean::Preprocessing, r1cs: &R1cs, x_out_target: [F; 4]) -> CcsInstance {
    let mut z = encode_f_prime_public_input(x_out_target);
    z.resize(prep.structure().m, F::ZERO);
    direct_ccs::build_instance(prep, r1cs, &z).expect("recursive-link instance")
}

fn peek_next_state(prep: &neo_fold_clean::Preprocessing, state: &State, batch: &[CcsInstance]) -> State {
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
    .expect("peek")
    .0
}

fn full_history_fixture() -> (neo_fold_clean::Preprocessing, neo_fold_clean::UncompressedAudit) {
    let r1cs = bit_carrier_r1cs();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 42).expect("preprocess");
    let placeholder_z = vec![F::ZERO; prep.structure().m];
    let dummy = || direct_ccs::build_instance(&prep, &r1cs, &placeholder_z).expect("dummy");
    let mut state = base_state(&prep);
    let mut steps = Vec::with_capacity(2);
    let mut public_batches = Vec::with_capacity(2);
    for _ in 0..2 {
        let predicted = peek_next_state(&prep, &state, &[dummy()]);
        let batch = build_link_instance(&prep, &r1cs, state_x_out(&prep, &predicted));
        let public_batch = vec![batch.claim.clone()];
        let (next_state, step_proof) = construction2::step(
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            prep.structure_digest(),
            &prep.log,
            prep.mix_rhos_commits(),
            prep.combine_b_pows(),
            &prep.vk,
            state,
            vec![batch],
        )
        .expect("step");
        state = next_state;
        steps.push(step_proof);
        public_batches.push(public_batch);
    }
    let in_flight = neo_fold_clean::UncompressedAudit {
        proof: neo_fold_clean::Uncompressed {
            state,
            final_fold: None,
        },
        steps,
        public_batches,
    };
    let finished = neo_fold_clean::finish_uncompressed_with_audit(&prep, in_flight).expect("finish");
    (prep, finished)
}

fn polynomial_eval(coefficients: &[K], point: K) -> K {
    coefficients
        .iter()
        .rev()
        .fold(K::ZERO, |suffix, coefficient| *coefficient + point * suffix)
}

struct Artifact {
    source: String,
    rows: usize,
    instructions: Vec<checked_program_artifact_support::Instruction>,
}

fn build_artifact() -> Artifact {
    // Production has d_sc=4, hence five K coefficients per round.
    let coefficients = [kval(2, 3), kval(5, 7), kval(11, 13), kval(17, 19), kval(23, 29)];
    let challenge = kval(31, 37);
    let claim_in = coefficients
        .iter()
        .copied()
        .fold(coefficients[0], |sum, coefficient| sum + coefficient);

    let mut builder = R1csBuilder::new();
    let coefficient_vars: Vec<_> = coefficients
        .iter()
        .copied()
        .map(|value| alloc_k(&mut builder, value))
        .collect();
    let challenge_var = alloc_k(&mut builder, challenge);
    let claim_in_var = alloc_k(&mut builder, claim_in);
    let row_start = builder.rows();
    let first_allocated_column = builder.cols();
    let claim_out_var = enforce_sumcheck_round(&mut builder, &coefficient_vars, challenge_var, claim_in_var);
    let row_end = builder.rows();

    assert!(builder.is_satisfied(), "isolated SumCheck round");
    assert_eq!(
        [
            builder.witness()[claim_out_var.c0.col()],
            builder.witness()[claim_out_var.c1.col()]
        ],
        polynomial_eval(&coefficients, challenge).as_coeffs(),
        "round output must be the claimed polynomial at the challenge"
    );
    let audits = builder.sumcheck_round_audits();
    assert_eq!(audits.len(), 1, "one isolated round audit");
    let audit = &audits[0];
    assert_eq!(
        (audit.row_start, audit.row_end, audit.first_allocated_column),
        (row_start, row_end, first_allocated_column)
    );

    let normalized = normalize_range(&builder, row_start, row_end, first_allocated_column);
    let canonical = canonicalize_program(&normalized);
    let global_to_local: HashMap<_, _> = canonical
        .column_map
        .iter()
        .copied()
        .enumerate()
        .map(|(local, global)| (global, local))
        .collect();
    let local = |column: usize| {
        *global_to_local
            .get(&column)
            .expect("canonical local column")
    };
    let local_pair = |pair: [usize; 2]| [local(pair[0]), local(pair[1])];
    let local_coefficients: Vec<_> = audit
        .coefficient_cols
        .iter()
        .copied()
        .map(local_pair)
        .collect();
    let local_inputs: Vec<_> = normalized
        .input_columns
        .iter()
        .copied()
        .map(local)
        .collect();
    let local_witness: Vec<u64> = canonical
        .column_map
        .iter()
        .map(|&global| builder.witness()[global].as_canonical_u64())
        .collect();

    let pairs = |values: &[[usize; 2]]| {
        format!(
            "[{}]",
            values
                .iter()
                .map(|pair| format!("({}, {})", pair[0], pair[1]))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let naturals = |values: &[usize]| {
        format!(
            "[{}]",
            values
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let witness = local_witness
        .iter()
        .enumerate()
        .map(|(column, value)| format!("  | {column} => {value}"))
        .collect::<Vec<_>>()
        .join("\n");

    let mut source = String::new();
    writeln!(
        source,
        "import Nightstream.Implementation.R1CS.Core.CheckedProgram\n\n\
         /-! Exact isolated production `enforce_sumcheck_round` artifact. -/\n\n\
         namespace Nightstream.Implementation.R1CS.SumcheckRoundArtifact\n\n\
         open Nightstream.Implementation.R1CS\n\
         open Nightstream.Implementation.R1CS.CheckedProgram\n\n\
         def degree : Nat := 4\n\
         def coefficientColumns : List (Nat × Nat) := {}\n\
         def challengeColumns : Nat × Nat := ({}, {})\n\
         def claimInColumns : Nat × Nat := ({}, {})\n\
         def claimOutColumns : Nat × Nat := ({}, {})\n\
         def inputColumns : List Nat := {}\n\
         def instructions : List Instruction :=\n   [{}]\n\
         def rows : List Row := CheckedProgram.rows instructions\n\
         def honestAssignment : Nat → Nat\n{}\n  | _ => 0\n\n\
         theorem coefficient_count : coefficientColumns.length = degree + 1 := by native_decide\n\
         theorem input_has_one : 0 ∈ inputColumns := by native_decide\n\
         theorem definitions_wellFormed : Program.WellFormed inputColumns\n    (CheckedProgram.definitions instructions) := by native_decide\n\
         theorem definitions_canonical : ∀ definition ∈ CheckedProgram.definitions instructions,\n    definition.Canonical := by native_decide\n\
         theorem checks_reference : CheckedProgram.ChecksReference\n    (Program.knownAfter inputColumns (CheckedProgram.definitions instructions))\n    instructions := by native_decide\n\n\
         end Nightstream.Implementation.R1CS.SumcheckRoundArtifact",
        pairs(&local_coefficients),
        local_pair(audit.challenge_cols)[0],
        local_pair(audit.challenge_cols)[1],
        local_pair(audit.claim_in_cols)[0],
        local_pair(audit.claim_in_cols)[1],
        local_pair(audit.claim_out_cols)[0],
        local_pair(audit.claim_out_cols)[1],
        naturals(&local_inputs),
        lean_instructions(&canonical.instructions),
        witness,
    )
    .expect("render Lean artifact");
    Artifact {
        source,
        rows: row_end - row_start,
        instructions: canonical.instructions,
    }
}

#[test]
fn isolated_sumcheck_round_artifact_matches() {
    let artifact = build_artifact();
    assert_eq!(artifact.rows, 30, "degree-four round row count drift");
    let path = repo_root().join(ARTIFACT_PATH);
    if std::env::var_os("UPDATE_NIFS_COMPILER_ARTIFACT").is_some() {
        fs::write(&path, &artifact.source).expect("write SumCheck round artifact");
    }
    let existing = fs::read_to_string(&path).unwrap_or_default();
    assert_eq!(
        existing, artifact.source,
        "{ARTIFACT_PATH} drift; regenerate with UPDATE_NIFS_COMPILER_ARTIFACT=1"
    );
}

fn maps_in_range(builder: &R1csBuilder, row_start: usize, row_end: usize, anchor: &Artifact) -> Vec<Vec<usize>> {
    let mut audits: Vec<_> = builder
        .sumcheck_round_audits()
        .iter()
        .filter(|audit| row_start <= audit.row_start && audit.row_end <= row_end)
        .collect();
    audits.sort_by_key(|audit| audit.row_start);
    let maps: Vec<Vec<usize>> = audits
        .into_iter()
        .map(|audit| {
            assert_eq!(audit.row_end - audit.row_start, anchor.rows, "SumCheck round row count");
            assert_eq!(
                audit.coefficient_cols.len(),
                5,
                "production degree-four coefficient count"
            );
            let normalized = normalize_range(builder, audit.row_start, audit.row_end, audit.first_allocated_column);
            let canonical = canonicalize_program(&normalized);
            assert_eq!(
                canonical.instructions, anchor.instructions,
                "production SumCheck round differs from isolated exact compiler"
            );
            let map = canonical.column_map;
            let pair = |low: usize, high: usize| [map[low], map[high]];
            assert_eq!(
                audit.coefficient_cols,
                vec![pair(2, 8), pair(3, 9), pair(4, 10), pair(5, 11), pair(6, 12)],
                "coefficient decoder map"
            );
            assert_eq!(audit.challenge_cols, pair(13, 15), "challenge decoder map");
            assert_eq!(audit.claim_in_cols, pair(1, 7), "claim-in decoder map");
            assert_eq!(audit.claim_out_cols, pair(41, 42), "claim-out decoder map");
            map
        })
        .collect();
    for pair in maps.windows(2) {
        assert_eq!(
            [pair[0][41], pair[0][42]],
            [pair[1][1], pair[1][7]],
            "SumCheck running claim must be reused wire-for-wire"
        );
    }
    maps
}

fn lean_maps(maps: &[Vec<usize>]) -> String {
    maps.iter()
        .map(|map| {
            format!(
                "  [{}]",
                map.iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
        .collect::<Vec<_>>()
        .join("\n, ")
}

fn full_history_call_artifact(anchor: &Artifact) -> String {
    let (prep, finished) = full_history_fixture();
    let statement = neo_fold_clean::build_decider_statement(&prep, &finished);
    let synthesis = synthesize_statement_r1cs(&prep, &statement).expect("full-history synthesis");
    assert!(
        synthesis.builder.is_satisfied(),
        "full-history SumCheck conformance fixture: {:?}",
        synthesis.builder.first_unsatisfied_row()
    );
    let builder = &synthesis.builder;
    let ranges = |name: &str| {
        let mut matches: Vec<_> = builder
            .row_family_ranges()
            .iter()
            .filter(|range| range.name == name)
            .collect();
        matches.sort_by_key(|range| range.row_start);
        matches
    };
    let fe = ranges("nifs.pi_ccs.fe_sumcheck");
    let nc = ranges("nifs.pi_ccs.nc_sumcheck");
    assert_eq!(fe.len(), 2, "recursive and terminal FE owners");
    assert_eq!(nc.len(), 2, "recursive and terminal NC owners");

    let recursive_fe = maps_in_range(builder, fe[0].row_start, fe[0].row_end, anchor);
    let terminal_fe = maps_in_range(builder, fe[1].row_start, fe[1].row_end, anchor);
    let recursive_nc = maps_in_range(builder, nc[0].row_start, nc[0].row_end, anchor);
    let terminal_nc = maps_in_range(builder, nc[1].row_start, nc[1].row_end, anchor);
    let audited = recursive_fe.len() + terminal_fe.len() + recursive_nc.len() + terminal_nc.len();
    assert_eq!(
        audited,
        builder.sumcheck_round_audits().len(),
        "every emitted SumCheck round belongs to one exported FE/NC owner"
    );

    format!(
        "import Nightstream.Implementation.R1CS.Correspondence.Sumcheck.SumcheckChainSound\n\n\
         /-!\n\
         Generated exact affine maps for every full-history Π_CCS FE/NC round.\n\n\
         Owns: call-site relabeling of the isolated SumCheck-round compiler.\n\
         Does not own: transcript challenges, initial claims, terminal identities,\n\
         or soundness of Π_CCS outside the isolated round equations.\n\
         Emits constraints: no; the four row lists reconstruct already emitted rows.\n\
         Authority boundary: every map is extracted from one exact production row\n\
         range and checked against the isolated canonical compiler artifact.\n\n\
         | Branch | Mathematical obligation | Emitted by |\n\
         |---|---|---|\n\
         | recursive FE | Seven linked degree-four SumCheck rounds | nifs.pi_ccs.fe_sumcheck |\n\
         | recursive NC | Fifteen linked degree-four SumCheck rounds | nifs.pi_ccs.nc_sumcheck |\n\
         | terminal FE | Seven linked degree-four SumCheck rounds | nifs.pi_ccs.fe_sumcheck |\n\
         | terminal NC | Fifteen linked degree-four SumCheck rounds | nifs.pi_ccs.nc_sumcheck |\n\
         -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySumcheckArtifact\n\n\
         open Nightstream.Implementation.R1CS.SumcheckChainSound\n\n\
         set_option maxRecDepth 1048576\n\n\
         def recursiveFeMaps : List ColumnMap :=\n[{}]\n\
         def recursiveNcMaps : List ColumnMap :=\n[{}]\n\
         def terminalFeMaps : List ColumnMap :=\n[{}]\n\
         def terminalNcMaps : List ColumnMap :=\n[{}]\n\n\
         def recursiveFeRows : List Nightstream.Implementation.R1CS.Row :=\n  recursiveFeMaps.flatMap Rows\n\
         def recursiveNcRows : List Nightstream.Implementation.R1CS.Row :=\n  recursiveNcMaps.flatMap Rows\n\
         def terminalFeRows : List Nightstream.Implementation.R1CS.Row :=\n  terminalFeMaps.flatMap Rows\n\
         def terminalNcRows : List Nightstream.Implementation.R1CS.Row :=\n  terminalNcMaps.flatMap Rows\n\n\
         theorem recursive_fe_shape : recursiveFeMaps.length = {} ∧\n    MapsOne recursiveFeMaps ∧ Linked recursiveFeMaps := by native_decide\n\
         theorem recursive_nc_shape : recursiveNcMaps.length = {} ∧\n    MapsOne recursiveNcMaps ∧ Linked recursiveNcMaps := by native_decide\n\
         theorem terminal_fe_shape : terminalFeMaps.length = {} ∧\n    MapsOne terminalFeMaps ∧ Linked terminalFeMaps := by native_decide\n\
         theorem terminal_nc_shape : terminalNcMaps.length = {} ∧\n    MapsOne terminalNcMaps ∧ Linked terminalNcMaps := by native_decide\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistorySumcheckArtifact\n",
        lean_maps(&recursive_fe),
        lean_maps(&recursive_nc),
        lean_maps(&terminal_fe),
        lean_maps(&terminal_nc),
        recursive_fe.len(),
        recursive_nc.len(),
        terminal_fe.len(),
        terminal_nc.len(),
    )
}

#[test]
fn full_history_sumcheck_call_sites_match_isolated_compiler() {
    let anchor = build_artifact();
    let source = full_history_call_artifact(&anchor);
    let path = repo_root().join(CALLS_PATH);
    let existing = fs::read_to_string(&path).unwrap_or_default();
    if existing != source {
        let expected = path.with_extension("lean.expected");
        fs::write(&expected, source).expect("write full-history SumCheck expected artifact");
        panic!("{CALLS_PATH} drifted: {}", expected.display());
    }
}
