//! Exact Rust-to-Lean row gate for one phased PiCCS SumCheck round.
//!
//! Owns the production degree-nine source layout, one honest assignment, the
//! real compact Rust phase emitter, an independent row recipe, exhaustive
//! source-matrix comparison, and deterministic Lean artifact output.
//!
//! Does not own Poseidon2 replay rows, recursive orchestration, the start or
//! finish phase, or the complete recursive and terminal F-prime relations.
//!
//! Emits constraints: 31 direct selective-CCS product rows over 54 columns.
//! Only the general-selector, A, B, and C ports are nonzero.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use neo_fold_clean::engine::r1cs_circuit::{enforce_sumcheck_round_phase, KVar, R1csBuilder, R1csSnapshot};
use neo_fold_clean::frontends::r1cs_f_prime::lean_manifest::GOLDILOCKS_MODULUS;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryPiCcsRoundSelectiveCcs.lean";

const SCHEMA_VERSION: u64 = 1;
const DEGREE: usize = 9;
const COEFFICIENT_COUNT: usize = DEGREE + 1;
const CURRENT_START: usize = 1;
const COEFFICIENT_START: usize = 3;
const CHALLENGE_START: usize = 23;
const NEXT_START: usize = 25;
const AUXILIARY_START: usize = 27;
const ROWS: usize = 31;
const COLUMNS: usize = 54;
const ROW_VARIABLES: usize = 5;
const PORT_COUNT: usize = 13;
const GENERAL_SELECTOR_PORT: usize = 1;
const A_PORT: usize = 2;
const B_PORT: usize = 3;
const C_PORT: usize = 4;

type Pair = [F; 2];
type Term = (usize, F);
type SourceRow = [Vec<Term>; 3];

fn coefficient_start(index: usize) -> usize {
    COEFFICIENT_START + 2 * index
}

fn frame_start(step: usize) -> usize {
    AUXILIARY_START + 3 * step
}

fn k_add(left: Pair, right: Pair) -> Pair {
    [left[0] + right[0], left[1] + right[1]]
}

fn k_mul(left: Pair, right: Pair) -> Pair {
    [
        left[0] * right[0] + F::from_u64(7) * left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    ]
}

fn evaluate(coefficients: &[Pair], point: Pair) -> Pair {
    coefficients
        .iter()
        .rev()
        .fold([F::ZERO; 2], |suffix, &coefficient| {
            k_add(coefficient, k_mul(point, suffix))
        })
}

fn round_initial(coefficients: &[Pair]) -> Pair {
    let sum = coefficients.iter().copied().fold([F::ZERO; 2], k_add);
    k_add(coefficients[0], sum)
}

fn alloc_pair(builder: &mut R1csBuilder, value: Pair) -> KVar {
    KVar::alloc(builder, value[0], value[1])
}

fn honest_source() -> (R1csBuilder, Vec<KVar>, KVar, KVar, KVar) {
    let coefficients = (0..COEFFICIENT_COUNT)
        .map(|index| [F::from_usize(3 + 5 * index), F::from_usize(7 + 11 * index)])
        .collect::<Vec<_>>();
    let challenge = [F::from_u64(19), F::from_u64(23)];
    let current = round_initial(&coefficients);
    let next = evaluate(&coefficients, challenge);

    let mut builder = R1csBuilder::new();
    let current_var = alloc_pair(&mut builder, current);
    let coefficient_vars = coefficients
        .iter()
        .copied()
        .map(|value| alloc_pair(&mut builder, value))
        .collect::<Vec<_>>();
    let challenge_var = alloc_pair(&mut builder, challenge);
    let next_var = alloc_pair(&mut builder, next);

    assert_eq!(
        [current_var.c0.col(), current_var.c1.col()],
        [CURRENT_START, CURRENT_START + 1]
    );
    for (index, coefficient) in coefficient_vars.iter().enumerate() {
        assert_eq!(
            [coefficient.c0.col(), coefficient.c1.col()],
            [coefficient_start(index), coefficient_start(index) + 1],
        );
    }
    assert_eq!(
        [challenge_var.c0.col(), challenge_var.c1.col()],
        [CHALLENGE_START, CHALLENGE_START + 1],
    );
    assert_eq!([next_var.c0.col(), next_var.c1.col()], [NEXT_START, NEXT_START + 1]);
    assert_eq!(builder.cols(), AUXILIARY_START);

    enforce_sumcheck_round_phase(&mut builder, &coefficient_vars, challenge_var, current_var, next_var);
    (builder, coefficient_vars, challenge_var, current_var, next_var)
}

fn singleton(column: usize) -> Vec<Term> {
    vec![(column, F::ONE)]
}

fn carried_output(index: usize) -> [Vec<Term>; 2] {
    if index == DEGREE {
        return [
            singleton(coefficient_start(index)),
            singleton(coefficient_start(index) + 1),
        ];
    }
    let frame = frame_start(index);
    [
        vec![
            (coefficient_start(index), F::ONE),
            (frame, F::ONE),
            (frame + 1, F::from_u64(7)),
        ],
        vec![
            (coefficient_start(index) + 1, F::ONE),
            (frame + 2, F::ONE),
            (frame, -F::ONE),
            (frame + 1, -F::ONE),
        ],
    ]
}

fn expected_source_row(row: usize) -> SourceRow {
    if row < 2 {
        let limb = row;
        let mut initial = vec![(coefficient_start(0) + limb, F::from_u64(2))];
        initial.extend((1..COEFFICIENT_COUNT).map(|index| (coefficient_start(index) + limb, F::ONE)));
        return [singleton(CURRENT_START + limb), singleton(0), initial];
    }

    if row < 2 + 3 * DEGREE {
        let within = row - 2;
        let step = within / 3;
        let kind = within % 3;
        let suffix = carried_output(step + 1);
        let frame = frame_start(step);
        return match kind {
            0 => [singleton(CHALLENGE_START), suffix[0].clone(), singleton(frame)],
            1 => [singleton(CHALLENGE_START + 1), suffix[1].clone(), singleton(frame + 1)],
            2 => {
                let mut challenge_sum = singleton(CHALLENGE_START);
                challenge_sum.extend(singleton(CHALLENGE_START + 1));
                let mut suffix_sum = suffix[0].clone();
                suffix_sum.extend(suffix[1].clone());
                [challenge_sum, suffix_sum, singleton(frame + 2)]
            }
            _ => unreachable!(),
        };
    }

    let limb = row - (2 + 3 * DEGREE);
    [
        carried_output(0)[limb].clone(),
        singleton(0),
        singleton(NEXT_START + limb),
    ]
}

fn canonical_terms(terms: Vec<Term>) -> Vec<Term> {
    let mut combined = BTreeMap::new();
    for (column, coefficient) in terms {
        *combined.entry(column).or_insert(F::ZERO) += coefficient;
    }
    combined.retain(|_, coefficient| *coefficient != F::ZERO);
    combined.into_iter().collect()
}

fn assert_snapshot_matches_recipe(snapshot: &R1csSnapshot) {
    assert_eq!(snapshot.rows(), ROWS);
    assert_eq!(snapshot.cols(), COLUMNS);
    for row in 0..ROWS {
        let expected = expected_source_row(row).map(canonical_terms);
        assert_eq!(snapshot.a_row(row), expected[0], "A row {row}");
        assert_eq!(snapshot.b_row(row), expected[1], "B row {row}");
        assert_eq!(snapshot.c_row(row), expected[2], "C row {row}");
    }
}

fn render_artifact() -> String {
    let mut rendered = String::new();
    rendered.push_str(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.PiCcsRoundSelectiveCcsSchema\n\n\
/-! Generated file: exact compact recipe for one production PiCCS round.\n\n\
Owns the Rust-declared degree-nine dimensions, canonical column starts,\n\
selective port indices, and Goldilocks coefficients checked against the real\n\
compact phase emitter.\n\n\
Does not own semantic truth, Poseidon2 replay, recursive orchestration, or the\n\
complete recursive and terminal F-prime relations. Lean recomputes every row.\n\n\
Emits constraints: no. This file contains recipe data, not a trusted digest.\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiCcsRoundSelectiveCcs\n\n\
open Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsRoundSelectiveCcs.Artifact\n\n\
def rawArtifact : RawArtifact where\n",
    );
    writeln!(rendered, "  schemaVersion := {SCHEMA_VERSION}").unwrap();
    writeln!(rendered, "  degree := {DEGREE}").unwrap();
    writeln!(rendered, "  coefficientCount := {COEFFICIENT_COUNT}").unwrap();
    writeln!(rendered, "  currentStart := {CURRENT_START}").unwrap();
    writeln!(rendered, "  coefficientStart := {COEFFICIENT_START}").unwrap();
    writeln!(rendered, "  challengeStart := {CHALLENGE_START}").unwrap();
    writeln!(rendered, "  nextStart := {NEXT_START}").unwrap();
    writeln!(rendered, "  auxiliaryStart := {AUXILIARY_START}").unwrap();
    writeln!(rendered, "  rows := {ROWS}").unwrap();
    writeln!(rendered, "  columns := {COLUMNS}").unwrap();
    writeln!(rendered, "  rowVariables := {ROW_VARIABLES}").unwrap();
    writeln!(rendered, "  portCount := {PORT_COUNT}").unwrap();
    writeln!(rendered, "  generalSelectorPort := {GENERAL_SELECTOR_PORT}").unwrap();
    writeln!(rendered, "  aPort := {A_PORT}").unwrap();
    writeln!(rendered, "  bPort := {B_PORT}").unwrap();
    writeln!(rendered, "  cPort := {C_PORT}").unwrap();
    writeln!(rendered, "  nonresidue := 7").unwrap();
    writeln!(rendered, "  minusOne := {}", GOLDILOCKS_MODULUS - 1).unwrap();
    rendered.push_str(
        "\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiCcsRoundSelectiveCcs\n",
    );
    rendered
}

fn artifact_path() -> String {
    format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_PATH)
}

#[test]
fn pi_ccs_round_emitter_matches_generated_selective_recipe() {
    assert_eq!(ROWS, 31);
    assert_eq!(COLUMNS, 54);
    assert!(ROWS <= 1 << ROW_VARIABLES);
    assert!((1 << (ROW_VARIABLES - 1)) < ROWS);

    let (builder, coefficients, challenge, current, next) = honest_source();
    assert_eq!(builder.rows(), ROWS);
    assert_eq!(builder.cols(), COLUMNS);
    assert!(builder.is_satisfied(), "honest compact PiCCS round must satisfy");
    assert!(builder.unconstrained_columns().is_empty());
    let audit = builder
        .sumcheck_round_audits()
        .first()
        .expect("one round audit");
    assert_eq!(audit.row_start, 0);
    assert_eq!(audit.row_end, ROWS);
    assert_eq!(audit.first_allocated_column, AUXILIARY_START);
    assert_eq!(audit.allocated_cols, (AUXILIARY_START..COLUMNS).collect::<Vec<_>>());
    assert_eq!(audit.coefficient_cols.len(), COEFFICIENT_COUNT);
    assert_eq!(audit.challenge_cols, [challenge.c0.col(), challenge.c1.col()]);
    assert_eq!(audit.claim_in_cols, [current.c0.col(), current.c1.col()]);
    assert_eq!(audit.claim_out_cols, [next.c0.col(), next.c1.col()]);
    assert_eq!(coefficients.len(), COEFFICIENT_COUNT);

    let snapshot = builder.snapshot();
    assert_snapshot_matches_recipe(&snapshot);
    assert!(snapshot.is_satisfied(snapshot.witness()));

    let path = artifact_path();
    let rendered = render_artifact();
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if rendered != committed {
        let expected = format!("{path}.expected");
        std::fs::write(&expected, rendered).expect("write PiCCS round artifact candidate");
        panic!("PiCCS round selective-CCS artifact drifted; inspect {expected} and promote it explicitly");
    }
}

#[test]
#[ignore = "deliberately writes the reviewed generated Lean artifact"]
fn regenerate_pi_ccs_round_selective_ccs_artifact() {
    std::fs::write(artifact_path(), render_artifact()).expect("write generated PiCCS round artifact");
}
