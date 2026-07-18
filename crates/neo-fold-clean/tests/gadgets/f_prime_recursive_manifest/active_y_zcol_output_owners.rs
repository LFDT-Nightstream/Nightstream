//! Three-matrix diagnostic owners for the two parent `y_zcol` output evaluations.
//!
//! Owns: selection of the two `YZColLimb` identities from the validated
//! recursive trace, exact binding of their 54 coefficient columns to the
//! returned F′ parent wires, and rendering of the active Lean artifact.
//!
//! Does not own: semantic parent-opening authority, transcript timing, the
//! ten padded-lane zero/canonicalization checks, bad-root probability,
//! compact lowering, cost estimates, or row removal.
//!
//! Emits constraints: no.
//!
//! Authority boundary: source R1CS rows and returned parent wires are checked
//! directly. Role labels are only selectors and cannot establish the parent
//! binding. The generated data is artifact evidence, never semantic authority.
//!
//! | Stage path | Equation/ownership | Multiplicity | Evidence |
//! |---|---|---:|---|
//! | `nifs.pi_rlc.verify.identities.y_zcol.evaluations.output.limb0` | `E0 = sum_i parent_y_zcol[i].c0 * beta^i` for `i < 54` | 108 source-R1CS rows | exact trace replay plus parent-column equality |
//! | `nifs.pi_rlc.verify.identities.y_zcol.evaluations.output.limb1` | `E1 = sum_i parent_y_zcol[i].c1 * beta^i` for `i < 54` | 108 source-R1CS rows | exact trace replay plus parent-column equality |
//! | `nifs.pi_rlc.verify.identities.y_zcol.evaluations.output` | both leaves share the same 54-entry beta ladder | one shared dependency | exact column equality |

use std::fmt::Write as _;
use std::fs;

use neo_fold_clean::engine::r1cs_circuit::projection_identity_trace::validate_projection_identity_traces;
use neo_fold_clean::engine::r1cs_circuit::{ProjectionIdentityRole, R1csEncodingTrace, R1csSnapshot};
use neo_fold_clean::paper::f_prime::r1cs::FPrimeStepOutput;
use neo_fold_clean::paper::reductions::pi_rlc_circuit::stage as pi_rlc_stage;
use neo_math::ring::D;
use neo_math::F;
use p3_field::PrimeField64;

use super::repo_root;

const LEAN_DATA_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeRecursive/Generated/FPrimeRecursiveYZcolProjectionData.lean";

#[derive(Debug, PartialEq, Eq)]
struct OutputEvaluationOwner {
    stage_path: &'static str,
    identity_index: usize,
    limb: usize,
    identity_row_start: usize,
    identity_row_end: usize,
    evaluation_row_start: usize,
    evaluation_row_end: usize,
    evaluation_allocated_start: usize,
    evaluation_allocated_end: usize,
    parent_coefficient_columns: Vec<usize>,
    power_columns: Vec<[usize; 2]>,
    evaluation_output_columns: [usize; 2],
}

fn exact_stage_interval(
    trace: &R1csEncodingTrace,
    stage_path: &'static str,
    expected_rows: std::ops::Range<usize>,
    expected_columns: std::ops::Range<usize>,
) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
    let matching = trace
        .stages()
        .windows(2)
        .filter(|pair| {
            pair[0].label == stage_path
                && pair[0].row == expected_rows.start
                && pair[1].row == expected_rows.end
                && pair[0].col == expected_columns.start
                && pair[1].col == expected_columns.end
        })
        .collect::<Vec<_>>();
    let [pair] = matching.as_slice() else {
        panic!(
            "expected exactly one `{stage_path}` checkpoint interval with rows {expected_rows:?} and columns {expected_columns:?}, found {}",
            matching.len()
        );
    };
    assert!(pair[0].row <= pair[1].row, "stage rows must be monotone");
    assert!(pair[0].col <= pair[1].col, "stage columns must be monotone");
    (pair[0].row..pair[1].row, pair[0].col..pair[1].col)
}

fn parent_limb_columns(output: &FPrimeStepOutput, limb: usize) -> Vec<usize> {
    assert!(limb < 2, "y_zcol extension limb must be 0 or 1");
    let parent = output
        .nifs_parent
        .as_ref()
        .expect("recursive F' exposes its NIFS parent");
    assert!(parent.y_zcol_lanes >= D, "parent y_zcol must contain all active lanes");
    assert_eq!(
        parent.y_zcol.len(),
        2 * parent.y_zcol_lanes,
        "parent y_zcol must use interleaved two-limb K encoding"
    );
    (0..D)
        .map(|lane| parent.y_zcol[2 * lane + limb].col())
        .collect()
}

fn output_evaluation_owners(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    output: &FPrimeStepOutput,
) -> Vec<OutputEvaluationOwner> {
    validate_projection_identity_traces(source, trace).expect("exact production projection trace");

    let selected = trace
        .projection_identities()
        .iter()
        .enumerate()
        .filter_map(|(identity_index, identity)| match identity.role {
            ProjectionIdentityRole::YZColLimb { limb } => Some((identity_index, limb, identity)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        selected
            .iter()
            .map(|(_, limb, _)| *limb)
            .collect::<Vec<_>>(),
        [0, 1],
        "the diagnostic profile must contain exactly the ordered c0/c1 y_zcol identities"
    );

    let mut owners = Vec::with_capacity(2);
    for (identity_index, limb, identity) in selected {
        let stage_path = match limb {
            0 => pi_rlc_stage::IDENTITIES_Y_ZCOL_EVALUATIONS_OUTPUT_LIMB0,
            1 => pi_rlc_stage::IDENTITIES_Y_ZCOL_EVALUATIONS_OUTPUT_LIMB1,
            _ => unreachable!("validated y_zcol limb"),
        };
        let evaluation = &trace.polynomial_evaluations()[identity.output_evaluation];
        let expected_parent_columns = parent_limb_columns(output, limb);
        assert_eq!(
            identity.output_columns, expected_parent_columns,
            "YZColLimb({limb}) identity must evaluate the returned NIFS parent y_zcol limb"
        );
        assert_eq!(
            evaluation.coefficient_cols, expected_parent_columns,
            "YZColLimb({limb}) output evaluation must retain the parent coefficient order"
        );
        assert_eq!(evaluation.coefficient_cols.len(), D, "active y_zcol width");
        assert_eq!(evaluation.power_cols.len(), D, "active beta-power width");
        assert!(
            identity.source_rows.start <= evaluation.row_start && evaluation.row_end <= identity.source_rows.end,
            "output evaluation must be contained by its identity owner"
        );
        assert_eq!(
            evaluation.allocated_columns,
            (evaluation.allocated_columns[0]..evaluation.allocated_columns[0] + evaluation.allocated_columns.len())
                .collect::<Vec<_>>(),
            "output-evaluation columns must be one exact contiguous SSA interval"
        );
        let evaluation_allocated_start = evaluation.allocated_columns[0];
        let evaluation_allocated_end = evaluation_allocated_start + evaluation.allocated_columns.len();
        assert!(
            evaluation
                .coefficient_cols
                .iter()
                .all(|&column| column < evaluation_allocated_start),
            "parent coefficients must precede the evaluation SSA interval"
        );
        assert!(
            evaluation
                .power_cols
                .iter()
                .flatten()
                .all(|&column| column < evaluation_allocated_start),
            "shared power inputs must precede the evaluation SSA interval"
        );
        assert_eq!(
            evaluation.output_cols,
            [evaluation_allocated_end - 2, evaluation_allocated_end - 1],
            "output columns must terminate the evaluation interval"
        );
        let (stage_rows, stage_columns) = exact_stage_interval(
            trace,
            stage_path,
            evaluation.row_start..evaluation.row_end,
            evaluation_allocated_start..evaluation_allocated_end,
        );
        assert_eq!(
            stage_rows,
            evaluation.row_start..evaluation.row_end,
            "the physical stage must own exactly the output-evaluation rows"
        );
        assert_eq!(
            stage_columns,
            evaluation_allocated_start..evaluation_allocated_end,
            "the physical stage must own exactly the output-evaluation columns"
        );
        owners.push(OutputEvaluationOwner {
            stage_path,
            identity_index,
            limb,
            identity_row_start: identity.source_rows.start,
            identity_row_end: identity.source_rows.end,
            evaluation_row_start: evaluation.row_start,
            evaluation_row_end: evaluation.row_end,
            evaluation_allocated_start,
            evaluation_allocated_end,
            parent_coefficient_columns: evaluation.coefficient_cols.clone(),
            power_columns: evaluation.power_cols.clone(),
            evaluation_output_columns: evaluation.output_cols,
        });
    }
    assert_eq!(
        owners[0].power_columns, owners[1].power_columns,
        "both parent limbs must use the same beta ladder"
    );
    assert!(
        owners[0].evaluation_row_end <= owners[1].evaluation_row_start,
        "the two output-evaluation row owners must be disjoint and ordered"
    );
    assert!(
        owners[0].evaluation_allocated_end <= owners[1].evaluation_allocated_start,
        "the two output-evaluation SSA intervals must be disjoint and ordered"
    );
    owners
}

fn lean_nat_list(values: impl IntoIterator<Item = usize>) -> String {
    format!(
        "[{}]",
        values
            .into_iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_k_columns(columns: [usize; 2]) -> String {
    format!("{{ c0 := {}, c1 := {} }}", columns[0], columns[1])
}

fn lean_k_columns_list(values: impl IntoIterator<Item = [usize; 2]>) -> String {
    format!(
        "[{}]",
        values
            .into_iter()
            .map(lean_k_columns)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_terms(terms: &[(usize, F)]) -> String {
    format!(
        "[{}]",
        terms
            .iter()
            .map(|&(column, coefficient)| format!("({column}, {})", coefficient.as_canonical_u64()))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_source_rows(source: &R1csSnapshot, owners: &[OutputEvaluationOwner]) -> String {
    let rows = owners
        .iter()
        .flat_map(|owner| owner.evaluation_row_start..owner.evaluation_row_end)
        .map(|row| {
            format!(
                "({row}, ⟨{}, {}, {}⟩)",
                lean_terms(source.a_row(row)),
                lean_terms(source.b_row(row)),
                lean_terms(source.c_row(row))
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 2 * (2 * (D - 1) + 2), "two exact evaluator leaves");
    format!("[{}]", rows.join(",\n   "))
}

fn render(source: &R1csSnapshot, trace: &R1csEncodingTrace, output: &FPrimeStepOutput) -> String {
    let owners = output_evaluation_owners(source, trace, output);
    let parent = output.nifs_parent.as_ref().expect("recursive F' parent");
    let mut rendered = String::new();
    rendered.push_str("import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.YZcolProjectionSchema\n\n");
    rendered.push_str(
        "/-! Generated by `active_pi_rlc_projection_artifacts_match_production_trace`; do not hand-edit. -/\n\n",
    );
    rendered.push_str("namespace Nightstream.Implementation.R1CS.FPrimeRecursiveYZcolProjectionData\n\n");
    writeln!(
        rendered,
        "def stagePath : String := \"{}\"",
        pi_rlc_stage::IDENTITIES_Y_ZCOL_EVALUATIONS_OUTPUT
    )
    .expect("render stage path");
    writeln!(rendered, "def activeLaneCount : Nat := {D}").expect("render active lanes");
    writeln!(rendered, "def paddedLaneCount : Nat := {}", parent.y_zcol_lanes).expect("render padded lanes");
    writeln!(
        rendered,
        "def sharedPowerColumns : List ProjectionProgram.KColumns := {}\n",
        lean_k_columns_list(owners[0].power_columns.iter().copied())
    )
    .expect("render power columns");
    rendered.push_str("def owners : List YZcolOutputEvaluationOwner :=\n");
    for (index, owner) in owners.iter().enumerate() {
        let prefix = if index == 0 { "  [ " } else { "  , " };
        writeln!(rendered, "{prefix}{{ stagePath := \"{}\"", owner.stage_path).expect("render owner");
        writeln!(rendered, "      identityIndex := {}", owner.identity_index).expect("render owner");
        writeln!(rendered, "      limb := {}", owner.limb).expect("render owner");
        writeln!(rendered, "      identityRowStart := {}", owner.identity_row_start).expect("render owner");
        writeln!(rendered, "      identityRowEnd := {}", owner.identity_row_end).expect("render owner");
        writeln!(rendered, "      evaluationRowStart := {}", owner.evaluation_row_start).expect("render owner");
        writeln!(rendered, "      evaluationRowEnd := {}", owner.evaluation_row_end).expect("render owner");
        writeln!(
            rendered,
            "      evaluationAllocatedStart := {}",
            owner.evaluation_allocated_start
        )
        .expect("render owner");
        writeln!(
            rendered,
            "      evaluationAllocatedEnd := {}",
            owner.evaluation_allocated_end
        )
        .expect("render owner");
        writeln!(
            rendered,
            "      parentCoefficientColumns := {}",
            lean_nat_list(owner.parent_coefficient_columns.iter().copied())
        )
        .expect("render owner");
        rendered.push_str("      powerColumns := sharedPowerColumns\n");
        writeln!(
            rendered,
            "      evaluationOutputColumns := {} }}",
            lean_k_columns(owner.evaluation_output_columns)
        )
        .expect("render owner");
    }
    rendered.push_str("  ]\n\n");
    writeln!(
        rendered,
        "def sourceRows : List (Nat × Row) :=\n  {}\n",
        lean_source_rows(source, &owners)
    )
    .expect("render exact source rows");
    rendered.push_str("end Nightstream.Implementation.R1CS.FPrimeRecursiveYZcolProjectionData\n");
    rendered
}

pub(super) fn check_generated_artifact(source: &R1csSnapshot, trace: &R1csEncodingTrace, output: &FPrimeStepOutput) {
    let rendered = render(source, trace, output);
    let path = repo_root().join(LEAN_DATA_PATH);
    let committed = fs::read_to_string(&path).unwrap_or_default();
    if committed != rendered {
        let expected = path.with_extension("lean.expected");
        fs::create_dir_all(expected.parent().expect("generated artifact parent"))
            .expect("create generated artifact directory");
        fs::write(&expected, &rendered).expect("write expected active y_zcol artifact");
    }
    assert_eq!(
        committed, rendered,
        "active y_zcol output-owner artifact drifted; review the generated .expected file"
    );
}
