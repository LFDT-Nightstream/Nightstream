//! Three-matrix diagnostic owner for the shared PiRLC rho evaluations.
//!
//! Owns: unique structural selection of the 15 polynomial evaluations reused
//! by both returned-parent `YZColLimb` identities, their exact source row and
//! SSA intervals, and rendering of three bounded Lean artifact shards.
//!
//! Does not own: transcript derivation or semantic authority of rho, beta
//! transcript authority, projection-identity soundness, compact lowering,
//! encoded costs, or row removal.
//!
//! Emits constraints: no.
//!
//! Authority boundary: exact source rows and ordered wire schedules are
//! replayed from production. Stage and role labels are selectors only.
//!
//! | Stage path | Mathematical obligation | Multiplicity | Evidence |
//! |---|---|---:|---|
//! | `nifs.pi_rlc.verify.projection_shared.rho_evaluations` | `rho_i(beta) = sum_j rho_i[j] beta^j` | 15 x 108 source-R1CS rows | exact trace replay |
//! | returned-parent `YZColLimb` users | both identities consume the same ordered rho evaluations | two consumers | exact coefficient/output/power-column equality |

use std::fmt::Write as _;
use std::fs;
use std::ops::Range;

use neo_fold_clean::engine::r1cs_circuit::{
    PolynomialEvaluationTraceEntry, ProjectionIdentityRole, R1csEncodingTrace, R1csSnapshot,
};
use neo_fold_clean::paper::reductions::pi_rlc_circuit::stage as pi_rlc_stage;
use neo_math::ring::D;
use neo_math::F;
use p3_field::PrimeField64;

use super::repo_root;

const LEAN_DATA_PATHS: [&str; 3] = [
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeRecursive/PiRlcProjection/Generated/RhoEvaluationsData0.lean",
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeRecursive/PiRlcProjection/Generated/RhoEvaluationsData1.lean",
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeRecursive/PiRlcProjection/Generated/RhoEvaluationsData2.lean",
];
const EVALUATION_COUNT: usize = 15;
const EVALUATIONS_PER_SHARD: usize = 5;
const EVALUATION_ROW_COUNT: usize = 2 * (D - 1) + 2;
const SOURCE_ROW_COUNT: usize = EVALUATION_COUNT * EVALUATION_ROW_COUNT;

#[derive(Clone, Debug, PartialEq, Eq)]
struct RhoEvaluationOwner {
    pair_index: usize,
    trace_index: usize,
    row_start: usize,
    row_end: usize,
    allocated_start: usize,
    allocated_end: usize,
    coefficient_columns: Vec<usize>,
    power_columns: Vec<[usize; 2]>,
    output_columns: [usize; 2],
}

#[derive(Debug, PartialEq, Eq)]
struct RhoEvaluationsOwner {
    stage_row_start: usize,
    stage_row_end: usize,
    stage_allocated_start: usize,
    stage_allocated_end: usize,
    consumer_identity_indices: [usize; 2],
    evaluations: Vec<RhoEvaluationOwner>,
}

fn exact_stage_interval(
    trace: &R1csEncodingTrace,
    expected_rows: Range<usize>,
    expected_columns: Range<usize>,
) -> (Range<usize>, Range<usize>) {
    let matching = trace
        .stages()
        .windows(2)
        .filter(|pair| {
            pair[0].label == pi_rlc_stage::PROJECTION_SHARED_RHO_EVALUATIONS
                && pair[0].row == expected_rows.start
                && pair[1].row == expected_rows.end
                && pair[0].col == expected_columns.start
                && pair[1].col == expected_columns.end
        })
        .collect::<Vec<_>>();
    let [pair] = matching.as_slice() else {
        panic!(
            "expected exactly one `{}` checkpoint interval with rows {expected_rows:?} and columns {expected_columns:?}, found {}",
            pi_rlc_stage::PROJECTION_SHARED_RHO_EVALUATIONS,
            matching.len()
        );
    };
    (pair[0].row..pair[1].row, pair[0].col..pair[1].col)
}

fn matches_shared_rho_evaluation(
    evaluation: &PolynomialEvaluationTraceEntry,
    pair_index: usize,
    rho_columns: &[Vec<usize>],
    rho_outputs: &[[usize; 2]],
    shared_power_columns: &[[usize; 2]],
) -> bool {
    evaluation.coefficient_cols == rho_columns[pair_index]
        && evaluation.output_cols == rho_outputs[pair_index]
        && evaluation.power_cols == shared_power_columns[..D]
}

fn selected_owner(trace: &R1csEncodingTrace) -> RhoEvaluationsOwner {
    let y_zcol_identities = trace
        .projection_identities()
        .iter()
        .enumerate()
        .filter_map(|(identity_index, identity)| match identity.role {
            ProjectionIdentityRole::YZColLimb { limb } => Some((identity_index, limb, identity)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        y_zcol_identities
            .iter()
            .map(|(_, limb, _)| *limb)
            .collect::<Vec<_>>(),
        [0, 1],
        "the diagnostic profile must contain exactly the ordered c0/c1 y_zcol identities"
    );
    let (_, _, first_identity) = y_zcol_identities[0];
    let (_, _, second_identity) = y_zcol_identities[1];
    assert_eq!(
        first_identity.rho_columns, second_identity.rho_columns,
        "both returned-parent y_zcol identities must consume the same ordered rho coefficients"
    );
    assert_eq!(
        first_identity.rho_evaluation_outputs, second_identity.rho_evaluation_outputs,
        "both returned-parent y_zcol identities must consume the same ordered rho outputs"
    );
    assert_eq!(
        first_identity.power_columns, second_identity.power_columns,
        "both returned-parent y_zcol identities must consume the same beta ladder"
    );
    assert_eq!(first_identity.rho_columns.len(), EVALUATION_COUNT, "rho arity");
    assert_eq!(
        first_identity.rho_evaluation_outputs.len(),
        EVALUATION_COUNT,
        "rho output arity"
    );
    assert_eq!(first_identity.power_columns.len(), D + 1, "complete beta ladder width");

    let traces = trace.polynomial_evaluations();
    assert!(
        traces.len() >= EVALUATION_COUNT,
        "production trace must contain at least {EVALUATION_COUNT} polynomial evaluations"
    );
    let starts = (0..=traces.len() - EVALUATION_COUNT)
        .filter(|&start| {
            (0..EVALUATION_COUNT).all(|pair_index| {
                matches_shared_rho_evaluation(
                    &traces[start + pair_index],
                    pair_index,
                    &first_identity.rho_columns,
                    &first_identity.rho_evaluation_outputs,
                    &first_identity.power_columns,
                )
            })
        })
        .collect::<Vec<_>>();
    let [trace_start] = starts.as_slice() else {
        panic!(
            "expected exactly one ordered block of {EVALUATION_COUNT} rho evaluations shared by both y_zcol identities, found {}",
            starts.len()
        );
    };

    let mut evaluations: Vec<RhoEvaluationOwner> = Vec::with_capacity(EVALUATION_COUNT);
    for pair_index in 0..EVALUATION_COUNT {
        let trace_index = trace_start + pair_index;
        let evaluation = &traces[trace_index];
        assert_eq!(evaluation.coefficient_cols.len(), D, "rho coefficient width");
        assert_eq!(evaluation.power_cols.len(), D, "rho power width");
        assert_eq!(
            evaluation.row_end - evaluation.row_start,
            EVALUATION_ROW_COUNT,
            "exact rho evaluation row count"
        );
        assert_eq!(
            evaluation.allocated_columns.len(),
            EVALUATION_ROW_COUNT,
            "exact rho evaluation allocated-column count"
        );
        assert_eq!(
            evaluation.allocated_columns,
            (evaluation.allocated_columns[0]..evaluation.allocated_columns[0] + EVALUATION_ROW_COUNT)
                .collect::<Vec<_>>(),
            "rho evaluation must allocate one exact contiguous SSA interval"
        );
        let allocated_start = evaluation.allocated_columns[0];
        let allocated_end = allocated_start + evaluation.allocated_columns.len();
        assert!(
            evaluation
                .coefficient_cols
                .iter()
                .all(|&column| column < allocated_start),
            "rho coefficient inputs must predate their evaluation interval"
        );
        assert!(
            evaluation
                .power_cols
                .iter()
                .flatten()
                .all(|&column| column < allocated_start),
            "shared power inputs must predate their evaluation interval"
        );
        assert_eq!(
            evaluation.output_cols,
            [allocated_end - 2, allocated_end - 1],
            "rho output must terminate its evaluation interval"
        );
        if let Some(previous) = evaluations.last() {
            assert_eq!(previous.row_end, evaluation.row_start, "rho rows must be contiguous");
            assert_eq!(
                previous.allocated_end, allocated_start,
                "rho allocations must be contiguous"
            );
        }
        evaluations.push(RhoEvaluationOwner {
            pair_index,
            trace_index,
            row_start: evaluation.row_start,
            row_end: evaluation.row_end,
            allocated_start,
            allocated_end,
            coefficient_columns: evaluation.coefficient_cols.clone(),
            power_columns: evaluation.power_cols.clone(),
            output_columns: evaluation.output_cols,
        });
    }

    let stage_row_start = evaluations[0].row_start;
    let stage_row_end = evaluations[EVALUATION_COUNT - 1].row_end;
    let stage_allocated_start = evaluations[0].allocated_start;
    let stage_allocated_end = evaluations[EVALUATION_COUNT - 1].allocated_end;
    assert_eq!(
        stage_row_end - stage_row_start,
        SOURCE_ROW_COUNT,
        "shared rho row total"
    );
    assert_eq!(
        stage_allocated_end - stage_allocated_start,
        SOURCE_ROW_COUNT,
        "shared rho allocation total"
    );
    let (stage_rows, stage_columns) = exact_stage_interval(
        trace,
        stage_row_start..stage_row_end,
        stage_allocated_start..stage_allocated_end,
    );
    assert_eq!(stage_rows, stage_row_start..stage_row_end, "exact rho stage rows");
    assert_eq!(
        stage_columns,
        stage_allocated_start..stage_allocated_end,
        "exact rho stage columns"
    );

    RhoEvaluationsOwner {
        stage_row_start,
        stage_row_end,
        stage_allocated_start,
        stage_allocated_end,
        consumer_identity_indices: [y_zcol_identities[0].0, y_zcol_identities[1].0],
        evaluations,
    }
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

fn lean_owner(owner: &RhoEvaluationOwner) -> String {
    format!(
        "{{ stagePath := \"{}\", pairIndex := {}, traceIndex := {}, rowStart := {}, rowEnd := {}, allocatedStart := {}, allocatedEnd := {}, coefficientColumns := {}, powerColumns := {}, outputColumns := {} }}",
        pi_rlc_stage::PROJECTION_SHARED_RHO_EVALUATIONS,
        owner.pair_index,
        owner.trace_index,
        owner.row_start,
        owner.row_end,
        owner.allocated_start,
        owner.allocated_end,
        lean_nat_list(owner.coefficient_columns.iter().copied()),
        lean_k_columns_list(owner.power_columns.iter().copied()),
        lean_k_columns(owner.output_columns),
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

fn lean_source_rows(source: &R1csSnapshot, evaluations: &[RhoEvaluationOwner]) -> String {
    let rows = evaluations
        .iter()
        .flat_map(|owner| owner.row_start..owner.row_end)
        .map(|row| {
            format!(
                "({row}, ⟨{}, {}, {}⟩)",
                lean_terms(source.a_row(row)),
                lean_terms(source.b_row(row)),
                lean_terms(source.c_row(row))
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        rows.len(),
        evaluations.len() * EVALUATION_ROW_COUNT,
        "exact rho shard source rows"
    );
    format!("[{}]", rows.join(",\n   "))
}

fn render_shard(source: &R1csSnapshot, owner: &RhoEvaluationsOwner, shard: usize) -> String {
    let start = shard * EVALUATIONS_PER_SHARD;
    let end = start + EVALUATIONS_PER_SHARD;
    let evaluations = &owner.evaluations[start..end];
    let mut rendered = String::new();
    rendered.push_str(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.RhoEvaluationsSchema\n\n",
    );
    rendered.push_str(
        "/-! Generated by `active_pi_rlc_projection_artifacts_match_production_trace`; do not hand-edit. -/\n\n",
    );
    writeln!(
        rendered,
        "namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjectionRhoEvaluationsData.Shard{shard}\n"
    )
    .expect("render shard namespace");
    if shard == 0 {
        writeln!(
            rendered,
            "def stagePath : String := \"{}\"",
            pi_rlc_stage::PROJECTION_SHARED_RHO_EVALUATIONS
        )
        .expect("render stage path");
        writeln!(rendered, "def stageRowStart : Nat := {}", owner.stage_row_start).expect("render stage row start");
        writeln!(rendered, "def stageRowEnd : Nat := {}", owner.stage_row_end).expect("render stage row end");
        writeln!(
            rendered,
            "def stageAllocatedStart : Nat := {}",
            owner.stage_allocated_start
        )
        .expect("render stage allocated start");
        writeln!(rendered, "def stageAllocatedEnd : Nat := {}", owner.stage_allocated_end)
            .expect("render stage allocated end");
        writeln!(
            rendered,
            "def consumerIdentityIndices : List Nat := {}\n",
            lean_nat_list(owner.consumer_identity_indices)
        )
        .expect("render consumer identity indices");
    }
    rendered.push_str("set_option maxRecDepth 100000 in\n");
    rendered.push_str("def owners : List PiRlcRhoEvaluationOwner :=\n");
    for (index, evaluation) in evaluations.iter().enumerate() {
        let prefix = if index == 0 { "  [ " } else { "  , " };
        writeln!(rendered, "{prefix}{}", lean_owner(evaluation)).expect("render rho owner");
    }
    rendered.push_str("  ]\n\n");
    rendered.push_str("set_option maxRecDepth 100000 in\n");
    writeln!(
        rendered,
        "def sourceRows : List (Nat × Row) :=\n  {}\n",
        lean_source_rows(source, evaluations)
    )
    .expect("render exact source rows");
    writeln!(
        rendered,
        "end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjectionRhoEvaluationsData.Shard{shard}"
    )
    .expect("render shard end");
    rendered
}

pub(super) fn check_generated_artifact(source: &R1csSnapshot, trace: &R1csEncodingTrace) {
    let owner = selected_owner(trace);
    let mut drifted = Vec::new();
    for (shard, relative_path) in LEAN_DATA_PATHS.iter().enumerate() {
        let rendered = render_shard(source, &owner, shard);
        let path = repo_root().join(relative_path);
        let committed = fs::read_to_string(&path).unwrap_or_default();
        if committed != rendered {
            let expected = path.with_extension("lean.expected");
            fs::create_dir_all(expected.parent().expect("generated artifact parent"))
                .expect("create generated artifact directory");
            fs::write(&expected, &rendered).expect("write expected active rho-evaluation artifact");
            drifted.push(shard);
        }
    }
    assert!(
        drifted.is_empty(),
        "active rho-evaluation shards {drifted:?} drifted; review the generated .expected files"
    );
}
