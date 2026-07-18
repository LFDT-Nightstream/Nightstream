//! Three-matrix diagnostic owners for the two complete PiRLC `y_zcol` identities.
//!
//! Owns: structural selection of the ordered `YZColLimb {0,1}` identities,
//! exact trace/audit agreement, every identity-local source-row interval, and
//! rendering of bounded Lean shards for rows not already owned by the shared
//! beta/rho blocks or the parent-output evaluators.
//!
//! Does not own: transcript authority, semantic column authority, shared
//! beta/rho rows, the existing output-evaluation rows, padding, encoded
//! lowering, bad-root probability, or row removal.
//!
//! Emits constraints: no.
//!
//! Authority boundary: source rows are authoritative. Role and stage labels
//! only select candidate intervals; every selected interval is replayed from
//! the validated production trace and cross-checked against the audit record.
//!
//! | Child path per limb | Mathematical obligation | Exported source rows |
//! |---|---|---:|
//! | `evaluations.inputs` | evaluate 15 input polynomials at beta | 1,620 |
//! | `k_products.rho_times_input` | multiply each rho/input evaluation pair | 75 |
//! | `evaluations.output` | evaluate the returned parent limb | 0 new; reuse 108 |
//! | `evaluations.quotient` | evaluate the degree-52 quotient | 106 |
//! | `k_products.quotient_times_phi` | multiply `q(beta) * Phi81(beta)` | 5 |
//! | `final_limb_checks` | compare both K limbs | 2 |

use std::fmt::Write as _;
use std::fs;
use std::ops::Range;

use neo_fold_clean::engine::r1cs_circuit::builder::ProjectionIdentityAudit;
use neo_fold_clean::engine::r1cs_circuit::projection_identity_trace::validate_projection_identity_traces;
use neo_fold_clean::engine::r1cs_circuit::{
    KMulTraceEntry, PolynomialEvaluationTraceEntry, ProjectionIdentityRole, ProjectionIdentityTraceEntry,
    R1csEncodingTrace, R1csSnapshot,
};
use neo_fold_clean::paper::reductions::pi_rlc_circuit::stage as pi_rlc_stage;
use neo_math::ring::D;
use neo_math::F;
use p3_field::PrimeField64;

use super::repo_root;

const PAIR_COUNT: usize = 15;
const INPUTS_PER_SHARD: usize = 5;
const INPUT_EVALUATION_ROWS: usize = 2 * (D - 1) + 2;
const PAIR_PRODUCT_ROWS: usize = 5;
const OUTPUT_EVALUATION_ROWS: usize = INPUT_EVALUATION_ROWS;
const QUOTIENT_COEFFICIENTS: usize = D - 1;
const QUOTIENT_EVALUATION_ROWS: usize = 2 * (QUOTIENT_COEFFICIENTS - 1) + 2;
const QUOTIENT_PHI_ROWS: usize = 5;
const FINAL_CHECK_ROWS: usize = 2;
const NEW_LOCAL_ROWS: usize = PAIR_COUNT * (INPUT_EVALUATION_ROWS + PAIR_PRODUCT_ROWS)
    + QUOTIENT_EVALUATION_ROWS
    + QUOTIENT_PHI_ROWS
    + FINAL_CHECK_ROWS;
const COMPLETE_IDENTITY_ROWS: usize = NEW_LOCAL_ROWS + OUTPUT_EVALUATION_ROWS;
const COMPLETE_IDENTITY_COLUMNS: usize = COMPLETE_IDENTITY_ROWS - FINAL_CHECK_ROWS;

const INPUT_DATA_PATHS: [[&str; 3]; 2] = [
    [
        "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeRecursive/PiRlcProjection/Generated/YZcolIdentityLimb0Inputs0Data.lean",
        "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeRecursive/PiRlcProjection/Generated/YZcolIdentityLimb0Inputs1Data.lean",
        "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeRecursive/PiRlcProjection/Generated/YZcolIdentityLimb0Inputs2Data.lean",
    ],
    [
        "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeRecursive/PiRlcProjection/Generated/YZcolIdentityLimb1Inputs0Data.lean",
        "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeRecursive/PiRlcProjection/Generated/YZcolIdentityLimb1Inputs1Data.lean",
        "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeRecursive/PiRlcProjection/Generated/YZcolIdentityLimb1Inputs2Data.lean",
    ],
];

const TAIL_DATA_PATHS: [&str; 2] = [
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeRecursive/PiRlcProjection/Generated/YZcolIdentityLimb0TailData.lean",
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeRecursive/PiRlcProjection/Generated/YZcolIdentityLimb1TailData.lean",
];

#[derive(Clone, Debug, PartialEq, Eq)]
struct IdentityPairOwner {
    pair_index: usize,
    input_evaluation_trace_index: usize,
    product_trace_index: usize,
    input_evaluation_row_start: usize,
    input_evaluation_row_end: usize,
    input_evaluation_allocated_start: usize,
    input_evaluation_allocated_end: usize,
    input_columns: Vec<usize>,
    input_evaluation_output: [usize; 2],
    product_row_start: usize,
    product_row_end: usize,
    product_allocated_start: usize,
    product_allocated_end: usize,
    product_output: [usize; 2],
}

#[derive(Debug, PartialEq, Eq)]
struct IdentityOwner {
    identity_index: usize,
    limb: usize,
    identity_row_start: usize,
    identity_row_end: usize,
    identity_allocated_start: usize,
    identity_allocated_end: usize,
    input_stage_path: &'static str,
    product_stage_path: &'static str,
    output_stage_path: &'static str,
    quotient_stage_path: &'static str,
    quotient_phi_stage_path: &'static str,
    final_checks_stage_path: &'static str,
    pairs: Vec<IdentityPairOwner>,
    output_columns: Vec<usize>,
    output_evaluation_trace_index: usize,
    output_evaluation_row_start: usize,
    output_evaluation_row_end: usize,
    output_evaluation_allocated_start: usize,
    output_evaluation_allocated_end: usize,
    output_evaluation_output: [usize; 2],
    quotient_columns: Vec<usize>,
    quotient_evaluation_trace_index: usize,
    quotient_evaluation_row_start: usize,
    quotient_evaluation_row_end: usize,
    quotient_evaluation_allocated_start: usize,
    quotient_evaluation_allocated_end: usize,
    quotient_evaluation_output: [usize; 2],
    quotient_phi_product_trace_index: usize,
    quotient_phi_row_start: usize,
    quotient_phi_row_end: usize,
    quotient_phi_allocated_start: usize,
    quotient_phi_allocated_end: usize,
    quotient_phi_output: [usize; 2],
    final_check_row_start: usize,
    final_check_row_end: usize,
}

#[derive(Clone, Copy)]
struct StagePaths {
    input: &'static str,
    product: &'static str,
    output: &'static str,
    quotient: &'static str,
    quotient_phi: &'static str,
    final_checks: &'static str,
}

fn stage_paths(limb: usize) -> StagePaths {
    match limb {
        0 => StagePaths {
            input: pi_rlc_stage::IDENTITIES_Y_ZCOL_EVALUATIONS_INPUTS_LIMB0,
            product: pi_rlc_stage::IDENTITIES_Y_ZCOL_K_PRODUCTS_RHO_TIMES_INPUT_LIMB0,
            output: pi_rlc_stage::IDENTITIES_Y_ZCOL_EVALUATIONS_OUTPUT_LIMB0,
            quotient: pi_rlc_stage::IDENTITIES_Y_ZCOL_EVALUATIONS_QUOTIENT_LIMB0,
            quotient_phi: pi_rlc_stage::IDENTITIES_Y_ZCOL_K_PRODUCTS_QUOTIENT_TIMES_PHI_LIMB0,
            final_checks: pi_rlc_stage::IDENTITIES_Y_ZCOL_FINAL_LIMB_CHECKS_LIMB0,
        },
        1 => StagePaths {
            input: pi_rlc_stage::IDENTITIES_Y_ZCOL_EVALUATIONS_INPUTS_LIMB1,
            product: pi_rlc_stage::IDENTITIES_Y_ZCOL_K_PRODUCTS_RHO_TIMES_INPUT_LIMB1,
            output: pi_rlc_stage::IDENTITIES_Y_ZCOL_EVALUATIONS_OUTPUT_LIMB1,
            quotient: pi_rlc_stage::IDENTITIES_Y_ZCOL_EVALUATIONS_QUOTIENT_LIMB1,
            quotient_phi: pi_rlc_stage::IDENTITIES_Y_ZCOL_K_PRODUCTS_QUOTIENT_TIMES_PHI_LIMB1,
            final_checks: pi_rlc_stage::IDENTITIES_Y_ZCOL_FINAL_LIMB_CHECKS_LIMB1,
        },
        _ => unreachable!("validated y_zcol limb"),
    }
}

fn exact_stage_interval(
    trace: &R1csEncodingTrace,
    stage_path: &'static str,
    expected_rows: Range<usize>,
    expected_columns: Range<usize>,
) {
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
        .count();
    assert_eq!(
        matching, 1,
        "expected one exact `{stage_path}` interval with rows {expected_rows:?} and columns {expected_columns:?}"
    );
}

fn evaluation_allocated_interval(evaluation: &PolynomialEvaluationTraceEntry) -> Range<usize> {
    assert!(!evaluation.allocated_columns.is_empty(), "evaluation allocation");
    let start = evaluation.allocated_columns[0];
    let end = start + evaluation.allocated_columns.len();
    assert_eq!(
        evaluation.allocated_columns,
        (start..end).collect::<Vec<_>>(),
        "evaluation allocation must be one contiguous SSA interval"
    );
    start..end
}

fn product_allocated_interval(product: &KMulTraceEntry) -> Range<usize> {
    let columns = product
        .intermediates
        .iter()
        .chain(&product.output)
        .map(|variable| variable.col())
        .collect::<Vec<_>>();
    assert_eq!(columns.len(), PAIR_PRODUCT_ROWS, "one exact K-mul allocation");
    let start = columns[0];
    let end = start + columns.len();
    assert_eq!(columns, (start..end).collect::<Vec<_>>(), "contiguous K-mul allocation");
    start..end
}

fn product_output(product: &KMulTraceEntry) -> [usize; 2] {
    [product.output[0].col(), product.output[1].col()]
}

fn assert_trace_audit_agree(
    identity: &ProjectionIdentityTraceEntry,
    audit: &ProjectionIdentityAudit,
    trace: &R1csEncodingTrace,
) {
    assert_eq!(identity.role, audit.role, "identity role");
    assert_eq!(identity.source_rows, audit.row_start..audit.row_end, "identity rows");
    assert_eq!(identity.power_columns, audit.power_columns, "power columns");
    assert_eq!(identity.rho_columns, audit.rho_columns, "rho columns");
    assert_eq!(
        identity.rho_evaluation_outputs, audit.rho_evaluation_outputs,
        "rho outputs"
    );
    assert_eq!(identity.input_columns, audit.input_columns, "input columns");
    assert_eq!(identity.output_columns, audit.output_columns, "output columns");
    assert_eq!(identity.quotient_columns, audit.quotient_columns, "quotient columns");

    let evaluations = trace.polynomial_evaluations();
    let input_outputs = identity
        .input_evaluations
        .clone()
        .map(|index| evaluations[index].output_cols)
        .collect::<Vec<_>>();
    assert_eq!(
        input_outputs, audit.input_evaluation_outputs,
        "input evaluation outputs"
    );
    assert_eq!(
        evaluations[identity.output_evaluation].output_cols, audit.output_evaluation,
        "output evaluation output"
    );
    assert_eq!(
        evaluations[identity.quotient_evaluation].output_cols, audit.quotient_evaluation,
        "quotient evaluation output"
    );

    let products = trace.k_muls();
    let pair_outputs = identity
        .pair_products
        .clone()
        .map(|index| product_output(&products[index]))
        .collect::<Vec<_>>();
    assert_eq!(pair_outputs, audit.pair_product_outputs, "pair product outputs");
    assert_eq!(
        product_output(&products[identity.quotient_phi_product]),
        audit.quotient_phi_product,
        "quotient/Phi output"
    );
}

fn selected_owners(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    audits: &[ProjectionIdentityAudit],
) -> Vec<IdentityOwner> {
    validate_projection_identity_traces(source, trace).expect("exact production projection trace");
    assert_eq!(
        trace.projection_identities().len(),
        audits.len(),
        "trace/audit identity census"
    );

    let selected = trace
        .projection_identities()
        .iter()
        .zip(audits)
        .enumerate()
        .filter_map(|(identity_index, (identity, audit))| match identity.role {
            ProjectionIdentityRole::YZColLimb { limb } => Some((identity_index, limb, identity, audit)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        selected
            .iter()
            .map(|(_, limb, _, _)| *limb)
            .collect::<Vec<_>>(),
        [0, 1],
        "exact ordered y_zcol identity pair"
    );
    let mut owners = Vec::with_capacity(2);
    for (identity_index, limb, identity, audit) in selected {
        assert_trace_audit_agree(identity, audit, trace);
        let paths = stage_paths(limb);
        assert_eq!(identity.input_columns.len(), PAIR_COUNT, "input arity");
        assert_eq!(identity.rho_columns.len(), PAIR_COUNT, "rho arity");
        assert_eq!(identity.input_evaluations.len(), PAIR_COUNT, "input evaluator arity");
        assert_eq!(identity.pair_products.len(), PAIR_COUNT, "pair-product arity");
        assert_eq!(identity.output_columns.len(), D, "output width");
        assert_eq!(identity.quotient_columns.len(), QUOTIENT_COEFFICIENTS, "quotient width");
        assert_eq!(
            identity.source_rows.len(),
            COMPLETE_IDENTITY_ROWS,
            "complete identity rows"
        );
        assert_eq!(
            identity.allocated_columns.len(),
            COMPLETE_IDENTITY_COLUMNS,
            "complete identity allocations"
        );

        let evaluations = trace.polynomial_evaluations();
        let products = trace.k_muls();
        let mut row_cursor = identity.source_rows.start;
        let mut column_cursor = identity.allocated_columns.start;
        let mut pairs = Vec::with_capacity(PAIR_COUNT);

        for pair_index in 0..PAIR_COUNT {
            let evaluation_index = identity.input_evaluations.start + pair_index;
            let evaluation = &evaluations[evaluation_index];
            let evaluation_columns = evaluation_allocated_interval(evaluation);
            assert_eq!(evaluation.coefficient_cols, identity.input_columns[pair_index]);
            assert_eq!(evaluation.power_cols, identity.power_columns[..D]);
            assert_eq!(evaluation.row_end - evaluation.row_start, INPUT_EVALUATION_ROWS);
            assert_eq!(evaluation.row_start, row_cursor, "input evaluation row cursor");
            assert_eq!(
                evaluation_columns.start, column_cursor,
                "input evaluation column cursor"
            );
            exact_stage_interval(
                trace,
                paths.input,
                evaluation.row_start..evaluation.row_end,
                evaluation_columns.clone(),
            );
            row_cursor = evaluation.row_end;
            column_cursor = evaluation_columns.end;

            let product_index = identity.pair_products.start + pair_index;
            let product = &products[product_index];
            let product_columns = product_allocated_interval(product);
            assert_eq!(product.source_rows.len(), PAIR_PRODUCT_ROWS, "pair K-mul rows");
            assert_eq!(product.source_rows.start, row_cursor, "pair product row cursor");
            assert_eq!(product_columns.start, column_cursor, "pair product column cursor");
            assert_eq!(product_output(product), audit.pair_product_outputs[pair_index]);
            exact_stage_interval(
                trace,
                paths.product,
                product.source_rows.clone(),
                product_columns.clone(),
            );
            row_cursor = product.source_rows.end;
            column_cursor = product_columns.end;

            pairs.push(IdentityPairOwner {
                pair_index,
                input_evaluation_trace_index: evaluation_index,
                product_trace_index: product_index,
                input_evaluation_row_start: evaluation.row_start,
                input_evaluation_row_end: evaluation.row_end,
                input_evaluation_allocated_start: evaluation_columns.start,
                input_evaluation_allocated_end: evaluation_columns.end,
                input_columns: evaluation.coefficient_cols.clone(),
                input_evaluation_output: evaluation.output_cols,
                product_row_start: product.source_rows.start,
                product_row_end: product.source_rows.end,
                product_allocated_start: product_columns.start,
                product_allocated_end: product_columns.end,
                product_output: product_output(product),
            });
        }

        let output_evaluation = &evaluations[identity.output_evaluation];
        let output_columns = evaluation_allocated_interval(output_evaluation);
        assert_eq!(output_evaluation.coefficient_cols, identity.output_columns);
        assert_eq!(output_evaluation.power_cols, identity.power_columns[..D]);
        assert_eq!(
            output_evaluation.row_end - output_evaluation.row_start,
            OUTPUT_EVALUATION_ROWS
        );
        assert_eq!(output_evaluation.row_start, row_cursor, "output evaluation row cursor");
        assert_eq!(output_columns.start, column_cursor, "output evaluation column cursor");
        exact_stage_interval(
            trace,
            paths.output,
            output_evaluation.row_start..output_evaluation.row_end,
            output_columns.clone(),
        );
        row_cursor = output_evaluation.row_end;
        column_cursor = output_columns.end;

        let quotient_evaluation = &evaluations[identity.quotient_evaluation];
        let quotient_columns = evaluation_allocated_interval(quotient_evaluation);
        assert_eq!(quotient_evaluation.coefficient_cols, identity.quotient_columns);
        assert_eq!(
            quotient_evaluation.power_cols,
            identity.power_columns[..QUOTIENT_COEFFICIENTS]
        );
        assert_eq!(
            quotient_evaluation.row_end - quotient_evaluation.row_start,
            QUOTIENT_EVALUATION_ROWS
        );
        assert_eq!(quotient_evaluation.row_start, row_cursor, "quotient row cursor");
        assert_eq!(quotient_columns.start, column_cursor, "quotient column cursor");
        exact_stage_interval(
            trace,
            paths.quotient,
            quotient_evaluation.row_start..quotient_evaluation.row_end,
            quotient_columns.clone(),
        );
        row_cursor = quotient_evaluation.row_end;
        column_cursor = quotient_columns.end;

        let quotient_phi = &products[identity.quotient_phi_product];
        let quotient_phi_columns = product_allocated_interval(quotient_phi);
        assert_eq!(quotient_phi.source_rows.len(), QUOTIENT_PHI_ROWS);
        assert_eq!(quotient_phi.source_rows.start, row_cursor, "quotient/Phi row cursor");
        assert_eq!(quotient_phi_columns.start, column_cursor, "quotient/Phi column cursor");
        exact_stage_interval(
            trace,
            paths.quotient_phi,
            quotient_phi.source_rows.clone(),
            quotient_phi_columns.clone(),
        );
        row_cursor = quotient_phi.source_rows.end;
        column_cursor = quotient_phi_columns.end;

        assert_eq!(identity.final_limb_rows, row_cursor..row_cursor + FINAL_CHECK_ROWS);
        exact_stage_interval(
            trace,
            paths.final_checks,
            identity.final_limb_rows.clone(),
            column_cursor..column_cursor,
        );
        row_cursor = identity.final_limb_rows.end;
        assert_eq!(row_cursor, identity.source_rows.end, "complete identity row cursor");
        assert_eq!(
            column_cursor, identity.allocated_columns.end,
            "complete identity column cursor"
        );

        owners.push(IdentityOwner {
            identity_index,
            limb,
            identity_row_start: identity.source_rows.start,
            identity_row_end: identity.source_rows.end,
            identity_allocated_start: identity.allocated_columns.start,
            identity_allocated_end: identity.allocated_columns.end,
            input_stage_path: paths.input,
            product_stage_path: paths.product,
            output_stage_path: paths.output,
            quotient_stage_path: paths.quotient,
            quotient_phi_stage_path: paths.quotient_phi,
            final_checks_stage_path: paths.final_checks,
            pairs,
            output_columns: identity.output_columns.clone(),
            output_evaluation_trace_index: identity.output_evaluation,
            output_evaluation_row_start: output_evaluation.row_start,
            output_evaluation_row_end: output_evaluation.row_end,
            output_evaluation_allocated_start: output_columns.start,
            output_evaluation_allocated_end: output_columns.end,
            output_evaluation_output: output_evaluation.output_cols,
            quotient_columns: identity.quotient_columns.clone(),
            quotient_evaluation_trace_index: identity.quotient_evaluation,
            quotient_evaluation_row_start: quotient_evaluation.row_start,
            quotient_evaluation_row_end: quotient_evaluation.row_end,
            quotient_evaluation_allocated_start: quotient_columns.start,
            quotient_evaluation_allocated_end: quotient_columns.end,
            quotient_evaluation_output: quotient_evaluation.output_cols,
            quotient_phi_product_trace_index: identity.quotient_phi_product,
            quotient_phi_row_start: quotient_phi.source_rows.start,
            quotient_phi_row_end: quotient_phi.source_rows.end,
            quotient_phi_allocated_start: quotient_phi_columns.start,
            quotient_phi_allocated_end: quotient_phi_columns.end,
            quotient_phi_output: product_output(quotient_phi),
            final_check_row_start: identity.final_limb_rows.start,
            final_check_row_end: identity.final_limb_rows.end,
        });
    }

    assert_eq!(owners[0].identity_index + 1, owners[1].identity_index);
    assert_eq!(owners[0].identity_row_end, owners[1].identity_row_start);
    assert_eq!(owners[0].identity_allocated_end, owners[1].identity_allocated_start);
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

fn lean_source_rows(source: &R1csSnapshot, rows: impl IntoIterator<Item = usize>) -> String {
    let rows = rows
        .into_iter()
        .map(|row| {
            format!(
                "({row}, ⟨{}, {}, {}⟩)",
                lean_terms(source.a_row(row)),
                lean_terms(source.b_row(row)),
                lean_terms(source.c_row(row))
            )
        })
        .collect::<Vec<_>>();
    format!("[{}]", rows.join(",\n   "))
}

fn lean_pair(owner: &IdentityPairOwner) -> String {
    format!(
        "{{ pairIndex := {}, inputEvaluationTraceIndex := {}, productTraceIndex := {}, inputEvaluationRowStart := {}, inputEvaluationRowEnd := {}, inputEvaluationAllocatedStart := {}, inputEvaluationAllocatedEnd := {}, inputColumns := {}, inputEvaluationOutput := {}, productRowStart := {}, productRowEnd := {}, productAllocatedStart := {}, productAllocatedEnd := {}, productOutput := {} }}",
        owner.pair_index,
        owner.input_evaluation_trace_index,
        owner.product_trace_index,
        owner.input_evaluation_row_start,
        owner.input_evaluation_row_end,
        owner.input_evaluation_allocated_start,
        owner.input_evaluation_allocated_end,
        lean_nat_list(owner.input_columns.iter().copied()),
        lean_k_columns(owner.input_evaluation_output),
        owner.product_row_start,
        owner.product_row_end,
        owner.product_allocated_start,
        owner.product_allocated_end,
        lean_k_columns(owner.product_output),
    )
}

fn render_input_shard(source: &R1csSnapshot, owner: &IdentityOwner, shard: usize) -> String {
    let pair_start = shard * INPUTS_PER_SHARD;
    let pairs = &owner.pairs[pair_start..pair_start + INPUTS_PER_SHARD];
    let rows = pairs
        .iter()
        .flat_map(|pair| pair.input_evaluation_row_start..pair.input_evaluation_row_end)
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), INPUTS_PER_SHARD * INPUT_EVALUATION_ROWS);
    let mut rendered = String::new();
    rendered.push_str(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.YZcolIdentitiesSchema\n\n",
    );
    rendered.push_str(
        "/-! Generated by `active_pi_rlc_projection_artifacts_match_production_trace`; do not hand-edit. -/\n\n",
    );
    writeln!(
        rendered,
        "namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjectionYZcolIdentitiesData.Limb{}\n",
        owner.limb
    )
    .expect("render input namespace");
    writeln!(rendered, "set_option maxRecDepth 100000 in").expect("render option");
    writeln!(
        rendered,
        "def inputSourceRows{shard} : List (Nat × Row) :=\n  {}\n",
        lean_source_rows(source, rows)
    )
    .expect("render input source rows");
    writeln!(
        rendered,
        "end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjectionYZcolIdentitiesData.Limb{}",
        owner.limb
    )
    .expect("render input namespace end");
    rendered
}

fn render_tail(source: &R1csSnapshot, owner: &IdentityOwner) -> String {
    let tail_definition_rows = owner
        .pairs
        .iter()
        .flat_map(|pair| pair.product_row_start..pair.product_row_end)
        .chain(owner.quotient_evaluation_row_start..owner.quotient_evaluation_row_end)
        .chain(owner.quotient_phi_row_start..owner.quotient_phi_row_end)
        .collect::<Vec<_>>();
    assert_eq!(
        tail_definition_rows.len(),
        PAIR_COUNT * PAIR_PRODUCT_ROWS + QUOTIENT_EVALUATION_ROWS + QUOTIENT_PHI_ROWS
    );
    let check_rows = owner.final_check_row_start..owner.final_check_row_end;
    assert_eq!(check_rows.len(), FINAL_CHECK_ROWS);

    let mut rendered = String::new();
    rendered.push_str(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.YZcolIdentitiesSchema\n\n",
    );
    rendered.push_str(
        "/-! Generated by `active_pi_rlc_projection_artifacts_match_production_trace`; do not hand-edit. -/\n\n",
    );
    writeln!(
        rendered,
        "namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjectionYZcolIdentitiesData.Limb{}\n",
        owner.limb
    )
    .expect("render tail namespace");
    writeln!(rendered, "def owner : PiRlcYZcolIdentityOwner where").expect("render owner");
    writeln!(rendered, "  identityIndex := {}", owner.identity_index).expect("render owner");
    writeln!(rendered, "  limb := {}", owner.limb).expect("render owner");
    writeln!(rendered, "  identityRowStart := {}", owner.identity_row_start).expect("render owner");
    writeln!(rendered, "  identityRowEnd := {}", owner.identity_row_end).expect("render owner");
    writeln!(
        rendered,
        "  identityAllocatedStart := {}",
        owner.identity_allocated_start
    )
    .expect("render owner");
    writeln!(rendered, "  identityAllocatedEnd := {}", owner.identity_allocated_end).expect("render owner");
    writeln!(rendered, "  inputStagePath := \"{}\"", owner.input_stage_path).expect("render owner");
    writeln!(rendered, "  productStagePath := \"{}\"", owner.product_stage_path).expect("render owner");
    writeln!(rendered, "  outputStagePath := \"{}\"", owner.output_stage_path).expect("render owner");
    writeln!(rendered, "  quotientStagePath := \"{}\"", owner.quotient_stage_path).expect("render owner");
    writeln!(
        rendered,
        "  quotientPhiStagePath := \"{}\"",
        owner.quotient_phi_stage_path
    )
    .expect("render owner");
    writeln!(
        rendered,
        "  finalChecksStagePath := \"{}\"",
        owner.final_checks_stage_path
    )
    .expect("render owner");
    rendered.push_str("  pairs :=\n");
    for (index, pair) in owner.pairs.iter().enumerate() {
        let prefix = if index == 0 { "    [ " } else { "    , " };
        writeln!(rendered, "{prefix}{}", lean_pair(pair)).expect("render pair");
    }
    rendered.push_str("    ]\n");
    writeln!(
        rendered,
        "  outputColumns := {}",
        lean_nat_list(owner.output_columns.iter().copied())
    )
    .expect("render owner");
    writeln!(
        rendered,
        "  outputEvaluationTraceIndex := {}",
        owner.output_evaluation_trace_index
    )
    .expect("render owner");
    writeln!(
        rendered,
        "  outputEvaluationRowStart := {}",
        owner.output_evaluation_row_start
    )
    .expect("render owner");
    writeln!(
        rendered,
        "  outputEvaluationRowEnd := {}",
        owner.output_evaluation_row_end
    )
    .expect("render owner");
    writeln!(
        rendered,
        "  outputEvaluationAllocatedStart := {}",
        owner.output_evaluation_allocated_start
    )
    .expect("render owner");
    writeln!(
        rendered,
        "  outputEvaluationAllocatedEnd := {}",
        owner.output_evaluation_allocated_end
    )
    .expect("render owner");
    writeln!(
        rendered,
        "  outputEvaluationOutput := {}",
        lean_k_columns(owner.output_evaluation_output)
    )
    .expect("render owner");
    writeln!(
        rendered,
        "  quotientColumns := {}",
        lean_nat_list(owner.quotient_columns.iter().copied())
    )
    .expect("render owner");
    writeln!(
        rendered,
        "  quotientEvaluationTraceIndex := {}",
        owner.quotient_evaluation_trace_index
    )
    .expect("render owner");
    writeln!(
        rendered,
        "  quotientEvaluationRowStart := {}",
        owner.quotient_evaluation_row_start
    )
    .expect("render owner");
    writeln!(
        rendered,
        "  quotientEvaluationRowEnd := {}",
        owner.quotient_evaluation_row_end
    )
    .expect("render owner");
    writeln!(
        rendered,
        "  quotientEvaluationAllocatedStart := {}",
        owner.quotient_evaluation_allocated_start
    )
    .expect("render owner");
    writeln!(
        rendered,
        "  quotientEvaluationAllocatedEnd := {}",
        owner.quotient_evaluation_allocated_end
    )
    .expect("render owner");
    writeln!(
        rendered,
        "  quotientEvaluationOutput := {}",
        lean_k_columns(owner.quotient_evaluation_output)
    )
    .expect("render owner");
    writeln!(
        rendered,
        "  quotientPhiProductTraceIndex := {}",
        owner.quotient_phi_product_trace_index
    )
    .expect("render owner");
    writeln!(rendered, "  quotientPhiRowStart := {}", owner.quotient_phi_row_start).expect("render owner");
    writeln!(rendered, "  quotientPhiRowEnd := {}", owner.quotient_phi_row_end).expect("render owner");
    writeln!(
        rendered,
        "  quotientPhiAllocatedStart := {}",
        owner.quotient_phi_allocated_start
    )
    .expect("render owner");
    writeln!(
        rendered,
        "  quotientPhiAllocatedEnd := {}",
        owner.quotient_phi_allocated_end
    )
    .expect("render owner");
    writeln!(
        rendered,
        "  quotientPhiOutput := {}",
        lean_k_columns(owner.quotient_phi_output)
    )
    .expect("render owner");
    writeln!(rendered, "  finalCheckRowStart := {}", owner.final_check_row_start).expect("render owner");
    writeln!(rendered, "  finalCheckRowEnd := {}\n", owner.final_check_row_end).expect("render owner");
    rendered.push_str("set_option maxRecDepth 100000 in\n");
    writeln!(
        rendered,
        "def tailDefinitionSourceRows : List (Nat × Row) :=\n  {}\n",
        lean_source_rows(source, tail_definition_rows)
    )
    .expect("render tail definition rows");
    writeln!(
        rendered,
        "def checkSourceRows : List (Nat × Row) :=\n  {}\n",
        lean_source_rows(source, check_rows)
    )
    .expect("render check rows");
    writeln!(
        rendered,
        "end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjectionYZcolIdentitiesData.Limb{}",
        owner.limb
    )
    .expect("render tail namespace end");
    rendered
}

fn compare_or_write_expected(relative_path: &str, rendered: &str) -> bool {
    let path = repo_root().join(relative_path);
    let committed = fs::read_to_string(&path).unwrap_or_default();
    if committed == rendered {
        return false;
    }
    let expected = path.with_extension("lean.expected");
    fs::create_dir_all(expected.parent().expect("generated artifact parent"))
        .expect("create generated artifact directory");
    fs::write(&expected, rendered).expect("write expected y_zcol identity artifact");
    true
}

pub(super) fn check_generated_artifact(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    audits: &[ProjectionIdentityAudit],
) {
    let owners = selected_owners(source, trace, audits);
    let mut drifted = Vec::new();
    for owner in &owners {
        for shard in 0..3 {
            let rendered = render_input_shard(source, owner, shard);
            if compare_or_write_expected(INPUT_DATA_PATHS[owner.limb][shard], &rendered) {
                drifted.push(format!("limb{}-inputs{shard}", owner.limb));
            }
        }
        let rendered = render_tail(source, owner);
        if compare_or_write_expected(TAIL_DATA_PATHS[owner.limb], &rendered) {
            drifted.push(format!("limb{}-tail", owner.limb));
        }
    }
    assert!(
        drifted.is_empty(),
        "active y_zcol identity shards {drifted:?} drifted; review the generated .expected files"
    );
}
