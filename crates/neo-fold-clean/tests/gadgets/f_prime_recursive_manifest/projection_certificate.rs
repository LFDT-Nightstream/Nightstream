//! Generated Lean certificate for the exact production projection compaction.
//!
//! Owns: source-schema normalization, compact-plan extraction, exact cost
//! recomputation, Lean rendering, and the drift regression.
//!
//! Does not own: production relation construction or semantic authority for
//! diagnostic identity roles.
//!
//! Emits constraints: no.
//!
//! Authority boundary: the production source rows and their validated compact
//! plan are authoritative; the rendered Lean data is non-authoritative
//! structure whose exact agreement is enforced by the generator regression.
//!
//! | Artifact branch | Mathematical content | Production evidence |
//! |---|---|---|
//! | evaluations | coefficient-zero, retained outputs, exact chunk tails | compact-plan audit |
//! | retained matrix | exact 34-entry diagonal | product-sum rank validator |
//! | terminal limbs | exact operands, signs, W, and 18/14 chunks | source-row replay |
//! | mixed cost tree | 41-column retained fields, 95-column synthetic carries, and per-stage rows | production stage-profile reconciliation |

use std::fmt::Write as _;
use std::fs;

use neo_fold_clean::engine::r1cs_circuit::projection_identity_trace::validate_projection_identity_traces;
use neo_fold_clean::engine::r1cs_circuit::{
    ProjectionIdentityRole, ProjectionNebulaCoordinate, R1csEncodingTrace, R1csSnapshot,
};
use neo_fold_clean::frontends::f_prime::gadget_native::{
    audit_projection_identity_compaction, profile_r1cs_gadget_native_stages, ProjectionCoefficientZero,
    ProjectionEvaluationKind, ProjectionFinalCoefficient, ProjectionFinalFactorAudit, ProjectionFinalOperand,
    ORDINARY_PRIVATE_DIGITS,
};
use neo_fold_clean::paper::reductions::pi_rlc_circuit::stage as pi_rlc_stage;
use neo_math::Fq;
use p3_field::extension::BinomiallyExtendable;
use p3_field::PrimeField64;

use super::{build_recursive_program, repo_root, sha256_hex};

const PROJECTION_CERTIFICATE_PATH: &str =
    "formal/superneo-lean/SuperNeo/FPrimeRecursiveVerifier/PiRlcAlgebra/Refinement/Generated/ProjectionIdentityCertificateData.lean";
const EVALUATION_CHUNK_WIDTH: usize = 18;
const EXTENSION_LIMBS: usize = 2;
const KARATSUBA_PRODUCTS: usize = 3;
const SYNTHETIC_CANONICAL_SLOT_WIDTH: usize = 95;
const SYNTHETIC_CANONICALITY_RELATIONS: usize = 32;
const SYNTHETIC_CANONICALITY_PAIR_ROWS: usize = SYNTHETIC_CANONICALITY_RELATIONS / 2;
const COST_STAGE_COUNT: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
struct ProjectionIdentitySchema {
    pair_count: usize,
    source_rows: usize,
    source_columns: usize,
    input_evaluations: Vec<(usize, usize, usize, usize, usize)>,
    pair_products: Vec<(usize, usize, usize, usize)>,
    output_evaluation: (usize, usize, usize, usize, usize),
    quotient_evaluation: (usize, usize, usize, usize, usize),
    quotient_phi_product: (usize, usize, usize, usize),
    final_limb_rows: (usize, usize),
}

fn normalized_projection_identity_schema(trace: &R1csEncodingTrace, identity_index: usize) -> ProjectionIdentitySchema {
    let identity = &trace.projection_identities()[identity_index];
    let row_base = identity.source_rows.start;
    let column_base = identity.allocated_columns.start;
    let evaluation = |index: usize| {
        let evaluation = &trace.polynomial_evaluations()[index];
        let column_start = *evaluation
            .allocated_columns
            .first()
            .expect("validated evaluation allocates columns");
        let column_end = evaluation
            .allocated_columns
            .last()
            .expect("validated evaluation allocates columns")
            + 1;
        (
            evaluation.row_start - row_base,
            evaluation.row_end - row_base,
            column_start - column_base,
            column_end - column_base,
            evaluation.coefficient_cols.len(),
        )
    };
    let product = |index: usize| {
        let product = &trace.k_muls()[index];
        (
            product.source_rows.start - row_base,
            product.source_rows.end - row_base,
            product.intermediates[0].col() - column_base,
            product.output[1].col() + 1 - column_base,
        )
    };
    ProjectionIdentitySchema {
        pair_count: identity.input_columns.len(),
        source_rows: identity.source_rows.len(),
        source_columns: identity.allocated_columns.len(),
        input_evaluations: identity.input_evaluations.clone().map(evaluation).collect(),
        pair_products: identity.pair_products.clone().map(product).collect(),
        output_evaluation: evaluation(identity.output_evaluation),
        quotient_evaluation: evaluation(identity.quotient_evaluation),
        quotient_phi_product: product(identity.quotient_phi_product),
        final_limb_rows: (
            identity.final_limb_rows.start - row_base,
            identity.final_limb_rows.end - row_base,
        ),
    }
}

fn common_projection_identity_schema(trace: &R1csEncodingTrace) -> ProjectionIdentitySchema {
    assert!(!trace.projection_identities().is_empty(), "projection identity census");
    let representative = normalized_projection_identity_schema(trace, 0);
    for identity_index in 1..trace.projection_identities().len() {
        assert_eq!(
            normalized_projection_identity_schema(trace, identity_index),
            representative,
            "projection identity {identity_index} must instantiate the representative schema"
        );
    }
    representative
}

fn lean_projection_role(role: ProjectionIdentityRole) -> String {
    match role {
        ProjectionIdentityRole::Standalone => ".standalone".to_owned(),
        ProjectionIdentityRole::CommitmentLane { lane } => format!(".commitmentLane {lane}"),
        ProjectionIdentityRole::NebulaCommitmentLane { coordinate, lane } => match coordinate {
            ProjectionNebulaCoordinate::Ops => format!(".adviceOpsLane {lane}"),
            ProjectionNebulaCoordinate::Is => format!(".adviceIsLane {lane}"),
            ProjectionNebulaCoordinate::Fs => format!(".adviceFsLane {lane}"),
        },
        ProjectionIdentityRole::ActiveXColumn { column } => format!(".activeXColumn {column}"),
        ProjectionIdentityRole::YRingLimb { row, limb } => format!(".yRingLimb {row} {limb}"),
        ProjectionIdentityRole::YZColLimb { limb } => format!(".yZcolLimb {limb}"),
    }
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

fn lean_nested_nat_lists(values: impl IntoIterator<Item = impl IntoIterator<Item = usize>>) -> String {
    format!(
        "[{}]",
        values
            .into_iter()
            .map(lean_nat_list)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_evaluation_kind(kind: ProjectionEvaluationKind) -> String {
    match kind {
        ProjectionEvaluationKind::Input { pair } => format!(".input {pair}"),
        ProjectionEvaluationKind::Output => ".output".to_owned(),
        ProjectionEvaluationKind::Quotient => ".quotient".to_owned(),
    }
}

fn lean_coefficient_zero(mode: ProjectionCoefficientZero) -> &'static str {
    match mode {
        ProjectionCoefficientZero::SubtractFromResult => ".subtractFromResult",
        ProjectionCoefficientZero::Absent => ".absent",
    }
}

fn lean_final_operand(operand: ProjectionFinalOperand) -> String {
    match operand {
        ProjectionFinalOperand::RhoEvaluation { pair, limb } => format!(".rhoEvaluation {pair} {limb}"),
        ProjectionFinalOperand::InputEvaluation { pair, limb } => format!(".inputEvaluation {pair} {limb}"),
        ProjectionFinalOperand::QuotientEvaluation { limb } => format!(".quotientEvaluation {limb}"),
        ProjectionFinalOperand::Phi { limb } => format!(".phi {limb}"),
    }
}

fn lean_final_coefficient(coefficient: ProjectionFinalCoefficient) -> &'static str {
    match coefficient {
        ProjectionFinalCoefficient::One => ".one",
        ProjectionFinalCoefficient::NegOne => ".negOne",
        ProjectionFinalCoefficient::W => ".w",
        ProjectionFinalCoefficient::NegW => ".negW",
    }
}

fn pair_tail_rows(coordinates: usize) -> usize {
    coordinates / 2 + coordinates % 2
}

fn cost_stage(kind: ProjectionEvaluationKind) -> usize {
    match kind {
        ProjectionEvaluationKind::Input { .. } => 0,
        ProjectionEvaluationKind::Output => 1,
        ProjectionEvaluationKind::Quotient => 2,
    }
}

fn lean_final_factor(factor: ProjectionFinalFactorAudit) -> String {
    format!(
        "{{ left := {}, right := {}, coefficient := {} }}",
        lean_final_operand(factor.left),
        lean_final_operand(factor.right),
        lean_final_coefficient(factor.coefficient)
    )
}

fn render_projection_identity_certificate(source: &R1csSnapshot, trace: &R1csEncodingTrace) -> String {
    let schema = common_projection_identity_schema(trace);
    let compaction = audit_projection_identity_compaction(source, trace).expect("exact production compaction plan");
    let roles = &compaction.roles;
    let compact = &compaction.schema;
    let mut evaluation_coefficient_counts = schema
        .input_evaluations
        .iter()
        .map(|evaluation| evaluation.4)
        .collect::<Vec<_>>();
    evaluation_coefficient_counts.push(schema.output_evaluation.4);
    evaluation_coefficient_counts.push(schema.quotient_evaluation.4);

    let mut compact_retained_fields_by_stage = [0usize; COST_STAGE_COUNT];
    let mut compact_synthetic_fields_by_stage = [0usize; COST_STAGE_COUNT];
    let mut compact_product_sum_rows_by_stage = [0usize; COST_STAGE_COUNT];
    for evaluation in &compact.evaluations {
        let stage = cost_stage(evaluation.kind);
        let retained = evaluation.retained_ordinals.len();
        let emitted = evaluation.chunk_sizes.iter().map(Vec::len).sum::<usize>();
        compact_retained_fields_by_stage[stage] += retained;
        compact_synthetic_fields_by_stage[stage] += emitted - retained;
        compact_product_sum_rows_by_stage[stage] += emitted;
    }
    let compact_final_fields = compact
        .final_limbs
        .iter()
        .map(|limb| limb.chunk_sizes.len() - 1)
        .sum::<usize>();
    let compact_final_product_rows = compact
        .final_limbs
        .iter()
        .map(|limb| limb.chunk_sizes.len())
        .sum::<usize>();
    compact_synthetic_fields_by_stage[3] = compact_final_fields;
    compact_product_sum_rows_by_stage[3] = compact_final_product_rows;

    let compact_ordinary_coordinates_by_stage =
        compact_retained_fields_by_stage.map(|fields| fields * ORDINARY_PRIVATE_DIGITS);
    let compact_synthetic_coordinates_by_stage =
        compact_synthetic_fields_by_stage.map(|fields| fields * SYNTHETIC_CANONICAL_SLOT_WIDTH);
    let compact_ordinary_centered_rows_by_stage = compact_ordinary_coordinates_by_stage.map(pair_tail_rows);
    let compact_synthetic_boolean_rows_by_stage = compact_synthetic_coordinates_by_stage.map(pair_tail_rows);
    let compact_synthetic_canonicality_rows_by_stage =
        compact_synthetic_fields_by_stage.map(|fields| fields * SYNTHETIC_CANONICALITY_PAIR_ROWS);
    let compact_encoded_columns_by_stage = std::array::from_fn::<_, COST_STAGE_COUNT, _>(|stage| {
        compact_ordinary_coordinates_by_stage[stage] + compact_synthetic_coordinates_by_stage[stage]
    });
    let compact_encoded_rows_by_stage = std::array::from_fn::<_, COST_STAGE_COUNT, _>(|stage| {
        compact_ordinary_centered_rows_by_stage[stage]
            + compact_synthetic_boolean_rows_by_stage[stage]
            + compact_synthetic_canonicality_rows_by_stage[stage]
            + compact_product_sum_rows_by_stage[stage]
    });

    let compact_evaluation_fields = compact_retained_fields_by_stage[..3].iter().sum::<usize>()
        + compact_synthetic_fields_by_stage[..3].iter().sum::<usize>();
    let compact_evaluation_product_rows = compact_product_sum_rows_by_stage[..3].iter().sum::<usize>();
    let compact_evaluation_columns = compact_encoded_columns_by_stage[..3].iter().sum::<usize>();
    let compact_evaluation_rows = compact_encoded_rows_by_stage[..3].iter().sum::<usize>();
    let compact_final_columns = compact_encoded_columns_by_stage[3];
    let compact_final_rows = compact_encoded_rows_by_stage[3];
    let compact_encoded_columns = compact_evaluation_columns + compact_final_columns;
    let compact_encoded_rows = compact_evaluation_rows + compact_final_rows;
    let compact_retained_fields = compact_retained_fields_by_stage.iter().sum::<usize>();
    let compact_synthetic_fields = compact_synthetic_fields_by_stage.iter().sum::<usize>();
    let compact_product_sum_rows = compact_evaluation_product_rows + compact_final_product_rows;

    assert_eq!(
        (schema.pair_count, schema.source_rows, schema.source_columns),
        (15, 1_916, 1_914)
    );
    assert_eq!(compact_retained_fields, compact.retained_column_offsets.len());
    assert_eq!(
        [
            compact_evaluation_fields,
            compact_evaluation_product_rows,
            compact_evaluation_rows,
            compact_evaluation_columns,
            compact_final_fields,
            compact_final_product_rows,
            compact_final_rows,
            compact_final_columns,
            compact_encoded_rows,
            compact_encoded_columns,
            compact_retained_fields,
            compact_synthetic_fields,
            compact_product_sum_rows,
        ],
        [102, 102, 5_117, 7_854, 2, 4, 131, 190, 5_248, 8_044, 34, 70, 106,]
    );
    assert_eq!(compact_retained_fields_by_stage, [30, 2, 2, 0]);
    assert_eq!(compact_synthetic_fields_by_stage, [60, 4, 4, 2]);
    assert_eq!(compact_product_sum_rows_by_stage, [90, 6, 6, 4]);
    assert_eq!(compact_encoded_columns_by_stage, [6_930, 462, 462, 190]);
    assert_eq!(compact_encoded_rows_by_stage, [4_515, 301, 301, 131]);

    let profile =
        profile_r1cs_gadget_native_stages(source, trace, &[]).expect("production gadget-native stage profile");
    let identities = profile
        .aggregate_prefix(pi_rlc_stage::IDENTITIES)
        .expect("production projection-identity profile");
    assert_eq!(
        (
            identities.ordinary_private_field_source_cols,
            identities.synthetic_product_sum_fields,
            identities.encoded_rows,
            identities.encoded_cols,
        ),
        (
            compact_retained_fields * roles.len(),
            compact_synthetic_fields * roles.len(),
            compact_encoded_rows * roles.len(),
            compact_encoded_columns * roles.len(),
        ),
        "generated mixed cost model must equal the production identity profile"
    );

    let mut rendered = String::new();
    rendered.push_str(
        "/-! Generated by `gadgets_f_prime_recursive_manifest`; do not hand-edit.\n\nOwns: exact normalized source geometry, compact product-sum schedule, and mixed retained/synthetic production cost model extracted from validated rows.\n\nDoes not own: semantic authority for diagnostic role labels, transcript authority, or the exact-or-bad-root reduction.\n\nEmits constraints: no.\n\nAuthority boundary: source R1CS rows are authoritative; generation fails unless trace replay, the compact-plan validator, and the production stage profile agree.\n\n| Data branch | Mathematical obligation | Rust validator |\n|---|---|---|\n| `evaluationPlans` | 17 two-limb evaluations, coefficient-zero handling, exact chunk tails | `projection_identity` plus `product_sum` |\n| `retainedBindings` | Exact 34-by-34 diagonal retained boundary | `product_sum::validate_identities` |\n| `finalLimbPlans` | Ordered `W`/sign schedule and retained terminal outputs | `projection_identity` plus source-row replay |\n| `compact*ByStage` | 41-coordinate retained fields and 95-coordinate synthetic carries reconcile to 5,248 rows by 8,044 columns | `profile_r1cs_gadget_native_stages` |\n-/\n\n",
    );
    rendered.push_str("namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.ProjectionIdentityCertificateData\n\n");
    rendered.push_str("inductive IdentityRole where\n");
    rendered.push_str("  | standalone\n");
    rendered.push_str("  | commitmentLane (lane : Nat)\n");
    rendered.push_str("  | adviceOpsLane (lane : Nat)\n");
    rendered.push_str("  | adviceIsLane (lane : Nat)\n");
    rendered.push_str("  | adviceFsLane (lane : Nat)\n");
    rendered.push_str("  | activeXColumn (column : Nat)\n");
    rendered.push_str("  | yRingLimb (row limb : Nat)\n");
    rendered.push_str("  | yZcolLimb (limb : Nat)\n");
    rendered.push_str("deriving DecidableEq, Repr\n\n");
    rendered.push_str("inductive EvaluationKind where\n");
    rendered.push_str("  | input (pair : Nat)\n");
    rendered.push_str("  | output\n");
    rendered.push_str("  | quotient\n");
    rendered.push_str("deriving DecidableEq, Repr\n\n");
    rendered.push_str("inductive CoefficientZero where\n");
    rendered.push_str("  | subtractFromResult\n");
    rendered.push_str("  | absent\n");
    rendered.push_str("deriving DecidableEq, Repr\n\n");
    rendered.push_str("structure EvaluationPlan where\n");
    rendered.push_str("  kind : EvaluationKind\n");
    rendered.push_str("  sourceRowOffset : Nat\n");
    rendered.push_str("  sourceRowCount : Nat\n");
    rendered.push_str("  coefficientCount : Nat\n");
    rendered.push_str("  retainedOrdinals : List Nat\n");
    rendered.push_str("  retainedColumnOffsets : List Nat\n");
    rendered.push_str("  coefficientZero : List CoefficientZero\n");
    rendered.push_str("  productCounts : List Nat\n");
    rendered.push_str("  chunkSizes : List (List Nat)\n");
    rendered.push_str("deriving DecidableEq, Repr\n\n");
    rendered.push_str("inductive FinalOperand where\n");
    rendered.push_str("  | rhoEvaluation (pair limb : Nat)\n");
    rendered.push_str("  | inputEvaluation (pair limb : Nat)\n");
    rendered.push_str("  | quotientEvaluation (limb : Nat)\n");
    rendered.push_str("  | phi (limb : Nat)\n");
    rendered.push_str("deriving DecidableEq, Repr\n\n");
    rendered.push_str("inductive FinalCoefficient where\n");
    rendered.push_str("  | one\n");
    rendered.push_str("  | negOne\n");
    rendered.push_str("  | w\n");
    rendered.push_str("  | negW\n");
    rendered.push_str("deriving DecidableEq, Repr\n\n");
    rendered.push_str("structure FinalFactor where\n");
    rendered.push_str("  left : FinalOperand\n");
    rendered.push_str("  right : FinalOperand\n");
    rendered.push_str("  coefficient : FinalCoefficient\n");
    rendered.push_str("deriving DecidableEq, Repr\n\n");
    rendered.push_str("structure FinalLimbPlan where\n");
    rendered.push_str("  limb : Nat\n");
    rendered.push_str("  sourceRowOffset : Nat\n");
    rendered.push_str("  resultRetainedOrdinal : Nat\n");
    rendered.push_str("  chunkSizes : List Nat\n");
    rendered.push_str("  factors : List FinalFactor\n");
    rendered.push_str("deriving DecidableEq, Repr\n\n");
    rendered.push_str("structure RetainedBinding where\n");
    rendered.push_str("  identity : Nat\n");
    rendered.push_str("  retainedOrdinal : Nat\n");
    rendered.push_str("  coefficient : FinalCoefficient\n");
    rendered.push_str("deriving DecidableEq, Repr\n\n");
    writeln!(rendered, "def schemaVersion : Nat := 3").expect("write certificate");
    writeln!(rendered, "def rolesAreDiagnostic : Bool := true").expect("write certificate");
    writeln!(rendered, "def identityCount : Nat := {}", roles.len()).expect("write certificate");
    writeln!(rendered, "def pairCount : Nat := {}", schema.pair_count).expect("write certificate");
    writeln!(rendered, "def extensionLimbs : Nat := {EXTENSION_LIMBS}").expect("write certificate");
    writeln!(rendered, "def karatsubaProducts : Nat := {KARATSUBA_PRODUCTS}").expect("write certificate");
    writeln!(
        rendered,
        "def karatsubaW : Nat := {}",
        <Fq as BinomiallyExtendable<2>>::W.as_canonical_u64()
    )
    .expect("write certificate");
    writeln!(rendered, "def evaluationChunkWidth : Nat := {EVALUATION_CHUNK_WIDTH}").expect("write certificate");
    writeln!(rendered, "def ordinarySlotWidth : Nat := {ORDINARY_PRIVATE_DIGITS}").expect("write certificate");
    writeln!(
        rendered,
        "def syntheticCanonicalSlotWidth : Nat := {SYNTHETIC_CANONICAL_SLOT_WIDTH}"
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def syntheticCanonicalityRelations : Nat := {SYNTHETIC_CANONICALITY_RELATIONS}"
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def syntheticCanonicalityPairRows : Nat := {SYNTHETIC_CANONICALITY_PAIR_ROWS}"
    )
    .expect("write certificate");
    rendered.push_str(
        "def EvaluationPlan.productCoefficientIndices (plan : EvaluationPlan) : List Nat :=\n  (List.range plan.coefficientCount).drop 1\n",
    );
    rendered.push_str(
        "def EvaluationPlan.powerIndicesByLimb (plan : EvaluationPlan) : List (List Nat) :=\n  List.replicate extensionLimbs plan.productCoefficientIndices\n",
    );
    writeln!(rendered, "def sourceRowsPerIdentity : Nat := {}", schema.source_rows).expect("write certificate");
    writeln!(
        rendered,
        "def sourceColumnsPerIdentity : Nat := {}",
        schema.source_columns
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def inputEvaluationRowOffsets : List Nat := {}",
        lean_nat_list(
            schema
                .input_evaluations
                .iter()
                .map(|evaluation| evaluation.0)
        )
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def inputEvaluationColumnOffsets : List Nat := {}",
        lean_nat_list(
            schema
                .input_evaluations
                .iter()
                .map(|evaluation| evaluation.2)
        )
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def rhoProductRowOffsets : List Nat := {}",
        lean_nat_list(schema.pair_products.iter().map(|product| product.0))
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def rhoProductColumnOffsets : List Nat := {}",
        lean_nat_list(schema.pair_products.iter().map(|product| product.2))
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def outputEvaluationRowOffset : Nat := {}",
        schema.output_evaluation.0
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def quotientEvaluationRowOffset : Nat := {}",
        schema.quotient_evaluation.0
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def quotientPhiProductRowOffset : Nat := {}",
        schema.quotient_phi_product.0
    )
    .expect("write certificate");
    writeln!(rendered, "def finalLimbRowOffset : Nat := {}", schema.final_limb_rows.0).expect("write certificate");
    writeln!(
        rendered,
        "def evaluationCoefficientCounts : List Nat := {}",
        lean_nat_list(evaluation_coefficient_counts.iter().copied())
    )
    .expect("write certificate");
    rendered.push_str(
        "def compactCostStageNames : List String :=\n  [ \"evaluations.inputs\", \"evaluations.output\", \"evaluations.quotient\", \"final_limb_checks\" ]\n",
    );
    writeln!(
        rendered,
        "def compactRetainedFieldsByStage : List Nat := {}",
        lean_nat_list(compact_retained_fields_by_stage)
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def compactSyntheticFieldsByStage : List Nat := {}",
        lean_nat_list(compact_synthetic_fields_by_stage)
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def compactProductSumRowsByStage : List Nat := {}",
        lean_nat_list(compact_product_sum_rows_by_stage)
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def compactOrdinaryCoordinatesByStage : List Nat := {}",
        lean_nat_list(compact_ordinary_coordinates_by_stage)
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def compactSyntheticCoordinatesByStage : List Nat := {}",
        lean_nat_list(compact_synthetic_coordinates_by_stage)
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def compactOrdinaryCenteredRowsByStage : List Nat := {}",
        lean_nat_list(compact_ordinary_centered_rows_by_stage)
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def compactSyntheticBooleanRowsByStage : List Nat := {}",
        lean_nat_list(compact_synthetic_boolean_rows_by_stage)
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def compactSyntheticCanonicalityRowsByStage : List Nat := {}",
        lean_nat_list(compact_synthetic_canonicality_rows_by_stage)
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def compactEncodedColumnsByStage : List Nat := {}",
        lean_nat_list(compact_encoded_columns_by_stage)
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def compactEncodedRowsByStage : List Nat := {}",
        lean_nat_list(compact_encoded_rows_by_stage)
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def compactEvaluationFields : Nat := {compact_evaluation_fields}"
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def compactEvaluationProductRows : Nat := {compact_evaluation_product_rows}"
    )
    .expect("write certificate");
    writeln!(rendered, "def compactEvaluationRows : Nat := {compact_evaluation_rows}").expect("write certificate");
    writeln!(
        rendered,
        "def compactEvaluationColumns : Nat := {compact_evaluation_columns}"
    )
    .expect("write certificate");
    writeln!(rendered, "def compactFinalFields : Nat := {compact_final_fields}").expect("write certificate");
    writeln!(
        rendered,
        "def compactFinalProductRows : Nat := {compact_final_product_rows}"
    )
    .expect("write certificate");
    writeln!(rendered, "def compactFinalRows : Nat := {compact_final_rows}").expect("write certificate");
    writeln!(rendered, "def compactFinalColumns : Nat := {compact_final_columns}").expect("write certificate");
    writeln!(rendered, "def compactRetainedFields : Nat := {compact_retained_fields}").expect("write certificate");
    writeln!(
        rendered,
        "def compactSyntheticFields : Nat := {compact_synthetic_fields}"
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def compactProductSumRows : Nat := {compact_product_sum_rows}"
    )
    .expect("write certificate");
    writeln!(rendered, "def compactEncodedRows : Nat := {compact_encoded_rows}").expect("write certificate");
    writeln!(rendered, "def compactEncodedColumns : Nat := {compact_encoded_columns}").expect("write certificate");
    writeln!(
        rendered,
        "def compactAllIdentityRows : Nat := {}",
        compact_encoded_rows * roles.len()
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def compactAllIdentityColumns : Nat := {}",
        compact_encoded_columns * roles.len()
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def representativeSourceSchemaSha256 : String := \"{}\"",
        sha256_hex(format!("{schema:?}").as_bytes())
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def compactPlanSha256 : String := \"{}\"",
        sha256_hex(format!("{compact:?}").as_bytes())
    )
    .expect("write certificate");
    writeln!(
        rendered,
        "def completeCertificateSha256 : String := \"{}\"\n",
        sha256_hex(format!("{schema:?}\n{compaction:?}").as_bytes())
    )
    .expect("write certificate");
    rendered.push_str("def roles : List IdentityRole :=\n");
    for (index, &role) in roles.iter().enumerate() {
        let prefix = if index == 0 { "  [ " } else { "  , " };
        writeln!(rendered, "{prefix}{}", lean_projection_role(role)).expect("write certificate");
    }
    rendered.push_str("  ]\n\n");
    writeln!(
        rendered,
        "def retainedColumnOffsets : List Nat := {}\n",
        lean_nat_list(compact.retained_column_offsets.iter().copied())
    )
    .expect("write certificate");
    rendered.push_str("def evaluationPlans : List EvaluationPlan :=\n");
    for (index, evaluation) in compact.evaluations.iter().enumerate() {
        let prefix = if index == 0 { "  [ " } else { "  , " };
        writeln!(rendered, "{prefix}{{ kind := {}", lean_evaluation_kind(evaluation.kind)).expect("write certificate");
        writeln!(rendered, "      sourceRowOffset := {}", evaluation.source_row_offset).expect("write certificate");
        writeln!(rendered, "      sourceRowCount := {}", evaluation.source_row_count).expect("write certificate");
        writeln!(rendered, "      coefficientCount := {}", evaluation.coefficient_count).expect("write certificate");
        writeln!(
            rendered,
            "      retainedOrdinals := {}",
            lean_nat_list(evaluation.retained_ordinals)
        )
        .expect("write certificate");
        writeln!(
            rendered,
            "      retainedColumnOffsets := {}",
            lean_nat_list(evaluation.retained_column_offsets)
        )
        .expect("write certificate");
        writeln!(
            rendered,
            "      coefficientZero := [{}]",
            evaluation
                .coefficient_zero
                .map(lean_coefficient_zero)
                .join(", ")
        )
        .expect("write certificate");
        writeln!(
            rendered,
            "      productCounts := {}",
            lean_nat_list(evaluation.product_counts)
        )
        .expect("write certificate");
        writeln!(
            rendered,
            "      chunkSizes := {} }}",
            lean_nested_nat_lists(
                evaluation
                    .chunk_sizes
                    .iter()
                    .map(|chunks| chunks.iter().copied())
            )
        )
        .expect("write certificate");
    }
    rendered.push_str("  ]\n\n");
    rendered.push_str("def retainedBindings : List RetainedBinding :=\n");
    for (index, binding) in compact.retained_bindings.iter().enumerate() {
        let prefix = if index == 0 { "  [ " } else { "  , " };
        writeln!(
            rendered,
            "{prefix}{{ identity := {}, retainedOrdinal := {}, coefficient := {} }}",
            binding.identity,
            binding.retained_ordinal,
            lean_final_coefficient(binding.coefficient)
        )
        .expect("write certificate");
    }
    rendered.push_str("  ]\n\n");
    rendered.push_str("def finalLimbPlans : List FinalLimbPlan :=\n");
    for (index, limb) in compact.final_limbs.iter().enumerate() {
        let prefix = if index == 0 { "  [ " } else { "  , " };
        writeln!(rendered, "{prefix}{{ limb := {}", limb.limb).expect("write certificate");
        writeln!(rendered, "      sourceRowOffset := {}", limb.source_row_offset).expect("write certificate");
        writeln!(
            rendered,
            "      resultRetainedOrdinal := {}",
            limb.result_retained_ordinal
        )
        .expect("write certificate");
        writeln!(
            rendered,
            "      chunkSizes := {}",
            lean_nat_list(limb.chunk_sizes.iter().copied())
        )
        .expect("write certificate");
        rendered.push_str("      factors :=\n");
        for (factor_index, factor) in limb.factors.iter().copied().enumerate() {
            let factor_prefix = if factor_index == 0 { "        [ " } else { "        , " };
            writeln!(rendered, "{factor_prefix}{}", lean_final_factor(factor)).expect("write certificate");
        }
        rendered.push_str("        ] }\n");
    }
    rendered.push_str("  ]\n\n");
    rendered.push_str("end SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.ProjectionIdentityCertificateData\n");
    rendered
}

#[test]
fn projection_identity_certificate_matches_production_trace() {
    let builder = build_recursive_program();
    let source = builder.snapshot();
    let trace = builder.encoding_trace();
    validate_projection_identity_traces(&source, trace).expect("exact production projection trace");

    let rendered = render_projection_identity_certificate(&source, trace);
    let path = repo_root().join(PROJECTION_CERTIFICATE_PATH);
    let committed = fs::read_to_string(&path).unwrap_or_default();
    if committed != rendered {
        let expected = path.with_extension("lean.expected");
        fs::create_dir_all(expected.parent().expect("certificate parent"))
            .expect("create certificate expected directory");
        fs::write(&expected, &rendered).expect("write expected projection identity certificate");
    }
    assert_eq!(
        committed, rendered,
        "projection identity certificate drifted; review the generated .expected file"
    );
}
