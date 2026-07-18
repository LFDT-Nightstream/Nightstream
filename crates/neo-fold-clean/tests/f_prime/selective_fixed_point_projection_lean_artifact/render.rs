//! Typed metadata renderer for the bounded tiny-fixture projection certificate.
//!
//! Owns: lossless rendering of the retained active scope, polynomial and
//! Karatsuba trace coordinates, and producer-side serializer field indices as
//! one instance of the handwritten Lean artifact schema.
//!
//! Does not own: the exact sparse rows, serializer-index semantics, producer
//! to consumer equality, source authority, trace soundness, or row removal.
//!
//! Emits constraints: no.
//!
//! | Artifact branch | Exported evidence | Independent Lean consumer |
//! |---|---|---|
//! | `scope` | exact test parameters, app shape, arm shape, and 15/13/23033 profile | `Scope.IsTinyFixture` |
//! | `shared` / `limbs` | exact row and column coordinates | `Artifact.StructureValid` |
//! | `producers` | raw `(field index, source column)` vectors | serializer-index correspondence |

use std::fmt::Write as _;
use std::ops::Range;

use neo_fold_clean::frontends::r1cs_f_prime::ivc::{
    PiCcsOutputYZcolProducerEntryAudit, PiCcsOutputYZcolProjectionAudit, PiRlcYZcolKMulAudit,
    PiRlcYZcolLinearCombinationAudit, PiRlcYZcolPolynomialEvaluationAudit,
};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{rows, GeneratedLeanFile, TinyFixtureScope};

const GENERATED_ROOT: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeSelectiveFixedPoint/PiRlcProjection/YZcol/Generated";
const IMPORT_ROOT: &str = "Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol";
const NAMESPACE: &str =
    "Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Generated.Metadata";

fn lean_row_block(rows: Range<usize>) -> String {
    format!("{{ start := {}, stop := {} }}", rows.start, rows.end)
}

fn lean_k_columns(columns: [usize; 2]) -> String {
    format!("{{ c0 := {}, c1 := {} }}", columns[0], columns[1])
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

fn lc_terms(lc: &PiRlcYZcolLinearCombinationAudit) -> Vec<(usize, F)> {
    let mut terms = lc.terms().to_vec();
    if lc.constant() != F::ZERO {
        terms.push((0, lc.constant()));
    }
    terms
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

fn lean_k_terms(c0: &PiRlcYZcolLinearCombinationAudit, c1: &PiRlcYZcolLinearCombinationAudit) -> String {
    format!(
        "{{ c0 := {}, c1 := {} }}",
        lean_terms(&lc_terms(c0)),
        lean_terms(&lc_terms(c1))
    )
}

fn lean_evaluation(owner: &PiRlcYZcolPolynomialEvaluationAudit) -> String {
    let coefficient_count = owner.coefficient_columns().len();
    assert_ne!(coefficient_count, 0, "polynomial certificate must be nonempty");
    assert_eq!(owner.power_columns().len(), coefficient_count, "power width");
    assert_eq!(
        owner.allocated_columns().len(),
        2 * coefficient_count,
        "evaluation allocation width"
    );
    assert_eq!(owner.rows().len(), 2 * coefficient_count, "evaluation row width");
    assert_eq!(
        owner.output_columns(),
        [
            owner.allocated_columns()[2 * coefficient_count - 2],
            owner.allocated_columns()[2 * coefficient_count - 1]
        ],
        "evaluation output coordinates"
    );
    let products = owner.allocated_columns()[..2 * (coefficient_count - 1)]
        .chunks_exact(2)
        .map(|pair| [pair[0], pair[1]])
        .collect::<Vec<_>>();
    format!(
        "{{ rows := {}, coefficients := {}, powers := {}, products := {}, output := {} }}",
        lean_row_block(owner.rows()),
        lean_nat_list(owner.coefficient_columns().iter().copied()),
        lean_k_columns_list(owner.power_columns().iter().copied()),
        lean_k_columns_list(products),
        lean_k_columns(owner.output_columns())
    )
}

fn lean_k_product(owner: &PiRlcYZcolKMulAudit) -> String {
    let [first, second] = owner.identities() else {
        panic!("K-product certificate must retain two extension-limb identities")
    };
    let [c0c0, c1c1] = first.factors() else {
        panic!("K-product c0 identity must retain two factors")
    };
    let [c0c1, c1c0] = second.factors() else {
        panic!("K-product c1 identity must retain two factors")
    };
    assert_eq!(lc_terms(c0c0.left()), lc_terms(c0c1.left()), "shared left c0");
    assert_eq!(lc_terms(c1c1.left()), lc_terms(c1c0.left()), "shared left c1");
    assert_eq!(lc_terms(c0c0.right()), lc_terms(c1c0.right()), "shared right c0");
    assert_eq!(lc_terms(c1c1.right()), lc_terms(c0c1.right()), "shared right c1");
    assert_eq!(owner.rows().len(), 5, "K-product row width");
    assert_eq!(owner.allocated_columns().len(), 5, "K-product allocation width");
    assert_eq!(
        owner.retained_columns(),
        owner.output_columns(),
        "retained K-product output"
    );

    let left_c0 = lc_terms(c0c0.left());
    let left_c1 = lc_terms(c1c1.left());
    let right_c0 = lc_terms(c0c0.right());
    let right_c1 = lc_terms(c1c1.right());
    let mut sum_left = left_c0.clone();
    sum_left.extend(left_c1.iter().copied());
    let mut sum_right = right_c0.clone();
    sum_right.extend(right_c1.iter().copied());
    let [product_c0, product_c1, product_sum] = owner.intermediate_columns();

    format!(
        "{{ rows := {}, left := {}, right := {}, sumLeft := {}, sumRight := {}, productC0 := {product_c0}, productC1 := {product_c1}, productSum := {product_sum}, output := {} }}",
        lean_row_block(owner.rows()),
        lean_k_terms(c0c0.left(), c1c1.left()),
        lean_k_terms(c0c0.right(), c1c1.right()),
        lean_terms(&sum_left),
        lean_terms(&sum_right),
        lean_k_columns(owner.output_columns())
    )
}

fn lean_evaluation_list<'a>(values: impl IntoIterator<Item = &'a PiRlcYZcolPolynomialEvaluationAudit>) -> String {
    format!(
        "[{}]",
        values
            .into_iter()
            .map(lean_evaluation)
            .collect::<Vec<_>>()
            .join(",\n       ")
    )
}

fn lean_k_product_list<'a>(values: impl IntoIterator<Item = &'a PiRlcYZcolKMulAudit>) -> String {
    format!(
        "[{}]",
        values
            .into_iter()
            .map(lean_k_product)
            .collect::<Vec<_>>()
            .join(",\n       ")
    )
}

fn lean_producer_entries(entries: &[PiCcsOutputYZcolProducerEntryAudit]) -> String {
    format!(
        "[{}]",
        entries
            .iter()
            .map(|entry| format!(
                "{{ serializerFieldIndex := {}, sourceColumn := {} }}",
                entry.field_index(),
                entry.column()
            ))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn render_shared(contents: &mut String, projection: &PiCcsOutputYZcolProjectionAudit) {
    let shared = projection.identity().shared();
    contents.push_str("set_option maxRecDepth 100000 in\n");
    writeln!(contents, "def shared : SharedOwner :=").expect("render shared owner");
    writeln!(
        contents,
        "  {{ betaLadderRows := {},",
        lean_row_block(shared.beta_ladder_rows())
    )
    .expect("render beta rows");
    writeln!(contents, "    beta := {},", lean_k_columns(shared.beta_columns())).expect("render beta columns");
    writeln!(
        contents,
        "    powers := {},",
        lean_k_columns_list(shared.power_columns().iter().copied())
    )
    .expect("render powers");
    writeln!(
        contents,
        "    ladderProducts := {},",
        lean_k_product_list(shared.beta_products())
    )
    .expect("render ladder products");
    writeln!(
        contents,
        "    rhoEvaluations := {} }}\n",
        lean_evaluation_list(shared.rho_evaluations())
    )
    .expect("render rho evaluations");
}

fn render_limb(contents: &mut String, projection: &PiCcsOutputYZcolProjectionAudit, limb: usize) {
    let owner = projection.identity().limb(limb);
    assert_eq!(
        owner.input_evaluations().len(),
        owner.rho_products().len(),
        "pair count"
    );
    contents.push_str("set_option maxRecDepth 100000 in\n");
    writeln!(contents, "def limb{limb} : LimbOwner :=").expect("render limb owner");
    writeln!(contents, "  {{ limb := {limb},").expect("render limb index");
    contents.push_str("    pairs :=\n      [");
    for source in 0..owner.input_evaluations().len() {
        if source != 0 {
            contents.push_str("       ,");
        }
        writeln!(
            contents,
            "{{ sourceIndex := {source}, inputEvaluation := {}, rhoProduct := {} }}",
            lean_evaluation(&owner.input_evaluations()[source]),
            lean_k_product(&owner.rho_products()[source])
        )
        .expect("render limb pair");
    }
    contents.push_str("      ],\n");
    writeln!(
        contents,
        "    parentEvaluation := {},",
        lean_evaluation(owner.parent_evaluation())
    )
    .expect("render parent evaluation");
    writeln!(
        contents,
        "    quotientEvaluation := {},",
        lean_evaluation(owner.quotient_evaluation())
    )
    .expect("render quotient evaluation");
    writeln!(
        contents,
        "    quotientPhiProduct := {},",
        lean_k_product(owner.quotient_phi_product())
    )
    .expect("render quotient product");
    writeln!(contents, "    finalRows := {},", lean_row_block(owner.final_rows())).expect("render final rows");
    writeln!(contents, "    maxDegree := 106 }}\n").expect("render max degree");
}

fn render_producers(contents: &mut String, projection: &PiCcsOutputYZcolProjectionAudit) {
    contents.push_str("set_option maxRecDepth 100000 in\n");
    contents.push_str("def producers : List ProducerVector :=\n  [");
    let mut first = true;
    for limb in 0..2 {
        for input in projection.limb(limb) {
            if !first {
                contents.push_str("   ,");
            }
            first = false;
            assert_eq!(input.limb(), limb, "producer limb order");
            writeln!(
                contents,
                "{{ sourceIndex := {}, limb := {limb}, entries := {} }}",
                input.source(),
                lean_producer_entries(input.producer_entries())
            )
            .expect("render producer vector");
        }
    }
    contents.push_str("  ]\n\n");
}

fn render_source_rows(contents: &mut String) {
    contents.push_str("set_option maxRecDepth 100000 in\n");
    contents.push_str("def sourceRows : List (Nat × Row) :=\n  ");
    writeln!(contents, "{}\n", rows::source_row_definitions().join(" ++\n  ")).expect("render source row census");
}

pub(super) fn metadata(projection: &PiCcsOutputYZcolProjectionAudit, fixture: TinyFixtureScope) -> GeneratedLeanFile {
    let profile = projection.profile();
    let identity = projection.identity();
    assert_eq!(profile.source_count(), 15, "active source count");
    assert_eq!(profile.matrix_count(), 13, "active matrix count");
    assert_eq!(profile.field_count(), 23_033, "active serializer width");
    assert_eq!(profile.lane_count(), 54, "active lane count");
    assert_eq!(profile.limb_count(), 2, "active limb count");
    assert_eq!(identity.source_rows().len(), 5_724, "exact retained row count");
    assert_eq!(identity.shared().power_columns().len(), 55, "power count");

    let mut contents = String::new();
    writeln!(contents, "import {IMPORT_ROOT}.Schema").expect("render schema import");
    for module in rows::generated_row_modules() {
        writeln!(contents, "import {IMPORT_ROOT}.Generated.Rows.{module}").expect("render row import");
    }
    contents.push_str("\n/-! Generated by `active_selective_fixed_point_projection_artifact_matches_retained_certificate`; do not hand-edit.\n\n");
    contents.push_str("Owns: exact bounded tiny-fixture scope, retained source-trace coordinates, raw producer field indices, and composition of all indexed source-row shards.\n\n");
    contents.push_str("Does not own: serializer semantics, producer-to-consumer equality, source authority, row satisfaction, selective lowering, final costs, or row removal.\n\n");
    contents.push_str("Emits constraints: no.\n\n");
    contents.push_str("| Branch | Mathematical obligation | Physical source |\n|---|---|---|\n");
    contents.push_str("| `shared` | beta ladder and 15 rho evaluations | 1,892 retained source-R1CS definitions |\n");
    contents.push_str("| `limbs` | 15 input/product pairs plus parent, quotient, Phi, and checks per limb | retained source-R1CS trace coordinates |\n");
    contents.push_str("| `producers` | 30 ordered 54-lane serializer vectors | primary SIS serializer map |\n");
    contents.push_str("| `sourceRows` | 1,892 shared + 3,832 `y_zcol` normalized A/B/C equations | 12 disjoint generator shards |\n-/\n\n");
    writeln!(contents, "namespace {NAMESPACE}\n").expect("render metadata namespace");
    writeln!(
        contents,
        "def scope : Scope := {{ parameterConstraintCount := {}, commitmentWidth := {}, securityBits := {}, applicationRowCount := {}, applicationColumnCount := {}, applicationPublicInputCount := {}, sourceCount := {}, matrixCount := {}, serializerFieldCount := {}, sourceArmRowCount := {}, sourceArmColumnCount := {}, laneCount := {}, powerCount := 55, quotientCount := 53, maxDegree := 106 }}\n",
        fixture.parameter_constraint_count,
        fixture.commitment_width,
        fixture.security_bits,
        fixture.application_row_count,
        fixture.application_column_count,
        fixture.application_public_input_count,
        profile.source_count(),
        profile.matrix_count(),
        profile.field_count(),
        profile.source_arm_row_count(),
        profile.source_arm_column_count(),
        profile.lane_count()
    )
    .expect("render active scope");
    writeln!(contents, "def retainedRowCount : Nat := {}", identity.row_count()).expect("render retained rows");
    writeln!(
        contents,
        "def retainedAllocatedColumnCount : Nat := {}\n",
        identity.allocated_column_count()
    )
    .expect("render retained columns");
    render_source_rows(&mut contents);
    render_shared(&mut contents, projection);
    render_limb(&mut contents, projection, 0);
    render_limb(&mut contents, projection, 1);
    render_producers(&mut contents, projection);
    contents.push_str("set_option maxRecDepth 100000 in\n");
    contents.push_str("def artifact : Artifact :=\n");
    contents.push_str("  { scope := scope, sourceRows := sourceRows, shared := shared,\n");
    contents.push_str("    limbs := [limb0, limb1], producers := producers }\n\n");
    writeln!(contents, "end {NAMESPACE}").expect("render metadata namespace end");

    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/Metadata.lean"),
        contents,
    }
}
