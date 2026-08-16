//! Exact tests for schedule selectors over shared lifecycle and phase-kind rows.

use std::fmt::Write as _;

use neo_ccs::CcsStructure;
use neo_fold_clean::engine::r1cs_circuit::boolean::enforce_bit;
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder};
use neo_fold_clean::frontends::r1cs_f_prime::{
    build_multi_branch_selective_low_norm_r1cs_with_alignment, build_scheduled_grouped_phase_low_norm_r1cs,
    build_scheduled_grouped_phase_low_norm_r1cs_with_field_links, lower_field_r1cs, ScheduledCommonPhaseFieldLink,
    ScheduledCursorBits, ScheduledGroupedPhaseError, ScheduledGroupedPhaseLowNormR1cs, ScheduledPhaseKindLinks,
    SparseR1cs,
};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

const PORT_COUNT: usize = 13;
const GENERAL_SELECTOR_PORT: usize = 1;
const A_PORT: usize = 2;
const B_PORT: usize = 3;
const C_PORT: usize = 4;
const SCHEMA_VERSION: usize = 1;
const ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryScheduledGroupedPhaseFixture.lean";

fn bit_arm(public: [F; 4], private: F) -> (SparseR1cs, Vec<F>) {
    let mut builder = R1csBuilder::new();
    let public_vars = public.map(|value| builder.alloc(value));
    let private_var = builder.alloc(private);
    for var in public_vars {
        enforce_bit(&mut builder, var);
    }
    enforce_bit(&mut builder, private_var);
    builder.enforce(
        &Lc::from_var(public_vars[0]),
        &Lc::from_var(private_var),
        &Lc::from_var(private_var),
    );
    assert!(builder.is_satisfied());
    lower_field_r1cs(builder, &public_vars)
        .expect("lower scheduled fixture")
        .into_parts()
}

fn transition_bits(arm: usize) -> [F; 4] {
    let before = arm;
    let after = arm + 1;
    [
        F::from_usize(before & 1),
        F::from_usize((before >> 1) & 1),
        F::from_usize(after & 1),
        F::from_usize((after >> 1) & 1),
    ]
}

fn affine_defined_arm(arm: usize, x_value: F) -> (SparseR1cs, Vec<F>) {
    let mut builder = R1csBuilder::new();
    let public_vars = transition_bits(arm).map(|value| builder.alloc(value));
    for var in public_vars {
        enforce_bit(&mut builder, var);
    }
    let x = builder.alloc(x_value);
    enforce_bit(&mut builder, x);
    let y = builder.alloc(F::ONE - x_value);
    let mut definition = Lc::from_var(y);
    definition.add_term(x, F::ONE);
    definition.add_constant(-F::ONE);
    builder.enforce_zero(&definition);
    assert!(builder.is_satisfied());
    lower_field_r1cs(builder, &public_vars)
        .expect("lower affine-definition fixture")
        .into_parts()
}

struct Fixture {
    relation: ScheduledGroupedPhaseLowNormR1cs,
    assignments: [Vec<F>; 3],
    mismatched: Vec<F>,
}

fn fixture() -> Fixture {
    let (shape, assignment0) = bit_arm(transition_bits(0), F::ZERO);
    let (_, assignment1) = bit_arm(transition_bits(1), F::ONE);
    let (_, assignment2) = bit_arm(transition_bits(2), F::ZERO);
    let (_, mismatched) = bit_arm([F::ONE, F::ONE, F::ONE, F::ONE], F::ZERO);
    let common = build_multi_branch_selective_low_norm_r1cs_with_alignment(&[shape.clone(), shape.clone()], 0, D, 0)
        .expect("compile lifecycle groups");
    let phase_kinds = build_multi_branch_selective_low_norm_r1cs_with_alignment(&[shape.clone(), shape], 0, D, 0)
        .expect("compile phase kinds");
    let relation = build_scheduled_grouped_phase_low_norm_r1cs(
        common,
        phase_kinds,
        vec![0, 1, 1],
        vec![0, 1, 0],
        ScheduledCursorBits::new(1..3, 3..5),
    )
    .expect("compose exact schedule");
    Fixture {
        relation,
        assignments: [assignment0, assignment1, assignment2],
        mismatched,
    }
}

fn assert_terms(structure: &CcsStructure<F>, row: usize, port: usize, mut expected: Vec<(usize, F)>) {
    expected.sort_unstable_by_key(|&(column, _)| column);
    assert_eq!(
        structure.matrices[port]
            .materialize_row(row)
            .expect("row in bounds"),
        expected,
        "row {row}, port {port}",
    );
}

fn assert_linear_zero_row(structure: &CcsStructure<F>, row: usize, expected_c: Vec<(usize, F)>) {
    for port in 0..PORT_COUNT {
        let expected = match port {
            GENERAL_SELECTOR_PORT => vec![(0, F::ONE)],
            C_PORT => expected_c.clone(),
            _ => Vec::new(),
        };
        assert_terms(structure, row, port, expected);
    }
}

fn assert_activation_row(structure: &CcsStructure<F>, row: usize, schedule_selector: usize, group_selector: usize) {
    for port in 0..PORT_COUNT {
        let expected = match port {
            GENERAL_SELECTOR_PORT => vec![(0, F::ONE)],
            A_PORT => vec![(schedule_selector, F::ONE)],
            B_PORT => vec![(group_selector, F::ONE)],
            C_PORT => vec![(schedule_selector, F::ONE)],
            _ => Vec::new(),
        };
        assert_terms(structure, row, port, expected);
    }
}

fn assert_cursor_row(
    structure: &CcsStructure<F>,
    row: usize,
    schedule_selector: usize,
    bit_start: usize,
    expected_value: usize,
) {
    let mut expected_b = vec![(bit_start, F::ONE), (bit_start + 1, F::from_u64(2))];
    if expected_value != 0 {
        expected_b.push((0, -F::from_usize(expected_value)));
    }
    for port in 0..PORT_COUNT {
        let expected = match port {
            GENERAL_SELECTOR_PORT => vec![(0, F::ONE)],
            A_PORT => vec![(schedule_selector, F::ONE)],
            B_PORT => expected_b.clone(),
            _ => Vec::new(),
        };
        assert_terms(structure, row, port, expected);
    }
}

fn lean_list(values: &[usize]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn render_artifact(fixture: &Fixture) -> String {
    let relation = &fixture.relation;
    let layout = relation.layout();
    let mut rendered = String::new();
    rendered.push_str(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.ScheduledGroupedPhaseFixtureSchema\n\n\
/-! Generated file: exact schedule-over-grouped-phase composition fixture.\n\n\
Owns the Rust-emitted row ranges, selector columns, schedule maps, cursor-bit\n\
ranges, and selective port indices used by the exhaustive matrix test.\n\n\
Does not own component semantics, the production 400-arm schedule, or the\n\
complete recursive and terminal F-prime relations. Lean recomputes each row.\n\n\
Emits constraints: this fixture's schedule total, group equality, activation,\n\
and exact cursor rows.\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryScheduledGroupedPhaseFixture\n\n\
open Nightstream.Implementation.R1CS.FPrimeFullHistoryScheduledGroupedPhaseFixture.Artifact\n\n\
def rawArtifact : RawArtifact where\n",
    );
    writeln!(rendered, "  schemaVersion := {SCHEMA_VERSION}").unwrap();
    writeln!(rendered, "  rows := {}", relation.structure().n).unwrap();
    writeln!(rendered, "  columns := {}", relation.structure().m).unwrap();
    writeln!(rendered, "  publicColumns := {}", relation.public_input_len()).unwrap();
    writeln!(rendered, "  commonRowEnd := {}", layout.common_rows().end).unwrap();
    writeln!(rendered, "  phaseRowEnd := {}", layout.phase_rows().end).unwrap();
    writeln!(
        rendered,
        "  scheduleTotalRowEnd := {}",
        layout.schedule_total_rows().end
    )
    .unwrap();
    writeln!(
        rendered,
        "  lifecycleEqualityRowEnd := {}",
        layout.lifecycle_equality_rows().end
    )
    .unwrap();
    writeln!(
        rendered,
        "  phaseKindEqualityRowEnd := {}",
        layout.phase_kind_equality_rows().end
    )
    .unwrap();
    writeln!(
        rendered,
        "  lifecycleActivationRowEnd := {}",
        layout.lifecycle_activation_rows().end
    )
    .unwrap();
    writeln!(
        rendered,
        "  phaseKindActivationRowEnd := {}",
        layout.phase_kind_activation_rows().end
    )
    .unwrap();
    writeln!(
        rendered,
        "  cursorBindingRowEnd := {}",
        layout.cursor_binding_rows().end
    )
    .unwrap();
    writeln!(rendered, "  portCount := {PORT_COUNT}").unwrap();
    writeln!(rendered, "  generalSelectorPort := {GENERAL_SELECTOR_PORT}").unwrap();
    writeln!(rendered, "  aPort := {A_PORT}").unwrap();
    writeln!(rendered, "  bPort := {B_PORT}").unwrap();
    writeln!(rendered, "  cPort := {C_PORT}").unwrap();
    writeln!(
        rendered,
        "  commonSelectorColumns := {}",
        lean_list(layout.common_selector_columns())
    )
    .unwrap();
    writeln!(
        rendered,
        "  phaseKindSelectorColumns := {}",
        lean_list(layout.phase_kind_selector_columns())
    )
    .unwrap();
    writeln!(
        rendered,
        "  scheduleSelectorColumns := {}",
        lean_list(layout.schedule_selector_columns())
    )
    .unwrap();
    writeln!(
        rendered,
        "  lifecycleGroups := {}",
        lean_list(layout.lifecycle_groups())
    )
    .unwrap();
    writeln!(rendered, "  phaseKinds := {}", lean_list(layout.phase_kinds())).unwrap();
    writeln!(
        rendered,
        "  beforeCursorStart := {}",
        layout.cursor_bits().before().start
    )
    .unwrap();
    writeln!(rendered, "  beforeCursorEnd := {}", layout.cursor_bits().before().end).unwrap();
    writeln!(rendered, "  afterCursorStart := {}", layout.cursor_bits().after().start).unwrap();
    writeln!(rendered, "  afterCursorEnd := {}", layout.cursor_bits().after().end).unwrap();
    rendered.push_str(
        "\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryScheduledGroupedPhaseFixture\n",
    );
    rendered
}

fn artifact_path() -> String {
    format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_PATH)
}

#[test]
fn scheduled_composition_stores_group_rows_once_and_binds_each_cursor() {
    let fixture = fixture();
    let relation = &fixture.relation;
    let layout = relation.layout();
    let structure = relation.structure();

    assert_eq!(layout.lifecycle_groups(), &[0, 1, 1]);
    assert_eq!(layout.phase_kinds(), &[0, 1, 0]);
    assert_eq!(layout.common_rows().len(), relation.common_relation().structure().n);
    assert_eq!(layout.phase_rows().len(), relation.phase_kind_relation().structure().n);
    assert_eq!(layout.schedule_total_rows().len(), 1);
    assert_eq!(layout.lifecycle_equality_rows().len(), 2);
    assert_eq!(layout.phase_kind_equality_rows().len(), 2);
    assert_eq!(layout.lifecycle_activation_rows().len(), 3);
    assert_eq!(layout.phase_kind_activation_rows().len(), 3);
    assert_eq!(layout.cursor_binding_rows().len(), 6);

    let schedule = layout.schedule_selector_columns();
    assert_linear_zero_row(
        structure,
        layout.schedule_total_rows().start,
        vec![
            (0, F::ONE),
            (schedule[0], -F::ONE),
            (schedule[1], -F::ONE),
            (schedule[2], -F::ONE),
        ],
    );
    for group in 0..2 {
        let mut terms = vec![(layout.common_selector_columns()[group], F::ONE)];
        terms.extend(
            layout
                .lifecycle_groups()
                .iter()
                .enumerate()
                .filter(|(_, arm_group)| **arm_group == group)
                .map(|(arm, _)| (schedule[arm], -F::ONE)),
        );
        assert_linear_zero_row(structure, layout.lifecycle_equality_rows().start + group, terms);
    }
    for kind in 0..2 {
        let mut terms = vec![(layout.phase_kind_selector_columns()[kind], F::ONE)];
        terms.extend(
            layout
                .phase_kinds()
                .iter()
                .enumerate()
                .filter(|(_, arm_kind)| **arm_kind == kind)
                .map(|(arm, _)| (schedule[arm], -F::ONE)),
        );
        assert_linear_zero_row(structure, layout.phase_kind_equality_rows().start + kind, terms);
    }
    for arm in 0..3 {
        assert_activation_row(
            structure,
            layout.lifecycle_activation_rows().start + arm,
            schedule[arm],
            layout.common_selector_columns()[layout.lifecycle_groups()[arm]],
        );
        assert_activation_row(
            structure,
            layout.phase_kind_activation_rows().start + arm,
            schedule[arm],
            layout.phase_kind_selector_columns()[layout.phase_kinds()[arm]],
        );
        assert_cursor_row(
            structure,
            layout.cursor_binding_rows().start + 2 * arm,
            schedule[arm],
            1,
            arm,
        );
        assert_cursor_row(
            structure,
            layout.cursor_binding_rows().start + 2 * arm + 1,
            schedule[arm],
            3,
            arm + 1,
        );

        let assignment = relation
            .encode(arm, &fixture.assignments[arm], &fixture.assignments[arm])
            .expect("encode honest scheduled arm");
        assert!(relation.is_satisfied(&assignment), "schedule arm {arm}");
    }
}

#[test]
fn scheduled_composition_rejects_wrong_schedule_selector_or_cursor() {
    let fixture = fixture();
    let relation = &fixture.relation;
    let layout = relation.layout();
    let mut assignment = relation
        .encode(1, &fixture.assignments[1], &fixture.assignments[1])
        .expect("encode schedule arm one");
    assignment[layout.schedule_selector_columns()[1]] = F::ZERO;
    assignment[layout.schedule_selector_columns()[2]] = F::ONE;
    assert!(!relation.is_satisfied(&assignment));

    let mut assignment = relation
        .encode(1, &fixture.assignments[1], &fixture.assignments[1])
        .expect("encode schedule arm one");
    assignment[layout.cursor_bits().before().start] = F::ZERO;
    assert!(!relation.is_satisfied(&assignment));
}

#[test]
fn scheduled_composition_rejects_shared_public_mismatch() {
    let fixture = fixture();
    let error = fixture
        .relation
        .encode(0, &fixture.assignments[0], &fixture.mismatched)
        .expect_err("shared public mismatch must fail");
    assert!(matches!(
        error,
        ScheduledGroupedPhaseError::Grouped(
            neo_fold_clean::frontends::r1cs_f_prime::GroupedPhaseError::PublicAssignmentMismatch { .. }
        )
    ));
}

#[test]
fn scheduled_private_link_reconstructs_and_binds_selected_source_fields() {
    let (shape, assignment0) = bit_arm(transition_bits(0), F::ZERO);
    let (_, assignment1) = bit_arm(transition_bits(1), F::ONE);
    let (_, assignment1_wrong_private) = bit_arm(transition_bits(1), F::ZERO);
    let (_, assignment2) = bit_arm(transition_bits(2), F::ZERO);
    let common = build_multi_branch_selective_low_norm_r1cs_with_alignment(&[shape.clone(), shape.clone()], 0, D, 0)
        .expect("compile linked lifecycle groups");
    let phases = build_multi_branch_selective_low_norm_r1cs_with_alignment(&[shape.clone(), shape], 0, D, 0)
        .expect("compile linked phase kinds");
    let relation = build_scheduled_grouped_phase_low_norm_r1cs_with_field_links(
        common,
        phases,
        vec![0, 1, 1],
        vec![0, 1, 0],
        ScheduledCursorBits::new(1..3, 3..5),
        vec![ScheduledPhaseKindLinks {
            lifecycle_group: 1,
            phase_kind: 1,
            fields: vec![ScheduledCommonPhaseFieldLink {
                common_field: 5,
                phase_field: 5,
            }],
        }],
    )
    .expect("compose linked schedule");

    let layout = relation.layout();
    assert_eq!(layout.common_phase_link_rows().len(), 1);
    assert_eq!(
        layout.common_phase_link_rows_for_kind(0),
        Some(layout.common_phase_link_rows().start..layout.common_phase_link_rows().start)
    );
    assert_eq!(
        layout.common_phase_link_rows_for_kind(1),
        Some(layout.common_phase_link_rows())
    );
    let common_slot = relation
        .common_relation()
        .field_slot(1, 5)
        .expect("common private bit slot");
    let phase_slot = relation
        .phase_kind_relation()
        .field_slot(1, 5)
        .expect("phase private bit slot");
    assert_eq!(common_slot.1, 1);
    assert_eq!(phase_slot.1, 1);
    let phase_column = layout.phase_private_columns().start + phase_slot.0 - relation.public_input_len();
    for port in 0..PORT_COUNT {
        let expected = match port {
            GENERAL_SELECTOR_PORT => vec![(0, F::ONE)],
            A_PORT => vec![(layout.phase_kind_selector_columns()[1], F::ONE)],
            B_PORT => vec![(common_slot.0, F::ONE), (phase_column, -F::ONE)],
            _ => Vec::new(),
        };
        assert_terms(
            relation.structure(),
            layout.common_phase_link_rows().start,
            port,
            expected,
        );
    }

    let honest = relation
        .encode(1, &assignment1, &assignment1)
        .expect("encode matching linked fields");
    assert!(relation.is_satisfied(&honest));
    let wrong = relation
        .encode(1, &assignment1, &assignment1_wrong_private)
        .expect("encode mismatching linked fields");
    assert!(!relation.is_satisfied(&wrong));
    let unlinked = relation
        .encode(0, &assignment0, &assignment0)
        .expect("encode unlinked phase kind");
    assert!(relation.is_satisfied(&unlinked));
    let later = relation
        .encode(2, &assignment2, &assignment2)
        .expect("encode later unlinked phase kind");
    assert!(relation.is_satisfied(&later));
}

#[test]
fn scheduled_private_link_reconstructs_compiler_eliminated_affine_fields() {
    let (shape, assignment0) = affine_defined_arm(0, F::ZERO);
    let (_, assignment1) = affine_defined_arm(1, F::ZERO);
    let (_, assignment1_wrong) = affine_defined_arm(1, F::ONE);
    let common = build_multi_branch_selective_low_norm_r1cs_with_alignment(&[shape.clone(), shape.clone()], 0, D, 0)
        .expect("compile affine lifecycle groups");
    let phases = build_multi_branch_selective_low_norm_r1cs_with_alignment(&[shape.clone(), shape], 0, D, 0)
        .expect("compile affine phase kinds");
    assert_eq!(common.field_slot(1, 6), None, "field y must use its affine definition");
    assert!(common
        .selective_compiler_audit()
        .expect("compiler audit")
        .source_arm_linear_definition(1, 6)
        .is_some());

    let relation = build_scheduled_grouped_phase_low_norm_r1cs_with_field_links(
        common,
        phases,
        vec![0, 1],
        vec![0, 1],
        ScheduledCursorBits::new(1..3, 3..5),
        vec![ScheduledPhaseKindLinks {
            lifecycle_group: 1,
            phase_kind: 1,
            fields: vec![ScheduledCommonPhaseFieldLink {
                common_field: 6,
                phase_field: 6,
            }],
        }],
    )
    .expect("compose affine field link");

    let honest = relation
        .encode(1, &assignment1, &assignment1)
        .expect("encode matching affine fields");
    assert!(relation.is_satisfied(&honest));

    let changed = relation
        .encode(1, &assignment1, &assignment1_wrong)
        .expect("encode mismatched affine fields");
    assert!(!relation.is_satisfied(&changed));

    let base = relation
        .encode(0, &assignment0, &assignment0)
        .expect("encode base affine fields");
    assert!(relation.is_satisfied(&base));
}

#[test]
fn scheduled_private_link_rejects_a_phase_kind_used_by_two_lifecycle_groups() {
    let (shape, _) = bit_arm(transition_bits(0), F::ZERO);
    let common = build_multi_branch_selective_low_norm_r1cs_with_alignment(&[shape.clone(), shape.clone()], 0, D, 0)
        .expect("compile lifecycle groups");
    let phases = build_multi_branch_selective_low_norm_r1cs_with_alignment(&[shape.clone(), shape], 0, D, 0)
        .expect("compile phase kinds");
    let error = build_scheduled_grouped_phase_low_norm_r1cs_with_field_links(
        common,
        phases,
        vec![0, 1, 1],
        vec![0, 1, 0],
        ScheduledCursorBits::new(1..3, 3..5),
        vec![ScheduledPhaseKindLinks {
            lifecycle_group: 0,
            phase_kind: 0,
            fields: vec![ScheduledCommonPhaseFieldLink {
                common_field: 5,
                phase_field: 5,
            }],
        }],
    )
    .expect_err("linked phase kind must have one lifecycle group");
    assert!(matches!(
        error,
        ScheduledGroupedPhaseError::LinkLifecycleGroupMismatch {
            arm: 2,
            phase_kind: 0,
            actual_group: 1,
            expected_group: 0,
        }
    ));
}

#[test]
fn scheduled_grouped_phase_generated_artifact_matches_emitted_recipe() {
    let fixture = fixture();
    let rendered = render_artifact(&fixture);
    let path = artifact_path();
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if committed != rendered {
        let expected = format!("{path}.expected");
        std::fs::write(&expected, rendered).expect("write scheduled grouped-phase artifact candidate");
        panic!("scheduled grouped-phase artifact drifted; inspect {expected} and promote it explicitly");
    }
}

#[test]
#[ignore = "deliberately writes the reviewed generated Lean artifact"]
fn regenerate_scheduled_grouped_phase_artifact() {
    std::fs::write(artifact_path(), render_artifact(&fixture()))
        .expect("write generated scheduled grouped-phase artifact");
}
