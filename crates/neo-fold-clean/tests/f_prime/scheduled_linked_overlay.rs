//! Exact tests for schedule-selected private overlays and field links.

use std::fmt::Write as _;

use neo_ccs::CcsStructure;
use neo_fold_clean::engine::r1cs_circuit::boolean::enforce_bit;
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder};
use neo_fold_clean::frontends::r1cs_f_prime::{
    build_multi_branch_selective_low_norm_r1cs_with_alignment, build_scheduled_linked_overlay_low_norm_r1cs,
    lower_field_r1cs, OverlayBaseFieldPin, OverlayFieldLink, OverlayKindLinks, ScheduledCursorBits, SparseR1cs,
};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

const GENERAL_SELECTOR: usize = 1;
const A: usize = 2;
const B: usize = 3;
const C: usize = 4;
const PORT_COUNT: usize = 13;
const SCHEMA_VERSION: usize = 1;
const PIN_KIND: usize = 1;
const PIN_PHASE_FIELD: usize = 5;
const PIN_VALUE: usize = 13;
const ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryScheduledLinkedOverlayFixture.lean";

fn scheduled_arm(before: usize, after: usize, value: u64) -> (SparseR1cs, Vec<F>) {
    let mut builder = R1csBuilder::new();
    let mut public = Vec::with_capacity(4);
    for word in [before, after] {
        for bit in 0..2 {
            let wire = builder.alloc(F::from_usize((word >> bit) & 1));
            enforce_bit(&mut builder, wire);
            public.push(wire);
        }
    }
    let field = builder.alloc(F::from_u64(value));
    let square = builder.alloc(F::from_u64(value * value));
    builder.enforce(&Lc::from_var(field), &Lc::from_var(field), &Lc::from_var(square));
    assert!(builder.is_satisfied());
    lower_field_r1cs(builder, &public)
        .expect("lower scheduled fixture")
        .into_parts()
}

fn overlay_arm(value: u64) -> (SparseR1cs, Vec<F>) {
    let mut builder = R1csBuilder::new();
    let field = builder.alloc(F::from_u64(value));
    let square = builder.alloc(F::from_u64(value * value));
    builder.enforce(&Lc::from_var(field), &Lc::from_var(field), &Lc::from_var(square));
    assert!(builder.is_satisfied());
    lower_field_r1cs(builder, &[])
        .expect("lower overlay fixture")
        .into_parts()
}

struct Fixture {
    relation: neo_fold_clean::frontends::r1cs_f_prime::ScheduledLinkedOverlayLowNormR1cs,
    common: [Vec<F>; 3],
    phases: [Vec<F>; 3],
    overlays: [Vec<F>; 3],
}

fn fixture() -> Fixture {
    let (scheduled_shape, assignment_0) = scheduled_arm(0, 1, 11);
    let (_, assignment_1) = scheduled_arm(1, 2, 13);
    let (_, assignment_2) = scheduled_arm(2, 3, 17);
    let (overlay_shape, overlay_0) = overlay_arm(11);
    let (_, overlay_1) = overlay_arm(13);
    let (_, overlay_2) = overlay_arm(17);

    let common = build_multi_branch_selective_low_norm_r1cs_with_alignment(
        &[scheduled_shape.clone(), scheduled_shape.clone()],
        0,
        D,
        0,
    )
    .expect("compile lifecycle fixtures");
    let phases =
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&[scheduled_shape.clone(), scheduled_shape], 0, D, 0)
            .expect("compile phase fixtures");
    let overlays =
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&[overlay_shape.clone(), overlay_shape], 0, 1, 0)
            .expect("compile overlay fixtures");
    let relation = build_scheduled_linked_overlay_low_norm_r1cs(
        common,
        phases,
        overlays,
        vec![0, 1, 1],
        vec![0, 1, 0],
        vec![0, 1, 0],
        ScheduledCursorBits::new(1..3, 3..5),
        vec![
            OverlayKindLinks {
                overlay_kind: 0,
                phase_kind: 0,
                fields: vec![OverlayFieldLink {
                    phase_field: 5,
                    overlay_field: 1,
                }],
                base_pins: Vec::new(),
            },
            OverlayKindLinks {
                overlay_kind: 1,
                phase_kind: 1,
                fields: vec![OverlayFieldLink {
                    phase_field: 5,
                    overlay_field: 1,
                }],
                base_pins: vec![OverlayBaseFieldPin {
                    phase_field: PIN_PHASE_FIELD,
                    value: F::from_usize(PIN_VALUE),
                }],
            },
        ],
    )
    .expect("compose linked overlay fixture");

    Fixture {
        relation,
        common: [assignment_0.clone(), assignment_1.clone(), assignment_2.clone()],
        phases: [assignment_0, assignment_1, assignment_2],
        overlays: [overlay_0, overlay_1, overlay_2],
    }
}

fn row_terms(structure: &CcsStructure<F>, row: usize, port: usize) -> Vec<(usize, F)> {
    structure.matrices[port]
        .materialize_row(row)
        .expect("row in bounds")
}

fn row_residual(structure: &CcsStructure<F>, row: usize, assignment: &[F]) -> F {
    let point = structure
        .matrices
        .iter()
        .map(|matrix| {
            matrix
                .materialize_row(row)
                .expect("row in bounds")
                .into_iter()
                .fold(F::ZERO, |sum, (column, coefficient)| {
                    sum + coefficient * assignment[column]
                })
        })
        .collect::<Vec<_>>();
    structure.f.eval(&point)
}

fn assert_terms(structure: &CcsStructure<F>, row: usize, port: usize, mut expected: Vec<(usize, F)>) {
    expected.sort_unstable_by_key(|&(column, _)| column);
    assert_eq!(row_terms(structure, row, port), expected, "row {row}, port {port}");
}

fn assert_linear_zero_row(structure: &CcsStructure<F>, row: usize, expected_c: Vec<(usize, F)>) {
    for port in 0..PORT_COUNT {
        let expected = match port {
            GENERAL_SELECTOR => vec![(0, F::ONE)],
            C => expected_c.clone(),
            _ => Vec::new(),
        };
        assert_terms(structure, row, port, expected);
    }
}

fn assert_activation_row(structure: &CcsStructure<F>, row: usize, schedule_selector: usize, overlay_selector: usize) {
    for port in 0..PORT_COUNT {
        let expected = match port {
            GENERAL_SELECTOR => vec![(0, F::ONE)],
            A => vec![(schedule_selector, F::ONE)],
            B => vec![(overlay_selector, F::ONE)],
            C => vec![(schedule_selector, F::ONE)],
            _ => Vec::new(),
        };
        assert_terms(structure, row, port, expected);
    }
}

fn embedded_field_geometry(fixture: &Fixture) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let relation = &fixture.relation;
    let scheduled = relation.scheduled_relation();
    let phase_relation = scheduled.phase_kind_relation();
    let phase_public = phase_relation.public_input_len();
    let phase_private = scheduled.layout().phase_private_columns().start;
    let overlay_private = relation.layout().overlay_private_columns().start;
    let mut phase_starts = Vec::with_capacity(2);
    let mut overlay_starts = Vec::with_capacity(2);
    let mut widths = Vec::with_capacity(2);
    let mut radices = Vec::with_capacity(2);
    for kind in 0..2 {
        let (phase_start, phase_width) = phase_relation
            .field_slot(kind, 5)
            .expect("linked phase field slot");
        let (overlay_start, overlay_width) = relation
            .overlay_relation()
            .field_slot(kind, 1)
            .expect("linked overlay field slot");
        assert_eq!(phase_width, overlay_width);
        assert!(phase_start >= phase_public);
        assert!(overlay_start >= 1);
        phase_starts.push(phase_private + phase_start - phase_public);
        overlay_starts.push(overlay_private + overlay_start - 1);
        widths.push(phase_width);
        radices.push(match phase_width {
            41 => 3,
            23 => 7,
            1..=64 => 2,
            width => panic!("unsupported linked field width {width}"),
        });
    }
    (phase_starts, overlay_starts, widths, radices)
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
    let scheduled = relation.scheduled_relation();
    let (phase_starts, overlay_starts, widths, radices) = embedded_field_geometry(fixture);
    let mut rendered = String::new();
    rendered.push_str(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.ScheduledLinkedOverlayFixtureSchema\n\n\
/-! Generated file: exact schedule-linked private-overlay fixture.\n\n\
Owns the Rust-emitted row ranges, selector columns, schedule maps, linked\n\
field digit ranges, and selective port indices used by the exhaustive matrix\n\
test.\n\n\
Does not own component semantics, production dimensions, or the complete\n\
recursive and terminal F-prime relations. Lean recomputes every link row.\n\n\
Emits constraints: overlay selector equality, activation, exact decoded-field\n\
equality, and ring-padding rows for this fixture.\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryScheduledLinkedOverlayFixture\n\n\
open Nightstream.Implementation.R1CS.FPrimeFullHistoryScheduledLinkedOverlayFixture.Artifact\n\n\
def rawArtifact : RawArtifact where\n",
    );
    writeln!(rendered, "  schemaVersion := {SCHEMA_VERSION}").unwrap();
    writeln!(rendered, "  rows := {}", relation.structure().n).unwrap();
    writeln!(rendered, "  columns := {}", relation.structure().m).unwrap();
    writeln!(rendered, "  publicColumns := {}", relation.public_input_len()).unwrap();
    writeln!(rendered, "  scheduledRowEnd := {}", layout.scheduled_rows().end).unwrap();
    writeln!(rendered, "  overlayRowEnd := {}", layout.overlay_rows().end).unwrap();
    writeln!(
        rendered,
        "  overlayKindEqualityRowEnd := {}",
        layout.overlay_kind_equality_rows().end
    )
    .unwrap();
    writeln!(
        rendered,
        "  overlayActivationRowEnd := {}",
        layout.overlay_activation_rows().end
    )
    .unwrap();
    writeln!(rendered, "  fieldLinkRowEnd := {}", layout.field_link_rows().end).unwrap();
    writeln!(rendered, "  baseFieldPinRowEnd := {}", layout.base_field_pin_rows().end).unwrap();
    writeln!(rendered, "  ringPaddingRowEnd := {}", layout.ring_padding_rows().end).unwrap();
    writeln!(
        rendered,
        "  ringPaddingColumnStart := {}",
        layout.ring_padding_columns().start
    )
    .unwrap();
    writeln!(rendered, "  portCount := {PORT_COUNT}").unwrap();
    writeln!(rendered, "  generalSelectorPort := {GENERAL_SELECTOR}").unwrap();
    writeln!(rendered, "  aPort := {A}").unwrap();
    writeln!(rendered, "  bPort := {B}").unwrap();
    writeln!(rendered, "  cPort := {C}").unwrap();
    writeln!(
        rendered,
        "  scheduleSelectorColumns := {}",
        lean_list(scheduled.layout().schedule_selector_columns())
    )
    .unwrap();
    writeln!(
        rendered,
        "  overlaySelectorColumns := {}",
        lean_list(layout.overlay_selector_columns())
    )
    .unwrap();
    writeln!(
        rendered,
        "  lifecycleGroups := {}",
        lean_list(scheduled.layout().lifecycle_groups())
    )
    .unwrap();
    writeln!(
        rendered,
        "  phaseKinds := {}",
        lean_list(scheduled.layout().phase_kinds())
    )
    .unwrap();
    writeln!(rendered, "  overlayKinds := {}", lean_list(layout.overlay_kinds())).unwrap();
    writeln!(rendered, "  phaseFieldStarts := {}", lean_list(&phase_starts)).unwrap();
    writeln!(rendered, "  overlayFieldStarts := {}", lean_list(&overlay_starts)).unwrap();
    writeln!(rendered, "  fieldWidths := {}", lean_list(&widths)).unwrap();
    writeln!(rendered, "  fieldRadices := {}", lean_list(&radices)).unwrap();
    writeln!(rendered, "  basePinKinds := {}", lean_list(&[PIN_KIND])).unwrap();
    writeln!(rendered, "  basePinPhaseFields := {}", lean_list(&[PIN_PHASE_FIELD])).unwrap();
    writeln!(rendered, "  basePinValues := {}", lean_list(&[PIN_VALUE])).unwrap();
    rendered.push_str(
        "\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryScheduledLinkedOverlayFixture\n",
    );
    rendered
}

fn artifact_path() -> String {
    format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_PATH)
}

#[test]
fn linked_overlay_stores_components_once_and_accepts_each_exact_arm() {
    let fixture = fixture();
    let relation = &fixture.relation;
    let layout = relation.layout();
    assert_eq!(relation.structure().n, 385);
    assert_eq!(relation.structure().m, 540);
    assert_eq!(relation.public_input_len(), 54);
    assert_eq!(layout.overlay_kinds(), &[0, 1, 0]);
    assert_eq!(
        layout.scheduled_rows().len(),
        relation.scheduled_relation().structure().n
    );
    assert_eq!(layout.overlay_rows().len(), relation.overlay_relation().structure().n);
    assert_eq!(layout.overlay_kind_equality_rows().len(), 2);
    assert_eq!(layout.overlay_activation_rows().len(), 3);
    assert_eq!(layout.field_link_rows().len(), 2);
    assert_eq!(layout.base_field_pin_rows().len(), 1);
    assert_eq!(
        layout
            .field_link_rows_for_kind(0)
            .expect("kind zero links")
            .len(),
        1
    );
    assert_eq!(
        layout
            .field_link_rows_for_kind(1)
            .expect("kind one links")
            .len(),
        1
    );
    assert_eq!(
        layout
            .base_field_pin_rows_for_kind(0)
            .expect("kind zero pins")
            .len(),
        0
    );
    assert_eq!(
        layout
            .base_field_pin_rows_for_kind(1)
            .expect("kind one pins")
            .len(),
        1
    );

    for arm in 0..3 {
        let assignment = relation
            .encode(arm, &fixture.common[arm], &fixture.phases[arm], &fixture.overlays[arm])
            .expect("encode honest linked arm");
        assert!(
            relation.is_satisfied(&assignment),
            "arm {arm}: {:?}",
            relation.first_unsatisfied_row(&assignment),
        );
    }
}

#[test]
fn linked_overlay_keeps_common_source_slots_at_their_final_columns() {
    let fixture = fixture();
    let relation = &fixture.relation;
    let scheduled = relation.scheduled_relation();

    for lifecycle_group in 0..2 {
        for source_field in 0..7 {
            assert_eq!(
                relation.common_field_slot(lifecycle_group, source_field),
                scheduled
                    .common_relation()
                    .field_slot(lifecycle_group, source_field),
                "group {lifecycle_group}, source field {source_field}",
            );
        }
    }
}

#[test]
fn linked_overlay_rows_bind_selector_and_exact_private_field() {
    let fixture = fixture();
    let relation = &fixture.relation;
    let layout = relation.layout();
    let structure = relation.structure();
    let scheduled = relation.scheduled_relation();
    let schedule_selectors = scheduled.layout().schedule_selector_columns();
    let (phase_starts, overlay_starts, widths, radices) = embedded_field_geometry(&fixture);

    for kind in 0..2 {
        let row = layout.overlay_kind_equality_rows().start + kind;
        let mut expected = vec![(layout.overlay_selector_columns()[kind], F::ONE)];
        expected.extend(
            layout
                .overlay_kinds()
                .iter()
                .enumerate()
                .filter(|(_, arm_kind)| **arm_kind == kind)
                .map(|(arm, _)| (schedule_selectors[arm], -F::ONE)),
        );
        assert_linear_zero_row(structure, row, expected);
    }

    for arm in 0..3 {
        let row = layout.overlay_activation_rows().start + arm;
        assert_activation_row(
            structure,
            row,
            schedule_selectors[arm],
            layout.overlay_selector_columns()[layout.overlay_kinds()[arm]],
        );
    }

    for kind in 0..2 {
        let row = layout
            .field_link_rows_for_kind(kind)
            .expect("linked kind")
            .start;
        let mut expected_b = Vec::with_capacity(2 * widths[kind]);
        let mut coefficient = F::ONE;
        for offset in 0..widths[kind] {
            expected_b.push((phase_starts[kind] + offset, coefficient));
            expected_b.push((overlay_starts[kind] + offset, -coefficient));
            coefficient *= F::from_usize(radices[kind]);
        }
        for port in 0..PORT_COUNT {
            let expected = match port {
                GENERAL_SELECTOR => vec![(0, F::ONE)],
                A => vec![(layout.overlay_selector_columns()[kind], F::ONE)],
                B => expected_b.clone(),
                _ => Vec::new(),
            };
            assert_terms(structure, row, port, expected);
        }
    }

    let pin_row = layout
        .base_field_pin_rows_for_kind(PIN_KIND)
        .expect("pinned kind")
        .start;
    let mut expected_b = vec![(0, -F::from_usize(PIN_VALUE))];
    let mut coefficient = F::ONE;
    for offset in 0..widths[PIN_KIND] {
        expected_b.push((phase_starts[PIN_KIND] + offset, coefficient));
        coefficient *= F::from_usize(radices[PIN_KIND]);
    }
    for port in 0..PORT_COUNT {
        let expected = match port {
            GENERAL_SELECTOR => vec![(0, F::ONE)],
            A => vec![(layout.overlay_selector_columns()[PIN_KIND], F::ONE)],
            B => expected_b.clone(),
            _ => Vec::new(),
        };
        assert_terms(structure, pin_row, port, expected);
    }

    for (row, column) in layout
        .ring_padding_rows()
        .zip(layout.ring_padding_columns())
    {
        assert_linear_zero_row(structure, row, vec![(column, F::ONE)]);
    }
}

#[test]
fn linked_overlay_base_pin_rejects_a_phase_digit_tamper() {
    let fixture = fixture();
    let relation = &fixture.relation;
    let layout = relation.layout();
    let (phase_starts, _, _, _) = embedded_field_geometry(&fixture);
    let pin_row = layout
        .base_field_pin_rows_for_kind(PIN_KIND)
        .expect("pinned kind")
        .start;
    let mut assignment = relation
        .encode(
            PIN_KIND,
            &fixture.common[PIN_KIND],
            &fixture.phases[PIN_KIND],
            &fixture.overlays[PIN_KIND],
        )
        .expect("encode pinned arm");
    assert_eq!(row_residual(relation.structure(), pin_row, &assignment), F::ZERO);
    assignment[phase_starts[PIN_KIND]] += F::ONE;
    assert_ne!(row_residual(relation.structure(), pin_row, &assignment), F::ZERO);
}

#[test]
fn linked_overlay_rejects_a_separately_valid_private_substitution() {
    let fixture = fixture();
    let relation = &fixture.relation;
    let (_, different_overlay) = overlay_arm(19);
    let assignment = relation
        .encode(1, &fixture.common[1], &fixture.phases[1], &different_overlay)
        .expect("encode component-valid substituted overlay");
    assert!(relation.overlay_relation().is_satisfied(
        &relation
            .overlay_relation()
            .encode(1, &different_overlay)
            .expect("standalone overlay assignment")
    ));
    assert_eq!(
        relation.first_unsatisfied_row(&assignment),
        Some(
            relation
                .layout()
                .field_link_rows_for_kind(1)
                .expect("kind one link")
                .start
        )
    );
}

#[test]
fn scheduled_linked_overlay_generated_artifact_matches_emitted_recipe() {
    let fixture = fixture();
    let rendered = render_artifact(&fixture);
    let path = artifact_path();
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if committed != rendered {
        let expected = format!("{path}.expected");
        std::fs::write(&expected, rendered).expect("write scheduled linked-overlay artifact candidate");
        panic!("scheduled linked-overlay artifact drifted; inspect {expected} and promote it explicitly");
    }
}

#[test]
#[ignore = "deliberately writes the reviewed generated Lean artifact"]
fn regenerate_scheduled_linked_overlay_artifact() {
    std::fs::write(artifact_path(), render_artifact(&fixture()))
        .expect("write generated scheduled linked-overlay artifact");
}
