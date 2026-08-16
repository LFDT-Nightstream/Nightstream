//! Exact composition tests for common lifecycle rows plus phase-local rows.

use std::fmt::Write as _;

use neo_ccs::CcsStructure;
use neo_fold_clean::engine::r1cs_circuit::boolean::enforce_bit;
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder};
use neo_fold_clean::frontends::r1cs_f_prime::{
    build_grouped_phase_low_norm_r1cs, build_multi_branch_selective_low_norm_r1cs_with_alignment, lower_field_r1cs,
    GroupedPhaseError, SparseR1cs,
};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

const ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryGroupedPhaseFixture.lean";
const SCHEMA_VERSION: usize = 1;
const PORT_COUNT: usize = 13;
const GENERAL_SELECTOR_PORT: usize = 1;
const A_PORT: usize = 2;
const B_PORT: usize = 3;
const C_PORT: usize = 4;

fn bit_arm(public: F, private: F) -> (SparseR1cs, Vec<F>) {
    let mut builder = R1csBuilder::new();
    let public_var = builder.alloc(public);
    let private_var = builder.alloc(private);
    enforce_bit(&mut builder, public_var);
    enforce_bit(&mut builder, private_var);
    builder.enforce(
        &Lc::from_var(public_var),
        &Lc::from_var(private_var),
        &Lc::from_var(private_var),
    );
    assert!(builder.is_satisfied());
    lower_field_r1cs(builder, &[public_var])
        .expect("lower bit fixture")
        .into_parts()
}

struct Fixture {
    relation: neo_fold_clean::frontends::r1cs_f_prime::GroupedPhaseLowNormR1cs,
    common_assignments: [Vec<F>; 2],
    phase_assignments: [Vec<F>; 3],
    mismatched_phase_assignment: Vec<F>,
}

fn fixture() -> Fixture {
    let (shape, common_zero) = bit_arm(F::ONE, F::ZERO);
    let (_, common_one) = bit_arm(F::ONE, F::ONE);
    let (_, phase_zero) = bit_arm(F::ONE, F::ZERO);
    let (_, phase_one) = bit_arm(F::ONE, F::ONE);
    let (_, mismatched_phase_assignment) = bit_arm(F::ZERO, F::ZERO);

    let common = build_multi_branch_selective_low_norm_r1cs_with_alignment(&[shape.clone(), shape.clone()], 0, D, 0)
        .expect("compile common lifecycle relation");
    let phases =
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&[shape.clone(), shape.clone(), shape], 0, D, 0)
            .expect("compile phase-local relation");
    let relation =
        build_grouped_phase_low_norm_r1cs(common, phases, vec![0, 1, 1]).expect("compose grouped phase relation");

    Fixture {
        relation,
        common_assignments: [common_zero, common_one],
        phase_assignments: [phase_zero.clone(), phase_one, phase_zero],
        mismatched_phase_assignment,
    }
}

fn row_residual(structure: &CcsStructure<F>, assignment: &[F], row: usize) -> F {
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
    assert_eq!(
        structure.matrices[port]
            .materialize_row(row)
            .expect("row in bounds"),
        expected,
        "row {row}, port {port}",
    );
}

fn assert_link_rows_match_recipe(fixture: &Fixture) {
    let relation = &fixture.relation;
    let structure = relation.structure();
    let layout = relation.layout();
    for group in 0..layout.common_selector_columns().len() {
        let row = layout.group_equality_rows().start + group;
        for port in 0..PORT_COUNT {
            let expected = match port {
                GENERAL_SELECTOR_PORT => vec![(0, F::ONE)],
                C_PORT => {
                    let mut terms = vec![(layout.common_selector_columns()[group], F::ONE)];
                    terms.extend(
                        layout
                            .phase_groups()
                            .iter()
                            .enumerate()
                            .filter(|(_, phase_group)| **phase_group == group)
                            .map(|(phase, _)| (layout.phase_selector_columns()[phase], -F::ONE)),
                    );
                    terms
                }
                _ => Vec::new(),
            };
            assert_terms(structure, row, port, expected);
        }
    }
    for phase in 0..layout.phase_selector_columns().len() {
        let row = layout.phase_activation_rows().start + phase;
        let phase_selector = layout.phase_selector_columns()[phase];
        let group_selector = layout.common_selector_columns()[layout.phase_groups()[phase]];
        for port in 0..PORT_COUNT {
            let expected = match port {
                GENERAL_SELECTOR_PORT => vec![(0, F::ONE)],
                A_PORT => vec![(phase_selector, F::ONE)],
                B_PORT => vec![(group_selector, F::ONE)],
                C_PORT => vec![(phase_selector, F::ONE)],
                _ => Vec::new(),
            };
            assert_terms(structure, row, port, expected);
        }
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
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.GroupedPhaseFixtureSchema\n\n\
/-! Generated file: exact grouped-phase composition fixture.\n\n\
Owns: Rust-emitted row ranges, shared width, selector columns, phase groups,\n\
and selective port indices for the exhaustive grouped-phase matrix test.\n\n\
Does not own: source-component rows, production phase counts, or Nebula F-prime\n\
semantics. Lean recomputes every group-equality and activation row.\n\n\
Emits constraints: no. This file contains checked recipe data.\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryGroupedPhaseFixture\n\n\
open Nightstream.Implementation.R1CS.FPrimeFullHistoryGroupedPhaseFixture.Artifact\n\n\
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
        "  groupEqualityRowEnd := {}",
        layout.group_equality_rows().end
    )
    .unwrap();
    writeln!(
        rendered,
        "  phaseActivationRowEnd := {}",
        layout.phase_activation_rows().end
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
        "  phaseSelectorColumns := {}",
        lean_list(layout.phase_selector_columns())
    )
    .unwrap();
    writeln!(rendered, "  phaseGroups := {}", lean_list(layout.phase_groups())).unwrap();
    rendered.push_str(
        "\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryGroupedPhaseFixture\n",
    );
    rendered
}

fn artifact_path() -> String {
    format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_PATH)
}

#[test]
fn grouped_phase_composition_stores_each_component_once_and_accepts_each_phase() {
    let fixture = fixture();
    let relation = &fixture.relation;
    let layout = relation.layout();
    assert_eq!(layout.phase_groups(), &[0, 1, 1]);
    assert_eq!(layout.common_rows().len(), relation.common_relation().structure().n);
    assert_eq!(layout.phase_rows().len(), relation.phase_relation().structure().n);
    assert_eq!(layout.group_equality_rows().len(), 2);
    assert_eq!(layout.phase_activation_rows().len(), 3);
    assert_eq!(
        relation.structure().n,
        relation.common_relation().structure().n
            + relation.phase_relation().structure().n
            + layout.phase_groups().len()
            + layout.common_selector_columns().len()
            + layout.ring_padding_rows().len(),
    );
    assert_link_rows_match_recipe(&fixture);
    assert_eq!(
        relation.structure().m,
        relation.common_relation().structure().m + relation.phase_relation().structure().m
            - relation.public_input_len()
            + layout.ring_padding_columns().len(),
    );

    for phase in 0..3 {
        let group = layout.phase_groups()[phase];
        let assignment = relation
            .encode(
                phase,
                &fixture.common_assignments[group],
                &fixture.phase_assignments[phase],
            )
            .expect("encode honest grouped phase");
        assert!(relation.is_satisfied(&assignment), "phase {phase}");
    }
}

#[test]
fn grouped_phase_generated_artifact_matches_emitted_recipe() {
    let fixture = fixture();
    let rendered = render_artifact(&fixture);
    let path = artifact_path();
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    assert_eq!(committed, rendered, "grouped-phase artifact drifted:\n{rendered}");
}

#[test]
fn grouped_phase_links_reject_selector_mutations() {
    let fixture = fixture();
    let relation = &fixture.relation;
    let layout = relation.layout();
    for phase in 0..3 {
        let group = layout.phase_groups()[phase];
        let mut assignment = relation
            .encode(
                phase,
                &fixture.common_assignments[group],
                &fixture.phase_assignments[phase],
            )
            .expect("encode honest grouped phase");
        assignment[layout.common_selector_columns()[group]] = F::ZERO;

        let group_row = layout.group_equality_rows().start + group;
        let activation_row = layout.phase_activation_rows().start + phase;
        assert_ne!(row_residual(relation.structure(), &assignment, group_row), F::ZERO);
        assert_ne!(row_residual(relation.structure(), &assignment, activation_row), F::ZERO);
        assert!(!relation.is_satisfied(&assignment), "mutated phase {phase}");
    }
}

#[test]
fn grouped_phase_encoding_rejects_a_shared_public_mismatch() {
    let fixture = fixture();
    let error = fixture
        .relation
        .encode(0, &fixture.common_assignments[0], &fixture.mismatched_phase_assignment)
        .expect_err("different shared public values must fail");
    assert!(matches!(error, GroupedPhaseError::PublicAssignmentMismatch { .. }));
}
