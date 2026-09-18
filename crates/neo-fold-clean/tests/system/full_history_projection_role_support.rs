use std::fmt::Write as _;

use neo_fold_clean::engine::r1cs_circuit::builder::{
    ProjectionGlueAudit, ProjectionGlueRole, ProjectionIdentityAudit, ProjectionIdentityRole,
};

use super::full_history_affine_artifact_support::{affine_pins, pin_runs, render_runs, Pin};
use super::*;

const ARTIFACT_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryProjectionRoles.lean";

fn projection_ranges(builder: &R1csBuilder) -> Vec<&RowFamilyRange> {
    let mut ranges = builder
        .row_family_ranges()
        .iter()
        .filter(|range| range.name == "nifs.pi_rlc.projection_identities")
        .collect::<Vec<_>>();
    ranges.sort_by_key(|range| range.row_start);
    assert_eq!(ranges.len(), 2, "recursive and terminal projection owners");
    ranges
}

fn identities_in<'a>(builder: &'a R1csBuilder, range: &RowFamilyRange) -> Vec<&'a ProjectionIdentityAudit> {
    let mut identities = builder
        .projection_identity_audits()
        .iter()
        .filter(|audit| range.row_start <= audit.row_start && audit.row_end <= range.row_end)
        .collect::<Vec<_>>();
    identities.sort_by_key(|audit| audit.row_start);
    identities
}

fn glue_in<'a>(builder: &'a R1csBuilder, range: &RowFamilyRange) -> Vec<&'a ProjectionGlueAudit> {
    let mut glue = builder
        .projection_glue_audits()
        .iter()
        .filter(|audit| range.row_start <= audit.row_start && audit.row_end <= range.row_end)
        .collect::<Vec<_>>();
    glue.sort_by_key(|audit| audit.row_start);
    glue
}

fn expected_roles() -> Vec<ProjectionIdentityRole> {
    let mut roles = (0..18)
        .map(|lane| ProjectionIdentityRole::CommitmentLane { lane })
        .collect::<Vec<_>>();
    roles.extend((0..5).map(|column| ProjectionIdentityRole::ActiveXColumn { column }));
    for row in 0..3 {
        roles.extend((0..2).map(|limb| ProjectionIdentityRole::YRingLimb { row, limb }));
    }
    roles.extend((0..2).map(|limb| ProjectionIdentityRole::YZColLimb { limb }));
    roles
}

fn expected_glue_roles() -> Vec<ProjectionGlueRole> {
    vec![
        ProjectionGlueRole::InactiveXZero,
        ProjectionGlueRole::YRingPaddingZero { row: 0 },
        ProjectionGlueRole::YRingPaddingZero { row: 1 },
        ProjectionGlueRole::YRingPaddingZero { row: 2 },
        ProjectionGlueRole::YZColPaddingZero,
    ]
}

fn assert_exact_partition(
    range: &RowFamilyRange,
    identities: &[&ProjectionIdentityAudit],
    glue: &[&ProjectionGlueAudit],
) {
    let mut intervals = identities
        .iter()
        .map(|audit| (audit.row_start, audit.row_end))
        .chain(glue.iter().map(|audit| (audit.row_start, audit.row_end)))
        .collect::<Vec<_>>();
    intervals.sort_unstable();
    let mut cursor = range.row_start;
    for (start, end) in intervals {
        assert_eq!(start, cursor, "unowned or overlapping projection row at {cursor}");
        assert!(start < end, "empty projection owner at row {start}");
        cursor = end;
    }
    assert_eq!(cursor, range.row_end, "projection owner leaves a residual suffix");
}

fn glue_pins(builder: &R1csBuilder, glue: &[&ProjectionGlueAudit]) -> Vec<Pin> {
    glue.iter()
        .flat_map(|audit| {
            affine_pins(
                builder,
                &RowFamilyRange {
                    name: "nifs.pi_rlc.projection_glue",
                    row_start: audit.row_start,
                    row_end: audit.row_end,
                },
            )
        })
        .inspect(|pin| assert!(matches!(pin, Pin::Zero(_)), "projection glue must be a zero pin"))
        .collect()
}

fn lean_role(role: ProjectionIdentityRole) -> String {
    match role {
        ProjectionIdentityRole::CommitmentLane { lane } => format!(".commitmentLane {lane}"),
        ProjectionIdentityRole::ActiveXColumn { column } => format!(".activeXColumn {column}"),
        ProjectionIdentityRole::YRingLimb { row, limb } => format!(".yRingLimb {row} {limb}"),
        ProjectionIdentityRole::YZColLimb { limb } => format!(".yZColLimb {limb}"),
        other => panic!("unsupported plain-profile projection role {other:?}"),
    }
}

fn lean_roles(identities: &[&ProjectionIdentityAudit]) -> String {
    identities
        .iter()
        .map(|identity| lean_role(identity.role))
        .collect::<Vec<_>>()
        .join(", ")
}

fn lean_glue_role(role: ProjectionGlueRole) -> String {
    match role {
        ProjectionGlueRole::InactiveXZero => ".inactiveXZero".into(),
        ProjectionGlueRole::YRingPaddingZero { row } => format!(".yRingPaddingZero {row}"),
        ProjectionGlueRole::YZColPaddingZero => ".yZColPaddingZero".into(),
    }
}

fn lean_glue(glue: &[&ProjectionGlueAudit]) -> String {
    let mut rendered = String::new();
    for (index, audit) in glue.iter().enumerate() {
        let prefix = if index == 0 { "  [" } else { "  ," };
        writeln!(
            rendered,
            "{prefix} ⟨{}, {}, {}⟩",
            lean_glue_role(audit.role),
            audit.row_start,
            audit.row_end
        )
        .expect("render glue owner");
    }
    rendered.push_str("  ]");
    rendered
}

fn render_artifact(
    recursive_identities: &[&ProjectionIdentityAudit],
    terminal_identities: &[&ProjectionIdentityAudit],
    recursive_glue: &[&ProjectionGlueAudit],
    terminal_glue: &[&ProjectionGlueAudit],
    recursive_pins: &[Pin],
    terminal_pins: &[Pin],
) -> String {
    format!(
        "import Nightstream.Implementation.R1CS.Core.AffinePins\n\n\
         /-! Generated semantic ownership for every plain-profile PiRLC projection row. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles\n\n\
         open Nightstream.Implementation.R1CS\n\n\
         inductive Role where\n\
           | commitmentLane (lane : Nat)\n\
           | activeXColumn (column : Nat)\n\
           | yRingLimb (row limb : Nat)\n\
           | yZColLimb (limb : Nat)\n\
         deriving DecidableEq, Repr\n\n\
         inductive GlueRole where\n\
           | inactiveXZero\n\
           | yRingPaddingZero (row : Nat)\n\
           | yZColPaddingZero\n\
         deriving DecidableEq, Repr\n\n\
         structure GlueOwner where\n\
           role : GlueRole\n\
           rowStart : Nat\n\
           rowEnd : Nat\n\
         deriving DecidableEq, Repr\n\n\
         def nativeVerifierOrder : List Role :=\n\
           (List.range 18).map .commitmentLane ++\n\
           (List.range 5).map .activeXColumn ++\n\
           ((List.range 3).flatMap fun row =>\n\
             (List.range 2).map fun limb => .yRingLimb row limb) ++\n\
           (List.range 2).map .yZColLimb\n\n\
         def recursiveRoles : List Role := [{}]\n\
         def terminalRoles : List Role := [{}]\n\n\
         def recursiveGlueOwners : List GlueOwner :=\n{}\n\n\
         def terminalGlueOwners : List GlueOwner :=\n{}\n\n\
         def recursiveGlueRuns : List AffinePins.Run :=\n{}\n\n\
         def terminalGlueRuns : List AffinePins.Run :=\n{}\n\n\
         def recursiveGluePins : List AffinePins.Pin :=\n\
           AffinePins.expandRuns recursiveGlueRuns\n\
         def terminalGluePins : List AffinePins.Pin :=\n\
           AffinePins.expandRuns terminalGlueRuns\n\n\
         def recursiveGlueRows : List Row := AffinePins.rows recursiveGluePins\n\
         def terminalGlueRows : List Row := AffinePins.rows terminalGluePins\n\n\
         theorem recursive_roles_native_order : recursiveRoles = nativeVerifierOrder := by native_decide\n\
         theorem terminal_roles_native_order : terminalRoles = nativeVerifierOrder := by native_decide\n\
         theorem role_census : recursiveRoles.length = 31 ∧ terminalRoles.length = 31 := by native_decide\n\
         theorem recursive_glue_owner_census :\n\
             recursiveGlueOwners.map (fun owner => owner.role) =\n\
               [.inactiveXZero, .yRingPaddingZero 0, .yRingPaddingZero 1,\n\
                .yRingPaddingZero 2, .yZColPaddingZero] := by native_decide\n\
         theorem terminal_glue_owner_census :\n\
             terminalGlueOwners.map (fun owner => owner.role) =\n\
               [.inactiveXZero, .yRingPaddingZero 0, .yRingPaddingZero 1,\n\
                .yRingPaddingZero 2, .yZColPaddingZero] := by native_decide\n\
         theorem recursive_glue_rows : recursiveGluePins.length = 162 := by\n\
           rw [recursiveGluePins, AffinePins.expandRuns_length]\n\
           native_decide\n\
         theorem terminal_glue_rows : terminalGluePins.length = 1296 := by\n\
           rw [terminalGluePins, AffinePins.expandRuns_length]\n\
           native_decide\n\n\
         def pinIsZero : AffinePins.Pin → Bool\n\
           | .zero _ => true\n\
           | _ => false\n\n\
         theorem recursive_glue_only_zero :\n\
             recursiveGluePins.all pinIsZero = true := by native_decide\n\
         theorem terminal_glue_only_zero :\n\
             terminalGluePins.all pinIsZero = true := by native_decide\n\n\
         theorem recursive_glue_sound\n\
             {{assignment : Nat → Nat}}\n\
             (canonical : ∀ column, assignment column < goldilocksP)\n\
             (one : assignment 0 = 1)\n\
             (satisfies : Satisfies recursiveGlueRows assignment) :\n\
             ∀ pin ∈ recursiveGluePins, AffinePins.Pin.Holds assignment pin := by\n\
         exact AffinePins.rows_sound (by native_decide) canonical one satisfies\n\n\
         theorem recursive_glue_complete\n\
             {{assignment : Nat → Nat}}\n\
             (canonical : ∀ column, assignment column < goldilocksP)\n\
             (one : assignment 0 = 1)\n\
             (holds : ∀ pin ∈ recursiveGluePins, AffinePins.Pin.Holds assignment pin) :\n\
             Satisfies recursiveGlueRows assignment := by\n\
           exact AffinePins.rows_complete (by native_decide) canonical one holds\n\n\
         theorem terminal_glue_sound\n\
             {{assignment : Nat → Nat}}\n\
             (canonical : ∀ column, assignment column < goldilocksP)\n\
             (one : assignment 0 = 1)\n\
             (satisfies : Satisfies terminalGlueRows assignment) :\n\
             ∀ pin ∈ terminalGluePins, AffinePins.Pin.Holds assignment pin := by\n\
         exact AffinePins.rows_sound (by native_decide) canonical one satisfies\n\n\
         theorem terminal_glue_complete\n\
             {{assignment : Nat → Nat}}\n\
             (canonical : ∀ column, assignment column < goldilocksP)\n\
             (one : assignment 0 = 1)\n\
             (holds : ∀ pin ∈ terminalGluePins, AffinePins.Pin.Holds assignment pin) :\n\
             Satisfies terminalGlueRows assignment := by\n\
           exact AffinePins.rows_complete (by native_decide) canonical one holds\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles\n",
        lean_roles(recursive_identities),
        lean_roles(terminal_identities),
        lean_glue(recursive_glue),
        lean_glue(terminal_glue),
        render_runs(&pin_runs(recursive_pins)),
        render_runs(&pin_runs(terminal_pins)),
    )
}

pub fn compare_projection_role_artifact(builder: &R1csBuilder) {
    let ranges = projection_ranges(builder);
    let groups = ranges
        .iter()
        .map(|range| {
            let identities = identities_in(builder, range);
            let glue = glue_in(builder, range);
            assert_eq!(
                identities
                    .iter()
                    .map(|identity| identity.role)
                    .collect::<Vec<_>>(),
                expected_roles(),
                "{} projection roles must follow native verifier order",
                range.row_start
            );
            assert_eq!(
                glue.iter().map(|audit| audit.role).collect::<Vec<_>>(),
                expected_glue_roles(),
                "{} projection glue role census",
                range.row_start
            );
            assert_exact_partition(range, &identities, &glue);
            let pins = glue_pins(builder, &glue);
            (identities, glue, pins)
        })
        .collect::<Vec<_>>();
    for ((identities, _, _), arity) in groups.iter().zip([1, 15]) {
        assert!(
            identities
                .iter()
                .all(|identity| identity.rho_columns.len() == arity && identity.input_columns.len() == arity),
            "projection role census must retain arity {arity}"
        );
    }
    assert_eq!(groups[0].2.len(), 162, "recursive affine projection glue rows");
    assert_eq!(groups[1].2.len(), 1296, "terminal affine projection glue rows");
    let rendered = render_artifact(
        &groups[0].0,
        &groups[1].0,
        &groups[0].1,
        &groups[1].1,
        &groups[0].2,
        &groups[1].2,
    );
    compare_full_history_artifact(&formal_repo_root().join(ARTIFACT_PATH), &rendered, "lean.expected");
}

#[test]
fn full_history_projection_roles_and_glue_match_exact_rows() {
    let (prep, finished) = build_honest_finished_proof(2);
    let statement = neo_fold_clean::build_decider_statement(&prep, &finished);
    let synth = synthesize_statement_r1cs(&prep, &statement).expect("synthesize full history");
    assert!(synth.builder.is_satisfied(), "honest full-history rows");
    compare_projection_role_artifact(&synth.builder);
}
