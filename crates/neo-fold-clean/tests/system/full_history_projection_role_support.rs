//! Exact Rust-to-Lean ownership metadata for full-history PiRLC projections.
//!
//! Owns: generated role order, identity wires, and affine zero-padding rows.
//!
//! Does not own: projection semantics, verifier acceptance, or protocol authority.
//!
//! Emits constraints: no.
//!
//! Authority boundary: every exported column and row is re-derived from the
//! production builder audit; the generated Lean module is evidence, not authority.
//! Each 10-column tail is inferred by continuing the checked active-column
//! stride and then requiring every inferred column to occur in the exact
//! `y_zcol` zero-pin rows. A direct typed-wire audit connecting those columns
//! to `combined_c{0,1}[54..64]` remains a separate Rust-semantic refinement.
//!
//! | Artifact branch | Mathematical obligation | Exported evidence |
//! |---|---|---|
//! | `identities.y_zcol.{0,1}` | Active degree-53 output evaluation at beta | 54 active columns, 2 K-limb result columns, exact rows |
//! | `padding.y_zcol` | Canonical zero tail through padded width 64 | 10 zero-pinned columns per limb, shared exact rows |
//! | padded output view | Active prefix followed by its checked tail | generator-checked 64-column composition |
//!
//! This artifact covers the full-history recursive profile (arity 1) and its
//! terminal profile (arity 15). It is not the fixed-F-prime recursive artifact.

use std::collections::HashSet;
use std::fmt::Write as _;

use neo_fold_clean::engine::r1cs_circuit::builder::{
    ProjectionGlueAudit, ProjectionGlueRole, ProjectionIdentityAudit, ProjectionIdentityRole,
};

use super::full_history_affine_artifact_support::{affine_pins, pin_runs, render_runs, Pin};
use super::*;

const ARTIFACT_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryProjectionRoles.lean";
const Y_ZCOL_PADDED_WIDTH: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
struct YZColOutputPadding {
    limb: usize,
    shared_row_start: usize,
    shared_row_end: usize,
    zero_columns: Vec<usize>,
}

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

fn y_zcol_identities<'a>(identities: &[&'a ProjectionIdentityAudit]) -> Vec<&'a ProjectionIdentityAudit> {
    identities
        .iter()
        .copied()
        .filter(|identity| matches!(identity.role, ProjectionIdentityRole::YZColLimb { .. }))
        .collect()
}

fn profile_arity(identities: &[&ProjectionIdentityAudit]) -> usize {
    let arity = identities
        .first()
        .expect("non-empty projection profile")
        .rho_columns
        .len();
    assert!(
        identities
            .iter()
            .all(|identity| identity.rho_columns.len() == arity && identity.input_columns.len() == arity),
        "projection profile must have one common arity"
    );
    arity
}

fn y_zcol_output_padding(
    builder: &R1csBuilder,
    identities: &[&ProjectionIdentityAudit],
    glue: &[&ProjectionGlueAudit],
    profile: &str,
) -> Vec<YZColOutputPadding> {
    let identities = y_zcol_identities(identities);
    assert_eq!(
        identities
            .iter()
            .map(|identity| identity.role)
            .collect::<Vec<_>>(),
        vec![
            ProjectionIdentityRole::YZColLimb { limb: 0 },
            ProjectionIdentityRole::YZColLimb { limb: 1 },
        ],
        "{profile} y_zcol identity role census"
    );
    assert!(
        identities
            .iter()
            .all(|identity| identity.output_columns.len() == D),
        "{profile} y_zcol identities must expose all {D} active coefficient columns"
    );
    assert!(
        identities
            .iter()
            .all(|identity| identity.output_evaluation.len() == 2),
        "{profile} y_zcol identities must expose both K-valued output-evaluation columns"
    );

    let padding_owner = glue
        .iter()
        .copied()
        .find(|audit| audit.role == ProjectionGlueRole::YZColPaddingZero)
        .expect("y_zcol padding owner");
    let padding_pins = glue_pins(builder, &[padding_owner]);
    let pinned_zero_columns = padding_pins
        .iter()
        .map(|pin| match pin {
            Pin::Zero(column) => *column,
            _ => unreachable!("glue_pins rejects non-zero projection glue"),
        })
        .collect::<HashSet<_>>();

    let mut all_composed_columns = HashSet::new();
    identities
        .iter()
        .map(|identity| {
            let ProjectionIdentityRole::YZColLimb { limb } = identity.role else {
                unreachable!("filtered y_zcol identity")
            };
            let first = *identity
                .output_columns
                .first()
                .expect("non-empty y_zcol output");
            let second = *identity.output_columns.get(1).expect("two y_zcol outputs");
            let step = second
                .checked_sub(first)
                .expect("increasing y_zcol output columns");
            assert!(step > 0, "{profile} y_zcol limb {limb} output column step");
            let padded = (0..Y_ZCOL_PADDED_WIDTH)
                .map(|index| first + index * step)
                .collect::<Vec<_>>();
            assert_eq!(
                identity.output_columns.as_slice(),
                &padded[..D],
                "{profile} y_zcol limb {limb} active output columns must be the ordered padded prefix"
            );
            let zero_columns = padded[D..].to_vec();
            assert!(
                zero_columns
                    .iter()
                    .all(|column| pinned_zero_columns.contains(column)),
                "{profile} y_zcol limb {limb} padded tail must be pinned by the y_zcol glue owner"
            );
            assert!(
                padded
                    .iter()
                    .all(|column| all_composed_columns.insert(*column)),
                "{profile} y_zcol padded limb views must be disjoint"
            );
            YZColOutputPadding {
                limb,
                shared_row_start: padding_owner.row_start,
                shared_row_end: padding_owner.row_end,
                zero_columns,
            }
        })
        .collect()
}

fn lean_y_zcol_identities(identities: &[&ProjectionIdentityAudit]) -> String {
    let mut rendered = String::new();
    for (index, identity) in y_zcol_identities(identities).iter().enumerate() {
        let ProjectionIdentityRole::YZColLimb { limb } = identity.role else {
            unreachable!("filtered y_zcol identity")
        };
        let prefix = if index == 0 { "  [" } else { "  ," };
        writeln!(
            rendered,
            "{prefix} ⟨{limb}, {}, {}, {}, {}⟩",
            identity.row_start,
            identity.row_end,
            lean_nat_list(identity.output_columns.iter().copied()),
            lean_nat_list(identity.output_evaluation),
        )
        .expect("render y_zcol identity owner");
    }
    rendered.push_str("  ]");
    rendered
}

fn lean_y_zcol_padding(padding: &[YZColOutputPadding]) -> String {
    let mut rendered = String::new();
    for (index, owner) in padding.iter().enumerate() {
        let prefix = if index == 0 { "  [" } else { "  ," };
        writeln!(
            rendered,
            "{prefix} ⟨{}, {}, {}, {}⟩",
            owner.limb,
            owner.shared_row_start,
            owner.shared_row_end,
            lean_nat_list(owner.zero_columns.iter().copied()),
        )
        .expect("render y_zcol output padding");
    }
    rendered.push_str("  ]");
    rendered
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
    recursive_y_zcol_padding: &[YZColOutputPadding],
    terminal_y_zcol_padding: &[YZColOutputPadding],
    recursive_arity: usize,
    terminal_arity: usize,
    recursive_glue: &[&ProjectionGlueAudit],
    terminal_glue: &[&ProjectionGlueAudit],
    recursive_pins: &[Pin],
    terminal_pins: &[Pin],
) -> String {
    format!(
        "import Nightstream.Implementation.R1CS.Core.AffinePins\n\n\
         /-! Generated semantic ownership for the full-history PiRLC projection rows.\n\n\
         Owns: exact production row and column evidence for projection identities and affine glue.\n\
         Does not own: protocol semantics, typed `combined_c` wire identity, witness authority,\n\
         or verifier acceptance.\n\
         Emits constraints: no.\n\n\
         Each 10-column tail is inferred by continuing the checked active-column stride and\n\
         requiring every inferred column to occur in the exact `y_zcol` zero-pin rows. It is\n\
         artifact-checked layout evidence; direct typed-wire refinement remains open.\n\n\
         Scope: recursive arity 1 and terminal arity 15 in the full-history audit relation.\n\
         This is not evidence for the fixed-F-prime recursive profile.\n\n\
         | Child path | Mathematical obligation | Rust evidence | Lean evidence |\n\
         |---|---|---|---|\n\
         | `nifs.pi_rlc.identities.y_zcol.{{0,1}}` | Evaluate each 54-coefficient active output limb at beta | `ProjectionIdentityAudit` | `YZColIdentityOwner` |\n\
         | `nifs.pi_rlc.padding.y_zcol` | Pin each inferred 10-column tail to zero | `ProjectionGlueAudit` plus stride check | `YZColOutputPadding` |\n\
         | derived padded output | Concatenate each active prefix with its checked tail | generator composition | `recursive/terminalYZColPaddedOutputColumns` |\n\
         -/\n\n\
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
         structure YZColIdentityOwner where\n\
           limb : Nat\n\
           rowStart : Nat\n\
           rowEnd : Nat\n\
           activeCoefficientColumns : List Nat\n\
           outputEvaluationColumns : List Nat\n\
         deriving DecidableEq, Repr\n\n\
         structure YZColOutputPadding where\n\
           limb : Nat\n\
           sharedRowStart : Nat\n\
           sharedRowEnd : Nat\n\
           zeroColumns : List Nat\n\
         deriving DecidableEq, Repr\n\n\
         def nativeVerifierOrder : List Role :=\n\
           (List.range 18).map .commitmentLane ++\n\
           (List.range 5).map .activeXColumn ++\n\
           ((List.range 3).flatMap fun row =>\n\
             (List.range 2).map fun limb => .yRingLimb row limb) ++\n\
           (List.range 2).map .yZColLimb\n\n\
         def recursiveProjectionArity : Nat := {}\n\
         def terminalProjectionArity : Nat := {}\n\n\
         def recursiveRoles : List Role := [{}]\n\
         def terminalRoles : List Role := [{}]\n\n\
         def recursiveYZColIdentities : List YZColIdentityOwner :=\n{}\n\n\
         def terminalYZColIdentities : List YZColIdentityOwner :=\n{}\n\n\
         def recursiveYZColOutputPadding : List YZColOutputPadding :=\n{}\n\n\
         def terminalYZColOutputPadding : List YZColOutputPadding :=\n{}\n\n\
         def paddedOutputColumns\n\
             (identity : YZColIdentityOwner) (padding : YZColOutputPadding) : List Nat :=\n\
           identity.activeCoefficientColumns ++ padding.zeroColumns\n\n\
         def recursiveYZColPaddedOutputColumns : List (List Nat) :=\n\
           List.zipWith paddedOutputColumns recursiveYZColIdentities recursiveYZColOutputPadding\n\
         def terminalYZColPaddedOutputColumns : List (List Nat) :=\n\
           List.zipWith paddedOutputColumns terminalYZColIdentities terminalYZColOutputPadding\n\n\
         def outputPaddingPins (owners : List YZColOutputPadding) : List AffinePins.Pin :=\n\
           owners.flatMap fun owner => owner.zeroColumns.map .zero\n\n\
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
         theorem full_history_profile_arities :\n\
             recursiveProjectionArity = 1 ∧ terminalProjectionArity = 15 := by native_decide\n\
         theorem recursive_y_zcol_identity_census :\n\
             recursiveYZColIdentities.map (fun owner => owner.limb) = [0, 1] := by native_decide\n\
         theorem terminal_y_zcol_identity_census :\n\
             terminalYZColIdentities.map (fun owner => owner.limb) = [0, 1] := by native_decide\n\
         theorem y_zcol_padding_census :\n\
             recursiveYZColOutputPadding.map (fun owner => owner.limb) = [0, 1] ∧\n\
             terminalYZColOutputPadding.map (fun owner => owner.limb) = [0, 1] := by native_decide\n\
         theorem y_zcol_active_coefficient_width :\n\
             (recursiveYZColIdentities ++ terminalYZColIdentities).all\n\
               (fun owner => owner.activeCoefficientColumns.length == 54) = true := by native_decide\n\
         theorem y_zcol_output_evaluation_width :\n\
             (recursiveYZColIdentities ++ terminalYZColIdentities).all\n\
               (fun owner => owner.outputEvaluationColumns.length == 2) = true := by native_decide\n\
         theorem y_zcol_padding_width :\n\
             (recursiveYZColOutputPadding ++ terminalYZColOutputPadding).all\n\
               (fun owner => owner.zeroColumns.length == 10) = true := by native_decide\n\
         theorem y_zcol_padded_output_width :\n\
             (recursiveYZColPaddedOutputColumns ++ terminalYZColPaddedOutputColumns).all\n\
               (fun columns => columns.length == 64) = true := by native_decide\n\
         theorem y_zcol_padded_output_columns_disjoint :\n\
             recursiveYZColPaddedOutputColumns.flatten.eraseDups.length = 128 ∧\n\
             terminalYZColPaddedOutputColumns.flatten.eraseDups.length = 128 := by native_decide\n\
         theorem y_zcol_output_padding_is_glue :\n\
             (outputPaddingPins recursiveYZColOutputPadding).all recursiveGluePins.contains = true ∧\n\
             (outputPaddingPins terminalYZColOutputPadding).all terminalGluePins.contains = true := by\n\
           native_decide\n\
         theorem y_zcol_output_padding_rows_match_glue_owner :\n\
             recursiveYZColOutputPadding.all (fun padding =>\n\
               recursiveGlueOwners.contains\n\
                 ⟨.yZColPaddingZero, padding.sharedRowStart, padding.sharedRowEnd⟩) = true ∧\n\
             terminalYZColOutputPadding.all (fun padding =>\n\
               terminalGlueOwners.contains\n\
                 ⟨.yZColPaddingZero, padding.sharedRowStart, padding.sharedRowEnd⟩) = true := by\n\
           native_decide\n\
         theorem y_zcol_identity_ranges_nonempty :\n\
             (recursiveYZColIdentities ++ terminalYZColIdentities).all\n\
               (fun owner => decide (owner.rowStart < owner.rowEnd)) = true := by native_decide\n\
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
        recursive_arity,
        terminal_arity,
        lean_roles(recursive_identities),
        lean_roles(terminal_identities),
        lean_y_zcol_identities(recursive_identities),
        lean_y_zcol_identities(terminal_identities),
        lean_y_zcol_padding(recursive_y_zcol_padding),
        lean_y_zcol_padding(terminal_y_zcol_padding),
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
            let y_zcol_padding = y_zcol_output_padding(
                builder,
                &identities,
                &glue,
                &format!("profile starting at row {}", range.row_start),
            );
            assert_exact_partition(range, &identities, &glue);
            let pins = glue_pins(builder, &glue);
            (identities, glue, pins, y_zcol_padding)
        })
        .collect::<Vec<_>>();
    let arities = groups
        .iter()
        .map(|group| profile_arity(&group.0))
        .collect::<Vec<_>>();
    assert_eq!(
        arities,
        [1, 15],
        "full-history recursive and terminal projection arities"
    );
    assert_eq!(groups[0].2.len(), 162, "recursive affine projection glue rows");
    assert_eq!(groups[1].2.len(), 1296, "terminal affine projection glue rows");
    let rendered = render_artifact(
        &groups[0].0,
        &groups[1].0,
        &groups[0].3,
        &groups[1].3,
        arities[0],
        arities[1],
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
