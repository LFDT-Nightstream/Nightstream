import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryTerminalCeSound
import Nightstream.Implementation.R1CS.Correspondence.Gadgets.TerminalPendingProjectionCompiler

/-!
Bounded full-history bridge for terminal delayed-`y_zcol` recomposition.

This leaf deliberately targets the existing generated terminal-CE profile:
one 257-column relation with a 54 x 5 witness. It is not the selective
fixed-point production relation whose logical width is 11,437,038. Importing
this theorem therefore cannot establish production-shape authority.

The important authority direction is nevertheless concrete. Every child
column used by the recomposition layout is constructed from the exact mapped
`ncEvaluationCols` already proved sound by `all_claims_sound`. Thus the child
values are derived from the terminal CE full-witness/s_col computation; no
child `CeClaim.y_zcol` value, digest, or caller-provided semantic equality is
a premise.

The Rust exporter still has to emit the parent columns and prove its 108 rows
equal `TerminalPendingProjectionCompiler.rows (projectionLayout parentCols)`.
It must also export the pending-old-block association. Those are structural
artifact obligations, not fields of the theorem below.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPendingProjectionBridge

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.TerminalCeCompiler
open Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCe
open Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound

def activeLaneCount : Nat := 54

/-- Pair a flat base-field limb list without inventing a trailing limb. -/
def pairColumns : List Nat → List KColumns
  | c0 :: c1 :: tail => { c0 := c0, c1 := c1 } :: pairColumns tail
  | _ => []

theorem pairColumns_values (assignment : Nat → Nat) :
    ∀ columns,
      (pairColumns columns).map (fun value => value.value assignment) =
        pairs (valuesAt assignment columns)
  | [] => rfl
  | [_] => rfl
  | c0 :: c1 :: tail => by
      simp only [pairColumns, List.map_cons, pairs, valuesAt,
        KColumns.value, baseAt, fieldAt, List.cons.injEq, true_and]
      exact pairColumns_values assignment tail

def relabelColumns (columnMap : List Nat) (columns : KColumns) : KColumns :=
  { c0 := Relabel.column columnMap columns.c0
    c1 := Relabel.column columnMap columns.c1 }

@[simp] theorem relabelColumns_value
    (columnMap : List Nat) (columns : KColumns) (assignment : Nat → Nat) :
    (relabelColumns columnMap columns).value assignment =
      columns.value (Relabel.assignment columnMap assignment) := by
  rfl

/-- All 64 mapped terminal-CE output lanes for one child. -/
def mappedNcColumns (columnMap : List Nat) : List KColumns :=
  (pairColumns layout.ncEvaluationCols).map (relabelColumns columnMap)

/-- The 54 active mapped terminal-CE output lanes used by recomposition. -/
def mappedActiveColumns (columnMap : List Nat) : List KColumns :=
  (mappedNcColumns columnMap).take activeLaneCount

theorem claimHolds_implies_ncEvaluations
    {assignment : Nat → Nat}
    (holds : TerminalCeCompiler.ClaimHolds program assignment) :
    (decodeSidecar layout assignment).evaluations =
      program.expectedNcEvaluations assignment := by
  unfold TerminalCeCompiler.ClaimHolds at holds
  unfold Nightstream.Protocol.TerminalCE.ClaimHolds at holds
  have sidecarAccepted := holds.2.2.2.2.2.2.2
  have expected : program.expectedNcEvaluations assignment =
      (decodeSidecar layout assignment).evaluations := by
    symm
    simpa [Program.semantics, TerminalCeCompiler.context,
      TerminalCeCompiler.claim, decodeSidecar, kValuesAt] using
      sidecarAccepted
  exact expected.symm

theorem mappedNcColumns_values
    (columnMap : List Nat) (assignment : Nat → Nat) :
    (mappedNcColumns columnMap).map (fun value => value.value assignment) =
      (decodeSidecar layout
        (Relabel.assignment columnMap assignment)).evaluations := by
  calc
    (mappedNcColumns columnMap).map (fun value => value.value assignment) =
        (pairColumns layout.ncEvaluationCols).map
          (fun value => value.value
            (Relabel.assignment columnMap assignment)) := by
      simp [mappedNcColumns, List.map_map, Function.comp_def]
    _ = pairs (valuesAt (Relabel.assignment columnMap assignment)
          layout.ncEvaluationCols) :=
      pairColumns_values (Relabel.assignment columnMap assignment)
        layout.ncEvaluationCols
    _ = (decodeSidecar layout
          (Relabel.assignment columnMap assignment)).evaluations := by
      rfl

/-- Exact mapped terminal CE soundness for all 64 lanes. The final ten lanes
are tied to the checked full-witness computation here; proving that the
computed values are zero is a separate generic compiler lemma still required
before a production padding claim. -/
theorem mappedNcColumns_values_eq_expected
    {assignment : Nat → Nat}
    (allClaims : AllClaimsHold assignment)
    (columnMap : List Nat) (columnMapMember : columnMap ∈ columnMaps) :
    (mappedNcColumns columnMap).map (fun value => value.value assignment) =
      program.expectedNcEvaluations
        (Relabel.assignment columnMap assignment) := by
  exact (mappedNcColumns_values columnMap assignment).trans
    (claimHolds_implies_ncEvaluations
      (allClaims columnMap columnMapMember))

theorem mappedActiveColumns_values_eq_expected
    {assignment : Nat → Nat}
    (allClaims : AllClaimsHold assignment)
    (columnMap : List Nat) (columnMapMember : columnMap ∈ columnMaps) :
    (mappedActiveColumns columnMap).map
        (fun value => value.value assignment) =
      (program.expectedNcEvaluations
        (Relabel.assignment columnMap assignment)).take activeLaneCount := by
  simpa [mappedActiveColumns] using congrArg (List.take activeLaneCount)
    (mappedNcColumns_values_eq_expected allClaims columnMap columnMapMember)

/-- Child-major terminal columns in exact PiDEC order. -/
def terminalChildColumns : List (List KColumns) :=
  columnMaps.map mappedActiveColumns

def terminalExpectedChildren (assignment : Nat → Nat) :
    List (List ProjectionProgram.K) :=
  columnMaps.map fun columnMap =>
    (program.expectedNcEvaluations
      (Relabel.assignment columnMap assignment)).take activeLaneCount

private theorem mapChildrenValues
    (assignment : Nat → Nat) :
    ∀ maps : List (List Nat),
      (∀ columnMap, columnMap ∈ maps →
        (mappedActiveColumns columnMap).map
            (fun value => value.value assignment) =
          (program.expectedNcEvaluations
            (Relabel.assignment columnMap assignment)).take
              activeLaneCount) →
      (maps.map mappedActiveColumns).map
          (fun child => child.map (fun value => value.value assignment)) =
        maps.map (fun columnMap =>
          (program.expectedNcEvaluations
            (Relabel.assignment columnMap assignment)).take activeLaneCount)
  | [], _ => rfl
  | columnMap :: maps, pointwise => by
      simp only [List.map_cons, List.cons.injEq]
      constructor
      · exact pointwise columnMap (by simp)
      · apply mapChildrenValues assignment maps
        intro current currentMember
        exact pointwise current (by simp [currentMember])

theorem terminalChildColumns_values_eq_expected
    {assignment : Nat → Nat}
    (allClaims : AllClaimsHold assignment) :
    terminalChildColumns.map
        (fun child => child.map (fun value => value.value assignment)) =
      terminalExpectedChildren assignment := by
  unfold terminalChildColumns terminalExpectedChildren
  apply mapChildrenValues assignment columnMaps
  intro columnMap columnMapMember
  exact mappedActiveColumns_values_eq_expected allClaims columnMap columnMapMember

def laneLayout (parentColumns : List KColumns) (lane : Nat) :
    TerminalPendingProjectionCompiler.LaneLayout where
  parent := parentColumns.getD lane default
  children := terminalChildColumns.map fun child => child.getD lane default

/-- The only permissible bounded layout: child columns come directly from
the exact terminal-CE maps, never from a separately exported sidecar list. -/
def projectionLayout (parentColumns : List KColumns) :
    TerminalPendingProjectionCompiler.Layout where
  radix := 2
  childCount := 14
  lanes := (List.range activeLaneCount).map (laneLayout parentColumns)

theorem projectionLayout_shape (parentColumns : List KColumns) :
    TerminalPendingProjectionCompiler.ShapeValid
      (projectionLayout parentColumns) := by
  constructor
  · intro lane laneMember
    rcases List.mem_map.mp laneMember with ⟨index, indexMember, rfl⟩
    simp [projectionLayout, laneLayout, terminalChildColumns,
      column_maps_length]
  · intro child childLt
    change child < 14 at childLt
    change 0 < 2 ^ child % goldilocksP ∧
      2 ^ child % goldilocksP < goldilocksP
    have childLe : child ≤ 13 := by omega
    have powerLe : 2 ^ child ≤ 2 ^ 13 :=
      Nat.pow_le_pow_right (by decide) childLe
    have powerLt : 2 ^ child < goldilocksP :=
      Nat.lt_of_le_of_lt powerLe (by decide)
    rw [Nat.mod_eq_of_lt powerLt]
    exact ⟨Nat.two_pow_pos child, powerLt⟩

/-- Bounded artifact-facing result. Exact terminal-CE rows derive all child
values from their mapped full-witness computations, while the dedicated 108
rows derive the parent radix equations. This is not a production-shape
`PackedYZcolBoundAtBlock` theorem. -/
structure BoundedHolds (parentColumns : List KColumns)
    (assignment : Nat → Nat) : Prop where
  parentRecomposition :
    TerminalPendingProjectionCompiler.Accepted
      (projectionLayout parentColumns) assignment
  childFullWitnessEvaluations :
    terminalChildColumns.map
        (fun child => child.map (fun value => value.value assignment)) =
      terminalExpectedChildren assignment

theorem rows_imply_boundedHolds
    (parentColumns : List KColumns)
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (terminalCeSatisfies : Satisfies terminalCeRows assignment)
    (recompositionSatisfies :
      Satisfies (TerminalPendingProjectionCompiler.rows
        (projectionLayout parentColumns))
        assignment) :
    BoundedHolds parentColumns assignment := by
  have allClaims := all_claims_sound prime canonical one terminalCeSatisfies
  exact {
    parentRecomposition := TerminalPendingProjectionCompiler.rows_sound
      (projectionLayout_shape parentColumns) canonical one
      recompositionSatisfies
    childFullWitnessEvaluations :=
      terminalChildColumns_values_eq_expected allClaims
  }

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPendingProjectionBridge
