import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Generated.SelectiveMatrixRows
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Generated.SelectiveRows
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Schema

/-!
Artifact census for the lossless compact rows emitted for the fixed-point
`y_zcol` projection slice.

Owns: exact row count and dimensions, exact emitted-index agreement with the
source-to-selective fragment artifact, unique fragment ownership, and
agreement between each diagnostic family tag and its unique fragment owner.

Does not own: field decoding, sparse-row semantics, selector truth, rewrite
correctness, source-column meaning, protocol authority, or row removal.

Emits constraints: no.

Authority boundary: the family comparison below is an artifact consistency
check only. Later semantic correspondence must inspect decoded coefficients;
it may not infer an equation from a family tag.

| Artifact leaf | Mathematical obligation | Authority class |
|---|---|---|
| emitted-row census | every selected physical row occurs exactly once | checked |
| owner join | emitted index and compiler fragment agree | computed |
| family tag | diagnostic tag agrees with the unique owner | checked, non-semantic |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Census

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Wire
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective

private abbrev ownership : Selective.Artifact :=
  Generated.SelectiveRows.artifact
private abbrev rawRows : List Materialized.RawRow :=
  Generated.SelectiveMatrixRows.rawRows

def emittedRows : List Nat := rawRows.map Materialized.RawRow.emittedRow

def expectedEmittedRows : List Nat :=
  ownership.fragments.flatMap fun fragment => fragment.emittedRows.indices

def fragmentOwns (fragment : LoweredFragment) (row : Nat) : Bool :=
  fragment.emittedRows.indices.contains row

def owners (row : Nat) : List LoweredFragment :=
  ownership.fragments.filter fun fragment => fragmentOwns fragment row

def expectedFamily : Disposition → Option RawFamily
  | .retained => some .retained
  | .rewrite _ .polynomialEvaluation => some .polynomialEvaluation
  | .rewrite _ .productSum => some .productSum
  | .rewrite _ .linearDefinition => none

def ownerAndFamilyAgree (row : Materialized.RawRow) : Bool :=
  match owners row.emittedRow with
  | [fragment] => expectedFamily fragment.disposition == some row.family
  | _ => false

private def strictlyIncreasingCheck : List Nat → Bool
  | [] => true
  | [_] => true
  | first :: second :: rest =>
      decide (first < second) && strictlyIncreasingCheck (second :: rest)

private theorem strictlyIncreasingCheck_eq_true_iff (values : List Nat) :
    strictlyIncreasingCheck values = true ↔ YZcol.StrictlyIncreasing values := by
  induction values with
  | nil => simp [strictlyIncreasingCheck, YZcol.StrictlyIncreasing]
  | cons first rest inductionHypothesis =>
      cases rest with
      | nil => simp [strictlyIncreasingCheck, YZcol.StrictlyIncreasing]
      | cons second tail =>
          simp [strictlyIncreasingCheck, YZcol.StrictlyIncreasing,
            inductionHypothesis]

theorem rowCount : rawRows.length = 1254 := by
  native_decide

theorem relationRowCountAgreement :
    Generated.SelectiveMatrixRows.finalRelationRows =
      ownership.finalRelationRowCount := by
  native_decide

theorem distinguishedColumns :
    Generated.SelectiveMatrixRows.constantOneColumn = 0 ∧
      Generated.SelectiveMatrixRows.steadySelectorColumn = 272 := by
  native_decide

/-- All compact records use the generated final relation dimensions, the
thirteen-port vocabulary, and the recursive steady arm. -/
theorem fixedShape :
    ∀ row ∈ rawRows,
      row.schemaVersion = 1 ∧
      row.rows = Generated.SelectiveMatrixRows.finalRelationRows ∧
      row.columns = Generated.SelectiveMatrixRows.finalRelationColumns ∧
      row.emittedRow < row.rows ∧
      row.arm = some 2 ∧
      row.ports.length = 13 := by
  native_decide

theorem emittedRowsStrictlyIncreasing :
    YZcol.StrictlyIncreasing emittedRows := by
  apply (strictlyIncreasingCheck_eq_true_iff emittedRows).mp
  native_decide

theorem emittedRowsNodup : emittedRows.Nodup :=
  YZcol.strictlyIncreasing_nodup emittedRowsStrictlyIncreasing

theorem expectedRowCount : expectedEmittedRows.length = 1254 := by
  native_decide

/-- Sorting erases only traversal order. Equality therefore checks the exact
physical emitted-row multiset against expansion of all nonempty fragment
intervals. -/
theorem exactEmittedRows :
    emittedRows.mergeSort (fun left right => decide (left ≤ right)) =
      expectedEmittedRows.mergeSort (fun left right => decide (left ≤ right)) := by
  native_decide

/-- Every materialized row is owned by exactly one source obligation fragment,
and its diagnostic family agrees with that owner. -/
theorem uniqueOwner :
    ∀ row ∈ rawRows, (owners row.emittedRow).length = 1 := by
  native_decide

theorem uniqueOwnerAndFamily :
    ∀ row ∈ rawRows, ownerAndFamilyAgree row = true := by
  native_decide

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Census
