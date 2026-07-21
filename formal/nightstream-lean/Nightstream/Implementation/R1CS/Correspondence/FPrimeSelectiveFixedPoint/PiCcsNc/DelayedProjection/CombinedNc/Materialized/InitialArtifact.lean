import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows.Chunk2
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows.Chunk3
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows.Chunk4
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows.Chunk5
import Nightstream.Implementation.R1CS.Artifacts.Projection.IndexedRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.InitialProgram
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDecodeBridge

/-!
Exact generated-row refinement for the production combined-NC claimed-initial
formula.

Owns: the 376 physical source rows at generated source-list positions
300 through 675, their exact absolute source indices, sparse A/B/C equality
with `InitialProgram`, and transport of their satisfaction to the typed
boundary formula.

Does not own: source-to-selective compiler soundness, assignment synthesis,
constant-one enforcement, transcript scheduling, pending-parent authority,
commitment binding, costs, or row removal.

The interval is split as 84/128/128/36 proof-free records. This is a data
certificate split only; mathematical ownership remains one coherent formula.

Assurance tier: artifact-checked for this fixed generated profile once this
module and its parent validate.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.InitialArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.ProjectionIndexedRows
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDecodeBridge
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Semantics

def shard0 : List RawSourceRow := SourceRows.Chunk2.values.drop 44
def shard1 : List RawSourceRow := SourceRows.Chunk3.values
def shard2 : List RawSourceRow := SourceRows.Chunk4.values
def shard3 : List RawSourceRow := SourceRows.Chunk5.values.take 36

/-- Exact local reconstruction of aggregate generated source positions
`300 ..< 676`: `2 * 128 + 44 = 300`, followed by 376 records. -/
def claimedInitialRows : List RawSourceRow :=
  shard0 ++ (shard1 ++ (shard2 ++ shard3))

def rawRows (rows : List RawSourceRow) : List Row :=
  rows.map SourceDecodeBridge.rawRow

def expected0 : List Row := InitialProgram.rows.take 84
def expected1 : List Row := (InitialProgram.rows.drop 84).take 128
def expected2 : List Row := (InitialProgram.rows.drop 212).take 128
/-- Deliberately no `take`: the final certificate rejects an oversized
remainder instead of silently truncating it. -/
def expected3 : List Row := InitialProgram.rows.drop 340

private def rowsPermutationEquivalentListDecidable :
    (source reconstructed : List Row) →
      Decidable (RowsPermutationEquivalentList source reconstructed)
  | [], [] => isTrue True.intro
  | [], _ :: _ => isFalse id
  | _ :: _, [] => isFalse id
  | source :: sources, reconstructed :: reconstructions =>
      match inferInstanceAs
          (Decidable (RowsPermutationEquivalent source reconstructed)),
        rowsPermutationEquivalentListDecidable sources reconstructions with
      | isTrue head, isTrue tail => isTrue ⟨head, tail⟩
      | isFalse head, isTrue _ => isFalse fun equivalent => head equivalent.1
      | isTrue _, isFalse tail => isFalse fun equivalent => tail equivalent.2
      | isFalse head, isFalse _ => isFalse fun equivalent => head equivalent.1

local instance (source reconstructed : List Row) :
    Decidable (RowsPermutationEquivalentList source reconstructed) :=
  rowsPermutationEquivalentListDecidable source reconstructed

/-- Fail-closed shard predicate. Absolute row indices are checked alongside
the sparse equations; neither a stage label nor a contiguous range alone can
establish program meaning. -/
def ShardValid (start : Nat) (source : List RawSourceRow)
    (expected : List Row) : Prop :=
  source.map RawSourceRow.sourceRow =
      ((List.range source.length).map fun offset => start + offset) ∧
  (∀ row ∈ source,
    RawSourceRowValid row ∧
      row.rows = InitialProgram.rawBoundary.sourceRows ∧
      row.columns = InitialProgram.rawBoundary.sourceColumns) ∧
  RowsPermutationEquivalentList (rawRows source) expected

local instance (start : Nat) (source : List RawSourceRow)
    (expected : List Row) : Decidable (ShardValid start source expected) := by
  unfold ShardValid RawSourceRowValid RawTermValid
  infer_instance

/-! Each certificate below evaluates only proof-free sparse data:

* `shard0Certificate`: 84 `RawSourceRow`/`Row` pairs;
* `shard1Certificate`: 128 `RawSourceRow`/`Row` pairs;
* `shard2Certificate`: 128 `RawSourceRow`/`Row` pairs;
* `shard3Certificate`: 36 `RawSourceRow`/`Row` pairs.

The maximum is 128 records. The last expected shard is the complete remainder,
so the certificates jointly reject overlap, gaps, and an oversized tail.
-/

set_option maxRecDepth 100000 in
theorem shard0Certificate : ShardValid 3972968 shard0 expected0 := by
  native_decide

set_option maxRecDepth 100000 in
theorem shard1Certificate : ShardValid 3973052 shard1 expected1 := by
  native_decide

set_option maxRecDepth 100000 in
theorem shard2Certificate : ShardValid 3973180 shard2 expected2 := by
  native_decide

set_option maxRecDepth 100000 in
theorem shard3Certificate : ShardValid 3973308 shard3 expected3 := by
  native_decide

private theorem rowsPermutationEquivalentList_append
    {leftSource rightSource leftExpected rightExpected : List Row}
    (left : RowsPermutationEquivalentList leftSource leftExpected)
    (right : RowsPermutationEquivalentList rightSource rightExpected) :
    RowsPermutationEquivalentList (leftSource ++ rightSource)
      (leftExpected ++ rightExpected) := by
  induction leftSource generalizing leftExpected with
  | nil =>
      cases leftExpected with
      | nil => simpa using right
      | cons _ _ => simp [RowsPermutationEquivalentList] at left
  | cons source sources inductionHypothesis =>
      cases leftExpected with
      | nil => simp [RowsPermutationEquivalentList] at left
      | cons expected expecteds =>
          change RowsPermutationEquivalent source expected ∧
            RowsPermutationEquivalentList sources expecteds at left
          change RowsPermutationEquivalent source expected ∧
            RowsPermutationEquivalentList
              (sources ++ rightSource) (expecteds ++ rightExpected)
          exact ⟨left.1, inductionHypothesis left.2⟩

private theorem splitAtThree (rows : List Row) :
    rows = rows.take 84 ++
      ((rows.drop 84).take 128 ++
        ((rows.drop 212).take 128 ++ rows.drop 340)) := by
  calc
    rows = rows.take 84 ++ rows.drop 84 :=
      (List.take_append_drop 84 rows).symm
    _ = rows.take 84 ++
        ((rows.drop 84).take 128 ++ (rows.drop 84).drop 128) := by
      exact congrArg (fun tail => rows.take 84 ++ tail)
        (List.take_append_drop 128 (rows.drop 84)).symm
    _ = rows.take 84 ++
        ((rows.drop 84).take 128 ++ rows.drop 212) := by
      simp only [List.drop_drop]
    _ = rows.take 84 ++
        ((rows.drop 84).take 128 ++
          ((rows.drop 212).take 128 ++ (rows.drop 212).drop 128)) := by
      exact congrArg
        (fun tail => rows.take 84 ++ ((rows.drop 84).take 128 ++ tail))
        (List.take_append_drop 128 (rows.drop 212)).symm
    _ = rows.take 84 ++
        ((rows.drop 84).take 128 ++
          ((rows.drop 212).take 128 ++ rows.drop 340)) := by
      simp only [List.drop_drop]

theorem initialProgramRows_exact :
    RowsPermutationEquivalentList (rawRows claimedInitialRows)
      InitialProgram.rows := by
  have combined := rowsPermutationEquivalentList_append
    shard0Certificate.2.2
    (rowsPermutationEquivalentList_append shard1Certificate.2.2
      (rowsPermutationEquivalentList_append shard2Certificate.2.2
        shard3Certificate.2.2))
  rw [claimedInitialRows, rawRows, List.map_append, List.map_append,
    List.map_append, splitAtThree InitialProgram.rows]
  simpa [expected0, expected1, expected2, expected3] using combined

private theorem rowHolds_iff_of_permutation
    (assignment : Nat → Nat) {left right : Row}
    (permutation : RowsPermutationEquivalent left right) :
    RowHolds assignment left ↔ RowHolds assignment right := by
  unfold RowHolds
  rw [lcEval_eq_of_perm assignment permutation.1,
    lcEval_eq_of_perm assignment permutation.2.1,
    lcEval_eq_of_perm assignment permutation.2.2]

/-- Generic kernel transport across lockstep sparse-term permutations. -/
private theorem satisfies_iff_of_rowsPermutationEquivalentList
    (assignment : Nat → Nat) {left right : List Row}
    (permutation : RowsPermutationEquivalentList left right) :
    Satisfies left assignment ↔ Satisfies right assignment := by
  induction left generalizing right with
  | nil =>
      cases right with
      | nil => simp [Satisfies]
      | cons _ _ => simp [RowsPermutationEquivalentList] at permutation
  | cons leftHead leftTail inductionHypothesis =>
      cases right with
      | nil => simp [RowsPermutationEquivalentList] at permutation
      | cons rightHead rightTail =>
          change RowsPermutationEquivalent leftHead rightHead ∧
            RowsPermutationEquivalentList leftTail rightTail at permutation
          simp only [Satisfies, List.mem_cons]
          constructor
          · intro holds row member
            rcases member with rfl | member
            · exact (rowHolds_iff_of_permutation assignment
                permutation.1).mp (holds leftHead (by simp))
            · have tailHolds : Satisfies leftTail assignment := by
                intro candidate candidateMember
                exact holds candidate (by simp [candidateMember])
              exact (inductionHypothesis permutation.2).mp tailHolds row member
          · intro holds row member
            rcases member with rfl | member
            · exact (rowHolds_iff_of_permutation assignment
                permutation.1).mpr (holds rightHead (by simp))
            · have tailHolds : Satisfies rightTail assignment := by
                intro candidate candidateMember
                exact holds candidate (by simp [candidateMember])
              exact (inductionHypothesis permutation.2).mpr tailHolds row member

/-- Satisfaction of the exact generated 376-row interval forces the typed
production boundary formula. The pending-parent values remain ordinary
assignment reads here; their authority is deliberately owned by the later
state/commitment composition. -/
theorem sourceRows_imply_boundaryClaimedInitial
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (satisfies : Satisfies (rawRows claimedInitialRows) assignment) :
    boundaryClaimedInitial productionBoundary assignment =
      K.mul (boundaryBatchWeight productionBoundary assignment)
        (Nightstream.SuperNeo.ProjectionCheck.eval K.ops
          (boundaryPendingParentYZcol productionBoundary assignment)
          (boundaryProducerBeta productionBoundary assignment)) := by
  have programSatisfies : Satisfies InitialProgram.rows assignment :=
    (satisfies_iff_of_rowsPermutationEquivalentList assignment
      initialProgramRows_exact).mp satisfies
  have computed := InitialProgram.sound canonical constantOne programSatisfies
  simpa [boundaryClaimedInitial, boundaryBatchWeight,
    boundaryPendingParentYZcol, boundaryProducerBeta,
    InitialProgram.claimedInitialColumns,
    InitialProgram.batchWeightColumns, InitialProgram.pendingColumns,
    InitialProgram.producerBetaColumns, InitialProgram.rawBoundary,
    productionBoundary_raw] using computed

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.InitialArtifact
