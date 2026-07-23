import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.TerminalArtifact.Certificates

/-!
Concrete generated-row bridge for the fixed production combined-NC terminal.

Owns: kernel composition of the 52 bounded certificates, exact transport of
source-row satisfaction to the independent 6,595-row terminal program, and
structural recovery of canonicality from the 52 bounded definition shards.

Does not own: source-to-selective lowering, padding truth, transcript order,
parent/child authority, commitment binding, costs, or row removal.

Assurance tier: artifact-checked for the fixed generated terminal interval.
-/

/-!
Emits constraints: none; this module checks the generated terminal-row artifact.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.terminal_artifact` | Establish exact ownership and decoding for generated terminal rows. | checked artifact |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.TerminalArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ProjectionIndexedRows
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized
open Certificates

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.TerminalProgram

private theorem splitRemainder_length {Alpha : Type}
    (lengths : List Nat) (values : List Alpha) :
    (splitRemainder lengths values).length = lengths.length + 1 := by
  induction lengths generalizing values with
  | nil => rfl
  | cons length lengths inductionHypothesis =>
      simp only [splitRemainder, List.length_cons]
      rw [inductionHypothesis]

private theorem splitRemainder_flatten {Alpha : Type}
    (lengths : List Nat) (values : List Alpha) :
    (splitRemainder lengths values).flatten = values := by
  induction lengths generalizing values with
  | nil => simp [splitRemainder]
  | cons length lengths inductionHypothesis =>
      simp only [splitRemainder, List.flatten_cons]
      rw [inductionHypothesis]
      exact List.take_append_drop length values

theorem expectedRowShardCount :
    expectedRowShards.length = terminalShardCount := by
  rw [expectedRowShards, splitRemainder_length]
  simp [terminalPrefixLengths, terminalShardCount]

theorem expectedDefinitionShardCount :
    expectedDefinitionShards.length = terminalShardCount := by
  rw [expectedDefinitionShards, splitRemainder_length]
  simp [terminalPrefixLengths, terminalShardCount]

private theorem ofFn_getD_eq_self
    {Alpha : Type} {count : Nat} (values : List Alpha) (default : Alpha)
    (lengthEq : values.length = count) :
    (List.ofFn fun index : Fin count => values.getD index.val default) =
      values := by
  apply List.ext_get
  · simp [lengthEq]
  · intro index leftLt rightLt
    simp only [List.get_eq_getElem, List.getElem_ofFn]
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem rightLt]
    rfl

def generatedTerminalSourceShards : List (List RawSourceRow) :=
  List.ofFn fun index : Fin terminalShardCount => sourceShard index.val

/-- The exact 6,595 generated records, without ever forming them inside one
native certificate. -/
def generatedTerminalRows : List RawSourceRow :=
  generatedTerminalSourceShards.flatten

def generatedTerminalRowShards : List (List Row) :=
  List.ofFn fun index : Fin terminalShardCount =>
    rawRows (sourceShard index.val)

def programRowShards : List (List Row) :=
  List.ofFn fun index : Fin terminalShardCount =>
    expectedRowShard index.val

def programDefinitionShards : List (List Definition) :=
  List.ofFn fun index : Fin terminalShardCount =>
    expectedDefinitionShard index.val

theorem programRowShards_eq_expected :
    programRowShards = expectedRowShards := by
  exact ofFn_getD_eq_self expectedRowShards []
    expectedRowShardCount

theorem programDefinitionShards_eq_expected :
    programDefinitionShards = expectedDefinitionShards := by
  exact ofFn_getD_eq_self expectedDefinitionShards []
    expectedDefinitionShardCount

theorem expectedRowShards_flatten :
    expectedRowShards.flatten = rows :=
  splitRemainder_flatten terminalPrefixLengths rows

theorem expectedDefinitionShards_flatten :
    expectedDefinitionShards.flatten = definitions :=
  splitRemainder_flatten terminalPrefixLengths definitions

theorem generatedTerminalShardRowIds (index : Fin terminalShardCount) :
    (sourceShard index.val).map RawSourceRow.sourceRow =
      List.range' (shardStart index.val) (shardLength index.val) :=
  (rowCertificateAt index.val index.isLt).2.1

theorem generatedTerminalShardLength (index : Fin terminalShardCount) :
    (sourceShard index.val).length = shardLength index.val :=
  (rowCertificateAt index.val index.isLt).1

theorem generatedTerminalRows_finalBoundary :
    ((sourceShard 51).take 83).map RawSourceRow.sourceRow =
        List.range' (rawBoundary.terminalFinalEqualityRows.start - 83) 83 ∧
      ((sourceShard 51).drop 83).map RawSourceRow.sourceRow =
        [rawBoundary.terminalFinalEqualityRows.start,
          rawBoundary.terminalFinalEqualityRows.start + 1] :=
  finalShardBoundary

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

private theorem rowsPermutationEquivalentList_ofFn_flatten :
    ∀ count (sources expected : Fin count → List Row),
      (∀ index, RowsPermutationEquivalentList
        (sources index) (expected index)) →
      RowsPermutationEquivalentList
        (List.ofFn sources).flatten (List.ofFn expected).flatten
  | 0, _, _, _ => trivial
  | count + 1, sources, expected, related => by
      rw [List.ofFn_succ, List.ofFn_succ,
        List.flatten_cons, List.flatten_cons]
      exact rowsPermutationEquivalentList_append (related 0)
        (rowsPermutationEquivalentList_ofFn_flatten count
          (fun index => sources index.succ)
          (fun index => expected index.succ)
          (fun index => related index.succ))

private theorem rawRows_generatedTerminalRows :
    rawRows generatedTerminalRows = generatedTerminalRowShards.flatten := by
  simp [rawRows, generatedTerminalRows, generatedTerminalSourceShards,
    generatedTerminalRowShards, Function.comp_def]

/-- The physical generated terminal rows and the independent terminal program
have lockstep equations, modulo sparse-term ordering within each A/B/C side. -/
theorem terminalProgramRows_exact :
    RowsPermutationEquivalentList
      (rawRows generatedTerminalRows) rows := by
  have flattened : RowsPermutationEquivalentList
      generatedTerminalRowShards.flatten programRowShards.flatten := by
    exact rowsPermutationEquivalentList_ofFn_flatten terminalShardCount
      (fun index => rawRows (sourceShard index.val))
      (fun index => expectedRowShard index.val)
      (fun index =>
        (rowCertificateAt index.val index.isLt).2.2.2)
  rw [programRowShards_eq_expected, expectedRowShards_flatten] at flattened
  rw [rawRows_generatedTerminalRows]
  exact flattened

private theorem all_ofFn
    {Alpha : Type} (Property : Alpha → Prop) :
    ∀ count (values : Fin count → Alpha),
      (∀ index, Property (values index)) →
      ∀ value ∈ List.ofFn values, Property value
  | 0, _, _ => by simp
  | count + 1, values, holds => by
      intro value member
      simp only [List.ofFn_succ, List.mem_cons] at member
      rcases member with rfl | member
      · exact holds 0
      · exact all_ofFn Property count
          (fun index => values index.succ)
          (fun index => holds index.succ)
          value member

private theorem definitionsCanonical_flatten
    {shards : List (List Definition)}
    (canonical :
      ∀ shard ∈ shards, DefinitionsCanonical shard) :
    DefinitionsCanonical shards.flatten := by
  intro definition member
  rcases List.mem_flatten.mp member with
    ⟨shard, shardMember, definitionMember⟩
  exact canonical shard shardMember definition definitionMember

/-- All 6,593 deterministic definitions are canonical. The proof only
composes the independently checked 110/128/83-record certificates. -/
theorem terminalDefinitionsCanonical :
    ∀ definition ∈ definitions, definition.Canonical := by
  have shardCanonical :
      ∀ shard ∈ programDefinitionShards, DefinitionsCanonical shard := by
    apply all_ofFn DefinitionsCanonical
    intro index
    have bound : index.val < terminalShardCount := index.isLt
    exact (definitionCertificateAt index.val bound).2
  have flattened := definitionsCanonical_flatten shardCanonical
  rw [programDefinitionShards_eq_expected,
    expectedDefinitionShards_flatten] at flattened
  exact flattened

private theorem rowHolds_iff_of_permutation
    (assignment : Nat → Nat) {left right : Row}
    (permutation : RowsPermutationEquivalent left right) :
    RowHolds assignment left ↔ RowHolds assignment right := by
  unfold RowHolds
  rw [lcEval_eq_of_perm assignment permutation.1,
    lcEval_eq_of_perm assignment permutation.2.1,
    lcEval_eq_of_perm assignment permutation.2.2]

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

def GeneratedTerminalRowsSatisfy (assignment : Nat → Nat) : Prop :=
  Satisfies (rawRows generatedTerminalRows) assignment

/-- Exact generated source-row satisfaction yields the independent terminal
semantics. The only external arithmetic premises are canonical assignment
values and the verifier's constant-one column. -/
theorem generatedTerminalRowsSatisfy_implies_computed
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (satisfies : GeneratedTerminalRowsSatisfy assignment) :
    Computed assignment := by
  have programSatisfies : Satisfies rows assignment :=
    (satisfies_iff_of_rowsPermutationEquivalentList assignment
      terminalProgramRows_exact).mp satisfies
  exact sound canonical constantOne
    terminalDefinitionsCanonical programSatisfies

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.TerminalArtifact
