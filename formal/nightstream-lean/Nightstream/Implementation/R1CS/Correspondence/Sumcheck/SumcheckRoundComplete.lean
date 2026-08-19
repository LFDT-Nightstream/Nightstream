import Nightstream.Implementation.R1CS.Correspondence.Sumcheck.SumcheckChainSound

/-!
Interpreter-witness completeness for the production SumCheck round compiler.

The public witness is a source assignment and the result of running the exact
SSA interpreter.  Decoded `Accepted` supplies the two verifier assertions.
No row-satisfaction predicate occurs in the witness.
-/

namespace Nightstream.Implementation.R1CS.SumcheckRoundSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.SumcheckRoundArtifact

private def lowTerms : List (Nat × Nat) :=
  [(2, 2), (3, 1), (4, 1), (5, 1), (6, 1)]

private def highTerms : List (Nat × Nat) :=
  [(8, 2), (9, 1), (10, 1), (11, 1), (12, 1)]

private theorem lowTermsValue (assignment : Nat → Nat) :
    residue (lcEval assignment lowTerms) =
      baseAt assignment 2 + (baseAt assignment 2 +
        (baseAt assignment 3 + (baseAt assignment 4 +
          (baseAt assignment 5 + baseAt assignment 6)))) := by
  apply Fin.ext
  simp [lowTerms, lcEval, residue, baseAt, Fin.val_add]
  congr 1
  omega

private theorem highTermsValue (assignment : Nat → Nat) :
    residue (lcEval assignment highTerms) =
      baseAt assignment 8 + (baseAt assignment 8 +
        (baseAt assignment 9 + (baseAt assignment 10 +
          (baseAt assignment 11 + baseAt assignment 12)))) := by
  apply Fin.ext
  simp [highTerms, lcEval, residue, baseAt, Fin.val_add]
  congr 1
  omega

private theorem initialEqualities
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (accepted : Accepted assignment) :
    assignment 1 = lcEval assignment lowTerms ∧
      assignment 7 = lcEval assignment highTerms := by
  have initial := accepted.initial
  unfold claimInValue polynomial coefficientValues at initial
  simp only [coefficientColumns, List.map_cons, List.map_nil,
    Nightstream.SuperNeo.ProjectionCheck.eval, List.foldr_cons,
    List.foldr_nil, K.ops, columns, claimInColumns,
    K.zero_mul, K.one_mul, K.add_zero] at initial
  have low := congrArg K.c0 initial
  have high := congrArg K.c1 initial
  simp only [KColumns.value, K.add] at low high
  have lowField : baseAt assignment 1 = residue (lcEval assignment lowTerms) :=
    low.trans (lowTermsValue assignment).symm
  have highField : baseAt assignment 7 = residue (lcEval assignment highTerms) :=
    high.trans (highTermsValue assignment).symm
  constructor
  · have equality := congrArg Fin.val lowField
    simp only [baseAt, residue] at equality
    rw [Nat.mod_eq_of_lt (canonical 1)] at equality
    have valueLt : lcEval assignment lowTerms < goldilocksP := by
      unfold lcEval
      exact Nat.mod_lt _ (by decide)
    simpa [Nat.mod_eq_of_lt valueLt] using equality
  · have equality := congrArg Fin.val highField
    simp only [baseAt, residue] at equality
    rw [Nat.mod_eq_of_lt (canonical 7)] at equality
    have valueLt : lcEval assignment highTerms < goldilocksP := by
      unfold lcEval
      exact Nat.mod_lt _ (by decide)
    simpa [Nat.mod_eq_of_lt valueLt] using equality

private theorem checks_complete
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : Accepted assignment) :
    Satisfies (checks instructions) assignment := by
  have equalities := initialEqualities canonical accepted
  have lowRow : RowHolds assignment (builderLinearRow 1 lowTerms) :=
    builderLinearRow_complete one 1 lowTerms (by
      simp [lowTerms, CanonicalTerms, goldilocksP]) equalities.1
  have highRow : RowHolds assignment (builderLinearRow 7 highTerms) :=
    builderLinearRow_complete one 7 highTerms (by
      simp [highTerms, CanonicalTerms, goldilocksP]) equalities.2
  intro row member
  have checksShape : checks instructions =
      [builderLinearRow 1 lowTerms, builderLinearRow 7 highTerms] := by
    native_decide
  rw [checksShape] at member
  simp at member
  rcases member with rfl | rfl
  · exact lowRow
  · exact highRow

/-- Actual interpreter execution plus decoded round acceptance. -/
structure ExecutionWitness (assignment : Nat → Nat) where
  source : Nat → Nat
  sourceCanonical : ∀ column, source column < goldilocksP
  sourceOne : source 0 = 1
  executed : interpret source instructions = assignment
  accepted : Accepted assignment

/-- One honest native round execution satisfies every exact emitted row. -/
theorem native_complete
    {assignment : Nat → Nat}
    (witness : ExecutionWitness assignment) :
    Satisfies rows assignment := by
  have checksHold : ChecksHold witness.source instructions := by
    unfold ChecksHold
    rw [witness.executed]
    exact checks_complete (by
      rw [← witness.executed]
      exact run_canonical witness.sourceCanonical) (by
      rw [← witness.executed]
      have preserves := run_preserves_known definitions_wellFormed witness.source
      exact (preserves 0 input_has_one).trans witness.sourceOne)
      witness.accepted
  have compiled := CheckedProgram.complete definitions_wellFormed
    definitions_canonical witness.sourceCanonical input_has_one
    witness.sourceOne checksHold
  rw [witness.executed] at compiled
  exact compiled

/-- Affine-renamed production round completeness. -/
theorem mapped_native_complete
    (columnMap : List Nat)
    {assignment : Nat → Nat}
    (witness : ExecutionWitness
      (Relabel.assignment columnMap assignment)) :
    Satisfies (rows.map (Relabel.row columnMap)) assignment := by
  apply (Relabel.satisfies_mapped_iff rows columnMap assignment).mpr
  exact native_complete witness

end Nightstream.Implementation.R1CS.SumcheckRoundSound

namespace Nightstream.Implementation.R1CS.SumcheckChainSound

open Nightstream.Implementation.R1CS

/-- Honest interpreter execution for every mapped round in a chain. -/
structure ExecutionWitness
    (maps : List ColumnMap) (assignment : Nat → Nat) where
  round : ∀ columnMap, columnMap ∈ maps →
    SumcheckRoundSound.ExecutionWitness
      (mappedAssignment columnMap assignment)

/-- Per-round interpreter witnesses reconstruct the entire exact FE/NC row
family. -/
theorem complete
    {maps : List ColumnMap} {assignment : Nat → Nat}
    (witness : ExecutionWitness maps assignment) :
    Holds maps assignment := by
  intro columnMap member
  exact SumcheckRoundSound.mapped_native_complete columnMap
    (witness.round columnMap member)

end Nightstream.Implementation.R1CS.SumcheckChainSound
