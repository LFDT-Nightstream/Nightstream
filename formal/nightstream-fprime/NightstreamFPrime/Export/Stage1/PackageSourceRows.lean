import NightstreamFPrime.Export.Stage1.Rows

/-!
Owns fail-closed reconstruction of ordinary R1CS source rows from the two
classified row fields stored in a circuit package. A row is returned only
when its physical row index has exactly one owner.

This module does not select Stage 1 phases or construct package rows.
-/

namespace NightstreamFPrime.Export.Stage1.PackageSourceRows

open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout

/-- Reconstruct the typed compiled rows carried by the two package fields. -/
def decodedRows (instructions : List WitnessInstruction)
    (assertions : List SparseRow) : List Rows.CompiledRow :=
  instructions.map Rows.CompiledRow.witness ++
    assertions.map Rows.CompiledRow.assertion

/-- Return one ordinary source row only when its physical row index has one
exact owner. Missing and duplicate ownership both fail. -/
def sourceRow? (instructions : List WitnessInstruction)
    (assertions : List SparseRow) (rowIndex : Nat) : Option R1CS.Row :=
  match (decodedRows instructions assertions).filter
      (fun row => row.rowIndex == rowIndex) with
  | [row] => some row.toR1CS
  | _ => none

/-- Source-row view of the exact classified fields carried by one package. -/
def packageSourceRow? (package : CircuitPackage) : Nat → Option R1CS.Row :=
  sourceRow? package.witnessInstructions package.assertionRows

def classifiedRows (rows : List Rows.CompiledRow) : List Rows.CompiledRow :=
  decodedRows (Rows.witnessInstructions rows) (Rows.assertionRows rows)

/-- Appending two independently classified fields is the same row multiset
as appending their two decoded row sets. -/
theorem decodedRows_append_perm
    (leftInstructions rightInstructions : List WitnessInstruction)
    (leftAssertions rightAssertions : List SparseRow) :
    List.Perm
      (decodedRows (leftInstructions ++ rightInstructions)
        (leftAssertions ++ rightAssertions))
      (decodedRows leftInstructions leftAssertions ++
        decodedRows rightInstructions rightAssertions) := by
  simp only [decodedRows, List.map_append]
  simpa only [List.append_assoc] using
    (List.perm_append_comm_assoc
      (rightInstructions.map Rows.CompiledRow.witness)
      (leftAssertions.map Rows.CompiledRow.assertion)
      (rightAssertions.map Rows.CompiledRow.assertion)).append_left
        (leftInstructions.map Rows.CompiledRow.witness)

/-- Stable classification preserves every compiled row exactly once. -/
theorem classifiedRows_perm (rows : List Rows.CompiledRow) :
    List.Perm (classifiedRows rows) rows := by
  induction rows with
  | nil => rfl
  | cons row rest inductionHypothesis =>
      cases row with
      | witness instruction =>
          exact List.Perm.cons _ inductionHypothesis
      | assertion assertion =>
          exact List.perm_middle.trans
            (List.Perm.cons _ inductionHypothesis)

private theorem filter_rowIndex_eq_singleton
    (rows : List Rows.CompiledRow)
    (unique : (rows.map Rows.CompiledRow.rowIndex).Nodup)
    (target : Rows.CompiledRow) (member : target ∈ rows) :
    rows.filter (fun row => row.rowIndex == target.rowIndex) = [target] := by
  induction rows with
  | nil => simp at member
  | cons head tail inductionHypothesis =>
      have headAbsent : head.rowIndex ∉
          tail.map Rows.CompiledRow.rowIndex :=
        (List.nodup_cons.mp unique).1
      have tailUnique :
          (tail.map Rows.CompiledRow.rowIndex).Nodup :=
        (List.nodup_cons.mp unique).2
      rcases List.mem_cons.mp member with targetEq | targetMember
      · subst target
        have tailFiltered :
            tail.filter (fun row => row.rowIndex == head.rowIndex) = [] := by
          rw [List.filter_eq_nil_iff]
          intro candidate candidateMember
          have different : candidate.rowIndex ≠ head.rowIndex := by
            intro equal
            apply headAbsent
            have mapped : candidate.rowIndex ∈
                tail.map Rows.CompiledRow.rowIndex :=
              List.mem_map_of_mem candidateMember
            simpa [equal] using mapped
          simp [different]
        simp [tailFiltered]
      · have different : head.rowIndex ≠ target.rowIndex := by
          intro equal
          apply headAbsent
          have mapped : target.rowIndex ∈
              tail.map Rows.CompiledRow.rowIndex :=
            List.mem_map_of_mem targetMember
          simpa [equal] using mapped
        simp [different,
          inductionHypothesis tailUnique targetMember]

/-- Any uniquely owned decoded package row is recovered exactly. -/
theorem sourceRow?_eq_some
    (instructions : List WitnessInstruction) (assertions : List SparseRow)
    (unique : ((decodedRows instructions assertions).map
      Rows.CompiledRow.rowIndex).Nodup)
    (target : Rows.CompiledRow)
    (member : target ∈ decodedRows instructions assertions) :
    sourceRow? instructions assertions target.rowIndex =
      some target.toR1CS := by
  have selected := filter_rowIndex_eq_singleton
    (decodedRows instructions assertions) unique target member
  unfold sourceRow?
  rw [selected]

/-- The classified package fields recover any row from a duplicate-free
compiled source list exactly. -/
theorem sourceRow?_classified
    (rows : List Rows.CompiledRow)
    (unique : (rows.map Rows.CompiledRow.rowIndex).Nodup)
    (target : Rows.CompiledRow) (member : target ∈ rows) :
    sourceRow? (Rows.witnessInstructions rows) (Rows.assertionRows rows)
        target.rowIndex = some target.toR1CS := by
  have permutation :=
    (classifiedRows_perm rows).filter
      (fun row => row.rowIndex == target.rowIndex)
  have reference := filter_rowIndex_eq_singleton rows unique target member
  rw [reference] at permutation
  have selected :
      (classifiedRows rows).filter
        (fun row => row.rowIndex == target.rowIndex) = [target] := by
    simpa using permutation
  unfold sourceRow?
  change (match (classifiedRows rows).filter
      (fun row => row.rowIndex == target.rowIndex) with
    | [row] => some row.toR1CS
    | _ => none) = some target.toR1CS
  rw [selected]

end NightstreamFPrime.Export.Stage1.PackageSourceRows
