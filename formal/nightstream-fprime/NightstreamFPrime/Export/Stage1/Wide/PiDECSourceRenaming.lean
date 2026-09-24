import NightstreamFPrime.Export.Stage1.PiDECLeafRenaming
import NightstreamFPrime.Export.Stage1.Wide.SourceAssignment

/-! Restore PiDEC source addresses to the reference layout. This is a proof
view of the physical rows; it does not allocate the removed sampler witness. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PiDECSourceRenaming

open NightstreamFPrime.Circuit NightstreamFPrime.Layout NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open CompactRows

/-- Parent products and local fields have separate source offsets. -/
def restore (column : Nat) : Nat :=
  if column < Layout.Stage1.Wide.PiDECInputs.proofInputStart then column + 208165
  else column + 925480

theorem restore_location (location : PiDECDirectPlan.Location) :
    restore (PiDECSource.column location) = location.sourceColumn := by
  rw [PiDECSource.source_offsets]
  cases location with
  | parentCommitment index =>
    apply if_pos
    have bound : index.val < 1188 := index.isLt
    change 19587528 + index.val < 27496062
    omega
  | parentPublicInput index =>
    apply if_pos
    have bound : index.val < 270 := index.isLt
    change 19593036 + index.val < 27496062
    omega
  | parentEvalK index =>
    apply if_pos
    have bound : index.val < 108 := index.isLt
    change 19595034 + index.val < 27496062
    omega
  | parentEvalA index =>
    apply if_pos
    have bound : index.val < 1512 := index.isLt
    change 19619334 + index.val < 27496062
    omega
  | proof index =>
    apply if_neg
    change ¬ 27496062 + index.val < 27496062
    omega
  | logical index =>
    apply if_neg
    change ¬ 27545310 + index.val < 27496062
    omega
  | fresh index =>
    apply if_neg
    change ¬ 27545580 + index.val < 27496062
    omega

theorem sourceEnv_restore (env : Env) (location : PiDECDirectPlan.Location) :
    SourceAssignment.sourceEnv env (restore (PiDECSource.column location)) =
      env (PiDECSource.column location) := by
  rw [restore_location, SourceAssignment.sourceEnv_piDec]
  rfl

variable {width : Nat}
  {fits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}

private abbrev oldInterface := Layout.Stage1.PiDECInputs.interface width fits
private abbrev newInterface := Layout.Stage1.Wide.PiDECInputs.interface width fits
private abbrev oldStart := Layout.Stage1.PiDECInputs.phaseOffset
private abbrev newStart := Layout.Stage1.Wide.PiDECInputs.phaseOffset

private theorem sign_restore (index : Nat) (bounded : index < 270) :
    restore (newStart + index) = oldStart + index := by
  exact restore_location (.logical ⟨index, bounded⟩)

private theorem parent_public (coordinate : Fin (PiDEC.v1_1.PublicInputSplit.coordinateCount width fits)) :
    renameExpr restore (((newInterface (width := width) (fits := fits)).parent newStart).publicInput coordinate) =
      ((oldInterface (width := width) (fits := fits)).parent oldStart).publicInput coordinate := by
  exact congrArg Expr.var (restore_location (.parentPublicInput
    (PiRLC.v1_1.CombinationStep.indexOf
      (blockCount := PiRLC.v1_1.PublicInputCombination.blockCount)
      (cellCount := PiRLC.v1_1.PublicInputCombination.cellCount)
      (Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicBlockIndex (FullShape width fits) coordinate)
      (Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicLaneIndex coordinate)
      PiRLC.v1_1.PublicInputCombination.cell)))

private theorem digit (child : Spec.Phi81Relation.PiDECAlgebra.Radix.ChildIndex)
    (coordinate : Fin (PiDEC.v1_1.PublicInputSplit.coordinateCount width fits)) :
    renameExpr restore ((newInterface (width := width) (fits := fits)).digit newStart child coordinate) =
      (oldInterface (width := width) (fits := fits)).digit oldStart child coordinate := by
  change Expr.var (restore (27540990 + child.val * 270 + coordinate.val)) =
    Expr.var (28466470 + child.val * 270 + coordinate.val)
  apply congrArg Expr.var
  unfold restore
  rw [if_neg (by change ¬27540990 + child.val * 270 + coordinate.val < 27496062; omega)]
  omega

theorem publicInput_rows :
    (flatConstraints (Circuit.ops (PiDEC.v1_1.Formal.publicInputCircuit
      (PiDEC.v1_1.Formal.atOffset (newInterface (width := width) (fits := fits)) newStart)).main newStart)).map (renameExpr restore) =
    flatConstraints (Circuit.ops (PiDEC.v1_1.Formal.publicInputCircuit
      (PiDEC.v1_1.Formal.atOffset (oldInterface (width := width) (fits := fits)) oldStart)).main oldStart) := by
  apply PiDECLeafRenaming.publicInput_rows
  · intro source bounded
    simp only [PiDEC.v1_1.PublicInputSplit.sourceOffset,
      PiDEC.v1_1.SignedSplitScalar.exactPrivateCount, Nat.mul_one]
    apply sign_restore
    simpa only [PiDEC.v1_1.PublicInputSplit.coordinateCount_eq] using bounded
  · exact parent_public
  · exact digit

private theorem parent_commitment (row : Fin productionProfile.commitmentWidth) (lane : Fin ringDegree) :
    renameExpr restore (((newInterface (width := width) (fits := fits)).parent newStart).commitment row lane) =
      ((oldInterface (width := width) (fits := fits)).parent oldStart).commitment row lane := by
  exact congrArg Expr.var (restore_location (.parentCommitment
    (PiRLC.v1_1.CombinationStep.indexOf row lane PiRLC.v1_1.CommitmentCombination.cell)))

private theorem child_commitment (child : Spec.Phi81Relation.PiDECAlgebra.Radix.ChildIndex)
    (row : Fin productionProfile.commitmentWidth) (lane : Fin ringDegree) :
    renameExpr restore (((newInterface (width := width) (fits := fits)).message newStart child).commitment row lane) =
      ((oldInterface (width := width) (fits := fits)).message oldStart child).commitment row lane := by
  change Expr.var (restore (27496062 + child.val * 1188 + row.val * 54 + lane.val)) =
    Expr.var (28421542 + child.val * 1188 + row.val * 54 + lane.val)
  apply congrArg Expr.var
  unfold restore
  rw [if_neg (by change ¬27496062 + child.val * 1188 + row.val * 54 + lane.val < 27496062; omega)]
  omega

theorem commitment_rows :
    (flatConstraints (Circuit.ops (PiDEC.v1_1.Formal.commitmentCircuit
      (PiDEC.v1_1.Formal.atOffset (newInterface (width := width) (fits := fits)) newStart)).main
      (newStart + 270))).map (renameExpr restore) =
    flatConstraints (Circuit.ops (PiDEC.v1_1.Formal.commitmentCircuit
      (PiDEC.v1_1.Formal.atOffset (oldInterface (width := width) (fits := fits)) oldStart)).main
      (oldStart + 270)) := by
  apply PiDECLeafRenaming.radix_rows
  · intro coordinate
    exact parent_commitment (width := width) (fits := fits) _ _
  · intro child coordinate
    exact child_commitment (width := width) (fits := fits) _ _ _

private theorem parent_evalK (coefficient : Fin productionShape.coefficientCount) :
    renameExpr restore (((newInterface (width := width) (fits := fits)).parent newStart).evaluation.eval_K coefficient).c0 =
      (((oldInterface (width := width) (fits := fits)).parent oldStart).evaluation.eval_K coefficient).c0 ∧
    renameExpr restore (((newInterface (width := width) (fits := fits)).parent newStart).evaluation.eval_K coefficient).c1 =
      (((oldInterface (width := width) (fits := fits)).parent oldStart).evaluation.eval_K coefficient).c1 := by
  constructor
  · exact congrArg Expr.var (restore_location (.parentEvalK (PiRLC.v1_1.CombinationStep.indexOf
      PiRLC.v1_1.EvalKCombination.block (Fin.cast PiRLC.v1_1.EvalKCombination.coefficientCount_eq coefficient)
      PiRLC.v1_1.RingKCombination.c0Cell)))
  · exact congrArg Expr.var (restore_location (.parentEvalK (PiRLC.v1_1.CombinationStep.indexOf
      PiRLC.v1_1.EvalKCombination.block (Fin.cast PiRLC.v1_1.EvalKCombination.coefficientCount_eq coefficient)
      PiRLC.v1_1.RingKCombination.c1Cell)))

private theorem parent_evalA (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) :
    renameExpr restore (((newInterface (width := width) (fits := fits)).parent newStart).evaluation.eval_A matrix coefficient).c0 =
      (((oldInterface (width := width) (fits := fits)).parent oldStart).evaluation.eval_A matrix coefficient).c0 ∧
    renameExpr restore (((newInterface (width := width) (fits := fits)).parent newStart).evaluation.eval_A matrix coefficient).c1 =
      (((oldInterface (width := width) (fits := fits)).parent oldStart).evaluation.eval_A matrix coefficient).c1 := by
  constructor
  · exact congrArg Expr.var (restore_location (.parentEvalA (PiRLC.v1_1.CombinationStep.indexOf
      matrix (Fin.cast PiRLC.v1_1.EvalKCombination.coefficientCount_eq coefficient)
      PiRLC.v1_1.RingKCombination.c0Cell)))
  · exact congrArg Expr.var (restore_location (.parentEvalA (PiRLC.v1_1.CombinationStep.indexOf
      matrix (Fin.cast PiRLC.v1_1.EvalKCombination.coefficientCount_eq coefficient)
      PiRLC.v1_1.RingKCombination.c1Cell)))

private theorem suffix_var (source : Nat) (afterProof : 27496062 ≤ source) :
    renameExpr restore (.var source) = .var (source + 925480) := by
  apply congrArg Expr.var
  exact if_neg (by change ¬source < 27496062; omega)

private theorem child_evalK (child : Spec.Phi81Relation.PiDECAlgebra.Radix.ChildIndex)
    (coefficient : Fin productionShape.coefficientCount) :
    renameExpr restore (((newInterface (width := width) (fits := fits)).message newStart child).evaluation.eval_K coefficient).c0 =
      (((oldInterface (width := width) (fits := fits)).message oldStart child).evaluation.eval_K coefficient).c0 ∧
    renameExpr restore (((newInterface (width := width) (fits := fits)).message newStart child).evaluation.eval_K coefficient).c1 =
      (((oldInterface (width := width) (fits := fits)).message oldStart child).evaluation.eval_K coefficient).c1 := by
  constructor
  · change renameExpr restore (.var (27515070 + child.val * 108 + coefficient.val * 2)) =
      .var (28440550 + child.val * 108 + coefficient.val * 2)
    rw [suffix_var _ (by omega)]
    apply congrArg Expr.var
    omega
  · change renameExpr restore (.var (27515070 + child.val * 108 + coefficient.val * 2 + 1)) =
      .var (28440550 + child.val * 108 + coefficient.val * 2 + 1)
    rw [suffix_var _ (by omega)]
    apply congrArg Expr.var
    omega

private theorem child_evalA (child : Spec.Phi81Relation.PiDECAlgebra.Radix.ChildIndex)
    (matrix : Fin productionShape.matrixCount) (coefficient : Fin productionShape.coefficientCount) :
    renameExpr restore (((newInterface (width := width) (fits := fits)).message newStart child).evaluation.eval_A matrix coefficient).c0 =
      (((oldInterface (width := width) (fits := fits)).message oldStart child).evaluation.eval_A matrix coefficient).c0 ∧
    renameExpr restore (((newInterface (width := width) (fits := fits)).message newStart child).evaluation.eval_A matrix coefficient).c1 =
      (((oldInterface (width := width) (fits := fits)).message oldStart child).evaluation.eval_A matrix coefficient).c1 := by
  constructor
  · change renameExpr restore (.var (27516798 + child.val * 1512 + matrix.val * 108 + coefficient.val * 2)) =
      .var (28442278 + child.val * 1512 + matrix.val * 108 + coefficient.val * 2)
    rw [suffix_var _ (by omega)]
    apply congrArg Expr.var
    omega
  · change renameExpr restore (.var (27516798 + child.val * 1512 + matrix.val * 108 + coefficient.val * 2 + 1)) =
      .var (28442278 + child.val * 1512 + matrix.val * 108 + coefficient.val * 2 + 1)
    rw [suffix_var _ (by omega)]
    apply congrArg Expr.var
    omega

theorem evalK_rows :
    (flatConstraints (Circuit.ops (PiDEC.v1_1.Formal.evalKCircuit
      (PiDEC.v1_1.Formal.atOffset (newInterface (width := width) (fits := fits)) newStart)).main
      (newStart + 270))).map (renameExpr restore) =
    flatConstraints (Circuit.ops (PiDEC.v1_1.Formal.evalKCircuit
      (PiDEC.v1_1.Formal.atOffset (oldInterface (width := width) (fits := fits)) oldStart)).main
      (oldStart + 270)) := by
  apply PiDECLeafRenaming.ringK_rows
  · intro block lane
    exact parent_evalK (width := width) (fits := fits) _
  · intro child block lane
    exact child_evalK (width := width) (fits := fits) _ _

theorem evalA_rows :
    (flatConstraints (Circuit.ops (PiDEC.v1_1.Formal.evalACircuit
      (PiDEC.v1_1.Formal.atOffset (newInterface (width := width) (fits := fits)) newStart)).main
      (newStart + 270))).map (renameExpr restore) =
    flatConstraints (Circuit.ops (PiDEC.v1_1.Formal.evalACircuit
      (PiDEC.v1_1.Formal.atOffset (oldInterface (width := width) (fits := fits)) oldStart)).main
      (oldStart + 270)) := by
  apply PiDECLeafRenaming.ringK_rows
  · intro block lane
    exact parent_evalA (width := width) (fits := fits) _ _
  · intro child block lane
    exact child_evalA (width := width) (fits := fits) _ _ _

theorem logicalConstraints (relation : ProductionKey.LogicalRelation width fits) :
    (PiDEC.v1_1.logicalConstraints relation (newInterface (width := width) (fits := fits)) newStart).map
      (renameExpr restore) =
    PiDEC.v1_1.logicalConstraints relation (oldInterface (width := width) (fits := fits)) oldStart := by
  rw [PiDEC.v1_1.logicalConstraints_eq_nonBoundary, PiDEC.v1_1.logicalConstraints_eq_nonBoundary]
  unfold PiDEC.v1_1.nonBoundaryConstraints PiDEC.v1_1.childConstraints
  simp only [List.map_append]
  exact congrArg₂ List.append
    (congrArg₂ List.append
      (congrArg₂ List.append (publicInput_rows (width := width) (fits := fits))
        (commitment_rows (width := width) (fits := fits)))
      (evalK_rows (width := width) (fits := fits)))
    (evalA_rows (width := width) (fits := fits))

private theorem relocate_restore : relocate 27545580 925480 restore = restore := by
  funext column
  unfold relocate
  split
  · rfl
  · unfold restore
    rw [if_neg (by change ¬column < 27496062; omega)]

/-- A valid PiDEC phase supplies the logical scope needed by lowering.
This is the circuit's existing completeness scope, not an extra input check. -/
theorem logical_scope (relation : ProductionKey.LogicalRelation width fits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := fits)) (env : Env)
    (assumptions : PiDEC.v1_1.Formal.Assumptions relation
      (newInterface (width := width) (fits := fits)) newStart env)
    (phase : PiDEC.v1_1.Semantics.PhaseHolds relation ajtai
      (newInterface (width := width) (fits := fits)) newStart env) :
    ∀ expression ∈ PiDEC.v1_1.logicalConstraints relation
      (newInterface (width := width) (fits := fits)) newStart, expression.VarsBelow 27545580 := by
  rcases PiDEC.v1_1.Formal.completePrefix relation ajtai
    (newInterface (width := width) (fits := fits)) env newStart assumptions phase with
    ⟨logical, operationsEq⟩
  have mainOpsEq : logical.operations = Circuit.ops
      (PiDEC.v1_1.Formal.main relation (newInterface (width := width) (fits := fits))) newStart := by
    rw [PiDEC.v1_1.Formal.main_ops]
    exact operationsEq
  have count : newStart + localLength logical.operations = 27545580 := by
    rw [mainOpsEq, PiDEC.v1_1.Formal.localLength_eq]
    rfl
  unfold PiDEC.v1_1.logicalConstraints
  rw [← mainOpsEq, ← count]
  exact logical.scope

/-- All physical PiDEC rows are the same ordered rows under the source map. -/
theorem physicalRows_of_phase (relation : ProductionKey.LogicalRelation width fits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := fits)) (env : Env)
    (assumptions : PiDEC.v1_1.Formal.Assumptions relation
      (newInterface (width := width) (fits := fits)) newStart env)
    (phase : PiDEC.v1_1.Semantics.PhaseHolds relation ajtai
      (newInterface (width := width) (fits := fits)) newStart env) :
    PiDEC.v1_1.physicalRows relation (oldInterface (width := width) (fits := fits)) oldStart =
      (PiDEC.v1_1.physicalRows relation (newInterface (width := width) (fits := fits)) newStart).map
        (renameRow restore) := by
  rw [PiDEC.v1_1.physicalRows_eq_lowerConstraints, PiDEC.v1_1.physicalRows_eq_lowerConstraints,
    ← logicalConstraints relation]
  have mapped := ConstraintRenaming.lowerConstraints_rename 27545580 27545580 925480 restore
    (PiDEC.v1_1.logicalConstraints relation (newInterface (width := width) (fits := fits)) newStart)
    (Nat.le_refl _) (logical_scope relation ajtai env assumptions phase)
  rw [relocate_restore] at mapped
  exact mapped

end NightstreamFPrime.Export.Stage1.Wide.PiDECSourceRenaming
