import NightstreamFPrime.Export.Stage1.ActualPiRLC
import NightstreamFPrime.Export.Stage1.ActualPiDEC
import NightstreamFPrime.Layout.Stage1.AccumulatorSemantics
import Mathlib.Algebra.BigOperators.Fin

/-!
Owns the exact PiDEC parent from arbitrary accepted selected rows and actual
public input. Every field is the verifier combination of the decoded PiCCS
outputs and exact sampler challenges. Child-message decoding is a separate owner.
-/

namespace NightstreamFPrime.Export.Stage1.ActualPiDECParent

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open DirectPiRLCSamplerCompletePrefixPlan (piDecGeometry)
open scoped BigOperators

variable {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
  {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

def inputs (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) :=
  Semantics.evalInputs relation (PiRLCInputs.interface
    (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
    PiRLCInputs.phaseOffset (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
      (PiDECRetainedGeometry.prefixGeometry (piDecGeometry geometry)) assignment))

def parent (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) :=
  (PiDEC.v1_1.Semantics.inputAttempt relation
    (PiDECArithmetic.phaseInterface relationLogicalWidth relationPublicFits) PiDECInputs.phaseOffset
    (Spartan.pullback (ActualPiDEC.decodedEnv (piDecGeometry geometry) assignment))).parent

private theorem contribution_sum
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (descriptor : PiRLCProductSchedule.Descriptor) :
    (Finset.range PiRLCCombinationInvocations.sourceCount).sum
        (ActualPiRLCValues.contributionAt geometry assignment descriptor) =
      ∑ source : Fin PiRLCCombinationInvocations.sourceCount,
        ActualPiRLCValues.contribution geometry assignment
          (ActualPiRLCValues.withSource descriptor source) := by
  rw [Finset.sum_fin_eq_sum_range]
  rfl

private theorem combineCommitments_apply {count rows : Nat}
    (challenges : Fin count → RingF)
    (values : Fin count → Phi81Relation.PiRLCAlgebra.Commitment.Value rows)
    (row : Fin rows) (lane : Fin ringDegree) :
    Phi81Relation.PiRLCAlgebra.Commitment.combineCommitments challenges values row lane =
      ∑ source : Fin count, ringFMul (challenges source) (values source row) lane := by
  induction count with
  | zero => simp [Phi81Relation.PiRLCAlgebra.Commitment.combineCommitments,
      Phi81Relation.PiRLCAlgebra.Commitment.commitmentZero, ringFZero]
  | succ count ih =>
      rw [Fin.sum_univ_succ]
      change ringFMul (challenges 0) (values 0 row) lane +
        Phi81Relation.PiRLCAlgebra.Commitment.combineCommitments
          (fun source => challenges source.succ) (fun source => values source.succ) row lane = _
      rw [ih]

private theorem finalDescriptor_at (family : PiRLCProductSchedule.Family)
    (block : Fin family.blockCount) (lane : Fin ringDegree) (cell : Fin family.cellCount) :
    PiDECValueWiring.finalDescriptor family (CombinationStep.indexOf block lane cell) =
      ⟨family, ⟨16, by decide⟩, block, lane, cell⟩ := by
  simp [PiDECValueWiring.finalDescriptor, CombinationStep.coordinates, CombinationStep.indexOf]

theorem sourceCommitment_eq
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Fin PiRLCCombinationInvocations.sourceCount)
    (row : Fin productionProfile.commitmentWidth) (lane : Fin ringDegree) :
    ActualPiRLCValues.piCcsValue (piDecGeometry geometry) assignment
        ⟨.commitment, source, row, lane, ⟨0, by decide⟩⟩ =
      (inputs relation geometry assignment source).commitment row := by
  funext coefficient
  have value := (congrArg (fun expression => expression.eval
    (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
      (PiDECRetainedGeometry.prefixGeometry (piDecGeometry geometry)) assignment)))
    (PiRLCCombinationInvocations.productionCommitmentValue_eq
      (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits) source row coefficient)).symm
  simpa only [PiRLCCombinationInvocations.sourceValue, Nat.mul_one, Expr.eval_var] using value

theorem parentCommitment_eq_form
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (row : Fin productionProfile.commitmentWidth) (lane : Fin ringDegree) :
    (parent relation geometry assignment).commitment row lane =
      ((PiDECDirectPlan.Location.parentCommitment
        (CombinationStep.indexOf row lane CommitmentCombination.cell)).form (piDecGeometry geometry)).eval assignment := by
  dsimp only [parent, PiDEC.v1_1.Semantics.inputAttempt, PiDEC.v1_1.InputBinding.evalAttempt,
    PiDEC.v1_1.InputBinding.evalParent, PiDEC.v1_1.Formal.inputBindingInterface,
    PiDEC.v1_1.Formal.atOffset, PiDECArithmetic.phaseInterface, PiDECInputs.interface,
    PiDECInputs.parent, PiDECInputs.piRlcOutputInterface, Formal.outputBindingInterface,
    CommitmentCombination.output, CombinationFamily.output, CombinationStep.output,
    Expr.eval]
  exact ActualPiDEC.decodedEnv_location (piDecGeometry geometry) assignment
    (.parentCommitment (CombinationStep.indexOf row lane CommitmentCombination.cell))

theorem rowsZero_implies_commitment
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (ActualPiRLCValues.inputs (piDecGeometry geometry)).oneColumn = 1)
    (rows : (PiRLCProductPlan.plan (ActualPiRLCValues.inputs (piDecGeometry geometry))).RowsZero assignment) :
    (parent relation geometry assignment).commitment =
      Phi81Relation.PiRLCAlgebra.Commitment.combineCommitments
        (ActualPiRLCSampling.challenge geometry assignment)
        (fun source => (inputs relation geometry assignment source).commitment) := by
  funext row lane
  rw [parentCommitment_eq_form, ActualPiRLCValues.rowsZero_implies_parentCommitment_sum
    _ _ one rows, contribution_sum, combineCommitments_apply]
  apply Finset.sum_congr rfl
  intro source _
  have descriptorEq := finalDescriptor_at .commitment row lane CommitmentCombination.cell
  have contributionEq := congrArg (fun descriptor =>
    ActualPiRLCValues.contribution (piDecGeometry geometry) assignment
      (ActualPiRLCValues.withSource descriptor source)) descriptorEq
  exact contributionEq.trans (congrArg₂ (fun challenge value => ringFMul challenge value lane)
    (ActualPiRLC.productChallenge_eq geometry assignment
      ⟨.commitment, source, row, lane, CommitmentCombination.cell⟩)
    (sourceCommitment_eq relation geometry assignment source row lane))

private theorem combinePublicInputs_apply {shape : Phi81Relation.Shape} {count : Nat}
    (challenges : Fin count → RingF) (values : Fin count → Phi81Relation.PublicInput shape)
    (column : Fin shape.publicWidth) :
    Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs challenges values column =
      ∑ source : Fin count, ringFMul (challenges source)
        (Phi81Relation.PiRLCAlgebra.PublicInput.publicBlock (values source)
          (Phi81Relation.PiRLCAlgebra.PublicInput.publicBlockIndex shape column))
        (Phi81Relation.PiRLCAlgebra.PublicInput.publicLaneIndex column) := by
  induction count with
  | zero => simp [Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs,
      Phi81Relation.PiRLCAlgebra.PublicInput.publicZero]
  | succ count ih =>
      rw [Fin.sum_univ_succ]
      change ringFMul (challenges 0)
          (Phi81Relation.PiRLCAlgebra.PublicInput.publicBlock (values 0)
            (Phi81Relation.PiRLCAlgebra.PublicInput.publicBlockIndex shape column))
          (Phi81Relation.PiRLCAlgebra.PublicInput.publicLaneIndex column) +
        Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs
          (fun source => challenges source.succ) (fun source => values source.succ) column = _
      rw [ih]

theorem sourcePublicInput_eq
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Fin PiRLCCombinationInvocations.sourceCount)
    (block : Fin publicRingColumns) (lane : Fin ringDegree) :
    ActualPiRLCValues.piCcsValue (piDecGeometry geometry) assignment
        ⟨.publicInput, source, block, lane, PublicInputCombination.cell⟩ =
      Phi81Relation.PiRLCAlgebra.PublicInput.publicBlock
        (inputs relation geometry assignment source).publicInput block := by
  funext coefficient
  have value := (congrArg (fun expression => expression.eval
    (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
      (PiDECRetainedGeometry.prefixGeometry (piDecGeometry geometry)) assignment)))
    (PiRLCCombinationInvocations.productionPublicInputValue_eq
      (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits) source block coefficient)).symm
  simpa only [PiRLCCombinationInvocations.sourceValue, Nat.mul_one, Expr.eval_var] using value

theorem parentPublicInput_eq_form
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (column : Fin (FullShape relationLogicalWidth relationPublicFits).publicWidth) :
    (parent relation geometry assignment).publicInput column =
      ((PiDECDirectPlan.Location.parentPublicInput
        (CombinationStep.indexOf
          (blockCount := PublicInputCombination.blockCount)
          (cellCount := PublicInputCombination.cellCount)
          (Phi81Relation.PiRLCAlgebra.PublicInput.publicBlockIndex
            (FullShape relationLogicalWidth relationPublicFits) column)
          (Phi81Relation.PiRLCAlgebra.PublicInput.publicLaneIndex
            (shape := FullShape relationLogicalWidth relationPublicFits) column)
          PublicInputCombination.cell)).form (piDecGeometry geometry)).eval assignment := by
  dsimp only [parent, PiDEC.v1_1.Semantics.inputAttempt, PiDEC.v1_1.InputBinding.evalAttempt,
    PiDEC.v1_1.InputBinding.evalParent, PiDEC.v1_1.Formal.inputBindingInterface,
    PiDEC.v1_1.Formal.atOffset, PiDECArithmetic.phaseInterface, PiDECInputs.interface,
    PiDECInputs.parent, PiDECInputs.piRlcOutputInterface, Formal.outputBindingInterface,
    PublicInputCombination.output, CombinationFamily.output, CombinationStep.output,
    Expr.eval]
  exact ActualPiDEC.decodedEnv_location (piDecGeometry geometry) assignment
    (.parentPublicInput (CombinationStep.indexOf
      (blockCount := PublicInputCombination.blockCount)
      (cellCount := PublicInputCombination.cellCount)
      (Phi81Relation.PiRLCAlgebra.PublicInput.publicBlockIndex
        (FullShape relationLogicalWidth relationPublicFits) column)
      (Phi81Relation.PiRLCAlgebra.PublicInput.publicLaneIndex
        (shape := FullShape relationLogicalWidth relationPublicFits) column)
      PublicInputCombination.cell))

theorem rowsZero_implies_publicInput
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (ActualPiRLCValues.inputs (piDecGeometry geometry)).oneColumn = 1)
    (rows : (PiRLCProductPlan.plan (ActualPiRLCValues.inputs (piDecGeometry geometry))).RowsZero assignment) :
    (parent relation geometry assignment).publicInput =
      Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs
        (ActualPiRLCSampling.challenge geometry assignment)
        (fun source => (inputs relation geometry assignment source).publicInput) := by
  funext column
  let block := Phi81Relation.PiRLCAlgebra.PublicInput.publicBlockIndex
    (FullShape relationLogicalWidth relationPublicFits) column
  let lane := Phi81Relation.PiRLCAlgebra.PublicInput.publicLaneIndex column
  rw [parentPublicInput_eq_form, ActualPiRLCValues.rowsZero_implies_parentPublicInput_sum
    _ _ one rows, contribution_sum, combinePublicInputs_apply]
  apply Finset.sum_congr rfl
  intro source _
  have descriptorEq := finalDescriptor_at .publicInput block lane PublicInputCombination.cell
  have contributionEq := congrArg (fun descriptor =>
    ActualPiRLCValues.contribution (piDecGeometry geometry) assignment
      (ActualPiRLCValues.withSource descriptor source)) descriptorEq
  exact contributionEq.trans (congrArg₂ (fun challenge value => ringFMul challenge value lane)
    (ActualPiRLC.productChallenge_eq geometry assignment
      ⟨.publicInput, source, block, lane, PublicInputCombination.cell⟩)
    (sourcePublicInput_eq relation geometry assignment source block lane))

def inputEvaluation (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Fin PiRLCCombinationInvocations.sourceCount) :
    PaperAlgebra.Evaluation :=
  (inputs relation geometry assignment source).evaluations.getD 0 PaperAlgebra.evaluationZero

def parentEvaluation (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) : PaperAlgebra.Evaluation :=
  (parent relation geometry assignment).evaluations.getD 0 PaperAlgebra.evaluationZero

theorem inputEvaluations_eq
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Fin PiRLCCombinationInvocations.sourceCount) :
    (inputs relation geometry assignment source).evaluations =
      #[inputEvaluation relation geometry assignment source] := by rfl

theorem parentEvaluations_eq
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) :
    (parent relation geometry assignment).evaluations =
      #[parentEvaluation relation geometry assignment] := by rfl

private theorem expressionCell_eval (cell : Fin RingKCombination.cellCount)
    (expression : Quadratic.KExpr) (env : Env) :
    (RingKCombination.expressionCell cell expression).eval env =
      RingKCombination.kCell cell (expression.eval env) := by
  unfold RingKCombination.expressionCell RingKCombination.kCell
  split <;> rfl

theorem sourceEvalK_eq
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Fin PiRLCCombinationInvocations.sourceCount)
    (lane : Fin ringDegree) (cell : Fin RingKCombination.cellCount) :
    ActualPiRLCValues.piCcsValue (piDecGeometry geometry) assignment
        ⟨.evalK, source, EvalKCombination.block, lane, cell⟩ =
      RingKCombination.ringKCell cell (inputEvaluation relation geometry assignment source).pad := by
  funext coefficient
  have value := (congrArg (fun expression => expression.eval
    (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
      (PiDECRetainedGeometry.prefixGeometry (piDecGeometry geometry)) assignment)))
    (PiRLCCombinationInvocations.productionEvalKValue_eq
      (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
      source EvalKCombination.block coefficient cell)).symm
  exact value.trans (expressionCell_eval cell _ _)

theorem sourceEvalA_eq
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Fin PiRLCCombinationInvocations.sourceCount)
    (matrix : Fin productionShape.matrixCount) (lane : Fin ringDegree)
    (cell : Fin RingKCombination.cellCount) :
    ActualPiRLCValues.piCcsValue (piDecGeometry geometry) assignment
        ⟨.evalA, source, matrix, lane, cell⟩ =
      RingKCombination.ringKCell cell
        ((inputEvaluation relation geometry assignment source).matrix matrix) := by
  funext coefficient
  have value := (congrArg (fun expression => expression.eval
    (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
      (PiDECRetainedGeometry.prefixGeometry (piDecGeometry geometry)) assignment)))
    (PiRLCCombinationInvocations.productionEvalAValue_eq
      (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
      source matrix coefficient cell)).symm
  exact value.trans (expressionCell_eval cell _ _)

private theorem output_cell {blockCount : Nat}
    (interface : RingKCombination.Interface blockCount) (offset : Nat)
    (block : Fin blockCount) (lane : Fin ringDegree) (env : Env)
    (cell : Fin RingKCombination.cellCount) :
    RingKCombination.kCell cell ((RingKCombination.output interface offset block lane).eval env) =
      (CombinationFamily.output (RingKCombination.familyInterface interface)
        offset block lane cell).eval env := by
  fin_cases cell <;> rfl

private theorem parentEvaluation_eq
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) :
    parentEvaluation relation geometry assignment =
      PiCCS.v1_1.StatementAbsorption.evalEvaluation
        (PiDECInputs.parent relationLogicalWidth relationPublicFits).evaluation
        (Spartan.pullback (ActualPiDEC.decodedEnv (piDecGeometry geometry) assignment)) := by rfl

theorem parentEvalK_eq_form
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (lane : Fin ringDegree)
    (cell : Fin RingKCombination.cellCount) :
    RingKCombination.kCell cell ((parentEvaluation relation geometry assignment).pad lane) =
      ((PiDECDirectPlan.Location.parentEvalK
        (CombinationStep.indexOf EvalKCombination.block lane cell)).form (piDecGeometry geometry)).eval
        assignment := by
  let env := Spartan.pullback (ActualPiDEC.decodedEnv (piDecGeometry geometry) assignment)
  let interface := EvalKCombination.ringInterface (Formal.evalKInterface
    (PiDECInputs.piRlcSharedInterface relationLogicalWidth relationPublicFits))
  let location := PiDECDirectPlan.Location.parentEvalK
    (CombinationStep.indexOf EvalKCombination.block lane cell)
  have evaluated : (parentEvaluation relation geometry assignment).pad lane =
      (RingKCombination.output interface PiRLCStarts.evalKLogicalStart
        EvalKCombination.block lane).eval env := by
    rw [parentEvaluation_eq]
    rfl
  have symbolic : CombinationFamily.output (RingKCombination.familyInterface interface)
      PiRLCStarts.evalKLogicalStart EvalKCombination.block lane cell =
        Expr.var location.sourceColumn := by rfl
  exact (congrArg (RingKCombination.kCell cell) evaluated).trans
    ((output_cell interface PiRLCStarts.evalKLogicalStart EvalKCombination.block lane env cell).trans
      ((congrArg (fun expression => expression.eval env) symbolic).trans
        ((Expr.eval_var env location.sourceColumn).trans
          (ActualPiDEC.decodedEnv_location (piDecGeometry geometry) assignment location))))

theorem parentEvalA_eq_form
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (matrix : Fin productionShape.matrixCount)
    (lane : Fin ringDegree) (cell : Fin RingKCombination.cellCount) :
    RingKCombination.kCell cell ((parentEvaluation relation geometry assignment).matrix matrix lane) =
      ((PiDECDirectPlan.Location.parentEvalA
        (CombinationStep.indexOf matrix lane cell)).form (piDecGeometry geometry)).eval assignment := by
  let env := Spartan.pullback (ActualPiDEC.decodedEnv (piDecGeometry geometry) assignment)
  let interface := EvalACombination.ringInterface (Formal.evalAInterface
    (PiDECInputs.piRlcSharedInterface relationLogicalWidth relationPublicFits))
  let location := PiDECDirectPlan.Location.parentEvalA (CombinationStep.indexOf matrix lane cell)
  have evaluated : (parentEvaluation relation geometry assignment).matrix matrix lane =
      (RingKCombination.output interface PiRLCStarts.evalALogicalStart matrix lane).eval env := by
    rw [parentEvaluation_eq]
    rfl
  have symbolic : CombinationFamily.output (RingKCombination.familyInterface interface)
      PiRLCStarts.evalALogicalStart matrix lane cell = Expr.var location.sourceColumn := by rfl
  exact (congrArg (RingKCombination.kCell cell) evaluated).trans
    ((output_cell interface PiRLCStarts.evalALogicalStart matrix lane env cell).trans
      ((congrArg (fun expression => expression.eval env) symbolic).trans
        ((Expr.eval_var env location.sourceColumn).trans
          (ActualPiDEC.decodedEnv_location (piDecGeometry geometry) assignment location))))

private theorem kCell_add (cell : Fin RingKCombination.cellCount) (left right : K) :
    RingKCombination.kCell cell (K.add left right) =
      RingKCombination.kCell cell left + RingKCombination.kCell cell right := by
  unfold RingKCombination.kCell K.add
  split <;> rfl

private theorem k_eq_of_cells (left right : K)
    (cells : ∀ cell : Fin RingKCombination.cellCount,
      RingKCombination.kCell cell left = RingKCombination.kCell cell right) : left = right := by
  cases left with
  | mk left0 left1 =>
      cases right with
      | mk right0 right1 =>
          have first : left0 = right0 := cells RingKCombination.c0Cell
          have second : left1 = right1 := cells RingKCombination.c1Cell
          cases first
          cases second
          rfl

private theorem combineEvaluation_cell_apply {count : Nat}
    (challenges : Fin count → RingF) (values : Fin count → RingK)
    (lane : Fin ringDegree) (cell : Fin RingKCombination.cellCount) :
    RingKCombination.kCell cell (PiRLCFinite.combineEvaluation challenges values lane) =
      ∑ source : Fin count,
        ringFMul (challenges source) (RingKCombination.ringKCell cell (values source)) lane := by
  induction count with
  | zero => simp [PiRLCFinite.combineEvaluation, BaseLinear.evaluationZero,
      RingKCombination.kCell, ringKZero, K.zero]
  | succ count ih =>
      rw [Fin.sum_univ_succ]
      change RingKCombination.kCell cell
        (K.add (ringKMul (RingKAction.embedChallenge (challenges 0)) (values 0) lane)
          (PiRLCFinite.combineEvaluation (fun source => challenges source.succ)
            (fun source => values source.succ) lane)) = _
      rw [kCell_add, ← RingKCombination.ringKMul_cell, ih]

theorem rowsZero_implies_evalK
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (ActualPiRLCValues.inputs (piDecGeometry geometry)).oneColumn = 1)
    (rows : (PiRLCProductPlan.plan (ActualPiRLCValues.inputs (piDecGeometry geometry))).RowsZero assignment) :
    (parentEvaluation relation geometry assignment).pad =
      PiRLCFinite.combineEvaluation (ActualPiRLCSampling.challenge geometry assignment)
        (fun source => (inputEvaluation relation geometry assignment source).pad) := by
  funext lane
  apply k_eq_of_cells
  intro cell
  rw [parentEvalK_eq_form, ActualPiRLCValues.rowsZero_implies_parentEvalK_sum
    _ _ one rows, contribution_sum, combineEvaluation_cell_apply]
  apply Finset.sum_congr rfl
  intro source _
  have descriptorEq := finalDescriptor_at .evalK EvalKCombination.block lane cell
  have contributionEq := congrArg (fun descriptor =>
    ActualPiRLCValues.contribution (piDecGeometry geometry) assignment
      (ActualPiRLCValues.withSource descriptor source)) descriptorEq
  exact contributionEq.trans (congrArg₂ (fun challenge value => ringFMul challenge value lane)
    (ActualPiRLC.productChallenge_eq geometry assignment
      ⟨.evalK, source, EvalKCombination.block, lane, cell⟩)
    (sourceEvalK_eq relation geometry assignment source lane cell))

theorem rowsZero_implies_evalA
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (ActualPiRLCValues.inputs (piDecGeometry geometry)).oneColumn = 1)
    (rows : (PiRLCProductPlan.plan (ActualPiRLCValues.inputs (piDecGeometry geometry))).RowsZero assignment)
    (matrix : Fin productionShape.matrixCount) :
    (parentEvaluation relation geometry assignment).matrix matrix =
      PiRLCFinite.combineEvaluation (ActualPiRLCSampling.challenge geometry assignment)
        (fun source => (inputEvaluation relation geometry assignment source).matrix matrix) := by
  funext lane
  apply k_eq_of_cells
  intro cell
  rw [parentEvalA_eq_form, ActualPiRLCValues.rowsZero_implies_parentEvalA_sum
    _ _ one rows, contribution_sum, combineEvaluation_cell_apply]
  apply Finset.sum_congr rfl
  intro source _
  have descriptorEq := finalDescriptor_at .evalA matrix lane cell
  have contributionEq := congrArg (fun descriptor =>
    ActualPiRLCValues.contribution (piDecGeometry geometry) assignment
      (ActualPiRLCValues.withSource descriptor source)) descriptorEq
  exact contributionEq.trans (congrArg₂ (fun challenge value => ringFMul challenge value lane)
    (ActualPiRLC.productChallenge_eq geometry assignment ⟨.evalA, source, matrix, lane, cell⟩)
    (sourceEvalA_eq relation geometry assignment source matrix lane cell))

private theorem evaluation_eq_of_fields (left right : PaperAlgebra.Evaluation)
    (pad : left.pad = right.pad) (matrix : left.matrix = right.matrix) : left = right := by
  cases left
  cases right
  cases pad
  cases matrix
  rfl

theorem rowsZero_implies_evaluations
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (ActualPiRLCValues.inputs (piDecGeometry geometry)).oneColumn = 1)
    (rows : (PiRLCProductPlan.plan (ActualPiRLCValues.inputs (piDecGeometry geometry))).RowsZero assignment) :
    (parent relation geometry assignment).evaluations =
      PaperAlgebra.combineEvaluations (ActualPiRLCSampling.challenge geometry assignment)
        (fun source => (inputs relation geometry assignment source).evaluations) := by
  have inputArrays : (fun source => (inputs relation geometry assignment source).evaluations) =
      (fun source => #[inputEvaluation relation geometry assignment source]) := by
    funext source
    exact inputEvaluations_eq relation geometry assignment source
  have combined := (congrArg
    (PaperAlgebra.combineEvaluations (ActualPiRLCSampling.challenge geometry assignment))
    inputArrays).trans (PaperAlgebra.combineEvaluations_singletons (by decide)
      (ActualPiRLCSampling.challenge geometry assignment) (inputEvaluation relation geometry assignment))
  calc
    _ = #[parentEvaluation relation geometry assignment] := parentEvaluations_eq relation geometry assignment
    _ = #[PaperAlgebra.combineEvaluationFamily (ActualPiRLCSampling.challenge geometry assignment)
        (inputEvaluation relation geometry assignment)] := by
      apply congrArg (fun value : PaperAlgebra.Evaluation => #[value])
      apply evaluation_eq_of_fields
      · exact rowsZero_implies_evalK relation geometry assignment one rows
      · funext matrix
        exact rowsZero_implies_evalA relation geometry assignment one rows matrix
    _ = _ := combined.symm

theorem parentPoint_eq_inputPoint
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) :
    (parent relation geometry assignment).point =
      (inputs relation geometry assignment ⟨0, by decide⟩).point := by
  exact ActualPiDEC.evalPoint_eq_piCcs (piDecGeometry geometry) assignment

private theorem instance_eq_of_fields
    (left right : InputBinding.InputInstance relationLogicalWidth relationPublicFits)
    (system : left.constraintSystem = right.constraintSystem)
    (commitment : left.commitment = right.commitment)
    (publicInput : left.publicInput = right.publicInput)
    (point : left.point = right.point)
    (evaluations : left.evaluations = right.evaluations)
    (stage : left.stage = right.stage) : left = right := by
  cases left
  cases right
  cases system
  cases commitment
  cases publicInput
  cases point
  cases evaluations
  cases stage
  rfl

/-- Every typed parent field is the exact 17-source combination. -/
theorem rowsZero_implies_combinedParent
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (ajtai : AjtaiKey (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (ActualPiRLCValues.inputs (piDecGeometry geometry)).oneColumn = 1)
    (rows : (PiRLCProductPlan.plan (ActualPiRLCValues.inputs (piDecGeometry geometry))).RowsZero assignment) :
    parent relation geometry assignment =
      Spec.Folding.PiRLC.combinedOutput (PaperAlgebra.piRlcAlgebra ajtai)
        (InputBinding.relationSource relation)
        (inputs relation geometry assignment ⟨0, by decide⟩).point
        (inputs relation geometry assignment) (ActualPiRLCSampling.challenge geometry assignment) := by
  apply instance_eq_of_fields
  · rfl
  · exact rowsZero_implies_commitment relation geometry assignment one rows
  · exact rowsZero_implies_publicInput relation geometry assignment one rows
  · exact parentPoint_eq_inputPoint relation geometry assignment
  · exact rowsZero_implies_evaluations relation geometry assignment one rows
  · rfl

private theorem keyParent_eq_of_piCcsFields
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (ajtai : AjtaiKey (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
    (running : Running (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
    (fresh : Fresh (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
    (left right : Proof (ProductionKey.degreeBound relation))
    (challenges : Fin Spec.Folding.Nifs.PaperProfile.arity.total → RingF)
    (rounds : left.piCcsRounds = right.piCcsRounds) (output : left.piCcsOutput = right.piCcsOutput) :
    (ProductionKey.key relation ajtai).parentForChallenges running fresh left challenges =
      (ProductionKey.key relation ajtai).parentForChallenges running fresh right challenges := by
  cases left
  cases right
  cases rounds
  cases output
  rfl

/-- The actual parent equals the NIFS parent for the exact decoded PiCCS
fields and retained challenge vector. PiDEC messages do not enter this calculation. -/
theorem selectedRowsAndPublic_imply_parentForChallenges
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (ajtai : AjtaiKey (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (template : Proof (ProductionKey.degreeBound (PerApplicationFixedPoint.relation application fits)))
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (digest : Digest)
    (publicEqual : Phi81Relation.projectPublicInput
      (shape := FullShape (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      (Phi81CarrierLayout.extendAssignment 0 assignment) =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application) digest)
    (accepted : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment) :
    let geometry := PerApplicationFixedPoint.geometry application
    let relation := PerApplicationFixedPoint.relation application fits
    let interface := PiCCSInvocations.parentInterface (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application)
    let env := Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
      (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment)
    parent relation (DirectApplicationPrefixPlan.prefixGeometry geometry) assignment =
      (ProductionKey.key relation ajtai).parentForChallenges
        (PiCCS.v1_1.Formal.evalRunning interface PiCCSInputs.phaseOffset env)
        (PiCCS.v1_1.Formal.evalFresh interface PiCCSInputs.phaseOffset env)
        (PiCCS.v1_1.Formal.evalProof relation interface PiCCSInputs.phaseOffset env template)
        (ActualPiRLCSampling.challenge (DirectApplicationPrefixPlan.prefixGeometry geometry) assignment) := by
  let geometry := PerApplicationFixedPoint.geometry application
  let relation := PerApplicationFixedPoint.relation application fits
  let env := Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
    (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment)
  let samplerGeometry := DirectApplicationPrefixPlan.prefixGeometry geometry
  let canonical := AccumulatorInputs.proof relation env
  have publicBound : RecursivePublicOutputPlan.publicInput geometry assignment =
      encHash (publicFits := RecursivePublicOutputPlan.carrierPublicFits geometry) digest := by
    rw [RecursivePublicOutputPlan.publicInput_eq_projectPublicInput]
    exact publicEqual
  have one := RecursivePublicOutputPlan.publicEqual_implies_one geometry assignment digest publicBound
  have selected : (DirectApplicationPrefixPlan.plan relation fits.package geometry).RowsZero assignment := by
    rw [PerApplicationFixedPoint.plan_fixedPoint]
    exact accepted
  have parts := (DirectApplicationPrefixPlan.rowsZero_iff relation fits.package geometry assignment).mp selected
  have prefixRows := (DirectPiRLCSamplerCompletePrefixPlan.rowsZero_iff relation samplerGeometry assignment).mp parts.1.1.1
  have piRlcRows := prefixRows.2.2.1
  change (PiRLCRetainedPlan.plan _ _).RowsZero assignment at piRlcRows
  have productRows := (PiRLCRetainedPlan.rowsZero_iff _ _ assignment).mp piRlcRows
  have combined := rowsZero_implies_combinedParent relation ajtai samplerGeometry assignment one productRows.1
  have piCcs := PiCCSDecodedPhase.selectedRowsZero_implies_phaseHolds
    application fits ajtai canonical assignment one accepted
  have keyInputs := AccumulatorSemantics.piRlcInputs_eq_keyOutputs relation ajtai env piCcs
  have keyPoint : (inputs relation samplerGeometry assignment ⟨0, by decide⟩).point =
      ((ProductionKey.key relation ajtai).piCcsExecution
        (AccumulatorInputs.running _ _ env) (AccumulatorInputs.fresh _ _ env) canonical).coins.roundPoint :=
    congrArg (fun values => (values ⟨0, by decide⟩).point) keyInputs
  have canonicalParent : parent relation samplerGeometry assignment =
      (ProductionKey.key relation ajtai).parentForChallenges
        (AccumulatorInputs.running _ _ env) (AccumulatorInputs.fresh _ _ env) canonical
        (ActualPiRLCSampling.challenge samplerGeometry assignment) := by
    exact combined.trans (congrArg₂ (fun point values =>
      Spec.Folding.PiRLC.combinedOutput (arity := Spec.Folding.Nifs.PaperProfile.arity)
        (PaperAlgebra.piRlcAlgebra ajtai)
        (InputBinding.relationSource relation) point values (ActualPiRLCSampling.challenge samplerGeometry assignment))
      keyPoint keyInputs)
  dsimp only
  exact canonicalParent.trans (keyParent_eq_of_piCcsFields relation ajtai _ _ canonical _
    (ActualPiRLCSampling.challenge samplerGeometry assignment) rfl rfl)

/-- Selected acceptance makes the verifier's optional parent exactly the
actual decoded parent, including successful verifier-owned challenge sampling. -/
theorem selectedRowsAndPublic_imply_parent
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (ajtai : AjtaiKey (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (template : Proof (ProductionKey.degreeBound (PerApplicationFixedPoint.relation application fits)))
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (digest : Digest)
    (publicEqual : Phi81Relation.projectPublicInput
      (shape := FullShape (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      (Phi81CarrierLayout.extendAssignment 0 assignment) =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application) digest)
    (accepted : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment) :
    let geometry := PerApplicationFixedPoint.geometry application
    let relation := PerApplicationFixedPoint.relation application fits
    let interface := PiCCSInvocations.parentInterface (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application)
    let env := Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
      (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment)
    (ProductionKey.key relation ajtai).parent
        (PiCCS.v1_1.Formal.evalRunning interface PiCCSInputs.phaseOffset env)
        (PiCCS.v1_1.Formal.evalFresh interface PiCCSInputs.phaseOffset env)
        (PiCCS.v1_1.Formal.evalProof relation interface PiCCSInputs.phaseOffset env template) =
      some (parent relation (DirectApplicationPrefixPlan.prefixGeometry geometry) assignment) := by
  have challenges := ActualPiRLC.selectedRowsAndPublic_imply_keyChallenges application fits ajtai
    template assignment digest publicEqual accepted
  have parentEq := selectedRowsAndPublic_imply_parentForChallenges application fits ajtai
    template assignment digest publicEqual accepted
  exact (congrArg (Option.map _) challenges).trans (congrArg some parentEq.symm)

end NightstreamFPrime.Export.Stage1.ActualPiDECParent
