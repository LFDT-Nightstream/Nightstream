import NightstreamFPrime.Export.Stage1.RunningTransitionReducedMatrixComplete
import NightstreamFPrime.Export.Stage1.RunningTransitionReducedRetainedSemantics

/-!
Connect the reduced matrix relation to the original physical source packet.
The encoding retains the shared inputs, one inverse field, and one flag bit.
No removed transition scratch value is required by the encoding or soundness.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.RunningTransitionReducedEncoding

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1 NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open RunningTransitionReducedMatrixSemantics
open RunningTransitionReducedRetainedBlocks

structure Encodes {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (assignment : Assignment F logicalWidth)
    (source : Fin (sourceWidth program) → F) : Prop where
  state : (RunningTransitionRetainedBlocks.stateBlock program).EncodesAt
    (RunningTransitionRetainedGeometry.stateStart program)
    (RunningTransitionRetainedGeometry.stateFits geometry) assignment source
  output : (RunningTransitionRetainedBlocks.outputBlock program).EncodesAt
    (RunningTransitionRetainedGeometry.outputStart program)
    (RunningTransitionRetainedGeometry.outputFits geometry) assignment source
  piDec : (RunningTransitionRetainedBlocks.piDecBlock program).EncodesAt
    (RunningTransitionRetainedGeometry.piDecStart program)
    (RunningTransitionRetainedGeometry.piDecFits geometry) assignment source
  sboxes : (PiCCSPoseidonPlan.retainedBlock program).EncodesAt
    (PiCCSPoseidonPlan.retainedStart program)
    (PiCCSPoseidonPlan.retainedFits
      (RunningTransitionRetainedGeometry.poseidonGeometry geometry)) assignment
    (PiCCSActionPayloadBlock.sourceAssignment program source)
  inverse : (inverseBlock program).EncodesAt
    (inverseStart program) (inverseFits geometry) assignment source
  flag : (flagBlock program).EncodesAt
    (flagStart program) (flagFits geometry) assignment source

abbrev SourceEnv (program : ApplicationProgram)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F) : Env :=
  Spartan.pullback (RunningTransitionDirectPlan.transitionEnv program base)

variable {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groups : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groups products))

include encodes

theorem decoded_external (column : Nat)
    (supported : RunningTransitionSourceSupport.External column) :
    decodedEnv geometry assignment column = SourceEnv program base column := by
  rcases supported with state | output | point | piDec
  · let index : Fin RunningTransitionSourceSupport.stateCount :=
      ⟨column - RunningTransitionSourceSupport.stateStart, by
        rcases state with ⟨lower, upper⟩; omega⟩
    have address : (RunningTransitionDirectPlan.Location.state index).sourceColumn = column := by
      dsimp only [RunningTransitionDirectPlan.Location.sourceColumn, index]
      exact Nat.add_sub_of_le state.1
    rw [← address, decodedEnv, sourceForm_state]
    exact RunningTransitionDirectPlan.Location.form_eval_of_encodes geometry assignment
      base groups products (.state index) encodes.state
  · let index : Fin RunningTransitionSourceSupport.outputCount :=
      ⟨column - RunningTransitionSourceSupport.outputStart, by
        rcases output with ⟨lower, upper⟩; omega⟩
    have address : (RunningTransitionDirectPlan.Location.output index).sourceColumn = column := by
      dsimp only [RunningTransitionDirectPlan.Location.sourceColumn, index]
      exact Nat.add_sub_of_le output.1
    rw [← address, decodedEnv, sourceForm_output]
    exact RunningTransitionDirectPlan.Location.form_eval_of_encodes geometry assignment
      base groups products (.output index) encodes.output
  · rcases point with ⟨coordinate, rfl | rfl⟩
    · rw [← PiCCSTranscriptOutputForms.pointSource_c0, decodedEnv, sourceForm_point]
      exact PiCCSTranscriptOutputForms.pointForm_eval
        (RunningTransitionRetainedGeometry.poseidonGeometry geometry)
        assignment base groups products encodes.sboxes coordinate 0
    · rw [← PiCCSTranscriptOutputForms.pointSource_c1, decodedEnv, sourceForm_point]
      exact PiCCSTranscriptOutputForms.pointForm_eval
        (RunningTransitionRetainedGeometry.poseidonGeometry geometry)
        assignment base groups products encodes.sboxes coordinate 1
  · have range := RunningTransitionSourceSupport.piDecField_inRange piDec
    let index : Fin RunningTransitionSourceSupport.piDecCount :=
      ⟨column - RunningTransitionSourceSupport.piDecStart, by
        rcases range with ⟨lower, upper⟩; omega⟩
    have address : (RunningTransitionDirectPlan.Location.piDec index).sourceColumn = column := by
      dsimp only [RunningTransitionDirectPlan.Location.sourceColumn, index]
      exact Nat.add_sub_of_le range.1
    rw [← address, decodedEnv, sourceForm_piDec]
    exact RunningTransitionDirectPlan.Location.form_eval_of_encodes geometry assignment
      base groups products (.piDec index) encodes.piDec

omit encodes in
private theorem source_inverse :
    PiRLCRetainedPreservation.sourceAssignment program base groups products (inverseSource program) =
      SourceEnv program base RunningTransitionInputs.phaseOffset := by
  rw [RunningTransitionReducedRetainedSemantics.inverseSource_value]
  apply Eq.symm
  apply RunningTransitionDirectPlan.transitionEnv_of_outside program base
  · exact (RunningTransitionDirectPlan.Location.fresh ⟨0, by
      rw [RunningTransitionRetainedBlocks.freshCount_eq]; decide⟩).sourceColumn_lt
  · apply Or.inr
    exact Nat.le_trans (show PiCCSInputs.phaseOffset +
      PiCCSOrdinarySourceSupport.transcriptInvocationCount * 592 ≤
        PiDECInputs.phaseOffset by decide) RunningTransitionInputs.piDecPhaseOffset_le

theorem decoded_inverse :
    decodedEnv geometry assignment RunningTransitionInputs.phaseOffset =
      SourceEnv program base RunningTransitionInputs.phaseOffset := by
  rw [decodedEnv, ← inverseForm_eq_source]
  change ((inverseBlock program).form (inverseStart program) (inverseFits geometry)
    ⟨0, by change 0 < 1; decide⟩).eval assignment = _
  rw [LowNormBlock.Block.form_eval _ _ _ assignment _ encodes.inverse]
  exact source_inverse base groups products

theorem decoded_flag :
    decodedEnv geometry assignment RunningTransitionReducedRows.flagIndex =
      SourceEnv program base RunningTransitionReducedRows.flagIndex := by
  rw [decodedEnv, sourceForm_flag]
  change ((flagBlock program).form (flagStart program) (flagFits geometry)
    ⟨0, by change 0 < 1; decide⟩).eval assignment = _
  rw [LowNormBlock.Block.form_eval _ _ _ assignment _ encodes.flag]
  change PiRLCRetainedPreservation.sourceAssignment program base groups products
    (flagSource program) = _
  rw [RunningTransitionReducedRetainedSemantics.flagSource_value]
  apply Eq.symm
  apply RunningTransitionDirectPlan.transitionEnv_of_outside program base
  · exact (RunningTransitionDirectPlan.Location.fresh ⟨1, by
      rw [RunningTransitionRetainedBlocks.freshCount_eq]; decide⟩).sourceColumn_lt
  · apply Or.inr
    exact Nat.le_trans (show PiCCSInputs.phaseOffset +
      PiCCSOrdinarySourceSupport.transcriptInvocationCount * 592 ≤
        PiDECInputs.phaseOffset by decide)
      (Nat.le_trans RunningTransitionInputs.piDecPhaseOffset_le (Nat.le_add_right _ 1))

theorem decoded_logical (column : Nat)
    (supported : RunningTransitionSourceSupport.Logical column) :
    decodedEnv geometry assignment column = SourceEnv program base column := by
  rcases supported with external | rfl
  · exact decoded_external geometry assignment base groups products encodes column external
  · exact decoded_inverse geometry assignment base groups products encodes

theorem rows_iff_source {relationWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth relationWidth}
    (relation : ProductionKey.LogicalRelation relationWidth publicFits) :
    R1CS.RowsHold (decodedEnv geometry assignment)
        (RunningTransitionReducedRows.rows relationWidth publicFits) ↔
      R1CS.RowsHold (SourceEnv program base)
        (RunningTransitionReducedRows.rows relationWidth publicFits) := by
  rw [RunningTransitionReducedRows.rows_iff_logical relation,
    RunningTransitionReducedRows.rows_iff_logical relation,
    decoded_flag geometry assignment base groups products encodes,
    decoded_inverse geometry assignment base groups products encodes]
  have support := RunningTransitionSourceSupport.inputsSupported relationWidth publicFits
  have iteration := Expr.eval_eq_of_agree_satisfy
    RunningTransitionInputs.iterationExpr RunningTransitionSourceSupport.Logical
    (decodedEnv geometry assignment) (SourceEnv program base) support.iteration
    (decoded_logical geometry assignment base groups products encodes)
  rw [iteration]
  apply and_congr_right
  intro _
  have values : ∀ expression ∈ RunningTransitionLayout.logicalConstraints relationWidth publicFits,
      expression.eval (decodedEnv geometry assignment) = expression.eval (SourceEnv program base) := by
    intro expression member
    exact Expr.eval_eq_of_agree_satisfy expression RunningTransitionSourceSupport.Logical _ _
      (RunningTransitionSourceSupport.logicalConstraints_varsSatisfy relationWidth publicFits expression member)
      (decoded_logical geometry assignment base groups products encodes)
  constructor <;> intro holds expression member
  · rw [← values expression member]
    exact holds expression member
  · rw [values expression member]
    exact holds expression member

theorem accepts_implies_spec {relationWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth relationWidth}
    (relation : ProductionKey.LogicalRelation relationWidth publicFits)
    (oneColumn : Fin logicalWidth) (sourceRow : Nat → Option R1CS.Row)
    (one : assignment oneColumn = 1)
    (accepted : RunningTransitionReducedMatrixComplete.Accepts
      (RunningTransitionReducedMatrixProgram.matrixProgram program oneColumn.val) sourceRow assignment) :
    Lifecycle.Stage1.RunningTransition.SpecHolds
      (RunningTransitionInputs.interface relationWidth publicFits)
      RunningTransitionInputs.phaseOffset (SourceEnv program base) := by
  apply RunningTransitionReducedRows.soundness relation
  apply (rows_iff_source geometry assignment base groups products encodes relation).mp
  exact (RunningTransitionReducedMatrixComplete.accepts_iff_rows relation geometry
    oneColumn sourceRow assignment one).mp accepted

theorem physical_implies_accepts {relationWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth relationWidth}
    (relation : ProductionKey.LogicalRelation relationWidth publicFits)
    (oneColumn : Fin logicalWidth) (sourceRow : Nat → Option R1CS.Row)
    (one : assignment oneColumn = 1)
    (physical : RunningTransitionLayout.PhysicalHolds relationWidth publicFits (SourceEnv program base)) :
    RunningTransitionReducedMatrixComplete.Accepts
      (RunningTransitionReducedMatrixProgram.matrixProgram program oneColumn.val) sourceRow assignment := by
  apply (RunningTransitionReducedMatrixComplete.accepts_iff_rows relation geometry
    oneColumn sourceRow assignment one).mpr
  apply (rows_iff_source geometry assignment base groups products encodes relation).mpr
  exact RunningTransitionReducedRows.physical_already_reduced relation _ physical

end NightstreamFPrime.Export.Stage1.RunningTransitionReducedEncoding
