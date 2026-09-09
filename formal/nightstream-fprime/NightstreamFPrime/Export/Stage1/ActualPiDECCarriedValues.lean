import NightstreamFPrime.Export.Stage1.ActualPiDEC
import NightstreamFPrime.Export.Stage1.ActualRunningTransition

/-!
Owns shared PiDEC values between the actual PiDEC decoder and the actual
running-transition decoder. Child fields use the same retained proof block,
and the point uses the same retained PiCCS transcript outputs.
-/

namespace NightstreamFPrime.Export.Stage1.ActualPiDECCarriedValues

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open DirectPiDECPrefixPlan (runningGeometry)

variable {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}

/-- All child commitments, public digits and both evaluation families share
one physical proof-input block across the two actual decoders. -/
theorem piDecField_eq
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth) (column : Nat)
    (field : RunningTransitionSourceSupport.PiDecField column) :
    (Spartan.pullback (ActualRunningTransition.decodedEnv
      (runningGeometry geometry) assignment)) column =
      (Spartan.pullback (ActualPiDEC.decodedEnv geometry assignment)) column := by
  have inside := RunningTransitionSourceSupport.piDecField_inRange field
  let index := RunningTransitionDirectPlan.rangeIndex inside
  have owns : (RunningTransitionDirectPlan.Location.piDec index).sourceColumn = column :=
    RunningTransitionDirectPlan.rangeIndex_source inside
  have support : RunningTransitionSourceSupport.Source
      (RunningTransitionDirectPlan.Location.piDec index).sourceColumn := by
    rw [owns]
    exact Or.inl (Or.inr (Or.inr (Or.inr field)))
  have running := ActualRunningTransition.decodedEnv_location
    (runningGeometry geometry) assignment (.piDec index) support
  have piDec := ActualPiDEC.decodedEnv_location geometry assignment (.proof index)
  have forms := congrArg (fun form => form.eval assignment)
    (PiDECDirectPlan.Location.proof_form_eq_running geometry index)
  rw [← owns]
  exact running.trans (forms.symm.trans piDec.symm)

private theorem transcriptOutputForm
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry application logicalWidth)
    (invocation : Fin PiCCSOrdinaryRetainedBlocks.transcriptInvocationCount)
    (lane : Fin Poseidon2.width) :
    (PiCCSOrdinaryDirectPlan.Location.proofLogical
      (PiCCSOrdinaryRetainedBlocks.transcriptOutputSlot
        (Fin.encodeProd (invocation, lane)))).form geometry =
      PiCCSTranscriptOutputForms.transcriptForm
        (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry) invocation lane := by
  have decoded := Fin.decodeProd_encodeProd
    (m := PiCCSOrdinaryRetainedBlocks.transcriptInvocationCount)
    (n := Poseidon2.width) (invocation, lane)
  have mapped := congrArg
    (fun pair : Fin PiCCSOrdinaryRetainedBlocks.transcriptInvocationCount ×
        Fin Poseidon2.width =>
      PiCCSTranscriptOutputForms.transcriptForm
        (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry) pair.1 pair.2) decoded
  exact (PiCCSOrdinaryDirectPlan.Location.form_transcriptOutput geometry
    (Fin.encodeProd (invocation, lane))).trans mapped

/-- Every decoded transcript word is the same retained Poseidon2 form. -/
theorem piCcsTranscriptWord_eq_form
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (invocation : Fin PiCCSOrdinaryRetainedBlocks.transcriptInvocationCount)
    (lane : Fin Poseidon2.width) :
    (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv geometry assignment))
        (PiCCSTranscriptOutputForms.transcriptSource invocation lane) =
      (PiCCSTranscriptOutputForms.transcriptForm
        (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry) invocation lane).eval
          assignment := by
  have source : (PiCCSOrdinaryDirectPlan.Location.proofLogical
      (PiCCSOrdinaryRetainedBlocks.transcriptOutputSlot
        (Fin.encodeProd (invocation, lane)))).sourceColumn =
      PiCCSTranscriptOutputForms.transcriptSource invocation lane := by
    simp only [PiCCSOrdinaryDirectPlan.Location.sourceColumn,
      PiCCSOrdinaryRetainedBlocks.proofLogicalSource_transcriptOutput,
      PiCCSOrdinaryRetainedBlocks.transcriptOutputSource_encodeProd,
      PiCCSTranscriptOutputForms.transcriptSource,
      PiCCSTranscriptOutputForms.transcriptSourceStart]
    omega
  have value := PiCCSAssignmentSoundness.decodedEnv_location geometry assignment
    (.proofLogical (PiCCSOrdinaryRetainedBlocks.transcriptOutputSlot
      (Fin.encodeProd (invocation, lane))))
  rw [source] at value
  exact value.trans (congrArg (fun form => form.eval assignment)
    (transcriptOutputForm geometry invocation lane))

/-- The running point uses the actual PiCCS transcript words. -/
theorem piCcsPointWord_eq_form
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (coordinate : Fin productionShape.cubeVariables) (component : Fin 2) :
    (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
      (PiDECRetainedGeometry.prefixGeometry geometry) assignment))
        (PiCCSTranscriptOutputForms.pointSource coordinate component) =
      (PiCCSTranscriptOutputForms.pointForm
        (PiCCSOrdinaryRetainedGeometry.poseidonGeometry
          (PiDECRetainedGeometry.prefixGeometry geometry)) coordinate component).eval assignment := by
  rw [PiCCSTranscriptOutputForms.pointSource_eq_transcriptSource]
  exact piCcsTranscriptWord_eq_form (PiDECRetainedGeometry.prefixGeometry geometry)
    assignment (PiCCSTranscriptOutputForms.pointInvocation coordinate component)
    ⟨0, by norm_num [Poseidon2.width]⟩

/-- The running transition reads those same retained point components. -/
theorem runningPointWord_eq_form
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (coordinate : Fin productionShape.cubeVariables) (component : Fin 2) :
    (Spartan.pullback (ActualRunningTransition.decodedEnv
      (runningGeometry geometry) assignment))
        (PiCCSTranscriptOutputForms.pointSource coordinate component) =
      (PiCCSTranscriptOutputForms.pointForm
        (PiCCSOrdinaryRetainedGeometry.poseidonGeometry
          (PiDECRetainedGeometry.prefixGeometry geometry)) coordinate component).eval assignment := by
  fin_cases component
  · have value := ActualRunningTransition.decodedEnv_location
      (runningGeometry geometry) assignment (.roundC0 coordinate)
      (Or.inl (Or.inr (Or.inr (Or.inl ⟨coordinate, Or.inl rfl⟩))))
    exact (congrArg (Spartan.pullback (ActualRunningTransition.decodedEnv
      (runningGeometry geometry) assignment))
      (PiCCSTranscriptOutputForms.pointSource_c0 coordinate)).trans value
  · have value := ActualRunningTransition.decodedEnv_location
      (runningGeometry geometry) assignment (.roundC1 coordinate)
      (Or.inl (Or.inr (Or.inr (Or.inl ⟨coordinate, Or.inr rfl⟩))))
    exact (congrArg (Spartan.pullback (ActualRunningTransition.decodedEnv
      (runningGeometry geometry) assignment))
      (PiCCSTranscriptOutputForms.pointSource_c1 coordinate)).trans value

variable {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

private theorem runningPoint_eq_piCcs
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth) :
    PiDEC.v1_1.InputBinding.evalPoint
        ((PiDECInputs.interface relationLogicalWidth relationPublicFits).point PiDECInputs.phaseOffset)
        (Spartan.pullback (ActualRunningTransition.decodedEnv
          (runningGeometry geometry) assignment)) =
      PiDEC.v1_1.InputBinding.evalPoint
        ((PiDECInputs.interface relationLogicalWidth relationPublicFits).point PiDECInputs.phaseOffset)
        (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
          (PiDECRetainedGeometry.prefixGeometry geometry) assignment)) := by
  unfold PiDEC.v1_1.InputBinding.evalPoint PiRLC.v1_1.InputBinding.evalPoint
    PiCCS.v1_1.StatementAbsorption.evalPoint
  congr 2
  funext coordinate
  change ((RunningTransitionInputs.recursiveRunningExpr
    relationLogicalWidth relationPublicFits).point coordinate).eval _ =
    ((RunningTransitionInputs.recursiveRunningExpr
      relationLogicalWidth relationPublicFits).point coordinate).eval _
  rw [RunningTransitionInputs.recursivePoint_eq_direct]
  simp only [RunningTransitionInputs.directRoundPoint, KExpr.eval]
  apply congrArg₂ K.mk
  · simpa only [Expr.eval, PiCCSTranscriptOutputForms.pointSource_c0] using
      (runningPointWord_eq_form geometry assignment coordinate 0).trans
        (piCcsPointWord_eq_form geometry assignment coordinate 0).symm
  · simpa only [Expr.eval, PiCCSTranscriptOutputForms.pointSource_c1] using
      (runningPointWord_eq_form geometry assignment coordinate 1).trans
        (piCcsPointWord_eq_form geometry assignment coordinate 1).symm

/-- The complete shared point is equal under both actual decoders. -/
theorem point_eq
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth) :
    PiDEC.v1_1.InputBinding.evalPoint
        ((PiDECInputs.interface relationLogicalWidth relationPublicFits).point PiDECInputs.phaseOffset)
        (Spartan.pullback (ActualRunningTransition.decodedEnv
          (runningGeometry geometry) assignment)) =
      PiDEC.v1_1.InputBinding.evalPoint
        ((PiDECInputs.interface relationLogicalWidth relationPublicFits).point PiDECInputs.phaseOffset)
        (Spartan.pullback (ActualPiDEC.decodedEnv geometry assignment)) :=
  (runningPoint_eq_piCcs geometry assignment).trans
    (ActualPiDEC.evalPoint_eq_piCcs geometry assignment).symm

private theorem running_ext
    (left right : Running (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
    (point : left.point = right.point) (commitments : left.commitments = right.commitments)
    (publicInputs : left.publicInputs = right.publicInputs)
    (evaluations : left.evaluations = right.evaluations) : left = right := by
  cases left
  cases right
  simp_all

private theorem evaluation_ext (left right : PaperAlgebra.Evaluation)
    (pad : left.pad = right.pad) (matrix : left.matrix = right.matrix) : left = right := by
  cases left
  cases right
  simp_all

/-- All sixteen complete PiDEC child claims use the same actual values in
the running transition, including the public digits and shared point. -/
theorem runningOutput_eq
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth) :
    RunningTransitionInputs.piDecRunningOutput relation
        (Spartan.pullback (ActualRunningTransition.decodedEnv (runningGeometry geometry) assignment)) =
      RunningTransitionInputs.piDecRunningOutput relation
        (Spartan.pullback (ActualPiDEC.decodedEnv geometry assignment)) := by
  apply running_ext
  · exact point_eq geometry assignment
  · funext source row coefficient
    change (Spartan.pullback (ActualRunningTransition.decodedEnv
      (runningGeometry geometry) assignment))
        (PiDECInputs.childCommitmentStart (RunningTransitionInputs.childOfRunning source) +
          row.val * ringDegree + coefficient.val) = _
    exact piDecField_eq geometry assignment _
      (Or.inl ⟨RunningTransitionInputs.childOfRunning source, row, coefficient, rfl⟩)
  · funext source column
    let coordinate : Fin 270 := ⟨column.val, by
      have bound := column.isLt
      norm_num [FullShape, fullShape, Phi81Relation.Shape.publicWidth,
        publicRingColumns, ringDegree] at bound ⊢
      exact bound⟩
    change (Spartan.pullback (ActualRunningTransition.decodedEnv
      (runningGeometry geometry) assignment))
        (PiDECInputs.childPublicInputStart (RunningTransitionInputs.childOfRunning source) +
          coordinate.val) = _
    exact piDecField_eq geometry assignment _
      (Or.inr (Or.inl ⟨RunningTransitionInputs.childOfRunning source, coordinate, rfl⟩))
  · funext source
    apply evaluation_ext
    · funext coefficient
      change (PiDECInputs.childEvalK (RunningTransitionInputs.childOfRunning source) coefficient).eval _ =
        (PiDECInputs.childEvalK (RunningTransitionInputs.childOfRunning source) coefficient).eval _
      simp only [PiDECInputs.childEvalK, KExpr.eval, Expr.eval]
      apply congrArg₂ K.mk
      · exact piDecField_eq geometry assignment _ (Or.inr (Or.inr (Or.inl
          ⟨RunningTransitionInputs.childOfRunning source, coefficient, Or.inl rfl⟩)))
      · exact piDecField_eq geometry assignment _ (Or.inr (Or.inr (Or.inl
          ⟨RunningTransitionInputs.childOfRunning source, coefficient, Or.inr rfl⟩)))
    · funext matrix coefficient
      change (PiDECInputs.childEvalA (RunningTransitionInputs.childOfRunning source) matrix coefficient).eval _ =
        (PiDECInputs.childEvalA (RunningTransitionInputs.childOfRunning source) matrix coefficient).eval _
      simp only [PiDECInputs.childEvalA, KExpr.eval, Expr.eval]
      apply congrArg₂ K.mk
      · exact piDecField_eq geometry assignment _ (Or.inr (Or.inr (Or.inr
          ⟨RunningTransitionInputs.childOfRunning source, matrix, coefficient, Or.inl rfl⟩)))
      · exact piDecField_eq geometry assignment _ (Or.inr (Or.inr (Or.inr
          ⟨RunningTransitionInputs.childOfRunning source, matrix, coefficient, Or.inr rfl⟩)))

/-- The selected package's running transition carries the exact selected
PiDEC output for the same arbitrary assignment. -/
theorem selectedRunningOutput_eq
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application)) :
    RunningTransitionInputs.piDecRunningOutput (PerApplicationFixedPoint.relation application fits)
        (Spartan.pullback (ActualRunningTransition.decodedEnv
          (ActualRunningTransition.selectedGeometry application) assignment)) =
      RunningTransitionInputs.piDecRunningOutput (PerApplicationFixedPoint.relation application fits)
        (Spartan.pullback (ActualPiDEC.decodedEnv
          (ActualPiDEC.selectedGeometry application) assignment)) :=
  runningOutput_eq (PerApplicationFixedPoint.relation application fits)
    (ActualPiDEC.selectedGeometry application) assignment

end NightstreamFPrime.Export.Stage1.ActualPiDECCarriedValues
