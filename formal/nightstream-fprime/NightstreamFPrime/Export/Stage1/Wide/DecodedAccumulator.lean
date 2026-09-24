import NightstreamFPrime.Export.Stage1.Wide.DecodedPiRLCInputs
import NightstreamFPrime.Export.Stage1.Wide.DecodedPiDEC
import NightstreamFPrime.Export.Stage1.Wide.PiDECSourceWitness
import NightstreamFPrime.Export.Stage1.Wide.PiRLCSourceOutput
import NightstreamFPrime.Layout.Stage1.Wide.PiRLCProtocolCompleteness
import NightstreamFPrime.Layout.Stage1.StepPhysicalCompleteness
import NightstreamFPrime.Layout.Stage1.PiDECProtocolCompleteness

/-! Connect arbitrary accepted candidate values to the existing phase
semantics. The total PiRLC constructor supplies an auxiliary proof view;
accepted product rows determine its actual retained outputs. -/

namespace NightstreamFPrime.Export.Stage1.Wide.DecodedAccumulator

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding Spec.Folding.PiCCS.PaperJoint Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open ProductionRelation Layout.Stage1
open PiCCS.v1_1

private theorem point_ext {n : Nat} (left right : CubePoint K n)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

variable {width : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}
  (program : RetainedLayout.Program) (relation : ProductionKey.LogicalRelation width publicFits)
  (ajtai : AjtaiKey (logicalWidth := width) (publicFits := publicFits))
  (assignment : Assignment F (RetainedLayout.logicalWidth program))

private theorem ccs_at (target : Env)
    (agrees : ∀ index, index < Layout.Stage1.Wide.PiRLCInputs.phaseOffset →
      target index = DecodedPrefix.piCcsEnv program assignment index)
    (template : Proof (ProductionKey.degreeBound relation))
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (accepted : (Stage1Plan.prefixPlan program relation).RowsZero assignment) :
    Formal.PhaseHolds relation ajtai (AccumulatorInputs.piCcsInterface width publicFits)
      PiCCSInputs.phaseOffset target template := by
  let before := DecodedPrefix.piCcsEnv program assignment
  have pointEq : StatementAbsorption.evalPoint
      (Formal.roundPoint (AccumulatorInputs.piCcsInterface width publicFits) PiCCSInputs.phaseOffset) before =
      StatementAbsorption.evalPoint
      (Formal.roundPoint (AccumulatorInputs.piCcsInterface width publicFits) PiCCSInputs.phaseOffset) target := by
    apply point_ext
    dsimp only [StatementAbsorption.evalPoint]
    apply congrArg List.ofFn
    funext coordinate
    change ((RunningTransitionInputs.recursiveRunningExpr width publicFits).point coordinate).eval before =
      ((RunningTransitionInputs.recursiveRunningExpr width publicFits).point coordinate).eval target
    apply Quadratic.KExpr.eval_eq_of_agree_below _ Layout.Stage1.Wide.PiRLCInputs.phaseOffset before target
    · rw [RunningTransitionInputs.recursivePoint_eq_direct, PiCCSStarts.roundTranscriptWitnessStart_eq]
      have bound : coordinate.val < 28 := coordinate.isLt
      change (15027676 + coordinate.val * 5328 + 4136 < 19513117) ∧
        (15027676 + coordinate.val * 5328 + 4728 < 19513117)
      constructor <;> omega
    · intro index below
      exact (agrees index below).symm
  apply Formal.PhaseTransport.phaseHolds_of_agree_satisfy relation ajtai
    (AccumulatorInputs.piCcsInterface width publicFits) PiCCSInputs.phaseOffset
    PiCCSOrdinarySourceSupport.External before target template
    (PiCCSOrdinarySourceSupport.externalInputsSupported width publicFits)
  · intro index allowed
    exact (agrees index (lt_of_lt_of_le
      (Layout.Stage1.StepPhysicalCompleteness.external_before_c index allowed) (by decide))).symm
  · exact (AccumulatorSemantics.piRlcPoint_eq_roundTranscript
      (logicalWidth := width) (publicFits := publicFits) before).symm.trans
      (pointEq.trans (AccumulatorSemantics.piRlcPoint_eq_roundTranscript
        (logicalWidth := width) (publicFits := publicFits) target))
  · have stateEq : (Layout.Stage1.Wide.PiRLCInputs.piCcsOutputState
        (logicalWidth := width) (publicFits := publicFits)) =
        Formal.outputBindingFinalState relation (AccumulatorInputs.piCcsInterface width publicFits)
          PiCCSInputs.phaseOffset := Layout.Stage1.Wide.PiRLCInputs.piCcsOutputState_eq_parent relation
    refine (congrArg (StatementAbsorption.evalState before) stateEq).symm.trans
      (Eq.trans ?_ (congrArg (StatementAbsorption.evalState target) stateEq))
    apply congrArg List.ofFn
    funext lane
    exact Expr.eval_eq_of_agree_below _ Layout.Stage1.Wide.PiRLCInputs.phaseOffset before target
      (Layout.Stage1.Wide.PiRLCInputBounds.samplerInitialBelow relation before lane)
      (fun index below => (agrees index below).symm)
  · exact DecodedPrefix.piCcs program relation assignment ajtai template one accepted

/-- The auxiliary PiRLC construction preserves the decoded PiCCS phase and
all decoded PiDEC rows. Their read intervals are disjoint from its allocation. -/
theorem witnessView
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (cRows : (Stage1Plan.prefixPlan program relation).RowsZero assignment)
    (dRows : (Stage1Plan.piDec program relation).RowsZero assignment) :
    ∃ r : Sequence.Prefix (DecodedPiDEC.env program assignment) Layout.Stage1.Wide.PiRLCInputs.phaseOffset,
      r.operations = PiRLC.Wide.Formal.opsAt relation Layout.Stage1.Wide.PiRLCInputs.interface
        Layout.Stage1.Wide.PiRLCInputs.phaseOffset ∧
      Formal.PhaseHolds relation ajtai (AccumulatorInputs.piCcsInterface width publicFits)
        PiCCSInputs.phaseOffset r.current (AccumulatorInputs.proof relation r.current) ∧
      PiRLC.Wide.Semantics.PhaseHolds relation ajtai Layout.Stage1.Wide.PiRLCInputs.interface
        Layout.Stage1.Wide.PiRLCInputs.phaseOffset r.current ∧
      PiDEC.v1_1.Semantics.PhaseHolds relation ajtai (PiDECInputs.interface width publicFits)
        PiDECInputs.phaseOffset r.current ∧
      (∀ index, index < Layout.Stage1.Wide.PiRLCInputs.phaseOffset →
        r.current index = DecodedPrefix.piCcsEnv program assignment index) ∧
      (∀ index, PiDECSourceSupport.Source index →
        r.current index = DecodedPiDEC.env program assignment index) := by
  obtain ⟨r, operations, rPhase⟩ := PiRLC.Wide.Formal.completePrefix_constructive relation ajtai
    Layout.Stage1.Wide.PiRLCInputs.interface (DecodedPiDEC.env program assignment)
    Layout.Stage1.Wide.PiRLCInputs.phaseOffset
    (Layout.Stage1.Wide.PiRLCInputBounds.assumptions relation (DecodedPiDEC.env program assignment))
  have early (index : Nat) (below : index < Layout.Stage1.Wide.PiRLCInputs.phaseOffset) :
      r.current index = DecodedPrefix.piCcsEnv program assignment index :=
    (r.agrees index (Or.inl below)).trans (DecodedPiDEC.before_parent program assignment index
      (lt_of_lt_of_le below (by decide)))
  have late (index : Nat) (owned : PiDECSourceSupport.Source index) :
      r.current index = DecodedPiDEC.env program assignment index := by
    apply r.agrees index
    apply Or.inr
    rw [operations, ← PiRLC.Wide.Formal.main_ops, PiRLC.Wide.Formal.localLength_eq]
    exact le_trans (by decide) (PiDECSourceSupport.parentStart_le_source owned)
  have dPhysical := R1CS.rowsHold_of_agree _ PiDECSourceSupport.Source
    (DecodedPiDEC.env program assignment) r.current (PiDECSourceWitness.old_rows_supported relation)
    late (DecodedPiDEC.physical program relation assignment one dRows)
  have dPhase := Layout.PiDEC.v1_1.physical_implies_phaseHolds relation ajtai
    (PiDECInputs.interface width publicFits) PiDECInputs.phaseOffset r.current
    (PiDECInputs.assumptions relation r.current) dPhysical
  exact ⟨r, operations, ccs_at program relation ajtai assignment r.current early _ one cRows,
    rPhase, dPhase, early, late⟩

theorem ordered_of_view
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (cRows : (Stage1Plan.prefixPlan program relation).RowsZero assignment)
    (r : Sequence.Prefix (DecodedPiDEC.env program assignment) Layout.Stage1.Wide.PiRLCInputs.phaseOffset)
    (operations : r.operations = PiRLC.Wide.Formal.opsAt relation Layout.Stage1.Wide.PiRLCInputs.interface
      Layout.Stage1.Wide.PiRLCInputs.phaseOffset)
    (phase : PiRLC.Wide.Semantics.PhaseHolds relation ajtai Layout.Stage1.Wide.PiRLCInputs.interface
      Layout.Stage1.Wide.PiRLCInputs.phaseOffset r.current)
    (early : ∀ index, index < Layout.Stage1.Wide.PiRLCInputs.phaseOffset →
      r.current index = DecodedPrefix.piCcsEnv program assignment index)
    (family : PiRLCOutput.Family) (block : Fin family.blockCount) (cell : Fin family.cellCount)
    (lane : Fin ringDegree) :
    PiRLCOutput.ordered (PiRLCWitness.initial (Stage1Plan.piRlcInterface program) assignment)
        (PiRLCWitness.inputValues (Stage1Plan.piRlcInterface program) assignment) family block cell lane =
      PiDECSource.value r.current (PiDECOutput.parentView family block cell lane) := by
  let initial := PiRLCWitness.initial (Stage1Plan.piRlcInterface program) assignment
  let values := PiRLCWitness.inputValues (Stage1Plan.piRlcInterface program) assignment
  let inputs := PiRLCSourceOutput.familyInterface (width := width) (fits := publicFits) family
  have rows : holds r.current (Circuit.ops (PiRLC.Wide.Formal.main relation
      Layout.Stage1.Wide.PiRLCInputs.interface) Layout.Stage1.Wide.PiRLCInputs.phaseOffset) := by
    rw [PiRLC.Wide.Formal.main_ops, ← operations]
    exact holdsFlat_implies_holds _ _ r.rows
  have specification := PiRLC.Wide.Formal.soundness relation _ Layout.Stage1.Wide.PiRLCInputs.phaseOffset r.current
    (Layout.Stage1.Wide.PiRLCInputBounds.assumptions relation r.current) rows
  have initialEq : List.ofFn initial = PiRLC.Wide.Scalar.evalState r.current
      (Layout.Stage1.Wide.PiRLCInputs.piCcsOutputState (logicalWidth := width) (publicFits := publicFits)) := by
    dsimp only [initial]
    rw [DecodedPiRLCInputs.initialState program relation assignment one cRows]
    apply congrArg List.ofFn
    funext outputLane
    exact Expr.eval_eq_of_agree_below _ Layout.Stage1.Wide.PiRLCInputs.phaseOffset _ _
      (Layout.Stage1.Wide.PiRLCInputBounds.samplerInitialBelow relation r.current outputLane)
      (fun index below => (early index below).symm)
  have challengeEq (source : Fin 17) : PiRLCValues.challenge initial source.val =
      PiRLC.v1_1.CombinationFamily.challengeValue inputs (PiRLCSourceOutput.familyStart family) r.current source := by
    funext coefficient
    rw [PiRLCSourceOutput.challenge_value]
    unfold PiRLCValues.challenge
    rw [initialEq]
    exact congrFun (congrFun phase.response source) coefficient
  have valueEq (source : Fin 17) :
      values ({family, source, block, cell} : PiRLCProductRingSchedule.Descriptor).invocation =
      PiRLC.v1_1.CombinationFamily.inputValue inputs (PiRLCSourceOutput.familyStart family) r.current source block cell := by
    funext coefficient
    have copied := DecodedPiRLCInputs.value program assignment
      ({family, source, block, cell} : PiRLCProductRingSchedule.Descriptor).invocation coefficient
    rw [PiRLCProductRingSchedule.descriptor_laneInvocation, PiRLCProductRingSchedule.descriptor_invocation] at copied
    have before := PiRLCValueWiring.valueSource_beforePhase
      ({family, source, block, lane := coefficient, cell} : PiRLCProductSchedule.Descriptor)
    have same := early _ (lt_of_lt_of_le before (by decide))
    exact copied.trans (same.symm.trans (PiRLCSourceOutput.input_value
      (width := width) (fits := publicFits) r.current family source block cell coefficient).symm)
  have sumEq : PiRLCOutput.ordered initial values family block cell =
      PiRLC.v1_1.CombinationFamily.orderedCombination inputs (PiRLCSourceOutput.familyStart family) r.current block cell := by
    unfold PiRLCOutput.ordered PiRLC.v1_1.CombinationFamily.orderedCombination
    apply congrArg PiRLC.v1_1.CombinationFamily.rightCombination
    funext source
    exact congrArg₂ ringFMul (challengeEq source) (valueEq source)
  change PiRLCOutput.ordered initial values family block cell lane = _
  rw [sumEq, ← PiRLCSourceOutput.canonical relation r.current specification family block cell]
  exact PiRLCSourceOutput.parent_value (width := width) (fits := publicFits) r.current family block cell lane

theorem parent_of_view (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (cRows : (Stage1Plan.prefixPlan program relation).RowsZero assignment)
    (rRows : (Stage1Plan.piRlc program compiled).RowsZero assignment)
    (r : Sequence.Prefix (DecodedPiDEC.env program assignment) Layout.Stage1.Wide.PiRLCInputs.phaseOffset)
    (operations : r.operations = PiRLC.Wide.Formal.opsAt relation Layout.Stage1.Wide.PiRLCInputs.interface
      Layout.Stage1.Wide.PiRLCInputs.phaseOffset)
    (phase : PiRLC.Wide.Semantics.PhaseHolds relation ajtai Layout.Stage1.Wide.PiRLCInputs.interface
      Layout.Stage1.Wide.PiRLCInputs.phaseOffset r.current)
    (early : ∀ index, index < Layout.Stage1.Wide.PiRLCInputs.phaseOffset →
      r.current index = DecodedPrefix.piCcsEnv program assignment index)
    (late : ∀ index, PiDECSourceSupport.Source index →
      r.current index = DecodedPiDEC.env program assignment index) :
    (PiDEC.v1_1.Semantics.inputAttempt relation (PiDECInputs.interface width publicFits)
      PiDECInputs.phaseOffset r.current).parent =
      PiRLC.Wide.Semantics.evalOutput relation Layout.Stage1.Wide.PiRLCInputs.interface
        Layout.Stage1.Wide.PiRLCInputs.phaseOffset r.current := by
  have same (family : PiRLCOutput.Family) (block : Fin family.blockCount) (cell : Fin family.cellCount)
      (lane : Fin ringDegree) :
      r.current (PiDECOutput.parentView family block cell lane).sourceColumn =
        PiDECSource.value r.current (PiDECOutput.parentView family block cell lane) :=
    (late _ (PiDECOutput.parentView family block cell lane).sourceSupport).trans
      ((DecodedPiDEC.parent program compiled assignment one rRows family block cell lane).trans
        (ordered_of_view program relation ajtai assignment one cRows r operations phase early family block cell lane))
  have startC : PiRLC.v1_1.CombinationFamily.stepOffset (PiRLC.v1_1.Formal.commitmentOffset PiRLCInputs.phaseOffset)
      PiRLC.v1_1.CombinationFamily.finalSource.val PiRLC.v1_1.CommitmentCombination.blockCount
      PiRLC.v1_1.CommitmentCombination.cellCount = 19795693 := by rfl
  have startX : PiRLC.v1_1.CombinationFamily.stepOffset (PiRLC.v1_1.Formal.publicInputOffset PiRLCInputs.phaseOffset)
      PiRLC.v1_1.CombinationFamily.finalSource.val PiRLC.v1_1.PublicInputCombination.blockCount
      PiRLC.v1_1.PublicInputCombination.cellCount = 19801201 := by rfl
  have startK : PiRLC.v1_1.CombinationFamily.stepOffset (PiRLC.v1_1.Formal.evalKOffset PiRLCInputs.phaseOffset)
      PiRLC.v1_1.CombinationFamily.finalSource.val PiRLC.v1_1.EvalKCombination.blockCount
      PiRLC.v1_1.RingKCombination.cellCount = 19803199 := by rfl
  have startA : PiRLC.v1_1.CombinationFamily.stepOffset (PiRLC.v1_1.Formal.evalAOffset PiRLCInputs.phaseOffset)
      PiRLC.v1_1.CombinationFamily.finalSource.val PiRLC.v1_1.EvalACombination.blockCount
      PiRLC.v1_1.RingKCombination.cellCount = 19827499 := by rfl
  apply PiDECProtocolCompleteness.instance_ext
  · rfl
  · funext row lane
    change r.current _ = r.current _
    convert same .commitment row ⟨0, by decide⟩ lane using 1 <;> (
      apply congrArg r.current
      try rw [startC]
      norm_num [PiDECOutput.parentView, PiDECDirectPlan.Location.sourceColumn, PiDECSource.value,
        PiDECSource.column, PiDECSource.parentCommitmentStart, PiDECSource.parentPublicInputStart,
        PiDECSource.parentEvalKStart, PiDECSource.parentEvalAStart,
        PiRLCCombinationInvocations.logicalIndex, PiRLCCombinationInvocations.indexOf_val,
        PiRLC.v1_1.CombinationFamily.stepOffset, PiRLC.v1_1.CombinationStep.indexOf,
        PiRLC.v1_1.CombinationFamily.finalSource, PiRLC.v1_1.CombinationFamily.stepSize,
        PiRLC.v1_1.CommitmentCombination.cell, PiRLC.v1_1.PublicInputCombination.cell,
        PiRLC.v1_1.RingKCombination.c0Cell, PiRLC.v1_1.RingKCombination.c1Cell, ringDegree,
        Layout.Stage1.Wide.PiRLCStarts.commitmentLogicalStart, Layout.Stage1.Wide.PiRLCStarts.publicInputLogicalStart,
        Layout.Stage1.Wide.PiRLCStarts.evalKLogicalStart, Layout.Stage1.Wide.PiRLCStarts.evalALogicalStart,
            Layout.Stage1.Wide.PiRLCStarts.phaseLogicalStart]
      ring)
  · funext column
    change r.current _ = r.current _
    convert same .publicInput (Phi81Relation.PiRLCAlgebra.PublicInput.publicBlockIndex (FullShape width publicFits) column)
      ⟨0, by decide⟩ (Phi81Relation.PiRLCAlgebra.PublicInput.publicLaneIndex column) using 1 <;> (
        apply congrArg r.current
        try rw [startX]
        norm_num [PiDECOutput.parentView, PiDECDirectPlan.Location.sourceColumn, PiDECSource.value,
          PiDECSource.column, PiDECSource.parentCommitmentStart, PiDECSource.parentPublicInputStart,
          PiDECSource.parentEvalKStart, PiDECSource.parentEvalAStart,
          PiRLCCombinationInvocations.logicalIndex, PiRLCCombinationInvocations.indexOf_val,
          PiRLC.v1_1.CombinationFamily.stepOffset, PiRLC.v1_1.CombinationStep.indexOf,
          PiRLC.v1_1.CombinationFamily.finalSource, PiRLC.v1_1.CombinationFamily.stepSize,
          PiRLC.v1_1.CommitmentCombination.cell, PiRLC.v1_1.PublicInputCombination.cell,
          PiRLC.v1_1.RingKCombination.c0Cell, PiRLC.v1_1.RingKCombination.c1Cell, ringDegree,
          Layout.Stage1.Wide.PiRLCStarts.commitmentLogicalStart, Layout.Stage1.Wide.PiRLCStarts.publicInputLogicalStart,
          Layout.Stage1.Wide.PiRLCStarts.evalKLogicalStart, Layout.Stage1.Wide.PiRLCStarts.evalALogicalStart,
            Layout.Stage1.Wide.PiRLCStarts.phaseLogicalStart]
        ring)
  · rfl
  · apply congrArg (fun evaluation : PaperAlgebra.Evaluation => #[evaluation])
    apply PiDECProtocolCompleteness.evaluation_ext
    · funext coefficient
      apply congrArg₂ K.mk
      · change r.current _ = r.current _
        convert same .evalK ⟨0, by decide⟩ ⟨0, by decide⟩ coefficient using 1 <;> (
          apply congrArg r.current
          try rw [startK]
          norm_num [PiDECOutput.parentView, PiDECDirectPlan.Location.sourceColumn, PiDECSource.value,
            PiDECSource.column, PiDECSource.parentCommitmentStart, PiDECSource.parentPublicInputStart,
            PiDECSource.parentEvalKStart, PiDECSource.parentEvalAStart,
            PiRLCCombinationInvocations.logicalIndex, PiRLCCombinationInvocations.indexOf_val,
            PiRLC.v1_1.CombinationFamily.stepOffset, PiRLC.v1_1.CombinationStep.indexOf,
            PiRLC.v1_1.CombinationFamily.finalSource, PiRLC.v1_1.CombinationFamily.stepSize,
            PiRLC.v1_1.CommitmentCombination.cell, PiRLC.v1_1.PublicInputCombination.cell,
            PiRLC.v1_1.RingKCombination.c0Cell, PiRLC.v1_1.RingKCombination.c1Cell, ringDegree,
            Layout.Stage1.Wide.PiRLCStarts.commitmentLogicalStart, Layout.Stage1.Wide.PiRLCStarts.publicInputLogicalStart,
            Layout.Stage1.Wide.PiRLCStarts.evalKLogicalStart, Layout.Stage1.Wide.PiRLCStarts.evalALogicalStart,
            Layout.Stage1.Wide.PiRLCStarts.phaseLogicalStart]
          ring)
      · change r.current _ = r.current _
        convert same .evalK ⟨0, by decide⟩ ⟨1, by decide⟩ coefficient using 1 <;> (
          apply congrArg r.current
          try rw [startK]
          norm_num [PiDECOutput.parentView, PiDECDirectPlan.Location.sourceColumn, PiDECSource.value,
            PiDECSource.column, PiDECSource.parentCommitmentStart, PiDECSource.parentPublicInputStart,
            PiDECSource.parentEvalKStart, PiDECSource.parentEvalAStart,
            PiRLCCombinationInvocations.logicalIndex, PiRLCCombinationInvocations.indexOf_val,
            PiRLC.v1_1.CombinationFamily.stepOffset, PiRLC.v1_1.CombinationStep.indexOf,
            PiRLC.v1_1.CombinationFamily.finalSource, PiRLC.v1_1.CombinationFamily.stepSize,
            PiRLC.v1_1.CommitmentCombination.cell, PiRLC.v1_1.PublicInputCombination.cell,
            PiRLC.v1_1.RingKCombination.c0Cell, PiRLC.v1_1.RingKCombination.c1Cell, ringDegree,
            Layout.Stage1.Wide.PiRLCStarts.commitmentLogicalStart, Layout.Stage1.Wide.PiRLCStarts.publicInputLogicalStart,
            Layout.Stage1.Wide.PiRLCStarts.evalKLogicalStart, Layout.Stage1.Wide.PiRLCStarts.evalALogicalStart,
            Layout.Stage1.Wide.PiRLCStarts.phaseLogicalStart]
          ring)
    · funext matrix coefficient
      apply congrArg₂ K.mk
      · change r.current _ = r.current _
        convert same .evalA matrix ⟨0, by decide⟩ coefficient using 1 <;> (
          apply congrArg r.current
          try rw [startA]
          norm_num [PiDECOutput.parentView, PiDECDirectPlan.Location.sourceColumn, PiDECSource.value,
            PiDECSource.column, PiDECSource.parentCommitmentStart, PiDECSource.parentPublicInputStart,
            PiDECSource.parentEvalKStart, PiDECSource.parentEvalAStart,
            PiRLCCombinationInvocations.logicalIndex, PiRLCCombinationInvocations.indexOf_val,
            PiRLC.v1_1.CombinationFamily.stepOffset, PiRLC.v1_1.CombinationStep.indexOf,
            PiRLC.v1_1.CombinationFamily.finalSource, PiRLC.v1_1.CombinationFamily.stepSize,
            PiRLC.v1_1.CommitmentCombination.cell, PiRLC.v1_1.PublicInputCombination.cell,
            PiRLC.v1_1.RingKCombination.c0Cell, PiRLC.v1_1.RingKCombination.c1Cell, ringDegree,
            Layout.Stage1.Wide.PiRLCStarts.commitmentLogicalStart, Layout.Stage1.Wide.PiRLCStarts.publicInputLogicalStart,
            Layout.Stage1.Wide.PiRLCStarts.evalKLogicalStart, Layout.Stage1.Wide.PiRLCStarts.evalALogicalStart,
            Layout.Stage1.Wide.PiRLCStarts.phaseLogicalStart]
          ring)
      · change r.current _ = r.current _
        convert same .evalA matrix ⟨1, by decide⟩ coefficient using 1 <;> (
          apply congrArg r.current
          try rw [startA]
          norm_num [PiDECOutput.parentView, PiDECDirectPlan.Location.sourceColumn, PiDECSource.value,
            PiDECSource.column, PiDECSource.parentCommitmentStart, PiDECSource.parentPublicInputStart,
            PiDECSource.parentEvalKStart, PiDECSource.parentEvalAStart,
            PiRLCCombinationInvocations.logicalIndex, PiRLCCombinationInvocations.indexOf_val,
            PiRLC.v1_1.CombinationFamily.stepOffset, PiRLC.v1_1.CombinationStep.indexOf,
            PiRLC.v1_1.CombinationFamily.finalSource, PiRLC.v1_1.CombinationFamily.stepSize,
            PiRLC.v1_1.CommitmentCombination.cell, PiRLC.v1_1.PublicInputCombination.cell,
            PiRLC.v1_1.RingKCombination.c0Cell, PiRLC.v1_1.RingKCombination.c1Cell, ringDegree,
            Layout.Stage1.Wide.PiRLCStarts.commitmentLogicalStart, Layout.Stage1.Wide.PiRLCStarts.publicInputLogicalStart,
            Layout.Stage1.Wide.PiRLCStarts.evalKLogicalStart, Layout.Stage1.Wide.PiRLCStarts.evalALogicalStart,
            Layout.Stage1.Wide.PiRLCStarts.phaseLogicalStart]
          ring)
  · rfl

/-- Accepted candidate C/R/D rows force the complete wide-key verifier on
the decoded external values. The auxiliary construction changes none of them. -/
theorem accepted (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (cRows : (Stage1Plan.prefixPlan program relation).RowsZero assignment)
    (rRows : (Stage1Plan.piRlc program compiled).RowsZero assignment)
    (dRows : (Stage1Plan.piDec program relation).RowsZero assignment) :
    ∃ env : Env,
      Nifs.PaperNonInteractive.verify (PiRLC.Wide.Key.key relation ajtai)
        (AccumulatorInputs.running width publicFits env) (AccumulatorInputs.fresh width publicFits env)
        (AccumulatorInputs.proof relation env) = some (AccumulatorInputs.output relation env) ∧
      (∀ index, index < Layout.Stage1.Wide.PiRLCInputs.phaseOffset →
        env index = DecodedPrefix.piCcsEnv program assignment index) ∧
      (∀ index, PiDECSourceSupport.Source index → env index = DecodedPiDEC.env program assignment index) := by
  obtain ⟨r, operations, cPhase, rPhase, dPhase, early, late⟩ :=
    witnessView program relation ajtai assignment one cRows dRows
  let env := r.current
  let key := PiRLC.Wide.Key.key relation ajtai
  let running := AccumulatorInputs.running width publicFits env
  let fresh := AccumulatorInputs.fresh width publicFits env
  let proof := AccumulatorInputs.proof relation env
  let output := AccumulatorInputs.output relation env
  have inputs := Layout.Stage1.Wide.AccumulatorSemantics.inputs_eq_keyOutputs relation ajtai env cPhase
  have parent := Layout.Stage1.Wide.AccumulatorSemantics.output_eq_keyParent relation ajtai env
    running fresh proof Layout.Stage1.Wide.PiRLCInputs.interface Layout.Stage1.Wide.PiRLCInputs.phaseOffset rPhase inputs
  have initial : PiRLC.Wide.Scalar.evalState env
      ((Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := publicFits)).initialState
        Layout.Stage1.Wide.PiRLCInputs.phaseOffset) = (key.piCcsExecution running fresh proof).outgoingState := by
    rw [PiRLC.Wide.Key.piCcsExecution_unchanged]
    exact PiRLCProtocolCompleteness.initialState_eq_of_phase relation ajtai env proof cPhase
  have challenges := Layout.Stage1.Wide.AccumulatorSemantics.challenges_eq_key relation ajtai env
    running fresh proof Layout.Stage1.Wide.PiRLCInputs.interface Layout.Stage1.Wide.PiRLCInputs.phaseOffset rPhase initial
  have joined := parent_of_view program relation ajtai assignment compiled one cRows rRows
    r operations rPhase early late
  have oldParent := AccumulatorSemantics.piDecParent_eq_piRlcOutput relation env
  have oldToNew := oldParent.symm.trans joined
  have attempt := AccumulatorSemantics.piDecAttempt_eq_keyAttemptForParent relation ajtai env
  rw [oldToNew] at attempt
  have attemptEq : key.piDecAttempt running fresh proof =
      some (PiDEC.v1_1.Semantics.inputAttempt relation (PiDECInputs.interface width publicFits)
        PiDECInputs.phaseOffset env) := by
    unfold Nifs.PaperNonInteractive.Key.piDecAttempt Nifs.PaperNonInteractive.Key.parent
    rw [challenges]
    change some (key.piDecAttemptForParent proof (key.parentForChallenges running fresh proof _)) = _
    rw [← parent]
    rw [PiRLC.Wide.Key.piDecAttemptForParent_unchanged]
    exact congrArg some attempt.symm
  have cCheck : Folding.PiCCS.Accepted key running fresh proof := by
    unfold Folding.PiCCS.Accepted
    rw [PiRLC.Wide.Key.piCcsCheck_unchanged]
    exact cPhase.accepted
  have dAccepted := PiDEC.v1_1.Semantics.accepted relation ajtai
    (PiDECInputs.interface width publicFits) PiDECInputs.phaseOffset env
    (PiDEC.v1_1.Semantics.phaseHolds_implies_spec relation ajtai _ _ _ dPhase)
  have dCheck := (Nifs.PaperNonInteractive.piDecCheck_eq_true_iff key running fresh proof).mpr
    ⟨_, attemptEq, dAccepted⟩
  have computed := key.output_eq_some_of_parentBounded running fresh proof _ attemptEq dAccepted.parentBounded
  have outputEq : key.outputForAttempt proof
      (PiDEC.v1_1.Semantics.inputAttempt relation (PiDECInputs.interface width publicFits) PiDECInputs.phaseOffset env)
      (key.piDecPublicInputSplit.split
        (PiDEC.v1_1.Semantics.inputAttempt relation (PiDECInputs.interface width publicFits) PiDECInputs.phaseOffset env).parent.publicInput) = output := by
    rw [PiRLC.Wide.Key.outputForAttempt_unchanged]
    exact AccumulatorSemantics.outputForAttempt_eq_accumulatorOutput relation ajtai env dPhase
  rw [outputEq] at computed
  exact ⟨env, (Nifs.PaperNonInteractive.verify_eq_some_iff key running fresh proof output).mpr
    ⟨cCheck, dCheck, computed⟩, early, late⟩

private theorem output_readback (env : Env)
    (early : ∀ index, index < Layout.Stage1.Wide.PiRLCInputs.phaseOffset →
      env index = DecodedPrefix.piCcsEnv program assignment index)
    (late : ∀ index, PiDECSourceSupport.Source index → env index = DecodedPiDEC.env program assignment index) :
    AccumulatorInputs.output relation env = AccumulatorInputs.output relation (DecodedPiDEC.env program assignment) := by
  let before := DecodedPiDEC.env program assignment
  have cells (column : Nat) (bound : 28421542 ≤ column ∧ column < 28470790) : env column = before column :=
    late column (PiDECSourceSupport.proof_source column bound)
  apply Formal.PhaseTransport.running_ext
  · apply point_ext
    dsimp only [AccumulatorInputs.output, RunningTransitionInputs.piDecRunningOutput, StatementAbsorption.evalPoint]
    apply congrArg List.ofFn
    funext coordinate
    change ((RunningTransitionInputs.recursiveRunningExpr width publicFits).point coordinate).eval env =
      ((RunningTransitionInputs.recursiveRunningExpr width publicFits).point coordinate).eval before
    apply Quadratic.KExpr.eval_eq_of_agree_below _ Layout.Stage1.Wide.PiRLCInputs.phaseOffset env before
    · rw [RunningTransitionInputs.recursivePoint_eq_direct, PiCCSStarts.roundTranscriptWitnessStart_eq]
      have bound : coordinate.val < 28 := coordinate.isLt
      change (15027676 + coordinate.val * 5328 + 4136 < 19513117) ∧
        (15027676 + coordinate.val * 5328 + 4728 < 19513117)
      constructor <;> omega
    · intro index below
      exact (early index below).trans (DecodedPiDEC.before_parent program assignment index
        (lt_of_lt_of_le below (by decide))).symm
  · funext source row lane
    have hs : source.val < 16 := source.isLt
    have hr : row.val < 22 := row.isLt
    have hl : lane.val < 54 := lane.isLt
    apply cells
    change 28421542 ≤ 28421542 + source.val * 1188 + row.val * 54 + lane.val ∧
      28421542 + source.val * 1188 + row.val * 54 + lane.val < 28470790
    constructor <;> omega
  · funext source column
    have hs : source.val < 16 := source.isLt
    have hc : column.val < 270 := column.isLt
    apply cells
    change 28421542 ≤ 28466470 + source.val * 270 + column.val ∧
      28466470 + source.val * 270 + column.val < 28470790
    constructor <;> omega
  · funext source
    have hs : source.val < 16 := source.isLt
    apply PiDECProtocolCompleteness.evaluation_ext
    · funext coefficient
      have hc : coefficient.val < 54 := coefficient.isLt
      apply congrArg₂ K.mk
      · apply cells
        change 28421542 ≤ 28440550 + source.val * 108 + coefficient.val * 2 ∧
          28440550 + source.val * 108 + coefficient.val * 2 < 28470790
        constructor <;> omega
      · apply cells
        change 28421542 ≤ 28440550 + source.val * 108 + coefficient.val * 2 + 1 ∧
          28440550 + source.val * 108 + coefficient.val * 2 + 1 < 28470790
        constructor <;> omega
    · funext matrix coefficient
      have hm : matrix.val < 14 := matrix.isLt
      have hc : coefficient.val < 54 := coefficient.isLt
      apply congrArg₂ K.mk
      · apply cells
        change 28421542 ≤ 28442278 + source.val * 1512 + matrix.val * 108 + coefficient.val * 2 ∧
          28442278 + source.val * 1512 + matrix.val * 108 + coefficient.val * 2 < 28470790
        constructor <;> omega
      · apply cells
        change 28421542 ≤ 28442278 + source.val * 1512 + matrix.val * 108 + coefficient.val * 2 + 1 ∧
          28442278 + source.val * 1512 + matrix.val * 108 + coefficient.val * 2 + 1 < 28470790
        constructor <;> omega

/-- The verifier reads only the candidate's decoded values. Auxiliary PiRLC
locals cannot change its running input, proof messages, or output. -/
theorem decodedAccepted (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (cRows : (Stage1Plan.prefixPlan program relation).RowsZero assignment)
    (rRows : (Stage1Plan.piRlc program compiled).RowsZero assignment)
    (dRows : (Stage1Plan.piDec program relation).RowsZero assignment) :
    let env := DecodedPiDEC.env program assignment
    Nifs.PaperNonInteractive.verify (PiRLC.Wide.Key.key relation ajtai)
      (AccumulatorInputs.running width publicFits env) (AccumulatorInputs.fresh width publicFits env)
      (AccumulatorInputs.proof relation env) = some (AccumulatorInputs.output relation env) := by
  let before := DecodedPiDEC.env program assignment
  obtain ⟨env, verified, early, late⟩ := accepted program relation ajtai assignment compiled one cRows rRows dRows
  have agree (index : Nat) (below : index < PiCCSInputs.phaseOffset) : env index = before index :=
    (early index (lt_of_lt_of_le below (by decide))).trans
      (DecodedPiDEC.before_parent program assignment index (lt_of_lt_of_le below (by decide))).symm
  have scope := PiCCSInputs.externalInputsBelow width publicFits
  have running := Formal.CompletenessSupport.evalRunning_eq_of_agree_below
    (AccumulatorInputs.piCcsInterface width publicFits) PiCCSInputs.phaseOffset env before scope agree
  have fresh := Formal.CompletenessSupport.evalFresh_eq_of_agree_below
    (AccumulatorInputs.piCcsInterface width publicFits) PiCCSInputs.phaseOffset env before scope agree
  have proofFields := Formal.CompletenessSupport.evalProof_eq_of_agree_below relation
    (AccumulatorInputs.piCcsInterface width publicFits) PiCCSInputs.phaseOffset env before
    (AccumulatorInputs.proof relation before) scope agree
  have output := output_readback program relation assignment env early late
  have proof : AccumulatorInputs.proof relation env = AccumulatorInputs.proof relation before := by
    apply Formal.PhaseTransport.proof_ext
    · exact congrArg (fun p => p.piCcsRounds) proofFields
    · exact congrArg (fun p => p.piCcsOutput) proofFields
    · exact congrArg (fun r => r.commitments) output
    · exact congrArg (fun r => r.evaluations) output
  change AccumulatorInputs.running width publicFits env = AccumulatorInputs.running width publicFits before at running
  change AccumulatorInputs.fresh width publicFits env = AccumulatorInputs.fresh width publicFits before at fresh
  rw [running, fresh, proof, output] at verified
  exact verified

end NightstreamFPrime.Export.Stage1.Wide.DecodedAccumulator
