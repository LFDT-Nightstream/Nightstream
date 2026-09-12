import NightstreamFPrime.Export.Stage1.PiCCSCompletedReadout
import NightstreamFPrime.Export.Stage1.PiCCSEndpointCompleteness
import NightstreamFPrime.Export.Stage1.PiCCSInvocationSlices
import NightstreamFPrime.Export.Stage1.InvocationInputLaw
import NightstreamFPrime.Export.Stage1.PiCCSCompilerAssertions
import NightstreamFPrime.Export.Stage1.PiCCSPhaseInputs
import NightstreamFPrime.Export.Stage1.PiCCSActionPayloadSupport
import NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxSourceCompleteness

/-!
Owns the direct C permutation-plan proof from the actual completed C rows.
Source input and retained S-box values are tied to the same compact compiler
invocation. Squeeze pins use the actual compiler assertion meaning.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSPoseidonCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Package
open PiCCSPoseidonPreservation
open PerApplicationAssignmentTransportExecution

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

section Completed

variable (application : Lifecycle.Stage1.Application.Program)
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (target : Env)
  (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
  (physical : NightstreamFPrime.Layout.PiCCS.v1_1.PhysicalHolds relation
    (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset (Spartan.pullback target))

local notation "raw" => canonicalRawValues application
  (PerApplicationSourceAssignment.ofCompleted application target suffix)
local notation "geometry" => PerApplicationCanonicalEncodes.poseidonGeometry application
local notation "payload" => DirectPiDECPrefixPlan.piCcsPayload
  (PerApplicationCanonicalEncodes.piDecGeometry application)

include relation physical in
private theorem source_value (column : Nat) (bound : column < Spartan.SourceColumnCount) :
    PiCCSActionPayloadBlock.packageEnv application raw.retainedSource column =
      Spartan.pullback target column := by
  exact (packageEnv_sourceAssignment application raw.base raw.groupValue raw.products column bound).trans
    (PiCCSCompletedReadout.transitionEnv_of_completed application relation target suffix physical column bound)

include relation physical in
private theorem payload_value (index : Fin PiCCSActionPayloadBlock.payloadCount) :
    PiCCSActionPayloadBlock.payloadValue application raw.retainedSource index =
      (PiCCSActionPayloadBlock.payloadExpression index).eval (Spartan.pullback target) := by
  apply Expr.eval_eq_of_agree_satisfy _ PiCCSOrdinarySourceSupport.Source
    (PiCCSActionPayloadBlock.packageEnv application raw.retainedSource) (Spartan.pullback target)
    (PiCCSActionPayloadSupport.payloadExpression_supported index)
  intro column supported
  exact source_value application relation target suffix physical column
    (PiCCSOrdinarySourceSupport.source_lt_sourceColumnCount supported)

private theorem source_sbox (index : InvocationIndex) (row : Fin PoseidonRetainedSlots.rows.length) :
    PiCCSPoseidonPreservation.sourceAssignment application raw.retainedSource
      ((PiCCSPoseidonPlan.schedule application).block.source
        (PoseidonRetainedFamily.slot (PiCCSPoseidonPlan.schedule application) index row)) =
      target ((physicalInvocation index).witnessStart + (PoseidonRetainedSlots.localOutput row).val) := by
  rw [retainedSource_sbox application raw.base raw.groupValue raw.products index row]
  apply PerApplicationSourceAssignment.packageEnv_ofCompleted
  have before := PoseidonRetainedBlock.laterWitnessStart_bound (laterIndex index)
  change (physicalInvocation index).witnessStart + 592 ≤
    PoseidonRetainedBlock.basePackage.layout.constantColumn at before
  have localBound := (PoseidonRetainedSlots.localOutput row).isLt
  change (PoseidonRetainedSlots.localOutput row).val < 592 at localBound
  have constant : PoseidonRetainedBlock.basePackage.layout.constantColumn = 29336446 :=
    NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.1
  rw [constant] at before
  rw [PerApplicationPackage.basePackage_totalColumnCount_eq]
  omega

private theorem sbox_form (index : InvocationIndex) (row : Fin PoseidonRetainedSlots.rows.length) :
    ((PoseidonSboxFamilyPlan.invocationInterface (PiCCSPoseidonPlan.interface payload geometry)
      index).sboxOutput row).eval raw.assignment =
      Pilot.canonicalInvocationEnv (physicalInvocation index) target
        (PoseidonRetainedSlots.rows.get row).step.output.val := by
  have sboxes := PiCCSPoseidonPlan.retainedBlock_encodesAt geometry raw.assignment raw.retainedSource
    (PiRLCRetainedGeometry.laterPoseidonFits (PiCCSPoseidonPlan.prefixGeometry geometry))
    (PerApplicationCanonicalEncodes.retainedEncodes raw).laterPoseidon
  change (PoseidonRetainedFamily.form (PiCCSPoseidonPlan.schedule application)
    (PiCCSPoseidonPlan.retainedStart application) (PiCCSPoseidonPlan.retainedFits geometry)
    index row).eval raw.assignment = _
  rw [PoseidonRetainedFamily.form_eval _ _ _ raw.assignment _ sboxes,
    source_sbox application target suffix index row,
    PoseidonRetainedSlots.output_eq_input_add_local]
  exact (Pilot.canonicalInvocationEnv_local (physicalInvocation index) target _).symm

include relation physical in
private theorem sboxes_of_input (index : InvocationIndex)
    (inputEq : SparseLayer.evalState raw.assignment (PiCCSPoseidonPlan.inputState payload geometry index) =
      Layer.evalState (Pilot.canonicalInvocationEnv (physicalInvocation index) target)
        PoseidonScheduleTrace.canonicalState) :
    PoseidonSboxPlan.SboxEquations
      (PoseidonSboxFamilyPlan.invocationInterface (PiCCSPoseidonPlan.interface payload geometry) index)
      raw.assignment := by
  apply PoseidonSboxSourceCompleteness.equations_of_sourceRows _ raw.assignment
    (Pilot.canonicalInvocationEnv (physicalInvocation index) target)
    (PerApplicationCanonicalAssignment.assignment_one raw) inputEq
    (sbox_form application target suffix index)
  exact Pilot.canonicalPermutationInvocation_implies_constraints (physicalInvocation index) target
    (PiCCSCompletedReadout.invocation_holds relation target physical index)

private theorem encoding : Encoding payload geometry raw.assignment raw.retainedSource := by
  exact encodingOfRetained payload geometry raw.assignment raw.retainedSource
    (PiRLCRetainedGeometry.laterPoseidonFits (PiCCSPoseidonPlan.prefixGeometry geometry))
    (PerApplicationCanonicalEncodes.retainedEncodes raw).laterPoseidon
    (PerApplicationCanonicalEncodes.runningPrefixEncodes raw).prior.payload

include relation physical in
private theorem payload_absorb_source
    (index : InvocationIndex) (lane : Fin 8) (block : List Expr)
    (found : PiCCSActionPayloadBlock.kindAt index = .absorb block) :
    payloadLaneValue application raw.retainedSource index lane =
      (block.getD lane.val (0 : Expr)).eval (Spartan.pullback target) := by
  by_cases rateLane : lane.val < Spec.Poseidon2.rate
  · rw [payloadLaneValue, dif_pos rateLane]
    have copied := payload_value application relation target suffix physical
      (Fin.encodeProd (index, ⟨lane.val, rateLane⟩))
    rw [PiCCSActionPayloadBlock.payloadExpression_encode] at copied
    simpa only [PiCCSActionPayloadBlock.payloadExpr, PiCCSActionPayloadBlock.selectedBlock,
      found, PiCCSActionPayloadBlock.selectedBlockForKind] using copied
  · have short := PiCCSActionPayloadBlock.kindAt_wellFormed index
    rw [found] at short
    change block.length ≤ Spec.Poseidon2.rate at short
    rw [payloadLaneValue, dif_neg rateLane,
      List.getD_eq_default block (0 : Expr) (short.trans (Nat.le_of_not_gt rateLane))]
    rfl

include relation physical in
private theorem expected_source
    (index : InvocationIndex) (expected : KExpr)
    (found : PiCCSActionPayloadBlock.kindAt index = .squeezeFirst expected) :
    expected.eval (PiCCSActionPayloadBlock.packageEnv application raw.retainedSource) =
      expected.eval (Spartan.pullback target) := by
  have zero : expected.c0.eval (PiCCSActionPayloadBlock.packageEnv application raw.retainedSource) =
      expected.c0.eval (Spartan.pullback target) := by
    have copied := payload_value application relation target suffix physical
      (Fin.encodeProd (index, (0 : Fin Spec.Poseidon2.rate)))
    simpa only [PiCCSActionPayloadBlock.payloadValue,
      PiCCSActionPayloadBlock.payloadExpression_encode,
      PiCCSActionPayloadBlock.payloadExpr, PiCCSActionPayloadBlock.selectedBlock,
      found, PiCCSActionPayloadBlock.selectedBlockForKind, List.getD_cons_zero] using copied
  have one : expected.c1.eval (PiCCSActionPayloadBlock.packageEnv application raw.retainedSource) =
      expected.c1.eval (Spartan.pullback target) := by
    have copied := payload_value application relation target suffix physical
      (Fin.encodeProd (index, (1 : Fin Spec.Poseidon2.rate)))
    simpa only [PiCCSActionPayloadBlock.payloadValue,
      PiCCSActionPayloadBlock.payloadExpression_encode,
      PiCCSActionPayloadBlock.payloadExpr, PiCCSActionPayloadBlock.selectedBlock,
      found, PiCCSActionPayloadBlock.selectedBlockForKind,
      List.getD_cons_succ, List.getD_cons_zero] using copied
  exact congrArg₂ K.mk zero one

include relation physical in
private theorem binding_zero
    (expectedValues : ∀ (index : InvocationIndex) (expected : KExpr),
      PiCCSActionPayloadBlock.kindAt index = .squeezeFirst expected →
      expected.eval (Spartan.pullback target) =
        K.mk (previousValue geometry raw.assignment index 0) (outputValue geometry raw.assignment index 0))
    (index : InvocationIndex) (component : Fin 2) :
    (PiCCSPoseidonPlan.bindingForm payload geometry index component).eval raw.assignment = 0 := by
  cases found : PiCCSActionPayloadBlock.kindAt index with
  | absorb block =>
      simp only [PiCCSPoseidonPlan.bindingForm, found, SparseForm.empty_eval]
  | squeezeSecond =>
      simp only [PiCCSPoseidonPlan.bindingForm, found, SparseForm.empty_eval]
  | squeezeFirst expected =>
      have same := (expected_source application relation target suffix physical index expected found).trans
        (expectedValues index expected found)
      have zero : expected.c0.eval (PiCCSActionPayloadBlock.packageEnv application raw.retainedSource) =
          previousValue geometry raw.assignment index 0 := congrArg K.c0 same
      have one : expected.c1.eval (PiCCSActionPayloadBlock.packageEnv application raw.retainedSource) =
          outputValue geometry raw.assignment index 0 := congrArg K.c1 same
      fin_cases component
      · rw [PiCCSPoseidonPlan.bindingForm_squeezeFirst_zero payload geometry index expected found,
          SparseForm.add_eval, SparseForm.scale_eval,
          payloadForm_eval payload geometry raw.assignment raw.retainedSource
            (encoding application target suffix),
          payloadLaneValue_squeezeFirst_zero application raw.retainedSource index expected found]
        rw [show (PiCCSPoseidonPlan.previousOutput geometry index 0).eval raw.assignment =
            previousValue geometry raw.assignment index 0 from
          congrFun (previousOutput_eval geometry raw.assignment index) 0, zero]
        simp
      · rw [PiCCSPoseidonPlan.bindingForm_squeezeFirst_one payload geometry index expected found,
          SparseForm.add_eval, SparseForm.scale_eval,
          payloadForm_eval payload geometry raw.assignment raw.retainedSource
            (encoding application target suffix),
          payloadLaneValue_squeezeFirst_one application raw.retainedSource index expected found]
        rw [show (PiCCSPoseidonPlan.outputState geometry index 0).eval raw.assignment =
            outputValue geometry raw.assignment index 0 from rfl, one]
        simp

include relation physical in
private theorem slice_values
    (phase rowStart witnessStart : Nat) (state : Layer.EState) (actions : List Formal.Action)
    (offset count : Nat) (counted : Invocations.invocationCount actions = count)
    (fits : offset + count ≤ PiCCSActionPayloadBlock.invocationCount)
    (offsetBound : offset < PiCCSActionPayloadBlock.invocationCount)
    (witnessLocal : Spartan.piCcsPhaseOffset ≤ witnessStart)
    (stateAffine : NightstreamFPrime.Layout.Poseidon2.StateAffine state)
    (actionsAffine : Invocations.ActionsInvocationInputsAffine actions)
    (selected : ∀ index : Fin count,
      physicalInvocation (PoseidonActionSemantics.sliceIndex offset count fits index) =
        (Invocations.compileActions phase rowStart witnessStart state actions).invocations.get
          ⟨index.val, by rw [Invocations.compileActions_invocations_length, counted]; exact index.isLt⟩)
    (kind : ∀ index : Fin count,
      PiCCSActionPayloadBlock.kindAt (PoseidonActionSemantics.sliceIndex offset count fits index) =
        PoseidonActionSchedule.kindAt actions (Fin.cast counted.symm index))
    (initial : List.ofFn (Layer.evalState (Spartan.pullback target) state) =
      PoseidonActionSemantics.sliceInitial Spec.Poseidon2.zeroState
        (valueState geometry raw.assignment) offset offsetBound)
    (assertions : ConstraintsHold (Spartan.pullback target)
      (Formal.compile witnessStart state actions).assertions)
    (index : Fin count) :
    (∀ lane : Fin 8,
      (invocationInputCombination
        (physicalInvocation (PoseidonActionSemantics.sliceIndex offset count fits index)) lane.val).toR1CS.eval target =
      canonicalInput geometry raw.assignment raw.retainedSource
        (PoseidonActionSemantics.sliceIndex offset count fits index) lane) ∧
    (∀ expected : KExpr,
      PiCCSActionPayloadBlock.kindAt (PoseidonActionSemantics.sliceIndex offset count fits index) =
        .squeezeFirst expected →
      expected.eval (Spartan.pullback target) =
        K.mk (previousValue geometry raw.assignment (PoseidonActionSemantics.sliceIndex offset count fits index) 0)
          (outputValue geometry raw.assignment (PoseidonActionSemantics.sliceIndex offset count fits index) 0)) := by
  subst count
  let trace := Invocations.compileActions phase rowStart witnessStart state actions
  let selectedInvocation := fun current : Fin (Invocations.invocationCount actions) => trace.invocations.get
    (Fin.cast (Invocations.compileActions_invocations_length phase rowStart witnessStart state actions).symm current)
  let outputs := fun current => List.ofFn fun coordinate : Fin 8 =>
    target ((selectedInvocation current).witnessStart + 584 + coordinate.val)
  let previous := PoseidonActionSemantics.previousState
    (List.ofFn (Layer.evalState (Spartan.pullback target) state)) outputs index
  let globalIndex := PoseidonActionSemantics.sliceIndex offset (Invocations.invocationCount actions) fits index
  have selectedEq (current : Fin (Invocations.invocationCount actions)) :
      physicalInvocation (PoseidonActionSemantics.sliceIndex offset _ fits current) = selectedInvocation current :=
    selected current
  have outputsEq : outputs = PoseidonActionSemantics.sliceOutput
      (valueState geometry raw.assignment) offset (Invocations.invocationCount actions) fits := by
    funext current
    apply congrArg List.ofFn
    funext lane
    have output := congrFun
      (PiCCSCompletedReadout.outputValue_of_completed application relation target suffix physical
        (PoseidonActionSemantics.sliceIndex offset _ fits current)) lane
    exact (output.trans (congrArg (fun invocation : PermutationInvocation =>
      target (invocation.witnessStart + 584 + lane.val)) (selectedEq current))).symm
  have previousEq : previous = List.ofFn (previousValue geometry raw.assignment globalIndex) := by
    change PoseidonActionSemantics.previousState _ outputs index = _
    rw [initial, outputsEq, PoseidonActionSemantics.previousState_slice]
    exact previousState_eq_previousValue geometry raw.assignment globalIndex
  have previousLane (lane : Fin 8) :
      previous.getD lane.val 0 = previousValue geometry raw.assignment globalIndex lane := by
    rw [previousEq]
    exact PriorStateHash.ofFn_getD _ lane (0 : F)
  have currentLane : (outputs index).getD 0 0 = outputValue geometry raw.assignment globalIndex 0 := by
    rw [outputsEq]
    exact PriorStateHash.ofFn_getD _ (0 : Fin 8) (0 : F)
  constructor
  · intro lane
    have inputLaw := InvocationInputLaw.compileActions_input_eval phase rowStart witnessStart
      state actions target witnessLocal stateAffine actionsAffine index lane
    change (invocationInputCombination (selectedInvocation index) lane.val).toR1CS.eval target =
      (match PoseidonActionSchedule.kindAt actions index with
      | .absorb block => previous.getD lane.val 0 + (block.getD lane.val (0 : Expr)).eval (Spartan.pullback target)
      | .squeezeFirst _ | .squeezeSecond => previous.getD lane.val 0) at inputLaw
    rw [← selectedEq index, ← kind index] at inputLaw
    simp only [previousLane] at inputLaw
    cases found : PiCCSActionPayloadBlock.kindAt globalIndex with
    | absorb block =>
        change _ = canonicalInput geometry raw.assignment raw.retainedSource globalIndex lane
        rw [canonicalInput, found,
          payload_absorb_source application relation target suffix physical globalIndex lane block found]
        simpa only [globalIndex, found] using inputLaw
    | squeezeFirst expected =>
        change _ = canonicalInput geometry raw.assignment raw.retainedSource globalIndex lane
        rw [canonicalInput, found]
        simpa only [globalIndex, found] using inputLaw
    | squeezeSecond =>
        change _ = canonicalInput geometry raw.assignment raw.retainedSource globalIndex lane
        rw [canonicalInput, found]
        simpa only [globalIndex, found] using inputLaw
  · intro expected found
    have localKind : PoseidonActionSchedule.kindAt actions index = .squeezeFirst expected :=
      (kind index).symm.trans found
    have expectedLaw := InvocationInputLaw.compileActions_expected_eval phase rowStart witnessStart
      state actions target witnessLocal assertions index expected localKind
    change expected.eval (Spartan.pullback target) = K.mk (previous.getD 0 0) ((outputs index).getD 0 0) at expectedLaw
    exact expectedLaw.trans (congrArg₂ K.mk (previousLane 0) currentLane)

include relation physical in
private theorem invocation_values (index : InvocationIndex) :
    (∀ lane : Fin 8,
      (invocationInputCombination (physicalInvocation index) lane.val).toR1CS.eval target =
        canonicalInput geometry raw.assignment raw.retainedSource index lane) ∧
    (∀ expected : KExpr, PiCCSActionPayloadBlock.kindAt index = .squeezeFirst expected →
      expected.eval (Spartan.pullback target) =
        K.mk (previousValue geometry raw.assignment index 0) (outputValue geometry raw.assignment index 0)) := by
  have bounded : index.val < 7604 := by
    simpa only [PiCCSPoseidonPlan.invocationCount_eq] using index.isLt
  by_cases inStatement : index.val < 379
  · let current : Fin PiCCSTranscriptDirectSemantics.statementCount := ⟨index.val, inStatement⟩
    have same : PoseidonActionSemantics.sliceIndex
        PiCCSTranscriptDirectSemantics.statementOffset PiCCSTranscriptDirectSemantics.statementCount
        PiCCSTranscriptDirectSemantics.statementFits current = index := by
      apply Fin.ext
      change 0 + index.val = index.val
      omega
    rw [← same]
    have affine := PiCCSPhaseInputs.statement_affine relation
    exact slice_values application relation target suffix physical
      PiCCSInvocations.statementPhase PiCCSInvocations.statementRowStart PiCCSInvocations.statementWitnessStart
      Hash.zeroE PiCCSActionPayloadBlock.statementActions
      PiCCSTranscriptDirectSemantics.statementOffset PiCCSTranscriptDirectSemantics.statementCount
      (PiCCSInvocations.statementInvocationCount_eq Data.logicalWidth Data.publicFits)
      PiCCSTranscriptDirectSemantics.statementFits PiCCSTranscriptDirectSemantics.statementOffsetBound
      (by
        rw [PiCCSInvocations.statementWitnessStart, PiCCSStarts.statementWitnessStart_eq]
        exact Nat.le_refl _) affine.1 affine.2
      PiCCSInvocationSlices.statement_invocation
      PiCCSTranscriptDirectSemantics.statementKindAt_eq
      (PiCCSPhaseInputs.statement_initial application target suffix)
      (PiCCSCompilerAssertions.statement_assertions (Spartan.pullback target)) current
  · by_cases inChallenge : index.val < 466
    · let current : Fin PiCCSTranscriptDirectSemantics.challengeCount := ⟨index.val - 379, by
        change index.val - 379 < 87
        omega⟩
      have same : PoseidonActionSemantics.sliceIndex
          PiCCSTranscriptDirectSemantics.challengeOffset PiCCSTranscriptDirectSemantics.challengeCount
          PiCCSTranscriptDirectSemantics.challengeFits current = index := by
        apply Fin.ext
        change 379 + (index.val - 379) = index.val
        omega
      rw [← same]
      have affine := PiCCSPhaseInputs.challenge_affine relation
      exact slice_values application relation target suffix physical
        PiCCSInvocations.challengePhase PiCCSInvocations.challengeRowStart PiCCSInvocations.challengeWitnessStart
        ((PiCCSInvocations.challengeInterface Data.logicalWidth Data.publicFits).initialState
          PiCCSInvocations.challengeWitnessStart) PiCCSActionPayloadBlock.challengeActions
        PiCCSTranscriptDirectSemantics.challengeOffset PiCCSTranscriptDirectSemantics.challengeCount
        PiCCSActionPayloadBlock.challengeInvocationCount_eq
        PiCCSTranscriptDirectSemantics.challengeFits PiCCSTranscriptDirectSemantics.challengeOffsetBound
        (by
          rw [PiCCSInvocations.challengeWitnessStart, PiCCSStarts.challengeWitnessStart_eq]
          norm_num [Spartan.piCcsPhaseOffset]) affine.1 affine.2
        PiCCSInvocationSlices.challenge_invocation
        PiCCSTranscriptDirectSemantics.challengeKindAt_eq
        (PiCCSPhaseInputs.challenge_initial application relation target suffix physical)
        (PiCCSCompilerAssertions.challenge_assertions (Spartan.pullback target)) current
    · by_cases inRound : index.val < 718
      · let current : Fin PiCCSTranscriptDirectSemantics.roundCount := ⟨index.val - 466, by
          change index.val - 466 < 252
          omega⟩
        have same : PoseidonActionSemantics.sliceIndex
            PiCCSTranscriptDirectSemantics.roundOffset PiCCSTranscriptDirectSemantics.roundCount
            PiCCSTranscriptDirectSemantics.roundFits current = index := by
          apply Fin.ext
          change 466 + (index.val - 466) = index.val
          omega
        rw [← same]
        have affine := PiCCSPhaseInputs.round_affine relation
        exact slice_values application relation target suffix physical
          PiCCSInvocations.roundPhase PiCCSInvocations.roundRowStart PiCCSInvocations.roundWitnessStart
          ((PiCCSInvocations.roundInterface Data.logicalWidth Data.publicFits).initialState
            PiCCSInvocations.roundWitnessStart) PiCCSActionPayloadBlock.roundActions
          PiCCSTranscriptDirectSemantics.roundOffset PiCCSTranscriptDirectSemantics.roundCount
          PiCCSActionPayloadBlock.roundInvocationCount_eq
          PiCCSTranscriptDirectSemantics.roundFits PiCCSTranscriptDirectSemantics.roundOffsetBound
          (by
            rw [PiCCSInvocations.roundWitnessStart, PiCCSStarts.roundTranscriptWitnessStart_eq]
            norm_num [Spartan.piCcsPhaseOffset]) affine.1 affine.2
          PiCCSInvocationSlices.round_invocation
          PiCCSTranscriptDirectSemantics.roundKindAt_eq
          (PiCCSPhaseInputs.round_initial application relation target suffix physical)
          (PiCCSCompilerAssertions.round_assertions (Spartan.pullback target)) current
      · let current : Fin PiCCSTranscriptDirectSemantics.outputCount := ⟨index.val - 718, by
          change index.val - 718 < 6886
          omega⟩
        have same : PoseidonActionSemantics.sliceIndex
            PiCCSTranscriptDirectSemantics.outputOffset PiCCSTranscriptDirectSemantics.outputCount
            PiCCSTranscriptDirectSemantics.outputFits current = index := by
          apply Fin.ext
          change 718 + (index.val - 718) = index.val
          omega
        rw [← same]
        have affine := PiCCSPhaseInputs.output_affine relation
        exact slice_values application relation target suffix physical
          PiCCSInvocations.outputPhase PiCCSInvocations.outputRowStart PiCCSInvocations.outputWitnessStart
          ((PiCCSInvocations.outputInterface Data.logicalWidth Data.publicFits).initialState
            PiCCSInvocations.outputWitnessStart) PiCCSActionPayloadBlock.outputActions
          PiCCSTranscriptDirectSemantics.outputOffset PiCCSTranscriptDirectSemantics.outputCount
          (PiCCSInvocations.outputInvocationCount_eq Data.logicalWidth Data.publicFits)
          PiCCSTranscriptDirectSemantics.outputFits PiCCSTranscriptDirectSemantics.outputOffsetBound
          (by
            rw [PiCCSInvocations.outputWitnessStart, PiCCSStarts.outputBindingWitnessStart_eq]
            norm_num [Spartan.piCcsPhaseOffset]) affine.1 affine.2
          PiCCSInvocationSlices.output_invocation
          PiCCSTranscriptDirectSemantics.outputKindAt_eq
          (PiCCSPhaseInputs.output_initial application relation target suffix physical)
          (PiCCSCompilerAssertions.output_assertions (Spartan.pullback target)) current

include relation physical in
/-- Actual physical C rows satisfy the canonical direct C Poseidon plan.
Inputs, retained S-box values and squeeze pins are derived from the same
compiler invocations and their exact source expressions. -/
theorem rowsZero_of_completed :
    (PiCCSPoseidonPlan.plan payload geometry).RowsZero raw.assignment := by
  apply PiCCSPoseidonPlan.equations_imply_rowsZero payload geometry raw.assignment
    (PerApplicationCanonicalAssignment.assignment_one raw)
  · intro index
    apply sboxes_of_input application relation target suffix physical index
    rw [inputState_eval payload geometry raw.assignment raw.retainedSource
      (encoding application target suffix)]
    funext lane
    change canonicalInput geometry raw.assignment raw.retainedSource index lane =
      Pilot.canonicalInvocationEnv (physicalInvocation index) target lane.val
    rw [Pilot.canonicalInvocationEnv_input]
    exact ((invocation_values application relation target suffix physical index).1 lane).symm
  · exact binding_zero application relation target suffix physical
      (fun index => (invocation_values application relation target suffix physical index).2)

end Completed

end NightstreamFPrime.Export.Stage1.PiCCSPoseidonCompleteness
