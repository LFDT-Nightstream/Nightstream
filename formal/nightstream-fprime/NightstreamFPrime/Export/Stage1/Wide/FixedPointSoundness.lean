import NightstreamFPrime.Export.Stage1.Wide.DecodedAccumulator
import NightstreamFPrime.Export.Stage1.Wide.FixedPoint
import NightstreamFPrime.Export.Stage1.ApplicationAssignmentSoundness
import NightstreamFPrime.Export.Stage1.RunningTransitionReducedMatrixComplete
import NightstreamFPrime.Layout.Stage1.StateDecoder
import NightstreamFPrime.Lifecycle.Stage1.Wide.Relation

/-! Decode and join the candidate's application, state, and verifier phases.
Every view reads the accepted assignment through the existing retained forms. -/

namespace NightstreamFPrime.Export.Stage1.Wide.FixedPointSoundness

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding Spec.Folding.PiCCS.PaperJoint Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Spec.HyperNova.Construction2.Paper ProductionRelation Layout.Stage1
open PiCCS.v1_1

variable (program : RetainedLayout.Program)
  (assignment : Assignment F (RetainedLayout.logicalWidth program))

def priorState : Nat → F := fun word =>
  DecodedPiDEC.env program assignment (PilotProduction.priorPreimageStart + word)

def nextState : Nat → F := fun word =>
  DecodedPiDEC.env program assignment (PilotProduction.outputPreimageStart + word)

def digest : Digest := List.ofFn fun lane : Fin PilotProduction.digestWords =>
  DecodedPrefix.pilotEnv program assignment (PilotProduction.outputDigestStart + lane.val)

def advice : AppWitness := List.ofFn fun lane : Fin program.witnessWordCount =>
  (ApplicationDirectPlan.witnessForm (Stage1Plan.referenceGeometry program) lane).eval
    (AssignmentPullback.assignment program assignment)

def runningGeometry := DirectPiDECPrefixPlan.runningGeometry (Stage1Plan.piDecGeometry program)

def runningEnv : Env := RunningTransitionReducedMatrixSemantics.decodedEnv (runningGeometry program)
  (AssignmentPullback.assignment program assignment)

private theorem prior_word (word : Fin PilotProduction.stateHashWords) :
    priorState program assignment word.val =
      DecodedPrefix.piCcsEnv program assignment (PilotProduction.priorPreimageStart + word.val) := by
  apply DecodedPiDEC.before_parent
  have h : word.val < 49393 := word.isLt
  change 0 + word.val < 19795693
  omega

private theorem next_word (word : Fin PilotProduction.stateHashWords) :
    nextState program assignment word.val =
      DecodedPrefix.piCcsEnv program assignment (PilotProduction.outputPreimageStart + word.val) := by
  apply DecodedPiDEC.before_parent
  have h : word.val < 49393 := word.isLt
  change 49663 + word.val < 19795693
  omega

private theorem prior_pilot (word : Fin PilotProduction.stateHashWords) :
    DecodedPrefix.pilotEnv program assignment (PilotProduction.priorPreimageStart + word.val) =
      priorState program assignment word.val :=
  (PilotDecodedEnvironment.priorWord_agrees _ _ word).trans (prior_word program assignment word).symm

private theorem next_pilot (word : Fin PilotProduction.stateHashWords) :
    DecodedPrefix.pilotEnv program assignment (PilotProduction.outputPreimageStart + word.val) =
      nextState program assignment word.val :=
  (PilotDecodedEnvironment.outputWord_agrees _ _ word).trans (next_word program assignment word).symm

private theorem prior_form (word : Fin PilotProduction.stateHashWords) :
    ((PiCCSOrdinaryDirectPlan.Location.priorInput word).form (Stage1Plan.piCcsGeometry program)).eval
      (AssignmentPullback.assignment program assignment) = priorState program assignment word.val :=
  (PiCCSAssignmentSoundness.decodedEnv_location _ _ (.priorInput word)).symm.trans
    (prior_word program assignment word).symm

private theorem next_form (word : Fin PilotProduction.stateHashWords) :
    ((PiCCSOrdinaryDirectPlan.Location.outputInput word).form (Stage1Plan.piCcsGeometry program)).eval
      (AssignmentPullback.assignment program assignment) = nextState program assignment word.val :=
  (PiCCSAssignmentSoundness.decodedEnv_location _ _ (.outputInput word)).symm.trans
    (next_word program assignment word).symm

private theorem running_state (index : Fin RunningTransitionSourceSupport.stateCount) :
    runningEnv program assignment (RunningTransitionDirectPlan.Location.state index).sourceColumn =
      priorState program assignment (28 + index.val) := by
  dsimp only [runningEnv, RunningTransitionReducedMatrixSemantics.decodedEnv]
  rw [RunningTransitionReducedMatrixSemantics.sourceForm_state, RunningTransitionDirectPlan.Location.state_form_eq_pilot]
  have same := prior_form program assignment (RunningTransitionDirectPlan.Location.statePreimageWord index)
  rw [PiCCSOrdinaryDirectPlan.Location.priorInput_form_eq_pilot] at same
  exact same

private theorem running_next (index : Fin RunningTransitionSourceSupport.outputCount) :
    runningEnv program assignment (RunningTransitionDirectPlan.Location.output index).sourceColumn =
      nextState program assignment index.val := by
  dsimp only [runningEnv, RunningTransitionReducedMatrixSemantics.decodedEnv]
  rw [RunningTransitionReducedMatrixSemantics.sourceForm_output, RunningTransitionDirectPlan.Location.output_form_eq_pilot]
  have same := next_form program assignment (RunningTransitionDirectPlan.Location.outputPreimageWord index)
  rw [PiCCSOrdinaryDirectPlan.Location.outputInput_form_eq_pilot] at same
  exact same

private theorem running_child (index : Fin RunningTransitionSourceSupport.piDecCount) :
    runningEnv program assignment (RunningTransitionDirectPlan.Location.piDec index).sourceColumn =
      DecodedPiDEC.env program assignment (PiDECDirectPlan.Location.proof index).sourceColumn := by
  dsimp only [runningEnv, RunningTransitionReducedMatrixSemantics.decodedEnv]
  rw [RunningTransitionReducedMatrixSemantics.sourceForm_piDec]
  rw [DecodedPiDEC.location, PiDECDirectPlan.Location.proof_form_eq_running]

private theorem point_form (coordinate : Fin productionShape.cubeVariables) (component : Fin 2) :
    (PiCCSTranscriptOutputForms.pointForm (Stage1Plan.poseidonGeometry program) coordinate component).eval
      (AssignmentPullback.assignment program assignment) =
      DecodedPrefix.piCcsEnv program assignment (PiCCSTranscriptOutputForms.pointSource coordinate component) := by
  let invocation := PiCCSTranscriptOutputForms.pointInvocation coordinate component
  let index := Fin.encodeProd (invocation, (0 : Fin 8))
  let location : PiCCSOrdinaryDirectPlan.Location :=
    .proofLogical (PiCCSOrdinaryRetainedBlocks.transcriptOutputSlot index)
  have sourceEq : location.sourceColumn = PiCCSTranscriptOutputForms.pointSource coordinate component := by
    dsimp only [location, PiCCSOrdinaryDirectPlan.Location.sourceColumn]
    rw [PiCCSOrdinaryRetainedBlocks.proofLogicalSource_transcriptOutput]
    have source := PiCCSOrdinaryRetainedBlocks.transcriptOutputSource_encodeProd invocation (0 : Fin 8)
    refine source.trans ?_
    rw [PiCCSTranscriptOutputForms.pointSource_eq_transcriptSource]
    dsimp only [PiCCSTranscriptOutputForms.transcriptSource, PiCCSTranscriptOutputForms.transcriptSourceStart]
    simp only [Fin.val_zero, Nat.add_zero]
    omega
  have formEq : location.form (Stage1Plan.piCcsGeometry program) =
      PiCCSTranscriptOutputForms.pointForm (Stage1Plan.poseidonGeometry program) coordinate component := by
    dsimp only [location]
    rw [PiCCSOrdinaryDirectPlan.Location.form_transcriptOutput]
    have geometryEq : PiCCSOrdinaryRetainedGeometry.poseidonGeometry (Stage1Plan.piCcsGeometry program) =
        Stage1Plan.poseidonGeometry program := rfl
    rw [geometryEq]
    have decoded : @Fin.decodeProd PiCCSOrdinaryRetainedBlocks.transcriptInvocationCount
        Poseidon2.width index = (invocation, (0 : Fin 8)) := Fin.decodeProd_encodeProd _
    dsimp only [PiCCSTranscriptOutputForms.pointForm]
    rw [decoded]
    rfl
  have same := PiCCSAssignmentSoundness.decodedEnv_location (Stage1Plan.piCcsGeometry program)
    (AssignmentPullback.assignment program assignment) location
  rw [sourceEq, formEq] at same
  exact same.symm

private theorem running_point (coordinate : Fin productionShape.cubeVariables) (component : Fin 2) :
    runningEnv program assignment (PiCCSTranscriptOutputForms.pointSource coordinate component) =
      DecodedPrefix.piCcsEnv program assignment (PiCCSTranscriptOutputForms.pointSource coordinate component) := by
  dsimp only [runningEnv, RunningTransitionReducedMatrixSemantics.decodedEnv]
  rw [RunningTransitionReducedMatrixSemantics.sourceForm_point]
  exact point_form program assignment coordinate component

private theorem running_external (column : Nat)
    (supported : RunningTransitionSourceSupport.External column) :
    runningEnv program assignment column = DecodedPiDEC.env program assignment column := by
  rcases supported with state | output | point | child
  · let index : Fin RunningTransitionSourceSupport.stateCount := ⟨column - 28, by
      change 28 ≤ column ∧ column < 28 + 11 at state
      change column - 28 < 11
      omega⟩
    have source : (RunningTransitionDirectPlan.Location.state index).sourceColumn = column := by
      change 28 ≤ column ∧ column < 28 + 11 at state
      change 28 + (column - 28) = column
      omega
    have value := running_state program assignment index
    change runningEnv program assignment _ = DecodedPiDEC.env program assignment (0 + (28 + index.val)) at value
    simpa only [Nat.zero_add, ← source] using! value
  · let index : Fin RunningTransitionSourceSupport.outputCount := ⟨column - 49663, by
      change 49663 ≤ column ∧ column < 49663 + 49393 at output
      change column - 49663 < 49393
      omega⟩
    have source : (RunningTransitionDirectPlan.Location.output index).sourceColumn = column := by
      change 49663 ≤ column ∧ column < 49663 + 49393 at output
      change 49663 + (column - 49663) = column
      omega
    have value := running_next program assignment index
    change runningEnv program assignment _ = DecodedPiDEC.env program assignment (49663 + index.val) at value
    simpa only [← source] using! value
  · rcases point with ⟨coordinate, left | right⟩
    · rw [← PiCCSTranscriptOutputForms.pointSource_c0 coordinate] at left
      subst column
      rw [running_point, DecodedPiDEC.before_parent]
      have h : coordinate.val < 28 := coordinate.isLt
      rw [PiCCSTranscriptOutputForms.pointSource_c0, PiCCSStarts.roundTranscriptWitnessStart_eq]
      change 15027676 + coordinate.val * 5328 + 4136 < 19795693
      omega
    · rw [← PiCCSTranscriptOutputForms.pointSource_c1 coordinate] at right
      subst column
      rw [running_point, DecodedPiDEC.before_parent]
      have h : coordinate.val < 28 := coordinate.isLt
      rw [PiCCSTranscriptOutputForms.pointSource_c1, PiCCSStarts.roundTranscriptWitnessStart_eq]
      change 15027676 + coordinate.val * 5328 + 4728 < 19795693
      omega
  · have bounds := RunningTransitionSourceSupport.piDecField_inRange child
    let index : Fin RunningTransitionSourceSupport.piDecCount := ⟨column - 28421542, by
      change 28421542 ≤ column ∧ column < 28421542 + 49248 at bounds
      change column - 28421542 < 49248
      omega⟩
    have source : 28421542 + index.val = column := by
      change 28421542 ≤ column ∧ column < 28421542 + 49248 at bounds
      dsimp only [index]
      omega
    have value := running_child program assignment index
    change runningEnv program assignment (28421542 + index.val) =
      DecodedPiDEC.env program assignment (28421542 + index.val) at value
    rwa [source] at value

private theorem running_expression (expression : Expr)
    (supported : expression.VarsSatisfy RunningTransitionSourceSupport.Logical)
    (below : expression.VarsBelow RunningTransitionInputs.phaseOffset) :
    expression.eval (runningEnv program assignment) = expression.eval (DecodedPiDEC.env program assignment) := by
  induction expression with
  | var index =>
    rcases supported with external | inverse
    · exact running_external program assignment index external
    · exact False.elim ((ne_of_lt below) inverse)
  | const value => rfl
  | add left right leftIH rightIH =>
    exact congrArg₂ (· + ·) (leftIH supported.1 below.1) (rightIH supported.2 below.2)
  | mul left right leftIH rightIH =>
    exact congrArg₂ (· * ·) (leftIH supported.1 below.1) (rightIH supported.2 below.2)

variable {width : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}

private theorem running_spec (relation : ProductionKey.LogicalRelation width publicFits)
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (rows : (Stage1Plan.running program).RowsZero assignment) :
    Lifecycle.Stage1.RunningTransition.SpecHolds (RunningTransitionInputs.interface width publicFits)
      RunningTransitionInputs.phaseOffset (DecodedPiDEC.env program assignment) := by
  let reference := AssignmentPullback.assignment program assignment
  have accepted := (AssignmentPullback.rowsZero_iff program assignment _ _).mp rows
  have matrix := (RunningTransitionReducedPlan.rowsZero_iff_accepts relation (runningGeometry program)
    (fun _ => none) reference).mp accepted
  have physical := (RunningTransitionReducedMatrixComplete.accepts_iff_rows relation (runningGeometry program)
    (RunningTransitionRetainedGeometry.oneColumn (runningGeometry program)) (fun _ => none) reference
    (DecodedPrefix.reference_one program assignment one)).mp matrix
  have specification := RunningTransitionReducedRows.soundness relation (runningEnv program assignment) physical
  let support := RunningTransitionSourceSupport.inputsSupported width publicFits
  let bounds := RunningTransitionInputs.assumptions width publicFits relation (runningEnv program assignment)
  apply Lifecycle.Stage1.RunningTransition.specHolds_of_values_eq _ _ (runningEnv program assignment) _
    (running_expression program assignment _ support.iteration bounds.iteration)
    (fun index => running_expression program assignment _ (support.initialState index) (bounds.initialState index))
    (fun index => running_expression program assignment _ (support.currentState index) (bounds.currentState index))
    (fun index => running_expression program assignment _
      (Lifecycle.Stage1.RunningTransition.runningWord_varsSatisfy _ _ support.recursive index) (bounds.recursive index))
    (fun index => running_expression program assignment _
      (Lifecycle.Stage1.RunningTransition.runningWord_varsSatisfy _ _ support.output index) (bounds.output index))
    specification

private theorem next_spec
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (rows : (Stage1Plan.nextPreimage program).RowsZero assignment) :
    Lifecycle.Stage1.NextPreimage.SpecHolds NextPreimageInputs.sourceInterface
      RunningTransitionInputs.phaseOffset (DecodedPrefix.piCcsEnv program assignment) := by
  let reference := AssignmentPullback.assignment program assignment
  let geometry := Stage1Plan.piCcsGeometry program
  let target := PiCCSAssignmentSoundness.decodedEnv geometry reference
  have accepted := (AssignmentPullback.rowsZero_iff program assignment _ _).mp rows
  have sourceRows := (OrdinarySourcePlan.Program.rowsZero_iff NextPreimageDirectPlan.program
    (NextPreimageDirectPlan.inputs geometry) reference target
    (DecodedPrefix.reference_one program assignment one) (by
      intro index
      refine ⟨?_, ?_, ?_⟩ <;> intro term member <;>
        exact PiCCSAssignmentSoundness.decodedEnv_preserves geometry reference _)).mp accepted
  have holdsRows := (NextPreimageDirectPlan.program_holds_iff_rowsHold target).mp sourceRows
  have specification := (NextPreimageInputs.spartanSpec_iff_sourceSpec NextPreimagePackage.privateStart target).mp
    (NextPreimagePackage.sourceRows_imply_spec target holdsRows)
  exact ⟨specification.iteration, specification.initialState⟩

def input (relation : ProductionKey.LogicalRelation width publicFits) :
    Input KeyDigest AppState AppWitness
      (Running (logicalWidth := width) (publicFits := publicFits))
      (Fresh (logicalWidth := width) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount where
  iteration := StateDecoder.iteration (priorState program assignment)
  z0 := StateDecoder.initialState (priorState program assignment)
  zi := StateDecoder.currentState (priorState program assignment)
  running := fun _ => StateDecoder.running width publicFits (priorState program assignment)
  fresh := AccumulatorInputs.fresh width publicFits (DecodedPiDEC.env program assignment)
  priorPc := 1
  witness := advice program assignment
  nifsProof := AccumulatorInputs.proof relation (DecodedPiDEC.env program assignment)

def output (width : Nat) (publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width) :
    Output Digest AppState (Running (logicalWidth := width) (publicFits := publicFits)) slotCount where
  zNext := StateDecoder.currentState (nextState program assignment)
  runningNext := fun _ => StateDecoder.running width publicFits (nextState program assignment)
  pcNext := functionIndex
  x := digest program assignment

def contextKey : KeyDigest := StateDecoder.keyDigest (priorState program assignment)

private theorem canonical_states (relation : ProductionKey.LogicalRelation width publicFits)
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (rows : (Stage1Plan.prefixPlan program relation).RowsZero assignment) :
    StateDecoder.Canonical (priorState program assignment) ∧ StateDecoder.Canonical (nextState program assignment) := by
  have binding := (DecodedPrefix.piCcsSpec program relation assignment one rows).statementBinding.state
  constructor
  · intro word member
    let index : Fin PilotProduction.stateHashWords := ⟨word.index, StateBinding.fixedWord_index_lt word member⟩
    rw [show word.index = index.val from rfl, prior_word]
    exact binding.priorCanonical word member
  · intro word member
    let index : Fin PilotProduction.stateHashWords := ⟨word.index, StateBinding.fixedWord_index_lt word member⟩
    rw [show word.index = index.val from rfl, next_word]
    exact binding.outputCanonical word member

private theorem context_keys (relation : ProductionKey.LogicalRelation width publicFits)
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (rows : (Stage1Plan.prefixPlan program relation).RowsZero assignment) :
    StateDecoder.keyDigest (nextState program assignment) = contextKey program assignment := by
  have preserved := (DecodedPrefix.piCcsSpec program relation assignment one rows).statementBinding.state.contextPreserved
  unfold contextKey StateDecoder.keyDigest
  apply StateDecoder.slice_congr
  intro lane
  let index : Fin PilotProduction.stateHashWords := ⟨24 + lane.val, by
    have h : lane.val < 4 := lane.isLt
    change 24 + lane.val < 49393
    omega⟩
  change nextState program assignment index.val = priorState program assignment index.val
  rw [next_word, prior_word]
  exact preserved ⟨lane.val, lane.isLt⟩

private theorem next_iteration
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (rows : (Stage1Plan.nextPreimage program).RowsZero assignment) :
    natWord (StateDecoder.iteration (priorState program assignment) + 1) =
      natWord (StateDecoder.iteration (nextState program assignment)) := by
  have equality := (next_spec program assignment one rows).iteration
  let index : Fin PilotProduction.stateHashWords := ⟨28, by decide⟩
  have fieldEq : nextState program assignment index.val = priorState program assignment index.val + 1 := by
    rw [next_word, prior_word]
    simpa only [NextPreimageInputs.sourceInterface, NextPreimageInputs.outputIterationSource,
      NextPreimageInputs.priorIterationSource, RunningTransitionInputs.iterationWordIndex,
      Expr.eval, index] using equality
  rw [StateDecoder.iteration, StateDecoder.iteration, StateDecoder.natWord_val_add_one,
    StateDecoder.natWord_val]
  exact fieldEq.symm

private theorem next_initial
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (rows : (Stage1Plan.nextPreimage program).RowsZero assignment) :
    StateDecoder.initialState (nextState program assignment) = StateDecoder.initialState (priorState program assignment) := by
  unfold StateDecoder.initialState
  apply StateDecoder.slice_congr
  intro lane
  let index : Fin PilotProduction.stateHashWords := ⟨30 + lane.val, by
    have h : lane.val < 4 := lane.isLt
    change 30 + lane.val < 49393
    omega⟩
  change nextState program assignment index.val = priorState program assignment index.val
  rw [next_word, prior_word]
  simpa only [NextPreimageInputs.sourceInterface, NextPreimageInputs.outputInitialStateSource,
    NextPreimageInputs.priorInitialStateSource, RunningTransitionInputs.initialStateWordStart,
    Expr.eval, index, Nat.add_assoc] using
      (next_spec program assignment one rows).initialState ⟨lane.val, lane.isLt⟩

private theorem next_serialization (relation : ProductionKey.LogicalRelation width publicFits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := publicFits))
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (prefixRows : (Stage1Plan.prefixPlan program relation).RowsZero assignment)
    (next : (Stage1Plan.nextPreimage program).RowsZero assignment) :
    serializePreimage (publicFits := publicFits)
      (nextHashPreimage (Lifecycle.Stage1.Wide.Relation.setup relation ajtai (contextKey program assignment))
        (input program assignment relation) (output program assignment width publicFits)) =
      serializePreimage (StateDecoder.preimage width publicFits (nextState program assignment)) := by
  have context := context_keys program assignment relation one prefixRows
  have iteration := next_iteration program assignment one next
  have initial := next_initial program assignment one next
  unfold serializePreimage nextHashPreimage Lifecycle.Stage1.Wide.Relation.setup StateDecoder.preimage
  change stateDomainTag ++ block (contextKey program assignment) ++
      [natWord (StateDecoder.iteration (priorState program assignment) + 1)] ++
      block (StateDecoder.initialState (priorState program assignment)) ++
      block (StateDecoder.currentState (nextState program assignment)) ++
      serializeRunning (StateDecoder.running width publicFits (nextState program assignment)) ++ [natWord 1] =
    stateDomainTag ++ block (StateDecoder.keyDigest (nextState program assignment)) ++
      [natWord (StateDecoder.iteration (nextState program assignment))] ++
      block (StateDecoder.initialState (nextState program assignment)) ++
      block (StateDecoder.currentState (nextState program assignment)) ++
      serializeRunning (StateDecoder.running width publicFits (nextState program assignment)) ++ [natWord 1]
  rw [context, iteration, initial]

private theorem application_step (fits : PerApplicationPackage.FitsTwoPow28 program)
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (rows : (Stage1Plan.application program fits).RowsZero assignment) :
    StateDecoder.currentState (nextState program assignment) =
      program.step (StateDecoder.currentState (priorState program assignment)) (advice program assignment) := by
  have appRows := (AssignmentPullback.rowsZero_iff program assignment _ _).mp rows
  have step := ApplicationAssignmentSoundness.rowsZero_implies_step fits (Stage1Plan.referenceGeometry program)
    (AssignmentPullback.assignment program assignment) (DecodedPrefix.reference_one program assignment one) appRows
  have prior (lane : Lifecycle.Stage1.Application.StateIndex) :=
    prior_form program assignment (ApplicationDirectPlan.Location.preimageWord lane)
  have next (lane : Lifecycle.Stage1.Application.StateIndex) :=
    next_form program assignment (ApplicationDirectPlan.Location.preimageWord lane)
  simp only [PiCCSOrdinaryDirectPlan.Location.priorInput_form_eq_pilot] at prior
  simp only [PiCCSOrdinaryDirectPlan.Location.outputInput_form_eq_pilot] at next
  simp only [prior, next] at step
  exact step

private theorem public_input (relation : ProductionKey.LogicalRelation width publicFits) :
    PriorStateHash.RepresentsPublicInput PilotProduction.priorInterface PilotProduction.witnessOffset
      (DecodedPrefix.pilotEnv program assignment)
      ((machineFor publicFits program).freshPublic (input program assignment relation).fresh) := by
  intro column
  change DecodedPrefix.pilotEnv program assignment (PilotProduction.priorPublicInputStart + column.val) =
    DecodedPiDEC.env program assignment (PilotProduction.priorPublicInputStart + column.val)
  rw [DecodedPiDEC.before_parent]
  · exact PilotDecodedEnvironment.priorPublic_agrees _ _ column
  · have bound : column.val < 270 := column.isLt
    change 49393 + column.val < 19795693
    omega

private theorem hash_slots (relation : ProductionKey.LogicalRelation width publicFits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := publicFits))
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (prefixRows : (Stage1Plan.prefixPlan program relation).RowsZero assignment)
    (next : (Stage1Plan.nextPreimage program).RowsZero assignment) :
    (machineFor publicFits program).freshPublic (input program assignment relation).fresh =
        encHash (stateHash (priorHashPreimage (Lifecycle.setup relation ajtai (contextKey program assignment))
          (input program assignment relation))) ∧
      (output program assignment width publicFits).x =
        stateHash (nextHashPreimage (Lifecycle.setup relation ajtai (contextKey program assignment))
          (input program assignment relation) (output program assignment width publicFits)) := by
  have canonical := canonical_states program assignment relation one prefixRows
  have prior := StateDecoder.priorRepresents width publicFits (DecodedPrefix.pilotEnv program assignment)
    (priorState program assignment) canonical.1 (prior_pilot program assignment)
  have nextRep := StateDecoder.outputRepresents width publicFits (DecodedPrefix.pilotEnv program assignment)
    (nextState program assignment) canonical.2 (next_pilot program assignment)
  have priorRep : PriorStateHash.RepresentsPreimage PilotProduction.priorInterface PilotProduction.witnessOffset
      (DecodedPrefix.pilotEnv program assignment)
      (priorHashPreimage (Lifecycle.setup relation ajtai (contextKey program assignment)) (input program assignment relation)) := prior
  have nextRep' : OutputHash.RepresentsPreimage PilotProduction.outputInterface
      (Lifecycle.Pilot.outputOffset PilotProduction.interface PilotProduction.witnessOffset)
      (DecodedPrefix.pilotEnv program assignment)
      (nextHashPreimage (Lifecycle.setup relation ajtai (contextKey program assignment))
        (input program assignment relation) (output program assignment width publicFits)) :=
    by
      have serial := next_serialization program assignment relation ajtai one prefixRows next
      rw [Lifecycle.Stage1.Wide.Relation.nextPreimage_unchanged] at serial
      exact nextRep.trans serial.symm
  exact Lifecycle.Pilot.builders_imply_hash_slots PilotProduction.interface PilotProduction.witnessOffset
    (DecodedPrefix.pilotEnv program assignment) relation ajtai (contextKey program assignment) program.step
    (input program assignment relation) (output program assignment width publicFits)
    (DecodedPrefix.pilot program relation assignment one prefixRows) priorRep (public_input program assignment relation) nextRep' rfl

private theorem slot_eq (slot : Fin slotCount) : slot = functionIndex := by
  apply Fin.ext
  have h : slot.val < 1 := slot.isLt
  change slot.val = 0
  omega

private theorem plan_semantics (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (relation : ProductionKey.LogicalRelation width publicFits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := publicFits))
    (fits : PerApplicationPackage.FitsTwoPow28 program)
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (rows : (Stage1Plan.plan program compiled relation fits).RowsZero assignment) :
    Lifecycle.Stage1.Wide.Relation.StepHoldsFor relation ajtai (contextKey program assignment) program
      (input program assignment relation) (output program assignment width publicFits) := by
  obtain ⟨cRows, rRows, dRows, runningRows, appRows, nextRows, _publicRows⟩ :=
    (Stage1Plan.rows_iff program compiled relation fits assignment).mp rows
  have hashes := hash_slots program assignment relation ajtai one cRows nextRows
  have runningSpec := running_spec program assignment relation one runningRows
  have application := application_step program assignment fits one appRows
  let env := DecodedPiDEC.env program assignment
  have outRunning := StateDecoder.evalOutputRunning_eq_running width publicFits env
  change StatementAbsorption.evalRunning (RunningTransitionInputs.outputRunningExpr width publicFits) env =
    (output program assignment width publicFits).runningNext functionIndex at outRunning
  have iterationZero : Lifecycle.Stage1.RunningTransition.iterationValue
      (RunningTransitionInputs.interface width publicFits) RunningTransitionInputs.phaseOffset env = 0 ↔
        (input program assignment relation).iteration = 0 := by
    dsimp only [Lifecycle.Stage1.RunningTransition.iterationValue, RunningTransitionInputs.interface,
      RunningTransitionInputs.iterationExpr, Expr.eval, input, StateDecoder.iteration, priorState,
      PilotProduction.priorPreimageStart, RunningTransitionInputs.iterationWordIndex, Nat.zero_add]
    change env 28 = 0 ↔ (env 28).val = 0
    constructor
    · intro h; rw [h]; rfl
    · intro h; exact Fin.ext h
  change FixedAugmentedTransition (Lifecycle.Stage1.Wide.Relation.setup relation ajtai (contextKey program assignment))
    (machineFor publicFits program) functionIndex (input program assignment relation)
    (output program assignment width publicFits)
  refine ⟨rfl, ?_, ?_, ?_⟩
  · exact application
  · change (output program assignment width publicFits).x =
      stateHash (nextHashPreimage (Lifecycle.Stage1.Wide.Relation.setup relation ajtai (contextKey program assignment))
        (input program assignment relation) (output program assignment width publicFits))
    rw [Lifecycle.Stage1.Wide.Relation.nextPreimage_unchanged]
    exact hashes.2
  rcases Nat.eq_zero_or_pos (input program assignment relation).iteration with base | positive
  · have fieldZero := iterationZero.mpr base
    have states : (input program assignment relation).z0 = (input program assignment relation).zi := by
      change (List.ofFn fun lane : Fin 4 => env (0 + (30 + lane.val))) =
        (List.ofFn fun lane : Fin 4 => env (0 + (35 + lane.val)))
      apply congrArg List.ofFn
      funext lane
      simpa only [RunningTransitionInputs.interface, RunningTransitionInputs.initialStateExpr,
        RunningTransitionInputs.currentStateExpr, Expr.eval, RunningTransitionInputs.initialStateWordStart,
        RunningTransitionInputs.currentStateWordStart, Nat.add_assoc] using! runningSpec.initialState fieldZero lane
    have baseOutput := RunningTransitionInputs.spec_typed_base runningSpec fieldZero
    have defaultOutput : (output program assignment width publicFits).runningNext = fun _ =>
        (Lifecycle.Stage1.Wide.Relation.setup relation ajtai (contextKey program assignment)).defaultRunning := by
      funext slot
      rw [slot_eq slot, ← outRunning]
      exact baseOutput
    exact Or.inl ⟨base, states, defaultOutput⟩
  · have nonzero : Lifecycle.Stage1.RunningTransition.iterationValue
        (RunningTransitionInputs.interface width publicFits) RunningTransitionInputs.phaseOffset env ≠ 0 := by
      intro zero
      exact (Nat.ne_of_gt positive) (iterationZero.mp zero)
    have recursiveOutput := RunningTransitionInputs.spec_typed_recursive_eq_piDecOutput relation runningSpec nonzero
    have accepted := DecodedAccumulator.decodedAccepted program relation ajtai assignment compiled one cRows rRows dRows
    have inRunning := StateDecoder.evalRunning_eq_running width publicFits env
    change AccumulatorInputs.running width publicFits env = (input program assignment relation).running functionIndex at inRunning
    have outEq : AccumulatorInputs.output relation env = (output program assignment width publicFits).runningNext functionIndex :=
      recursiveOutput.symm.trans outRunning
    change Nifs.PaperNonInteractive.verify (PiRLC.Wide.Key.key relation ajtai)
      (AccumulatorInputs.running width publicFits env) (input program assignment relation).fresh
      (input program assignment relation).nifsProof = some (AccumulatorInputs.output relation env) at accepted
    rw [inRunning, outEq] at accepted
    have valid : InRange slotCount (input program assignment relation).priorPc := by
      change InRange 1 1
      norm_num [InRange]
    refine Or.inr ⟨valid, positive, ?_, ?_, ?_⟩
    · change (machineFor publicFits program).freshPublic (input program assignment relation).fresh =
        encHash (stateHash (priorHashPreimage
          (Lifecycle.Stage1.Wide.Relation.setup relation ajtai (contextKey program assignment))
          (input program assignment relation)))
      rw [Lifecycle.Stage1.Wide.Relation.priorPreimage_unchanged]
      exact hashes.1
    · rw [slot_eq (selectedIndex valid)]
      simpa only [Spec.HyperNova.NonInteractiveMultiFold.Accepts, Lifecycle.Stage1.Wide.Relation.setup,
        Lifecycle.Stage1.Wide.Relation.nifsVerifier] using accepted
    · intro slot different
      exact False.elim (different ((slot_eq slot).trans (slot_eq (selectedIndex valid)).symm))

/-- Any accepted candidate assignment with constant column one satisfies the
complete HyperNova augmented step, using the candidate relation and wide key.
Input, output, application advice and context are decoded from that assignment. -/
theorem rowsZero_implies_stepHoldsFor (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 program)
    (ajtai : AjtaiKey (logicalWidth := RetainedLayout.logicalWidth program)
      (publicFits := FixedPoint.publicFits program))
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (rows : (FixedPoint.structuralPlan program compiled fits).RowsZero assignment) :
    Lifecycle.Stage1.Wide.Relation.StepHoldsFor (FixedPoint.relation program compiled fits) ajtai
      (contextKey program assignment) program
      (input program assignment (FixedPoint.relation program compiled fits))
      (output program assignment (RetainedLayout.logicalWidth program) (FixedPoint.publicFits program)) := by
  apply plan_semantics program assignment compiled (FixedPoint.relation program compiled fits) ajtai fits.package one
  rwa [FixedPoint.plan_fixedPoint]

end NightstreamFPrime.Export.Stage1.Wide.FixedPointSoundness
