import NightstreamFPrime.Export.Stage1.DirectPiCCSCommonPhaseSemantics
import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalAssignment
import NightstreamFPrime.Export.Stage1.PiDECDirectSupport
import NightstreamFPrime.Export.Stage1.PiDECEnvironmentCustody
import NightstreamFPrime.Export.Stage1.PiCCSCommonEnvironmentCustody
import NightstreamFPrime.Export.Stage1.PiRLCSamplerRetainedCustody
import NightstreamFPrime.Layout.Stage1.AccumulatorInputs
import NightstreamFPrime.Layout.Stage1.StateDecoder

/-!
Owns the typed HyperNova input and output decoded from one canonical raw
per-application value packet. The application and relation are Lean inputs;
the caller does not supply a typed lifecycle value or representation record.

This module decodes values only. Row acceptance and semantic soundness remain
in the per-application fixed-point soundness module.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationDecodedIO

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper

abbrev Program := Lifecycle.Stage1.Application.Program
abbrev RawValues := PerApplicationCanonicalAssignment.RawValues
abbrev FitsTwoPow28 (application : Program) :=
  PerApplicationFixedPoint.FitsTwoPow28 application

private theorem cubePoint_ext
    {Field : Type} {variableCount : Nat}
    (left right : CubePoint Field variableCount)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

private theorem evaluationFamily_ext
    (left right : StrongReduction.EvaluationFamily K productionShape)
    (pad : left.pad = right.pad)
    (matrix : left.matrix = right.matrix) : left = right := by
  cases left
  cases right
  simp_all

private theorem piDecOutput_ext
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : Lifecycle.PiDEC.v1_1.OutputBinding.Output
      logicalWidth publicFits)
    (constraintSystem : left.constraintSystem = right.constraintSystem)
    (commitment : left.commitment = right.commitment)
    (publicInput : left.publicInput = right.publicInput)
    (point : left.point = right.point)
    (evaluations : left.evaluations = right.evaluations)
    (stage : left.stage = right.stage) : left = right := by
  cases left
  cases right
  simp_all

private theorem running_ext
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (point : left.point = right.point)
    (commitments : left.commitments = right.commitments)
    (publicInputs : left.publicInputs = right.publicInputs)
    (evaluations : left.evaluations = right.evaluations) : left = right := by
  cases left
  cases right
  simp_all

private theorem kExpr_eval_eq_of_support
    (value : Circuit.Quadratic.KExpr) (allowed : Nat → Prop)
    (left right : Env)
    (support : value.c0.VarsSatisfy allowed ∧ value.c1.VarsSatisfy allowed)
    (agrees : ∀ source, allowed source → left source = right source) :
    value.eval left = value.eval right := by
  exact congrArg₂ K.mk
    (value.c0.eval_eq_of_agree_satisfy allowed left right support.1 agrees)
    (value.c1.eval_eq_of_agree_satisfy allowed left right support.2 agrees)

def relation (application : Program) (fits : FitsTwoPow28 application) :=
  PerApplicationFixedPoint.relation application fits

def geometry (application : Program) :=
  PerApplicationFixedPoint.geometry application

def prefixGeometry (application : Program) :=
  DirectApplicationPrefixPlan.prefixGeometry (geometry application)

def transitionEnv {application : Program} (raw : RawValues application) : Env :=
  Spartan.pullback
    (RunningTransitionDirectPlan.transitionEnv application raw.base)

def commonEnv {application : Program} (raw : RawValues application) : Env :=
  Spartan.pullback
    (PiRLCSamplerRetainedCustody.semanticEnv (prefixGeometry application)
      raw.assignment raw.base)

def pilotEnv {application : Program} (raw : RawValues application) : Env :=
  PilotSpartan.pullback
    (PilotOrdinaryDirectPlan.pilotEnv application raw.base)

def applicationEnv {application : Program} (raw : RawValues application) : Env :=
  ApplicationDirectPlan.sourceEnv
    (DirectApplicationPrefixPlan.applicationSource application raw.base)

def priorState {application : Program} (raw : RawValues application) : Nat → F :=
  fun word => commonEnv raw (PilotProduction.priorPreimageStart + word)

def outputState {application : Program} (raw : RawValues application) : Nat → F :=
  fun word => transitionEnv raw (PilotProduction.outputPreimageStart + word)

/-- The context key carried by the raw state. Final package closure must prove
that this value is the verifier-owned final verification-key digest. -/
def contextKey {application : Program} (raw : RawValues application) : KeyDigest :=
  StateDecoder.keyDigest (priorState raw)

theorem commonEnv_eq_transitionEnv_of_source
    {application : Program} (raw : RawValues application)
    (source : Nat) (support : PiCCSOrdinarySourceSupport.Source source) :
    commonEnv raw source = transitionEnv raw source := by
  unfold commonEnv transitionEnv Spartan.pullback
  exact PiCCSCommonEnvironmentCustody.semanticEnv_eq_transitionEnv_of_target
    (prefixGeometry application) raw.assignment raw.base
    (PiCCSOrdinarySourceSupport.source_target source support)

theorem commonEnv_eq_transitionEnv_of_piDecSource
    {application : Program} (raw : RawValues application)
    (source : Nat) (support : PiDECSourceSupport.Source source) :
    commonEnv raw source = transitionEnv raw source := by
  unfold commonEnv transitionEnv Spartan.pullback
  exact PiDECEnvironmentCustody.semanticEnv_source_eq_transitionEnv
    (prefixGeometry application) raw.assignment raw.base support

private theorem roundC0Source
    (coordinate : Fin productionShape.cubeVariables) :
    PiCCSOrdinarySourceSupport.Source
      (PiCCSStarts.roundTranscriptWitnessStart +
        coordinate.val * RunningTransitionInputs.roundStride +
        RunningTransitionInputs.roundSampleC0Offset) := by
  apply PiCCSOrdinarySourceSupport.transcript_output_source
  refine ⟨⟨472 + coordinate.val * 9, ?_⟩, ⟨0, ?_⟩, ?_⟩
  · have bound := coordinate.isLt
    change coordinate.val < 28 at bound
    rw [PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq]
    omega
  · norm_num [Spec.Poseidon2.width]
  · rw [PiCCSStarts.roundTranscriptWitnessStart_eq,
      PiCCSInputs.phaseOffset_eq]
    norm_num [RunningTransitionInputs.roundStride,
      RunningTransitionInputs.roundSampleC0Offset]
    omega

private theorem roundC1Source
    (coordinate : Fin productionShape.cubeVariables) :
    PiCCSOrdinarySourceSupport.Source
      (PiCCSStarts.roundTranscriptWitnessStart +
        coordinate.val * RunningTransitionInputs.roundStride +
        RunningTransitionInputs.roundSampleC1Offset) := by
  apply PiCCSOrdinarySourceSupport.transcript_output_source
  refine ⟨⟨473 + coordinate.val * 9, ?_⟩, ⟨0, ?_⟩, ?_⟩
  · have bound := coordinate.isLt
    change coordinate.val < 28 at bound
    rw [PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq]
    omega
  · norm_num [Spec.Poseidon2.width]
  · rw [PiCCSStarts.roundTranscriptWitnessStart_eq,
      PiCCSInputs.phaseOffset_eq]
    norm_num [RunningTransitionInputs.roundStride,
      RunningTransitionInputs.roundSampleC1Offset]
    omega

theorem pilotEnv_eq_transitionEnv_of_lt
    {application : Program} (raw : RawValues application)
    (source : Nat) (bound : source < Spartan.pilotSourceColumnCount) :
    pilotEnv raw source = transitionEnv raw source := by
  unfold pilotEnv transitionEnv PilotSpartan.pullback Spartan.pullback
    PilotOrdinaryDirectPlan.pilotEnv Spartan.sourceToSpartan
  rw [if_pos bound]

private theorem applicationInputEnv_eq_transition
    {application : Program} (raw : RawValues application)
    (index : Lifecycle.Stage1.Application.StateIndex) :
    applicationEnv raw (ApplicationInputs.inputColumn index) =
      transitionEnv raw (ApplicationInputs.inputSourceColumn index) := by
  have sourceBound : ApplicationInputs.inputColumn index <
      ApplicationRetainedBlocks.sourceWidth application :=
    (ApplicationRetainedBlocks.inputBlock application).source index |>.isLt
  have privateBound : ApplicationInputs.inputColumn index <
      PiRLCProductPlan.basePackage.layout.constantColumn := by
    have indexBound := index.isLt
    have constant : PiRLCProductPlan.basePackage.layout.constantColumn =
        29336446 :=
      NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.1
    rw [constant, ApplicationInputs.inputColumn_value]
    norm_num [ApplicationInputs.currentWordStart,
      Lifecycle.Stage1.Application.stateWordCount] at indexBound ⊢
    omega
  have packageBound : ApplicationInputs.inputColumn index <
      PiRLCProductPlan.basePackage.layout.totalColumnCount := by
    have total : PiRLCProductPlan.basePackage.layout.totalColumnCount =
        29336725 :=
      NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.2.2
    rw [total, ApplicationInputs.inputColumn_value]
    have indexBound := index.isLt
    norm_num [ApplicationInputs.currentWordStart,
      Lifecycle.Stage1.Application.stateWordCount] at indexBound ⊢
    omega
  unfold applicationEnv ApplicationDirectPlan.sourceEnv
  rw [dif_pos sourceBound]
  unfold DirectApplicationPrefixPlan.applicationSource
  unfold transitionEnv Spartan.pullback
  change raw.base _ = RunningTransitionDirectPlan.transitionEnv application
    raw.base (ApplicationInputs.inputColumn index)
  unfold RunningTransitionDirectPlan.transitionEnv
  rw [dif_pos packageBound]
  apply congrArg raw.base
  apply Fin.ext
  change ApplicationInputs.inputColumn index =
    PerApplicationPackage.shiftColumn application
      (ApplicationInputs.inputColumn index)
  rw [PerApplicationPackage.shiftColumn_private application _ privateBound]

private theorem applicationOutputEnv_eq_transition
    {application : Program} (raw : RawValues application)
    (index : Lifecycle.Stage1.Application.StateIndex) :
    applicationEnv raw (ApplicationInputs.outputColumn index) =
      transitionEnv raw (ApplicationInputs.outputSourceColumn index) := by
  have sourceBound : ApplicationInputs.outputColumn index <
      ApplicationRetainedBlocks.sourceWidth application :=
    (ApplicationRetainedBlocks.outputBlock application).source index |>.isLt
  have privateBound : ApplicationInputs.outputColumn index <
      PiRLCProductPlan.basePackage.layout.constantColumn := by
    have indexBound := index.isLt
    have constant : PiRLCProductPlan.basePackage.layout.constantColumn =
        29336446 :=
      NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.1
    rw [constant, ApplicationInputs.outputColumn_value]
    norm_num [Lifecycle.Stage1.Application.stateWordCount] at indexBound ⊢
    omega
  have packageBound : ApplicationInputs.outputColumn index <
      PiRLCProductPlan.basePackage.layout.totalColumnCount := by
    have total : PiRLCProductPlan.basePackage.layout.totalColumnCount =
        29336725 :=
      NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.2.2
    rw [total, ApplicationInputs.outputColumn_value]
    have indexBound := index.isLt
    norm_num [Lifecycle.Stage1.Application.stateWordCount] at indexBound ⊢
    omega
  unfold applicationEnv ApplicationDirectPlan.sourceEnv
  rw [dif_pos sourceBound]
  unfold DirectApplicationPrefixPlan.applicationSource
  unfold transitionEnv Spartan.pullback
  change raw.base _ = RunningTransitionDirectPlan.transitionEnv application
    raw.base (ApplicationInputs.outputColumn index)
  unfold RunningTransitionDirectPlan.transitionEnv
  rw [dif_pos packageBound]
  apply congrArg raw.base
  apply Fin.ext
  change ApplicationInputs.outputColumn index =
    PerApplicationPackage.shiftColumn application
      (ApplicationInputs.outputColumn index)
  rw [PerApplicationPackage.shiftColumn_private application _ privateBound]

private theorem priorSource
    (word : Fin PilotProduction.stateHashWords) :
    PiCCSOrdinarySourceSupport.Source
      (PilotProduction.priorPreimageStart + word.val) := by
  apply PiCCSOrdinarySourceSupport.external_source
  apply PiCCSOrdinarySourceSupport.external_prior
  exact ⟨by omega, by
    have bound := word.isLt
    omega⟩

private theorem outputSource
    (word : Fin PilotProduction.stateHashWords) :
    PiCCSOrdinarySourceSupport.Source
      (PilotProduction.outputPreimageStart + word.val) := by
  apply PiCCSOrdinarySourceSupport.external_source
  apply PiCCSOrdinarySourceSupport.external_output
  exact ⟨by omega, by
    have bound := word.isLt
    omega⟩

private theorem priorPublicSource
    (column : Fin Lifecycle.PriorStateHash.publicWidth) :
    PiCCSOrdinarySourceSupport.Source
      (PilotProduction.priorPublicInputStart + column.val) := by
  apply PiCCSOrdinarySourceSupport.external_source
  apply PiCCSOrdinarySourceSupport.external_public
  exact ⟨by omega, by
    have bound := column.isLt
    norm_num [Lifecycle.PriorStateHash.publicWidth_eq] at bound ⊢
    omega⟩

private theorem priorPilotBound
    (word : Fin PilotProduction.stateHashWords) :
    PilotProduction.priorPreimageStart + word.val <
      Spartan.pilotSourceColumnCount := by
  have bound := word.isLt
  norm_num [Spartan.pilotSourceColumnCount,
    PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq]
    at bound ⊢
  omega

private theorem outputPilotBound
    (word : Fin PilotProduction.stateHashWords) :
    PilotProduction.outputPreimageStart + word.val <
      Spartan.pilotSourceColumnCount := by
  have bound := word.isLt
  norm_num [Spartan.pilotSourceColumnCount,
    PilotProduction.outputPreimageStart,
    PilotProduction.priorPublicInputStart,
    PilotProduction.priorPreimageStart,
    PilotProduction.stateHashWords_eq, Lifecycle.PriorStateHash.publicWidth,
    ringDegree, publicRingColumns] at bound ⊢
  omega

private theorem priorPublicPilotBound
    (column : Fin Lifecycle.PriorStateHash.publicWidth) :
    PilotProduction.priorPublicInputStart + column.val <
      Spartan.pilotSourceColumnCount := by
  have bound := column.isLt
  norm_num [Spartan.pilotSourceColumnCount,
    PilotProduction.priorPublicInputStart,
    PilotProduction.priorPreimageStart,
    PilotProduction.stateHashWords_eq,
    Lifecycle.PriorStateHash.publicWidth_eq] at bound ⊢
  omega

theorem priorState_eq_transition
    {application : Program} (raw : RawValues application)
    (word : Fin PilotProduction.stateHashWords) :
    priorState raw word.val = transitionEnv raw
      (PilotProduction.priorPreimageStart + word.val) :=
  commonEnv_eq_transitionEnv_of_source raw _ (priorSource word)

theorem outputState_eq_transition
    {application : Program} (raw : RawValues application)
    (word : Fin PilotProduction.stateHashWords) :
    outputState raw word.val = transitionEnv raw
      (PilotProduction.outputPreimageStart + word.val) := by
  rfl

theorem priorState_eq_pilot
    {application : Program} (raw : RawValues application)
    (word : Fin PilotProduction.stateHashWords) :
    priorState raw word.val = pilotEnv raw
      (PilotProduction.priorPreimageStart + word.val) := by
  rw [priorState_eq_transition raw word]
  exact (pilotEnv_eq_transitionEnv_of_lt raw _ (priorPilotBound word)).symm

theorem outputState_eq_pilot
    {application : Program} (raw : RawValues application)
    (word : Fin PilotProduction.stateHashWords) :
    outputState raw word.val = pilotEnv raw
      (PilotProduction.outputPreimageStart + word.val) := by
  rw [outputState_eq_transition raw word]
  exact (pilotEnv_eq_transitionEnv_of_lt raw _ (outputPilotBound word)).symm

theorem priorCanonical_of_stateBinding
    {application : Program} (raw : RawValues application)
    (canonical : Lifecycle.PiCCS.v1_1.StateBinding.StateCanonical
      PiCCSInputs.priorStateWord (commonEnv raw)) :
    StateDecoder.Canonical (priorState raw) := by
  intro word member
  simpa [priorState, PiCCSInputs.priorStateWord] using canonical word member

theorem outputCanonical_of_stateBinding
    {application : Program} (raw : RawValues application)
    (canonical : Lifecycle.PiCCS.v1_1.StateBinding.StateCanonical
      PiCCSInputs.outputStateWord (commonEnv raw)) :
    StateDecoder.Canonical (outputState raw) := by
  intro word member
  let bounded : Fin PilotProduction.stateHashWords := ⟨word.index, by
    have indexBound :=
      Lifecycle.PiCCS.v1_1.StateBinding.fixedWord_index_lt word member
    simpa [PilotProduction.stateHashWords_eq] using indexBound⟩
  calc
    outputState raw word.index = transitionEnv raw
        (PilotProduction.outputPreimageStart + word.index) := by rfl
    _ = commonEnv raw
        (PilotProduction.outputPreimageStart + word.index) :=
      (commonEnv_eq_transitionEnv_of_source raw _
        (outputSource bounded)).symm
    _ = word.value := by
      simpa [PiCCSInputs.outputStateWord] using canonical word member

theorem priorRepresents
    {application : Program} (raw : RawValues application)
    (canonical : StateDecoder.Canonical (priorState raw)) :
    Lifecycle.PriorStateHash.RepresentsPreimage
      PilotProduction.priorInterface PilotProduction.witnessOffset
      (pilotEnv raw)
      (StateDecoder.preimage
        (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application) (priorState raw)) := by
  unfold Lifecycle.PriorStateHash.RepresentsPreimage
  rw [PilotProduction.priorInterface_preimage_apply]
  simp only [PilotProduction.priorPreimage, Hash.evalList,
    PilotProduction.variableExprs, List.map_ofFn]
  rw [StateDecoder.serializePreimage_preimage _ _ canonical]
  apply congrArg List.ofFn
  funext word
  exact (priorState_eq_pilot raw word).symm

theorem outputRepresents
    {application : Program} (raw : RawValues application)
    (canonical : StateDecoder.Canonical (outputState raw)) :
    Lifecycle.OutputHash.RepresentsPreimage PilotProduction.outputInterface
      (Lifecycle.Pilot.outputOffset PilotProduction.interface
        PilotProduction.witnessOffset)
      (pilotEnv raw)
      (StateDecoder.preimage
        (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application) (outputState raw)) := by
  unfold Lifecycle.OutputHash.RepresentsPreimage
  rw [PilotProduction.outputInterface_preimage_apply]
  simp only [PilotProduction.outputPreimage, Hash.evalList,
    PilotProduction.variableExprs, List.map_ofFn]
  rw [StateDecoder.serializePreimage_preimage _ _ canonical]
  apply congrArg List.ofFn
  funext word
  exact (outputState_eq_pilot raw word).symm

def outputDigest {application : Program} (raw : RawValues application) : Digest :=
  raw.outputDigest

theorem outputDigestRepresents
    {application : Program} (raw : RawValues application) :
    Lifecycle.OutputHash.RepresentsDigest PilotProduction.outputInterface
      (Lifecycle.Pilot.outputOffset PilotProduction.interface
        PilotProduction.witnessOffset)
      (pilotEnv raw) (outputDigest raw) := by
  rfl

theorem semantics_imply_canonicalStates
    (application : Program) (fits : FitsTwoPow28 application)
    (raw : RawValues application)
    (semantics : DirectApplicationPrefixPlan.Semantics
      (relation application fits) (geometry application) raw.assignment
      raw.base raw.groupValue raw.products) :
    StateDecoder.Canonical (priorState raw) ∧
      StateDecoder.Canonical (outputState raw) := by
  have piCcs :=
    DirectPiCCSCommonPhaseSemantics.semantics_imply_piCcsSpecHolds
      (relation application fits) (prefixGeometry application) raw.assignment
      raw.base raw.groupValue raw.products semantics.runningPrefix
  have binding := piCcs.statementBinding.state
  constructor
  · apply priorCanonical_of_stateBinding raw
    simpa [PiCCSInvocations.parentInterface, PiCCSInputs.interface,
      Lifecycle.PiCCS.v1_1.Formal.statementBindingInterface,
      Lifecycle.PiCCS.v1_1.Formal.atOffset, commonEnv] using
        binding.priorCanonical
  · apply outputCanonical_of_stateBinding raw
    simpa [PiCCSInvocations.parentInterface, PiCCSInputs.interface,
      Lifecycle.PiCCS.v1_1.Formal.statementBindingInterface,
      Lifecycle.PiCCS.v1_1.Formal.atOffset, commonEnv] using
        binding.outputCanonical

theorem semantics_imply_contextKeys
    (application : Program) (fits : FitsTwoPow28 application)
    (raw : RawValues application)
    (semantics : DirectApplicationPrefixPlan.Semantics
      (relation application fits) (geometry application) raw.assignment
      raw.base raw.groupValue raw.products) :
    StateDecoder.keyDigest (outputState raw) = contextKey raw := by
  have piCcs :=
    DirectPiCCSCommonPhaseSemantics.semantics_imply_piCcsSpecHolds
      (relation application fits) (prefixGeometry application) raw.assignment
      raw.base raw.groupValue raw.products semantics.runningPrefix
  have preserved := piCcs.statementBinding.state.contextPreserved
  unfold contextKey StateDecoder.keyDigest
  apply StateDecoder.slice_congr
  intro lane
  let contextLane : Fin 4 := ⟨lane.val, by
    have bound := lane.isLt
    simpa [PilotProduction.digestWords, PilotValues.digestWords] using bound⟩
  have word := preserved contextLane
  let outputWord : Fin PilotProduction.stateHashWords :=
    ⟨Lifecycle.PiCCS.v1_1.StateBinding.contextWordStart + contextLane.val, by
      have bound := contextLane.isLt
      norm_num [Lifecycle.PiCCS.v1_1.StateBinding.contextWordStart,
        PilotProduction.stateHashWords_eq] at bound ⊢
      omega⟩
  have outputCustody : transitionEnv raw
      (PilotProduction.outputPreimageStart + outputWord.val) =
    commonEnv raw
      (PilotProduction.outputPreimageStart + outputWord.val) :=
    (commonEnv_eq_transitionEnv_of_source raw _
      (outputSource outputWord)).symm
  rw [show outputState raw outputWord.val = transitionEnv raw
    (PilotProduction.outputPreimageStart + outputWord.val) by rfl,
    outputCustody]
  simpa [priorState, contextLane, outputWord,
    PiCCSInvocations.parentInterface, PiCCSInputs.interface,
    Lifecycle.PiCCS.v1_1.Formal.statementBindingInterface,
    Lifecycle.PiCCS.v1_1.Formal.atOffset,
    Lifecycle.PiCCS.v1_1.StateBinding.contextWordStart,
    PiCCSInputs.priorStateWord, PiCCSInputs.outputStateWord,
    commonEnv] using word

def input (application : Program) (fits : FitsTwoPow28 application)
    (raw : RawValues application) :
    Input KeyDigest AppState AppWitness
      (Running
        (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      (Fresh
        (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      (Proof (ProductionKey.degreeBound (relation application fits))) slotCount where
  iteration := StateDecoder.iteration (priorState raw)
  z0 := StateDecoder.initialState (priorState raw)
  zi := StateDecoder.currentState (priorState raw)
  running := fun _ => StateDecoder.running
    (PerApplicationFixedPoint.logicalWidth application)
    (PerApplicationFixedPoint.publicFits application) (priorState raw)
  fresh := AccumulatorInputs.fresh
    (PerApplicationFixedPoint.logicalWidth application)
    (PerApplicationFixedPoint.publicFits application) (commonEnv raw)
  priorPc := 1
  witness := Lifecycle.Stage1.Application.witnessValue
    (ApplicationInputs.interface application)
    (ApplicationInputs.localStart application) (applicationEnv raw)
  nifsProof := AccumulatorInputs.proof (relation application fits) (commonEnv raw)

def output (application : Program) (raw : RawValues application) :
    Output Digest AppState
      (Running
        (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      slotCount where
  zNext := StateDecoder.currentState (outputState raw)
  runningNext := fun _ => StateDecoder.running
    (PerApplicationFixedPoint.logicalWidth application)
    (PerApplicationFixedPoint.publicFits application) (outputState raw)
  pcNext := functionIndex
  x := outputDigest raw

theorem priorPublicInputRepresents
    (application : Program) (fits : FitsTwoPow28 application)
    (raw : RawValues application) :
    Lifecycle.PriorStateHash.RepresentsPublicInput
      PilotProduction.priorInterface PilotProduction.witnessOffset
      (pilotEnv raw)
      ((machineFor (PerApplicationFixedPoint.publicFits application)
        application).freshPublic (input application fits raw).fresh) := by
  intro column
  change pilotEnv raw
      (PilotProduction.priorPublicInputStart + column.val) =
    commonEnv raw
      (PilotProduction.priorPublicInputStart + column.val)
  calc
    pilotEnv raw (PilotProduction.priorPublicInputStart + column.val) =
        transitionEnv raw
          (PilotProduction.priorPublicInputStart + column.val) :=
      pilotEnv_eq_transitionEnv_of_lt raw _
        (priorPublicPilotBound column)
    _ = commonEnv raw
          (PilotProduction.priorPublicInputStart + column.val) :=
      (commonEnv_eq_transitionEnv_of_source raw _
        (priorPublicSource column)).symm

theorem runningInputRepresents
    (application : Program) (fits : FitsTwoPow28 application)
    (raw : RawValues application) :
    AccumulatorInputs.running
        (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application) (commonEnv raw) =
      (input application fits raw).running functionIndex := by
  change PiCCS.v1_1.StatementAbsorption.evalRunning
      (PiCCSInputs.runningExpr
        (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application)) (commonEnv raw) =
    StateDecoder.running
      (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application) (priorState raw)
  simpa [priorState] using StateDecoder.evalRunning_eq_running
    (PerApplicationFixedPoint.logicalWidth application)
    (PerApplicationFixedPoint.publicFits application) (commonEnv raw)

theorem runningOutputRepresents
    (application : Program) (raw : RawValues application) :
    PiCCS.v1_1.StatementAbsorption.evalRunning
        (RunningTransitionInputs.outputRunningExpr
          (PerApplicationFixedPoint.logicalWidth application)
          (PerApplicationFixedPoint.publicFits application))
        (transitionEnv raw) =
      (output application raw).runningNext functionIndex := by
  change PiCCS.v1_1.StatementAbsorption.evalRunning
      (RunningTransitionInputs.outputRunningExpr
        (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      (transitionEnv raw) =
    StateDecoder.running
      (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application) (outputState raw)
  simpa [outputState] using StateDecoder.evalOutputRunning_eq_running
    (PerApplicationFixedPoint.logicalWidth application)
    (PerApplicationFixedPoint.publicFits application) (transitionEnv raw)

theorem iterationZeroRepresents
    (application : Program) (fits : FitsTwoPow28 application)
    (raw : RawValues application) :
    Lifecycle.Stage1.RunningTransition.iterationValue
        (RunningTransitionInputs.interface
          (PerApplicationFixedPoint.logicalWidth application)
          (PerApplicationFixedPoint.publicFits application))
        RunningTransitionInputs.phaseOffset (transitionEnv raw) = 0 ↔
      (input application fits raw).iteration = 0 := by
  let word : Fin PilotProduction.stateHashWords :=
    ⟨RunningTransitionInputs.iterationWordIndex, by
      norm_num [RunningTransitionInputs.iterationWordIndex,
        PilotProduction.stateHashWords_eq]⟩
  have custody := priorState_eq_transition raw word
  have transitionIteration :
      Lifecycle.Stage1.RunningTransition.iterationValue
          (RunningTransitionInputs.interface
            (PerApplicationFixedPoint.logicalWidth application)
            (PerApplicationFixedPoint.publicFits application))
          RunningTransitionInputs.phaseOffset (transitionEnv raw) =
        priorState raw word.val := by
    simpa [Lifecycle.Stage1.RunningTransition.iterationValue,
      RunningTransitionInputs.interface,
      RunningTransitionInputs.iterationExpr, word] using custody.symm
  rw [transitionIteration]
  rw [show (input application fits raw).iteration =
    StateDecoder.iteration (priorState raw) by rfl]
  change priorState raw word.val = 0 ↔ (priorState raw word.val).val = 0
  constructor
  · intro equality
    rw [equality]
    rfl
  · intro valueZero
    apply Fin.ext
    simpa using valueZero

theorem initialStateRepresents
    (application : Program) (fits : FitsTwoPow28 application)
    (raw : RawValues application) :
    List.ofFn (fun index =>
        (RunningTransitionInputs.initialStateExpr index).eval
          (transitionEnv raw)) =
      (input application fits raw).z0 := by
  rw [show (input application fits raw).z0 =
    StateDecoder.initialState (priorState raw) by rfl]
  unfold StateDecoder.initialState StateDecoder.slice
  apply congrArg List.ofFn
  funext index
  let word : Fin PilotProduction.stateHashWords :=
    ⟨RunningTransitionInputs.initialStateWordStart + index.val, by
      have bound := index.isLt
      norm_num [RunningTransitionInputs.initialStateWordStart,
        Lifecycle.Stage1.RunningTransition.stateWordCount,
        PilotProduction.stateHashWords_eq] at bound ⊢
      omega⟩
  simpa [input, StateDecoder.initialState, StateDecoder.slice,
    RunningTransitionInputs.initialStateExpr, word, Nat.add_assoc] using
      (priorState_eq_transition raw word).symm

theorem currentStateRepresents
    (application : Program) (fits : FitsTwoPow28 application)
    (raw : RawValues application) :
    List.ofFn (fun index =>
        (RunningTransitionInputs.currentStateExpr index).eval
          (transitionEnv raw)) =
      (input application fits raw).zi := by
  rw [show (input application fits raw).zi =
    StateDecoder.currentState (priorState raw) by rfl]
  unfold StateDecoder.currentState StateDecoder.slice
  apply congrArg List.ofFn
  funext index
  let word : Fin PilotProduction.stateHashWords :=
    ⟨RunningTransitionInputs.currentStateWordStart + index.val, by
      have bound := index.isLt
      norm_num [RunningTransitionInputs.currentStateWordStart,
        Lifecycle.Stage1.RunningTransition.stateWordCount,
        PilotProduction.stateHashWords_eq] at bound ⊢
      omega⟩
  simpa [input, StateDecoder.currentState, StateDecoder.slice,
    RunningTransitionInputs.currentStateExpr, word, Nat.add_assoc] using
      (priorState_eq_transition raw word).symm

theorem applicationInputRepresents
    (application : Program) (fits : FitsTwoPow28 application)
    (raw : RawValues application) :
    Lifecycle.Stage1.Application.inputState
        (ApplicationInputs.interface application)
        (ApplicationInputs.localStart application) (applicationEnv raw) =
      (input application fits raw).zi := by
  rw [show (input application fits raw).zi =
    StateDecoder.currentState (priorState raw) by rfl]
  unfold Lifecycle.Stage1.Application.inputState ApplicationInputs.interface
    StateDecoder.currentState StateDecoder.slice
  apply congrArg List.ofFn
  funext index
  let word : Fin PilotProduction.stateHashWords :=
    ⟨ApplicationInputs.currentWordStart + index.val, by
      have bound := index.isLt
      norm_num [ApplicationInputs.currentWordStart,
        Lifecycle.Stage1.Application.stateWordCount,
        PilotProduction.stateHashWords_eq] at bound ⊢
      omega⟩
  calc
    applicationEnv raw (ApplicationInputs.inputColumn index) =
        transitionEnv raw (ApplicationInputs.inputSourceColumn index) :=
      applicationInputEnv_eq_transition raw index
    _ = priorState raw word.val := by
      simpa [ApplicationInputs.inputSourceColumn, word, Nat.add_assoc] using
        (priorState_eq_transition raw word).symm

theorem applicationOutputRepresents
    (application : Program) (raw : RawValues application) :
    Lifecycle.Stage1.Application.outputState
        (ApplicationInputs.interface application)
        (ApplicationInputs.localStart application) (applicationEnv raw) =
      (output application raw).zNext := by
  rw [show (output application raw).zNext =
    StateDecoder.currentState (outputState raw) by rfl]
  unfold Lifecycle.Stage1.Application.outputState ApplicationInputs.interface
    StateDecoder.currentState StateDecoder.slice
  apply congrArg List.ofFn
  funext index
  let word : Fin PilotProduction.stateHashWords :=
    ⟨ApplicationInputs.currentWordStart + index.val, by
      have bound := index.isLt
      norm_num [ApplicationInputs.currentWordStart,
        Lifecycle.Stage1.Application.stateWordCount,
        PilotProduction.stateHashWords_eq] at bound ⊢
      omega⟩
  calc
    applicationEnv raw (ApplicationInputs.outputColumn index) =
        transitionEnv raw (ApplicationInputs.outputSourceColumn index) :=
      applicationOutputEnv_eq_transition raw index
    _ = outputState raw word.val := by
      simpa [ApplicationInputs.outputSourceColumn, word, Nat.add_assoc] using
        (outputState_eq_transition raw word).symm

private theorem piDecPoint_eq
    {application : Program} (raw : RawValues application) :
    PiCCS.v1_1.StatementAbsorption.evalPoint
        ((PiDECInputs.interface
          (PerApplicationFixedPoint.logicalWidth application)
          (PerApplicationFixedPoint.publicFits application)).point
            PiDECInputs.phaseOffset)
        (commonEnv raw) =
      PiCCS.v1_1.StatementAbsorption.evalPoint
        ((PiDECInputs.interface
          (PerApplicationFixedPoint.logicalWidth application)
          (PerApplicationFixedPoint.publicFits application)).point
            PiDECInputs.phaseOffset)
        (transitionEnv raw) := by
  apply cubePoint_ext
  change List.ofFn (fun coordinate =>
      ((PiDECInputs.interface
        (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application)).point
          PiDECInputs.phaseOffset coordinate).eval (commonEnv raw)) =
    List.ofFn (fun coordinate =>
      ((PiDECInputs.interface
        (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application)).point
          PiDECInputs.phaseOffset coordinate).eval (transitionEnv raw))
  apply congrArg List.ofFn
  funext coordinate
  have pointEq :
      (PiDECInputs.interface
        (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application)).point
          PiDECInputs.phaseOffset coordinate =
        RunningTransitionInputs.directRoundPoint
          PiCCSStarts.roundTranscriptWitnessStart coordinate := by
    simpa [RunningTransitionInputs.recursiveRunningExpr,
      RunningTransitionInputs.piDecInterface] using
        (RunningTransitionInputs.recursivePoint_eq_direct
          (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
          (publicFits := PerApplicationFixedPoint.publicFits application)
          coordinate)
  rw [pointEq]
  unfold RunningTransitionInputs.directRoundPoint Circuit.Quadratic.KExpr.eval
  apply congrArg₂ K.mk
  · exact commonEnv_eq_transitionEnv_of_source raw _ (roundC0Source coordinate)
  · exact commonEnv_eq_transitionEnv_of_source raw _ (roundC1Source coordinate)

private theorem piDecOutputs_eq
    (application : Program) (fits : FitsTwoPow28 application)
    (raw : RawValues application) :
    Lifecycle.PiDEC.v1_1.Semantics.output (relation application fits)
        (PiDECInputs.interface
          (PerApplicationFixedPoint.logicalWidth application)
          (PerApplicationFixedPoint.publicFits application))
        PiDECInputs.phaseOffset (commonEnv raw) =
      Lifecycle.PiDEC.v1_1.Semantics.output (relation application fits)
        (PiDECInputs.interface
          (PerApplicationFixedPoint.logicalWidth application)
          (PerApplicationFixedPoint.publicFits application))
        PiDECInputs.phaseOffset (transitionEnv raw) := by
  funext child
  apply piDecOutput_ext
  · rfl
  · funext row coefficient
    apply Expr.eval_eq_of_agree_satisfy _ PiDECSourceSupport.Source
      (commonEnv raw) (transitionEnv raw)
    · exact PiDECDirectSupport.messageCommitment_supported child row coefficient
    · exact commonEnv_eq_transitionEnv_of_piDecSource raw
  · funext coordinate
    apply Expr.eval_eq_of_agree_satisfy _ PiDECSourceSupport.Source
      (commonEnv raw) (transitionEnv raw)
    · exact PiDECDirectSupport.digit_supported child coordinate
    · exact commonEnv_eq_transitionEnv_of_piDecSource raw
  · exact piDecPoint_eq raw
  · apply congrArg (fun value => #[value])
    apply evaluationFamily_ext
    · funext coefficient
      exact kExpr_eval_eq_of_support _ PiDECSourceSupport.Source
        (commonEnv raw) (transitionEnv raw)
        (PiDECDirectSupport.messageEvalK_supported child coefficient)
        (commonEnv_eq_transitionEnv_of_piDecSource raw)
    · funext matrix coefficient
      exact kExpr_eval_eq_of_support _ PiDECSourceSupport.Source
        (commonEnv raw) (transitionEnv raw)
        (PiDECDirectSupport.messageEvalA_supported child matrix coefficient)
        (commonEnv_eq_transitionEnv_of_piDecSource raw)
  · rfl

theorem accumulatorOutputEnvRepresents
    (application : Program) (fits : FitsTwoPow28 application)
    (raw : RawValues application) :
    AccumulatorInputs.output (relation application fits) (commonEnv raw) =
      AccumulatorInputs.output (relation application fits)
        (transitionEnv raw) := by
  unfold AccumulatorInputs.output RunningTransitionInputs.piDecRunningOutput
  have outputs := piDecOutputs_eq application fits raw
  apply running_ext
  · exact piDecPoint_eq raw
  · funext source
    exact congrArg (fun family =>
      (family (RunningTransitionInputs.childOfRunning source)).commitment)
      outputs
  · funext source
    exact congrArg (fun family =>
      (family (RunningTransitionInputs.childOfRunning source)).publicInput)
      outputs
  · funext source
    exact congrArg (fun family =>
      (family (RunningTransitionInputs.childOfRunning source)).evaluations.getD
        0 evaluationZero) outputs

@[simp] theorem outputDigest_length {application : Program}
    (raw : RawValues application) :
    (outputDigest raw).length = PilotProduction.digestWords := by
  simp [outputDigest,
    PerApplicationCanonicalAssignment.RawValues.outputDigest]

theorem semantics_imply_nextIterationWord
    (application : Program) (fits : FitsTwoPow28 application)
    (raw : RawValues application)
    (semantics : DirectApplicationPrefixPlan.Semantics
      (relation application fits) (geometry application) raw.assignment
      raw.base raw.groupValue raw.products) :
    natWord ((input application fits raw).iteration + 1) =
      natWord (StateDecoder.iteration (outputState raw)) := by
  let word : Fin PilotProduction.stateHashWords :=
    ⟨RunningTransitionInputs.iterationWordIndex, by
      norm_num [RunningTransitionInputs.iterationWordIndex,
        PilotProduction.stateHashWords_eq]⟩
  have wired := semantics.nextPreimage.iteration
  have fieldEquality : outputState raw word.val = priorState raw word.val + 1 := by
    calc
      outputState raw word.val = transitionEnv raw
          (PilotProduction.outputPreimageStart + word.val) :=
        outputState_eq_transition raw word
      _ = transitionEnv raw
          (PilotProduction.priorPreimageStart + word.val) + 1 := by
        simpa [NextPreimageInputs.sourceInterface,
          NextPreimageInputs.outputIterationSource,
          NextPreimageInputs.priorIterationSource, transitionEnv, word] using
            wired
      _ = priorState raw word.val + 1 := by
        rw [priorState_eq_transition raw word]
  calc
    natWord ((input application fits raw).iteration + 1) =
        priorState raw word.val + 1 := by
      rw [show (input application fits raw).iteration =
        StateDecoder.iteration (priorState raw) by rfl]
      simpa [StateDecoder.iteration, word] using
        StateDecoder.natWord_val_add_one (priorState raw word.val)
    _ = outputState raw word.val := fieldEquality.symm
    _ = natWord (StateDecoder.iteration (outputState raw)) := by
      change outputState raw word.val = natWord (outputState raw word.val).val
      exact (StateDecoder.natWord_val _).symm

theorem semantics_imply_nextInitialState
    (application : Program) (fits : FitsTwoPow28 application)
    (raw : RawValues application)
    (semantics : DirectApplicationPrefixPlan.Semantics
      (relation application fits) (geometry application) raw.assignment
      raw.base raw.groupValue raw.products) :
    StateDecoder.initialState (outputState raw) =
      StateDecoder.initialState (priorState raw) := by
  unfold StateDecoder.initialState
  apply StateDecoder.slice_congr
  intro lane
  let stateLane : Lifecycle.Stage1.RunningTransition.StateIndex :=
    ⟨lane.val, by simpa [Lifecycle.Stage1.Application.stateWordCount,
      Lifecycle.Stage1.RunningTransition.stateWordCount] using lane.isLt⟩
  let word : Fin PilotProduction.stateHashWords :=
    ⟨RunningTransitionInputs.initialStateWordStart + stateLane.val, by
      have bound := stateLane.isLt
      norm_num [RunningTransitionInputs.initialStateWordStart,
        Lifecycle.Stage1.RunningTransition.stateWordCount,
        PilotProduction.stateHashWords_eq] at bound ⊢
      omega⟩
  have wired := semantics.nextPreimage.initialState stateLane
  calc
    outputState raw word.val = transitionEnv raw
        (PilotProduction.outputPreimageStart + word.val) :=
      outputState_eq_transition raw word
    _ = transitionEnv raw
        (PilotProduction.priorPreimageStart + word.val) := by
      simpa [NextPreimageInputs.sourceInterface,
        NextPreimageInputs.outputInitialStateSource,
        NextPreimageInputs.priorInitialStateSource, transitionEnv, word,
        Nat.add_assoc] using
          wired
    _ = priorState raw word.val := (priorState_eq_transition raw word).symm

theorem semantics_imply_nextSerialization
    (application : Program) (fits : FitsTwoPow28 application)
    (ajtai : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (raw : RawValues application)
    (semantics : DirectApplicationPrefixPlan.Semantics
      (relation application fits) (geometry application) raw.assignment
      raw.base raw.groupValue raw.products) :
    serializePreimage (publicFits := PerApplicationFixedPoint.publicFits application)
        (nextHashPreimage
          (setup (relation application fits) ajtai (contextKey raw))
          (input application fits raw) (output application raw)) =
      serializePreimage
        (publicFits := PerApplicationFixedPoint.publicFits application)
        (StateDecoder.preimage
          (PerApplicationFixedPoint.logicalWidth application)
          (PerApplicationFixedPoint.publicFits application)
          (outputState raw)) := by
  have context := semantics_imply_contextKeys application fits raw semantics
  have iteration := semantics_imply_nextIterationWord application fits raw semantics
  have initial := semantics_imply_nextInitialState application fits raw semantics
  unfold serializePreimage nextHashPreimage setup StateDecoder.preimage
  change stateDomainTag ++ block (contextKey raw) ++
      [natWord ((input application fits raw).iteration + 1)] ++
      block (input application fits raw).z0 ++
      block (output application raw).zNext ++
      serializeRunning
        ((output application raw).runningNext functionIndex) ++
      [natWord (oneBased (output application raw).pcNext)] =
    stateDomainTag ++ block (StateDecoder.keyDigest (outputState raw)) ++
      [natWord (StateDecoder.iteration (outputState raw))] ++
      block (StateDecoder.initialState (outputState raw)) ++
      block (StateDecoder.currentState (outputState raw)) ++
      serializeRunning (StateDecoder.running
        (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application) (outputState raw)) ++
      [natWord 1]
  rw [show (input application fits raw).z0 =
    StateDecoder.initialState (priorState raw) by rfl]
  rw [show (output application raw).zNext =
    StateDecoder.currentState (outputState raw) by rfl]
  rw [show (output application raw).runningNext functionIndex =
    StateDecoder.running
      (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application) (outputState raw) by rfl]
  rw [show oneBased (output application raw).pcNext = 1 by rfl]
  rw [← context, iteration, ← initial]

theorem priorHashPreimageRepresents
    (application : Program) (fits : FitsTwoPow28 application)
    (ajtai : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (raw : RawValues application)
    (canonical : StateDecoder.Canonical (priorState raw)) :
    Lifecycle.PriorStateHash.RepresentsPreimage
      PilotProduction.priorInterface PilotProduction.witnessOffset
      (pilotEnv raw)
      (priorHashPreimage
        (setup (relation application fits) ajtai (contextKey raw))
        (input application fits raw)) := by
  simpa [priorHashPreimage, setup, input, contextKey,
    StateDecoder.preimage] using priorRepresents raw canonical

theorem nextHashPreimageRepresents
    (application : Program) (fits : FitsTwoPow28 application)
    (ajtai : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (raw : RawValues application)
    (canonical : StateDecoder.Canonical (outputState raw))
    (semantics : DirectApplicationPrefixPlan.Semantics
      (relation application fits) (geometry application) raw.assignment
      raw.base raw.groupValue raw.products) :
    Lifecycle.OutputHash.RepresentsPreimage PilotProduction.outputInterface
      (Lifecycle.Pilot.outputOffset PilotProduction.interface
        PilotProduction.witnessOffset)
      (pilotEnv raw)
      (nextHashPreimage
        (setup (relation application fits) ajtai (contextKey raw))
        (input application fits raw) (output application raw)) := by
  exact (outputRepresents raw canonical).trans
    (semantics_imply_nextSerialization application fits ajtai raw semantics).symm

end NightstreamFPrime.Export.Stage1.PerApplicationDecodedIO
