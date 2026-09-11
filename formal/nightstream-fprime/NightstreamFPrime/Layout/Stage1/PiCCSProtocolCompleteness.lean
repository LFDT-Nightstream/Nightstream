import NightstreamFPrime.Layout.Stage1.PiCCSProofInputs
import NightstreamFPrime.Layout.Stage1.StateEncodingCanonical
import NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions
import NightstreamFPrime.Layout.Stage1.PiCCSInputSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.PhaseTransport

/-!
Owns PiCCS witness construction from the existing typed protocol input.
Canonical prior/output states and the verifier-owned context establish the
state-binding checks. Actual PiCCS acceptance then constructs the local
prefix and derives its output specification. No generated output is assumed.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Layout.Stage1.PiCCSProtocolCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PiCCSInputs (phaseOffset)
open PiCCSProofInputs (relationInterface relationProof)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}
  (prior : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
  (priorPublic : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
  (output : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
  (digest : Digest)
  (priorFixed : PilotProduction.FixedPreimage prior)
  (outputFixed : PilotProduction.FixedPreimage output)
  (digestFixed : digest.length = PilotProduction.digestWords)
  (values : PiCCSProofInputs.ProofValues)
  (context : VerifierContext.Digest4)

/-- Load existing protocol values and the verifier's expected context. -/
def environment : Env :=
  PiCCSProofInputs.loadExpectedContext
    (PiCCSProofInputs.protocolEnv prior priorPublic output digest
      priorFixed outputFixed digestFixed values) context

/-- The canonical protocol environment preserves the complete pilot input prefix. -/
theorem pilot_word (index : Nat) (bound : index < PilotProduction.externalColumnCount) :
    environment prior priorPublic output digest priorFixed outputFixed digestFixed values context index =
      PilotProduction.protocolEnv prior priorPublic output digest priorFixed outputFixed digestFixed index := by
  have contextBound : index < PiCCSInputs.expectedContextStart := by
    rw [PilotProduction.externalColumnCount_eq] at bound
    rw [PiCCSInputs.expectedContextStart_eq]
    omega
  have proofBound : index < PiCCSInputs.proofInputStart := by
    unfold PiCCSInputs.proofInputStart
    omega
  rw [environment, PiCCSProofInputs.loadExpectedContext_agreesOutside _ _ index (Or.inl contextBound)]
  exact PiCCSProofInputs.eval_pilotPrefix
    (PiCCSProofInputs.protocolValues prior priorPublic output digest priorFixed outputFixed digestFixed values)
    index proofBound

private theorem fixedList_word {count : Nat} (words : List F)
    (lengths : words.length = count) (index : Fin count) :
    PilotProduction.fixedList words lengths index = words.getD index.val 0 := by
  exact (List.getD_eq_get words 0 (Fin.cast lengths.symm index)).symm

private theorem pilot_prior_word (values : PilotProduction.ExternalValues)
    (index : Fin PilotProduction.stateHashWords) :
    PilotProduction.loadExternal values (PilotProduction.priorPreimageStart + index.val) =
      values.priorPreimage index := by
  have inside : index.val < PilotProduction.priorPublicInputStart := by
    simpa only [PilotProduction.priorPublicInputStart, PilotProduction.priorPreimageStart,
      Nat.zero_add] using index.isLt
  simp only [PilotProduction.loadExternal, PilotProduction.priorPreimageStart,
    Nat.zero_add, dif_pos inside]

private theorem pilot_output_word (values : PilotProduction.ExternalValues)
    (index : Fin PilotProduction.stateHashWords) :
    PilotProduction.loadExternal values (PilotProduction.outputPreimageStart + index.val) =
      values.outputPreimage index := by
  have afterPrior : ¬ PilotProduction.outputPreimageStart + index.val <
      PilotProduction.priorPublicInputStart := by
    unfold PilotProduction.outputPreimageStart
    omega
  have afterPublic : ¬ PilotProduction.outputPreimageStart + index.val <
      PilotProduction.outputPreimageStart := by omega
  have inside : PilotProduction.outputPreimageStart + index.val < PilotProduction.outputDigestStart := by
    unfold PilotProduction.outputDigestStart
    omega
  simp only [PilotProduction.loadExternal, dif_neg afterPrior, dif_neg afterPublic,
    dif_pos inside, Nat.add_sub_cancel_left]

private theorem prior_word (index : Fin PilotProduction.stateHashWords) :
    (PiCCSInputs.priorStateWord index.val).eval
      (environment prior priorPublic output digest priorFixed outputFixed digestFixed values context) =
        (serializePreimage (publicFits := publicFits) prior).getD index.val 0 := by
  change environment prior priorPublic output digest priorFixed outputFixed digestFixed values context
    (PilotProduction.priorPreimageStart + index.val) = _
  rw [pilot_word prior priorPublic output digest priorFixed outputFixed digestFixed values context]
  · rw [PilotProduction.protocolEnv, pilot_prior_word]
    exact fixedList_word _ (PilotProduction.serializePreimage_length_fixed prior priorFixed) index
  · have indexBound : index.val < 49393 := by
      simpa only [PilotProduction.stateHashWords_eq] using index.isLt
    rw [PilotProduction.externalColumnCount_eq]
    simp only [PilotProduction.priorPreimageStart]
    omega

private theorem output_word (index : Fin PilotProduction.stateHashWords) :
    (PiCCSInputs.outputStateWord index.val).eval
      (environment prior priorPublic output digest priorFixed outputFixed digestFixed values context) =
        (serializePreimage (publicFits := publicFits) output).getD index.val 0 := by
  change environment prior priorPublic output digest priorFixed outputFixed digestFixed values context
    (PilotProduction.outputPreimageStart + index.val) = _
  rw [pilot_word prior priorPublic output digest priorFixed outputFixed digestFixed values context]
  · rw [PilotProduction.protocolEnv, pilot_output_word]
    exact fixedList_word _ (PilotProduction.serializePreimage_length_fixed output outputFixed) index
  · have indexBound : index.val < 49393 := by
      simpa only [PilotProduction.stateHashWords_eq] using index.isLt
    rw [PilotProduction.externalColumnCount_eq]
    norm_num [PilotProduction.outputPreimageStart, PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq,
      PriorStateHash.publicWidth, ringDegree, publicRingColumns] <;> omega

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)

/-- The typed state serializer and loaded selected context establish all
state-binding checks. There is no caller-supplied state-binding conclusion. -/
theorem stateBinding
    (priorPc : prior.pc = 1) (outputPc : output.pc = 1)
    (priorContext : prior.verifierKeys functionIndex = context.toList)
    (outputContext : output.verifierKeys functionIndex = context.toList) :
    StateBinding.SpecHolds
      (Formal.statementBindingInterface (Formal.atOffset (relationInterface relation) phaseOffset)).state
      phaseOffset
      (environment prior priorPublic output digest priorFixed outputFixed digestFixed values context) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro word member
    change (PiCCSInputs.priorStateWord word.index).eval
      (environment prior priorPublic output digest priorFixed outputFixed digestFixed values context) = word.value
    let index : Fin PilotProduction.stateHashWords :=
      ⟨word.index, by simpa only [PilotProduction.stateHashWords_eq] using
        StateBinding.fixedWord_index_lt word member⟩
    exact (prior_word prior priorPublic output digest priorFixed outputFixed digestFixed values context index).trans
      (StateEncodingCanonical.serializePreimage_canonical prior priorFixed priorPc word member)
  · intro word member
    change (PiCCSInputs.outputStateWord word.index).eval
      (environment prior priorPublic output digest priorFixed outputFixed digestFixed values context) = word.value
    let index : Fin PilotProduction.stateHashWords :=
      ⟨word.index, by simpa only [PilotProduction.stateHashWords_eq] using
        StateBinding.fixedWord_index_lt word member⟩
    exact (output_word prior priorPublic output digest priorFixed outputFixed digestFixed values context index).trans
      (StateEncodingCanonical.serializePreimage_canonical output outputFixed outputPc word member)
  · intro lane
    change (PiCCSInputs.priorStateWord (24 + lane.val)).eval
        (environment prior priorPublic output digest priorFixed outputFixed digestFixed values context) =
      (PiCCSInputs.expectedContext lane).eval
        (environment prior priorPublic output digest priorFixed outputFixed digestFixed values context)
    let index : Fin PilotProduction.stateHashWords := ⟨24 + lane.val, by
      rw [PilotProduction.stateHashWords_eq]
      omega⟩
    calc
      _ = (serializePreimage (publicFits := publicFits) prior).getD (24 + lane.val) 0 :=
        prior_word prior priorPublic output digest priorFixed outputFixed digestFixed values context index
      _ = (prior.verifierKeys functionIndex).getD lane.val 0 :=
        StateEncodingCanonical.serializePreimage_context_word prior priorFixed lane
      _ = context.toList.getD lane.val 0 := by rw [priorContext]
      _ = _ := (PiCCSProofInputs.loadExpectedContext_read _ context lane).symm
  · intro lane
    change (PiCCSInputs.outputStateWord (24 + lane.val)).eval
        (environment prior priorPublic output digest priorFixed outputFixed digestFixed values context) =
      (PiCCSInputs.expectedContext lane).eval
        (environment prior priorPublic output digest priorFixed outputFixed digestFixed values context)
    let index : Fin PilotProduction.stateHashWords := ⟨24 + lane.val, by
      rw [PilotProduction.stateHashWords_eq]
      omega⟩
    calc
      _ = (serializePreimage (publicFits := publicFits) output).getD (24 + lane.val) 0 :=
        output_word prior priorPublic output digest priorFixed outputFixed digestFixed values context index
      _ = (output.verifierKeys functionIndex).getD lane.val 0 :=
        StateEncodingCanonical.serializePreimage_context_word output outputFixed lane
      _ = context.toList.getD lane.val 0 := by rw [outputContext]
      _ = _ := (PiCCSProofInputs.loadExpectedContext_read _ context lane).symm

/-- An environment with the same existing external source columns reads the
same typed running instance, fresh instance, and proof. Generated pilot cells
are not part of that source agreement. -/
theorem inputs_eq_of_external
    (template : Proof 9) (initial : Env)
    (source : ∀ index, PiCCSOrdinarySourceSupport.External index → initial index =
      environment prior priorPublic output digest priorFixed outputFixed digestFixed values context index) :
    Formal.evalRunning (relationInterface relation) phaseOffset initial = prior.running functionIndex ∧
    Formal.evalFresh (relationInterface relation) phaseOffset initial =
      PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values ∧
    Formal.evalProof relation (relationInterface relation) phaseOffset initial
      (relationProof relation values template) = relationProof relation values template := by
  have support : Formal.ExternalInputsSupported (relationInterface relation) phaseOffset
      PiCCSOrdinarySourceSupport.External :=
    PiCCSOrdinarySourceSupport.externalInputsSupported logicalWidth publicFits
  have original := PiCCSProofInputs.protocolInputs_eq relation prior priorPublic output digest
    priorFixed outputFixed digestFixed values template
  have loaded := PiCCSProofInputs.loadExpectedContext_inputs_eq relation
    (PiCCSProofInputs.protocolEnv prior priorPublic output digest priorFixed outputFixed digestFixed values)
    context (relationProof relation values template)
  have running := Formal.PhaseTransport.evalRunning_eq_of_agree_satisfy (relationInterface relation)
    phaseOffset PiCCSOrdinarySourceSupport.External initial _ support source
  have fresh := Formal.PhaseTransport.evalFresh_eq_of_agree_satisfy (relationInterface relation)
    phaseOffset PiCCSOrdinarySourceSupport.External initial _ support source
  have proof := Formal.PhaseTransport.evalProof_eq_of_agree_satisfy relation (relationInterface relation)
    phaseOffset PiCCSOrdinarySourceSupport.External initial _
    (relationProof relation values template) support source
  exact ⟨running.trans (loaded.1.trans original.1),
    fresh.trans (loaded.2.1.trans original.2.1),
    proof.trans (loaded.2.2.trans original.2.2)⟩

/-- Accepted protocol inputs construct the local PiCCS prefix in any environment
with the same external source values. Existing sibling witnesses are retained
outside this phase, and generated phase outputs are derived from its rows. -/
theorem completePrefix_from
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (template : Proof 9)
    (priorPc : prior.pc = 1) (outputPc : output.pc = 1)
    (priorContext : prior.verifierKeys functionIndex = context.toList)
    (outputContext : output.verifierKeys functionIndex = context.toList)
    (accepted : NightstreamFPrime.Spec.Folding.PiCCS.Accepted (ProductionKey.key relation ajtai)
      (prior.running functionIndex) (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values)
      (relationProof relation values template))
    (initial : Env)
    (source : ∀ index, PiCCSOrdinarySourceSupport.External index → initial index =
      environment prior priorPublic output digest priorFixed outputFixed digestFixed values context index) :
    ∃ completed : Sequence.Prefix
      initial phaseOffset,
      completed.operations = Formal.opsAt relation (relationInterface relation) phaseOffset ∧
        Formal.PhaseHolds relation ajtai (relationInterface relation) phaseOffset completed.current
          (relationProof relation values template) := by
  have external : NightstreamFPrime.Layout.PiCCS.v1_1.ProductionInputs.ExternalInputsLinear
      (relationInterface relation) phaseOffset :=
    PiCCSInputs.externalInputsLinear logicalWidth publicFits
  have assumptions := NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions.production
    relation (relationInterface relation) phaseOffset external initial
  have inputs := inputs_eq_of_external prior priorPublic output digest priorFixed outputFixed
    digestFixed values context relation template initial source
  have acceptedEnv : NightstreamFPrime.Spec.Folding.PiCCS.Accepted (ProductionKey.key relation ajtai)
      (Formal.evalRunning (relationInterface relation) phaseOffset initial)
      (Formal.evalFresh (relationInterface relation) phaseOffset initial)
      (Formal.evalProof relation (relationInterface relation) phaseOffset initial
        (relationProof relation values template)) := by
    rw [inputs.1, inputs.2.1, inputs.2.2]
    exact accepted
  have support : Formal.ExternalInputsSupported (relationInterface relation) phaseOffset
      PiCCSOrdinarySourceSupport.External :=
    PiCCSOrdinarySourceSupport.externalInputsSupported logicalWidth publicFits
  have binding := Formal.PhaseTransport.stateBinding_of_agree_satisfy (relationInterface relation) phaseOffset
    PiCCSOrdinarySourceSupport.External _ initial support
    (fun index supported => (source index supported).symm)
    (stateBinding prior priorPublic output digest priorFixed outputFixed digestFixed values context
      relation priorPc outputPc priorContext outputContext)
  exact Formal.completePrefix_of_accepted relation ajtai (relationInterface relation)
    (relationProof relation values template) initial phaseOffset assumptions binding acceptedEnv

/-- Actual accepted protocol inputs construct the complete local PiCCS
prefix. Canonical state and context checks are derived above, syntactic
bounds come from the existing interface, and generated phase outputs are
proved rather than supplied. -/
theorem completePrefix
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (template : Proof 9)
    (priorPc : prior.pc = 1) (outputPc : output.pc = 1)
    (priorContext : prior.verifierKeys functionIndex = context.toList)
    (outputContext : output.verifierKeys functionIndex = context.toList)
    (accepted : NightstreamFPrime.Spec.Folding.PiCCS.Accepted (ProductionKey.key relation ajtai)
      (prior.running functionIndex) (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values)
      (relationProof relation values template)) :
    ∃ completed : Sequence.Prefix
      (environment prior priorPublic output digest priorFixed outputFixed digestFixed values context) phaseOffset,
      completed.operations = Formal.opsAt relation (relationInterface relation) phaseOffset ∧
        Formal.PhaseHolds relation ajtai (relationInterface relation) phaseOffset completed.current
          (relationProof relation values template) := by
  exact completePrefix_from prior priorPublic output digest priorFixed outputFixed digestFixed values
    context relation ajtai template priorPc outputPc priorContext outputContext accepted
    (environment prior priorPublic output digest priorFixed outputFixed digestFixed values context)
    (fun _ _ => rfl)

end NightstreamFPrime.Layout.Stage1.PiCCSProtocolCompleteness
