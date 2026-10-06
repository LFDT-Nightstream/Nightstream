import NightstreamFPrime.Layout.Stage1.PiCCSProofInputs

/-!
Paper authority: SuperNeo v1_1, section 7.3, PiCCS running input.
Obligation: Show that the honest protocol environment presents the exact prior
running instance to PiCCS.

The prior block supplies the running fields in place. The child region holds
the canonical child digits and the honest sign bits, so every prior child-split
row holds, and the state decoder returns the prior running instance. This
module owns value identities only; it adds no row or column.
-/

namespace NightstreamFPrime.Layout.Stage1.PiCCSPriorRunning

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Layout.Stage1.PiCCSInputs
open NightstreamFPrime.Layout.Stage1.PiCCSProofInputs

section Protocol

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}
  (prior : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
  (priorPublic : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
  (outputPreimage : HashPreimage
    (logicalWidth := logicalWidth) (publicFits := publicFits))
  (digest : Digest)
  (priorFixed : PilotProduction.FixedPreimage prior)
  (outputFixed : PilotProduction.FixedPreimage outputPreimage)
  (digestFixed : digest.length = PilotProduction.digestWords)
  (proofValues : ProofValues)

/-- Each loaded prior-state word is the canonical serialized word. -/
theorem protocolEnv_priorWord (index : Nat)
    (bound : index < PilotProduction.stateHashWords) :
    protocolEnv prior priorPublic outputPreimage digest priorFixed outputFixed
        digestFixed proofValues (PilotProduction.priorPreimageStart + index) =
      (serializePreimage (publicFits := publicFits) prior).getD index 0 := by
  have stateBound : index < 27819 := by
    simpa only [PilotProduction.stateHashWords_eq] using bound
  unfold protocolEnv
  rw [eval_pilotPrefix _ _ (by
    norm_num [PilotProduction.priorPreimageStart, priorChildrenStart,
      expectedContextStart, expectedContextWords]
    omega)]
  have inside : PilotProduction.priorPreimageStart + index <
      PilotProduction.priorPublicInputStart := by
    simpa only [PilotProduction.priorPublicInputStart] using Nat.add_lt_add_left bound _
  change PilotProduction.loadExternal (PilotProduction.protocolValues prior priorPublic
      outputPreimage digest priorFixed outputFixed digestFixed)
      (PilotProduction.priorPreimageStart + index) = _
  unfold PilotProduction.loadExternal
  rw [dif_pos inside]
  unfold PilotProduction.protocolValues
  calc
    _ = (serializePreimage (publicFits := publicFits) prior).getD
        (PilotProduction.priorPreimageStart + index) 0 := by
      unfold PilotProduction.fixedList
      exact (List.getD_eq_get _ 0 _).symm
    _ = _ := by rw [show PilotProduction.priorPreimageStart = 0 from rfl, Nat.zero_add]

/-- Each loaded region digit is the prior child public input. -/
theorem protocolEnv_priorDigit (source : Fin productionShape.runningCount)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    protocolEnv prior priorPublic outputPreimage digest priorFixed outputFixed
        digestFixed proofValues (runningPublicStart source.val + column.val) =
      (prior.running functionIndex).publicInputs source column := by
  have sourceBound : source.val < 16 := source.isLt
  have columnBound : column.val < 270 := column.isLt
  unfold protocolEnv
  rw [show runningPublicStart source.val + column.val =
      priorChildrenStart + (source.val * 270 + column.val) by
    unfold runningPublicStart runningPublicWords
    omega]
  rw [eval_childWord _ _ (by unfold priorChildrenWords; omega)]
  change (serializeChildPublicInputs (publicFits := publicFits)
      (prior.running functionIndex) ++ _).getD _ 0 = _
  rw [List.getD_append _ _ _ _ (by
    rw [serializeChildPublicInputs_length]
    omega)]
  exact serializeChildPublicInputs_getD _ source column

/-- Each loaded region sign is the honest sign bit of the prior parent. -/
theorem protocolEnv_priorSign
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    protocolEnv prior priorPublic outputPreimage digest priorFixed outputFixed
        digestFixed proofValues (priorSignStart + column.val) =
      signWord (parentPublic (prior.running functionIndex) column) := by
  have columnBound : column.val < 270 := column.isLt
  unfold protocolEnv
  rw [show priorSignStart + column.val = priorChildrenStart + (4320 + column.val) by
    unfold priorSignStart
    omega]
  rw [eval_childWord _ _ (by unfold priorChildrenWords; omega)]
  change (serializeChildPublicInputs (publicFits := publicFits)
      (prior.running functionIndex) ++ _).getD _ 0 = _
  rw [List.getD_append_right _ _ _ _ (by rw [serializeChildPublicInputs_length]; omega),
    serializeChildPublicInputs_length, Nat.add_sub_cancel_left]
  exact finRange_map_getD _ column

/-- The honest region satisfies every prior child-split row. -/
theorem childrenSplit_protocolEnv
    (canonical : Lifecycle.ChildrenCanonical (prior.running functionIndex)) :
    StateBinding.ChildrenSplit
      (Formal.statementBindingInterface
        (Formal.atOffset (PiCCSInputs.interface logicalWidth publicFits) phaseOffset)).state
      phaseOffset
      (protocolEnv prior priorPublic outputPreimage digest priorFixed outputFixed
        digestFixed proofValues) := by
  let running := prior.running functionIndex
  have signValue (word : Fin packedParentWords) (lane : Fin 3) :
      StateBinding.priorSignValue
          (Formal.statementBindingInterface
            (Formal.atOffset (PiCCSInputs.interface logicalWidth publicFits) phaseOffset)).state
          phaseOffset
          (protocolEnv prior priorPublic outputPreimage digest priorFixed outputFixed
            digestFixed proofValues) word lane =
        signWord (parentPublic running (packedColumn word lane)) :=
    protocolEnv_priorSign prior priorPublic outputPreimage digest priorFixed outputFixed
      digestFixed proofValues (packedColumn word lane)
  have digitsValue (word : Fin packedParentWords) (lane : Fin 3) :
      StateBinding.priorDigits
          (Formal.statementBindingInterface
            (Formal.atOffset (PiCCSInputs.interface logicalWidth publicFits) phaseOffset)).state
          phaseOffset
          (protocolEnv prior priorPublic outputPreimage digest priorFixed outputFixed
            digestFixed proofValues) word lane =
        Lifecycle.childDigits running (packedColumn word lane) :=
    funext fun child => protocolEnv_priorDigit prior priorPublic outputPreimage digest
      priorFixed outputFixed digestFixed proofValues _ (packedColumn word lane)
  refine ⟨?_, ?_, ?_⟩
  · intro word lane
    rw [signValue]
    unfold signWord
    split_ifs
    · exact Or.inl rfl
    · exact Or.inr rfl
  · intro word lane child
    rw [digitsValue, signValue, canonical.childDigits_eq]
    have bounded : centeredMagnitude (parentPublic running (packedColumn word lane)) <
        Radix.combinedBound := by
      rw [Radix.production_parameters.2.2]
      simpa using canonical.parentBounded (packedColumn word lane)
    have honest := (Radix.UniformSignedDigits.honest_complete _ bounded).constraint.2 child
    have signEq : Radix.UniformSignedDigits.honestSign
        (parentPublic running (packedColumn word lane)) =
          1 - 2 * signWord (parentPublic running (packedColumn word lane)) := by
      unfold Radix.UniformSignedDigits.honestSign signWord
      split_ifs <;> decide
    rw [← signEq]
    exact honest
  · intro word
    rw [digitsValue, digitsValue, digitsValue]
    change protocolEnv prior priorPublic outputPreimage digest priorFixed outputFixed
        digestFixed proofValues
        (PilotProduction.priorPreimageStart + (StateBinding.packedWordStart + word.val)) =
      packWord (parentPublic running (packedColumn word 0))
        (parentPublic running (packedColumn word 1))
        (parentPublic running (packedColumn word 2))
    have wordBound : word.val < 90 := word.isLt
    rw [protocolEnv_priorWord prior priorPublic outputPreimage digest priorFixed
        outputFixed digestFixed proofValues _ (by
          rw [PilotProduction.stateHashWords_eq]
          unfold StateBinding.packedWordStart
          omega),
      show StateBinding.packedWordStart + word.val = 12 + (27704 + word.val) by
        unfold StateBinding.packedWordStart
        omega,
      StateEncodingCanonical.serializePreimage_running_word prior (27704 + word.val) (by omega)]
    unfold serializeRunning
    rw [List.getD_append_right _ _ _ _ (by rw [serializeRunningFields_length]; omega),
      serializeRunningFields_length, Nat.add_sub_cancel_left]
    exact finRange_map_getD _ word

/-- The combined protocol environment presents the exact authoritative prior
running instance to PiCCS. -/
theorem evalRunning_protocolEnv_eq_priorRunning
    (canonical : Lifecycle.ChildrenCanonical (prior.running functionIndex)) :
    StatementAbsorption.evalRunning (runningExpr logicalWidth publicFits)
        (protocolEnv prior priorPublic outputPreimage digest
          priorFixed outputFixed digestFixed proofValues) =
      prior.running functionIndex := by
  rw [StateDecoder.evalRunning_eq_running _ (childrenSplit_protocolEnv prior priorPublic
    outputPreimage digest priorFixed outputFixed digestFixed proofValues canonical)]
  apply StateDecoder.running_eq_of_serialized canonical
  apply List.ext_getElem
  · simp [serializeRunning_length]
  · intro index leftBound rightBound
    have indexBound : index < 27794 := by simpa using leftBound
    simp only [StateDecoder.slice, List.getElem_ofFn]
    rw [show PiCCSInputs.priorRunningStart + index = 12 + index from rfl,
      protocolEnv_priorWord prior priorPublic outputPreimage digest priorFixed
        outputFixed digestFixed proofValues _ (by
          rw [PilotProduction.stateHashWords_eq]
          omega),
      StateEncodingCanonical.serializePreimage_running_word prior index indexBound,
      List.getD_eq_getElem _ _ rightBound]

end Protocol

/-- Relation-typed parent view of the authoritative running instance. -/
theorem formalEvalRunning_protocolEnv_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (prior : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (priorPublic : PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (outputPreimage : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (digest : Digest)
    (priorFixed : PilotProduction.FixedPreimage prior)
    (outputFixed : PilotProduction.FixedPreimage outputPreimage)
    (digestFixed : digest.length = PilotProduction.digestWords)
    (proofValues : ProofValues)
    (canonical : Lifecycle.ChildrenCanonical (prior.running functionIndex)) :
    Formal.evalRunning (relationInterface relation) phaseOffset
        (protocolEnv prior priorPublic outputPreimage digest
          priorFixed outputFixed digestFixed proofValues) =
      prior.running functionIndex := by
  rw [evalRunning_relationInterface]
  exact evalRunning_protocolEnv_eq_priorRunning prior priorPublic outputPreimage
    digest priorFixed outputFixed digestFixed proofValues canonical

/-- Complete semantic coverage of every caller-owned value read by the
production PiCCS parent. -/
theorem protocolInputs_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (prior : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (priorPublic : PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (outputPreimage : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (digest : Digest)
    (priorFixed : PilotProduction.FixedPreimage prior)
    (outputFixed : PilotProduction.FixedPreimage outputPreimage)
    (digestFixed : digest.length = PilotProduction.digestWords)
    (proofValues : ProofValues)
    (template : Proof 8)
    (canonical : Lifecycle.ChildrenCanonical (prior.running functionIndex)) :
    Formal.evalRunning (relationInterface relation) phaseOffset
        (protocolEnv prior priorPublic outputPreimage digest
          priorFixed outputFixed digestFixed proofValues) =
        prior.running functionIndex ∧
      Formal.evalFresh (relationInterface relation) phaseOffset
          (protocolEnv prior priorPublic outputPreimage digest
            priorFixed outputFixed digestFixed proofValues) =
        protocolFresh logicalWidth publicFits priorPublic proofValues ∧
      Formal.evalProof relation (relationInterface relation) phaseOffset
          (protocolEnv prior priorPublic outputPreimage digest
            priorFixed outputFixed digestFixed proofValues)
          (relationProof relation proofValues template) =
        relationProof relation proofValues template := by
  exact ⟨formalEvalRunning_protocolEnv_eq relation prior priorPublic
      outputPreimage digest priorFixed outputFixed digestFixed proofValues canonical,
    formalEvalFresh_protocolEnv_eq relation prior priorPublic outputPreimage
      digest priorFixed outputFixed digestFixed proofValues,
    formalEvalProof_protocolEnv_eq relation prior priorPublic outputPreimage
      digest priorFixed outputFixed digestFixed proofValues template⟩

end NightstreamFPrime.Layout.Stage1.PiCCSPriorRunning
