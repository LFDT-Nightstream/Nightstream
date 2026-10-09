import NightstreamFPrime.Layout.Stage1.StateEncodingCanonical

/-!
Owns exact typed readback of the existing state-preimage word interval.
Canonical framing and serializer injectivity establish the decoder inverse;
no digest is used as authority and no new state representation is introduced.
-/

namespace NightstreamFPrime.Layout.Stage1.StateEncodingReadback

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem serialized_words
    (value : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fixed : PilotProduction.FixedPreimage value) :
    (List.ofFn fun index : Fin PilotProduction.stateHashWords =>
      (serializePreimage (publicFits := publicFits) value).getD index.val 0) =
      serializePreimage (publicFits := publicFits) value := by
  apply List.ext_get
  · rw [List.length_ofFn, PilotProduction.serializePreimage_length_fixed value fixed]
  · intro index leftBound rightBound
    rw [List.get_ofFn]
    exact List.getD_eq_getElem (l := serializePreimage (publicFits := publicFits) value)
      (d := 0) rightBound

/-- Exact words on the declared ABI interval decode to the same well-formed
state. WellFormed includes the natural counter bound, pc=1 and canonical
children, so this theorem does not identify distinct natural counters through
modular field encoding or distinct children through the packed parent. -/
theorem preimage_eq_of_words
    (value : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (wellFormed : StateEncoding.WellFormed value)
    (words : Nat → F)
    (matching : ∀ index : Fin PilotProduction.stateHashWords,
      words index.val = (serializePreimage (publicFits := publicFits) value).getD index.val 0) :
    StateDecoder.preimage logicalWidth publicFits words = value := by
  have encodedCanonical := StateEncodingCanonical.serializePreimage_canonical value
  have canonical : StateDecoder.Canonical words := by
    intro word member
    let index : Fin PilotProduction.stateHashWords := ⟨word.index, by
      rw [PilotProduction.stateHashWords_eq]
      exact PiCCS.v1_2.StateBinding.fixedWord_index_lt word member⟩
    exact (matching index).trans (encodedCanonical word member)
  have runningEq : StateDecoder.running logicalWidth publicFits words =
      value.running functionIndex := by
    apply StateDecoder.running_eq_of_serialized wellFormed.2.2.2
    apply List.ext_getElem
    · simp [serializeRunning_length]
    · intro index leftBound rightBound
      have indexBound : index < 27794 := by simpa using leftBound
      simp only [StateDecoder.slice, List.getElem_ofFn]
      rw [show PiCCSInputs.priorRunningStart + index = 12 + index from rfl,
        matching ⟨12 + index, by rw [PilotProduction.stateHashWords_eq]; omega⟩,
        StateEncodingCanonical.serializePreimage_running_word value index indexBound,
        List.getD_eq_getElem _ _ rightBound]
  have decodedWellFormed : StateEncoding.WellFormed
      (StateDecoder.preimage logicalWidth publicFits words) := by
    refine ⟨StateDecoder.preimage_fixed logicalWidth publicFits words,
      StateDecoder.iteration_lt words, rfl, ?_⟩
    change Lifecycle.ChildrenCanonical (StateDecoder.running logicalWidth publicFits words)
    rw [runningEq]
    exact wellFormed.2.2.2
  apply StateEncoding.serializePreimage_injective decodedWellFormed wellFormed
  calc
    serializePreimage (publicFits := publicFits)
        (StateDecoder.preimage logicalWidth publicFits words) =
      List.ofFn (fun index : Fin PilotProduction.stateHashWords => words index.val) :=
        StateDecoder.serializePreimage_preimage logicalWidth publicFits canonical
    _ = List.ofFn (fun index : Fin PilotProduction.stateHashWords =>
        (serializePreimage (publicFits := publicFits) value).getD index.val 0) :=
      congrArg List.ofFn (funext matching)
    _ = serializePreimage (publicFits := publicFits) value := serialized_words value wellFormed.1

end NightstreamFPrime.Layout.Stage1.StateEncodingReadback
