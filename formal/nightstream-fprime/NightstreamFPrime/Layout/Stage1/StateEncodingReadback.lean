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
state. WellFormed includes the natural counter bound and pc=1, so this theorem
does not identify distinct natural counters through modular field encoding. -/
theorem preimage_eq_of_words
    (value : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (wellFormed : StateEncoding.WellFormed value)
    (words : Nat → F)
    (matching : ∀ index : Fin PilotProduction.stateHashWords,
      words index.val = (serializePreimage (publicFits := publicFits) value).getD index.val 0) :
    StateDecoder.preimage logicalWidth publicFits words = value := by
  have encodedCanonical := StateEncodingCanonical.serializePreimage_canonical
    value wellFormed.1 wellFormed.2.2
  have canonical : StateDecoder.Canonical words := by
    intro word member
    let index : Fin PilotProduction.stateHashWords := ⟨word.index, by
      rw [PilotProduction.stateHashWords_eq]
      exact PiCCS.v1_1.StateBinding.fixedWord_index_lt word member⟩
    exact (matching index).trans (encodedCanonical word member)
  apply StateEncoding.serializePreimage_injective
    (StateDecoder.preimage_wellFormed logicalWidth publicFits words) wellFormed
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
