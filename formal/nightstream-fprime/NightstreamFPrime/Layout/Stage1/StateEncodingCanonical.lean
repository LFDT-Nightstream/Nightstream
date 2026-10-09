import NightstreamFPrime.Layout.Stage1.StateDecoder

/-!
Owns the fixed-word checks for an honestly serialized Stage 1 state. The
serializer starts with the constant domain chunk that PiCCS StateBinding pins,
and places the context words at the start of the tail. This adds no rows or
alternate state representation.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Layout.Stage1.StateEncodingCanonical

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The canonical serializer satisfies every fixed-word check used by the
PiCCS state boundary. -/
theorem serializePreimage_canonical
    (preimage : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    StateDecoder.Canonical
      (fun index => (serializePreimage (publicFits := publicFits) preimage).getD index 0) := by
  intro word member
  rw [StateBinding.fixedWords, List.mem_map] at member
  rcases member with ⟨index, _member, rfl⟩
  simp only [serializePreimage, List.append_assoc]
  exact List.getD_append _ _ _ _ index.isLt

/-- Running word `index` occurs after the 12-word domain chunk. -/
theorem serializePreimage_running_word
    (preimage : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (index : Nat) (bound : index < 27794) :
    (serializePreimage (publicFits := publicFits) preimage).getD (12 + index) 0 =
      (serializeRunning (publicFits := publicFits) (preimage.running functionIndex)).getD
        index 0 := by
  unfold serializePreimage
  rw [List.getD_append _ _ _ _ (by
      simp only [List.length_append, stateDomainChunk_length, serializeRunning_length]
      omega),
    List.getD_append_right _ _ _ _ (by rw [stateDomainChunk_length]; omega),
    stateDomainChunk_length, Nat.add_sub_cancel_left]

/-- Context payload words occur at the start of the tail. -/
theorem serializePreimage_context_word
    (preimage : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fixed : PilotProduction.FixedPreimage preimage) (lane : Fin 4) :
    (serializePreimage (publicFits := publicFits) preimage).getD
        (StateBinding.contextWordStart + lane.val) 0 =
      (preimage.verifierKeys functionIndex).getD lane.val 0 := by
  have keyLength := fixed.1
  change (preimage.verifierKeys functionIndex).length = 4 at keyLength
  unfold serializePreimage serializeTail
  rw [List.getD_append_right _ _ _ _ (by
    simp only [List.length_append, stateDomainChunk_length, serializeRunning_length,
      StateBinding.contextWordStart]
    omega)]
  simp only [List.length_append, stateDomainChunk_length, serializeRunning_length,
    List.append_assoc]
  rw [show StateBinding.contextWordStart + lane.val - (12 + 27794) = lane.val by
    simp [StateBinding.contextWordStart]]
  exact List.getD_append _ _ _ _ (by rw [keyLength]; exact lane.isLt)

end NightstreamFPrime.Layout.Stage1.StateEncodingCanonical
