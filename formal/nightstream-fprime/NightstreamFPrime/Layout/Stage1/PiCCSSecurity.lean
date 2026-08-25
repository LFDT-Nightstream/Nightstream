import NightstreamFPrime.Layout.Stage1.StateEncoding
import NightstreamFPrime.Lifecycle.VerifierContext
import NightstreamFPrime.Spec.Folding.PiCCS.TranscriptReplay

/-!
Owns the committed-statement reduction for digest-only PiCCS.

The deterministic result stops at explicit collision events. It assigns no
probability to Poseidon2 and does not mix commitment binding, Fiat--Shamir
security, sampling security, or the SuperNeo knowledge reduction into row
soundness. Those assumptions belong to the complete Stage 1 security
composition.
-/

namespace NightstreamFPrime.Layout.Stage1.PiCCSSecurity

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Two different canonical verifier-context encodings produce one context
digest. This is the exact Poseidon2 context-hash collision event. -/
def ContextDigestCollision
    (left right : VerifierContext.Descriptor) : Prop :=
  left.serialize ≠ right.serialize ∧ left.digest4 = right.digest4

/-- Two different canonical Stage 1 statement encodings produce one state
digest. This is the exact Poseidon2 state-hash collision event. -/
def StateHashCollision
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits)) : Prop :=
  serializePreimage (publicFits := publicFits) left ≠
      serializePreimage (publicFits := publicFits) right ∧
    stateHash (publicFits := publicFits) left =
      stateHash (publicFits := publicFits) right

/-- Equal context digests identify one canonical descriptor unless the
context hash collides. -/
theorem contextDigest_identifies_descriptor_or_collision
    (left right : VerifierContext.Descriptor)
    (digestEqual : left.digest4 = right.digest4) :
    left = right ∨ ContextDigestCollision left right := by
  by_cases same : left = right
  · exact Or.inl same
  · apply Or.inr
    refine ⟨?_, digestEqual⟩
    intro encodedEqual
    exact same (VerifierContext.Descriptor.serialize_injective encodedEqual)

/-- Equal state digests identify one complete well-formed statement unless
the state hash collides. The encoding-injectivity theorem covers the context,
iteration, application state, program counter, point, commitments, public
inputs, separate `Eval_K`, and every separate `Eval_A` value. -/
theorem stateHash_identifies_statement_or_collision
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (leftWellFormed : StateEncoding.WellFormed left)
    (rightWellFormed : StateEncoding.WellFormed right)
    (digestEqual : stateHash (publicFits := publicFits) left =
      stateHash (publicFits := publicFits) right) :
    left = right ∨ StateHashCollision left right := by
  by_cases same : left = right
  · exact Or.inl same
  · apply Or.inr
    refine ⟨?_, digestEqual⟩
    intro encodedEqual
    exact same (StateEncoding.serializePreimage_injective
      leftWellFormed rightWellFormed encodedEqual)

/-- A valid state encoding cannot be a nonempty trailing extension of another
valid state encoding. This rules out the ambiguity branch before any
cryptographic assumption is used. -/
theorem no_valid_trailing_extension
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {left right : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (leftWellFormed : StateEncoding.WellFormed left)
    (rightWellFormed : StateEncoding.WellFormed right)
    {suffix : List F} (suffixNonempty : suffix ≠ []) :
    serializePreimage (publicFits := publicFits) left ≠
      serializePreimage (publicFits := publicFits) right ++ suffix :=
  StateEncoding.serializePreimage_not_trailing_extension
    leftWellFormed rightWellFormed suffixNonempty

end NightstreamFPrime.Layout.Stage1.PiCCSSecurity
