import DirectCcsFPrime.CompressedFPrimeAuthority
import Mathlib.Tactic

/-!
Trace-carrying folded F' authority.

This module owns the direct induction bridge for a verifier that opens an
accepted prior proof to an explicit sequence of F' public images. A trace is
valid only when every adjacent pair satisfies the indexed F' transition
relation. Digest replay may identify the trace, but it is not authority by
itself.
-/

namespace DirectCcsFPrime

namespace FPrimeTraceAuthority

/--
Concrete trace authority for a claimed folded F' prior image.

`imageAt i` names the public image after `i` F' transitions. The authority is
valid only when it starts at `initial`, ends at the claimed public image, and
every adjacent pair satisfies the indexed transition relation.
-/
structure Authority
    {Image : Type}
    (Transition : Nat → Image → Image → Prop)
    (initial : Image)
    (steps : Nat)
    (image : Image) where
  imageAt : Nat → Image
  startsAtInitial : imageAt 0 = initial
  endsAtClaimed : imageAt steps = image
  stepValid :
    ∀ i, i < steps →
      Transition i (imageAt i) (imageAt (i + 1))

/-- A valid trace authority gives ordinary F' reachability. -/
theorem reachable
    {Image : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image}
    {steps : Nat}
    {image : Image}
    (trace : Authority Transition initial steps image) :
    FPrimeInduction.Reachable Transition initial steps image := by
  induction steps generalizing image with
  | zero =>
      have hImage : image = initial := by
        exact trace.endsAtClaimed.symm.trans trace.startsAtInitial
      subst image
      exact FPrimeInduction.Reachable.base
  | succ steps ih =>
      let priorImage := trace.imageAt steps
      have hPrior :
          FPrimeInduction.Reachable
            Transition
            initial
            steps
            priorImage := by
        apply ih
        exact
          { imageAt := trace.imageAt
            startsAtInitial := trace.startsAtInitial
            endsAtClaimed := rfl
            stepValid := by
              intro i hi
              exact trace.stepValid i (Nat.lt_trans hi (Nat.lt_succ_self steps)) }
      have hStep :
          Transition
            steps
            priorImage
            (trace.imageAt (steps + 1)) :=
        trace.stepValid steps (Nat.lt_succ_self steps)
      have hEnd : trace.imageAt (steps + 1) = image :=
        trace.endsAtClaimed
      rw [← hEnd]
      exact FPrimeInduction.Reachable.step hPrior hStep

/-- Turn trace authority into the existing proof-carrying folded authority. -/
def toFoldedAuthority
    {Image : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image}
    {steps : Nat}
    {image : Image}
    (trace : Authority Transition initial steps image) :
    FoldedFPrimeAuthority.Authority Transition initial :=
  { steps := steps
    image := image
    reachable := reachable trace }

/-- The folded authority obtained from a trace accepts the same public pair. -/
theorem toFoldedAuthority_accepts
    {Image : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image}
    {steps : Nat}
    {image : Image}
    (trace : Authority Transition initial steps image) :
    FoldedFPrimeAuthority.Accepts
      (Transition := Transition)
      (initial := initial)
      steps
      (toFoldedAuthority trace)
      image :=
  ⟨rfl, rfl⟩

/--
Soundness shape for a compressed verifier that opens accepted proofs to traces.
-/
def VerifierSound
    {Image Proof : Type}
    (Transition : Nat → Image → Image → Prop)
    (initial : Image)
    (VerifyPrior : Nat → Proof → Image → Prop) : Prop :=
  ∀ steps proof image,
    VerifyPrior steps proof image →
      Nonempty (Authority Transition initial steps image)

/--
Trace-opening verifier soundness is enough to satisfy the prior-authority
soundness interface used by terminal F' compression.
-/
theorem verifierSound_priorAuthoritySound
    {Image Proof : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image}
    {VerifyPrior : Nat → Proof → Image → Prop}
    (hSound : VerifierSound Transition initial VerifyPrior) :
    FPrimeInduction.PriorAuthoritySound
      Transition
      initial
      VerifyPrior := by
  intro steps proof image hVerify
  rcases hSound steps proof image hVerify with ⟨trace⟩
  exact reachable trace

/--
Verifier object whose accepted proofs open to explicit folded F' traces.
-/
structure SoundVerifier
    {Image Proof : Type}
    (Transition : Nat → Image → Image → Prop)
    (initial : Image) where
  verify : Nat → Proof → Image → Prop
  opensToTrace :
    ∀ steps proof image,
      verify steps proof image →
        Nonempty (Authority Transition initial steps image)

namespace SoundVerifier

/-- A trace-opening verifier satisfies the terminal prior-authority contract. -/
theorem priorAuthoritySound
    {Image Proof : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image}
    (verifier :
      SoundVerifier
        (Image := Image)
        (Proof := Proof)
        Transition
        initial) :
    FPrimeInduction.PriorAuthoritySound
      Transition
      initial
      verifier.verify :=
  verifierSound_priorAuthoritySound verifier.opensToTrace

/--
Every trace-opening verifier is also a `CompressedFPrimeAuthority.SoundVerifier`.
-/
def toCompressedSoundVerifier
    {Image Proof : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image}
    (verifier :
      SoundVerifier
        (Image := Image)
        (Proof := Proof)
        Transition
        initial) :
    CompressedFPrimeAuthority.SoundVerifier
      (Image := Image)
      (Proof := Proof)
      Transition
      initial where
  verify := verifier.verify
  opensToFoldedAuthority := by
    intro steps proof image hVerify
    rcases verifier.opensToTrace steps proof image hVerify with
      ⟨trace⟩
    exact
      ⟨toFoldedAuthority trace,
        toFoldedAuthority_accepts trace⟩

end SoundVerifier

end FPrimeTraceAuthority

end DirectCcsFPrime
