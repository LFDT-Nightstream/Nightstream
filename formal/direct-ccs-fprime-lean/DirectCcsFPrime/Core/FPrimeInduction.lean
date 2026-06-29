/-!
F' induction authority boundary.

This module formalizes the proof-critical Construction-2 shape needed by the
direct CCS path: the terminal proof may check only the latest F' step, but that
step must consume sound prior induction authority. A self-consistent digest of
the prior image is not modeled as authority here.
-/

namespace DirectCcsFPrime

namespace FPrimeInduction

/--
Reachability of an F' public image after `steps` applications of the verifier
transition relation.

`Transition i current next` is the theorem-facing shape of one accepted F'
step at step index `i`.
-/
inductive Reachable
    {Image : Type}
    (Transition : Nat → Image → Image → Prop)
    (initial : Image) :
    Nat → Image → Prop where
  | base :
      Reachable Transition initial 0 initial
  | step
      {i : Nat}
      {current next : Image} :
      Reachable Transition initial i current →
      Transition i current next →
        Reachable Transition initial (i + 1) next

/--
Soundness requirement for a folded F' accumulator or other prior-authority
object.

The concrete accumulator is allowed to be opaque. What matters is that
accepting it as authority for `(steps, image)` implies actual reachability of
that image from the base under the F' transition relation.
-/
def PriorAuthoritySound
    {Image Authority : Type}
    (Transition : Nat → Image → Image → Prop)
    (initial : Image)
    (AuthorityAccepts : Nat → Authority → Image → Prop) : Prop :=
  ∀ steps authority image,
    AuthorityAccepts steps authority image →
      Reachable Transition initial steps image

/--
Soundness requirement for the latest F' step verifier.

The verifier can carry proof/advice data, but accepting it must imply exactly
the abstract F' transition from the prior image to the next image.
-/
def LatestStepSound
    {Image Authority Proof : Type}
    (Transition : Nat → Image → Image → Prop)
    (VerifyLatestStep :
      Nat → Authority → Image → Image → Proof → Prop) : Prop :=
  ∀ step authority priorImage nextImage proof,
    VerifyLatestStep step authority priorImage nextImage proof →
      Transition step priorImage nextImage

/--
Accepted terminal F' compression object.

The final proof checks one latest F' step and a prior-authority object. The
structure deliberately stores the prior authority as a predicate proof, not as
a digest field.
-/
structure TerminalCompressionAccepted
    {Image Authority Proof : Type}
    (AuthorityAccepts : Nat → Authority → Image → Prop)
    (VerifyLatestStep :
      Nat → Authority → Image → Image → Proof → Prop)
    (priorSteps : Nat)
    (priorAuthority : Authority)
    (priorImage nextImage : Image)
    (proof : Proof) : Prop where
  priorAccepted :
    AuthorityAccepts priorSteps priorAuthority priorImage
  latestAccepted :
    VerifyLatestStep
      priorSteps
      priorAuthority
      priorImage
      nextImage
      proof

/--
Terminal F' compression proves reachability of the final image when:

* the prior authority is sound, and
* the latest F' step verifier is sound.

This is the minimal Construction-2 induction theorem needed by the direct CCS
path. It does not claim that any particular accumulator or SNARK implements
`AuthorityAccepts`; that is the concrete instantiation obligation.
-/
theorem terminal_compression_reaches_final
    {Image Authority Proof : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image}
    {AuthorityAccepts : Nat → Authority → Image → Prop}
    {VerifyLatestStep :
      Nat → Authority → Image → Image → Proof → Prop}
    {priorSteps : Nat}
    {priorAuthority : Authority}
    {priorImage nextImage : Image}
    {proof : Proof}
    (hAuthority :
      PriorAuthoritySound
        Transition
        initial
        AuthorityAccepts)
    (hLatest :
      LatestStepSound
        Transition
        VerifyLatestStep)
    (hAccepted :
      TerminalCompressionAccepted
        AuthorityAccepts
        VerifyLatestStep
        priorSteps
        priorAuthority
        priorImage
        nextImage
        proof) :
    Reachable Transition initial (priorSteps + 1) nextImage :=
  Reachable.step
    (hAuthority priorSteps priorAuthority priorImage hAccepted.priorAccepted)
    (hLatest
      priorSteps
      priorAuthority
      priorImage
      nextImage
      proof
      hAccepted.latestAccepted)

/--
Base-case authority that accepts only the initial image at step zero.
-/
def BaseAuthorityAccepts
    {Image : Type}
    (initial : Image) :
    Nat → Unit → Image → Prop :=
  fun steps _ image => steps = 0 ∧ image = initial

/--
The base authority is sound for any transition relation because it only
authorizes the initial zero-step image.
-/
theorem base_authority_sound
    {Image : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image} :
    PriorAuthoritySound
      Transition
      initial
      (BaseAuthorityAccepts initial) := by
  intro steps authority image hAccept
  rcases hAccept with ⟨hSteps, hImage⟩
  subst hSteps
  subst hImage
  exact Reachable.base

/--
Digest-only acceptance is not an induction authority unless a separate
soundness theorem is supplied.

The theorem is phrased constructively: if a digest-only predicate accepts a
non-reachable image, then it cannot satisfy `PriorAuthoritySound`.
-/
theorem digest_only_acceptance_not_sound_when_it_accepts_unreachable
    {Image Authority : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image}
    {AuthorityAccepts : Nat → Authority → Image → Prop}
    {steps : Nat}
    {authority : Authority}
    {image : Image}
    (hAccept : AuthorityAccepts steps authority image)
    (hUnreachable : ¬ Reachable Transition initial steps image) :
    ¬ PriorAuthoritySound Transition initial AuthorityAccepts := by
  intro hSound
  exact hUnreachable (hSound steps authority image hAccept)

end FPrimeInduction

end DirectCcsFPrime
