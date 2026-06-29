import DirectCcsFPrime.Core.Construction2DirectFPrime

/-!
Proof-carrying folded F' prior authority.

The direct terminal proof is allowed to check only the latest F' step if the
prior object it consumes is already sound induction authority. This module gives
the minimal concrete authority shape: an accepted prior authority carries a
proof that the prior image is reachable from the base image under the F'
transition relation.

This is intentionally not a digest chain. A digest may name or commit to such
an authority in an implementation, but acceptance here is sound only because it
contains reachability evidence.
-/

namespace DirectCcsFPrime

namespace FoldedFPrimeAuthority

/--
Proof-carrying folded F' authority for a concrete transition relation.

`reachable` is the theorem-facing object that a concrete folded accumulator
SNARK/NIFS proof must ultimately supply or imply.
-/
structure Authority
    {Image : Type}
    (Transition : Nat → Image → Image → Prop)
    (initial : Image) where
  steps : Nat
  image : Image
  reachable : FPrimeInduction.Reachable Transition initial steps image

/--
Accepted folded authority.

The terminal verifier asks for authority over a specific `(steps, image)`.
Acceptance is only the field match; soundness comes from the authority's stored
reachability proof.
-/
def Accepts
    {Image : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image}
    (steps : Nat)
    (authority : Authority Transition initial)
    (image : Image) : Prop :=
  authority.steps = steps ∧ authority.image = image

/-- Proof-carrying folded authority is sound prior authority. -/
theorem accepts_sound
    {Image : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image} :
    FPrimeInduction.PriorAuthoritySound
      Transition
      initial
      (Accepts
        (Transition := Transition)
        (initial := initial)) := by
  intro steps authority image hAccept
  rcases hAccept with ⟨hSteps, hImage⟩
  cases authority with
  | mk authoritySteps authorityImage reachable =>
      dsimp at hSteps hImage
      subst hSteps
      subst hImage
      exact reachable

/-- Base folded authority for the zero-step initial image. -/
def base
    {Image : Type}
    {Transition : Nat → Image → Image → Prop}
    (initial : Image) :
    Authority Transition initial :=
  { steps := 0
    image := initial
    reachable := FPrimeInduction.Reachable.base }

/-- The base authority authorizes exactly the zero-step initial image. -/
theorem base_accepts
    {Image : Type}
    {Transition : Nat → Image → Image → Prop}
    (initial : Image) :
    Accepts
      (Transition := Transition)
      (initial := initial)
      0
      (base (Transition := Transition) initial)
      initial := by
  exact ⟨rfl, rfl⟩

/--
Extend folded authority by one accepted F' transition.

This is the theorem-facing shape of append: a concrete implementation may
compress or commit to the authority, but the authority it represents advances
only by applying one accepted transition to a previously reachable image.
-/
def extend
    {Image : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial next : Image}
    (authority : Authority Transition initial)
    (hTransition : Transition authority.steps authority.image next) :
    Authority Transition initial :=
  { steps := authority.steps + 1
    image := next
    reachable :=
      FPrimeInduction.Reachable.step
        authority.reachable
        hTransition }

/--
An extended authority authorizes exactly the one-step successor image produced
by the accepted transition.
-/
theorem extend_accepts
    {Image : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial next : Image}
    (authority : Authority Transition initial)
    (hTransition : Transition authority.steps authority.image next) :
    Accepts
      (Transition := Transition)
      (initial := initial)
      (authority.steps + 1)
      (extend authority hTransition)
      next := by
  exact ⟨rfl, rfl⟩

/--
Terminal compression with proof-carrying folded prior authority proves final
reachability.

This specializes `FPrimeInduction.terminal_compression_reaches_final` by
discharging the prior-authority soundness premise with `accepts_sound`.
-/
theorem terminal_compression_reaches_final
    {Image Proof : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image}
    {VerifyLatestStep :
      Nat → Authority Transition initial → Image → Image → Proof → Prop}
    {priorSteps : Nat}
    {priorAuthority : Authority Transition initial}
    {priorImage nextImage : Image}
    {proof : Proof}
    (hLatest :
      FPrimeInduction.LatestStepSound
        Transition
        VerifyLatestStep)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (Accepts
          (Transition := Transition)
          (initial := initial))
        VerifyLatestStep
        priorSteps
        priorAuthority
        priorImage
        nextImage
        proof) :
    FPrimeInduction.Reachable
      Transition
      initial
      (priorSteps + 1)
      nextImage :=
  FPrimeInduction.terminal_compression_reaches_final
    accepts_sound
    hLatest
    hAccepted

/--
Direct Construction-2 specialization with proof-carrying folded prior
authority.
-/
theorem construction2_terminal_reaches_final
    {Digest Boundary AccHandle : Type}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {AccumulatorStep : Nat → AccHandle → AccHandle → Prop}
    {initial :
      Construction2DirectFPrime.PublicImage Digest Boundary AccHandle}
    {priorSteps : Nat}
    {priorAuthority :
      Authority
        (Construction2DirectFPrime.Transition
          BoundaryStep
          AccumulatorStep)
        initial}
    {priorImage nextImage :
      Construction2DirectFPrime.PublicImage Digest Boundary AccHandle}
    {proof : Unit}
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (Accepts
          (Transition :=
            Construction2DirectFPrime.Transition
              BoundaryStep
              AccumulatorStep)
          (initial := initial))
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority :=
            Authority
              (Construction2DirectFPrime.Transition
                BoundaryStep
                AccumulatorStep)
              initial)
          BoundaryStep
          AccumulatorStep)
        priorSteps
        priorAuthority
        priorImage
        nextImage
        proof) :
    FPrimeInduction.Reachable
      (Construction2DirectFPrime.Transition
        BoundaryStep
        AccumulatorStep)
      initial
      (priorSteps + 1)
      nextImage :=
  terminal_compression_reaches_final
    Construction2DirectFPrime.latest_step_sound
    hAccepted

/--
Digest-only acceptance cannot be substituted for proof-carrying authority.

If an authority predicate accepts an unreachable prior image, it is not a valid
prior-authority predicate for terminal compression.
-/
theorem accepted_unreachable_is_not_sound_authority
    {Image Authority : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image}
    {AuthorityAccepts : Nat → Authority → Image → Prop}
    {steps : Nat}
    {authority : Authority}
    {image : Image}
    (hAccept : AuthorityAccepts steps authority image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable Transition initial steps image) :
    ¬ FPrimeInduction.PriorAuthoritySound
      Transition
      initial
      AuthorityAccepts :=
  FPrimeInduction.digest_only_acceptance_not_sound_when_it_accepts_unreachable
    hAccept
    hUnreachable

end FoldedFPrimeAuthority

end DirectCcsFPrime
