import DirectCcsFPrime.Core.FoldedFPrimeAuthority

/-!
Compressed folded F' prior authority.

This module is the theorem boundary for a terminal proof that consumes a
compressed prior F' proof instead of a proof-carrying Lean reachability object.
The compressed verifier may be opaque, but its acceptance predicate must imply
reachability under the same F' transition and base image.
-/

namespace DirectCcsFPrime

namespace CompressedFPrimeAuthority

/--
Soundness requirement for a compressed prior-F' verifier.

`VerifyPrior steps proof image` is the verifier predicate for the compressed
prior proof. It is sound only when acceptance implies that `image` is reachable
from `initial` after `steps` applications of the same `Transition` relation.
-/
def VerifierSound
    {Image Proof : Type}
    (Transition : Nat → Image → Image → Prop)
    (initial : Image)
    (VerifyPrior : Nat → Proof → Image → Prop) : Prop :=
  ∀ steps proof image,
    VerifyPrior steps proof image →
      FPrimeInduction.Reachable Transition initial steps image

/-- The prior-authority acceptance predicate induced by a compressed verifier. -/
def Accepts
    {Image Proof : Type}
    (VerifyPrior : Nat → Proof → Image → Prop)
    (steps : Nat)
    (proof : Proof)
    (image : Image) : Prop :=
  VerifyPrior steps proof image

/--
Verifier object for a compressed prior F' proof system.

The verifier predicate itself may be opaque. Its required theorem is precise:
every accepted proof must open to proof-carrying folded F' authority for the
same `(steps, image)`.
-/
structure SoundVerifier
    {Image Proof : Type}
    (Transition : Nat → Image → Image → Prop)
    (initial : Image) where
  verify : Nat → Proof → Image → Prop
  opensToFoldedAuthority :
    ∀ steps proof image,
      verify steps proof image →
        ∃ authority : FoldedFPrimeAuthority.Authority Transition initial,
          FoldedFPrimeAuthority.Accepts
            (Transition := Transition)
            (initial := initial)
            steps
            authority
            image

namespace SoundVerifier

/-- Acceptance predicate induced by a sound compressed verifier object. -/
def Accepts
    {Image Proof : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image}
    (verifier : SoundVerifier (Image := Image) (Proof := Proof) Transition initial)
    (steps : Nat)
    (proof : Proof)
    (image : Image) : Prop :=
  verifier.verify steps proof image

end SoundVerifier

/--
Same-proof functionality for a compressed prior verifier.

This is stronger than soundness. It says one opaque proof cannot be accepted as
authority for two different prior `(steps, image)` pairs.
-/
def ProofFunctional
    {Image Proof : Type}
    (VerifyPrior : Nat → Proof → Image → Prop) : Prop :=
  ∀ {stepsA stepsB : Nat}
    {proof : Proof}
    {imageA imageB : Image},
    VerifyPrior stepsA proof imageA →
    VerifyPrior stepsB proof imageB →
      stepsA = stepsB ∧ imageA = imageB

namespace SoundVerifier

/--
Same-proof functionality for the acceptance predicate induced by a
`SoundVerifier`.
-/
def ProofFunctional
    {Image Proof : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image}
    (verifier : SoundVerifier (Image := Image) (Proof := Proof) Transition initial) :
    Prop :=
  CompressedFPrimeAuthority.ProofFunctional verifier.verify

end SoundVerifier

namespace ReplayNecessity

inductive TwoImage where
  | initial
  | next
  deriving DecidableEq

def twoImageTransition : Nat → TwoImage → TwoImage → Prop :=
  fun _ _ _ => True

def twoImageInitial : TwoImage :=
  TwoImage.initial

def sameProofTwoImageVerify : Nat → Unit → TwoImage → Prop :=
  fun steps _ image =>
    (steps = 0 ∧ image = TwoImage.initial) ∨
      (steps = 1 ∧ image = TwoImage.next)

def nextAuthority :
    FoldedFPrimeAuthority.Authority twoImageTransition twoImageInitial :=
  { steps := 1
    image := TwoImage.next
    reachable :=
      FPrimeInduction.Reachable.step
        FPrimeInduction.Reachable.base
        trivial }

def sameProofTwoImageSoundVerifier :
    SoundVerifier
      (Image := TwoImage)
      (Proof := Unit)
      twoImageTransition
      twoImageInitial where
  verify := sameProofTwoImageVerify
  opensToFoldedAuthority := by
    intro steps proof image hVerify
    rcases hVerify with hBase | hNext
    · rcases hBase with ⟨hSteps, hImage⟩
      subst hSteps
      subst hImage
      exact
        ⟨FoldedFPrimeAuthority.base
            (Transition := twoImageTransition)
            twoImageInitial,
          FoldedFPrimeAuthority.base_accepts
            (Transition := twoImageTransition)
            twoImageInitial⟩
    · rcases hNext with ⟨hSteps, hImage⟩
      subst hSteps
      subst hImage
      exact ⟨nextAuthority, ⟨rfl, rfl⟩⟩

theorem sameProofTwoImage_accepts_base :
    sameProofTwoImageSoundVerifier.verify
      0
      ()
      TwoImage.initial := by
  exact Or.inl ⟨rfl, rfl⟩

theorem sameProofTwoImage_accepts_next :
    sameProofTwoImageSoundVerifier.verify
      1
      ()
      TwoImage.next := by
  exact Or.inr ⟨rfl, rfl⟩

end ReplayNecessity

/--
A sound compressed verifier supplies valid Construction-2 prior authority.
-/
theorem accepts_sound_of_verifier_sound
    {Image Proof : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image}
    {VerifyPrior : Nat → Proof → Image → Prop}
    (hVerifier : VerifierSound Transition initial VerifyPrior) :
    FPrimeInduction.PriorAuthoritySound
      Transition
      initial
      (Accepts VerifyPrior) := by
  intro steps proof image hAccept
  exact hVerifier steps proof image hAccept

/--
A compressed prior verifier is sound if every accepted proof opens to a
proof-carrying folded F' authority for the same `(steps, image)`.

This is the concrete bridge from a compressed verifier to the theorem-level
authority object. It keeps the cryptographic proof system opaque while making
the required obligation exact.
-/
theorem verifier_sound_of_opens_to_folded_authority
    {Image Proof : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image}
    {VerifyPrior : Nat → Proof → Image → Prop}
    (hOpens :
      ∀ steps proof image,
        VerifyPrior steps proof image →
          ∃ authority : FoldedFPrimeAuthority.Authority Transition initial,
            FoldedFPrimeAuthority.Accepts
              (Transition := Transition)
              (initial := initial)
              steps
              authority
              image) :
    VerifierSound Transition initial VerifyPrior := by
  intro steps proof image hVerify
  rcases hOpens steps proof image hVerify with ⟨authority, hAccept⟩
  exact
    FoldedFPrimeAuthority.accepts_sound
      (Transition := Transition)
      (initial := initial)
      steps
      authority
      image
      hAccept

/--
A `SoundVerifier` object satisfies the compressed-verifier soundness
requirement.
-/
theorem verifier_sound_of_sound_verifier
    {Image Proof : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image}
    (verifier : SoundVerifier (Image := Image) (Proof := Proof) Transition initial) :
    VerifierSound Transition initial verifier.verify :=
  verifier_sound_of_opens_to_folded_authority
    verifier.opensToFoldedAuthority

/--
A `SoundVerifier` object supplies valid prior authority for terminal
compression.
-/
theorem sound_verifier_prior_authority_sound
    {Image Proof : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image}
    (verifier : SoundVerifier (Image := Image) (Proof := Proof) Transition initial) :
    FPrimeInduction.PriorAuthoritySound
      Transition
      initial
      (SoundVerifier.Accepts verifier) :=
  accepts_sound_of_verifier_sound
    (verifier_sound_of_sound_verifier verifier)

/--
Terminal compression reaches the final image when it consumes a sound compressed
prior proof and a sound latest-step verifier.
-/
theorem terminal_compression_reaches_final
    {Image PriorProof LatestProof : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image}
    {VerifyPrior : Nat → PriorProof → Image → Prop}
    {VerifyLatestStep :
      Nat → PriorProof → Image → Image → LatestProof → Prop}
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : Image}
    {latestProof : LatestProof}
    (hVerifier : VerifierSound Transition initial VerifyPrior)
    (hLatest :
      FPrimeInduction.LatestStepSound
        Transition
        VerifyLatestStep)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (Accepts VerifyPrior)
        VerifyLatestStep
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof) :
    FPrimeInduction.Reachable
      Transition
      initial
      (priorSteps + 1)
      nextImage :=
  FPrimeInduction.terminal_compression_reaches_final
    (accepts_sound_of_verifier_sound hVerifier)
    hLatest
    hAccepted

/--
Terminal compression reaches the final image when the compressed prior verifier
opens every accepted proof to proof-carrying folded authority.

This is the production-shaped theorem boundary for replacing a Lean
proof-carrying prior authority object with a compressed verifier: the verifier
does not need to expose the authority publicly, but its accepted proof must
imply such authority exists.
-/
theorem terminal_compression_reaches_final_of_opens_to_folded_authority
    {Image PriorProof LatestProof : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image}
    {VerifyPrior : Nat → PriorProof → Image → Prop}
    {VerifyLatestStep :
      Nat → PriorProof → Image → Image → LatestProof → Prop}
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : Image}
    {latestProof : LatestProof}
    (hOpens :
      ∀ steps proof image,
        VerifyPrior steps proof image →
          ∃ authority : FoldedFPrimeAuthority.Authority Transition initial,
            FoldedFPrimeAuthority.Accepts
              (Transition := Transition)
              (initial := initial)
              steps
              authority
              image)
    (hLatest :
      FPrimeInduction.LatestStepSound
        Transition
        VerifyLatestStep)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (Accepts VerifyPrior)
        VerifyLatestStep
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof) :
    FPrimeInduction.Reachable
      Transition
      initial
      (priorSteps + 1)
      nextImage :=
  terminal_compression_reaches_final
    (verifier_sound_of_opens_to_folded_authority hOpens)
    hLatest
    hAccepted

/--
Terminal compression reaches the final image when it consumes a sound compressed
prior verifier object.

This theorem is the compact production boundary: the proof object remains
opaque, but the verifier object carries the exact extraction/opening theorem
needed for induction authority.
-/
theorem terminal_compression_reaches_final_of_sound_verifier
    {Image PriorProof LatestProof : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image}
    (verifier :
      SoundVerifier
        (Image := Image)
        (Proof := PriorProof)
        Transition
        initial)
    {VerifyLatestStep :
      Nat → PriorProof → Image → Image → LatestProof → Prop}
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : Image}
    {latestProof : LatestProof}
    (hLatest :
      FPrimeInduction.LatestStepSound
        Transition
        VerifyLatestStep)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (SoundVerifier.Accepts verifier)
        VerifyLatestStep
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof) :
    FPrimeInduction.Reachable
      Transition
      initial
      (priorSteps + 1)
      nextImage :=
  FPrimeInduction.terminal_compression_reaches_final
    (sound_verifier_prior_authority_sound verifier)
    hLatest
    hAccepted

/--
If a compressed prior verifier accepts an unreachable image, it cannot satisfy
the verifier-soundness requirement.
-/
theorem accepted_unreachable_is_not_sound_verifier
    {Image Proof : Type}
    {Transition : Nat → Image → Image → Prop}
    {initial : Image}
    {VerifyPrior : Nat → Proof → Image → Prop}
    {steps : Nat}
    {proof : Proof}
    {image : Image}
    (hAccept : VerifyPrior steps proof image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable Transition initial steps image) :
    ¬ VerifierSound Transition initial VerifyPrior := by
  intro hSound
  exact hUnreachable (hSound steps proof image hAccept)

/--
`SoundVerifier` alone does not imply same-proof replay functionality.

The witness verifier is sound: both accepted prior pairs are reachable and open
to proof-carrying folded authority. Nevertheless the same opaque proof value
`()`, accepted by the same verifier, authorizes two different prior pairs.

This is why replay-stable production endpoints require a fixed
`PriorVerifierAuthorityOpening`/opener, not only a per-acceptance existence
theorem for folded authority.
-/
theorem sound_verifier_does_not_imply_same_proof_functional :
    ∃ verifier :
        SoundVerifier
          (Image := ReplayNecessity.TwoImage)
          (Proof := Unit)
          ReplayNecessity.twoImageTransition
          ReplayNecessity.twoImageInitial,
      ¬ SoundVerifier.ProofFunctional verifier := by
  refine ⟨ReplayNecessity.sameProofTwoImageSoundVerifier, ?_⟩
  intro hFunctional
  rcases
      hFunctional
        ReplayNecessity.sameProofTwoImage_accepts_base
        ReplayNecessity.sameProofTwoImage_accepts_next with
    ⟨hSteps, _hImage⟩
  exact Nat.succ_ne_zero 0 hSteps.symm

end CompressedFPrimeAuthority

end DirectCcsFPrime
