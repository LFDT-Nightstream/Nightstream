import DirectCcsFPrime.Core.FPrimeInduction

/-!
Direct CCS Construction-2 public image and latest F' step.

This module fixes the concrete public-image shape used by the non-VM direct
CCS path. It instantiates the latest-step side of `FPrimeInduction`; the
folded prior-authority soundness remains a separate theorem obligation.
-/

namespace DirectCcsFPrime

namespace Construction2DirectFPrime

/--
Compact public image for the single-relation direct CCS `F'` path.

`accumulator` is a handle/root for the folded F' accumulator authority, not raw
semantic CE projection data.
-/
structure PublicImage
    (Digest Boundary AccHandle : Type) where
  vkDigest : Digest
  step : Nat
  initialBoundary : Boundary
  currentBoundary : Boundary
  accumulator : AccHandle
  pc : Nat
deriving DecidableEq

/-- Single-relation direct CCS uses fixed program counter `pc = 1`. -/
def WellFormed
    {Digest Boundary AccHandle : Type}
    (image : PublicImage Digest Boundary AccHandle) : Prop :=
  image.pc = 1

/--
One direct CCS `F'` transition.

`BoundaryStep` is the direct computation boundary update. `AccumulatorStep` is
the verifier-side accumulator update performed by NIFS.V for the latest step.
-/
def Transition
    {Digest Boundary AccHandle : Type}
    (BoundaryStep : Nat → Boundary → Boundary → Prop)
    (AccumulatorStep : Nat → AccHandle → AccHandle → Prop)
    (i : Nat)
    (prior next : PublicImage Digest Boundary AccHandle) : Prop :=
  prior.step = i ∧
  next.step = i + 1 ∧
  prior.vkDigest = next.vkDigest ∧
  prior.initialBoundary = next.initialBoundary ∧
  prior.pc = 1 ∧
  next.pc = 1 ∧
  BoundaryStep i prior.currentBoundary next.currentBoundary ∧
  AccumulatorStep i prior.accumulator next.accumulator

/--
Canonical latest-step verifier relation for direct CCS `F'`.

The terminal proof may carry additional proof/advice data in a concrete
implementation. At the theorem boundary, acceptance is exactly the direct
Construction-2 transition.
-/
def VerifyLatestStep
    {Digest Boundary AccHandle Authority : Type}
    (BoundaryStep : Nat → Boundary → Boundary → Prop)
    (AccumulatorStep : Nat → AccHandle → AccHandle → Prop)
    (i : Nat)
    (_authority : Authority)
    (prior next : PublicImage Digest Boundary AccHandle)
    (_proof : Unit) : Prop :=
  Transition BoundaryStep AccumulatorStep i prior next

/--
The canonical latest-step verifier is sound for the direct Construction-2
transition by definition.
-/
theorem latest_step_sound
    {Digest Boundary AccHandle Authority : Type}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {AccumulatorStep : Nat → AccHandle → AccHandle → Prop} :
    FPrimeInduction.LatestStepSound
      (Transition
        (Digest := Digest)
        (Boundary := Boundary)
        (AccHandle := AccHandle)
        BoundaryStep
        AccumulatorStep)
      (VerifyLatestStep
        (Digest := Digest)
        (Boundary := Boundary)
        (AccHandle := AccHandle)
        (Authority := Authority)
        BoundaryStep
        AccumulatorStep) := by
  intro step authority priorImage nextImage proof hVerified
  exact hVerified

/--
Specialized terminal theorem for the direct CCS Construction-2 image.

The only nontrivial premise left is `PriorAuthoritySound`: the accepted folded
F' authority must imply reachability of the prior public image.
-/
theorem terminal_direct_fprime_reaches_final
    {Digest Boundary AccHandle Authority : Type}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {AccumulatorStep : Nat → AccHandle → AccHandle → Prop}
    {initial : PublicImage Digest Boundary AccHandle}
    {AuthorityAccepts :
      Nat →
        Authority →
        PublicImage Digest Boundary AccHandle →
        Prop}
    {priorSteps : Nat}
    {priorAuthority : Authority}
    {priorImage nextImage : PublicImage Digest Boundary AccHandle}
    {proof : Unit}
    (hAuthority :
      FPrimeInduction.PriorAuthoritySound
        (Transition BoundaryStep AccumulatorStep)
        initial
        AuthorityAccepts)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        AuthorityAccepts
        (VerifyLatestStep
          (Authority := Authority)
          BoundaryStep
          AccumulatorStep)
        priorSteps
        priorAuthority
        priorImage
        nextImage
        proof) :
    FPrimeInduction.Reachable
      (Transition BoundaryStep AccumulatorStep)
      initial
      (priorSteps + 1)
      nextImage :=
  FPrimeInduction.terminal_compression_reaches_final
    hAuthority
    latest_step_sound
    hAccepted

/-- A direct F' transition always advances the public step counter by one. -/
theorem transition_next_step
    {Digest Boundary AccHandle : Type}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {AccumulatorStep : Nat → AccHandle → AccHandle → Prop}
    {i : Nat}
    {prior next : PublicImage Digest Boundary AccHandle}
    (hTransition : Transition BoundaryStep AccumulatorStep i prior next) :
    next.step = prior.step + 1 := by
  rcases hTransition with
    ⟨hPrior, hNext, _hVk, _hInitial, _hPriorPc, _hNextPc, _hBoundary, _hAcc⟩
  subst hPrior
  exact hNext

/-- A direct F' transition preserves the verifier-key digest. -/
theorem transition_preserves_vkDigest
    {Digest Boundary AccHandle : Type}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {AccumulatorStep : Nat → AccHandle → AccHandle → Prop}
    {i : Nat}
    {prior next : PublicImage Digest Boundary AccHandle}
    (hTransition : Transition BoundaryStep AccumulatorStep i prior next) :
    prior.vkDigest = next.vkDigest :=
  hTransition.2.2.1

/-- A direct F' transition preserves the initial computation boundary. -/
theorem transition_preserves_initialBoundary
    {Digest Boundary AccHandle : Type}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {AccumulatorStep : Nat → AccHandle → AccHandle → Prop}
    {i : Nat}
    {prior next : PublicImage Digest Boundary AccHandle}
    (hTransition : Transition BoundaryStep AccumulatorStep i prior next) :
    prior.initialBoundary = next.initialBoundary :=
  hTransition.2.2.2.1

/-- A direct F' transition preserves fixed `pc = 1` on both sides. -/
theorem transition_pc_fixed
    {Digest Boundary AccHandle : Type}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {AccumulatorStep : Nat → AccHandle → AccHandle → Prop}
    {i : Nat}
    {prior next : PublicImage Digest Boundary AccHandle}
    (hTransition : Transition BoundaryStep AccumulatorStep i prior next) :
    WellFormed prior ∧ WellFormed next :=
  ⟨hTransition.2.2.2.2.1, hTransition.2.2.2.2.2.1⟩

/--
Reachability from a zero-step base image fixes the public step counter.

The final compressed proof exposes `image.step` as public state. This theorem
connects that counter to the number of accepted F' transitions rather than
leaving it as an independently chosen field.
-/
theorem reachable_step_counter
    {Digest Boundary AccHandle : Type}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {AccumulatorStep : Nat → AccHandle → AccHandle → Prop}
    {initial image : PublicImage Digest Boundary AccHandle}
    {steps : Nat}
    (hInitialStep : initial.step = 0)
    (hReach :
      FPrimeInduction.Reachable
        (Transition BoundaryStep AccumulatorStep)
        initial
        steps
        image) :
    image.step = steps := by
  induction hReach with
  | base =>
      exact hInitialStep
  | step _hPrior hTransition _ih =>
      exact hTransition.2.1

/--
Reachability preserves the verifier-key digest across the whole F' chain.
-/
theorem reachable_preserves_vkDigest
    {Digest Boundary AccHandle : Type}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {AccumulatorStep : Nat → AccHandle → AccHandle → Prop}
    {initial image : PublicImage Digest Boundary AccHandle}
    {steps : Nat}
    (hReach :
      FPrimeInduction.Reachable
        (Transition BoundaryStep AccumulatorStep)
        initial
        steps
        image) :
    initial.vkDigest = image.vkDigest := by
  induction hReach with
  | base =>
      rfl
  | step _hPrior hTransition ih =>
      exact ih.trans hTransition.2.2.1

/--
Reachability preserves the initial direct-computation boundary.
-/
theorem reachable_preserves_initialBoundary
    {Digest Boundary AccHandle : Type}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {AccumulatorStep : Nat → AccHandle → AccHandle → Prop}
    {initial image : PublicImage Digest Boundary AccHandle}
    {steps : Nat}
    (hReach :
      FPrimeInduction.Reachable
        (Transition BoundaryStep AccumulatorStep)
        initial
        steps
        image) :
    initial.initialBoundary = image.initialBoundary := by
  induction hReach with
  | base =>
      rfl
  | step _hPrior hTransition ih =>
      exact ih.trans hTransition.2.2.2.1

/--
Reachability preserves the fixed single-relation program counter.
-/
theorem reachable_wellFormed_of_initial
    {Digest Boundary AccHandle : Type}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {AccumulatorStep : Nat → AccHandle → AccHandle → Prop}
    {initial image : PublicImage Digest Boundary AccHandle}
    {steps : Nat}
    (hInitial : WellFormed initial)
    (hReach :
      FPrimeInduction.Reachable
        (Transition BoundaryStep AccumulatorStep)
        initial
        steps
        image) :
    WellFormed image := by
  induction hReach with
  | base =>
      exact hInitial
  | step _hPrior hTransition _ih =>
      exact hTransition.2.2.2.2.2.1

/--
Terminal direct compression inherits the public-image invariants of reachable
Construction-2 chains.
-/
theorem terminal_direct_fprime_public_image_invariants
    {Digest Boundary AccHandle Authority : Type}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {AccumulatorStep : Nat → AccHandle → AccHandle → Prop}
    {initial : PublicImage Digest Boundary AccHandle}
    {AuthorityAccepts :
      Nat →
        Authority →
        PublicImage Digest Boundary AccHandle →
        Prop}
    {priorSteps : Nat}
    {priorAuthority : Authority}
    {priorImage nextImage : PublicImage Digest Boundary AccHandle}
    {proof : Unit}
    (hInitialStep : initial.step = 0)
    (hInitialWellFormed : WellFormed initial)
    (hAuthority :
      FPrimeInduction.PriorAuthoritySound
        (Transition BoundaryStep AccumulatorStep)
        initial
        AuthorityAccepts)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        AuthorityAccepts
        (VerifyLatestStep
          (Authority := Authority)
          BoundaryStep
          AccumulatorStep)
        priorSteps
        priorAuthority
        priorImage
        nextImage
        proof) :
    nextImage.step = priorSteps + 1 ∧
      initial.vkDigest = nextImage.vkDigest ∧
      initial.initialBoundary = nextImage.initialBoundary ∧
      WellFormed nextImage := by
  let hReach :
      FPrimeInduction.Reachable
        (Transition BoundaryStep AccumulatorStep)
        initial
        (priorSteps + 1)
        nextImage :=
    terminal_direct_fprime_reaches_final hAuthority hAccepted
  exact
    ⟨reachable_step_counter hInitialStep hReach,
      reachable_preserves_vkDigest hReach,
      reachable_preserves_initialBoundary hReach,
      reachable_wellFormed_of_initial hInitialWellFormed hReach⟩

end Construction2DirectFPrime

end DirectCcsFPrime
