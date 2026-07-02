import DirectCcsFPrime.ProofSystem.Production.Impl.PublicIO.Raw.Core

/-!
Strict sound-verifier package for the raw public-vector F' prior verifier.

This module packages the Rust-shaped raw public-IO verifier surface as the
strict compressed F' `SoundVerifier` consumed by terminal production soundness.
The cryptographic boundary stays in the raw backend soundness obligation; this
file only exposes the certified verifier as a reusable strict verifier object.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundVerifier

/-- Section 7.1-backed production context. -/
abbrev ProductionContext :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIO.ProductionContext

/-- Public image for the parent-only production path. -/
abbrev PublicImage :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIO.PublicImage

/-- Proof-carrying folded prior authority for the induced production context. -/
abbrev ProofCarryingPriorProof :=
  @DirectParentOnlyProductionConcreteFPrimePriorRawIO.ProofCarryingPriorProof

/-- Runtime raw public-vector verifier surface. -/
abbrev ConcreteRuntimeRawPublicIOSurface :=
  @DirectParentOnlyProductionConcreteFPrimePriorRawIO.ConcreteRuntimeRawPublicIOSurface

/-- Runtime raw public-vector acceptance predicate. -/
abbrev RuntimeVerifyPriorOfRawPublicIO :=
  @DirectParentOnlyProductionConcreteFPrimePriorRawIO.RuntimeVerifyPriorOfRawPublicIO

/-- Certified verifier induced by raw terminal public IO equality. -/
abbrev certifiedPriorVerifierOfRawPublicIO :=
  @DirectParentOnlyProductionConcreteFPrimePriorRawIO.certifiedPriorVerifierOfRawPublicIO

/-- Strict `SoundVerifier` induced by raw public-vector checks. -/
def soundVerifierOfRawPublicIO
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.SoundPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifier
    (certifiedPriorVerifierOfRawPublicIO surface)

/-- The raw public-vector `SoundVerifier` accepts exactly raw public-IO checks. -/
theorem soundVerifierOfRawPublicIO_accepts_iff
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary} :
    CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfRawPublicIO surface)
        steps
        proof
        image <->
      RuntimeVerifyPriorOfRawPublicIO
        surface
        steps
        proof
        image := by
  simpa [soundVerifierOfRawPublicIO,
    DirectParentOnlyProductionConcreteFPrimePriorRawIO.certifiedPriorVerifierOfRawPublicIO_verify]
    using
      DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifier_accepts_iff
        (certifiedPriorVerifierOfRawPublicIO surface)

/-- Raw public-vector acceptance opens to folded F' authority. -/
theorem soundVerifierOfRawPublicIO_opensToFoldedAuthority
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfRawPublicIO surface)
        steps
        proof
        image) :
    ∃ authority : ProofCarryingPriorProof ctx,
      FoldedFPrimeAuthority.Accepts
        (Transition :=
          DirectParentOnlyProductionSoundness.Transition
            ctx.toProductionContext)
        (initial := ctx.initial)
        steps
        authority
        image :=
  (soundVerifierOfRawPublicIO surface).opensToFoldedAuthority
    steps
    proof
    image
    hVerify

/-- Raw public-vector strict acceptance reaches the claimed prior image. -/
theorem soundVerifierOfRawPublicIO_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfRawPublicIO surface)
        steps
        proof
        image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image := by
  rcases soundVerifierOfRawPublicIO_opensToFoldedAuthority hVerify with
    ⟨authority, hAccepts⟩
  exact
    FoldedFPrimeAuthority.accepts_sound
      (Transition :=
        DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
      (initial := ctx.initial)
      steps
      authority
      image
      hAccepts

/-- Raw public-vector strict acceptance exposes prior public-image invariants. -/
theorem soundVerifierOfRawPublicIO_publicImageInvariants
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfRawPublicIO surface)
        steps
        proof
        image) :
    DirectParentOnlyProductionConcreteFPrimePrior.AcceptedPriorPublicImageInvariants
      ctx
      steps
      image := by
  have hReach :
      FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
        ctx.initial
        steps
        image :=
    soundVerifierOfRawPublicIO_reaches_prior hVerify
  exact
    { stepMatches :=
        Construction2DirectFPrime.reachable_step_counter
          ctx.initialStep
          hReach
      vkDigestMatches :=
        Construction2DirectFPrime.reachable_preserves_vkDigest
          hReach
      initialBoundaryMatches :=
        Construction2DirectFPrime.reachable_preserves_initialBoundary
          hReach
      wellFormed :=
        Construction2DirectFPrime.reachable_wellFormed_of_initial
          ctx.initialWellFormed
          hReach }

/-- Raw public-vector strict acceptance rejects unreachable prior images. -/
theorem soundVerifierOfRawPublicIO_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfRawPublicIO surface)
        steps
        proof
        image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable
    (soundVerifierOfRawPublicIO_reaches_prior hVerify)

/-- The raw public-vector `SoundVerifier` is same-proof functional. -/
theorem soundVerifierOfRawPublicIO_proofFunctional
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx) :
    CompressedFPrimeAuthority.SoundVerifier.ProofFunctional
      (soundVerifierOfRawPublicIO surface) :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifierProofFunctional
    (certifiedPriorVerifierOfRawPublicIO surface)

/-- One raw public-vector proof cannot verify for two different public pairs. -/
theorem soundVerifierOfRawPublicIO_sameProof
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx}
    {stepsA stepsB : Nat}
    {proof : PriorProof}
    {imageA imageB : PublicImage Digest Boundary}
    (hA :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfRawPublicIO surface)
        stepsA
        proof
        imageA)
    (hB :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfRawPublicIO surface)
        stepsB
        proof
        imageB) :
    stepsA = stepsB ∧ imageA = imageB :=
  soundVerifierOfRawPublicIO_proofFunctional surface hA hB

/--
Terminal acceptance from raw public-vector checks passes through the strict
`SoundVerifier` object.
-/
theorem acceptedTerminalWithSoundVerifierOfRawPublicIOLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfRawPublicIO
        surface
        priorSteps
        priorProof
        priorImage)
    (hLatest :
      DirectParentOnlyProductionSoundness.VerifyLatestStep
        ctx.toProductionContext
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.AcceptedTerminal
      ctx
      (soundVerifierOfRawPublicIO surface)
      priorSteps
      priorProof
      priorImage
      nextImage
      latestProof := by
  refine
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.acceptedTerminalWithSoundVerifier
      (certifiedPriorVerifierOfRawPublicIO surface)
      ?_
  exact
    { priorAccepted := by
        simpa [DirectParentOnlyProductionConcreteFPrimePriorRawIO.certifiedPriorVerifierOfRawPublicIO_verify]
          using hPrior
      latestAccepted := hLatest }

end DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundVerifier

end DirectCcsFPrime
