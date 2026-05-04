import DirectCcsFPrime.DirectParentOnlyProductionConcreteFPrimePriorBackendOpening

/-!
Strict sound-verifier package for direct backend exact public-IO checks.

This module packages the split backend exact-public-IO verifier checks as the
compressed F' `SoundVerifier` consumed by production terminal soundness. The
trusted cryptographic obligations stay in the backend opening surface; this
file turns the accepted-opens theorem into the canonical certified verifier
and strict sound verifier objects.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorBackendOpeningSoundVerifier

/-- Section 7.1-backed production context. -/
abbrev ProductionContext :=
  DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.ProductionContext

/-- Public image for the parent-only production path. -/
abbrev PublicImage :=
  DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.PublicImage

/-- Proof-carrying folded prior authority for the induced production context. -/
abbrev ProofCarryingPriorProof :=
  @DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.ProofCarryingPriorProof

/-- Backend-shaped exact public-IO opening surface. -/
abbrev ConcreteRuntimeExactPublicIOOpeningSurface :=
  @DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.ConcreteRuntimeExactPublicIOOpeningSurface

/-- Direct backend exact public-IO verifier predicate. -/
abbrev RuntimeVerifyPriorOfRuntimeExactPublicIOChecks :=
  @DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.RuntimeVerifyPriorOfRuntimeExactPublicIOChecks

/-- Certified prior verifier whose predicate is direct backend exact-IO checks. -/
def certifiedPriorVerifierOfRuntimeExactPublicIOChecks
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.ofAcceptedOpens
    ctx
    (RuntimeVerifyPriorOfRuntimeExactPublicIOChecks surface)
    ({ openAuthority := surface.openAuthority } :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
        (PriorProof := PriorProof)
        ctx)
    (DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.runtimeVerifyPriorOfRuntimeExactPublicIOChecks_acceptedOpens
      surface)

/-- The direct backend-checks certified verifier uses exact-IO checks. -/
theorem certifiedPriorVerifierOfRuntimeExactPublicIOChecks_verify
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx) :
    (certifiedPriorVerifierOfRuntimeExactPublicIOChecks surface).verify =
      RuntimeVerifyPriorOfRuntimeExactPublicIOChecks surface :=
  rfl

/-- Strict `SoundVerifier` induced by direct backend exact-IO checks. -/
def soundVerifierOfRuntimeExactPublicIOChecks
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.SoundPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifier
    (certifiedPriorVerifierOfRuntimeExactPublicIOChecks surface)

/-- The direct backend-checks `SoundVerifier` accepts exactly exact-IO checks. -/
theorem soundVerifierOfRuntimeExactPublicIOChecks_accepts_iff
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary} :
    CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfRuntimeExactPublicIOChecks surface)
        steps
        proof
        image <->
      RuntimeVerifyPriorOfRuntimeExactPublicIOChecks
        surface
        steps
        proof
        image := by
  simpa [soundVerifierOfRuntimeExactPublicIOChecks,
    certifiedPriorVerifierOfRuntimeExactPublicIOChecks_verify]
    using
      DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifier_accepts_iff
        (certifiedPriorVerifierOfRuntimeExactPublicIOChecks surface)

/-- Direct backend-checks acceptance opens to folded F' authority. -/
theorem soundVerifierOfRuntimeExactPublicIOChecks_opensToFoldedAuthority
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfRuntimeExactPublicIOChecks surface)
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
  (soundVerifierOfRuntimeExactPublicIOChecks surface).opensToFoldedAuthority
    steps
    proof
    image
    hVerify

/-- The direct backend-checks `SoundVerifier` is same-proof functional. -/
theorem soundVerifierOfRuntimeExactPublicIOChecks_proofFunctional
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx) :
    CompressedFPrimeAuthority.SoundVerifier.ProofFunctional
      (soundVerifierOfRuntimeExactPublicIOChecks surface) :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifierProofFunctional
    (certifiedPriorVerifierOfRuntimeExactPublicIOChecks surface)

/-- One backend proof cannot verify for two different prior public pairs. -/
theorem soundVerifierOfRuntimeExactPublicIOChecks_sameProof
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx}
    {stepsA stepsB : Nat}
    {proof : PriorProof}
    {imageA imageB : PublicImage Digest Boundary}
    (hA :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfRuntimeExactPublicIOChecks surface)
        stepsA
        proof
        imageA)
    (hB :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfRuntimeExactPublicIOChecks surface)
        stepsB
        proof
        imageB) :
    stepsA = stepsB ∧ imageA = imageB :=
  soundVerifierOfRuntimeExactPublicIOChecks_proofFunctional surface hA hB

/--
Terminal acceptance from direct backend checks passes through the strict
`SoundVerifier` object.
-/
theorem acceptedTerminalWithSoundVerifierOfRuntimeExactPublicIOChecksLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfRuntimeExactPublicIOChecks
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
      (soundVerifierOfRuntimeExactPublicIOChecks surface)
      priorSteps
      priorProof
      priorImage
      nextImage
      latestProof := by
  refine
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.acceptedTerminalWithSoundVerifier
      (certifiedPriorVerifierOfRuntimeExactPublicIOChecks surface)
      ?_
  exact
    { priorAccepted := by
        simpa [certifiedPriorVerifierOfRuntimeExactPublicIOChecks_verify]
          using hPrior
      latestAccepted := hLatest }

end DirectParentOnlyProductionConcreteFPrimePriorBackendOpeningSoundVerifier

end DirectCcsFPrime
