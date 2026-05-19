import DirectCcsFPrime.DirectParentOnlyProductionConcreteFPrimePriorBackend

/-!
Raw public-IO authority consequences for the concrete prior F' verifier.

This module keeps the Rust-shaped raw public-vector verifier path audit-ready:
accepted raw public IO opens authority, fixes the prior public image invariants,
and inherits same-proof functionality from the certified verifier object.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorRawIO

/-- Section 7.1-backed production context. -/
abbrev ProductionContext :=
  DirectParentOnlyProductionConcreteFPrimePriorBackend.ProductionContext

/-- Public image for the parent-only production path. -/
abbrev PublicImage :=
  DirectParentOnlyProductionConcreteFPrimePriorBackend.PublicImage

/-- Proof-carrying folded prior authority for the induced production context. -/
abbrev ProofCarryingPriorProof :=
  @DirectParentOnlyProductionConcreteFPrimePriorBackend.ProofCarryingPriorProof

/-- Runtime raw public-vector verifier surface. -/
abbrev ConcreteRuntimeRawPublicIOSurface :=
  @DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeRawPublicIOSurface

/-- Runtime raw public-vector acceptance predicate. -/
abbrev RuntimeVerifyPriorOfRawPublicIO :=
  @DirectParentOnlyProductionConcreteFPrimePriorBackend.RuntimeVerifyPriorOfRawPublicIO

/-- Accepted raw public-vector evidence package. -/
abbrev AcceptedRawPublicIOEvidence :=
  @DirectParentOnlyProductionConcreteFPrimePriorBackend.AcceptedRawPublicIOEvidence

/-- Certified verifier induced by raw terminal public IO equality. -/
abbrev certifiedPriorVerifierOfRawPublicIO :=
  @DirectParentOnlyProductionConcreteFPrimePriorBackend.certifiedPriorVerifierOfRawPublicIO

/-- Fixed authority opener induced by raw terminal public IO equality. -/
abbrev authorityOpenerOfRawPublicIO :=
  @DirectParentOnlyProductionConcreteFPrimePriorBackend.authorityOpenerOfRawPublicIO

/-- Central accepted-opens theorem for the raw public-vector path. -/
abbrev runtimeVerifyPriorOfRawPublicIO_acceptedOpens :=
  @DirectParentOnlyProductionConcreteFPrimePriorBackend.runtimeVerifyPriorOfRawPublicIO_acceptedOpens

/-- Fully exposed accepted raw public-vector evidence. -/
abbrev runtimeVerifyPriorOfRawPublicIO_evidence :=
  @DirectParentOnlyProductionConcreteFPrimePriorBackend.runtimeVerifyPriorOfRawPublicIO_evidence

/-- Opened raw public-vector authority accepts the same public pair. -/
abbrev runtimeVerifyPriorOfRawPublicIO_openedAuthority_accepts_of_open :=
  @DirectParentOnlyProductionConcreteFPrimePriorBackend.runtimeVerifyPriorOfRawPublicIO_openedAuthority_accepts_of_open

/-- Raw public-vector acceptance reaches its claimed prior image. -/
abbrev runtimeVerifyPriorOfRawPublicIO_reaches_prior :=
  @DirectParentOnlyProductionConcreteFPrimePriorBackend.runtimeVerifyPriorOfRawPublicIO_reaches_prior

/-- The certified verifier induced by raw public IO uses the raw predicate. -/
theorem certifiedPriorVerifierOfRawPublicIO_verify
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx) :
    (certifiedPriorVerifierOfRawPublicIO surface).verify =
      RuntimeVerifyPriorOfRawPublicIO surface :=
  rfl

/-- A raw public-vector prior acceptance always opens some authority. -/
theorem runtimeVerifyPriorOfRawPublicIO_openAuthority_ne_none
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfRawPublicIO surface steps proof image) :
    surface.openAuthority proof ≠ none := by
  rcases
    DirectParentOnlyProductionConcreteFPrimePriorBackend.runtimeVerifyPriorOfRawPublicIO_evidence
      surface
      hVerify with
    ⟨_hCompact,
      _hBoundaryReplay,
      _hTranscript,
      _hStatement,
      _hValid,
      _hBoundary,
      _hRawPublicIO,
      hOpened⟩
  rcases hOpened with ⟨authority, hOpen, _hAccepts⟩
  intro hNone
  rw [hNone] at hOpen
  cases hOpen

/-- A raw public-vector verifier cannot accept an unreachable prior image. -/
theorem runtimeVerifyPriorOfRawPublicIO_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfRawPublicIO surface steps proof image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable
    (DirectParentOnlyProductionConcreteFPrimePriorBackend.runtimeVerifyPriorOfRawPublicIO_reaches_prior
      surface
      hVerify)

/--
Accepted raw public-vector verifier proofs expose prior public-image invariants.
-/
theorem runtimeVerifyPriorOfRawPublicIO_publicImageInvariants
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfRawPublicIO surface steps proof image) :
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
    DirectParentOnlyProductionConcreteFPrimePriorBackend.runtimeVerifyPriorOfRawPublicIO_reaches_prior
      surface
      hVerify
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

/-- The raw public-vector concrete prior verifier is same-proof functional. -/
theorem proofFunctionalOfRawPublicIO
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx) :
    CompressedFPrimeAuthority.ProofFunctional
      (RuntimeVerifyPriorOfRawPublicIO surface) := by
  intro stepsA stepsB proof imageA imageB hA hB
  change
    (certifiedPriorVerifierOfRawPublicIO surface).verify
      stepsA
      proof
      imageA at hA
  change
    (certifiedPriorVerifierOfRawPublicIO surface).verify
      stepsB
      proof
      imageB at hB
  exact
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.proofFunctional
      (certifiedPriorVerifierOfRawPublicIO surface)
      hA
      hB

/--
Raw public-vector prior-plus-latest end-to-end theorem.
-/
theorem certifiedSingleTerminalEndToEnd_ofRawPublicIOLatestStep
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
    DirectParentOnlyProductionSuperNeoReuseEndToEnd.CertifiedTerminalEndToEnd
      ctx
      (certifiedPriorVerifierOfRawPublicIO surface).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.certifiedSingleTerminalEndToEnd_ofCertifiedPriorVerifierLatestStep
    (certifiedPriorVerifierOfRawPublicIO surface)
    hPrior
    hLatest

/--
Raw public-vector projection to the non-aggregate private DEC and stage facts.
-/
theorem nonAggregatePrivateDecStageFacts_ofRawPublicIOLatestStep
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
    DirectParentOnlyProductionSuperNeoReuseEndToEnd.CertifiedTerminalNonAggregatePrivateDecStageFacts
      ctx
      priorSteps
      priorImage
      nextImage
      nextImage :=
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.nonAggregatePrivateDecStageFacts_ofCertifiedPriorVerifierLatestStep
    (certifiedPriorVerifierOfRawPublicIO surface)
    hPrior
    hLatest

/--
Raw public-vector projection to the Section 7.1 owner-target stage audit.
-/
theorem section71StageTargetAuditTrail_ofRawPublicIOLatestStep
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
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.TerminalSection71StageTargetAuditTrail
      ctx
      priorSteps
      priorImage
      nextImage
      nextImage :=
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.section71StageTargetAuditTrail_ofCertifiedPriorVerifierLatestStep
    (certifiedPriorVerifierOfRawPublicIO surface)
    hPrior
    hLatest

end DirectParentOnlyProductionConcreteFPrimePriorRawIO

end DirectCcsFPrime
