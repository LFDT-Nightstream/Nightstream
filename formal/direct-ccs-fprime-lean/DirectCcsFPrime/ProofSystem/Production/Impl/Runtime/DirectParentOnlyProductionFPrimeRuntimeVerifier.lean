import DirectCcsFPrime.ProofSystem.Production.Impl.RawProduction.DirectParentOnlyProductionConcreteFPrimePriorRawProductionRuntimeAuthorityCompressed

/-!
Reader-facing surface for the production prior F' runtime verifier.

This file owns the compact API for the concrete prior verifier path. The only
backend boundary is runtime authority soundness: accepted verifier audit facts
must open the fixed authority object to folded F' reachability for the same
`(steps, image)` pair.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionFPrimeRuntimeVerifier

open DirectParentOnlyProductionConcreteFPrimePriorRawProduction

/-- Production exact runtime opening surface induced by verifier authority soundness. -/
def openingSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks) :
    ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx :=
  productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
    checks
    soundness

/--
Runtime verifier acceptance opens folded F' authority for the same public pair.
-/
theorem verifyOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionExact
        (openingSurface checks soundness)
        steps
        proof
        image) :
    ∃ authority : ProofCarryingPriorProof ctx,
      checks.openAuthority proof = some authority ∧
        FoldedFPrimeAuthority.Accepts
          (Transition :=
            DirectParentOnlyProductionSoundness.Transition
              ctx.toProductionContext)
          (initial := ctx.initial)
          steps
          authority
          image := by
  simpa [openingSurface]
    using
      runtimeVerifyPriorOfProductionExactRuntimeAuthoritySoundness_acceptedOpens
        checks
        soundness
        hVerify

/-- Audit evidence opens folded F' authority for the same public pair. -/
theorem auditOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAudit :
      ProductionExactVerifierAcceptedAudit checks steps proof image publicIO) :
    ∃ authority : ProofCarryingPriorProof ctx,
      checks.openAuthority proof = some authority ∧
        FoldedFPrimeAuthority.Accepts
          (Transition :=
            DirectParentOnlyProductionSoundness.Transition
              ctx.toProductionContext)
          (initial := ctx.initial)
          steps
          authority
          image :=
  runtimeAuthorityAuditOpens
    checks
    soundness
    hAudit

/-- Audit evidence cannot pass if the fixed opener returns no authority. -/
theorem auditOpenAuthority_ne_none
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAudit :
      ProductionExactVerifierAcceptedAudit checks steps proof image publicIO) :
    checks.openAuthority proof ≠ none := by
  rcases auditOpens checks soundness hAudit with
    ⟨authority, hOpen, _hAccepts⟩
  intro hNone
  rw [hNone] at hOpen
  cases hOpen

/--
Any authority returned by the fixed opener for audit evidence accepts the same
`(steps, image)` pair.
-/
theorem auditOpenedAuthorityAccepts
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    {authority : ProofCarryingPriorProof ctx}
    (hAudit :
      ProductionExactVerifierAcceptedAudit checks steps proof image publicIO)
    (hOpen : checks.openAuthority proof = some authority) :
    FoldedFPrimeAuthority.Accepts
      (Transition :=
        DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
      (initial := ctx.initial)
      steps
      authority
      image := by
  rcases auditOpens checks soundness hAudit with
    ⟨openedAuthority, hOpened, hAccepts⟩
  have hSame : some authority = some openedAuthority := by
    rw [← hOpen, hOpened]
  cases hSame
  exact hAccepts

/-- Audit evidence reaches the claimed prior public image. -/
theorem auditReaches
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAudit :
      ProductionExactVerifierAcceptedAudit checks steps proof image publicIO) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image := by
  rcases auditOpens checks soundness hAudit with
    ⟨authority, _hOpen, hAccepts⟩
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

/-- Audit evidence exposes the prior public-image invariants. -/
theorem auditPublicImageInvariants
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAudit :
      ProductionExactVerifierAcceptedAudit checks steps proof image publicIO) :
    DirectParentOnlyProductionConcreteFPrimePrior.AcceptedPriorPublicImageInvariants
      ctx
      steps
      image := by
  have hReach :=
    auditReaches
      checks
      soundness
      hAudit
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

/-- Audit evidence cannot authorize an unreachable prior public image. -/
theorem auditCannotAcceptUnreachable
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAudit :
      ProductionExactVerifierAcceptedAudit checks steps proof image publicIO)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable
    (auditReaches
      checks
      soundness
      hAudit)

/-- Runtime authority soundness as compressed-verifier soundness. -/
def compressedSoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks) :
    ProductionExactCompressedVerifierSoundness checks :=
  compressedSoundnessOfRuntimeAuthority
    checks
    soundness

/-- Certified prior verifier induced by runtime authority soundness. -/
def certified
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  certifiedVerifierOfRuntimeAuthorityAudit
    checks
    soundness

/-- Audit evidence is accepted by the certified prior verifier. -/
theorem certifiedAcceptsAudit
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAudit :
      ProductionExactVerifierAcceptedAudit checks steps proof image publicIO) :
    (certified checks soundness).verify
      steps
      proof
      image := by
  simpa [certified]
    using
      certifiedVerifierAcceptsRuntimeAuthorityAudit
        checks
        soundness
        hAudit

/-- Strict sound verifier induced by runtime authority soundness. -/
def sound
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.SoundPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  soundVerifierOfRuntimeAuthorityAudit
    checks
    soundness

/-- Audit evidence is accepted by the strict sound verifier. -/
theorem soundAcceptsAudit
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAudit :
      ProductionExactVerifierAcceptedAudit checks steps proof image publicIO) :
    CompressedFPrimeAuthority.SoundVerifier.Accepts
      (sound checks soundness)
      steps
      proof
      image := by
  simpa [sound]
    using
      soundVerifierAcceptsRuntimeAuthorityAudit
        checks
        soundness
        hAudit

/-- The strict sound verifier opens folded F' authority for accepted proofs. -/
theorem soundOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (sound checks soundness)
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
  (sound checks soundness).opensToFoldedAuthority
    steps
    proof
    image
    hVerify

/-- One proof accepted by this strict verifier has one public `(steps, image)`. -/
theorem soundSameProof
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {soundness : ProductionExactRuntimeAuthoritySoundness checks}
    {stepsA stepsB : Nat}
    {proof : PriorProof}
    {imageA imageB : PublicImage Digest Boundary}
    (hA :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (sound checks soundness)
        stepsA
        proof
        imageA)
    (hB :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (sound checks soundness)
        stepsB
        proof
        imageB) :
    stepsA = stepsB ∧ imageA = imageB :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifierProofFunctional
    (certified checks soundness)
    hA
    hB

/-- Audit evidence plus the latest Construction-2 step is terminal-accepted. -/
theorem acceptedTerminal
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    {latestProof : Unit}
    (hPrior :
      ProductionExactVerifierAcceptedAudit
        checks
        priorSteps
        priorProof
        priorImage
        publicIO)
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
      (sound checks soundness)
      priorSteps
      priorProof
      priorImage
      nextImage
      latestProof := by
  simpa [sound]
    using
      acceptedTerminalOfRuntimeAuthorityAudit
        checks
        soundness
        hPrior
        hLatest

/-- Audit evidence plus the latest Construction-2 step gives the end-to-end result. -/
theorem endToEnd
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    {latestProof : Unit}
    (hPrior :
      ProductionExactVerifierAcceptedAudit
        checks
        priorSteps
        priorProof
        priorImage
        publicIO)
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
      (certified checks soundness).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage := by
  simpa [certified]
    using
      certifiedEndToEndOfRuntimeAuthorityAudit
        checks
        soundness
        hPrior
        hLatest

/-- Extract the exact non-aggregate private DEC/stage facts from `endToEnd`. -/
theorem privateDecFacts
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    {latestProof : Unit}
    (hPrior :
      ProductionExactVerifierAcceptedAudit
        checks
        priorSteps
        priorProof
        priorImage
        publicIO)
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
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.nonAggregatePrivateDecStageFacts_of_certifiedTerminalEndToEnd
    (endToEnd
      checks
      soundness
      hPrior
      hLatest)

/-- Extract the exact Section 7.1 owner-target stage audit from `endToEnd`. -/
theorem stageAudit
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    {latestProof : Unit}
    (hPrior :
      ProductionExactVerifierAcceptedAudit
        checks
        priorSteps
        priorProof
        priorImage
        publicIO)
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
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.section71StageTargetAuditTrail_of_certifiedTerminalEndToEnd
    (endToEnd
      checks
      soundness
      hPrior
      hLatest)

end DirectParentOnlyProductionFPrimeRuntimeVerifier

end DirectCcsFPrime
