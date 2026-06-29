import DirectCcsFPrime.ProofSystem.Production.Security.Audit.DirectParentOnlyProductionConcreteFPrimePriorRawProductionCompressedSoundness

/-!
Audit-first consequences for the production exact F' prior verifier.

This module keeps the implementation-facing entry points over the flat verifier
audit package. The audit is the object a concrete compressed verifier should
produce after replaying the public statement, transcript, terminal public IO,
and final-claim checks.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorRawProduction

/-- Audit evidence is accepted by the compressed-soundness certified verifier. -/
theorem certifiedPriorVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact_acceptsAudit
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactCompressedVerifierSoundness checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAudit :
      ProductionExactVerifierAcceptedAudit checks steps proof image publicIO) :
    (certifiedPriorVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact
        checks
        soundness).verify
      steps
      proof
      image :=
  certifiedPriorVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact_acceptsVerifierAccepted
    checks
    soundness
    (productionExactVerifierAccepted_of_audit hAudit)

/-- Audit evidence is accepted by the compressed-soundness strict verifier. -/
theorem soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact_acceptsAudit
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactCompressedVerifierSoundness checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAudit :
      ProductionExactVerifierAcceptedAudit checks steps proof image publicIO) :
    CompressedFPrimeAuthority.SoundVerifier.Accepts
      (soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact
        checks
        soundness)
      steps
      proof
      image :=
  soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact_acceptsVerifierAccepted
    checks
    soundness
    (productionExactVerifierAccepted_of_audit hAudit)

/-- Audit evidence cannot fail to open through the fixed authority opener. -/
theorem productionExactCompressedVerifierSoundness_openAuthority_ne_none_ofAudit
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactCompressedVerifierSoundness checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAudit :
      ProductionExactVerifierAcceptedAudit checks steps proof image publicIO) :
    checks.openAuthority proof ≠ none :=
  productionExactCompressedVerifierSoundness_openAuthority_ne_none
    checks
    soundness
    (productionExactVerifierAccepted_of_audit hAudit)

/-- Any opened authority for accepted audit evidence reaches the same pair. -/
theorem productionExactCompressedVerifierSoundness_openedAuthority_accepts_ofAudit
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactCompressedVerifierSoundness checks)
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
      image :=
  productionExactCompressedVerifierSoundness_openedAuthority_accepts_ofAccepted
    checks
    soundness
    (productionExactVerifierAccepted_of_audit hAudit)
    hOpen

/-- Audit evidence reaches the claimed prior image. -/
theorem productionExactCompressedVerifierSoundness_reaches_prior_ofAudit
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {soundness : ProductionExactCompressedVerifierSoundness checks}
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
      image :=
  productionExactCompressedVerifierSoundness_reaches_prior
    (soundness := soundness)
    (productionExactVerifierAccepted_of_audit hAudit)

/-- Audit evidence exposes prior public-image invariants. -/
theorem productionExactCompressedVerifierSoundness_publicImageInvariants_ofAudit
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {soundness : ProductionExactCompressedVerifierSoundness checks}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAudit :
      ProductionExactVerifierAcceptedAudit checks steps proof image publicIO) :
    DirectParentOnlyProductionConcreteFPrimePrior.AcceptedPriorPublicImageInvariants
      ctx
      steps
      image :=
  productionExactCompressedVerifierSoundness_publicImageInvariants
    (soundness := soundness)
    (productionExactVerifierAccepted_of_audit hAudit)

/-- Audit evidence cannot authorize an unreachable prior. -/
theorem productionExactCompressedVerifierSoundness_cannot_accept_unreachable_prior_ofAudit
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {soundness : ProductionExactCompressedVerifierSoundness checks}
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
  productionExactCompressedVerifierSoundness_cannot_accept_unreachable_prior
    (soundness := soundness)
    (productionExactVerifierAccepted_of_audit hAudit)
    hUnreachable

/--
Audit evidence plus the latest Construction-2 step is accepted by the strict
compressed-soundness verifier.
-/
theorem acceptedTerminalWithSoundVerifierOfProductionExactAuditCompressedVerifierSoundnessLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactCompressedVerifierSoundness checks)
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
      (soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact
        checks
        soundness)
      priorSteps
      priorProof
      priorImage
      nextImage
      latestProof :=
  acceptedTerminalWithSoundVerifierOfProductionExactVerifierAcceptedCompressedVerifierSoundnessLatestStep
    checks
    soundness
    (productionExactVerifierAccepted_of_audit hPrior)
    hLatest

/--
Audit evidence plus the latest Construction-2 step returns the certified
terminal end-to-end package.
-/
theorem certifiedSingleTerminalEndToEnd_ofProductionExactAuditCompressedVerifierSoundnessLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactCompressedVerifierSoundness checks)
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
      (certifiedPriorVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact
        checks
        soundness).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  certifiedSingleTerminalEndToEnd_ofProductionExactVerifierAcceptedCompressedVerifierSoundnessLatestStep
    checks
    soundness
    (productionExactVerifierAccepted_of_audit hPrior)
    hLatest

end DirectParentOnlyProductionConcreteFPrimePriorRawProduction

end DirectCcsFPrime
