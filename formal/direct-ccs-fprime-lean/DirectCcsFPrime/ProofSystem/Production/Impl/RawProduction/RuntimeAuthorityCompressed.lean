import DirectCcsFPrime.ProofSystem.Production.Security.Audit.RawProductionAuditSoundness

/-!
Runtime-authority bridge for the production exact F' prior verifier.

This module connects the backend verifier soundness boundary to the
audit-first compressed-verifier path. The trusted backend obligation remains
`ProductionExactRuntimeAuthoritySoundness`; the derived object is the
`ProductionExactCompressedVerifierSoundness` consumed by the production
terminal theorem stack.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorRawProduction

/--
Runtime authority soundness induces audit-first compressed-verifier soundness.

The runtime boundary already receives every verifier replay fact separately.
The flat audit package carries those same facts, so no caller has to supply an
extra authority-opening premise after the verifier audit has been established.
-/
def compressedSoundnessOfRuntimeAuthority
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks) :
    ProductionExactCompressedVerifierSoundness checks where
  acceptedAuditOpens := by
    intro steps proof image publicIO hAudit
    exact
      soundness.acceptedExactRuntimeOpens
        steps
        proof
        image
        publicIO
        hAudit.compactImageReplay
        hAudit.construction2BoundaryReplay
        hAudit.poseidon2TranscriptReplay
        hAudit.proofIvcPublicImage_eq_canonical
        hAudit.statementPublicValid
        hAudit.construction2Boundary_eq
        hAudit.terminalVerifierPublicIO_eq
        hAudit.terminalPublicValues_eq
        hAudit.boundaryPublicValues_eq

/--
Runtime authority soundness opens folded F' authority from flat verifier audit
evidence.
-/
theorem runtimeAuthorityAuditOpens
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
  (compressedSoundnessOfRuntimeAuthority
      checks
      soundness).acceptedAuditOpens
    steps
    proof
    image
    publicIO
    hAudit

/--
Certified prior verifier induced from runtime authority soundness through the
audit-first compressed-verifier bridge.
-/
def certifiedVerifierOfRuntimeAuthorityAudit
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
  certifiedPriorVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact
    checks
    (compressedSoundnessOfRuntimeAuthority
      checks
      soundness)

/-- Runtime-authority audit evidence is accepted by the certified prior verifier. -/
theorem certifiedVerifierAcceptsRuntimeAuthorityAudit
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
    (certifiedVerifierOfRuntimeAuthorityAudit
        checks
        soundness).verify
      steps
      proof
      image :=
  certifiedPriorVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact_acceptsAudit
    checks
    (compressedSoundnessOfRuntimeAuthority
      checks
      soundness)
    hAudit

/--
Strict `SoundVerifier` induced from runtime authority soundness through the
audit-first compressed-verifier bridge.
-/
def soundVerifierOfRuntimeAuthorityAudit
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
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifier
    (certifiedVerifierOfRuntimeAuthorityAudit
      checks
      soundness)

/-- Runtime-authority audit evidence is accepted by the strict `SoundVerifier`. -/
theorem soundVerifierAcceptsRuntimeAuthorityAudit
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
      (soundVerifierOfRuntimeAuthorityAudit
        checks
        soundness)
      steps
      proof
      image := by
  simpa [
    soundVerifierOfRuntimeAuthorityAudit,
    certifiedVerifierOfRuntimeAuthorityAudit]
    using
      soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact_acceptsAudit
        checks
        (compressedSoundnessOfRuntimeAuthority
          checks
          soundness)
        hAudit

/--
Runtime authority soundness plus audit evidence and the latest Construction-2
step is accepted by the strict prior verifier.
-/
theorem acceptedTerminalOfRuntimeAuthorityAudit
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
      (soundVerifierOfRuntimeAuthorityAudit
        checks
        soundness)
      priorSteps
      priorProof
      priorImage
      nextImage
      latestProof := by
  simpa [
    soundVerifierOfRuntimeAuthorityAudit,
    certifiedVerifierOfRuntimeAuthorityAudit]
    using
      acceptedTerminalWithSoundVerifierOfProductionExactAuditCompressedVerifierSoundnessLatestStep
        checks
        (compressedSoundnessOfRuntimeAuthority
          checks
          soundness)
        hPrior
        hLatest

/--
Runtime authority soundness plus audit evidence returns the certified terminal
end-to-end package.
-/
theorem certifiedEndToEndOfRuntimeAuthorityAudit
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
      (certifiedVerifierOfRuntimeAuthorityAudit
        checks
        soundness).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage := by
  simpa [
    certifiedVerifierOfRuntimeAuthorityAudit]
    using
      certifiedSingleTerminalEndToEnd_ofProductionExactAuditCompressedVerifierSoundnessLatestStep
        checks
        (compressedSoundnessOfRuntimeAuthority
          checks
          soundness)
        hPrior
        hLatest

end DirectParentOnlyProductionConcreteFPrimePriorRawProduction

end DirectCcsFPrime
