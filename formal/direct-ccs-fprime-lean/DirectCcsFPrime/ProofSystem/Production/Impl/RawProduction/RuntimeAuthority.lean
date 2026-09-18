import DirectCcsFPrime.ProofSystem.Production.Impl.RawProduction.RuntimeExact

/-!
Runtime authority soundness for the production exact F' prior verifier.

This file owns the bridge from verifier-replayed production exact facts to the
single authority certificate consumed by the terminal theorem stack.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorRawProduction

/--
Production exact runtime authority soundness.

This is the backend F' soundness obligation at the verifier boundary: once the
production verifier has replayed the compact image, Construction-2 boundary,
Poseidon2 transcript, canonical statement, and exact terminal/boundary public
IO, the fixed opener must expose folded F' authority for the same public pair.
-/
structure ProductionExactRuntimeAuthoritySoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx) where
  acceptedExactRuntimeOpens :
    ∀ steps proof image publicIO,
      ProductionCompactImageReplay
        (rawProductionVerifierChecksOfExact checks)
        steps
        proof
        image →
      ProductionConstruction2BoundaryReplay
        (rawProductionVerifierChecksOfExact checks)
        steps
        proof
        image →
      ProductionPoseidon2TranscriptReplay
        (rawProductionVerifierChecksOfExact checks)
        steps
        proof
        image →
      checks.proofIvcPublicImage proof =
        checks.canonicalIvcPublicImage steps image →
      checks.statementPublicValid
        (checks.canonicalIvcPublicImage steps image) →
      checks.construction2Boundary
          (checks.proofIvcPublicImage proof) =
        checks.construction2Boundary
          (checks.canonicalIvcPublicImage steps image) →
      checks.terminalVerifierPublicIO
          (checks.terminalCommittedProof proof) =
        some publicIO →
      publicIO.terminal =
        checks.terminalPublicValues
          (checks.canonicalIvcPublicImage steps image) →
      publicIO.boundary =
        checks.boundaryPublicValues
          (checks.construction2Boundary
            (checks.canonicalIvcPublicImage steps image)) →
        ∃ authority : ProofCarryingPriorProof ctx,
          checks.openAuthority proof = some authority ∧
            FoldedFPrimeAuthority.Accepts
              (Transition :=
                DirectParentOnlyProductionSoundness.Transition
                  ctx.toProductionContext)
              (initial := ctx.initial)
              steps
              authority
              image

/--
Runtime authority soundness instantiates the single production exact authority
certificate.
-/
def productionExactAuthorityCertificateOfRuntimeAuthoritySoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks) :
    ProductionExactAuthorityCertificate checks where
  acceptedOpens := by
    intro steps proof image publicIO hBound
    rcases hBound with
      ⟨hAccepted, hStatement⟩
    rcases hAccepted with
      ⟨hCompact,
        hBoundaryReplay,
        hTranscript,
        hValid,
        hBoundary,
        hPublicIO,
        hTerminal,
        hBoundaryValues⟩
    exact
      soundness.acceptedExactRuntimeOpens
        steps
        proof
        image
        publicIO
        hCompact
        hBoundaryReplay
        hTranscript
        hStatement
        hValid
        hBoundary
        hPublicIO
        hTerminal
        hBoundaryValues

/--
Production exact opening surface induced by runtime authority soundness.
-/
def productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks) :
    ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx :=
  productionExactPriorOpeningSurfaceOfAuthorityCertificate
    checks
    (productionExactAuthorityCertificateOfRuntimeAuthoritySoundness
      checks
      soundness)

/--
Production exact verification opens folded F' authority from the runtime
authority-soundness boundary.
-/
theorem runtimeVerifyPriorOfProductionExactRuntimeAuthoritySoundness_acceptedOpens
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
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness)
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
  simpa [productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness] using
    runtimeVerifyPriorOfProductionExactAuthorityCertificate_acceptedOpens
      checks
      (productionExactAuthorityCertificateOfRuntimeAuthoritySoundness
        checks
        soundness)
      hVerify

/--
Generic exact-runtime backend surface induced by the production exact verifier
facts and the production runtime authority-soundness boundary.
-/
def concreteRuntimeExactPublicIOSurfaceOfProductionExactRuntimeAuthoritySoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks) :
    DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
      (PriorProof := PriorProof)
      ctx where
  Statement := checks.IvcPublicImage
  PublicBoundary := checks.Construction2Boundary
  PublicField := checks.PublicField
  TerminalCommittedProof := checks.TerminalCommittedProof
  canonicalStatement := checks.canonicalIvcPublicImage
  proofStatement := checks.proofIvcPublicImage
  statementBoundary := checks.construction2Boundary
  proofBoundary := fun proof =>
    checks.construction2Boundary (checks.proofIvcPublicImage proof)
  terminalPublicValues := checks.terminalPublicValues
  boundaryPublicValues := checks.boundaryPublicValues
  terminalCommittedProof := checks.terminalCommittedProof
  statementPublicValid := checks.statementPublicValid
  terminalVerifierPublicIO := checks.terminalVerifierPublicIO
  compactImageReplay :=
    ProductionCompactImageReplay (rawProductionVerifierChecksOfExact checks)
  construction2BoundaryReplay :=
    ProductionConstruction2BoundaryReplay
      (rawProductionVerifierChecksOfExact checks)
  transcriptReplay :=
    ProductionPoseidon2TranscriptReplay
      (rawProductionVerifierChecksOfExact checks)
  openAuthority := checks.openAuthority
  replayBindsProofStatement := by
    intro steps proof image hCompact _hBoundaryReplay _hTranscript
    exact hCompact.1
  exactRuntimeSound := by
    intro steps proof image publicIO
      hCompact hBoundaryReplay hTranscript hStatement hValid hBoundary
      hPublicIO hTerminal hBoundaryValues
    exact
      soundness.acceptedExactRuntimeOpens
        steps
        proof
        image
        publicIO
        hCompact
        hBoundaryReplay
        hTranscript
        hStatement
        hValid
        hBoundary
        hPublicIO
        hTerminal
        hBoundaryValues

/--
Production exact verification is the generic exact-runtime verifier for the
backend surface induced by runtime authority soundness.
-/
theorem runtimeVerifyPriorOfProductionExactRuntimeAuthoritySoundness_toConcreteRuntimeExactPublicIO
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
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness)
        steps
        proof
        image) :
    DirectParentOnlyProductionConcreteFPrimePriorBackend.RuntimeVerifyPriorOfExactPublicIO
      (concreteRuntimeExactPublicIOSurfaceOfProductionExactRuntimeAuthoritySoundness
        checks
        soundness)
      steps
      proof
      image := by
  rcases hVerify with
    ⟨publicIO, hAccepted⟩
  rcases hAccepted with
    ⟨hCompact,
      hBoundaryReplay,
      hTranscript,
      hValid,
      hBoundary,
      hPublicIO,
      hTerminal,
      hBoundaryValues⟩
  exact
    ⟨hCompact,
      hBoundaryReplay,
      hTranscript,
      hValid,
      hBoundary,
      ⟨publicIO, hPublicIO, hTerminal, hBoundaryValues⟩⟩

/--
Accepted production exact verification opens folded F' authority through the
generic exact-runtime backend surface induced by runtime authority soundness.
-/
theorem runtimeVerifyPriorOfProductionExactRuntimeAuthoritySoundness_concreteRuntimeExactPublicIOEvidence
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
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness)
        steps
        proof
        image) :
    DirectParentOnlyProductionConcreteFPrimePriorBackend.AcceptedExactPublicIOEvidence
      (concreteRuntimeExactPublicIOSurfaceOfProductionExactRuntimeAuthoritySoundness
        checks
        soundness)
      steps
      proof
      image :=
  DirectParentOnlyProductionConcreteFPrimePriorBackend.runtimeVerifyPriorOfExactPublicIO_evidence
    (concreteRuntimeExactPublicIOSurfaceOfProductionExactRuntimeAuthoritySoundness
      checks
      soundness)
    (runtimeVerifyPriorOfProductionExactRuntimeAuthoritySoundness_toConcreteRuntimeExactPublicIO
      checks
      soundness
      hVerify)

/--
Certified prior verifier induced directly by production exact runtime authority
soundness through the generic exact-runtime backend surface.
-/
def certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact
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
  DirectParentOnlyProductionConcreteFPrimePriorBackend.certifiedPriorVerifierOfExactPublicIO
    (concreteRuntimeExactPublicIOSurfaceOfProductionExactRuntimeAuthoritySoundness
      checks
      soundness)

/--
Production exact verification is accepted by the concrete-exact certified
prior verifier induced by runtime authority soundness.
-/
theorem certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact_accepts
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
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness)
        steps
        proof
        image) :
    (certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact
        checks
        soundness).verify
      steps
      proof
      image := by
  simpa [
    certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact]
    using
      runtimeVerifyPriorOfProductionExactRuntimeAuthoritySoundness_toConcreteRuntimeExactPublicIO
        checks
        soundness
        hVerify

/--
Strict `SoundVerifier` induced directly by production exact runtime authority
soundness through the concrete-exact certified verifier.
-/
def soundVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact
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
    (certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact
      checks
      soundness)

/--
The concrete-exact runtime-soundness `SoundVerifier` accepts exactly the
generic exact-runtime backend predicate induced by the production checks.
-/
theorem soundVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact_accepts_iff
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary} :
    CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact
          checks
          soundness)
        steps
        proof
        image <->
      DirectParentOnlyProductionConcreteFPrimePriorBackend.RuntimeVerifyPriorOfExactPublicIO
        (concreteRuntimeExactPublicIOSurfaceOfProductionExactRuntimeAuthoritySoundness
          checks
          soundness)
        steps
        proof
        image := by
  simpa [
    soundVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact,
    certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact]
    using
      DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifier_accepts_iff
        (certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact
          checks
          soundness)

/--
Terminal acceptance from production exact checks passes through the
concrete-exact runtime-soundness strict `SoundVerifier`.
-/
theorem acceptedTerminalWithSoundVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExactLatestStep
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
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness)
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
      (soundVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact
        checks
        soundness)
      priorSteps
      priorProof
      priorImage
      nextImage
      latestProof := by
  refine
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.acceptedTerminalWithSoundVerifier
      (certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact
        checks
        soundness)
      ?_
  exact {
    priorAccepted :=
      certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact_accepts
        checks
        soundness
        hPrior
    latestAccepted := hLatest
  }

/--
Production exact prior-plus-latest end-to-end theorem through the
concrete-exact runtime authority-soundness backend surface.
-/
theorem certifiedSingleTerminalEndToEnd_ofProductionExactRuntimeAuthoritySoundnessConcreteExactLatestStep
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
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness)
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
      (certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact
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
    certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact]
    using
      DirectParentOnlyProductionConcreteFPrimePriorBackend.certifiedSingleTerminalEndToEnd_ofExactPublicIOLatestStep
        (concreteRuntimeExactPublicIOSurfaceOfProductionExactRuntimeAuthoritySoundness
          checks
          soundness)
        (runtimeVerifyPriorOfProductionExactRuntimeAuthoritySoundness_toConcreteRuntimeExactPublicIO
          checks
          soundness
          hPrior)
        hLatest

/--
Concrete-exact runtime-soundness projection to non-aggregate private DEC and
stage facts.
-/
theorem nonAggregatePrivateDecStageFacts_ofProductionExactRuntimeAuthoritySoundnessConcreteExactLatestStep
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
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness)
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
  DirectParentOnlyProductionConcreteFPrimePriorBackend.nonAggregatePrivateDecStageFacts_ofExactPublicIOLatestStep
    (concreteRuntimeExactPublicIOSurfaceOfProductionExactRuntimeAuthoritySoundness
      checks
      soundness)
    (runtimeVerifyPriorOfProductionExactRuntimeAuthoritySoundness_toConcreteRuntimeExactPublicIO
      checks
      soundness
      hPrior)
    hLatest

/--
Concrete-exact runtime-soundness projection to the Section 7.1 owner-target
stage audit.
-/
theorem section71StageTargetAuditTrail_ofProductionExactRuntimeAuthoritySoundnessConcreteExactLatestStep
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
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness)
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
  DirectParentOnlyProductionConcreteFPrimePriorBackend.section71StageTargetAuditTrail_ofExactPublicIOLatestStep
    (concreteRuntimeExactPublicIOSurfaceOfProductionExactRuntimeAuthoritySoundness
      checks
      soundness)
    (runtimeVerifyPriorOfProductionExactRuntimeAuthoritySoundness_toConcreteRuntimeExactPublicIO
      checks
      soundness
      hPrior)
    hLatest

/--
Runtime exact public-IO opening surface induced by production exact checks,
runtime authority soundness, and canonical terminal-slice binding.
-/
def runtimeExactPublicIOOpeningSurfaceOfProductionExactRuntimeAuthoritySoundnessTerminalSlice
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    (sliceBinding :
      ProductionExactTerminalSliceBinding
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness)) :
    DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.ConcreteRuntimeExactPublicIOOpeningSurface
      (PriorProof := PriorProof)
      ctx :=
  runtimeExactPublicIOOpeningSurfaceOfProductionExactTerminalSlice
    (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
      checks
      soundness)
    sliceBinding

/--
Certified prior verifier induced by production exact runtime authority
soundness and canonical terminal-slice binding.
-/
def certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessTerminalSlice
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    (sliceBinding :
      ProductionExactTerminalSliceBinding
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness)) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  certifiedPriorVerifierOfProductionExactTerminalSlice
    (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
      checks
      soundness)
    sliceBinding

/--
Production exact verification is accepted by the runtime-soundness
terminal-slice certified verifier.
-/
theorem certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessTerminalSlice_accepts
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    (sliceBinding :
      ProductionExactTerminalSliceBinding
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness))
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness)
        steps
        proof
        image) :
    (certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessTerminalSlice
        checks
        soundness
        sliceBinding).verify
      steps
      proof
      image := by
  simpa [
    certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessTerminalSlice]
    using
      certifiedPriorVerifierOfProductionExactTerminalSlice_accepts
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness)
        sliceBinding
        hVerify

/--
Strict `SoundVerifier` induced by production exact runtime authority soundness.
-/
def soundVerifierOfProductionExactRuntimeAuthoritySoundnessTerminalSlice
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    (sliceBinding :
      ProductionExactTerminalSliceBinding
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness)) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.SoundPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifier
    (certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessTerminalSlice
      checks
      soundness
      sliceBinding)

/--
The runtime-soundness terminal-slice `SoundVerifier` accepts exactly the
induced backend-shaped exact public-IO opening predicate.
-/
theorem soundVerifierOfProductionExactRuntimeAuthoritySoundnessTerminalSlice_accepts_iff
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    (sliceBinding :
      ProductionExactTerminalSliceBinding
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness))
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary} :
    CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionExactRuntimeAuthoritySoundnessTerminalSlice
          checks
          soundness
          sliceBinding)
        steps
        proof
        image <->
      DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.RuntimeVerifyPriorOfRuntimeExactPublicIOOpening
        (runtimeExactPublicIOOpeningSurfaceOfProductionExactRuntimeAuthoritySoundnessTerminalSlice
          checks
          soundness
          sliceBinding)
        steps
        proof
        image := by
  simpa [
    soundVerifierOfProductionExactRuntimeAuthoritySoundnessTerminalSlice,
    certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessTerminalSlice,
    runtimeExactPublicIOOpeningSurfaceOfProductionExactRuntimeAuthoritySoundnessTerminalSlice]
    using
      soundVerifierOfProductionExactTerminalSlice_accepts_iff
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness)
        sliceBinding

/--
Terminal acceptance from production exact checks passes through the
runtime-soundness strict `SoundVerifier`.
-/
theorem acceptedTerminalWithSoundVerifierOfProductionExactRuntimeAuthoritySoundnessTerminalSliceLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    (sliceBinding :
      ProductionExactTerminalSliceBinding
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness))
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness)
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
      (soundVerifierOfProductionExactRuntimeAuthoritySoundnessTerminalSlice
        checks
        soundness
        sliceBinding)
      priorSteps
      priorProof
      priorImage
      nextImage
      latestProof := by
  simpa [
    soundVerifierOfProductionExactRuntimeAuthoritySoundnessTerminalSlice,
    certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessTerminalSlice]
    using
      acceptedTerminalWithSoundVerifierOfProductionExactTerminalSliceLatestStep
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness)
        sliceBinding
        hPrior
        hLatest

/--
Production exact prior-plus-latest end-to-end theorem from runtime authority
soundness and canonical terminal-slice binding.
-/
theorem certifiedSingleTerminalEndToEnd_ofProductionExactRuntimeAuthoritySoundnessTerminalSliceLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    (sliceBinding :
      ProductionExactTerminalSliceBinding
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness))
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness)
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
      (certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessTerminalSlice
        checks
        soundness
        sliceBinding).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage := by
  simpa [
    certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessTerminalSlice]
    using
      certifiedSingleTerminalEndToEnd_ofProductionExactTerminalSliceLatestStep
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness)
        sliceBinding
        hPrior
        hLatest

/--
Runtime-soundness production exact projection to non-aggregate private DEC and
stage facts.
-/
theorem nonAggregatePrivateDecStageFacts_ofProductionExactRuntimeAuthoritySoundnessTerminalSliceLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    (sliceBinding :
      ProductionExactTerminalSliceBinding
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness))
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness)
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
  nonAggregatePrivateDecStageFacts_ofProductionExactTerminalSliceLatestStep
    (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
      checks
      soundness)
    sliceBinding
    hPrior
    hLatest

/--
Runtime-soundness production exact projection to the Section 7.1 owner-target
stage audit.
-/
theorem section71StageTargetAuditTrail_ofProductionExactRuntimeAuthoritySoundnessTerminalSliceLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactRuntimeAuthoritySoundness checks)
    (sliceBinding :
      ProductionExactTerminalSliceBinding
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness))
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
          checks
          soundness)
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
  section71StageTargetAuditTrail_ofProductionExactTerminalSliceLatestStep
    (productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
      checks
      soundness)
    sliceBinding
    hPrior
    hLatest

end DirectParentOnlyProductionConcreteFPrimePriorRawProduction

end DirectCcsFPrime
