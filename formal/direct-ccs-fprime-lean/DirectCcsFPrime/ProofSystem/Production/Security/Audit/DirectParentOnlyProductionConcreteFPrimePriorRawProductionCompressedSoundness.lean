import DirectCcsFPrime.ProofSystem.Production.Impl.RawProduction.DirectParentOnlyProductionConcreteFPrimePriorRawProductionRuntimeAuthority

/-!
Compressed-verifier soundness for the production exact F' prior path.

This file owns the implementation-facing F' authority boundary: accepted
production exact verifier evidence must open the fixed authority object to
folded F' reachability for the same `(steps, image)` pair.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorRawProduction

/--
Production exact compressed-verifier soundness.

This is the concrete F' backend boundary over the flat verifier audit:
once the verifier has established the exact public statement, transcript,
terminal public IO, and final-claim checks, the fixed opener exposes folded
F' authority for that same pair.
-/
structure ProductionExactCompressedVerifierSoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx) where
  acceptedAuditOpens :
    ∀ steps proof image publicIO,
      ProductionExactVerifierAcceptedAudit checks steps proof image publicIO →
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
Accepted production exact verifier audit opens folded F' authority for the same
public pair.
-/
theorem productionExactCompressedVerifierSoundness_acceptedAuditOpens
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
  soundness.acceptedAuditOpens steps proof image publicIO hAudit

/--
Accepted production exact verifier evidence opens folded F' authority for the
same public pair.
-/
theorem productionExactCompressedVerifierSoundness_acceptedVerifierOpens
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
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
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
  soundness.acceptedAuditOpens
    steps
    proof
    image
    publicIO
    (productionExactVerifierAccepted_audit hAccepted)

/--
Accepted production exact verifier evidence opens folded F' authority through a
production exact opening surface.
-/
theorem productionExactVerifierAcceptedOfOpeningSurface_acceptedOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO surface.checks.PublicField}
    (hAccepted :
      ProductionExactVerifierAccepted
        surface.checks
        steps
        proof
        image
        publicIO) :
    ∃ authority : ProofCarryingPriorProof ctx,
      surface.checks.openAuthority proof = some authority ∧
        FoldedFPrimeAuthority.Accepts
          (Transition :=
            DirectParentOnlyProductionSoundness.Transition
              ctx.toProductionContext)
          (initial := ctx.initial)
          steps
          authority
          image :=
  runtimeVerifyPriorOfProductionExact_acceptedOpens
    surface
    steps
    proof
    image
    ⟨publicIO, hAccepted⟩

/--
An existing production exact opening surface induces compressed-verifier
soundness over its verifier checks.
-/
def productionExactCompressedVerifierSoundnessOfOpeningSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx) :
    ProductionExactCompressedVerifierSoundness surface.checks where
  acceptedAuditOpens := by
    intro steps proof image publicIO hAudit
    exact
      productionExactVerifierAcceptedOfOpeningSurface_acceptedOpens
        surface
        (productionExactVerifierAccepted_of_audit hAudit)

/--
A single production exact authority certificate induces compressed-verifier
soundness over the same verifier checks.
-/
def productionExactCompressedVerifierSoundnessOfAuthorityCertificate
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (certificate : ProductionExactAuthorityCertificate checks) :
    ProductionExactCompressedVerifierSoundness checks :=
  productionExactCompressedVerifierSoundnessOfOpeningSurface
    (productionExactPriorOpeningSurfaceOfAuthorityCertificate
      checks
      certificate)

/--
Compressed-verifier soundness induces the single production exact authority
certificate consumed by the older production exact terminal surfaces.
-/
def productionExactAuthorityCertificateOfCompressedVerifierSoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactCompressedVerifierSoundness checks) :
    ProductionExactAuthorityCertificate checks where
  acceptedOpens := by
    intro steps proof image publicIO hBound
    exact
      soundness.acceptedAuditOpens
        steps
        proof
        image
        publicIO
        (productionExactVerifierAccepted_audit hBound.1)

/--
Compressed-verifier soundness and the single authority certificate expose the
same direct accepted-verifier opening consequence.
-/
theorem productionExactCompressedVerifierSoundnessOfAuthorityCertificate_acceptedVerifierOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (certificate : ProductionExactAuthorityCertificate checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO checks.PublicField}
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
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
  productionExactCompressedVerifierSoundness_acceptedVerifierOpens
    checks
    (productionExactCompressedVerifierSoundnessOfAuthorityCertificate
      checks
      certificate)
    hAccepted

/--
An authority certificate also opens folded F' authority from the flat verifier
audit.
-/
theorem productionExactCompressedVerifierSoundnessOfAuthorityCertificate_acceptedAuditOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (certificate : ProductionExactAuthorityCertificate checks)
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
  (productionExactCompressedVerifierSoundnessOfAuthorityCertificate
      checks
      certificate).acceptedAuditOpens
    steps
    proof
    image
    publicIO
    hAudit

/--
Compressed-verifier soundness instantiates the runtime authority-soundness
surface.
-/
def productionExactRuntimeAuthoritySoundnessOfCompressedVerifierSoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactCompressedVerifierSoundness checks) :
    ProductionExactRuntimeAuthoritySoundness checks where
  acceptedExactRuntimeOpens := by
    intro steps proof image publicIO
      hCompact hBoundaryReplay hTranscript _hStatement hValid hBoundary
      hPublicIO hTerminal hBoundaryValues
    exact
      soundness.acceptedAuditOpens
        steps
        proof
        image
        publicIO
        (productionExactVerifierAccepted_audit
          ⟨hCompact,
            hBoundaryReplay,
            hTranscript,
            hValid,
            hBoundary,
            hPublicIO,
            hTerminal,
            hBoundaryValues⟩)

/--
Production exact opening surface induced by compressed-verifier soundness.
-/
def productionExactPriorOpeningSurfaceOfCompressedVerifierSoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactCompressedVerifierSoundness checks) :
    ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx :=
  productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
    checks
    (productionExactRuntimeAuthoritySoundnessOfCompressedVerifierSoundness
      checks
      soundness)

/--
Accepted production exact verification opens folded F' authority directly from
compressed-verifier soundness.
-/
theorem runtimeVerifyPriorOfProductionExactCompressedVerifierSoundness_acceptedOpens
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
    (hVerify :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfCompressedVerifierSoundness
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
  rcases hVerify with
    ⟨publicIO, hAccepted⟩
  exact
    soundness.acceptedAuditOpens
      steps
      proof
      image
      publicIO
      (productionExactVerifierAccepted_audit hAccepted)

/--
Generic exact-runtime backend surface induced by production exact verifier
checks and compressed-verifier soundness.
-/
def concreteRuntimeExactPublicIOSurfaceOfProductionExactCompressedVerifierSoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactCompressedVerifierSoundness checks) :
    DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
      (PriorProof := PriorProof)
      ctx :=
  concreteRuntimeExactPublicIOSurfaceOfProductionExactRuntimeAuthoritySoundness
    checks
    (productionExactRuntimeAuthoritySoundnessOfCompressedVerifierSoundness
      checks
      soundness)

/--
Production exact verification is the induced generic exact-runtime backend
predicate under compressed-verifier soundness.
-/
theorem runtimeVerifyPriorOfProductionExactCompressedVerifierSoundness_toConcreteRuntimeExactPublicIO
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
    (hVerify :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfCompressedVerifierSoundness
          checks
          soundness)
        steps
        proof
        image) :
    DirectParentOnlyProductionConcreteFPrimePriorBackend.RuntimeVerifyPriorOfExactPublicIO
      (concreteRuntimeExactPublicIOSurfaceOfProductionExactCompressedVerifierSoundness
        checks
        soundness)
      steps
      proof
      image := by
  simpa [
    productionExactPriorOpeningSurfaceOfCompressedVerifierSoundness,
    concreteRuntimeExactPublicIOSurfaceOfProductionExactCompressedVerifierSoundness]
    using
      runtimeVerifyPriorOfProductionExactRuntimeAuthoritySoundness_toConcreteRuntimeExactPublicIO
        checks
        (productionExactRuntimeAuthoritySoundnessOfCompressedVerifierSoundness
          checks
          soundness)
        hVerify

/--
Accepted production exact verification exposes generic exact-runtime evidence
under compressed-verifier soundness.
-/
theorem runtimeVerifyPriorOfProductionExactCompressedVerifierSoundness_concreteRuntimeExactPublicIOEvidence
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
    (hVerify :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfCompressedVerifierSoundness
          checks
          soundness)
        steps
        proof
        image) :
    DirectParentOnlyProductionConcreteFPrimePriorBackend.AcceptedExactPublicIOEvidence
      (concreteRuntimeExactPublicIOSurfaceOfProductionExactCompressedVerifierSoundness
        checks
        soundness)
      steps
      proof
      image := by
  simpa [
    concreteRuntimeExactPublicIOSurfaceOfProductionExactCompressedVerifierSoundness]
    using
      runtimeVerifyPriorOfProductionExactRuntimeAuthoritySoundness_concreteRuntimeExactPublicIOEvidence
        checks
        (productionExactRuntimeAuthoritySoundnessOfCompressedVerifierSoundness
          checks
          soundness)
        (by
          simpa [productionExactPriorOpeningSurfaceOfCompressedVerifierSoundness]
            using hVerify)

/--
Certified prior verifier induced directly by compressed-verifier soundness.
-/
def certifiedPriorVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactCompressedVerifierSoundness checks) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact
    checks
    (productionExactRuntimeAuthoritySoundnessOfCompressedVerifierSoundness
      checks
      soundness)

/--
Production exact verification is accepted by the compressed-soundness certified
prior verifier.
-/
theorem certifiedPriorVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact_accepts
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
    (hVerify :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfCompressedVerifierSoundness
          checks
          soundness)
        steps
        proof
        image) :
    (certifiedPriorVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact
        checks
        soundness).verify
      steps
      proof
      image := by
  simpa [
    certifiedPriorVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact,
    productionExactPriorOpeningSurfaceOfCompressedVerifierSoundness]
    using
      certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact_accepts
        checks
        (productionExactRuntimeAuthoritySoundnessOfCompressedVerifierSoundness
          checks
          soundness)
        hVerify

/--
Accepted production exact verifier evidence is accepted directly by the
compressed-soundness certified prior verifier.
-/
theorem certifiedPriorVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact_acceptsVerifierAccepted
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
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    (certifiedPriorVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact
        checks
        soundness).verify
      steps
      proof
      image :=
  certifiedPriorVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact_accepts
    checks
    soundness
    ⟨publicIO, hAccepted⟩

/--
Strict `SoundVerifier` induced directly by compressed-verifier soundness.
-/
def soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactCompressedVerifierSoundness checks) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.SoundPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifier
    (certifiedPriorVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact
      checks
      soundness)

/--
The compressed-soundness `SoundVerifier` accepts exactly the induced generic
exact-runtime backend predicate.
-/
theorem soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact_accepts_iff
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactCompressedVerifierSoundness checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary} :
    CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact
          checks
          soundness)
        steps
        proof
        image <->
      DirectParentOnlyProductionConcreteFPrimePriorBackend.RuntimeVerifyPriorOfExactPublicIO
        (concreteRuntimeExactPublicIOSurfaceOfProductionExactCompressedVerifierSoundness
          checks
          soundness)
        steps
        proof
        image := by
  simpa [
    soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact,
    certifiedPriorVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact,
    concreteRuntimeExactPublicIOSurfaceOfProductionExactCompressedVerifierSoundness]
    using
      soundVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact_accepts_iff
        checks
        (productionExactRuntimeAuthoritySoundnessOfCompressedVerifierSoundness
          checks
          soundness)

/--
Accepted production exact verifier evidence is accepted directly by the
compressed-soundness strict `SoundVerifier`.
-/
theorem soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact_acceptsVerifierAccepted
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
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    CompressedFPrimeAuthority.SoundVerifier.Accepts
      (soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact
        checks
        soundness)
      steps
      proof
      image := by
  exact
    (soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact_accepts_iff
        checks
        soundness).2
      (runtimeVerifyPriorOfProductionExactCompressedVerifierSoundness_toConcreteRuntimeExactPublicIO
        checks
        soundness
        ⟨publicIO, hAccepted⟩)

/--
The compressed-soundness `SoundVerifier` opens folded F' authority for accepted
proofs.
-/
theorem soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact_opensToFoldedAuthority
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
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact
          checks
          soundness)
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
  (soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact
      checks
      soundness).opensToFoldedAuthority
    steps
    proof
    image
    hVerify

/-- The compressed-soundness `SoundVerifier` is same-proof functional. -/
theorem soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact_proofFunctional
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : ProductionExactCompressedVerifierSoundness checks) :
    CompressedFPrimeAuthority.SoundVerifier.ProofFunctional
      (soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact
        checks
        soundness) :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifierProofFunctional
    (certifiedPriorVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact
      checks
      soundness)

/-- One compressed proof cannot verify for two different public pairs. -/
theorem soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact_sameProof
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {soundness : ProductionExactCompressedVerifierSoundness checks}
    {stepsA stepsB : Nat}
    {proof : PriorProof}
    {imageA imageB : PublicImage Digest Boundary}
    (hA :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact
          checks
          soundness)
        stepsA
        proof
        imageA)
    (hB :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact
          checks
          soundness)
        stepsB
        proof
        imageB) :
    stepsA = stepsB ∧ imageA = imageB :=
  soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact_proofFunctional
    checks
    soundness
    hA
    hB

/--
Accepted production exact verifier evidence cannot fail to open through the
fixed authority opener.
-/
theorem productionExactCompressedVerifierSoundness_openAuthority_ne_none
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
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    checks.openAuthority proof ≠ none := by
  rcases
    soundness.acceptedAuditOpens
      steps
      proof
      image
      publicIO
      (productionExactVerifierAccepted_audit hAccepted) with
    ⟨authority, hOpen, _hAccepts⟩
  intro hNone
  rw [hNone] at hOpen
  cases hOpen

/--
Any authority returned by the fixed opener for accepted production verifier
evidence accepts the same `(steps, image)` pair.
-/
theorem productionExactCompressedVerifierSoundness_openedAuthority_accepts_ofAccepted
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
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO)
    (hOpen : checks.openAuthority proof = some authority) :
    FoldedFPrimeAuthority.Accepts
      (Transition :=
        DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
      (initial := ctx.initial)
      steps
      authority
      image := by
  rcases
    soundness.acceptedAuditOpens
      steps
      proof
      image
      publicIO
      (productionExactVerifierAccepted_audit hAccepted) with
    ⟨openedAuthority, hOpened, hAccepts⟩
  have hSame : some authority = some openedAuthority := by
    rw [← hOpen, hOpened]
  cases hSame
  exact hAccepts

/-- Accepted production verifier evidence binds the opened authority step. -/
theorem productionExactCompressedVerifierSoundness_bindsOpenedAuthoritySteps
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
    {authority : ProofCarryingPriorProof ctx}
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO)
    (hOpen : checks.openAuthority proof = some authority) :
    authority.steps = steps :=
  (productionExactCompressedVerifierSoundness_openedAuthority_accepts_ofAccepted
    checks
    soundness
    hAccepted
    hOpen).1

/-- Accepted production verifier evidence binds the opened authority image. -/
theorem productionExactCompressedVerifierSoundness_bindsOpenedAuthorityImage
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
    {authority : ProofCarryingPriorProof ctx}
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO)
    (hOpen : checks.openAuthority proof = some authority) :
    authority.image = image :=
  (productionExactCompressedVerifierSoundness_openedAuthority_accepts_ofAccepted
    checks
    soundness
    hAccepted
    hOpen).2

/-- Accepted production verifier evidence cannot coexist with no opening. -/
theorem productionExactCompressedVerifierSoundness_cannot_accept_without_opening
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
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO)
    (hNone : checks.openAuthority proof = none) :
    False :=
  productionExactCompressedVerifierSoundness_openAuthority_ne_none
    checks
    soundness
    hAccepted
    hNone

/-- Accepted production verifier evidence reaches the claimed prior image. -/
theorem productionExactCompressedVerifierSoundness_reaches_prior
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
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image := by
  rcases
    soundness.acceptedAuditOpens
      steps
      proof
      image
      publicIO
      (productionExactVerifierAccepted_audit hAccepted) with
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

/-- Accepted production verifier evidence exposes prior public-image invariants. -/
theorem productionExactCompressedVerifierSoundness_publicImageInvariants
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
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO) :
    DirectParentOnlyProductionConcreteFPrimePrior.AcceptedPriorPublicImageInvariants
      ctx
      steps
      image := by
  have hReach :=
    productionExactCompressedVerifierSoundness_reaches_prior
      (soundness := soundness)
      hAccepted
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

/-- Accepted production verifier evidence cannot authorize an unreachable prior. -/
theorem productionExactCompressedVerifierSoundness_cannot_accept_unreachable_prior
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
    (hAccepted :
      ProductionExactVerifierAccepted checks steps proof image publicIO)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable
    (productionExactCompressedVerifierSoundness_reaches_prior
      (soundness := soundness)
      hAccepted)

/--
Terminal acceptance from production exact checks passes through the strict
`SoundVerifier` induced by compressed-verifier soundness.
-/
theorem acceptedTerminalWithSoundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExactLatestStep
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
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfCompressedVerifierSoundness
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
      (soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact
        checks
        soundness)
      priorSteps
      priorProof
      priorImage
      nextImage
      latestProof := by
  simpa [
    productionExactPriorOpeningSurfaceOfCompressedVerifierSoundness,
    soundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact,
    certifiedPriorVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact]
    using
      acceptedTerminalWithSoundVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExactLatestStep
        checks
        (productionExactRuntimeAuthoritySoundnessOfCompressedVerifierSoundness
          checks
          soundness)
        hPrior
        hLatest

/--
Terminal acceptance from direct production exact verifier evidence passes
through the strict `SoundVerifier` induced by compressed-verifier soundness.
-/
theorem acceptedTerminalWithSoundVerifierOfProductionExactVerifierAcceptedCompressedVerifierSoundnessLatestStep
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
      ProductionExactVerifierAccepted
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
  acceptedTerminalWithSoundVerifierOfProductionExactCompressedVerifierSoundnessConcreteExactLatestStep
    checks
    soundness
    ⟨publicIO, hPrior⟩
    hLatest

/--
Production exact prior-plus-latest end-to-end theorem from compressed-verifier
soundness.
-/
theorem certifiedSingleTerminalEndToEnd_ofProductionExactCompressedVerifierSoundnessConcreteExactLatestStep
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
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfCompressedVerifierSoundness
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
      (certifiedPriorVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact
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
    productionExactPriorOpeningSurfaceOfCompressedVerifierSoundness,
    certifiedPriorVerifierOfProductionExactCompressedVerifierSoundnessConcreteExact]
    using
      certifiedSingleTerminalEndToEnd_ofProductionExactRuntimeAuthoritySoundnessConcreteExactLatestStep
        checks
        (productionExactRuntimeAuthoritySoundnessOfCompressedVerifierSoundness
          checks
          soundness)
        hPrior
        hLatest

/--
Direct production exact verifier evidence plus the latest Construction-2 step
returns the terminal end-to-end package.
-/
theorem certifiedSingleTerminalEndToEnd_ofProductionExactVerifierAcceptedCompressedVerifierSoundnessLatestStep
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
      ProductionExactVerifierAccepted
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
  certifiedSingleTerminalEndToEnd_ofProductionExactCompressedVerifierSoundnessConcreteExactLatestStep
    checks
    soundness
    ⟨publicIO, hPrior⟩
    hLatest

/--
Compressed-soundness projection to non-aggregate private DEC and stage facts.
-/
theorem nonAggregatePrivateDecStageFacts_ofProductionExactCompressedVerifierSoundnessConcreteExactLatestStep
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
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfCompressedVerifierSoundness
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
      nextImage := by
  simpa [productionExactPriorOpeningSurfaceOfCompressedVerifierSoundness]
    using
      nonAggregatePrivateDecStageFacts_ofProductionExactRuntimeAuthoritySoundnessConcreteExactLatestStep
        checks
        (productionExactRuntimeAuthoritySoundnessOfCompressedVerifierSoundness
          checks
          soundness)
        hPrior
        hLatest

/--
Direct production exact verifier evidence projection to non-aggregate private
DEC and stage facts.
-/
theorem nonAggregatePrivateDecStageFacts_ofProductionExactVerifierAcceptedCompressedVerifierSoundnessLatestStep
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
      ProductionExactVerifierAccepted
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
  nonAggregatePrivateDecStageFacts_ofProductionExactCompressedVerifierSoundnessConcreteExactLatestStep
    checks
    soundness
    ⟨publicIO, hPrior⟩
    hLatest

/--
Compressed-soundness projection to the Section 7.1 owner-target stage audit.
-/
theorem section71StageTargetAuditTrail_ofProductionExactCompressedVerifierSoundnessConcreteExactLatestStep
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
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfCompressedVerifierSoundness
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
      nextImage := by
  simpa [productionExactPriorOpeningSurfaceOfCompressedVerifierSoundness]
    using
      section71StageTargetAuditTrail_ofProductionExactRuntimeAuthoritySoundnessConcreteExactLatestStep
        checks
        (productionExactRuntimeAuthoritySoundnessOfCompressedVerifierSoundness
          checks
          soundness)
        hPrior
        hLatest

/--
Direct production exact verifier evidence projection to the Section 7.1
owner-target stage audit.
-/
theorem section71StageTargetAuditTrail_ofProductionExactVerifierAcceptedCompressedVerifierSoundnessLatestStep
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
      ProductionExactVerifierAccepted
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
  section71StageTargetAuditTrail_ofProductionExactCompressedVerifierSoundnessConcreteExactLatestStep
    checks
    soundness
    ⟨publicIO, hPrior⟩
    hLatest

end DirectParentOnlyProductionConcreteFPrimePriorRawProduction

end DirectCcsFPrime
