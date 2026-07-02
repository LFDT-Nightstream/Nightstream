import DirectCcsFPrime.ProofSystem.Production.Impl.RawProduction.DirectParentOnlyProductionConcreteFPrimePriorRawProductionRuntimeAuthority
import DirectCcsFPrime.Audit.RedTeam.DirectParentOnlyProductionPrivateDecNoSwapAudit

/-!
Production exact-runtime instantiation for the parent-only terminal theorem.

Spec: `specs/Production/Impl/Runtime/DirectParentOnlyProductionExactRuntimeInstantiation.spec.md`

This file owns the short production entry points for the concrete exact-runtime
F' prior verifier path. It packages production verifier checks plus the trusted
runtime authority-soundness boundary into the certified prior verifier consumed
by the parent-only terminal theorem.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionExactRuntimeInstantiation

open DirectParentOnlyProductionConcreteFPrimePriorRawProduction

/-- Production exact verifier checks used by this instantiation. -/
abbrev ExactChecks
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) :=
  ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx

/--
Trusted runtime authority-soundness boundary for the production exact verifier.

This is the only backend assumption in this file: accepted exact-runtime
verifier evidence must open folded F' reachability authority for the same
`(steps, image)` pair.
-/
abbrev RuntimeAuthoritySoundness :=
  @ProductionExactRuntimeAuthoritySoundness

/-- Production exact prior-verifier acceptance predicate. -/
abbrev VerifyPrior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx) :=
  RuntimeVerifyPriorOfProductionExact surface

/-- Opening surface induced by production exact checks and runtime soundness. -/
def openingSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ExactChecks (PriorProof := PriorProof) ctx)
    (soundness : RuntimeAuthoritySoundness checks) :
    ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx :=
  productionExactPriorOpeningSurfaceOfRuntimeAuthoritySoundness
    checks
    soundness

/-- Generic exact-runtime surface induced by production exact checks. -/
def runtimeSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ExactChecks (PriorProof := PriorProof) ctx)
    (soundness : RuntimeAuthoritySoundness checks) :
    DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
      (PriorProof := PriorProof)
      ctx :=
  concreteRuntimeExactPublicIOSurfaceOfProductionExactRuntimeAuthoritySoundness
    checks
    soundness

/-- Certified prior verifier induced by production exact checks. -/
def certifiedPriorVerifier
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ExactChecks (PriorProof := PriorProof) ctx)
    (soundness : RuntimeAuthoritySoundness checks) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact
    checks
    soundness

/-- Strict sound prior verifier induced by production exact checks. -/
def soundPriorVerifier
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ExactChecks (PriorProof := PriorProof) ctx)
    (soundness : RuntimeAuthoritySoundness checks) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.SoundPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  soundVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact
    checks
    soundness

/--
Accepted production exact prior verification opens folded F' authority for the
same public pair.
-/
theorem verifyPriorOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ExactChecks (PriorProof := PriorProof) ctx)
    (soundness : RuntimeAuthoritySoundness checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      VerifyPrior (openingSurface checks soundness) steps proof image) :
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
  simpa [openingSurface] using
    runtimeVerifyPriorOfProductionExactRuntimeAuthoritySoundness_acceptedOpens
      checks
      soundness
      hVerify

/--
Accepted production exact prior verification is the generic exact-runtime
predicate for the induced runtime surface.
-/
theorem verifyPriorAsRuntimeSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ExactChecks (PriorProof := PriorProof) ctx)
    (soundness : RuntimeAuthoritySoundness checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      VerifyPrior (openingSurface checks soundness) steps proof image) :
    DirectParentOnlyProductionConcreteFPrimePriorBackend.RuntimeVerifyPriorOfExactPublicIO
      (runtimeSurface checks soundness)
      steps
      proof
      image := by
  simpa [openingSurface, runtimeSurface] using
    runtimeVerifyPriorOfProductionExactRuntimeAuthoritySoundness_toConcreteRuntimeExactPublicIO
      checks
      soundness
      hVerify

/-- Accepted production exact prior verification is accepted by the certified verifier. -/
theorem certifiedPriorVerifierAccepts
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ExactChecks (PriorProof := PriorProof) ctx)
    (soundness : RuntimeAuthoritySoundness checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      VerifyPrior (openingSurface checks soundness) steps proof image) :
    (certifiedPriorVerifier checks soundness).verify steps proof image := by
  simpa [openingSurface, certifiedPriorVerifier] using
    certifiedPriorVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact_accepts
      checks
      soundness
      hVerify

/-- Accepted production exact prior verification is accepted by the strict sound verifier. -/
theorem soundPriorVerifierAccepts
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ExactChecks (PriorProof := PriorProof) ctx)
    (soundness : RuntimeAuthoritySoundness checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      VerifyPrior (openingSurface checks soundness) steps proof image) :
    CompressedFPrimeAuthority.SoundVerifier.Accepts
      (soundPriorVerifier checks soundness)
      steps
      proof
      image := by
  exact
    (soundVerifierOfProductionExactRuntimeAuthoritySoundnessConcreteExact_accepts_iff
      checks
      soundness).2
      (verifyPriorAsRuntimeSurface checks soundness hVerify)

/-- Accepted production exact prior verification proves prior reachability. -/
theorem verifyPriorReaches
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ExactChecks (PriorProof := PriorProof) ctx)
    (soundness : RuntimeAuthoritySoundness checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      VerifyPrior (openingSurface checks soundness) steps proof image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image := by
  rcases verifyPriorOpens checks soundness hVerify with
    ⟨authority, _hOpen, hAccept⟩
  exact
    FoldedFPrimeAuthority.accepts_sound
      steps
      authority
      image
      hAccept

/-- Production exact prior verification cannot accept an unreachable prior image. -/
theorem verifyPriorRejectsUnreachable
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ExactChecks (PriorProof := PriorProof) ctx)
    (soundness : RuntimeAuthoritySoundness checks)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      VerifyPrior (openingSurface checks soundness) steps proof image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
          (DirectParentOnlyProductionSoundness.Transition
            ctx.toProductionContext)
          ctx.initial
          steps
          image) :
    False :=
  hUnreachable (verifyPriorReaches checks soundness hVerify)

/--
Production exact prior verification plus the latest Construction-2 step gives
the parent-only terminal end-to-end package.
-/
theorem parentOnlyTerminalSoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ExactChecks (PriorProof := PriorProof) ctx)
    (soundness : RuntimeAuthoritySoundness checks)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      VerifyPrior
        (openingSurface checks soundness)
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
      (certifiedPriorVerifier checks soundness).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage := by
  simpa [openingSurface, certifiedPriorVerifier] using
    certifiedSingleTerminalEndToEnd_ofProductionExactRuntimeAuthoritySoundnessConcreteExactLatestStep
      checks
      soundness
      hPrior
      hLatest

/-- Non-aggregate private DEC and stage facts from the production exact path. -/
theorem privateDecFacts
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ExactChecks (PriorProof := PriorProof) ctx)
    (soundness : RuntimeAuthoritySoundness checks)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      VerifyPrior
        (openingSurface checks soundness)
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
  simpa [openingSurface] using
    nonAggregatePrivateDecStageFacts_ofProductionExactRuntimeAuthoritySoundnessConcreteExactLatestStep
      checks
      soundness
      hPrior
      hLatest

/--
Concrete no-swap audit for an alternate child table satisfying the full
pointwise private DEC requirements for the same parent source.
-/
theorem privateDecNoSwapAudit
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ExactChecks (PriorProof := PriorProof) ctx)
    (soundness : RuntimeAuthoritySoundness checks)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      VerifyPrior
        (openingSurface checks soundness)
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
        latestProof)
    {otherInputs : DecDigitUniqueness.ColumnDigits n}
    (hOther :
      DirectParentOnlyProductionPrivateDecNoSwapAudit.PointwiseRequirements
        ctx
        priorImage.accumulator.parentSource
        otherInputs) :
    ∃
      (priorInputs : DecDigitUniqueness.ColumnDigits n)
      (auditedCert :
        ParentOnlyAccumulatorStep.PointwisePrivateDecCertificate
          (n := n)
          (hashEncoded := ctx.parentHash.hashEncoded)
          (params := params)
          (ce := ctx.data.ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              ctx.commitmentOfParent)
          priorImage.accumulator.parentSource
          priorInputs)
      (otherCert :
        ParentOnlyAccumulatorStep.PointwisePrivateDecCertificate
          (n := n)
          (hashEncoded := ctx.parentHash.hashEncoded)
          (params := params)
          (ce := ctx.data.ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              ctx.commitmentOfParent)
          priorImage.accumulator.parentSource
          otherInputs),
        ParentOnlyAccumulatorStep.PrivateDecNoSwapAudit
          auditedCert
          otherCert :=
  DirectParentOnlyProductionPrivateDecNoSwapAudit.auditOfFacts
    (privateDecFacts checks soundness hPrior hLatest)
    hOther

/-- Section 7.1 owner-target stage audit from the production exact path. -/
theorem section71StageAudit
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ExactChecks (PriorProof := PriorProof) ctx)
    (soundness : RuntimeAuthoritySoundness checks)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      VerifyPrior
        (openingSurface checks soundness)
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
  simpa [openingSurface] using
    section71StageTargetAuditTrail_ofProductionExactRuntimeAuthoritySoundnessConcreteExactLatestStep
      checks
      soundness
      hPrior
      hLatest

end DirectParentOnlyProductionExactRuntimeInstantiation

end DirectCcsFPrime
