import DirectCcsFPrime.ProofSystem.Production.Impl.Runtime.Verifier
import DirectCcsFPrime.ProofSystem.Production.Impl.ConcretePrior.VerifierCertified
import DirectCcsFPrime.Audit.RedTeam.ProductionPrivateDecNoSwapAudit

/-!
Production-facing prior F' verifier API.

This file owns the concise entry point for the concrete prior-verifier path.
The backend boundary is the exact compressed-verifier audit: once the verifier
has replayed the public statement, Poseidon2 transcript, exact terminal public
IO, and final claims, acceptance must open folded F' authority for the same
`(steps, image)` pair.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionFPrimePriorVerifier

open DirectParentOnlyProductionConcreteFPrimePriorRawProduction

/--
Concrete F' backend soundness at the verifier audit boundary.

This is the trusted backend obligation: a proof accepted by the exact
compressed verifier must open a real folded F' authority object for the same
public statement. Digests are replayed as binding data, not accepted as
authority by themselves.
-/
abbrev BackendSoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx) :=
  ProductionExactCompressedVerifierSoundness checks

/--
Split exact-verifier opening surface.

This is the preferred concrete verifier boundary when an implementation can
separate the backend obligations: accepted exact statements must open through
the fixed opener, and every opened authority must bind the same public
`(steps, image)` pair.
-/
abbrev OpeningSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) :=
  ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx

/--
Implementation-shaped exact-runtime verifier surface.

This is the concrete backend entry point: it carries verifier replay facts,
terminal/boundary public-IO layout, the fixed authority opener, and the minimal
backend soundness theorem needed to turn acceptance into folded F' authority.
-/
abbrev RuntimeExactSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) :=
  DirectParentOnlyProductionConcreteFPrimePriorVerifierCertified.ConcreteRuntimeExactPublicIOSurface
    (PriorProof := PriorProof)
    ctx

/--
Layout binding for the implementation-shaped exact-runtime verifier.

The layout witness connects the verifier's raw terminal public IO to the
structured terminal/boundary split consumed by the authority-opening proof.
-/
abbrev RuntimeExactLayout
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : RuntimeExactSurface (PriorProof := PriorProof) ctx) :=
  DirectParentOnlyProductionConcreteFPrimePriorVerifierCertified.ExactPublicIOLayoutBinding
    surface

/-- Concrete exact-runtime prior verification predicate. -/
abbrev RuntimeExactVerify
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : RuntimeExactSurface (PriorProof := PriorProof) ctx) :=
  DirectParentOnlyProductionConcreteFPrimePriorVerifierCertified.RuntimeVerifyPriorOfExactPublicIO
    surface

/-- Runtime authority soundness derived from backend audit soundness. -/
def runtimeSoundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : BackendSoundness checks) :
    ProductionExactRuntimeAuthoritySoundness checks :=
  productionExactRuntimeAuthoritySoundnessOfCompressedVerifierSoundness
    checks
    soundness

/-- Backend audit soundness derived from the split opening surface. -/
def backendSoundnessOfOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : OpeningSurface (PriorProof := PriorProof) ctx) :
    BackendSoundness surface.checks :=
  productionExactCompressedVerifierSoundnessOfOpeningSurface surface

/-- Runtime authority soundness derived from the split opening surface. -/
def runtimeSoundnessOfOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : OpeningSurface (PriorProof := PriorProof) ctx) :
    ProductionExactRuntimeAuthoritySoundness surface.checks :=
  runtimeSoundness
    surface.checks
    (backendSoundnessOfOpening surface)

/-- Production exact opening surface induced by backend audit soundness. -/
def openingSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : BackendSoundness checks) :
    ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx :=
  DirectParentOnlyProductionFPrimeRuntimeVerifier.openingSurface
    checks
    (runtimeSoundness checks soundness)

/-- Exact verifier acceptance through the split surface opens folded F' authority. -/
theorem verifyOpensOfOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : OpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionExact surface steps proof image) :
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
    hVerify

/-- Exact verifier acceptance through the split surface exposes audit evidence. -/
theorem verifyAuditOfOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : OpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionExact surface steps proof image) :
    ∃ publicIO : ExactTerminalBoundaryPublicIO surface.checks.PublicField,
      ProductionExactVerifierAcceptedAudit
        surface.checks
        steps
        proof
        image
        publicIO := by
  rcases hVerify with ⟨publicIO, hAccepted⟩
  exact ⟨publicIO, productionExactVerifierAccepted_audit hAccepted⟩

/-- Exact verifier audit evidence opens folded F' authority for the same pair. -/
theorem auditOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : BackendSoundness checks)
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
  productionExactCompressedVerifierSoundness_acceptedAuditOpens
    checks
    soundness
    hAudit

/-- Audit evidence for the split surface opens folded F' authority. -/
theorem auditOpensOfOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : OpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO surface.checks.PublicField}
    (hAudit :
      ProductionExactVerifierAcceptedAudit
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
  auditOpens
    surface.checks
    (backendSoundnessOfOpening surface)
    hAudit

/-- Audit evidence reaches the claimed prior public image. -/
theorem auditReaches
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : BackendSoundness checks)
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
  DirectParentOnlyProductionFPrimeRuntimeVerifier.auditReaches
    checks
    (runtimeSoundness checks soundness)
    hAudit

/-- Split-surface audit evidence reaches the claimed prior public image. -/
theorem auditReachesOfOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : OpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {publicIO : ExactTerminalBoundaryPublicIO surface.checks.PublicField}
    (hAudit :
      ProductionExactVerifierAcceptedAudit
        surface.checks
        steps
        proof
        image
        publicIO) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image :=
  auditReaches
    surface.checks
    (backendSoundnessOfOpening surface)
    hAudit

/-- Certified prior verifier induced by backend audit soundness. -/
def certified
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : BackendSoundness checks) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionFPrimeRuntimeVerifier.certified
    checks
    (runtimeSoundness checks soundness)

/-- Certified prior verifier induced by the split opening surface. -/
def certifiedOfOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : OpeningSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  certified
    surface.checks
    (backendSoundnessOfOpening surface)

/--
Certified prior verifier induced directly from the implementation-shaped
exact-runtime verifier surface.
-/
def certifiedOfRuntimeExact
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : RuntimeExactSurface (PriorProof := PriorProof) ctx)
    (layout : RuntimeExactLayout surface) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionConcreteFPrimePriorVerifierCertified.certifiedPriorVerifierOfExactRuntimeSoundAndLayout
    surface
    layout

/-- Strict sound verifier induced by backend audit soundness. -/
def sound
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : BackendSoundness checks) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.SoundPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionFPrimeRuntimeVerifier.sound
    checks
    (runtimeSoundness checks soundness)

/-- Strict sound verifier induced by the split opening surface. -/
def soundOfOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : OpeningSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.SoundPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  sound
    surface.checks
    (backendSoundnessOfOpening surface)

/--
Strict prior verifier induced directly from the implementation-shaped
exact-runtime verifier surface.
-/
def soundOfRuntimeExact
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : RuntimeExactSurface (PriorProof := PriorProof) ctx)
    (layout : RuntimeExactLayout surface) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.SoundPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionConcreteFPrimePriorVerifierCertified.soundVerifierOfExactRuntimeSoundAndLayout
    surface
    layout

/-- Audit evidence is accepted by the strict sound verifier. -/
theorem soundAcceptsAudit
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : BackendSoundness checks)
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
      DirectParentOnlyProductionFPrimeRuntimeVerifier.soundAcceptsAudit
        checks
        (runtimeSoundness checks soundness)
        hAudit

/-- Exact verifier acceptance through the split surface is accepted by the strict verifier. -/
theorem soundAcceptsVerifyOfOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : OpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionExact surface steps proof image) :
    CompressedFPrimeAuthority.SoundVerifier.Accepts
      (soundOfOpening surface)
      steps
      proof
      image := by
  rcases verifyAuditOfOpening surface hVerify with ⟨publicIO, hAudit⟩
  exact
    soundAcceptsAudit
      surface.checks
      (backendSoundnessOfOpening surface)
      (publicIO := publicIO)
      hAudit

/--
Implementation-shaped exact-runtime acceptance opens folded F' authority for
the same public pair through the fixed opener.
-/
theorem verifyOpensOfRuntimeExact
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : RuntimeExactSurface (PriorProof := PriorProof) ctx}
    {layout : RuntimeExactLayout surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeExactVerify surface steps proof image) :
    ∃ authority : ProofCarryingPriorProof ctx,
      surface.openAuthority proof = some authority ∧
        FoldedFPrimeAuthority.Accepts
          (Transition :=
            DirectParentOnlyProductionSoundness.Transition
              ctx.toProductionContext)
          (initial := ctx.initial)
          steps
          authority
          image := by
  rcases
    DirectParentOnlyProductionConcreteFPrimePriorVerifierCertified.runtimeVerifyPriorOfExactRuntimeSoundAndLayout_opensAuthority
      (layout := layout)
      hVerify with
    ⟨authority, hOpen⟩
  exact
    ⟨authority,
      hOpen,
      DirectParentOnlyProductionConcreteFPrimePriorVerifierCertified.runtimeVerifyPriorOfExactRuntimeSoundAndLayout_openedAuthority_accepts
        (layout := layout)
        hVerify
        hOpen⟩

/--
Any authority opened by implementation-shaped exact-runtime acceptance is bound
to the verifier's same claimed `(steps, image)` pair.
-/
theorem openedAuthorityAcceptsOfRuntimeExact
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : RuntimeExactSurface (PriorProof := PriorProof) ctx}
    {layout : RuntimeExactLayout surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      RuntimeExactVerify surface steps proof image)
    (hOpen : surface.openAuthority proof = some authority) :
    FoldedFPrimeAuthority.Accepts
      (Transition :=
        DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
      (initial := ctx.initial)
      steps
      authority
      image :=
  DirectParentOnlyProductionConcreteFPrimePriorVerifierCertified.runtimeVerifyPriorOfExactRuntimeSoundAndLayout_openedAuthority_accepts
    (layout := layout)
    hVerify
    hOpen

/-- Implementation-shaped exact-runtime acceptance cannot succeed if opening fails. -/
theorem cannotAcceptWithoutOpeningOfRuntimeExact
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : RuntimeExactSurface (PriorProof := PriorProof) ctx}
    {layout : RuntimeExactLayout surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeExactVerify surface steps proof image)
    (hNone : surface.openAuthority proof = none) :
    False :=
  DirectParentOnlyProductionConcreteFPrimePriorVerifierCertified.runtimeVerifyPriorOfExactRuntimeSoundAndLayout_cannot_accept_without_opening
    (layout := layout)
    hVerify
    hNone

/--
Implementation-shaped exact-runtime acceptance is accepted by the derived
strict verifier.
-/
theorem soundAcceptsVerifyOfRuntimeExact
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : RuntimeExactSurface (PriorProof := PriorProof) ctx}
    {layout : RuntimeExactLayout surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeExactVerify surface steps proof image) :
    CompressedFPrimeAuthority.SoundVerifier.Accepts
      (soundOfRuntimeExact surface layout)
      steps
      proof
      image :=
  (DirectParentOnlyProductionConcreteFPrimePriorVerifierCertified.soundVerifierOfExactRuntimeSoundAndLayout_accepts_iff
    surface
    layout).2 hVerify

/-- One proof accepted by this strict verifier has one public `(steps, image)`. -/
theorem soundSameProof
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx}
    {soundness : BackendSoundness checks}
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
  DirectParentOnlyProductionFPrimeRuntimeVerifier.soundSameProof
    (soundness := runtimeSoundness checks soundness)
    hA
    hB

/-- One proof accepted through the split surface has one public `(steps, image)`. -/
theorem soundSameProofOfOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : OpeningSurface (PriorProof := PriorProof) ctx}
    {stepsA stepsB : Nat}
    {proof : PriorProof}
    {imageA imageB : PublicImage Digest Boundary}
    (hA :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundOfOpening surface)
        stepsA
        proof
        imageA)
    (hB :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundOfOpening surface)
        stepsB
        proof
        imageB) :
    stepsA = stepsB ∧ imageA = imageB :=
  soundSameProof
    (soundness := backendSoundnessOfOpening surface)
    hA
    hB

/--
One implementation-shaped exact-runtime proof cannot verify for two public
`(steps, image)` pairs under the derived strict verifier.
-/
theorem soundSameProofOfRuntimeExact
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : RuntimeExactSurface (PriorProof := PriorProof) ctx}
    {layout : RuntimeExactLayout surface}
    {stepsA stepsB : Nat}
    {proof : PriorProof}
    {imageA imageB : PublicImage Digest Boundary}
    (hA :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundOfRuntimeExact surface layout)
        stepsA
        proof
        imageA)
    (hB :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundOfRuntimeExact surface layout)
        stepsB
        proof
        imageB) :
    stepsA = stepsB ∧ imageA = imageB :=
  DirectParentOnlyProductionConcreteFPrimePriorVerifierCertified.soundVerifierOfExactRuntimeSoundAndLayout_sameProof
    hA
    hB

/--
Implementation-shaped exact-runtime acceptance exposes prior public-image
invariants.
-/
theorem publicImageInvariantsOfRuntimeExact
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : RuntimeExactSurface (PriorProof := PriorProof) ctx}
    {layout : RuntimeExactLayout surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeExactVerify surface steps proof image) :
    DirectParentOnlyProductionConcreteFPrimePrior.AcceptedPriorPublicImageInvariants
      ctx
      steps
      image :=
  DirectParentOnlyProductionConcreteFPrimePriorVerifierCertified.runtimeVerifyPriorOfExactRuntimeSoundAndLayout_publicImageInvariants
    (layout := layout)
    hVerify

/-- Audit evidence plus the latest Construction-2 step gives the end-to-end result. -/
theorem endToEnd
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : BackendSoundness checks)
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
      DirectParentOnlyProductionFPrimeRuntimeVerifier.endToEnd
        checks
        (runtimeSoundness checks soundness)
        hPrior
        hLatest

/-- Split-surface exact verifier acceptance plus the latest step gives end-to-end soundness. -/
theorem endToEndOfOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : OpeningSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionExact
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
      (certifiedOfOpening surface).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage := by
  rcases verifyAuditOfOpening surface hPrior with ⟨publicIO, hAudit⟩
  simpa [certifiedOfOpening]
    using
      endToEnd
        surface.checks
        (backendSoundnessOfOpening surface)
        (publicIO := publicIO)
        hAudit
        hLatest

/--
Implementation-shaped exact-runtime acceptance plus the latest step gives the
production terminal end-to-end result.
-/
theorem endToEndOfRuntimeExact
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : RuntimeExactSurface (PriorProof := PriorProof) ctx)
    (layout : RuntimeExactLayout surface)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeExactVerify
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
      (certifiedOfRuntimeExact surface layout).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  DirectParentOnlyProductionConcreteFPrimePriorVerifierCertified.certifiedSingleTerminalEndToEnd_ofExactRuntimeSoundAndLayoutLatestStep
    surface
    layout
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
    (soundness : BackendSoundness checks)
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
  DirectParentOnlyProductionFPrimeRuntimeVerifier.privateDecFacts
    checks
    (runtimeSoundness checks soundness)
    hPrior
    hLatest

/-- Extract the non-aggregate private DEC/stage facts from split-surface acceptance. -/
theorem privateDecFactsOfOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : OpeningSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionExact
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
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.nonAggregatePrivateDecStageFacts_of_certifiedTerminalEndToEnd
    (endToEndOfOpening surface hPrior hLatest)

/--
Extract non-aggregate private DEC/stage facts from implementation-shaped
exact-runtime acceptance.
-/
theorem privateDecFactsOfRuntimeExact
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : RuntimeExactSurface (PriorProof := PriorProof) ctx)
    (layout : RuntimeExactLayout surface)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeExactVerify
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
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.nonAggregatePrivateDecStageFacts_of_certifiedTerminalEndToEnd
    (endToEndOfRuntimeExact surface layout hPrior hLatest)

/--
Extract the concrete private DEC no-swap audit from implementation-shaped
exact-runtime acceptance.

The alternate child table must satisfy the full pointwise private DEC
requirements for the same parent source; the returned audit records the
pointwise equalities that rule out hidden child substitution.
-/
theorem privateDecNoSwapAuditOfRuntimeExact
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : RuntimeExactSurface (PriorProof := PriorProof) ctx)
    (layout : RuntimeExactLayout surface)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeExactVerify
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
    (privateDecFactsOfRuntimeExact surface layout hPrior hLatest)
    hOther

/-- Extract the exact Section 7.1 owner-target stage audit from `endToEnd`. -/
theorem stageAudit
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (soundness : BackendSoundness checks)
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
  DirectParentOnlyProductionFPrimeRuntimeVerifier.stageAudit
    checks
    (runtimeSoundness checks soundness)
    hPrior
    hLatest

/-- Extract the Section 7.1 owner-target stage audit from split-surface acceptance. -/
theorem stageAuditOfOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : OpeningSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionExact
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
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.section71StageTargetAuditTrail_of_certifiedTerminalEndToEnd
    (endToEndOfOpening surface hPrior hLatest)

/--
Extract the Section 7.1 owner-target stage audit from implementation-shaped
exact-runtime acceptance.
-/
theorem stageAuditOfRuntimeExact
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : RuntimeExactSurface (PriorProof := PriorProof) ctx)
    (layout : RuntimeExactLayout surface)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeExactVerify
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
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.section71StageTargetAuditTrail_of_certifiedTerminalEndToEnd
    (endToEndOfRuntimeExact surface layout hPrior hLatest)

end DirectParentOnlyProductionFPrimePriorVerifier

end DirectCcsFPrime
