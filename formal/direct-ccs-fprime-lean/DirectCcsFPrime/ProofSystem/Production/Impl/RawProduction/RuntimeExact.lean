import DirectCcsFPrime.ProofSystem.Production.Impl.RawProduction.Exact

/-!
Strict verifier consequences for the backend-routed production exact F' path.

This file keeps the backend-routed exact public-IO verifier package out of the
main production-exact theorem file so the production proof files stay small.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorRawProduction

/--
Canonical terminal-slice discipline for the production exact verifier.

The verifier output exposes a structured terminal/boundary split plus its raw
public vector. This requirement says the terminal part is the canonical-length
prefix of that raw vector whenever the raw vector is the canonical
terminal/boundary concatenation.
-/
structure ProductionExactTerminalSliceBinding
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx) where
  terminalSliceMatchesCanonicalPrefix :
    ∀ steps proof image publicIO,
      surface.checks.terminalVerifierPublicIO
        (surface.checks.terminalCommittedProof proof) =
          some publicIO →
      publicIO.raw =
        surface.checks.terminalPublicValues
          (surface.checks.canonicalIvcPublicImage steps image) ++
        surface.checks.boundaryPublicValues
          (surface.checks.construction2Boundary
            (surface.checks.canonicalIvcPublicImage steps image)) →
      publicIO.terminal =
        publicIO.raw.take
          (surface.checks.terminalPublicValues
            (surface.checks.canonicalIvcPublicImage steps image)).length

/--
Canonical terminal-slice discipline induces the production exact terminal
length binding consumed by the backend-routed verifier bridge.
-/
def productionExactTerminalLengthBindingOfTerminalSlice
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    (sliceBinding : ProductionExactTerminalSliceBinding surface) :
    ProductionExactTerminalLengthBinding surface where
  terminalLengthBindsExpected := by
    intro steps proof image publicIO hPublicIO hRaw
    let expectedTerminal :=
      surface.checks.terminalPublicValues
        (surface.checks.canonicalIvcPublicImage steps image)
    let expectedBoundary :=
      surface.checks.boundaryPublicValues
        (surface.checks.construction2Boundary
          (surface.checks.canonicalIvcPublicImage steps image))
    have hTerminal :
        publicIO.terminal =
          publicIO.raw.take expectedTerminal.length := by
      simpa [expectedTerminal, expectedBoundary] using
        sliceBinding.terminalSliceMatchesCanonicalPrefix
          steps
          proof
          image
          publicIO
          hPublicIO
          hRaw
    have hRawLength :
        expectedTerminal.length ≤ publicIO.raw.length := by
      rw [hRaw]
      simp [expectedTerminal]
    calc
      publicIO.terminal.length =
          (publicIO.raw.take expectedTerminal.length).length := by
        rw [hTerminal]
      _ = expectedTerminal.length := by
        simp [List.length_take, hRawLength]

/--
Single backend authority certificate for production exact verifier checks.

This is the compact trusted boundary: accepted production exact evidence must
open through the fixed opener to folded F' authority for the same public pair.
-/
structure ProductionExactAuthorityCertificate
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx) where
  acceptedOpens :
    ∀ steps proof image publicIO,
      ProductionExactBoundStatementAccepted
        checks
        steps
        proof
        image
        publicIO →
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
Instantiate the production exact opening surface from a single authority
certificate.

Lean derives the split `opens` and `opened authority binds` obligations from
the single folded-authority opening theorem.
-/
def productionExactPriorOpeningSurfaceOfAuthorityCertificate
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (certificate : ProductionExactAuthorityCertificate checks) :
    ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx where
  checks := checks
  productionExactBackendOpens := by
    intro steps proof image publicIO hBound
    rcases
      certificate.acceptedOpens
        steps
        proof
        image
        publicIO
        hBound with
      ⟨authority, hOpen, _hAccepts⟩
    exact ⟨authority, hOpen⟩
  openedAuthorityBindsProductionExactStatement := by
    intro steps proof image publicIO authority hBound hOpen
    rcases
      certificate.acceptedOpens
        steps
        proof
        image
        publicIO
        hBound with
      ⟨openedAuthority, hOpened, hAccepts⟩
    have hSame : openedAuthority = authority := by
      have hSome : some openedAuthority = some authority :=
        hOpened.symm.trans hOpen
      cases hSome
      rfl
    cases hSame
    exact hAccepts

/--
The surface induced by a single authority certificate has the certificate's
accepted-opening consequence.
-/
theorem runtimeVerifyPriorOfProductionExactAuthorityCertificate_acceptedOpens
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
    (hVerify :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfAuthorityCertificate
          checks
          certificate)
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
  exact
    runtimeVerifyPriorOfProductionExact_acceptedOpens
      (productionExactPriorOpeningSurfaceOfAuthorityCertificate
        checks
        certificate)
      steps
      proof
      image
      hVerify

/--
The backend-routed production exact `SoundVerifier` accepts exactly the
backend-shaped exact public-IO opening predicate.
-/
theorem soundVerifierOfProductionExactRuntimeExactPublicIOOpening_accepts_iff
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    (terminalLengthBinding :
      ProductionExactTerminalLengthBinding surface)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary} :
    CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionExactRuntimeExactPublicIOOpening
          surface
          terminalLengthBinding)
        steps
        proof
        image <->
      DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.RuntimeVerifyPriorOfRuntimeExactPublicIOOpening
        (runtimeExactPublicIOOpeningSurfaceOfProductionExact
          surface
          terminalLengthBinding)
        steps
        proof
        image := by
  simpa [
    soundVerifierOfProductionExactRuntimeExactPublicIOOpening,
    certifiedPriorVerifierOfProductionExactRuntimeExactPublicIOOpening_verify]
    using
      DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifier_accepts_iff
        (certifiedPriorVerifierOfProductionExactRuntimeExactPublicIOOpening
          surface
          terminalLengthBinding)

/--
Terminal acceptance from production exact checks passes through the
backend-routed strict `SoundVerifier`.
-/
theorem acceptedTerminalWithSoundVerifierOfProductionExactRuntimeExactPublicIOOpeningLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    (terminalLengthBinding :
      ProductionExactTerminalLengthBinding surface)
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
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.AcceptedTerminal
      ctx
      (soundVerifierOfProductionExactRuntimeExactPublicIOOpening
        surface
        terminalLengthBinding)
      priorSteps
      priorProof
      priorImage
      nextImage
      latestProof := by
  refine
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.acceptedTerminalWithSoundVerifier
      (certifiedPriorVerifierOfProductionExactRuntimeExactPublicIOOpening
        surface
        terminalLengthBinding)
      ?_
  exact
    { priorAccepted :=
        certifiedPriorVerifierOfProductionExactRuntimeExactPublicIOOpening_accepts
          surface
          terminalLengthBinding
          hPrior
      latestAccepted := hLatest }

/--
Backend-routed production exact projection to non-aggregate private DEC and
stage facts.
-/
theorem nonAggregatePrivateDecStageFacts_ofProductionExactRuntimeExactPublicIOOpeningLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    (terminalLengthBinding :
      ProductionExactTerminalLengthBinding surface)
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
  DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.nonAggregatePrivateDecStageFacts_ofRuntimeExactPublicIOOpeningLatestStep
    (runtimeExactPublicIOOpeningSurfaceOfProductionExact
      surface
      terminalLengthBinding)
    (runtimeVerifyPriorOfProductionExact_toRuntimeExactPublicIOOpening
      surface
      terminalLengthBinding
      hPrior)
    hLatest

/--
Backend-routed production exact projection to the Section 7.1 owner-target
stage audit.
-/
theorem section71StageTargetAuditTrail_ofProductionExactRuntimeExactPublicIOOpeningLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    (terminalLengthBinding :
      ProductionExactTerminalLengthBinding surface)
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
  DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.section71StageTargetAuditTrail_ofRuntimeExactPublicIOOpeningLatestStep
    (runtimeExactPublicIOOpeningSurfaceOfProductionExact
      surface
      terminalLengthBinding)
    (runtimeVerifyPriorOfProductionExact_toRuntimeExactPublicIOOpening
      surface
      terminalLengthBinding
      hPrior)
    hLatest

/--
Backend-shaped exact public-IO opening surface induced by canonical terminal
slice binding.
-/
def runtimeExactPublicIOOpeningSurfaceOfProductionExactTerminalSlice
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    (sliceBinding : ProductionExactTerminalSliceBinding surface) :
    DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.ConcreteRuntimeExactPublicIOOpeningSurface
      (PriorProof := PriorProof)
      ctx :=
  runtimeExactPublicIOOpeningSurfaceOfProductionExact
    surface
    (productionExactTerminalLengthBindingOfTerminalSlice
      surface
      sliceBinding)

/--
Certified prior verifier for production exact checks using canonical terminal
slice binding.
-/
def certifiedPriorVerifierOfProductionExactTerminalSlice
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    (sliceBinding : ProductionExactTerminalSliceBinding surface) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  certifiedPriorVerifierOfProductionExactRuntimeExactPublicIOOpening
    surface
    (productionExactTerminalLengthBindingOfTerminalSlice
      surface
      sliceBinding)

/--
Production exact verification is accepted by the canonical terminal-slice
certified verifier.
-/
theorem certifiedPriorVerifierOfProductionExactTerminalSlice_accepts
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    (sliceBinding : ProductionExactTerminalSliceBinding surface)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionExact
        surface
        steps
        proof
        image) :
    (certifiedPriorVerifierOfProductionExactTerminalSlice
        surface
        sliceBinding).verify
      steps
      proof
      image := by
  simpa [certifiedPriorVerifierOfProductionExactTerminalSlice] using
    certifiedPriorVerifierOfProductionExactRuntimeExactPublicIOOpening_accepts
      surface
      (productionExactTerminalLengthBindingOfTerminalSlice
        surface
        sliceBinding)
      hVerify

/--
Strict `SoundVerifier` for the production exact path using canonical terminal
slice binding.
-/
def soundVerifierOfProductionExactTerminalSlice
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    (sliceBinding : ProductionExactTerminalSliceBinding surface) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.SoundPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifier
    (certifiedPriorVerifierOfProductionExactTerminalSlice
      surface
      sliceBinding)

/--
The canonical terminal-slice production exact `SoundVerifier` accepts exactly
the induced backend-shaped exact public-IO opening predicate.
-/
theorem soundVerifierOfProductionExactTerminalSlice_accepts_iff
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    (sliceBinding : ProductionExactTerminalSliceBinding surface)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary} :
    CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionExactTerminalSlice
          surface
          sliceBinding)
        steps
        proof
        image <->
      DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.RuntimeVerifyPriorOfRuntimeExactPublicIOOpening
        (runtimeExactPublicIOOpeningSurfaceOfProductionExactTerminalSlice
          surface
          sliceBinding)
        steps
        proof
        image := by
  simpa [
    soundVerifierOfProductionExactTerminalSlice,
    certifiedPriorVerifierOfProductionExactTerminalSlice,
    runtimeExactPublicIOOpeningSurfaceOfProductionExactTerminalSlice]
    using
      soundVerifierOfProductionExactRuntimeExactPublicIOOpening_accepts_iff
        surface
        (productionExactTerminalLengthBindingOfTerminalSlice
          surface
          sliceBinding)

/--
Production exact verifier acceptance opens folded F' authority through the
canonical terminal-slice bridge.
-/
theorem runtimeVerifyPriorOfProductionExact_viaTerminalSlice_acceptedOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    (sliceBinding : ProductionExactTerminalSliceBinding surface)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionExact
        surface
        steps
        proof
        image) :
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
  runtimeVerifyPriorOfProductionExact_viaRuntimeExactPublicIOOpening_acceptedOpens
    surface
    (productionExactTerminalLengthBindingOfTerminalSlice
      surface
      sliceBinding)
    hVerify

/--
Terminal acceptance from production exact checks passes through the canonical
terminal-slice strict `SoundVerifier`.
-/
theorem acceptedTerminalWithSoundVerifierOfProductionExactTerminalSliceLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    (sliceBinding : ProductionExactTerminalSliceBinding surface)
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
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.AcceptedTerminal
      ctx
      (soundVerifierOfProductionExactTerminalSlice
        surface
        sliceBinding)
      priorSteps
      priorProof
      priorImage
      nextImage
      latestProof := by
  refine
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.acceptedTerminalWithSoundVerifier
      (certifiedPriorVerifierOfProductionExactTerminalSlice
        surface
        sliceBinding)
      ?_
  exact
    { priorAccepted :=
        certifiedPriorVerifierOfProductionExactTerminalSlice_accepts
          surface
          sliceBinding
          hPrior
      latestAccepted := hLatest }

/--
Canonical terminal-slice production exact prior-plus-latest end-to-end theorem.
-/
theorem certifiedSingleTerminalEndToEnd_ofProductionExactTerminalSliceLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    (sliceBinding : ProductionExactTerminalSliceBinding surface)
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
      (certifiedPriorVerifierOfProductionExactTerminalSlice
        surface
        sliceBinding).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  certifiedSingleTerminalEndToEnd_ofProductionExactRuntimeExactPublicIOOpeningLatestStep
    surface
    (productionExactTerminalLengthBindingOfTerminalSlice
      surface
      sliceBinding)
    hPrior
    hLatest

/--
Canonical terminal-slice production exact projection to non-aggregate private
DEC and stage facts.
-/
theorem nonAggregatePrivateDecStageFacts_ofProductionExactTerminalSliceLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    (sliceBinding : ProductionExactTerminalSliceBinding surface)
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
  nonAggregatePrivateDecStageFacts_ofProductionExactRuntimeExactPublicIOOpeningLatestStep
    surface
    (productionExactTerminalLengthBindingOfTerminalSlice
      surface
      sliceBinding)
    hPrior
    hLatest

/--
Canonical terminal-slice production exact projection to the Section 7.1
owner-target stage audit.
-/
theorem section71StageTargetAuditTrail_ofProductionExactTerminalSliceLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    (sliceBinding : ProductionExactTerminalSliceBinding surface)
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
  section71StageTargetAuditTrail_ofProductionExactRuntimeExactPublicIOOpeningLatestStep
    surface
    (productionExactTerminalLengthBindingOfTerminalSlice
      surface
      sliceBinding)
    hPrior
    hLatest

/--
Runtime exact public-IO opening surface induced directly from production exact
checks, the single authority certificate, and canonical terminal-slice binding.
-/
def runtimeExactPublicIOOpeningSurfaceOfProductionExactAuthorityCertificateTerminalSlice
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (certificate : ProductionExactAuthorityCertificate checks)
    (sliceBinding :
      ProductionExactTerminalSliceBinding
        (productionExactPriorOpeningSurfaceOfAuthorityCertificate
          checks
          certificate)) :
    DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.ConcreteRuntimeExactPublicIOOpeningSurface
      (PriorProof := PriorProof)
      ctx :=
  runtimeExactPublicIOOpeningSurfaceOfProductionExactTerminalSlice
    (productionExactPriorOpeningSurfaceOfAuthorityCertificate
      checks
      certificate)
    sliceBinding

/--
Certified prior verifier induced directly from production exact checks, the
single authority certificate, and canonical terminal-slice binding.
-/
def certifiedPriorVerifierOfProductionExactAuthorityCertificateTerminalSlice
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (certificate : ProductionExactAuthorityCertificate checks)
    (sliceBinding :
      ProductionExactTerminalSliceBinding
        (productionExactPriorOpeningSurfaceOfAuthorityCertificate
          checks
          certificate)) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  certifiedPriorVerifierOfProductionExactTerminalSlice
    (productionExactPriorOpeningSurfaceOfAuthorityCertificate
      checks
      certificate)
    sliceBinding

/--
Production exact verification is accepted by the certificate-native
terminal-slice certified verifier.
-/
theorem certifiedPriorVerifierOfProductionExactAuthorityCertificateTerminalSlice_accepts
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (certificate : ProductionExactAuthorityCertificate checks)
    (sliceBinding :
      ProductionExactTerminalSliceBinding
        (productionExactPriorOpeningSurfaceOfAuthorityCertificate
          checks
          certificate))
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfAuthorityCertificate
          checks
          certificate)
        steps
        proof
        image) :
    (certifiedPriorVerifierOfProductionExactAuthorityCertificateTerminalSlice
        checks
        certificate
        sliceBinding).verify
      steps
      proof
      image := by
  simpa [
    certifiedPriorVerifierOfProductionExactAuthorityCertificateTerminalSlice]
    using
      certifiedPriorVerifierOfProductionExactTerminalSlice_accepts
        (productionExactPriorOpeningSurfaceOfAuthorityCertificate
          checks
          certificate)
        sliceBinding
        hVerify

/--
Strict `SoundVerifier` induced directly from production exact checks, the
single authority certificate, and canonical terminal-slice binding.
-/
def soundVerifierOfProductionExactAuthorityCertificateTerminalSlice
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (certificate : ProductionExactAuthorityCertificate checks)
    (sliceBinding :
      ProductionExactTerminalSliceBinding
        (productionExactPriorOpeningSurfaceOfAuthorityCertificate
          checks
          certificate)) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.SoundPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifier
    (certifiedPriorVerifierOfProductionExactAuthorityCertificateTerminalSlice
      checks
      certificate
      sliceBinding)

/--
The certificate-native terminal-slice `SoundVerifier` accepts exactly the
induced backend-shaped exact public-IO opening predicate.
-/
theorem soundVerifierOfProductionExactAuthorityCertificateTerminalSlice_accepts_iff
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (certificate : ProductionExactAuthorityCertificate checks)
    (sliceBinding :
      ProductionExactTerminalSliceBinding
        (productionExactPriorOpeningSurfaceOfAuthorityCertificate
          checks
          certificate))
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary} :
    CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionExactAuthorityCertificateTerminalSlice
          checks
          certificate
          sliceBinding)
        steps
        proof
        image <->
      DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.RuntimeVerifyPriorOfRuntimeExactPublicIOOpening
        (runtimeExactPublicIOOpeningSurfaceOfProductionExactAuthorityCertificateTerminalSlice
          checks
          certificate
          sliceBinding)
        steps
        proof
        image := by
  simpa [
    soundVerifierOfProductionExactAuthorityCertificateTerminalSlice,
    certifiedPriorVerifierOfProductionExactAuthorityCertificateTerminalSlice,
    runtimeExactPublicIOOpeningSurfaceOfProductionExactAuthorityCertificateTerminalSlice]
    using
      soundVerifierOfProductionExactTerminalSlice_accepts_iff
        (productionExactPriorOpeningSurfaceOfAuthorityCertificate
          checks
          certificate)
        sliceBinding

/--
Terminal acceptance from certificate-native production exact checks passes
through the certificate-native strict `SoundVerifier`.
-/
theorem acceptedTerminalWithSoundVerifierOfProductionExactAuthorityCertificateTerminalSliceLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (certificate : ProductionExactAuthorityCertificate checks)
    (sliceBinding :
      ProductionExactTerminalSliceBinding
        (productionExactPriorOpeningSurfaceOfAuthorityCertificate
          checks
          certificate))
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfAuthorityCertificate
          checks
          certificate)
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
      (soundVerifierOfProductionExactAuthorityCertificateTerminalSlice
        checks
        certificate
        sliceBinding)
      priorSteps
      priorProof
      priorImage
      nextImage
      latestProof := by
  simpa [
    soundVerifierOfProductionExactAuthorityCertificateTerminalSlice,
    certifiedPriorVerifierOfProductionExactAuthorityCertificateTerminalSlice]
    using
      acceptedTerminalWithSoundVerifierOfProductionExactTerminalSliceLatestStep
        (productionExactPriorOpeningSurfaceOfAuthorityCertificate
          checks
          certificate)
        sliceBinding
        hPrior
        hLatest

/--
Certificate-native production exact prior-plus-latest end-to-end theorem.

This is the production theorem surface that consumes the concrete verifier
checks through one authority certificate rather than a caller-assembled opening
surface.
-/
theorem certifiedSingleTerminalEndToEnd_ofProductionExactAuthorityCertificateTerminalSliceLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (certificate : ProductionExactAuthorityCertificate checks)
    (sliceBinding :
      ProductionExactTerminalSliceBinding
        (productionExactPriorOpeningSurfaceOfAuthorityCertificate
          checks
          certificate))
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfAuthorityCertificate
          checks
          certificate)
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
      (certifiedPriorVerifierOfProductionExactAuthorityCertificateTerminalSlice
        checks
        certificate
        sliceBinding).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage := by
  simpa [
    certifiedPriorVerifierOfProductionExactAuthorityCertificateTerminalSlice]
    using
      certifiedSingleTerminalEndToEnd_ofProductionExactTerminalSliceLatestStep
        (productionExactPriorOpeningSurfaceOfAuthorityCertificate
          checks
          certificate)
        sliceBinding
        hPrior
        hLatest

/--
Certificate-native production exact projection to non-aggregate private DEC and
stage facts.
-/
theorem nonAggregatePrivateDecStageFacts_ofProductionExactAuthorityCertificateTerminalSliceLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (certificate : ProductionExactAuthorityCertificate checks)
    (sliceBinding :
      ProductionExactTerminalSliceBinding
        (productionExactPriorOpeningSurfaceOfAuthorityCertificate
          checks
          certificate))
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfAuthorityCertificate
          checks
          certificate)
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
    (productionExactPriorOpeningSurfaceOfAuthorityCertificate
      checks
      certificate)
    sliceBinding
    hPrior
    hLatest

/--
Certificate-native production exact projection to the Section 7.1 owner-target
stage audit.
-/
theorem section71StageTargetAuditTrail_ofProductionExactAuthorityCertificateTerminalSliceLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks :
      ProductionExactPriorVerifierChecks (PriorProof := PriorProof) ctx)
    (certificate : ProductionExactAuthorityCertificate checks)
    (sliceBinding :
      ProductionExactTerminalSliceBinding
        (productionExactPriorOpeningSurfaceOfAuthorityCertificate
          checks
          certificate))
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionExact
        (productionExactPriorOpeningSurfaceOfAuthorityCertificate
          checks
          certificate)
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
    (productionExactPriorOpeningSurfaceOfAuthorityCertificate
      checks
      certificate)
    sliceBinding
    hPrior
    hLatest

end DirectParentOnlyProductionConcreteFPrimePriorRawProduction

end DirectCcsFPrime
