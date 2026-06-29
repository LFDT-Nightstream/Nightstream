import DirectCcsFPrime.ProofSystem.Production.Impl.PublicIO.Terminal.DirectParentOnlyProductionConcreteFPrimePriorTerminalIO

/-!
Runtime backend authority for the concrete prior F' verifier.

This module separates runtime verifier acceptance from authority extraction.
The verifier predicate contains only the replay, public-boundary, and terminal
public-IO checks that the direct CCS compressed verifier can perform. The one
trusted backend obligation says that those checks, for the exact verifier public
IO, open the proof to folded F' reachability authority for the same
`(steps, image)` pair.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorBackend

/-- Section 7.1-backed production context. -/
abbrev ProductionContext :=
  DirectParentOnlyProductionConcreteFPrimePriorTerminalIO.ProductionContext

/-- Public image for the parent-only production path. -/
abbrev PublicImage :=
  DirectParentOnlyProductionConcreteFPrimePriorTerminalIO.PublicImage

/-- Proof-carrying folded prior authority for the induced production context. -/
abbrev ProofCarryingPriorProof :=
  @DirectParentOnlyProductionConcreteFPrimePriorTerminalIO.ProofCarryingPriorProof

/--
Runtime-shaped backend verifier surface for the production F' prior verifier.

`RuntimeVerifyPrior` built from this surface does not contain an authority
opening premise. It models verifier-visible checks only: replay binding,
statement/public-boundary validation, terminal committed-proof public-IO
verification, and public-IO prefix/suffix checks. The `runtimeSound` field is
the minimal cryptographic backend assumption that turns those checks into a
real folded F' authority opening.
-/
structure ConcreteRuntimeBackendSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) where
  Statement : Type
  PublicBoundary : Type
  TerminalPublicValues : Type
  TerminalCommittedProof : Type
  TerminalVerifierPublicIO : Type
  canonicalStatement :
    Nat →
      PublicImage Digest Boundary →
        Statement
  proofStatement :
    PriorProof →
      Statement
  statementBoundary :
    Statement →
      PublicBoundary
  proofBoundary :
    PriorProof →
      PublicBoundary
  terminalPublicValues :
    Statement →
      TerminalPublicValues
  terminalCommittedProof :
    PriorProof →
      TerminalCommittedProof
  statementPublicValid :
    Statement →
      Prop
  terminalVerifierPublicIO :
    TerminalCommittedProof →
      Option TerminalVerifierPublicIO
  terminalPublicValuesPrefix :
    TerminalPublicValues →
      TerminalVerifierPublicIO →
        Prop
  terminalBoundaryValuesSuffix :
    PublicBoundary →
      TerminalVerifierPublicIO →
        Prop
  compactImageReplay :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop
  construction2BoundaryReplay :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop
  transcriptReplay :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop
  openAuthority :
    PriorProof →
      Option (ProofCarryingPriorProof ctx)
  replayBindsProofStatement :
    ∀ steps proof image,
      compactImageReplay steps proof image →
      construction2BoundaryReplay steps proof image →
      transcriptReplay steps proof image →
        proofStatement proof = canonicalStatement steps image
  runtimeSound :
    ∀ steps proof image publicIO,
      compactImageReplay steps proof image →
      construction2BoundaryReplay steps proof image →
      transcriptReplay steps proof image →
      proofStatement proof = canonicalStatement steps image →
      statementPublicValid (canonicalStatement steps image) →
      proofBoundary proof =
        statementBoundary (canonicalStatement steps image) →
      terminalVerifierPublicIO (terminalCommittedProof proof) =
        some publicIO →
      terminalPublicValuesPrefix
        (terminalPublicValues (canonicalStatement steps image))
        publicIO →
      terminalBoundaryValuesSuffix
        (statementBoundary (canonicalStatement steps image))
        publicIO →
        ∃ authority : ProofCarryingPriorProof ctx,
          openAuthority proof = some authority ∧
            FoldedFPrimeAuthority.Accepts
              (Transition :=
                DirectParentOnlyProductionSoundness.Transition
                  ctx.toProductionContext)
              (initial := ctx.initial)
              steps
              authority
              image

/--
Runtime prior verifier predicate.

This is deliberately opener-free. A proof accepted here has passed the concrete
runtime checks, but the authority opening is derived separately from
`ConcreteRuntimeBackendSurface.runtimeSound`.
-/
def RuntimeVerifyPrior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeBackendSurface (PriorProof := PriorProof) ctx) :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop :=
  fun steps proof image =>
    surface.compactImageReplay steps proof image ∧
      surface.construction2BoundaryReplay steps proof image ∧
      surface.transcriptReplay steps proof image ∧
      surface.statementPublicValid
        (surface.canonicalStatement steps image) ∧
      surface.proofBoundary proof =
        surface.statementBoundary
          (surface.canonicalStatement steps image) ∧
      ∃ publicIO : surface.TerminalVerifierPublicIO,
        surface.terminalVerifierPublicIO
          (surface.terminalCommittedProof proof) =
            some publicIO ∧
          surface.terminalPublicValuesPrefix
            (surface.terminalPublicValues
              (surface.canonicalStatement steps image))
            publicIO ∧
          surface.terminalBoundaryValuesSuffix
            (surface.statementBoundary
              (surface.canonicalStatement steps image))
            publicIO

/-- Fixed authority opener induced by the runtime backend surface. -/
def authorityOpenerOfRuntimeBackend
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeBackendSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
      (PriorProof := PriorProof)
      ctx where
  openAuthority := surface.openAuthority

/--
Fully exposed evidence returned by runtime backend verifier acceptance.

The evidence form records the replay facts, the proof/canonical-statement
identity derived from replay, verifier public-boundary checks, terminal
public-IO checks, and the opened folded F' authority. This keeps transcript
replay visible as statement binding rather than letting it disappear behind the
backend soundness assumption.
-/
def AcceptedRuntimeBackendEvidence
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeBackendSurface (PriorProof := PriorProof) ctx)
    (steps : Nat)
    (proof : PriorProof)
    (image : PublicImage Digest Boundary) : Prop :=
  surface.compactImageReplay steps proof image ∧
    surface.construction2BoundaryReplay steps proof image ∧
    surface.transcriptReplay steps proof image ∧
    surface.proofStatement proof =
      surface.canonicalStatement steps image ∧
    surface.statementPublicValid
      (surface.canonicalStatement steps image) ∧
    surface.proofBoundary proof =
      surface.statementBoundary
        (surface.canonicalStatement steps image) ∧
    (∃ publicIO : surface.TerminalVerifierPublicIO,
      surface.terminalVerifierPublicIO
        (surface.terminalCommittedProof proof) =
          some publicIO ∧
      surface.terminalPublicValuesPrefix
        (surface.terminalPublicValues
          (surface.canonicalStatement steps image))
        publicIO ∧
      surface.terminalBoundaryValuesSuffix
        (surface.statementBoundary
          (surface.canonicalStatement steps image))
        publicIO) ∧
    ∃ authority : ProofCarryingPriorProof ctx,
      surface.openAuthority proof = some authority ∧
      FoldedFPrimeAuthority.Accepts
        (Transition :=
          DirectParentOnlyProductionSoundness.Transition
            ctx.toProductionContext)
        (initial := ctx.initial)
        steps
        authority
        image

/--
Runtime verifier acceptance proves the accepted-opens obligation.

This is the F' authority bridge for the backend-shaped verifier: runtime checks
alone do not carry authority, but the backend soundness field converts those
checks into a concrete opened authority that accepts the same public image.
-/
theorem runtimeVerifyPrior_acceptedOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeBackendSurface (PriorProof := PriorProof) ctx) :
    ∀ steps proof image,
      RuntimeVerifyPrior surface steps proof image →
        ∃ authority :
            DirectParentOnlyProductionSoundness.ProofCarryingPriorProof
              ctx.toProductionContext,
          (authorityOpenerOfRuntimeBackend surface).openAuthority proof =
              some authority ∧
            FoldedFPrimeAuthority.Accepts
              (Transition :=
                DirectParentOnlyProductionSoundness.Transition
                  ctx.toProductionContext)
              (initial := ctx.initial)
              steps
              authority
              image := by
  intro steps proof image hVerify
  rcases hVerify with
    ⟨hCompact,
      hBoundaryReplay,
      hTranscript,
      hValid,
      hBoundary,
      hPublicIO⟩
  rcases hPublicIO with ⟨publicIO, hPublicIO, hPrefix, hSuffix⟩
  have hStatement :
      surface.proofStatement proof =
        surface.canonicalStatement steps image :=
    surface.replayBindsProofStatement
      steps
      proof
      image
      hCompact
      hBoundaryReplay
      hTranscript
  exact
    surface.runtimeSound
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
      hPrefix
      hSuffix

/--
Runtime verifier acceptance exposes all authority-relevant evidence.
-/
theorem runtimeVerifyPrior_evidence
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeBackendSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : RuntimeVerifyPrior surface steps proof image) :
    AcceptedRuntimeBackendEvidence surface steps proof image := by
  rcases hVerify with
    ⟨hCompact,
      hBoundaryReplay,
      hTranscript,
      hValid,
      hBoundary,
      hPublicIO⟩
  rcases hPublicIO with ⟨publicIO, hPublicIO, hPrefix, hSuffix⟩
  have hStatement :
      surface.proofStatement proof =
        surface.canonicalStatement steps image :=
    surface.replayBindsProofStatement
      steps
      proof
      image
      hCompact
      hBoundaryReplay
      hTranscript
  rcases
    surface.runtimeSound
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
      hPrefix
      hSuffix with
    ⟨authority, hOpen, hAccepts⟩
  exact
    ⟨hCompact,
      hBoundaryReplay,
      hTranscript,
      hStatement,
      hValid,
      hBoundary,
      ⟨publicIO, hPublicIO, hPrefix, hSuffix⟩,
      ⟨authority, hOpen, hAccepts⟩⟩

/-- Certified prior verifier induced by runtime backend checks. -/
def certifiedPriorVerifierOfRuntimeBackend
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeBackendSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.ofAcceptedOpens
    ctx
    (RuntimeVerifyPrior surface)
    (authorityOpenerOfRuntimeBackend surface)
    (runtimeVerifyPrior_acceptedOpens surface)

/--
If the runtime backend opener returns a concrete authority for an accepted proof,
that exact authority accepts the same `(steps, image)` pair.
-/
theorem runtimeVerifyPrior_openedAuthority_accepts_of_open
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeBackendSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify : RuntimeVerifyPrior surface steps proof image)
    (hOpen : surface.openAuthority proof = some authority) :
    FoldedFPrimeAuthority.Accepts
      (Transition :=
        DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
      (initial := ctx.initial)
      steps
      authority
      image := by
  rcases runtimeVerifyPrior_acceptedOpens surface steps proof image hVerify with
    ⟨openedAuthority, hOpened, hAccepts⟩
  have hAuthority : openedAuthority = authority := by
    have hSome : some openedAuthority = some authority :=
      hOpened.symm.trans hOpen
    cases hSome
    rfl
  cases hAuthority
  exact hAccepts

/-- Every runtime-backend accepted prior proof reaches its claimed image. -/
theorem runtimeVerifyPrior_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeBackendSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : RuntimeVerifyPrior surface steps proof image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image := by
  rcases runtimeVerifyPrior_acceptedOpens surface steps proof image hVerify with
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

/-- A runtime-backend verifier cannot accept an unreachable prior image. -/
theorem runtimeVerifyPrior_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeBackendSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : RuntimeVerifyPrior surface steps proof image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable (runtimeVerifyPrior_reaches_prior surface hVerify)

/--
Runtime backend prior-plus-latest end-to-end theorem.

The caller supplies runtime verifier acceptance and latest-step acceptance.
Lean packages the certified verifier internally and returns the existing
terminal end-to-end package, including parent-only CE binding, no-swap, stage
audit, and public-image invariants.
-/
theorem certifiedSingleTerminalEndToEnd_ofRuntimeBackendLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeBackendSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPrior surface priorSteps priorProof priorImage)
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
      (certifiedPriorVerifierOfRuntimeBackend surface).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.certifiedSingleTerminalEndToEnd_ofCertifiedPriorVerifierLatestStep
    (certifiedPriorVerifierOfRuntimeBackend surface)
    hPrior
    hLatest

/--
Runtime backend projection to exact non-aggregate private DEC and stage facts.
-/
theorem nonAggregatePrivateDecStageFacts_ofRuntimeBackendLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeBackendSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPrior surface priorSteps priorProof priorImage)
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
    (certifiedPriorVerifierOfRuntimeBackend surface)
    hPrior
    hLatest

/-- Runtime backend projection to the Section 7.1 owner-target audit. -/
theorem section71StageTargetAuditTrail_ofRuntimeBackendLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeBackendSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPrior surface priorSteps priorProof priorImage)
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
    (certifiedPriorVerifierOfRuntimeBackend surface)
    hPrior
    hLatest

end DirectParentOnlyProductionConcreteFPrimePriorBackend

end DirectCcsFPrime
