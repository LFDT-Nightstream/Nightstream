import DirectCcsFPrime.ProofSystem.Production.Impl.Backend.DirectParentOnlyProductionConcreteFPrimePriorBackendBase

/-!
Exact terminal/boundary public-IO adapter for the concrete prior F' backend.

This module owns the structured exact public-IO surface and its adapter into
the generic runtime backend authority path.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorBackend

/--
Structured terminal public IO for the concrete committed F' verifier.

The intended Rust verifier ABI receives one public vector from Spartan. The
production check compares that vector to the exact concatenation of terminal F'
public values and the Construction-2 boundary public values. Keeping the split
explicit here prevents an implementation from satisfying independent
prefix/suffix predicates while smuggling unrelated public fields between them.
-/
structure ExactTerminalBoundaryPublicIO (Field : Type) where
  terminal : List Field
  boundary : List Field
  raw : List Field
  raw_eq : raw = terminal ++ boundary

/--
Runtime backend surface for the exact terminal-public-IO layout used by the
production F' verifier.

The verifier-visible public IO is structured as the exact terminal/boundary
concatenation. The backend soundness obligation receives that exact split,
together with the compact-image replay, Construction-2 boundary replay,
transcript replay, and replay-derived canonical statement equality.
-/
structure ConcreteRuntimeExactPublicIOSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) where
  Statement : Type
  PublicBoundary : Type
  PublicField : Type
  TerminalCommittedProof : Type
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
      List PublicField
  boundaryPublicValues :
    PublicBoundary →
      List PublicField
  terminalCommittedProof :
    PriorProof →
      TerminalCommittedProof
  statementPublicValid :
    Statement →
      Prop
  terminalVerifierPublicIO :
    TerminalCommittedProof →
      Option (ExactTerminalBoundaryPublicIO PublicField)
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
  exactRuntimeSound :
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
      publicIO.terminal =
        terminalPublicValues (canonicalStatement steps image) →
      publicIO.boundary =
        boundaryPublicValues
          (statementBoundary (canonicalStatement steps image)) →
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
Adapter from exact public-IO layout to the general runtime backend surface.
-/
def runtimeBackendSurfaceOfExactPublicIO
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx) :
    ConcreteRuntimeBackendSurface (PriorProof := PriorProof) ctx where
  Statement := surface.Statement
  PublicBoundary := surface.PublicBoundary
  TerminalPublicValues := List surface.PublicField
  TerminalCommittedProof := surface.TerminalCommittedProof
  TerminalVerifierPublicIO :=
    ExactTerminalBoundaryPublicIO surface.PublicField
  canonicalStatement := surface.canonicalStatement
  proofStatement := surface.proofStatement
  statementBoundary := surface.statementBoundary
  proofBoundary := surface.proofBoundary
  terminalPublicValues := surface.terminalPublicValues
  terminalCommittedProof := surface.terminalCommittedProof
  statementPublicValid := surface.statementPublicValid
  terminalVerifierPublicIO := surface.terminalVerifierPublicIO
  terminalPublicValuesPrefix := fun expected publicIO =>
    publicIO.terminal = expected
  terminalBoundaryValuesSuffix := fun boundary publicIO =>
    publicIO.boundary = surface.boundaryPublicValues boundary
  compactImageReplay := surface.compactImageReplay
  construction2BoundaryReplay := surface.construction2BoundaryReplay
  transcriptReplay := surface.transcriptReplay
  openAuthority := surface.openAuthority
  replayBindsProofStatement := surface.replayBindsProofStatement
  runtimeSound := by
    intro steps proof image publicIO
      hCompact hBoundaryReplay hTranscript hStatement hValid hBoundary
      hPublicIO hTerminal hBoundaryValues
    exact
      surface.exactRuntimeSound
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

/-- Runtime verifier predicate induced by exact terminal public IO. -/
def RuntimeVerifyPriorOfExactPublicIO
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx) :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop :=
  RuntimeVerifyPrior (runtimeBackendSurfaceOfExactPublicIO surface)

/--
Exact public-IO evidence exposed by accepted runtime verification.
-/
def AcceptedExactPublicIOEvidence
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx)
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
    (∃ publicIO : ExactTerminalBoundaryPublicIO surface.PublicField,
      surface.terminalVerifierPublicIO
        (surface.terminalCommittedProof proof) =
          some publicIO ∧
      publicIO.terminal =
        surface.terminalPublicValues
          (surface.canonicalStatement steps image) ∧
      publicIO.boundary =
        surface.boundaryPublicValues
          (surface.statementBoundary
            (surface.canonicalStatement steps image)) ∧
      publicIO.raw =
        surface.terminalPublicValues
          (surface.canonicalStatement steps image) ++
        surface.boundaryPublicValues
          (surface.statementBoundary
            (surface.canonicalStatement steps image))) ∧
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
Accepted exact public-IO runtime verification exposes replay, exact public IO,
and the opened folded F' authority.
-/
theorem runtimeVerifyPriorOfExactPublicIO_evidence
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIO surface steps proof image) :
    AcceptedExactPublicIOEvidence surface steps proof image := by
  rcases hVerify with
    ⟨hCompact,
      hBoundaryReplay,
      hTranscript,
      hValid,
      hBoundary,
      hPublicIO⟩
  rcases hPublicIO with
    ⟨publicIO, hPublicIO, hTerminal, hBoundaryValues⟩
  change
      publicIO.terminal =
        surface.terminalPublicValues
          (surface.canonicalStatement steps image) at hTerminal
  change
      publicIO.boundary =
        surface.boundaryPublicValues
          (surface.statementBoundary
            (surface.canonicalStatement steps image)) at hBoundaryValues
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
  have hRaw :
      publicIO.raw =
        surface.terminalPublicValues
          (surface.canonicalStatement steps image) ++
        surface.boundaryPublicValues
          (surface.statementBoundary
            (surface.canonicalStatement steps image)) := by
    calc
      publicIO.raw = publicIO.terminal ++ publicIO.boundary :=
        publicIO.raw_eq
      _ =
          surface.terminalPublicValues
              (surface.canonicalStatement steps image) ++
            surface.boundaryPublicValues
              (surface.statementBoundary
                (surface.canonicalStatement steps image)) := by
        rw [hTerminal, hBoundaryValues]
  rcases
    surface.exactRuntimeSound
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
      hBoundaryValues with
    ⟨authority, hOpen, hAccepts⟩
  exact
    ⟨hCompact,
      hBoundaryReplay,
      hTranscript,
      hStatement,
      hValid,
      hBoundary,
      ⟨publicIO, hPublicIO, hTerminal, hBoundaryValues, hRaw⟩,
      ⟨authority, hOpen, hAccepts⟩⟩

/-- Certified prior verifier induced by exact terminal public IO. -/
def certifiedPriorVerifierOfExactPublicIO
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  certifiedPriorVerifierOfRuntimeBackend
    (runtimeBackendSurfaceOfExactPublicIO surface)

/--
Exact-public-IO prior-plus-latest end-to-end theorem.
-/
theorem certifiedSingleTerminalEndToEnd_ofExactPublicIOLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfExactPublicIO
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
      (certifiedPriorVerifierOfExactPublicIO surface).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  certifiedSingleTerminalEndToEnd_ofRuntimeBackendLatestStep
    (runtimeBackendSurfaceOfExactPublicIO surface)
    hPrior
    hLatest

/--
Exact-public-IO projection to non-aggregate private DEC and stage facts.
-/
theorem nonAggregatePrivateDecStageFacts_ofExactPublicIOLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfExactPublicIO
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
  nonAggregatePrivateDecStageFacts_ofRuntimeBackendLatestStep
    (runtimeBackendSurfaceOfExactPublicIO surface)
    hPrior
    hLatest

/-- Exact-public-IO projection to the Section 7.1 owner-target audit. -/
theorem section71StageTargetAuditTrail_ofExactPublicIOLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfExactPublicIO
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
  section71StageTargetAuditTrail_ofRuntimeBackendLatestStep
    (runtimeBackendSurfaceOfExactPublicIO surface)
    hPrior
    hLatest

end DirectParentOnlyProductionConcreteFPrimePriorBackend

end DirectCcsFPrime
