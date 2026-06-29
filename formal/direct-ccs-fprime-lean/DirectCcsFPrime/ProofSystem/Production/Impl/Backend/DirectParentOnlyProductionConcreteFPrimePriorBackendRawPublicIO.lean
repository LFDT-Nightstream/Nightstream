import DirectCcsFPrime.ProofSystem.Production.Impl.Backend.DirectParentOnlyProductionConcreteFPrimePriorBackendExactPublicIO

/-!
Raw public-vector adapter for the concrete prior F' backend.

This module owns the raw terminal/boundary public-vector verifier surface and
its certified prior-verifier consequences.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorBackend

/--
Runtime backend surface for the raw public vector returned by the production
terminal committed F' verifier.

This specifies the intended raw Rust verifier ABI: the terminal verifier returns
one raw public vector, and the verifier accepts only when that vector is exactly
the expected concatenation of terminal F' public values and the Construction-2
boundary public values. The authority opening remains a backend soundness
consequence, not a verifier predicate supplied by callers.
-/
structure ConcreteRuntimeRawPublicIOSurface
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
      Option (List PublicField)
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
  rawRuntimeSound :
    ∀ steps proof image rawPublicIO,
      compactImageReplay steps proof image →
      construction2BoundaryReplay steps proof image →
      transcriptReplay steps proof image →
      proofStatement proof = canonicalStatement steps image →
      statementPublicValid (canonicalStatement steps image) →
      proofBoundary proof =
        statementBoundary (canonicalStatement steps image) →
      terminalVerifierPublicIO (terminalCommittedProof proof) =
        some rawPublicIO →
      rawPublicIO =
        terminalPublicValues (canonicalStatement steps image) ++
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
Raw-vector runtime verifier predicate.

The predicate contains only checks available to the verifier: replay,
canonical-statement validity, proof/statement boundary equality, and exact
terminal committed verifier public-vector equality.
-/
def RuntimeVerifyPriorOfRawPublicIO
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx) :
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
      ∃ rawPublicIO : List surface.PublicField,
        surface.terminalVerifierPublicIO
          (surface.terminalCommittedProof proof) =
            some rawPublicIO ∧
          rawPublicIO =
            surface.terminalPublicValues
              (surface.canonicalStatement steps image) ++
            surface.boundaryPublicValues
              (surface.statementBoundary
                (surface.canonicalStatement steps image))

/-- Fixed authority opener induced by the raw-vector runtime backend. -/
def authorityOpenerOfRawPublicIO
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
      (PriorProof := PriorProof)
      ctx where
  openAuthority := surface.openAuthority

/--
Raw-vector runtime verifier acceptance proves the accepted-opens obligation.
-/
theorem runtimeVerifyPriorOfRawPublicIO_acceptedOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx) :
    ∀ steps proof image,
      RuntimeVerifyPriorOfRawPublicIO surface steps proof image →
        ∃ authority :
            DirectParentOnlyProductionSoundness.ProofCarryingPriorProof
              ctx.toProductionContext,
          (authorityOpenerOfRawPublicIO surface).openAuthority proof =
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
      hRawPublicIO⟩
  rcases hRawPublicIO with
    ⟨rawPublicIO, hRawPublicIO, hRawEq⟩
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
    surface.rawRuntimeSound
      steps
      proof
      image
      rawPublicIO
      hCompact
      hBoundaryReplay
      hTranscript
      hStatement
      hValid
      hBoundary
      hRawPublicIO
      hRawEq

/--
Fully exposed evidence returned by raw-vector runtime verifier acceptance.
-/
def AcceptedRawPublicIOEvidence
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx)
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
    (∃ rawPublicIO : List surface.PublicField,
      surface.terminalVerifierPublicIO
        (surface.terminalCommittedProof proof) =
          some rawPublicIO ∧
      rawPublicIO =
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
Accepted raw-vector runtime verification exposes replay, exact raw public IO,
and the opened folded F' authority.
-/
theorem runtimeVerifyPriorOfRawPublicIO_evidence
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfRawPublicIO surface steps proof image) :
    AcceptedRawPublicIOEvidence surface steps proof image := by
  rcases hVerify with
    ⟨hCompact,
      hBoundaryReplay,
      hTranscript,
      hValid,
      hBoundary,
      hRawPublicIO⟩
  rcases hRawPublicIO with
    ⟨rawPublicIO, hRawPublicIO, hRawEq⟩
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
    surface.rawRuntimeSound
      steps
      proof
      image
      rawPublicIO
      hCompact
      hBoundaryReplay
      hTranscript
      hStatement
      hValid
      hBoundary
      hRawPublicIO
      hRawEq with
    ⟨authority, hOpen, hAccepts⟩
  exact
    ⟨hCompact,
      hBoundaryReplay,
      hTranscript,
      hStatement,
      hValid,
      hBoundary,
      ⟨rawPublicIO, hRawPublicIO, hRawEq⟩,
      ⟨authority, hOpen, hAccepts⟩⟩

/-- Certified prior verifier induced by raw terminal public IO equality. -/
def certifiedPriorVerifierOfRawPublicIO
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.ofAcceptedOpens
    ctx
    (RuntimeVerifyPriorOfRawPublicIO surface)
    (authorityOpenerOfRawPublicIO surface)
    (runtimeVerifyPriorOfRawPublicIO_acceptedOpens surface)

/--
If the raw-vector backend opener returns a concrete authority for an accepted
proof, that exact authority accepts the same `(steps, image)` pair.
-/
theorem runtimeVerifyPriorOfRawPublicIO_openedAuthority_accepts_of_open
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      RuntimeVerifyPriorOfRawPublicIO surface steps proof image)
    (hOpen : surface.openAuthority proof = some authority) :
    FoldedFPrimeAuthority.Accepts
      (Transition :=
        DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
      (initial := ctx.initial)
      steps
      authority
      image := by
  rcases
    runtimeVerifyPriorOfRawPublicIO_acceptedOpens
      surface
      steps
      proof
      image
      hVerify with
    ⟨openedAuthority, hOpened, hAccepts⟩
  have hAuthority : openedAuthority = authority := by
    have hSome : some openedAuthority = some authority :=
      hOpened.symm.trans hOpen
    cases hSome
    rfl
  cases hAuthority
  exact hAccepts

/-- Every raw-vector accepted prior proof reaches its claimed image. -/
theorem runtimeVerifyPriorOfRawPublicIO_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfRawPublicIO surface steps proof image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image := by
  rcases
    runtimeVerifyPriorOfRawPublicIO_acceptedOpens
      surface
      steps
      proof
      image
      hVerify with
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

/--
Raw-vector prior-plus-latest end-to-end theorem.
-/
theorem certifiedSingleTerminalEndToEnd_ofRawPublicIOLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeRawPublicIOSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfRawPublicIO
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
      (certifiedPriorVerifierOfRawPublicIO surface).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.certifiedSingleTerminalEndToEnd_ofCertifiedPriorVerifierLatestStep
    (certifiedPriorVerifierOfRawPublicIO surface)
    hPrior
    hLatest


end DirectParentOnlyProductionConcreteFPrimePriorBackend

end DirectCcsFPrime
