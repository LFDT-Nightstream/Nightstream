import DirectCcsFPrime.ProofSystem.Production.Security.Opening.ExactIOOpening

/-!
Backend-shaped opening certificate for the production prior F' verifier.

This module gives implementers a flat surface matching the exact public-IO
backend verifier, but replaces the old monolithic `exactRuntimeSound`
obligation with exact opener and opened-authority binding obligations.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorBackendOpening

/-- Section 7.1-backed production context. -/
abbrev ProductionContext :=
  DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.ProductionContext

/-- Public image for the parent-only production path. -/
abbrev PublicImage :=
  DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.PublicImage

/-- Proof-carrying folded prior authority for the induced production context. -/
abbrev ProofCarryingPriorProof :=
  @DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.ProofCarryingPriorProof

/-- Structured terminal/boundary public IO returned by the production verifier. -/
abbrev ExactTerminalBoundaryPublicIO :=
  DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.ExactTerminalBoundaryPublicIO

/-- Verifier-visible exact public-IO checks. -/
abbrev ConcreteExactPublicIOVerifierChecks :=
  @DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.ConcreteExactPublicIOVerifierChecks

/-- Bound exact public-IO acceptance for the canonical statement. -/
abbrev ExactPublicIOBoundStatementAccepted :=
  @DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.ExactPublicIOBoundStatementAccepted

/-- Opening-level exact public-IO surface. -/
abbrev ConcreteExactPublicIOOpeningSurface :=
  @DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.ConcreteExactPublicIOOpeningSurface

/--
Backend-shaped exact public-IO opening surface.

The verifier data matches the exact public-IO backend shape. The authority
obligations are deliberately split: accepted exact statements must open through
the fixed opener, and any opened authority for that exact statement must bind
the same `(steps, image)` pair.
-/
structure ConcreteRuntimeExactPublicIOOpeningSurface
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
  terminalLengthBindsExpected :
    ∀ steps proof image publicIO,
      terminalVerifierPublicIO
        (terminalCommittedProof proof) =
          some publicIO →
      publicIO.raw =
        terminalPublicValues
          (canonicalStatement steps image) ++
        boundaryPublicValues
          (statementBoundary
            (canonicalStatement steps image)) →
      publicIO.terminal.length =
        (terminalPublicValues
          (canonicalStatement steps image)).length
  replayBindsProofStatement :
    ∀ steps proof image,
      compactImageReplay steps proof image →
      construction2BoundaryReplay steps proof image →
      transcriptReplay steps proof image →
        proofStatement proof = canonicalStatement steps image
  exactBoundStatementOpens :
    ∀ steps proof image publicIO,
      compactImageReplay steps proof image →
      construction2BoundaryReplay steps proof image →
      transcriptReplay steps proof image →
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
      proofStatement proof = canonicalStatement steps image →
        ∃ authority : ProofCarryingPriorProof ctx,
          openAuthority proof = some authority
  openedAuthorityBindsExactStatement :
    ∀ steps proof image publicIO authority,
      compactImageReplay steps proof image →
      construction2BoundaryReplay steps proof image →
      transcriptReplay steps proof image →
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
      proofStatement proof = canonicalStatement steps image →
      openAuthority proof = some authority →
        authority.steps = steps ∧ authority.image = image

/-- Exact public-IO checks extracted from the backend-shaped opening surface. -/
def exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx) :
    ConcreteExactPublicIOVerifierChecks (PriorProof := PriorProof) ctx where
  Statement := surface.Statement
  PublicBoundary := surface.PublicBoundary
  PublicField := surface.PublicField
  TerminalCommittedProof := surface.TerminalCommittedProof
  canonicalStatement := surface.canonicalStatement
  proofStatement := surface.proofStatement
  statementBoundary := surface.statementBoundary
  proofBoundary := surface.proofBoundary
  terminalPublicValues := surface.terminalPublicValues
  boundaryPublicValues := surface.boundaryPublicValues
  terminalCommittedProof := surface.terminalCommittedProof
  statementPublicValid := surface.statementPublicValid
  terminalVerifierPublicIO := surface.terminalVerifierPublicIO
  compactImageReplay := surface.compactImageReplay
  construction2BoundaryReplay := surface.construction2BoundaryReplay
  transcriptReplay := surface.transcriptReplay
  openAuthority := surface.openAuthority

/--
Exact verifier checks, stated directly over the backend-shaped surface.
-/
def RuntimeVerifyPriorOfRuntimeExactPublicIOChecks
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx) :
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
      ∃ publicIO : ExactTerminalBoundaryPublicIO surface.PublicField,
        surface.terminalVerifierPublicIO
          (surface.terminalCommittedProof proof) =
            some publicIO ∧
          publicIO.terminal =
            surface.terminalPublicValues
              (surface.canonicalStatement steps image) ∧
          publicIO.boundary =
            surface.boundaryPublicValues
              (surface.statementBoundary
                (surface.canonicalStatement steps image))

/-- Terminal-length binding induced by the backend-shaped opening surface. -/
def terminalLengthBindingOfRuntimeExactPublicIOOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx) :
    DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.ExactPublicIOTerminalLengthBinding
      (exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening surface) where
  terminalLengthBindsExpected := by
    intro steps proof image publicIO hPublicIO hRaw
    exact
      surface.terminalLengthBindsExpected
        steps
        proof
        image
        publicIO
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hPublicIO)
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hRaw)

/-- Exact opening surface induced by the backend-shaped opening surface. -/
def exactPublicIOOpeningSurfaceOfRuntimeExactPublicIOOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx) :
    ConcreteExactPublicIOOpeningSurface (PriorProof := PriorProof) ctx where
  checks := exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening surface
  terminalLengthBinding :=
    terminalLengthBindingOfRuntimeExactPublicIOOpening surface
  replayBindsProofStatement := by
    intro steps proof image hCompact hBoundaryReplay hTranscript
    exact
      surface.replayBindsProofStatement
        steps
        proof
        image
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hCompact)
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hBoundaryReplay)
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hTranscript)
  exactBoundStatementOpens := by
    intro steps proof image publicIO hBound
    rcases hBound with ⟨hAccepted, hStatement⟩
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
      surface.exactBoundStatementOpens
        steps
        proof
        image
        publicIO
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hCompact)
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hBoundaryReplay)
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hTranscript)
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hValid)
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hBoundary)
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hPublicIO)
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hTerminal)
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hBoundaryValues)
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hStatement)
  openedAuthorityBindsExactStatement := by
    intro steps proof image publicIO authority hBound hOpen
    rcases hBound with ⟨hAccepted, hStatement⟩
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
      surface.openedAuthorityBindsExactStatement
        steps
        proof
        image
        publicIO
        authority
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hCompact)
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hBoundaryReplay)
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hTranscript)
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hValid)
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hBoundary)
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hPublicIO)
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hTerminal)
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hBoundaryValues)
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hStatement)
        (by
          simpa [exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening]
            using hOpen)

/-- Runtime verifier predicate induced by backend-shaped opening evidence. -/
def RuntimeVerifyPriorOfRuntimeExactPublicIOOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx) :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop :=
  DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.RuntimeVerifyPriorOfExactPublicIOOpening
    (exactPublicIOOpeningSurfaceOfRuntimeExactPublicIOOpening surface)

/-- Certified prior verifier induced by backend-shaped opening evidence. -/
def certifiedPriorVerifierOfRuntimeExactPublicIOOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.certifiedPriorVerifierOfExactPublicIOOpening
    (exactPublicIOOpeningSurfaceOfRuntimeExactPublicIOOpening surface)

/--
Exact backend checks imply the raw-vector opening verifier predicate.
-/
theorem runtimeVerifyPriorOfRuntimeExactPublicIOOpening_ofExactPublicIOChecks
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfRuntimeExactPublicIOChecks
        surface
        steps
        proof
        image) :
    RuntimeVerifyPriorOfRuntimeExactPublicIOOpening
      surface
      steps
      proof
      image := by
  rcases hVerify with
    ⟨hCompact,
      hBoundaryReplay,
      hTranscript,
      hValid,
      hBoundary,
      hPublicIO⟩
  rcases hPublicIO with
    ⟨publicIO, hPublicIO, hTerminal, hBoundaryValues⟩
  unfold RuntimeVerifyPriorOfRuntimeExactPublicIOOpening
  unfold DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.RuntimeVerifyPriorOfExactPublicIOOpening
  unfold DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.RuntimeVerifyPriorOfRawPublicIOOpening
  unfold DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.RuntimeVerifyPriorOfRawPublicIOSoundness
  unfold DirectParentOnlyProductionConcreteFPrimePriorRawIO.RuntimeVerifyPriorOfRawPublicIO
  refine
    ⟨by
      simpa [
        DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.rawPublicIOSoundnessSurfaceOfExactPublicIOOpening,
        DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.rawPublicIOOpeningSurfaceOfExactPublicIOOpening,
        exactPublicIOOpeningSurfaceOfRuntimeExactPublicIOOpening,
        exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening,
        DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.runtimeRawPublicIOSurfaceOfSoundness]
        using hCompact,
    by
      simpa [
        DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.rawPublicIOSoundnessSurfaceOfExactPublicIOOpening,
        DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.rawPublicIOOpeningSurfaceOfExactPublicIOOpening,
        exactPublicIOOpeningSurfaceOfRuntimeExactPublicIOOpening,
        exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening,
        DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.runtimeRawPublicIOSurfaceOfSoundness]
        using hBoundaryReplay,
    by
      simpa [
        DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.rawPublicIOSoundnessSurfaceOfExactPublicIOOpening,
        DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.rawPublicIOOpeningSurfaceOfExactPublicIOOpening,
        exactPublicIOOpeningSurfaceOfRuntimeExactPublicIOOpening,
        exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening,
        DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.runtimeRawPublicIOSurfaceOfSoundness]
        using hTranscript,
    by
      simpa [
        DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.rawPublicIOSoundnessSurfaceOfExactPublicIOOpening,
        DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.rawPublicIOOpeningSurfaceOfExactPublicIOOpening,
        exactPublicIOOpeningSurfaceOfRuntimeExactPublicIOOpening,
        exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening,
        DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.runtimeRawPublicIOSurfaceOfSoundness]
        using hValid,
    by
      simpa [
        DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.rawPublicIOSoundnessSurfaceOfExactPublicIOOpening,
        DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.rawPublicIOOpeningSurfaceOfExactPublicIOOpening,
        exactPublicIOOpeningSurfaceOfRuntimeExactPublicIOOpening,
        exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening,
        DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.runtimeRawPublicIOSurfaceOfSoundness]
        using hBoundary,
    ?_⟩
  refine ⟨publicIO.raw, ?_, ?_⟩
  · simp [
      DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening.rawPublicIOSoundnessSurfaceOfOpening,
      DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.rawPublicIOOpeningSurfaceOfExactPublicIOOpening,
      DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.rawPublicIOVerifierChecksOfExactPublicIOOpening,
      exactPublicIOOpeningSurfaceOfRuntimeExactPublicIOOpening,
      exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening,
      DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.runtimeRawPublicIOSurfaceOfSoundness,
      hPublicIO]
  · calc
      publicIO.raw = publicIO.terminal ++ publicIO.boundary :=
        publicIO.raw_eq
      _ =
          surface.terminalPublicValues
              (surface.canonicalStatement steps image) ++
            surface.boundaryPublicValues
              (surface.statementBoundary
                (surface.canonicalStatement steps image)) := by
        rw [hTerminal, hBoundaryValues]

/--
Backend-shaped opening acceptance opens folded F' authority for the same public
pair.
-/
theorem runtimeVerifyPriorOfRuntimeExactPublicIOOpening_acceptedOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx) :
    ∀ steps proof image,
      RuntimeVerifyPriorOfRuntimeExactPublicIOOpening
        surface
        steps
        proof
        image →
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
  intro steps proof image hVerify
  rcases
    DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.runtimeVerifyPriorOfExactPublicIOOpening_acceptedOpens
      (exactPublicIOOpeningSurfaceOfRuntimeExactPublicIOOpening surface)
      steps
      proof
      image
      hVerify with
    ⟨authority, hOpen, hAccepts⟩
  exact
    ⟨authority,
      by
        simpa [exactPublicIOOpeningSurfaceOfRuntimeExactPublicIOOpening,
          exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening] using hOpen,
      hAccepts⟩

/--
If the fixed backend opener returns a concrete authority for an accepted proof,
that authority accepts the same `(steps, image)` pair.
-/
theorem runtimeVerifyPriorOfRuntimeExactPublicIOOpening_openedAuthority_accepts_of_open
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      RuntimeVerifyPriorOfRuntimeExactPublicIOOpening
        surface
        steps
        proof
        image)
    (hOpen : surface.openAuthority proof = some authority) :
    FoldedFPrimeAuthority.Accepts
      (Transition :=
        DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
      (initial := ctx.initial)
      steps
      authority
      image := by
  exact
    DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.runtimeVerifyPriorOfExactPublicIOOpening_openedAuthority_accepts_of_open
      (exactPublicIOOpeningSurfaceOfRuntimeExactPublicIOOpening surface)
      hVerify
      (by
        simpa [exactPublicIOOpeningSurfaceOfRuntimeExactPublicIOOpening,
          exactPublicIOVerifierChecksOfRuntimeExactPublicIOOpening] using hOpen)

/-- Backend-shaped opening acceptance cannot succeed if the fixed opener fails. -/
theorem runtimeVerifyPriorOfRuntimeExactPublicIOOpening_cannot_accept_without_opening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfRuntimeExactPublicIOOpening
        surface
        steps
        proof
        image)
    (hNone : surface.openAuthority proof = none) :
    False := by
  rcases
    runtimeVerifyPriorOfRuntimeExactPublicIOOpening_acceptedOpens
      surface
      steps
      proof
      image
      hVerify with
    ⟨authority, hOpen, _⟩
  rw [hNone] at hOpen
  cases hOpen

/-- Backend-shaped opening acceptance reaches its claimed prior image. -/
theorem runtimeVerifyPriorOfRuntimeExactPublicIOOpening_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfRuntimeExactPublicIOOpening
        surface
        steps
        proof
        image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image :=
  DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.runtimeVerifyPriorOfExactPublicIOOpening_reaches_prior
    (exactPublicIOOpeningSurfaceOfRuntimeExactPublicIOOpening surface)
    hVerify

/-- Backend-shaped opening verifier is same-proof functional. -/
theorem proofFunctionalOfRuntimeExactPublicIOOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx) :
    CompressedFPrimeAuthority.ProofFunctional
      (RuntimeVerifyPriorOfRuntimeExactPublicIOOpening surface) :=
  DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.proofFunctionalOfExactPublicIOOpening
    (exactPublicIOOpeningSurfaceOfRuntimeExactPublicIOOpening surface)

/-- Backend-shaped opening acceptance exposes prior public-image invariants. -/
theorem runtimeVerifyPriorOfRuntimeExactPublicIOOpening_publicImageInvariants
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfRuntimeExactPublicIOOpening
        surface
        steps
        proof
        image) :
    DirectParentOnlyProductionConcreteFPrimePrior.AcceptedPriorPublicImageInvariants
      ctx
      steps
      image :=
  DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.runtimeVerifyPriorOfExactPublicIOOpening_publicImageInvariants
    (exactPublicIOOpeningSurfaceOfRuntimeExactPublicIOOpening surface)
    hVerify

/-- Backend-shaped opening acceptance cannot authorize an unreachable image. -/
theorem runtimeVerifyPriorOfRuntimeExactPublicIOOpening_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfRuntimeExactPublicIOOpening
        surface
        steps
        proof
        image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable
    (runtimeVerifyPriorOfRuntimeExactPublicIOOpening_reaches_prior
      surface
      hVerify)

/--
Direct backend checks open folded F' authority for the same public pair.
-/
theorem runtimeVerifyPriorOfRuntimeExactPublicIOChecks_acceptedOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx) :
    ∀ steps proof image,
      RuntimeVerifyPriorOfRuntimeExactPublicIOChecks
        surface
        steps
        proof
        image →
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
  intro steps proof image hVerify
  exact
    runtimeVerifyPriorOfRuntimeExactPublicIOOpening_acceptedOpens
      surface
      steps
      proof
      image
      (runtimeVerifyPriorOfRuntimeExactPublicIOOpening_ofExactPublicIOChecks
        surface
        hVerify)

/--
If direct backend checks accept and the fixed opener returns an authority, that
authority accepts the same `(steps, image)` pair.
-/
theorem runtimeVerifyPriorOfRuntimeExactPublicIOChecks_openedAuthority_accepts_of_open
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      RuntimeVerifyPriorOfRuntimeExactPublicIOChecks
        surface
        steps
        proof
        image)
    (hOpen : surface.openAuthority proof = some authority) :
    FoldedFPrimeAuthority.Accepts
      (Transition :=
        DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
      (initial := ctx.initial)
      steps
      authority
      image :=
  runtimeVerifyPriorOfRuntimeExactPublicIOOpening_openedAuthority_accepts_of_open
    surface
    (runtimeVerifyPriorOfRuntimeExactPublicIOOpening_ofExactPublicIOChecks
      surface
      hVerify)
    hOpen

/-- Direct backend checks cannot verify if the fixed opener returns `none`. -/
theorem runtimeVerifyPriorOfRuntimeExactPublicIOChecks_cannot_accept_without_opening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfRuntimeExactPublicIOChecks
        surface
        steps
        proof
        image)
    (hNone : surface.openAuthority proof = none) :
    False :=
  runtimeVerifyPriorOfRuntimeExactPublicIOOpening_cannot_accept_without_opening
    surface
    (runtimeVerifyPriorOfRuntimeExactPublicIOOpening_ofExactPublicIOChecks
      surface
      hVerify)
    hNone

/-- Direct backend checks reach their claimed prior image. -/
theorem runtimeVerifyPriorOfRuntimeExactPublicIOChecks_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfRuntimeExactPublicIOChecks
        surface
        steps
        proof
        image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image :=
  runtimeVerifyPriorOfRuntimeExactPublicIOOpening_reaches_prior
    surface
    (runtimeVerifyPriorOfRuntimeExactPublicIOOpening_ofExactPublicIOChecks
      surface
      hVerify)

/-- Direct backend checks expose prior public-image invariants. -/
theorem runtimeVerifyPriorOfRuntimeExactPublicIOChecks_publicImageInvariants
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfRuntimeExactPublicIOChecks
        surface
        steps
        proof
        image) :
    DirectParentOnlyProductionConcreteFPrimePrior.AcceptedPriorPublicImageInvariants
      ctx
      steps
      image :=
  runtimeVerifyPriorOfRuntimeExactPublicIOOpening_publicImageInvariants
    surface
    (runtimeVerifyPriorOfRuntimeExactPublicIOOpening_ofExactPublicIOChecks
      surface
      hVerify)

/-- Direct backend checks cannot authorize an unreachable prior image. -/
theorem runtimeVerifyPriorOfRuntimeExactPublicIOChecks_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfRuntimeExactPublicIOChecks
        surface
        steps
        proof
        image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable
    (runtimeVerifyPriorOfRuntimeExactPublicIOChecks_reaches_prior
      surface
      hVerify)

/-- Direct backend checks are same-proof functional. -/
theorem proofFunctionalOfRuntimeExactPublicIOChecks
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx) :
    CompressedFPrimeAuthority.ProofFunctional
      (RuntimeVerifyPriorOfRuntimeExactPublicIOChecks surface) := by
  intro stepsA stepsB proof imageA imageB hA hB
  exact
    proofFunctionalOfRuntimeExactPublicIOOpening
      surface
      (runtimeVerifyPriorOfRuntimeExactPublicIOOpening_ofExactPublicIOChecks
        surface
        hA)
      (runtimeVerifyPriorOfRuntimeExactPublicIOOpening_ofExactPublicIOChecks
        surface
        hB)

/--
Backend-shaped opening prior-plus-latest end-to-end theorem.
-/
theorem certifiedSingleTerminalEndToEnd_ofRuntimeExactPublicIOOpeningLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfRuntimeExactPublicIOOpening
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
      (certifiedPriorVerifierOfRuntimeExactPublicIOOpening surface).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.certifiedSingleTerminalEndToEnd_ofExactPublicIOOpeningLatestStep
    (exactPublicIOOpeningSurfaceOfRuntimeExactPublicIOOpening surface)
    hPrior
    hLatest

/--
Backend-shaped opening projection to non-aggregate private DEC and stage facts.
-/
theorem nonAggregatePrivateDecStageFacts_ofRuntimeExactPublicIOOpeningLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfRuntimeExactPublicIOOpening
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
  DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.nonAggregatePrivateDecStageFacts_ofExactPublicIOOpeningLatestStep
    (exactPublicIOOpeningSurfaceOfRuntimeExactPublicIOOpening surface)
    hPrior
    hLatest

/--
Backend-shaped opening projection to the Section 7.1 owner-target stage audit.
-/
theorem section71StageTargetAuditTrail_ofRuntimeExactPublicIOOpeningLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfRuntimeExactPublicIOOpening
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
  DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening.section71StageTargetAuditTrail_ofExactPublicIOOpeningLatestStep
    (exactPublicIOOpeningSurfaceOfRuntimeExactPublicIOOpening surface)
    hPrior
    hLatest

/--
Exact backend checks feed the backend-shaped opening end-to-end theorem.
-/
theorem certifiedSingleTerminalEndToEnd_ofRuntimeExactPublicIOChecksLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfRuntimeExactPublicIOChecks
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
      (certifiedPriorVerifierOfRuntimeExactPublicIOOpening surface).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  certifiedSingleTerminalEndToEnd_ofRuntimeExactPublicIOOpeningLatestStep
    surface
    (runtimeVerifyPriorOfRuntimeExactPublicIOOpening_ofExactPublicIOChecks
      surface
      hPrior)
    hLatest

/--
Direct backend checks project to non-aggregate private DEC and stage facts.
-/
theorem nonAggregatePrivateDecStageFacts_ofRuntimeExactPublicIOChecksLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfRuntimeExactPublicIOChecks
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
  nonAggregatePrivateDecStageFacts_ofRuntimeExactPublicIOOpeningLatestStep
    surface
    (runtimeVerifyPriorOfRuntimeExactPublicIOOpening_ofExactPublicIOChecks
      surface
      hPrior)
    hLatest

/--
Direct backend checks project to the Section 7.1 owner-target stage audit.
-/
theorem section71StageTargetAuditTrail_ofRuntimeExactPublicIOChecksLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
        ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfRuntimeExactPublicIOChecks
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
  section71StageTargetAuditTrail_ofRuntimeExactPublicIOOpeningLatestStep
    surface
    (runtimeVerifyPriorOfRuntimeExactPublicIOOpening_ofExactPublicIOChecks
      surface
      hPrior)
    hLatest

/--
Reuse an exact-runtime soundness surface through the stronger opening surface.

This is a compatibility bridge for concrete prior verifier work that already
packages backend SNARK soundness as `exactRuntimeSound`. The bridge also
requires the existing exact public-IO layout binding; raw concatenation alone
does not determine the terminal/boundary split. The bridge does not turn
digests into authority: it derives the fixed opener and opened-authority
binding obligations from the authority returned by `exactRuntimeSound`, using
the functionality of `Option.some`.
-/
def runtimeExactPublicIOOpeningSurfaceOfExactRuntimeSoundAndLayout
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (layout :
      DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.ExactPublicIOLayoutBinding
        surface) :
    ConcreteRuntimeExactPublicIOOpeningSurface (PriorProof := PriorProof)
      ctx where
  Statement := surface.Statement
  PublicBoundary := surface.PublicBoundary
  PublicField := surface.PublicField
  TerminalCommittedProof := surface.TerminalCommittedProof
  canonicalStatement := surface.canonicalStatement
  proofStatement := surface.proofStatement
  statementBoundary := surface.statementBoundary
  proofBoundary := surface.proofBoundary
  terminalPublicValues := surface.terminalPublicValues
  boundaryPublicValues := surface.boundaryPublicValues
  terminalCommittedProof := surface.terminalCommittedProof
  statementPublicValid := surface.statementPublicValid
  terminalVerifierPublicIO := surface.terminalVerifierPublicIO
  compactImageReplay := surface.compactImageReplay
  construction2BoundaryReplay := surface.construction2BoundaryReplay
  transcriptReplay := surface.transcriptReplay
  openAuthority := surface.openAuthority
  terminalLengthBindsExpected := by
    intro steps proof image publicIO hPublicIO hRaw
    rcases
      layout.rawOutputBindsTerminalBoundary
        steps
        proof
        image
        publicIO
        hPublicIO
        hRaw with
      ⟨hTerminal, _hBoundary⟩
    simp [hTerminal]
  replayBindsProofStatement := surface.replayBindsProofStatement
  exactBoundStatementOpens := by
    intro steps proof image publicIO
      hCompact hBoundaryReplay hTranscript hValid hBoundary hPublicIO
      hTerminal hBoundaryValues hStatement
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
      ⟨authority, hOpen, _hAccepts⟩
    exact ⟨authority, hOpen⟩
  openedAuthorityBindsExactStatement := by
    intro steps proof image publicIO authority
      hCompact hBoundaryReplay hTranscript hValid hBoundary hPublicIO
      hTerminal hBoundaryValues hStatement hOpen
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
      ⟨opened, hOpened, hAccepts⟩
    have hSome : some authority = some opened := hOpen.symm.trans hOpened
    cases hSome
    exact hAccepts

/--
The existing exact-runtime verifier predicate feeds the stronger fixed-opener
prior verifier induced by
`runtimeExactPublicIOOpeningSurfaceOfExactRuntimeSoundAndLayout`.
-/
theorem runtimeVerifyPriorOfExactRuntimeSoundAndLayout_toRuntimeExactPublicIOOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (layout :
      DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.ExactPublicIOLayoutBinding
        surface)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.RuntimeVerifyPriorOfExactPublicIO
        surface
        steps
        proof
        image) :
    RuntimeVerifyPriorOfRuntimeExactPublicIOOpening
      (runtimeExactPublicIOOpeningSurfaceOfExactRuntimeSoundAndLayout surface layout)
      steps
      proof
      image := by
  apply runtimeVerifyPriorOfRuntimeExactPublicIOOpening_ofExactPublicIOChecks
  simpa [
    DirectParentOnlyProductionConcreteFPrimePriorBackend.RuntimeVerifyPriorOfExactPublicIO,
    DirectParentOnlyProductionConcreteFPrimePriorBackend.RuntimeVerifyPrior,
    DirectParentOnlyProductionConcreteFPrimePriorBackend.runtimeBackendSurfaceOfExactPublicIO,
    RuntimeVerifyPriorOfRuntimeExactPublicIOChecks,
    runtimeExactPublicIOOpeningSurfaceOfExactRuntimeSoundAndLayout
  ] using hVerify

/--
Accepted exact-runtime verification opens folded F' authority through the fixed
opener supplied by the exact-runtime surface.
-/
theorem runtimeVerifyPriorOfExactRuntimeSoundAndLayout_acceptedOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface
        (PriorProof := PriorProof)
        ctx)
    (layout :
      DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.ExactPublicIOLayoutBinding
        surface) :
    ∀ steps proof image,
      DirectParentOnlyProductionConcreteFPrimePriorBackend.RuntimeVerifyPriorOfExactPublicIO
        surface
        steps
        proof
        image →
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
  intro steps proof image hVerify
  exact
    runtimeVerifyPriorOfRuntimeExactPublicIOOpening_acceptedOpens
      (runtimeExactPublicIOOpeningSurfaceOfExactRuntimeSoundAndLayout surface layout)
      steps
      proof
      image
      (runtimeVerifyPriorOfExactRuntimeSoundAndLayout_toRuntimeExactPublicIOOpening
        surface
        layout
        hVerify)

end DirectParentOnlyProductionConcreteFPrimePriorBackendOpening

end DirectCcsFPrime
