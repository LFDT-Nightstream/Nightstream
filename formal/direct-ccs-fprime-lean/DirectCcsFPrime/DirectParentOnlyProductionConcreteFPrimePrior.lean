import DirectCcsFPrime.DirectParentOnlyProductionSuperNeoReuseEndToEnd

/-!
Concrete prior F' verifier authority for the Section 7.1-backed endpoint.

This module owns the focused F' authority bridge. It models the implementation
verifier body by the checks that matter for authority: compact-image replay,
Construction-2 boundary replay, Fiat-Shamir/transcript replay, committed-step
verifier acceptance, and a fixed opener for folded F' reachability. Digest
replay is not authority by itself; an accepted proof must open to proof-carrying
folded authority for the same `(steps, image)` pair.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePrior

/-- Section 7.1-backed production context. -/
abbrev ProductionContext :=
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.ProductionContext

/-- Public image for the parent-only production path. -/
abbrev PublicImage (Digest Boundary : Type) :=
  DirectParentOnlyProductionSoundness.PublicImage Digest Boundary

/-- Proof-carrying folded prior authority for the induced production context. -/
abbrev ProofCarryingPriorProof
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) :=
  DirectParentOnlyProductionSoundness.ProofCarryingPriorProof
    ctx.toProductionContext

/--
Concrete verifier-body contract for the compressed prior F' proof.

The predicates are the implementation checks that may be backed by Poseidon2
transcript replay and the committed-step verifier. The final field is the
authority theorem: when those checks all hold and the fixed opener returns an
authority object, that authority accepts the same `(steps, image)` pair.
-/
structure ConcreteVerifierBody
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) where
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
  committedStepAccepted :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop
  openAuthority :
    PriorProof →
      Option (ProofCarryingPriorProof ctx)
  acceptedBodyOpens :
    ∀ steps proof image authority,
      compactImageReplay steps proof image →
      construction2BoundaryReplay steps proof image →
      transcriptReplay steps proof image →
      committedStepAccepted steps proof image →
      openAuthority proof = some authority →
        FoldedFPrimeAuthority.Accepts
          (Transition :=
            DirectParentOnlyProductionSoundness.Transition
              ctx.toProductionContext)
          (initial := ctx.initial)
          steps
          authority
          image

/-- Concrete prior verifier induced by the implementation verifier body. -/
abbrev ConcretePriorVerifier :=
  @ConcreteVerifierBody

/--
Rust-shaped verifier-body checks before packaging as a certified verifier.

The first three replay predicates model the compact-image, Construction-2
boundary, and Fiat-Shamir transcript checks that bind the public verifier
statement. The committed-step verifier is then sound only for that bound
statement. This separates digest/transcript replay from authority: replay binds
the statement, while committed-step soundness plus the fixed opener supplies
folded F' authority.
-/
structure ConcreteVerifierBodyChecks
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) where
  verifierStatement :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
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
  committedStepAccepted :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop
  openAuthority :
    PriorProof →
      Option (ProofCarryingPriorProof ctx)
  replayBindsStatement :
    ∀ steps proof image,
      compactImageReplay steps proof image →
      construction2BoundaryReplay steps proof image →
      transcriptReplay steps proof image →
        verifierStatement steps proof image
  committedStepVerifierSound :
    ∀ steps proof image authority,
      verifierStatement steps proof image →
      committedStepAccepted steps proof image →
      openAuthority proof = some authority →
        FoldedFPrimeAuthority.Accepts
          (Transition :=
            DirectParentOnlyProductionSoundness.Transition
              ctx.toProductionContext)
          (initial := ctx.initial)
          steps
          authority
          image

/--
Canonical-statement binding surface for the implementation-facing verifier ABI.

The implementation verifier first reconstructs the statement that should be
verified from the caller's `(steps, image)` pair, then checks that the opaque
proof is verified against that same statement. The statement type stays
abstract here, but the equality is not: replay must bind the proof statement to
the canonical statement for the exact public pair.
-/
structure ConcreteVerifierStatementBinding
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) where
  Statement : Type
  canonicalStatement :
    Nat →
      PublicImage Digest Boundary →
        Statement
  proofStatement :
    PriorProof →
      Statement
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
  committedStepAccepted :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop
  openAuthority :
    PriorProof →
      Option (ProofCarryingPriorProof ctx)
  replayBindsCanonicalStatement :
    ∀ steps proof image,
      compactImageReplay steps proof image →
      construction2BoundaryReplay steps proof image →
      transcriptReplay steps proof image →
        proofStatement proof = canonicalStatement steps image
  committedCanonicalStatementSound :
    ∀ steps proof image authority,
      proofStatement proof = canonicalStatement steps image →
      committedStepAccepted steps proof image →
      openAuthority proof = some authority →
        FoldedFPrimeAuthority.Accepts
          (Transition :=
            DirectParentOnlyProductionSoundness.Transition
              ctx.toProductionContext)
          (initial := ctx.initial)
        steps
        authority
        image

/--
Statement-level verifier surface matching the direct CCS Rust verifier contract.

The intended Rust path reconstructs a public statement from the expected public
image, validates its final Construction-2 boundary, checks that the opaque proof
uses the same public boundary, and then verifies the terminal committed proof
against both the terminal public values and that boundary. The final soundness
field is the single cryptographic backend obligation: successful committed
verification for that exact public IO opens to folded F' reachability authority.
-/
structure ConcreteVerifierStatementSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) where
  Statement : Type
  PublicBoundary : Type
  TerminalPublicValues : Type
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
      TerminalPublicValues
  terminalCommittedProof :
    PriorProof →
      TerminalCommittedProof
  statementPublicValid :
    Statement →
      Prop
  terminalVerifierAccepted :
    TerminalPublicValues →
      PublicBoundary →
      TerminalCommittedProof →
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
  terminalVerifierSound :
    ∀ steps proof image authority,
      statementPublicValid (canonicalStatement steps image) →
      proofBoundary proof =
        statementBoundary (canonicalStatement steps image) →
      terminalVerifierAccepted
        (terminalPublicValues (canonicalStatement steps image))
        (statementBoundary (canonicalStatement steps image))
        (terminalCommittedProof proof) →
      openAuthority proof = some authority →
        FoldedFPrimeAuthority.Accepts
          (Transition :=
            DirectParentOnlyProductionSoundness.Transition
              ctx.toProductionContext)
          (initial := ctx.initial)
          steps
          authority
          image

/--
Turn Rust-shaped verifier checks plus committed-step soundness into the
`ConcreteVerifierBody` consumed by the production endpoint.
-/
def concreteVerifierBodyOfChecks
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ConcreteVerifierBodyChecks (PriorProof := PriorProof) ctx) :
    ConcreteVerifierBody (PriorProof := PriorProof) ctx where
  compactImageReplay := checks.compactImageReplay
  construction2BoundaryReplay := checks.construction2BoundaryReplay
  transcriptReplay := checks.transcriptReplay
  committedStepAccepted := checks.committedStepAccepted
  openAuthority := checks.openAuthority
  acceptedBodyOpens := by
    intro steps proof image authority hCompact hBoundary hTranscript hCommitted hOpen
    exact
      checks.committedStepVerifierSound
        steps
        proof
        image
        authority
        (checks.replayBindsStatement
          steps
          proof
          image
          hCompact
          hBoundary
          hTranscript)
        hCommitted
        hOpen

/--
Turn canonical-statement replay into the checks-first verifier surface.
-/
def concreteVerifierBodyChecksOfStatementBinding
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (binding :
      ConcreteVerifierStatementBinding (PriorProof := PriorProof) ctx) :
    ConcreteVerifierBodyChecks (PriorProof := PriorProof) ctx where
  verifierStatement := fun steps proof image =>
    binding.proofStatement proof =
      binding.canonicalStatement steps image
  compactImageReplay := binding.compactImageReplay
  construction2BoundaryReplay := binding.construction2BoundaryReplay
  transcriptReplay := binding.transcriptReplay
  committedStepAccepted := binding.committedStepAccepted
  openAuthority := binding.openAuthority
  replayBindsStatement := by
    intro steps proof image hCompact hBoundary hTranscript
    exact
      binding.replayBindsCanonicalStatement
        steps
        proof
        image
        hCompact
        hBoundary
        hTranscript
  committedStepVerifierSound := by
    intro steps proof image authority hStatement hCommitted hOpen
    exact
      binding.committedCanonicalStatementSound
        steps
        proof
        image
        authority
        hStatement
        hCommitted
        hOpen

/--
Turn canonical-statement replay directly into the verifier body consumed by
the production endpoint.
-/
def concreteVerifierBodyOfStatementBinding
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (binding :
      ConcreteVerifierStatementBinding (PriorProof := PriorProof) ctx) :
    ConcreteVerifierBody (PriorProof := PriorProof) ctx :=
  concreteVerifierBodyOfChecks
    (concreteVerifierBodyChecksOfStatementBinding binding)

/--
Turn the verifier-shaped statement surface into canonical-statement binding.
-/
def concreteVerifierStatementBindingOfSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteVerifierStatementSurface (PriorProof := PriorProof) ctx) :
    ConcreteVerifierStatementBinding (PriorProof := PriorProof) ctx where
  Statement := surface.Statement
  canonicalStatement := surface.canonicalStatement
  proofStatement := surface.proofStatement
  compactImageReplay := surface.compactImageReplay
  construction2BoundaryReplay := surface.construction2BoundaryReplay
  transcriptReplay := surface.transcriptReplay
  committedStepAccepted := fun steps proof image =>
    surface.statementPublicValid
      (surface.canonicalStatement steps image) ∧
      surface.proofBoundary proof =
        surface.statementBoundary
          (surface.canonicalStatement steps image) ∧
      surface.terminalVerifierAccepted
        (surface.terminalPublicValues
          (surface.canonicalStatement steps image))
        (surface.statementBoundary
          (surface.canonicalStatement steps image))
        (surface.terminalCommittedProof proof)
  openAuthority := surface.openAuthority
  replayBindsCanonicalStatement := by
    intro steps proof image hCompact hBoundary hTranscript
    exact
      surface.replayBindsProofStatement
        steps
        proof
        image
        hCompact
        hBoundary
        hTranscript
  committedCanonicalStatementSound := by
    intro steps proof image authority _hStatement hCommitted hOpen
    rcases hCommitted with ⟨hValid, hBoundaryEq, hTerminal⟩
    exact
      surface.terminalVerifierSound
        steps
        proof
        image
        authority
        hValid
        hBoundaryEq
        hTerminal
        hOpen

/--
Turn the verifier-shaped statement surface into the checks-first verifier
surface.
-/
def concreteVerifierBodyChecksOfStatementSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteVerifierStatementSurface (PriorProof := PriorProof) ctx) :
    ConcreteVerifierBodyChecks (PriorProof := PriorProof) ctx :=
  concreteVerifierBodyChecksOfStatementBinding
    (concreteVerifierStatementBindingOfSurface surface)

/--
Turn the verifier-shaped statement surface directly into the verifier body
consumed by the production endpoint.
-/
def concreteVerifierBodyOfStatementSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteVerifierStatementSurface (PriorProof := PriorProof) ctx) :
    ConcreteVerifierBody (PriorProof := PriorProof) ctx :=
  concreteVerifierBodyOfStatementBinding
    (concreteVerifierStatementBindingOfSurface surface)

/--
Acceptance predicate for the concrete compressed prior F' verifier.

All replay checks must pass, and the fixed opener must return an authority
object. The opened authority is then justified by `acceptedBodyOpens`.
-/
def VerifyPrior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (body : ConcreteVerifierBody (PriorProof := PriorProof) ctx) :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop :=
  fun steps proof image =>
    body.compactImageReplay steps proof image ∧
      body.construction2BoundaryReplay steps proof image ∧
      body.transcriptReplay steps proof image ∧
      body.committedStepAccepted steps proof image ∧
      ∃ authority : ProofCarryingPriorProof ctx,
        body.openAuthority proof = some authority

/-- Acceptance predicate induced directly by Rust-shaped verifier checks. -/
def VerifyPriorOfChecks
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ConcreteVerifierBodyChecks (PriorProof := PriorProof) ctx) :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop :=
  VerifyPrior (concreteVerifierBodyOfChecks checks)

/-- Acceptance predicate induced by canonical-statement replay binding. -/
def VerifyPriorOfStatementBinding
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (binding :
      ConcreteVerifierStatementBinding (PriorProof := PriorProof) ctx) :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop :=
  VerifyPriorOfChecks
    (concreteVerifierBodyChecksOfStatementBinding binding)

/--
Acceptance predicate induced by the verifier-shaped statement surface.
-/
def VerifyPriorOfStatementSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteVerifierStatementSurface (PriorProof := PriorProof) ctx) :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop :=
  VerifyPriorOfStatementBinding
    (concreteVerifierStatementBindingOfSurface surface)

/-- Fixed authority opener induced by the concrete verifier body. -/
def authorityOpener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (body : ConcreteVerifierBody (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
      (PriorProof := PriorProof)
      ctx where
  openAuthority := body.openAuthority

/-- Fixed authority opener induced directly by Rust-shaped verifier checks. -/
def authorityOpenerOfChecks
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ConcreteVerifierBodyChecks (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
      (PriorProof := PriorProof)
      ctx :=
  authorityOpener (concreteVerifierBodyOfChecks checks)

/-- Fixed authority opener induced by canonical-statement replay binding. -/
def authorityOpenerOfStatementBinding
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (binding :
      ConcreteVerifierStatementBinding (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
      (PriorProof := PriorProof)
      ctx :=
  authorityOpenerOfChecks
    (concreteVerifierBodyChecksOfStatementBinding binding)

/-- Fixed authority opener induced by the verifier-shaped statement surface. -/
def authorityOpenerOfStatementSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteVerifierStatementSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
      (PriorProof := PriorProof)
      ctx :=
  authorityOpenerOfStatementBinding
    (concreteVerifierStatementBindingOfSurface surface)

/--
The concrete verifier body proves the accepted-opens obligation.

This is the central F' authority theorem for the concrete prior verifier: a
passing verifier body opens the same proof to folded authority for the same
`(steps, image)` pair.
-/
theorem acceptedOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (body : ConcreteVerifierBody (PriorProof := PriorProof) ctx) :
    ∀ steps proof image,
      VerifyPrior body steps proof image →
        ∃ authority :
            DirectParentOnlyProductionSoundness.ProofCarryingPriorProof
              ctx.toProductionContext,
          (authorityOpener body).openAuthority proof = some authority ∧
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
    ⟨hCompact, hBoundary, hTranscript, hCommitted, hOpened⟩
  rcases hOpened with ⟨authority, hOpen⟩
  refine ⟨authority, hOpen, ?_⟩
  exact
    body.acceptedBodyOpens
      steps
      proof
      image
      authority
      hCompact
      hBoundary
      hTranscript
      hCommitted
      hOpen

/--
Rust-shaped verifier checks prove the accepted-opens obligation through
statement replay and committed-step verifier soundness.
-/
theorem acceptedOpensOfChecks
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ConcreteVerifierBodyChecks (PriorProof := PriorProof) ctx) :
    ∀ steps proof image,
      VerifyPriorOfChecks checks steps proof image →
        ∃ authority :
            DirectParentOnlyProductionSoundness.ProofCarryingPriorProof
              ctx.toProductionContext,
          (authorityOpenerOfChecks checks).openAuthority proof =
              some authority ∧
            FoldedFPrimeAuthority.Accepts
              (Transition :=
                DirectParentOnlyProductionSoundness.Transition
                  ctx.toProductionContext)
              (initial := ctx.initial)
              steps
              authority
              image :=
  acceptedOpens (concreteVerifierBodyOfChecks checks)

/--
Canonical-statement replay binding proves the accepted-opens obligation.
-/
theorem acceptedOpensOfStatementBinding
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (binding :
      ConcreteVerifierStatementBinding (PriorProof := PriorProof) ctx) :
    ∀ steps proof image,
      VerifyPriorOfStatementBinding binding steps proof image →
        ∃ authority :
            DirectParentOnlyProductionSoundness.ProofCarryingPriorProof
              ctx.toProductionContext,
          (authorityOpenerOfStatementBinding binding).openAuthority proof =
              some authority ∧
            FoldedFPrimeAuthority.Accepts
              (Transition :=
                DirectParentOnlyProductionSoundness.Transition
                  ctx.toProductionContext)
              (initial := ctx.initial)
              steps
              authority
              image :=
  acceptedOpensOfChecks
    (concreteVerifierBodyChecksOfStatementBinding binding)

/--
Verifier-shaped statement replay proves the accepted-opens obligation.
-/
theorem acceptedOpensOfStatementSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteVerifierStatementSurface (PriorProof := PriorProof) ctx) :
    ∀ steps proof image,
      VerifyPriorOfStatementSurface surface steps proof image →
        ∃ authority :
            DirectParentOnlyProductionSoundness.ProofCarryingPriorProof
              ctx.toProductionContext,
          (authorityOpenerOfStatementSurface surface).openAuthority proof =
              some authority ∧
            FoldedFPrimeAuthority.Accepts
              (Transition :=
                DirectParentOnlyProductionSoundness.Transition
                  ctx.toProductionContext)
              (initial := ctx.initial)
              steps
              authority
              image :=
  acceptedOpensOfStatementBinding
    (concreteVerifierStatementBindingOfSurface surface)

/--
Opening certificate induced by the concrete verifier body.
-/
def priorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (body : ConcreteVerifierBody (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
      ctx
      (VerifyPrior body) where
  opener := authorityOpener body
  acceptedOpens := acceptedOpens body

/--
Opening certificate induced by Rust-shaped verifier checks.
-/
def priorVerifierAuthorityOpeningOfChecks
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ConcreteVerifierBodyChecks (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
      ctx
      (VerifyPriorOfChecks checks) :=
  priorVerifierAuthorityOpening (concreteVerifierBodyOfChecks checks)

/--
Opening certificate induced by canonical-statement replay binding.
-/
def priorVerifierAuthorityOpeningOfStatementBinding
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (binding :
      ConcreteVerifierStatementBinding (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
      ctx
      (VerifyPriorOfStatementBinding binding) :=
  priorVerifierAuthorityOpeningOfChecks
    (concreteVerifierBodyChecksOfStatementBinding binding)

/--
Opening certificate induced by the verifier-shaped statement surface.
-/
def priorVerifierAuthorityOpeningOfStatementSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteVerifierStatementSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
      ctx
      (VerifyPriorOfStatementSurface surface) :=
  priorVerifierAuthorityOpeningOfStatementBinding
    (concreteVerifierStatementBindingOfSurface surface)

/--
Certified prior verifier induced by the concrete verifier body.

Callers pass the verifier body object, not a loose accepted-opens theorem.
-/
def certifiedPriorVerifier
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (body : ConcreteVerifierBody (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.ofAcceptedOpens
    ctx
    (VerifyPrior body)
    (authorityOpener body)
    (acceptedOpens body)

/--
Certified prior verifier induced by Rust-shaped verifier checks.

The caller supplies verifier-body replay and committed-step soundness, not a
loose accepted-opens theorem.
-/
def certifiedPriorVerifierOfChecks
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ConcreteVerifierBodyChecks (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  certifiedPriorVerifier (concreteVerifierBodyOfChecks checks)

/--
Certified prior verifier induced by canonical-statement replay binding.
-/
def certifiedPriorVerifierOfStatementBinding
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (binding :
      ConcreteVerifierStatementBinding (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  certifiedPriorVerifierOfChecks
    (concreteVerifierBodyChecksOfStatementBinding binding)

/--
Certified prior verifier induced by the verifier-shaped statement surface.
-/
def certifiedPriorVerifierOfStatementSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteVerifierStatementSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  certifiedPriorVerifierOfStatementBinding
    (concreteVerifierStatementBindingOfSurface surface)

/-- The certified verifier induced by the body uses the body verifier predicate. -/
theorem certifiedPriorVerifier_verify
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (body : ConcreteVerifierBody (PriorProof := PriorProof) ctx) :
    (certifiedPriorVerifier body).verify = VerifyPrior body :=
  rfl

/-- The certified verifier induced by checks uses the checks verifier predicate. -/
theorem certifiedPriorVerifierOfChecks_verify
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ConcreteVerifierBodyChecks (PriorProof := PriorProof) ctx) :
    (certifiedPriorVerifierOfChecks checks).verify =
      VerifyPriorOfChecks checks :=
  rfl

/-- The certified verifier induced by statement binding uses that predicate. -/
theorem certifiedPriorVerifierOfStatementBinding_verify
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (binding :
      ConcreteVerifierStatementBinding (PriorProof := PriorProof) ctx) :
    (certifiedPriorVerifierOfStatementBinding binding).verify =
      VerifyPriorOfStatementBinding binding :=
  rfl

/-- The certified verifier induced by the statement surface uses that predicate. -/
theorem certifiedPriorVerifierOfStatementSurface_verify
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteVerifierStatementSurface (PriorProof := PriorProof) ctx) :
    (certifiedPriorVerifierOfStatementSurface surface).verify =
      VerifyPriorOfStatementSurface surface :=
  rfl

/-- A concrete prior acceptance always opens some authority. -/
theorem verifyPrior_openAuthority_ne_none
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (body : ConcreteVerifierBody (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : VerifyPrior body steps proof image) :
    body.openAuthority proof ≠ none := by
  rcases hVerify with
    ⟨_hCompact, _hBoundary, _hTranscript, _hCommitted, hOpened⟩
  rcases hOpened with ⟨authority, hOpen⟩
  intro hNone
  rw [hNone] at hOpen
  cases hOpen

/-- A checks-based prior acceptance always opens some authority. -/
theorem verifyPriorOfChecks_openAuthority_ne_none
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ConcreteVerifierBodyChecks (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : VerifyPriorOfChecks checks steps proof image) :
    checks.openAuthority proof ≠ none :=
  verifyPrior_openAuthority_ne_none
    (concreteVerifierBodyOfChecks checks)
    hVerify

/-- A statement-surface prior acceptance always opens some authority. -/
theorem verifyPriorOfStatementSurface_openAuthority_ne_none
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteVerifierStatementSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      VerifyPriorOfStatementSurface surface steps proof image) :
    surface.openAuthority proof ≠ none :=
  verifyPriorOfChecks_openAuthority_ne_none
    (concreteVerifierBodyChecksOfStatementSurface surface)
    hVerify

/--
Public-image invariants forced by accepted prior F' authority.

This is the verifier-facing prior-image counterpart of the terminal public
invariant theorem: once the concrete prior proof opens to folded authority, the
opened reachability proof fixes the public step counter and preserves the base
`vkDigest`, initial boundary, and well-formedness.
-/
structure AcceptedPriorPublicImageInvariants
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (steps : Nat)
    (image : PublicImage Digest Boundary) : Prop where
  stepMatches : image.step = steps
  vkDigestMatches : ctx.initial.vkDigest = image.vkDigest
  initialBoundaryMatches :
    ctx.initial.initialBoundary = image.initialBoundary
  wellFormed : Construction2DirectFPrime.WellFormed image

/--
Accepted concrete prior verifier proofs expose prior public-image invariants.
-/
theorem verifyPrior_publicImageInvariants
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (body : ConcreteVerifierBody (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : VerifyPrior body steps proof image) :
    AcceptedPriorPublicImageInvariants ctx steps image := by
  rcases acceptedOpens body steps proof image hVerify with
    ⟨authority, _hOpen, hAccept⟩
  have hReach :
      FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
        ctx.initial
        steps
        image :=
    FoldedFPrimeAuthority.accepts_sound
      (Transition :=
        DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
      (initial := ctx.initial)
      steps
      authority
      image
      hAccept
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

/--
Checks-based verifier acceptance exposes prior public-image invariants.
-/
theorem verifyPriorOfChecks_publicImageInvariants
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ConcreteVerifierBodyChecks (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : VerifyPriorOfChecks checks steps proof image) :
    AcceptedPriorPublicImageInvariants ctx steps image :=
  verifyPrior_publicImageInvariants
    (concreteVerifierBodyOfChecks checks)
    hVerify

/--
Canonical-statement binding verifier acceptance exposes prior public-image
invariants.
-/
theorem verifyPriorOfStatementBinding_publicImageInvariants
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (binding :
      ConcreteVerifierStatementBinding (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      VerifyPriorOfStatementBinding binding steps proof image) :
    AcceptedPriorPublicImageInvariants ctx steps image :=
  verifyPriorOfChecks_publicImageInvariants
    (concreteVerifierBodyChecksOfStatementBinding binding)
    hVerify

/--
Verifier-shaped statement acceptance exposes prior public-image invariants.
-/
theorem verifyPriorOfStatementSurface_publicImageInvariants
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteVerifierStatementSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      VerifyPriorOfStatementSurface surface steps proof image) :
    AcceptedPriorPublicImageInvariants ctx steps image :=
  verifyPriorOfStatementBinding_publicImageInvariants
    (concreteVerifierStatementBindingOfSurface surface)
    hVerify

/--
The concrete prior verifier is same-proof functional.

The fixed opener forces both acceptances of one opaque proof to expose the same
folded authority, and that authority carries one `(steps, image)` pair.
-/
theorem proofFunctional
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (body : ConcreteVerifierBody (PriorProof := PriorProof) ctx) :
    CompressedFPrimeAuthority.ProofFunctional (VerifyPrior body) := by
  intro stepsA stepsB proof imageA imageB hA hB
  rcases acceptedOpens body stepsA proof imageA hA with
    ⟨authorityA, hOpenA, hAcceptA⟩
  rcases acceptedOpens body stepsB proof imageB hB with
    ⟨authorityB, hOpenB, hAcceptB⟩
  have hAuthority : authorityA = authorityB := by
    have hSome : some authorityA = some authorityB :=
      hOpenA.symm.trans hOpenB
    cases hSome
    rfl
  cases hAuthority
  rcases hAcceptA with ⟨hStepsA, hImageA⟩
  rcases hAcceptB with ⟨hStepsB, hImageB⟩
  exact
    ⟨hStepsA.symm.trans hStepsB,
      hImageA.symm.trans hImageB⟩

/--
The checks-based concrete prior verifier is same-proof functional.
-/
theorem proofFunctionalOfChecks
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ConcreteVerifierBodyChecks (PriorProof := PriorProof) ctx) :
    CompressedFPrimeAuthority.ProofFunctional
      (VerifyPriorOfChecks checks) :=
  proofFunctional (concreteVerifierBodyOfChecks checks)

/--
The statement-surface concrete prior verifier is same-proof functional.
-/
theorem proofFunctionalOfStatementSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteVerifierStatementSurface (PriorProof := PriorProof) ctx) :
    CompressedFPrimeAuthority.ProofFunctional
      (VerifyPriorOfStatementSurface surface) :=
  proofFunctional
    (concreteVerifierBodyOfStatementSurface surface)

/--
Concrete-verifier prior-plus-latest end-to-end theorem.

The caller supplies the concrete verifier-body acceptance and the accepted
latest Construction-2 step. Lean packages the certified verifier internally
and returns the existing end-to-end terminal package.
-/
theorem certifiedSingleTerminalEndToEnd_ofConcretePriorVerifierLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (body : ConcreteVerifierBody (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior : VerifyPrior body priorSteps priorProof priorImage)
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
      (certifiedPriorVerifier body).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.certifiedSingleTerminalEndToEnd_ofCertifiedPriorVerifierLatestStep
    (certifiedPriorVerifier body)
    hPrior
    hLatest

/--
Checks-based concrete-verifier prior-plus-latest end-to-end theorem.
-/
theorem certifiedSingleTerminalEndToEnd_ofConcreteVerifierChecksLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ConcreteVerifierBodyChecks (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      VerifyPriorOfChecks checks priorSteps priorProof priorImage)
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
      (certifiedPriorVerifierOfChecks checks).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  certifiedSingleTerminalEndToEnd_ofConcretePriorVerifierLatestStep
    (concreteVerifierBodyOfChecks checks)
    hPrior
    hLatest

/--
Statement-surface prior-plus-latest end-to-end theorem.
-/
theorem certifiedSingleTerminalEndToEnd_ofConcreteVerifierStatementSurfaceLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteVerifierStatementSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      VerifyPriorOfStatementSurface
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
      (certifiedPriorVerifierOfStatementSurface surface).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  certifiedSingleTerminalEndToEnd_ofConcreteVerifierChecksLatestStep
    (concreteVerifierBodyChecksOfStatementSurface surface)
    hPrior
    hLatest

/--
Concrete-verifier projection to the non-aggregate private DEC and stage facts.
-/
theorem nonAggregatePrivateDecStageFacts_ofConcretePriorVerifierLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (body : ConcreteVerifierBody (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior : VerifyPrior body priorSteps priorProof priorImage)
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
    (certifiedPriorVerifier body)
    hPrior
    hLatest

/--
Checks-based concrete-verifier projection to the non-aggregate private DEC and
stage facts.
-/
theorem nonAggregatePrivateDecStageFacts_ofConcreteVerifierChecksLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ConcreteVerifierBodyChecks (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      VerifyPriorOfChecks checks priorSteps priorProof priorImage)
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
  nonAggregatePrivateDecStageFacts_ofConcretePriorVerifierLatestStep
    (concreteVerifierBodyOfChecks checks)
    hPrior
    hLatest

/--
Concrete-verifier projection to the Section 7.1 owner-target stage audit.
-/
theorem section71StageTargetAuditTrail_ofConcretePriorVerifierLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (body : ConcreteVerifierBody (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior : VerifyPrior body priorSteps priorProof priorImage)
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
    (certifiedPriorVerifier body)
    hPrior
    hLatest

/--
Checks-based concrete-verifier projection to the Section 7.1 owner-target stage
audit.
-/
theorem section71StageTargetAuditTrail_ofConcreteVerifierChecksLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (checks : ConcreteVerifierBodyChecks (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      VerifyPriorOfChecks checks priorSteps priorProof priorImage)
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
  section71StageTargetAuditTrail_ofConcretePriorVerifierLatestStep
    (concreteVerifierBodyOfChecks checks)
    hPrior
    hLatest

end DirectParentOnlyProductionConcreteFPrimePrior

end DirectCcsFPrime
