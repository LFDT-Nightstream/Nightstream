import DirectCcsFPrime.ProofSystem.Production.Security.Opening.DirectParentOnlyProductionConcreteFPrimePriorBackendOpening

/-!
Concrete F' prior verifier target.

This module isolates the first verifier-facing F' obligations: the concrete
runtime verifier predicate, the exact statement pair it checks, and the theorem
target saying acceptance opens folded F' authority for that same `(steps, image)`
pair.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorVerifierTarget

/-- Section 7.1-backed production context. -/
abbrev ProductionContext :=
  DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.ProductionContext

/-- Public image for the parent-only production path. -/
abbrev PublicImage :=
  DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.PublicImage

/-- Proof-carrying folded prior authority for the induced production context. -/
abbrev ProofCarryingPriorProof :=
  @DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.ProofCarryingPriorProof

/-- Structured terminal/boundary public IO returned by the production verifier. -/
abbrev ExactTerminalBoundaryPublicIO :=
  DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.ExactTerminalBoundaryPublicIO

/-- Concrete production prior verifier surface used by the F' target. -/
abbrev ConcretePriorVerifierSurface :=
  @DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.ConcreteRuntimeExactPublicIOOpeningSurface

/--
Concrete production prior verifier predicate.

The predicate is intentionally the runtime exact-public-IO verifier surface:
acceptance replays the compact image, Construction-2 boundary, transcript, and
terminal/boundary public IO against the canonical statement for the claimed
`(steps, image)` pair.
-/
abbrev ConcreteVerifyPrior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : ConcretePriorVerifierSurface (PriorProof := PriorProof) ctx) :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop :=
  DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.RuntimeVerifyPriorOfRuntimeExactPublicIOChecks
    surface

/--
Pinned F' authority target for the concrete prior verifier.

The target is deliberately authority-opening based: a verified prior proof must
open to proof-carrying folded F' authority accepted for the same public
`(steps, image)` pair.
-/
def ConcreteFPrimeAuthorityTarget
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : ConcretePriorVerifierSurface (PriorProof := PriorProof) ctx) :
    Prop :=
  ∀ steps proof image,
    ConcreteVerifyPrior surface steps proof image →
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
Replay evidence extracted from concrete prior verification.

This is the verifier-facing public pair evidence before authority opening: the
proof statement, boundary, transcript, and exact public IO all point at the
canonical statement for the same `steps` and `image`.
-/
def ConcreteVerifyPriorReplayEvidence
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : ConcretePriorVerifierSurface (PriorProof := PriorProof) ctx)
    (steps : Nat)
    (proof : PriorProof)
    (image : PublicImage Digest Boundary) : Prop :=
  surface.compactImageReplay steps proof image ∧
    surface.construction2BoundaryReplay steps proof image ∧
    surface.transcriptReplay steps proof image ∧
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
              (surface.canonicalStatement steps image))) ∧
    surface.proofStatement proof =
      surface.canonicalStatement steps image

/-- Concrete verification exposes the exact statement and public-IO replay facts. -/
theorem concreteVerifyPrior_replayEvidence
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcretePriorVerifierSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : ConcreteVerifyPrior surface steps proof image) :
    ConcreteVerifyPriorReplayEvidence surface steps proof image := by
  rcases hVerify with
    ⟨hCompact, hBoundaryReplay, hTranscript, hValid, hProofBoundary,
      hPublicIO⟩
  exact
    ⟨hCompact, hBoundaryReplay, hTranscript, hValid, hProofBoundary,
      hPublicIO,
      surface.replayBindsProofStatement
        steps
        proof
        image
        hCompact
        hBoundaryReplay
        hTranscript⟩

/-- Concrete verification binds the proof statement to the claimed public pair. -/
theorem concreteVerifyPrior_bindsProofStatement
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcretePriorVerifierSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : ConcreteVerifyPrior surface steps proof image) :
    surface.proofStatement proof =
      surface.canonicalStatement steps image :=
by
  rcases hVerify with
    ⟨hCompact, hBoundaryReplay, hTranscript, _hValid, _hProofBoundary,
      _hPublicIO⟩
  exact
    surface.replayBindsProofStatement
      steps
      proof
      image
      hCompact
      hBoundaryReplay
      hTranscript

/-- Concrete verification must open through the fixed authority opener. -/
theorem concreteVerifyPrior_opensAuthority
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcretePriorVerifierSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : ConcreteVerifyPrior surface steps proof image) :
    ∃ authority : ProofCarryingPriorProof ctx,
      surface.openAuthority proof = some authority := by
  rcases
    DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.runtimeVerifyPriorOfRuntimeExactPublicIOChecks_acceptedOpens
        surface
        steps
        proof
        image
        hVerify with
    ⟨authority, hOpen, _hAccepts⟩
  exact ⟨authority, hOpen⟩

/-- Any opened authority accepted by verification is for the same public pair. -/
theorem concreteVerifyPrior_openedAuthority_accepts
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcretePriorVerifierSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify : ConcreteVerifyPrior surface steps proof image)
    (hOpen : surface.openAuthority proof = some authority) :
    FoldedFPrimeAuthority.Accepts
      (Transition :=
        DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
      (initial := ctx.initial)
      steps
      authority
      image :=
  DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.runtimeVerifyPriorOfRuntimeExactPublicIOChecks_openedAuthority_accepts_of_open
      surface
      hVerify
      hOpen

/-- Opened authority step count is bound to the verifier's claimed step count. -/
theorem concreteVerifyPrior_bindsOpenedAuthoritySteps
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcretePriorVerifierSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify : ConcreteVerifyPrior surface steps proof image)
    (hOpen : surface.openAuthority proof = some authority) :
    authority.steps = steps :=
  (concreteVerifyPrior_openedAuthority_accepts hVerify hOpen).1

/-- Opened authority image is bound to the verifier's claimed public image. -/
theorem concreteVerifyPrior_bindsOpenedAuthorityImage
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcretePriorVerifierSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify : ConcreteVerifyPrior surface steps proof image)
    (hOpen : surface.openAuthority proof = some authority) :
    authority.image = image :=
  (concreteVerifyPrior_openedAuthority_accepts hVerify hOpen).2

/-- The concrete verifier surface satisfies the pinned F' authority target. -/
theorem concreteFPrimeAuthorityTarget_holds
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : ConcretePriorVerifierSurface (PriorProof := PriorProof) ctx) :
    ConcreteFPrimeAuthorityTarget surface := by
  intro steps proof image hVerify
  exact
    DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.runtimeVerifyPriorOfRuntimeExactPublicIOChecks_acceptedOpens
        surface
        steps
        proof
        image
        hVerify

end DirectParentOnlyProductionConcreteFPrimePriorVerifierTarget

end DirectCcsFPrime
