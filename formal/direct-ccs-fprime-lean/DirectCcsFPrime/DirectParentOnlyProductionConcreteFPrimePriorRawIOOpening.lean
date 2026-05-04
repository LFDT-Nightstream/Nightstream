import DirectCcsFPrime.DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness

/-!
Split opening certificate for the raw public-IO prior F' verifier.

This module refines the raw public-IO authority boundary. Instead of assuming a
single theorem that accepted raw public IO already yields `Accepts`, it asks for
two implementation-facing facts: the fixed opener returns proof-carrying folded
authority, and the opened authority binds the same `(steps, image)` statement.
Lean then derives the folded `F'` acceptance theorem itself.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening

/-- Section 7.1-backed production context. -/
abbrev ProductionContext :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.ProductionContext

/-- Public image for the parent-only production path. -/
abbrev PublicImage :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.PublicImage

/-- Proof-carrying folded prior authority for the induced production context. -/
abbrev ProofCarryingPriorProof :=
  @DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.ProofCarryingPriorProof

/-- Verifier-visible raw public-vector checks. -/
abbrev ConcreteRawPublicIOVerifierChecks :=
  @DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.ConcreteRawPublicIOVerifierChecks

/-- Bound raw public-vector acceptance for the canonical statement. -/
abbrev RawPublicIOBoundStatementAccepted :=
  @DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.RawPublicIOBoundStatementAccepted

/-- Split raw public-IO soundness surface. -/
abbrev ConcreteRawPublicIOSoundnessSurface :=
  @DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.ConcreteRawPublicIOSoundnessSurface

/--
Opening-level certificate for raw public-IO authority.

The cryptographic boundary is now explicit at the opener: accepted bound raw IO
must open through the fixed opener, and the opened authority's public fields
must match the same bound `(steps, image)` pair.
-/
structure ConcreteRawPublicIOOpeningSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) where
  checks :
    ConcreteRawPublicIOVerifierChecks (PriorProof := PriorProof) ctx
  replayBindsProofStatement :
    ∀ steps proof image,
      checks.compactImageReplay steps proof image →
      checks.construction2BoundaryReplay steps proof image →
      checks.transcriptReplay steps proof image →
        checks.proofStatement proof =
          checks.canonicalStatement steps image
  rawBoundStatementOpens :
    ∀ steps proof image rawPublicIO,
      RawPublicIOBoundStatementAccepted
        checks
        steps
        proof
        image
        rawPublicIO →
        ∃ authority : ProofCarryingPriorProof ctx,
          checks.openAuthority proof = some authority
  openedAuthorityBindsBoundStatement :
    ∀ steps proof image rawPublicIO authority,
      RawPublicIOBoundStatementAccepted
        checks
        steps
        proof
        image
        rawPublicIO →
      checks.openAuthority proof = some authority →
        authority.steps = steps ∧ authority.image = image

/--
Opening-level evidence derives the raw public-IO authority-soundness theorem.
-/
theorem rawBoundStatementAuthoritySound_ofOpeningSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {rawPublicIO : List surface.checks.PublicField}
    (hBound :
      RawPublicIOBoundStatementAccepted
        surface.checks
        steps
        proof
        image
        rawPublicIO) :
    ∃ authority : ProofCarryingPriorProof ctx,
      surface.checks.openAuthority proof = some authority ∧
        FoldedFPrimeAuthority.Accepts
          (Transition :=
            DirectParentOnlyProductionSoundness.Transition
              ctx.toProductionContext)
          (initial := ctx.initial)
          steps
          authority
          image := by
  rcases
    surface.rawBoundStatementOpens
      steps
      proof
      image
      rawPublicIO
      hBound with
    ⟨authority, hOpen⟩
  rcases
    surface.openedAuthorityBindsBoundStatement
      steps
      proof
      image
      rawPublicIO
      authority
      hBound
      hOpen with
    ⟨hSteps, hImage⟩
  exact ⟨authority, hOpen, ⟨hSteps, hImage⟩⟩

/--
Opening-level evidence instantiates the split raw public-IO soundness surface.
-/
def rawPublicIOSoundnessSurfaceOfOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOOpeningSurface (PriorProof := PriorProof) ctx) :
    ConcreteRawPublicIOSoundnessSurface (PriorProof := PriorProof) ctx where
  checks := surface.checks
  replayBindsProofStatement := surface.replayBindsProofStatement
  rawBoundStatementAuthoritySound := by
    intro steps proof image rawPublicIO hBound
    exact
      rawBoundStatementAuthoritySound_ofOpeningSurface
        surface
        hBound

/-- Runtime verifier predicate induced by opening-level raw public-IO evidence. -/
def RuntimeVerifyPriorOfRawPublicIOOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOOpeningSurface (PriorProof := PriorProof) ctx) :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
        Prop :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.RuntimeVerifyPriorOfRawPublicIOSoundness
    (rawPublicIOSoundnessSurfaceOfOpening surface)

/-- Certified prior verifier induced by opening-level raw public-IO evidence. -/
def certifiedPriorVerifierOfRawPublicIOOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOOpeningSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.certifiedPriorVerifierOfRawPublicIOSoundness
    (rawPublicIOSoundnessSurfaceOfOpening surface)

/-- The opening-level certified verifier uses the opening-level predicate. -/
theorem certifiedPriorVerifierOfRawPublicIOOpening_verify
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOOpeningSurface (PriorProof := PriorProof) ctx) :
    (certifiedPriorVerifierOfRawPublicIOOpening surface).verify =
      RuntimeVerifyPriorOfRawPublicIOOpening surface :=
  rfl

/--
Opening-level raw public-IO acceptance opens folded F' authority for the same
public pair.
-/
theorem runtimeVerifyPriorOfRawPublicIOOpening_acceptedOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOOpeningSurface (PriorProof := PriorProof) ctx) :
    ∀ steps proof image,
      RuntimeVerifyPriorOfRawPublicIOOpening surface steps proof image →
        ∃ authority : ProofCarryingPriorProof ctx,
          surface.checks.openAuthority proof = some authority ∧
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
    DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.runtimeVerifyPriorOfRawPublicIOSoundness_acceptedOpens
      (rawPublicIOSoundnessSurfaceOfOpening surface)
      steps
      proof
      image
      hVerify

/--
If the fixed opener returns a concrete authority for an accepted proof, that
authority accepts the same `(steps, image)` pair.
-/
theorem runtimeVerifyPriorOfRawPublicIOOpening_openedAuthority_accepts_of_open
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      RuntimeVerifyPriorOfRawPublicIOOpening
        surface
        steps
        proof
        image)
    (hOpen : surface.checks.openAuthority proof = some authority) :
    FoldedFPrimeAuthority.Accepts
      (Transition :=
        DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
      (initial := ctx.initial)
      steps
      authority
      image := by
  rcases
    runtimeVerifyPriorOfRawPublicIOOpening_acceptedOpens
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

/-- Opening-level raw public-IO acceptance reaches its claimed prior image. -/
theorem runtimeVerifyPriorOfRawPublicIOOpening_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfRawPublicIOOpening surface steps proof image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.runtimeVerifyPriorOfRawPublicIOSoundness_reaches_prior
    (rawPublicIOSoundnessSurfaceOfOpening surface)
    hVerify

/-- The opening-level raw public-IO prior verifier is same-proof functional. -/
theorem proofFunctionalOfRawPublicIOOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOOpeningSurface (PriorProof := PriorProof) ctx) :
    CompressedFPrimeAuthority.ProofFunctional
      (RuntimeVerifyPriorOfRawPublicIOOpening surface) :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.proofFunctionalOfRawPublicIOSoundness
    (rawPublicIOSoundnessSurfaceOfOpening surface)

/--
Opening-level raw public-IO acceptance exposes prior public-image invariants.
-/
theorem runtimeVerifyPriorOfRawPublicIOOpening_publicImageInvariants
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfRawPublicIOOpening surface steps proof image) :
    DirectParentOnlyProductionConcreteFPrimePrior.AcceptedPriorPublicImageInvariants
      ctx
      steps
      image :=
  DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.runtimeVerifyPriorOfRawPublicIOSoundness_publicImageInvariants
    (rawPublicIOSoundnessSurfaceOfOpening surface)
    hVerify

/--
Opening-level raw public-IO acceptance cannot authorize an unreachable prior
image.
-/
theorem runtimeVerifyPriorOfRawPublicIOOpening_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ConcreteRawPublicIOOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfRawPublicIOOpening surface steps proof image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable
    (runtimeVerifyPriorOfRawPublicIOOpening_reaches_prior
      surface
      hVerify)

end DirectParentOnlyProductionConcreteFPrimePriorRawIOOpening

end DirectCcsFPrime
