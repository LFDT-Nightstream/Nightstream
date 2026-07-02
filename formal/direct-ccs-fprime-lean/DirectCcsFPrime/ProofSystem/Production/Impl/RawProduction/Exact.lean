import DirectCcsFPrime.ProofSystem.Production.Impl.RawProduction.Core
import DirectCcsFPrime.ProofSystem.Production.Security.Opening.BackendOpening

/-!
Consequences for the structured production exact-IO prior F' verifier.

The definitions live in `DirectParentOnlyProductionConcreteFPrimePriorRawProduction`;
this file keeps the extra theorem surface out of the near-cap production module.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorRawProduction

/-- A production raw acceptance cannot fail to open. -/
theorem runtimeVerifyPriorOfProductionRaw_openAuthority_ne_none
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionRaw surface steps proof image) :
    surface.checks.openAuthority proof ≠ none := by
  rcases
    runtimeVerifyPriorOfProductionRaw_acceptedOpens
      surface
      steps
      proof
      image
      hVerify with
    ⟨authority, hOpen, _hAccepts⟩
  intro hNone
  rw [hNone] at hOpen
  cases hOpen

/-- Any opened production raw authority accepts the same public pair. -/
theorem runtimeVerifyPriorOfProductionRaw_openedAuthority_accepts_of_open
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      RuntimeVerifyPriorOfProductionRaw surface steps proof image)
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
    runtimeVerifyPriorOfProductionRaw_acceptedOpens
      surface
      steps
      proof
      image
      hVerify with
    ⟨openedAuthority, hOpened, hAccepts⟩
  have hSame : some authority = some openedAuthority := by
    rw [← hOpen, hOpened]
  cases hSame
  exact hAccepts

/-- Production raw verification binds opened authority to the step count. -/
theorem runtimeVerifyPriorOfProductionRaw_bindsOpenedAuthoritySteps
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      RuntimeVerifyPriorOfProductionRaw surface steps proof image)
    (hOpen : surface.checks.openAuthority proof = some authority) :
    authority.steps = steps :=
  (runtimeVerifyPriorOfProductionRaw_openedAuthority_accepts_of_open
    surface
    hVerify
    hOpen).1

/-- Production raw verification binds opened authority to the public image. -/
theorem runtimeVerifyPriorOfProductionRaw_bindsOpenedAuthorityImage
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      RuntimeVerifyPriorOfProductionRaw surface steps proof image)
    (hOpen : surface.checks.openAuthority proof = some authority) :
    authority.image = image :=
  (runtimeVerifyPriorOfProductionRaw_openedAuthority_accepts_of_open
    surface
    hVerify
    hOpen).2

/-- Production raw verification cannot accept without a fixed opening. -/
theorem runtimeVerifyPriorOfProductionRaw_cannot_accept_without_opening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionRaw surface steps proof image)
    (hNone : surface.checks.openAuthority proof = none) :
    False :=
  runtimeVerifyPriorOfProductionRaw_openAuthority_ne_none
    surface
    hVerify
    hNone

/--
The production raw `SoundVerifier` opens the fixed authority for the same
public pair.
-/
theorem soundVerifierOfProductionRaw_openedAuthority_accepts_of_open
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionRaw surface)
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
      image :=
  runtimeVerifyPriorOfProductionRaw_openedAuthority_accepts_of_open
    surface
    ((soundVerifierOfProductionRaw_accepts_iff surface).1 hVerify)
    hOpen

/-- Production raw `SoundVerifier` acceptance binds the opened authority step. -/
theorem soundVerifierOfProductionRaw_bindsOpenedAuthoritySteps
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionRaw surface)
        steps
        proof
        image)
    (hOpen : surface.checks.openAuthority proof = some authority) :
    authority.steps = steps :=
  (soundVerifierOfProductionRaw_openedAuthority_accepts_of_open
    surface
    hVerify
    hOpen).1

/-- Production raw `SoundVerifier` acceptance binds the opened authority image. -/
theorem soundVerifierOfProductionRaw_bindsOpenedAuthorityImage
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionRaw surface)
        steps
        proof
        image)
    (hOpen : surface.checks.openAuthority proof = some authority) :
    authority.image = image :=
  (soundVerifierOfProductionRaw_openedAuthority_accepts_of_open
    surface
    hVerify
    hOpen).2

/-- Production raw `SoundVerifier` acceptance cannot fail to open. -/
theorem soundVerifierOfProductionRaw_openAuthority_ne_none
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionRaw surface)
        steps
        proof
        image) :
    surface.checks.openAuthority proof ≠ none :=
  runtimeVerifyPriorOfProductionRaw_openAuthority_ne_none
    surface
    ((soundVerifierOfProductionRaw_accepts_iff surface).1 hVerify)

/-- Production raw `SoundVerifier` cannot accept without a fixed opening. -/
theorem soundVerifierOfProductionRaw_cannot_accept_without_opening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionRaw surface)
        steps
        proof
        image)
    (hNone : surface.checks.openAuthority proof = none) :
    False :=
  soundVerifierOfProductionRaw_openAuthority_ne_none
    surface
    hVerify
    hNone

/-- The production raw `SoundVerifier` reaches the prior image. -/
theorem soundVerifierOfProductionRaw_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionRaw surface)
        steps
        proof
        image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image :=
  runtimeVerifyPriorOfProductionRaw_reaches_prior
    surface
    ((soundVerifierOfProductionRaw_accepts_iff surface).1 hVerify)

/-- Production raw `SoundVerifier` acceptance exposes public-image invariants. -/
theorem soundVerifierOfProductionRaw_publicImageInvariants
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionRaw surface)
        steps
        proof
        image) :
    DirectParentOnlyProductionConcreteFPrimePrior.AcceptedPriorPublicImageInvariants
      ctx
      steps
      image :=
  runtimeVerifyPriorOfProductionRaw_publicImageInvariants
    surface
    ((soundVerifierOfProductionRaw_accepts_iff surface).1 hVerify)

/-- Production raw `SoundVerifier` cannot authorize an unreachable prior. -/
theorem soundVerifierOfProductionRaw_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionRaw surface)
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
    (soundVerifierOfProductionRaw_reaches_prior hVerify)

/-- The production raw `SoundVerifier` is same-proof functional. -/
theorem soundVerifierOfProductionRaw_proofFunctional
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx) :
    CompressedFPrimeAuthority.SoundVerifier.ProofFunctional
      (soundVerifierOfProductionRaw surface) :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifierProofFunctional
    (certifiedPriorVerifierOfProductionRaw surface)

/--
Terminal acceptance from production raw checks passes through the strict
`SoundVerifier` object.
-/
theorem acceptedTerminalWithSoundVerifierOfProductionRawLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionRawPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      RuntimeVerifyPriorOfProductionRaw
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
      (soundVerifierOfProductionRaw surface)
      priorSteps
      priorProof
      priorImage
      nextImage
      latestProof := by
  refine
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.acceptedTerminalWithSoundVerifier
      (certifiedPriorVerifierOfProductionRaw surface)
      ?_
  exact
    { priorAccepted := by
        simpa [certifiedPriorVerifierOfProductionRaw_verify]
          using hPrior
      latestAccepted := hLatest }

/-- A structured production exact acceptance cannot fail to open. -/
theorem runtimeVerifyPriorOfProductionExact_openAuthority_ne_none
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionExact surface steps proof image) :
    surface.checks.openAuthority proof ≠ none := by
  rcases
    runtimeVerifyPriorOfProductionExact_acceptedOpens
      surface
      steps
      proof
      image
      hVerify with
    ⟨authority, hOpen, _hAccepts⟩
  intro hNone
  rw [hNone] at hOpen
  cases hOpen

/-- Any opened exact production authority accepts the same public pair. -/
theorem runtimeVerifyPriorOfProductionExact_openedAuthority_accepts_of_open
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      RuntimeVerifyPriorOfProductionExact surface steps proof image)
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
    runtimeVerifyPriorOfProductionExact_boundStatement
      surface
      hVerify with
    ⟨publicIO, hBound⟩
  exact
    surface.openedAuthorityBindsProductionExactStatement
      steps
      proof
      image
      publicIO
      authority
      hBound
      hOpen

/-- Exact production verification binds opened authority to the step count. -/
theorem runtimeVerifyPriorOfProductionExact_bindsOpenedAuthoritySteps
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      RuntimeVerifyPriorOfProductionExact surface steps proof image)
    (hOpen : surface.checks.openAuthority proof = some authority) :
    authority.steps = steps :=
  (runtimeVerifyPriorOfProductionExact_openedAuthority_accepts_of_open
    surface
    hVerify
    hOpen).1

/-- Exact production verification binds opened authority to the public image. -/
theorem runtimeVerifyPriorOfProductionExact_bindsOpenedAuthorityImage
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      RuntimeVerifyPriorOfProductionExact surface steps proof image)
    (hOpen : surface.checks.openAuthority proof = some authority) :
    authority.image = image :=
  (runtimeVerifyPriorOfProductionExact_openedAuthority_accepts_of_open
    surface
    hVerify
    hOpen).2

/-- Exact production verification cannot accept without a fixed opening. -/
theorem runtimeVerifyPriorOfProductionExact_cannot_accept_without_opening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionExact surface steps proof image)
    (hNone : surface.checks.openAuthority proof = none) :
    False :=
  runtimeVerifyPriorOfProductionExact_openAuthority_ne_none
    surface
    hVerify
    hNone

/-- Exact production verification cannot authorize an unreachable image. -/
theorem runtimeVerifyPriorOfProductionExact_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionExact surface steps proof image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable
    (runtimeVerifyPriorOfProductionExact_reaches_prior
      surface
      hVerify)

/-- Exact production verification exposes prior public-image invariants. -/
theorem runtimeVerifyPriorOfProductionExact_publicImageInvariants
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionExact surface steps proof image) :
    DirectParentOnlyProductionConcreteFPrimePrior.AcceptedPriorPublicImageInvariants
      ctx
      steps
      image := by
  have hReach :=
    runtimeVerifyPriorOfProductionExact_reaches_prior
      surface
      hVerify
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

/-- The structured production exact `SoundVerifier` opens folded F' authority. -/
theorem soundVerifierOfProductionExact_opensToFoldedAuthority
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionExact surface)
        steps
        proof
        image) :
    ∃ authority : ProofCarryingPriorProof ctx,
      FoldedFPrimeAuthority.Accepts
        (Transition :=
          DirectParentOnlyProductionSoundness.Transition
            ctx.toProductionContext)
        (initial := ctx.initial)
        steps
        authority
        image :=
  (soundVerifierOfProductionExact surface).opensToFoldedAuthority
    steps
    proof
    image
    hVerify

/--
The structured production exact `SoundVerifier` opens the fixed authority for
the same public pair.
-/
theorem soundVerifierOfProductionExact_openedAuthority_accepts_of_open
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionExact surface)
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
      image :=
  runtimeVerifyPriorOfProductionExact_openedAuthority_accepts_of_open
    surface
    ((soundVerifierOfProductionExact_accepts_iff surface).1 hVerify)
    hOpen

/-- Exact production `SoundVerifier` acceptance binds the opened authority step. -/
theorem soundVerifierOfProductionExact_bindsOpenedAuthoritySteps
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionExact surface)
        steps
        proof
        image)
    (hOpen : surface.checks.openAuthority proof = some authority) :
    authority.steps = steps :=
  (soundVerifierOfProductionExact_openedAuthority_accepts_of_open
    surface
    hVerify
    hOpen).1

/-- Exact production `SoundVerifier` acceptance binds the opened authority image. -/
theorem soundVerifierOfProductionExact_bindsOpenedAuthorityImage
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionExact surface)
        steps
        proof
        image)
    (hOpen : surface.checks.openAuthority proof = some authority) :
    authority.image = image :=
  (soundVerifierOfProductionExact_openedAuthority_accepts_of_open
    surface
    hVerify
    hOpen).2

/-- Exact production `SoundVerifier` acceptance cannot fail to open. -/
theorem soundVerifierOfProductionExact_openAuthority_ne_none
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionExact surface)
        steps
        proof
        image) :
    surface.checks.openAuthority proof ≠ none :=
  runtimeVerifyPriorOfProductionExact_openAuthority_ne_none
    surface
    ((soundVerifierOfProductionExact_accepts_iff surface).1 hVerify)

/-- Exact production `SoundVerifier` cannot accept without a fixed opening. -/
theorem soundVerifierOfProductionExact_cannot_accept_without_opening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionExact surface)
        steps
        proof
        image)
    (hNone : surface.checks.openAuthority proof = none) :
    False :=
  soundVerifierOfProductionExact_openAuthority_ne_none
    surface
    hVerify
    hNone

/-- The structured production exact `SoundVerifier` reaches the prior image. -/
theorem soundVerifierOfProductionExact_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionExact surface)
        steps
        proof
        image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image := by
  rcases soundVerifierOfProductionExact_opensToFoldedAuthority hVerify with
    ⟨authority, hAccepts⟩
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

/-- Exact production `SoundVerifier` acceptance exposes public-image invariants. -/
theorem soundVerifierOfProductionExact_publicImageInvariants
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionExact surface)
        steps
        proof
        image) :
    DirectParentOnlyProductionConcreteFPrimePrior.AcceptedPriorPublicImageInvariants
      ctx
      steps
      image := by
  have hReach :=
    soundVerifierOfProductionExact_reaches_prior hVerify
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

/-- Exact production `SoundVerifier` cannot authorize an unreachable prior. -/
theorem soundVerifierOfProductionExact_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfProductionExact surface)
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
    (soundVerifierOfProductionExact_reaches_prior hVerify)

/-- The structured production exact `SoundVerifier` is same-proof functional. -/
theorem soundVerifierOfProductionExact_proofFunctional
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx) :
    CompressedFPrimeAuthority.SoundVerifier.ProofFunctional
      (soundVerifierOfProductionExact surface) :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifierProofFunctional
    (certifiedPriorVerifierOfProductionExact surface)

/--
Terminal acceptance from exact production checks passes through the strict
`SoundVerifier` object.
-/
theorem acceptedTerminalWithSoundVerifierOfProductionExactLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
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
      (soundVerifierOfProductionExact surface)
      priorSteps
      priorProof
      priorImage
      nextImage
      latestProof := by
  refine
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.acceptedTerminalWithSoundVerifier
      (certifiedPriorVerifierOfProductionExact surface)
      ?_
  exact
    { priorAccepted := by
        simpa [certifiedPriorVerifierOfProductionExact_verify]
          using hPrior
      latestAccepted := hLatest }

/--
Production exact terminal public-IO layout binding.

The structured verifier already checks pointwise terminal and boundary values
for accepted proofs. This additional verifier-layout fact says the terminal
slice exposed by the terminal committed proof has the canonical terminal length
whenever its raw public vector matches the canonical terminal/boundary
concatenation.
-/
structure ProductionExactTerminalLengthBinding
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx) where
  terminalLengthBindsExpected :
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
      publicIO.terminal.length =
        (surface.checks.terminalPublicValues
          (surface.checks.canonicalIvcPublicImage steps image)).length

/--
Instantiate the backend-shaped exact public-IO opening surface from the
structured production exact verifier.

This is the production-facing bridge from the Rust-shaped exact terminal public
IO view to the reusable exact public-IO authority-opening proof path.
-/
def runtimeExactPublicIOOpeningSurfaceOfProductionExact
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    (terminalLengthBinding :
      ProductionExactTerminalLengthBinding surface) :
    DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.ConcreteRuntimeExactPublicIOOpeningSurface
      (PriorProof := PriorProof)
      ctx where
  Statement := surface.checks.IvcPublicImage
  PublicBoundary := surface.checks.Construction2Boundary
  PublicField := surface.checks.PublicField
  TerminalCommittedProof := surface.checks.TerminalCommittedProof
  canonicalStatement := surface.checks.canonicalIvcPublicImage
  proofStatement := surface.checks.proofIvcPublicImage
  statementBoundary := surface.checks.construction2Boundary
  proofBoundary := fun proof =>
    surface.checks.construction2Boundary
      (surface.checks.proofIvcPublicImage proof)
  terminalPublicValues := surface.checks.terminalPublicValues
  boundaryPublicValues := surface.checks.boundaryPublicValues
  terminalCommittedProof := surface.checks.terminalCommittedProof
  statementPublicValid := surface.checks.statementPublicValid
  terminalVerifierPublicIO := surface.checks.terminalVerifierPublicIO
  compactImageReplay :=
    ProductionCompactImageReplay
      (rawProductionVerifierChecksOfExact surface.checks)
  construction2BoundaryReplay :=
    ProductionConstruction2BoundaryReplay
      (rawProductionVerifierChecksOfExact surface.checks)
  transcriptReplay :=
    ProductionPoseidon2TranscriptReplay
      (rawProductionVerifierChecksOfExact surface.checks)
  openAuthority := surface.checks.openAuthority
  terminalLengthBindsExpected := by
    intro steps proof image publicIO hPublicIO hRaw
    exact
      terminalLengthBinding.terminalLengthBindsExpected
        steps
        proof
        image
        publicIO
        hPublicIO
        hRaw
  replayBindsProofStatement := by
    intro steps proof image hCompact _hBoundaryReplay _hTranscript
    exact hCompact.1
  exactBoundStatementOpens := by
    intro steps proof image publicIO
      hCompact
      hBoundaryReplay
      hTranscript
      hValid
      hBoundary
      hPublicIO
      hTerminal
      hBoundaryValues
      hStatement
    exact
      surface.productionExactBackendOpens
        steps
        proof
        image
        publicIO
        ⟨⟨hCompact,
            hBoundaryReplay,
            hTranscript,
            hValid,
            hBoundary,
            hPublicIO,
            hTerminal,
            hBoundaryValues⟩,
          hStatement⟩
  openedAuthorityBindsExactStatement := by
    intro steps proof image publicIO authority
      hCompact
      hBoundaryReplay
      hTranscript
      hValid
      hBoundary
      hPublicIO
      hTerminal
      hBoundaryValues
      hStatement
      hOpen
    exact
      surface.openedAuthorityBindsProductionExactStatement
        steps
        proof
        image
        publicIO
        authority
        ⟨⟨hCompact,
            hBoundaryReplay,
            hTranscript,
            hValid,
            hBoundary,
            hPublicIO,
            hTerminal,
            hBoundaryValues⟩,
          hStatement⟩
        hOpen

/--
Production exact runtime verification satisfies the backend-shaped exact
public-IO opening verifier predicate.
-/
theorem runtimeVerifyPriorOfProductionExact_toRuntimeExactPublicIOOpening
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
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionExact
        surface
        steps
        proof
        image) :
    DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.RuntimeVerifyPriorOfRuntimeExactPublicIOOpening
      (runtimeExactPublicIOOpeningSurfaceOfProductionExact
        surface
        terminalLengthBinding)
      steps
      proof
      image := by
  rcases hVerify with ⟨publicIO, hAccepted⟩
  rcases hAccepted with
    ⟨hCompact,
      hBoundaryReplay,
      hTranscript,
      hValid,
      hBoundary,
      hPublicIO,
      hTerminal,
      hBoundaryValues⟩
  apply
    DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.runtimeVerifyPriorOfRuntimeExactPublicIOOpening_ofExactPublicIOChecks
  exact
    ⟨hCompact,
      hBoundaryReplay,
      hTranscript,
      hValid,
      hBoundary,
      ⟨publicIO, hPublicIO, hTerminal, hBoundaryValues⟩⟩

/--
The production exact verifier opens folded F' authority through the backend
exact public-IO opening bridge.
-/
theorem runtimeVerifyPriorOfProductionExact_viaRuntimeExactPublicIOOpening_acceptedOpens
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
          image := by
  rcases
    DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.runtimeVerifyPriorOfRuntimeExactPublicIOOpening_acceptedOpens
      (runtimeExactPublicIOOpeningSurfaceOfProductionExact
        surface
        terminalLengthBinding)
      steps
      proof
      image
      (runtimeVerifyPriorOfProductionExact_toRuntimeExactPublicIOOpening
        surface
        terminalLengthBinding
        hVerify) with
    ⟨authority, hOpen, hAccepts⟩
  exact ⟨authority, hOpen, hAccepts⟩

/--
Certified prior verifier for production exact checks routed through the backend
exact public-IO opening bridge.
-/
def certifiedPriorVerifierOfProductionExactRuntimeExactPublicIOOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    (terminalLengthBinding :
      ProductionExactTerminalLengthBinding surface) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.certifiedPriorVerifierOfRuntimeExactPublicIOOpening
    (runtimeExactPublicIOOpeningSurfaceOfProductionExact
      surface
      terminalLengthBinding)

/--
The backend-routed production exact certified verifier uses the backend-shaped
exact public-IO opening predicate.
-/
theorem certifiedPriorVerifierOfProductionExactRuntimeExactPublicIOOpening_verify
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    (terminalLengthBinding :
      ProductionExactTerminalLengthBinding surface) :
    (certifiedPriorVerifierOfProductionExactRuntimeExactPublicIOOpening
        surface
        terminalLengthBinding).verify =
      DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.RuntimeVerifyPriorOfRuntimeExactPublicIOOpening
        (runtimeExactPublicIOOpeningSurfaceOfProductionExact
          surface
          terminalLengthBinding) :=
  rfl

/-- Production exact verification is accepted by the backend-routed verifier. -/
theorem certifiedPriorVerifierOfProductionExactRuntimeExactPublicIOOpening_accepts
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
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfProductionExact
        surface
        steps
        proof
        image) :
    (certifiedPriorVerifierOfProductionExactRuntimeExactPublicIOOpening
        surface
        terminalLengthBinding).verify
      steps
      proof
      image := by
  simpa [
    certifiedPriorVerifierOfProductionExactRuntimeExactPublicIOOpening_verify]
    using
      runtimeVerifyPriorOfProductionExact_toRuntimeExactPublicIOOpening
        surface
        terminalLengthBinding
        hVerify

/-- Strict `SoundVerifier` for the backend-routed production exact path. -/
def soundVerifierOfProductionExactRuntimeExactPublicIOOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
    (terminalLengthBinding :
      ProductionExactTerminalLengthBinding surface) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.SoundPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifier
    (certifiedPriorVerifierOfProductionExactRuntimeExactPublicIOOpening
      surface
      terminalLengthBinding)

/--
Backend-routed production exact prior-plus-latest end-to-end theorem.
-/
theorem certifiedSingleTerminalEndToEnd_ofProductionExactRuntimeExactPublicIOOpeningLatestStep
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
    DirectParentOnlyProductionSuperNeoReuseEndToEnd.CertifiedTerminalEndToEnd
      ctx
      (certifiedPriorVerifierOfProductionExactRuntimeExactPublicIOOpening
        surface
        terminalLengthBinding).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.certifiedSingleTerminalEndToEnd_ofRuntimeExactPublicIOOpeningLatestStep
    (runtimeExactPublicIOOpeningSurfaceOfProductionExact
      surface
      terminalLengthBinding)
    (runtimeVerifyPriorOfProductionExact_toRuntimeExactPublicIOOpening
      surface
      terminalLengthBinding
      hPrior)
    hLatest

/--
Canonical production-exact runtime end-to-end theorem.

The caller supplies exact production prior verification and the latest
Construction-2 step. The result is the existing terminal package containing
opened F' authority, same-proof replay, parent-only CE binding, no-swap
evidence, stage audit, and final public-image invariants.
-/
theorem productionExactTerminalEndToEnd_ofRuntimeVerifyPriorLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
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
      (certifiedPriorVerifierOfProductionExact surface).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  certifiedSingleTerminalEndToEnd_ofProductionExactLatestStep
    surface
    hPrior
    hLatest

/-- Production exact projection to non-aggregate private DEC and stage facts. -/
theorem nonAggregatePrivateDecStageFacts_ofProductionExactLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
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
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.nonAggregatePrivateDecStageFacts_ofCertifiedPriorVerifierLatestStep
    (certifiedPriorVerifierOfProductionExact surface)
    hPrior
    hLatest

/-- Production exact projection to the Section 7.1 owner-target stage audit. -/
theorem section71StageTargetAuditTrail_ofProductionExactLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface :
      ProductionExactPriorOpeningSurface (PriorProof := PriorProof) ctx)
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
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.section71StageTargetAuditTrail_ofCertifiedPriorVerifierLatestStep
    (certifiedPriorVerifierOfProductionExact surface)
    hPrior
    hLatest

end DirectParentOnlyProductionConcreteFPrimePriorRawProduction

end DirectCcsFPrime
