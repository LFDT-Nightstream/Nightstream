import DirectCcsFPrime.ProofSystem.Production.Impl.ConcretePrior.VerifierTarget
import DirectCcsFPrime.ProofSystem.Production.Security.SuperNeoReuseCertifiedVerifier
import DirectCcsFPrime.ProofSystem.Production.Security.SuperNeoReuseEndToEnd

/-!
Certified concrete F' prior verifier.

This module turns the concrete prior-verifier target into the production
`CertifiedPriorVerifier` object. The verifier predicate is the concrete
runtime exact-public-IO predicate, and certification comes only from the fixed
authority opener proved in the target module.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorVerifierCertified

/-- Section 7.1-backed production context. -/
abbrev ProductionContext :=
  DirectParentOnlyProductionConcreteFPrimePriorVerifierTarget.ProductionContext

/-- Public image for the parent-only production path. -/
abbrev PublicImage :=
  DirectParentOnlyProductionConcreteFPrimePriorVerifierTarget.PublicImage

/-- Proof-carrying folded prior authority for the induced production context. -/
abbrev ProofCarryingPriorProof :=
  @DirectParentOnlyProductionConcreteFPrimePriorVerifierTarget.ProofCarryingPriorProof

/-- Concrete production prior verifier surface. -/
abbrev ConcretePriorVerifierSurface :=
  @DirectParentOnlyProductionConcreteFPrimePriorVerifierTarget.ConcretePriorVerifierSurface

/-- Concrete production prior verifier predicate. -/
abbrev ConcreteVerifyPrior :=
  @DirectParentOnlyProductionConcreteFPrimePriorVerifierTarget.ConcreteVerifyPrior

/-- Certified concrete compressed prior verifier. -/
abbrev CertifiedPriorVerifier :=
  @DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier

/-- Strict compressed prior verifier accepted by the production theorem. -/
abbrev SoundPriorVerifier :=
  @DirectParentOnlyProductionSuperNeoReuse.ProductionContext.SoundPriorVerifier

/-- Implementation-shaped exact public-IO runtime verifier surface. -/
abbrev ConcreteRuntimeExactPublicIOSurface :=
  @DirectParentOnlyProductionConcreteFPrimePriorBackend.ConcreteRuntimeExactPublicIOSurface

/-- Exact public-IO terminal/boundary layout binding. -/
abbrev ExactPublicIOLayoutBinding :=
  @DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness.ExactPublicIOLayoutBinding

/-- Implementation-shaped exact public-IO runtime prior verifier predicate. -/
abbrev RuntimeVerifyPriorOfExactPublicIO :=
  @DirectParentOnlyProductionConcreteFPrimePriorBackend.RuntimeVerifyPriorOfExactPublicIO

/-- Fixed opener induced by the concrete verifier surface. -/
def priorAuthorityOpenerOfConcretePriorVerifierSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : ConcretePriorVerifierSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
      (PriorProof := PriorProof)
      ctx where
  openAuthority := surface.openAuthority

/-- The induced opener is exactly the surface opener. -/
theorem priorAuthorityOpenerOfConcretePriorVerifierSurface_openAuthority
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : ConcretePriorVerifierSurface (PriorProof := PriorProof) ctx)
    (proof : PriorProof) :
    (priorAuthorityOpenerOfConcretePriorVerifierSurface surface).openAuthority
        proof =
      surface.openAuthority proof :=
  rfl

/-- Concrete verifier opening certificate for the certified prior verifier. -/
def priorVerifierAuthorityOpeningOfConcretePriorVerifierSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : ConcretePriorVerifierSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
      ctx
      (ConcreteVerifyPrior surface) where
  opener := priorAuthorityOpenerOfConcretePriorVerifierSurface surface
  acceptedOpens :=
    DirectParentOnlyProductionConcreteFPrimePriorVerifierTarget.concreteFPrimeAuthorityTarget_holds
      surface

/--
Certified verifier whose predicate is exactly concrete prior verification.

This is the implementation-facing F' object: callers provide the concrete
surface and Lean supplies the `CertifiedPriorVerifier` package from the fixed
opener plus the target authority-opening theorem.
-/
def certifiedPriorVerifierOfConcretePriorVerifierSurface
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : ConcretePriorVerifierSurface (PriorProof := PriorProof) ctx) :
    CertifiedPriorVerifier (PriorProof := PriorProof) ctx :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.ofAcceptedOpens
    ctx
    (ConcreteVerifyPrior surface)
    (priorAuthorityOpenerOfConcretePriorVerifierSurface surface)
    (DirectParentOnlyProductionConcreteFPrimePriorVerifierTarget.concreteFPrimeAuthorityTarget_holds
      surface)

/-- The certified verifier uses the concrete runtime verifier predicate. -/
theorem certifiedPriorVerifierOfConcretePriorVerifierSurface_verify
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : ConcretePriorVerifierSurface (PriorProof := PriorProof) ctx) :
    (certifiedPriorVerifierOfConcretePriorVerifierSurface surface).verify =
      ConcreteVerifyPrior surface :=
  rfl

/-- The certified verifier uses the fixed opener induced by the surface. -/
theorem certifiedPriorVerifierOfConcretePriorVerifierSurface_opener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : ConcretePriorVerifierSurface (PriorProof := PriorProof) ctx) :
    (certifiedPriorVerifierOfConcretePriorVerifierSurface surface).opening.opener =
      priorAuthorityOpenerOfConcretePriorVerifierSurface surface :=
  rfl

/-- Concrete prior verification opens authority through the certified verifier. -/
theorem concreteVerifyPrior_certifiedAcceptedOpens
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
      (certifiedPriorVerifierOfConcretePriorVerifierSurface surface).opening.opener.openAuthority
          proof =
        some authority ∧
        FoldedFPrimeAuthority.Accepts
          (Transition :=
            DirectParentOnlyProductionSoundness.Transition
              ctx.toProductionContext)
          (initial := ctx.initial)
          steps
          authority
          image :=
  (certifiedPriorVerifierOfConcretePriorVerifierSurface surface).opening.acceptedOpens
    steps
    proof
    image
    hVerify

/--
Concrete prior-verifier surface induced by implementation-shaped exact runtime
soundness plus exact public-IO layout binding.

The verifier predicate remains the concrete exact-public-IO replay predicate;
the layout binding is used to connect the verifier's raw terminal output to the
terminal/boundary split needed by the authority-opening proof.
-/
def concretePriorVerifierSurfaceOfExactRuntimeSoundAndLayout
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx)
    (layout : ExactPublicIOLayoutBinding surface) :
    ConcretePriorVerifierSurface (PriorProof := PriorProof) ctx :=
  DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.runtimeExactPublicIOOpeningSurfaceOfExactRuntimeSoundAndLayout
    surface
    layout

/-- The induced concrete verifier surface keeps the implementation opener. -/
theorem concretePriorVerifierSurfaceOfExactRuntimeSoundAndLayout_openAuthority
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx)
    (layout : ExactPublicIOLayoutBinding surface)
    (proof : PriorProof) :
    (concretePriorVerifierSurfaceOfExactRuntimeSoundAndLayout surface layout).openAuthority
        proof =
      surface.openAuthority proof :=
  rfl

/--
Implementation exact-runtime verification is the concrete verifier predicate on
the induced certified surface.
-/
theorem runtimeVerifyPriorOfExactRuntimeSoundAndLayout_toConcreteVerifyPrior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx}
    (layout : ExactPublicIOLayoutBinding surface)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIO surface steps proof image) :
    ConcreteVerifyPrior
      (concretePriorVerifierSurfaceOfExactRuntimeSoundAndLayout surface layout)
      steps
      proof
      image := by
  simpa [
    RuntimeVerifyPriorOfExactPublicIO,
    ConcreteVerifyPrior,
    DirectParentOnlyProductionConcreteFPrimePriorVerifierTarget.ConcreteVerifyPrior,
    DirectParentOnlyProductionConcreteFPrimePriorBackend.RuntimeVerifyPriorOfExactPublicIO,
    DirectParentOnlyProductionConcreteFPrimePriorBackend.RuntimeVerifyPrior,
    DirectParentOnlyProductionConcreteFPrimePriorBackend.runtimeBackendSurfaceOfExactPublicIO,
    DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.RuntimeVerifyPriorOfRuntimeExactPublicIOChecks,
    concretePriorVerifierSurfaceOfExactRuntimeSoundAndLayout,
    DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.runtimeExactPublicIOOpeningSurfaceOfExactRuntimeSoundAndLayout]
    using hVerify

/--
Certified verifier induced by implementation-shaped exact runtime soundness plus
layout binding.
-/
def certifiedPriorVerifierOfExactRuntimeSoundAndLayout
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx)
    (layout : ExactPublicIOLayoutBinding surface) :
    CertifiedPriorVerifier (PriorProof := PriorProof) ctx :=
  certifiedPriorVerifierOfConcretePriorVerifierSurface
    (concretePriorVerifierSurfaceOfExactRuntimeSoundAndLayout surface layout)

/-- The implementation-shaped certified verifier uses exact runtime verification. -/
theorem certifiedPriorVerifierOfExactRuntimeSoundAndLayout_verify
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx)
    (layout : ExactPublicIOLayoutBinding surface) :
    (certifiedPriorVerifierOfExactRuntimeSoundAndLayout surface layout).verify =
      RuntimeVerifyPriorOfExactPublicIO surface := by
  funext steps proof image
  rfl

/--
Strict `SoundVerifier` induced by implementation-shaped exact runtime
soundness plus layout binding.

This is the compressed F' authority object consumed by production terminal
soundness: verification is exact-runtime replay, and acceptance opens to folded
F' reachability through the fixed authority opener.
-/
def soundVerifierOfExactRuntimeSoundAndLayout
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx)
    (layout : ExactPublicIOLayoutBinding surface) :
    SoundPriorVerifier (PriorProof := PriorProof) ctx :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifier
    (certifiedPriorVerifierOfExactRuntimeSoundAndLayout surface layout)

/-- The exact-runtime `SoundVerifier` accepts exactly exact public-IO replay. -/
theorem soundVerifierOfExactRuntimeSoundAndLayout_accepts_iff
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx)
    (layout : ExactPublicIOLayoutBinding surface)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary} :
    CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfExactRuntimeSoundAndLayout surface layout)
        steps
        proof
        image <->
      RuntimeVerifyPriorOfExactPublicIO surface steps proof image := by
  simpa [soundVerifierOfExactRuntimeSoundAndLayout,
    certifiedPriorVerifierOfExactRuntimeSoundAndLayout_verify]
    using
      DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifier_accepts_iff
        (certifiedPriorVerifierOfExactRuntimeSoundAndLayout surface layout)

/--
Acceptance by the exact-runtime `SoundVerifier` opens to folded F' authority for
the same `(steps, image)` pair.
-/
theorem soundVerifierOfExactRuntimeSoundAndLayout_opensToFoldedAuthority
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx}
    {layout : ExactPublicIOLayoutBinding surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfExactRuntimeSoundAndLayout surface layout)
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
  (soundVerifierOfExactRuntimeSoundAndLayout surface layout).opensToFoldedAuthority
    steps
    proof
    image
    hVerify

/--
Any authority opened by exact-runtime `SoundVerifier` acceptance is for the
same claimed `(steps, image)` pair.
-/
theorem soundVerifierOfExactRuntimeSoundAndLayout_openedAuthority_accepts
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx}
    {layout : ExactPublicIOLayoutBinding surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfExactRuntimeSoundAndLayout surface layout)
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
  DirectParentOnlyProductionConcreteFPrimePriorVerifierTarget.concreteVerifyPrior_openedAuthority_accepts
    (runtimeVerifyPriorOfExactRuntimeSoundAndLayout_toConcreteVerifyPrior
      layout
      ((soundVerifierOfExactRuntimeSoundAndLayout_accepts_iff
        surface
        layout).1 hVerify))
    (by
      simpa [concretePriorVerifierSurfaceOfExactRuntimeSoundAndLayout]
        using hOpen)

/--
Exact-runtime `SoundVerifier` acceptance binds the opened authority to the
claimed step count.
-/
theorem soundVerifierOfExactRuntimeSoundAndLayout_bindsOpenedAuthoritySteps
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx}
    {layout : ExactPublicIOLayoutBinding surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfExactRuntimeSoundAndLayout surface layout)
        steps
        proof
        image)
    (hOpen : surface.openAuthority proof = some authority) :
    authority.steps = steps :=
  (soundVerifierOfExactRuntimeSoundAndLayout_openedAuthority_accepts
    (layout := layout)
    hVerify
    hOpen).1

/--
Exact-runtime `SoundVerifier` acceptance binds the opened authority to the
claimed public image.
-/
theorem soundVerifierOfExactRuntimeSoundAndLayout_bindsOpenedAuthorityImage
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx}
    {layout : ExactPublicIOLayoutBinding surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfExactRuntimeSoundAndLayout surface layout)
        steps
        proof
        image)
    (hOpen : surface.openAuthority proof = some authority) :
    authority.image = image :=
  (soundVerifierOfExactRuntimeSoundAndLayout_openedAuthority_accepts
    (layout := layout)
    hVerify
    hOpen).2

/-- Exact-runtime `SoundVerifier` acceptance cannot have an empty opener. -/
theorem soundVerifierOfExactRuntimeSoundAndLayout_openAuthority_ne_none
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx}
    {layout : ExactPublicIOLayoutBinding surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
      (soundVerifierOfExactRuntimeSoundAndLayout surface layout)
        steps
        proof
        image) :
    surface.openAuthority proof ≠ none := by
  rcases
    DirectParentOnlyProductionConcreteFPrimePriorVerifierTarget.concreteVerifyPrior_opensAuthority
      (runtimeVerifyPriorOfExactRuntimeSoundAndLayout_toConcreteVerifyPrior
        layout
        ((soundVerifierOfExactRuntimeSoundAndLayout_accepts_iff
          surface
          layout).1 hVerify)) with
    ⟨authority, hOpen⟩
  intro hNone
  have hOpen' : surface.openAuthority proof = some authority := by
    simpa [concretePriorVerifierSurfaceOfExactRuntimeSoundAndLayout]
      using hOpen
  rw [hNone] at hOpen'
  cases hOpen'

/-- Exact-runtime `SoundVerifier` cannot accept without a fixed opening. -/
theorem soundVerifierOfExactRuntimeSoundAndLayout_cannot_accept_without_opening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx}
    {layout : ExactPublicIOLayoutBinding surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfExactRuntimeSoundAndLayout surface layout)
        steps
        proof
        image)
    (hNone : surface.openAuthority proof = none) :
    False :=
  soundVerifierOfExactRuntimeSoundAndLayout_openAuthority_ne_none
    (layout := layout)
    hVerify
    hNone

/-- Exact-runtime `SoundVerifier` acceptance reaches the claimed prior image. -/
theorem soundVerifierOfExactRuntimeSoundAndLayout_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx}
    {layout : ExactPublicIOLayoutBinding surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfExactRuntimeSoundAndLayout surface layout)
        steps
        proof
        image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image :=
  by
    rcases
      soundVerifierOfExactRuntimeSoundAndLayout_opensToFoldedAuthority
        hVerify with
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

/-- Exact-runtime `SoundVerifier` acceptance exposes public-image invariants. -/
theorem soundVerifierOfExactRuntimeSoundAndLayout_publicImageInvariants
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx}
    {layout : ExactPublicIOLayoutBinding surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfExactRuntimeSoundAndLayout surface layout)
        steps
        proof
        image) :
    DirectParentOnlyProductionConcreteFPrimePrior.AcceptedPriorPublicImageInvariants
      ctx
      steps
      image := by
  have hReach :=
    soundVerifierOfExactRuntimeSoundAndLayout_reaches_prior hVerify
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

/-- Exact-runtime `SoundVerifier` cannot authorize an unreachable prior image. -/
theorem soundVerifierOfExactRuntimeSoundAndLayout_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx}
    {layout : ExactPublicIOLayoutBinding surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfExactRuntimeSoundAndLayout surface layout)
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
    (soundVerifierOfExactRuntimeSoundAndLayout_reaches_prior hVerify)

/-- The exact-runtime `SoundVerifier` is same-proof functional. -/
theorem soundVerifierOfExactRuntimeSoundAndLayout_proofFunctional
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx)
    (layout : ExactPublicIOLayoutBinding surface) :
    CompressedFPrimeAuthority.SoundVerifier.ProofFunctional
      (soundVerifierOfExactRuntimeSoundAndLayout surface layout) :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifierProofFunctional
    (certifiedPriorVerifierOfExactRuntimeSoundAndLayout surface layout)

/--
One exact-runtime prior proof cannot verify as authority for two different
prior `(steps, image)` pairs under the induced `SoundVerifier`.
-/
theorem soundVerifierOfExactRuntimeSoundAndLayout_sameProof
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx}
    {layout : ExactPublicIOLayoutBinding surface}
    {stepsA stepsB : Nat}
    {proof : PriorProof}
    {imageA imageB : PublicImage Digest Boundary}
    (hA :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfExactRuntimeSoundAndLayout surface layout)
        stepsA
        proof
        imageA)
    (hB :
      CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifierOfExactRuntimeSoundAndLayout surface layout)
        stepsB
        proof
        imageB) :
    stepsA = stepsB ∧ imageA = imageB :=
  soundVerifierOfExactRuntimeSoundAndLayout_proofFunctional surface layout hA hB

/--
Latest-step terminal acceptance through the exact-runtime induced
`SoundVerifier`.
-/
theorem acceptedTerminalWithSoundVerifierOfExactRuntimeSoundAndLayoutLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx)
    (layout : ExactPublicIOLayoutBinding surface)
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
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.AcceptedTerminal
      ctx
      (soundVerifierOfExactRuntimeSoundAndLayout surface layout)
      priorSteps
      priorProof
      priorImage
      nextImage
      latestProof := by
  refine
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.acceptedTerminalWithSoundVerifier
      (certifiedPriorVerifierOfExactRuntimeSoundAndLayout surface layout)
      ?_
  exact
    { priorAccepted := by
        simpa [certifiedPriorVerifierOfExactRuntimeSoundAndLayout_verify]
          using hPrior
      latestAccepted := hLatest }

/--
Exact-runtime verification opens authority through the certified verifier for
the same `(steps, image)` pair.
-/
theorem runtimeVerifyPriorOfExactRuntimeSoundAndLayout_certifiedAcceptedOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx}
    {layout : ExactPublicIOLayoutBinding surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIO surface steps proof image) :
    ∃ authority : ProofCarryingPriorProof ctx,
      (certifiedPriorVerifierOfExactRuntimeSoundAndLayout surface layout).opening.opener.openAuthority
          proof =
        some authority ∧
        FoldedFPrimeAuthority.Accepts
          (Transition :=
            DirectParentOnlyProductionSoundness.Transition
              ctx.toProductionContext)
          (initial := ctx.initial)
          steps
          authority
          image :=
  concreteVerifyPrior_certifiedAcceptedOpens
    (runtimeVerifyPriorOfExactRuntimeSoundAndLayout_toConcreteVerifyPrior
      layout
      hVerify)

/-- Exact-runtime prior verification must open through the fixed authority opener. -/
theorem runtimeVerifyPriorOfExactRuntimeSoundAndLayout_opensAuthority
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx}
    {layout : ExactPublicIOLayoutBinding surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIO surface steps proof image) :
    ∃ authority : ProofCarryingPriorProof ctx,
      surface.openAuthority proof = some authority := by
  rcases
    DirectParentOnlyProductionConcreteFPrimePriorVerifierTarget.concreteVerifyPrior_opensAuthority
      (runtimeVerifyPriorOfExactRuntimeSoundAndLayout_toConcreteVerifyPrior
        layout
        hVerify) with
    ⟨authority, hOpen⟩
  exact
    ⟨authority,
      by
        simpa [concretePriorVerifierSurfaceOfExactRuntimeSoundAndLayout]
          using hOpen⟩

/--
Any authority opened by exact-runtime verification accepts the same claimed
`(steps, image)` pair.
-/
theorem runtimeVerifyPriorOfExactRuntimeSoundAndLayout_openedAuthority_accepts
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx}
    {layout : ExactPublicIOLayoutBinding surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIO surface steps proof image)
    (hOpen : surface.openAuthority proof = some authority) :
    FoldedFPrimeAuthority.Accepts
      (Transition :=
        DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
      (initial := ctx.initial)
      steps
      authority
      image :=
  DirectParentOnlyProductionConcreteFPrimePriorVerifierTarget.concreteVerifyPrior_openedAuthority_accepts
    (runtimeVerifyPriorOfExactRuntimeSoundAndLayout_toConcreteVerifyPrior
      layout
      hVerify)
    (by
      simpa [concretePriorVerifierSurfaceOfExactRuntimeSoundAndLayout]
        using hOpen)

/-- Exact-runtime verification binds opened authority to the claimed step count. -/
theorem runtimeVerifyPriorOfExactRuntimeSoundAndLayout_bindsOpenedAuthoritySteps
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx}
    {layout : ExactPublicIOLayoutBinding surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIO surface steps proof image)
    (hOpen : surface.openAuthority proof = some authority) :
    authority.steps = steps :=
  (runtimeVerifyPriorOfExactRuntimeSoundAndLayout_openedAuthority_accepts
    (layout := layout)
    hVerify
    hOpen).1

/-- Exact-runtime verification binds opened authority to the claimed public image. -/
theorem runtimeVerifyPriorOfExactRuntimeSoundAndLayout_bindsOpenedAuthorityImage
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx}
    {layout : ExactPublicIOLayoutBinding surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIO surface steps proof image)
    (hOpen : surface.openAuthority proof = some authority) :
    authority.image = image :=
  (runtimeVerifyPriorOfExactRuntimeSoundAndLayout_openedAuthority_accepts
    (layout := layout)
    hVerify
    hOpen).2

/-- Exact-runtime verification cannot accept if the fixed opener returns `none`. -/
theorem runtimeVerifyPriorOfExactRuntimeSoundAndLayout_cannot_accept_without_opening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx}
    {layout : ExactPublicIOLayoutBinding surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIO surface steps proof image)
    (hNone : surface.openAuthority proof = none) :
    False := by
  rcases
    runtimeVerifyPriorOfExactRuntimeSoundAndLayout_opensAuthority
      (layout := layout)
      hVerify with
    ⟨authority, hOpen⟩
  rw [hNone] at hOpen
  cases hOpen

/-- Exact-runtime verification implies the fixed opener returns an authority. -/
theorem runtimeVerifyPriorOfExactRuntimeSoundAndLayout_openAuthority_ne_none
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx}
    {layout : ExactPublicIOLayoutBinding surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIO surface steps proof image) :
    surface.openAuthority proof ≠ none := by
  intro hNone
  exact
    runtimeVerifyPriorOfExactRuntimeSoundAndLayout_cannot_accept_without_opening
      (layout := layout)
      hVerify
      hNone

/-- Exact-runtime verification exposes the prior public-image invariants. -/
theorem runtimeVerifyPriorOfExactRuntimeSoundAndLayout_publicImageInvariants
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx}
    {layout : ExactPublicIOLayoutBinding surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIO surface steps proof image) :
    DirectParentOnlyProductionConcreteFPrimePrior.AcceptedPriorPublicImageInvariants
      ctx
      steps
      image :=
  DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.runtimeVerifyPriorOfRuntimeExactPublicIOChecks_publicImageInvariants
    (concretePriorVerifierSurfaceOfExactRuntimeSoundAndLayout surface layout)
    (runtimeVerifyPriorOfExactRuntimeSoundAndLayout_toConcreteVerifyPrior
      layout
      hVerify)

/-- Exact-runtime verification is same-proof functional. -/
theorem proofFunctionalOfExactRuntimeSoundAndLayout
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx)
    (layout : ExactPublicIOLayoutBinding surface) :
    CompressedFPrimeAuthority.ProofFunctional
      (RuntimeVerifyPriorOfExactPublicIO surface) := by
  intro stepsA stepsB proof imageA imageB hA hB
  exact
    DirectParentOnlyProductionConcreteFPrimePriorBackendOpening.proofFunctionalOfRuntimeExactPublicIOChecks
      (concretePriorVerifierSurfaceOfExactRuntimeSoundAndLayout surface layout)
      (runtimeVerifyPriorOfExactRuntimeSoundAndLayout_toConcreteVerifyPrior
        layout
        hA)
      (runtimeVerifyPriorOfExactRuntimeSoundAndLayout_toConcreteVerifyPrior
        layout
        hB)

/-- Concrete prior verification reaches the claimed prior public image. -/
theorem concreteVerifyPrior_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcretePriorVerifierSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : ConcreteVerifyPrior surface steps proof image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.reaches_prior
    (certifiedPriorVerifierOfConcretePriorVerifierSurface surface)
    hVerify

/-- Concrete prior verification cannot accept an unreachable prior image. -/
theorem concreteVerifyPrior_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcretePriorVerifierSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : ConcreteVerifyPrior surface steps proof image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable
    (concreteVerifyPrior_reaches_prior hVerify)

/-- Exact-runtime prior verification reaches the claimed prior public image. -/
theorem runtimeVerifyPriorOfExactRuntimeSoundAndLayout_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx}
    {layout : ExactPublicIOLayoutBinding surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIO surface steps proof image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image :=
  concreteVerifyPrior_reaches_prior
    (runtimeVerifyPriorOfExactRuntimeSoundAndLayout_toConcreteVerifyPrior
      layout
      hVerify)

/-- Exact-runtime verification cannot accept an unreachable prior image. -/
theorem runtimeVerifyPriorOfExactRuntimeSoundAndLayout_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx}
    {layout : ExactPublicIOLayoutBinding surface}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify :
      RuntimeVerifyPriorOfExactPublicIO surface steps proof image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable
    (runtimeVerifyPriorOfExactRuntimeSoundAndLayout_reaches_prior
      (layout := layout)
      hVerify)

/--
Concrete certified-verifier latest-step theorem.

The caller provides concrete prior verification and the latest Construction-2
step. Lean constructs the certified verifier object internally and returns the
existing terminal end-to-end package.
-/
theorem certifiedSingleTerminalEndToEnd_ofConcretePriorVerifierLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : ConcretePriorVerifierSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      ConcreteVerifyPrior
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
      (certifiedPriorVerifierOfConcretePriorVerifierSurface surface).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.certifiedSingleTerminalEndToEnd_ofCertifiedPriorVerifierLatestStep
    (certifiedPriorVerifierOfConcretePriorVerifierSurface surface)
    hPrior
    hLatest

/--
Implementation-shaped exact-runtime latest-step theorem.

The caller supplies exact-runtime prior verification and the latest
Construction-2 step. Lean builds the concrete certified verifier internally and
returns the terminal end-to-end package.
-/
theorem certifiedSingleTerminalEndToEnd_ofExactRuntimeSoundAndLayoutLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : ConcreteRuntimeExactPublicIOSurface (PriorProof := PriorProof) ctx)
    (layout : ExactPublicIOLayoutBinding surface)
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
      (certifiedPriorVerifierOfExactRuntimeSoundAndLayout surface layout).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  certifiedSingleTerminalEndToEnd_ofConcretePriorVerifierLatestStep
    (concretePriorVerifierSurfaceOfExactRuntimeSoundAndLayout surface layout)
    (runtimeVerifyPriorOfExactRuntimeSoundAndLayout_toConcreteVerifyPrior
      layout
      hPrior)
    hLatest

end DirectParentOnlyProductionConcreteFPrimePriorVerifierCertified

end DirectCcsFPrime
