import DirectCcsFPrime.DirectParentOnlyProductionConcreteFPrimePrior
import DirectCcsFPrime.DirectParentOnlyProductionSuperNeoReuseEndToEnd

/-!
Opening-level terminal public-IO verifier for the concrete prior F' path.

This module models the Rust terminal committed verifier shape directly:
accepted verification replays the compact image, Construction-2 boundary,
transcript, statement boundary, and terminal public-IO prefix/suffix checks.
Authority is not part of the verifier predicate. Instead, accepted verifier
evidence must open through the fixed opener to proof-carrying folded F'
authority for the same `(steps, image)` pair.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionConcreteFPrimePriorTerminalIOOpening

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
Terminal public-IO opening surface.

The verifier-visible fields model the implementation checks. The final two
fields are the minimal backend soundness boundary: an accepted bound terminal
IO statement opens through the fixed opener, and any opened authority for that
statement binds the same `(steps, image)` pair.
-/
structure TerminalIOOpeningSurface
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
  acceptedTerminalIOOpens :
    ∀ steps proof image publicIO,
      compactImageReplay steps proof image →
      construction2BoundaryReplay steps proof image →
      transcriptReplay steps proof image →
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
      proofStatement proof = canonicalStatement steps image →
        ∃ authority : ProofCarryingPriorProof ctx,
          openAuthority proof = some authority
  openedAuthorityBindsTerminalIO :
    ∀ steps proof image publicIO authority,
      compactImageReplay steps proof image →
      construction2BoundaryReplay steps proof image →
      transcriptReplay steps proof image →
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
      proofStatement proof = canonicalStatement steps image →
      openAuthority proof = some authority →
        authority.steps = steps ∧ authority.image = image

/--
Terminal public-IO verifier predicate.

This predicate contains only verifier-visible replay and public-IO checks. It
does not require callers to prove that the authority opener returned a value.
-/
def VerifyPrior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : TerminalIOOpeningSurface (PriorProof := PriorProof) ctx) :
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

/-- Fixed authority opener induced by the terminal public-IO surface. -/
def authorityOpener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : TerminalIOOpeningSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
      (PriorProof := PriorProof)
      ctx where
  openAuthority := surface.openAuthority

/--
Audit evidence exposed by accepted terminal public-IO verification.

The evidence contains the verifier-visible checks, the replay-derived proof
statement equality, and the opened folded F' authority for the same public
pair.
-/
def AcceptedTerminalIOEvidence
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : TerminalIOOpeningSurface (PriorProof := PriorProof) ctx)
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

/-- Accepted terminal public-IO verification opens folded F' authority. -/
theorem acceptedOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : TerminalIOOpeningSurface (PriorProof := PriorProof) ctx) :
    ∀ steps proof image,
      VerifyPrior surface steps proof image →
        ∃ authority :
            DirectParentOnlyProductionSoundness.ProofCarryingPriorProof
              ctx.toProductionContext,
          (authorityOpener surface).openAuthority proof =
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
  rcases
    surface.acceptedTerminalIOOpens
      steps
      proof
      image
      publicIO
      hCompact
      hBoundaryReplay
      hTranscript
      hValid
      hBoundary
      hPublicIO
      hPrefix
      hSuffix
      hStatement with
    ⟨authority, hOpen⟩
  have hAccepts :
      FoldedFPrimeAuthority.Accepts
        (Transition :=
          DirectParentOnlyProductionSoundness.Transition
            ctx.toProductionContext)
        (initial := ctx.initial)
        steps
        authority
        image :=
    surface.openedAuthorityBindsTerminalIO
      steps
      proof
      image
      publicIO
      authority
      hCompact
      hBoundaryReplay
      hTranscript
      hValid
      hBoundary
      hPublicIO
      hPrefix
      hSuffix
      hStatement
      hOpen
  exact ⟨authority, hOpen, hAccepts⟩

/-- Accepted terminal public-IO verification exposes the full audit evidence. -/
theorem verifyPrior_evidence
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : TerminalIOOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : VerifyPrior surface steps proof image) :
    AcceptedTerminalIOEvidence surface steps proof image := by
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
    acceptedOpens
      surface
      steps
      proof
      image
      ⟨hCompact,
        hBoundaryReplay,
        hTranscript,
        hValid,
        hBoundary,
        ⟨publicIO, hPublicIO, hPrefix, hSuffix⟩⟩ with
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

/-- Terminal public-IO acceptance cannot hide a missing fixed opening. -/
theorem verifyPrior_openAuthority_ne_none
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : TerminalIOOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : VerifyPrior surface steps proof image) :
    surface.openAuthority proof ≠ none := by
  rcases acceptedOpens surface steps proof image hVerify with
    ⟨authority, hOpen, _hAccepts⟩
  intro hNone
  have hOpen' : surface.openAuthority proof = some authority := by
    simpa [authorityOpener] using hOpen
  rw [hNone] at hOpen'
  cases hOpen'

/--
If the fixed opener returns an authority for accepted terminal public-IO
verification, that exact authority accepts the same `(steps, image)` pair.
-/
theorem verifyPrior_openedAuthority_accepts
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : TerminalIOOpeningSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    {authority : ProofCarryingPriorProof ctx}
    (hVerify : VerifyPrior surface steps proof image)
    (hOpen : surface.openAuthority proof = some authority) :
    FoldedFPrimeAuthority.Accepts
      (Transition :=
        DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
      (initial := ctx.initial)
      steps
      authority
      image := by
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
    surface.openedAuthorityBindsTerminalIO
      steps
      proof
      image
      publicIO
      authority
      hCompact
      hBoundaryReplay
      hTranscript
      hValid
      hBoundary
      hPublicIO
      hPrefix
      hSuffix
      hStatement
      hOpen

/-- Accepted terminal public-IO verification reaches the claimed prior image. -/
theorem verifyPrior_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : TerminalIOOpeningSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : VerifyPrior surface steps proof image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
      ctx.initial
      steps
      image := by
  rcases acceptedOpens surface steps proof image hVerify with
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

/-- Terminal public-IO verification cannot accept an unreachable prior image. -/
theorem verifyPrior_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : TerminalIOOpeningSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : VerifyPrior surface steps proof image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable (verifyPrior_reaches_prior hVerify)

/-- Accepted terminal public-IO verification exposes prior-image invariants. -/
theorem verifyPrior_publicImageInvariants
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {surface : TerminalIOOpeningSurface (PriorProof := PriorProof) ctx}
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : VerifyPrior surface steps proof image) :
    DirectParentOnlyProductionConcreteFPrimePrior.AcceptedPriorPublicImageInvariants
      ctx
      steps
      image := by
  have hReach := verifyPrior_reaches_prior hVerify
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

/-- Canonical opening certificate induced by terminal public-IO verification. -/
def priorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : TerminalIOOpeningSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
      ctx
      (VerifyPrior surface) where
  opener := authorityOpener surface
  acceptedOpens := acceptedOpens surface

/-- Certified prior verifier induced by terminal public-IO verification. -/
def certifiedVerifier
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : TerminalIOOpeningSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.ofAcceptedOpens
    ctx
    (VerifyPrior surface)
    (authorityOpener surface)
    (acceptedOpens surface)

/-- The certified verifier uses the terminal public-IO predicate. -/
theorem certifiedVerifier_verify
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : TerminalIOOpeningSurface (PriorProof := PriorProof) ctx) :
    (certifiedVerifier surface).verify =
      VerifyPrior surface :=
  rfl

/-- Strict `SoundVerifier` induced by terminal public-IO verification. -/
def soundVerifier
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : TerminalIOOpeningSurface (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.SoundPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifier
    (certifiedVerifier surface)

/-- The induced strict verifier accepts exactly terminal public-IO verification. -/
theorem soundVerifier_accepts_iff
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : TerminalIOOpeningSurface (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary} :
    CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifier surface)
        steps
        proof
        image <->
      VerifyPrior surface steps proof image := by
  simpa [soundVerifier, certifiedVerifier_verify]
    using
      DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifier_accepts_iff
        (certifiedVerifier surface)

/-- Terminal public-IO verification is same-proof functional. -/
theorem proofFunctional
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : TerminalIOOpeningSurface (PriorProof := PriorProof) ctx) :
    CompressedFPrimeAuthority.ProofFunctional
      (VerifyPrior surface) := by
  intro stepsA stepsB proof imageA imageB hA hB
  exact
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.proofFunctional
      (certifiedVerifier surface)
      hA
      hB

/-- The induced strict terminal public-IO verifier is same-proof functional. -/
theorem soundVerifierProofFunctional
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : TerminalIOOpeningSurface (PriorProof := PriorProof) ctx) :
    CompressedFPrimeAuthority.SoundVerifier.ProofFunctional
      (soundVerifier surface) :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.soundVerifierProofFunctional
    (certifiedVerifier surface)

/--
Terminal public-IO prior-plus-latest theorem.

This is the implementation-facing call path: prior verification is just the
terminal public-IO predicate, and Lean packages the folded F' opening evidence
internally through the certified verifier.
-/
theorem certifiedEndToEndOfLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (surface : TerminalIOOpeningSurface (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage : PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      VerifyPrior surface priorSteps priorProof priorImage)
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
      (certifiedVerifier surface).opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage := by
  simpa [certifiedVerifier_verify] using
    DirectParentOnlyProductionSuperNeoReuseEndToEnd.certifiedSingleTerminalEndToEnd_ofCertifiedPriorVerifierLatestStep
      (certifiedVerifier surface)
      hPrior
      hLatest

end DirectParentOnlyProductionConcreteFPrimePriorTerminalIOOpening

end DirectCcsFPrime
