import DirectCcsFPrime.ProofSystem.Production.Security.DirectParentOnlyProductionSuperNeoReuseReplayEndpoint

/-!
Certified concrete compressed verifier for the Section 7.1-backed endpoint.

This module is the implementation-facing object for the compressed prior
verifier: callers provide one verifier predicate together with the fixed
opening certificate that turns every accepted proof into proof-carrying folded
F' authority for the same `(steps, image)` pair.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier

/-- Section 7.1-backed production context. -/
abbrev ProductionContext :=
  DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.ProductionContext

/--
Concrete compressed prior verifier plus its fixed authority-opening certificate.

This is the object the optimized endpoint should consume. The verifier may be
opaque, but it is certified only if accepted proofs open to proof-carrying
folded F' authority under the induced production transition.
-/
structure CertifiedPriorVerifier
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) where
  verify :
    Nat ->
      PriorProof ->
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary ->
        Prop
  opening :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
      ctx
      verify

namespace CertifiedPriorVerifier

/-- Terminal acceptance through a certified concrete compressed verifier. -/
abbrev AcceptedTerminal
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier : CertifiedPriorVerifier (PriorProof := PriorProof) ctx)
    (priorSteps : Nat)
    (priorProof : PriorProof)
    (priorImage nextImage :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary)
    (latestProof : Unit) : Prop :=
  DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
    ctx
    verifier.verify
    priorSteps
    priorProof
    priorImage
    nextImage
    latestProof

/-- Alternate latest transition for the certified verifier's context. -/
abbrev AlternateLatestStep
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (priorSteps : Nat)
    (priorImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary) :
    Prop :=
  DirectParentOnlyProductionSuperNeoReuse.ProductionContext.AlternateLatestStep
    ctx
    priorSteps
    priorImage
    altNext

/-- Flattened computed-stage evidence for one certified terminal endpoint. -/
abbrev ComputedStageEndpointEvidence
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (priorSteps : Nat)
    (priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary) : Prop :=
  DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.ComputedStageReplayEvidence
    ctx
    priorSteps
    priorImage
    nextImage
    altNext

/--
Explicit replay no-swap evidence for the audited private DEC child table.

This is the implementation-facing projection of computed replay evidence: it
names the audited child table and exposes the full pointwise uniqueness
quantifier for the replayed parent source.
-/
def ExplicitReplayNoSwapEvidence
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (priorSteps : Nat)
    (priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary) : Prop :=
  ∃ priorInputs,
    ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
      (n := n)
      (hashEncoded := ctx.parentHash.hashEncoded)
      (params := params)
      (ce := ctx.data.ce)
      (StatementEncodes :=
        ParentOpeningAuthorization.StatementEncodesByCommitment
          ctx.commitmentOfParent)
      priorImage.accumulator.parentSource
      priorInputs ∧
    DirectParentOnlyProductionChildMembership.PointwiseChildAuditTrail
      ctx.toProductionContext
      priorImage.accumulator.parentSource
      priorInputs ∧
    nextImage =
      DirectParentOnlyProductionSoundness.ComputedNextImage
        ctx.toProductionContext
        priorSteps
        priorImage
        priorInputs ∧
    altNext =
      DirectParentOnlyProductionSoundness.ComputedNextImage
        ctx.toProductionContext
        priorSteps
        priorImage
        priorInputs ∧
    (∀ otherInputs,
      ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := ctx.parentHash.hashEncoded)
        (params := params)
        (ce := ctx.data.ce)
        (StatementEncodes :=
          ParentOpeningAuthorization.StatementEncodesByCommitment
            ctx.commitmentOfParent)
        priorImage.accumulator.parentSource
        otherInputs →
          otherInputs = priorInputs)

/--
Build the certified verifier object from the raw implementation obligation.

The implementation supplies its concrete verifier predicate, a fixed opener,
and the exact theorem that every accepted proof opens to folded F' authority
for the same `(steps, image)` pair. This is the required authority boundary;
no digest self-consistency is accepted as authority here.
-/
def ofAcceptedOpens
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (verify :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop)
    (opener :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
        (PriorProof := PriorProof)
        ctx)
    (acceptedOpens :
      ∀ steps proof image,
        verify steps proof image →
          ∃ authority :
              DirectParentOnlyProductionSoundness.ProofCarryingPriorProof
                ctx.toProductionContext,
            opener.openAuthority proof = some authority ∧
              FoldedFPrimeAuthority.Accepts
                (Transition :=
                  DirectParentOnlyProductionSoundness.Transition
                    ctx.toProductionContext)
                (initial := ctx.initial)
                steps
                authority
                image) :
    CertifiedPriorVerifier (PriorProof := PriorProof) ctx where
  verify := verify
  opening := {
    opener := opener
    acceptedOpens := acceptedOpens
  }

/-- The certified verifier built from raw obligations uses the supplied predicate. -/
theorem ofAcceptedOpens_verify
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (verify :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop)
    (opener :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
        (PriorProof := PriorProof)
        ctx)
    (acceptedOpens :
      ∀ steps proof image,
        verify steps proof image →
          ∃ authority :
              DirectParentOnlyProductionSoundness.ProofCarryingPriorProof
                ctx.toProductionContext,
            opener.openAuthority proof = some authority ∧
              FoldedFPrimeAuthority.Accepts
                (Transition :=
                  DirectParentOnlyProductionSoundness.Transition
                    ctx.toProductionContext)
                (initial := ctx.initial)
                steps
                authority
                image) :
    (ofAcceptedOpens ctx verify opener acceptedOpens).verify = verify :=
  rfl

/-- The certified verifier induces the strict `SoundVerifier` object. -/
def soundVerifier
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier : CertifiedPriorVerifier (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.SoundPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionSuperNeoReusePriorOpening.soundVerifier_of_priorVerifierAuthorityOpening
    ctx
    verifier.opening

/-- The induced strict `SoundVerifier` accepts exactly the certified predicate. -/
theorem soundVerifier_accepts_iff
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier : CertifiedPriorVerifier (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary} :
    CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifier verifier)
        steps
        proof
        image <->
      verifier.verify steps proof image := by
  rfl

/--
Terminal acceptance through the certified verifier is terminal acceptance
through its induced strict `SoundVerifier`.
-/
theorem acceptedTerminalWithSoundVerifier
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier : CertifiedPriorVerifier (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProof : Unit}
    (hAccepted :
      AcceptedTerminal
        verifier
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.AcceptedTerminal
      ctx
      (soundVerifier verifier)
      priorSteps
      priorProof
      priorImage
      nextImage
      latestProof := by
  simpa
    [AcceptedTerminal,
      DirectParentOnlyProductionSuperNeoReuse.ProductionContext.AcceptedTerminal,
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier,
      DirectParentOnlyProductionSuperNeoReusePriorOpening.AcceptedTerminalWithPriorVerifier,
      soundVerifier,
      DirectParentOnlyProductionSuperNeoReusePriorOpening.soundVerifier_of_priorVerifierAuthorityOpening,
      DirectParentOnlyProductionPriorOpening.soundVerifier_of_priorVerifierAuthorityOpening,
      CompressedFPrimeAuthority.SoundVerifier.Accepts]
    using hAccepted

/--
The certified verifier is same-proof functional.

This replay-stability fact comes from the fixed opening certificate, not from a
bare `SoundVerifier`: one opaque proof cannot certify two different prior
`(steps, image)` pairs under the same fixed opener.
-/
theorem proofFunctional
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier : CertifiedPriorVerifier (PriorProof := PriorProof) ctx) :
    CompressedFPrimeAuthority.ProofFunctional verifier.verify :=
  DirectParentOnlyProductionSuperNeoReusePriorOpening.proofFunctional_of_priorVerifierAuthorityOpening
    ctx
    verifier.opening

/-- The induced strict `SoundVerifier` is same-proof functional. -/
theorem soundVerifierProofFunctional
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier : CertifiedPriorVerifier (PriorProof := PriorProof) ctx) :
    CompressedFPrimeAuthority.SoundVerifier.ProofFunctional
      (soundVerifier verifier) :=
  DirectParentOnlyProductionSuperNeoReusePriorOpening.soundVerifier_of_priorVerifierAuthorityOpening_proofFunctional
    ctx
    verifier.opening

/--
Every accepted certified prior proof exposes the concrete folded authority
returned by the verifier's fixed opener.

This is stronger than reachability alone: the same opaque proof is opened by
the certified opener, and that opened authority accepts the same `(steps,
image)` pair.
-/
theorem openedAuthority
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier : CertifiedPriorVerifier (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hVerify : verifier.verify steps proof image) :
    ∃ authority :
        DirectParentOnlyProductionSoundness.ProofCarryingPriorProof
          ctx.toProductionContext,
      verifier.opening.opener.openAuthority proof = some authority ∧
        FoldedFPrimeAuthority.Accepts
          (Transition :=
            DirectParentOnlyProductionSoundness.Transition
              ctx.toProductionContext)
          (initial := ctx.initial)
          steps
          authority
          image :=
  DirectParentOnlyProductionSuperNeoReusePriorOpening.priorVerifierAuthorityOpening_opened_authority
    ctx
    verifier.opening
    hVerify

/-- An accepted certified prior proof cannot fail to open under the fixed opener. -/
theorem openAuthority_ne_none
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier : CertifiedPriorVerifier (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hVerify : verifier.verify steps proof image) :
    verifier.opening.opener.openAuthority proof ≠ none :=
  DirectParentOnlyProductionSuperNeoReusePriorOpening.priorVerifierAuthorityOpening_openAuthority_ne_none
    ctx
    verifier.opening
    hVerify

/--
If the fixed opener returns a concrete authority for an accepted proof, that
exact authority accepts the same `(steps, image)` pair.
-/
theorem openedAuthority_accepts_of_open
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier : CertifiedPriorVerifier (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {authority :
      DirectParentOnlyProductionSoundness.ProofCarryingPriorProof
        ctx.toProductionContext}
    (hVerify : verifier.verify steps proof image)
    (hOpen : verifier.opening.opener.openAuthority proof = some authority) :
    FoldedFPrimeAuthority.Accepts
      (Transition :=
        DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
      (initial := ctx.initial)
      steps
      authority
      image :=
  DirectParentOnlyProductionSuperNeoReusePriorOpening.priorVerifierAuthorityOpening_opened_authority_accepts_of_open
    ctx
    verifier.opening
    hVerify
    hOpen

/-- Every accepted certified prior proof opens to folded F' reachability. -/
theorem reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier : CertifiedPriorVerifier (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hVerify : verifier.verify steps proof image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition ctx.toProductionContext)
      ctx.initial
      steps
      image :=
  DirectParentOnlyProductionSuperNeoReusePriorOpening.priorVerifierAuthorityOpening_reaches_prior
    ctx
    verifier.opening
    hVerify

/-- A certified verifier cannot accept an unreachable prior public image. -/
theorem cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier : CertifiedPriorVerifier (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hVerify : verifier.verify steps proof image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition ctx.toProductionContext)
        ctx.initial
        steps
        image) :
    False :=
  DirectParentOnlyProductionSuperNeoReusePriorOpening.priorVerifierAuthorityOpening_cannot_accept_unreachable_prior
    ctx
    verifier.opening
    hVerify
    hUnreachable

/-- Preferred short name for certified verifier acceptance equivalence. -/
abbrev verifyAcceptsIff :=
  @soundVerifier_accepts_iff

/-- Preferred short name for same-proof functionality. -/
abbrev sameProofFunctional :=
  @proofFunctional

/-- Preferred short name for the opened prior authority certificate. -/
abbrev priorAuthority :=
  @openedAuthority

/-- Preferred short name for nonempty prior authority opening. -/
abbrev priorProofOpens :=
  @openAuthority_ne_none

/-- Preferred short name for authority acceptance after opening. -/
abbrev priorAuthorityAccepts :=
  @openedAuthority_accepts_of_open

/-- Preferred short name for prior folded `F'` reachability. -/
abbrev priorReachable :=
  @reaches_prior

/-- Preferred short name for rejecting unreachable prior images. -/
abbrev rejectsUnreachablePrior :=
  @cannot_accept_unreachable_prior

end CertifiedPriorVerifier

end DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier

end DirectCcsFPrime
