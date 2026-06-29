import DirectCcsFPrime.ProofSystem.Production.Security.DirectParentOnlyProductionPriorOpeningFunctional
import DirectCcsFPrime.ProofSystem.Production.Impl.SuperNeoReuse.DirectParentOnlyProductionSuperNeoReuse

/-!
Compressed-prior opening for the Section 7.1-backed production context.

This module composes the production context whose stages reuse upstream
SuperNeo Section 7.1 authority with the existing prior-authority opening
boundary. The compressed prior proof is accepted only when it opens to
proof-carrying folded F' authority for the induced production transition.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionSuperNeoReusePriorOpening

/-- Section 7.1-backed production context. -/
abbrev ProductionContext :=
  DirectParentOnlyProductionSuperNeoReuse.ProductionContext

/-- Opaque prior-proof opener for the induced production context. -/
abbrev PriorAuthorityOpener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) :=
  DirectParentOnlyProductionPriorOpening.PriorAuthorityOpener
    (PriorProof := PriorProof)
    ctx.toProductionContext

/-- Verifier induced by a prior-authority opener for the induced context. -/
abbrev VerifyWithAuthorityOpener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx) :
    Nat →
      PriorProof →
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
        Prop :=
  DirectParentOnlyProductionPriorOpening.VerifyWithAuthorityOpener
    ctx.toProductionContext
    opener

/-- Concrete compressed-verifier opening certificate for the induced context. -/
abbrev PriorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop) :=
  DirectParentOnlyProductionPriorOpening.PriorVerifierAuthorityOpening
    ctx.toProductionContext
    VerifyPrior

/-- Terminal acceptance through an opener-induced verifier. -/
abbrev AcceptedTerminalWithAuthorityOpener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx)
    (priorSteps : Nat)
    (priorProof : PriorProof)
    (priorImage nextImage :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary)
    (latestProof : Unit) : Prop :=
  FPrimeInduction.TerminalCompressionAccepted
    (CompressedFPrimeAuthority.Accepts
      (VerifyWithAuthorityOpener ctx opener))
    (DirectParentOnlyProductionSoundness.VerifyLatestStep
      ctx.toProductionContext)
    priorSteps
    priorProof
    priorImage
    nextImage
    latestProof

/-- Terminal acceptance through an externally defined compressed verifier. -/
abbrev AcceptedTerminalWithPriorVerifier
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop)
    (priorSteps : Nat)
    (priorProof : PriorProof)
    (priorImage nextImage :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary)
    (latestProof : Unit) : Prop :=
  FPrimeInduction.TerminalCompressionAccepted
    (CompressedFPrimeAuthority.Accepts VerifyPrior)
    (DirectParentOnlyProductionSoundness.VerifyLatestStep
      ctx.toProductionContext)
    priorSteps
    priorProof
    priorImage
    nextImage
    latestProof

/-- Sound verifier induced by an authority opener. -/
def soundVerifier_of_authority_opener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.SoundPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionPriorOpening.soundVerifier_of_authority_opener
    ctx.toProductionContext
    opener

/-- Sound verifier induced by a concrete verifier plus an opening certificate. -/
def soundVerifier_of_priorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.SoundPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionPriorOpening.soundVerifier_of_priorVerifierAuthorityOpening
    ctx.toProductionContext
    opening

/--
The opener-induced verifier over the Section 7.1-backed context is same-proof
functional.
-/
theorem proofFunctional_of_authority_opener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx) :
    CompressedFPrimeAuthority.ProofFunctional
      (VerifyWithAuthorityOpener ctx opener) :=
  DirectParentOnlyProductionPriorOpening.proofFunctional_of_authority_opener
    ctx.toProductionContext
    opener

/--
The opener-induced `SoundVerifier` over the Section 7.1-backed context is
same-proof functional.
-/
theorem soundVerifier_of_authority_opener_proofFunctional
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx) :
    CompressedFPrimeAuthority.SoundVerifier.ProofFunctional
      (soundVerifier_of_authority_opener ctx opener) :=
  DirectParentOnlyProductionPriorOpening.soundVerifier_of_authority_opener_proofFunctional
    ctx.toProductionContext
    opener

/--
An externally defined verifier with a fixed opening certificate over the
Section 7.1-backed context is same-proof functional.
-/
theorem proofFunctional_of_priorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior) :
    CompressedFPrimeAuthority.ProofFunctional VerifyPrior :=
  DirectParentOnlyProductionPriorOpening.proofFunctional_of_priorVerifierAuthorityOpening
    ctx.toProductionContext
    opening

/--
The `SoundVerifier` induced by a fixed opening certificate over the
Section 7.1-backed context is same-proof functional.
-/
theorem soundVerifier_of_priorVerifierAuthorityOpening_proofFunctional
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior) :
    CompressedFPrimeAuthority.SoundVerifier.ProofFunctional
      (soundVerifier_of_priorVerifierAuthorityOpening ctx opening) :=
  DirectParentOnlyProductionPriorOpening.soundVerifier_of_priorVerifierAuthorityOpening_proofFunctional
    ctx.toProductionContext
    opening

/--
An opener-induced verifier over the Section 7.1-backed context cannot accept a
proof whose opener returns no authority.
-/
theorem verifyWithAuthorityOpener_openAuthority_ne_none
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hVerify : VerifyWithAuthorityOpener ctx opener steps proof image) :
    opener.openAuthority proof ≠ none :=
  DirectParentOnlyProductionPriorOpening.verifyWithAuthorityOpener_openAuthority_ne_none
    ctx.toProductionContext
    opener
    hVerify

/--
If the Section 7.1-backed opener-induced verifier accepts and the opener
returns a concrete authority, that exact authority accepts the same
`(steps, image)` pair.
-/
theorem verifyWithAuthorityOpener_opened_authority_accepts_of_open
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {authority :
      DirectParentOnlyProductionSoundness.ProofCarryingPriorProof
        ctx.toProductionContext}
    (hVerify : VerifyWithAuthorityOpener ctx opener steps proof image)
    (hOpen : opener.openAuthority proof = some authority) :
    FoldedFPrimeAuthority.Accepts
      (Transition :=
        DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
      (initial := ctx.initial)
      steps
      authority
      image :=
  DirectParentOnlyProductionPriorOpening.verifyWithAuthorityOpener_opened_authority_accepts_of_open
    ctx.toProductionContext
    opener
    hVerify
    hOpen

/--
An opener-induced verifier over the Section 7.1-backed context reaches every
prior image it accepts.
-/
theorem authority_opener_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hVerify : VerifyWithAuthorityOpener ctx opener steps proof image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition ctx.toProductionContext)
      ctx.initial
      steps
      image :=
  DirectParentOnlyProductionPriorOpening.authority_opener_reaches_prior
    ctx.toProductionContext
    opener
    hVerify

/--
An opener-induced verifier over the Section 7.1-backed context cannot accept
an unreachable prior image.
-/
theorem authority_opener_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hVerify : VerifyWithAuthorityOpener ctx opener steps proof image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition ctx.toProductionContext)
        ctx.initial
        steps
        image) :
    False :=
  DirectParentOnlyProductionPriorOpening.authority_opener_cannot_accept_unreachable_prior
    ctx.toProductionContext
    opener
    hVerify
    hUnreachable

/--
Accepted concrete verifier output over the Section 7.1-backed context exposes
an opened proof-carrying folded authority for the same `(steps, image)` pair.
-/
theorem priorVerifierAuthorityOpening_opened_authority
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior)
    {steps : Nat}
    {proof : PriorProof}
    {image : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hVerify : VerifyPrior steps proof image) :
    ∃ authority :
        DirectParentOnlyProductionSoundness.ProofCarryingPriorProof
          ctx.toProductionContext,
      opening.opener.openAuthority proof = some authority ∧
        FoldedFPrimeAuthority.Accepts
          (Transition :=
            DirectParentOnlyProductionSoundness.Transition
              ctx.toProductionContext)
          (initial := ctx.initial)
          steps
          authority
          image :=
  DirectParentOnlyProductionPriorOpening.priorVerifierAuthorityOpening_opened_authority
    opening
    hVerify

/--
A concrete verifier with a Section 7.1-backed opening certificate cannot accept
a proof whose fixed opener returns no authority.
-/
theorem priorVerifierAuthorityOpening_openAuthority_ne_none
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior)
    {steps : Nat}
    {proof : PriorProof}
    {image : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hVerify : VerifyPrior steps proof image) :
    opening.opener.openAuthority proof ≠ none :=
  DirectParentOnlyProductionPriorOpening.priorVerifierAuthorityOpening_openAuthority_ne_none
    opening
    hVerify

/--
If a concrete verifier accepts over the Section 7.1-backed context and its
fixed opener returns a concrete authority, that exact authority accepts the
same `(steps, image)` pair.
-/
theorem priorVerifierAuthorityOpening_opened_authority_accepts_of_open
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior)
    {steps : Nat}
    {proof : PriorProof}
    {image : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {authority :
      DirectParentOnlyProductionSoundness.ProofCarryingPriorProof
        ctx.toProductionContext}
    (hVerify : VerifyPrior steps proof image)
    (hOpen : opening.opener.openAuthority proof = some authority) :
    FoldedFPrimeAuthority.Accepts
      (Transition :=
        DirectParentOnlyProductionSoundness.Transition
          ctx.toProductionContext)
      (initial := ctx.initial)
      steps
      authority
      image :=
  DirectParentOnlyProductionPriorOpening.priorVerifierAuthorityOpening_opened_authority_accepts_of_open
    opening
    hVerify
    hOpen

/--
Accepted concrete verifier output over the Section 7.1-backed context reaches
the same prior public image.
-/
theorem priorVerifierAuthorityOpening_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior)
    {steps : Nat}
    {proof : PriorProof}
    {image : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hVerify : VerifyPrior steps proof image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition ctx.toProductionContext)
      ctx.initial
      steps
      image :=
  DirectParentOnlyProductionPriorOpening.priorVerifierAuthorityOpening_reaches_prior
    ctx.toProductionContext
    opening
    hVerify

/--
A concrete verifier with a Section 7.1-backed opening certificate cannot accept
an unreachable prior image.
-/
theorem priorVerifierAuthorityOpening_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior)
    {steps : Nat}
    {proof : PriorProof}
    {image : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hVerify : VerifyPrior steps proof image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition ctx.toProductionContext)
        ctx.initial
        steps
        image) :
    False :=
  DirectParentOnlyProductionPriorOpening.priorVerifierAuthorityOpening_cannot_accept_unreachable_prior
    ctx.toProductionContext
    opening
    hVerify
    hUnreachable

/--
Opener-induced endpoint for the Section 7.1-backed production context.

The conclusion includes prior/final reachability, public-image invariants,
pointwise private-child audit, and contextual `Pi_CCS -> Pi_RLC` stage audit.
-/
theorem audited_public_endpoint_with_stage_audit_of_authority_opener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hAccepted :
      AcceptedTerminalWithAuthorityOpener
        ctx
        opener
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      DirectParentOnlyProductionSuperNeoReuse.ProductionContext.AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.AuditedPublicEndpointWithStageAudit
      ctx
      priorSteps
      priorImage
      nextImage
      altNext :=
  DirectParentOnlyProductionPriorOpening.audited_public_endpoint_with_stage_audit_of_authority_opener
    ctx.toProductionContext
    opener
    hAccepted
    hAlt

/--
Concrete-verifier endpoint for the Section 7.1-backed production context.

This is the raw compressed-prior entry point: `VerifyPrior` is acceptable only
through `PriorVerifierAuthorityOpening`, which opens every accepted proof to
folded `F'` authority for the same `(steps, image)`.
-/
theorem audited_public_endpoint_with_stage_audit_of_priorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hAccepted :
      AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      DirectParentOnlyProductionSuperNeoReuse.ProductionContext.AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.AuditedPublicEndpointWithStageAudit
      ctx
      priorSteps
      priorImage
      nextImage
      altNext :=
  DirectParentOnlyProductionPriorOpening.audited_public_endpoint_with_stage_audit_of_priorVerifierAuthorityOpening
    ctx.toProductionContext
    opening
    hAccepted
    hAlt

/--
Opener-induced replay guard for the prior pair.

The same opaque proof cannot open to two different prior step/image pairs under
the same Section 7.1-backed production context.
-/
theorem terminal_prior_pair_functional_for_same_proof_of_authority_opener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx)
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      AcceptedTerminalWithAuthorityOpener
        ctx
        opener
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      AcceptedTerminalWithAuthorityOpener
        ctx
        opener
        priorStepsB
        priorProof
        priorImageB
        nextImageB
        latestProofB) :
    priorStepsA = priorStepsB ∧ priorImageA = priorImageB :=
  DirectParentOnlyProductionPriorOpening.terminal_prior_pair_functional_for_same_proof_of_authority_opener
    ctx.toProductionContext
    opener
    hA
    hB

/--
Concrete compressed-verifier replay guard for the prior pair.

The same opaque prior proof cannot be reused under the same externally opened
verifier to authorize a different prior step count or prior public image.
-/
theorem terminal_prior_pair_functional_for_same_proof_of_priorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior)
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsB
        priorProof
        priorImageB
        nextImageB
        latestProofB) :
    priorStepsA = priorStepsB ∧ priorImageA = priorImageB :=
  DirectParentOnlyProductionPriorOpening.terminal_prior_pair_functional_for_same_proof_of_priorVerifierAuthorityOpening
    ctx.toProductionContext
    opening
    hA
    hB

/--
Opener-induced replay guard for the terminal image.

Once the same opened prior proof fixes the prior pair, the latest transition is
forced to the same computed public image with the same pointwise-audited private
child table.
-/
theorem terminal_next_image_functional_for_same_proof_of_authority_opener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx)
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      AcceptedTerminalWithAuthorityOpener
        ctx
        opener
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      AcceptedTerminalWithAuthorityOpener
        ctx
        opener
        priorStepsB
        priorProof
        priorImageB
        nextImageB
        latestProofB) :
    nextImageA = nextImageB :=
  DirectParentOnlyProductionPriorOpening.terminal_next_image_functional_for_same_proof_of_authority_opener
    ctx.toProductionContext
    opener
    hA
    hB

/--
Concrete compressed-verifier replay guard for one opaque prior proof.

The same prior proof cannot be retargeted to another prior pair, and cannot
yield a different terminal public image under the same Section 7.1-backed
production context.
-/
theorem terminal_next_image_functional_for_same_proof_of_priorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior)
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsB
        priorProof
        priorImageB
        nextImageB
        latestProofB) :
    nextImageA = nextImageB :=
  DirectParentOnlyProductionPriorOpening.terminal_next_image_functional_for_same_proof_of_priorVerifierAuthorityOpening
    ctx.toProductionContext
    opening
    hA
    hB

end DirectParentOnlyProductionSuperNeoReusePriorOpening

end DirectCcsFPrime
