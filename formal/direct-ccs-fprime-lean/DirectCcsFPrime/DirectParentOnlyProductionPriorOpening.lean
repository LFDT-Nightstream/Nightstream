import DirectCcsFPrime.DirectParentOnlyProductionStageAudit

/-!
Production prior-authority opening surface for the parent-only direct CCS F'
path.

This module makes the compressed-prior requirement concrete without modeling
the compression scheme itself. An accepted opaque prior proof must open to
proof-carrying folded F' authority for the exact `(steps, image)` consumed by
terminal compression.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionPriorOpening

/--
Opaque prior-proof opener.

The opener is the theorem-facing artifact that a compressed proof verifier must
provide or imply: from a prior proof object, recover proof-carrying folded F'
authority under the production context's exact transition and initial image.
-/
structure PriorAuthorityOpener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params) where
  openAuthority :
    PriorProof →
      Option
        (DirectParentOnlyProductionSoundness.ProofCarryingPriorProof ctx)

/--
Verifier induced by a prior-authority opener.

Acceptance means the proof opens to folded F' authority and that authority
authorizes the same `(steps, image)` pair. The digest or proof bytes may be
compressed, but the accepted object is authority because of the opened
reachability proof, not because the image fields hash consistently.
-/
def VerifyWithAuthorityOpener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx) :
    Nat →
      PriorProof →
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
        Prop :=
  fun steps proof image =>
    ∃ authority : DirectParentOnlyProductionSoundness.ProofCarryingPriorProof ctx,
      opener.openAuthority proof = some authority ∧
        FoldedFPrimeAuthority.Accepts
          (Transition := DirectParentOnlyProductionSoundness.Transition ctx)
          (initial := ctx.initial)
          steps
          authority
          image

/--
The opener-induced verifier satisfies the production prior-opening obligation.
-/
theorem opensToProofCarryingPriorAuthority_of_verifyWithAuthorityOpener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSoundness.OpensToProofCarryingPriorAuthority
      ctx
      (VerifyWithAuthorityOpener ctx opener) := by
  intro steps proof image hVerify
  rcases hVerify with ⟨authority, _hOpen, hAccept⟩
  exact ⟨authority, hAccept⟩

/--
Sound prior verifier induced by an authority opener.
-/
def soundPriorVerifier_of_authority_opener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx) :
    DirectParentOnlyProductionSoundness.SoundPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionSoundness.soundPriorVerifier_of_opens_to_proof_carrying_prior_authority
    ctx
    (VerifyWithAuthorityOpener ctx opener)
    (opensToProofCarryingPriorAuthority_of_verifyWithAuthorityOpener ctx opener)

/--
Canonical `SoundVerifier` induced by an authority opener.

This is the object expected by the strict production theorem surfaces. Its
accepted predicate is exactly `VerifyWithAuthorityOpener ctx opener`.
-/
def soundVerifier_of_authority_opener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx) :
    CompressedFPrimeAuthority.SoundVerifier
      (Image := DirectParentOnlyProductionSoundness.PublicImage Digest Boundary)
      (Proof := PriorProof)
      (DirectParentOnlyProductionSoundness.Transition ctx)
      ctx.initial :=
  soundPriorVerifier_of_authority_opener ctx opener

/--
The opener-induced `SoundVerifier` accepts exactly the opener-induced verifier
predicate.
-/
theorem soundVerifier_of_authority_opener_accepts_iff
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary} :
    CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifier_of_authority_opener ctx opener)
        steps
        proof
        image ↔
      VerifyWithAuthorityOpener ctx opener steps proof image := by
  rfl

/--
An opener-induced verifier reaches every prior image it accepts.
-/
theorem authority_opener_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hVerify : VerifyWithAuthorityOpener ctx opener steps proof image) :
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition ctx)
      ctx.initial
      steps
      image :=
  DirectParentOnlyProductionSoundness.prior_verifier_reaches_prior_of_opening
    ctx
    (opensToProofCarryingPriorAuthority_of_verifyWithAuthorityOpener ctx opener)
    hVerify

/--
An opener-induced verifier cannot accept an unreachable prior image.
-/
theorem authority_opener_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hVerify : VerifyWithAuthorityOpener ctx opener steps proof image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition ctx)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable
    (authority_opener_reaches_prior ctx opener hVerify)

/--
One opener cannot open the same opaque prior proof to two different
proof-carrying authorities.
-/
theorem authority_opener_opened_authority_unique
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params}
    {opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx}
    {proof : PriorProof}
    {authorityA authorityB :
      DirectParentOnlyProductionSoundness.ProofCarryingPriorProof ctx}
    (hOpenA : opener.openAuthority proof = some authorityA)
    (hOpenB : opener.openAuthority proof = some authorityB) :
    authorityA = authorityB := by
  have hSome : some authorityA = some authorityB :=
    hOpenA.symm.trans hOpenB
  cases hSome
  rfl

/--
An opener-induced verifier cannot accept a proof whose opener returns no
authority.
-/
theorem verifyWithAuthorityOpener_openAuthority_ne_none
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hVerify : VerifyWithAuthorityOpener ctx opener steps proof image) :
    opener.openAuthority proof ≠ none := by
  rcases hVerify with ⟨authority, hOpen, _hAccept⟩
  intro hNone
  rw [hNone] at hOpen
  cases hOpen

/--
If an opener-induced verifier accepts and the opener returns a concrete
authority, that exact authority accepts the same `(steps, image)` pair.
-/
theorem verifyWithAuthorityOpener_opened_authority_accepts_of_open
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {authority :
      DirectParentOnlyProductionSoundness.ProofCarryingPriorProof ctx}
    (hVerify : VerifyWithAuthorityOpener ctx opener steps proof image)
    (hOpen : opener.openAuthority proof = some authority) :
    FoldedFPrimeAuthority.Accepts
      (Transition := DirectParentOnlyProductionSoundness.Transition ctx)
      (initial := ctx.initial)
      steps
      authority
      image := by
  rcases hVerify with ⟨openedAuthority, hOpened, hAccept⟩
  have hAuthority : openedAuthority = authority :=
    authority_opener_opened_authority_unique hOpened hOpen
  cases hAuthority
  exact hAccept

/--
An opener-induced verifier is functional for one opaque prior proof.

The same proof cannot authorize two different prior step/image pairs, because
both acceptances must open to the same proof-carrying folded `F'` authority.
-/
theorem verifyWithAuthorityOpener_functional_for_same_proof
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx)
    {stepsA stepsB : Nat}
    {proof : PriorProof}
    {imageA imageB : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hA : VerifyWithAuthorityOpener ctx opener stepsA proof imageA)
    (hB : VerifyWithAuthorityOpener ctx opener stepsB proof imageB) :
    stepsA = stepsB ∧ imageA = imageB := by
  rcases hA with ⟨authorityA, hOpenA, hAcceptA⟩
  rcases hB with ⟨authorityB, hOpenB, hAcceptB⟩
  have hAuthority : authorityA = authorityB :=
    authority_opener_opened_authority_unique hOpenA hOpenB
  subst hAuthority
  rcases hAcceptA with ⟨hStepsA, hImageA⟩
  rcases hAcceptB with ⟨hStepsB, hImageB⟩
  exact
    ⟨hStepsA.symm.trans hStepsB,
      hImageA.symm.trans hImageB⟩

/--
Terminal acceptances through an opener-induced verifier are functional on the
prior pair for one opaque prior proof.

This lifts the prior-verifier replay guard to the terminal object: the same
compressed prior proof cannot be consumed as authority for two different prior
step/image pairs.
-/
theorem terminal_prior_pair_functional_for_same_proof_of_authority_opener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx)
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.Accepts
          (VerifyWithAuthorityOpener ctx opener))
        (DirectParentOnlyProductionSoundness.VerifyLatestStep ctx)
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.Accepts
          (VerifyWithAuthorityOpener ctx opener))
        (DirectParentOnlyProductionSoundness.VerifyLatestStep ctx)
        priorStepsB
        priorProof
        priorImageB
        nextImageB
        latestProofB) :
    priorStepsA = priorStepsB ∧ priorImageA = priorImageB :=
  verifyWithAuthorityOpener_functional_for_same_proof
    ctx
    opener
    hA.priorAccepted
    hB.priorAccepted

/--
Concrete opening obligation for an externally defined compressed prior
verifier.

This is the adapter shape for a real verifier predicate: acceptance must imply
that a fixed opener recovers folded F' authority accepted for the same
`(steps, image)` pair.
-/
structure PriorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop) where
  opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx
  acceptedOpens :
    ∀ steps proof image,
      VerifyPrior steps proof image →
        ∃ authority :
            DirectParentOnlyProductionSoundness.ProofCarryingPriorProof ctx,
          opener.openAuthority proof = some authority ∧
            FoldedFPrimeAuthority.Accepts
              (Transition := DirectParentOnlyProductionSoundness.Transition ctx)
              (initial := ctx.initial)
              steps
              authority
              image

/--
A verifier equipped with a prior-authority opening satisfies the production
prior-opening obligation.
-/
theorem opensToProofCarryingPriorAuthority_of_priorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params}
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior) :
    DirectParentOnlyProductionSoundness.OpensToProofCarryingPriorAuthority
      ctx
      VerifyPrior := by
  intro steps proof image hVerify
  rcases opening.acceptedOpens steps proof image hVerify with
    ⟨authority, _hOpen, hAccept⟩
  exact ⟨authority, hAccept⟩

/--
Accepted concrete verifier output exposes an opened proof-carrying folded
authority for the same `(steps, image)` pair.
-/
theorem priorVerifierAuthorityOpening_opened_authority
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params}
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
        DirectParentOnlyProductionSoundness.ProofCarryingPriorProof ctx,
      opening.opener.openAuthority proof = some authority ∧
        FoldedFPrimeAuthority.Accepts
          (Transition := DirectParentOnlyProductionSoundness.Transition ctx)
          (initial := ctx.initial)
          steps
          authority
          image :=
  opening.acceptedOpens steps proof image hVerify

/--
A concrete verifier with an opening certificate cannot accept a proof whose
fixed opener returns no authority.
-/
theorem priorVerifierAuthorityOpening_openAuthority_ne_none
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params}
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
    opening.opener.openAuthority proof ≠ none := by
  rcases priorVerifierAuthorityOpening_opened_authority opening hVerify with
    ⟨authority, hOpen, _hAccept⟩
  intro hNone
  rw [hNone] at hOpen
  cases hOpen

/--
If a concrete verifier accepts and its fixed opener returns a concrete
authority, that exact authority accepts the same `(steps, image)` pair.
-/
theorem priorVerifierAuthorityOpening_opened_authority_accepts_of_open
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params}
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
      DirectParentOnlyProductionSoundness.ProofCarryingPriorProof ctx}
    (hVerify : VerifyPrior steps proof image)
    (hOpen : opening.opener.openAuthority proof = some authority) :
    FoldedFPrimeAuthority.Accepts
      (Transition := DirectParentOnlyProductionSoundness.Transition ctx)
      (initial := ctx.initial)
      steps
      authority
      image := by
  rcases priorVerifierAuthorityOpening_opened_authority opening hVerify with
    ⟨openedAuthority, hOpened, hAccept⟩
  have hAuthority : openedAuthority = authority :=
    authority_opener_opened_authority_unique
      (opener := opening.opener)
      hOpened
      hOpen
  cases hAuthority
  exact hAccept

/--
Sound prior verifier induced by an externally defined verifier plus an
authority-opening theorem.
-/
def soundPriorVerifier_of_priorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior) :
    DirectParentOnlyProductionSoundness.SoundPriorVerifier
      (PriorProof := PriorProof)
      ctx :=
  DirectParentOnlyProductionSoundness.soundPriorVerifier_of_opens_to_proof_carrying_prior_authority
    ctx
    VerifyPrior
    (opensToProofCarryingPriorAuthority_of_priorVerifierAuthorityOpening opening)

/--
Canonical `SoundVerifier` induced by an externally defined verifier plus its
prior-authority opening certificate.

This is the object expected by the strict production theorem surfaces. Its
accepted predicate is exactly the supplied `VerifyPrior`.
-/
def soundVerifier_of_priorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior) :
    CompressedFPrimeAuthority.SoundVerifier
      (Image := DirectParentOnlyProductionSoundness.PublicImage Digest Boundary)
      (Proof := PriorProof)
      (DirectParentOnlyProductionSoundness.Transition ctx)
      ctx.initial :=
  soundPriorVerifier_of_priorVerifierAuthorityOpening ctx opening

/--
The `SoundVerifier` induced by a concrete opening certificate accepts exactly
the supplied verifier predicate.
-/
theorem soundVerifier_of_priorVerifierAuthorityOpening_accepts_iff
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior)
    {steps : Nat}
    {proof : PriorProof}
    {image : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary} :
    CompressedFPrimeAuthority.SoundVerifier.Accepts
        (soundVerifier_of_priorVerifierAuthorityOpening ctx opening)
        steps
        proof
        image ↔
      VerifyPrior steps proof image := by
  rfl

/--
An externally defined verifier with an authority-opening theorem reaches every
prior image it accepts.
-/
theorem priorVerifierAuthorityOpening_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
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
      (DirectParentOnlyProductionSoundness.Transition ctx)
      ctx.initial
      steps
      image :=
  DirectParentOnlyProductionSoundness.prior_verifier_reaches_prior_of_opening
    ctx
    (opensToProofCarryingPriorAuthority_of_priorVerifierAuthorityOpening opening)
    hVerify

/--
An externally defined verifier with an authority-opening theorem cannot accept
an unreachable prior image.
-/
theorem priorVerifierAuthorityOpening_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
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
        (DirectParentOnlyProductionSoundness.Transition ctx)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable
    (priorVerifierAuthorityOpening_reaches_prior
      ctx
      opening
      hVerify)

/--
An externally defined verifier equipped with an authority-opening theorem is
functional for one opaque prior proof.

If the same proof is accepted for two prior step/image pairs, the fixed opener
forces both acceptances to refer to the same proof-carrying folded `F'`
authority, hence to the same pair.
-/
theorem priorVerifierAuthorityOpening_functional_for_same_proof
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior)
    {stepsA stepsB : Nat}
    {proof : PriorProof}
    {imageA imageB : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hA : VerifyPrior stepsA proof imageA)
    (hB : VerifyPrior stepsB proof imageB) :
    stepsA = stepsB ∧ imageA = imageB := by
  rcases opening.acceptedOpens stepsA proof imageA hA with
    ⟨authorityA, hOpenA, hAcceptA⟩
  rcases opening.acceptedOpens stepsB proof imageB hB with
    ⟨authorityB, hOpenB, hAcceptB⟩
  have hAuthority : authorityA = authorityB :=
    authority_opener_opened_authority_unique
      (opener := opening.opener)
      hOpenA
      hOpenB
  subst hAuthority
  rcases hAcceptA with ⟨hStepsA, hImageA⟩
  rcases hAcceptB with ⟨hStepsB, hImageB⟩
  exact
    ⟨hStepsA.symm.trans hStepsB,
      hImageA.symm.trans hImageB⟩

/--
Terminal acceptances through an externally defined verifier with an
authority-opening theorem are functional on the prior pair for one opaque prior
proof.

This is the terminal replay guard expected from the concrete compressed prior
verifier: reusing one proof cannot authorize a different prior public image or
step count.
-/
theorem terminal_prior_pair_functional_for_same_proof_of_priorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
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
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.Accepts VerifyPrior)
        (DirectParentOnlyProductionSoundness.VerifyLatestStep ctx)
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.Accepts VerifyPrior)
        (DirectParentOnlyProductionSoundness.VerifyLatestStep ctx)
        priorStepsB
        priorProof
        priorImageB
        nextImageB
        latestProofB) :
    priorStepsA = priorStepsB ∧ priorImageA = priorImageB :=
  priorVerifierAuthorityOpening_functional_for_same_proof
    ctx
    opening
    hA.priorAccepted
    hB.priorAccepted

/--
Strongest parent-only terminal conclusion for an opener-induced verifier:
reachability, computed latest images, unique pointwise private children, and
fixed-CE membership for that unique private child table.
-/
theorem terminal_soundness_with_fixed_ce_membership_of_authority_opener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.Accepts
          (VerifyWithAuthorityOpener ctx opener))
        (DirectParentOnlyProductionSoundness.VerifyLatestStep ctx)
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      DirectParentOnlyProductionSoundness.AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    DirectParentOnlyProductionSoundness.TerminalSoundness
        ctx
        priorSteps
        priorImage
        nextImage
        altNext ∧
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
        ParentOnlyAccumulatorStep.FixedCEChildMembership
          (n := n)
          params
          ctx.data.ce
          priorInputs ∧
        nextImage =
          DirectParentOnlyProductionSoundness.ComputedNextImage
            ctx
            priorSteps
            priorImage
            priorInputs ∧
        altNext =
          DirectParentOnlyProductionSoundness.ComputedNextImage
            ctx
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
              otherInputs = priorInputs) :=
  DirectParentOnlyProductionChildMembership.terminal_soundness_with_unique_children_and_fixed_ce_membership_of_prior_verifier_opening
    ctx
    (opensToProofCarryingPriorAuthority_of_verifyWithAuthorityOpener ctx opener)
    hAccepted
    hAlt

/--
Strongest parent-only terminal conclusion for an externally defined compressed
prior verifier equipped with a prior-authority opening theorem.
-/
theorem terminal_soundness_with_fixed_ce_membership_of_priorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.Accepts VerifyPrior)
        (DirectParentOnlyProductionSoundness.VerifyLatestStep ctx)
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      DirectParentOnlyProductionSoundness.AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    DirectParentOnlyProductionSoundness.TerminalSoundness
        ctx
        priorSteps
        priorImage
        nextImage
        altNext ∧
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
        ParentOnlyAccumulatorStep.FixedCEChildMembership
          (n := n)
          params
          ctx.data.ce
          priorInputs ∧
        nextImage =
          DirectParentOnlyProductionSoundness.ComputedNextImage
            ctx
            priorSteps
            priorImage
            priorInputs ∧
        altNext =
          DirectParentOnlyProductionSoundness.ComputedNextImage
            ctx
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
              otherInputs = priorInputs) :=
  DirectParentOnlyProductionChildMembership.terminal_soundness_with_unique_children_and_fixed_ce_membership_of_prior_verifier_opening
    ctx
    (opensToProofCarryingPriorAuthority_of_priorVerifierAuthorityOpening opening)
    hAccepted
    hAlt

/--
Parent-only terminal conclusion with the explicit pointwise child audit trail
for an opener-induced verifier.

This is the same compressed-prior opening boundary as
`terminal_soundness_with_fixed_ce_membership_of_authority_opener`, but exposes
the non-aggregate private `Pi_DEC` facts carried by the unique child table:
accepted private decomposition, binary fixed-length child columns, per-column
Goldilocks recomposition, witness-table identity, and next-`Pi_CCS` wire
identity.
-/
theorem terminal_soundness_with_pointwise_child_audit_trail_of_authority_opener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.Accepts
          (VerifyWithAuthorityOpener ctx opener))
        (DirectParentOnlyProductionSoundness.VerifyLatestStep ctx)
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      DirectParentOnlyProductionSoundness.AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    DirectParentOnlyProductionSoundness.TerminalSoundness
        ctx
        priorSteps
        priorImage
        nextImage
        altNext ∧
      DirectParentOnlyProductionChildMembership.TerminalChildAuditTrail
        ctx
        priorSteps
        priorImage
        nextImage
        altNext :=
  DirectParentOnlyProductionChildMembership.terminal_soundness_with_pointwise_child_audit_trail_of_prior_verifier_opening
    ctx
    (opensToProofCarryingPriorAuthority_of_verifyWithAuthorityOpener ctx opener)
    hAccepted
    hAlt

/--
Parent-only terminal conclusion with the explicit pointwise child audit trail
for an externally defined compressed prior verifier equipped with a
prior-authority opening theorem.
-/
theorem terminal_soundness_with_pointwise_child_audit_trail_of_priorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.Accepts VerifyPrior)
        (DirectParentOnlyProductionSoundness.VerifyLatestStep ctx)
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      DirectParentOnlyProductionSoundness.AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    DirectParentOnlyProductionSoundness.TerminalSoundness
        ctx
        priorSteps
        priorImage
        nextImage
        altNext ∧
      DirectParentOnlyProductionChildMembership.TerminalChildAuditTrail
        ctx
        priorSteps
        priorImage
        nextImage
        altNext :=
  DirectParentOnlyProductionChildMembership.terminal_soundness_with_pointwise_child_audit_trail_of_prior_verifier_opening
    ctx
    (opensToProofCarryingPriorAuthority_of_priorVerifierAuthorityOpening opening)
    hAccepted
    hAlt

/--
Opener-induced terminal conclusion with explicit prior reachability.

This packages the two authority facts that terminal compression relies on:
the opaque prior proof opens to a folded `F'` authority for the exact prior
`(steps, image)` pair, and the accepted latest step then reaches the terminal
image while exposing the pointwise private child audit trail.
-/
theorem terminal_soundness_with_prior_reachability_and_pointwise_child_audit_trail_of_authority_opener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.Accepts
          (VerifyWithAuthorityOpener ctx opener))
        (DirectParentOnlyProductionSoundness.VerifyLatestStep ctx)
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      DirectParentOnlyProductionSoundness.AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition ctx)
        ctx.initial
        priorSteps
        priorImage ∧
      DirectParentOnlyProductionSoundness.TerminalSoundness
        ctx
        priorSteps
        priorImage
        nextImage
        altNext ∧
      DirectParentOnlyProductionChildMembership.TerminalChildAuditTrail
        ctx
        priorSteps
        priorImage
        nextImage
        altNext := by
  have hPriorAccepted :
      VerifyWithAuthorityOpener ctx opener priorSteps priorProof priorImage := by
    exact hAccepted.priorAccepted
  have hPriorReach :
      FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition ctx)
        ctx.initial
        priorSteps
        priorImage :=
    authority_opener_reaches_prior
      ctx
      opener
      hPriorAccepted
  have hTerminal :=
    terminal_soundness_with_pointwise_child_audit_trail_of_authority_opener
      ctx
      opener
      hAccepted
      hAlt
  exact ⟨hPriorReach, hTerminal.1, hTerminal.2⟩

/--
Concrete-verifier terminal conclusion with explicit prior reachability.

This is the production compressed-verifier endpoint: accepted verifier output
must open to folded `F'` authority for the prior image, and the accepted latest
step exposes terminal soundness plus the private `Pi_DEC` child audit trail.
-/
theorem terminal_soundness_with_prior_reachability_and_pointwise_child_audit_trail_of_priorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.Accepts VerifyPrior)
        (DirectParentOnlyProductionSoundness.VerifyLatestStep ctx)
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      DirectParentOnlyProductionSoundness.AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition ctx)
        ctx.initial
        priorSteps
        priorImage ∧
      DirectParentOnlyProductionSoundness.TerminalSoundness
        ctx
        priorSteps
        priorImage
        nextImage
        altNext ∧
      DirectParentOnlyProductionChildMembership.TerminalChildAuditTrail
        ctx
        priorSteps
        priorImage
        nextImage
        altNext := by
  have hPriorAccepted : VerifyPrior priorSteps priorProof priorImage := by
    exact hAccepted.priorAccepted
  have hPriorReach :
      FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition ctx)
        ctx.initial
        priorSteps
        priorImage :=
    priorVerifierAuthorityOpening_reaches_prior
      ctx
      opening
      hPriorAccepted
  have hTerminal :=
    terminal_soundness_with_pointwise_child_audit_trail_of_priorVerifierAuthorityOpening
      ctx
      opening
      hAccepted
      hAlt
  exact ⟨hPriorReach, hTerminal.1, hTerminal.2⟩

/--
Opener-induced terminal endpoint with flattened public-image facts and the
pointwise private-child audit trail.

This is the direct theorem surface for an opaque compressed proof that opens to
folded `F'` authority: prior reachability, terminal reachability, public-image
invariants, deterministic boundary update, and the private child audit trail
are all exposed in one conclusion.
-/
theorem audited_public_endpoint_of_authority_opener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.Accepts
          (VerifyWithAuthorityOpener ctx opener))
        (DirectParentOnlyProductionSoundness.VerifyLatestStep ctx)
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      DirectParentOnlyProductionSoundness.AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    DirectParentOnlyProductionEndpoint.AuditedPublicEndpoint
      ctx
      priorSteps
      priorImage
      nextImage
      altNext :=
  DirectParentOnlyProductionEndpoint.audited_public_endpoint_of_prior_verifier_opening
    ctx
    (opensToProofCarryingPriorAuthority_of_verifyWithAuthorityOpener ctx opener)
    hAccepted
    hAlt

/--
Concrete compressed-verifier endpoint with flattened public-image facts and the
pointwise private-child audit trail.

This is the production endpoint for an externally defined verifier once its
accepted proofs are shown to open to proof-carrying folded `F'` authority for
the same `(steps, image)` pair.
-/
theorem audited_public_endpoint_of_priorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.Accepts VerifyPrior)
        (DirectParentOnlyProductionSoundness.VerifyLatestStep ctx)
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      DirectParentOnlyProductionSoundness.AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    DirectParentOnlyProductionEndpoint.AuditedPublicEndpoint
      ctx
      priorSteps
      priorImage
      nextImage
      altNext :=
  DirectParentOnlyProductionEndpoint.audited_public_endpoint_of_prior_verifier_opening
    ctx
    (opensToProofCarryingPriorAuthority_of_priorVerifierAuthorityOpening opening)
    hAccepted
    hAlt

/--
Opener-induced endpoint with flattened public-image facts, child audit, and
contextual `Pi_CCS -> Pi_RLC` stage audit.
-/
theorem audited_public_endpoint_with_stage_audit_of_authority_opener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.Accepts
          (VerifyWithAuthorityOpener ctx opener))
        (DirectParentOnlyProductionSoundness.VerifyLatestStep ctx)
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      DirectParentOnlyProductionSoundness.AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    DirectParentOnlyProductionStageAudit.AuditedPublicEndpointWithStageAudit
      ctx
      priorSteps
      priorImage
      nextImage
      altNext :=
  DirectParentOnlyProductionStageAudit.audited_public_endpoint_with_stage_audit_of_prior_verifier_opening
    ctx
    (opensToProofCarryingPriorAuthority_of_verifyWithAuthorityOpener ctx opener)
    hAccepted
    hAlt

/--
Concrete compressed-verifier endpoint with flattened public-image facts, child
audit, and contextual `Pi_CCS -> Pi_RLC` stage audit.
-/
theorem audited_public_endpoint_with_stage_audit_of_priorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening : PriorVerifierAuthorityOpening ctx VerifyPrior)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.Accepts VerifyPrior)
        (DirectParentOnlyProductionSoundness.VerifyLatestStep ctx)
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      DirectParentOnlyProductionSoundness.AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    DirectParentOnlyProductionStageAudit.AuditedPublicEndpointWithStageAudit
      ctx
      priorSteps
      priorImage
      nextImage
      altNext :=
  DirectParentOnlyProductionStageAudit.audited_public_endpoint_with_stage_audit_of_prior_verifier_opening
    ctx
    (opensToProofCarryingPriorAuthority_of_priorVerifierAuthorityOpening opening)
    hAccepted
    hAlt

/--
Terminal acceptances through an opener-induced verifier are functional on the
final public image for one opaque prior proof.

Once the prior replay guard fixes the prior step/image pair, the second
accepted latest step is an alternate latest transition for the first terminal
object. Parent-only terminal soundness then forces both terminal images to be
the same computed latest image over the unique pointwise-valid private child
table.
-/
theorem terminal_next_image_functional_for_same_proof_of_authority_opener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (opener : PriorAuthorityOpener (PriorProof := PriorProof) ctx)
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.Accepts
          (VerifyWithAuthorityOpener ctx opener))
        (DirectParentOnlyProductionSoundness.VerifyLatestStep ctx)
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.Accepts
          (VerifyWithAuthorityOpener ctx opener))
        (DirectParentOnlyProductionSoundness.VerifyLatestStep ctx)
        priorStepsB
        priorProof
        priorImageB
        nextImageB
        latestProofB) :
    nextImageA = nextImageB := by
  rcases
      terminal_prior_pair_functional_for_same_proof_of_authority_opener
        ctx
        opener
        hA
        hB with
    ⟨hSteps, hPriorImage⟩
  subst priorStepsB
  subst priorImageB
  have hAltB :
      DirectParentOnlyProductionSoundness.AlternateLatestStep
        ctx
        priorStepsA
        priorImageA
        nextImageB := by
    exact hB.latestAccepted
  exact
    (terminal_soundness_with_fixed_ce_membership_of_authority_opener
      ctx
      opener
      hA
      hAltB).1.2.1

/--
Terminal acceptances through an externally defined verifier with an
authority-opening theorem are functional on the final public image for one
opaque prior proof.

This is the concrete compressed-verifier replay guard: the same prior proof
cannot be retargeted to another prior pair, and it also cannot yield a
different accepted terminal image under the same production context.
-/
theorem terminal_next_image_functional_for_same_proof_of_priorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
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
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.Accepts VerifyPrior)
        (DirectParentOnlyProductionSoundness.VerifyLatestStep ctx)
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.Accepts VerifyPrior)
        (DirectParentOnlyProductionSoundness.VerifyLatestStep ctx)
        priorStepsB
        priorProof
        priorImageB
        nextImageB
        latestProofB) :
    nextImageA = nextImageB := by
  rcases
      terminal_prior_pair_functional_for_same_proof_of_priorVerifierAuthorityOpening
        ctx
        opening
        hA
        hB with
    ⟨hSteps, hPriorImage⟩
  subst priorStepsB
  subst priorImageB
  have hAltB :
      DirectParentOnlyProductionSoundness.AlternateLatestStep
        ctx
        priorStepsA
        priorImageA
        nextImageB := by
    exact hB.latestAccepted
  exact
    (terminal_soundness_with_fixed_ce_membership_of_priorVerifierAuthorityOpening
      ctx
      opening
      hA
      hAltB).1.2.1

end DirectParentOnlyProductionPriorOpening

end DirectCcsFPrime
