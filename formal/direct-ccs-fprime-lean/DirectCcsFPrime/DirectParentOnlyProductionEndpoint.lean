import DirectCcsFPrime.DirectParentOnlyProductionChildMembership

/-!
Packaged production endpoint for the parent-only direct CCS F' path.

This module keeps the base production context compact while exposing the final
compressed-prior conclusion callers want: accepted prior authority reaches the
exact prior public image, and the latest step exposes one unique audited
pointwise private `Pi_DEC` child table for the parent-only handle.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionEndpoint

/--
Combined production conclusion for the parent-only terminal path.

The prior compressed proof is authority only because it reaches the exact prior
public image. The private post-DEC children are accepted only because they are
pointwise authorized for the context-fixed CE relation and cannot be substituted
by another pointwise-valid table for the same parent source.
-/
def PriorReachabilityAndUniquePointwiseChildren
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (priorSteps : Nat)
    (priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary) : Prop :=
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
            otherInputs = priorInputs)

/--
Combined audit-facing production conclusion for the parent-only terminal path.

This is the stronger endpoint shape for implementation callers: prior authority
reaches the exact prior public image, terminal soundness holds, and the private
post-DEC children carry the concrete pointwise audit trail. In particular the
conclusion exposes accepted private `Pi_DEC`, fixed CE/Ajtai parameters,
binary fixed-length child columns, per-column Goldilocks recomposition,
witness-table identity, next-`Pi_CCS` wire identity, and uniqueness for the
same parent source.
-/
def PriorReachabilityAndPointwiseChildAuditTrail
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (priorSteps : Nat)
    (priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary) : Prop :=
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
      altNext

/--
Audit-facing endpoint with public-image facts flattened for callers.

This is the theorem shape the final implementation should consume: the prior
image is reachable, the accepted terminal image is reachable, public-image
invariants are preserved, the alternate latest transition is the same image,
and the private post-DEC children are exposed through the pointwise audit trail.
-/
def AuditedPublicEndpoint
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (priorSteps : Nat)
    (priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary) : Prop :=
  FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition ctx)
      ctx.initial
      priorSteps
      priorImage ∧
    FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition ctx)
      ctx.initial
      (priorSteps + 1)
      nextImage ∧
    nextImage = altNext ∧
    nextImage.accumulator.parentSource =
      altNext.accumulator.parentSource ∧
    nextImage.currentBoundary =
      ctx.computeBoundary priorSteps priorImage.currentBoundary ∧
    altNext.currentBoundary =
      ctx.computeBoundary priorSteps priorImage.currentBoundary ∧
    nextImage.step = priorSteps + 1 ∧
    ctx.initial.vkDigest = nextImage.vkDigest ∧
    ctx.initial.initialBoundary = nextImage.initialBoundary ∧
    Construction2DirectFPrime.WellFormed nextImage ∧
    DirectParentOnlyProductionChildMembership.TerminalChildAuditTrail
      ctx
      priorSteps
      priorImage
      nextImage
      altNext

/--
Flatten the existing prior-reachability plus child-audit conclusion into the
public endpoint shape.
-/
theorem audited_public_endpoint_of_prior_reachability_and_pointwise_child_audit_trail
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params}
    {priorSteps : Nat}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (h :
      PriorReachabilityAndPointwiseChildAuditTrail
        ctx
        priorSteps
        priorImage
        nextImage
        altNext) :
    AuditedPublicEndpoint
      ctx
      priorSteps
      priorImage
      nextImage
      altNext := by
  rcases h with ⟨hPriorReach, hTerminal, hAudit⟩
  rcases hTerminal with
    ⟨hFinalReach,
      hSameImage,
      hParentSource,
      _hChildren,
      hNextBoundary,
      hAltBoundary,
      hStep,
      hVk,
      hInitialBoundary,
      hWellFormed⟩
  exact
    ⟨hPriorReach,
      hFinalReach,
      hSameImage,
      hParentSource,
      hNextBoundary,
      hAltBoundary,
      hStep,
      hVk,
      hInitialBoundary,
      hWellFormed,
      hAudit⟩

/--
Production terminal endpoint with explicit prior reachability and unique
pointwise private children.
-/
theorem terminal_soundness_with_prior_reachability_and_unique_pointwise_children
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (verifier :
      DirectParentOnlyProductionSoundness.SoundPriorVerifier
        (PriorProof := PriorProof)
        ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hAccepted :
      DirectParentOnlyProductionSoundness.AcceptedTerminal
        ctx
        verifier
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
    PriorReachabilityAndUniquePointwiseChildren
      ctx
      priorSteps
      priorImage
      nextImage
      altNext := by
  have hPrior :
      FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition ctx)
        ctx.initial
        priorSteps
        priorImage :=
    DirectParentOnlyProductionSoundness.soundPriorVerifier_reaches_prior
      ctx
      verifier
      hAccepted.priorAccepted
  have hTerminal :=
    DirectParentOnlyProductionSoundness.terminal_soundness_with_unique_pointwise_children
      ctx
      verifier
      hAccepted
      hAlt
  exact ⟨hPrior, hTerminal.1, hTerminal.2⟩

/--
Raw compressed-verifier endpoint with explicit prior reachability and unique
pointwise private children.

The compressed verifier may remain opaque, but this theorem consumes its
opening theorem to folded `F'` authority for the same `(steps, image)` pair.
-/
theorem terminal_soundness_with_prior_reachability_and_unique_pointwise_children_of_prior_verifier_opening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (hOpens :
      DirectParentOnlyProductionSoundness.OpensToProofCarryingPriorAuthority
        ctx
        VerifyPrior)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
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
    PriorReachabilityAndUniquePointwiseChildren
      ctx
      priorSteps
      priorImage
      nextImage
      altNext := by
  have hPriorAccepted : VerifyPrior priorSteps priorProof priorImage := by
    exact hAccepted.priorAccepted
  have hPrior :
      FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition ctx)
        ctx.initial
        priorSteps
        priorImage :=
    DirectParentOnlyProductionSoundness.prior_verifier_reaches_prior_of_opening
      ctx
      hOpens
      hPriorAccepted
  have hTerminal :=
    DirectParentOnlyProductionSoundness.terminal_soundness_with_unique_pointwise_children_of_prior_verifier_opening
      ctx
      hOpens
      hAccepted
      hAlt
  exact ⟨hPrior, hTerminal.1, hTerminal.2⟩

/--
Production terminal endpoint with explicit prior reachability and the
pointwise private-child audit trail.
-/
theorem terminal_soundness_with_prior_reachability_and_pointwise_child_audit_trail
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (verifier :
      DirectParentOnlyProductionSoundness.SoundPriorVerifier
        (PriorProof := PriorProof)
        ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hAccepted :
      DirectParentOnlyProductionSoundness.AcceptedTerminal
        ctx
        verifier
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
    PriorReachabilityAndPointwiseChildAuditTrail
      ctx
      priorSteps
      priorImage
      nextImage
      altNext := by
  have hPrior :
      FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition ctx)
        ctx.initial
        priorSteps
        priorImage :=
    DirectParentOnlyProductionSoundness.soundPriorVerifier_reaches_prior
      ctx
      verifier
      hAccepted.priorAccepted
  have hTerminal :=
    DirectParentOnlyProductionChildMembership.terminal_soundness_with_pointwise_child_audit_trail
      ctx
      verifier
      hAccepted
      hAlt
  exact ⟨hPrior, hTerminal.1, hTerminal.2⟩

/--
Production endpoint with flattened public-image facts and the pointwise
private-child audit trail.
-/
theorem audited_public_endpoint_of_sound_verifier
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (verifier :
      DirectParentOnlyProductionSoundness.SoundPriorVerifier
        (PriorProof := PriorProof)
        ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hAccepted :
      DirectParentOnlyProductionSoundness.AcceptedTerminal
        ctx
        verifier
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
    AuditedPublicEndpoint
      ctx
      priorSteps
      priorImage
      nextImage
      altNext :=
  audited_public_endpoint_of_prior_reachability_and_pointwise_child_audit_trail
    (terminal_soundness_with_prior_reachability_and_pointwise_child_audit_trail
      ctx
      verifier
      hAccepted
      hAlt)

/--
Raw compressed-verifier endpoint with explicit prior reachability and the
pointwise private-child audit trail.

The compressed verifier may remain opaque, but this theorem consumes its
opening theorem to folded `F'` authority for the same `(steps, image)` pair.
-/
theorem terminal_soundness_with_prior_reachability_and_pointwise_child_audit_trail_of_prior_verifier_opening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (hOpens :
      DirectParentOnlyProductionSoundness.OpensToProofCarryingPriorAuthority
        ctx
        VerifyPrior)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
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
    PriorReachabilityAndPointwiseChildAuditTrail
      ctx
      priorSteps
      priorImage
      nextImage
      altNext := by
  have hPriorAccepted : VerifyPrior priorSteps priorProof priorImage := by
    exact hAccepted.priorAccepted
  have hPrior :
      FPrimeInduction.Reachable
        (DirectParentOnlyProductionSoundness.Transition ctx)
        ctx.initial
        priorSteps
        priorImage :=
    DirectParentOnlyProductionSoundness.prior_verifier_reaches_prior_of_opening
      ctx
      hOpens
      hPriorAccepted
  have hTerminal :=
    DirectParentOnlyProductionChildMembership.terminal_soundness_with_pointwise_child_audit_trail_of_prior_verifier_opening
      ctx
      hOpens
      hAccepted
      hAlt
  exact ⟨hPrior, hTerminal.1, hTerminal.2⟩

/--
Raw compressed-verifier endpoint with flattened public-image facts and the
pointwise private-child audit trail.
-/
theorem audited_public_endpoint_of_prior_verifier_opening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (hOpens :
      DirectParentOnlyProductionSoundness.OpensToProofCarryingPriorAuthority
        ctx
        VerifyPrior)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
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
    AuditedPublicEndpoint
      ctx
      priorSteps
      priorImage
      nextImage
      altNext :=
  audited_public_endpoint_of_prior_reachability_and_pointwise_child_audit_trail
    (terminal_soundness_with_prior_reachability_and_pointwise_child_audit_trail_of_prior_verifier_opening
      ctx
      hOpens
      hAccepted
      hAlt)

end DirectParentOnlyProductionEndpoint

end DirectCcsFPrime
