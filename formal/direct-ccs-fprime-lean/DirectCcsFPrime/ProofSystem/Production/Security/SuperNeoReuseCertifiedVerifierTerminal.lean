import DirectCcsFPrime.ProofSystem.Production.Security.SuperNeoReuseCertifiedVerifierCore

/-!
Terminal audit package for the Section 7.1-backed certified prior verifier.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier

namespace CertifiedPriorVerifier

/--
Certified-verifier production endpoint with public-image invariants,
pointwise private-child audit, and contextual stage audit.

This is the single-acceptance theorem for the implementation-facing verifier:
accepted compressed prior proof authority comes from the fixed opener, while
the latest-step comparison is checked against the same context-fixed transition.
-/
theorem auditedPublicEndpointWithStageAudit
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier : CertifiedPriorVerifier (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProof : Unit}
    (hAccepted :
      AcceptedTerminal
        verifier
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      AlternateLatestStep
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
  DirectParentOnlyProductionSuperNeoReuse.ProductionContext.audited_public_endpoint_with_stage_audit_of_sound_verifier
    ctx
    (soundVerifier verifier)
    (acceptedTerminalWithSoundVerifier verifier hAccepted)
    hAlt

/--
Flattened computed-stage evidence for one certified terminal endpoint.

This exposes the exact private child table and deterministic `Pi_CCS -> Pi_RLC`
parent-source computation from the audited endpoint, without requiring callers
to peel apart the endpoint conjunction manually.
-/
theorem computedStageEndpointEvidence
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier : CertifiedPriorVerifier (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProof : Unit}
    (hAccepted :
      AcceptedTerminal
        verifier
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    ComputedStageEndpointEvidence
      ctx
      priorSteps
      priorImage
      nextImage
      altNext := by
  have hEndpoint :=
    auditedPublicEndpointWithStageAudit
      verifier
      hAccepted
      hAlt
  simpa
    [ComputedStageEndpointEvidence,
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.ComputedStageReplayEvidence,
      DirectParentOnlyProductionSuperNeoReuse.ProductionContext.AuditedPublicEndpointWithStageAudit]
    using
      DirectParentOnlyProductionStageAudit.computed_stage_evidence_of_terminal_stage_audit_trail
        hEndpoint.2

/--
Single-call certified terminal audit package.

This is the implementation-facing one-terminal theorem: an accepted compressed
prior proof opens to folded F' authority for the same prior pair, the terminal
endpoint satisfies the production public-image and child/stage audit, and the
computed-stage endpoint evidence exposes the exact pointwise child table and
`Pi_CCS -> Pi_RLC` parent computation.
-/
theorem terminalAuditPackage
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier : CertifiedPriorVerifier (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProof : Unit}
    (hAccepted :
      AcceptedTerminal
        verifier
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    verifier.opening.opener.openAuthority priorProof ≠ none ∧
      (∃ authority :
          DirectParentOnlyProductionSoundness.ProofCarryingPriorProof
            ctx.toProductionContext,
        verifier.opening.opener.openAuthority priorProof = some authority ∧
          FoldedFPrimeAuthority.Accepts
            (Transition :=
              DirectParentOnlyProductionSoundness.Transition
                ctx.toProductionContext)
            (initial := ctx.initial)
            priorSteps
            authority
            priorImage) ∧
      DirectParentOnlyProductionSuperNeoReuse.ProductionContext.AuditedPublicEndpointWithStageAudit
        ctx
        priorSteps
        priorImage
        nextImage
        altNext ∧
      ComputedStageEndpointEvidence
        ctx
        priorSteps
        priorImage
        nextImage
        altNext :=
  ⟨DirectParentOnlyProductionSuperNeoReusePriorOpening.priorVerifierAuthorityOpening_openAuthority_ne_none
      ctx
      verifier.opening
      hAccepted.priorAccepted,
    DirectParentOnlyProductionSuperNeoReusePriorOpening.priorVerifierAuthorityOpening_opened_authority
      ctx
      verifier.opening
      hAccepted.priorAccepted,
    auditedPublicEndpointWithStageAudit
      verifier
      hAccepted
      hAlt,
    computedStageEndpointEvidence
      verifier
      hAccepted
      hAlt⟩

/--
Raw-verifier form of the single-call terminal audit package.

This is the direct implementation entry point: once the concrete compressed
verifier is paired with a fixed opener and the accepted-opens theorem, one
terminal acceptance exposes opened folded F' authority and the exact pointwise
stage evidence.
-/
theorem terminalAuditPackage_ofAcceptedOpens
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
                image)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProof : Unit}
    (hAccepted :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        verify
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    opener.openAuthority priorProof ≠ none ∧
      (∃ authority :
          DirectParentOnlyProductionSoundness.ProofCarryingPriorProof
            ctx.toProductionContext,
        opener.openAuthority priorProof = some authority ∧
          FoldedFPrimeAuthority.Accepts
            (Transition :=
              DirectParentOnlyProductionSoundness.Transition
                ctx.toProductionContext)
            (initial := ctx.initial)
            priorSteps
            authority
            priorImage) ∧
      DirectParentOnlyProductionSuperNeoReuse.ProductionContext.AuditedPublicEndpointWithStageAudit
        ctx
        priorSteps
        priorImage
        nextImage
        altNext ∧
      ComputedStageEndpointEvidence
        ctx
        priorSteps
        priorImage
        nextImage
        altNext := by
  let verifier := ofAcceptedOpens ctx verify opener acceptedOpens
  simpa [verifier, ofAcceptedOpens, AcceptedTerminal]
    using terminalAuditPackage verifier hAccepted hAlt

/--
Single-terminal no-swap theorem for private DEC children.

An arbitrary alternate child table must satisfy the full pointwise private DEC
requirements for the accepted parent source before Lean concludes that it is
the same table audited by the terminal `Pi_CCS -> Pi_RLC` computation.
-/
theorem terminalPointwiseChildTableUnique
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier : CertifiedPriorVerifier (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProof : Unit}
    (hAccepted :
      AcceptedTerminal
        verifier
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext)
    {otherInputs : DecDigitUniqueness.ColumnDigits n}
    (hOther :
      ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := ctx.parentHash.hashEncoded)
        (params := params)
        (ce := ctx.data.ce)
        (StatementEncodes :=
          ParentOpeningAuthorization.StatementEncodesByCommitment
            ctx.commitmentOfParent)
        priorImage.accumulator.parentSource
        otherInputs) :
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
      otherInputs = priorInputs ∧
      DirectParentOnlyProductionStageAudit.ParentSourceStageAudit
        ctx.toProductionContext
        priorSteps
        priorImage.accumulator
        priorInputs
        nextImage.accumulator.parentSource ∧
      DirectParentOnlyProductionStageAudit.ParentSourceStageAudit
        ctx.toProductionContext
        priorSteps
        priorImage.accumulator
        priorInputs
        altNext.accumulator.parentSource := by
  have hEndpoint :=
    auditedPublicEndpointWithStageAudit
      verifier
      hAccepted
      hAlt
  exact
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.pointwise_child_table_unique_of_pointwise_child_replay_binding
      (ctx := ctx)
      (by
        simpa
          [DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PointwiseChildReplayBinding]
          using hEndpoint.2)
      hOther

/--
Raw-verifier form of the single-terminal no-swap theorem for private DEC
children.

The alternate child table is accepted only through the full pointwise private
DEC requirements for the same parent source; satisfying an aggregate summary is
not enough to use this theorem.
-/
theorem terminalPointwiseChildTableUnique_ofAcceptedOpens
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
                image)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProof : Unit}
    (hAccepted :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        verify
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext)
    {otherInputs : DecDigitUniqueness.ColumnDigits n}
    (hOther :
      ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := ctx.parentHash.hashEncoded)
        (params := params)
        (ce := ctx.data.ce)
        (StatementEncodes :=
          ParentOpeningAuthorization.StatementEncodesByCommitment
            ctx.commitmentOfParent)
        priorImage.accumulator.parentSource
        otherInputs) :
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
      otherInputs = priorInputs ∧
      DirectParentOnlyProductionStageAudit.ParentSourceStageAudit
        ctx.toProductionContext
        priorSteps
        priorImage.accumulator
        priorInputs
        nextImage.accumulator.parentSource ∧
      DirectParentOnlyProductionStageAudit.ParentSourceStageAudit
        ctx.toProductionContext
        priorSteps
        priorImage.accumulator
        priorInputs
        altNext.accumulator.parentSource := by
  let verifier := ofAcceptedOpens ctx verify opener acceptedOpens
  simpa [verifier, ofAcceptedOpens, AcceptedTerminal]
    using
      terminalPointwiseChildTableUnique
        verifier
        hAccepted
        hAlt
        hOther

/-- Preferred short name for the one-terminal endpoint audit. -/
abbrev terminalEndpointAudit :=
  @auditedPublicEndpointWithStageAudit

/-- Preferred short name for one-terminal computed stage evidence. -/
abbrev terminalStageEvidence :=
  @computedStageEndpointEvidence

/-- Preferred short name for the one-terminal audit package. -/
abbrev terminalAudit :=
  @terminalAuditPackage

/-- Preferred short name for the raw-verifier one-terminal audit package. -/
abbrev terminalAuditOfAcceptedOpens :=
  @terminalAuditPackage_ofAcceptedOpens

/-- Preferred short name for one-terminal private-child no-swap. -/
abbrev terminalChildNoSwap :=
  @terminalPointwiseChildTableUnique

/-- Preferred short name for raw-verifier one-terminal private-child no-swap. -/
abbrev terminalChildNoSwapOfAcceptedOpens :=
  @terminalPointwiseChildTableUnique_ofAcceptedOpens

end CertifiedPriorVerifier

end DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier

end DirectCcsFPrime
