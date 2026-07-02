import DirectCcsFPrime.ProofSystem.Production.Security.Endpoint

/-!
Production stage-audit surface for the parent-only direct CCS F' path.

This module keeps the optimized public handle parent-only, but exposes the
stage facts that matter for audit: the private child table feeds the contextual
`Pi_CCS` computation for the current step, and the parent source is the
deterministic `Pi_RLC` result over that exact output with the imported SuperNeo
stage statements attached.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionStageAudit

/-- Child-carrying prior handle used internally by the contextual stage code. -/
def childCarryingPrior
    {Digest : Type}
    {n : Nat}
    (prior : DirectParentOnlyProductionSoundness.AccHandle Digest)
    (priorInputs : DecDigitUniqueness.ColumnDigits n) :
    DirectTerminalSoundness.AccHandle Digest n where
  parentSource := prior.parentSource
  nextPiCCSInputs := priorInputs

/--
Audit facts for one parent source produced by the context's SuperNeo stages.

This is intentionally about the pointwise child table, not an aggregate digest
or norm summary. The source must be the deterministic `Pi_RLC` result over the
deterministic contextual `Pi_CCS` output for the same step and child-carrying
prior handle, and the imported SuperNeo stage statements must hold for the
contexts carried by those computed objects.
-/
def ParentSourceStageAudit
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (i : Nat)
    (prior : DirectParentOnlyProductionSoundness.AccHandle Digest)
    (priorInputs : DecDigitUniqueness.ColumnDigits n)
    (source : DigestParentBinding.Source Digest) : Prop :=
  let childPrior := childCarryingPrior prior priorInputs
  let out := ctx.stage.computePiCCS i childPrior
  source = ctx.stage.computePiRLC i out ∧
    out.step = i ∧
    SuperNeo.PiCCSInterface.piCCSStrongStatement out.ctx ∧
    SuperNeo.PiDECInterface.piDECKnowledgeStatement out.ctx ∧
    SuperNeo.PiRLCInterface.piRLCWeakStatement
      (ctx.stage.piRLCContext out (ctx.stage.computePiRLC i out)) ∧
    SuperNeo.PiDECInterface.piDECKnowledgeStatement
      (ctx.stage.piRLCContext out (ctx.stage.computePiRLC i out))

/--
An accepted parent-only stage relation exposes the deterministic contextual
stage audit facts.
-/
theorem parent_source_stage_audit_of_parentSourceFromPiStages
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    {i : Nat}
    {prior : DirectParentOnlyProductionSoundness.AccHandle Digest}
    {priorInputs : DecDigitUniqueness.ColumnDigits n}
    {source : DigestParentBinding.Source Digest}
    (hSource :
      ParentOnlyAccumulatorStep.ParentSourceFromPiStages
        (n := n)
        (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
          (DirectParentOnlyProductionSoundness.verifiedStage ctx))
        (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
          (DirectParentOnlyProductionSoundness.verifiedStage ctx))
        i
        prior
        priorInputs
        source) :
    ParentSourceStageAudit ctx i prior priorInputs source := by
  rcases hSource with ⟨out, hPiCCS, hPiRLC⟩
  let childPrior := childCarryingPrior prior priorInputs
  have hOut : out = ctx.stage.computePiCCS i childPrior := by
    simpa
      [DirectParentOnlyProductionSoundness.verifiedStage,
        DirectParentOnlyStageSemantics.verifiedStageOfContextual,
        DirectStageSemantics.ReusedStageComputations.toVerified,
        DirectStageSemantics.ContextualReusedStageComputations.toReused,
        childPrior,
        childCarryingPrior]
      using hPiCCS.1
  subst out
  have hSourceEq :
      source =
        ctx.stage.computePiRLC
          i
          (ctx.stage.computePiCCS i childPrior) := by
    simpa
      [DirectParentOnlyProductionSoundness.verifiedStage,
        DirectParentOnlyStageSemantics.verifiedStageOfContextual,
        DirectStageSemantics.ReusedStageComputations.toVerified,
        DirectStageSemantics.ContextualReusedStageComputations.toReused,
        childPrior,
        childCarryingPrior]
      using hPiRLC.1
  have hPiCCSStrong :
      SuperNeo.PiCCSInterface.piCCSStrongStatement
        (ctx.stage.computePiCCS i childPrior).ctx := by
    simpa
      [DirectParentOnlyProductionSoundness.verifiedStage,
        DirectParentOnlyStageSemantics.verifiedStageOfContextual,
        DirectStageSemantics.ReusedStageComputations.toVerified,
        DirectStageSemantics.ContextualReusedStageComputations.toReused,
        childPrior,
        childCarryingPrior]
      using hPiCCS.2.2
  have hPiCCSDECKnowledge :
      SuperNeo.PiDECInterface.piDECKnowledgeStatement
        (ctx.stage.computePiCCS i childPrior).ctx :=
    SuperNeoBridge.ReusedStageAuthority.piDECKnowledge
      (ctx.stage.piCCSAuthority i childPrior)
  have hPiRLCWeak :
      SuperNeo.PiRLCInterface.piRLCWeakStatement
        (ctx.stage.piRLCContext
          (ctx.stage.computePiCCS i childPrior)
          (ctx.stage.computePiRLC
            i
            (ctx.stage.computePiCCS i childPrior))) := by
    simpa
      [DirectParentOnlyProductionSoundness.verifiedStage,
        DirectParentOnlyStageSemantics.verifiedStageOfContextual,
        DirectStageSemantics.ReusedStageComputations.toVerified,
        DirectStageSemantics.ContextualReusedStageComputations.toReused,
        childPrior,
        childCarryingPrior]
      using hPiRLC.2.2
  have hPiDECKnowledge :
      SuperNeo.PiDECInterface.piDECKnowledgeStatement
        (ctx.stage.piRLCContext
          (ctx.stage.computePiCCS i childPrior)
          (ctx.stage.computePiRLC
            i
            (ctx.stage.computePiCCS i childPrior))) :=
    SuperNeoBridge.ReusedStageAuthority.piDECKnowledge
      (ctx.stage.piRLCAuthority
        i
        (ctx.stage.computePiCCS i childPrior))
  exact
    ⟨hSourceEq,
      ctx.stage.computePiCCS_step i childPrior,
      hPiCCSStrong,
      hPiCCSDECKnowledge,
      hPiRLCWeak,
      hPiDECKnowledge⟩

/--
Project the imported `Pi_DEC` knowledge statement for the exact computed
`Pi_CCS` context from a parent-source stage audit.
-/
theorem piDEC_knowledge_of_piCCS_parent_source_stage_audit
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params}
    {i : Nat}
    {prior : DirectParentOnlyProductionSoundness.AccHandle Digest}
    {priorInputs : DecDigitUniqueness.ColumnDigits n}
    {source : DigestParentBinding.Source Digest}
    (hAudit : ParentSourceStageAudit ctx i prior priorInputs source) :
    let childPrior := childCarryingPrior prior priorInputs
    let out := ctx.stage.computePiCCS i childPrior
    SuperNeo.PiDECInterface.piDECKnowledgeStatement out.ctx :=
  hAudit.2.2.2.1

/--
Project the imported `Pi_DEC` knowledge statement for the exact computed
`Pi_RLC` context whose parent source is reused as the compact public handle.
-/
theorem piDEC_knowledge_of_parent_source_stage_audit
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params}
    {i : Nat}
    {prior : DirectParentOnlyProductionSoundness.AccHandle Digest}
    {priorInputs : DecDigitUniqueness.ColumnDigits n}
    {source : DigestParentBinding.Source Digest}
    (hAudit : ParentSourceStageAudit ctx i prior priorInputs source) :
    let childPrior := childCarryingPrior prior priorInputs
    let out := ctx.stage.computePiCCS i childPrior
    SuperNeo.PiDECInterface.piDECKnowledgeStatement
      (ctx.stage.piRLCContext out (ctx.stage.computePiRLC i out)) :=
  hAudit.2.2.2.2.2

/--
Project the deterministic `Pi_RLC` parent-source computation from a stage
audit.
-/
theorem source_eq_computePiRLC_of_parent_source_stage_audit
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params}
    {i : Nat}
    {prior : DirectParentOnlyProductionSoundness.AccHandle Digest}
    {priorInputs : DecDigitUniqueness.ColumnDigits n}
    {source : DigestParentBinding.Source Digest}
    (hAudit : ParentSourceStageAudit ctx i prior priorInputs source) :
    let childPrior := childCarryingPrior prior priorInputs
    let out := ctx.stage.computePiCCS i childPrior
    source = ctx.stage.computePiRLC i out :=
  hAudit.1

/-- Project the current-step fact from the contextual `Pi_CCS` output. -/
theorem piCCS_step_of_parent_source_stage_audit
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params}
    {i : Nat}
    {prior : DirectParentOnlyProductionSoundness.AccHandle Digest}
    {priorInputs : DecDigitUniqueness.ColumnDigits n}
    {source : DigestParentBinding.Source Digest}
    (hAudit : ParentSourceStageAudit ctx i prior priorInputs source) :
    let childPrior := childCarryingPrior prior priorInputs
    let out := ctx.stage.computePiCCS i childPrior
    out.step = i :=
  hAudit.2.1

/-- Project the imported `Pi_CCS` strong statement from a stage audit. -/
theorem piCCS_strong_of_parent_source_stage_audit
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params}
    {i : Nat}
    {prior : DirectParentOnlyProductionSoundness.AccHandle Digest}
    {priorInputs : DecDigitUniqueness.ColumnDigits n}
    {source : DigestParentBinding.Source Digest}
    (hAudit : ParentSourceStageAudit ctx i prior priorInputs source) :
    let childPrior := childCarryingPrior prior priorInputs
    let out := ctx.stage.computePiCCS i childPrior
    SuperNeo.PiCCSInterface.piCCSStrongStatement out.ctx :=
  hAudit.2.2.1

/--
Project the imported `Pi_RLC` weak statement for the exact context that
computes the parent source.
-/
theorem piRLC_weak_of_parent_source_stage_audit
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params}
    {i : Nat}
    {prior : DirectParentOnlyProductionSoundness.AccHandle Digest}
    {priorInputs : DecDigitUniqueness.ColumnDigits n}
    {source : DigestParentBinding.Source Digest}
    (hAudit : ParentSourceStageAudit ctx i prior priorInputs source) :
    let childPrior := childCarryingPrior prior priorInputs
    let out := ctx.stage.computePiCCS i childPrior
    SuperNeo.PiRLCInterface.piRLCWeakStatement
      (ctx.stage.piRLCContext out (ctx.stage.computePiRLC i out)) :=
  hAudit.2.2.2.2.1

/--
Terminal audit trail with the contextual `Pi_CCS -> Pi_RLC` stage facts exposed.

The same private child table carries the pointwise DEC/CE audit trail and feeds
both the accepted and alternate latest parent-source computations.
-/
def TerminalStageAuditTrail
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
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
      ctx
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
          otherInputs = priorInputs) ∧
    ParentSourceStageAudit
      ctx
      priorSteps
      priorImage.accumulator
      priorInputs
      nextImage.accumulator.parentSource ∧
    ParentSourceStageAudit
      ctx
      priorSteps
      priorImage.accumulator
      priorInputs
      altNext.accumulator.parentSource

/--
Flatten the terminal stage audit into the exact pointwise child table and
computed `Pi_CCS -> Pi_RLC` parent-source facts.

This is the direct audit projection for the optimized parent-only public state:
both terminal images use the same pointwise-authorized private child table, and
both parent sources are the deterministic `Pi_RLC` result over the contextual
`Pi_CCS` output for that child-carrying prior handle.
-/
theorem computed_stage_evidence_of_terminal_stage_audit_trail
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params}
    {priorSteps : Nat}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hTrail :
      TerminalStageAuditTrail
        ctx
        priorSteps
        priorImage
        nextImage
        altNext) :
    ∃ priorInputs,
      let childPrior :=
        childCarryingPrior priorImage.accumulator priorInputs
      let out := ctx.stage.computePiCCS priorSteps childPrior
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
        ctx
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
      nextImage.accumulator.parentSource =
        ctx.stage.computePiRLC priorSteps out ∧
      altNext.accumulator.parentSource =
        ctx.stage.computePiRLC priorSteps out ∧
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
            otherInputs = priorInputs) ∧
      out.step = priorSteps ∧
      SuperNeo.PiCCSInterface.piCCSStrongStatement out.ctx ∧
      SuperNeo.PiDECInterface.piDECKnowledgeStatement out.ctx ∧
      SuperNeo.PiRLCInterface.piRLCWeakStatement
        (ctx.stage.piRLCContext out (ctx.stage.computePiRLC priorSteps out)) ∧
      SuperNeo.PiDECInterface.piDECKnowledgeStatement
        (ctx.stage.piRLCContext out (ctx.stage.computePiRLC priorSteps out)) := by
  rcases hTrail with
    ⟨priorInputs,
      hPointwise,
      hAudit,
      hNext,
      hAlt,
      hUnique,
      hStage,
      hAltStage⟩
  refine ⟨priorInputs, ?_⟩
  exact
    ⟨hPointwise,
      hAudit,
      hNext,
      hAlt,
      hStage.1,
      hAltStage.1,
      hUnique,
      hStage.2.1,
      hStage.2.2.1,
      hStage.2.2.2.1,
      hStage.2.2.2.2.1,
      hStage.2.2.2.2.2⟩

/--
Combine terminal soundness and the child audit trail into the stage audit trail.
-/
theorem terminal_stage_audit_trail_of_terminal_soundness_and_child_audit
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    {priorSteps : Nat}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hSound :
      DirectParentOnlyProductionSoundness.TerminalSoundness
        ctx
        priorSteps
        priorImage
        nextImage
        altNext)
    (hChild :
      DirectParentOnlyProductionChildMembership.TerminalChildAuditTrail
        ctx
        priorSteps
        priorImage
        nextImage
        altNext) :
    TerminalStageAuditTrail
      ctx
      priorSteps
      priorImage
      nextImage
      altNext := by
  rcases hSound with
    ⟨_hReach,
      _hImageEq,
      _hParentSource,
      hShared,
      _hNextBoundary,
      _hAltBoundary,
      _hStep,
      _hVk,
      _hInitialBoundary,
      _hWellFormed⟩
  rcases hShared with
    ⟨stageInputs, hStagePointwise, hNextSource, hAltSource⟩
  rcases hChild with
    ⟨priorInputs,
      hPointwise,
      hAudit,
      hNextComputed,
      hAltComputed,
      hUnique⟩
  have hInputs : stageInputs = priorInputs :=
    hUnique stageInputs hStagePointwise
  subst stageInputs
  exact
    ⟨priorInputs,
      hPointwise,
      hAudit,
      hNextComputed,
      hAltComputed,
      hUnique,
      parent_source_stage_audit_of_parentSourceFromPiStages
        ctx
        hNextSource,
      parent_source_stage_audit_of_parentSourceFromPiStages
        ctx
        hAltSource⟩

/--
Production endpoint with flattened public-image facts and the contextual stage
audit trail.
-/
def AuditedPublicEndpointWithStageAudit
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (priorSteps : Nat)
    (priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary) : Prop :=
  DirectParentOnlyProductionEndpoint.AuditedPublicEndpoint
      ctx
      priorSteps
      priorImage
      nextImage
      altNext ∧
    TerminalStageAuditTrail
      ctx
      priorSteps
      priorImage
      nextImage
      altNext

/--
Sound-verifier endpoint with public-image facts, pointwise child audit, and
contextual stage audit.
-/
theorem audited_public_endpoint_with_stage_audit_of_sound_verifier
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
    AuditedPublicEndpointWithStageAudit
      ctx
      priorSteps
      priorImage
      nextImage
      altNext := by
  have hChild :=
    DirectParentOnlyProductionChildMembership.terminal_soundness_with_pointwise_child_audit_trail
      ctx
      verifier
      hAccepted
      hAlt
  exact
    ⟨DirectParentOnlyProductionEndpoint.audited_public_endpoint_of_sound_verifier
        ctx
        verifier
        hAccepted
        hAlt,
      terminal_stage_audit_trail_of_terminal_soundness_and_child_audit
        ctx
        hChild.1
        hChild.2⟩

/--
Raw compressed-prior endpoint with public-image facts, pointwise child audit,
and contextual stage audit.
-/
theorem audited_public_endpoint_with_stage_audit_of_prior_verifier_opening
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
    AuditedPublicEndpointWithStageAudit
      ctx
      priorSteps
      priorImage
      nextImage
      altNext := by
  have hChild :=
    DirectParentOnlyProductionChildMembership.terminal_soundness_with_pointwise_child_audit_trail_of_prior_verifier_opening
      ctx
      hOpens
      hAccepted
      hAlt
  exact
    ⟨DirectParentOnlyProductionEndpoint.audited_public_endpoint_of_prior_verifier_opening
        ctx
        hOpens
        hAccepted
        hAlt,
      terminal_stage_audit_trail_of_terminal_soundness_and_child_audit
        ctx
        hChild.1
        hChild.2⟩

end DirectParentOnlyProductionStageAudit

end DirectCcsFPrime
