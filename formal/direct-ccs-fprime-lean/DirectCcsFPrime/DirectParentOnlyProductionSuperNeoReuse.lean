import DirectCcsFPrime.DirectParentOnlyProductionStageAudit
import DirectCcsFPrime.DirectStageSuperNeoReuse

/-!
Production context constructor with Section 7.1-backed SuperNeo stages.

This module keeps the parent-only production theorem surface intact, but makes
the intended implementation instantiation sharper: stage authority comes from
the upstream SuperNeo Section 7.1 context adapter.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionSuperNeoReuse

/--
Production context inputs when direct stages are backed by theorem-native
SuperNeo Section 7.1 contexts.

The fields match the existing parent-only production context, except the stage
field is the stronger `Section71ContextualStageComputations` package. Poseidon2
and MSIS remain external theorem-facing assumptions, as in the production
context.
-/
structure ProductionContext
    (Digest Boundary : Type)
    (n : Nat)
    (params : SuperNeo.ProofSystem.AjtaiParams) where
  parentHash : Poseidon2ParentCEBHash.Hash Digest
  data : DirectConcreteInstantiation.ConcreteCEData n params
  stage : DirectStageSuperNeoReuse.Section71ContextualStageComputations Digest n
  computeBoundary : Nat → Boundary → Boundary
  commitmentOfParent :
    ParentEncoding.SomeParentCEB →
      SuperNeo.ProofSystem.Commitment
  initial : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary
  initialStep : initial.step = 0
  initialWellFormed : Construction2DirectFPrime.WellFormed initial
  msisReduction : SuperNeo.ProofSystem.MSISToAjtaiReductions params
  msisHardness : SuperNeo.ProofSystem.MSISHardnessAssumption params

namespace ProductionContext

/-- Convert Section 7.1-backed inputs into the existing production context. -/
def toProductionContext
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) :
    DirectParentOnlyProductionSoundness.Context Digest Boundary n params where
  parentHash := ctx.parentHash
  data := ctx.data
  stage := ctx.stage.toContextualReused
  computeBoundary := ctx.computeBoundary
  commitmentOfParent := ctx.commitmentOfParent
  initial := ctx.initial
  initialStep := ctx.initialStep
  initialWellFormed := ctx.initialWellFormed
  msisReduction := ctx.msisReduction
  msisHardness := ctx.msisHardness

/-- Sound compressed prior verifier for the induced production context. -/
abbrev SoundPriorVerifier
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) :=
  DirectParentOnlyProductionSoundness.SoundPriorVerifier
    (PriorProof := PriorProof)
    (ctx.toProductionContext)

/-- Terminal acceptance for the induced production context. -/
abbrev AcceptedTerminal
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (verifier : SoundPriorVerifier (PriorProof := PriorProof) ctx)
    (priorSteps : Nat)
    (priorProof : PriorProof)
    (priorImage nextImage :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary)
    (latestProof : Unit) : Prop :=
  DirectParentOnlyProductionSoundness.AcceptedTerminal
    ctx.toProductionContext
    verifier
    priorSteps
    priorProof
    priorImage
    nextImage
    latestProof

/-- Alternate latest step for the induced production context. -/
abbrev AlternateLatestStep
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (priorSteps : Nat)
    (priorImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary) :
    Prop :=
  DirectParentOnlyProductionSoundness.AlternateLatestStep
    ctx.toProductionContext
    priorSteps
    priorImage
    altNext

/-- Stage-audited public endpoint for the induced production context. -/
abbrev AuditedPublicEndpointWithStageAudit
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (priorSteps : Nat)
    (priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary) :
    Prop :=
  DirectParentOnlyProductionStageAudit.AuditedPublicEndpointWithStageAudit
    ctx.toProductionContext
    priorSteps
    priorImage
    nextImage
    altNext

/--
Section 7.1 owner-target audit for one deterministic parent-source stage.

This is the production wrapper's exact SuperNeo-reuse witness: the child table
feeds the contextual `Pi_CCS` computation, the compact parent source is the
deterministic `Pi_RLC` result over that output, and both computed contexts are
the targets of the Section 7.1 owner objects carried by the production stage.
-/
def Section71StageTargetAudit
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (i : Nat)
    (prior : DirectParentOnlyProductionSoundness.AccHandle Digest)
    (priorInputs : DecDigitUniqueness.ColumnDigits n)
    (source : DigestParentBinding.Source Digest) : Prop :=
  let childPrior :=
    DirectParentOnlyProductionStageAudit.childCarryingPrior prior priorInputs
  let out := ctx.stage.computePiCCS i childPrior
  source = ctx.stage.computePiRLC i out ∧
    out.step = i ∧
    (ctx.stage.piCCSSection71 i childPrior).target = out.ctx ∧
    SuperNeo.ceRelation (ctx.stage.piCCSSection71 i childPrior).target ∧
    (ctx.stage.piRLCSection71 i out).target =
      ctx.stage.piRLCContext out (ctx.stage.computePiRLC i out) ∧
    SuperNeo.ceRelation (ctx.stage.piRLCSection71 i out).target ∧
    SuperNeo.PiCCSInterface.piCCSStrongStatement out.ctx ∧
    SuperNeo.PiDECInterface.piDECKnowledgeStatement out.ctx ∧
    SuperNeo.PiRLCInterface.piRLCWeakStatement
      (ctx.stage.piRLCContext out (ctx.stage.computePiRLC i out)) ∧
    SuperNeo.PiDECInterface.piDECKnowledgeStatement
      (ctx.stage.piRLCContext out (ctx.stage.computePiRLC i out))

/--
Terminal Section 7.1 owner-target audit for the shared private child table.

Both terminal parent sources are shown to arise from the same pointwise-valid
private DEC child table and from the exact Section 7.1 target contexts carried
by the production stage package.
-/
def TerminalSection71StageTargetAuditTrail
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (priorSteps : Nat)
    (priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary) :
    Prop :=
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
          otherInputs = priorInputs) ∧
    Section71StageTargetAudit
      ctx
      priorSteps
      priorImage.accumulator
      priorInputs
      nextImage.accumulator.parentSource ∧
    Section71StageTargetAudit
      ctx
      priorSteps
      priorImage.accumulator
      priorInputs
      altNext.accumulator.parentSource

/-- The production context's computed `Pi_CCS` output satisfies the imported strong statement. -/
theorem piCCSStrong_of_compute
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (i : Nat)
    (prior : DirectTerminalSoundness.AccHandle Digest n) :
    SuperNeo.PiCCSInterface.piCCSStrongStatement
      (ctx.stage.computePiCCS i prior).ctx :=
  DirectStageSuperNeoReuse.Section71ContextualStageComputations.piCCSStrong_of_compute
    ctx.stage
    i
    prior

/-- The production context's computed `Pi_RLC` context satisfies the imported weak statement. -/
theorem piRLCWeak_of_compute
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (i : Nat)
    (out : DirectStageSemantics.ContextualPiCCSOut) :
    SuperNeo.PiRLCInterface.piRLCWeakStatement
      (ctx.stage.piRLCContext out (ctx.stage.computePiRLC i out)) :=
  DirectStageSuperNeoReuse.Section71ContextualStageComputations.piRLCWeak_of_compute
    ctx.stage
    i
    out

/-- The production context's computed `Pi_CCS` output exposes the imported DEC knowledge surface. -/
theorem piDECKnowledge_of_piCCS_compute
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (i : Nat)
    (prior : DirectTerminalSoundness.AccHandle Digest n) :
    SuperNeo.PiDECInterface.piDECKnowledgeStatement
      (ctx.stage.computePiCCS i prior).ctx :=
  DirectStageSuperNeoReuse.Section71ContextualStageComputations.piDECKnowledge_of_piCCS_compute
    ctx.stage
    i
    prior

/-- The production context's computed `Pi_RLC` context exposes the imported DEC knowledge surface. -/
theorem piDECKnowledge_of_piRLC_compute
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (i : Nat)
    (out : DirectStageSemantics.ContextualPiCCSOut) :
    SuperNeo.PiDECInterface.piDECKnowledgeStatement
      (ctx.stage.piRLCContext out (ctx.stage.computePiRLC i out)) :=
  DirectStageSuperNeoReuse.Section71ContextualStageComputations.piDECKnowledge_of_piRLC_compute
    ctx.stage
    i
    out

/--
Lift a contextual parent-source stage audit to the exact Section 7.1
owner-target audit carried by the production wrapper.
-/
theorem section71_stage_target_audit_of_parent_source_stage_audit
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {i : Nat}
    {prior : DirectParentOnlyProductionSoundness.AccHandle Digest}
    {priorInputs : DecDigitUniqueness.ColumnDigits n}
    {source : DigestParentBinding.Source Digest}
    (hAudit :
      DirectParentOnlyProductionStageAudit.ParentSourceStageAudit
        ctx.toProductionContext
        i
        prior
        priorInputs
        source) :
    Section71StageTargetAudit
      ctx
      i
      prior
      priorInputs
      source := by
  let childPrior :=
    DirectParentOnlyProductionStageAudit.childCarryingPrior prior priorInputs
  let out := ctx.stage.computePiCCS i childPrior
  have hSource :
      source = ctx.stage.computePiRLC i out := by
    simpa
      [Section71StageTargetAudit,
        DirectParentOnlyProductionStageAudit.ParentSourceStageAudit,
        toProductionContext,
        DirectStageSuperNeoReuse.Section71ContextualStageComputations.toContextualReused,
        childPrior,
        out]
      using hAudit.1
  exact
    ⟨hSource,
      by
        simpa [out, childPrior] using ctx.stage.computePiCCS_step i childPrior,
      by
        simpa [out, childPrior] using ctx.stage.piCCSSection71_target i childPrior,
      SuperNeo.ProtocolSection71Context.ceRelation
        (ctx.stage.piCCSSection71 i childPrior),
      by
        simpa [out, childPrior] using ctx.stage.piRLCSection71_target i out,
      SuperNeo.ProtocolSection71Context.ceRelation
        (ctx.stage.piRLCSection71 i out),
      by
        simpa [out, childPrior] using piCCSStrong_of_compute ctx i childPrior,
      by
        simpa [out, childPrior] using
          piDECKnowledge_of_piCCS_compute ctx i childPrior,
      by
        simpa [out, childPrior] using piRLCWeak_of_compute ctx i out,
      by
        simpa [out, childPrior] using
          piDECKnowledge_of_piRLC_compute ctx i out⟩

/--
Project the terminal Section 7.1 owner-target audit from the stage-audited
endpoint.
-/
theorem terminal_section71_stage_target_audit_trail_of_audited_public_endpoint
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {priorSteps : Nat}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hEndpoint :
      AuditedPublicEndpointWithStageAudit
        ctx
        priorSteps
        priorImage
        nextImage
        altNext) :
    TerminalSection71StageTargetAuditTrail
      ctx
      priorSteps
      priorImage
      nextImage
      altNext := by
  rcases hEndpoint.2 with
    ⟨priorInputs,
      hPointwise,
      _hChildAudit,
      hNext,
      hAlt,
      hUnique,
      hNextStage,
      hAltStage⟩
  exact
    ⟨priorInputs,
      hPointwise,
      hNext,
      hAlt,
      hUnique,
      section71_stage_target_audit_of_parent_source_stage_audit
        ctx
        hNextStage,
      section71_stage_target_audit_of_parent_source_stage_audit
        ctx
        hAltStage⟩

/--
Section 7.1-backed production endpoint with public-image facts, pointwise child
audit, and contextual stage audit.
-/
theorem audited_public_endpoint_with_stage_audit_of_sound_verifier
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (verifier : SoundPriorVerifier (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hAccepted :
      AcceptedTerminal
        ctx
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
    AuditedPublicEndpointWithStageAudit
      ctx
      priorSteps
      priorImage
      nextImage
      altNext :=
  DirectParentOnlyProductionStageAudit.audited_public_endpoint_with_stage_audit_of_sound_verifier
    ctx.toProductionContext
    verifier
    hAccepted
    hAlt

/--
Sound-verifier terminal endpoint with the Section 7.1 owner-target audit
projected directly.
-/
theorem terminal_section71_stage_target_audit_trail_of_sound_verifier
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (verifier : SoundPriorVerifier (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hAccepted :
      AcceptedTerminal
        ctx
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
    TerminalSection71StageTargetAuditTrail
      ctx
      priorSteps
      priorImage
      nextImage
      altNext :=
  terminal_section71_stage_target_audit_trail_of_audited_public_endpoint
    ctx
    (audited_public_endpoint_with_stage_audit_of_sound_verifier
      ctx
      verifier
      hAccepted
      hAlt)

end ProductionContext

end DirectParentOnlyProductionSuperNeoReuse

end DirectCcsFPrime
