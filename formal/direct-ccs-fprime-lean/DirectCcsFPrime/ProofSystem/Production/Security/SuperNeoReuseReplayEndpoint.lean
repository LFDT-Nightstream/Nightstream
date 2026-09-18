import DirectCcsFPrime.ProofSystem.Production.Impl.SuperNeoReuse.PriorOpening

/-!
Replay endpoint for the Section 7.1-backed concrete compressed verifier.

This module packages the adversarial replay case directly: the same opaque
prior proof is consumed by two terminal acceptances under the same concrete
verifier. If the verifier has the required opening certificate, both
acceptances are forced to the same prior pair and terminal image, while the
first terminal image carries the full pointwise child and stage audit.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionSuperNeoReuseReplayEndpoint

/-- Section 7.1-backed production context. -/
abbrev ProductionContext :=
  DirectParentOnlyProductionSuperNeoReusePriorOpening.ProductionContext

/-- Opaque prior-proof opener for the induced context. -/
abbrev PriorAuthorityOpener
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) :=
  DirectParentOnlyProductionSuperNeoReusePriorOpening.PriorAuthorityOpener
    (PriorProof := PriorProof)
    ctx

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
  DirectParentOnlyProductionSuperNeoReusePriorOpening.PriorVerifierAuthorityOpening
    ctx
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
  DirectParentOnlyProductionSuperNeoReusePriorOpening.AcceptedTerminalWithAuthorityOpener
    ctx
    opener
    priorSteps
    priorProof
    priorImage
    nextImage
    latestProof

/-- Terminal acceptance through a concrete compressed verifier. -/
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
  DirectParentOnlyProductionSuperNeoReusePriorOpening.AcceptedTerminalWithPriorVerifier
    ctx
    VerifyPrior
    priorSteps
    priorProof
    priorImage
    nextImage
    latestProof

/--
Replay-stable endpoint for one concrete compressed prior proof.

The same proof fixes the prior step/image pair and the terminal public image.
The endpoint also exposes final reachability, public-image invariants,
deterministic boundary update, pointwise private-child audit, and contextual
`Pi_CCS -> Pi_RLC` stage audit for the first terminal image, with the second
terminal image serving as the alternate latest image.
-/
def SameProofReplayEndpoint
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (priorStepsA priorStepsB : Nat)
    (priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary) : Prop :=
  priorStepsA = priorStepsB ∧
    priorImageA = priorImageB ∧
    nextImageA = nextImageB ∧
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.AuditedPublicEndpointWithStageAudit
      ctx
      priorStepsA
      priorImageA
      nextImageA
      nextImageB

/--
Pointwise private-child replay binding exposed by the replay endpoint.

This is the non-aggregate anti-swap conclusion: the two replayed terminal
images are computed from one pointwise-authorized private DEC child table, and
any other pointwise-valid table for the same parent source is equal to it.
-/
def PointwiseChildReplayBinding
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (priorSteps : Nat)
    (priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary) : Prop :=
  DirectParentOnlyProductionStageAudit.TerminalStageAuditTrail
    ctx.toProductionContext
    priorSteps
    priorImage
    nextImage
    altNext

/--
Computed-stage replay evidence for the unique pointwise private child table.

This flattens the terminal stage audit into the exact values callers need to
check the optimized parent-only public handle: the same private child table is
pointwise-valid, both terminal parent sources are the deterministic
`Pi_RLC(Pi_CCS(child table))` result, and the imported SuperNeo stage
statements hold for the exact computed contexts.
-/
def ComputedStageReplayEvidence
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (priorSteps : Nat)
    (priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary) : Prop :=
  ∃ priorInputs,
    let childPrior :=
      DirectParentOnlyProductionStageAudit.childCarryingPrior
        priorImage.accumulator
        priorInputs
    let out :=
      ctx.toProductionContext.stage.computePiCCS priorSteps childPrior
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
    nextImage.accumulator.parentSource =
      ctx.toProductionContext.stage.computePiRLC priorSteps out ∧
    altNext.accumulator.parentSource =
      ctx.toProductionContext.stage.computePiRLC priorSteps out ∧
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
      (ctx.toProductionContext.stage.piRLCContext
        out
        (ctx.toProductionContext.stage.computePiRLC priorSteps out)) ∧
    SuperNeo.PiDECInterface.piDECKnowledgeStatement
      (ctx.toProductionContext.stage.piRLCContext
        out
        (ctx.toProductionContext.stage.computePiRLC priorSteps out))

/--
Extract the pointwise private-child replay binding from the endpoint.

This is the theorem callers should use when auditing that the optimized path is
not relying on an aggregate child summary: it exposes the unique child table and
the contextual `Pi_CCS -> Pi_RLC` stage audit for that table.
-/
theorem pointwise_child_replay_binding_of_same_proof_replay_endpoint
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {priorStepsA priorStepsB : Nat}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hReplay :
      SameProofReplayEndpoint
        ctx
        priorStepsA
        priorStepsB
        priorImageA
        priorImageB
        nextImageA
        nextImageB) :
    PointwiseChildReplayBinding
      ctx
      priorStepsA
      priorImageA
      nextImageA
      nextImageB := by
  exact hReplay.2.2.2.2

/--
Extract flattened computed-stage evidence from the pointwise child replay
binding.
-/
theorem computed_stage_replay_evidence_of_pointwise_child_replay_binding
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {priorSteps : Nat}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hBinding :
      PointwiseChildReplayBinding
        ctx
        priorSteps
        priorImage
        nextImage
        altNext) :
    ComputedStageReplayEvidence
      ctx
      priorSteps
      priorImage
      nextImage
      altNext := by
  simpa [ComputedStageReplayEvidence, PointwiseChildReplayBinding]
    using
      DirectParentOnlyProductionStageAudit.computed_stage_evidence_of_terminal_stage_audit_trail
        hBinding

/--
Extract flattened computed-stage evidence from a same-proof replay endpoint.
-/
theorem computed_stage_replay_evidence_of_same_proof_replay_endpoint
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {priorStepsA priorStepsB : Nat}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hReplay :
      SameProofReplayEndpoint
        ctx
        priorStepsA
        priorStepsB
        priorImageA
        priorImageB
        nextImageA
        nextImageB) :
    ComputedStageReplayEvidence
      ctx
      priorStepsA
      priorImageA
      nextImageA
      nextImageB :=
  computed_stage_replay_evidence_of_pointwise_child_replay_binding
    (pointwise_child_replay_binding_of_same_proof_replay_endpoint hReplay)

/--
Computed-stage evidence exposes the exact `Pi_CCS` context that consumes the
pointwise private DEC child table.
-/
theorem piCCS_dec_knowledge_of_computed_stage_replay_evidence
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {priorSteps : Nat}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hEvidence :
      ComputedStageReplayEvidence
        ctx
        priorSteps
        priorImage
        nextImage
        altNext) :
    ∃ priorInputs,
      let childPrior :=
        DirectParentOnlyProductionStageAudit.childCarryingPrior
          priorImage.accumulator
          priorInputs
      let out :=
        ctx.toProductionContext.stage.computePiCCS priorSteps childPrior
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
      out.step = priorSteps ∧
      SuperNeo.PiCCSInterface.piCCSStrongStatement out.ctx ∧
      SuperNeo.PiDECInterface.piDECKnowledgeStatement out.ctx := by
  rcases hEvidence with
    ⟨priorInputs,
      hPointwise,
      hChildAudit,
      _hNext,
      _hAlt,
      _hNextSource,
      _hAltSource,
      _hUnique,
      hStep,
      hPiCCSStrong,
      hPiCCSDec,
      _hPiRLCWeak,
      _hPiRLCDec⟩
  exact
    ⟨priorInputs,
      hPointwise,
      hChildAudit,
      hStep,
      hPiCCSStrong,
      hPiCCSDec⟩

/--
Computed-stage evidence exposes the exact `Pi_RLC` context that produces the
compact parent source from the computed `Pi_CCS` output.
-/
theorem piRLC_dec_knowledge_of_computed_stage_replay_evidence
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {priorSteps : Nat}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hEvidence :
      ComputedStageReplayEvidence
        ctx
        priorSteps
        priorImage
        nextImage
        altNext) :
    ∃ priorInputs,
      let childPrior :=
        DirectParentOnlyProductionStageAudit.childCarryingPrior
          priorImage.accumulator
          priorInputs
      let out :=
        ctx.toProductionContext.stage.computePiCCS priorSteps childPrior
      nextImage.accumulator.parentSource =
        ctx.toProductionContext.stage.computePiRLC priorSteps out ∧
      altNext.accumulator.parentSource =
        ctx.toProductionContext.stage.computePiRLC priorSteps out ∧
      SuperNeo.PiRLCInterface.piRLCWeakStatement
        (ctx.toProductionContext.stage.piRLCContext
          out
          (ctx.toProductionContext.stage.computePiRLC priorSteps out)) ∧
      SuperNeo.PiDECInterface.piDECKnowledgeStatement
        (ctx.toProductionContext.stage.piRLCContext
          out
          (ctx.toProductionContext.stage.computePiRLC priorSteps out)) := by
  rcases hEvidence with
    ⟨priorInputs,
      _hPointwise,
      _hChildAudit,
      _hNext,
      _hAlt,
      hNextSource,
      hAltSource,
      _hUnique,
      _hStep,
      _hPiCCSStrong,
      _hPiCCSDec,
      hPiRLCWeak,
      hPiRLCDec⟩
  exact
    ⟨priorInputs,
      hNextSource,
      hAltSource,
      hPiRLCWeak,
      hPiRLCDec⟩

/--
Any pointwise-valid child table for the replayed parent source is the audited
table used by both terminal images.

This is the direct no-swap projection: an adversary may propose another private
child table, but if it satisfies the full pointwise private-DEC requirements for
the same parent source, it is equal to the table already feeding the audited
`Pi_CCS -> Pi_RLC` stage computations.
-/
theorem pointwise_child_table_unique_of_pointwise_child_replay_binding
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {priorSteps : Nat}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hBinding :
      PointwiseChildReplayBinding
        ctx
        priorSteps
        priorImage
        nextImage
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
  rcases hBinding with
    ⟨priorInputs,
      hPointwise,
      hAudit,
      hNext,
      hAlt,
      hUnique,
      hStage,
      hAltStage⟩
  refine
    ⟨priorInputs,
      ?_,
      hAudit,
      hNext,
      hAlt,
      ?_,
      hStage,
      hAltStage⟩
  · simpa [DirectParentOnlyProductionSuperNeoReuse.ProductionContext.toProductionContext]
      using hPointwise
  · exact
      hUnique
        otherInputs
        (by
          simpa [DirectParentOnlyProductionSuperNeoReuse.ProductionContext.toProductionContext]
            using hOther)

/--
Opener-induced replay endpoint for one opaque prior proof.

This is the direct anti-retargeting theorem for the authority-opening path:
same opener and same prior proof imply the same prior pair and terminal image,
while retaining the full non-aggregate child and stage audit.
-/
theorem same_proof_replay_endpoint_of_authority_opener
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
    SameProofReplayEndpoint
      ctx
      priorStepsA
      priorStepsB
      priorImageA
      priorImageB
      nextImageA
      nextImageB := by
  rcases
      DirectParentOnlyProductionSuperNeoReusePriorOpening.terminal_prior_pair_functional_for_same_proof_of_authority_opener
        ctx
        opener
        hA
        hB with
    ⟨hSteps, hPriorImage⟩
  subst priorStepsB
  subst priorImageB
  have hNext : nextImageA = nextImageB :=
    DirectParentOnlyProductionSuperNeoReusePriorOpening.terminal_next_image_functional_for_same_proof_of_authority_opener
      ctx
      opener
      hA
      hB
  have hEndpoint :
      DirectParentOnlyProductionSuperNeoReuse.ProductionContext.AuditedPublicEndpointWithStageAudit
        ctx
        priorStepsA
        priorImageA
        nextImageA
        nextImageB :=
    DirectParentOnlyProductionSuperNeoReusePriorOpening.audited_public_endpoint_with_stage_audit_of_authority_opener
      ctx
      opener
      hA
      hB.latestAccepted
  exact ⟨rfl, rfl, hNext, hEndpoint⟩

/--
Opener-induced replay endpoint with the pointwise child binding projected.

The same opener and same opaque proof force both accepted terminal images to
reuse one unique pointwise-authorized private child table.
-/
theorem pointwise_child_replay_binding_of_authority_opener
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
    PointwiseChildReplayBinding
      ctx
      priorStepsA
      priorImageA
      nextImageA
      nextImageB :=
  pointwise_child_replay_binding_of_same_proof_replay_endpoint
      (same_proof_replay_endpoint_of_authority_opener
        ctx
        opener
        hA
        hB)

/--
Opener-induced no-swap theorem for replayed private DEC children.

Any alternate pointwise-valid child table for the replayed parent source is the
same table used by both accepted terminal images.
-/
theorem pointwise_child_table_unique_of_authority_opener
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
        latestProofB)
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
        priorImageA.accumulator.parentSource
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
        priorImageA.accumulator.parentSource
        priorInputs ∧
      DirectParentOnlyProductionChildMembership.PointwiseChildAuditTrail
        ctx.toProductionContext
        priorImageA.accumulator.parentSource
        priorInputs ∧
      nextImageA =
        DirectParentOnlyProductionSoundness.ComputedNextImage
          ctx.toProductionContext
          priorStepsA
          priorImageA
          priorInputs ∧
      nextImageB =
        DirectParentOnlyProductionSoundness.ComputedNextImage
          ctx.toProductionContext
          priorStepsA
          priorImageA
          priorInputs ∧
      otherInputs = priorInputs ∧
      DirectParentOnlyProductionStageAudit.ParentSourceStageAudit
        ctx.toProductionContext
        priorStepsA
        priorImageA.accumulator
        priorInputs
        nextImageA.accumulator.parentSource ∧
      DirectParentOnlyProductionStageAudit.ParentSourceStageAudit
        ctx.toProductionContext
        priorStepsA
        priorImageA.accumulator
        priorInputs
        nextImageB.accumulator.parentSource :=
  pointwise_child_table_unique_of_pointwise_child_replay_binding
    (pointwise_child_replay_binding_of_authority_opener
      ctx
      opener
      hA
      hB)
    hOther

/--
Opener-induced replay endpoint with prior authority and computed stage evidence
packaged together.

This is the single-call audit theorem for the opener path: the prior proof
opens to folded authority for the first accepted prior pair, the same proof
cannot retarget the prior pair or terminal image, and the hidden child table is
the unique pointwise-valid table feeding the computed `Pi_CCS -> Pi_RLC`
stages.
-/
theorem authority_opener_replay_audit_package
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
            priorStepsA
            authority
            priorImageA) ∧
      SameProofReplayEndpoint
        ctx
        priorStepsA
        priorStepsB
        priorImageA
        priorImageB
        nextImageA
        nextImageB ∧
      ComputedStageReplayEvidence
        ctx
        priorStepsA
        priorImageA
        nextImageA
        nextImageB := by
  have hVerify :
      DirectParentOnlyProductionSuperNeoReusePriorOpening.VerifyWithAuthorityOpener
        ctx
        opener
        priorStepsA
        priorProof
        priorImageA :=
    hA.priorAccepted
  have hReplay :=
    same_proof_replay_endpoint_of_authority_opener
      ctx
      opener
      hA
      hB
  exact
    ⟨DirectParentOnlyProductionSuperNeoReusePriorOpening.verifyWithAuthorityOpener_openAuthority_ne_none
        ctx
        opener
        hVerify,
      hVerify,
      hReplay,
      computed_stage_replay_evidence_of_same_proof_replay_endpoint hReplay⟩

/--
Concrete-verifier replay endpoint for one opaque prior proof.

This is the direct anti-retargeting theorem for the optimized parent-only path:
same verifier, same opening certificate, same prior proof implies same prior
pair and same terminal image, while retaining the full non-aggregate child and
stage audit.
-/
theorem same_proof_replay_endpoint_of_priorVerifierAuthorityOpening
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
    SameProofReplayEndpoint
      ctx
      priorStepsA
      priorStepsB
      priorImageA
      priorImageB
      nextImageA
      nextImageB := by
  rcases
      DirectParentOnlyProductionSuperNeoReusePriorOpening.terminal_prior_pair_functional_for_same_proof_of_priorVerifierAuthorityOpening
        ctx
        opening
        hA
        hB with
    ⟨hSteps, hPriorImage⟩
  subst priorStepsB
  subst priorImageB
  have hNext : nextImageA = nextImageB :=
    DirectParentOnlyProductionSuperNeoReusePriorOpening.terminal_next_image_functional_for_same_proof_of_priorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB
  have hEndpoint :
      DirectParentOnlyProductionSuperNeoReuse.ProductionContext.AuditedPublicEndpointWithStageAudit
        ctx
        priorStepsA
        priorImageA
        nextImageA
        nextImageB :=
    DirectParentOnlyProductionSuperNeoReusePriorOpening.audited_public_endpoint_with_stage_audit_of_priorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB.latestAccepted
  exact ⟨rfl, rfl, hNext, hEndpoint⟩

/--
Concrete-verifier replay endpoint with the pointwise child binding projected.

The same concrete verifier, opening certificate, and opaque proof force both
accepted terminal images to reuse one unique pointwise-authorized private child
table.
-/
theorem pointwise_child_replay_binding_of_priorVerifierAuthorityOpening
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
    PointwiseChildReplayBinding
      ctx
      priorStepsA
      priorImageA
      nextImageA
      nextImageB :=
    pointwise_child_replay_binding_of_same_proof_replay_endpoint
      (same_proof_replay_endpoint_of_priorVerifierAuthorityOpening
        ctx
        opening
        hA
        hB)

/--
Concrete-verifier no-swap theorem for replayed private DEC children.

Any alternate pointwise-valid child table for the replayed parent source is the
same table used by both accepted terminal images under the opened compressed
prior verifier.
-/
theorem pointwise_child_table_unique_of_priorVerifierAuthorityOpening
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
        latestProofB)
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
        priorImageA.accumulator.parentSource
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
        priorImageA.accumulator.parentSource
        priorInputs ∧
      DirectParentOnlyProductionChildMembership.PointwiseChildAuditTrail
        ctx.toProductionContext
        priorImageA.accumulator.parentSource
        priorInputs ∧
      nextImageA =
        DirectParentOnlyProductionSoundness.ComputedNextImage
          ctx.toProductionContext
          priorStepsA
          priorImageA
          priorInputs ∧
      nextImageB =
        DirectParentOnlyProductionSoundness.ComputedNextImage
          ctx.toProductionContext
          priorStepsA
          priorImageA
          priorInputs ∧
      otherInputs = priorInputs ∧
      DirectParentOnlyProductionStageAudit.ParentSourceStageAudit
        ctx.toProductionContext
        priorStepsA
        priorImageA.accumulator
        priorInputs
        nextImageA.accumulator.parentSource ∧
      DirectParentOnlyProductionStageAudit.ParentSourceStageAudit
        ctx.toProductionContext
        priorStepsA
        priorImageA.accumulator
        priorInputs
        nextImageB.accumulator.parentSource :=
  pointwise_child_table_unique_of_pointwise_child_replay_binding
    (pointwise_child_replay_binding_of_priorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB)
    hOther

/--
Concrete-verifier replay endpoint with prior authority and computed stage
evidence packaged together.

This is the single-call audit theorem for the implementation-facing path:
accepted compressed prior verification opens to folded authority for the first
accepted prior pair, the same opaque proof cannot retarget the prior pair or
terminal image, and the hidden child table is the unique pointwise-valid table
feeding the computed `Pi_CCS -> Pi_RLC` stages.
-/
theorem concrete_verifier_replay_audit_package_of_priorVerifierAuthorityOpening
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
    opening.opener.openAuthority priorProof ≠ none ∧
      (∃ authority :
          DirectParentOnlyProductionSoundness.ProofCarryingPriorProof
            ctx.toProductionContext,
        opening.opener.openAuthority priorProof = some authority ∧
          FoldedFPrimeAuthority.Accepts
            (Transition :=
              DirectParentOnlyProductionSoundness.Transition
                ctx.toProductionContext)
            (initial := ctx.initial)
            priorStepsA
            authority
            priorImageA) ∧
      SameProofReplayEndpoint
        ctx
        priorStepsA
        priorStepsB
        priorImageA
        priorImageB
        nextImageA
        nextImageB ∧
      ComputedStageReplayEvidence
        ctx
        priorStepsA
        priorImageA
        nextImageA
        nextImageB := by
  have hVerify : VerifyPrior priorStepsA priorProof priorImageA :=
    hA.priorAccepted
  have hReplay :=
    same_proof_replay_endpoint_of_priorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB
  exact
    ⟨DirectParentOnlyProductionSuperNeoReusePriorOpening.priorVerifierAuthorityOpening_openAuthority_ne_none
        ctx
        opening
        hVerify,
      DirectParentOnlyProductionSuperNeoReusePriorOpening.priorVerifierAuthorityOpening_opened_authority
        ctx
        opening
        hVerify,
      hReplay,
      computed_stage_replay_evidence_of_same_proof_replay_endpoint hReplay⟩

end DirectParentOnlyProductionSuperNeoReuseReplayEndpoint

end DirectCcsFPrime
