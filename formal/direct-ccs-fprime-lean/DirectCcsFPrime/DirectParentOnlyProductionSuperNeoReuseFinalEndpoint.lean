import DirectCcsFPrime.DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier

/-!
Final replay endpoint for the Section 7.1-backed parent-only production path.

This module is the implementation-facing wrapper around the certified raw
compressed-verifier replay package. It keeps the authority boundary explicit:
the concrete verifier contributes authority only through an opener theorem that
turns every accepted proof into folded `F'` reachability for the same public
image.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionSuperNeoReuseFinalEndpoint

/-- Section 7.1-backed production context. -/
abbrev ProductionContext :=
  DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.ProductionContext

/-- Explicit pointwise no-swap evidence from the certified replay package. -/
abbrev ExplicitReplayNoSwapEvidence :=
  @DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.ExplicitReplayNoSwapEvidence

/-- Flattened public endpoint exposed by the replay-stable terminal package. -/
abbrev AuditedPublicEndpoint :=
  @DirectParentOnlyProductionEndpoint.AuditedPublicEndpoint

/-- Contextual stage audit exposed by the replay-stable terminal package. -/
abbrev TerminalStageAuditTrail :=
  @DirectParentOnlyProductionStageAudit.TerminalStageAuditTrail

/--
Prior folded authority opened from the opaque compressed proof.

This is the authority side of the final endpoint stated on its own: the proof
must open to a folded `F'` authority object, and that object must accept the
same prior `(steps, image)` pair used by the terminal stage.
-/
def OpenedPriorAuthority
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (opener :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
        (PriorProof := PriorProof)
        ctx)
    (priorSteps : Nat)
    (priorProof : PriorProof)
    (priorImage :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary) :
    Prop :=
  opener.openAuthority priorProof ≠ none ∧
    ∃ authority :
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
          priorImage

/--
Final endpoint no-swap conclusion for one alternate private DEC child table.

The alternate table is admitted only through the full pointwise private-DEC
requirements for the same parent source. The conclusion identifies it with the
audited child table used by both terminal images and returns the corresponding
`Pi_CCS -> Pi_RLC` parent-source audits.
-/
def PointwiseChildTableNoSwap
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (priorSteps : Nat)
    (priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary)
    (otherInputs : DecDigitUniqueness.ColumnDigits n) : Prop :=
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
      altNext.accumulator.parentSource

/--
Final public-image facts exposed by the raw-verifier replay endpoint.

This packages the Construction-2 facts the terminal verifier needs after
replaying a concrete compressed prior proof: the final image is reachable under
the induced `F'` transition, its step is the next step, the verifier-key and
initial-boundary digests are preserved, and the final public image is
well-formed.
-/
def FinalPublicImageInvariants
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (priorSteps : Nat)
    (nextImage :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary) :
    Prop :=
  FPrimeInduction.Reachable
      (DirectParentOnlyProductionSoundness.Transition ctx.toProductionContext)
      ctx.initial
      (priorSteps + 1)
      nextImage ∧
    nextImage.step = priorSteps + 1 ∧
    ctx.initial.vkDigest = nextImage.vkDigest ∧
    ctx.initial.initialBoundary = nextImage.initialBoundary ∧
    Construction2DirectFPrime.WellFormed nextImage

/--
Final replay endpoint for a raw compressed prior verifier.

The endpoint combines the exact facts required by the optimized parent-only
public state: the opaque prior proof opens to folded `F'` authority, replaying
that same proof fixes the prior and terminal public images, the computed
`Pi_CCS -> Pi_RLC -> Pi_DEC` stage evidence is exposed, and the private DEC
child table has an explicit pointwise no-swap witness.
-/
def RawVerifierReplayTerminalEndpoint
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (opener :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
        (PriorProof := PriorProof)
        ctx)
    (priorStepsA priorStepsB : Nat)
    (priorProof : PriorProof)
    (priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary) : Prop :=
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
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.SameProofReplayEndpoint
      ctx
      priorStepsA
      priorStepsB
      priorImageA
      priorImageB
      nextImageA
      nextImageB ∧
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.ComputedStageReplayEvidence
      ctx
      priorStepsA
      priorImageA
      nextImageA
      nextImageB ∧
    DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.ExplicitReplayNoSwapEvidence
      ctx
      priorStepsA
      priorImageA
      nextImageA
      nextImageB

/--
Build the final replay endpoint from a raw verifier and its opening theorem.

This is the theorem the concrete implementation should call after supplying
the real compressed verifier predicate, the fixed authority opener, and the
proof that every accepted verifier result opens to folded `F'` authority for
the same `(steps, image)` pair.
-/
theorem rawVerifierReplayTerminalEndpoint_ofAcceptedOpens
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
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        verify
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        verify
        priorStepsB
        priorProof
        priorImageB
        nextImageB
        latestProofB) :
    RawVerifierReplayTerminalEndpoint
      ctx
      opener
      priorStepsA
      priorStepsB
      priorProof
      priorImageA
      priorImageB
      nextImageA
      nextImageB := by
  simpa [RawVerifierReplayTerminalEndpoint]
    using
      DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.replayAuditPackageWithExplicitNoSwap_ofAcceptedOpens
        ctx
        verify
        opener
        acceptedOpens
        hA
        hB

/--
Build the final replay endpoint from the canonical prior-verifier opening
certificate.

This is the preferred production call surface: the concrete compressed
verifier supplies one `PriorVerifierAuthorityOpening`, whose opener and
accepted-opens theorem are reused unchanged by the final endpoint.
-/
theorem rawVerifierReplayTerminalEndpoint_ofPriorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
        ctx
        VerifyPrior)
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsB
        priorProof
        priorImageB
        nextImageB
        latestProofB) :
    RawVerifierReplayTerminalEndpoint
      ctx
      opening.opener
      priorStepsA
      priorStepsB
      priorProof
      priorImageA
      priorImageB
      nextImageA
      nextImageB :=
  rawVerifierReplayTerminalEndpoint_ofAcceptedOpens
    ctx
    VerifyPrior
    opening.opener
    opening.acceptedOpens
    hA
    hB

/-- The final endpoint exposes the opened prior folded authority directly. -/
theorem openedPriorAuthority_of_rawVerifierReplayTerminalEndpoint
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {opener :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
        (PriorProof := PriorProof)
        ctx}
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (h :
      RawVerifierReplayTerminalEndpoint
        ctx
        opener
        priorStepsA
        priorStepsB
        priorProof
        priorImageA
        priorImageB
        nextImageA
        nextImageB) :
    OpenedPriorAuthority ctx opener priorStepsA priorProof priorImageA :=
  ⟨h.1, h.2.1⟩

/--
The final endpoint rules out alternate private DEC child tables pointwise.

An alternate table must satisfy the full private-DEC requirements for the same
parent source before this theorem applies. Aggregate norm equalities or
self-consistent recomputed digests do not supply this hypothesis.
-/
theorem pointwiseChildTableNoSwap_of_rawVerifierReplayTerminalEndpoint
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {opener :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
        (PriorProof := PriorProof)
        ctx}
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (h :
      RawVerifierReplayTerminalEndpoint
        ctx
        opener
        priorStepsA
        priorStepsB
        priorProof
        priorImageA
        priorImageB
        nextImageA
        nextImageB)
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
    PointwiseChildTableNoSwap
      ctx
      priorStepsA
      priorImageA
      nextImageA
      nextImageB
      otherInputs := by
  simpa [PointwiseChildTableNoSwap]
    using
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.pointwise_child_table_unique_of_pointwise_child_replay_binding
        (DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.pointwise_child_replay_binding_of_same_proof_replay_endpoint
          h.2.2.1)
        hOther

/--
Production-certificate form of the final endpoint no-swap theorem.

This is the direct call surface for a concrete compressed verifier once its
`PriorVerifierAuthorityOpening` certificate is available.
-/
theorem pointwiseChildTableNoSwap_ofPriorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
        ctx
        VerifyPrior)
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
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
    PointwiseChildTableNoSwap
      ctx
      priorStepsA
      priorImageA
      nextImageA
      nextImageB
      otherInputs :=
  pointwiseChildTableNoSwap_of_rawVerifierReplayTerminalEndpoint
    (rawVerifierReplayTerminalEndpoint_ofPriorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB)
    hOther

/-- The final endpoint exposes same-proof replay stability directly. -/
theorem sameProofReplayEndpoint_of_rawVerifierReplayTerminalEndpoint
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {opener :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
        (PriorProof := PriorProof)
        ctx}
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (h :
      RawVerifierReplayTerminalEndpoint
        ctx
        opener
        priorStepsA
        priorStepsB
        priorProof
        priorImageA
        priorImageB
        nextImageA
        nextImageB) :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.SameProofReplayEndpoint
      ctx
      priorStepsA
      priorStepsB
      priorImageA
      priorImageB
      nextImageA
      nextImageB :=
  h.2.2.1

/-- The final endpoint exposes the computed stage replay evidence directly. -/
theorem computedStageReplayEvidence_of_rawVerifierReplayTerminalEndpoint
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {opener :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
        (PriorProof := PriorProof)
        ctx}
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (h :
      RawVerifierReplayTerminalEndpoint
        ctx
        opener
        priorStepsA
        priorStepsB
        priorProof
        priorImageA
        priorImageB
        nextImageA
        nextImageB) :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.ComputedStageReplayEvidence
      ctx
      priorStepsA
      priorImageA
      nextImageA
      nextImageB :=
  h.2.2.2.1

/--
The final endpoint exposes the exact computed `Pi_CCS` context that consumes
the pointwise private DEC child table.
-/
theorem piCCSDecKnowledge_of_rawVerifierReplayTerminalEndpoint
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {opener :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
        (PriorProof := PriorProof)
        ctx}
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (h :
      RawVerifierReplayTerminalEndpoint
        ctx
        opener
        priorStepsA
        priorStepsB
        priorProof
        priorImageA
        priorImageB
        nextImageA
        nextImageB) :
    ∃ priorInputs,
      let childPrior :=
        DirectParentOnlyProductionStageAudit.childCarryingPrior
          priorImageA.accumulator
          priorInputs
      let out :=
        ctx.toProductionContext.stage.computePiCCS priorStepsA childPrior
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
      out.step = priorStepsA ∧
      SuperNeo.PiCCSInterface.piCCSStrongStatement out.ctx ∧
      SuperNeo.PiDECInterface.piDECKnowledgeStatement out.ctx :=
  DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.piCCS_dec_knowledge_of_computed_stage_replay_evidence
    (computedStageReplayEvidence_of_rawVerifierReplayTerminalEndpoint h)

/--
The final endpoint exposes the exact computed `Pi_RLC` context that produces
the compact parent source.
-/
theorem piRLCDecKnowledge_of_rawVerifierReplayTerminalEndpoint
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {opener :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
        (PriorProof := PriorProof)
        ctx}
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (h :
      RawVerifierReplayTerminalEndpoint
        ctx
        opener
        priorStepsA
        priorStepsB
        priorProof
        priorImageA
        priorImageB
        nextImageA
        nextImageB) :
    ∃ priorInputs,
      let childPrior :=
        DirectParentOnlyProductionStageAudit.childCarryingPrior
          priorImageA.accumulator
          priorInputs
      let out :=
        ctx.toProductionContext.stage.computePiCCS priorStepsA childPrior
      nextImageA.accumulator.parentSource =
        ctx.toProductionContext.stage.computePiRLC priorStepsA out ∧
      nextImageB.accumulator.parentSource =
        ctx.toProductionContext.stage.computePiRLC priorStepsA out ∧
      SuperNeo.PiRLCInterface.piRLCWeakStatement
        (ctx.toProductionContext.stage.piRLCContext
          out
          (ctx.toProductionContext.stage.computePiRLC priorStepsA out)) ∧
      SuperNeo.PiDECInterface.piDECKnowledgeStatement
        (ctx.toProductionContext.stage.piRLCContext
          out
          (ctx.toProductionContext.stage.computePiRLC priorStepsA out)) :=
  DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.piRLC_dec_knowledge_of_computed_stage_replay_evidence
    (computedStageReplayEvidence_of_rawVerifierReplayTerminalEndpoint h)

/-- The final endpoint exposes pointwise no-swap evidence directly. -/
theorem explicitReplayNoSwapEvidence_of_rawVerifierReplayTerminalEndpoint
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {opener :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
        (PriorProof := PriorProof)
        ctx}
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (h :
      RawVerifierReplayTerminalEndpoint
        ctx
        opener
        priorStepsA
        priorStepsB
        priorProof
        priorImageA
        priorImageB
        nextImageA
        nextImageB) :
    ExplicitReplayNoSwapEvidence
      ctx
      priorStepsA
      priorImageA
      nextImageA
      nextImageB :=
  h.2.2.2.2

/-- The final endpoint exposes the flattened public endpoint directly. -/
theorem auditedPublicEndpoint_of_rawVerifierReplayTerminalEndpoint
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {opener :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
        (PriorProof := PriorProof)
        ctx}
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (h :
      RawVerifierReplayTerminalEndpoint
        ctx
        opener
        priorStepsA
        priorStepsB
        priorProof
        priorImageA
        priorImageB
        nextImageA
        nextImageB) :
    AuditedPublicEndpoint
      ctx.toProductionContext
      priorStepsA
      priorImageA
      nextImageA
      nextImageB :=
  (sameProofReplayEndpoint_of_rawVerifierReplayTerminalEndpoint h).2.2.2.1

/-- The final endpoint exposes the contextual stage audit directly. -/
theorem terminalStageAuditTrail_of_rawVerifierReplayTerminalEndpoint
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {opener :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
        (PriorProof := PriorProof)
        ctx}
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (h :
      RawVerifierReplayTerminalEndpoint
        ctx
        opener
        priorStepsA
        priorStepsB
        priorProof
        priorImageA
        priorImageB
        nextImageA
        nextImageB) :
    TerminalStageAuditTrail
      ctx.toProductionContext
      priorStepsA
      priorImageA
      nextImageA
      nextImageB :=
  (sameProofReplayEndpoint_of_rawVerifierReplayTerminalEndpoint h).2.2.2.2

/-- The final endpoint exposes final reachability and public-image invariants. -/
theorem finalPublicImageInvariants_of_rawVerifierReplayTerminalEndpoint
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {opener :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
        (PriorProof := PriorProof)
        ctx}
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (h :
      RawVerifierReplayTerminalEndpoint
        ctx
        opener
        priorStepsA
        priorStepsB
        priorProof
        priorImageA
        priorImageB
        nextImageA
        nextImageB) :
    FinalPublicImageInvariants ctx priorStepsA nextImageA := by
  rcases auditedPublicEndpoint_of_rawVerifierReplayTerminalEndpoint h with
    ⟨_hPriorReach,
      hFinalReach,
      _hNextEqAlt,
      _hParentSource,
      _hNextBoundary,
      _hAltBoundary,
      hStep,
      hVk,
      hInitialBoundary,
      hWellFormed,
      _hAudit⟩
  exact
    ⟨hFinalReach,
      hStep,
      hVk,
      hInitialBoundary,
      hWellFormed⟩

/-- Production-certificate form of the opened prior folded authority. -/
theorem openedPriorAuthority_ofPriorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
        ctx
        VerifyPrior)
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsB
        priorProof
        priorImageB
        nextImageB
        latestProofB) :
    OpenedPriorAuthority
      ctx
      opening.opener
      priorStepsA
      priorProof
      priorImageA :=
  openedPriorAuthority_of_rawVerifierReplayTerminalEndpoint
    (rawVerifierReplayTerminalEndpoint_ofPriorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB)

/-- Production-certificate form of same-proof replay stability. -/
theorem sameProofReplayEndpoint_ofPriorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
        ctx
        VerifyPrior)
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsB
        priorProof
        priorImageB
        nextImageB
        latestProofB) :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.SameProofReplayEndpoint
      ctx
      priorStepsA
      priorStepsB
      priorImageA
      priorImageB
      nextImageA
      nextImageB :=
  sameProofReplayEndpoint_of_rawVerifierReplayTerminalEndpoint
    (rawVerifierReplayTerminalEndpoint_ofPriorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB)

/-- Production-certificate form of computed replay-stage evidence. -/
theorem computedStageReplayEvidence_ofPriorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
        ctx
        VerifyPrior)
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsB
        priorProof
        priorImageB
        nextImageB
        latestProofB) :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.ComputedStageReplayEvidence
      ctx
      priorStepsA
      priorImageA
      nextImageA
      nextImageB :=
  computedStageReplayEvidence_of_rawVerifierReplayTerminalEndpoint
    (rawVerifierReplayTerminalEndpoint_ofPriorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB)

/-- Production-certificate form of computed `Pi_CCS` DEC knowledge. -/
theorem piCCSDecKnowledge_ofPriorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
        ctx
        VerifyPrior)
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsB
        priorProof
        priorImageB
        nextImageB
        latestProofB) :
    ∃ priorInputs,
      let childPrior :=
        DirectParentOnlyProductionStageAudit.childCarryingPrior
          priorImageA.accumulator
          priorInputs
      let out :=
        ctx.toProductionContext.stage.computePiCCS priorStepsA childPrior
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
      out.step = priorStepsA ∧
      SuperNeo.PiCCSInterface.piCCSStrongStatement out.ctx ∧
      SuperNeo.PiDECInterface.piDECKnowledgeStatement out.ctx :=
  piCCSDecKnowledge_of_rawVerifierReplayTerminalEndpoint
    (rawVerifierReplayTerminalEndpoint_ofPriorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB)

/-- Production-certificate form of computed `Pi_RLC` DEC knowledge. -/
theorem piRLCDecKnowledge_ofPriorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
        ctx
        VerifyPrior)
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsB
        priorProof
        priorImageB
        nextImageB
        latestProofB) :
    ∃ priorInputs,
      let childPrior :=
        DirectParentOnlyProductionStageAudit.childCarryingPrior
          priorImageA.accumulator
          priorInputs
      let out :=
        ctx.toProductionContext.stage.computePiCCS priorStepsA childPrior
      nextImageA.accumulator.parentSource =
        ctx.toProductionContext.stage.computePiRLC priorStepsA out ∧
      nextImageB.accumulator.parentSource =
        ctx.toProductionContext.stage.computePiRLC priorStepsA out ∧
      SuperNeo.PiRLCInterface.piRLCWeakStatement
        (ctx.toProductionContext.stage.piRLCContext
          out
          (ctx.toProductionContext.stage.computePiRLC priorStepsA out)) ∧
      SuperNeo.PiDECInterface.piDECKnowledgeStatement
        (ctx.toProductionContext.stage.piRLCContext
          out
          (ctx.toProductionContext.stage.computePiRLC priorStepsA out)) :=
  piRLCDecKnowledge_of_rawVerifierReplayTerminalEndpoint
    (rawVerifierReplayTerminalEndpoint_ofPriorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB)

/-- Production-certificate form of explicit pointwise no-swap evidence. -/
theorem explicitReplayNoSwapEvidence_ofPriorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
        ctx
        VerifyPrior)
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsB
        priorProof
        priorImageB
        nextImageB
        latestProofB) :
    ExplicitReplayNoSwapEvidence
      ctx
      priorStepsA
      priorImageA
      nextImageA
      nextImageB :=
  explicitReplayNoSwapEvidence_of_rawVerifierReplayTerminalEndpoint
    (rawVerifierReplayTerminalEndpoint_ofPriorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB)

/-- Production-certificate form of the flattened public endpoint. -/
theorem auditedPublicEndpoint_ofPriorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
        ctx
        VerifyPrior)
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsB
        priorProof
        priorImageB
        nextImageB
        latestProofB) :
    AuditedPublicEndpoint
      ctx.toProductionContext
      priorStepsA
      priorImageA
      nextImageA
      nextImageB :=
  auditedPublicEndpoint_of_rawVerifierReplayTerminalEndpoint
    (rawVerifierReplayTerminalEndpoint_ofPriorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB)

/-- Production-certificate form of the contextual terminal stage audit. -/
theorem terminalStageAuditTrail_ofPriorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
        ctx
        VerifyPrior)
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsB
        priorProof
        priorImageB
        nextImageB
        latestProofB) :
    TerminalStageAuditTrail
      ctx.toProductionContext
      priorStepsA
      priorImageA
      nextImageA
      nextImageB :=
  terminalStageAuditTrail_of_rawVerifierReplayTerminalEndpoint
    (rawVerifierReplayTerminalEndpoint_ofPriorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB)

/-- Production-certificate form of final reachability and public invariants. -/
theorem finalPublicImageInvariants_ofPriorVerifierAuthorityOpening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    (opening :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
        ctx
        VerifyPrior)
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorStepsB
        priorProof
        priorImageB
        nextImageB
        latestProofB) :
    FinalPublicImageInvariants ctx priorStepsA nextImageA :=
  finalPublicImageInvariants_of_rawVerifierReplayTerminalEndpoint
    (rawVerifierReplayTerminalEndpoint_ofPriorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB)

end DirectParentOnlyProductionSuperNeoReuseFinalEndpoint

end DirectCcsFPrime
