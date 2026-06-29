import DirectCcsFPrime.ProofSystem.Production.Security.DirectParentOnlyProductionSuperNeoReuseFinalEndpoint

/-!
End-to-end soundness surface for the Section 7.1-backed parent-only path.

This module composes the final endpoint into the theorem shape the production
instantiation needs after it supplies the two real obligations: a concrete
Section 7.1-backed production context and a concrete compressed-prior verifier
opening certificate.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionSuperNeoReuseEndToEnd

/-- Section 7.1-backed production context. -/
abbrev ProductionContext :=
  DirectParentOnlyProductionSuperNeoReuseFinalEndpoint.ProductionContext

/--
End-to-end terminal soundness package for a concrete verifier-opening
certificate.

The package keeps authority and private-child reuse explicit. The compressed
prior proof must open to folded `F'` authority for the same prior public image,
the same proof must replay to the same terminal image, the final Construction-2
public image must be reachable with preserved public-image invariants, and
every alternate private DEC child table is accepted only through the full
pointwise DEC requirements for the same parent source.
-/
structure CertifiedTerminalEndToEnd
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
    (priorStepsA priorStepsB : Nat)
    (priorProof : PriorProof)
    (priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary) :
    Prop where
  openedAuthority :
    DirectParentOnlyProductionSuperNeoReuseFinalEndpoint.OpenedPriorAuthority
      ctx
      opening.opener
      priorStepsA
      priorProof
      priorImageA
  sameProofReplay :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.SameProofReplayEndpoint
      ctx
      priorStepsA
      priorStepsB
      priorImageA
      priorImageB
      nextImageA
      nextImageB
  computedStage :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.ComputedStageReplayEvidence
      ctx
      priorStepsA
      priorImageA
      nextImageA
      nextImageB
  piCCSDecKnowledge :
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
      SuperNeo.PiDECInterface.piDECKnowledgeStatement out.ctx
  piRLCDecKnowledge :
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
          (ctx.toProductionContext.stage.computePiRLC priorStepsA out))
  explicitNoSwap :
    DirectParentOnlyProductionSuperNeoReuseFinalEndpoint.ExplicitReplayNoSwapEvidence
      ctx
      priorStepsA
      priorImageA
      nextImageA
      nextImageB
  everyAlternateChildNoSwap :
    ∀ otherInputs : DecDigitUniqueness.ColumnDigits n,
      ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := ctx.parentHash.hashEncoded)
        (params := params)
        (ce := ctx.data.ce)
        (StatementEncodes :=
          ParentOpeningAuthorization.StatementEncodesByCommitment
            ctx.commitmentOfParent)
        priorImageA.accumulator.parentSource
        otherInputs →
        DirectParentOnlyProductionSuperNeoReuseFinalEndpoint.PointwiseChildTableNoSwap
          ctx
          priorStepsA
          priorImageA
          nextImageA
          nextImageB
          otherInputs
  auditedPublicEndpoint :
    DirectParentOnlyProductionSuperNeoReuseFinalEndpoint.AuditedPublicEndpoint
      ctx.toProductionContext
      priorStepsA
      priorImageA
      nextImageA
      nextImageB
  terminalStageAudit :
    DirectParentOnlyProductionSuperNeoReuseFinalEndpoint.TerminalStageAuditTrail
      ctx.toProductionContext
      priorStepsA
      priorImageA
      nextImageA
      nextImageB
  section71TargetAudit :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.TerminalSection71StageTargetAuditTrail
      ctx
      priorStepsA
      priorImageA
      nextImageA
      nextImageB
  finalInvariants :
    DirectParentOnlyProductionSuperNeoReuseFinalEndpoint.FinalPublicImageInvariants
      ctx
      priorStepsA
      nextImageA

def auditedChildCarryingPrior
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (_ctx : ProductionContext Digest Boundary n params)
    (priorImage :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary)
    (priorInputs : DecDigitUniqueness.ColumnDigits n) :
    DirectTerminalSoundness.AccHandle Digest n :=
  DirectParentOnlyProductionStageAudit.childCarryingPrior
    priorImage.accumulator
    priorInputs

def auditedPiCCSOutput
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (priorSteps : Nat)
    (priorImage :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary)
    (priorInputs : DecDigitUniqueness.ColumnDigits n) :
    DirectStageSemantics.ContextualPiCCSOut :=
  ctx.toProductionContext.stage.computePiCCS
    priorSteps
    (auditedChildCarryingPrior ctx priorImage priorInputs)

def auditedPiRLCParentSource
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (priorSteps : Nat)
    (priorImage :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary)
    (priorInputs : DecDigitUniqueness.ColumnDigits n) :
    DigestParentBinding.Source Digest :=
  ctx.toProductionContext.stage.computePiRLC
    priorSteps
    (auditedPiCCSOutput ctx priorSteps priorImage priorInputs)

/-- Named certificate for hidden private DEC facts and `Pi_CCS -> Pi_RLC` reuse. -/
structure CertifiedTerminalNonAggregatePrivateDecStageCertificate
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (priorSteps : Nat)
    (priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary) where
  priorInputs : DecDigitUniqueness.ColumnDigits n
  privateDec :
    ParentOnlyAccumulatorStep.PointwisePrivateDecCertificate
      (n := n)
      (hashEncoded := ctx.parentHash.hashEncoded)
      (params := params)
      (ce := ctx.data.ce)
      (StatementEncodes :=
        ParentOpeningAuthorization.StatementEncodesByCommitment
          ctx.commitmentOfParent)
      priorImage.accumulator.parentSource
      priorInputs
  childAudit :
    DirectParentOnlyProductionChildMembership.PointwiseChildAuditTrail
      ctx.toProductionContext
      priorImage.accumulator.parentSource
      priorInputs
  nextImageComputed :
    nextImage =
      DirectParentOnlyProductionSoundness.ComputedNextImage
        ctx.toProductionContext
        priorSteps
        priorImage
        priorInputs
  altImageComputed :
    altNext =
      DirectParentOnlyProductionSoundness.ComputedNextImage
        ctx.toProductionContext
        priorSteps
        priorImage
        priorInputs
  nextParentSourceComputed :
    nextImage.accumulator.parentSource =
      auditedPiRLCParentSource ctx priorSteps priorImage priorInputs
  altParentSourceComputed :
    altNext.accumulator.parentSource =
      auditedPiRLCParentSource ctx priorSteps priorImage priorInputs
  uniquePointwisePrivateChildren :
    ∀ otherInputs : DecDigitUniqueness.ColumnDigits n,
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
          otherInputs = priorInputs
  piCCSStep :
    (auditedPiCCSOutput ctx priorSteps priorImage priorInputs).step =
      priorSteps
  piCCSStrong :
    SuperNeo.PiCCSInterface.piCCSStrongStatement
      (auditedPiCCSOutput ctx priorSteps priorImage priorInputs).ctx
  piCCSDecKnowledge :
    SuperNeo.PiDECInterface.piDECKnowledgeStatement
      (auditedPiCCSOutput ctx priorSteps priorImage priorInputs).ctx
  piRLCWeak :
    SuperNeo.PiRLCInterface.piRLCWeakStatement
      (ctx.toProductionContext.stage.piRLCContext
        (auditedPiCCSOutput ctx priorSteps priorImage priorInputs)
        (auditedPiRLCParentSource ctx priorSteps priorImage priorInputs))
  piRLCDecKnowledge :
    SuperNeo.PiDECInterface.piDECKnowledgeStatement
      (ctx.toProductionContext.stage.piRLCContext
        (auditedPiCCSOutput ctx priorSteps priorImage priorInputs)
        (auditedPiRLCParentSource ctx priorSteps priorImage priorInputs))

/--
Flattened non-aggregate private DEC and stage facts exposed by the final
end-to-end package.

This is the audit shape that matters for the parent-only optimization: the
private child table is accepted by the private `Pi_DEC` verifier against the
opened parent, uses the fixed production CE relation and Ajtai parameters,
passes bitness/exact-length/per-column recomposition, is exactly the
CE-witness-derived table, and is the same table used by the contextual
`Pi_CCS -> Pi_RLC` stage computations. Any alternate child table must satisfy
the full pointwise private DEC requirements for the same parent source before
the theorem identifies it with the audited table.
-/
def CertifiedTerminalNonAggregatePrivateDecStageFacts
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (priorSteps : Nat)
    (priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary) :
    Prop :=
  Nonempty
    (CertifiedTerminalNonAggregatePrivateDecStageCertificate
      ctx
      priorSteps
      priorImage
      nextImage
      altNext)

/--
Project the exact non-aggregate private DEC and stage facts from the
end-to-end package.
-/
theorem nonAggregatePrivateDecStageFacts_of_certifiedTerminalEndToEnd
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    {opening :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
        ctx
        VerifyPrior}
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hEnd :
      CertifiedTerminalEndToEnd
        ctx
        opening
        priorStepsA
        priorStepsB
        priorProof
        priorImageA
        priorImageB
        nextImageA
        nextImageB) :
    CertifiedTerminalNonAggregatePrivateDecStageFacts
      ctx
      priorStepsA
      priorImageA
      nextImageA
      nextImageB := by
  rcases
      DirectParentOnlyProductionStageAudit.computed_stage_evidence_of_terminal_stage_audit_trail
        hEnd.terminalStageAudit with
    ⟨priorInputs,
      hPointwise,
      hAudit,
      hNext,
      hAlt,
      hNextSource,
      hAltSource,
      hUnique,
      hStep,
      hPiCCS,
      hPiCCSDec,
      hPiRLC,
      hPiRLCDec⟩
  rcases hPointwise with ⟨privateDec⟩
  exact
    ⟨{
      priorInputs := priorInputs
      privateDec := privateDec
      childAudit := hAudit
      nextImageComputed := hNext
      altImageComputed := hAlt
      nextParentSourceComputed := hNextSource
      altParentSourceComputed := hAltSource
      uniquePointwisePrivateChildren := hUnique
      piCCSStep := hStep
      piCCSStrong := hPiCCS
      piCCSDecKnowledge := hPiCCSDec
      piRLCWeak := hPiRLC
      piRLCDecKnowledge := hPiRLCDec
    }⟩

/-- The final non-aggregate facts expose the exact private DEC certificate. -/
theorem privateDecCertificate_of_nonAggregatePrivateDecStageFacts
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {priorSteps : Nat}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hFacts :
      CertifiedTerminalNonAggregatePrivateDecStageFacts
        ctx
        priorSteps
        priorImage
        nextImage
        altNext) :
    ∃ priorInputs,
      Nonempty
        (ParentOnlyAccumulatorStep.PointwisePrivateDecCertificate
          (n := n)
          (hashEncoded := ctx.parentHash.hashEncoded)
          (params := params)
          (ce := ctx.data.ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              ctx.commitmentOfParent)
          priorImage.accumulator.parentSource
          priorInputs) := by
  rcases hFacts with ⟨cert⟩
  exact ⟨cert.priorInputs, ⟨cert.privateDec⟩⟩

/-- The final non-aggregate facts expose the real child-audit trail. -/
theorem childAuditTrail_of_nonAggregatePrivateDecStageFacts
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {priorSteps : Nat}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hFacts :
      CertifiedTerminalNonAggregatePrivateDecStageFacts
        ctx
        priorSteps
        priorImage
        nextImage
        altNext) :
    ∃ priorInputs,
      DirectParentOnlyProductionChildMembership.PointwiseChildAuditTrail
        ctx.toProductionContext
        priorImage.accumulator.parentSource
        priorInputs := by
  rcases hFacts with ⟨cert⟩
  exact ⟨cert.priorInputs, cert.childAudit⟩

/-- Every accepted alternate child table is the audited private DEC table. -/
theorem uniquePrivateChildren_of_nonAggregatePrivateDecStageFacts
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {priorSteps : Nat}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hFacts :
      CertifiedTerminalNonAggregatePrivateDecStageFacts
        ctx
        priorSteps
        priorImage
        nextImage
        altNext) :
    ∃ priorInputs,
      ∀ otherInputs : DecDigitUniqueness.ColumnDigits n,
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
            otherInputs = priorInputs := by
  rcases hFacts with ⟨cert⟩
  exact ⟨cert.priorInputs, cert.uniquePointwisePrivateChildren⟩

/-- The audited next `Pi_CCS` wires are the child witness digit table. -/
theorem nextPiCCSInputs_eq_childWitnessDigitTable_of_nonAggregatePrivateDecStageFacts
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {priorSteps : Nat}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hFacts :
      CertifiedTerminalNonAggregatePrivateDecStageFacts
        ctx
        priorSteps
        priorImage
        nextImage
        altNext) :
    ∃
      (priorInputs : DecDigitUniqueness.ColumnDigits n)
      (bundle : SuperNeoBridge.AjtaiCEChildBundle 14 n),
        bundle.ce = ctx.data.ce ∧
          bundle.ajtaiParams = params ∧
          priorInputs =
            SuperNeoBridge.childWitnessDigitTable
              (k := 14)
              (n := n)
              bundle.witness := by
  rcases hFacts with ⟨cert⟩
  refine
    ⟨cert.priorInputs,
      cert.privateDec.bundle,
      cert.privateDec.fixedCE,
      cert.privateDec.fixedAjtaiParams,
      ?_⟩
  calc
    cert.priorInputs = cert.privateDec.bundle.nextPiCCSInputs :=
      cert.privateDec.nextInputIdentity
    _ = cert.privateDec.bundle.digitTable :=
      cert.privateDec.wireIdentity
    _ =
        SuperNeoBridge.childWitnessDigitTable
          (k := 14)
          (n := n)
          cert.privateDec.bundle.witness :=
      cert.privateDec.witnessTable

/-- Project the exact Section 7.1 owner-target audit. -/
theorem section71StageTargetAuditTrail_of_certifiedTerminalEndToEnd
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {VerifyPrior :
      Nat →
        PriorProof →
        DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
          Prop}
    {opening :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
        ctx
        VerifyPrior}
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hEnd :
      CertifiedTerminalEndToEnd
        ctx
        opening
        priorStepsA
        priorStepsB
        priorProof
        priorImageA
        priorImageB
        nextImageA
        nextImageB) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.TerminalSection71StageTargetAuditTrail
      ctx
      priorStepsA
      priorImageA
      nextImageA
      nextImageB :=
  hEnd.section71TargetAudit

/--
Compose the final endpoint into the end-to-end production theorem surface.

This is the theorem the concrete implementation should target after proving
its compressed verifier opening certificate. No hash implementation is
formalized here; Poseidon2 enters only through the parent hash binding carried
by `ctx`.
-/
theorem certifiedTerminalEndToEnd_ofPriorVerifierAuthorityOpening
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
    CertifiedTerminalEndToEnd
      ctx
      opening
      priorStepsA
      priorStepsB
      priorProof
      priorImageA
      priorImageB
      nextImageA
      nextImageB where
  openedAuthority :=
    DirectParentOnlyProductionSuperNeoReuseFinalEndpoint.openedPriorAuthority_ofPriorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB
  sameProofReplay :=
    DirectParentOnlyProductionSuperNeoReuseFinalEndpoint.sameProofReplayEndpoint_ofPriorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB
  computedStage :=
    DirectParentOnlyProductionSuperNeoReuseFinalEndpoint.computedStageReplayEvidence_ofPriorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB
  piCCSDecKnowledge :=
    DirectParentOnlyProductionSuperNeoReuseFinalEndpoint.piCCSDecKnowledge_ofPriorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB
  piRLCDecKnowledge :=
    DirectParentOnlyProductionSuperNeoReuseFinalEndpoint.piRLCDecKnowledge_ofPriorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB
  explicitNoSwap :=
    DirectParentOnlyProductionSuperNeoReuseFinalEndpoint.explicitReplayNoSwapEvidence_ofPriorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB
  everyAlternateChildNoSwap := by
    intro otherInputs hOther
    exact
      DirectParentOnlyProductionSuperNeoReuseFinalEndpoint.pointwiseChildTableNoSwap_ofPriorVerifierAuthorityOpening
        ctx
        opening
        hA
        hB
        hOther
  auditedPublicEndpoint :=
    DirectParentOnlyProductionSuperNeoReuseFinalEndpoint.auditedPublicEndpoint_ofPriorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB
  terminalStageAudit :=
    DirectParentOnlyProductionSuperNeoReuseFinalEndpoint.terminalStageAuditTrail_ofPriorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB
  section71TargetAudit :=
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.terminal_section71_stage_target_audit_trail_of_audited_public_endpoint
      ctx
      ⟨DirectParentOnlyProductionSuperNeoReuseFinalEndpoint.auditedPublicEndpoint_ofPriorVerifierAuthorityOpening
          ctx
          opening
          hA
          hB,
        DirectParentOnlyProductionSuperNeoReuseFinalEndpoint.terminalStageAuditTrail_ofPriorVerifierAuthorityOpening
          ctx
          opening
          hA
          hB⟩
  finalInvariants :=
    DirectParentOnlyProductionSuperNeoReuseFinalEndpoint.finalPublicImageInvariants_ofPriorVerifierAuthorityOpening
      ctx
      opening
      hA
      hB

/--
Raw-verifier form of the end-to-end production theorem.

The concrete implementation may supply its verifier predicate, fixed authority
opener, and accepted-opens theorem directly. Lean constructs the canonical
`PriorVerifierAuthorityOpening` certificate internally and then exposes the same
end-to-end package as the certificate-based theorem.
-/
theorem certifiedTerminalEndToEnd_ofAcceptedOpens
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
    CertifiedTerminalEndToEnd
      ctx
      ({ opener := opener
         acceptedOpens := acceptedOpens } :
        DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
          ctx
          verify)
      priorStepsA
      priorStepsB
      priorProof
      priorImageA
      priorImageB
      nextImageA
      nextImageB :=
  certifiedTerminalEndToEnd_ofPriorVerifierAuthorityOpening
    ctx
    ({ opener := opener
       acceptedOpens := acceptedOpens } :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
        ctx
        verify)
    hA
    hB

/--
Single-terminal certificate form of the end-to-end theorem.

The ordinary verifier call has one terminal acceptance. Lean replays that same
acceptance against itself, so callers get the full replay-stable and pointwise
no-swap package without manufacturing a duplicate acceptance argument.
-/
theorem certifiedSingleTerminalEndToEnd_ofPriorVerifierAuthorityOpening
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
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProof : Unit}
    (hAccepted :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        VerifyPrior
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof) :
    CertifiedTerminalEndToEnd
      ctx
      opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  certifiedTerminalEndToEnd_ofPriorVerifierAuthorityOpening
    ctx
    opening
    hAccepted
    hAccepted

/--
Raw-verifier single-terminal form of the end-to-end theorem.

The concrete implementation supplies the verifier predicate, fixed opener, and
accepted-opens theorem directly. The single accepted terminal proof is replayed
against itself to expose the same authority, public-image, stage, and pointwise
child no-swap facts as the two-acceptance theorem.
-/
theorem certifiedSingleTerminalEndToEnd_ofAcceptedOpens
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
    {priorImage nextImage :
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
        latestProof) :
    CertifiedTerminalEndToEnd
      ctx
      ({ opener := opener
         acceptedOpens := acceptedOpens } :
        DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
          ctx
          verify)
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  certifiedTerminalEndToEnd_ofAcceptedOpens
    ctx
    verify
    opener
    acceptedOpens
    hAccepted
    hAccepted

/--
Raw-verifier latest-step projection to the exact non-aggregate private DEC and
stage facts.

This is the direct implementation-facing audit theorem: once the concrete
prior verifier has an accepted-opens theorem, an accepted prior proof and one
accepted latest step expose the accepted private `Pi_DEC` proof, fixed
CE/Ajtai objects, per-column recomposition, CE-witness table identity,
next-`Pi_CCS` wire identity, and contextual `Pi_CCS -> Pi_RLC` parent-source
computation.
-/
theorem nonAggregatePrivateDecStageFacts_ofAcceptedOpensLatestStep
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
    {priorImage nextImage :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      verify priorSteps priorProof priorImage)
    (hLatest :
      DirectParentOnlyProductionSoundness.VerifyLatestStep
        ctx.toProductionContext
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof) :
    CertifiedTerminalNonAggregatePrivateDecStageFacts
      ctx
      priorSteps
      priorImage
      nextImage
      nextImage :=
  nonAggregatePrivateDecStageFacts_of_certifiedTerminalEndToEnd
    (certifiedSingleTerminalEndToEnd_ofAcceptedOpens
      ctx
      verify
      opener
      acceptedOpens
      ({ priorAccepted := hPrior
         latestAccepted := hLatest } :
        DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
          ctx
          verify
          priorSteps
          priorProof
          priorImage
          nextImage
          latestProof))

/--
Raw-verifier latest-step projection to the Section 7.1 owner-target audit.
-/
theorem section71StageTargetAuditTrail_ofAcceptedOpensLatestStep
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
    {priorImage nextImage :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      verify priorSteps priorProof priorImage)
    (hLatest :
      DirectParentOnlyProductionSoundness.VerifyLatestStep
        ctx.toProductionContext
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.TerminalSection71StageTargetAuditTrail
      ctx
      priorSteps
      priorImage
      nextImage
      nextImage :=
  section71StageTargetAuditTrail_of_certifiedTerminalEndToEnd
    (certifiedSingleTerminalEndToEnd_ofAcceptedOpens
      ctx
      verify
      opener
      acceptedOpens
      ({ priorAccepted := hPrior
         latestAccepted := hLatest } :
        DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
          ctx
          verify
          priorSteps
          priorProof
          priorImage
          nextImage
          latestProof))

/-- Proof-carrying folded prior proof for the induced Section 7.1-backed context. -/
abbrev ProofCarryingPriorProof
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) :=
  DirectParentOnlyProductionSoundness.ProofCarryingPriorProof
    ctx.toProductionContext

/--
Verifier predicate for the proof-carrying prior-authority baseline.

Acceptance is exactly folded `F'` authority acceptance for the context's
transition and initial image.
-/
def ProofCarryingVerifyPrior
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) :
    Nat →
      ProofCarryingPriorProof ctx →
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary →
        Prop :=
  FoldedFPrimeAuthority.Accepts
    (Transition :=
      DirectParentOnlyProductionSoundness.Transition
        ctx.toProductionContext)
    (initial := ctx.initial)

/--
The proof-carrying baseline opener is identity: the prior proof already is the
folded `F'` authority object consumed by the final theorem.
-/
def proofCarryingPriorOpener
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorAuthorityOpener
      (PriorProof := ProofCarryingPriorProof ctx)
      ctx where
  openAuthority proof := some proof

/--
The proof-carrying baseline satisfies the accepted-opens obligation
definitionally.
-/
theorem proofCarryingPriorAcceptedOpens
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) :
    ∀ steps proof image,
      ProofCarryingVerifyPrior ctx steps proof image →
        ∃ authority :
            DirectParentOnlyProductionSoundness.ProofCarryingPriorProof
              ctx.toProductionContext,
          (proofCarryingPriorOpener ctx).openAuthority proof =
              some authority ∧
            FoldedFPrimeAuthority.Accepts
              (Transition :=
                DirectParentOnlyProductionSoundness.Transition
                  ctx.toProductionContext)
              (initial := ctx.initial)
              steps
              authority
              image := by
  intro steps proof image hAccept
  exact ⟨proof, rfl, hAccept⟩

/--
Canonical prior-opening certificate for proof-carrying folded prior authority.

This closes the non-compressed baseline at the final end-to-end surface. The
compressed verifier path must still prove its own accepted-opens theorem to
reduce to this same authority shape.
-/
def proofCarryingPriorVerifierAuthorityOpening
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PriorVerifierAuthorityOpening
      ctx
      (ProofCarryingVerifyPrior ctx) where
  opener := proofCarryingPriorOpener ctx
  acceptedOpens := proofCarryingPriorAcceptedOpens ctx

/--
Proof-carrying single-terminal end-to-end theorem.

This is the final theorem-level baseline: if the prior proof itself carries
folded `F'` reachability authority and the terminal latest step is accepted,
the final package exposes the same public-image invariants, stage evidence, and
pointwise private-child no-swap guarantees as the compressed verifier path.
-/
theorem certifiedSingleTerminalEndToEnd_ofProofCarryingPriorAuthority
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {priorSteps : Nat}
    {priorProof : ProofCarryingPriorProof ctx}
    {priorImage nextImage :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProof : Unit}
    (hAccepted :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        (ProofCarryingVerifyPrior ctx)
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof) :
    CertifiedTerminalEndToEnd
      ctx
      (proofCarryingPriorVerifierAuthorityOpening ctx)
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  certifiedSingleTerminalEndToEnd_ofPriorVerifierAuthorityOpening
    ctx
    (proofCarryingPriorVerifierAuthorityOpening ctx)
    hAccepted

/--
Proof-carrying latest-step form of the single-terminal theorem.

This is the shortest proof-carrying call path: the prior proof supplies its own
authorized `(steps, image)` pair, and the caller supplies only the accepted
latest Construction-2 step from that image. Lean builds the terminal acceptance
record internally before returning the same end-to-end package.
-/
theorem certifiedSingleTerminalEndToEnd_ofProofCarryingLatestStep
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {priorProof : ProofCarryingPriorProof ctx}
    {nextImage :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProof : Unit}
    (hLatest :
      DirectParentOnlyProductionSoundness.VerifyLatestStep
        ctx.toProductionContext
        priorProof.steps
        priorProof
        priorProof.image
        nextImage
        latestProof) :
    CertifiedTerminalEndToEnd
      ctx
      (proofCarryingPriorVerifierAuthorityOpening ctx)
      priorProof.steps
      priorProof.steps
      priorProof
      priorProof.image
      priorProof.image
      nextImage
      nextImage :=
  certifiedSingleTerminalEndToEnd_ofProofCarryingPriorAuthority
    ctx
    ({ priorAccepted := ⟨rfl, rfl⟩
       latestAccepted := hLatest } :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.AcceptedTerminalWithPriorVerifier
        ctx
        (ProofCarryingVerifyPrior ctx)
        priorProof.steps
        priorProof
        priorProof.image
        nextImage
        latestProof)

/--
Proof-carrying latest-step projection to the exact non-aggregate private DEC
and stage facts.
-/
theorem nonAggregatePrivateDecStageFacts_ofProofCarryingLatestStep
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {priorProof : ProofCarryingPriorProof ctx}
    {nextImage :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProof : Unit}
    (hLatest :
      DirectParentOnlyProductionSoundness.VerifyLatestStep
        ctx.toProductionContext
        priorProof.steps
        priorProof
        priorProof.image
        nextImage
        latestProof) :
    CertifiedTerminalNonAggregatePrivateDecStageFacts
      ctx
      priorProof.steps
      priorProof.image
      nextImage
      nextImage :=
  nonAggregatePrivateDecStageFacts_of_certifiedTerminalEndToEnd
    (certifiedSingleTerminalEndToEnd_ofProofCarryingLatestStep
      ctx
      hLatest)

/--
Proof-carrying latest-step projection to the Section 7.1 owner-target audit.
-/
theorem section71StageTargetAuditTrail_ofProofCarryingLatestStep
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    {priorProof : ProofCarryingPriorProof ctx}
    {nextImage :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProof : Unit}
    (hLatest :
      DirectParentOnlyProductionSoundness.VerifyLatestStep
        ctx.toProductionContext
        priorProof.steps
        priorProof
        priorProof.image
        nextImage
        latestProof) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.TerminalSection71StageTargetAuditTrail
      ctx
      priorProof.steps
      priorProof.image
      nextImage
      nextImage :=
  section71StageTargetAuditTrail_of_certifiedTerminalEndToEnd
    (certifiedSingleTerminalEndToEnd_ofProofCarryingLatestStep
      ctx
      hLatest)

/--
Certified-verifier single-terminal end-to-end theorem.

This is the packaged production call path: once a concrete compressed prior
verifier is certified by a fixed authority-opening certificate, one accepted
terminal proof returns the full end-to-end package directly.
-/
theorem certifiedSingleTerminalEndToEnd_ofCertifiedPriorVerifier
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier :
      DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
        (PriorProof := PriorProof)
        ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProof : Unit}
    (hAccepted :
      DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.AcceptedTerminal
        verifier
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof) :
    CertifiedTerminalEndToEnd
      ctx
      verifier.opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  certifiedSingleTerminalEndToEnd_ofPriorVerifierAuthorityOpening
    ctx
    verifier.opening
    hAccepted

/--
Certified-verifier prior-plus-latest form of the single-terminal theorem.

The caller supplies the concrete verifier acceptance for the prior folded `F'`
proof and the accepted latest Construction-2 step. Lean constructs the terminal
acceptance record internally and returns the same final end-to-end package.
-/
theorem certifiedSingleTerminalEndToEnd_ofCertifiedPriorVerifierLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier :
      DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
        (PriorProof := PriorProof)
        ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      verifier.verify priorSteps priorProof priorImage)
    (hLatest :
      DirectParentOnlyProductionSoundness.VerifyLatestStep
        ctx.toProductionContext
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof) :
    CertifiedTerminalEndToEnd
      ctx
      verifier.opening
      priorSteps
      priorSteps
      priorProof
      priorImage
      priorImage
      nextImage
      nextImage :=
  certifiedSingleTerminalEndToEnd_ofCertifiedPriorVerifier
    verifier
    ({ priorAccepted := hPrior
       latestAccepted := hLatest } :
      DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier.AcceptedTerminal
        verifier
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)

/--
Packaged certified-verifier latest-step projection to the exact non-aggregate
private DEC and stage facts.

This is the shortest compressed-verifier audit call path: the implementation
passes the packaged `CertifiedPriorVerifier`, concrete prior acceptance, and
latest-step acceptance; Lean returns the pointwise private DEC facts and exact
stage computation facts directly.
-/
theorem nonAggregatePrivateDecStageFacts_ofCertifiedPriorVerifierLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier :
      DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
        (PriorProof := PriorProof)
        ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      verifier.verify priorSteps priorProof priorImage)
    (hLatest :
      DirectParentOnlyProductionSoundness.VerifyLatestStep
        ctx.toProductionContext
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof) :
    CertifiedTerminalNonAggregatePrivateDecStageFacts
      ctx
      priorSteps
      priorImage
      nextImage
      nextImage :=
  nonAggregatePrivateDecStageFacts_of_certifiedTerminalEndToEnd
    (certifiedSingleTerminalEndToEnd_ofCertifiedPriorVerifierLatestStep
      verifier
      hPrior
      hLatest)

/--
Packaged certified-verifier latest-step projection to the Section 7.1
owner-target audit.
-/
theorem section71StageTargetAuditTrail_ofCertifiedPriorVerifierLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier :
      DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier.CertifiedPriorVerifier
        (PriorProof := PriorProof)
        ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {priorImage nextImage :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProof : Unit}
    (hPrior :
      verifier.verify priorSteps priorProof priorImage)
    (hLatest :
      DirectParentOnlyProductionSoundness.VerifyLatestStep
        ctx.toProductionContext
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof) :
    DirectParentOnlyProductionSuperNeoReuse.ProductionContext.TerminalSection71StageTargetAuditTrail
      ctx
      priorSteps
      priorImage
      nextImage
      nextImage :=
  section71StageTargetAuditTrail_of_certifiedTerminalEndToEnd
    (certifiedSingleTerminalEndToEnd_ofCertifiedPriorVerifierLatestStep
      verifier
      hPrior
      hLatest)

end DirectParentOnlyProductionSuperNeoReuseEndToEnd

end DirectCcsFPrime
