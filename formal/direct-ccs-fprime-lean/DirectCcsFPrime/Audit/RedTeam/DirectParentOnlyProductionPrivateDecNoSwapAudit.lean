import DirectCcsFPrime.ProofSystem.Production.Security.DirectParentOnlyProductionSuperNeoReuseEndToEnd

/-!
Production projection for private DEC no-swap audits.

This module owns the small bridge from the final non-aggregate terminal facts
to the certificate-level `PrivateDecNoSwapAudit`. It keeps the large
end-to-end theorem file focused on constructing terminal evidence, while this
file exposes the concrete anti-substitution equalities for production callers.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionPrivateDecNoSwapAudit

/-- Section 7.1-backed production context. -/
abbrev ProductionContext :=
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.ProductionContext

/-- Non-aggregate private DEC/stage facts from the final terminal package. -/
abbrev NonAggregateFacts
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params) :=
  DirectParentOnlyProductionSuperNeoReuseEndToEnd.CertifiedTerminalNonAggregatePrivateDecStageFacts
    ctx

/--
Pointwise private DEC requirements for the production parent-only context.
-/
abbrev PointwiseRequirements
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : ProductionContext Digest Boundary n params)
    (source : DigestParentBinding.Source Digest)
    (inputs : DecDigitUniqueness.ColumnDigits n) : Prop :=
  ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
    (n := n)
    (hashEncoded := ctx.parentHash.hashEncoded)
    (params := params)
    (ce := ctx.data.ce)
    (StatementEncodes :=
      ParentOpeningAuthorization.StatementEncodesByCommitment
        ctx.commitmentOfParent)
    source
    inputs

/--
Certificate-level private DEC no-swap audit projected from final terminal
facts.

The alternate table is not merely compared by an aggregate checksum. It must
satisfy the full pointwise private DEC requirements for the same parent source;
then the returned audit records equality of parent residues, private digit
tables, CE witness-derived digit tables, next-`Pi_CCS` wires, and requested
input tables.
-/
theorem auditOfFacts
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {priorSteps : Nat}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hFacts :
      NonAggregateFacts
        ctx
        priorSteps
        priorImage
        nextImage
        altNext)
    {otherInputs : DecDigitUniqueness.ColumnDigits n}
    (hOther :
      PointwiseRequirements
        ctx
        priorImage.accumulator.parentSource
        otherInputs) :
    ∃
      (priorInputs : DecDigitUniqueness.ColumnDigits n)
      (auditedCert :
        ParentOnlyAccumulatorStep.PointwisePrivateDecCertificate
          (n := n)
          (hashEncoded := ctx.parentHash.hashEncoded)
          (params := params)
          (ce := ctx.data.ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              ctx.commitmentOfParent)
          priorImage.accumulator.parentSource
          priorInputs)
      (otherCert :
        ParentOnlyAccumulatorStep.PointwisePrivateDecCertificate
          (n := n)
          (hashEncoded := ctx.parentHash.hashEncoded)
          (params := params)
          (ce := ctx.data.ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              ctx.commitmentOfParent)
          priorImage.accumulator.parentSource
          otherInputs),
        ParentOnlyAccumulatorStep.PrivateDecNoSwapAudit
          auditedCert
          otherCert := by
  rcases hFacts with ⟨cert⟩
  rcases hOther with ⟨otherCert⟩
  refine
    ⟨cert.priorInputs,
      cert.privateDec,
      otherCert,
      ?_⟩
  exact
    ParentOnlyAccumulatorStep.privateDecNoSwapAudit_of_certificates
      (Poseidon2ParentCEBHash.encodedParentCEBDigestBinding
        ctx.parentHash)
      ParentOpeningAuthorization.statementEncodesByCommitment_functional
      (AjtaiResidueBinding.noAjtaiBindingCollision_of_msis
        ctx.msisReduction
        ctx.msisHardness)
      (AjtaiResidueBinding.ceOpeningAdapter_of_assignmentOpeningAdapter
        (AjtaiResidueBinding.assignmentOpeningAdapter_of_ajtaiBackedCommitMap
          ctx.data.ajtaiBackedCommitMap))
      cert.privateDec
      otherCert

/--
Certificate-level private DEC no-swap audit projected directly from the final
end-to-end package.
-/
theorem auditOfEndToEnd
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
      DirectParentOnlyProductionSuperNeoReuseEndToEnd.CertifiedTerminalEndToEnd
        ctx
        opening
        priorStepsA
        priorStepsB
        priorProof
        priorImageA
        priorImageB
        nextImageA
        nextImageB)
    {otherInputs : DecDigitUniqueness.ColumnDigits n}
    (hOther :
      PointwiseRequirements
        ctx
        priorImageA.accumulator.parentSource
        otherInputs) :
    ∃
      (priorInputs : DecDigitUniqueness.ColumnDigits n)
      (auditedCert :
        ParentOnlyAccumulatorStep.PointwisePrivateDecCertificate
          (n := n)
          (hashEncoded := ctx.parentHash.hashEncoded)
          (params := params)
          (ce := ctx.data.ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              ctx.commitmentOfParent)
          priorImageA.accumulator.parentSource
          priorInputs)
      (otherCert :
        ParentOnlyAccumulatorStep.PointwisePrivateDecCertificate
          (n := n)
          (hashEncoded := ctx.parentHash.hashEncoded)
          (params := params)
          (ce := ctx.data.ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              ctx.commitmentOfParent)
          priorImageA.accumulator.parentSource
          otherInputs),
        ParentOnlyAccumulatorStep.PrivateDecNoSwapAudit
          auditedCert
          otherCert :=
  auditOfFacts
    (DirectParentOnlyProductionSuperNeoReuseEndToEnd.nonAggregatePrivateDecStageFacts_of_certifiedTerminalEndToEnd
      hEnd)
    hOther

end DirectParentOnlyProductionPrivateDecNoSwapAudit

end DirectCcsFPrime
