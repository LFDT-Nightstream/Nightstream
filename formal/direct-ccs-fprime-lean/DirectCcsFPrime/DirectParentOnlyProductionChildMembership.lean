import DirectCcsFPrime.DirectParentOnlyProductionSoundness

/-!
Production child-membership surface for the parent-only direct CCS F' path.

This module keeps the production soundness file compact while exposing the
fixed-CE membership facts for the private DEC children extracted by the
optimized terminal theorem.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionChildMembership

/--
Named production audit certificate for the private child table extracted by
the parent-only terminal theorem.

This exposes the actual non-aggregate obligations hidden behind the optimized
parent handle: a concrete private `Pi_DEC` acceptance over a real SuperNeo child
bundle, fixed CE/Ajtai parameters, binary child digits, exact length 14,
per-column Goldilocks recomposition to the opened parent residues, witness-table
identity, and wire identity into the next `Pi_CCS` inputs.
-/
structure PointwiseChildAuditCertificate
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (source : DigestParentBinding.Source Digest)
    (nextInputs : DecDigitUniqueness.ColumnDigits n) where
  privateDec :
    ParentOnlyAccumulatorStep.PointwisePrivateDecCertificate
      (n := n)
      (hashEncoded := ctx.parentHash.hashEncoded)
      (params := params)
      (ce := ctx.data.ce)
      (StatementEncodes :=
        ParentOpeningAuthorization.StatementEncodesByCommitment
          ctx.commitmentOfParent)
      source
      nextInputs
  fixedMembership :
    ParentOnlyAccumulatorStep.FixedCEChildMembership
      (n := n)
      params
      ctx.data.ce
      nextInputs

/--
Audit trail for the private child table extracted by the parent-only terminal
theorem.
-/
def PointwiseChildAuditTrail
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (source : DigestParentBinding.Source Digest)
    (nextInputs : DecDigitUniqueness.ColumnDigits n) : Prop :=
  Nonempty (PointwiseChildAuditCertificate ctx source nextInputs)

/--
Full terminal child-audit conclusion for one accepted parent-only latest step
and one alternate latest step.
-/
def TerminalChildAuditTrail
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
    PointwiseChildAuditTrail
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
          otherInputs = priorInputs)

/--
Pointwise private-DEC requirements expose the production child audit trail.
-/
theorem pointwise_child_audit_trail_of_pointwise_private_dec_requirements
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    {source : DigestParentBinding.Source Digest}
    {nextInputs : DecDigitUniqueness.ColumnDigits n}
    (hPointwise :
      ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := ctx.parentHash.hashEncoded)
        (params := params)
        (ce := ctx.data.ce)
        (StatementEncodes :=
          ParentOpeningAuthorization.StatementEncodesByCommitment
            ctx.commitmentOfParent)
        source
        nextInputs) :
    PointwiseChildAuditTrail ctx source nextInputs := by
  have hFixed :
      ParentOnlyAccumulatorStep.FixedCEChildMembership
        (n := n)
        params
        ctx.data.ce
        nextInputs :=
    ParentOnlyAccumulatorStep.fixedCEChildMembership_of_pointwise_private_dec_requirements
      hPointwise
  rcases hPointwise with ⟨cert⟩
  exact ⟨{
    privateDec := cert
    fixedMembership := hFixed
  }⟩

/--
Pointwise child audit exposes the accepted private `Pi_DEC` proof directly.

This is the proof-carrying fact behind the optimized parent-only handle: the
child table is not just a set of values with a matching aggregate summary, it
is accepted by the private DEC verifier against opened parent residues for the
same parent source.
-/
theorem privatePiDECAccepts_of_pointwise_child_audit_trail
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params}
    {source : DigestParentBinding.Source Digest}
    {nextInputs : DecDigitUniqueness.ColumnDigits n}
    (hAudit : PointwiseChildAuditTrail ctx source nextInputs) :
    ∃
      (parentResidues : Fin n → Nat)
      (bundle : SuperNeoBridge.AjtaiCEChildBundle 14 n)
      (proof : Unit),
        PrivatePiDecSoundness.AcceptedPrivatePiDec
          (ParentOpeningAuthorization.BindsOpenedParentCEBResiduesFor
            (n := n)
            ctx.parentHash.hashEncoded
            ctx.data.ce
            (ParentOpeningAuthorization.StatementEncodesByCommitment
              ctx.commitmentOfParent))
          (CanonicalPrivatePiDecVerifier.Verify
            (n := n)
            (Source := DigestParentBinding.Source Digest))
          source
          parentResidues
          bundle
          proof := by
  rcases hAudit with ⟨cert⟩
  exact
    ⟨cert.privateDec.parentResidues,
      cert.privateDec.bundle,
      cert.privateDec.proof,
      cert.privateDec.accepted⟩

/--
Pointwise child audit exposes fixed-CE membership for the child table wired
into the next `Pi_CCS`.
-/
theorem fixedCEChildMembership_of_pointwise_child_audit_trail
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params}
    {source : DigestParentBinding.Source Digest}
    {nextInputs : DecDigitUniqueness.ColumnDigits n}
    (hAudit : PointwiseChildAuditTrail ctx source nextInputs) :
    ParentOnlyAccumulatorStep.FixedCEChildMembership
      (n := n)
      params
      ctx.data.ce
      nextInputs := by
  rcases hAudit with ⟨cert⟩
  exact cert.fixedMembership

/--
Pointwise child audit exposes the exact non-aggregate DEC table facts.

The conclusion keeps the child bundle explicit and includes bitness, exact
length, and per-column Goldilocks recomposition. This is the theorem-facing
shape that rules out validating only a total norm, checksum, or other aggregate
summary.
-/
theorem nonaggregate_dec_facts_of_pointwise_child_audit_trail
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params}
    {source : DigestParentBinding.Source Digest}
    {nextInputs : DecDigitUniqueness.ColumnDigits n}
    (hAudit : PointwiseChildAuditTrail ctx source nextInputs) :
    ∃
      (parentResidues : Fin n → Nat)
      (bundle : SuperNeoBridge.AjtaiCEChildBundle 14 n)
      (proof : Unit),
        PrivatePiDecSoundness.AcceptedPrivatePiDec
            (ParentOpeningAuthorization.BindsOpenedParentCEBResiduesFor
              (n := n)
              ctx.parentHash.hashEncoded
              ctx.data.ce
              (ParentOpeningAuthorization.StatementEncodesByCommitment
                ctx.commitmentOfParent))
            (CanonicalPrivatePiDecVerifier.Verify
              (n := n)
              (Source := DigestParentBinding.Source Digest))
            source
            parentResidues
            bundle
            proof ∧
          bundle.ce = ctx.data.ce ∧
          bundle.ajtaiParams = params ∧
          DecDigitUniqueness.binaryColumnDigits bundle.digitTable ∧
          BinaryChildTableAuthorization.fixedColumnLength 14 bundle.digitTable ∧
          (∀ j,
            DecDigitUniqueness.recomposeNatDigits (bundle.digitTable j) %
                SuperNeo.Goldilocks.q =
              parentResidues j % SuperNeo.Goldilocks.q) ∧
          bundle.digitTable =
            SuperNeoBridge.childWitnessDigitTable
              (k := 14)
              (n := n)
              bundle.witness ∧
          bundle.nextPiCCSInputs = bundle.digitTable ∧
          nextInputs = bundle.nextPiCCSInputs ∧
          (∀ i,
            SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
              ctx.data.ce
              (bundle.statement i)
              (bundle.witness i)) ∧
            (∀ i,
              SuperNeo.ProofSystem.opensTo
                params
                (bundle.statement i).commitment
                (bundle.opening i)) := by
  rcases hAudit with ⟨cert⟩
  refine
    ⟨cert.privateDec.parentResidues,
      cert.privateDec.bundle,
      cert.privateDec.proof,
      cert.privateDec.accepted,
      cert.privateDec.fixedCE,
      cert.privateDec.fixedAjtaiParams,
      cert.privateDec.binaryDigits,
      cert.privateDec.length14,
      cert.privateDec.recomposesToParent,
      cert.privateDec.witnessTable,
      cert.privateDec.wireIdentity,
      cert.privateDec.nextInputIdentity,
      ?_,
      ?_⟩
  · intro i
    simpa [cert.privateDec.fixedCE] using cert.privateDec.bundle.ceHolds i
  · intro i
    simpa [cert.privateDec.fixedAjtaiParams] using cert.privateDec.bundle.ajtaiOpens i

/--
Pointwise child audit proves that the next `Pi_CCS` input wires are exactly the
CE witness-derived child digit table.
-/
theorem nextPiCCSInputs_eq_childWitnessDigitTable_of_pointwise_child_audit_trail
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params}
    {source : DigestParentBinding.Source Digest}
    {nextInputs : DecDigitUniqueness.ColumnDigits n}
    (hAudit : PointwiseChildAuditTrail ctx source nextInputs) :
    ∃ bundle : SuperNeoBridge.AjtaiCEChildBundle 14 n,
      bundle.ce = ctx.data.ce ∧
        bundle.ajtaiParams = params ∧
        nextInputs =
        SuperNeoBridge.childWitnessDigitTable
            (k := 14)
            (n := n)
            bundle.witness := by
  rcases hAudit with ⟨cert⟩
  refine
    ⟨cert.privateDec.bundle,
      cert.privateDec.fixedCE,
      cert.privateDec.fixedAjtaiParams,
      ?_⟩
  calc
    nextInputs = cert.privateDec.bundle.nextPiCCSInputs :=
      cert.privateDec.nextInputIdentity
    _ = cert.privateDec.bundle.digitTable := cert.privateDec.wireIdentity
    _ =
        SuperNeoBridge.childWitnessDigitTable
          (k := 14)
          (n := n)
          cert.privateDec.bundle.witness := cert.privateDec.witnessTable

/--
Terminal child audit exposes the single audited child table and its
pointwise/private-DEC facts.
-/
theorem pointwise_child_audit_trail_of_terminal_child_audit_trail
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params}
    {priorSteps : Nat}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hAudit :
      TerminalChildAuditTrail
        ctx
        priorSteps
        priorImage
        nextImage
        altNext) :
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
      PointwiseChildAuditTrail
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
            otherInputs = priorInputs) :=
  hAudit

/--
Production terminal soundness with fixed-CE membership for the unique private
child table.

The optimized terminal theorem extracts one pointwise-authorized private child
table and proves it is unique for the parent source. This theorem additionally
exposes that the same table comes from children satisfying the context-fixed CE
relation and Ajtai opening parameters.
-/
theorem terminal_soundness_with_unique_children_and_fixed_ce_membership
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (verifier : DirectParentOnlyProductionSoundness.SoundPriorVerifier (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
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
              otherInputs = priorInputs) := by
  rcases
      DirectParentOnlyProductionSoundness.terminal_soundness_with_unique_pointwise_children
        ctx
        verifier
        hAccepted
        hAlt with
    ⟨hSound,
      priorInputs,
      hPointwise,
      hNextComputed,
      hAltComputed,
      hUnique⟩
  exact
    ⟨hSound,
      priorInputs,
      hPointwise,
      ParentOnlyAccumulatorStep.fixedCEChildMembership_of_pointwise_private_dec_requirements
        hPointwise,
      hNextComputed,
      hAltComputed,
      hUnique⟩

/--
Production terminal soundness with the explicit pointwise child audit trail.

This theorem is the compact audit-facing surface: the latest public image is
sound, both latest images are computed from the same private child table, and
that table carries the concrete per-child DEC/CE/wire facts instead of an
aggregate checksum-style obligation.
-/
theorem terminal_soundness_with_pointwise_child_audit_trail
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (verifier : DirectParentOnlyProductionSoundness.SoundPriorVerifier (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
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
    DirectParentOnlyProductionSoundness.TerminalSoundness
        ctx
        priorSteps
        priorImage
        nextImage
        altNext ∧
      TerminalChildAuditTrail
        ctx
        priorSteps
        priorImage
        nextImage
        altNext := by
  rcases
      terminal_soundness_with_unique_children_and_fixed_ce_membership
        ctx
        verifier
        hAccepted
        hAlt with
    ⟨hSound,
      priorInputs,
      hPointwise,
      _hFixed,
      hNextComputed,
      hAltComputed,
      hUnique⟩
  exact
    ⟨hSound,
      priorInputs,
      hPointwise,
      pointwise_child_audit_trail_of_pointwise_private_dec_requirements
        ctx
        hPointwise,
      hNextComputed,
      hAltComputed,
      hUnique⟩

/--
Proof-carrying folded-prior entry point with fixed-CE membership for the
unique private child table.

This is the theorem-level reference path before plugging in a compressed prior
verifier: the prior proof itself carries folded `F'` reachability, while the
latest parent-only step still exposes the same pointwise DEC authorization,
fixed child CE membership, computed latest images, and uniqueness of the
private child table.
-/
theorem terminal_soundness_with_unique_children_and_fixed_ce_membership_of_proof_carrying_prior_authority
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    {priorSteps : Nat}
    {priorProof : DirectParentOnlyProductionSoundness.ProofCarryingPriorProof ctx}
    {latestProof : Unit}
    {priorImage nextImage altNext : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hAccepted :
      DirectParentOnlyProductionSoundness.AcceptedProofCarryingTerminal
        ctx
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
  terminal_soundness_with_unique_children_and_fixed_ce_membership
    ctx
    (DirectParentOnlyProductionSoundness.proofCarryingPriorVerifier ctx)
    hAccepted
    hAlt

/--
Proof-carrying folded-prior entry point with the explicit pointwise child audit
trail.
-/
theorem terminal_soundness_with_pointwise_child_audit_trail_of_proof_carrying_prior_authority
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    {priorSteps : Nat}
    {priorProof : DirectParentOnlyProductionSoundness.ProofCarryingPriorProof ctx}
    {latestProof : Unit}
    {priorImage nextImage altNext : DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hAccepted :
      DirectParentOnlyProductionSoundness.AcceptedProofCarryingTerminal
        ctx
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
      TerminalChildAuditTrail
        ctx
        priorSteps
        priorImage
        nextImage
        altNext :=
  terminal_soundness_with_pointwise_child_audit_trail
    ctx
    (DirectParentOnlyProductionSoundness.proofCarryingPriorVerifier ctx)
    hAccepted
    hAlt

/--
Raw compressed-prior verifier entry point with fixed-CE membership for the
unique private child table.
-/
theorem terminal_soundness_with_unique_children_and_fixed_ce_membership_of_prior_verifier_opening
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
      DirectParentOnlyProductionSoundness.OpensToProofCarryingPriorAuthority ctx VerifyPrior)
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
              otherInputs = priorInputs) := by
  let verifier :
      DirectParentOnlyProductionSoundness.SoundPriorVerifier (PriorProof := PriorProof) ctx :=
    DirectParentOnlyProductionSoundness.soundPriorVerifier_of_opens_to_proof_carrying_prior_authority
      ctx
      VerifyPrior
      hOpens
  have hAcceptedSound :
      DirectParentOnlyProductionSoundness.AcceptedTerminal
        ctx
        verifier
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof := by
    constructor
    · change VerifyPrior priorSteps priorProof priorImage
      exact hAccepted.priorAccepted
    · exact hAccepted.latestAccepted
  exact
    terminal_soundness_with_unique_children_and_fixed_ce_membership
      ctx
      verifier
      hAcceptedSound
      hAlt

/--
Raw compressed-prior verifier entry point with the explicit pointwise child
audit trail.
-/
theorem terminal_soundness_with_pointwise_child_audit_trail_of_prior_verifier_opening
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
      DirectParentOnlyProductionSoundness.OpensToProofCarryingPriorAuthority ctx VerifyPrior)
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
      TerminalChildAuditTrail
        ctx
        priorSteps
        priorImage
        nextImage
        altNext := by
  let verifier :
      DirectParentOnlyProductionSoundness.SoundPriorVerifier (PriorProof := PriorProof) ctx :=
    DirectParentOnlyProductionSoundness.soundPriorVerifier_of_opens_to_proof_carrying_prior_authority
      ctx
      VerifyPrior
      hOpens
  have hAcceptedSound :
      DirectParentOnlyProductionSoundness.AcceptedTerminal
        ctx
        verifier
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof := by
    constructor
    · change VerifyPrior priorSteps priorProof priorImage
      exact hAccepted.priorAccepted
    · exact hAccepted.latestAccepted
  exact
    terminal_soundness_with_pointwise_child_audit_trail
      ctx
      verifier
      hAcceptedSound
      hAlt

end DirectParentOnlyProductionChildMembership

end DirectCcsFPrime
