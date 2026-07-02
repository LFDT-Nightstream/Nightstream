import DirectCcsFPrime.ProofSystem.Production.Security.SuperNeoReuseCertifiedVerifierTerminal

/-!
Same-proof replay and no-swap package for the Section 7.1-backed certified
prior verifier.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier

namespace CertifiedPriorVerifier

/--
Same certified verifier and same opaque proof imply the same replay endpoint.

The conclusion includes equality of the two prior pairs, equality of the two
terminal images, and the audited public endpoint with pointwise private-child
and contextual stage evidence.
-/
theorem sameProofReplayEndpoint
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier : CertifiedPriorVerifier (PriorProof := PriorProof) ctx)
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      AcceptedTerminal
        verifier
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      AcceptedTerminal
        verifier
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
  DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.same_proof_replay_endpoint_of_priorVerifierAuthorityOpening
    ctx
    verifier.opening
    hA
    hB

/-- Pointwise private-child replay binding for a certified verifier. -/
theorem pointwiseChildReplayBinding
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier : CertifiedPriorVerifier (PriorProof := PriorProof) ctx)
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      AcceptedTerminal
        verifier
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      AcceptedTerminal
        verifier
        priorStepsB
        priorProof
        priorImageB
        nextImageB
        latestProofB) :
    DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.PointwiseChildReplayBinding
      ctx
      priorStepsA
      priorImageA
      nextImageA
      nextImageB :=
  DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.pointwise_child_replay_binding_of_priorVerifierAuthorityOpening
    ctx
    verifier.opening
    hA
    hB

/-- Flattened computed-stage replay evidence for a certified verifier. -/
theorem computedStageReplayEvidence
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier : CertifiedPriorVerifier (PriorProof := PriorProof) ctx)
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      AcceptedTerminal
        verifier
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      AcceptedTerminal
        verifier
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
  DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.computed_stage_replay_evidence_of_same_proof_replay_endpoint
    (sameProofReplayEndpoint verifier hA hB)

/--
Computed replay evidence exposes the explicit no-swap child-table projection.
-/
theorem explicitReplayNoSwapEvidence_of_computedStageReplayEvidence
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    {priorSteps : Nat}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hEvidence :
      DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.ComputedStageReplayEvidence
        ctx
        priorSteps
        priorImage
        nextImage
        altNext) :
    ExplicitReplayNoSwapEvidence
      ctx
      priorSteps
      priorImage
      nextImage
      altNext := by
  rcases hEvidence with
    ⟨priorInputs,
      hPointwise,
      hAudit,
      hNext,
      hAlt,
      _hNextSource,
      _hAltSource,
      hUnique,
      _hOutStep,
      _hPiCCS,
      _hPiRLC,
      _hPiDEC⟩
  exact
    ⟨priorInputs,
      hPointwise,
      hAudit,
      hNext,
      hAlt,
      hUnique⟩

/--
Certified-verifier no-swap theorem for replayed private DEC children.

Any alternate pointwise-valid child table for the replayed parent source is the
same table used by both accepted terminal images under the fixed opener.
-/
theorem pointwiseChildTableUnique
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier : CertifiedPriorVerifier (PriorProof := PriorProof) ctx)
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      AcceptedTerminal
        verifier
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      AcceptedTerminal
        verifier
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
  DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.pointwise_child_table_unique_of_priorVerifierAuthorityOpening
    ctx
    verifier.opening
    hA
    hB
    hOther

/--
Raw-verifier form of same-proof replay no-swap uniqueness for private DEC
children.

The theorem is quantified over an arbitrary alternate child table. Equality is
available only after that table satisfies the full pointwise private DEC
requirements for the replayed parent source.
-/
theorem pointwiseChildTableUnique_ofAcceptedOpens
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
        nextImageB.accumulator.parentSource := by
  let verifier := ofAcceptedOpens ctx verify opener acceptedOpens
  simpa [verifier, ofAcceptedOpens, AcceptedTerminal]
    using pointwiseChildTableUnique verifier hA hB hOther

/--
Single-call certified replay audit package.

This is the intended final theorem-facing shape for the concrete compressed
prior verifier: the same proof opens to folded authority for the first prior
pair, cannot retarget the prior pair or terminal image, and exposes computed
stage evidence for the unique pointwise-valid private child table.
-/
theorem replayAuditPackage
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : ProductionContext Digest Boundary n params}
    (verifier : CertifiedPriorVerifier (PriorProof := PriorProof) ctx)
    {priorStepsA priorStepsB : Nat}
    {priorProof : PriorProof}
    {priorImageA priorImageB nextImageA nextImageB :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    {latestProofA latestProofB : Unit}
    (hA :
      AcceptedTerminal
        verifier
        priorStepsA
        priorProof
        priorImageA
        nextImageA
        latestProofA)
    (hB :
      AcceptedTerminal
        verifier
        priorStepsB
        priorProof
        priorImageB
        nextImageB
        latestProofB) :
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
        nextImageB :=
  DirectParentOnlyProductionSuperNeoReuseReplayEndpoint.concrete_verifier_replay_audit_package_of_priorVerifierAuthorityOpening
    ctx
    verifier.opening
    hA
    hB

/--
Raw-verifier form of the certified replay audit package.

The concrete verifier supplies only its predicate, fixed opener, and
accepted-opens theorem. Same-proof replay stability and computed-stage evidence
then follow through the certified verifier object built from those obligations.
-/
theorem replayAuditPackage_ofAcceptedOpens
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
        nextImageB := by
  let verifier := ofAcceptedOpens ctx verify opener acceptedOpens
  simpa [verifier, ofAcceptedOpens, AcceptedTerminal]
    using replayAuditPackage verifier hA hB

/--
Raw-verifier replay audit package with the no-swap child-table projection made
explicit.

This is the implementation-facing theorem for replay audits: the same opaque
proof opens to prior folded authority, replay fixes the prior and terminal
public images, computed-stage evidence is available, and the audited private
DEC child table is unique against every other fully pointwise-valid table for
the same parent source.
-/
theorem replayAuditPackageWithExplicitNoSwap_ofAcceptedOpens
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
      ExplicitReplayNoSwapEvidence
        ctx
        priorStepsA
        priorImageA
        nextImageA
        nextImageB := by
  rcases
      replayAuditPackage_ofAcceptedOpens
        ctx
        verify
        opener
        acceptedOpens
        hA
        hB with
    ⟨hOpen,
      hAuthority,
      hReplay,
      hEvidence⟩
  exact
    ⟨hOpen,
      hAuthority,
      hReplay,
      hEvidence,
      explicitReplayNoSwapEvidence_of_computedStageReplayEvidence hEvidence⟩

/-- Preferred short name for same-proof replay. -/
abbrev sameProofReplay :=
  @sameProofReplayEndpoint

/-- Preferred short name for pointwise child replay binding. -/
abbrev childReplayBinding :=
  @pointwiseChildReplayBinding

/-- Preferred short name for computed replay stage evidence. -/
abbrev replayStageEvidence :=
  @computedStageReplayEvidence

/-- Preferred short name for extracting no-swap evidence from replay evidence. -/
abbrev noSwapFromReplayEvidence :=
  @explicitReplayNoSwapEvidence_of_computedStageReplayEvidence

/-- Preferred short name for replayed private-child no-swap. -/
abbrev replayChildNoSwap :=
  @pointwiseChildTableUnique

/-- Preferred short name for raw-verifier replayed private-child no-swap. -/
abbrev replayChildNoSwapOfAcceptedOpens :=
  @pointwiseChildTableUnique_ofAcceptedOpens

/-- Preferred short name for the same-proof replay audit package. -/
abbrev replayAudit :=
  @replayAuditPackage

/-- Preferred short name for the raw-verifier replay audit package. -/
abbrev replayAuditOfAcceptedOpens :=
  @replayAuditPackage_ofAcceptedOpens

/-- Preferred short name for raw-verifier replay audit with no-swap evidence. -/
abbrev replayAuditWithNoSwapOfAcceptedOpens :=
  @replayAuditPackageWithExplicitNoSwap_ofAcceptedOpens

end CertifiedPriorVerifier

end DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier

end DirectCcsFPrime
