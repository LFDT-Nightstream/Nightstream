import DirectCcsFPrime.CompressedFPrimeAuthority
import DirectCcsFPrime.DirectConcreteInstantiation
import DirectCcsFPrime.DirectProgramStep
import DirectCcsFPrime.ParentOnlyAccumulatorStep

/-!
Terminal direct CCS F' soundness for the parent-only public handle.

This module closes the theorem slice for the Spartan-facing optimization where
the direct public accumulator carries only the parent `CE(B)` source. The
post-DEC `CE(b)^k` children are private advice, but every accepted latest step
must authorize them from the prior parent source and use that exact child table
in the next `Pi_CCS -> Pi_RLC` parent-source computation.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyTerminalSoundness

/-- Parent-only accumulator handle used by the optimized public image. -/
abbrev AccHandle (Digest : Type) :=
  ParentOnlyAccumulatorStep.AccumulatorHandle
    (DigestParentBinding.Source Digest)

/-- Parent-only accumulator update for the optimized terminal direct step. -/
def AccumulatorStep
    {Digest PiCCSOut : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (PiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle
          (DigestParentBinding.Source Digest)
          n →
        PiCCSOut →
          Prop)
    (PiRLC :
      Nat →
        PiCCSOut →
        DigestParentBinding.Source Digest →
          Prop)
    (hashEncoded : List Nat → Digest)
    (ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment)
    (StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment) :
    Nat → AccHandle Digest → AccHandle Digest → Prop :=
  ParentOnlyAccumulatorStep.Step
    (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
      (n := n)
      (hashEncoded := hashEncoded)
      (params := params)
      (ce := ce)
      (StatementEncodes := StatementEncodes))
    (ParentOnlyAccumulatorStep.ParentSourceFromPiStages
      (n := n)
      PiCCS
      PiRLC)

/-- Canonical parent-only direct CCS Construction-2 transition. -/
def Transition
    {Digest Boundary PiCCSOut : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (BoundaryStep : Nat → Boundary → Boundary → Prop)
    (PiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle
          (DigestParentBinding.Source Digest)
          n →
        PiCCSOut →
          Prop)
    (PiRLC :
      Nat →
        PiCCSOut →
        DigestParentBinding.Source Digest →
          Prop)
    (hashEncoded : List Nat → Digest)
    (ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment)
    (StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment) :
    Nat →
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (AccHandle Digest) →
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (AccHandle Digest) →
        Prop :=
  Construction2DirectFPrime.Transition
    BoundaryStep
    (AccumulatorStep
      (params := params)
      PiCCS
      PiRLC
      hashEncoded
      ce
      StatementEncodes)

/--
For deterministic boundary updates, if the parent-only accumulator source is
functional, then the whole latest public image is functional.
-/
theorem latest_publicImage_functional_of_parentSource
    {Digest Boundary Source : Type}
    {computeBoundary : Nat → Boundary → Boundary}
    {AccumulatorStep :
      Nat →
        ParentOnlyAccumulatorStep.AccumulatorHandle Source →
        ParentOnlyAccumulatorStep.AccumulatorHandle Source →
          Prop}
    {i : Nat}
    {prior nextA nextB :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (ParentOnlyAccumulatorStep.AccumulatorHandle Source)}
    (hParentSource :
      nextA.accumulator.parentSource =
        nextB.accumulator.parentSource)
    (hA :
      Construction2DirectFPrime.Transition
        (DirectProgramStep.ComputedBoundaryStep computeBoundary)
        AccumulatorStep
        i
        prior
        nextA)
    (hB :
      Construction2DirectFPrime.Transition
        (DirectProgramStep.ComputedBoundaryStep computeBoundary)
        AccumulatorStep
        i
        prior
        nextB) :
    nextA = nextB := by
  rcases hA with
    ⟨_hPriorA, hNextA, hVkA, hInitialA, _hPriorPcA,
      hNextPcA, hBoundaryA, _hAccA⟩
  rcases hB with
    ⟨_hPriorB, hNextB, hVkB, hInitialB, _hPriorPcB,
      hNextPcB, hBoundaryB, _hAccB⟩
  cases nextA with
  | mk vkA stepA initialA boundaryA accA pcA =>
      cases nextB with
      | mk vkB stepB initialB boundaryB accB pcB =>
          cases accA with
          | mk sourceA =>
              cases accB with
              | mk sourceB =>
                  simp only
                    [Construction2DirectFPrime.PublicImage.mk.injEq,
                      ParentOnlyAccumulatorStep.AccumulatorHandle.mk.injEq]
                  exact
                    ⟨hVkA.symm.trans hVkB,
                      hNextA.trans hNextB.symm,
                      hInitialA.symm.trans hInitialB,
                      hBoundaryA.trans hBoundaryB.symm,
                      hParentSource,
                      hNextPcA.trans hNextPcB.symm⟩

/-- Proof-carrying prior authority for the parent-only direct transition. -/
abbrev Authority
    {Digest Boundary PiCCSOut : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (BoundaryStep : Nat → Boundary → Boundary → Prop)
    (PiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle
          (DigestParentBinding.Source Digest)
          n →
        PiCCSOut →
          Prop)
    (PiRLC :
      Nat →
        PiCCSOut →
        DigestParentBinding.Source Digest →
          Prop)
    (hashEncoded : List Nat → Digest)
    (ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment)
    (StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment)
    (initial :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (AccHandle Digest)) :=
  FoldedFPrimeAuthority.Authority
    (Transition
      (params := params)
      BoundaryStep
      PiCCS
      PiRLC
      hashEncoded
      ce
      StatementEncodes)
    initial

/--
An accepted parent-only latest transition exposes pointwise private-DEC
requirements for the hidden prior child table.
-/
theorem latest_step_pointwise_prior_dec_requirements
    {Digest Boundary PiCCSOut : Type}
    {n : Nat}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {PiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle
          (DigestParentBinding.Source Digest)
          n →
        PiCCSOut →
          Prop}
    {PiRLC :
      Nat →
        PiCCSOut →
        DigestParentBinding.Source Digest →
          Prop}
    {i : Nat}
    {prior next :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (AccHandle Digest)}
    (hLatest :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (AccumulatorStep
          (params := params)
          PiCCS
          PiRLC
          hashEncoded
          ce
          StatementEncodes)
        i
        prior
        next) :
    ∃ priorInputs,
      ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes := StatementEncodes)
        prior.accumulator.parentSource
        priorInputs ∧
      ParentOnlyAccumulatorStep.ParentSourceFromPiStages
        (n := n)
        PiCCS
        PiRLC
        i
        prior.accumulator
        priorInputs
        next.accumulator.parentSource := by
  rcases hLatest with
    ⟨_hPrior, _hNext, _hVk, _hInitial, _hPriorPc,
      _hNextPc, _hBoundary, hAcc⟩
  exact ParentOnlyAccumulatorStep.pointwise_prior_dec_requirements_of_step hAcc

/--
Two parent-only latest transitions from the same prior image agree on the next
parent source under the concrete private-DEC and stage-functionality
requirements.
-/
theorem latest_parentSource_functional_of_stages_and_ajtaiCEOpening
    {Digest Boundary PiCCSOut : Type}
    {n : Nat}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {PiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle
          (DigestParentBinding.Source Digest)
          n →
        PiCCSOut →
          Prop}
    {PiRLC :
      Nat →
        PiCCSOut →
        DigestParentBinding.Source Digest →
          Prop}
    {i : Nat}
    {prior nextA nextB :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (AccHandle Digest)}
    (hPiCCS : ParentSourceStep.PiCCSFunctional PiCCS)
    (hPiRLC : ParentSourceStep.PiRLCFunctional PiRLC)
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hEncoding :
      ParentOpeningAuthorization.StatementEncodingCommitmentFunctional
        StatementEncodes)
    (hNoCollision :
      AjtaiResidueBinding.NoAjtaiBindingCollision params)
    (adapter :
      AjtaiResidueBinding.CEOpeningAdapter
        n
        params
        ce)
    (hA :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (AccumulatorStep
          (params := params)
          PiCCS
          PiRLC
          hashEncoded
          ce
          StatementEncodes)
        i
        prior
        nextA)
    (hB :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (AccumulatorStep
          (params := params)
          PiCCS
          PiRLC
          hashEncoded
          ce
          StatementEncodes)
        i
        prior
        nextB) :
    nextA.accumulator.parentSource = nextB.accumulator.parentSource := by
  rcases hA with
    ⟨_hPriorA, _hNextA, _hVkA, _hInitialA, _hPriorPcA,
      _hNextPcA, _hBoundaryA, hAccA⟩
  rcases hB with
    ⟨_hPriorB, _hNextB, _hVkB, _hInitialB, _hPriorPcB,
      _hNextPcB, _hBoundaryB, hAccB⟩
  exact
    ParentOnlyAccumulatorStep.step_parentSource_functional_of_stages_and_ajtaiCEOpening
      hPiCCS
      hPiRLC
      hDigest
      hEncoding
      hNoCollision
      adapter
    hAccA
    hAccB

/--
Two accepted parent-only latest transitions from the same prior image expose a
common pointwise-authorized private-DEC child table.
-/
theorem latest_common_prior_dec_requirements_of_stages_and_ajtaiCEOpening
    {Digest Boundary PiCCSOut : Type}
    {n : Nat}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {PiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle
          (DigestParentBinding.Source Digest)
          n →
        PiCCSOut →
          Prop}
    {PiRLC :
      Nat →
        PiCCSOut →
        DigestParentBinding.Source Digest →
          Prop}
    {i : Nat}
    {prior nextA nextB :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (AccHandle Digest)}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hEncoding :
      ParentOpeningAuthorization.StatementEncodingCommitmentFunctional
        StatementEncodes)
    (hNoCollision :
      AjtaiResidueBinding.NoAjtaiBindingCollision params)
    (adapter :
      AjtaiResidueBinding.CEOpeningAdapter
        n
        params
        ce)
    (hA :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (AccumulatorStep
          (params := params)
          PiCCS
          PiRLC
          hashEncoded
          ce
          StatementEncodes)
        i
        prior
        nextA)
    (hB :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (AccumulatorStep
          (params := params)
          PiCCS
          PiRLC
          hashEncoded
          ce
          StatementEncodes)
        i
        prior
        nextB) :
    ∃ priorInputs,
      ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes := StatementEncodes)
        prior.accumulator.parentSource
        priorInputs ∧
        ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          PiCCS
          PiRLC
          i
          prior.accumulator
          priorInputs
          nextA.accumulator.parentSource ∧
        ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          PiCCS
          PiRLC
          i
          prior.accumulator
          priorInputs
          nextB.accumulator.parentSource := by
  rcases hA with
    ⟨_hPriorA, _hNextA, _hVkA, _hInitialA, _hPriorPcA,
      _hNextPcA, _hBoundaryA, hAccA⟩
  rcases hB with
    ⟨_hPriorB, _hNextB, _hVkB, _hInitialB, _hPriorPcB,
      _hNextPcB, _hBoundaryB, hAccB⟩
  exact
    ParentOnlyAccumulatorStep.pointwise_common_prior_dec_requirements_of_steps
      (ReducedAccumulatorStep.canonical_authorized_functional_of_ajtaiCEOpening
        hDigest
        hEncoding
        hNoCollision
        adapter)
      hAccA
      hAccB

/--
Implementation-shaped parent-source functionality with deterministic parent
statement encoding and MSIS-backed Ajtai binding.
-/
theorem latest_parentSource_functional_of_statementCommitment_stages_and_msis
    {Digest Boundary PiCCSOut : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data : DirectConcreteInstantiation.ConcreteCEData n params)
    {hashEncoded : List Nat → Digest}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {PiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle
          (DigestParentBinding.Source Digest)
          n →
        PiCCSOut →
          Prop}
    {PiRLC :
      Nat →
        PiCCSOut →
        DigestParentBinding.Source Digest →
          Prop}
    {i : Nat}
    {prior nextA nextB :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (AccHandle Digest)}
    (hPiCCS : ParentSourceStep.PiCCSFunctional PiCCS)
    (hPiRLC : ParentSourceStep.PiRLCFunctional PiRLC)
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hRed :
      SuperNeo.ProofSystem.MSISToAjtaiReductions params)
    (hMsis :
      SuperNeo.ProofSystem.MSISHardnessAssumption params)
    (hA :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (AccumulatorStep
          (params := params)
          PiCCS
          PiRLC
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        i
        prior
        nextA)
    (hB :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (AccumulatorStep
          (params := params)
          PiCCS
          PiRLC
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        i
        prior
        nextB) :
    nextA.accumulator.parentSource = nextB.accumulator.parentSource :=
  latest_parentSource_functional_of_stages_and_ajtaiCEOpening
    hPiCCS
    hPiRLC
    hDigest
    ParentOpeningAuthorization.statementEncodesByCommitment_functional
    (AjtaiResidueBinding.noAjtaiBindingCollision_of_msis hRed hMsis)
    (AjtaiResidueBinding.ceOpeningAdapter_of_assignmentOpeningAdapter
      (AjtaiResidueBinding.assignmentOpeningAdapter_of_ajtaiBackedCommitMap
        data.ajtaiBackedCommitMap))
    hA
    hB

/--
Implementation-shaped common private-DEC child table for two accepted
parent-only latest transitions from the same prior image.
-/
theorem latest_common_prior_dec_requirements_of_statementCommitment_stages_and_msis
    {Digest Boundary PiCCSOut : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data : DirectConcreteInstantiation.ConcreteCEData n params)
    {hashEncoded : List Nat → Digest}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {PiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle
          (DigestParentBinding.Source Digest)
          n →
        PiCCSOut →
          Prop}
    {PiRLC :
      Nat →
        PiCCSOut →
        DigestParentBinding.Source Digest →
          Prop}
    {i : Nat}
    {prior nextA nextB :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (AccHandle Digest)}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hRed :
      SuperNeo.ProofSystem.MSISToAjtaiReductions params)
    (hMsis :
      SuperNeo.ProofSystem.MSISHardnessAssumption params)
    (hA :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (AccumulatorStep
          (params := params)
          PiCCS
          PiRLC
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        i
        prior
        nextA)
    (hB :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (AccumulatorStep
          (params := params)
          PiCCS
          PiRLC
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        i
        prior
        nextB) :
    ∃ priorInputs,
      ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := data.ce)
        (StatementEncodes :=
          ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent)
        prior.accumulator.parentSource
        priorInputs ∧
        ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          PiCCS
          PiRLC
          i
          prior.accumulator
          priorInputs
          nextA.accumulator.parentSource ∧
        ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          PiCCS
          PiRLC
          i
          prior.accumulator
          priorInputs
          nextB.accumulator.parentSource :=
  latest_common_prior_dec_requirements_of_stages_and_ajtaiCEOpening
    hDigest
    ParentOpeningAuthorization.statementEncodesByCommitment_functional
    (AjtaiResidueBinding.noAjtaiBindingCollision_of_msis hRed hMsis)
    (AjtaiResidueBinding.ceOpeningAdapter_of_assignmentOpeningAdapter
      (AjtaiResidueBinding.assignmentOpeningAdapter_of_ajtaiBackedCommitMap
        data.ajtaiBackedCommitMap))
    hA
    hB

/--
Terminal parent-only soundness with arbitrary sound prior authority.

The conclusion includes reachability, latest parent-source uniqueness, the
common pointwise private-DEC obligations for the hidden prior child table used
by the accepted latest transition and any alternate accepted transition from
the same prior image, and the Construction-2 public-image invariants.
-/
theorem terminal_soundness_of_prior_authority_sound_and_msis
    {Digest Boundary PiCCSOut Authority : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data : DirectConcreteInstantiation.ConcreteCEData n params)
    {hashEncoded : List Nat → Digest}
    {computeBoundary : Nat → Boundary → Boundary}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {PiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle
          (DigestParentBinding.Source Digest)
          n →
        PiCCSOut →
          Prop}
    {PiRLC :
      Nat →
        PiCCSOut →
        DigestParentBinding.Source Digest →
          Prop}
    {initial priorImage nextImage altNext :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (AccHandle Digest)}
    {AuthorityAccepts :
      Nat →
        Authority →
        Construction2DirectFPrime.PublicImage
          Digest
          Boundary
          (AccHandle Digest) →
          Prop}
    {priorSteps : Nat}
    {priorAuthority : Authority}
    {proof : Unit}
    (hPiCCS : ParentSourceStep.PiCCSFunctional PiCCS)
    (hPiRLC : ParentSourceStep.PiRLCFunctional PiRLC)
    (hInitialStep : initial.step = 0)
    (hInitialWellFormed : Construction2DirectFPrime.WellFormed initial)
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hRed :
      SuperNeo.ProofSystem.MSISToAjtaiReductions params)
    (hMsis :
      SuperNeo.ProofSystem.MSISHardnessAssumption params)
    (hAuthority :
      FPrimeInduction.PriorAuthoritySound
        (Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          PiCCS
          PiRLC
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        AuthorityAccepts)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        AuthorityAccepts
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority := Authority)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (AccumulatorStep
            (params := params)
            PiCCS
            PiRLC
            hashEncoded
            data.ce
            (ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent)))
        priorSteps
        priorAuthority
        priorImage
        nextImage
        proof)
    (hAlt :
      Construction2DirectFPrime.Transition
        (DirectProgramStep.ComputedBoundaryStep computeBoundary)
        (AccumulatorStep
          (params := params)
          PiCCS
          PiRLC
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          PiCCS
          PiRLC
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage.accumulator.parentSource =
        altNext.accumulator.parentSource ∧
      (∃ priorInputs,
        ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
          (n := n)
          (hashEncoded := hashEncoded)
          (params := params)
          (ce := data.ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent)
          priorImage.accumulator.parentSource
          priorInputs ∧
        ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          PiCCS
          PiRLC
          priorSteps
          priorImage.accumulator
          priorInputs
          nextImage.accumulator.parentSource ∧
        ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          PiCCS
          PiRLC
          priorSteps
          priorImage.accumulator
          priorInputs
          altNext.accumulator.parentSource) ∧
      nextImage.step = priorSteps + 1 ∧
      initial.vkDigest = nextImage.vkDigest ∧
      initial.initialBoundary = nextImage.initialBoundary ∧
      Construction2DirectFPrime.WellFormed nextImage := by
  have hReach :
      FPrimeInduction.Reachable
        (Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          PiCCS
          PiRLC
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage :=
    Construction2DirectFPrime.terminal_direct_fprime_reaches_final
      hAuthority
      hAccepted
  have hParent :
      nextImage.accumulator.parentSource =
        altNext.accumulator.parentSource :=
    latest_parentSource_functional_of_statementCommitment_stages_and_msis
      data
      hPiCCS
      hPiRLC
      hDigest
      hRed
      hMsis
      hAccepted.latestAccepted
      hAlt
  have hPointwise :
      ∃ priorInputs,
        ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
          (n := n)
          (hashEncoded := hashEncoded)
          (params := params)
          (ce := data.ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent)
          priorImage.accumulator.parentSource
          priorInputs ∧
        ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          PiCCS
          PiRLC
          priorSteps
          priorImage.accumulator
          priorInputs
          nextImage.accumulator.parentSource ∧
        ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          PiCCS
          PiRLC
          priorSteps
          priorImage.accumulator
          priorInputs
          altNext.accumulator.parentSource :=
    latest_common_prior_dec_requirements_of_statementCommitment_stages_and_msis
      data
      hDigest
      hRed
      hMsis
      hAccepted.latestAccepted
      hAlt
  have hPublic :=
    Construction2DirectFPrime.terminal_direct_fprime_public_image_invariants
      hInitialStep
      hInitialWellFormed
      hAuthority
      hAccepted
  rcases hPublic with ⟨hStep, hVk, hInitialBoundary, hWellFormed⟩
  exact
    ⟨hReach,
      hParent,
      hPointwise,
      hStep,
      hVk,
      hInitialBoundary,
      hWellFormed⟩

/--
Terminal parent-only soundness through a compressed prior verifier object.
-/
theorem terminal_soundness_of_sound_verifier_and_msis
    {Digest Boundary PiCCSOut PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data : DirectConcreteInstantiation.ConcreteCEData n params)
    {hashEncoded : List Nat → Digest}
    {computeBoundary : Nat → Boundary → Boundary}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {PiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle
          (DigestParentBinding.Source Digest)
          n →
        PiCCSOut →
          Prop}
    {PiRLC :
      Nat →
        PiCCSOut →
        DigestParentBinding.Source Digest →
          Prop}
    {initial priorImage nextImage altNext :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (AccHandle Digest)}
    (verifier :
      CompressedFPrimeAuthority.SoundVerifier
        (Image :=
          Construction2DirectFPrime.PublicImage
            Digest
            Boundary
            (AccHandle Digest))
        (Proof := PriorProof)
        (Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          PiCCS
          PiRLC
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    (hPiCCS : ParentSourceStep.PiCCSFunctional PiCCS)
    (hPiRLC : ParentSourceStep.PiRLCFunctional PiRLC)
    (hInitialStep : initial.step = 0)
    (hInitialWellFormed : Construction2DirectFPrime.WellFormed initial)
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hRed :
      SuperNeo.ProofSystem.MSISToAjtaiReductions params)
    (hMsis :
      SuperNeo.ProofSystem.MSISHardnessAssumption params)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.SoundVerifier.Accepts verifier)
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority := PriorProof)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (AccumulatorStep
            (params := params)
            PiCCS
            PiRLC
            hashEncoded
            data.ce
            (ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent)))
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      Construction2DirectFPrime.Transition
        (DirectProgramStep.ComputedBoundaryStep computeBoundary)
        (AccumulatorStep
          (params := params)
          PiCCS
          PiRLC
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          PiCCS
          PiRLC
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage.accumulator.parentSource =
        altNext.accumulator.parentSource ∧
      (∃ priorInputs,
        ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
          (n := n)
          (hashEncoded := hashEncoded)
          (params := params)
          (ce := data.ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent)
          priorImage.accumulator.parentSource
          priorInputs ∧
        ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          PiCCS
          PiRLC
          priorSteps
          priorImage.accumulator
          priorInputs
          nextImage.accumulator.parentSource ∧
        ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          PiCCS
          PiRLC
          priorSteps
          priorImage.accumulator
          priorInputs
          altNext.accumulator.parentSource) ∧
      nextImage.step = priorSteps + 1 ∧
      initial.vkDigest = nextImage.vkDigest ∧
      initial.initialBoundary = nextImage.initialBoundary ∧
      Construction2DirectFPrime.WellFormed nextImage :=
  terminal_soundness_of_prior_authority_sound_and_msis
    data
    hPiCCS
    hPiRLC
    hInitialStep
    hInitialWellFormed
    hDigest
    hRed
    hMsis
    (CompressedFPrimeAuthority.sound_verifier_prior_authority_sound verifier)
    hAccepted
    hAlt

end DirectParentOnlyTerminalSoundness

end DirectCcsFPrime
