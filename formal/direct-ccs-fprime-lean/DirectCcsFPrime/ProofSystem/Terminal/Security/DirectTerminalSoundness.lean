import DirectCcsFPrime.Core.FoldedFPrimeAuthority
import DirectCcsFPrime.ProofSystem.Stage.Spec.ParentSourceStep

/-!
Terminal direct CCS F' soundness composition.

This module composes the necessary theorem layers for the reduced-handle direct
CCS terminal proof:

* proof-carrying prior F' authority,
* one latest Construction-2 direct F' transition,
* `Pi_CCS -> Pi_RLC` parent-source functionality, and
* Ajtai-backed canonical private `Pi_DEC` authorization.

It still does not instantiate the concrete direct CCS boundary relation,
concrete `Pi_CCS`/`Pi_RLC` transcript computations, or concrete Ajtai/MSIS
security. Those are the remaining implementation/paper instantiation
obligations.
-/

namespace DirectCcsFPrime

namespace DirectTerminalSoundness

/-- Reduced direct CCS accumulator handle specialized to a digest-bound parent source. -/
abbrev AccHandle (Digest : Type) (n : Nat) :=
  ReducedAccumulatorStep.AccumulatorHandle
    (DigestParentBinding.Source Digest)
    n

/-- Canonical reduced accumulator step for the terminal direct CCS F' theorem. -/
def AccumulatorStep
    {Digest PiCCSOut : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (PiCCS :
      Nat →
        AccHandle Digest n →
        PiCCSOut →
          Prop)
    (PiRLC :
      Nat → PiCCSOut → DigestParentBinding.Source Digest → Prop)
    (hashEncoded : List Nat → Digest)
    (ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment)
    (StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment) :
    Nat → AccHandle Digest n → AccHandle Digest n → Prop :=
  ReducedAccumulatorStep.Step
    (ParentSourceStep.Step PiCCS PiRLC)
    (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
      (n := n)
      (hashEncoded := hashEncoded)
      (params := params)
      (ce := ce)
      (StatementEncodes := StatementEncodes))

/-- Canonical direct CCS Construction-2 transition used by terminal compression. -/
def Transition
    {Digest Boundary PiCCSOut : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (BoundaryStep : Nat → Boundary → Boundary → Prop)
    (PiCCS :
      Nat →
        AccHandle Digest n →
        PiCCSOut →
          Prop)
    (PiRLC :
      Nat → PiCCSOut → DigestParentBinding.Source Digest → Prop)
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
        (AccHandle Digest n) →
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (AccHandle Digest n) →
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

/-- Proof-carrying prior authority for the canonical terminal direct transition. -/
abbrev Authority
    {Digest Boundary PiCCSOut : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (BoundaryStep : Nat → Boundary → Boundary → Prop)
    (PiCCS :
      Nat →
        AccHandle Digest n →
        PiCCSOut →
          Prop)
    (PiRLC :
      Nat → PiCCSOut → DigestParentBinding.Source Digest → Prop)
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
        (AccHandle Digest n)) :=
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
Terminal direct compression reaches the final public image when its prior F'
authority is proof-carrying and the latest step is accepted.
-/
theorem terminal_reaches_final
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
        AccHandle Digest n →
        PiCCSOut →
          Prop}
    {PiRLC :
      Nat → PiCCSOut → DigestParentBinding.Source Digest → Prop}
    {initial priorImage nextImage :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (AccHandle Digest n)}
    {priorSteps : Nat}
    {priorAuthority :
      Authority
        (params := params)
        BoundaryStep
        PiCCS
        PiRLC
        hashEncoded
        ce
        StatementEncodes
        initial}
    {proof : Unit}
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (FoldedFPrimeAuthority.Accepts
          (Transition :=
            Transition
              (params := params)
              BoundaryStep
              PiCCS
              PiRLC
              hashEncoded
              ce
              StatementEncodes)
          (initial := initial))
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority :=
            Authority
              (params := params)
              BoundaryStep
              PiCCS
              PiRLC
              hashEncoded
              ce
              StatementEncodes
              initial)
          BoundaryStep
          (AccumulatorStep
            (params := params)
            PiCCS
            PiRLC
            hashEncoded
            ce
            StatementEncodes))
        priorSteps
        priorAuthority
        priorImage
        nextImage
        proof) :
    FPrimeInduction.Reachable
      (Transition
        (params := params)
        BoundaryStep
        PiCCS
        PiRLC
        hashEncoded
        ce
        StatementEncodes)
      initial
      (priorSteps + 1)
      nextImage :=
  FoldedFPrimeAuthority.construction2_terminal_reaches_final
    hAccepted

/--
Composed terminal theorem.

The final image is reachable, and the latest reduced accumulator update cannot
be swapped for a different parent source or next `Pi_CCS` child table from the
same prior image.
-/
theorem terminal_reaches_final_and_latest_accumulator_functional
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
        AccHandle Digest n →
        PiCCSOut →
          Prop}
    {PiRLC :
      Nat → PiCCSOut → DigestParentBinding.Source Digest → Prop}
    {initial priorImage nextImage altNext :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (AccHandle Digest n)}
    {priorSteps : Nat}
    {priorAuthority :
      Authority
        (params := params)
        BoundaryStep
        PiCCS
        PiRLC
        hashEncoded
        ce
        StatementEncodes
        initial}
    {proof : Unit}
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
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (FoldedFPrimeAuthority.Accepts
          (Transition :=
            Transition
              (params := params)
              BoundaryStep
              PiCCS
              PiRLC
              hashEncoded
              ce
              StatementEncodes)
          (initial := initial))
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority :=
            Authority
              (params := params)
              BoundaryStep
              PiCCS
              PiRLC
              hashEncoded
              ce
              StatementEncodes
              initial)
          BoundaryStep
          (AccumulatorStep
            (params := params)
            PiCCS
            PiRLC
            hashEncoded
            ce
            StatementEncodes))
        priorSteps
        priorAuthority
        priorImage
        nextImage
        proof)
    (hAlt :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (AccumulatorStep
          (params := params)
          PiCCS
          PiRLC
          hashEncoded
          ce
          StatementEncodes)
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (Transition
          (params := params)
          BoundaryStep
          PiCCS
          PiRLC
          hashEncoded
          ce
          StatementEncodes)
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage.accumulator.parentSource =
        altNext.accumulator.parentSource ∧
      nextImage.accumulator.nextPiCCSInputs =
        altNext.accumulator.nextPiCCSInputs := by
  constructor
  · exact terminal_reaches_final hAccepted
  · exact
      ParentSourceStep.transition_accumulator_fields_functional_of_stages_and_ajtaiCEOpening
        hPiCCS
        hPiRLC
        hDigest
        hEncoding
        hNoCollision
        adapter
        hAccepted.latestAccepted
        hAlt

/--
Computed-stage terminal theorem.

This is the implementation-facing specialization of
`terminal_reaches_final_and_latest_accumulator_functional`: when `Pi_CCS` and
`Pi_RLC` are represented by deterministic computations, their functionality is
discharged by construction instead of supplied as separate assumptions.
-/
theorem terminal_reaches_final_and_latest_accumulator_functional_of_computed_stages
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
    {computePiCCS :
      Nat →
        AccHandle Digest n →
        PiCCSOut}
    {computePiRLC :
      Nat → PiCCSOut → DigestParentBinding.Source Digest}
    {initial priorImage nextImage altNext :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (AccHandle Digest n)}
    {priorSteps : Nat}
    {priorAuthority :
      Authority
        (params := params)
        BoundaryStep
        (ParentSourceStep.ComputedPiCCS
          (n := n)
          computePiCCS)
        (ParentSourceStep.ComputedPiRLC computePiRLC)
        hashEncoded
        ce
        StatementEncodes
        initial}
    {proof : Unit}
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
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (FoldedFPrimeAuthority.Accepts
          (Transition :=
            Transition
              (params := params)
              BoundaryStep
              (ParentSourceStep.ComputedPiCCS
                (n := n)
                computePiCCS)
              (ParentSourceStep.ComputedPiRLC computePiRLC)
              hashEncoded
              ce
              StatementEncodes)
          (initial := initial))
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority :=
            Authority
              (params := params)
              BoundaryStep
              (ParentSourceStep.ComputedPiCCS
                (n := n)
                computePiCCS)
              (ParentSourceStep.ComputedPiRLC computePiRLC)
              hashEncoded
              ce
              StatementEncodes
              initial)
          BoundaryStep
          (AccumulatorStep
            (params := params)
            (ParentSourceStep.ComputedPiCCS
              (n := n)
              computePiCCS)
            (ParentSourceStep.ComputedPiRLC computePiRLC)
            hashEncoded
            ce
            StatementEncodes))
        priorSteps
        priorAuthority
        priorImage
        nextImage
        proof)
    (hAlt :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (AccumulatorStep
          (params := params)
          (ParentSourceStep.ComputedPiCCS
            (n := n)
            computePiCCS)
          (ParentSourceStep.ComputedPiRLC computePiRLC)
          hashEncoded
          ce
          StatementEncodes)
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (Transition
          (params := params)
          BoundaryStep
          (ParentSourceStep.ComputedPiCCS
            (n := n)
            computePiCCS)
          (ParentSourceStep.ComputedPiRLC computePiRLC)
          hashEncoded
          ce
          StatementEncodes)
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage.accumulator.parentSource =
        altNext.accumulator.parentSource ∧
      nextImage.accumulator.nextPiCCSInputs =
        altNext.accumulator.nextPiCCSInputs :=
  terminal_reaches_final_and_latest_accumulator_functional
    ParentSourceStep.computedPiCCS_functional
    ParentSourceStep.computedPiRLC_functional
    hDigest
    hEncoding
    hNoCollision
    adapter
    hAccepted
    hAlt

/--
Computed-stage terminal theorem with deterministic parent-statement
commitment encoding.

This is the narrower implementation-facing theorem: deterministic `Pi_CCS`,
deterministic `Pi_RLC`, and deterministic parent-statement commitment encoding
discharge all non-cryptographic functionality/serializer premises. The
remaining assumptions are the real binding boundaries.
-/
theorem terminal_reaches_final_and_latest_accumulator_functional_of_computed_stages_and_statement_commitment
    {Digest Boundary PiCCSOut : Type}
    {n : Nat}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {computePiCCS :
      Nat →
        AccHandle Digest n →
        PiCCSOut}
    {computePiRLC :
      Nat → PiCCSOut → DigestParentBinding.Source Digest}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {initial priorImage nextImage altNext :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (AccHandle Digest n)}
    {priorSteps : Nat}
    {priorAuthority :
      Authority
        (params := params)
        BoundaryStep
        (ParentSourceStep.ComputedPiCCS
          (n := n)
          computePiCCS)
        (ParentSourceStep.ComputedPiRLC computePiRLC)
        hashEncoded
        ce
        (ParentOpeningAuthorization.StatementEncodesByCommitment
          commitmentOfParent)
        initial}
    {proof : Unit}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hNoCollision :
      AjtaiResidueBinding.NoAjtaiBindingCollision params)
    (adapter :
      AjtaiResidueBinding.CEOpeningAdapter
        n
        params
        ce)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (FoldedFPrimeAuthority.Accepts
          (Transition :=
            Transition
              (params := params)
              BoundaryStep
              (ParentSourceStep.ComputedPiCCS
                (n := n)
                computePiCCS)
              (ParentSourceStep.ComputedPiRLC computePiRLC)
              hashEncoded
              ce
              (ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent))
          (initial := initial))
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority :=
            Authority
              (params := params)
              BoundaryStep
              (ParentSourceStep.ComputedPiCCS
                (n := n)
                computePiCCS)
              (ParentSourceStep.ComputedPiRLC computePiRLC)
              hashEncoded
              ce
              (ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent)
              initial)
          BoundaryStep
          (AccumulatorStep
            (params := params)
            (ParentSourceStep.ComputedPiCCS
              (n := n)
              computePiCCS)
            (ParentSourceStep.ComputedPiRLC computePiRLC)
            hashEncoded
            ce
            (ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent)))
        priorSteps
        priorAuthority
        priorImage
        nextImage
        proof)
    (hAlt :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (AccumulatorStep
          (params := params)
          (ParentSourceStep.ComputedPiCCS
            (n := n)
            computePiCCS)
          (ParentSourceStep.ComputedPiRLC computePiRLC)
          hashEncoded
          ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (Transition
          (params := params)
          BoundaryStep
          (ParentSourceStep.ComputedPiCCS
            (n := n)
            computePiCCS)
          (ParentSourceStep.ComputedPiRLC computePiRLC)
          hashEncoded
          ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage.accumulator.parentSource =
        altNext.accumulator.parentSource ∧
      nextImage.accumulator.nextPiCCSInputs =
        altNext.accumulator.nextPiCCSInputs :=
  terminal_reaches_final_and_latest_accumulator_functional_of_computed_stages
    hDigest
    ParentOpeningAuthorization.statementEncodesByCommitment_functional
    hNoCollision
    adapter
    hAccepted
    hAlt

/--
Narrow implementation-facing terminal theorem.

Deterministic `Pi_CCS`, deterministic `Pi_RLC`, deterministic parent-statement
commitment encoding, and an assignment-level Ajtai opening adapter for the
fixed CE commitment map discharge the local deterministic/adapter premises.
The remaining assumptions are exactly the parent digest binding and Ajtai
no-collision security boundaries.
-/
theorem terminal_reaches_final_and_latest_accumulator_functional_of_computed_stages_statement_commitment_and_assignment_opening
    {Digest Boundary PiCCSOut : Type}
    {n : Nat}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {computePiCCS :
      Nat →
        AccHandle Digest n →
        PiCCSOut}
    {computePiRLC :
      Nat → PiCCSOut → DigestParentBinding.Source Digest}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {initial priorImage nextImage altNext :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (AccHandle Digest n)}
    {priorSteps : Nat}
    {priorAuthority :
      Authority
        (params := params)
        BoundaryStep
        (ParentSourceStep.ComputedPiCCS
          (n := n)
          computePiCCS)
        (ParentSourceStep.ComputedPiRLC computePiRLC)
        hashEncoded
        ce
        (ParentOpeningAuthorization.StatementEncodesByCommitment
          commitmentOfParent)
        initial}
    {proof : Unit}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hNoCollision :
      AjtaiResidueBinding.NoAjtaiBindingCollision params)
    (assignmentAdapter :
      AjtaiResidueBinding.AssignmentOpeningAdapter
        n
        params
        ce.commitMap)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (FoldedFPrimeAuthority.Accepts
          (Transition :=
            Transition
              (params := params)
              BoundaryStep
              (ParentSourceStep.ComputedPiCCS
                (n := n)
                computePiCCS)
              (ParentSourceStep.ComputedPiRLC computePiRLC)
              hashEncoded
              ce
              (ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent))
          (initial := initial))
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority :=
            Authority
              (params := params)
              BoundaryStep
              (ParentSourceStep.ComputedPiCCS
                (n := n)
                computePiCCS)
              (ParentSourceStep.ComputedPiRLC computePiRLC)
              hashEncoded
              ce
              (ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent)
              initial)
          BoundaryStep
          (AccumulatorStep
            (params := params)
            (ParentSourceStep.ComputedPiCCS
              (n := n)
              computePiCCS)
            (ParentSourceStep.ComputedPiRLC computePiRLC)
            hashEncoded
            ce
            (ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent)))
        priorSteps
        priorAuthority
        priorImage
        nextImage
        proof)
    (hAlt :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (AccumulatorStep
          (params := params)
          (ParentSourceStep.ComputedPiCCS
            (n := n)
            computePiCCS)
          (ParentSourceStep.ComputedPiRLC computePiRLC)
          hashEncoded
          ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (Transition
          (params := params)
          BoundaryStep
          (ParentSourceStep.ComputedPiCCS
            (n := n)
            computePiCCS)
          (ParentSourceStep.ComputedPiRLC computePiRLC)
          hashEncoded
          ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage.accumulator.parentSource =
        altNext.accumulator.parentSource ∧
      nextImage.accumulator.nextPiCCSInputs =
        altNext.accumulator.nextPiCCSInputs :=
  terminal_reaches_final_and_latest_accumulator_functional_of_computed_stages_and_statement_commitment
    hDigest
    hNoCollision
    (AjtaiResidueBinding.ceOpeningAdapter_of_assignmentOpeningAdapter
      assignmentAdapter)
    hAccepted
    hAlt

/--
Same implementation-facing theorem, but using the theorem-facing Ajtai binding
assumption from `formal/superneo-lean` instead of the local Prop-level
no-collision statement.
-/
theorem terminal_reaches_final_and_latest_accumulator_functional_of_computed_stages_statement_commitment_assignment_opening_and_ajtai_binding
    {Digest Boundary PiCCSOut : Type}
    {n : Nat}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {computePiCCS :
      Nat →
        AccHandle Digest n →
        PiCCSOut}
    {computePiRLC :
      Nat → PiCCSOut → DigestParentBinding.Source Digest}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {initial priorImage nextImage altNext :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (AccHandle Digest n)}
    {priorSteps : Nat}
    {priorAuthority :
      Authority
        (params := params)
        BoundaryStep
        (ParentSourceStep.ComputedPiCCS
          (n := n)
          computePiCCS)
        (ParentSourceStep.ComputedPiRLC computePiRLC)
        hashEncoded
        ce
        (ParentOpeningAuthorization.StatementEncodesByCommitment
          commitmentOfParent)
        initial}
    {proof : Unit}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hAjtaiBinding :
      SuperNeo.ProofSystem.AjtaiBindingAssumption params)
    (assignmentAdapter :
      AjtaiResidueBinding.AssignmentOpeningAdapter
        n
        params
        ce.commitMap)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (FoldedFPrimeAuthority.Accepts
          (Transition :=
            Transition
              (params := params)
              BoundaryStep
              (ParentSourceStep.ComputedPiCCS
                (n := n)
                computePiCCS)
              (ParentSourceStep.ComputedPiRLC computePiRLC)
              hashEncoded
              ce
              (ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent))
          (initial := initial))
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority :=
            Authority
              (params := params)
              BoundaryStep
              (ParentSourceStep.ComputedPiCCS
                (n := n)
                computePiCCS)
              (ParentSourceStep.ComputedPiRLC computePiRLC)
              hashEncoded
              ce
              (ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent)
              initial)
          BoundaryStep
          (AccumulatorStep
            (params := params)
            (ParentSourceStep.ComputedPiCCS
              (n := n)
              computePiCCS)
            (ParentSourceStep.ComputedPiRLC computePiRLC)
            hashEncoded
            ce
            (ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent)))
        priorSteps
        priorAuthority
        priorImage
        nextImage
        proof)
    (hAlt :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (AccumulatorStep
          (params := params)
          (ParentSourceStep.ComputedPiCCS
            (n := n)
            computePiCCS)
          (ParentSourceStep.ComputedPiRLC computePiRLC)
          hashEncoded
          ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (Transition
          (params := params)
          BoundaryStep
          (ParentSourceStep.ComputedPiCCS
            (n := n)
            computePiCCS)
          (ParentSourceStep.ComputedPiRLC computePiRLC)
          hashEncoded
          ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage.accumulator.parentSource =
        altNext.accumulator.parentSource ∧
      nextImage.accumulator.nextPiCCSInputs =
        altNext.accumulator.nextPiCCSInputs :=
  terminal_reaches_final_and_latest_accumulator_functional_of_computed_stages_statement_commitment_and_assignment_opening
    hDigest
    (AjtaiResidueBinding.noAjtaiBindingCollision_of_ajtaiBindingAssumption
      hAjtaiBinding)
    assignmentAdapter
    hAccepted
    hAlt

/--
Same strongest implementation-facing theorem, but deriving Ajtai binding from
the SuperNeo MSIS hardness boundary and its MSIS-to-Ajtai reduction surface.
-/
theorem terminal_reaches_final_and_latest_accumulator_functional_of_computed_stages_statement_commitment_assignment_opening_and_msis
    {Digest Boundary PiCCSOut : Type}
    {n : Nat}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {computePiCCS :
      Nat →
        AccHandle Digest n →
        PiCCSOut}
    {computePiRLC :
      Nat → PiCCSOut → DigestParentBinding.Source Digest}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {initial priorImage nextImage altNext :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (AccHandle Digest n)}
    {priorSteps : Nat}
    {priorAuthority :
      Authority
        (params := params)
        BoundaryStep
        (ParentSourceStep.ComputedPiCCS
          (n := n)
          computePiCCS)
        (ParentSourceStep.ComputedPiRLC computePiRLC)
        hashEncoded
        ce
        (ParentOpeningAuthorization.StatementEncodesByCommitment
          commitmentOfParent)
        initial}
    {proof : Unit}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hRed :
      SuperNeo.ProofSystem.MSISToAjtaiReductions params)
    (hMsis :
      SuperNeo.ProofSystem.MSISHardnessAssumption params)
    (assignmentAdapter :
      AjtaiResidueBinding.AssignmentOpeningAdapter
        n
        params
        ce.commitMap)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (FoldedFPrimeAuthority.Accepts
          (Transition :=
            Transition
              (params := params)
              BoundaryStep
              (ParentSourceStep.ComputedPiCCS
                (n := n)
                computePiCCS)
              (ParentSourceStep.ComputedPiRLC computePiRLC)
              hashEncoded
              ce
              (ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent))
          (initial := initial))
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority :=
            Authority
              (params := params)
              BoundaryStep
              (ParentSourceStep.ComputedPiCCS
                (n := n)
                computePiCCS)
              (ParentSourceStep.ComputedPiRLC computePiRLC)
              hashEncoded
              ce
              (ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent)
              initial)
          BoundaryStep
          (AccumulatorStep
            (params := params)
            (ParentSourceStep.ComputedPiCCS
              (n := n)
              computePiCCS)
            (ParentSourceStep.ComputedPiRLC computePiRLC)
            hashEncoded
            ce
            (ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent)))
        priorSteps
        priorAuthority
        priorImage
        nextImage
        proof)
    (hAlt :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (AccumulatorStep
          (params := params)
          (ParentSourceStep.ComputedPiCCS
            (n := n)
            computePiCCS)
          (ParentSourceStep.ComputedPiRLC computePiRLC)
          hashEncoded
          ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (Transition
          (params := params)
          BoundaryStep
          (ParentSourceStep.ComputedPiCCS
            (n := n)
            computePiCCS)
          (ParentSourceStep.ComputedPiRLC computePiRLC)
          hashEncoded
          ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage.accumulator.parentSource =
        altNext.accumulator.parentSource ∧
      nextImage.accumulator.nextPiCCSInputs =
        altNext.accumulator.nextPiCCSInputs :=
  terminal_reaches_final_and_latest_accumulator_functional_of_computed_stages_statement_commitment_and_assignment_opening
    hDigest
    (AjtaiResidueBinding.noAjtaiBindingCollision_of_msis hRed hMsis)
    assignmentAdapter
    hAccepted
    hAlt

/--
Concrete-backed version of the strongest terminal theorem.

Instead of asking the caller for an arbitrary assignment-opening adapter, this
version requires the fixed CE commitment map to be backed by canonical Ajtai
commitments `(M || Mz)` for a fixed public matrix. The adapter is then derived
inside Lean from the canonical opening proof.
-/
theorem terminal_reaches_final_and_latest_accumulator_functional_of_computed_stages_statement_commitment_ajtai_backed_commit_map_and_msis
    {Digest Boundary PiCCSOut : Type}
    {n : Nat}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {computePiCCS :
      Nat →
        AccHandle Digest n →
        PiCCSOut}
    {computePiRLC :
      Nat → PiCCSOut → DigestParentBinding.Source Digest}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {initial priorImage nextImage altNext :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (AccHandle Digest n)}
    {priorSteps : Nat}
    {priorAuthority :
      Authority
        (params := params)
        BoundaryStep
        (ParentSourceStep.ComputedPiCCS
          (n := n)
          computePiCCS)
        (ParentSourceStep.ComputedPiRLC computePiRLC)
        hashEncoded
        ce
        (ParentOpeningAuthorization.StatementEncodesByCommitment
          commitmentOfParent)
        initial}
    {proof : Unit}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hRed :
      SuperNeo.ProofSystem.MSISToAjtaiReductions params)
    (hMsis :
      SuperNeo.ProofSystem.MSISHardnessAssumption params)
    (backing :
      AjtaiResidueBinding.AjtaiBackedCommitMap
        n
        params
        ce.commitMap)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (FoldedFPrimeAuthority.Accepts
          (Transition :=
            Transition
              (params := params)
              BoundaryStep
              (ParentSourceStep.ComputedPiCCS
                (n := n)
                computePiCCS)
              (ParentSourceStep.ComputedPiRLC computePiRLC)
              hashEncoded
              ce
              (ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent))
          (initial := initial))
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority :=
            Authority
              (params := params)
              BoundaryStep
              (ParentSourceStep.ComputedPiCCS
                (n := n)
                computePiCCS)
              (ParentSourceStep.ComputedPiRLC computePiRLC)
              hashEncoded
              ce
              (ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent)
              initial)
          BoundaryStep
          (AccumulatorStep
            (params := params)
            (ParentSourceStep.ComputedPiCCS
              (n := n)
              computePiCCS)
            (ParentSourceStep.ComputedPiRLC computePiRLC)
            hashEncoded
            ce
            (ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent)))
        priorSteps
        priorAuthority
        priorImage
        nextImage
        proof)
    (hAlt :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (AccumulatorStep
          (params := params)
          (ParentSourceStep.ComputedPiCCS
            (n := n)
            computePiCCS)
          (ParentSourceStep.ComputedPiRLC computePiRLC)
          hashEncoded
          ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (Transition
          (params := params)
          BoundaryStep
          (ParentSourceStep.ComputedPiCCS
            (n := n)
            computePiCCS)
          (ParentSourceStep.ComputedPiRLC computePiRLC)
          hashEncoded
          ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage.accumulator.parentSource =
        altNext.accumulator.parentSource ∧
      nextImage.accumulator.nextPiCCSInputs =
        altNext.accumulator.nextPiCCSInputs :=
  terminal_reaches_final_and_latest_accumulator_functional_of_computed_stages_statement_commitment_assignment_opening_and_msis
    hDigest
    hRed
    hMsis
    (AjtaiResidueBinding.assignmentOpeningAdapter_of_ajtaiBackedCommitMap
      backing)
    hAccepted
    hAlt

end DirectTerminalSoundness

end DirectCcsFPrime
