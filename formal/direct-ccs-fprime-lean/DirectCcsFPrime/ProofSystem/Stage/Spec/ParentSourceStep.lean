import DirectCcsFPrime.ProofSystem.Terminal.Security.ReducedAccumulatorStep

/-!
Parent-source derivation for the reduced direct CCS F' accumulator.

This module splits the remaining `ParentSourceStep` obligation into the two
SuperNeo stages that actually produce the reduced parent handle:

* `Pi_CCS` derives the `CE(b)^(K+k)` output claims, and
* `Pi_RLC` folds those claims into one parent `CE(B)` source.

The module proves only the necessary composition theorem: if those two accepted
stage relations are functional, then the reduced parent-source relation is
functional. It does not define a parallel `Pi_CCS` or `Pi_RLC` protocol.
-/

namespace DirectCcsFPrime

namespace ParentSourceStep

/--
The accepted `Pi_CCS -> Pi_RLC` parent-source step.

`PiCCS` owns the accepted `Pi_CCS` output claims for the current step and prior
accumulator. `PiRLC` owns the accepted fold from those output claims to one
parent `CE(B)` source.
-/
def Step
    {Source PiCCSOut : Type}
    {n : Nat}
    (PiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle Source n →
        PiCCSOut →
          Prop)
    (PiRLC : Nat → PiCCSOut → Source → Prop)
    (i : Nat)
    (prior : ReducedAccumulatorStep.AccumulatorHandle Source n)
    (source : Source) : Prop :=
  ∃ piCCSOutput,
    PiCCS i prior piCCSOutput ∧
      PiRLC i piCCSOutput source

/--
`Pi_CCS` output functionality for a fixed step and prior accumulator.

For the concrete protocol this is the deterministic/non-malleable accepted
`Pi_CCS` transcript-output obligation.
-/
def PiCCSFunctional
    {Source PiCCSOut : Type}
    {n : Nat}
    (PiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle Source n →
        PiCCSOut →
          Prop) : Prop :=
  ∀ i prior outA outB,
    PiCCS i prior outA →
    PiCCS i prior outB →
      outA = outB

/--
`Pi_RLC` parent functionality for fixed `Pi_CCS` outputs.

For the concrete protocol this is the deterministic Fiat-Shamir challenge and
linear-combination parent `CE(B)` obligation.
-/
def PiRLCFunctional
    {PiCCSOut Source : Type}
    (PiRLC : Nat → PiCCSOut → Source → Prop) : Prop :=
  ∀ i out sourceA sourceB,
    PiRLC i out sourceA →
    PiRLC i out sourceB →
      sourceA = sourceB

/--
Functional `Pi_CCS` and functional `Pi_RLC` imply functional parent-source
derivation.
-/
theorem functional_of_stage_functional
    {Source PiCCSOut : Type}
    {n : Nat}
    {PiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle Source n →
        PiCCSOut →
          Prop}
    {PiRLC : Nat → PiCCSOut → Source → Prop}
    (hPiCCS : PiCCSFunctional PiCCS)
    (hPiRLC : PiRLCFunctional PiRLC) :
    ReducedAccumulatorStep.ParentSourceFunctional
      (Step PiCCS PiRLC) := by
  intro i prior sourceA sourceB hA hB
  rcases hA with ⟨outA, hPiCCSA, hPiRLCA⟩
  rcases hB with ⟨outB, hPiCCSB, hPiRLCB⟩
  have hOut : outA = outB :=
    hPiCCS i prior outA outB hPiCCSA hPiCCSB
  rw [hOut] at hPiRLCA
  exact hPiRLC i outB sourceA sourceB hPiRLCA hPiRLCB

/-- Function-computed `Pi_CCS` relation. -/
def ComputedPiCCS
    {Source PiCCSOut : Type}
    {n : Nat}
    (compute :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle Source n →
          PiCCSOut)
    (i : Nat)
    (prior : ReducedAccumulatorStep.AccumulatorHandle Source n)
    (out : PiCCSOut) : Prop :=
  out = compute i prior

/-- Function-computed `Pi_RLC` relation. -/
def ComputedPiRLC
    {PiCCSOut Source : Type}
    (compute : Nat → PiCCSOut → Source)
    (i : Nat)
    (out : PiCCSOut)
    (source : Source) : Prop :=
  source = compute i out

/-- A function-computed `Pi_CCS` relation is functional. -/
theorem computedPiCCS_functional
    {Source PiCCSOut : Type}
    {n : Nat}
    {compute :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle Source n →
          PiCCSOut} :
    PiCCSFunctional
      (ComputedPiCCS (n := n) compute) := by
  intro i prior outA outB hA hB
  exact hA.trans hB.symm

/-- A function-computed `Pi_RLC` relation is functional. -/
theorem computedPiRLC_functional
    {PiCCSOut Source : Type}
    {compute : Nat → PiCCSOut → Source} :
    PiRLCFunctional (ComputedPiRLC compute) := by
  intro i out sourceA sourceB hA hB
  exact hA.trans hB.symm

/--
Function-computed `Pi_CCS` and `Pi_RLC` stages induce a functional parent-source
step.
-/
theorem functional_of_computed_stages
    {Source PiCCSOut : Type}
    {n : Nat}
    {computePiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle Source n →
          PiCCSOut}
    {computePiRLC : Nat → PiCCSOut → Source} :
    ReducedAccumulatorStep.ParentSourceFunctional
      (Step
        (ComputedPiCCS (n := n) computePiCCS)
        (ComputedPiRLC computePiRLC)) :=
  functional_of_stage_functional
    computedPiCCS_functional
    computedPiRLC_functional

/--
Construction-2 accumulator functionality with the parent-source obligation
discharged from functional `Pi_CCS` and `Pi_RLC` stages.
-/
theorem transition_accumulator_fields_functional_of_stage_functional
    {Digest Boundary Source PiCCSOut : Type}
    {n : Nat}
    {BoundaryStep :
      Nat → Boundary → Boundary → Prop}
    {PiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle Source n →
        PiCCSOut →
          Prop}
    {PiRLC : Nat → PiCCSOut → Source → Prop}
    {Authorized : Source → DecDigitUniqueness.ColumnDigits n → Prop}
    {i : Nat}
    {prior nextA nextB :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (ReducedAccumulatorStep.AccumulatorHandle Source n)}
    (hPiCCS : PiCCSFunctional PiCCS)
    (hPiRLC : PiRLCFunctional PiRLC)
    (hAuthorized :
      ReducedAccumulatorStep.AuthorizedFunctional Authorized)
    (hA :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (ReducedAccumulatorStep.Step
          (Step PiCCS PiRLC)
          Authorized)
        i
        prior
        nextA)
    (hB :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (ReducedAccumulatorStep.Step
          (Step PiCCS PiRLC)
          Authorized)
        i
        prior
        nextB) :
    nextA.accumulator.parentSource = nextB.accumulator.parentSource ∧
      nextA.accumulator.nextPiCCSInputs =
        nextB.accumulator.nextPiCCSInputs :=
  ReducedAccumulatorStep.transition_accumulator_fields_functional
    (functional_of_stage_functional hPiCCS hPiRLC)
    hAuthorized
    hA
    hB

/--
Canonical reduced-handle theorem with both pieces composed:

* `Pi_CCS -> Pi_RLC` parent-source functionality, and
* Ajtai-backed canonical private `Pi_DEC` authorization functionality.
-/
theorem transition_accumulator_fields_functional_of_stages_and_ajtaiCEOpening
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
    {BoundaryStep :
      Nat → Boundary → Boundary → Prop}
    {PiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle
          (DigestParentBinding.Source Digest)
          n →
        PiCCSOut →
          Prop}
    {PiRLC :
      Nat → PiCCSOut → DigestParentBinding.Source Digest → Prop}
    {i : Nat}
    {prior nextA nextB :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (ReducedAccumulatorStep.AccumulatorHandle
          (DigestParentBinding.Source Digest)
          n)}
    (hPiCCS : PiCCSFunctional PiCCS)
    (hPiRLC : PiRLCFunctional PiRLC)
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
        (ReducedAccumulatorStep.Step
          (Step PiCCS PiRLC)
          (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
            (n := n)
            (hashEncoded := hashEncoded)
            (params := params)
            (ce := ce)
            (StatementEncodes := StatementEncodes)))
        i
        prior
        nextA)
    (hB :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (ReducedAccumulatorStep.Step
          (Step PiCCS PiRLC)
          (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
            (n := n)
            (hashEncoded := hashEncoded)
            (params := params)
            (ce := ce)
            (StatementEncodes := StatementEncodes)))
        i
        prior
        nextB) :
    nextA.accumulator.parentSource = nextB.accumulator.parentSource ∧
      nextA.accumulator.nextPiCCSInputs =
        nextB.accumulator.nextPiCCSInputs :=
  ReducedAccumulatorStep.transition_accumulator_fields_functional_of_ajtaiCEOpening
    (functional_of_stage_functional hPiCCS hPiRLC)
    hDigest
    hEncoding
    hNoCollision
    adapter
    hA
    hB

/--
Canonical reduced-handle theorem with deterministic parent-statement
commitment encoding.
-/
theorem transition_accumulator_fields_functional_of_stages_statementCommitment_and_ajtaiCEOpening
    {Digest Boundary PiCCSOut : Type}
    {n : Nat}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {BoundaryStep :
      Nat → Boundary → Boundary → Prop}
    {PiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle
          (DigestParentBinding.Source Digest)
          n →
        PiCCSOut →
          Prop}
    {PiRLC :
      Nat → PiCCSOut → DigestParentBinding.Source Digest → Prop}
    {i : Nat}
    {prior nextA nextB :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (ReducedAccumulatorStep.AccumulatorHandle
          (DigestParentBinding.Source Digest)
          n)}
    (hPiCCS : PiCCSFunctional PiCCS)
    (hPiRLC : PiRLCFunctional PiRLC)
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
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
        (ReducedAccumulatorStep.Step
          (Step PiCCS PiRLC)
          (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
            (n := n)
            (hashEncoded := hashEncoded)
            (params := params)
            (ce := ce)
            (StatementEncodes :=
              ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent)))
        i
        prior
        nextA)
    (hB :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (ReducedAccumulatorStep.Step
          (Step PiCCS PiRLC)
          (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
            (n := n)
            (hashEncoded := hashEncoded)
            (params := params)
            (ce := ce)
            (StatementEncodes :=
              ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent)))
        i
        prior
        nextB) :
    nextA.accumulator.parentSource = nextB.accumulator.parentSource ∧
      nextA.accumulator.nextPiCCSInputs =
        nextB.accumulator.nextPiCCSInputs :=
  transition_accumulator_fields_functional_of_stages_and_ajtaiCEOpening
    hPiCCS
    hPiRLC
    hDigest
    ParentOpeningAuthorization.statementEncodesByCommitment_functional
    hNoCollision
    adapter
    hA
    hB

end ParentSourceStep

end DirectCcsFPrime
