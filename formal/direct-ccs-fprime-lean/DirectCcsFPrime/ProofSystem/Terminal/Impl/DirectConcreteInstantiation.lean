import DirectCcsFPrime.ProofSystem.Terminal.Security.DirectTerminalSoundness

/-!
Concrete instantiation boundary for direct CCS terminal F' soundness.

This module removes the remaining abstract CE commitment-map freedom from the
terminal theorem. The CE relation used here is built with the canonical Ajtai
commitment map `assignment ↦ M || Mz`.

It intentionally does not define new `Pi_CCS`, `Pi_RLC`, boundary-step, or
folded-authority semantics. Those are still the concrete protocol objects that
must be supplied by the end-to-end direct CCS instantiation.
-/

namespace DirectCcsFPrime

namespace DirectConcreteInstantiation

/--
Concrete CE data for the direct terminal theorem.

The commitment map is not an arbitrary field. It is the canonical Ajtai map for
a fixed public matrix and assignment-opening encoder. The remaining fields are
exactly the shape/input/evaluation data needed to form the SuperNeo CE relation.
-/
structure ConcreteCEData
    (n : Nat)
    (params : SuperNeo.ProofSystem.AjtaiParams) where
  normBound : Nat
  inputProjector :
    SuperNeo.ProofSystem.ConstraintSystem.InputProjector
  shape :
    SuperNeo.ProofSystem.ConstraintSystem.CCSStructure
  matrixFlat : Array SuperNeo.Coeffs
  toOpening :
    SuperNeo.Coeffs →
      SuperNeo.ProofSystem.Opening
  projectWitnessResidues : Array SuperNeo.Coeffs → Fin n → Nat
  matrixShape : matrixFlat.size = params.matrixFlatLen
  openingWellFormed :
    ∀ assignment,
      SuperNeo.ProofSystem.Opening.WellFormed
        params
        (toOpening assignment)
  openingNormSound :
    ∀ assignment,
      SuperNeo.ProofSystem.Opening.NormSound
        (toOpening assignment)
  bounded :
    ∀ assignment,
      (toOpening assignment).normBound < params.bindingNormBound
  residueSound :
    ∀ assignment,
      projectWitnessResidues (toOpening assignment).witness =
        ParentOpeningAuthorization.assignmentResidues
          (n := n)
          assignment

namespace ConcreteCEData

/-- Canonical Ajtai commitment map owned by concrete CE data. -/
def commitMap
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data : ConcreteCEData n params) :
    SuperNeo.Coeffs → SuperNeo.ProofSystem.Commitment :=
  AjtaiResidueBinding.ajtaiCommitMap
    params
    data.matrixFlat
    data.toOpening

/-- SuperNeo CE relation built with the canonical Ajtai commitment map. -/
def ce
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data : ConcreteCEData n params) :
    SuperNeo.ProofSystem.ConstraintSystem.CE
      SuperNeo.ProofSystem.Commitment where
  normBound := data.normBound
  commitMap := data.commitMap
  inputProjector := data.inputProjector
  shape := data.shape

/--
The CE commitment map induced by concrete CE data is Ajtai-backed by
construction.
-/
def ajtaiBackedCommitMap
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data : ConcreteCEData n params) :
    AjtaiResidueBinding.AjtaiBackedCommitMap
      n
      params
      data.ce.commitMap :=
  AjtaiResidueBinding.ajtaiBackedCommitMap_of_ajtaiCommitMap
    data.matrixShape
    data.openingWellFormed
    data.openingNormSound
    data.bounded
    data.residueSound

end ConcreteCEData

/--
Terminal direct CCS soundness for a CE relation built from the canonical Ajtai
commitment map.

This theorem deliberately still requires:

* encoded parent `CE(B)` digest binding,
* the SuperNeo MSIS-to-Ajtai reduction and MSIS hardness assumptions,
* accepted folded F' prior authority and latest-step proof, and
* concrete deterministic `computePiCCS` / `computePiRLC` functions.

Those are real remaining end-to-end obligations, not local adapter assumptions.
-/
theorem terminal_soundness_of_concrete_ce_and_msis
    {Digest Boundary PiCCSOut : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data : ConcreteCEData n params)
    {hashEncoded : List Nat → Digest}
    {BoundaryStep : Nat → Boundary → Boundary → Prop}
    {computePiCCS :
      Nat →
        DirectTerminalSoundness.AccHandle Digest n →
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
        (DirectTerminalSoundness.AccHandle Digest n)}
    {priorSteps : Nat}
    {priorAuthority :
      DirectTerminalSoundness.Authority
        (params := params)
        BoundaryStep
        (ParentSourceStep.ComputedPiCCS
          (n := n)
          computePiCCS)
        (ParentSourceStep.ComputedPiRLC computePiRLC)
        hashEncoded
        data.ce
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
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (FoldedFPrimeAuthority.Accepts
          (Transition :=
            DirectTerminalSoundness.Transition
              (params := params)
              BoundaryStep
              (ParentSourceStep.ComputedPiCCS
                (n := n)
                computePiCCS)
              (ParentSourceStep.ComputedPiRLC computePiRLC)
              hashEncoded
              data.ce
              (ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent))
          (initial := initial))
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority :=
            DirectTerminalSoundness.Authority
              (params := params)
              BoundaryStep
              (ParentSourceStep.ComputedPiCCS
                (n := n)
                computePiCCS)
              (ParentSourceStep.ComputedPiRLC computePiRLC)
              hashEncoded
              data.ce
              (ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent)
              initial)
          BoundaryStep
          (DirectTerminalSoundness.AccumulatorStep
            (params := params)
            (ParentSourceStep.ComputedPiCCS
              (n := n)
              computePiCCS)
            (ParentSourceStep.ComputedPiRLC computePiRLC)
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
        BoundaryStep
        (DirectTerminalSoundness.AccumulatorStep
          (params := params)
          (ParentSourceStep.ComputedPiCCS
            (n := n)
            computePiCCS)
          (ParentSourceStep.ComputedPiRLC computePiRLC)
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (DirectTerminalSoundness.Transition
          (params := params)
          BoundaryStep
          (ParentSourceStep.ComputedPiCCS
            (n := n)
            computePiCCS)
          (ParentSourceStep.ComputedPiRLC computePiRLC)
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage.accumulator.parentSource =
        altNext.accumulator.parentSource ∧
      nextImage.accumulator.nextPiCCSInputs =
        altNext.accumulator.nextPiCCSInputs :=
  DirectTerminalSoundness.terminal_reaches_final_and_latest_accumulator_functional_of_computed_stages_statement_commitment_ajtai_backed_commit_map_and_msis
    hDigest
    hRed
    hMsis
    data.ajtaiBackedCommitMap
    hAccepted
    hAlt

end DirectConcreteInstantiation

end DirectCcsFPrime
