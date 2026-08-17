import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityComposition
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.FullOracleSoundness
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.InteractiveCompositionBridge

/-!
Contract: exact noninteractive NIFS key for `PaddedRowIdentity`.

Owns: construction of the one-joint NIFS key from the selected 24-variable,
14-matrix relation; exact `1 + 14` arity; degree nine; one identity-first
matrix; definitional equality between the NIFS interactive context and the
selected finite-security context; the exact selected interactive budget; and
specialization of the complete finite random-oracle soundness theorem to this
key.

Does not own: a concrete Poseidon2 state or codec, random-oracle collision
bounds, Phi81 low-norm invertibility, Ajtai/MSIS binding, Rust, R1CS,
artifacts, or outer-proof security. Each remains an explicit input or a
separate theorem boundary.

Emits constraints: no.

Assurance tier: model-level bridge into the security-reduced NIFS theorem.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityNifs

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.Paper
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityComposition
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySecurity
open MatrixCoefficientSource
open PaperLinearAlgebra

universe uCommitment uPublicInput uState

/-- Exact interactive error budget for the selected one-joint profile. The
two `Pi_CCS` terms are fixed by root counting over the complete quadratic
extension. Only the relaxed-binding terms remain caller supplied. -/
def selectedInteractiveBudget
    (piRlcAlphabetCardinality : Nat)
    (relaxedBindingRaw relaxedBindingRoot : Rat) :
    InteractiveErrorBudget Rat where
  piCcsSumCheck := ratio 216 (goldilocksP * goldilocksP)
  piCcsSchwartzZippel := ratio 10599 (goldilocksP * goldilocksP)
  piRlcForkSampling := ratio 16 piRlcAlphabetCardinality
  relaxedBindingRaw := relaxedBindingRaw
  relaxedBindingRoot := relaxedBindingRoot

/-- Accepted-NIFS child-witness extraction is a separate boundary from the
zero-loss intrinsic `Pi_DEC` reduction. -/
def selectedExtractionBudget (piDecTargetWitnessFailure : Rat) :
    NifsExtractionErrorBudget Rat where
  piDecTargetWitnessFailure := piDecTargetWitnessFailure

/-- Static selected NIFS key. Transcript operations are typed inputs here so
the later Poseidon2 refinement cannot change the protocol dataflow. -/
noncomputable def selectedKey
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {State : Type uState}
    (openingMaps : OpeningMaps Commitment PublicInput assignmentColumns)
    (matrices : ApplicationMatrices)
    (piRlcAlgebra : PiRLC.Algebra
      (MatrixSource F shape assignmentColumns
        (Phi81ColumnLayout.blockCount assignmentColumns))
      (Assignment F assignmentColumns)
      PublicInput
      (CubePoint K shape.cubeVariables)
      (EvaluationFamily K shape)
      Commitment RingF
      (paperRelationSemantics baseOps extensionOps K.embed openingMaps)
      productionGlobalParams)
    (piDecAlgebra : PiDEC.Algebra
      (MatrixSource F shape assignmentColumns
        (Phi81ColumnLayout.blockCount assignmentColumns))
      (Assignment F assignmentColumns)
      PublicInput
      (CubePoint K shape.cubeVariables)
      (EvaluationFamily K shape)
      Commitment
      (paperRelationSemantics baseOps extensionOps K.embed openingMaps)
      productionGlobalParams)
    (piDecPublicInputSplit :
      PiDEC.PaperVerifier.PublicInputSplit piDecAlgebra)
    (piDecEvaluationArity : PiDEC.PaperVerifier.EvaluationArity
      (paperRelationSemantics baseOps extensionOps K.embed openingMaps))
    (piDecEvaluationCount :
      piDecEvaluationArity.count (matrixSource matrices) = 1)
    (oracle : ProtocolVerifier.Oracle K State shape)
    (initialTranscriptState : State)
    (absorbPublicInput : State ->
      Running K Commitment PublicInput shape ->
      Fresh Commitment PublicInput shape -> State)
    (absorbPiCcsOutput : State ->
      FullOutputCoordinates.FullOutput K shape -> State)
    (piRlcResponse : State -> Fin PaperProfile.arity.total -> RingF)
    (piRlcResponseValid : forall state index,
      piRlcAlgebra.challengeValid (piRlcResponse state index)) :
    Key K Commitment PublicInput RingF State shape assignmentColumns
      (Phi81ColumnLayout.blockCount assignmentColumns) 9 where
  baseOps := baseOps
  baseLaws := baseLaws
  baseZero := baseZeroAgreement
  noZeroDivisors :=
    NormRange.baseFieldNoZeroDivisors_of_modulusEuclid
      Nightstream.Implementation.R1CS.Canonical.GoldilocksField.goldilocks_euclidPrime
  extensionOps := extensionOps
  extensionLaws := extensionLaws
  extensionZeroLaws := extensionZeroLaws
  lift := K.embed
  liftLaws := protocolLift
  openingMaps := openingMaps
  params := productionGlobalParams
  freshBound := rfl
  arity := PaperProfile.arity
  freshCount_eq := rfl
  runningCount_eq := rfl
  outputCount_eq := rfl
  kPositive := by decide
  cubeLayout := assignmentLayout
  matrixSource := matrixSource matrices
  degreeBoundExact := by
    rw [ConstraintPolynomialLift.liftConstraintPolynomial_canonicalEqualityGatedDegreeBound]
    change Nat.max
      ((identityFirstSystem matrices).constraintPolynomial).canonicalEqualityGatedDegreeBound
        4 = 9
    rw [identityFirstDegree_exact]
    decide
  matrixCountPositive := by decide
  identityFirstEntry := by
    intro vertex column
    rfl
  constantLaw := Phi81CoefficientKernel.phi81ConstantTermLaw
  challengeSetSize := fullChallengeSupport.cardinality
  piRlcSemantics :=
    paperRelationSemantics baseOps extensionOps K.embed openingMaps
  openingAgreement := by
    intro normBound commitment publicInput assignment
    exact Iff.rfl
  ambientAgreement := by
    intro statement assignment sourceEq
    exact Iff.rfl
  evaluationAgreement := by
    intro assignment point
    exact ⟨True.intro, rfl⟩
  piRlcEvaluationsSize := fun _ _ _ => rfl
  piRlcAlgebra := piRlcAlgebra
  piDecAlgebra := piDecAlgebra
  piDecPublicInputSplit := piDecPublicInputSplit
  piDecEvaluationArity := piDecEvaluationArity
  piDecEvaluationCount := piDecEvaluationCount
  piDecDecision := fun _ => Classical.propDecidable _
  oracle := oracle
  initialTranscriptState := initialTranscriptState
  absorbPublicInput := absorbPublicInput
  absorbPiCcsOutput := absorbPiCcsOutput
  piRlcResponse := piRlcResponse
  piRlcResponseValid := piRlcResponseValid

section ContextBridge

variable
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {State : Type uState}
    (openingMaps : OpeningMaps Commitment PublicInput assignmentColumns)
    (matrices : ApplicationMatrices)
    (piRlcAlgebra : PiRLC.Algebra
      (MatrixSource F shape assignmentColumns
        (Phi81ColumnLayout.blockCount assignmentColumns))
      (Assignment F assignmentColumns)
      PublicInput
      (CubePoint K shape.cubeVariables)
      (EvaluationFamily K shape)
      Commitment RingF
      (paperRelationSemantics baseOps extensionOps K.embed openingMaps)
      productionGlobalParams)
    (piDecAlgebra : PiDEC.Algebra
      (MatrixSource F shape assignmentColumns
        (Phi81ColumnLayout.blockCount assignmentColumns))
      (Assignment F assignmentColumns)
      PublicInput
      (CubePoint K shape.cubeVariables)
      (EvaluationFamily K shape)
      Commitment
      (paperRelationSemantics baseOps extensionOps K.embed openingMaps)
      productionGlobalParams)
    (piDecPublicInputSplit :
      PiDEC.PaperVerifier.PublicInputSplit piDecAlgebra)
    (piDecEvaluationArity : PiDEC.PaperVerifier.EvaluationArity
      (paperRelationSemantics baseOps extensionOps K.embed openingMaps))
    (piDecEvaluationCount :
      piDecEvaluationArity.count (matrixSource matrices) = 1)
    (oracle : ProtocolVerifier.Oracle K State shape)
    (initialTranscriptState : State)
    (absorbPublicInput : State ->
      Running K Commitment PublicInput shape ->
      Fresh Commitment PublicInput shape -> State)
    (absorbPiCcsOutput : State ->
      FullOutputCoordinates.FullOutput K shape -> State)
    (piRlcResponse : State -> Fin PaperProfile.arity.total -> RingF)
    (piRlcResponseValid : forall state index,
      piRlcAlgebra.challengeValid (piRlcResponse state index))

local notation "key" => selectedKey openingMaps matrices piRlcAlgebra
  piDecAlgebra piDecPublicInputSplit piDecEvaluationArity
  piDecEvaluationCount oracle initialTranscriptState absorbPublicInput
  absorbPiCcsOutput piRlcResponse piRlcResponseValid

/-- The NIFS bridge selects exactly the same interactive `Pi_CCS` context as
the finite selected-profile proof. This equality includes the statement,
degree, challenge-space size, and ambient relation. -/
theorem strongExecutionContext_eq_selectedContext
    (running : Running K Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape) :
    (key).strongExecutionContext running fresh =
      selectedContext openingMaps matrices
        (Fin.addCases fresh.commitments running.commitments)
        (Fin.addCases fresh.publicInputs running.publicInputs)
        running.point
        (fun coordinate =>
          running.evaluations coordinate.running coordinate.matrix
            coordinate.coefficient)
        fullChallengeSupport := by
  rfl

/-- The whole adjacent `Pi_CCS`/`Pi_RLC` context is also definitionally the
selected composition context. Thus the NIFS bridge cannot silently change
the source partition or relation semantics. -/
theorem compatibleContext_eq_selectedCompatibleContext
    (running : Running K Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape) :
    (key).compatibleContext running fresh =
      selectedCompatibleContext openingMaps matrices
        (Fin.addCases fresh.commitments running.commitments)
        (Fin.addCases fresh.publicInputs running.publicInputs)
        running.point
        (fun coordinate =>
          running.evaluations coordinate.running coordinate.matrix
            coordinate.coefficient)
        piRlcAlgebra := by
  rfl

/-- The selected profile has seventeen `Pi_RLC` coordinates, so both the
coordinate-fork loss and the finite-oracle programming loss have numerator
eighteen. -/
theorem selectedProgrammingNumerator : (key).arity.total + 1 = 18 := by
  rfl

/-- Headline finite random-oracle reduction for the exact selected key.

This is a security reduction, not an unconditional cryptographic theorem.
The four transcript-collision bounds, accepted-child extraction bound,
`Pi_RLC` fork and binding bounds, and strong-set laws remain explicit. The
selected `Pi_CCS` SumCheck and mixing entries are fixed to the root-counting
values proved in `PaddedRowIdentitySecurity`; no caller can replace them
with smaller values in this theorem. -/
theorem selectedFullOracleSoundness
    [DecidableEq State]
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      (key).nifsPiRlcContext.semantics (key).nifsPiRlcContext.params
      (key).nifsPiRlcContext.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      (key).nifsPiRlcContext.algebra.challengeValid)
    (prefixExperiment : PiCcsPrefixExperiment key)
    (alphabet : Support RingF)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      (key).piRlcAlgebra.challengeValid scalar)
    (piDecTargetWitnessFailure relaxedBindingRaw relaxedBindingRoot : Rat)
    (collisionBudget : PostPrefixCollisionBudget)
    (interactiveContract :
      FullOracleInteractiveResidualContract
        (fullOracleForkMixture prefixExperiment alphabet
          alphabetValid).toProbabilityExperiment
        (selectedExtractionBudget piDecTargetWitnessFailure)
        (selectedInteractiveBudget alphabet.cardinality relaxedBindingRaw
          relaxedBindingRoot))
    (collisionContract :
      FullOracleCollisionContract prefixExperiment alphabet alphabetValid
        collisionBudget) :
    scale.le
      (scale.subtract
        ((fullOracleForkMixture prefixExperiment alphabet
          alphabetValid).probability FullOracleAcceptedOutcome)
        (nonInteractiveTotal scale
          (selectedExtractionBudget piDecTargetWitnessFailure)
          (selectedInteractiveBudget alphabet.cardinality relaxedBindingRaw
            relaxedBindingRoot)
          (postPrefixFiatShamirBudget key alphabet collisionBudget)))
      ((fullOracleForkMixture prefixExperiment alphabet
        alphabetValid).probability FullOracleTransitionOutcome) := by
  exact
    fullOracleMixtureAccepted_probability_sub_total_le_transition laws
      strongSet prefixExperiment alphabet alphabetValid
      (selectedExtractionBudget piDecTargetWitnessFailure)
      (selectedInteractiveBudget alphabet.cardinality relaxedBindingRaw
        relaxedBindingRoot)
      collisionBudget interactiveContract collisionContract

end ContextBridge

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityNifs
