import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityCodec
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityConcreteComposition
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityNifs
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentitySamplerSecurity
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Verifier

/-!
Contract: concrete noninteractive key and named security boundaries for
`PaddedRowIdentity`.

Owns: the exact production Phi81 relation, the selected width-8 Poseidon2
transcript, the fixed bounded `Pi_RLC` response, the one-joint NIFS key, and
the definitional bridge to the selected interactive composition.

Does not own: the analytic Phi81 low-norm invertibility theorem, Module-SIS
hardness, Poseidon2 random-oracle security, Rust, generated R1CS rows, or
outer-proof security. These are named premises, not hidden Lean axioms. The
finite bounded-sampler loss is proved separately and imported here.

Emits constraints: no.

Assurance tier: concrete model and security reduction. The key is executable
at the value level. The final probability statement remains conditional only
on explicit cryptographic and extraction contracts.
-/

set_option autoImplicit false
set_option maxHeartbeats 800000
set_option maxRecDepth 30000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteNifs

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.Folding.PiRLC.PaperForkExtraction
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.Paper
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySecurity
open MatrixCoefficientSource
open PaperLinearAlgebra

noncomputable local instance : DecidableEq PaddedRowIdentityPoseidon2.State :=
  Classical.decEq _

namespace Algebra
export PaddedRowIdentityConcreteAlgebra
  (AjtaiKey Commitment PublicInput Assignment openingMaps semantics
    semantics_evaluations_size evaluations_eq_paper piRlcAlgebra piDecAlgebra
    publicInputSplit evaluationArity ambientAgreement openingAgreement)
end Algebra

namespace Extraction
export PaddedRowIdentityConcreteExtraction
  (extractionAlgebra extractionStrongSetUnits)
end Extraction

namespace Composition
export PaddedRowIdentityConcreteComposition
  (compatibleContext piDecContext)
end Composition

namespace Poseidon2
export PaddedRowIdentityPoseidon2
  (State StatementId oracle initialState initialStateForStatement absorbPublicInput
    absorbFullOutput sampleCoefficient samplerSucceeded scalarResponse
    piRlcResponse
    piRlcResponse_valid SamplerAvailable SamplerShortfall
    available_or_shortfall available_excludes_shortfall
    not_available_iff_shortfall samplerSucceeded_eq_true_iff
    samplerSucceeded_eq_false_iff piRlcResponse_refines_of_available
    piRlcResponse_refines_of_no_shortfall)
end Poseidon2

abbrev StatementId := Poseidon2.StatementId

namespace SamplerSecurity
export PaddedRowIdentitySamplerSecurity
  (CandidateTriple IdealThreeRejections Poseidon2IdealSamplerTransfer
    candidateTriple completeSamplerShortfallBound
    idealCandidateTripleExperiment idealCandidateTripleSupport
    idealCandidateTripleSupport_cardinality
    idealCandidateTriple_joint_probability
    idealThreeRejections_probability samplerSecurityTarget
    shortfall_requires_three_rejections threeRejections_probability_eq
    sampleCoefficient_eq_none_iff_threeRejections
    completeSamplerShortfallBound_le_target
    samplerShortfall_probability_le
    samplerShortfall_probability_le_182_bits)
end SamplerSecurity

namespace Generic
export PaddedRowIdentityNifs
  (selectedInteractiveBudget selectedExtractionBudget)
end Generic

/-- Exact production NIFS key. Every algebraic operation and transcript
operation is selected here; a caller supplies the statement identifier, the
verifier-owned Ajtai key, and the application matrix family. -/
noncomputable def key
    (statementId : StatementId)
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices) :
    Key K Algebra.Commitment Algebra.PublicInput RingF Poseidon2.State shape
      assignmentColumns (Phi81ColumnLayout.blockCount assignmentColumns) 9 where
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
  openingMaps := Algebra.openingMaps ajtaiKey
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
  piRlcSemantics := Algebra.semantics ajtaiKey
  openingAgreement := Algebra.openingAgreement ajtaiKey
  ambientAgreement := Algebra.ambientAgreement ajtaiKey matrices
  evaluationAgreement := by
    intro assignment point
    exact ⟨True.intro,
      Algebra.evaluations_eq_paper ajtaiKey matrices assignment point⟩
  piRlcEvaluationsSize := Algebra.semantics_evaluations_size ajtaiKey
  piRlcAlgebra := Algebra.piRlcAlgebra ajtaiKey
  piDecAlgebra := Algebra.piDecAlgebra ajtaiKey
  piDecPublicInputSplit := Algebra.publicInputSplit ajtaiKey
  piDecEvaluationArity := Algebra.evaluationArity ajtaiKey
  piDecEvaluationCount := rfl
  piDecDecision := fun _ => Classical.propDecidable _
  oracle := Poseidon2.oracle
  initialTranscriptState := Poseidon2.initialStateForStatement statementId
  absorbPublicInput := Poseidon2.absorbPublicInput
  absorbPiCcsOutput := Poseidon2.absorbFullOutput
  piRlcResponse := Poseidon2.piRlcResponse
  piRlcResponseValid := Poseidon2.piRlcResponse_valid

/-- Fixed non-authoritative Ajtai value used only to specialize the compact
recursive verifier. The refinement theorem below must show that executable
verification does not read this value. -/
def compactAjtaiKey : Algebra.AjtaiKey :=
  fun _ _ => ringFZero

/-- Fixed non-authoritative matrix value used only to specialize the compact
recursive verifier. The full matrix family remains bound by the statement
identifier and owned by the outer verifier. -/
def compactMatrices : ApplicationMatrices where
  matrices := fun _ _ _ => 0

/-- Fixed executable key for the compact recursive verifier. Only the
statement identifier varies inside the recursive circuit. -/
noncomputable def compactKey (statementId : StatementId) :=
  key statementId compactAjtaiKey compactMatrices

@[simp] theorem key_arity_total
    (statementId : StatementId) (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices) :
    (key statementId ajtaiKey matrices).arity.total = 15 := by
  rfl

theorem key_initialTranscriptState
    (statementId : StatementId)
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices) :
    (key statementId ajtaiKey matrices).initialTranscriptState =
      Poseidon2.initialStateForStatement statementId := by
  simp only [key]

/-- The executable transcript prefix reads the statement identifier and the
public claims. It does not read the Ajtai base or application-matrix entries. -/
theorem key_publicInputState
    (statementId : StatementId)
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape) :
    (key statementId ajtaiKey matrices).publicInputState running fresh =
      Poseidon2.absorbPublicInput
        (Poseidon2.initialStateForStatement statementId) running fresh := by
  rfl

theorem key_oracle
    (statementId : StatementId)
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices) :
    (key statementId ajtaiKey matrices).oracle = Poseidon2.oracle := by
  rfl

theorem key_absorbPiCcsOutput
    (statementId : StatementId)
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (state : Poseidon2.State)
    (output : FullOutputCoordinates.FullOutput K shape) :
    (key statementId ajtaiKey matrices).absorbPiCcsOutput state output =
      Poseidon2.absorbFullOutput state output := by
  simp only [key]

/-- The joint-SumCheck verifier input reads the fixed selected polynomial and
the public running claim. It does not read key or matrix entries. -/
def verifierInput
    (running : Running K Algebra.Commitment Algebra.PublicInput shape) :
    ProtocolPolynomial.VerifierInput K shape where
  constraintPolynomial :=
    ConstraintPolynomialLift.liftConstraintPolynomial K.embed
      (ConstraintPolynomialPrepend.prependIgnoredVariable
        Semantics.polynomial)
  priorPoint := running.point
  claimedCoefficient := fun coordinate =>
    running.evaluations coordinate.running coordinate.matrix
      coordinate.coefficient

theorem key_verifierInput
    (statementId : StatementId)
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape) :
    ((key statementId ajtaiKey matrices).statement running fresh).verifierInput
        (key statementId ajtaiKey matrices).lift =
      verifierInput running := by
  rfl

/-- Key-independent projection of the prover message onto the joint-SumCheck
certificate. The selected kernel and first matrix are protocol constants. -/
def piCcsCertificate
    (proof : Proof K Algebra.Commitment shape 9) :
    ProtocolVerifier.Certificate K shape where
  rounds := fun round => (proof.piCcsRounds round).toMessage
  output := {
    freshMatrixImage := fun source matrix =>
      proof.piCcsOutput.coordinate (freshSourceIndex source) matrix
        Phi81CoefficientKernel.phi81Kernel.constant
    sourceAssignment := fun source =>
      proof.piCcsOutput.coordinate source ⟨0, by decide⟩
        Phi81CoefficientKernel.phi81Kernel.constant
    carriedImage := fun coordinate =>
      proof.piCcsOutput.coordinate (runningSourceIndex coordinate.running)
        coordinate.matrix coordinate.coefficient
  }

theorem key_piCcsCertificate
    (statementId : StatementId)
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape)
    (proof : Proof K Algebra.Commitment shape 9) :
    (key statementId ajtaiKey matrices).piCcsCertificate running fresh proof =
      piCcsCertificate proof := by
  rfl

/-- Complete joint-SumCheck replay from only the statement ID, public claims,
and prover message. -/
def piCcsExecution
    (statementId : StatementId)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape)
    (proof : Proof K Algebra.Commitment shape 9) :=
  let execution :=
    ProtocolVerifier.derive Poseidon2.oracle
      (Poseidon2.absorbPublicInput
        (Poseidon2.initialStateForStatement statementId) running fresh)
      (verifierInput running) (piCcsCertificate proof)
  { execution with
    outgoingState :=
      Poseidon2.absorbFullOutput execution.coins.finalState proof.piCcsOutput }

theorem key_piCcsExecution
    (statementId : StatementId)
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape)
    (proof : Proof K Algebra.Commitment shape 9) :
    (key statementId ajtaiKey matrices).piCcsExecution running fresh proof =
      piCcsExecution statementId running fresh proof := by
  unfold Key.piCcsExecution piCcsExecution
  rw [key_publicInputState, key_verifierInput, key_piCcsCertificate,
    key_oracle]
  simp only [key_absorbPiCcsOutput]

abbrev VerifierKey :=
  Key K Algebra.Commitment Algebra.PublicInput RingF Poseidon2.State shape
    assignmentColumns (Phi81ColumnLayout.blockCount assignmentColumns) 9

/-- Exact post-`Pi_CCS` state from which all selected `Pi_RLC` coefficients
are sampled. -/
def samplerState
    (verifierKey : VerifierKey)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape)
    (proof : Proof K Algebra.Commitment shape 9) : Poseidon2.State :=
  (verifierKey.piCcsExecution running fresh proof).outgoingState

/-- Selected executable verifier. A bounded-sampler shortfall rejects before
the generic paper verifier can consume its total internal response. -/
noncomputable def verify
    (verifierKey : VerifierKey)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape)
    (proof : Proof K Algebra.Commitment shape 9) :
    Option (Running K Algebra.Commitment Algebra.PublicInput shape) :=
  if Poseidon2.samplerSucceeded
      (samplerState verifierKey running fresh proof) then
    Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.verify
      verifierKey running fresh proof
  else
    none

theorem verify_eq_paper_of_samplerAvailable
    (verifierKey : VerifierKey)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape)
    (proof : Proof K Algebra.Commitment shape 9)
    (available : Poseidon2.SamplerAvailable
      (samplerState verifierKey running fresh proof)) :
    verify verifierKey running fresh proof =
      Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.verify
        verifierKey running fresh proof := by
  have succeeded : Poseidon2.samplerSucceeded
      (samplerState verifierKey running fresh proof) = true :=
    (Poseidon2.samplerSucceeded_eq_true_iff _).2 available
  simp [verify, succeeded]

theorem verify_eq_none_of_samplerShortfall
    (verifierKey : VerifierKey)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape)
    (proof : Proof K Algebra.Commitment shape 9)
    (shortfall : Poseidon2.SamplerShortfall
      (samplerState verifierKey running fresh proof)) :
    verify verifierKey running fresh proof = none := by
  have failed : Poseidon2.samplerSucceeded
      (samplerState verifierKey running fresh proof) = false :=
    (Poseidon2.samplerSucceeded_eq_false_iff _).2 shortfall
  simp [verify, failed]

theorem verify_implies_samplerAvailable
    (verifierKey : VerifierKey)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape)
    (proof : Proof K Algebra.Commitment shape 9)
    (result : Running K Algebra.Commitment Algebra.PublicInput shape)
    (accepted : verify verifierKey running fresh proof = some result) :
    Poseidon2.SamplerAvailable
      (samplerState verifierKey running fresh proof) := by
  by_contra unavailable
  have shortfall : Poseidon2.SamplerShortfall
      (samplerState verifierKey running fresh proof) :=
    (Poseidon2.not_available_iff_shortfall _).mp unavailable
  have rejected := verify_eq_none_of_samplerShortfall
    verifierKey running fresh proof shortfall
  rw [rejected] at accepted
  contradiction

/-- Selected-verifier acceptance has the same independent paper transition
or named paper bad event as generic acceptance. Sampler shortfall is not an
accepted bad event because the selected verifier rejects it first. -/
theorem verify_sound
    (verifierKey : VerifierKey)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape)
    (proof : Proof K Algebra.Commitment shape 9)
    (result : Running K Algebra.Commitment Algebra.PublicInput shape)
    (accepted : verify verifierKey running fresh proof = some result) :
    Transition verifierKey running fresh result \/
      BadEvent verifierKey running fresh proof result := by
  have available := verify_implies_samplerAvailable
    verifierKey running fresh proof result accepted
  have paperAccepted :
      Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.verify
          verifierKey running fresh proof = some result := by
    rw [← verify_eq_paper_of_samplerAvailable
      verifierKey running fresh proof available]
    exact accepted
  exact
    Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.verify_sound
      verifierKey running fresh proof result paperAccepted

/-- The joint-SumCheck Boolean is independent of the semantic Ajtai key and
matrix entries. These values remain bound by the statement identifier. -/
theorem piCcsCheck_eq_compact
    (statementId : StatementId)
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape)
    (proof : Proof K Algebra.Commitment shape 9) :
    piCcsCheck (key statementId ajtaiKey matrices) running fresh proof =
      piCcsCheck (compactKey statementId) running fresh proof := by
  unfold compactKey
  unfold piCcsCheck
  rw [key_verifierInput statementId ajtaiKey matrices running fresh,
    key_piCcsExecution statementId ajtaiKey matrices running fresh proof,
    key_piCcsCertificate statementId ajtaiKey matrices running fresh proof,
    key_verifierInput statementId compactAjtaiKey compactMatrices running fresh,
    key_piCcsExecution statementId compactAjtaiKey compactMatrices running fresh proof,
    key_piCcsCertificate statementId compactAjtaiKey compactMatrices running fresh proof]
  simp only [Key.piCcsFixedCertificate, key]

/-- The verifier-computed combined commitment uses only public commitments and
transcript-derived challenges. -/
theorem parent_commitment_eq_compact
    (statementId : StatementId)
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape)
    (proof : Proof K Algebra.Commitment shape 9) :
    ((key statementId ajtaiKey matrices).parent running fresh proof).commitment =
      ((compactKey statementId).parent running fresh proof).commitment := by
  unfold compactKey Key.parent PiRLC.combinedOutput Key.piRlcChallenges
  rw [key_piCcsExecution statementId ajtaiKey matrices running fresh proof,
    key_piCcsExecution statementId compactAjtaiKey compactMatrices running fresh proof]
  simp only [Key.piCcsOutputs, StrongReduction.Statement.publicOutput,
    Key.statement, key, Algebra.piRlcAlgebra]

/-- The verifier-computed combined public input uses only public inputs and
transcript-derived challenges. -/
theorem parent_publicInput_eq_compact
    (statementId : StatementId)
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape)
    (proof : Proof K Algebra.Commitment shape 9) :
    ((key statementId ajtaiKey matrices).parent running fresh proof).publicInput =
      ((compactKey statementId).parent running fresh proof).publicInput := by
  unfold compactKey Key.parent PiRLC.combinedOutput Key.piRlcChallenges
  rw [key_piCcsExecution statementId ajtaiKey matrices running fresh proof,
    key_piCcsExecution statementId compactAjtaiKey compactMatrices running fresh proof]
  simp only [Key.piCcsOutputs, StrongReduction.Statement.publicOutput,
    Key.statement, key, Algebra.piRlcAlgebra]

/-- The verifier-computed parent point is the transcript-derived SumCheck
point. -/
theorem parent_point_eq_compact
    (statementId : StatementId)
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape)
    (proof : Proof K Algebra.Commitment shape 9) :
    ((key statementId ajtaiKey matrices).parent running fresh proof).point =
      ((compactKey statementId).parent running fresh proof).point := by
  unfold compactKey Key.parent PiRLC.combinedOutput
  rw [key_piCcsExecution statementId ajtaiKey matrices running fresh proof,
    key_piCcsExecution statementId compactAjtaiKey compactMatrices running fresh proof]

/-- The verifier-computed combined evaluations use only the complete joint
SumCheck output and transcript-derived challenges. -/
theorem parent_evaluations_eq_compact
    (statementId : StatementId)
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape)
    (proof : Proof K Algebra.Commitment shape 9) :
    ((key statementId ajtaiKey matrices).parent running fresh proof).evaluations =
      ((compactKey statementId).parent running fresh proof).evaluations := by
  unfold compactKey Key.parent PiRLC.combinedOutput Key.piRlcChallenges
  rw [key_piCcsExecution statementId ajtaiKey matrices running fresh proof,
    key_piCcsExecution statementId compactAjtaiKey compactMatrices running fresh proof]
  simp only [Key.piCcsOutputs, Key.piCcsProbe,
    StrongReduction.Statement.publicOutput,
    Key.statement, key, Algebra.piRlcAlgebra]

/-- The operational `Pi_DEC` proposition reads only the public parent fields
and prover child messages. Its result is independent of semantic key data. -/
theorem piDecAccepted_iff_compact
    (statementId : StatementId)
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape)
    (proof : Proof K Algebra.Commitment shape 9) :
    PiDEC.PaperVerifier.Accepted
        (key statementId ajtaiKey matrices).piDecAlgebra
        (key statementId ajtaiKey matrices).piDecEvaluationArity
        ((key statementId ajtaiKey matrices).piDecAttempt running fresh proof) <->
      PiDEC.PaperVerifier.Accepted
        (compactKey statementId).piDecAlgebra
        (compactKey statementId).piDecEvaluationArity
        ((compactKey statementId).piDecAttempt running fresh proof) := by
  constructor
  · intro accepted
    refine {
      parentCombined := by rfl
      parentEvaluationSize := by rfl
      messageEvaluationSize := by intro child; rfl
      commitmentEquation := ?_
      evaluationEquation := ?_
    }
    · simp only [Key.piDecAttempt]
      rw [← parent_commitment_eq_compact statementId ajtaiKey matrices
        running fresh proof]
      simpa only [Key.piDecAttempt, compactKey, key, Algebra.piDecAlgebra] using
        accepted.commitmentEquation
    · simp only [Key.piDecAttempt]
      rw [← parent_evaluations_eq_compact statementId ajtaiKey matrices
        running fresh proof]
      simpa only [Key.piDecAttempt, compactKey, key, Algebra.piDecAlgebra] using
        accepted.evaluationEquation
  · intro accepted
    refine {
      parentCombined := by rfl
      parentEvaluationSize := by rfl
      messageEvaluationSize := by intro child; rfl
      commitmentEquation := ?_
      evaluationEquation := ?_
    }
    · simp only [Key.piDecAttempt]
      rw [parent_commitment_eq_compact statementId ajtaiKey matrices
        running fresh proof]
      simpa only [Key.piDecAttempt, compactKey, key, Algebra.piDecAlgebra] using
        accepted.commitmentEquation
    · simp only [Key.piDecAttempt]
      rw [parent_evaluations_eq_compact statementId ajtaiKey matrices
        running fresh proof]
      simpa only [Key.piDecAttempt, compactKey, key, Algebra.piDecAlgebra] using
        accepted.evaluationEquation

/-- The executable `Pi_DEC` Boolean is independent of semantic key data. -/
theorem piDecCheck_eq_compact
    (statementId : StatementId)
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape)
    (proof : Proof K Algebra.Commitment shape 9) :
    piDecCheck (key statementId ajtaiKey matrices) running fresh proof =
      piDecCheck (compactKey statementId) running fresh proof := by
  unfold piDecCheck
  exact decide_eq_decide.mpr
    (piDecAccepted_iff_compact statementId ajtaiKey matrices running fresh proof)

/-- The accepted running output uses only the checked parent point and public
input plus the prover child commitment and evaluation messages. -/
theorem output_eq_compact
    (statementId : StatementId)
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape)
    (proof : Proof K Algebra.Commitment shape 9) :
    (key statementId ajtaiKey matrices).output running fresh proof =
      (compactKey statementId).output running fresh proof := by
  unfold Key.output
  simp only [Key.piDecAttempt, PiDEC.PaperVerifier.children]
  rw [parent_point_eq_compact statementId ajtaiKey matrices running fresh proof,
    parent_publicInput_eq_compact statementId ajtaiKey matrices running fresh proof]
  simp only [compactKey, key, Algebra.publicInputSplit]

/-- The generic paper verifier has the same result for the full and compact
keys because it does not read the semantic Ajtai key or matrix entries. -/
private theorem paperVerify_eq_compact
    (statementId : StatementId)
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape)
    (proof : Proof K Algebra.Commitment shape 9) :
    Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.verify
        (key statementId ajtaiKey matrices) running fresh proof =
      Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.verify
        (compactKey statementId) running fresh proof := by
  unfold Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.verify
  rw [piCcsCheck_eq_compact statementId ajtaiKey matrices running fresh proof,
    piDecCheck_eq_compact statementId ajtaiKey matrices running fresh proof,
    output_eq_compact statementId ajtaiKey matrices running fresh proof]

/-- Exact executable refinement used by the recursive circuit. The full and
compact verifiers also compute the same sampler state, so both reject the
same bounded-sampler shortfall. -/
theorem verify_eq_compact
    (statementId : StatementId)
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape)
    (proof : Proof K Algebra.Commitment shape 9) :
    verify (key statementId ajtaiKey matrices) running fresh proof =
      verify (compactKey statementId) running fresh proof := by
  have compactExecution :
      (compactKey statementId).piCcsExecution running fresh proof =
        piCcsExecution statementId running fresh proof := by
    unfold compactKey
    exact key_piCcsExecution statementId compactAjtaiKey compactMatrices
      running fresh proof
  unfold verify samplerState
  rw [key_piCcsExecution statementId ajtaiKey matrices running fresh proof,
    compactExecution,
    paperVerify_eq_compact statementId ajtaiKey matrices running fresh proof]

section ContextBridge

variable
    (statementId : StatementId)
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape)

/-- The concrete noninteractive key uses the exact selected interactive
`Pi_CCS` and concrete Phi81 `Pi_RLC` context. -/
theorem compatibleContext_eq_concrete :
    (key statementId ajtaiKey matrices).compatibleContext running fresh =
      Composition.compatibleContext ajtaiKey matrices
        (Fin.addCases fresh.commitments running.commitments)
        (Fin.addCases fresh.publicInputs running.publicInputs)
        running.point
        (fun coordinate =>
          running.evaluations coordinate.running coordinate.matrix
            coordinate.coefficient) := by
  rfl

end ContextBridge

/-! ## Named irreducible security boundaries -/

/-- The analytic number-theory boundary used to prove that distinct selected
`Pi_RLC` challenges differ by a unit in `RingF`. -/
abbrev Phi81LowNormInvertibilityBoundary : Prop :=
  Phi81StrongSet.LowNormInvertibility

/-- Exact Module-SIS/Ajtai binding boundary used by the selected interactive
reduction. The two conjuncts refer to the concrete relation and the concrete
extraction algebra. -/
def ModuleSisBindingBoundary
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (commitments : Fin shape.sourceCount -> Algebra.Commitment)
    (publicInputs : Fin shape.sourceCount -> Algebra.PublicInput)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K)
    (theorem8 : Phi81LowNormInvertibilityBoundary)
    (verifier : PiRLC.PaperWeakFiniteUniform.VerifierData
      (Composition.compatibleContext ajtaiKey matrices commitments publicInputs
        priorPoint claimedCoefficient).piRlc)
    (ops : PiRLC.RelaxedBindingOps Algebra.Assignment Algebra.Commitment RingF)
    (relaxedBindingRaw : Rat) : Prop :=
  let context :=
    Composition.compatibleContext ajtaiKey matrices commitments publicInputs
      priorPoint claimedCoefficient
  let laws := Extraction.extractionAlgebra ajtaiKey
  let strongSet := Extraction.extractionStrongSetUnits ajtaiKey theorem8
  PiRLC.PaperForkCollision.RelaxedBindingLaws
      context.piRlc.semantics context.piRlc.params context.piRlc.algebra laws ops /\
    PiRLC.PaperWeakFiniteUniform.RelaxedBindingSecurity
      (context := context.piRlc) laws strongSet ops verifier relaxedBindingRaw

/-- Exact accepted-child extraction and interactive residual probability
boundary. This type names the PiDEC extraction loss and the Module-SIS binding
losses instead of treating them as algebraic facts. -/
abbrev InteractiveSecurityBoundary
    {statementId : StatementId}
    {ajtaiKey : Algebra.AjtaiKey}
    {matrices : ApplicationMatrices}
    (prefixExperiment :
      PiCcsPrefixExperiment (key statementId ajtaiKey matrices))
    (alphabet : Support RingF)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      (key statementId ajtaiKey matrices).piRlcAlgebra.challengeValid scalar)
    (piDecTargetWitnessFailure relaxedBindingRaw relaxedBindingRoot : Rat) :=
  FullOracleInteractiveResidualContract
    (fullOracleForkMixture prefixExperiment alphabet
      alphabetValid).toProbabilityExperiment
    (Generic.selectedExtractionBudget piDecTargetWitnessFailure)
    (Generic.selectedInteractiveBudget alphabet.cardinality relaxedBindingRaw
      relaxedBindingRoot)

/-- Exact four-collision Poseidon2/random-oracle boundary. The bounded
sampler shortfall is a separate event because it concerns the concrete
finite sampler, not transcript collision resistance. -/
abbrev Poseidon2RandomOracleBoundary
    {statementId : StatementId}
    {ajtaiKey : Algebra.AjtaiKey}
    {matrices : ApplicationMatrices}
    (prefixExperiment :
      PiCcsPrefixExperiment (key statementId ajtaiKey matrices))
    (alphabet : Support RingF)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      (key statementId ajtaiKey matrices).piRlcAlgebra.challengeValid scalar)
    (collisionBudget : PostPrefixCollisionBudget) :=
  FullOracleCollisionContract prefixExperiment alphabet alphabetValid
    collisionBudget

/-- Per-state bounded-sampler refinement boundary. Outside this exact event,
the key response is the canonical three-attempt full-field rejection sampler. -/
def BoundedSamplerSecurityBoundary (state : Poseidon2.State) : Prop :=
  ¬ Poseidon2.SamplerShortfall state

theorem boundedSampler_refines
    {state : Poseidon2.State}
    (boundary : BoundedSamplerSecurityBoundary state) :
    PaddedRowIdentityPoseidon2.ResponseRefinesAt
      PaddedRowIdentityPoseidon2.scalarResponse state :=
  Poseidon2.piRlcResponse_refines_of_no_shortfall boundary

/-! ## Concrete finite-oracle theorem -/

/-- Headline finite random-oracle reduction for the exact concrete key.

Lean supplies the full Phi81 algebra and PiDEC recomposition. The caller must
supply only the analytic unit theorem and explicit probability contracts for
accepted-child extraction, Module-SIS binding, and Poseidon2 collisions. A
production certification must also bound `SamplerShortfall`; this theorem
does not hide that concrete distribution event. -/
theorem concreteFullOracleSoundness
    [DecidableEq Poseidon2.State]
    (statementId : StatementId)
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (theorem8 : Phi81LowNormInvertibilityBoundary)
    (prefixExperiment :
      PiCcsPrefixExperiment (key statementId ajtaiKey matrices))
    (alphabet : Support RingF)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      (key statementId ajtaiKey matrices).piRlcAlgebra.challengeValid scalar)
    (piDecTargetWitnessFailure relaxedBindingRaw relaxedBindingRoot : Rat)
    (collisionBudget : PostPrefixCollisionBudget)
    (interactiveContract :
      FullOracleInteractiveResidualContract
        (fullOracleForkMixture prefixExperiment alphabet
          alphabetValid).toProbabilityExperiment
        (Generic.selectedExtractionBudget piDecTargetWitnessFailure)
        (Generic.selectedInteractiveBudget alphabet.cardinality
          relaxedBindingRaw relaxedBindingRoot))
    (collisionContract :
      FullOracleCollisionContract prefixExperiment alphabet alphabetValid
        collisionBudget) :
    scale.le
      (scale.subtract
        ((fullOracleForkMixture prefixExperiment alphabet
          alphabetValid).probability FullOracleAcceptedOutcome)
        (nonInteractiveTotal scale
          (Generic.selectedExtractionBudget piDecTargetWitnessFailure)
          (Generic.selectedInteractiveBudget alphabet.cardinality
            relaxedBindingRaw relaxedBindingRoot)
          (postPrefixFiatShamirBudget
            (key statementId ajtaiKey matrices) alphabet
            collisionBudget)))
      ((fullOracleForkMixture prefixExperiment alphabet
        alphabetValid).probability FullOracleTransitionOutcome) := by
  let laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      (key statementId ajtaiKey matrices).nifsPiRlcContext.semantics
      (key statementId ajtaiKey matrices).nifsPiRlcContext.params
      (key statementId ajtaiKey matrices).nifsPiRlcContext.algebra :=
    Extraction.extractionAlgebra ajtaiKey
  let strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      (key statementId ajtaiKey matrices).nifsPiRlcContext.algebra.challengeValid := by
    exact Extraction.extractionStrongSetUnits ajtaiKey theorem8
  refine
    fullOracleMixtureAccepted_probability_sub_total_le_transition
      laws strongSet
      prefixExperiment alphabet alphabetValid
      (Generic.selectedExtractionBudget piDecTargetWitnessFailure)
      (Generic.selectedInteractiveBudget alphabet.cardinality
        relaxedBindingRaw relaxedBindingRoot)
      collisionBudget ?_ ?_
  · exact interactiveContract
  · exact collisionContract

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteNifs
