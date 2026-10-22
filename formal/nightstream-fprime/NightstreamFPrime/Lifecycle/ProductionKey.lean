import NightstreamFPrime.Lifecycle.PaperAlgebra
import NightstreamFPrime.Lifecycle.Transcript
import NightstreamFPrime.Spec.GoldilocksPrime
import NightstreamFPrime.Spec.Phi81StrongSet.Cardinality
import NightstreamFPrime.Spec.ProductionRelation
import NightstreamFPrime.Spec.Folding.Nifs.PaperProfile
import NightstreamFPrime.Spec.Folding.PiCCS.CanonicalRowLayout
import NightstreamFPrime.Spec.Folding.Nifs

/-!
Owns the one concrete Stage 1 SuperNeo NIFS verifier key. Every algebraic law
field is discharged by a concrete theorem (Goldilocks primality, the Φ₈₁
carrier laws, the Ajtai commitment algebra); the transcript fields are the
Poseidon2 sponge of `Lifecycle.Transcript`. The only inputs are the F′ logical
matrix family, its cube capacity, and the verifier-owned Ajtai key. The CCS
polynomial is the fixed production low-norm gate. Nothing is a
caller-supplied verifier predicate or polynomial.
-/

namespace NightstreamFPrime.Lifecycle.ProductionKey

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra

structure LogicalRelation (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns <=
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  matrices : Fin productionProfile.ccsMatrices →
    PaperLinearAlgebra.BooleanMatrix F cubeVariables logicalWidth
  cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth <= 2 ^ cubeVariables

def LogicalRelation.system
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns <=
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : LogicalRelation logicalWidth publicFits) :
    Phi81Relation.Structure (FullShape logicalWidth publicFits) where
  matrices := relation.matrices
  constraintPolynomial := ProductionRelation.polynomial

@[simp] theorem LogicalRelation.system_constraintPolynomial
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns <=
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : LogicalRelation logicalWidth publicFits) :
    relation.system.constraintPolynomial = ProductionRelation.polynomial := by
  rfl

section

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns <=
    Phi81CarrierLayout.carrierWidth logicalWidth}


def degreeBound (_relation : LogicalRelation logicalWidth publicFits) : Nat := 8

theorem degreeBound_eq
    (relation : LogicalRelation logicalWidth publicFits) :
    degreeBound relation = 8 := by
  rfl

theorem derivedDegreeBound_eq
    (relation : LogicalRelation logicalWidth publicFits) :
    Nat.max
      (ConstraintPolynomialLift.liftConstraintPolynomial K.embed
        (matrixSource relation.system).constraintPolynomial
        ).canonicalEqualityGatedDegreeBound 4 = degreeBound relation := by
  unfold degreeBound matrixSource Phi81MatrixSource.source
  rw [ConstraintPolynomialLift.liftConstraintPolynomial_canonicalEqualityGatedDegreeBound]
  change Nat.max
    ProductionRelation.polynomial.canonicalEqualityGatedDegreeBound 4 = 8
  rw [ProductionRelation.polynomial_canonicalEqualityGatedDegreeBound]
  rfl

abbrev KeyType (relation : LogicalRelation logicalWidth publicFits) :=
  Key K PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    RingF Transcript.State productionShape (Phi81CarrierLayout.carrierWidth logicalWidth)
    (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth))
    (degreeBound relation)

def priorDigest
    (fresh : Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      productionShape) : List F :=
  decodeHash (fresh.publicInputs ⟨0, by decide⟩)

def publicInputBlocks
    (fresh : Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) productionShape) :
    List (List F) :=
  [priorDigest fresh] ++
    ((List.finRange productionShape.freshCount).flatMap fun index =>
      [serializeCommitment (fresh.commitments index),
        serializePublicInput (fresh.publicInputs index)])

def absorbPublicInput (state : Transcript.State)
    (_running : Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) productionShape)
    (fresh : Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) productionShape) :
    Transcript.State :=
  Transcript.absorb state (publicInputBlocks fresh).flatten

def fullOutputWords (out : FullOutputCoordinates.FullOutput K productionShape) : List F :=
  (List.finRange productionShape.sourceCount).flatMap fun source =>
    serializeEvaluations ⟨out.padCoordinate source, out.matrixCoordinate source⟩

def absorbFullOutput (s : Transcript.State)
    (out : FullOutputCoordinates.FullOutput K productionShape) : Transcript.State :=
  Transcript.absorb s (fullOutputWords out)

def piRlcResponse (state : Transcript.State) :
    Option (Fin (Nifs.PaperProfile.arity).total → RingF) :=
  some (Transcript.PiRlcSampler.piRlcChallenges state (Nifs.PaperProfile.arity).total)

theorem piRlcResponse_valid
    (state : Transcript.State)
    (response : Fin (Nifs.PaperProfile.arity).total → RingF)
    (returned : piRlcResponse state = some response)
    (index : Fin (Nifs.PaperProfile.arity).total) :
    Phi81Relation.PiRLCAlgebra.Challenge.challengeValid (response index) := by
  have same := Option.some.inj returned
  rw [← same]
  exact Transcript.PiRlcSampler.piRlcChallenges_member state _ index

theorem piRlcResponse_ne_none (state : Transcript.State) :
    piRlcResponse state ≠ none := by
  simp [piRlcResponse]

noncomputable def key (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    KeyType relation where
  baseOps := baseOps
  baseLaws := baseLaws
  baseZero := baseZeroAgreement
  noZeroDivisors := GoldilocksPrime.baseFieldNoZeroDivisors
  extensionOps := extensionOps
  extensionLaws := extensionLaws
  extensionZeroLaws := extensionZeroLaws
  lift := K.embed
  liftLaws := protocolLift
  openingMaps := openingMaps ajtai
  params := productionGlobalParams
  freshBound := rfl
  arity := Nifs.PaperProfile.arity
  freshCount_eq := rfl
  runningCount_eq := rfl
  outputCount_eq := rfl
  kPositive := by decide
  cubeLayout := NightstreamFPrime.Spec.Folding.PiCCS.CanonicalRowLayout.layout cubeVariables
    (Phi81CarrierLayout.carrierWidth logicalWidth) relation.cubeFits
  matrixSource := matrixSource relation.system
  degreeBoundExact := derivedDegreeBound_eq relation
  constantLaw := Phi81CoefficientKernel.phi81ConstantTermLaw
  challengeSetSize := productionChallengeSetCardinality
  piRlcSemantics := semantics ajtai
  openingAgreement := openingAgreement ajtai
  ambientAgreement := ambientAgreement ajtai
    (NightstreamFPrime.Spec.Folding.PiCCS.CanonicalRowLayout.layout cubeVariables
      (Phi81CarrierLayout.carrierWidth logicalWidth) relation.cubeFits)
    relation.system
  evaluationAgreement := by
    intro assignment point
    exact ⟨True.intro, evaluations_eq_paper ajtai
      (NightstreamFPrime.Spec.Folding.PiCCS.CanonicalRowLayout.layout cubeVariables
        (Phi81CarrierLayout.carrierWidth logicalWidth) relation.cubeFits)
      relation.system assignment point⟩
  piRlcEvaluationsSize := semantics_evaluations_size ajtai
  piRlcAlgebra := piRlcAlgebra ajtai
  piDecAlgebra := piDecAlgebra ajtai
  piDecPublicInputSplit := publicInputSplit ajtai
  piDecEvaluationArity := evaluationArity ajtai
  piDecEvaluationCount := rfl
  piDecDecision := PaperAlgebra.piDecDecision ajtai
  oracle := Transcript.piCcsOracle
  initialTranscriptState := Transcript.foldInitialState
  absorbPublicInput := absorbPublicInput
  absorbPiCcsOutput := absorbFullOutput
  piRlcResponse := piRlcResponse
  piRlcResponseValid := piRlcResponse_valid

theorem key_challengeSetSize_cardinality
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (key relation ajtai).challengeSetSize =
      Nat.card {value : RingF // Phi81StrongSet.ProductionMember value} := by
  change productionChallengeSetCardinality = _
  exact Phi81StrongSet.productionMember_cardinality.symm

theorem key_bigB_below_half_modulus
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    2 * (key relation ajtai).params.bigB < (key relation ajtai).params.q :=
  production_bigB_below_half_modulus

theorem key_absorbPiCcsOutput
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (state : Transcript.State)
    (output : FullOutputCoordinates.FullOutput K productionShape) :
    (key relation ajtai).absorbPiCcsOutput state output =
      absorbFullOutput state output := by
  rfl

theorem key_publicInputState_eq
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      productionShape)
    (fresh : Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      productionShape) :
    (key relation ajtai).publicInputState running fresh =
      Transcript.absorb Transcript.foldInitialState
        (publicInputBlocks fresh).flatten := by
  rfl

theorem key_oracle_initialState_eq
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (context : NightstreamFPrime.Spec.Folding.PiCCS.TranscriptReplay.Statement
      K Transcript.State productionShape) :
    (key relation ajtai).oracle.transcript.initialState context =
      context.priorState := by
  rfl

theorem key_oracle_eq
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (key relation ajtai).oracle = Transcript.piCcsOracle := by
  rfl

end

end NightstreamFPrime.Lifecycle.ProductionKey
