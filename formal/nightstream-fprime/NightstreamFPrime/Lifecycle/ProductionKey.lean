import NightstreamFPrime.Lifecycle.PaperAlgebra
import NightstreamFPrime.Lifecycle.Transcript
import NightstreamFPrime.Spec.GoldilocksPrime
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
polynomial is the fixed production selective low-norm gate. Nothing is a
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

/-- The F′ logical relation as the key consumes it: 14 verifier-key matrices
and proof that the completed carrier fits the selected cube. The constraint
polynomial is not a field of this record. SuperNeo v1_1 Pad comes from the
verifier-owned `cubeLayout`; it is not a CCS matrix. -/
structure LogicalRelation (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns <=
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  matrices : Fin productionProfile.ccsMatrices →
    PaperLinearAlgebra.BooleanMatrix F cubeVariables logicalWidth
  cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth <= 2 ^ cubeVariables

/-- Construct the sole production structure. Matrix data can vary with the
verifier key; the selective constraint polynomial cannot. -/
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


/-- The one production SumCheck degree. The verifier-key law below proves
that this fixed profile value equals `max(D_f + 1, 4)` for the relation-owned
selective polynomial. -/
def degreeBound (_relation : LogicalRelation logicalWidth publicFits) : Nat := 9

theorem degreeBound_eq
    (relation : LogicalRelation logicalWidth publicFits) :
    degreeBound relation = 9 := by
  rfl

/-- The fixed exposed degree is exactly the degree computed from the lifted
relation polynomial. -/
theorem derivedDegreeBound_eq
    (relation : LogicalRelation logicalWidth publicFits) :
    Nat.max
      (ConstraintPolynomialLift.liftConstraintPolynomial K.embed
        (matrixSource relation.system).constraintPolynomial
        ).canonicalEqualityGatedDegreeBound 4 = degreeBound relation := by
  unfold degreeBound matrixSource Phi81MatrixSource.source
  rw [ConstraintPolynomialLift.liftConstraintPolynomial_canonicalEqualityGatedDegreeBound]
  change Nat.max
    ProductionRelation.polynomial.canonicalEqualityGatedDegreeBound 4 = 9
  rw [ProductionRelation.polynomial_canonicalEqualityGatedDegreeBound]
  rfl

abbrev KeyType (relation : LogicalRelation logicalWidth publicFits) :=
  Key K PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    RingF Transcript.State productionShape (Phi81CarrierLayout.carrierWidth logicalWidth)
    (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth))
    (degreeBound relation)

/-- The four digest lanes occupy public-input columns 1 through 4. -/
def priorDigestIndex
    (lane : Fin 4) :
    Fin (FullShape logicalWidth publicFits).publicWidth :=
  ⟨lane.val + 1, by
    have laneBound := lane.isLt
    norm_num [FullShape, fullShape, Phi81Relation.Shape.publicWidth,
      publicRingColumns, ringDegree] at laneBound ⊢
    omega⟩

/-- The pilot-recomputed prior-state digest carried by the fresh public
instance. The pilot separately enforces marker 1 and the zero tail. -/
def priorDigest
    (fresh : Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      productionShape) : List F :=
  List.ofFn fun lane : Fin 4 =>
    fresh.publicInputs ⟨0, by decide⟩ (priorDigestIndex lane)

/-- Digest-only PiCCS public statement in canonical block order: the
pilot-recomputed prior digest, then the one fresh commitment and public
input. The complete running statement is bound through the constrained state
digest and is not absorbed again. -/
def publicInputBlocks
    (_running : Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) productionShape)
    (fresh : Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) productionShape) :
    List (List F) :=
  [priorDigest fresh] ++
    ((List.finRange productionShape.freshCount).flatMap fun index =>
      [serializeCommitment (fresh.commitments index),
        serializePublicInput (fresh.publicInputs index)])

/-- Absorb the digest-only PiCCS statement from its canonical block list. -/
def absorbPublicInput (state : Transcript.State)
    (running : Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) productionShape)
    (fresh : Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) productionShape) :
    Transcript.State :=
  Transcript.absorbBlocks state (publicInputBlocks running fresh)

/-- Absorb the complete paper `y′` family after the sum-check. -/
def absorbFullOutput (s : Transcript.State)
    (out : FullOutputCoordinates.FullOutput K productionShape) : Transcript.State :=
  Transcript.absorbBlock s ((List.finRange productionShape.sourceCount).flatMap fun i =>
    ((List.finRange productionShape.coefficientCount).flatMap fun l =>
      serializeK (out.padCoordinate i l)) ++
    ((List.finRange productionShape.matrixCount).flatMap fun j =>
      (List.finRange productionShape.coefficientCount).flatMap fun l =>
        serializeK (out.matrixCoordinate i j l)))

/-- The exact 17-value `ρ` batch from the post-output state, or explicit
sampler shortfall. -/
def piRlcResponse (s : Transcript.State) :
    Option (Fin (Nifs.PaperProfile.arity).total → RingF) :=
  Transcript.PiRlcSampler.piRlcChallenges s
    (Nifs.PaperProfile.arity).total

theorem piRlcResponse_valid
    (s : Transcript.State)
    (response : Fin (Nifs.PaperProfile.arity).total → RingF)
    (success : piRlcResponse s = some response)
    (index : Fin (Nifs.PaperProfile.arity).total) :
    Phi81Relation.PiRLCAlgebra.Challenge.challengeValid (response index) := by
  unfold piRlcResponse Transcript.PiRlcSampler.piRlcChallenges at success
  rw [Option.map_eq_some_iff] at success
  rcases success with ⟨batch, batchEq, responseEq⟩
  rw [← responseEq]
  exact Transcript.PiRlcSampler.piRlcChallenges_member
    (by simpa using batchEq) index

/-- The Stage 1 production NIFS key for one logical relation and one
verifier-owned Ajtai key. -/
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
  challengeSetSize := 5 ^ ringDegree
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
  piDecDecision := fun _ => Classical.propDecidable _
  oracle := Transcript.piCcsOracle
  initialTranscriptState :=
    Transcript.absorb Transcript.initialState Transcript.piCcsDigestDomainTag
  absorbPublicInput := absorbPublicInput
  absorbPiCcsOutput := absorbFullOutput
  piRlcResponse := piRlcResponse
  piRlcResponseValid := piRlcResponse_valid

/-- The production key uses the canonical complete `y′` absorber. This
projection theorem lets circuit coverage proofs keep the rest of the key
opaque. -/
theorem key_absorbPiCcsOutput
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (state : Transcript.State)
    (output : FullOutputCoordinates.FullOutput K productionShape) :
    (key relation ajtai).absorbPiCcsOutput state output =
      absorbFullOutput state output := by
  rfl

/-- The key-owned public-input state is the digest-only PiCCS tag followed by
the prior digest and fresh claim. -/
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
      Transcript.absorbBlocks
        (Transcript.absorb Transcript.initialState
          Transcript.piCcsDigestDomainTag)
        (publicInputBlocks running fresh) := by
  rfl

/-- The verifier input is bound through the pilot digest and the
definitionally shared statement view. The PiCCS oracle therefore starts from
the complete digest-only public-input state without reabsorbing it. -/
theorem key_oracle_initialState_eq
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (context : NightstreamFPrime.Spec.Folding.PiCCS.TranscriptReplay.Statement
      K Transcript.State productionShape) :
    (key relation ajtai).oracle.transcript.initialState context =
      context.priorState := by
  rfl

/-- The production key uses the one additive Poseidon2 PiCCS oracle. -/
theorem key_oracle_eq
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (key relation ajtai).oracle = Transcript.piCcsOracle := by
  rfl

end

end NightstreamFPrime.Lifecycle.ProductionKey
