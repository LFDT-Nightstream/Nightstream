import NightstreamFPrime.Lifecycle.PaperAlgebra
import NightstreamFPrime.Lifecycle.Transcript
import NightstreamFPrime.Spec.GoldilocksPrime
import NightstreamFPrime.Spec.Folding.Nifs.PaperProfile
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PrefixLayout
import NightstreamFPrime.Spec.Folding.Nifs

/-!
Owns the one concrete Stage 1 SuperNeo NIFS verifier key. Every algebraic law
field is discharged by a concrete theorem (Goldilocks primality, the Φ₈₁
carrier laws, the Ajtai commitment algebra); the transcript fields are the
Poseidon2 sponge of `Lifecycle.Transcript`. The only inputs are the F′ logical
relation itself (matrices, constraint polynomial, its cube capacity and its
canonical Pad layout) and the verifier-owned Ajtai key. Nothing is a
caller-supplied verifier predicate.
-/

namespace NightstreamFPrime.Lifecycle.ProductionKey

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra

/-- The F′ logical relation as the key consumes it: the Φ₈₁ structure at the
Stage 1 shape and the proof that its carrier fits the selected cube. The
v1.1 Pad matrix comes from the verifier-owned `cubeLayout`; it is not a CCS
matrix supplied by this record. -/
structure LogicalRelation (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns <=
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  system : Phi81Relation.Structure (FullShape logicalWidth publicFits)
  cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth <= 2 ^ cubeVariables

section

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns <=
    Phi81CarrierLayout.carrierWidth logicalWidth}


/-- Per-round sum-check degree bound computed from the lifted constraint
polynomial: `max(D_f + 1, 4)` (paper D.4 with `b = 2`). -/
def degreeBound (relation : LogicalRelation logicalWidth publicFits) : Nat :=
  Nat.max
    (ConstraintPolynomialLift.liftConstraintPolynomial K.embed
      (matrixSource relation.system).constraintPolynomial).canonicalEqualityGatedDegreeBound 4

abbrev KeyType (relation : LogicalRelation logicalWidth publicFits) :=
  Key K PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    RingF Transcript.State productionShape (Phi81CarrierLayout.carrierWidth logicalWidth)
    (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth))
    (degreeBound relation)

/-- The complete public NIFS statement in canonical block order: shared
point, 16 running commitment/public/evaluation groups, then the one fresh
commitment/public group. -/
def publicInputBlocks
    (running : Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) productionShape)
    (fresh : Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) productionShape) :
    List (List F) :=
  [serializePoint running.point] ++
    ((List.finRange productionShape.runningCount).flatMap fun index =>
      [serializeCommitment (running.commitments index),
        serializePublicInput (running.publicInputs index),
        serializeEvaluations (running.evaluations index)]) ++
    ((List.finRange productionShape.freshCount).flatMap fun index =>
      [serializeCommitment (fresh.commitments index),
        serializePublicInput (fresh.publicInputs index)])

/-- Absorb the complete public NIFS statement from its one canonical block
list. -/
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

/-- `ρ_i` for source `i`, squeezed from the post-output state. -/
def piRlcResponse (s : Transcript.State) (index : Fin (Nifs.PaperProfile.arity).total) : RingF :=
  (Transcript.piRlcChallenges s (Nifs.PaperProfile.arity).total).getD index.val ringFZero

theorem piRlcResponse_valid (s : Transcript.State) (index : Fin (Nifs.PaperProfile.arity).total) :
    Phi81Relation.PiRLCAlgebra.Challenge.challengeValid (piRlcResponse s index) := by
  unfold piRlcResponse
  have hlen : index.val < (Transcript.piRlcChallenges s (Nifs.PaperProfile.arity).total).length := by
    rw [Transcript.piRlcChallenges_length]; exact index.isLt
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hlen, Option.getD_some]
  exact Transcript.piRlcChallenges_member s _ _ (List.getElem_mem hlen)

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
  cubeLayout := PrefixLayout.layout cubeVariables
    (Phi81CarrierLayout.carrierWidth logicalWidth) relation.cubeFits
  matrixSource := matrixSource relation.system
  degreeBoundExact := rfl
  constantLaw := Phi81CoefficientKernel.phi81ConstantTermLaw
  challengeSetSize := 5 ^ ringDegree
  piRlcSemantics := semantics ajtai
  openingAgreement := openingAgreement ajtai
  ambientAgreement := ambientAgreement ajtai
    (PrefixLayout.layout cubeVariables
      (Phi81CarrierLayout.carrierWidth logicalWidth) relation.cubeFits)
    relation.system
  evaluationAgreement := by
    intro assignment point
    exact ⟨True.intro, evaluations_eq_paper ajtai
      (PrefixLayout.layout cubeVariables
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
  initialTranscriptState := Transcript.absorb Transcript.initialState Transcript.domainTag
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

/-- The key-owned public-input state is the domain tag followed by the one
canonical public block list. -/
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
        (Transcript.absorb Transcript.initialState Transcript.domainTag)
        (publicInputBlocks running fresh) := by
  rfl

/-- The production oracle initializes PiCCS from the two canonical verifier
input blocks. -/
theorem key_oracle_initialState_eq
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (context : ProtocolVerifier.Statement K Transcript.State productionShape) :
    (key relation ajtai).oracle.transcript.initialState context =
      Transcript.absorbBlocks context.priorState
        (Transcript.verifierInputBlocks context.input) := by
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
