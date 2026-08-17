import Nightstream.Implementation.Nebula.NIFS.Core.PaperAlgebra
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Verifier
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PrefixLayout

/-!
Contract: exact executable paper-NIFS key for Nebula-on-SuperNeo V2.

Assurance tier: concrete verifier model.

Owns the exact 25-variable product-commitment relation, one-fresh and
sixteen-running arity, degree-nine paper verifier, complete Poseidon2
transcript, bounded PiRLC response, and all PiRLC/PiDEC algebra fields.

The generated relation artifact must prove three directly checkable facts:
the 25-variable cube fits its assignment carrier, its lifted polynomial has
exact verifier degree nine, and matrix zero is the padded identity. These are
artifact-refinement obligations. They do not assume NIFS acceptance,
extraction, memory soundness, or execution.

Does not own generated NIFS rows, Module-SIS binding, Poseidon2 random-oracle
security, Rust refinement, the compact terminal proof, or recursive-size
closure.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxHeartbeats 800000
set_option maxRecDepth 30000

namespace Nightstream.Implementation.Nebula.ProductConcreteNifs

open Nightstream.Implementation.Nebula
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources

abbrev State := ProductPoseidon2.State
abbrev StatementId := ProductPoseidon2.StatementId
abbrev Commitment := ProductPaperAlgebra.Commitment

/-- Verifier-owned evidence extracted from the generated V2 relation
artifact. Each field is a concrete structural equality or bound. -/
structure RelationArtifact
    (logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) where
  system : Phi81Relation.Structure
    (ProductPaperAlgebra.FullShape logicalWidth publicFits)
  cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth <=
    2 ^ ProductNifsCodec.shape.cubeVariables
  degreeBoundExact :
    Nat.max
      (ConstraintPolynomialLift.liftConstraintPolynomial K.embed
        (ProductPaperAlgebra.matrixSource system).constraintPolynomial
      ).canonicalEqualityGatedDegreeBound 4 = 9
  identityFirstEntry : forall
      (vertex : BooleanVertex ProductNifsCodec.shape.cubeVariables)
      (column : Fin (Phi81CarrierLayout.carrierWidth logicalWidth)),
    (ProductPaperAlgebra.matrixSource system).matrices
        ⟨0, by decide⟩ vertex column =
      (Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PrefixLayout.layout
        ProductNifsCodec.shape.cubeVariables
        (Phi81CarrierLayout.carrierWidth logicalWidth) cubeFits
      ).paddedIdentityEntry baseOps.zero baseOps.one vertex column

namespace RelationArtifact

def cubeLayout
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (artifact : RelationArtifact logicalWidth publicFits) :
    ColumnLayout ProductNifsCodec.shape.cubeVariables
      (Phi81CarrierLayout.carrierWidth logicalWidth) :=
  Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PrefixLayout.layout
    ProductNifsCodec.shape.cubeVariables
    (Phi81CarrierLayout.carrierWidth logicalWidth) artifact.cubeFits

end RelationArtifact

/-- Build the exact product paper-NIFS key with one selected public-input
absorber. This is the only transcript hook that differs between the reference
V2 profile and the measured production candidates. -/
noncomputable def keyWithPublicAbsorption
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : RelationArtifact logicalWidth publicFits)
    (publicAbsorber : State ->
      ProductNifsCodec.Running
        (ProductPaperAlgebra.FullShape logicalWidth publicFits) ->
      ProductNifsCodec.Fresh
        (ProductPaperAlgebra.FullShape logicalWidth publicFits) -> State) :
    Key K Commitment
      (ProductPaperAlgebra.PublicInput logicalWidth publicFits)
      RingF State ProductNifsCodec.shape
      (Phi81CarrierLayout.carrierWidth logicalWidth)
      (Phi81ColumnLayout.blockCount
        (Phi81CarrierLayout.carrierWidth logicalWidth)) 9 where
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
  openingMaps := ProductPaperAlgebra.openingMaps config
  params := productionGlobalParams
  freshBound := rfl
  arity := PaperProfile.arity
  freshCount_eq := rfl
  runningCount_eq := rfl
  outputCount_eq := rfl
  kPositive := by decide
  cubeLayout := artifact.cubeLayout
  matrixSource := ProductPaperAlgebra.matrixSource artifact.system
  degreeBoundExact := artifact.degreeBoundExact
  matrixCountPositive := by decide
  identityFirstEntry := artifact.identityFirstEntry
  constantLaw := Phi81CoefficientKernel.phi81ConstantTermLaw
  challengeSetSize := goldilocksModulus * goldilocksModulus
  piRlcSemantics := ProductPaperAlgebra.semantics config
  openingAgreement := ProductPaperAlgebra.openingAgreement config
  ambientAgreement := ProductPaperAlgebra.ambientAgreement config artifact.system
  evaluationAgreement := by
    intro assignment point
    exact
      ⟨True.intro,
        ProductPaperAlgebra.evaluations_eq_paper
          config artifact.system assignment point⟩
  piRlcEvaluationsSize := ProductPaperAlgebra.semantics_evaluations_size config
  piRlcAlgebra := ProductPaperAlgebra.piRlcAlgebra config
  piDecAlgebra := ProductPaperAlgebra.piDecAlgebra config
  piDecPublicInputSplit := ProductPaperAlgebra.publicInputSplit config
  piDecEvaluationArity := ProductPaperAlgebra.evaluationArity config
  piDecEvaluationCount := rfl
  piDecDecision := fun _ => Classical.propDecidable _
  oracle := ProductPoseidon2.oracle
  initialTranscriptState := ProductPoseidon2.initialStateForStatement statementId
  absorbPublicInput := publicAbsorber
  absorbPiCcsOutput := ProductPoseidon2.absorbFullOutput
  piRlcResponse := ProductPoseidon2.piRlcResponse
  piRlcResponseValid := ProductPoseidon2.piRlcResponse_valid

@[simp] theorem keyWithPublicAbsorption_absorbPublicInput
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : RelationArtifact logicalWidth publicFits)
    (publicAbsorber : State ->
      ProductNifsCodec.Running
        (ProductPaperAlgebra.FullShape logicalWidth publicFits) ->
      ProductNifsCodec.Fresh
        (ProductPaperAlgebra.FullShape logicalWidth publicFits) -> State) :
    (keyWithPublicAbsorption statementId config artifact
      publicAbsorber).absorbPublicInput = publicAbsorber := by
  rfl

@[simp] theorem keyWithPublicAbsorption_initialTranscriptState
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : RelationArtifact logicalWidth publicFits)
    (publicAbsorber : State ->
      ProductNifsCodec.Running
        (ProductPaperAlgebra.FullShape logicalWidth publicFits) ->
      ProductNifsCodec.Fresh
        (ProductPaperAlgebra.FullShape logicalWidth publicFits) -> State) :
    (keyWithPublicAbsorption statementId config artifact
      publicAbsorber).initialTranscriptState =
        ProductPoseidon2.initialStateForStatement statementId := by
  rfl

@[simp] theorem keyWithPublicAbsorption_lift
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : RelationArtifact logicalWidth publicFits)
    (publicAbsorber : State ->
      ProductNifsCodec.Running
        (ProductPaperAlgebra.FullShape logicalWidth publicFits) ->
      ProductNifsCodec.Fresh
        (ProductPaperAlgebra.FullShape logicalWidth publicFits) -> State) :
    (keyWithPublicAbsorption statementId config artifact publicAbsorber).lift =
      K.embed := by
  rfl

@[simp] theorem keyWithPublicAbsorption_baseOps
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : RelationArtifact logicalWidth publicFits)
    (publicAbsorber : State ->
      ProductNifsCodec.Running
        (ProductPaperAlgebra.FullShape logicalWidth publicFits) ->
      ProductNifsCodec.Fresh
        (ProductPaperAlgebra.FullShape logicalWidth publicFits) -> State) :
    (keyWithPublicAbsorption statementId config artifact
      publicAbsorber).baseOps = baseOps := by
  rfl

@[simp] theorem keyWithPublicAbsorption_matrixSource
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : RelationArtifact logicalWidth publicFits)
    (publicAbsorber : State ->
      ProductNifsCodec.Running
        (ProductPaperAlgebra.FullShape logicalWidth publicFits) ->
      ProductNifsCodec.Fresh
        (ProductPaperAlgebra.FullShape logicalWidth publicFits) -> State) :
    (keyWithPublicAbsorption statementId config artifact
      publicAbsorber).matrixSource =
        ProductPaperAlgebra.matrixSource artifact.system := by
  rfl

@[simp] theorem keyWithPublicAbsorption_oracle
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : RelationArtifact logicalWidth publicFits)
    (publicAbsorber : State ->
      ProductNifsCodec.Running
        (ProductPaperAlgebra.FullShape logicalWidth publicFits) ->
      ProductNifsCodec.Fresh
        (ProductPaperAlgebra.FullShape logicalWidth publicFits) -> State) :
    (keyWithPublicAbsorption statementId config artifact
      publicAbsorber).oracle = ProductPoseidon2.oracle := by
  rfl

set_option maxRecDepth 100000 in
set_option maxHeartbeats 2000000 in
@[simp] theorem keyWithPublicAbsorption_absorbPiCcsOutput
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : RelationArtifact logicalWidth publicFits)
    (publicAbsorber : State ->
      ProductNifsCodec.Running
        (ProductPaperAlgebra.FullShape logicalWidth publicFits) ->
      ProductNifsCodec.Fresh
        (ProductPaperAlgebra.FullShape logicalWidth publicFits) -> State) :
    (keyWithPublicAbsorption statementId config artifact
      publicAbsorber).absorbPiCcsOutput =
        ProductPoseidon2.absorbFullOutput := by
  rfl

/-- Exact reference-V2 paper-NIFS key. The caller supplies only the statement
frame, mandatory product commitment keys and lane layout, and the generated
relation artifact with its three structural checks. -/
noncomputable def key
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : RelationArtifact logicalWidth publicFits) :=
  keyWithPublicAbsorption statementId config artifact
    (ProductPoseidon2.absorbPublicInput 9)

@[simp] theorem key_arity_total
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : RelationArtifact logicalWidth publicFits) :
    (key statementId config artifact).arity.total = 17 := by
  rfl

theorem key_initialTranscriptState
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : RelationArtifact logicalWidth publicFits) :
    (key statementId config artifact).initialTranscriptState =
      ProductPoseidon2.initialStateForStatement statementId := by
  rfl

/- The selected key hands PiCCS to PiRLC only after absorbing the complete
coefficient family with the V2 Poseidon2 output frame. -/
set_option maxRecDepth 100000 in
set_option maxHeartbeats 2000000 in
theorem key_absorbPiCcsOutput
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : RelationArtifact logicalWidth publicFits)
    (state : State)
    (message : FullOutputCoordinates.FullOutput K ProductNifsCodec.shape) :
    (key statementId config artifact).absorbPiCcsOutput state message =
      ProductPoseidon2.absorbFullOutput state message := by
  rfl

theorem key_publicInputState
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : RelationArtifact logicalWidth publicFits)
    (running : ProductNifsCodec.Running
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (fresh : ProductNifsCodec.Fresh
      (ProductPaperAlgebra.FullShape logicalWidth publicFits)) :
    (key statementId config artifact).publicInputState running fresh =
      ProductPoseidon2.absorbPublicInput 9
        (ProductPoseidon2.initialStateForStatement statementId)
        running fresh := by
  rfl

theorem key_matrixSource
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : RelationArtifact logicalWidth publicFits) :
    (key statementId config artifact).matrixSource =
      ProductPaperAlgebra.matrixSource artifact.system := by
  rfl

theorem key_commitment_map
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : RelationArtifact logicalWidth publicFits) :
    (key statementId config artifact).openingMaps.commit =
      ProductCommitmentAlgebra.commit config := by
  rfl

end Nightstream.Implementation.Nebula.ProductConcreteNifs
