import Nightstream.Implementation.NebulaV2.ProductPaperAlgebraFor
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Verifier

/-!
Contract: executable product-commitment paper NIFS indexed by the exact
augmented-relation exponent.

The generated augmented relation, the paper cube, the SumCheck schedule, and
the NIFS key use one `rowVariables` value. The fixed-25
`ProductConcreteNifs` module is only a reference profile.

The relation artifact contains structural facts about generated data. It does
not contain verifier acceptance, extraction, memory soundness, or execution.

Assurance tier: concrete verifier model.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxHeartbeats 800000
set_option maxRecDepth 30000

namespace Nightstream.Implementation.NebulaV2.ProductConcreteNifsFor

open Nightstream.Implementation.NebulaV2
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
abbrev Commitment := ProductPaperAlgebraFor.Commitment
abbrev PiCcsOutput (rowVariables : Nat) :=
  FullOutputCoordinates.FullOutput K (ProductNifsCodec.shapeFor rowVariables)

/-- Verifier-owned structure extracted from the exact generated augmented
relation. -/
structure RelationArtifact
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) where
  system : Phi81Relation.Structure
    (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)
  cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth <=
    2 ^ (ProductNifsCodec.shapeFor rowVariables).cubeVariables
  degreeBoundExact :
    Nat.max
      (ConstraintPolynomialLift.liftConstraintPolynomial K.embed
        (ProductPaperAlgebraFor.matrixSource system).constraintPolynomial
      ).canonicalEqualityGatedDegreeBound 4 = 9
  identityFirstEntry : forall
      (vertex : BooleanVertex
        (ProductNifsCodec.shapeFor rowVariables).cubeVariables)
      (column : Fin (Phi81CarrierLayout.carrierWidth logicalWidth)),
    (ProductPaperAlgebraFor.matrixSource system).matrices
        ⟨0, by simp [ProductNifsCodec.shapeFor]⟩ vertex column =
      (PrefixLayout.layout
        (ProductNifsCodec.shapeFor rowVariables).cubeVariables
        (Phi81CarrierLayout.carrierWidth logicalWidth) cubeFits
      ).paddedIdentityEntry baseOps.zero baseOps.one vertex column

namespace RelationArtifact

def cubeLayout
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (artifact : RelationArtifact rowVariables logicalWidth publicFits) :
    ColumnLayout (ProductNifsCodec.shapeFor rowVariables).cubeVariables
      (Phi81CarrierLayout.carrierWidth logicalWidth) :=
  PrefixLayout.layout
    (ProductNifsCodec.shapeFor rowVariables).cubeVariables
    (Phi81CarrierLayout.carrierWidth logicalWidth) artifact.cubeFits

end RelationArtifact

/-- Construct the paper NIFS key from both transcript absorbers. Keeping the
post-PiCCS absorber abstract makes its constructor equality cheap and opaque. -/
noncomputable def keyWithAbsorbers
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth
      publicFits operationsShape snapshotShape)
    (artifact : RelationArtifact rowVariables logicalWidth publicFits)
    (publicAbsorber : State ->
      ProductNifsCodec.RunningFor rowVariables
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
          publicFits) ->
      ProductNifsCodec.FreshFor rowVariables
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
          publicFits) -> State)
    (outputAbsorber : State -> PiCcsOutput rowVariables -> State) :
    Key K Commitment
      (ProductPaperAlgebraFor.PublicInput rowVariables logicalWidth publicFits)
      RingF State (ProductNifsCodec.shapeFor rowVariables)
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
  openingMaps := ProductPaperAlgebraFor.openingMaps config
  params := productionGlobalParams
  freshBound := rfl
  arity := PaperProfile.arity
  freshCount_eq := rfl
  runningCount_eq := rfl
  outputCount_eq := rfl
  kPositive := by decide
  cubeLayout := artifact.cubeLayout
  matrixSource := ProductPaperAlgebraFor.matrixSource artifact.system
  degreeBoundExact := artifact.degreeBoundExact
  matrixCountPositive := by simp [ProductNifsCodec.shapeFor]
  identityFirstEntry := artifact.identityFirstEntry
  constantLaw := Phi81CoefficientKernel.phi81ConstantTermLaw
  challengeSetSize := goldilocksModulus * goldilocksModulus
  piRlcSemantics := ProductPaperAlgebraFor.semantics config
  openingAgreement := ProductPaperAlgebraFor.openingAgreement config
  ambientAgreement :=
    ProductPaperAlgebraFor.ambientAgreement config artifact.system
  evaluationAgreement := by
    intro assignment point
    exact
      ⟨True.intro,
        ProductPaperAlgebraFor.evaluations_eq_paper
          config artifact.system assignment point⟩
  piRlcEvaluationsSize :=
    ProductPaperAlgebraFor.semantics_evaluations_size config
  piRlcAlgebra := ProductPaperAlgebraFor.piRlcAlgebra config
  piDecAlgebra := ProductPaperAlgebraFor.piDecAlgebra config
  piDecPublicInputSplit := ProductPaperAlgebraFor.publicInputSplit config
  piDecEvaluationArity := ProductPaperAlgebraFor.evaluationArity config
  piDecEvaluationCount := rfl
  piDecDecision := fun _ => Classical.propDecidable _
  oracle := ProductPoseidon2.oracleFor rowVariables
  initialTranscriptState :=
    ProductPoseidon2.initialStateForStatement statementId
  absorbPublicInput := publicAbsorber
  absorbPiCcsOutput := outputAbsorber
  piRlcResponse := ProductPoseidon2.piRlcResponse
  piRlcResponseValid := ProductPoseidon2.piRlcResponse_valid

/-- Construct the production paper NIFS key at the exact generated-relation
exponent and with the exact complete-output Poseidon2 absorber. -/
noncomputable def keyWithPublicAbsorption
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth
      publicFits operationsShape snapshotShape)
    (artifact : RelationArtifact rowVariables logicalWidth publicFits)
    (publicAbsorber : State ->
      ProductNifsCodec.RunningFor rowVariables
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
          publicFits) ->
      ProductNifsCodec.FreshFor rowVariables
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
          publicFits) -> State) :
    Key K Commitment
      (ProductPaperAlgebraFor.PublicInput rowVariables logicalWidth publicFits)
      RingF State (ProductNifsCodec.shapeFor rowVariables)
      (Phi81CarrierLayout.carrierWidth logicalWidth)
      (Phi81ColumnLayout.blockCount
        (Phi81CarrierLayout.carrierWidth logicalWidth)) 9 :=
  keyWithAbsorbers statementId config artifact publicAbsorber
    (ProductPoseidon2.absorbFullOutputFor rowVariables)

@[simp] theorem keyWithAbsorbers_absorbPiCcsOutput
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth
      publicFits operationsShape snapshotShape)
    (artifact : RelationArtifact rowVariables logicalWidth publicFits)
    (publicAbsorber : State ->
      ProductNifsCodec.RunningFor rowVariables
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
          publicFits) ->
      ProductNifsCodec.FreshFor rowVariables
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
          publicFits) -> State)
    (outputAbsorber : State -> PiCcsOutput rowVariables -> State) :
    (keyWithAbsorbers statementId config artifact publicAbsorber
      outputAbsorber).absorbPiCcsOutput = outputAbsorber := by
  rfl

@[simp] theorem keyWithPublicAbsorption_absorbPublicInput
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth
      publicFits operationsShape snapshotShape)
    (artifact : RelationArtifact rowVariables logicalWidth publicFits)
    (publicAbsorber : State ->
      ProductNifsCodec.RunningFor rowVariables
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
          publicFits) ->
      ProductNifsCodec.FreshFor rowVariables
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
          publicFits) -> State) :
    (keyWithPublicAbsorption statementId config artifact
      publicAbsorber).absorbPublicInput = publicAbsorber := by
  rfl

@[simp] theorem keyWithPublicAbsorption_initialTranscriptState
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth
      publicFits operationsShape snapshotShape)
    (artifact : RelationArtifact rowVariables logicalWidth publicFits)
    (publicAbsorber : State ->
      ProductNifsCodec.RunningFor rowVariables
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
          publicFits) ->
      ProductNifsCodec.FreshFor rowVariables
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
          publicFits) -> State) :
    (keyWithPublicAbsorption statementId config artifact
      publicAbsorber).initialTranscriptState =
        ProductPoseidon2.initialStateForStatement statementId := by
  rfl

@[simp] theorem keyWithPublicAbsorption_lift
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth
      publicFits operationsShape snapshotShape)
    (artifact : RelationArtifact rowVariables logicalWidth publicFits)
    (publicAbsorber : State ->
      ProductNifsCodec.RunningFor rowVariables
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
          publicFits) ->
      ProductNifsCodec.FreshFor rowVariables
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
          publicFits) -> State) :
    (keyWithPublicAbsorption statementId config artifact publicAbsorber).lift =
      K.embed := by
  rfl

@[simp] theorem keyWithPublicAbsorption_matrixSource
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth
      publicFits operationsShape snapshotShape)
    (artifact : RelationArtifact rowVariables logicalWidth publicFits)
    (publicAbsorber : State ->
      ProductNifsCodec.RunningFor rowVariables
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
          publicFits) ->
      ProductNifsCodec.FreshFor rowVariables
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
          publicFits) -> State) :
    (keyWithPublicAbsorption statementId config artifact
      publicAbsorber).matrixSource =
        ProductPaperAlgebraFor.matrixSource artifact.system := by
  rfl

@[simp] theorem keyWithPublicAbsorption_oracle
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth
      publicFits operationsShape snapshotShape)
    (artifact : RelationArtifact rowVariables logicalWidth publicFits)
    (publicAbsorber : State ->
      ProductNifsCodec.RunningFor rowVariables
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
          publicFits) ->
      ProductNifsCodec.FreshFor rowVariables
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
          publicFits) -> State) :
    (keyWithPublicAbsorption statementId config artifact publicAbsorber).oracle =
      ProductPoseidon2.oracleFor rowVariables := by
  rfl

@[simp] theorem keyWithPublicAbsorption_piDecPublicInputSplit
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth
      publicFits operationsShape snapshotShape)
    (artifact : RelationArtifact rowVariables logicalWidth publicFits)
    (publicAbsorber : State ->
      ProductNifsCodec.RunningFor rowVariables
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
          publicFits) ->
      ProductNifsCodec.FreshFor rowVariables
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
          publicFits) -> State) :
    (keyWithPublicAbsorption statementId config artifact
      publicAbsorber).piDecPublicInputSplit =
        ProductPaperAlgebraFor.publicInputSplit config := by
  rfl

set_option maxRecDepth 100000 in
set_option maxHeartbeats 2000000 in
@[simp] theorem keyWithPublicAbsorption_absorbPiCcsOutput
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth
      publicFits operationsShape snapshotShape)
    (artifact : RelationArtifact rowVariables logicalWidth publicFits)
    (publicAbsorber : State ->
      ProductNifsCodec.RunningFor rowVariables
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
          publicFits) ->
      ProductNifsCodec.FreshFor rowVariables
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
          publicFits) -> State) :
    (keyWithPublicAbsorption statementId config artifact
      publicAbsorber).absorbPiCcsOutput =
        ProductPoseidon2.absorbFullOutputFor rowVariables := by
  apply keyWithAbsorbers_absorbPiCcsOutput

end Nightstream.Implementation.NebulaV2.ProductConcreteNifsFor
