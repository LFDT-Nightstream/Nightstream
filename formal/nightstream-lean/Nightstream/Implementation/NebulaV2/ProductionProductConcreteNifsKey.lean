import Nightstream.Implementation.NebulaV2.ProductConcreteNifsFor
import Nightstream.Implementation.NebulaV2.ProductionProductNifsPublicTranscript
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex

/-!
Contract: candidate-specific executable paper-NIFS key selection.

Owns only the construction of the production paper-NIFS key and the two
constructor-derived equalities for its candidate-specific public absorber and
statement state. The caller cannot select or assume these equalities.

Does not own row placement, transcript-row soundness, generated artifacts,
cryptographic security, Rust refinement, or terminal verification.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxHeartbeats 200000
set_option maxRecDepth 30000

namespace Nightstream.Implementation.NebulaV2.ProductionProductConcreteNifsKey

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductionProductNifsPublicTranscript
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev State := ProductPoseidon2.State
abbrev StatementId := ProductPoseidon2.StatementId
abbrev Commitment := ProductPaperAlgebraFor.Commitment
abbrev RelationArtifact := ProductConcreteNifsFor.RelationArtifact

noncomputable def publicAbsorber
    (candidate : Id) {fullShape : Phi81Relation.Shape} :
    State -> ProductionFieldNativeFullClaim.Running fullShape ->
      ProductionFieldNativeFullClaim.Fresh fullShape -> State :=
  fun state running fresh =>
    Poseidon2Duplex.absorbList ProductPoseidon2.constants
      (publicNifsFields candidate 9 running fresh) state

/-- The exact production paper key with construction-derived evidence for its
only candidate-specific fields. -/
structure SelectedKey
    (candidate : Id) (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth)
    (statementId : StatementId) where
  paper : Key K Commitment
    (ProductPaperAlgebraFor.PublicInput rowVariables logicalWidth publicFits)
    RingF State (ProductNifsCodec.shapeFor rowVariables)
    (Phi81CarrierLayout.carrierWidth logicalWidth)
    (Phi81ColumnLayout.blockCount
      (Phi81CarrierLayout.carrierWidth logicalWidth)) 9
  absorbPublicInput_eq :
    paper.absorbPublicInput = publicAbsorber candidate
  initialTranscriptState_eq :
    paper.initialTranscriptState =
      ProductPoseidon2.initialStateForStatement statementId
  lift_eq : paper.lift = K.embed
  oracle_eq : paper.oracle = ProductPoseidon2.oracleFor rowVariables
  absorbPiCcsOutput_eq : paper.absorbPiCcsOutput =
    ProductPoseidon2.absorbFullOutputFor rowVariables

/-- Construct the candidate-specific production key. The degree is exactly
nine in both the key type and the public frame. -/
noncomputable def selectedKey
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : RelationArtifact rowVariables logicalWidth publicFits) :
    SelectedKey candidate rowVariables logicalWidth publicFits statementId where
  paper := ProductConcreteNifsFor.keyWithPublicAbsorption statementId config
    artifact (publicAbsorber candidate)
  absorbPublicInput_eq :=
    ProductConcreteNifsFor.keyWithPublicAbsorption_absorbPublicInput
      statementId config artifact (publicAbsorber candidate)
  initialTranscriptState_eq :=
    ProductConcreteNifsFor.keyWithPublicAbsorption_initialTranscriptState
      statementId config artifact (publicAbsorber candidate)
  lift_eq := ProductConcreteNifsFor.keyWithPublicAbsorption_lift
    statementId config artifact (publicAbsorber candidate)
  oracle_eq := ProductConcreteNifsFor.keyWithPublicAbsorption_oracle
    statementId config artifact (publicAbsorber candidate)
  absorbPiCcsOutput_eq :=
    ProductConcreteNifsFor.keyWithPublicAbsorption_absorbPiCcsOutput
      statementId config artifact (publicAbsorber candidate)

/-- The selected key uses the exact generated relation matrix source. -/
theorem selectedKey_matrixSource
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : RelationArtifact rowVariables logicalWidth publicFits) :
    (selectedKey candidate statementId config artifact).paper.matrixSource =
      ProductPaperAlgebraFor.matrixSource artifact.system := by
  change
    (ProductConcreteNifsFor.keyWithPublicAbsorption statementId config artifact
      (publicAbsorber candidate)).matrixSource =
        ProductPaperAlgebraFor.matrixSource artifact.system
  exact ProductConcreteNifsFor.keyWithPublicAbsorption_matrixSource
    statementId config artifact (publicAbsorber candidate)

/-- Any selected key computes its public state with the absorber and initial
state stored in its construction certificate. This theorem does not unfold
the large paper key. -/
theorem SelectedKey.publicInputState_eq
    {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {statementId : StatementId}
    (selected : SelectedKey candidate rowVariables logicalWidth publicFits
      statementId)
    (running : ProductNifsCodec.RunningFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (fresh : ProductNifsCodec.FreshFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)) :
    selected.paper.publicInputState running fresh =
      publicAbsorber candidate
        (ProductPoseidon2.initialStateForStatement statementId)
        running fresh := by
  unfold Key.publicInputState
  rw [selected.initialTranscriptState_eq, selected.absorbPublicInput_eq]

/-- Expanded form of the selected public-input state. This keeps downstream
proofs from unfolding the complete paper key. -/
theorem SelectedKey.publicInputState_eq_absorbList
    {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {statementId : StatementId}
    (selected : SelectedKey candidate rowVariables logicalWidth publicFits
      statementId)
    (running : ProductNifsCodec.RunningFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (fresh : ProductNifsCodec.FreshFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)) :
    selected.paper.publicInputState running fresh =
      Poseidon2Duplex.absorbList ProductPoseidon2.constants
        (publicNifsFields candidate 9 running fresh)
        (ProductPoseidon2.initialStateForStatement statementId) := by
  rw [selected.publicInputState_eq]
  rfl

end Nightstream.Implementation.NebulaV2.ProductionProductConcreteNifsKey
