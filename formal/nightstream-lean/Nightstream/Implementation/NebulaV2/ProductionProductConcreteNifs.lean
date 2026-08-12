import Nightstream.Implementation.NebulaV2.ProductionProductConcreteNifsKey
import Nightstream.Implementation.NebulaV2.ProductionProductPiCcsPublicBridge

/-!
Contract: candidate-specific executable paper-NIFS key for production.

Owns the production successor of the exact product NIFS key. It preserves all
V2 algebra, relation-artifact, PiCCS, PiRLC, PiDEC, and output-transcript data.
It replaces only public-input absorption with the candidate-specific `NSNF`
frame over the exact paper running and fresh inputs.

Does not own generated row placement, cryptographic security, Rust
refinement, terminal verification, or candidate selection after measurement.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxHeartbeats 200000
set_option maxRecDepth 30000

namespace Nightstream.Implementation.NebulaV2.ProductionProductConcreteNifs

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductionProductPiCcsPublicBridge
open Nightstream.Implementation.NebulaV2.ProductionProductConcreteNifsKey
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- The selected paper key's public-input state is exactly the state proved by
the production PiCCS public-field row bridge. -/
theorem selectedKey_publicInputState
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : RelationArtifact rowVariables logicalWidth publicFits)
    (running : ProductNifsCodec.RunningFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (fresh : ProductNifsCodec.FreshFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)) :
    (selectedKey candidate statementId config artifact).paper.publicInputState
        running fresh =
      successorPublicState statementId candidate 9 running fresh := by
  calc
    (selectedKey candidate statementId config artifact).paper.publicInputState
        running fresh =
        publicAbsorber candidate
          (ProductPoseidon2.initialStateForStatement statementId)
          running fresh :=
      SelectedKey.publicInputState_eq
        (selectedKey candidate statementId config artifact) running fresh
    _ = successorPublicState statementId candidate 9 running fresh := rfl

/-- Exact public-field rows imply the actual production key state. The only
placement premise is equality of physical fields with independent paper
inputs. -/
theorem rows_imply_selectedKey_publicInputState
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : RelationArtifact rowVariables logicalWidth publicFits)
    (running : ProductNifsCodec.RunningFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (fresh : ProductNifsCodec.FreshFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (input : PrefixInput
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (statementIdExact : input.statementId = statementId)
    (wires : PublicWires
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (assignment : Nat -> Nat)
    (residues : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placement : Placement candidate
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits)
      9 running fresh wires assignment)
    (satisfied : Satisfies
      (ProductNifsPublicAbsorptionRowsFor.rows
        (installPublicWires input wires))
      assignment) :
    SymbolicDuplexSemantics.decodedBuilder assignment
        (ProductNifsPublicAbsorptionRowsFor.absorbPublicInput
          (installPublicWires input wires)) =
      (selectedKey candidate statementId config artifact).paper.publicInputState
        running fresh := by
  rw [selectedKey_publicInputState]
  rw [← statementIdExact]
  exact rows_imply_successor_public_state candidate
    (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
      publicFits)
    9 running fresh input wires assignment residues one placement satisfied

end Nightstream.Implementation.NebulaV2.ProductionProductConcreteNifs
