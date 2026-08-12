import Nightstream.Implementation.NebulaV2.Production.FPrime.Recursive.RelationRowsSoundFor

/-!
Contract: exact producer-to-consumer state continuity at the generated
relation exponent.

The producer and consumer match the same complete fresh CCS public word.
Equality of their recomputed Poseidon2 state digests gives complete state
equality or one named field-canonical transcript collision.

Assurance tier: exponent-indexed implementation reduction.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductionPaperStateContinuityFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev FullShape
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits

abbrev Collision
    (candidate : Id) (fullShape : Phi81Relation.Shape)
    (statementId : ProductPoseidon2.StatementId) :=
  ProductionSuccessorStateBinding.FieldCanonicalSuccessorTranscriptCollision
    candidate fullShape statementId

theorem matched_state_digests_equal
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    {carrier : ProductionFieldNativeFullClaim.CcsPublic}
    {batch : ProductionMemoryBatchPoseidonBinding.Batch candidate}
    {statementId : ProductPoseidon2.StatementId}
    {producer consumer : ProductionSuccessorStateBinding.Value candidate
      fullShape}
    (producerMatches : ProductionMemoryBoundCcsPublic.FullMatches carrier
      (ProductionSuccessorStateBinding.outputDigest statementId producer)
      batch)
    (consumerMatches : ProductionMemoryBoundCcsPublic.FullMatches carrier
      (ProductionSuccessorStateBinding.outputDigest statementId consumer)
      batch) :
    ProductionSuccessorStateBinding.outputDigest statementId producer =
      ProductionSuccessorStateBinding.outputDigest statementId consumer :=
  ProductionMemoryBoundCcsPublic.matched_state_eq
    producerMatches.stateMatches consumerMatches.stateMatches

theorem state_equal_or_collision
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {statementId : ProductPoseidon2.StatementId}
    {headers : ChainHeaders Digest.Value}
    {producer consumer : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {carrier : ProductionFieldNativeFullClaim.CcsPublic}
    {batch : ProductionMemoryBatchPoseidonBinding.Batch candidate}
    {layout : ProductionSuccessorStateBindingRowsFor.Layout rowVariables}
    {assignment : Nat -> Nat}
    (producerCanonical : producer.Canonical headers)
    (consumerPlaced : ProductionSuccessorStateBindingRowsFor.Placed
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits)
      layout assignment consumer)
    (assignmentCanonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (producerMatches : ProductionMemoryBoundCcsPublic.FullMatches carrier
      (ProductionSuccessorStateBinding.outputDigest statementId producer)
      batch)
    (consumerMatches : ProductionMemoryBoundCcsPublic.FullMatches carrier
      (ProductionSuccessorStateBinding.outputDigest statementId consumer)
      batch) :
    producer = consumer ∨
      Collision candidate (FullShape rowVariables logicalWidth publicFits)
        statementId := by
  let left : ProductionSuccessorStateBinding.FieldCanonicalState candidate
      (FullShape rowVariables logicalWidth publicFits) :=
    { value := producer
      fieldsCanonical := fun field member =>
        ProductionSuccessorStateBinding.successorFrame_fields_canonical
          producerCanonical member }
  let right : ProductionSuccessorStateBinding.FieldCanonicalState candidate
      (FullShape rowVariables logicalWidth publicFits) :=
    { value := consumer
      fieldsCanonical :=
        ProductionSuccessorStateBindingRowsFor.successorFrame_fields_canonical_of_placed
          (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
            publicFits)
          assignmentCanonical one consumer consumerPlaced }
  have digestsEqual := matched_state_digests_equal producerMatches
    consumerMatches
  exact
    ProductionSuccessorStateBinding.equal_outputDigest_recovers_field_state_or_named_failure
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits).toShape
      left right digestsEqual

theorem state_equal_of_no_collision
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {statementId : ProductPoseidon2.StatementId}
    {headers : ChainHeaders Digest.Value}
    {producer consumer : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {carrier : ProductionFieldNativeFullClaim.CcsPublic}
    {batch : ProductionMemoryBatchPoseidonBinding.Batch candidate}
    {layout : ProductionSuccessorStateBindingRowsFor.Layout rowVariables}
    {assignment : Nat -> Nat}
    (producerCanonical : producer.Canonical headers)
    (consumerPlaced : ProductionSuccessorStateBindingRowsFor.Placed
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits)
      layout assignment consumer)
    (assignmentCanonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (producerMatches : ProductionMemoryBoundCcsPublic.FullMatches carrier
      (ProductionSuccessorStateBinding.outputDigest statementId producer)
      batch)
    (consumerMatches : ProductionMemoryBoundCcsPublic.FullMatches carrier
      (ProductionSuccessorStateBinding.outputDigest statementId consumer)
      batch)
    (noCollision : Not (Collision candidate
      (FullShape rowVariables logicalWidth publicFits) statementId)) :
    producer = consumer := by
  rcases state_equal_or_collision producerCanonical consumerPlaced
      assignmentCanonical one producerMatches consumerMatches with
    equal | collision
  · exact equal
  · exact False.elim (noCollision collision)

end Nightstream.Implementation.NebulaV2.ProductionPaperStateContinuityFor
