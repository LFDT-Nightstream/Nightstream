import Nightstream.Implementation.NebulaV2.ProductionBaseChallengeAuthorityRowsFor

/-!
Contract: semantic soundness of the base memory-challenge authority rows.

This module is separate from the row manifest so Lean checks the two large
Poseidon2 row programs once. It derives each dynamic digest lane and then the
complete 28-field authority placed in the segment-open transcript.

Does not own typed source placement, generated-artifact construction,
cryptographic collision bounds, external bytes, or Rust refinement.

Assurance tier: exponent-indexed row soundness.

Emits constraints: no new rows.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

namespace Nightstream.Implementation.NebulaV2.ProductionBaseChallengeAuthorityRowsFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

theorem Program.rows_imply_initialStateAuthorityLane
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    {initial : ProductionSuccessorStateBinding.Value candidate
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)}
    (initialPlaced : ProductionSuccessorStateBindingRowsFor.Placed
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits) program.initialLayout assignment initial)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies program.rows assignment)
    (lane : Fin 4) :
    assignment
        (program.openingLayout.transcript.frame.authorityColumn
          (initialStateAuthorityPosition lane)) =
      (ProductionSuccessorStateBinding.outputDigest program.statementId
        initial lane).val := by
  have dynamic := program.rows_imply_dynamicAuthorityExact canonical one
    satisfied
  have digestExact :=
    ProductionSuccessorStateBindingRowsFor.rows_imply_outputDigest_lane
      (contract := ProductPaperAlgebraFor.fullShapeContract rowVariables
        logicalWidth publicFits)
      (layout := program.initialLayout) (assignment := assignment)
      canonical one program.statementId initial initialPlaced
      program.initialHashBase (program.initialHash_satisfied satisfied) lane
  exact (dynamic.initialState lane).trans digestExact

theorem Program.rows_imply_preCarryAuthorityLane
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    {successor : ProductionSuccessorStateBinding.Value candidate
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)}
    (successorPlaced : ProductionSuccessorStateBindingRowsFor.Placed
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits) program.successorLayout assignment successor)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies program.rows assignment)
    (lane : Fin 4) :
    assignment
        (program.openingLayout.transcript.frame.authorityColumn
          (preCarryAuthorityPosition lane)) =
      (ProductionSuccessorStateBinding.preCarryDigest program.statementId
        successor.preCarry lane).val := by
  have dynamic := program.rows_imply_dynamicAuthorityExact canonical one
    satisfied
  have sourceRows : Satisfies
      (ProductionSuccessorStateBindingRowsFor.rows candidate
        program.successorHashBase program.successorLayout program.statementId)
      assignment :=
    program.successorHash_satisfied satisfied
  have digestRows : Satisfies
      (ProductionPreCarryDigestRowsFor.rows candidate program.preCarryLayout
        program.statementId) assignment :=
    program.preCarryDigest_satisfied satisfied
  have digestExact := ProductionPreCarryDigestRowsFor.rows_imply_digest_lane
    (contract := ProductPaperAlgebraFor.fullShapeContract rowVariables
      logicalWidth publicFits)
    (layout := program.preCarryLayout) (assignment := assignment)
    canonical one program.statementId successor successorPlaced sourceRows
    digestRows lane
  exact (dynamic.preCarry lane).trans digestExact

/-- Satisfying rows derive the complete 28-field base authority. No authority
record, digest equality, challenge value, or F-prime transition is assumed. -/
theorem Program.rows_imply_openingAuthorityPlaced
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    {initial successor : ProductionSuccessorStateBinding.Value candidate
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)}
    (initialPlaced : ProductionSuccessorStateBindingRowsFor.Placed
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits) program.initialLayout assignment initial)
    (successorPlaced : ProductionSuccessorStateBindingRowsFor.Placed
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits) program.successorLayout assignment successor)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies program.rows assignment) :
    MemoryOpenSegmentSound.AuthorityPlaced program.openingLayout assignment
      (program.openingAuthority initial successor) := by
  apply MemoryOpenSegmentSound.authorityPlaced_of_lanes
  intro digest lane
  fin_cases digest
  · exact program.rows_imply_staticAuthorityLane canonical one satisfied
      .verifierKey lane
  · exact program.rows_imply_staticAuthorityLane canonical one satisfied
      .applicationRelation lane
  · exact program.rows_imply_staticAuthorityLane canonical one satisfied
      .program lane
  · exact program.rows_imply_staticAuthorityLane canonical one satisfied
      .memoryPlan lane
  · exact program.rows_imply_staticAuthorityLane canonical one satisfied
      .laneLayout lane
  · exact program.rows_imply_initialStateAuthorityLane initialPlaced canonical
      one satisfied lane
  · exact program.rows_imply_preCarryAuthorityLane successorPlaced canonical
      one satisfied lane

end Nightstream.Implementation.NebulaV2.ProductionBaseChallengeAuthorityRowsFor
