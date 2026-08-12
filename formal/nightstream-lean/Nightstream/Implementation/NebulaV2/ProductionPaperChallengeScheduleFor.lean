import Nightstream.Implementation.NebulaV2.ProductionPaperBaseInvocationFor
import Nightstream.Implementation.NebulaV2.ProductionRecursiveSuccessorFor

/-!
Contract: local adapter from a closed-carry-indexed authority function to the
protocol-level `Continues` relation.

The concrete challenge transcript contains authority digests that can change
at a segment boundary. The protocol-level `Continues` relation accepts one
derive function. This module transports an explicit authority-equality premise
through that interface.

This module does not derive verifier authority. A function of the closed carry
alone cannot bind the complete prior F-prime state or the successor pre-carry.
The exact base and recursive node modules derive those values from their full
row-owned states. The global lifetime theorem does not use this adapter as an
authority proof.

Assurance tier: exponent-indexed challenge-schedule bridge.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductionPaperChallengeScheduleFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.AugmentedLifecycle
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.Protocol.NebulaV2.WasmState
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

local instance concreteKOne : One K := ⟨K.one⟩

/-- A local compatibility adapter. It is not the V2 global authority schedule
because its input omits the prior full state and successor prefix. -/
abbrev AuthoritySchedule := ClosedCarry Digest.Value ->
  MemoryOpenSegment.Authority

def derive (candidate : Id) (schedule : AuthoritySchedule) :
    ClosedCarry Digest.Value -> Roots Digest.Value -> Nat -> Challenges K :=
  fun closed precommit activeAccessCount =>
    MemoryOpenSegment.deriveFor (identity candidate) (schedule closed) closed
      precommit activeAccessCount

theorem base_open_exact
    (candidate : Id)
    (schedule : AuthoritySchedule)
    (headers : ChainHeaders Digest.Value)
    (opening : ProductionPaperBaseInvocationFor.Opening)
    (authorityExact :
      schedule
          (ProductionPaperBaseInvocationFor.initialClosed
            opening.initialMemoryRoot) = opening.authority) :
    openSegment (derive candidate schedule) headers opening.precommit
        opening.activeAccessCount
        (ProductionPaperBaseInvocationFor.initialClosed
          opening.initialMemoryRoot)
        (ProductionPaperBaseInvocationFor.initialClosed_canOpen
          opening.initialMemoryRoot)
        opening.activeCountInRange opening.initialEndTimestampInRange =
      .active (opening.activeFor candidate headers) := by
  change MemoryOpenSegment.openCarryFor (identity candidate)
      (schedule
        (ProductionPaperBaseInvocationFor.initialClosed
          opening.initialMemoryRoot))
      headers opening.precommit opening.activeAccessCount
      (ProductionPaperBaseInvocationFor.initialClosed opening.initialMemoryRoot)
      (ProductionPaperBaseInvocationFor.initialClosed_canOpen
        opening.initialMemoryRoot)
      opening.activeCountInRange opening.initialEndTimestampInRange =
    .active (opening.activeFor candidate headers)
  rw [authorityExact]
  exact opening.open_exact_for candidate headers

theorem continuation_exact
    {Program : Type}
    {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {priorAuthority :
      ProductionPaperPriorStateAuthorityRowsFor.Layout candidate rowVariables}
    {assignment : Nat -> Nat}
    {headers : ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)}
    {claim : ProductionFieldNativeFullClaim.Value candidate
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      claim proof}
    {machine : Machine Program} {program : Program}
    (evidence : ProductionRecursiveSuccessorFor.Evidence candidate statementId
      config artifact priorAuthority assignment headers priorPrefix claim proof
      recursive machine program)
    (schedule : AuthoritySchedule)
    (authorityExact : forall closed,
      recursive.memoryResult.semantic
          (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate)) =
        .closed closed ->
      schedule closed = evidence.authority) :
    Continues (derive candidate schedule) headers
      (recursive.memoryResult.semantic
        (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate)))
      (MemoryCarryParser.semanticCarry evidence.outgoing
        evidence.outgoingParsed.parserCanonical.stepIndex) := by
  have localContinuation := evidence.memory_continues_from_rows
  apply localContinuation.changeDeriveAtBoundary
  intro closed closedExact precommit activeAccessCount
  rw [ProductionRecursiveSuccessorFor.Evidence.memoryDerive, derive,
    authorityExact closed closedExact]

end Nightstream.Implementation.NebulaV2.ProductionPaperChallengeScheduleFor
