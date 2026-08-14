import Nightstream.Implementation.Nebula.Production.FPrime.Recursive.AcceptedRowsFor

/-!
Contract: application continuation of one row-derived recursive F-prime call.

This module derives the exact successor, 28-field challenge authority, and
local invocation theorem from the recursive row result. It does not own the
application compiler or fresh-claim producer rows.

Assurance tier: exponent-indexed recursive-row composition.

Emits constraints: no new rows.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 100000

namespace Nightstream.Implementation.Nebula.ProductionPaperRecursiveAcceptedRowsFor

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ApplicationBatch
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.Protocol.Nebula.WasmState
open Nightstream.Protocol.Nebula.WasmStatement
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

namespace Application

theorem successorRows
    {ProgramType : Type} {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {headers : ChainHeaders Digest.Value}
    {statement : ProductionStatement ProgramType}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    (rows : Rows candidate statementId config artifact headers statement value
      proof) :
    Satisfies
      (ProductionRecursiveSuccessorRowsFor.rows rows.program.successorLayout
        rows.program.fold.priorLayout statementId) rows.assignment := by
  have rowListExact :
      ProductionRecursiveSuccessorRowsFor.rows rows.program.successorLayout
          rows.program.fold.priorLayout rows.program.fold.statementId =
        ProductionRecursiveSuccessorRowsFor.rows rows.program.successorLayout
          rows.program.fold.priorLayout statementId :=
    congrArg
      (fun id => ProductionRecursiveSuccessorRowsFor.rows
        rows.program.successorLayout rows.program.fold.priorLayout id)
      rows.statementIdExact
  rw [← rowListExact]
  exact rows.program.successor_satisfied rows.satisfied

theorem successorPlaced
    {ProgramType : Type} {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {headers : ChainHeaders Digest.Value}
    {statement : ProductionStatement ProgramType}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {rows : Rows candidate statementId config artifact headers statement value
      proof}
    {machine : Machine ProgramType} {programValue : ProgramType}
    (application : Application rows machine programValue) :
    ProductionSuccessorStateBindingRowsFor.Placed
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits) rows.program.successorLayout.successor rows.assignment
      application.successor := by
  rw [successor]
  apply ProductionRecursiveSuccessorRowsFor.rows_imply_successorPlaced_explicit
    application.batch
  · exact rows.program.continuationValid
  · exact rows.program.continuationIntermediate
  · exact application.outgoingParsed
  · exact rows.program.continuation_satisfied rows.satisfied
  · exact rows.assignmentCanonical
  · exact rows.one
  · exact application.priorCanonical
  · exact rows.program.successorValid
  · exact application.applicationPlaced
  · exact rows.nifsOutputAlias
  · exact successorRows rows

noncomputable def authority
    {ProgramType : Type} {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {headers : ChainHeaders Digest.Value}
    {statement : ProductionStatement ProgramType}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {rows : Rows candidate statementId config artifact headers statement value
      proof}
    {machine : Machine ProgramType} {programValue : ProgramType}
    (application : Application rows machine programValue) :
    MemoryOpenSegment.Authority :=
  rows.program.openingAuthority rows.recursive.priorState
    application.successor

theorem authorityPlaced
    {ProgramType : Type} {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {headers : ChainHeaders Digest.Value}
    {statement : ProductionStatement ProgramType}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {rows : Rows candidate statementId config artifact headers statement value
      proof}
    {machine : Machine ProgramType} {programValue : ProgramType}
    (application : Application rows machine programValue) :
    MemoryOpenSegmentSound.AuthorityPlaced
      rows.program.continuationLayout.opening rows.assignment
      application.authority := by
  have priorDigestPlaced :
      ProductionMemoryBatchCcsLinkRowsFor.StateDigestPlaced
        rows.program.fold.priorLayout.ccs rows.assignment
        (ProductionSuccessorStateBinding.outputDigest
          rows.program.fold.statementId rows.recursive.priorState) := by
    rw [rows.statementIdExact]
    exact rows.recursive.priorAuthorityResult.stateDigestPlaced
  have successorPlaced := application.successorPlaced
  exact rows.program.rows_imply_openingAuthorityPlaced
    (logicalWidth := logicalWidth)
    (publicFits := publicFits)
    (prior := rows.recursive.priorState)
    (successor := application.successor)
    priorDigestPlaced successorPlaced
    rows.assignmentCanonical rows.one rows.satisfied

noncomputable def evidence
    {ProgramType : Type} {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {headers : ChainHeaders Digest.Value}
    {statement : ProductionStatement ProgramType}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {rows : Rows candidate statementId config artifact headers statement value
      proof}
    {machine : Machine ProgramType} {programValue : ProgramType}
    (application : Application rows machine programValue) :
    ProductionRecursiveSuccessorFor.Evidence candidate statementId config
      artifact rows.program.fold.priorLayout rows.assignment headers
      rows.priorPrefix value proof rows.recursive machine programValue where
  toCoreEvidence := application.coreEvidence
  authority := application.authority
  authorityPlaced := application.authorityPlaced

noncomputable def supplement
    {ProgramType : Type} {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {headers : ChainHeaders Digest.Value}
    {statement : ProductionStatement ProgramType}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {rows : Rows candidate statementId config artifact headers statement value
      proof}
    {machine : Machine ProgramType} {programValue : ProgramType}
    (application : Application rows machine programValue) :
    ProductionPaperRecursiveInvocationRowsSoundFor.Supplement candidate
      statementId config artifact rows.program.fold.priorLayout rows.assignment
      headers rows.priorPrefix value proof rows.recursive machine programValue
      rows.program.successorLayout where
  evidence := application.evidence
  layoutValid := rows.program.successorValid
  applicationPlaced := application.applicationPlaced
  nifsOutputAlias := rows.nifsOutputAlias
  successorRows := successorRows rows

/-- Exact local recursive invocation with row-derived authority. -/
theorem exactInvocation
    {ProgramType : Type} {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {headers : ChainHeaders Digest.Value}
    {statement : ProductionStatement ProgramType}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {rows : Rows candidate statementId config artifact headers statement value
      proof}
    {machine : Machine ProgramType} {programValue : ProgramType}
    (application : Application rows machine programValue) :
    ProductionPaperRecursiveInvocationRowsSoundFor.ExactInvocation candidate
      statementId config artifact rows.program.fold.priorLayout rows.assignment
      headers rows.priorPrefix value proof rows.recursive machine programValue
      rows.program.successorLayout application.supplement :=
  ProductionPaperRecursiveInvocationRowsSoundFor.exact_of_supplement
    application.supplement

end Application

end Nightstream.Implementation.Nebula.ProductionPaperRecursiveAcceptedRowsFor
