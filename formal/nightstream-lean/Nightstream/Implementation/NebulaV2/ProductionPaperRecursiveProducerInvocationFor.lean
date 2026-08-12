import Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveInvocationRowsSoundFor
import Nightstream.Implementation.NebulaV2.ProductionFreshClaimProducerFor

/-!
Contract: exact consume-before-produce recursive F-prime call at the generated
relation exponent.

The recursive core consumes claim `i - 1`. Its successor evaluates the
current application batch. A distinct current memory batch then produces
claim `i`. The type keeps the delayed and current batches separate.

Assurance tier: exponent-indexed implementation model.

Does not own generated application or memory rows, compiler refinement,
lifetime induction, terminal verification, recursive-size closure, Rust, or
cryptography.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveProducerInvocationFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ApplicationBatch
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.Protocol.NebulaV2.WasmPublicStatementEncoding
open Nightstream.Protocol.NebulaV2.WasmStatement
open Nightstream.Protocol.NebulaV2.WasmState
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev FullShape
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits

abbrev FreshAssignment
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductPaperAlgebraFor.Assignment rowVariables logicalWidth publicFits

noncomputable def claim
    {Program : Type} (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {priorAuthority :
      ProductionPaperPriorStateAuthorityRowsFor.Layout candidate rowVariables}
    {sourceAssignment : Nat -> Nat} {headers : ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {previous : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority sourceAssignment headers
      priorPrefix previous proof}
    {machine : Machine Program} {program : Program}
    {successorLayout : ProductionRecursiveSuccessorRowsFor.Layout rowVariables}
    (supplement : ProductionPaperRecursiveInvocationRowsSoundFor.Supplement
      candidate statementId config artifact priorAuthority sourceAssignment
      headers priorPrefix previous proof recursive machine program
      successorLayout)
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits) :
    ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits) :=
  ProductionFreshClaimProducerFor.value candidate statementId config
    supplement.successor memory assignment

structure Evidence
    {Program : Type} (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (relationAuthority : ProductionFreshClaimProducerFor.RelationAuthority
      publicFits artifact)
    (statement : ProductionStatement Program)
    (priorAuthority :
      ProductionPaperPriorStateAuthorityRowsFor.Layout candidate rowVariables)
    (sourceAssignment : Nat -> Nat) (headers : ChainHeaders Digest.Value)
    (priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits))
    (previous : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority sourceAssignment headers
      priorPrefix previous proof)
    (machine : Machine Program) (program : Program)
    (successorLayout : ProductionRecursiveSuccessorRowsFor.Layout rowVariables)
    (supplement : ProductionPaperRecursiveInvocationRowsSoundFor.Supplement
      candidate statementId config artifact priorAuthority sourceAssignment
      headers priorPrefix previous proof recursive machine program
      successorLayout)
    {memoryLayout : ProductionMemoryCheckedBatchRows.Layout candidate}
    (currentMemory : ProductionMemoryCheckedBatchRows.Result memoryLayout
      sourceAssignment headers)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits) : Prop where
  statementCanonical :
    (PublicImage.ofStatement statement).DecodesFor (identity candidate) statement
  firstBoundaryCarryExact :
    (memoryLayout.boundaries 0).carry =
      supplement.evidence.continuation.outgoing.carry
  currentApplicationMatched :
    ProductionApplicationBatchBridge.Matches currentMemory
      supplement.evidence.batch
  freshRelation : ProductionFreshClaimProducerFor.FreshRelationWitnessForRows
    statementId config artifact relationAuthority supplement.successor
      currentMemory.suffixBatch assignment sourceAssignment

namespace Evidence

private theorem semanticCarry_congr_value
    {left right : MemoryCarryCodec.Value}
    (same : left = right)
    (leftBound : left.stepIndex < Lifecycle.claimsPerSegment)
    (rightBound : right.stepIndex < Lifecycle.claimsPerSegment) :
    MemoryCarryParser.semanticCarry left leftBound =
      MemoryCarryParser.semanticCarry right rightBound := by
  cases same
  rfl

/-- One assignment and one physical carry alias force the recursive
successor's outgoing value to be the current producer boundary. -/
theorem outgoingValue_eq_firstBoundary
    {Program : Type} {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {relationAuthority : ProductionFreshClaimProducerFor.RelationAuthority
      publicFits artifact}
    {statement : ProductionStatement Program}
    {priorAuthority :
      ProductionPaperPriorStateAuthorityRowsFor.Layout candidate rowVariables}
    {sourceAssignment : Nat -> Nat} {headers : ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {previous : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority sourceAssignment headers
      priorPrefix previous proof}
    {machine : Machine Program} {program : Program}
    {successorLayout : ProductionRecursiveSuccessorRowsFor.Layout rowVariables}
    {supplement : ProductionPaperRecursiveInvocationRowsSoundFor.Supplement
      candidate statementId config artifact priorAuthority sourceAssignment
      headers priorPrefix previous proof recursive machine program
      successorLayout}
    {memoryLayout : ProductionMemoryCheckedBatchRows.Layout candidate}
    {currentMemory : ProductionMemoryCheckedBatchRows.Result memoryLayout
      sourceAssignment headers}
    {assignment : FreshAssignment rowVariables logicalWidth publicFits}
    (evidence : Evidence candidate statementId config artifact relationAuthority statement
      priorAuthority sourceAssignment headers priorPrefix previous proof
      recursive machine program successorLayout supplement currentMemory
      assignment) :
    supplement.evidence.outgoing = currentMemory.boundary 0 := by
  apply MemoryCarryCodec.Value.fieldValue_injective
  funext tag
  calc
    supplement.evidence.outgoing.fieldValue tag =
        sourceAssignment
          (supplement.evidence.continuation.outgoing.carry.fieldColumn tag) :=
      (supplement.evidence.outgoingParsed.placed tag).symm
    _ = sourceAssignment
        ((memoryLayout.boundaries 0).carry.fieldColumn tag) := by
      rw [evidence.firstBoundaryCarryExact]
    _ = (currentMemory.boundary 0).fieldValue tag :=
      (currentMemory.boundaryParsed 0).placed tag

/-- The semantic start of the current producer batch is reconstructed from
the shared recursive assignment. -/
theorem currentMemoryStartsAt
    {Program : Type} {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {relationAuthority : ProductionFreshClaimProducerFor.RelationAuthority
      publicFits artifact}
    {statement : ProductionStatement Program}
    {priorAuthority :
      ProductionPaperPriorStateAuthorityRowsFor.Layout candidate rowVariables}
    {sourceAssignment : Nat -> Nat} {headers : ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {previous : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority sourceAssignment headers
      priorPrefix previous proof}
    {machine : Machine Program} {program : Program}
    {successorLayout : ProductionRecursiveSuccessorRowsFor.Layout rowVariables}
    {supplement : ProductionPaperRecursiveInvocationRowsSoundFor.Supplement
      candidate statementId config artifact priorAuthority sourceAssignment
      headers priorPrefix previous proof recursive machine program
      successorLayout}
    {memoryLayout : ProductionMemoryCheckedBatchRows.Layout candidate}
    {currentMemory : ProductionMemoryCheckedBatchRows.Result memoryLayout
      sourceAssignment headers}
    {assignment : FreshAssignment rowVariables logicalWidth publicFits}
    (evidence : Evidence candidate statementId config artifact relationAuthority statement
      priorAuthority sourceAssignment headers priorPrefix previous proof
      recursive machine program successorLayout supplement currentMemory
      assignment) :
    currentMemory.semantic 0 =
      MemoryCarryParser.semanticCarry supplement.evidence.outgoing
        supplement.evidence.outgoingParsed.parserCanonical.stepIndex := by
  calc
    currentMemory.semantic 0 =
        MemoryCarryParser.semanticCarry (currentMemory.boundary 0)
          (currentMemory.boundaryParsed 0).parserCanonical.stepIndex :=
      currentMemory.semanticExact 0
    _ = MemoryCarryParser.semanticCarry supplement.evidence.outgoing
        supplement.evidence.outgoingParsed.parserCanonical.stepIndex :=
      semanticCarry_congr_value
        evidence.outgoingValue_eq_firstBoundary.symm _ _

end Evidence

structure ExactInvocation
    {Program : Type} (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (relationAuthority : ProductionFreshClaimProducerFor.RelationAuthority
      publicFits artifact)
    (statement : ProductionStatement Program)
    (priorAuthority :
      ProductionPaperPriorStateAuthorityRowsFor.Layout candidate rowVariables)
    (sourceAssignment : Nat -> Nat) (headers : ChainHeaders Digest.Value)
    (priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits))
    (previous : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority sourceAssignment headers
      priorPrefix previous proof)
    (machine : Machine Program) (program : Program)
    (successorLayout : ProductionRecursiveSuccessorRowsFor.Layout rowVariables)
    (supplement : ProductionPaperRecursiveInvocationRowsSoundFor.Supplement
      candidate statementId config artifact priorAuthority sourceAssignment
      headers priorPrefix previous proof recursive machine program
      successorLayout)
    {memoryLayout : ProductionMemoryCheckedBatchRows.Layout candidate}
    (currentMemory : ProductionMemoryCheckedBatchRows.Result memoryLayout
      sourceAssignment headers)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits)
    (evidence : Evidence candidate statementId config artifact relationAuthority statement
      priorAuthority sourceAssignment headers priorPrefix previous proof
      recursive machine program successorLayout supplement currentMemory
      assignment) : Prop where
  previousConsumed :
    ProductionPaperRecursiveInvocationRowsSoundFor.ExactInvocation candidate
      statementId config artifact priorAuthority sourceAssignment headers
      priorPrefix previous proof recursive machine program successorLayout
      supplement
  currentMemoryStartsAfterContinuation : currentMemory.semantic 0 =
    MemoryCarryParser.semanticCarry supplement.evidence.outgoing
      supplement.evidence.outgoingParsed.parserCanonical.stepIndex
  currentPortsExact : ApplicationBatch.accesses supplement.evidence.batch.rows =
    ProductionApplicationBatchBridge.memoryAccesses currentMemory
  nextStateExact :
    (claim candidate statementId config supplement
      currentMemory.suffixBatch assignment).recursiveState =
      supplement.successor.running
  nextMemoryExact :
    (claim candidate statementId config supplement
      currentMemory.suffixBatch assignment).memory = currentMemory.suffixBatch
  nextMemoryBound :
    (claim candidate statementId config supplement
      currentMemory.suffixBatch assignment).MemoryBound
  nextCanonical :
    (claim candidate statementId config supplement
      currentMemory.suffixBatch assignment).Canonical
  nextBundleOpens :
    ProductNifsCodec.codecBundle
        (claim candidate statementId config supplement
          currentMemory.suffixBatch assignment).commitmentBundle =
      ProductCommitmentAlgebra.commit config assignment
  nextPublicExact : ProductionMemoryBoundCcsPublic.FullMatches
    (claim candidate statementId config supplement
      currentMemory.suffixBatch assignment).ccsPublic
    (ProductionSuccessorStateBinding.outputDigest statementId
      supplement.successor)
    currentMemory.suffixBatch
  nextFreshRelationHolds : CCS.Holds
    (ProductPaperAlgebraFor.semantics config) productionGlobalParams
    (ProductionFreshClaimProducerFor.freshStatement candidate statementId config
      artifact supplement.successor currentMemory.suffixBatch
      assignment)
    assignment

theorem exact
    {Program : Type} {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {relationAuthority : ProductionFreshClaimProducerFor.RelationAuthority
      publicFits artifact}
    {statement : ProductionStatement Program}
    {priorAuthority :
      ProductionPaperPriorStateAuthorityRowsFor.Layout candidate rowVariables}
    {sourceAssignment : Nat -> Nat} {headers : ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {previous : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority sourceAssignment headers
      priorPrefix previous proof}
    {machine : Machine Program} {program : Program}
    {successorLayout : ProductionRecursiveSuccessorRowsFor.Layout rowVariables}
    {supplement : ProductionPaperRecursiveInvocationRowsSoundFor.Supplement
      candidate statementId config artifact priorAuthority sourceAssignment
      headers priorPrefix previous proof recursive machine program
      successorLayout}
    {memoryLayout : ProductionMemoryCheckedBatchRows.Layout candidate}
    {currentMemory : ProductionMemoryCheckedBatchRows.Result memoryLayout
      sourceAssignment headers}
    {assignment : FreshAssignment rowVariables logicalWidth publicFits}
    (evidence : Evidence candidate statementId config artifact relationAuthority statement
      priorAuthority sourceAssignment headers priorPrefix previous proof
      recursive machine program successorLayout supplement currentMemory
      assignment) :
    ExactInvocation candidate statementId config artifact relationAuthority statement
      priorAuthority sourceAssignment headers priorPrefix previous proof
      recursive machine program successorLayout supplement currentMemory
      assignment evidence := by
  exact
    { previousConsumed :=
        ProductionPaperRecursiveInvocationRowsSoundFor.exact_of_supplement
          supplement
      currentMemoryStartsAfterContinuation := evidence.currentMemoryStartsAt
      currentPortsExact := evidence.currentApplicationMatched.accesses_exact
      nextStateExact := ProductionFreshClaimProducerFor.value_running
        candidate statementId config supplement.successor
          currentMemory.suffixBatch assignment
      nextMemoryExact := ProductionFreshClaimProducerFor.value_memory
        candidate statementId config supplement.successor
          currentMemory.suffixBatch assignment
      nextMemoryBound := ProductionFreshClaimProducerFor.value_memoryBound
        candidate statementId config supplement.successor
          currentMemory.suffixBatch assignment
      nextCanonical := ProductionFreshClaimProducerFor.value_canonical candidate
        statementId config supplement.successor currentMemory assignment
      nextBundleOpens := ProductionFreshClaimProducerFor.value_bundle_opens
        candidate statementId config supplement.successor
          currentMemory.suffixBatch assignment
      nextPublicExact := ProductionFreshClaimProducerFor.value_ccs_fullMatches
        candidate statementId config supplement.successor
          currentMemory.suffixBatch assignment
      nextFreshRelationHolds :=
        ProductionFreshClaimProducerFor.freshStatement_holds_from_rows candidate
          statementId config artifact relationAuthority supplement.successor
          currentMemory.suffixBatch assignment sourceAssignment
          evidence.freshRelation }

end Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveProducerInvocationFor
