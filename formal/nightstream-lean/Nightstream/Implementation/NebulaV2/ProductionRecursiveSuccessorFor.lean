import Nightstream.Implementation.NebulaV2.ProductionMemorySegmentContinuationRows
import Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveRelationRowsSoundFor
import Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBinding
import Nightstream.Protocol.NebulaV2.ApplicationBatch
import Nightstream.Protocol.NebulaV2.AugmentedLifecycle

/-!
Contract: exact typed successor of one exponent-indexed production recursive
F-prime call.

The successor installs the paper-NIFS output computed from the authenticated
prior running state and exact consumed claim. It starts the application batch
from the decoded authenticated prior application state, preserves immutable
initial state, advances counters, and installs the row-derived memory carry.

Assurance tier: exponent-indexed value-level composition.

Does not own generated application rows, successor placement rows, terminal
verification, recursive-size closure, Rust refinement, or cryptographic
reductions.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.NebulaV2.ProductionRecursiveSuccessorFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ApplicationBatch
open Nightstream.Protocol.NebulaV2.AugmentedLifecycle
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.Protocol.NebulaV2.WasmState
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

local instance concreteKOne : One K := ⟨K.one⟩

abbrev FullShape
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits

noncomputable def nextRunning
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (claim : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables) :
    ProductNifsCodec.RunningFor rowVariables
      (FullShape rowVariables logicalWidth publicFits) :=
  let selected := ProductionProductPiCcsTypedBridgeFor.paperKey candidate
    statementId config artifact
  let fresh := ProductionFieldNativeFullClaim.freshOfValue
    (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
      publicFits).toShape claim
  selected.output claim.recursiveState fresh proof

noncomputable def value
    {Program : Type}
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (claim : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (prior : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    {machine : Machine Program} {program : Program}
    {after : AppStateVector}
    (batch : Batch candidate machine program
      (WasmStateEncoding.decode prior.applicationState) after)
    (outgoing : MemoryCarryCodec.Value) :
    ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits) :=
  { augmentedInvocationIndex := prior.augmentedInvocationIndex + 1
    realApplicationRowCount :=
      prior.realApplicationRowCount + realRowCount batch.rows
    initialApplicationState := prior.initialApplicationState
    applicationState := WasmStateEncoding.encode after
    running := nextRunning candidate statementId config artifact claim proof
    initialMemoryCarry := prior.initialMemoryCarry
    memoryCarry := outgoing }

@[simp] theorem value_invocation
    {Program : Type}
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (claim : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (prior : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    {machine : Machine Program} {program : Program}
    {after : AppStateVector}
    (batch : Batch candidate machine program
      (WasmStateEncoding.decode prior.applicationState) after)
    (outgoing : MemoryCarryCodec.Value) :
    (value candidate statementId config artifact claim proof prior batch
      outgoing).augmentedInvocationIndex =
        prior.augmentedInvocationIndex + 1 := rfl

@[simp] theorem value_running
    {Program : Type}
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (claim : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (prior : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    {machine : Machine Program} {program : Program}
    {after : AppStateVector}
    (batch : Batch candidate machine program
      (WasmStateEncoding.decode prior.applicationState) after)
    (outgoing : MemoryCarryCodec.Value) :
    (value candidate statementId config artifact claim proof prior batch
      outgoing).running =
        nextRunning candidate statementId config artifact claim proof := rfl

/-- Application, carry, and assignment data needed to reconstruct the
recursive successor.  It deliberately contains no memory-challenge authority.
This separation lets the complete recursive manifest derive that authority
before the continuation theorem uses it. -/
structure CoreEvidence
    {Program : Type}
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout
      candidate rowVariables)
    (assignment : Nat -> Nat)
    (headers : ChainHeaders Digest.Value)
    (priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits))
    (claim : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      claim proof)
    (machine : Machine Program) (program : Program) where
  applicationAfter : AppStateVector
  batch : Batch candidate machine program
    (WasmStateEncoding.decode recursive.priorState.applicationState)
    applicationAfter
  continuation : ProductionMemorySegmentContinuationRows.Layout candidate
  continuationValid : continuation.Valid
  continuationIntermediate : continuation.intermediate =
    priorAuthority.ccs.core.batch.frame.memory.boundaries
      (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate))
  outgoing : MemoryCarryCodec.Value
  outgoingParsed : MemoryCarryPublicRows.ParsedColumnsMatch
    continuation.outgoing.reference assignment headers outgoing
  continuationRows : Satisfies
    (ProductionMemorySegmentContinuationRows.rows continuation) assignment
  assignmentCanonical : forall column, assignment column < goldilocksP
  one : assignment 0 = 1
  priorCanonical : recursive.priorState.Canonical headers

/-- Complete continuation evidence.  The authority fields are separate from
the successor inputs because they must be derived from the full recursive
manifest, not selected by the prover. -/
structure Evidence
    {Program : Type}
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout
      candidate rowVariables)
    (assignment : Nat -> Nat)
    (headers : ChainHeaders Digest.Value)
    (priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits))
    (claim : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      claim proof)
    (machine : Machine Program) (program : Program)
    extends CoreEvidence candidate statementId config artifact priorAuthority
      assignment headers priorPrefix claim proof recursive machine program where
  authority : MemoryOpenSegment.Authority
  authorityPlaced : MemoryOpenSegmentSound.AuthorityPlaced
    continuation.opening assignment authority

namespace CoreEvidence

theorem prior_running_is_nifs_input
    {Program : Type} {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout
      candidate rowVariables}
    {assignment : Nat -> Nat} {headers : ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {claim : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      claim proof}
    {machine : Machine Program} {program : Program}
    (evidence : CoreEvidence candidate statementId config artifact
      priorAuthority assignment headers priorPrefix claim proof recursive
      machine program) :
    recursive.priorState.running = claim.recursiveState :=
  recursive.priorAuthorityResult.priorRunningExact

theorem successor_canonical
    {Program : Type} {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout
      candidate rowVariables}
    {assignment : Nat -> Nat} {headers : ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {claim : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      claim proof}
    {machine : Machine Program} {program : Program}
    (evidence : CoreEvidence candidate statementId config artifact
      priorAuthority assignment headers priorPrefix claim proof recursive
      machine program)
    (invocationInRange :
      recursive.priorState.augmentedInvocationIndex + 1 <
        maximumAugmentedInvocations candidate)
    (realRowsInRange :
      recursive.priorState.realApplicationRowCount +
          realRowCount evidence.batch.rows < 2 ^ 18) :
    (value candidate statementId config artifact claim proof
      recursive.priorState evidence.batch evidence.outgoing).Canonical
        headers := by
  refine
    { invocationIndex := invocationInRange
      realApplicationRowCount := realRowsInRange
      initialApplicationState := evidence.priorCanonical.initialApplicationState
      applicationState := ?_
      initialMemoryCarry := evidence.priorCanonical.initialMemoryCarry
      memoryCarry := evidence.outgoingParsed.parserCanonical }
  change (WasmStateEncoding.encode evidence.applicationAfter).Canonical
  rw [WasmStateEncoding.canonical_encode_iff]
  exact evidence.batch.after_valid evidence.priorCanonical.applicationState

end CoreEvidence

namespace Evidence

def memoryDerive
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
    {priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout
      candidate rowVariables}
    {assignment : Nat -> Nat}
    {headers : ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {claim : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      claim proof}
    {machine : Machine Program} {program : Program}
    (evidence : Evidence candidate statementId config artifact priorAuthority
      assignment headers priorPrefix claim proof recursive machine program) :
    ClosedCarry Digest.Value -> Roots Digest.Value -> Nat ->
      ProductState.Challenges K :=
  fun closed precommit activeAccessCount =>
    MemoryOpenSegment.deriveFor (identity candidate) evidence.authority closed
      precommit activeAccessCount

theorem memory_continues_from_rows
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
    {priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout
      candidate rowVariables}
    {assignment : Nat -> Nat}
    {headers : ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {claim : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      claim proof}
    {machine : Machine Program} {program : Program}
    (evidence : Evidence candidate statementId config artifact priorAuthority
      assignment headers priorPrefix claim proof recursive machine program) :
    Continues evidence.memoryDerive headers
      (recursive.memoryResult.semantic
        (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate)))
      (MemoryCarryParser.semanticCarry evidence.outgoing
        evidence.outgoingParsed.parserCanonical.stepIndex) := by
  let finalIndex :=
    Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate)
  have intermediateParsed : MemoryCarryPublicRows.ParsedColumnsMatch
      evidence.continuation.intermediate.reference assignment headers
      (recursive.memoryResult.boundary finalIndex) := by
    rw [evidence.continuationIntermediate]
    exact recursive.memoryResult.boundaryParsed finalIndex
  have continued := ProductionMemorySegmentContinuationRows.sound
    evidence.continuationValid evidence.assignmentCanonical evidence.one
    intermediateParsed evidence.outgoingParsed evidence.authorityPlaced
    evidence.continuationRows
  change Continues evidence.memoryDerive headers
      (recursive.memoryResult.semantic finalIndex)
      (MemoryCarryParser.semanticCarry evidence.outgoing
        evidence.outgoingParsed.parserCanonical.stepIndex)
  rw [recursive.memoryResult.semanticExact finalIndex]
  exact continued

theorem application_before_exact
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
    {priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout
      candidate rowVariables}
    {assignment : Nat -> Nat}
    {headers : ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {claim : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      claim proof}
    {machine : Machine Program} {program : Program}
    (evidence : Evidence candidate statementId config artifact priorAuthority
      assignment headers priorPrefix claim proof recursive machine program) :
    WasmStateEncoding.encode
        (WasmStateEncoding.decode recursive.priorState.applicationState) =
      recursive.priorState.applicationState :=
  WasmStateEncoding.encode_decode _

theorem prior_running_is_nifs_input
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
    {priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout
      candidate rowVariables}
    {assignment : Nat -> Nat}
    {headers : ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {claim : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      claim proof}
    {machine : Machine Program} {program : Program}
    (evidence : Evidence candidate statementId config artifact priorAuthority
      assignment headers priorPrefix claim proof recursive machine program) :
    recursive.priorState.running = claim.recursiveState :=
  evidence.toCoreEvidence.prior_running_is_nifs_input

theorem successor_canonical
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
    {priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout
      candidate rowVariables}
    {assignment : Nat -> Nat}
    {headers : ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {claim : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      claim proof}
    {machine : Machine Program} {program : Program}
    (evidence : Evidence candidate statementId config artifact priorAuthority
      assignment headers priorPrefix claim proof recursive machine program)
    (invocationInRange :
      recursive.priorState.augmentedInvocationIndex + 1 <
        maximumAugmentedInvocations candidate)
    (realRowsInRange :
      recursive.priorState.realApplicationRowCount +
          realRowCount evidence.batch.rows < 2 ^ 18) :
    (value candidate statementId config artifact claim proof
      recursive.priorState evidence.batch evidence.outgoing).Canonical
        headers := by
  exact evidence.toCoreEvidence.successor_canonical invocationInRange
    realRowsInRange

theorem exact_successor
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
    {priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout
      candidate rowVariables}
    {assignment : Nat -> Nat}
    {headers : ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {claim : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      claim proof}
    {machine : Machine Program} {program : Program}
    (evidence : Evidence candidate statementId config artifact priorAuthority
      assignment headers priorPrefix claim proof recursive machine program)
    (invocationInRange :
      recursive.priorState.augmentedInvocationIndex + 1 <
        maximumAugmentedInvocations candidate)
    (realRowsInRange :
      recursive.priorState.realApplicationRowCount +
          realRowCount evidence.batch.rows < 2 ^ 18) :
    let successor := value candidate statementId config artifact claim proof
      recursive.priorState evidence.batch evidence.outgoing
    successor.augmentedInvocationIndex =
        recursive.priorState.augmentedInvocationIndex + 1 /\
      successor.realApplicationRowCount =
        recursive.priorState.realApplicationRowCount +
          realRowCount evidence.batch.rows /\
      successor.initialApplicationState =
        recursive.priorState.initialApplicationState /\
      successor.applicationState =
        WasmStateEncoding.encode evidence.applicationAfter /\
      successor.running =
        nextRunning candidate statementId config artifact claim proof /\
      successor.initialMemoryCarry =
        recursive.priorState.initialMemoryCarry /\
      successor.memoryCarry = evidence.outgoing /\
      recursive.priorState.running = claim.recursiveState /\
      Continues evidence.memoryDerive headers
        (recursive.memoryResult.semantic
          (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate)))
        (MemoryCarryParser.semanticCarry evidence.outgoing
          evidence.outgoingParsed.parserCanonical.stepIndex) /\
      successor.Canonical headers := by
  dsimp only
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl,
    evidence.prior_running_is_nifs_input, evidence.memory_continues_from_rows,
    evidence.successor_canonical invocationInRange realRowsInRange⟩

end Evidence

end Nightstream.Implementation.NebulaV2.ProductionRecursiveSuccessorFor
