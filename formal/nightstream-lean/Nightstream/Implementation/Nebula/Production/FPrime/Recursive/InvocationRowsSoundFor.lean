import Nightstream.Implementation.Nebula.Production.FPrime.Recursive.RelationRowsSoundFor
import Nightstream.Implementation.Nebula.Production.FPrime.Recursive.RecursiveSuccessorRowsFor

/-!
Contract: exact local result of one exponent-indexed recursive F-prime
invocation.

The core authenticates the complete prior state, verifies and consumes one
complete fresh claim, and checks its delayed memory batch. The successor then
installs the exact application result, paper-NIFS output, continued memory
carry, counters, and state hash. Thus the consumed claim is the prior claim;
the successor is the state from which the next claim is produced.

Assurance tier: exponent-indexed local F-prime composition.

Does not own generated application rows, the later fresh-claim producer, base
or terminal branches, global adjacency, recursive-size closure, external
bytes, Rust refinement, or cryptographic reductions.

Emits constraints: no; it composes named row sections.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

namespace Nightstream.Implementation.Nebula.ProductionPaperRecursiveInvocationRowsSoundFor

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ApplicationBatch
open Nightstream.Protocol.Nebula.AugmentedLifecycle
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.ProductState
open Nightstream.Protocol.Nebula.ProductionBatchedFPrime
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.Protocol.Nebula.WasmState
open Nightstream.Protocol.Nebula.WasmStatement
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

local instance concreteKOne : One K := ⟨K.one⟩

abbrev FullShape := ProductionPaperRecursiveRelationRowsSoundFor.FullShape
abbrev ProtocolSchema :=
  ProductionPaperRecursiveRelationRowsSoundFor.ProtocolSchema

noncomputable def fresh
    (candidate : Id) {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)) :=
  ProductionFieldNativeFullClaim.freshOfValue
    (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
      publicFits).toShape value

noncomputable def wires
    (candidate : Id) (rowVariables : Nat) {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout
      candidate rowVariables)
    (baseWires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables) :=
  ProductionPaperRecursiveRelationRowsSoundFor.boundWires
    (logicalWidth := logicalWidth) (publicFits := publicFits) candidate
    rowVariables priorAuthority.ccs.carrier baseWires

noncomputable def sampleInput
    (candidate : Id) {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout
      candidate rowVariables)
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (baseWires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables)
    (samplerBase : Nat) :=
  ProductionProductPiRlcParentBridgeFor.samplerInput candidate statementId
    config artifact value.recursiveState (fresh candidate value)
    (wires (logicalWidth := logicalWidth) (publicFits := publicFits) candidate
      rowVariables priorAuthority baseWires) samplerBase

structure CoreChecks
    (candidate : Id) {rowVariables logicalWidth : Nat}
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
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (baseWires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables)
    (samplerBase : Nat) : Prop where
  samplerSucceeded :
    ProductPoseidon2.samplerSucceeded
        (ProductPiRlcFirstAcceptedBatchSound.samplerState
          (sampleInput candidate statementId config artifact priorAuthority
            value baseWires samplerBase) assignment) = true
  piCcsAccepted :
    piCcsCheck
      (ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
        config artifact) value.recursiveState (fresh candidate value) proof =
      true
  piDecAccepted :
    piDecCheck
      (ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
        config artifact) value.recursiveState (fresh candidate value) proof =
      true
  verifierOutputExact :
    verify
      (ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
        config artifact) value.recursiveState (fresh candidate value) proof =
      some
        ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
          config artifact).output value.recursiveState (fresh candidate value)
            proof)

structure Supplement
    {Program : Type} (candidate : Id)
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
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      value proof)
    (machine : Machine Program) (program : Program)
    (layout : ProductionRecursiveSuccessorRowsFor.Layout rowVariables) where
  evidence : ProductionRecursiveSuccessorFor.Evidence candidate statementId
    config artifact priorAuthority assignment headers priorPrefix value proof
    recursive machine program
  layoutValid : layout.Valid priorAuthority evidence.continuation
  applicationPlaced :
    ProductionRecursiveSuccessorRowsFor.ApplicationProducerPlaced layout
      assignment evidence.batch
  nifsOutputAlias : forall index,
    layout.nifsOutputColumn index =
      recursive.nifsOutputLayout.carrierColumn index
  successorRows : Satisfies
    (ProductionRecursiveSuccessorRowsFor.rows layout priorAuthority statementId)
      assignment

namespace Supplement

noncomputable def successor
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
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      value proof}
    {machine : Machine Program} {program : Program}
    {layout : ProductionRecursiveSuccessorRowsFor.Layout rowVariables}
    (supplement : Supplement candidate statementId config artifact
      priorAuthority assignment headers priorPrefix value proof recursive
      machine program layout) :
    ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits) :=
  ProductionRecursiveSuccessorFor.value candidate statementId config artifact
    value proof recursive.priorState supplement.evidence.batch
      supplement.evidence.outgoing

end Supplement

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
    (priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout
      candidate rowVariables)
    (assignment : Nat -> Nat) (headers : ChainHeaders Digest.Value)
    (priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits))
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      value proof)
    (machine : Machine Program) (program : Program)
    (layout : ProductionRecursiveSuccessorRowsFor.Layout rowVariables)
    (supplement : Supplement candidate statementId config artifact
      priorAuthority assignment headers priorPrefix value proof recursive
      machine program layout) : Prop where
  verifiedClaimExact : recursive.verified.claim = value.toProtocolClaim
    (NifsProof := ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
  verifiedProofExact : recursive.verified.proof = proof
  verifierAccepted :
    ProductionPaperRecursiveRelationRowsSoundFor.paperVerifier candidate
      statementId config artifact recursive.verified.proof
        recursive.verified.claim
  delayedMemoryTransition : ProductionBatchedFPrime.Transition
    (ProductionPaperRecursiveRelationRowsSoundFor.paperVerifier candidate
      statementId config artifact)
    MemoryProductBalanceRows.ConcreteBalanced
    (recursive.memoryResult.semantic 0) recursive.verified
    (recursive.memoryResult.semantic
      (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate)))
  applicationRun : Runs machine program
    (WasmStateEncoding.decode recursive.priorState.applicationState)
      supplement.evidence.batch.rows supplement.evidence.applicationAfter
      (realRowCount supplement.evidence.batch.rows)
  priorRunningExact : recursive.priorState.running = value.recursiveState
  memoryContinues : Continues supplement.evidence.memoryDerive headers
    (recursive.memoryResult.semantic
      (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate)))
    (MemoryCarryParser.semanticCarry supplement.evidence.outgoing
      supplement.evidence.outgoingParsed.parserCanonical.stepIndex)
  successorPlaced : ProductionSuccessorStateBindingRowsFor.Placed
    (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
      publicFits) layout.successor assignment supplement.successor
  successorOutputStateExact :
    SymbolicDuplexSemantics.decodedBuilder assignment
        (ProductionSuccessorStateBindingRowsFor.builder candidate
          layout.successorHashBase layout.successor statementId) =
      ProductionSuccessorStateBinding.outputState statementId
        supplement.successor
  successorCanonical : supplement.successor.Canonical headers

theorem exact_of_supplement
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
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      value proof}
    {machine : Machine Program} {program : Program}
    {layout : ProductionRecursiveSuccessorRowsFor.Layout rowVariables}
    (supplement : Supplement candidate statementId config artifact
      priorAuthority assignment headers priorPrefix value proof recursive
      machine program layout) :
    ExactInvocation candidate statementId config artifact priorAuthority
      assignment headers priorPrefix value proof recursive machine program
      layout supplement := by
  have successorResult :=
    ProductionRecursiveSuccessorRowsFor.rows_imply_exact_successor_and_outputState
      supplement.evidence.toCoreEvidence supplement.layoutValid
        supplement.applicationPlaced
        supplement.nifsOutputAlias supplement.successorRows
  exact
    { verifiedClaimExact := recursive.claimExact
      verifiedProofExact := recursive.proofExact
      verifierAccepted := recursive.verified.accepted
      delayedMemoryTransition := recursive.transition
      applicationRun := supplement.evidence.batch.run
      priorRunningExact := supplement.evidence.prior_running_is_nifs_input
      memoryContinues := supplement.evidence.memory_continues_from_rows
      successorPlaced := successorResult.1
      successorOutputStateExact := successorResult.2.1
      successorCanonical := successorResult.2.2 }

def CoreRowsResult
    {Program : Type} (candidate : Id)
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
    (assignment : Nat -> Nat) (headers : ChainHeaders Digest.Value)
    (priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits))
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (baseWires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables)
    (samplerBase : Nat) (_machine : Machine Program) (_program : Program)
    (_layout : ProductionRecursiveSuccessorRowsFor.Layout rowVariables) : Prop :=
  Nonempty (ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      value proof) /\
    CoreChecks candidate statementId config artifact priorAuthority assignment
      value proof baseWires samplerBase

/-- The mandatory recursive relation rows derive the accepted prior claim,
the exact NIFS output carrier, and the delayed memory transition. They do not
derive an application transition or prove that a successor supplement exists.
`exact_of_supplement` is the separate completion theorem once a concrete
application and successor-row supplement is available. -/
theorem rows_imply_exact_core
    {Program : Type} (candidate : Id)
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
    (priorAuthorityValid : priorAuthority.Valid)
    (headers : ChainHeaders Digest.Value)
    (priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits))
    (assignment : Nat -> Nat)
    (priorPrefixPlaced : ProductionPaperPriorStateAuthorityRowsFor.PrefixPlaced
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits) priorAuthority assignment priorPrefix)
    (statement : ProductionStatement Program)
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (valueCanonical : value.Canonical)
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (baseWires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables)
    (samplerBase : Nat)
    (algebraLayout : ProductPiRlcAlgebraRows.Layout)
    (piDecLayout : ProductPiDecRows.Layout)
    (nifsOutputLayout : ProductionProductNifsOutputRowsFor.Layout rowVariables)
    (compactManifest : SeedSchedule.Manifest)
    (compactLayout : ProductionFieldNativeCompactChainRowsFor.Layout)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (prefixCanonical : ProductionFullClaimNifsPublicCarrierFor.PrefixCanonical
      candidate (FullShape rowVariables logicalWidth publicFits) 9)
    (headersPlaced : ProductionMemoryCheckedBatchRows.HeadersPlaced
      priorAuthority.ccs.core.batch.frame.memory assignment headers)
    (carryHeadersPlaced : MemoryCarryRows.HeadersPlaced
      priorAuthority.carry.carry assignment headers)
    (rowsHold : ProductionPaperRecursiveRelationRowsSoundFor.RowsHold candidate
      statementId config artifact value proof
      (wires (logicalWidth := logicalWidth) (publicFits := publicFits) candidate
        rowVariables priorAuthority baseWires) samplerBase algebraLayout
      piDecLayout nifsOutputLayout priorAuthority compactManifest compactLayout
      assignment)
    (placement : ProductionPaperRecursiveRelationRowsSoundFor.Placement candidate
      statementId config artifact priorAuthority value proof baseWires
      samplerBase algebraLayout piDecLayout assignment canonical)
    (machine : Machine Program) (program : Program)
    (layout : ProductionRecursiveSuccessorRowsFor.Layout rowVariables) :
    CoreRowsResult candidate statementId config artifact priorAuthority assignment
      headers priorPrefix value proof baseWires samplerBase machine program
      layout := by
  rcases
      ProductionPaperRecursiveRelationRowsSoundFor.rows_imply_verified_exact_claim_and_memory_transition
        candidate statementId config artifact priorAuthority
        priorAuthorityValid headers priorPrefix assignment priorPrefixPlaced
        statement value valueCanonical proof baseWires samplerBase
        algebraLayout piDecLayout nifsOutputLayout compactManifest compactLayout
        canonical one
        prefixCanonical headersPlaced carryHeadersPlaced rowsHold placement with
    ⟨recursive, _nifsOutputLayoutExact, _compactManifestExact,
      samplerSucceeded, piCcsAccepted, piDecAccepted, verifierOutputExact⟩
  exact ⟨⟨recursive⟩,
    { samplerSucceeded := samplerSucceeded
      piCcsAccepted := piCcsAccepted
      piDecAccepted := piDecAccepted
      verifierOutputExact := verifierOutputExact }⟩

end Nightstream.Implementation.Nebula.ProductionPaperRecursiveInvocationRowsSoundFor
