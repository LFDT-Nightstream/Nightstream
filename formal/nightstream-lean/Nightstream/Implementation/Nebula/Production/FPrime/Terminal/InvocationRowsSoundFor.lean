import Nightstream.Implementation.Nebula.Production.FPrime.Recursive.RelationRowsSoundFor
import Nightstream.Implementation.Nebula.Production.FPrime.Recursive.RecursiveSuccessorFor
import Nightstream.Implementation.Nebula.NIFS.Terminal.Relation
import Nightstream.Implementation.Nebula.Memory.Transition.OpenSegmentSound
import Nightstream.Implementation.Nebula.FPrime.Terminal.ClosedCarryRows
import Nightstream.Protocol.Nebula.WasmStatement

/-!
Contract: exact terminal F-prime invocation at the generated relation
exponent.

The terminal invocation verifies and consumes the one trailing fresh claim,
derives a closed final memory carry from rows, opens all fourteen PiDEC
children against complete bounded assignments, and checks the external result
image. It has no successor state and produces no fresh claim.

The same `rowVariables` value selects the augmented relation, paper NIFS,
claim carrier, proof, final-fold output, and terminal children. This module
does not prove cryptographic binding, generated terminal-row refinement,
recursive-size closure, Rust refinement, or compact-backend soundness.

Assurance tier: exponent-indexed implementation model.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Nebula.ProductionPaperTerminalInvocationRowsSoundFor

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.ProductState
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.Protocol.Nebula.Terminal
open Nightstream.Protocol.Nebula.WasmStatement
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev FullShape
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits

abbrev ProtocolSchema
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductionPaperRecursiveRelationRowsSoundFor.ProtocolSchema rowVariables
    logicalWidth publicFits

/-! ## Deterministic final fold -/

noncomputable def finalRunning
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables) :
    ProductNifsCodec.RunningFor rowVariables
      (FullShape rowVariables logicalWidth publicFits) :=
  ProductionRecursiveSuccessorFor.nextRunning candidate statementId config
    artifact value proof

noncomputable def children
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables) :
    ProductTerminalRelation.Children
      (FullShape rowVariables logicalWidth publicFits) :=
  fun child =>
    let running := finalRunning candidate statementId config artifact value proof
    { constraintSystem := artifact.system
      commitment := running.commitments child
      publicInput := running.publicInputs child
      point := running.point
      evaluations := Array.ofFn (running.evaluations child)
      stage := .fresh }

@[simp] theorem children_stage
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (child : FoldedChild) :
    (children candidate statementId config artifact value proof child).stage =
      NormStage.fresh := rfl

/-! ## Terminal memory and public result -/

def closingLayout
    (candidate : Id) (rowVariables : Nat)
    (priorAuthority :
      ProductionPaperPriorStateAuthorityRowsFor.Layout candidate rowVariables) :
    TerminalClosedCarryRows.Layout where
  carry :=
    (priorAuthority.ccs.core.batch.frame.memory.boundaries
      (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate))).reference

def finalWire
    {candidate : Id} {rowVariables : Nat}
    {priorAuthority :
      ProductionPaperPriorStateAuthorityRowsFor.Layout candidate rowVariables}
    {assignment : Nat -> Nat} {headers : ChainHeaders Digest.Value}
    (memoryResult : ProductionMemoryCheckedBatchRows.Result
      priorAuthority.ccs.core.batch.frame.memory assignment headers) :
    MemoryCarryCodec.Value :=
  memoryResult.boundary
    (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate))

def finalClosed
    {candidate : Id} {rowVariables : Nat}
    {priorAuthority :
      ProductionPaperPriorStateAuthorityRowsFor.Layout candidate rowVariables}
    {assignment : Nat -> Nat} {headers : ChainHeaders Digest.Value}
    (memoryResult : ProductionMemoryCheckedBatchRows.Result
      priorAuthority.ccs.core.batch.frame.memory assignment headers) :
    ClosedCarry Digest.Value :=
  MemoryOpenSegmentSound.closedOfWire (finalWire memoryResult)

structure PublicChecks
    {Program : Type} (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (statement : ProductionStatement Program)
    (priorState : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (closed : ClosedCarry Digest.Value) : Prop where
  invocationIndex : priorState.augmentedInvocationIndex =
    statement.base.segmentCount * claimsPerSegment candidate
  initialApplication : priorState.initialApplicationState =
    WasmStateEncoding.encode statement.base.initialApplicationState
  finalApplication : priorState.applicationState =
    statement.resultImage.finalApplicationState
  realApplicationRows : priorState.realApplicationRowCount =
    statement.resultImage.realApplicationRowCount
  finalSegment : closed.segmentIndex = statement.base.segmentCount
  finalTimestamp : closed.globalTimestamp = statement.base.finalGlobalTimestamp
  finalMemoryRoot : closed.memoryRoot = statement.resultImage.finalMemoryRoot

/-! ## Same-witness product opening -/

structure ProductOpening
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables) where
  assignments : ProductTerminalRelation.Assignments
    (FullShape rowVariables logicalWidth publicFits)
  bounded : forall child, assignmentNormBounded 2 (assignments child)
  commitments : forall child,
    ProductCommitmentAlgebra.commit config (assignments child) =
      (children candidate statementId config artifact value proof child).commitment
  publicInputs : forall child,
    projectPublicInput (assignments child) =
      (children candidate statementId config artifact value proof child).publicInput
  evaluations : forall child,
    Phi81Relation.evaluations artifact.system (assignments child)
        (children candidate statementId config artifact value proof child).point =
      (children candidate statementId config artifact value proof child).evaluations

namespace ProductOpening

noncomputable def ofHolds
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    (assignments : ProductTerminalRelation.Assignments
      (FullShape rowVariables logicalWidth publicFits))
    (holds : ProductTerminalRelation.Holds config
      (children candidate statementId config artifact value proof)
      assignments) :
    ProductOpening candidate statementId config artifact value proof where
  assignments := assignments
  bounded := by
    intro child
    have member :=
      (Phi81Relation.ceMembership_iff
        (ProductCommitmentAlgebra.commit config) productionGlobalParams
        (children candidate statementId config artifact value proof child)
        (assignments child)).1 (holds child).2
    have fresh := (holds child).1
    rw [fresh] at member
    exact member.2.2.1
  commitments := by
    intro child
    exact ProductTerminalRelation.commitment_of_holds config _ _ holds child
  publicInputs := by
    intro child
    exact (ProductTerminalRelation.core_of_holds config _ _ holds child).2.1
  evaluations := by
    intro child
    exact (ProductTerminalRelation.core_of_holds config _ _ holds child).2.2

theorem coreHolds
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    (opening : ProductOpening candidate statementId config artifact value proof) :
    ProductTerminalRelation.CoreHolds
      (children candidate statementId config artifact value proof)
      opening.assignments := by
  intro child
  exact
    ⟨children_stage candidate statementId config artifact value proof child,
      opening.publicInputs child, opening.evaluations child⟩

theorem holds
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    (opening : ProductOpening candidate statementId config artifact value proof) :
    ProductTerminalRelation.Holds config
      (children candidate statementId config artifact value proof)
      opening.assignments :=
  ProductTerminalRelation.holds_of_common_openings config _ _ opening.bounded
    opening.commitments opening.coreHolds

end ProductOpening

/-! ## Exact terminal invocation -/

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
    (priorAuthority :
      ProductionPaperPriorStateAuthorityRowsFor.Layout candidate rowVariables)
    (assignment : Nat -> Nat) (headers : ChainHeaders Digest.Value)
    (priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits))
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      value proof)
    (opening : ProductOpening candidate statementId config artifact value proof)
    (statement : ProductionStatement Program) : Prop where
  assignmentCanonical : forall column, assignment column < goldilocksP
  one : assignment 0 = 1
  trailingVerified :
    (ProductionPaperRecursiveRelationRowsSoundFor.paperVerifier candidate
      statementId config artifact) recursive.verified.proof
        recursive.verified.claim
  trailingClaimExact : recursive.verified.claim =
    value.toProtocolClaim
      (NifsProof := ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
  trailingProofExact : recursive.verified.proof = proof
  finalFoldOutput : finalRunning candidate statementId config artifact value proof =
    (ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId config
      artifact).output value.recursiveState
        (ProductionFieldNativeFullClaim.freshOfValue
          (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
            publicFits).toShape value)
        proof
  finalPhase : (finalWire recursive.memoryResult).phase = .closed
  finalSemantic : recursive.memoryResult.semantic
      (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate)) =
    .closed (finalClosed recursive.memoryResult)
  consumesTrailing : ProductionBatchedFPrime.Transition
    (ProductionPaperRecursiveRelationRowsSoundFor.paperVerifier candidate
      statementId config artifact)
    MemoryProductBalanceRows.ConcreteBalanced
    (recursive.memoryResult.semantic 0) recursive.verified
    (.closed (finalClosed recursive.memoryResult))
  childrenHold : ProductTerminalRelation.Holds config
    (children candidate statementId config artifact value proof)
    opening.assignments
  publicResult : PublicChecks candidate statement recursive.priorState
    (finalClosed recursive.memoryResult)

theorem exact
    {Program : Type} (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (priorAuthority :
      ProductionPaperPriorStateAuthorityRowsFor.Layout candidate rowVariables)
    (assignment : Nat -> Nat) (headers : ChainHeaders Digest.Value)
    (priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits))
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      value proof)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (closingRows : Satisfies
      (TerminalClosedCarryRows.rows
        (closingLayout candidate rowVariables priorAuthority)) assignment)
    (opening : ProductOpening candidate statementId config artifact value proof)
    (statement : ProductionStatement Program)
    (publicChecks : PublicChecks candidate statement recursive.priorState
      (finalClosed recursive.memoryResult)) :
    ExactInvocation candidate statementId config artifact priorAuthority
      assignment headers priorPrefix value proof recursive opening statement := by
  have phaseClosed : (finalWire recursive.memoryResult).phase = .closed := by
    exact TerminalClosedCarryRows.parsed_phase_closed canonical one
      (recursive.memoryResult.boundaryParsed
        (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate)))
      closingRows
  have semanticClosed : recursive.memoryResult.semantic
      (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate)) =
        .closed (finalClosed recursive.memoryResult) := by
    rw [recursive.memoryResult.semanticExact]
    have phaseClosedRaw :
        (recursive.memoryResult.boundary
          (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate))).phase =
            .closed := by
      simpa [finalWire] using phaseClosed
    unfold MemoryCarryParser.semanticCarry
    rw [phaseClosedRaw]
    rfl
  have transitionClosed : ProductionBatchedFPrime.Transition
      (ProductionPaperRecursiveRelationRowsSoundFor.paperVerifier candidate
        statementId config artifact)
      MemoryProductBalanceRows.ConcreteBalanced
      (recursive.memoryResult.semantic 0) recursive.verified
      (.closed (finalClosed recursive.memoryResult)) := by
    rw [← semanticClosed]
    exact recursive.transition
  exact
    { assignmentCanonical := canonical
      one := one
      trailingVerified := recursive.verified.accepted
      trailingClaimExact := recursive.claimExact
      trailingProofExact := recursive.proofExact
      finalFoldOutput := rfl
      finalPhase := phaseClosed
      finalSemantic := semanticClosed
      consumesTrailing := transitionClosed
      childrenHold := opening.holds
      publicResult := publicChecks }

theorem exactOfHolds
    {Program : Type} (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (priorAuthority :
      ProductionPaperPriorStateAuthorityRowsFor.Layout candidate rowVariables)
    (assignment : Nat -> Nat) (headers : ChainHeaders Digest.Value)
    (priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits))
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      value proof)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (closingRows : Satisfies
      (TerminalClosedCarryRows.rows
        (closingLayout candidate rowVariables priorAuthority)) assignment)
    (terminalAssignments : ProductTerminalRelation.Assignments
      (FullShape rowVariables logicalWidth publicFits))
    (terminalHolds : ProductTerminalRelation.Holds config
      (children candidate statementId config artifact value proof)
      terminalAssignments)
    (statement : ProductionStatement Program)
    (publicChecks : PublicChecks candidate statement recursive.priorState
      (finalClosed recursive.memoryResult)) :
    ExactInvocation candidate statementId config artifact priorAuthority
      assignment headers priorPrefix value proof recursive
      (ProductOpening.ofHolds terminalAssignments terminalHolds) statement := by
  exact exact candidate statementId config artifact priorAuthority assignment
    headers priorPrefix value proof recursive canonical one closingRows
    (ProductOpening.ofHolds terminalAssignments terminalHolds) statement
    publicChecks

end Nightstream.Implementation.Nebula.ProductionPaperTerminalInvocationRowsSoundFor
