import Nightstream.Implementation.NebulaV2.Production.FPrime.Recursive.RecursiveSuccessorFor
import Nightstream.Implementation.NebulaV2.Production.Carrier.PreCarryDigestRowsFor
import Nightstream.Implementation.NebulaV2.Core.LessThanConstantRows
import Nightstream.Implementation.NebulaV2.Core.UnsignedAdditionRows

/-!
Contract: exponent-indexed physical carrier and Poseidon2 rows for one exact
recursive F-prime successor.

The section owns the two counter additions, strict lifetime bounds, zero-copy
aliases from the prior state, application output, paper-NIFS output, and
continued memory carry, and the complete successor hash. The application and
NIFS producer interfaces are narrow placement boundaries; neither contains
the desired successor conclusion.

Assurance tier: exponent-indexed row implementation.

Does not own the complete generated artifact, application compiler, terminal
verifier, recursive-size closure, Rust refinement, or cryptographic reductions.

Emits constraints: `78 + (successorPermutationCount(rowVariables) +
preCarryPermutationCount(rowVariables)) * 352` rows.
-/

set_option autoImplicit false
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

namespace Nightstream.Implementation.NebulaV2.ProductionRecursiveSuccessorRowsFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ApplicationBatch
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.Protocol.NebulaV2.WasmState
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev FullShape := ProductionRecursiveSuccessorFor.FullShape

structure Layout (rowVariables : Nat) where
  successor : ProductionSuccessorStateBindingRowsFor.Layout rowVariables
  successorHashBase : Nat
  preCarryDigestBase : Nat
  applicationRowCountColumn : Nat
  applicationStateColumn : Fin 85 -> Nat
  nifsOutputColumn :
    Fin (ProductNifsCodec.runningFieldCountFor rowVariables) -> Nat
  invocationValueBitStart : Nat
  invocationSlackColumn : Nat
  invocationSlackBitStart : Nat
  realRowsValueBitStart : Nat
  realRowsSlackColumn : Nat
  realRowsSlackBitStart : Nat

def Layout.preCarryDigest
    {rowVariables : Nat} (layout : Layout rowVariables) :
    ProductionPreCarryDigestRowsFor.Layout rowVariables :=
  { source := layout.successor
    sourceBase := layout.successorHashBase
    digestBase := layout.preCarryDigestBase }

def Layout.invocationAddition
    {candidate : Id} {rowVariables : Nat} (layout : Layout rowVariables)
    (prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables) : UnsignedAdditionRows.Layout :=
  { leftWidth := 17
    rightWidth := 1
    leftColumn := prior.state.invocationColumn
    rightColumn := 0
    outputColumn := layout.successor.invocationColumn }

def Layout.realRowsAddition
    {candidate : Id} {rowVariables : Nat} (layout : Layout rowVariables)
    (prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables) : UnsignedAdditionRows.Layout :=
  { leftWidth := 18
    rightWidth := 18
    leftColumn := prior.state.realRowsColumn
    rightColumn := layout.applicationRowCountColumn
    outputColumn := layout.successor.realRowsColumn }

def Layout.invocationBound
    {rowVariables : Nat} (candidate : Id) (layout : Layout rowVariables) :
    LessThanConstantRows.Layout :=
  { width := 17
    limit := maximumAugmentedInvocations candidate
    valueColumn := layout.successor.invocationColumn
    valueBitStart := layout.invocationValueBitStart
    slackColumn := layout.invocationSlackColumn
    slackBitStart := layout.invocationSlackBitStart }

def Layout.realRowsBound
    {rowVariables : Nat} (layout : Layout rowVariables) :
    LessThanConstantRows.Layout :=
  { width := 18
    limit := 2 ^ 18
    valueColumn := layout.successor.realRowsColumn
    valueBitStart := layout.realRowsValueBitStart
    slackColumn := layout.realRowsSlackColumn
    slackBitStart := layout.realRowsSlackBitStart }

theorem invocationBound_valid
    {rowVariables : Nat} (candidate : Id) (layout : Layout rowVariables) :
    (layout.invocationBound candidate).Valid := by
  constructor
  · cases candidate <;>
      norm_num [Layout.invocationBound, maximumAugmentedInvocations,
        maximumClaims, maximumSegments, claimsPerSegment, stepsPerSegment,
        checkedStepsPerFreshClaim]
  · cases candidate <;>
      norm_num [Layout.invocationBound, maximumAugmentedInvocations,
        maximumClaims, maximumSegments, claimsPerSegment, stepsPerSegment,
        checkedStepsPerFreshClaim]
  · norm_num [Layout.invocationBound, goldilocksP]

theorem realRowsBound_valid
    {rowVariables : Nat} (layout : Layout rowVariables) :
    layout.realRowsBound.Valid := by
  constructor <;> norm_num [Layout.realRowsBound, goldilocksP]

structure Layout.Valid
    {candidate : Id} {rowVariables : Nat} (layout : Layout rowVariables)
    (prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables)
    (continuation : ProductionMemorySegmentContinuationRows.Layout candidate) : Prop where
  initialApplication : forall index : Fin 85,
    layout.successor.initialApplicationColumn index =
      prior.state.initialApplicationColumn index
  application : forall index : Fin 85,
    layout.successor.applicationColumn index =
      layout.applicationStateColumn index
  running : forall index :
      Fin (ProductNifsCodec.runningFieldCountFor rowVariables),
    layout.successor.runningColumn index = layout.nifsOutputColumn index
  initialCarry : forall index : Fin 59,
    layout.successor.initialCarryColumn index =
      prior.state.initialCarryColumn index
  carry : forall index : Fin 59,
    layout.successor.carryColumn index =
      continuation.outgoing.carry.fieldColumn
        (ProductionMemoryCarryFields.tagAt index)

theorem invocationAddition_valid
    {candidate : Id} {rowVariables : Nat} (layout : Layout rowVariables)
    (prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables) : (layout.invocationAddition prior).Valid := by
  constructor
  norm_num [Layout.invocationAddition, goldilocksP]

theorem realRowsAddition_valid
    {candidate : Id} {rowVariables : Nat} (layout : Layout rowVariables)
    (prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables) : (layout.realRowsAddition prior).Valid := by
  constructor
  norm_num [Layout.realRowsAddition, goldilocksP]

def rows
    {candidate : Id} {rowVariables : Nat} (layout : Layout rowVariables)
    (prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables)
    (statementId : ProductPoseidon2.StatementId) : List Row :=
  UnsignedAdditionRows.rows (layout.invocationAddition prior) ++
    UnsignedAdditionRows.rows (layout.realRowsAddition prior) ++
    LessThanConstantRows.rows (layout.invocationBound candidate) ++
    LessThanConstantRows.rows layout.realRowsBound ++
    ProductionSuccessorStateBindingRowsFor.rows candidate
      layout.successorHashBase layout.successor statementId ++
    ProductionPreCarryDigestRowsFor.rows candidate layout.preCarryDigest
      statementId

def rowCount (rowVariables : Nat) : Nat :=
  78 + ProductionSuccessorStateBindingRowsFor.successorPermutationCount
    rowVariables * 352 +
    ProductionPreCarryDigestRowsFor.permutationCount rowVariables * 352

theorem rows_length_exact
    {candidate : Id} {rowVariables : Nat} (layout : Layout rowVariables)
    (prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables)
    (statementId : ProductPoseidon2.StatementId) :
    (rows layout prior statementId).length = rowCount rowVariables := by
  simp [rows, rowCount, UnsignedAdditionRows.rows_length,
    LessThanConstantRows.rows_length, Layout.invocationBound,
    Layout.realRowsBound,
    ProductionSuccessorStateBindingRowsFor.rows_length_exact,
    ProductionPreCarryDigestRowsFor.rows_length_exact]
  omega

private theorem invocation_rows_hold
    {candidate : Id} {rowVariables : Nat} {layout : Layout rowVariables}
    {prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables}
    {statementId : ProductPoseidon2.StatementId} {assignment : Nat -> Nat}
    (holds : Satisfies (rows layout prior statementId) assignment) :
    Satisfies (UnsignedAdditionRows.rows (layout.invocationAddition prior))
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem real_rows_hold
    {candidate : Id} {rowVariables : Nat} {layout : Layout rowVariables}
    {prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables}
    {statementId : ProductPoseidon2.StatementId} {assignment : Nat -> Nat}
    (holds : Satisfies (rows layout prior statementId) assignment) :
    Satisfies (UnsignedAdditionRows.rows (layout.realRowsAddition prior))
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem invocation_bound_rows_hold
    {candidate : Id} {rowVariables : Nat} {layout : Layout rowVariables}
    {prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables}
    {statementId : ProductPoseidon2.StatementId} {assignment : Nat -> Nat}
    (holds : Satisfies (rows layout prior statementId) assignment) :
    Satisfies (LessThanConstantRows.rows
      (layout.invocationBound candidate)) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem real_rows_bound_rows_hold
    {candidate : Id} {rowVariables : Nat} {layout : Layout rowVariables}
    {prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables}
    {statementId : ProductPoseidon2.StatementId} {assignment : Nat -> Nat}
    (holds : Satisfies (rows layout prior statementId) assignment) :
    Satisfies (LessThanConstantRows.rows layout.realRowsBound) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

theorem successor_rows_hold
    {candidate : Id} {rowVariables : Nat} {layout : Layout rowVariables}
    {prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables}
    {statementId : ProductPoseidon2.StatementId} {assignment : Nat -> Nat}
    (holds : Satisfies (rows layout prior statementId) assignment) :
    Satisfies
      (ProductionSuccessorStateBindingRowsFor.rows candidate
        layout.successorHashBase layout.successor statementId) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

theorem preCarry_digest_rows_hold
    {candidate : Id} {rowVariables : Nat} {layout : Layout rowVariables}
    {prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables}
    {statementId : ProductPoseidon2.StatementId} {assignment : Nat -> Nat}
    (holds : Satisfies (rows layout prior statementId) assignment) :
    Satisfies
      (ProductionPreCarryDigestRowsFor.rows candidate layout.preCarryDigest
        statementId) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

/-- The combined successor row family exposes the exact four-lane gated
pre-carry digest.  Callers do not need to unfold the large successor and
Poseidon2 row split. -/
theorem rows_imply_preCarry_digest_lane
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    {layout : Layout rowVariables} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (statementId : ProductPoseidon2.StatementId)
    (successor : ProductionSuccessorStateBinding.Value candidate fullShape)
    (successorPlaced : ProductionSuccessorStateBindingRowsFor.Placed contract
      layout.successor assignment successor)
    {prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables}
    (holds : Satisfies (rows layout prior statementId) assignment)
    (lane : Fin 4) :
    lcEval assignment
        (ProductionPreCarryDigestRowsFor.digestExpression candidate
          layout.preCarryDigest statementId lane) =
      (ProductionSuccessorStateBinding.preCarryDigest statementId
        successor.preCarry lane).val := by
  apply ProductionPreCarryDigestRowsFor.rows_imply_digest_lane contract
    canonical one statementId successor successorPlaced
  · exact successor_rows_hold holds
  · exact preCarry_digest_rows_hold holds

structure ApplicationProducerPlaced
    {Program : Type} {candidate : Id} {rowVariables : Nat}
    {machine : Machine Program} {program : Program}
    {before after : AppStateVector}
    (layout : Layout rowVariables) (assignment : Nat -> Nat)
    (batch : Batch candidate machine program before after) : Prop where
  rowCount : assignment layout.applicationRowCountColumn =
    realRowCount batch.rows
  application : forall index : Fin 85,
    assignment (layout.applicationStateColumn index) =
      ((ProductionWasmStateFields.encode (WasmStateEncoding.encode after)).get
        (Fin.cast
          (ProductionWasmStateFields.encode_length
            (WasmStateEncoding.encode after)).symm index))

private theorem candidate_invocations_lt_pow17 (candidate : Id) :
    maximumAugmentedInvocations candidate < 2 ^ 17 := by
  cases candidate <;>
    norm_num [maximumAugmentedInvocations, maximumClaims, maximumSegments,
      claimsPerSegment, stepsPerSegment, checkedStepsPerFreshClaim]

private theorem batch_real_rows_lt_pow18
    {Program : Type} {candidate : Id}
    {machine : Machine Program} {program : Program}
    {before after : AppStateVector}
    (batch : Batch candidate machine program before after) :
    realRowCount batch.rows < 2 ^ 18 := by
  exact batch.realRowCount_le_rowsPerFreshClaim.trans_lt (by
    cases candidate <;> decide)

theorem rows_imply_successor_ranges
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
    {assignment : Nat -> Nat}
    {headers : FPrime.ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {claim : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      claim proof}
    {machine : Machine Program} {program : Program}
    (evidence : ProductionRecursiveSuccessorFor.CoreEvidence candidate statementId
      config artifact priorAuthority assignment headers priorPrefix claim proof
      recursive machine program)
    {layout : Layout rowVariables}
    (applicationPlaced : ApplicationProducerPlaced layout assignment
      evidence.batch)
    (rowsHold : Satisfies (rows layout priorAuthority statementId) assignment) :
    recursive.priorState.augmentedInvocationIndex + 1 <
        maximumAugmentedInvocations candidate /\
      recursive.priorState.realApplicationRowCount +
          realRowCount evidence.batch.rows < 2 ^ 18 := by
  have priorPlaced := recursive.priorAuthorityResult.priorPlaced
  have priorInvocationBound :
      assignment priorAuthority.state.invocationColumn < 2 ^ 17 := by
    rw [priorPlaced.invocation]
    exact evidence.priorCanonical.invocationIndex.trans
      (candidate_invocations_lt_pow17 candidate)
  have oneBound :
      assignment (layout.invocationAddition priorAuthority).rightColumn <
        2 ^ (layout.invocationAddition priorAuthority).rightWidth := by
    change assignment 0 < 2 ^ 1
    rw [evidence.one]
    decide
  have invocationExact := UnsignedAdditionRows.output_eq_add
    (invocationAddition_valid layout priorAuthority) priorInvocationBound
    oneBound evidence.assignmentCanonical evidence.one
    (invocation_rows_hold rowsHold)
  have priorRowsBound :
      assignment priorAuthority.state.realRowsColumn < 2 ^ 18 := by
    rw [priorPlaced.realRows]
    exact evidence.priorCanonical.realApplicationRowCount
  have batchRowsBound :
      assignment layout.applicationRowCountColumn < 2 ^ 18 := by
    rw [applicationPlaced.rowCount]
    exact batch_real_rows_lt_pow18 evidence.batch
  have rowsExact := UnsignedAdditionRows.output_eq_add
    (realRowsAddition_valid layout priorAuthority) priorRowsBound
    batchRowsBound evidence.assignmentCanonical evidence.one
    (real_rows_hold rowsHold)
  have invocationPhysical := LessThanConstantRows.value_lt_limit
    (invocationBound_valid candidate layout) evidence.assignmentCanonical
    evidence.one (invocation_bound_rows_hold rowsHold)
  have realRowsPhysical := LessThanConstantRows.value_lt_limit
    (realRowsBound_valid layout) evidence.assignmentCanonical evidence.one
    (real_rows_bound_rows_hold rowsHold)
  constructor
  · change assignment layout.successor.invocationColumn <
      maximumAugmentedInvocations candidate at invocationPhysical
    change (assignment layout.successor.invocationColumn =
      assignment priorAuthority.state.invocationColumn + assignment 0) at invocationExact
    rw [invocationExact, priorPlaced.invocation, evidence.one] at invocationPhysical
    exact invocationPhysical
  · change assignment layout.successor.realRowsColumn < 2 ^ 18 at realRowsPhysical
    change (assignment layout.successor.realRowsColumn =
      assignment priorAuthority.state.realRowsColumn +
        assignment layout.applicationRowCountColumn) at rowsExact
    rw [rowsExact, priorPlaced.realRows, applicationPlaced.rowCount] at realRowsPhysical
    exact realRowsPhysical

theorem rows_imply_successorPlaced
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
    {assignment : Nat -> Nat}
    {headers : FPrime.ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {claim : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      claim proof}
    {machine : Machine Program} {program : Program}
    (evidence : ProductionRecursiveSuccessorFor.CoreEvidence candidate statementId
      config artifact priorAuthority assignment headers priorPrefix claim proof
      recursive machine program)
    {layout : Layout rowVariables}
    (valid : layout.Valid priorAuthority evidence.continuation)
    (applicationPlaced : ApplicationProducerPlaced layout assignment
      evidence.batch)
    (nifsOutputAlias : forall index,
      layout.nifsOutputColumn index =
        recursive.nifsOutputLayout.carrierColumn index)
    (rowsHold : Satisfies (rows layout priorAuthority statementId) assignment) :
    let successor := ProductionRecursiveSuccessorFor.value candidate statementId
      config artifact claim proof recursive.priorState evidence.batch
        evidence.outgoing
    ProductionSuccessorStateBindingRowsFor.Placed
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits) layout.successor assignment successor := by
  dsimp only
  let successor := ProductionRecursiveSuccessorFor.value candidate statementId
    config artifact claim proof recursive.priorState evidence.batch
      evidence.outgoing
  have priorPlaced := recursive.priorAuthorityResult.priorPlaced
  have priorInvocationBound :
      assignment priorAuthority.state.invocationColumn < 2 ^ 17 := by
    rw [priorPlaced.invocation]
    exact evidence.priorCanonical.invocationIndex.trans
      (candidate_invocations_lt_pow17 candidate)
  have oneBound :
      assignment (layout.invocationAddition priorAuthority).rightColumn <
        2 ^ (layout.invocationAddition priorAuthority).rightWidth := by
    change assignment 0 < 2 ^ 1
    rw [evidence.one]
    decide
  have invocationExact := UnsignedAdditionRows.output_eq_add
    (invocationAddition_valid layout priorAuthority) priorInvocationBound
    oneBound evidence.assignmentCanonical evidence.one
      (invocation_rows_hold rowsHold)
  have priorRowsBound :
      assignment priorAuthority.state.realRowsColumn < 2 ^ 18 := by
    rw [priorPlaced.realRows]
    exact evidence.priorCanonical.realApplicationRowCount
  have batchRowsBound : assignment layout.applicationRowCountColumn < 2 ^ 18 := by
    rw [applicationPlaced.rowCount]
    exact batch_real_rows_lt_pow18 evidence.batch
  have rowsExact := UnsignedAdditionRows.output_eq_add
    (realRowsAddition_valid layout priorAuthority) priorRowsBound
    batchRowsBound evidence.assignmentCanonical evidence.one
    (real_rows_hold rowsHold)
  refine
    { invocation := ?_
      realRows := ?_
      initialApplication := ?_
      application := ?_
      running := ?_
      initialCarry := ?_
      carry := ?_ }
  · change assignment layout.successor.invocationColumn =
      recursive.priorState.augmentedInvocationIndex + 1
    change assignment (layout.invocationAddition priorAuthority).outputColumn = _
    rw [invocationExact]
    change assignment priorAuthority.state.invocationColumn + assignment 0 = _
    rw [priorPlaced.invocation, evidence.one]
  · change assignment layout.successor.realRowsColumn =
      recursive.priorState.realApplicationRowCount +
        realRowCount evidence.batch.rows
    change assignment (layout.realRowsAddition priorAuthority).outputColumn = _
    rw [rowsExact]
    change assignment priorAuthority.state.realRowsColumn +
      assignment layout.applicationRowCountColumn = _
    rw [priorPlaced.realRows, applicationPlaced.rowCount]
  · intro index
    change assignment (layout.successor.initialApplicationColumn index) =
      ((ProductionWasmStateFields.encode
          recursive.priorState.initialApplicationState).get
        (Fin.cast
          (ProductionWasmStateFields.encode_length
            recursive.priorState.initialApplicationState).symm index))
    rw [valid.initialApplication index]
    exact priorPlaced.initialApplication index
  · intro index
    change assignment (layout.successor.applicationColumn index) =
      ((ProductionWasmStateFields.encode
          (WasmStateEncoding.encode evidence.applicationAfter)).get
        (Fin.cast
          (ProductionWasmStateFields.encode_length
            (WasmStateEncoding.encode evidence.applicationAfter)).symm index))
    rw [valid.application index]
    exact applicationPlaced.application index
  · intro index
    change assignment (layout.successor.runningColumn index) =
      ((ProductionSuccessorStateBinding.runningNativeFields
          (ProductionRecursiveSuccessorFor.nextRunning candidate statementId
            config artifact claim proof)).get
        (Fin.cast
          (ProductionSuccessorStateBindingRowsFor.runningNativeFields_length
            (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
              publicFits)
            (ProductionRecursiveSuccessorFor.nextRunning candidate statementId
              config artifact claim proof)).symm index))
    rw [valid.running index, nifsOutputAlias index]
    have outputCoordinate :=
      recursive.nifsOutputPlaced.assignment_coordinate index
    have nativeCoordinate :=
      ProductionSuccessorStateBindingRowsFor.runningNativeFields_get_eq_codec_getD
        (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
          publicFits)
        (ProductionRecursiveSuccessorFor.nextRunning candidate statementId
          config artifact claim proof) index
    exact outputCoordinate.trans nativeCoordinate.symm
  · intro index
    change assignment (layout.successor.initialCarryColumn index) =
      ((ProductionMemoryCarryFields.encode
          recursive.priorState.initialMemoryCarry).get
        (Fin.cast
          (ProductionMemoryCarryFields.encode_length
            recursive.priorState.initialMemoryCarry).symm index))
    rw [valid.initialCarry index]
    exact priorPlaced.initialCarry index
  · intro index
    change assignment (layout.successor.carryColumn index) =
      ((ProductionMemoryCarryFields.encode evidence.outgoing).get
        (Fin.cast
          (ProductionMemoryCarryFields.encode_length evidence.outgoing).symm
          index))
    rw [valid.carry index, ProductionMemoryCarryFields.encode_get]
    exact evidence.outgoingParsed.placed
      (ProductionMemoryCarryFields.tagAt index)

/-- Explicit form of `rows_imply_successorPlaced`.  This theorem keeps every
application and continuation input visible at the call boundary.  In
particular, it does not accept the successor or its placement as evidence. -/
theorem rows_imply_successorPlaced_explicit
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
    {assignment : Nat -> Nat}
    {headers : FPrime.ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {claim : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      claim proof}
    {machine : Machine Program} {program : Program}
    {after : AppStateVector}
    (batch : Batch candidate machine program
      (WasmStateEncoding.decode recursive.priorState.applicationState) after)
    {continuation : ProductionMemorySegmentContinuationRows.Layout candidate}
    (continuationValid : continuation.Valid)
    (continuationIntermediate : continuation.intermediate =
      priorAuthority.ccs.core.batch.frame.memory.boundaries
        (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate)))
    (outgoing : MemoryCarryCodec.Value)
    (outgoingParsed : MemoryCarryPublicRows.ParsedColumnsMatch
      continuation.outgoing.reference assignment headers outgoing)
    (continuationRows : Satisfies
      (ProductionMemorySegmentContinuationRows.rows continuation) assignment)
    (assignmentCanonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (priorCanonical : recursive.priorState.Canonical headers)
    {layout : Layout rowVariables}
    (valid : layout.Valid priorAuthority continuation)
    (applicationPlaced : ApplicationProducerPlaced layout assignment batch)
    (nifsOutputAlias : forall index,
      layout.nifsOutputColumn index =
        recursive.nifsOutputLayout.carrierColumn index)
    (rowsHold : Satisfies (rows layout priorAuthority statementId) assignment) :
    ProductionSuccessorStateBindingRowsFor.Placed
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits) layout.successor assignment
      (ProductionRecursiveSuccessorFor.value candidate statementId config
        artifact claim proof recursive.priorState batch outgoing) := by
  let evidence : ProductionRecursiveSuccessorFor.CoreEvidence candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      claim proof recursive machine program :=
    { applicationAfter := after
      batch := batch
      continuation := continuation
      continuationValid := continuationValid
      continuationIntermediate := continuationIntermediate
      outgoing := outgoing
      outgoingParsed := outgoingParsed
      continuationRows := continuationRows
      assignmentCanonical := assignmentCanonical
      one := one
      priorCanonical := priorCanonical }
  exact rows_imply_successorPlaced evidence valid applicationPlaced
    nifsOutputAlias rowsHold

theorem rows_imply_exact_successor_and_outputState
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
    {assignment : Nat -> Nat}
    {headers : FPrime.ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {claim : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact priorAuthority assignment headers priorPrefix
      claim proof}
    {machine : Machine Program} {program : Program}
    (evidence : ProductionRecursiveSuccessorFor.CoreEvidence candidate statementId
      config artifact priorAuthority assignment headers priorPrefix claim proof
      recursive machine program)
    {layout : Layout rowVariables}
    (valid : layout.Valid priorAuthority evidence.continuation)
    (applicationPlaced : ApplicationProducerPlaced layout assignment
      evidence.batch)
    (nifsOutputAlias : forall index,
      layout.nifsOutputColumn index =
        recursive.nifsOutputLayout.carrierColumn index)
    (rowsHold : Satisfies (rows layout priorAuthority statementId) assignment) :
    let successor := ProductionRecursiveSuccessorFor.value candidate statementId
      config artifact claim proof recursive.priorState evidence.batch
        evidence.outgoing
    ProductionSuccessorStateBindingRowsFor.Placed
        (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
          publicFits) layout.successor assignment successor /\
      SymbolicDuplexSemantics.decodedBuilder assignment
          (ProductionSuccessorStateBindingRowsFor.builder candidate
            layout.successorHashBase layout.successor statementId) =
        ProductionSuccessorStateBinding.outputState statementId successor /\
      successor.Canonical headers := by
  dsimp only
  let successor := ProductionRecursiveSuccessorFor.value candidate statementId
    config artifact claim proof recursive.priorState evidence.batch
      evidence.outgoing
  have placed : ProductionSuccessorStateBindingRowsFor.Placed
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits) layout.successor assignment successor :=
    rows_imply_successorPlaced evidence valid applicationPlaced nifsOutputAlias
      rowsHold
  have outputExact :=
    ProductionSuccessorStateBindingRowsFor.rows_imply_outputState
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits) evidence.assignmentCanonical evidence.one statementId
      successor placed layout.successorHashBase (successor_rows_hold rowsHold)
  have ranges := rows_imply_successor_ranges evidence applicationPlaced rowsHold
  exact ⟨placed, outputExact,
    evidence.successor_canonical ranges.1 ranges.2⟩

end Nightstream.Implementation.NebulaV2.ProductionRecursiveSuccessorRowsFor
