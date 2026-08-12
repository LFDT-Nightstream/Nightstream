import Nightstream.Implementation.NebulaV2.Production.Memory.BatchCcsLinkRowsFor
import Nightstream.Implementation.NebulaV2.Production.Memory.CarryRows
import Nightstream.Implementation.NebulaV2.Production.FPrime.Recursive.SuccessorStateBindingRowsFor
import Nightstream.Implementation.R1CS.Canonical.KEquality

/-!
Contract: exponent-indexed consumer authority for the exact HyperNova
Construction-2 prior state used by Nebula-on-SuperNeo.

The rows parse one current memory carry, hash the complete prior F-prime state
at the same generated relation exponent, link all four digest lanes to the CCS
public image, and bind the running state to the exact full claim consumed by
NIFS. The current carry is the first physical boundary of the delayed memory
batch. There is no second carry witness and no digest-only authority shortcut.

Assurance tier: exponent-indexed generated-row soundness.

Does not own Poseidon2 collision resistance, NIFS arithmetic, the application
transition, generated-artifact containment, Rust refinement, or terminal
verification.

Emits constraints: yes.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

namespace Nightstream.Implementation.NebulaV2.ProductionPaperPriorStateAuthorityRowsFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

abbrev CanonicalDigest :=
  ProductionMemoryBoundCcsPublic.CanonicalDigest

@[ext]
structure Prefix (candidate : Id) (fullShape : Phi81Relation.Shape) where
  augmentedInvocationIndex : Nat
  realApplicationRowCount : Nat
  initialApplicationState : ProductionSuccessorStateBinding.ApplicationState
  applicationState : ProductionSuccessorStateBinding.ApplicationState
  running : ProductionFieldNativeFullClaim.Running fullShape
  initialMemoryCarry : ProductionSuccessorStateBinding.MemoryCarry

def Prefix.withCarry
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (priorPrefix : Prefix candidate fullShape)
    (carry : ProductionSuccessorStateBinding.MemoryCarry) :
    ProductionSuccessorStateBinding.Value candidate fullShape :=
  { augmentedInvocationIndex := priorPrefix.augmentedInvocationIndex
    realApplicationRowCount := priorPrefix.realApplicationRowCount
    initialApplicationState := priorPrefix.initialApplicationState
    applicationState := priorPrefix.applicationState
    running := priorPrefix.running
    initialMemoryCarry := priorPrefix.initialMemoryCarry
    memoryCarry := carry }

structure Layout (candidate : Id) (rowVariables : Nat) where
  state : ProductionSuccessorStateBindingRowsFor.Layout rowVariables
  stateHashBase : Nat
  carry : ProductionMemoryCarryRows.Layout
  ccs : ProductionMemoryBatchCcsLinkRowsFor.Layout candidate rowVariables

/-- Structural identities only. No assignment or acceptance result occurs in
this certificate. -/
structure Layout.Valid
    {candidate : Id} {rowVariables : Nat}
    (layout : Layout candidate rowVariables) : Prop where
  ccsValid : ProductionMemoryBatchCcsLinkRowsFor.Valid layout.ccs
  initialCarryBoundary :
    layout.carry = layout.ccs.core.batch.frame.memory.boundaries 0
  exactRunningColumns : forall index :
      Fin (ProductNifsCodec.runningFieldCountFor rowVariables),
    layout.state.runningColumn index =
      layout.ccs.carrier.runningColumn index
  exactCarryColumns : forall index : Fin 59,
    layout.state.carryColumn index =
      layout.carry.carry.fieldColumn
        (ProductionMemoryCarryFields.tagAt index)

structure PrefixPlaced
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    (layout : Layout candidate rowVariables) (assignment : Nat -> Nat)
    (priorPrefix : Prefix candidate fullShape) : Prop where
  invocation : assignment layout.state.invocationColumn =
    priorPrefix.augmentedInvocationIndex
  realRows : assignment layout.state.realRowsColumn =
    priorPrefix.realApplicationRowCount
  initialApplication : forall index : Fin 85,
    assignment (layout.state.initialApplicationColumn index) =
      (ProductionWasmStateFields.encode priorPrefix.initialApplicationState).get
        (Fin.cast
          (ProductionWasmStateFields.encode_length
            priorPrefix.initialApplicationState).symm index)
  application : forall index : Fin 85,
    assignment (layout.state.applicationColumn index) =
      (ProductionWasmStateFields.encode priorPrefix.applicationState).get
        (Fin.cast
          (ProductionWasmStateFields.encode_length
            priorPrefix.applicationState).symm index)
  running : forall index :
      Fin (ProductNifsCodec.runningFieldCountFor rowVariables),
    assignment (layout.state.runningColumn index) =
      (ProductionSuccessorStateBinding.runningNativeFields
        priorPrefix.running).get
        (Fin.cast
          (ProductionSuccessorStateBindingRowsFor.runningNativeFields_length
            contract priorPrefix.running).symm index)
  initialCarry : forall index : Fin 59,
    assignment (layout.state.initialCarryColumn index) =
      (ProductionMemoryCarryFields.encode priorPrefix.initialMemoryCarry).get
        (Fin.cast
          (ProductionMemoryCarryFields.encode_length
            priorPrefix.initialMemoryCarry).symm index)

def digestLinkRows
    {candidate : Id} {rowVariables : Nat}
    (layout : Layout candidate rowVariables)
    (statementId : ProductPoseidon2.StatementId) : List Row :=
  List.ofFn fun lane : Fin 4 =>
    KEquality.equalityRow
      ((ProductionSuccessorStateBindingRowsFor.builder candidate
        layout.stateHashBase layout.state statementId).lanes
          (ProductionSuccessorStateBinding.outputLane lane))
      [(layout.ccs.core.stateDigestColumn lane, 1)]

def rows
    {candidate : Id} {rowVariables : Nat}
    (layout : Layout candidate rowVariables)
    (statementId : ProductPoseidon2.StatementId) : List Row :=
  ProductionMemoryCarryRows.rows layout.carry ++
    ProductionSuccessorStateBindingRowsFor.rows candidate
      layout.stateHashBase layout.state statementId ++
    ProductionMemoryBatchCcsLinkRowsFor.rows layout.ccs ++
    digestLinkRows layout statementId

private theorem carry_rows_hold
    {candidate : Id} {rowVariables : Nat}
    {layout : Layout candidate rowVariables}
    {statementId : ProductPoseidon2.StatementId}
    {assignment : Nat -> Nat}
    (holds : Satisfies (rows layout statementId) assignment) :
    Satisfies (ProductionMemoryCarryRows.rows layout.carry) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem state_rows_hold
    {candidate : Id} {rowVariables : Nat}
    {layout : Layout candidate rowVariables}
    {statementId : ProductPoseidon2.StatementId}
    {assignment : Nat -> Nat}
    (holds : Satisfies (rows layout statementId) assignment) :
    Satisfies
      (ProductionSuccessorStateBindingRowsFor.rows candidate
        layout.stateHashBase layout.state statementId) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem ccs_rows_hold
    {candidate : Id} {rowVariables : Nat}
    {layout : Layout candidate rowVariables}
    {statementId : ProductPoseidon2.StatementId}
    {assignment : Nat -> Nat}
    (holds : Satisfies (rows layout statementId) assignment) :
    Satisfies (ProductionMemoryBatchCcsLinkRowsFor.rows layout.ccs)
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem digest_link_rows_hold
    {candidate : Id} {rowVariables : Nat}
    {layout : Layout candidate rowVariables}
    {statementId : ProductPoseidon2.StatementId}
    {assignment : Nat -> Nat}
    (holds : Satisfies (rows layout statementId) assignment) :
    Satisfies (digestLinkRows layout statementId) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem prefix_with_carry_placed
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    {layout : Layout candidate rowVariables} (valid : layout.Valid)
    {assignment : Nat -> Nat}
    {priorPrefix : Prefix candidate fullShape}
    {headers : FPrime.ChainHeaders Digest.Value}
    {carry : ProductionSuccessorStateBinding.MemoryCarry}
    (prefixPlaced : PrefixPlaced contract layout assignment priorPrefix)
    (carryParsed : MemoryCarryPublicRows.ParsedColumnsMatch
      layout.carry.reference assignment headers carry) :
    ProductionSuccessorStateBindingRowsFor.Placed contract layout.state
      assignment (priorPrefix.withCarry carry) := by
  refine
    { invocation := prefixPlaced.invocation
      realRows := prefixPlaced.realRows
      initialApplication := prefixPlaced.initialApplication
      application := prefixPlaced.application
      running := prefixPlaced.running
      initialCarry := prefixPlaced.initialCarry
      carry := ?_ }
  intro index
  rw [valid.exactCarryColumns index,
    ProductionMemoryCarryFields.encode_get]
  exact carryParsed.placed (ProductionMemoryCarryFields.tagAt index)

/-- The state hash and NIFS verifier read the same physical running window. -/
theorem prefix_running_eq_claim
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    {layout : Layout candidate rowVariables} (valid : layout.Valid)
    {assignment : Nat -> Nat}
    {priorPrefix : Prefix candidate fullShape}
    {claim : ProductionFieldNativeFullClaim.Value candidate fullShape}
    (prefixPlaced : PrefixPlaced contract layout assignment priorPrefix)
    (claimPlaced : ProductionFullClaimCarrierLayoutFor.Placed contract
      layout.ccs.carrier assignment claim) :
    priorPrefix.running = claim.recursiveState := by
  apply ProductionSuccessorStateBinding.runningNativeFields_injective
  apply List.ext_get
  · rw [ProductionSuccessorStateBindingRowsFor.runningNativeFields_length
        contract priorPrefix.running,
      ProductionSuccessorStateBindingRowsFor.runningNativeFields_length
        contract claim.recursiveState]
  · intro index leftBound rightBound
    have coordinateBound :
        index < ProductNifsCodec.runningFieldCountFor rowVariables := by
      simpa [ProductionSuccessorStateBindingRowsFor.runningNativeFields_length
        contract priorPrefix.running] using leftBound
    let coordinate : Fin
        (ProductNifsCodec.runningFieldCountFor rowVariables) :=
      ⟨index, coordinateBound⟩
    have prefixCoordinate := prefixPlaced.running coordinate
    have claimCoordinate := claimPlaced.running coordinate
    rw [valid.exactRunningColumns coordinate] at prefixCoordinate
    calc
      (ProductionSuccessorStateBinding.runningNativeFields
          priorPrefix.running)[index]'leftBound =
          assignment (layout.ccs.carrier.runningColumn coordinate) := by
        exact (by simpa [coordinate] using prefixCoordinate.symm)
      _ = (ProductionSuccessorStateBinding.runningNativeFields
          claim.recursiveState)[index]'rightBound := by
        exact (by simpa [coordinate,
          ProductionSuccessorStateBinding.runningNativeFields] using
            claimCoordinate)

private theorem digest_link_sound
    {candidate : Id} {rowVariables : Nat}
    {layout : Layout candidate rowVariables}
    {statementId : ProductPoseidon2.StatementId}
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout statementId) assignment)
    (lane : Fin 4) :
    lcEval assignment
        ((ProductionSuccessorStateBindingRowsFor.builder candidate
          layout.stateHashBase layout.state statementId).lanes
            (ProductionSuccessorStateBinding.outputLane lane)) =
      assignment (layout.ccs.core.stateDigestColumn lane) := by
  have rowHolds : RowHolds assignment
      (KEquality.equalityRow
        ((ProductionSuccessorStateBindingRowsFor.builder candidate
          layout.stateHashBase layout.state statementId).lanes
            (ProductionSuccessorStateBinding.outputLane lane))
        [(layout.ccs.core.stateDigestColumn lane, 1)]) :=
    digest_link_rows_hold holds _ (by
      fin_cases lane <;> simp [digestLinkRows])
  have equal := (KEquality.equalityRow_iff assignment _ _ one).1 rowHolds
  simpa [lcEval, Nat.mod_eq_of_lt
    (canonical (layout.ccs.core.stateDigestColumn lane))] using equal

theorem stateDigestPlaced
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    {layout : Layout candidate rowVariables} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (statementId : ProductPoseidon2.StatementId)
    (prior : ProductionSuccessorStateBinding.Value candidate fullShape)
    (priorPlaced : ProductionSuccessorStateBindingRowsFor.Placed contract
      layout.state assignment prior)
    (holds : Satisfies (rows layout statementId) assignment) :
    ProductionMemoryBatchCcsLinkRowsFor.StateDigestPlaced layout.ccs assignment
      (ProductionSuccessorStateBinding.outputDigest statementId prior) := by
  intro lane
  have hashLane :=
    ProductionSuccessorStateBindingRowsFor.rows_imply_outputDigest_lane
      contract canonical one statementId prior priorPlaced
      layout.stateHashBase (state_rows_hold holds) lane
  exact (digest_link_sound canonical one holds lane).symm.trans hashLane

structure Result
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    (layout : Layout candidate rowVariables) (assignment : Nat -> Nat)
    (headers : FPrime.ChainHeaders Digest.Value)
    (statementId : ProductPoseidon2.StatementId)
    (priorPrefix : Prefix candidate fullShape)
    (claim : ProductionFieldNativeFullClaim.Value candidate fullShape)
    (memory : ProductionMemoryCheckedBatchRows.Result
      layout.ccs.core.batch.frame.memory assignment headers)
    (carry : ProductionMemoryCarryRows.Sound
      layout.carry assignment headers)
    (prior : ProductionSuccessorStateBinding.Value candidate fullShape) : Prop where
  priorExact : prior = priorPrefix.withCarry carry.value
  priorRunningExact : prior.running = claim.recursiveState
  priorPlaced : ProductionSuccessorStateBindingRowsFor.Placed
    contract layout.state assignment prior
  memoryStartWire : memory.boundary 0 = carry.value
  stateDigestPlaced : ProductionMemoryBatchCcsLinkRowsFor.StateDigestPlaced
    layout.ccs assignment
      (ProductionSuccessorStateBinding.outputDigest statementId prior)
  ccsFullMatches : ProductionMemoryBoundCcsPublic.FullMatches claim.ccsPublic
    (ProductionSuccessorStateBinding.outputDigest statementId prior)
    memory.suffixBatch

theorem rows_imply_exact_prior_state_and_fullMatches
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    {layout : Layout candidate rowVariables} (valid : layout.Valid)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (headers : FPrime.ChainHeaders Digest.Value)
    (statementId : ProductPoseidon2.StatementId)
    (priorPrefix : Prefix candidate fullShape)
    (prefixPlaced : PrefixPlaced contract layout assignment priorPrefix)
    (claim : ProductionFieldNativeFullClaim.Value candidate fullShape)
    (claimPlaced : ProductionFullClaimCarrierLayoutFor.Placed contract
      layout.ccs.carrier assignment claim)
    (carryHeadersPlaced : MemoryCarryRows.HeadersPlaced
      layout.carry.carry assignment headers)
    (memory : ProductionMemoryCheckedBatchRows.Result
      layout.ccs.core.batch.frame.memory assignment headers)
    (holds : Satisfies (rows layout statementId) assignment) :
    exists carry : ProductionMemoryCarryRows.Sound
        layout.carry assignment headers,
      exists prior : ProductionSuccessorStateBinding.Value candidate fullShape,
        Result contract layout assignment headers statementId priorPrefix claim
          memory carry prior := by
  let carry := ProductionMemoryCarryRows.derive headers canonical one
    carryHeadersPlaced (carry_rows_hold holds)
  let prior := priorPrefix.withCarry carry.value
  have priorPlaced : ProductionSuccessorStateBindingRowsFor.Placed contract
      layout.state assignment prior :=
    prefix_with_carry_placed contract valid prefixPlaced carry.parsed
  have priorRunningExact : prior.running = claim.recursiveState := by
    change priorPrefix.running = claim.recursiveState
    exact prefix_running_eq_claim contract valid prefixPlaced claimPlaced
  have digestPlaced := stateDigestPlaced contract canonical one statementId
    prior priorPlaced holds
  have startParsed : MemoryCarryPublicRows.ParsedColumnsMatch
      layout.carry.reference assignment headers (memory.boundary 0) := by
    rw [valid.initialCarryBoundary]
    exact memory.boundaryParsed 0
  have memoryStartWire : memory.boundary 0 = carry.value :=
    ProductionMemoryCarryRows.parsed_unique startParsed carry.parsed
  have ccsFullMatches :=
    ProductionMemoryBatchCcsLinkRowsFor.rows_imply_fullMatches valid.ccsValid
      canonical one claimPlaced digestPlaced memory (ccs_rows_hold holds)
  exact ⟨carry, prior,
    { priorExact := rfl
      priorRunningExact := priorRunningExact
      priorPlaced := priorPlaced
      memoryStartWire := memoryStartWire
      stateDigestPlaced := digestPlaced
      ccsFullMatches := ccsFullMatches }⟩

namespace Result

theorem prior_memoryCarry_eq_memory_start
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    {contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape}
    {layout : Layout candidate rowVariables} {assignment : Nat -> Nat}
    {headers : FPrime.ChainHeaders Digest.Value}
    {statementId : ProductPoseidon2.StatementId}
    {priorPrefix : Prefix candidate fullShape}
    {claim : ProductionFieldNativeFullClaim.Value candidate fullShape}
    {memory : ProductionMemoryCheckedBatchRows.Result
      layout.ccs.core.batch.frame.memory assignment headers}
    {carry : ProductionMemoryCarryRows.Sound
      layout.carry assignment headers}
    {prior : ProductionSuccessorStateBinding.Value candidate fullShape}
    (result : Result contract layout assignment headers statementId
      priorPrefix claim memory carry prior) :
    prior.memoryCarry = memory.boundary 0 := by
  rw [result.priorExact]
  exact result.memoryStartWire.symm

end Result

def rowCount (candidate : Id) (rowVariables : Nat) : Nat :=
  178 +
    ProductionSuccessorStateBindingRowsFor.successorPermutationCount
      rowVariables * 352 +
    ProductionMemoryBatchCcsLinkRows.rowCount candidate + 4

theorem rows_length_exact
    {candidate : Id} {rowVariables : Nat}
    {layout : Layout candidate rowVariables}
    (valid : layout.Valid) (statementId : ProductPoseidon2.StatementId) :
    (rows layout statementId).length = rowCount candidate rowVariables := by
  simp [rows, rowCount, ProductionMemoryCarryRows.rows_length_exact,
    ProductionSuccessorStateBindingRowsFor.rows_length_exact,
    ProductionMemoryBatchCcsLinkRowsFor.rows_length_exact valid.ccsValid,
    digestLinkRows]
  omega

end Nightstream.Implementation.NebulaV2.ProductionPaperPriorStateAuthorityRowsFor
