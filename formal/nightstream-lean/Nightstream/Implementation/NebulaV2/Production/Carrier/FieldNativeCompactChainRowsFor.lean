import Nightstream.Implementation.NebulaV2.Commitment.Compact.CheckedStepChainRows
import Nightstream.Implementation.NebulaV2.Commitment.Compact.ChainHeaderRows
import Nightstream.Implementation.NebulaV2.NIFS.Running.RunningParser
import Nightstream.Implementation.NebulaV2.Production.Carrier.CarrierLayoutFor
import Nightstream.Implementation.NebulaV2.Production.Memory.CheckedBatchRows

/-!
Contract: field-native compact-chain rows for the selected V2 paper claim.

The V2 profile has one checked memory step in each fresh claim. This block
computes the operations, initial-snapshot, and final-snapshot compact-chain
updates from the exact native bundle columns in that claim. It also derives
the two fixed chain headers from the verifier-key seed manifest.

There is no second authority-bearing bundle decoder. Structural column
equalities make the compact-token input the same physical field carrier that
PiCCS and NIFS verify.

Does not own the application-to-lane projection, prechallenge extraction,
Poseidon2 security, Ajtai binding, generated-artifact containment, or Rust
refinement.

Emits constraints: fixed header rows and three compact lane updates.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductionFieldNativeCompactChainRowsFor

open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.CommitmentBundle
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete

/-- Every candidate has a nonempty checked-step batch. The production V2
validity certificate below further restricts the candidate to `e1`. -/
def firstStep (candidate : Id) :
    Fin (ProductionMemoryCheckedBatchRows.StepCount candidate) :=
  ⟨0, by cases candidate <;> decide⟩

def componentIndex : Component -> Nat
  | .full => 0
  | .operations => 1
  | .initialSnapshot => 2
  | .finalSnapshot => 3

/-- Exact position of one component coordinate in the field-native bundle
carrier. -/
def bundleCoordinate (component : Component)
    (coordinate : CommitmentBundleCodec.Coordinate) :
    Fin ProductionProfileCandidates.bundleFieldCoordinates :=
  ⟨componentIndex component * MemoryWireGeometry.commitmentFieldCount +
      coordinate.val, by
    have coordinateBound : coordinate.val < 972 := coordinate.isLt
    cases component <;>
      norm_num [componentIndex,
        MemoryWireGeometry.commitmentFieldCount_exact,
        ProductionProfileCandidates.bundleFieldCoordinates] <;>
      omega⟩

structure Layout where
  headers : CompactChainHeaderRows.Layout
  operations : CompactLaneStepRows.Layout
  initialSnapshot : CompactLaneStepRows.Layout
  finalSnapshot : CompactLaneStepRows.Layout

def claimLayout
    {candidate : Id}
    (checked : ProductionMemoryCheckedBatchRows.Layout candidate) :
    ProductionMemoryClaimRows.Layout :=
  (checked.steps (firstStep candidate)).claim

def claimRootColumn
    {candidate : Id}
    (checked : ProductionMemoryCheckedBatchRows.Layout candidate)
    (stage : MemoryClaimCodec.RootStage)
    (role : MemoryClaimCodec.RootRole) (lane : Fin 4) : Nat :=
  Relabel.column
    ((claimLayout checked).reference.fieldColumnMap (.root stage role lane))
    CanonicalU64.varCol

/-- Static alias and schedule certificate. `candidateExact` prevents a
different checked-step batching relation from sharing the V2 key identity. -/
structure Layout.Valid
    (manifest : SeedSchedule.Manifest)
    {candidate : Id} {rowVariables : Nat}
    (carrier : ProductionFullClaimCarrierLayoutFor.Layout candidate rowVariables)
    (checked : ProductionMemoryCheckedBatchRows.Layout candidate)
    (layout : Layout) : Prop where
  candidateExact : candidate = .e1
  manifestProfile : manifest.profile = identity candidate
  headersValid : layout.headers.Valid manifest
  operationsValid : layout.operations.Valid manifest .operations
  initialSnapshotValid : layout.initialSnapshot.Valid manifest .memory
  finalSnapshotValid : layout.finalSnapshot.Valid manifest .memory
  operationsHeader : layout.headers.operations.digestColumn =
    (checked.boundaries 0).carry.headerColumn .operations
  initialSnapshotHeader : layout.headers.memory.digestColumn =
    (checked.boundaries 0).carry.headerColumn .initialSnapshot
  finalSnapshotHeader : layout.headers.memory.digestColumn =
    (checked.boundaries 0).carry.headerColumn .finalSnapshot
  operationsCommitment : layout.operations.token.commitmentFieldColumn =
    fun coordinate => carrier.bundleColumn
      (bundleCoordinate .operations coordinate)
  initialSnapshotCommitment :
    layout.initialSnapshot.token.commitmentFieldColumn =
      fun coordinate => carrier.bundleColumn
        (bundleCoordinate .initialSnapshot coordinate)
  finalSnapshotCommitment :
    layout.finalSnapshot.token.commitmentFieldColumn =
      fun coordinate => carrier.bundleColumn
        (bundleCoordinate .finalSnapshot coordinate)
  operationsPrior : layout.operations.priorDigestColumn =
    claimRootColumn checked .seenBefore .operations
  operationsAfter : layout.operations.afterDigestColumn =
    claimRootColumn checked .seenAfter .operations
  initialSnapshotPrior : layout.initialSnapshot.priorDigestColumn =
    claimRootColumn checked .seenBefore .initialSnapshot
  initialSnapshotAfter : layout.initialSnapshot.afterDigestColumn =
    claimRootColumn checked .seenAfter .initialSnapshot
  finalSnapshotPrior : layout.finalSnapshot.priorDigestColumn =
    claimRootColumn checked .seenBefore .finalSnapshot
  finalSnapshotAfter : layout.finalSnapshot.afterDigestColumn =
    claimRootColumn checked .seenAfter .finalSnapshot
  operationsIndex : layout.operations.linkFrame.indexColumn =
    (claimLayout checked).counterValueColumn .stepIndex
  initialSnapshotIndex : layout.initialSnapshot.linkFrame.indexColumn =
    (claimLayout checked).counterValueColumn .stepIndex
  finalSnapshotIndex : layout.finalSnapshot.linkFrame.indexColumn =
    (claimLayout checked).counterValueColumn .stepIndex

def rows (manifest : SeedSchedule.Manifest) (layout : Layout) : List Row :=
  CompactChainHeaderRows.rows manifest layout.headers ++
    CompactLaneStepRows.rows manifest .operations layout.operations ++
    CompactLaneStepRows.rows manifest .memory layout.initialSnapshot ++
    CompactLaneStepRows.rows manifest .memory layout.finalSnapshot

def rowCount : Nat := 445167

theorem rows_length_exact
    {manifest : SeedSchedule.Manifest}
    {candidate : Id} {rowVariables : Nat}
    {carrier : ProductionFullClaimCarrierLayoutFor.Layout candidate rowVariables}
    {checked : ProductionMemoryCheckedBatchRows.Layout candidate}
    {layout : Layout} (valid : layout.Valid manifest carrier checked) :
    (rows manifest layout).length = rowCount := by
  simp [rows, rowCount,
    CompactChainHeaderRows.rows_length_exact valid.headersValid,
    CompactLaneStepRows.rows_length_exact valid.operationsValid,
    CompactLaneStepRows.rows_length_exact valid.initialSnapshotValid,
    CompactLaneStepRows.rows_length_exact valid.finalSnapshotValid]

private theorem header_rows_hold
    {manifest : SeedSchedule.Manifest} {layout : Layout}
    {assignment : Nat -> Nat}
    (holds : Satisfies (rows manifest layout) assignment) :
    Satisfies (CompactChainHeaderRows.rows manifest layout.headers)
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem operations_rows_hold
    {manifest : SeedSchedule.Manifest} {layout : Layout}
    {assignment : Nat -> Nat}
    (holds : Satisfies (rows manifest layout) assignment) :
    Satisfies
      (CompactLaneStepRows.rows manifest .operations layout.operations)
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem initial_snapshot_rows_hold
    {manifest : SeedSchedule.Manifest} {layout : Layout}
    {assignment : Nat -> Nat}
    (holds : Satisfies (rows manifest layout) assignment) :
    Satisfies
      (CompactLaneStepRows.rows manifest .memory layout.initialSnapshot)
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem final_snapshot_rows_hold
    {manifest : SeedSchedule.Manifest} {layout : Layout}
    {assignment : Nat -> Nat}
    (holds : Satisfies (rows manifest layout) assignment) :
    Satisfies
      (CompactLaneStepRows.rows manifest .memory layout.finalSnapshot)
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private def coordinateRow
    (coordinate : CommitmentBundleCodec.Coordinate) :
    Fin ProductCommitmentAlgebra.Rank :=
  ⟨coordinate.val / CompactCommit.ringDegree, by
    have bound := coordinate.isLt
    change coordinate.val < 972 at bound
    change coordinate.val / 54 < 18
    omega⟩

private def coordinateLane
    (coordinate : CommitmentBundleCodec.Coordinate) :
    Fin CompactCommit.ringDegree :=
  ⟨coordinate.val % CompactCommit.ringDegree,
    Nat.mod_lt _ (by decide)⟩

private theorem coordinate_recompose
    (coordinate : CommitmentBundleCodec.Coordinate) :
    (coordinateRow coordinate).val * CompactCommit.ringDegree +
        (coordinateLane coordinate).val = coordinate.val := by
  simpa [coordinateRow, coordinateLane, Nat.mul_comm] using
    Nat.div_add_mod coordinate.val CompactCommit.ringDegree

private theorem encoded_bundle_coordinate
    (bundle : CommitmentBundleCodec.Value) (component : Component)
    (coordinate : CommitmentBundleCodec.Coordinate) :
    (ProductionFieldNativeFullClaim.bundleFields bundle).getD
        (bundleCoordinate component coordinate).val 0 =
      ProductNifsCodec.codecField (bundle component coordinate) := by
  let row := coordinateRow coordinate
  let lane := coordinateLane coordinate
  have mappedCoordinate :
      (⟨row.val * CompactCommit.ringDegree + lane.val, by
        change row.val * 54 + lane.val < 972
        have rowBound := row.isLt
        have laneBound := lane.isLt
        change row.val < 18 at rowBound
        change lane.val < 54 at laneBound
        omega⟩ : CommitmentBundleCodec.Coordinate) = coordinate := by
    apply Fin.ext
    exact coordinate_recompose coordinate
  have encoded := ProductNifsRunningParser.bundleCodec_getD
    (ProductNifsCodec.codecBundle bundle) component row lane
  change
    (ProductNifsCodec.bundleCodec.encode
      (ProductNifsCodec.codecBundle bundle)).getD
        (bundleCoordinate component coordinate).val 0 = _
  have indexExact :
      ProductNifsRunningParser.componentIndex component * 972 +
          row.val * CompactCommit.ringDegree + lane.val =
        (bundleCoordinate component coordinate).val := by
    have recomposed :
        row.val * CompactCommit.ringDegree + lane.val = coordinate.val :=
      coordinate_recompose coordinate
    cases component <;>
      simp [ProductNifsRunningParser.componentIndex, bundleCoordinate,
        componentIndex, MemoryWireGeometry.commitmentFieldCount_exact] <;>
      omega
  rw [← indexExact]
  have encodedCompact :
      (ProductNifsCodec.bundleCodec.encode
        (ProductNifsCodec.codecBundle bundle)).getD
          (ProductNifsRunningParser.componentIndex component * 972 +
            row.val * CompactCommit.ringDegree + lane.val) 0 =
        ProductNifsCodec.codecField
          (bundle component
            ⟨row.val * CompactCommit.ringDegree + lane.val, by
              have rowBound := row.isLt
              have laneBound := lane.isLt
              change row.val < 18 at rowBound
              change lane.val < 54 at laneBound
              change row.val * 54 + lane.val < 972
              omega⟩) := by
    simpa [CompactCommit.ringDegree, ringDegree] using encoded
  rw [mappedCoordinate] at encodedCompact
  exact encodedCompact

private theorem component_placed
    {candidate : Id} {rowVariables : Nat}
    {carrier : ProductionFullClaimCarrierLayoutFor.Layout candidate rowVariables}
    {assignment : Nat -> Nat}
    {fullShape : Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    {contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape}
    {value : ProductionFieldNativeFullClaim.Value candidate fullShape}
    (placed : ProductionFullClaimCarrierLayoutFor.Placed contract carrier
      assignment value)
    (laneLayout : CompactLaneStepRows.Layout) (component : Component)
    (columns : laneLayout.token.commitmentFieldColumn =
      fun coordinate => carrier.bundleColumn
        (bundleCoordinate component coordinate)) :
    CompactTokenRows.CommitmentPlaced laneLayout.token assignment
      (value.commitmentBundle component) := by
  intro coordinate
  change assignment (laneLayout.token.commitmentFieldColumn coordinate) =
    (value.commitmentBundle component coordinate).val
  rw [columns]
  let index := bundleCoordinate component coordinate
  let listIndex : Fin
      (ProductionFieldNativeFullClaim.bundleFields
        value.commitmentBundle).length :=
    ⟨index.val, by
      rw [ProductionFieldNativeFullClaim.bundleFields_length]
      exact index.isLt⟩
  calc
    assignment (carrier.bundleColumn index) =
        ((ProductionFieldNativeFullClaim.bundleFields
          value.commitmentBundle).get listIndex).val := by
      simpa [index, listIndex] using placed.bundle index
    _ = ((ProductionFieldNativeFullClaim.bundleFields
          value.commitmentBundle).getD index.val 0).val := by
      rw [List.getD_eq_get
        (ProductionFieldNativeFullClaim.bundleFields value.commitmentBundle)
        0 listIndex]
    _ = (value.commitmentBundle component coordinate).val := by
      exact congrArg Fin.val
        (encoded_bundle_coordinate value.commitmentBundle component coordinate)

private def selectedRoot (claim : MemoryClaimCodec.Claim)
    (stage : MemoryClaimCodec.RootStage)
    (role : MemoryClaimCodec.RootRole) : Digest.Value :=
  let roots :=
    match stage with
    | .precommit => claim.dPre
    | .seenBefore => claim.dSeenBefore
    | .seenAfter => claim.dSeenAfter
  match role with
  | .operations => roots.operations
  | .initialSnapshot => roots.initialSnapshot
  | .finalSnapshot => roots.finalSnapshot

private theorem root_placed
    {candidate : Id}
    {checked : ProductionMemoryCheckedBatchRows.Layout candidate}
    {assignment : Nat -> Nat} {claim : MemoryClaimCodec.Claim}
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      (claimLayout checked).reference assignment claim)
    (stage : MemoryClaimCodec.RootStage)
    (role : MemoryClaimCodec.RootRole) :
    CompactChainHashFrameRows.DigestPlaced
      (claimRootColumn checked stage role) assignment
      (selectedRoot claim stage role) := by
  intro lane
  have exactField := parsed.fields
    (MemoryClaimFieldRows.Slot.root stage role lane)
  cases stage <;> cases role <;> exact exactField

structure Result
    (manifest : SeedSchedule.Manifest)
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    (carrier : ProductionFullClaimCarrierLayoutFor.Layout candidate rowVariables)
    (checked : ProductionMemoryCheckedBatchRows.Layout candidate)
    (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (headers : ChainHeaders Digest.Value)
    (value : ProductionFieldNativeFullClaim.Value candidate fullShape)
    (claim : MemoryClaimCodec.Claim) : Prop where
  headersExact :
    (forall lane : Fin 4,
      (headers.operations.lanes lane).val =
        CompactChainPoseidonRows.pureHash
          (.header .operations manifest.profile manifest.plan) lane.val) /\
    (forall lane : Fin 4,
      (headers.memory.lanes lane).val =
        CompactChainPoseidonRows.pureHash
          (.header .memory manifest.profile manifest.plan) lane.val)
  operations : CompactCheckedStepChainRows.LaneExact manifest .operations
    claim.stepIndex layout.operations assignment canonical
    (value.commitmentBundle .operations)
    claim.dSeenBefore.operations claim.dSeenAfter.operations
  initialSnapshot : CompactCheckedStepChainRows.LaneExact manifest .memory
    claim.stepIndex layout.initialSnapshot assignment canonical
    (value.commitmentBundle .initialSnapshot)
    claim.dSeenBefore.initialSnapshot claim.dSeenAfter.initialSnapshot
  finalSnapshot : CompactCheckedStepChainRows.LaneExact manifest .memory
    claim.stepIndex layout.finalSnapshot assignment canonical
    (value.commitmentBundle .finalSnapshot)
    claim.dSeenBefore.finalSnapshot claim.dSeenAfter.finalSnapshot

/-- A satisfying block derives all three chain updates from the exact bundle
in the NIFS-selected field-native claim. -/
theorem exact
    {manifest : SeedSchedule.Manifest}
    {candidate : Id} {rowVariables : Nat}
    {carrier : ProductionFullClaimCarrierLayoutFor.Layout candidate rowVariables}
    {checked : ProductionMemoryCheckedBatchRows.Layout candidate}
    {layout : Layout} {assignment : Nat -> Nat}
    {fullShape : Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    {contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape}
    {value : ProductionFieldNativeFullClaim.Value candidate fullShape}
    {headers : ChainHeaders Digest.Value}
    (valid : layout.Valid manifest carrier checked)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (headersPlaced : ProductionMemoryCheckedBatchRows.HeadersPlaced checked
      assignment headers)
    (placed : ProductionFullClaimCarrierLayoutFor.Placed contract carrier
      assignment value)
    (memory : ProductionMemoryCheckedBatchRows.Result checked assignment headers)
    (holds : Satisfies (rows manifest layout) assignment) :
    Result manifest carrier checked layout assignment canonical headers value
      (memory.claim (firstStep candidate)) := by
  let claim := memory.claim (firstStep candidate)
  have parsed : MemoryClaimRows.ParsedColumnsMatch
      (claimLayout checked).reference assignment claim :=
    memory.claimParsed (firstStep candidate)
  have operationsCommitment := component_placed placed layout.operations
    .operations valid.operationsCommitment
  have initialCommitment := component_placed placed layout.initialSnapshot
    .initialSnapshot valid.initialSnapshotCommitment
  have finalCommitment := component_placed placed layout.finalSnapshot
    .finalSnapshot valid.finalSnapshotCommitment
  have operationsPrior := root_placed parsed .seenBefore .operations
  have operationsAfter := root_placed parsed .seenAfter .operations
  have initialPrior := root_placed parsed .seenBefore .initialSnapshot
  have initialAfter := root_placed parsed .seenAfter .initialSnapshot
  have finalPrior := root_placed parsed .seenBefore .finalSnapshot
  have finalAfter := root_placed parsed .seenAfter .finalSnapshot
  have indexExact :
      assignment ((claimLayout checked).counterValueColumn .stepIndex) =
        claim.stepIndex.val := parsed.counters .stepIndex
  have headerOutputs := CompactChainHeaderRows.outputs_exact
    valid.headersValid canonical one (header_rows_hold holds)
  have headerExact :
      (forall lane : Fin 4,
        (headers.operations.lanes lane).val =
          CompactChainPoseidonRows.pureHash
            (.header .operations manifest.profile manifest.plan) lane.val) /\
      (forall lane : Fin 4,
        (headers.memory.lanes lane).val =
          CompactChainPoseidonRows.pureHash
            (.header .memory manifest.profile manifest.plan) lane.val) := by
    constructor
    · intro lane
      calc
        (headers.operations.lanes lane).val =
            assignment
              ((checked.boundaries 0).carry.headerColumn .operations lane) := by
          symm
          simpa [ChainHeaders.roots, MemoryClaimCodec.rootValue] using
            headersPlaced 0 .operations lane
        _ = assignment (layout.headers.operations.digestColumn lane) := by
          rw [valid.operationsHeader]
        _ = CompactChainPoseidonRows.pureHash
            (.header .operations manifest.profile manifest.plan) lane.val :=
          headerOutputs.1 lane
    · intro lane
      calc
        (headers.memory.lanes lane).val =
            assignment
              ((checked.boundaries 0).carry.headerColumn
                .initialSnapshot lane) := by
          symm
          simpa [ChainHeaders.roots, MemoryClaimCodec.rootValue] using
            headersPlaced 0 .initialSnapshot lane
        _ = assignment (layout.headers.memory.digestColumn lane) := by
          rw [valid.initialSnapshotHeader]
        _ = CompactChainPoseidonRows.pureHash
            (.header .memory manifest.profile manifest.plan) lane.val :=
          headerOutputs.2 lane
  refine
    { headersExact := headerExact
      operations := ?_
      initialSnapshot := ?_
      finalSnapshot := ?_ }
  · apply CompactLaneStepRows.after_root_exact valid.operationsValid canonical
      one
    · rw [valid.operationsIndex]
      exact indexExact
    · exact operationsCommitment
    · rw [valid.operationsPrior]
      exact operationsPrior
    · rw [valid.operationsAfter]
      exact operationsAfter
    · exact operations_rows_hold holds
  · apply CompactLaneStepRows.after_root_exact valid.initialSnapshotValid
      canonical one
    · rw [valid.initialSnapshotIndex]
      exact indexExact
    · exact initialCommitment
    · rw [valid.initialSnapshotPrior]
      exact initialPrior
    · rw [valid.initialSnapshotAfter]
      exact initialAfter
    · exact initial_snapshot_rows_hold holds
  · apply CompactLaneStepRows.after_root_exact valid.finalSnapshotValid
      canonical one
    · rw [valid.finalSnapshotIndex]
      exact indexExact
    · exact finalCommitment
    · rw [valid.finalSnapshotPrior]
      exact finalPrior
    · rw [valid.finalSnapshotAfter]
      exact finalAfter
    · exact final_snapshot_rows_hold holds

end Nightstream.Implementation.NebulaV2.ProductionFieldNativeCompactChainRowsFor
