import Nightstream.Implementation.NebulaV2.Commitment.Compact.ChainPoseidonRows

/-!
Contract: exact V2 operations and memory compact-chain header roots.

Assurance tier: implementation-to-protocol bridge.

Owns two fixed header frames, two Poseidon2 traces, their exact row count,
and the row-derived operations and shared-memory header equations.

Does not own carry-column reuse, generated absolute columns, Poseidon2
collision resistance, or Rust conformance.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.CompactChainHeaderRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.NebulaV2.CompactChainHashFrame
open Nightstream.Implementation.NebulaV2.CompactChainHashFrameRows
open Nightstream.Implementation.NebulaV2.CompactChainPoseidonRows
open Nightstream.Protocol.NebulaV2

structure LaneLayout where
  frame : HeaderLayout
  trace : Trace
  digestColumn : Fin 4 → Nat

structure Layout where
  operations : LaneLayout
  memory : LaneLayout

def laneRows (manifest : SeedSchedule.Manifest) (role : CompactCommit.Role)
    (layout : LaneLayout) : List Row :=
  Framed.rows (headerRows manifest role layout.frame) layout.trace

def rows (manifest : SeedSchedule.Manifest) (layout : Layout) : List Row :=
  laneRows manifest .operations layout.operations ++
    laneRows manifest .memory layout.memory

structure LaneLayout.Valid
    (manifest : SeedSchedule.Manifest) (role : CompactCommit.Role)
    (layout : LaneLayout) : Prop where
  inputColumns : layout.trace.inputColumns = layout.frame.inputColumns
  schedule : valueSchedules layout.trace.rounds =
    compactSchedule (.header role manifest.profile manifest.plan)
  traceValid : layout.trace.Valid
    (laneRows manifest role layout)
  outputColumns :
    (fun lane : Fin 4 => layout.trace.outputColumns.getD lane.val 0) =
      layout.digestColumn
  traceRowsLength : layout.trace.rows.length = 2413

structure Layout.Valid (manifest : SeedSchedule.Manifest)
    (layout : Layout) : Prop where
  operations : layout.operations.Valid manifest .operations
  memory : layout.memory.Valid manifest .memory

theorem laneRows_length
    {manifest : SeedSchedule.Manifest} {role : CompactCommit.Role}
    {layout : LaneLayout} (valid : layout.Valid manifest role) :
    (laneRows manifest role layout).length = 2424 := by
  simp [laneRows, Framed.rows, headerRows_length, valid.traceRowsLength]

theorem rows_length_exact
    {manifest : SeedSchedule.Manifest} {layout : Layout}
    (valid : layout.Valid manifest) :
    (rows manifest layout).length = 4848 := by
  simp [rows, laneRows_length valid.operations, laneRows_length valid.memory]

private theorem operations_rows_hold
    {manifest : SeedSchedule.Manifest} {layout : Layout}
    {assignment : Nat → Nat}
    (holds : Satisfies (rows manifest layout) assignment) :
    Satisfies (laneRows manifest .operations layout.operations)
      assignment := by
  intro row member
  exact holds row (List.mem_append_left _ member)

private theorem memory_rows_hold
    {manifest : SeedSchedule.Manifest} {layout : Layout}
    {assignment : Nat → Nat}
    (holds : Satisfies (rows manifest layout) assignment) :
    Satisfies (laneRows manifest .memory layout.memory) assignment := by
  intro row member
  exact holds row (List.mem_append_right _ member)

private theorem laneValid
    {manifest : SeedSchedule.Manifest} {role : CompactCommit.Role}
    {layout : LaneLayout} (valid : layout.Valid manifest role) :
    Framed.Valid (.header role manifest.profile manifest.plan)
      (headerRows manifest role layout.frame)
      layout.frame.inputColumns layout.trace where
  exactInputColumns := valid.inputColumns
  exactSchedule := valid.schedule
  traceValid := valid.traceValid

private theorem one_lane_exact
    {manifest : SeedSchedule.Manifest} {role : CompactCommit.Role}
    {layout : LaneLayout} {assignment : Nat → Nat}
    (valid : layout.Valid manifest role)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (laneRows manifest role layout) assignment) :
    ∀ lane : Fin 4,
      assignment (layout.digestColumn lane) =
        pureHash (.header role manifest.profile manifest.plan) lane.val := by
  have frameHolds := Framed.frame_rows_hold holds
  have frameExact := header_input_exact canonical one frameHolds
  have exact := Framed.output_exact (laneValid valid) canonical one
    frameExact holds
  intro lane
  calc
    assignment (layout.digestColumn lane) =
        assignment (layout.trace.outputColumns.getD lane.val 0) := by
      rw [← congrFun valid.outputColumns lane]
    _ = pureHash (.header role manifest.profile manifest.plan) lane.val := exact lane

/-- Both canonical chain headers are row consequences. The memory header is
shared by initial and final snapshot roles at the carry-link layer. -/
theorem outputs_exact
    {manifest : SeedSchedule.Manifest} {layout : Layout}
    {assignment : Nat → Nat}
    (valid : layout.Valid manifest)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows manifest layout) assignment) :
    (∀ lane : Fin 4,
      assignment (layout.operations.digestColumn lane) =
        pureHash (.header .operations manifest.profile manifest.plan) lane.val) ∧
      (∀ lane : Fin 4,
        assignment (layout.memory.digestColumn lane) =
          pureHash (.header .memory manifest.profile manifest.plan) lane.val) :=
  ⟨one_lane_exact valid.operations canonical one
      (operations_rows_hold holds),
    one_lane_exact valid.memory canonical one (memory_rows_hold holds)⟩

end Nightstream.Implementation.NebulaV2.CompactChainHeaderRows
