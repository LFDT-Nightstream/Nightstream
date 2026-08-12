import Nightstream.Implementation.NebulaV2.Memory.Snapshot.Rows

set_option autoImplicit false

namespace tests.NebulaV2SnapshotRows

open Nightstream.Implementation.NebulaV2.SnapshotRows
open Nightstream.Protocol.NebulaV2

def zeroSnapshot : Snapshot :=
  fun _ => ⟨0, 0⟩

def futureSnapshot : Snapshot :=
  fun index =>
    if index.val = 0 then ⟨0, 8⟩ else ⟨0, 0⟩

theorem honest_zero_scan_is_accepted :
    Accepts zeroSnapshot zeroSnapshot 0 0 := by
  apply accepts_complete (by decide) (by decide)
  · exact Snapshot.ofImage_validAt_zero (fun _ => by decide)
  · exact Snapshot.ofImage_validAt_zero (fun _ => by decide)

/-- A future timestamp is within 23 bits but cannot satisfy the exact
segment-relative comparison relation. -/
theorem future_initial_timestamp_is_rejected :
    ¬ Accepts futureSnapshot zeroSnapshot 7 7 := by
  intro accepted
  have valid := (accepts_sound accepted).1
  have atZero := valid (⟨0, by decide⟩ : Fin scannedCells)
  simp [futureSnapshot] at atZero

theorem leq_witness_cannot_wrap
    (witness : LeqWitness (timestampLimit - 1) 0) : False := by
  have ordered := witness.sound
  simp [timestampLimit, timestampBits] at ordered

end tests.NebulaV2SnapshotRows
