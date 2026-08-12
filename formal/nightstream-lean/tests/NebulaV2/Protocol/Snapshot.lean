import Nightstream.Protocol.NebulaV2

set_option autoImplicit false

namespace tests.NebulaV2Snapshot

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.Memory

def zeroImage : Fin scannedCells → Nat := fun _ => 0

def honestInitial : Snapshot := Snapshot.ofImage zeroImage

theorem zero_image_in_range : Snapshot.ImageInRange zeroImage := by
  intro _
  change 0 < valueLimit
  decide

theorem honest_initial_valid : honestInitial.ValidAt 0 :=
  Snapshot.ofImage_validAt_zero zero_image_in_range

theorem honest_scan_has_exact_length :
    honestInitial.tupleList.length = scannedCells :=
  Snapshot.tupleList_length honestInitial

theorem honest_scan_has_no_repeated_index :
    (honestInitial.tupleList.map MemTuple.globalIndex).Nodup :=
  Snapshot.tupleList_indices_nodup honestInitial

/- Relative memory consistency does not establish the verifier's initial
memory. A constant-one image forms a valid empty segment but differs from the
verifier-owned zero image. -/
namespace MissingInitialAuthority

def fakeSnapshot : Snapshot := fun _ => ⟨1, 0⟩

theorem fake_snapshot_valid : fakeSnapshot.ValidAt 0 := by
  intro _
  constructor
  · change 1 < valueLimit
    decide
  · exact Nat.le_refl 0

theorem fake_empty_segment :
    ValidSegment fakeSnapshot fakeSnapshot 0 [] 0 where
  initialValid := fake_snapshot_valid
  finalValid := fake_snapshot_valid
  ordered := .nil 0
  balanced := by
    simp [Balanced, readTuples, writeTuples]

def zeroIndex : Fin scannedCells := ⟨0, by decide⟩

theorem fake_is_not_authoritative : fakeSnapshot ≠ honestInitial := by
  intro equal
  have valueEqual := congrArg (fun snapshot => (snapshot zeroIndex).value) equal
  change 1 = 0 at valueEqual
  omega

end MissingInitialAuthority

/- A raw multiset can satisfy relative balance while omitting almost all
cells. The Snapshot function and structural scan are therefore necessary. -/
namespace MissingExactCover

def onlyCell : MemTuple := ⟨0, 0, 0⟩

def incomplete : Multiset MemTuple := {onlyCell}

theorem incomplete_empty_balance : Balanced incomplete [] incomplete := by
  simp [Balanced, readTuples, writeTuples]

theorem incomplete_empty_execution : Executes incomplete 0 [] incomplete 0 :=
  .nil incomplete 0

def absentCell : MemTuple := ⟨0, 1, 0⟩

theorem absent_cell_is_missing : absentCell ∉ incomplete := by
  intro present
  have equal : absentCell = onlyCell := by
    simpa [incomplete] using present
  have indexEqual := congrArg MemTuple.globalIndex equal
  change 1 = 0 at indexEqual
  omega

end MissingExactCover

end tests.NebulaV2Snapshot
