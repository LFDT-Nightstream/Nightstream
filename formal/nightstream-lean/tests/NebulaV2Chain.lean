import Nightstream.Protocol.NebulaV2

set_option autoImplicit false

namespace tests.NebulaV2Chain

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.Memory

def zeroSnapshot : Snapshot := fun _ => ⟨0, 0⟩
def oneSnapshot : Snapshot := fun _ => ⟨1, 0⟩

theorem zero_valid : zeroSnapshot.ValidAt 0 := by
  intro _
  exact ⟨by change 0 < valueLimit; decide, Nat.le_refl 0⟩

theorem one_valid : oneSnapshot.ValidAt 0 := by
  intro _
  exact ⟨by change 1 < valueLimit; decide, Nat.le_refl 0⟩

def emptySegment
    (snapshot : Snapshot)
    (valid : snapshot.ValidAt 0) :
    ValidSegment snapshot snapshot 0 [] 0 where
  initialValid := valid
  finalValid := valid
  ordered := .nil 0
  balanced := by simp [Balanced, readTuples, writeTuples]

def twoEmptySegments :
    ValidChain zeroSnapshot 0 [[], []] zeroSnapshot 0 :=
  .cons (emptySegment zeroSnapshot zero_valid)
    (.cons (emptySegment zeroSnapshot zero_valid) (.nil zeroSnapshot 0))

theorem two_empty_segments_execute :
    Executes zeroSnapshot.tuples 0 [[], []].flatten zeroSnapshot.tuples 0 :=
  twoEmptySegments.executes

/- Two segments can each be valid relative to their own boundaries while the
shared boundary is false. A separate boundary-authority obligation is needed
before the ValidChain constructor can compose them. -/
namespace MissingBoundaryLink

def zeroSegment : ValidSegment zeroSnapshot zeroSnapshot 0 [] 0 :=
  emptySegment zeroSnapshot zero_valid

def oneSegment : ValidSegment oneSnapshot oneSnapshot 0 [] 0 :=
  emptySegment oneSnapshot one_valid

def zeroIndex : Fin scannedCells := ⟨0, by decide⟩

theorem boundary_snapshots_differ : zeroSnapshot ≠ oneSnapshot := by
  intro equal
  have valueEqual := congrArg (fun snapshot => (snapshot zeroIndex).value) equal
  change 0 = 1 at valueEqual
  omega

theorem individual_validity_does_not_supply_link :
    ValidSegment zeroSnapshot zeroSnapshot 0 [] 0 ∧
      ValidSegment oneSnapshot oneSnapshot 0 [] 0 ∧
      zeroSnapshot ≠ oneSnapshot :=
  ⟨zeroSegment, oneSegment, boundary_snapshots_differ⟩

end MissingBoundaryLink

end tests.NebulaV2Chain
