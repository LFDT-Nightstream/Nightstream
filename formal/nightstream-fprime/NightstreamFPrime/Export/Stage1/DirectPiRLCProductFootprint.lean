import NightstreamFPrime.Lifecycle.Types

namespace NightstreamFPrime.Export.Stage1.DirectPiRLCProductFootprint

def invocationCount : Nat := 31680
def rowCount : Nat := invocationCount * 2
def retainedFieldCount : Nat := invocationCount * 1
def retainedCoordinateCount : Nat := retainedFieldCount * 41

@[simp] theorem invocationCount_eq : invocationCount = 31680 := by
  rfl

@[simp] theorem rowCount_eq : rowCount = 63360 := by
  unfold rowCount
  rw [invocationCount_eq]

@[simp] theorem retainedFieldCount_eq : retainedFieldCount = 31680 := by
  unfold retainedFieldCount
  rw [invocationCount_eq]

@[simp] theorem retainedCoordinateCount_eq :
    retainedCoordinateCount = 1298880 := by
  unfold retainedCoordinateCount
  rw [retainedFieldCount_eq]

end NightstreamFPrime.Export.Stage1.DirectPiRLCProductFootprint
