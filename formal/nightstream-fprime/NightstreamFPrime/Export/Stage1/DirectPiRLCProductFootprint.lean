import NightstreamFPrime.Lifecycle.Types

/-!
Owns the fast closed-form footprint of the direct quotient-check compiler for
every PiRLC combination invocation. The separate bridge module proves that
the fixed invocation count equals the canonical Lean invocation list.

This module does not construct the final assignment or claim the complete
Stage 1 fit.
-/

namespace NightstreamFPrime.Export.Stage1.DirectPiRLCProductFootprint

def invocationCount : Nat := 33966
def rowCount : Nat := invocationCount * 2
def retainedFieldCount : Nat := invocationCount * 1
def retainedCoordinateCount : Nat := retainedFieldCount * 41

@[simp] theorem invocationCount_eq : invocationCount = 33966 := by
  rfl

@[simp] theorem rowCount_eq : rowCount = 67932 := by
  unfold rowCount
  rw [invocationCount_eq]

@[simp] theorem retainedFieldCount_eq : retainedFieldCount = 33966 := by
  unfold retainedFieldCount
  rw [invocationCount_eq]

@[simp] theorem retainedCoordinateCount_eq :
    retainedCoordinateCount = 1392606 := by
  unfold retainedCoordinateCount
  rw [retainedFieldCount_eq]

end NightstreamFPrime.Export.Stage1.DirectPiRLCProductFootprint
