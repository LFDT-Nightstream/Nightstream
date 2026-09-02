import NightstreamFPrime.Lifecycle.Types

/-!
Owns the fast closed-form footprint of the direct five-product compiler for
every PiRLC combination invocation. The separate bridge module proves that
the fixed invocation count equals the canonical Lean invocation list.

This module does not construct the final assignment or claim the complete
Stage 1 fit.
-/

namespace NightstreamFPrime.Export.Stage1.DirectPiRLCProductFootprint

def invocationCount : Nat := 52326
def rowCount : Nat := invocationCount * 34
def retainedFieldCount : Nat := invocationCount * 33
def retainedCoordinateCount : Nat := retainedFieldCount * 41

@[simp] theorem invocationCount_eq : invocationCount = 52326 := by
  rfl

@[simp] theorem rowCount_eq : rowCount = 1779084 := by
  unfold rowCount
  rw [invocationCount_eq]

@[simp] theorem retainedFieldCount_eq : retainedFieldCount = 1726758 := by
  unfold retainedFieldCount
  rw [invocationCount_eq]

@[simp] theorem retainedCoordinateCount_eq :
    retainedCoordinateCount = 70797078 := by
  unfold retainedCoordinateCount
  rw [retainedFieldCount_eq]

end NightstreamFPrime.Export.Stage1.DirectPiRLCProductFootprint
