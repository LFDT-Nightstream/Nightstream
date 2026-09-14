import NightstreamFPrime.Export.Stage1.PiDECParentSparseRead
import Mathlib.Data.ZMod.ValMinAbs
import Init.Data.Nat.Bitwise.Lemmas

/-!
Read signed binary digits from the existing centered integer view of one
parent block. The cache uses Vector Int, preserves all coordinates, and
changes neither the strict parent bound nor the prepared sparse forms.
The loader and parent input codec are outside this module.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECParentIntRead

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic (StoredAssignment)
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace

/-- Nat.testBit reads the existing little-endian binary position directly.
Only the centered integer sign selects the positive or negative unit. -/
def cachedDigit (value : Int) (child : Radix.ChildIndex) : F :=
  if value.natAbs.testBit child.val then
    if 0 ≤ value then 1 else -1
  else 0

private theorem cachedDigit_as_field (value : Int) (child : Radix.ChildIndex) :
    cachedDigit value child =
      if 0 ≤ value then
        Radix.fieldOfNat ((value.natAbs.testBit child.val).toNat)
      else -(Radix.fieldOfNat ((value.natAbs.testBit child.val).toNat)) := by
  cases bit : value.natAbs.testBit child.val <;> simp [cachedDigit, bit]

private theorem centered_natAbs (value : F) :
    (ZMod.valMinAbs (n := goldilocksModulus) value).natAbs =
      centeredMagnitude value := by
  simpa only [centeredMagnitude] using
    ZMod.valMinAbs_natAbs_eq_min (n := goldilocksModulus) value

private theorem centered_nonneg (value : F) :
    (0 ≤ ZMod.valMinAbs (n := goldilocksModulus) value) ↔
      Radix.isNonnegative value := by
  change (0 ≤ ZMod.valMinAbs (n := goldilocksModulus) value) ↔
    value.val ≤ goldilocksModulus / 2
  exact ZMod.valMinAbs_nonneg_iff (n := goldilocksModulus) value

/-- Under the unchanged strict B bound, the cached bit and sign give the
exact existing scalar split. No digit witness or cached-value premise is used. -/
theorem cachedDigit_eq_splitScalar (value : F) (child : Radix.ChildIndex)
    (bounded : centeredMagnitude value < Radix.combinedBound) :
    cachedDigit (ZMod.valMinAbs (n := goldilocksModulus) value) child =
      Radix.splitScalar value child := by
  simp only [cachedDigit_as_field, Nat.toNat_testBit, centered_natAbs,
    centered_nonneg, Radix.splitScalar, if_pos bounded, Radix.boundedDigit,
    Radix.magnitudeDigit, Radix.natBit]

/-- The existing sparse entries request only their cached scalar digits.
No field conversion, bound check, power or child array is built per read. -/
def sparseRead
    (forms : FixedArray (FixedArray (SparseForm ringDegree) ringDegree) ringDegree)
    (parent : Vector Int ringDegree) (basis : Fin ringDegree)
    (child : Radix.ChildIndex) (output : Fin ringDegree) : F :=
  ((forms.get basis).get output).evalSparse
    (fun input => cachedDigit (parent.get input) child)

/-- Mapping one parent block to its existing centered integer view preserves
every read of the same sparse forms. The only premise is the original
strict bound, which is checked before the parent cache is exposed. -/
theorem sparseRead_map_valMinAbs
    (forms : FixedArray (FixedArray (SparseForm ringDegree) ringDegree) ringDegree)
    (parent : StoredAssignment ringDegree)
    (bounded : ∀ input, centeredMagnitude (parent.get input) < Radix.combinedBound)
    (basis : Fin ringDegree) (child : Radix.ChildIndex) (output : Fin ringDegree) :
    sparseRead forms
        (parent.map (fun value => ZMod.valMinAbs (n := goldilocksModulus) value))
        basis child output =
      PiDECParentSparseRead.read forms parent basis child output := by
  unfold sparseRead PiDECParentSparseRead.read
  apply congrArg (fun source : Fin ringDegree → F =>
    ((forms.get basis).get output).evalSparse source)
  funext input
  change cachedDigit
      ((parent.map (fun value => ZMod.valMinAbs (n := goldilocksModulus) value))[input.val])
      child = Radix.splitScalar (parent.get input) child
  rw [Vector.getElem_map]
  exact cachedDigit_eq_splitScalar (parent.get input) child (bounded input)

end NightstreamFPrime.Export.Stage1.PiDECParentIntRead
