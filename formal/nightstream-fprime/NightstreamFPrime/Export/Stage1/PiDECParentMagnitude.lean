import NightstreamFPrime.Export.Stage1.PiDECParentIntRead
import NightstreamFPrime.Export.Stage1.PiDECEvaluationBatch
import Init.Data.Array.Lemmas
import Init.Data.Vector.Lemmas
import Init.Data.Nat.Bitwise.Lemmas

/-!
Compute the maximum magnitude from every coefficient of the actual stored
integer parent. Both loops traverse existing vectors. No coefficient list,
external maximum, fixed cutoff, or active-child count is supplied.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECParentMagnitude

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)

/-- Scan every parent block and all 54 lanes, including carried tails. -/
def maximumMagnitude {count : Nat}
    (parents : Vector (Vector Int ringDegree) count) : Nat :=
  parents.foldl (fun maximum block =>
    max maximum (block.foldl (fun maximum value => max maximum value.natAbs) 0)) 0

private theorem coefficient_le_foldMaximum {Alpha : Type} {count : Nat}
    (values : Vector Alpha count) (magnitude : Alpha → Nat)
    (initial : Nat) (index : Fin count) :
    magnitude (values.get index) ≤
      values.foldl (fun maximum value => max maximum (magnitude value)) initial := by
  rcases values with ⟨values, rfl⟩
  change magnitude values[index.val] ≤
    values.foldl (fun maximum value => max maximum (magnitude value)) initial
  have accumulated := Array.foldl_induction
    (as := values) (init := initial)
    (f := fun maximum value => max maximum (magnitude value))
    (fun visited maximum =>
      ∀ requested : Fin values.size, requested.val < visited →
        magnitude values[requested.val] ≤ maximum)
    (by
      intro requested beforeZero
      omega)
    (by
      intro current maximum previous requested beforeNext
      by_cases earlier : requested.val < current.val
      · exact (previous requested earlier).trans
          (Nat.le_max_left maximum (magnitude values[current.val]))
      · have same : requested = current := Fin.ext (by omega)
        subst requested
        exact Nat.le_max_right maximum (magnitude values[current.val]))
  exact accumulated index index.isLt

/-- Every input magnitude is bounded by this execution's computed maximum. -/
theorem coefficient_le_maximumMagnitude {count : Nat}
    (parents : Vector (Vector Int ringDegree) count)
    (block : Fin count) (lane : Fin ringDegree) :
    ((parents.get block).get lane).natAbs ≤ maximumMagnitude parents := by
  have inner := coefficient_le_foldMaximum (parents.get block) Int.natAbs 0 lane
  have outer := coefficient_le_foldMaximum parents
    (fun values : Vector Int ringDegree =>
      values.foldl (fun maximum value => max maximum value.natAbs) 0) 0 block
  exact inner.trans outer

/-- A child above the computed magnitude has zero digits at every actual
parent coordinate. The threshold is the child's existing binary position. -/
theorem cachedDigit_eq_zero {count : Nat}
    (parents : Vector (Vector Int ringDegree) count)
    (child : Radix.ChildIndex)
    (below : maximumMagnitude parents < 2 ^ child.val)
    (block : Fin count) (lane : Fin ringDegree) :
    PiDECParentIntRead.cachedDigit ((parents.get block).get lane) child = 0 := by
  have small : ((parents.get block).get lane).natAbs < 2 ^ child.val :=
    (coefficient_le_maximumMagnitude parents block lane).trans_lt below
  have bitZero := Nat.testBit_lt_two_pow small
  simp only [PiDECParentIntRead.cachedDigit, bitZero, Bool.false_eq_true, if_false]

/-- The runtime computes the maximum from the complete parent once. The
calculation is delayed until this child's bit can occur. All load guards
run before this function; zero rings still contain every output coefficient. -/
def ifActive (maximum : Nat) (child : Radix.ChildIndex)
    (calculate : Unit → Vector MaterializedRingK matrixCount) :
    Vector MaterializedRingK matrixCount :=
  if maximum < 2 ^ child.val then PiDECEvaluationBatch.zero matrixCount
  else calculate ()

end NightstreamFPrime.Export.Stage1.PiDECParentMagnitude
