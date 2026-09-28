import NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix
import Init.Data.Vector.Lemmas
import Init.Data.Vector.OfFn

/-!
Stored signed-binary PiDEC splitting. The existing assignment vector stores
all sixteen children, including zero children and the complete carrier tail.
The returned coordinates are proved equal to the existing scalar split.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.StoredSplit

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic
  (StoredAssignment view)

private def bounded {width : Nat} (parent : StoredAssignment width) : Bool :=
  parent.all (fun value => decide (centeredMagnitude value < Radix.combinedBound))

private theorem bounded_iff {width : Nat} (parent : StoredAssignment width) :
    bounded parent = true ↔
      ∀ column, centeredMagnitude (parent.get column) < Radix.combinedBound := by
  rw [bounded, Vector.all_eq_true]
  constructor
  · intro checked column
    simpa [Vector.get] using checked column.val column.isLt
  · intro checked column live
    simpa [Vector.get] using checked ⟨column, live⟩

/-- Reject an out-of-bound parent before constructing any short witness.
All 16 children are stored, including zero children. -/
def splitChecked {width : Nat} (parent : StoredAssignment width) :
    Option (Vector (StoredAssignment width) productionGlobalParams.k) :=
  if bounded parent then
    some (Vector.ofFn fun child =>
      parent.map (fun value => Radix.boundedDigit value child))
  else none

/-- Successful computation establishes the bound and identifies the exact
stored result. Neither a supplied digit witness nor an equality test selects
the computed digits. -/
theorem splitChecked_eq_some_iff {width : Nat}
    (parent : StoredAssignment width)
    (children : Vector (StoredAssignment width) productionGlobalParams.k) :
    splitChecked parent = some children ↔
      (∀ column, centeredMagnitude (parent.get column) < Radix.combinedBound) ∧
      children = Vector.ofFn (fun child =>
        parent.map (fun value => Radix.boundedDigit value child)) := by
  by_cases checked : bounded parent = true
  · have parentBound := (bounded_iff parent).mp checked
    simp [splitChecked, checked, parentBound, eq_comm]
  · have parentUnbounded := mt (bounded_iff parent).mpr checked
    simp [splitChecked, checked, parentUnbounded]

/-- Every returned private coefficient is the existing semantic scalar
split. The same statement applies to a ring block or the complete carrier. -/
theorem splitChecked_value {width : Nat}
    (parent : StoredAssignment width)
    (children : Vector (StoredAssignment width) productionGlobalParams.k)
    (success : splitChecked parent = some children)
    (child : Radix.ChildIndex) (column : Fin width) :
    (children.get child).get column = Radix.splitScalar (parent.get column) child := by
  rcases (splitChecked_eq_some_iff parent children).mp success with ⟨parentBound, rfl⟩
  change (Vector.ofFn (fun digit =>
    parent.map (fun value => Radix.boundedDigit value digit)))[child.val][column.val] = _
  rw [Vector.getElem_ofFn, Vector.getElem_map]
  rw [Radix.splitScalar, if_pos (parentBound column)]
  rfl

/-- At the complete typed width, these are precisely the honest private
assignments consumed by PiDEC.childrenOf and PiDEC.complete. -/
theorem splitChecked_assignment {shape : Shape}
    (parent : StoredAssignment shape.carrierWidth)
    (children : Vector (StoredAssignment shape.carrierWidth) productionGlobalParams.k)
    (success : splitChecked parent = some children) :
    ∀ child, view (children.get child) =
      Radix.splitAssignment (shape := shape) (view parent) child := by
  intro child
  funext column
  exact splitChecked_value parent children success child column

/-- The kernel returns the existing scalar split for exactly the bounded
inputs. A bounded parent cannot abort. -/
theorem kernel_eq_spec {width : Nat} (parent : StoredAssignment width) :
    splitChecked parent =
      if ∀ column, centeredMagnitude (parent.get column) < Radix.combinedBound then
        some (Vector.ofFn fun child : Radix.ChildIndex =>
          Vector.ofFn fun column : Fin width => Radix.splitScalar (parent.get column) child)
      else none := by
  by_cases parentBound :
      ∀ column, centeredMagnitude (parent.get column) < Radix.combinedBound
  · rw [if_pos parentBound]
    apply (splitChecked_eq_some_iff parent _).2
    refine ⟨parentBound, ?_⟩
    apply Vector.ext
    intro child childLt
    rw [Vector.getElem_ofFn, Vector.getElem_ofFn]
    apply Vector.ext
    intro column columnLt
    rw [Vector.getElem_ofFn, Vector.getElem_map]
    rw [Radix.splitScalar, if_pos (parentBound ⟨column, columnLt⟩)]
    rfl
  · rw [if_neg parentBound]
    cases result : splitChecked parent with
    | none => rfl
    | some children =>
        exact False.elim
          (parentBound ((splitChecked_eq_some_iff parent children).mp result).1)

end NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.StoredSplit
