import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.PiDEC
import Init.Data.Vector.OfFn

/-!
Stored assignment arithmetic for B.3 subtraction and B.4 recomposition.
The array builder charges its invoked coordinate program, loop branch and
push, initial capacity, and final return. Field operations have fixed-size
Goldilocks operands. Function-valued semantic assignments are only views.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic

open NightstreamFPrime.Spec
open Phi81Relation.EvaluationHomomorphism
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

abbrev StoredAssignment (width : Nat) := Vector F width

def view {width : Nat} (stored : StoredAssignment width) : Fin width → F := stored.get

private def coordinateAction {width : Nat} (coordinate : Fin width → Result F)
    (index : Fin width) : StateM Nat F := fun work =>
  let result := coordinate index
  (result.value, work + result.work + 2)

/-- A tail-recursive array traversal, with no intermediate list of indices. -/
def build {width : Nat} (coordinate : Fin width → Result F) : Result (StoredAssignment width) :=
  let result := (Vector.ofFnM (coordinateAction coordinate)).run (width + 1)
  ⟨result.1, result.2 + 1⟩

/-- Two array reads, one field subtraction, and one result return. -/
def subtract {width : Nat} (left right : StoredAssignment width) : Result (StoredAssignment width) :=
  build fun column =>
    let leftRead : Result F := ⟨left.get column, 1⟩
    let rightRead : Result F := ⟨right.get column, 1⟩
    ⟨leftRead.value - rightRead.value, leftRead.work + rightRead.work + 1 + 1⟩

end NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic
