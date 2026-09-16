import NightstreamFPrime.Export.Stage1.PiCCSFirstRoundPair
import NightstreamFPrime.Export.Stage1.PiCCSPolynomialRange
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanReproduction
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-! Scalar buckets for one source with arbitrary cached K endpoints.
All endpoint pairs contribute, including equal endpoints. Source gamma weights
remain outside this single-source sum. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSPrefixNormBuckets

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiniteSumAlgebra

/-- Prepare the existing cubic for every pair of supplied endpoint values. -/
def prepare {count : Nat} (values : Vector K count) :
    Vector (Vector (FixedPolynomial K 3) count) count :=
  Vector.ofFn fun low => Vector.ofFn fun high =>
    PiCCSFirstRoundPair.normPair extensionOps (values.get low) (values.get high)

/-- Initial scalar weights; this does not construct any polynomials. -/
def empty (count : Nat) : Vector (Vector K count) count :=
  Vector.replicate count (Vector.replicate count K.zero)

private def lookup {count : Nat} (buckets : Vector (Vector K count) count)
    (low high : Fin count) : K := (buckets.get low).get high

/-- Add the selected weight even when both endpoint codes are equal. -/
def add {count : Nat} (buckets : Vector (Vector K count) count)
    (low high : Fin count) (weight : K) : Vector (Vector K count) count :=
  let row := buckets.get low
  buckets.set low.val (row.set high.val (K.add (row.get high) weight) high.isLt) low.isLt

/-- Finish in canonical low-code, high-code order, with no source gamma scale. -/
def finish {count : Nat} (values : Vector K count)
    (buckets : Vector (Vector K count) count) : FixedPolynomial K 3 :=
  let table := prepare values
  FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices count) fun low =>
    FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices count) fun high =>
      FixedPolynomial.scale extensionOps.toOps (lookup buckets low high)
        ((table.get low).get high)

/-- The prepared table preserves the existing norm-pair coefficients. -/
theorem prepare_value {count : Nat} (values : Vector K count) (low high : Fin count) :
    ((prepare values).get low).get high =
      PiCCSFirstRoundPair.normPair extensionOps (values.get low) (values.get high) := by
  have get_ofFn {Alpha : Type} (value : Fin count → Alpha) (index : Fin count) :
      (Vector.ofFn value).get index = value index := by
    change (Vector.ofFn value)[index.val] = value index
    rw [Vector.getElem_ofFn]
  simp only [prepare, get_ofFn]

private theorem get_set {Alpha : Type} {count : Nat} (values : Vector Alpha count)
    (selected index : Fin count) (value : Alpha) :
    (values.set selected.val value selected.isLt).get index =
      if index = selected then value else values.get index := by
  by_cases equal : index = selected
  · subst index
    change (values.set selected.val value selected.isLt)[selected.val] = _
    simp [Vector.getElem_set_self]
  · have distinct : selected.val ≠ index.val := by
      intro same
      exact equal (Fin.ext same.symm)
    change (values.set selected.val value selected.isLt)[index.val] = _
    rw [if_neg equal]
    exact Vector.getElem_set_ne selected.isLt index.isLt distinct

private theorem lookup_empty {count : Nat} (low high : Fin count) :
    lookup (empty count) low high = extensionOps.zero := by
  have get_replicate {Alpha : Type} (value : Alpha) (index : Fin count) :
      (Vector.replicate count value).get index = value := by
    change (Vector.replicate count value)[index.val] = value
    rw [Vector.getElem_replicate]
  simp only [lookup, empty, get_replicate]
  rfl

private theorem lookup_add {count : Nat} (buckets : Vector (Vector K count) count)
    (low high a b : Fin count) (weight : K) :
    lookup (add buckets low high weight) a b =
      extensionOps.add (lookup buckets a b)
        (if a = low then if b = high then weight else extensionOps.zero else extensionOps.zero) := by
  have addZero := extensionLaws.add_zero
  by_cases lowEqual : a = low <;> by_cases highEqual : b = high <;>
    simp [add, lookup, get_set, lowEqual, highEqual, addZero]
  rfl

private theorem zero_left (value : K) :
    extensionOps.mul extensionOps.zero value = extensionOps.zero := by
  rw [extensionLaws.mul_comm]
  exact extensionLaws.mul_zero value

private theorem sumMap_guard {Index : Type} (indices : List Index)
    (condition : Prop) [Decidable condition] (value : Index → K) :
    sumMap extensionOps indices (fun index => if condition then value index else extensionOps.zero) =
      if condition then sumMap extensionOps indices value else extensionOps.zero := by
  by_cases holds : condition
  · simp only [if_pos holds]
  · simp only [if_neg holds, sumMap_zero extensionOps extensionLaws]

private theorem sumFin_single {count : Nat} (selected : Fin count) (value : Fin count → K) :
    sumMap extensionOps (canonicalFinIndices count)
      (fun index => if index = selected then value index else extensionOps.zero) = value selected := by
  exact BooleanReproduction.sumMap_ite_eq_of_mem_nodup extensionOps extensionLaws
    (canonicalFinIndices count) selected value
    (by simp [canonicalFinIndices]) (canonicalFinIndices_nodup count)

/-- Empty buckets have exactly the four zero coefficients, for any endpoint values. -/
theorem finish_empty {count : Nat} (values : Vector K count) :
    finish values (empty count) = FixedPolynomial.zero extensionOps.toOps 3 := by
  apply PiCCSPolynomialRange.coefficient_ext
  intro index
  simp only [finish, PiCCSPolynomialRange.coefficient_sum,
    PiCCSPolynomialRange.coefficient_scale, PiCCSPolynomialRange.coefficient_zero,
    lookup_empty, zero_left, sumMap_zero extensionOps extensionLaws]

/-- Each insertion adds its exact original cubic, including diagonal insertions.
This is equality of coefficient objects, without endpoint or challenge premises. -/
theorem finish_add {count : Nat} (values : Vector K count)
    (buckets : Vector (Vector K count) count) (low high : Fin count) (weight : K) :
    finish values (add buckets low high weight) =
      FixedPolynomial.add extensionOps.toOps (finish values buckets)
        (FixedPolynomial.scale extensionOps.toOps weight
          (PiCCSFirstRoundPair.normPair extensionOps (values.get low) (values.get high))) := by
  rw [← prepare_value values low high]
  apply PiCCSPolynomialRange.coefficient_ext
  intro index
  have delta (a b : Fin count) (value : K) :
      extensionOps.mul
          (if a = low then if b = high then weight else extensionOps.zero else extensionOps.zero) value =
        if a = low then if b = high then extensionOps.mul weight value
          else extensionOps.zero else extensionOps.zero := by
    by_cases lowEqual : a = low <;> by_cases highEqual : b = high <;>
      simp only [lowEqual, highEqual, if_true, if_false, zero_left]
  simp only [finish, PiCCSPolynomialRange.coefficient_sum,
    PiCCSPolynomialRange.coefficient_add, PiCCSPolynomialRange.coefficient_scale,
    lookup_add, extensionLaws.right_distrib, sumMap_add extensionOps extensionLaws,
    delta, sumMap_guard, sumFin_single]

/-- Accumulate one source's scalar weights over the complete ascending range,
then prepare and scale its cached norm cubics once. -/
def range {codeCount : Nat} (values : Vector K codeCount)
    (low high : Nat → Fin codeCount) (weight : Nat → K) (start count : Nat) :
    FixedPolynomial K 3 :=
  finish values (Nat.fold count (fun offset _ accumulated =>
    add accumulated (low (start + offset)) (high (start + offset))
      (weight (start + offset))) (empty codeCount))

/-- The implemented bucket loop preserves every coefficient of the original
per-pair range, including equal endpoints and zero-length ranges. -/
theorem range_eq_reference {codeCount : Nat} (values : Vector K codeCount)
    (low high : Nat → Fin codeCount) (weight : Nat → K) (start count : Nat) :
    range values low high weight start count =
      PiCCSPolynomialRange.range extensionOps start count (fun index =>
        FixedPolynomial.scale extensionOps.toOps (weight index)
          (PiCCSFirstRoundPair.normPair extensionOps
            (values.get (low index)) (values.get (high index)))) := by
  induction count with
  | zero =>
      change finish values (empty codeCount) = FixedPolynomial.zero extensionOps.toOps 3
      exact finish_empty values
  | succ count ih =>
      rw [range, Nat.fold_succ, finish_add]
      change FixedPolynomial.add extensionOps.toOps
        (range values low high weight start count)
        (FixedPolynomial.scale extensionOps.toOps (weight (start + count))
          (PiCCSFirstRoundPair.normPair extensionOps
            (values.get (low (start + count))) (values.get (high (start + count))))) = _
      rw [ih]
      simp only [PiCCSPolynomialRange.range, Nat.fold_succ]

end NightstreamFPrime.Export.Stage1.PiCCSPrefixNormBuckets
