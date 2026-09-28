import NightstreamFPrime.Export.Stage1.PiCCSNormCache
import NightstreamFPrime.Export.Stage1.PiCCSPolynomialRange
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanReproduction

/-! Accumulate scalar weights for the existing signed-unit norm cubics.
Each bucket retains one original source and its two signed endpoint codes. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSNormBuckets

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
open NightstreamFPrime.Lifecycle (productionShape)

abbrev Buckets := Vector (Vector (Vector K 3) 3) productionShape.sourceCount

/-- Initial scalar weights; no polynomial is constructed here. -/
def empty : Buckets :=
  Vector.replicate productionShape.sourceCount
    (Vector.replicate 3 (Vector.replicate 3 K.zero))

private def lookup (buckets : Buckets) (source : Fin productionShape.sourceCount)
    (low high : Fin 3) : K :=
  ((buckets.get source).get low).get high

/-- Add only the selected K weight. Equal signed endpoints have zero cubic. -/
def add (buckets : Buckets) (source : Fin productionShape.sourceCount)
    (low high : Fin 3) (weight : K) : Buckets :=
  if low = high then buckets
  else
    let sourceBucket := buckets.get source
    let lowBucket := sourceBucket.get low
    buckets.set source.val
      (sourceBucket.set low.val
        (lowBucket.set high.val (K.add (lowBucket.get high) weight) high.isLt)
        low.isLt)
      source.isLt

/-- Scale only after scalar accumulation. Traversal is canonical source,
low-code, high-code order, using the existing prepared norm cubics. -/
def finish (powers : Nat → K) (buckets : Buckets) : FixedPolynomial K 3 :=
  let table := PiCCSNormCache.prepare powers
  FixedPolynomial.sum extensionOps.toOps
    (canonicalFinIndices productionShape.sourceCount) fun source =>
      FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices 3) fun low =>
        FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices 3) fun high =>
          FixedPolynomial.scale extensionOps.toOps (lookup buckets source low high)
            (PiCCSNormCache.weightedLookup table source low high)

/-- Every equal signed endpoint gives exactly four zero coefficients. -/
theorem pairTable_diagonal (code : Fin 3) :
    ((PiCCSNormCache.pairTable ()).get code).get code =
      FixedPolynomial.zero extensionOps.toOps 3 := by
  apply PiCCSPolynomialRange.coefficients_ext
  rw [PiCCSNormCache.pairTable_value]
  fin_cases code <;> decide

private theorem weighted_diagonal (powers : Nat → K)
    (source : Fin productionShape.sourceCount) (code : Fin 3) :
    PiCCSNormCache.weightedLookup (PiCCSNormCache.prepare powers) source code code =
      FixedPolynomial.zero extensionOps.toOps 3 := by
  rw [PiCCSNormCache.weightedLookup_prepare,
    ← PiCCSNormCache.pairTable_value, pairTable_diagonal]
  exact PiCCSPolynomialRange.scale_zero_polynomial extensionOps extensionLaws 3 _

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

private theorem lookup_empty (source : Fin productionShape.sourceCount) (low high : Fin 3) :
    lookup empty source low high = extensionOps.zero := by
  have get_replicate {Alpha : Type} {count : Nat} (value : Alpha) (index : Fin count) :
      (Vector.replicate count value).get index = value := by
    change (Vector.replicate count value)[index.val] = value
    rw [Vector.getElem_replicate]
  simp only [lookup, empty, get_replicate]
  rfl

private theorem lookup_add_of_ne (buckets : Buckets)
    (source query : Fin productionShape.sourceCount) (low high a b : Fin 3)
    (weight : K) (different : low ≠ high) :
    lookup (add buckets source low high weight) query a b =
      extensionOps.add (lookup buckets query a b)
        (if query = source then if a = low then if b = high then weight else extensionOps.zero
          else extensionOps.zero else extensionOps.zero) := by
  have addZero : ∀ value : K, extensionOps.add value extensionOps.zero = value := extensionLaws.add_zero
  by_cases sourceEqual : query = source <;>
    by_cases lowEqual : a = low <;>
      by_cases highEqual : b = high <;>
        simp [add, different, lookup, get_set, sourceEqual, lowEqual, highEqual, addZero] <;> rfl

private def sumKeys
    (value : Fin productionShape.sourceCount → Fin 3 → Fin 3 → K) : K :=
  sumMap extensionOps (canonicalFinIndices productionShape.sourceCount) fun source =>
    sumMap extensionOps (canonicalFinIndices 3) fun low =>
      sumMap extensionOps (canonicalFinIndices 3) fun high => value source low high

private theorem sumKeys_add
    (left right : Fin productionShape.sourceCount → Fin 3 → Fin 3 → K) :
    sumKeys (fun source low high => extensionOps.add (left source low high) (right source low high)) =
      extensionOps.add (sumKeys left) (sumKeys right) := by
  simp only [sumKeys, sumMap_add extensionOps extensionLaws]

private theorem sumKeys_zero : sumKeys (fun _ _ _ => extensionOps.zero) = extensionOps.zero := by
  simp only [sumKeys, sumMap_zero extensionOps extensionLaws]

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

private theorem sumKeys_single (source : Fin productionShape.sourceCount) (low high : Fin 3)
    (value : Fin productionShape.sourceCount → Fin 3 → Fin 3 → K) :
    sumKeys (fun query a b =>
        if query = source then if a = low then if b = high then value query a b else extensionOps.zero
          else extensionOps.zero else extensionOps.zero) = value source low high := by
  simp only [sumKeys, sumMap_guard, sumFin_single]

private theorem sumKeys_lookup_add (buckets : Buckets)
    (source : Fin productionShape.sourceCount) (low high : Fin 3) (weight : K)
    (different : low ≠ high)
    (value : Fin productionShape.sourceCount → Fin 3 → Fin 3 → K) :
    sumKeys (fun query a b => extensionOps.mul (lookup (add buckets source low high weight) query a b)
        (value query a b)) =
      extensionOps.add (sumKeys (fun query a b => extensionOps.mul (lookup buckets query a b) (value query a b)))
        (extensionOps.mul weight (value source low high)) := by
  calc
    _ = sumKeys (fun query a b => extensionOps.add
        (extensionOps.mul (lookup buckets query a b) (value query a b))
        (extensionOps.mul (if query = source then if a = low then if b = high then weight else extensionOps.zero
          else extensionOps.zero else extensionOps.zero) (value query a b))) := by
      apply congrArg sumKeys
      funext query a b
      rw [lookup_add_of_ne buckets source query low high a b weight different]
      exact extensionLaws.right_distrib _ _ _
    _ = extensionOps.add (sumKeys (fun query a b => extensionOps.mul (lookup buckets query a b) (value query a b)))
        (sumKeys (fun query a b =>
          extensionOps.mul (if query = source then if a = low then if b = high then weight else extensionOps.zero
            else extensionOps.zero else extensionOps.zero) (value query a b))) := sumKeys_add _ _
    _ = _ := by
      apply congrArg (extensionOps.add (sumKeys (fun query a b =>
        extensionOps.mul (lookup buckets query a b) (value query a b))))
      calc
        _ = sumKeys (fun query a b =>
            if query = source then if a = low then if b = high then extensionOps.mul weight (value query a b)
              else extensionOps.zero else extensionOps.zero else extensionOps.zero) := by
          apply congrArg sumKeys
          funext query a b
          by_cases sourceEqual : query = source <;>
            by_cases lowEqual : a = low <;>
              by_cases highEqual : b = high <;>
                simp [sourceEqual, lowEqual, highEqual, zero_left]
        _ = _ := sumKeys_single source low high (fun query a b => extensionOps.mul weight (value query a b))

private theorem finish_coefficient (powers : Nat → K) (buckets : Buckets) (index : Fin 4) :
    PiCCSPolynomialRange.coefficient (finish powers buckets) index =
      sumKeys (fun source low high =>
        extensionOps.mul (lookup buckets source low high)
          (PiCCSPolynomialRange.coefficient
            (PiCCSNormCache.weightedLookup (PiCCSNormCache.prepare powers) source low high) index)) := by
  unfold finish sumKeys
  simp only [PiCCSPolynomialRange.coefficient_sum, PiCCSPolynomialRange.coefficient_scale]

/-- No weights gives the existing fixed-width zero polynomial. -/
theorem finish_empty (powers : Nat → K) :
    finish powers empty = FixedPolynomial.zero extensionOps.toOps 3 := by
  apply PiCCSPolynomialRange.coefficient_ext
  intro index
  rw [finish_coefficient, PiCCSPolynomialRange.coefficient_zero]
  calc
    _ = sumKeys (fun _ _ _ => extensionOps.zero) := by
      apply congrArg sumKeys
      funext source low high
      rw [lookup_empty]
      exact zero_left _
    _ = _ := sumKeys_zero

/-- One scalar insertion contributes exactly its original weighted norm cubic.
Diagonal insertions may be skipped because their coefficient objects are zero. -/
theorem finish_add (powers : Nat → K) (buckets : Buckets)
    (source : Fin productionShape.sourceCount) (low high : Fin 3) (weight : K) :
    finish powers (add buckets source low high weight) =
      FixedPolynomial.add extensionOps.toOps (finish powers buckets)
        (FixedPolynomial.scale extensionOps.toOps weight
          (PiCCSNormCache.weightedLookup (PiCCSNormCache.prepare powers) source low high)) := by
  by_cases diagonal : low = high
  · subst high
    simp only [add]
    rw [weighted_diagonal, PiCCSPolynomialRange.scale_zero_polynomial extensionOps extensionLaws]
    apply PiCCSPolynomialRange.coefficient_ext
    intro index
    rw [PiCCSPolynomialRange.coefficient_add, PiCCSPolynomialRange.coefficient_zero]
    exact (extensionLaws.add_zero _).symm
  · apply PiCCSPolynomialRange.coefficient_ext
    intro index
    rw [finish_coefficient, PiCCSPolynomialRange.coefficient_add,
      finish_coefficient, PiCCSPolynomialRange.coefficient_scale]
    exact sumKeys_lookup_add buckets source low high weight diagonal
      (fun query a b => PiCCSPolynomialRange.coefficient
        (PiCCSNormCache.weightedLookup (PiCCSNormCache.prepare powers) query a b) index)


end NightstreamFPrime.Export.Stage1.PiCCSNormBuckets
