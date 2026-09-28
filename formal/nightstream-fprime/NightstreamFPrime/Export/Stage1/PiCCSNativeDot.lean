import NightstreamFPrime.Export.Stage1.PiCCSWeightedBasis
import NightstreamFPrime.Export.Stage1.PiCCSNumericSumOrder
import NightstreamFPrime.Export.NativePoseidon2RoundCore

/-! Native-word accumulation for the existing PiCCS weighted dot product.
The input and output types remain the specified quadratic extension. Word
arithmetic uses the proved Goldilocks operations; no expected output is an input. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSNativeDot

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Export.NativePoseidon2

private abbrev CanonicalWord := { word : UInt64 // word.toNat < goldilocksModulus }

@[inline] private def fromF (value : F) : CanonicalWord :=
  ⟨UInt64.ofNatLT value.val (Nat.lt_trans value.isLt (by decide)), by
    simpa only [UInt64.toNat_ofNatLT] using value.isLt⟩

@[inline] private def toF (value : CanonicalWord) : F :=
  ⟨value.val.toNat, value.property⟩

private theorem toF_fromF (value : F) : toF (fromF value) = value := by
  apply Fin.ext
  simp only [toF, fromF, UInt64.toNat_ofNatLT]

private theorem toF_denote (value : CanonicalWord) : toF value = value.val.denote := by
  apply Fin.ext
  change value.val.toNat = value.val.toNat % goldilocksModulus
  exact (Nat.mod_eq_of_lt value.property).symm

@[inline] private def addWord (left right : CanonicalWord) : CanonicalWord :=
  ⟨add64 left.val right.val, add64_canonical _ _ left.property right.property⟩

@[inline] private def mulWord (left right : CanonicalWord) : CanonicalWord :=
  ⟨mul64 left.val right.val, mul64_canonical _ _⟩

private theorem addWord_value (left right : CanonicalWord) :
    toF (addWord left right) = toF left + toF right := by
  rw [toF_denote, toF_denote left, toF_denote right]
  simpa only [addWord] using add64_denote left.val right.val left.property right.property

private theorem mulWord_value (left right : CanonicalWord) :
    toF (mulWord left right) = toF left * toF right := by
  rw [toF_denote, toF_denote left, toF_denote right]
  simpa only [mulWord] using mul64_denote left.val right.val left.property right.property

private structure Accumulator where
  real : CanonicalWord
  imaginary : CanonicalWord

@[inline] private def value (acc : Accumulator) : K :=
  ⟨toF acc.real, toF acc.imaginary⟩

private def zero : Accumulator := ⟨fromF 0, fromF 0⟩

private theorem zero_value : value zero = K.zero := by
  simp only [value, zero, toF_fromF, K.zero]

@[inline] private def addProduct (acc : Accumulator) (left right : K) : Accumulator :=
  let a := fromF left.c0
  let b := fromF left.c1
  let c := fromF right.c0
  let d := fromF right.c1
  ⟨addWord acc.real (addWord (mulWord a c) (mulWord (fromF 7) (mulWord b d))),
   addWord acc.imaginary (addWord (mulWord a d) (mulWord b c))⟩

private theorem addProduct_value (acc : Accumulator) (left right : K) :
    value (addProduct acc left right) = K.add (value acc) (K.mul left right) := by
  simp only [addProduct, value, addWord_value, mulWord_value, toF_fromF,
    K.add, K.mul, Fin.mul_assoc]

private def accumulate {lanes : Nat} (count : Nat)
    (prepared : Vector K lanes) (source : Fin lanes → K) : Accumulator :=
  Nat.fold count (fun index _ acc =>
    if bounded : index < lanes then
      addProduct acc (prepared.get ⟨index, bounded⟩) (source ⟨index, bounded⟩)
    else acc) zero

private theorem accumulate_value {lanes : Nat} (count : Nat)
    (prepared : Vector K lanes) (source : Fin lanes → K) :
    value (accumulate count prepared source) =
      NumericCompletionSum.numericSum extensionOps count (fun index =>
        if bounded : index < lanes then
          extensionOps.mul (prepared.get ⟨index, bounded⟩) (source ⟨index, bounded⟩)
        else extensionOps.zero) := by
  induction count with
  | zero => exact zero_value
  | succ count ih =>
      simp only [accumulate, Nat.fold_succ] at ⊢
      change value (if bounded : count < lanes then
          addProduct (accumulate count prepared source)
            (prepared.get ⟨count, bounded⟩) (source ⟨count, bounded⟩)
        else accumulate count prepared source) = _
      by_cases bounded : count < lanes
      · simp only [dif_pos bounded, addProduct_value,
          NumericCompletionSum.numericSum, Nat.fold_succ]
        change extensionOps.add (value (accumulate count prepared source)) _ = _
        rw [ih]
        rfl
      · simp only [dif_neg bounded, NumericCompletionSum.numericSum,
          Nat.fold_succ, extensionLaws.add_zero]
        exact ih

/-- Compute every lane of the specified dot product with a native accumulator. -/
def dotK {lanes : Nat} (prepared : Vector K lanes) (source : Fin lanes → K) : K :=
  value (accumulate lanes prepared source)

/-- The native loop equals the existing dot product for arbitrary inputs and
lengths, including an empty vector. There is no source-validity premise. -/
theorem dotK_eq_spec {lanes : Nat} (prepared : Vector K lanes)
    (source : Fin lanes → K) : dotK prepared source = PiCCSWeightedBasis.dotK prepared source := by
  rw [dotK, accumulate_value,
    PiCCSNumericSumOrder.numericSum_eq_finSum extensionOps extensionLaws]
  unfold PiCCSWeightedBasis.dotK
  apply FiniteSumAlgebra.sumMap_congr
  intro lane _
  exact dif_pos lane.isLt

/-- Compile the proved dot-product implementation at existing call sites. -/
@[csimp] theorem dotK_eq_native : @PiCCSWeightedBasis.dotK = @dotK := by
  funext lanes prepared source
  exact (dotK_eq_spec prepared source).symm

end NightstreamFPrime.Export.Stage1.PiCCSNativeDot
