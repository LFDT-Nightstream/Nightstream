import NightstreamFPrime.Export.Stage1.PiCCSSignedFirstFold
import NightstreamFPrime.Export.Stage1.PiCCSPrefixNormBuckets
import NightstreamFPrime.Export.Stage1.PiCCSNormBuckets

/-! Norm endpoints after the first sum-check challenge. Each endpoint is
the existing interpolation of two original signed coefficients. Four original
coefficients form the next pair; the cubic is applied after interpolation. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSPrefixNorm

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-- The nine ordered pairs in the existing signed-code order. -/
def pairCode (low high : Fin 3) : Fin 9 :=
  ⟨low.val * 3 + high.val, by omega⟩

/-- Flatten the proved first-fold cache without new arithmetic. -/
def values (challenge : K) : Vector K 9 :=
  let table := PiCCSSignedFirstFold.prepare challenge
  Vector.ofFn fun code =>
    (table.get ⟨code.val / 3, by omega⟩).get
      ⟨code.val % 3, Nat.mod_lt _ (by decide)⟩

/-- Each cached endpoint is the specification's first prefix fold. -/
theorem values_pairCode (challenge : K) (low high : Fin 3) :
    (values challenge).get (pairCode low high) =
      PrefixFold.interpolate extensionOps challenge
        (K.embed (PiCCSNormCache.signedValue low))
        (K.embed (PiCCSNormCache.signedValue high)) := by
  change (Vector.ofFn _)[(pairCode low high).val] = _
  rw [Vector.getElem_ofFn]
  have first : (⟨(pairCode low high).val / 3, by
      have := (pairCode low high).isLt; omega⟩ : Fin 3) = low := by
    apply Fin.ext
    simp only [pairCode]
    omega
  have second : (⟨(pairCode low high).val % 3,
      Nat.mod_lt _ (by decide)⟩ : Fin 3) = high := by
    apply Fin.ext
    simp only [pairCode]
    omega
  rw [first, second, PiCCSSignedFirstFold.prepare_value]

/-- Compute a second-round norm range directly from original signed codes.
The range is in four-scalar groups; the weights are the remaining alpha factors. -/
def range (challenge : K) (codes : Nat → Fin 3) (weight : Nat → K)
    (start count : Nat) : FixedPolynomial K 3 :=
  PiCCSPrefixNormBuckets.range (values challenge)
    (fun index => pairCode (codes (4 * index)) (codes (4 * index + 1)))
    (fun index => pairCode (codes (4 * index + 2)) (codes (4 * index + 3)))
    weight start count

/-- The cached range is exactly the original norm polynomial applied after
the original prefix interpolation. No signed-endpoint assumption is made
about the interpolated values, and diagonal cubics are not removed. -/
theorem range_eq_reference (challenge : K) (codes : Nat → Fin 3) (weight : Nat → K)
    (start count : Nat) :
    range challenge codes weight start count =
      PiCCSPolynomialRange.range extensionOps start count (fun index =>
        FixedPolynomial.scale extensionOps.toOps (weight index)
          (PiCCSFirstRoundPair.normPair extensionOps
            (PrefixFold.interpolate extensionOps challenge
              (K.embed (PiCCSNormCache.signedValue (codes (4 * index))))
              (K.embed (PiCCSNormCache.signedValue (codes (4 * index + 1)))))
            (PrefixFold.interpolate extensionOps challenge
              (K.embed (PiCCSNormCache.signedValue (codes (4 * index + 2))))
              (K.embed (PiCCSNormCache.signedValue (codes (4 * index + 3))))))) := by
  rw [range, PiCCSPrefixNormBuckets.range_eq_reference]
  simp only [values_pairCode]

private theorem interpolate_zero (challenge : K) :
    PrefixFold.interpolate extensionOps challenge K.zero K.zero = K.zero := by
  change PrefixFold.interpolate extensionOps challenge extensionOps.zero extensionOps.zero =
    extensionOps.zero
  unfold PrefixFold.interpolate InterpolationOps.sub
  rw [extensionLaws.add_neg, extensionLaws.mul_zero, extensionLaws.add_zero]

private theorem normPair_zero :
    PiCCSFirstRoundPair.normPair extensionOps K.zero K.zero =
      FixedPolynomial.zero extensionOps.toOps 3 := by
  have diagonal := PiCCSNormBuckets.pairTable_diagonal ⟨1, by decide⟩
  rw [PiCCSNormCache.pairTable_value] at diagonal
  exact diagonal

private theorem referenceTerm_zero (challenge : K) (codes : Nat → Fin 3)
    (weight : Nat → K) (bound index : Nat)
    (zeroAfter : ∀ i, bound ≤ i → codes i = ⟨1, by decide⟩)
    (outside : bound ≤ 4 * index) :
    FixedPolynomial.scale extensionOps.toOps (weight index)
      (PiCCSFirstRoundPair.normPair extensionOps
        (PrefixFold.interpolate extensionOps challenge
          (K.embed (PiCCSNormCache.signedValue (codes (4 * index))))
          (K.embed (PiCCSNormCache.signedValue (codes (4 * index + 1)))))
        (PrefixFold.interpolate extensionOps challenge
          (K.embed (PiCCSNormCache.signedValue (codes (4 * index + 2))))
          (K.embed (PiCCSNormCache.signedValue (codes (4 * index + 3)))))) =
      FixedPolynomial.zero extensionOps.toOps 3 := by
  rw [zeroAfter (4 * index) outside,
    zeroAfter (4 * index + 1) (by omega),
    zeroAfter (4 * index + 2) (by omega),
    zeroAfter (4 * index + 3) (by omega)]
  change FixedPolynomial.scale extensionOps.toOps (weight index)
    (PiCCSFirstRoundPair.normPair extensionOps
      (PrefixFold.interpolate extensionOps challenge K.zero K.zero)
      (PrefixFold.interpolate extensionOps challenge K.zero K.zero)) = _
  simp only [interpolate_zero, normPair_zero,
    PiCCSPolynomialRange.scale_zero_polynomial extensionOps extensionLaws]

/-- Every four-scalar group after the source bound contributes exactly zero.
The range length, challenge, and scalar weights remain arbitrary. -/
theorem range_zero (challenge : K) (codes : Nat → Fin 3) (weight : Nat → K)
    (bound start count : Nat)
    (zeroAfter : ∀ i, bound ≤ i → codes i = ⟨1, by decide⟩)
    (outside : bound ≤ 4 * start) :
    range challenge codes weight start count = FixedPolynomial.zero extensionOps.toOps 3 := by
  rw [range_eq_reference, ← Nat.zero_add count]
  apply PiCCSPolynomialRange.range_append_zero extensionOps extensionLaws start 0 count
  intro index lower _
  exact referenceTerm_zero challenge codes weight bound index zeroAfter (by omega)

/-- A complete four-scalar prefix equals any longer full range: all appended
pairs are the proved zero suffix of the same original code reader. -/
theorem prefix_eq_full (challenge : K) (codes : Nat → Fin 3) (weight : Nat → K)
    (bound count fullCount : Nat)
    (zeroAfter : ∀ i, bound ≤ i → codes i = ⟨1, by decide⟩)
    (covered : bound ≤ 4 * count) (included : count ≤ fullCount) :
    range challenge codes weight 0 count = range challenge codes weight 0 fullCount := by
  rw [range_eq_reference, range_eq_reference]
  have length : fullCount = count + (fullCount - count) := by omega
  rw [length]
  symm
  apply PiCCSPolynomialRange.range_append_zero extensionOps extensionLaws 0 count
    (fullCount - count)
  intro index lower _
  exact referenceTerm_zero challenge codes weight bound index zeroAfter (by omega)

end NightstreamFPrime.Export.Stage1.PiCCSPrefixNorm
