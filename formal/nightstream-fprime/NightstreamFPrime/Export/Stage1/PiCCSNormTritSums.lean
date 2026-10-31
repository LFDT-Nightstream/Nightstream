import NightstreamFPrime.Export.Stage1.PiCCSNormCache
import NightstreamFPrime.Export.Stage1.PiCCSPolynomialRange
import NightstreamFPrime.Spec.GoldilocksExtensionRing

set_option autoImplicit false

/-! SuperNeo v1.2 Section 7.3, base round of the joint sum-check with `b = 2`.
The Metal kernel `joint_round_partials` adds the weight of each unequal pair of
signed digits to one of three sums, then adds one cubic made from the sums. -/

namespace NightstreamFPrime.Export.Stage1.PiCCSNormTritSums

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open scoped NightstreamFPrime.Spec.GoldilocksExtensionRing
open PiCCSNormCache (signedValue)

/-- `u` holds the pairs `(∓1, 0)`, `v` the pairs `(∓1, ±1)`, `w` the pairs `(0, ∓1)`. -/
structure Sums where
  u : K
  v : K
  w : K

def Sums.zero : Sums := ⟨K.zero, K.zero, K.zero⟩

def Sums.add (sums : Sums) (low high : Fin 3) (weight : K) : Sums :=
  if low = high then sums
  else if high.val = 1 then
    { sums with u := if low.val = 0 then K.add sums.u weight else K.sub sums.u weight }
  else if low.val = 1 then
    { sums with w := if high.val = 0 then K.add sums.w weight else K.sub sums.w weight }
  else
    { sums with v := if high.val = 2 then K.add sums.v weight else K.sub sums.v weight }

def Sums.finish (sums : Sums) : FixedPolynomial K 3 :=
  let twoU := K.add sums.u sums.u
  let fourV := K.add (K.add sums.v sums.v) (K.add sums.v sums.v)
  let eightV := K.add fourV fourV
  ⟨[K.zero,
    K.add (K.add twoU fourV) sums.w,
    K.sub K.zero (K.add (K.add twoU sums.u) (K.add eightV fourV)),
    K.sub (K.add sums.u eightV) sums.w], rfl⟩

def pairTerm (low high : Fin 3) (weight : K) : FixedPolynomial K 3 :=
  FixedPolynomial.scale extensionOps.toOps weight
    (PiCCSFirstRoundPair.normPair extensionOps
      (K.embed (signedValue low)) (K.embed (signedValue high)))

private theorem normPair_unit (low high : Fin 3) :
    PiCCSFirstRoundPair.normPair extensionOps
        (K.embed (signedValue low)) (K.embed (signedValue high)) =
      (Sums.zero.add low high K.one).finish := by
  apply PiCCSPolynomialRange.coefficients_ext
  fin_cases low <;> fin_cases high <;> decide

theorem finish_zero : Sums.zero.finish = FixedPolynomial.zero extensionOps.toOps 3 := by
  apply PiCCSPolynomialRange.coefficients_ext
  decide

theorem finish_add (sums : Sums) (low high : Fin 3) (weight : K) :
    (sums.add low high weight).finish =
      FixedPolynomial.add extensionOps.toOps sums.finish (pairTerm low high weight) := by
  rw [pairTerm, normPair_unit]
  apply PiCCSPolynomialRange.coefficients_ext
  rw [PiCCSPolynomialRange.add_coefficients, PiCCSPolynomialRange.scale_coefficients]
  obtain ⟨u, v, w⟩ := sums
  fin_cases low <;> fin_cases high <;>
    simp (config := { decide := true }) only [Sums.add, Sums.finish, Sums.zero, extensionOps,
      ↓reduceIte, List.map_cons, List.map_nil, List.zipWith_cons_cons, List.zipWith_nil_left,
      List.cons.injEq, and_true] <;>
    simp only [← GoldilocksExtensionRing.add_eq, ← GoldilocksExtensionRing.sub_eq,
      ← GoldilocksExtensionRing.mul_eq, ← GoldilocksExtensionRing.zero_eq,
      ← GoldilocksExtensionRing.one_eq] <;>
    refine ⟨?_, ?_, ?_, ?_⟩ <;> ring

theorem finish_foldl (items : List (Fin 3 × Fin 3 × K)) (sums : Sums) :
    (items.foldl (fun state item => state.add item.1 item.2.1 item.2.2) sums).finish =
      FixedPolynomial.add extensionOps.toOps sums.finish
        (FixedPolynomial.sum extensionOps.toOps items fun item =>
          pairTerm item.1 item.2.1 item.2.2) := by
  induction items generalizing sums with
  | nil =>
      exact (PiCCSPolynomialRange.add_zero extensionOps extensionLaws _).symm
  | cons item items inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis, finish_add, FixedPolynomial.sum,
        PiCCSPolynomialRange.add_assoc extensionOps extensionLaws]

theorem finish_sums (items : List (Fin 3 × Fin 3 × K)) :
    (items.foldl (fun state item => state.add item.1 item.2.1 item.2.2) Sums.zero).finish =
      FixedPolynomial.sum extensionOps.toOps items fun item =>
        pairTerm item.1 item.2.1 item.2.2 := by
  rw [finish_foldl, finish_zero]
  apply PiCCSPolynomialRange.coefficient_ext
  intro index
  rw [PiCCSPolynomialRange.coefficient_add, PiCCSPolynomialRange.coefficient_zero]
  exact extensionLaws.zero_add _

/-- The kernel takes a value into the sums only when its digit test passes. -/
theorem pairTerm_of_codes {low high : K} {lowCode highCode : Fin 3}
    (lowFound : PiCCSNormCache.signedCode? low = some lowCode)
    (highFound : PiCCSNormCache.signedCode? high = some highCode) (weight : K) :
    FixedPolynomial.scale extensionOps.toOps weight
        (PiCCSFirstRoundPair.normPair extensionOps low high) =
      pairTerm lowCode highCode weight := by
  rw [pairTerm, PiCCSNormCache.signedCode?_some low lowCode lowFound,
    PiCCSNormCache.signedCode?_some high highCode highFound]

end NightstreamFPrime.Export.Stage1.PiCCSNormTritSums
