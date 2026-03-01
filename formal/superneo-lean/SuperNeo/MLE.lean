import SuperNeo.EqPoly

/-!
Multilinear-extension scaffold.

This file provides a small executable MLE evaluator plus theorem-facing
assumption surfaces used by protocol-level composition.
-/

namespace SuperNeo

open F

/-- Bit-vector embedding for an index mask. -/
def bitsToFieldArray (width mask : Nat) : Array F :=
  Array.ofFn (fun i : Fin width =>
    F.ofNat ((mask / (2 ^ i.1)) % 2))

/-- Standard multilinear extension evaluation from a truth table `f`. -/
def mleEval (f r : Array F) : F :=
  if _h : f.size = (2 ^ r.size) then
    (List.range f.size).foldl
      (fun acc i => acc + f[i]! * eqPoly (bitsToFieldArray r.size i) r)
      0
  else
    0

/-- Inner-product form used as the theorem-facing target identity. -/
def mleInnerProductForm (f r : Array F) : F :=
  (List.range f.size).foldl
    (fun acc i => acc + f[i]! * eqPoly (bitsToFieldArray r.size i) r)
    0

/-- Theorem-facing boundary: MLE equals inner-product form on valid table sizes. -/
def mleIdentityAssumption : Prop :=
  ∀ f r : Array F,
    f.size = (2 ^ r.size) →
    mleEval f r = mleInnerProductForm f r

theorem mleEval_eq_innerProductForm_of_size
  {f r : Array F}
  (hSize : f.size = (2 ^ r.size)) :
  mleEval f r = mleInnerProductForm f r := by
  unfold mleEval
  simp [hSize, mleInnerProductForm]


end SuperNeo
