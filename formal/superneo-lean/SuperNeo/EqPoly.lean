import SuperNeo.Field

/-!
Equality-polynomial scaffold.

This module keeps a compact executable definition of the `eq` polynomial and a
clear theorem-facing assumption boundary for selector behavior.
-/

namespace SuperNeo

open F

/-- Bit predicate for field elements. -/
def IsBit (x : F) : Prop :=
  x = 0 ∨ x = 1

/-- Bit-vector predicate for arrays. -/
def IsBitVec (v : Array F) : Prop :=
  ∀ i : Fin v.size, IsBit v[i]

/-- Single-coordinate equality term `x*y + (1-x)*(1-y)`. -/
def eqTerm (x y : F) : F :=
  x * y + (1 - x) * (1 - y)

/-- Product equality polynomial over all coordinates (size-matched inputs only). -/
def eqPoly (x y : Array F) : F :=
  if _h : x.size = y.size then
    (List.range x.size).foldl (fun acc i => acc * eqTerm x[i]! y[i]!) 1
  else
    0

/-- Boolean bit-vector embedding `mask ↦ {0,1}^ell` as field elements. -/
def bitsToFArray (ell mask : Nat) : Array F :=
  Array.ofFn (fun i : Fin ell => F.ofNat ((mask / (2 ^ i.1)) % 2))

/-- Selector-style proposition for the equality polynomial. -/
def eqPolySelectorProp (x y : Array F) : Prop :=
  x.size = y.size →
    IsBitVec x →
    IsBitVec y →
    eqPoly x y = (if x = y then 1 else 0)

/-- Theorem-facing boundary: selector behavior on size-compatible vectors. -/
def eqPolyAssumption : Prop :=
  ∀ x y : Array F, eqPolySelectorProp x y

theorem eqPoly_eq_zero_of_size_ne
  {x y : Array F}
  (hNe : x.size ≠ y.size) :
  eqPoly x y = 0 := by
  unfold eqPoly
  simp [hNe]

theorem eqTerm_eq_delta_of_isBit
  {x y : F}
  (hx : IsBit x)
  (hy : IsBit y) :
  eqTerm x y = (if x = y then 1 else 0) := by
  rcases hx with rfl | rfl <;> rcases hy with rfl | rfl <;> decide

private theorem f_zero_mul (a : F) : 0 * a = 0 := by
  apply Fin.ext
  change (0 * a.val) % Goldilocks.q = 0
  simp

private theorem f_mul_zero (a : F) : a * 0 = 0 := by
  apply Fin.ext
  change (a.val * 0) % Goldilocks.q = 0
  simp

private theorem foldl_mul_eq_one_of_all_one
  (l : List Nat)
  (t : Nat → F)
  (hOne : ∀ i, i ∈ l → t i = 1) :
  l.foldl (fun acc i => acc * t i) 1 = 1 := by
  induction l with
  | nil =>
      rfl
  | cons a tl ih =>
      have hA : t a = 1 := hOne a (by simp)
      have hTl : ∀ i, i ∈ tl → t i = 1 := by
        intro i hi
        exact hOne i (by simp [hi])
      calc
        (a :: tl).foldl (fun acc i => acc * t i) 1
            = tl.foldl (fun acc i => acc * t i) (1 * t a) := by
                rfl
        _ = tl.foldl (fun acc i => acc * t i) 1 := by
              simp [hA]
        _ = 1 := ih hTl

private theorem foldl_mul_eq_zero_of_zero_or_exists_zero
  (l : List Nat)
  (t : Nat → F) :
  ∀ init : F,
    (init = 0 ∨ ∃ i, i ∈ l ∧ t i = 0) →
      l.foldl (fun acc i => acc * t i) init = 0 := by
  intro init hZeroOr
  induction l generalizing init with
  | nil =>
      cases hZeroOr with
      | inl hInit =>
          simpa [hInit]
      | inr hEx =>
          rcases hEx with ⟨i, hi, _⟩
          cases hi
  | cons a tl ih =>
      cases hZeroOr with
      | inl hInit =>
          have hInitMul : init * t a = 0 := by
            calc
              init * t a = 0 * t a := by simpa [hInit]
              _ = 0 := f_zero_mul (t a)
          calc
            (a :: tl).foldl (fun acc i => acc * t i) init
                = tl.foldl (fun acc i => acc * t i) (init * t a) := by
                    rfl
            _ = tl.foldl (fun acc i => acc * t i) 0 := by
                  simp [hInitMul]
            _ = 0 := ih 0 (Or.inl rfl)
      | inr hEx =>
          rcases hEx with ⟨i, hiMem, hiZero⟩
          rcases List.mem_cons.mp hiMem with hHead | hTail
          · have hiZeroA : t a = 0 := by
              simpa [hHead] using hiZero
            have hMulZero : init * t a = 0 := by
              calc
                init * t a = init * 0 := by simpa [hiZeroA]
                _ = 0 := f_mul_zero init
            calc
              (a :: tl).foldl (fun acc j => acc * t j) init
                  = tl.foldl (fun acc j => acc * t j) (init * t a) := by
                      rfl
              _ = tl.foldl (fun acc j => acc * t j) 0 := by
                    simp [hMulZero]
              _ = 0 := ih 0 (Or.inl rfl)
          · calc
              (a :: tl).foldl (fun acc j => acc * t j) init
                  = tl.foldl (fun acc j => acc * t j) (init * t a) := by
                      rfl
              _ = 0 := ih (init * t a) (Or.inr ⟨i, hTail, hiZero⟩)

private theorem exists_index_ne_of_ne_of_size_eq
  {x y : Array F}
  (hSize : x.size = y.size)
  (hNe : x ≠ y) :
  ∃ i, i < x.size ∧ x[i]! ≠ y[i]! := by
  classical
  by_cases hEx : ∃ i, i < x.size ∧ x[i]! ≠ y[i]!
  · exact hEx
  · exfalso
    apply hNe
    apply Array.ext
    · exact hSize
    · intro i hiX hiY
      have hNotNe : ¬ x[i]! ≠ y[i]! := by
        intro hNeAt
        exact hEx ⟨i, hiX, hNeAt⟩
      have hEqBang : x[i]! = y[i]! := by
        exact Classical.not_not.mp hNotNe
      simpa [hiX, hiY] using hEqBang

theorem eqPoly_eq_delta_of_isBitVec
  {x y : Array F}
  (hSize : x.size = y.size)
  (hx : IsBitVec x)
  (hy : IsBitVec y) :
  eqPoly x y = (if x = y then 1 else 0) := by
  by_cases hxy : x = y
  · subst hxy
    unfold eqPoly
    simp
    apply foldl_mul_eq_one_of_all_one
    intro i hiMem
    have hi : i < x.size := List.mem_range.mp hiMem
    let fi : Fin x.size := ⟨i, hi⟩
    have hBit : IsBit x[fi] := hx fi
    have hTerm :
        eqTerm x[i]! x[i]! =
          (if x[i]! = x[i]! then 1 else 0) :=
      eqTerm_eq_delta_of_isBit (x := x[i]!) (y := x[i]!)
        (by simpa [fi, hi] using hBit)
        (by simpa [fi, hi] using hBit)
    simpa using hTerm.trans (by simp)
  · have hNeAt : ∃ i, i < x.size ∧ x[i]! ≠ y[i]! :=
      exists_index_ne_of_ne_of_size_eq hSize hxy
    rcases hNeAt with ⟨i, hi, hxyi⟩
    unfold eqPoly
    simp [hSize]
    simp [hxy]
    apply foldl_mul_eq_zero_of_zero_or_exists_zero (l := List.range y.size)
      (t := fun j => eqTerm x[j]! y[j]!) (init := 1)
    apply Or.inr
    refine ⟨i, ?_, ?_⟩
    · have hiY : i < y.size := by simpa [hSize] using hi
      exact List.mem_range.mpr hiY
    · let fiX : Fin x.size := ⟨i, hi⟩
      have hBitX : IsBit x[fiX] := hx fiX
      have hiY : i < y.size := by simpa [hSize] using hi
      let fiY : Fin y.size := ⟨i, hiY⟩
      have hBitY : IsBit y[fiY] := hy fiY
      have hTerm :
          eqTerm x[i]! y[i]! =
            (if x[i]! = y[i]! then 1 else 0) :=
        eqTerm_eq_delta_of_isBit (x := x[i]!) (y := y[i]!)
          (by simpa [fiX, hi] using hBitX)
          (by simpa [fiY, hiY] using hBitY)
      have hIf : (if x[i]! = y[i]! then (1 : F) else 0) = 0 := by
        simp [hxyi]
      exact hTerm.trans hIf

theorem eqPolyAssumption_holds : eqPolyAssumption := by
  intro x y hSize hx hy
  exact eqPoly_eq_delta_of_isBitVec hSize hx hy


end SuperNeo
