import SuperNeo.Field

/-! Equality-polynomial and eq-lift lemmas used in P18. -/


namespace SuperNeo

open F

def oneMinus (x : F) : F := (1 : F) - x

def eqTerm (x y : F) : F :=
  x * y + oneMinus x * oneMinus y

/-- Prefix product for `eqPoly` over indices `[0,n)`. -/
private def eqPolyPrefix (x y : Array F) : Nat → F
  | 0 => 1
  | Nat.succ n => eqPolyPrefix x y n * eqTerm x[n]! y[n]!

/-- eq(x,y) = Π_i (x_i y_i + (1-x_i)(1-y_i)). -/
def eqPoly (x y : Array F) : F :=
  if x.size != y.size then
    0
  else
    eqPolyPrefix x y x.size

def bitsToFArray (width mask : Nat) : Array F :=
  Id.run do
    let mut out := Array.replicate width (0 : F)
    for i in [0:width] do
      let bit := (mask / (2 ^ i)) % 2
      out := out.set! i (F.ofNat bit)
    return out

def isBoolF (x : F) : Bool :=
  decide (x = 0 ∨ x = 1)

def isBoolFProp (x : F) : Prop :=
  x = 0 ∨ x = 1

private theorem eqTerm_eq_one_of_bool_eq
  {x y : F}
  (hx : isBoolFProp x)
  (hy : isBoolFProp y)
  (hEq : x = y) :
  eqTerm x y = 1 := by
  subst hEq
  rcases hx with rfl | rfl <;> native_decide

private theorem eqTerm_eq_zero_of_bool_ne
  {x y : F}
  (hx : isBoolFProp x)
  (hy : isBoolFProp y)
  (hNe : x ≠ y) :
  eqTerm x y = 0 := by
  rcases hx with rfl | rfl
  · rcases hy with rfl | rfl
    · exact (False.elim (hNe rfl))
    · native_decide
  · rcases hy with rfl | rfl
    · native_decide
    · exact (False.elim (hNe rfl))

/-- Boolean-selector form of `eqTerm`. -/
theorem eqTerm_eq_if_bool
  {x y : F}
  (hx : isBoolFProp x)
  (hy : isBoolFProp y) :
  eqTerm x y = (if x = y then (1 : F) else (0 : F)) := by
  by_cases hEq : x = y
  · simpa [hEq] using eqTerm_eq_one_of_bool_eq hx hy hEq
  · simpa [hEq] using eqTerm_eq_zero_of_bool_ne hx hy hEq

private theorem one_mul_one_f : ((1 : F) * (1 : F)) = 1 := by
  native_decide

private theorem zero_mul_f (a : F) : ((0 : F) * a) = 0 := by
  change F.ofNat (0 * a.val) = F.ofNat 0
  simp

private theorem mul_zero_f (a : F) : (a * (0 : F)) = 0 := by
  change F.ofNat (a.val * 0) = F.ofNat 0
  simp

private theorem eqPolyPrefix_eq_one_of_bool_eq
  {x y : Array F} :
  ∀ n : Nat,
    (∀ i, i < n → isBoolFProp x[i]!) →
    (∀ i, i < n → isBoolFProp y[i]!) →
    (∀ i, i < n → x[i]! = y[i]!) →
    eqPolyPrefix x y n = 1
  | 0 => by
      intro _ _ _
      simp [eqPolyPrefix]
  | Nat.succ n => by
      intro hx hy hEq
      have hPrev :
          eqPolyPrefix x y n = 1 := by
        apply eqPolyPrefix_eq_one_of_bool_eq n
        · intro i hi
          exact hx i (Nat.lt_trans hi (Nat.lt_succ_self n))
        · intro i hi
          exact hy i (Nat.lt_trans hi (Nat.lt_succ_self n))
        · intro i hi
          exact hEq i (Nat.lt_trans hi (Nat.lt_succ_self n))
      have hxN : isBoolFProp x[n]! := hx n (Nat.lt_succ_self n)
      have hyN : isBoolFProp y[n]! := hy n (Nat.lt_succ_self n)
      have hEqN : x[n]! = y[n]! := hEq n (Nat.lt_succ_self n)
      have hTerm : eqTerm x[n]! y[n]! = 1 :=
        eqTerm_eq_one_of_bool_eq hxN hyN hEqN
      calc
        eqPolyPrefix x y (Nat.succ n)
            = eqPolyPrefix x y n * eqTerm x[n]! y[n]! := by
                simp [eqPolyPrefix]
        _ = 1 * 1 := by simp [hPrev, hTerm]
        _ = 1 := one_mul_one_f

private theorem eqPolyPrefix_eq_zero_of_bool_ne
  {x y : Array F} :
  ∀ n : Nat,
    (∀ i, i < n → isBoolFProp x[i]!) →
    (∀ i, i < n → isBoolFProp y[i]!) →
    (∃ i, i < n ∧ x[i]! ≠ y[i]!) →
    eqPolyPrefix x y n = 0
  | 0 => by
      intro _ _ hNe
      rcases hNe with ⟨i, hi, _⟩
      exact (False.elim (Nat.not_lt_zero i hi))
  | Nat.succ n => by
      intro hx hy hNe
      rcases hNe with ⟨i, hiSucc, hDiff⟩
      have hLe : i ≤ n := Nat.le_of_lt_succ hiSucc
      have hiCases : i < n ∨ i = n := Nat.lt_or_eq_of_le hLe
      cases hiCases with
      | inl hi =>
          have hPrev :
              eqPolyPrefix x y n = 0 := by
            apply eqPolyPrefix_eq_zero_of_bool_ne n
            · intro j hj
              exact hx j (Nat.lt_trans hj (Nat.lt_succ_self n))
            · intro j hj
              exact hy j (Nat.lt_trans hj (Nat.lt_succ_self n))
            · exact ⟨i, hi, hDiff⟩
          calc
            eqPolyPrefix x y (Nat.succ n)
                = eqPolyPrefix x y n * eqTerm x[n]! y[n]! := by
                    simp [eqPolyPrefix]
            _ = 0 * eqTerm x[n]! y[n]! := by simp [hPrev]
            _ = 0 := zero_mul_f _
      | inr hiEq =>
          have hxN : isBoolFProp x[n]! := hx n (Nat.lt_succ_self n)
          have hyN : isBoolFProp y[n]! := hy n (Nat.lt_succ_self n)
          have hDiffN : x[n]! ≠ y[n]! := by
            simpa [hiEq] using hDiff
          have hTerm : eqTerm x[n]! y[n]! = 0 :=
            eqTerm_eq_zero_of_bool_ne hxN hyN hDiffN
          calc
            eqPolyPrefix x y (Nat.succ n)
                = eqPolyPrefix x y n * eqTerm x[n]! y[n]! := by
                    simp [eqPolyPrefix]
            _ = eqPolyPrefix x y n * 0 := by simp [hTerm]
            _ = 0 := mul_zero_f _

private theorem exists_index_ne_of_array_ne
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
    · intro i hix hiy
      have hEqAt : x[i]! = y[i]! := by
        by_cases hEqAt : x[i]! = y[i]!
        · exact hEqAt
        · exact False.elim (hEx ⟨i, hix, hEqAt⟩)
      simpa [hix, hiy] using hEqAt

theorem isBoolF_sound
  {x : F}
  (hOk : isBoolF x = true) :
  isBoolFProp x := by
  unfold isBoolF at hOk
  unfold isBoolFProp
  exact decide_eq_true_eq.mp hOk

theorem isBoolF_complete
  {x : F}
  (hProp : isBoolFProp x) :
  isBoolF x = true := by
  unfold isBoolF
  exact decide_eq_true hProp

theorem isBoolF_iff_prop
  {x : F} :
  isBoolF x = true ↔ isBoolFProp x := by
  constructor
  · exact isBoolF_sound
  · exact isBoolF_complete

/-- Theorem-native selector theorem: on Boolean vectors, `eqPoly` is an equality indicator. -/
theorem eqPoly_selector_on_bool
  {x y : Array F}
  (hSize : x.size = y.size)
  (hxBool : x.all isBoolF = true)
  (hyBool : y.all isBoolF = true) :
  if x = y then eqPoly x y = 1 else eqPoly x y = 0 := by
  have hx :
      ∀ i, i < x.size → isBoolFProp x[i]! := by
    intro i hi
    have hAt : isBoolF (x[i]'hi) = true := (Array.all_eq_true.mp hxBool) i hi
    have hAtProp : isBoolFProp (x[i]'hi) := isBoolF_sound hAt
    simpa [hi] using hAtProp
  have hy :
      ∀ i, i < x.size → isBoolFProp y[i]! := by
    intro i hi
    have hiy : i < y.size := by simpa [hSize] using hi
    have hAt : isBoolF (y[i]'hiy) = true := (Array.all_eq_true.mp hyBool) i hiy
    have hAtProp : isBoolFProp (y[i]'hiy) := isBoolF_sound hAt
    simpa [hiy] using hAtProp
  by_cases hEq : x = y
  · subst hEq
    have hEqIdx : ∀ i, i < x.size → x[i]! = x[i]! := by
      intro i hi
      rfl
    have hPrefix : eqPolyPrefix x x x.size = 1 :=
      eqPolyPrefix_eq_one_of_bool_eq (x := x) (y := x) x.size hx hy hEqIdx
    simpa [eqPoly] using hPrefix
  · simp [hEq]
    unfold eqPoly
    have hNotNe : (x.size != y.size) = false := by simp [hSize]
    rw [hNotNe]
    have hDiff : ∃ i, i < x.size ∧ x[i]! ≠ y[i]! :=
      exists_index_ne_of_array_ne hSize hEq
    have hPrefix : eqPolyPrefix x y x.size = 0 :=
      eqPolyPrefix_eq_zero_of_bool_ne (x := x) (y := y) x.size hx hy hDiff
    exact hPrefix

/-- Prop-native Boolean selector theorem for `eqPoly`. -/
theorem eqPoly_selector_on_bool_prop
  {x y : Array F}
  (hSize : x.size = y.size)
  (hxBool : ∀ i, i < x.size → isBoolFProp x[i]!)
  (hyBool : ∀ i, i < y.size → isBoolFProp y[i]!) :
  if x = y then eqPoly x y = 1 else eqPoly x y = 0 := by
  have hxBoolB : x.all isBoolF = true := by
    apply (Array.all_eq_true).2
    intro i hi
    have hXi : isBoolFProp (x[i]'hi) := by
      simpa [hi] using (hxBool i hi)
    exact isBoolF_complete hXi
  have hyBoolB : y.all isBoolF = true := by
    apply (Array.all_eq_true).2
    intro i hi
    have hYi : isBoolFProp (y[i]'hi) := by
      simpa [hi] using (hyBool i hi)
    exact isBoolF_complete hYi
  exact eqPoly_selector_on_bool hSize hxBoolB hyBoolB

theorem eqPoly_eq_one_of_bool_eq
  {x y : Array F}
  (hSize : x.size = y.size)
  (hxBool : x.all isBoolF = true)
  (hyBool : y.all isBoolF = true)
  (hEq : x = y) :
  eqPoly x y = 1 := by
  simpa [hEq] using (eqPoly_selector_on_bool hSize hxBool hyBool)

theorem eqPoly_eq_zero_of_bool_ne
  {x y : Array F}
  (hSize : x.size = y.size)
  (hxBool : x.all isBoolF = true)
  (hyBool : y.all isBoolF = true)
  (hNe : x ≠ y) :
  eqPoly x y = 0 := by
  simpa [hNe] using (eqPoly_selector_on_bool hSize hxBool hyBool)

/-- Compatibility alias for selector theorem naming. -/
theorem eqPoly_bool_selector
  {x y : Array F}
  (hSize : x.size = y.size)
  (hxBool : x.all isBoolF = true)
  (hyBool : y.all isBoolF = true) :
  if x = y then eqPoly x y = 1 else eqPoly x y = 0 := by
  exact eqPoly_selector_on_bool hSize hxBool hyBool

theorem eqPoly_eq_one_iff_bool_eq
  {x y : Array F}
  (hSize : x.size = y.size)
  (hxBool : x.all isBoolF = true)
  (hyBool : y.all isBoolF = true) :
  eqPoly x y = 1 ↔ x = y := by
  constructor
  · intro hPoly
    by_cases hEq : x = y
    · exact hEq
    · have hZero : eqPoly x y = 0 := eqPoly_eq_zero_of_bool_ne hSize hxBool hyBool hEq
      have h01 : (0 : F) = 1 := by
        calc
          (0 : F) = eqPoly x y := hZero.symm
          _ = 1 := hPoly
      have hZeroNeOne : (0 : F) ≠ 1 := by
        native_decide
      exact False.elim (hZeroNeOne h01)
  · intro hEq
    exact eqPoly_eq_one_of_bool_eq hSize hxBool hyBool hEq

theorem eqPoly_eq_zero_iff_bool_ne
  {x y : Array F}
  (hSize : x.size = y.size)
  (hxBool : x.all isBoolF = true)
  (hyBool : y.all isBoolF = true) :
  eqPoly x y = 0 ↔ x ≠ y := by
  constructor
  · intro hZero hEq
    have hOne : eqPoly x y = 1 := eqPoly_eq_one_of_bool_eq hSize hxBool hyBool hEq
    have h01 : (0 : F) = 1 := by
      calc
        (0 : F) = eqPoly x y := hZero.symm
        _ = 1 := hOne
    have hZeroNeOne : (0 : F) ≠ 1 := by
      native_decide
    exact False.elim (hZeroNeOne h01)
  · intro hNe
    exact eqPoly_eq_zero_of_bool_ne hSize hxBool hyBool hNe

/-- Indicator behavior on Boolean points: eq(x,y)=1 iff x=y, else 0. -/
def eqHypercubeIndicator (x y : Array F) : Bool :=
  if x.size != y.size then
    false
  else if !(x.all isBoolF && y.all isBoolF) then
    false
  else
    let e := eqPoly x y
    if decide (x = y) then
      decide (e = 1)
    else
      decide (e = 0)

def eqHypercubeIndicatorProp (x y : Array F) : Prop :=
  (x.size != y.size) = false ∧
  (x.all isBoolF && y.all isBoolF) = true ∧
  let e := eqPoly x y
  if x = y then e = 1 else e = 0

theorem eqHypercubeIndicator_sound
  {x y : Array F}
  (hOk : eqHypercubeIndicator x y = true) :
  eqHypercubeIndicatorProp x y := by
  unfold eqHypercubeIndicator at hOk
  by_cases hSize : x.size != y.size
  · simp [hSize] at hOk
  · have hSizeFalse : (x.size != y.size) = false := by simp [hSize]
    by_cases hBool : (x.all isBoolF && y.all isBoolF) = true
    · simp [hSize, hBool] at hOk
      refine ⟨hSizeFalse, hBool, ?_⟩
      by_cases hEq : x = y
      · have hEqPoly : eqPoly x y = 1 := by simpa [hEq] using hOk
        simpa [hEq] using hEqPoly
      · have hEqPoly : eqPoly x y = 0 := by simpa [hEq] using hOk
        simpa [hEq] using hEqPoly
    · simp [hSize, hBool] at hOk

theorem eqHypercubeIndicator_complete
  {x y : Array F}
  (hProp : eqHypercubeIndicatorProp x y) :
  eqHypercubeIndicator x y = true := by
  rcases hProp with ⟨hSize, hBool, hCase⟩
  unfold eqHypercubeIndicator
  by_cases hEq : x = y
  · have hEqPoly : eqPoly x y = 1 := by simpa [hEq] using hCase
    have hDecEq : decide (x = y) = true := decide_eq_true hEq
    simp [hSize, hBool, hDecEq, hEqPoly]
  · have hEqPoly : eqPoly x y = 0 := by simpa [hEq] using hCase
    have hDecEq : decide (x = y) = false := decide_eq_false hEq
    simp [hSize, hBool, hDecEq, hEqPoly]

theorem eqHypercubeIndicatorProp_size_eq
  {x y : Array F}
  (hProp : eqHypercubeIndicatorProp x y) :
  x.size = y.size := by
  have hSizeFalse : (x.size != y.size) = false := hProp.1
  by_cases hEq : x.size = y.size
  · exact hEq
  · have hNeTrue : (x.size != y.size) = true := by simp [hEq]
    rw [hNeTrue] at hSizeFalse
    cases hSizeFalse

theorem eqHypercubeIndicatorProp_bool_rows
  {x y : Array F}
  (hProp : eqHypercubeIndicatorProp x y) :
  x.all isBoolF = true ∧ y.all isBoolF = true := by
  simpa [Bool.and_eq_true] using hProp.2.1

theorem eqHypercubeIndicatorProp_eval_if
  {x y : Array F}
  (hProp : eqHypercubeIndicatorProp x y) :
  (if x = y then eqPoly x y = 1 else eqPoly x y = 0) := by
  exact hProp.2.2

theorem eqHypercubeIndicatorProp_eval_eq
  {x y : Array F}
  (hProp : eqHypercubeIndicatorProp x y)
  (hEq : x = y) :
  eqPoly x y = 1 := by
  simpa [hEq] using (eqHypercubeIndicatorProp_eval_if hProp)

theorem eqHypercubeIndicatorProp_eval_ne
  {x y : Array F}
  (hProp : eqHypercubeIndicatorProp x y)
  (hNe : x ≠ y) :
  eqPoly x y = 0 := by
  simpa [hNe] using (eqHypercubeIndicatorProp_eval_if hProp)

theorem eqHypercubeIndicator_iff_prop
  {x y : Array F} :
  eqHypercubeIndicator x y = true ↔ eqHypercubeIndicatorProp x y := by
  constructor
  · exact eqHypercubeIndicator_sound
  · exact eqHypercubeIndicator_complete

theorem eqHypercubeIndicatorProp_of_bool
  {x y : Array F}
  (hSize : x.size = y.size)
  (hxBool : x.all isBoolF = true)
  (hyBool : y.all isBoolF = true) :
  eqHypercubeIndicatorProp x y := by
  refine ⟨?_, ?_, ?_⟩
  · simp [hSize]
  · simp [hxBool, hyBool]
  · exact eqPoly_selector_on_bool hSize hxBool hyBool

theorem eqHypercubeIndicator_true_of_bool
  {x y : Array F}
  (hSize : x.size = y.size)
  (hxBool : x.all isBoolF = true)
  (hyBool : y.all isBoolF = true) :
  eqHypercubeIndicator x y = true := by
  exact eqHypercubeIndicator_complete
    (eqHypercubeIndicatorProp_of_bool hSize hxBool hyBool)

def eqPolySanity : Bool :=
  let x := #[0, 1, 0, 1]
  let y := #[0, 1, 0, 1]
  let z := #[1, 0, 1, 0]
  decide (eqPoly x y = 1 ∧ eqPoly x z = 0)

end SuperNeo
