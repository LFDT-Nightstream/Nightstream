import SuperNeo.MLE
import Mathlib.Data.Nat.Bitwise
import Mathlib.Data.Nat.Bits

namespace SuperNeo

open F

private theorem goldilocks_q_ne_one : Goldilocks.q ≠ 1 := by
  decide

private theorem f_zero_mul (a : F) : (0 : F) * a = 0 := by
  apply Fin.ext
  change (0 * a.val) % Goldilocks.q = 0
  simp

private theorem f_one_mul (a : F) : (1 : F) * a = a := by
  apply Fin.ext
  change (1 * a.val) % Goldilocks.q = a.val
  simpa using Nat.mod_eq_of_lt a.isLt

private theorem f_add_zero (a : F) : a + 0 = a := by
  apply Fin.ext
  change (a.val + 0) % Goldilocks.q = a.val
  simpa using Nat.mod_eq_of_lt a.isLt

/-- Sum_x eq(x,z) * q(x) over the Boolean hypercube table q. -/
def eqLiftFromTable (qVals z : Array F) : F :=
  let ell := z.size
  let n := 2 ^ ell
  if qVals.size != n then
    0
  else
    (List.range n).foldl
      (fun acc mask => acc + eqPoly (bitsToFArray ell mask) z * qVals[mask]!)
      0

private theorem bitsToFArray_size (ell mask : Nat) :
    (bitsToFArray ell mask).size = ell := by
  simp [bitsToFArray]

private theorem bitsToFArray_eq_bitsToFieldArray (ell mask : Nat) :
    bitsToFArray ell mask = bitsToFieldArray ell mask := by
  apply Array.ext
  · simp [bitsToFArray, bitsToFieldArray]
  · intro i hiBits hiField
    have hi : i < ell := by
      simpa [bitsToFArray_size] using hiBits
    by_cases hBit : Nat.testBit mask i
    · have hMod : (mask / 2 ^ i) % 2 = 1 := by
        simpa [Nat.testBit, Nat.shiftRight_eq_div_pow, Nat.mod_two_of_bodd] using hBit
      calc
        (bitsToFArray ell mask)[i] = F.ofNat ((mask / 2 ^ i) % 2) := by
          simp [bitsToFArray]
        _ = 1 := by simp [hMod]
        _ = (bitsToFieldArray ell mask)[i] := by
          simp [bitsToFieldArray, hBit]
    · have hMod : (mask / 2 ^ i) % 2 = 0 := by
        simpa [Nat.testBit, Nat.shiftRight_eq_div_pow, Nat.mod_two_of_bodd] using hBit
      calc
        (bitsToFArray ell mask)[i] = F.ofNat ((mask / 2 ^ i) % 2) := by
          simp [bitsToFArray]
        _ = 0 := by simp [hMod]
        _ = (bitsToFieldArray ell mask)[i] := by
          simp [bitsToFieldArray, hBit]

private theorem bitsToFieldArray_isBitVec (ell mask : Nat) :
    IsBitVec (bitsToFieldArray ell mask) := by
  intro i
  have hGet :
      (bitsToFieldArray ell mask)[i] =
        (if Nat.testBit mask i.1 then (1 : F) else 0) := by
    simpa using
      (Array.getElem_ofFn
        (f := fun j : Fin ell =>
          if Nat.testBit mask j.1 then (1 : F) else 0)
        (i := i.1)
        (h := by simpa [bitsToFieldArray] using i.2))
  by_cases hBit : Nat.testBit mask i.1
  · right
    simpa [hBit] using hGet
  · left
    simpa [hBit] using hGet

private theorem bitsToFArray_isBitVec (ell mask : Nat) :
    IsBitVec (bitsToFArray ell mask) := by
  simpa [bitsToFArray_eq_bitsToFieldArray] using bitsToFieldArray_isBitVec ell mask

private theorem bitsToFieldArray_injective_of_lt_pow
    {ell a b : Nat}
    (ha : a < 2 ^ ell)
    (hb : b < 2 ^ ell)
    (hEq : bitsToFieldArray ell a = bitsToFieldArray ell b) :
    a = b := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < ell
  · have hEntry :
        (bitsToFieldArray ell a)[i]! = (bitsToFieldArray ell b)[i]! := by
      simpa [bitsToFieldArray, hi] using congrArg (fun arr => arr[i]!) hEq
    by_cases hAi : Nat.testBit a i <;> by_cases hBi : Nat.testBit b i <;>
      simp [bitsToFieldArray, hi, hAi, hBi] at hEntry ⊢
    all_goals simp [goldilocks_q_ne_one] at hEntry
  · have hLe : ell ≤ i := Nat.le_of_not_lt hi
    have hPow : 2 ^ ell ≤ 2 ^ i := Nat.pow_le_pow_right (by decide : 1 ≤ 2) hLe
    have hAi : Nat.testBit a i = false :=
      Nat.testBit_eq_false_of_lt (lt_of_lt_of_le ha hPow)
    have hBi : Nat.testBit b i = false :=
      Nat.testBit_eq_false_of_lt (lt_of_lt_of_le hb hPow)
    simp [hAi, hBi]

private theorem bitsToFArray_injective_of_lt_pow
    {ell a b : Nat}
    (ha : a < 2 ^ ell)
    (hb : b < 2 ^ ell)
    (hEq : bitsToFArray ell a = bitsToFArray ell b) :
    a = b := by
  apply bitsToFieldArray_injective_of_lt_pow ha hb
  simpa [bitsToFArray_eq_bitsToFieldArray] using hEq

private theorem foldl_congr
    (l : List Nat)
    (init : F)
    (step1 step2 : F → Nat → F)
    (hEq : ∀ acc i, i ∈ l → step1 acc i = step2 acc i) :
    l.foldl step1 init = l.foldl step2 init := by
  induction l generalizing init with
  | nil =>
      rfl
  | cons a tl ih =>
      have hHead : step1 init a = step2 init a := hEq init a (by simp)
      calc
        (a :: tl).foldl step1 init = tl.foldl step1 (step1 init a) := by rfl
        _ = tl.foldl step1 (step2 init a) := by rw [hHead]
        _ = tl.foldl step2 (step2 init a) := by
              apply ih
              intro acc i hi
              exact hEq acc i (by simp [hi])
        _ = (a :: tl).foldl step2 init := by rfl

private theorem foldl_range_add_indicator
    (n mask : Nat)
    (value : F)
    (hMask : mask < n) :
    (List.range n).foldl
        (fun acc i => acc + (if i = mask then value else 0))
        0 = value := by
  induction n with
  | zero =>
      cases Nat.not_lt_zero _ hMask
  | succ n ih =>
      by_cases hEq : mask = n
      · subst hEq
        have hPrefix :
            (List.range mask).foldl
                (fun acc i => acc + (if i = mask then value else 0))
                0 = 0 := by
          have hCongr :
              (List.range mask).foldl
                  (fun acc i => acc + (if i = mask then value else 0))
                  0
                =
              (List.range mask).foldl
                  (fun acc _ => acc)
                  0 := by
            apply foldl_congr
              (l := List.range mask)
              (init := 0)
              (step1 := fun acc i => acc + (if i = mask then value else 0))
              (step2 := fun acc _ => acc)
            intro acc i hi
            have hiLt : i < mask := List.mem_range.mp hi
            simp [Nat.ne_of_lt hiLt]
          calc
            (List.range mask).foldl
                (fun acc i => acc + (if i = mask then value else 0))
                0
              =
            (List.range mask).foldl
                (fun acc _ => acc)
                0 := hCongr
            _ = 0 := by
                  induction List.range mask with
                  | nil => rfl
                  | cons a tl ih =>
                      simp [ih]
        calc
          (List.range (mask + 1)).foldl
              (fun acc i => acc + (if i = mask then value else 0))
              0
            = (List.range mask).foldl
                (fun acc i => acc + (if i = mask then value else 0))
                0 + value := by
                  simp [List.range_succ, List.foldl_append]
          _ = 0 + value := by simp [hPrefix]
          _ = value := by simp
      · have hMaskLt : mask < n := by omega
        calc
          (List.range (n + 1)).foldl
              (fun acc i => acc + (if i = mask then value else 0))
              0
            = (List.range n).foldl
                (fun acc i => acc + (if i = mask then value else 0))
                0 + (if n = mask then value else 0) := by
                  simp [List.range_succ, List.foldl_append]
          _ = (List.range n).foldl
                (fun acc i => acc + (if i = mask then value else 0))
                0 + 0 := by
                  have hNe : n ≠ mask := by
                    intro h
                    exact hEq h.symm
                  simp [hNe]
          _ = value + 0 := by simpa using ih hMaskLt
          _ = value := by simp

theorem eqLiftFromTable_bitsToFArray
    {qVals : Array F}
    {ell mask : Nat}
    (hSize : qVals.size = 2 ^ ell)
    (hMask : mask < 2 ^ ell) :
    eqLiftFromTable qVals (bitsToFArray ell mask) = qVals[mask]! := by
  unfold eqLiftFromTable
  simp [bitsToFArray_size, hSize]
  have hMaskBits : IsBitVec (bitsToFArray ell mask) :=
    bitsToFArray_isBitVec ell mask
  have hFold :
      (List.range (2 ^ ell)).foldl
          (fun acc i => acc + eqPoly (bitsToFArray ell i) (bitsToFArray ell mask) * qVals[i]!)
          0
        =
      (List.range (2 ^ ell)).foldl
          (fun acc i => acc + (if i = mask then qVals[mask]! else 0))
          0 := by
    apply foldl_congr
    intro acc i hiMem
    have hi : i < 2 ^ ell := List.mem_range.mp hiMem
    have hIBits : IsBitVec (bitsToFArray ell i) :=
      bitsToFArray_isBitVec ell i
    have hDelta :
        eqPoly (bitsToFArray ell i) (bitsToFArray ell mask) = (if i = mask then 1 else 0) := by
      have hEqPoly :=
        eqPoly_eq_delta_of_isBitVec (by simp [bitsToFArray_size]) hIBits hMaskBits
      have hArrEq : bitsToFArray ell i = bitsToFArray ell mask ↔ i = mask := by
        constructor
        · intro hArr
          exact bitsToFArray_injective_of_lt_pow hi hMask hArr
        · intro hIM
          subst hIM
          rfl
      simpa [hArrEq] using hEqPoly
    by_cases hIM : i = mask
    · subst hIM
      calc
        acc + eqPoly (bitsToFArray ell i) (bitsToFArray ell i) * qVals[i]!
          = acc + (1 : F) * qVals[i]! := by simp [hDelta]
        _ = acc + qVals[i]! := by
              exact congrArg (fun t => acc + t) (f_one_mul (qVals[i]!))
        _ = acc + (if i = i then qVals[i]! else 0) := by simp
    · calc
        acc + eqPoly (bitsToFArray ell i) (bitsToFArray ell mask) * qVals[i]!
          = acc + (0 : F) * qVals[i]! := by simp [hDelta, hIM]
        _ = acc + 0 := by
              exact congrArg (fun t => acc + t) (f_zero_mul (qVals[i]!))
        _ = acc := by exact f_add_zero acc
        _ = acc + (if i = mask then qVals[mask]! else 0) := by simp [hIM]
  calc
    (List.range (2 ^ ell)).foldl
        (fun acc i => acc + eqPoly (bitsToFArray ell i) (bitsToFArray ell mask) * qVals[i]!)
        0
      =
    (List.range (2 ^ ell)).foldl
        (fun acc i => acc + (if i = mask then qVals[mask]! else 0))
        0 := hFold
    _ = qVals[mask]! := foldl_range_add_indicator (2 ^ ell) mask (qVals[mask]!) hMask

def eqLiftBooleanIndicator (qVals : Array F) (ell mask : Nat) : Bool :=
  if qVals.size != 2 ^ ell then
    false
  else if mask >= 2 ^ ell then
    false
  else
    let z := bitsToFArray ell mask
    decide (eqLiftFromTable qVals z = qVals[mask]!)

def eqLiftAllBoolean (qVals : Array F) (ell : Nat) : Bool :=
  if qVals.size != 2 ^ ell then
    false
  else
    (List.range (2 ^ ell)).all (fun mask => eqLiftBooleanIndicator qVals ell mask)

/-- Proposition-level single-point eq-lift surface. -/
def eqLiftBooleanIndicatorProp (qVals : Array F) (ell mask : Nat) : Prop :=
  qVals.size = 2 ^ ell ∧
  mask < 2 ^ ell ∧
  eqLiftFromTable qVals (bitsToFArray ell mask) = qVals[mask]!

/-- Proposition-level all-points eq-lift surface. -/
def eqLiftAllBooleanProp (qVals : Array F) (ell : Nat) : Prop :=
  qVals.size = 2 ^ ell ∧
  ∀ mask, mask < 2 ^ ell →
    eqLiftFromTable qVals (bitsToFArray ell mask) = qVals[mask]!

/-- Table-level Boolean-cube vanishing predicate (the `ZS_ell` truth-table view). -/
def zeroOnBooleanCubeProp (qVals : Array F) (ell : Nat) : Prop :=
  qVals.size = 2 ^ ell ∧
  ∀ mask, mask < 2 ^ ell → qVals[mask]! = 0

/-- Eq-lifted Boolean-cube vanishing predicate. -/
def eqLiftZeroOnBooleanCubeProp (qVals : Array F) (ell : Nat) : Prop :=
  qVals.size = 2 ^ ell ∧
  ∀ mask, mask < 2 ^ ell →
    eqLiftFromTable qVals (bitsToFArray ell mask) = 0

theorem eqLiftBooleanIndicator_sound
  {qVals : Array F} {ell mask : Nat}
  (hOk : eqLiftBooleanIndicator qVals ell mask = true) :
  eqLiftBooleanIndicatorProp qVals ell mask := by
  unfold eqLiftBooleanIndicator at hOk
  by_cases hSize : qVals.size = 2 ^ ell
  · by_cases hMask : mask < 2 ^ ell
    · have hMaskGe : ¬ mask >= 2 ^ ell := Nat.not_le_of_lt hMask
      have hDec :
          decide (eqLiftFromTable qVals (bitsToFArray ell mask) = qVals[mask]!) = true := by
        simpa [hSize, hMaskGe] using hOk
      exact ⟨hSize, hMask, decide_eq_true_eq.mp hDec⟩
    · have hMaskGe : mask >= 2 ^ ell := Nat.ge_of_not_lt hMask
      simp [hSize, hMaskGe] at hOk
  · simp [hSize] at hOk

theorem eqLiftBooleanIndicator_complete
  {qVals : Array F} {ell mask : Nat}
  (hProp : eqLiftBooleanIndicatorProp qVals ell mask) :
  eqLiftBooleanIndicator qVals ell mask = true := by
  rcases hProp with ⟨hSize, hMask, hEq⟩
  unfold eqLiftBooleanIndicator
  have hMaskGe : ¬ mask >= 2 ^ ell := Nat.not_le_of_lt hMask
  simp [hSize, hMaskGe, decide_eq_true hEq]

theorem eqLiftBooleanIndicator_eq_true_iff
  {qVals : Array F} {ell mask : Nat} :
  eqLiftBooleanIndicator qVals ell mask = true ↔
    eqLiftBooleanIndicatorProp qVals ell mask := by
  constructor
  · exact eqLiftBooleanIndicator_sound
  · exact eqLiftBooleanIndicator_complete

theorem eqLiftAllBoolean_holds
  {qVals : Array F} {ell : Nat}
  (hSize : qVals.size = 2 ^ ell) :
  eqLiftAllBooleanProp qVals ell := by
  refine ⟨hSize, ?_⟩
  intro mask hMask
  exact eqLiftFromTable_bitsToFArray hSize hMask

theorem eqLiftAllBoolean_sound
  {qVals : Array F} {ell : Nat}
  (hOk : eqLiftAllBoolean qVals ell = true) :
  eqLiftAllBooleanProp qVals ell := by
  unfold eqLiftAllBoolean at hOk
  by_cases hSize : qVals.size = 2 ^ ell
  · have hAll :
        (List.range (2 ^ ell)).all (fun mask => eqLiftBooleanIndicator qVals ell mask) = true := by
      simpa [hSize] using hOk
    refine ⟨hSize, ?_⟩
    intro mask hMask
    have hMem : mask ∈ List.range (2 ^ ell) := List.mem_range.mpr hMask
    have hIndicator : eqLiftBooleanIndicator qVals ell mask = true := by
      exact (List.all_eq_true.mp hAll) mask hMem
    exact (eqLiftBooleanIndicator_sound hIndicator).2.2
  · simp [hSize] at hOk

theorem eqLiftAllBoolean_complete
  {qVals : Array F} {ell : Nat}
  (hProp : eqLiftAllBooleanProp qVals ell) :
  eqLiftAllBoolean qVals ell = true := by
  rcases hProp with ⟨hSize, hAll⟩
  unfold eqLiftAllBoolean
  rw [if_neg (by simp [hSize])]
  apply List.all_eq_true.mpr
  intro mask hMem
  exact eqLiftBooleanIndicator_complete
    ⟨hSize, List.mem_range.mp hMem, hAll mask (List.mem_range.mp hMem)⟩

theorem eqLiftAllBoolean_eq_true_iff
  {qVals : Array F} {ell : Nat} :
  eqLiftAllBoolean qVals ell = true ↔
    eqLiftAllBooleanProp qVals ell := by
  constructor
  · exact eqLiftAllBoolean_sound
  · exact eqLiftAllBoolean_complete

theorem eqLiftZeroOnBooleanCube_iff_zeroOnBooleanCube
  {qVals : Array F} {ell : Nat} :
  eqLiftZeroOnBooleanCubeProp qVals ell ↔
    zeroOnBooleanCubeProp qVals ell := by
  constructor <;> intro h
  · rcases h with ⟨hSize, hZero⟩
    refine ⟨hSize, ?_⟩
    intro mask hMask
    have hEval : eqLiftFromTable qVals (bitsToFArray ell mask) = 0 := hZero mask hMask
    simpa [eqLiftFromTable_bitsToFArray hSize hMask] using hEval
  · rcases h with ⟨hSize, hZero⟩
    refine ⟨hSize, ?_⟩
    intro mask hMask
    have hEval : qVals[mask]! = 0 := hZero mask hMask
    simpa [eqLiftFromTable_bitsToFArray hSize hMask] using hEval

/-- Schwartz-Zippel failure bound interface: d / |S|. -/
def schwartzZippelBoundLeOne (totalDegree setSize : Nat) : Bool :=
  if setSize = 0 then
    false
  else
    decide (totalDegree <= setSize)

/-- Proposition-level SZ precondition surface. -/
def schwartzZippelBoundLeOneProp (totalDegree setSize : Nat) : Prop :=
  setSize ≠ 0 ∧ totalDegree <= setSize

theorem schwartzZippelBoundLeOne_sound
  {totalDegree setSize : Nat}
  (hOk : schwartzZippelBoundLeOne totalDegree setSize = true) :
  setSize ≠ 0 ∧ totalDegree <= setSize := by
  unfold schwartzZippelBoundLeOne at hOk
  by_cases hzero : setSize = 0
  · simp [hzero] at hOk
  · have hDec : decide (totalDegree <= setSize) = true := by
      simpa [hzero] using hOk
    exact ⟨hzero, decide_eq_true_eq.mp hDec⟩

theorem schwartzZippelBoundLeOne_complete
  {totalDegree setSize : Nat}
  (hNonzero : setSize ≠ 0)
  (hBound : totalDegree <= setSize) :
  schwartzZippelBoundLeOne totalDegree setSize = true := by
  unfold schwartzZippelBoundLeOne
  simp [hNonzero, decide_eq_true hBound]

theorem schwartzZippelBoundLeOne_eq_true_iff_prop
  {totalDegree setSize : Nat} :
  schwartzZippelBoundLeOne totalDegree setSize = true ↔
    schwartzZippelBoundLeOneProp totalDegree setSize := by
  constructor
  · exact schwartzZippelBoundLeOne_sound
  · intro hProp
    exact schwartzZippelBoundLeOne_complete hProp.1 hProp.2

def polyLemmaSanity : Bool :=
  let qVals : Array F := #[F.ofNat 3, F.ofNat 1, F.ofNat 4, F.ofNat 1, F.ofNat 5, F.ofNat 9, F.ofNat 2, F.ofNat 6]
  eqLiftAllBoolean qVals 3 && schwartzZippelBoundLeOne 5 17

end SuperNeo
