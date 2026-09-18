import SuperNeo.Primitives.ExtensionField
import Mathlib.Data.Nat.Bitwise

/-!
Extension-field multilinear-extension scaffold.

This module mirrors the theorem-facing equality-polynomial and MLE evaluator
surfaces over `SuperNeo.KExt`. It exists because opening convergence Phase 1
works over the quadratic extension carrier, while the original `MLE.lean` and
`EqPoly.lean` are base-field-only.
-/

namespace SuperNeo

/-- Bit-vector embedding for an index mask (little-endian) into `KExt`. -/
def bitsToKExtArray (width mask : Nat) : Array KExt :=
  Array.ofFn (fun i : Fin width =>
    if Nat.testBit mask i.1 then (1 : KExt) else 0)

private theorem div_two_odd (j : Nat) : (2 * j + 1) / 2 = j := by
  calc
    (2 * j + 1) / 2 = (1 + j * 2) / 2 := by simp [Nat.mul_comm, Nat.add_comm]
    _ = 1 / 2 + j := Nat.add_mul_div_right 1 j (by decide : 0 < 2)
    _ = j := by simp

private theorem testBit_even_succ (j i : Nat) :
    Nat.testBit (2 * j) (i + 1) = Nat.testBit j i := by
  rw [Nat.testBit_add (x := 2 * j) (i := i) (n := 1)]
  simp

private theorem testBit_odd_succ (j i : Nat) :
    Nat.testBit (2 * j + 1) (i + 1) = Nat.testBit j i := by
  rw [Nat.testBit_add (x := 2 * j + 1) (i := i) (n := 1)]
  rw [div_two_odd]

private theorem bitsToKExtArray_even_head (width j : Nat) :
    (bitsToKExtArray (width + 1) (2 * j))[0]! = (0 : KExt) := by
  simp [bitsToKExtArray, Nat.testBit_zero]

private theorem bitsToKExtArray_odd_head (width j : Nat) :
    (bitsToKExtArray (width + 1) (2 * j + 1))[0]! = (1 : KExt) := by
  simp [bitsToKExtArray, Nat.testBit_zero]

private theorem foldl_mul_head
    (n : Nat)
    (t : Nat → KExt) :
    (List.range (n + 1)).foldl (fun acc i => acc * t i) 1
      = t 0 * (List.range n).foldl (fun acc i => acc * t (i + 1)) 1 := by
  induction n with
  | zero =>
      simp
  | succ n ih =>
      calc
        (List.range (Nat.succ n + 1)).foldl (fun acc i => acc * t i) 1
            = ((List.range (n + 1)).foldl (fun acc i => acc * t i) 1) * t (n + 1) := by
                simp [List.range_succ, List.foldl_append]
        _ = (t 0 * (List.range n).foldl (fun acc i => acc * t (i + 1)) 1) * t (n + 1) := by
              simp [ih]
        _ = t 0 * ((List.range n).foldl (fun acc i => acc * t (i + 1)) 1 * t (n + 1)) := by
              ring
        _ = t 0 * (List.range (n + 1)).foldl (fun acc i => acc * t (i + 1)) 1 := by
              simp [List.range_succ, List.foldl_append]

private theorem foldl_congr
    (l : List Nat)
    (init : KExt)
    (step1 step2 : KExt → Nat → KExt)
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

private theorem bitsToKExtArray_even_extract
    (width j : Nat) :
    (bitsToKExtArray (width + 1) (2 * j)).extract 1 (width + 1) =
      bitsToKExtArray width j := by
  apply Array.ext
  · simp [bitsToKExtArray]
  · intro i hiL hiR
    have hBit : Nat.testBit (2 * j) (1 + i) = Nat.testBit j i := by
      simpa [Nat.add_comm] using testBit_even_succ j i
    simp [Array.getElem_extract, bitsToKExtArray, hBit]

private theorem bitsToKExtArray_odd_extract
    (width j : Nat) :
    (bitsToKExtArray (width + 1) (2 * j + 1)).extract 1 (width + 1) =
      bitsToKExtArray width j := by
  apply Array.ext
  · simp [bitsToKExtArray]
  · intro i hiL hiR
    have hBit : Nat.testBit (2 * j + 1) (1 + i) = Nat.testBit j i := by
      simpa [Nat.add_comm] using testBit_odd_succ j i
    simp [Array.getElem_extract, bitsToKExtArray, hBit]

/-- Single-coordinate equality term over `KExt`. -/
def eqTermK (x y : KExt) : KExt :=
  x * y + (1 - x) * (1 - y)

/-- Product equality polynomial over all coordinates (size-matched inputs only). -/
def eqPolyK (x y : Array KExt) : KExt :=
  if _h : x.size = y.size then
    (List.range x.size).foldl (fun acc i => acc * eqTermK x[i]! y[i]!) 1
  else
    0

/-- Standard multilinear-extension evaluation from a truth table over `KExt`. -/
def mleEvalK (f r : Array KExt) : KExt :=
  if _h : f.size = (2 ^ r.size) then
    (List.range f.size).foldl
      (fun acc i => acc + f[i]! * eqPolyK (bitsToKExtArray r.size i) r)
      0
  else
    0

/-- Inner-product form used as the theorem-facing target identity. -/
def mleInnerProductFormK (f r : Array KExt) : KExt :=
  (List.range f.size).foldl
    (fun acc i => acc + f[i]! * eqPolyK (bitsToKExtArray r.size i) r)
    0

/-- Theorem-facing boundary: extension-field MLE equals the inner-product form on valid table sizes. -/
def mleIdentityAssumptionK : Prop :=
  ∀ f r : Array KExt,
    f.size = (2 ^ r.size) →
    mleEvalK f r = mleInnerProductFormK f r

theorem mleEvalK_eq_innerProductForm_of_size
  {f r : Array KExt}
  (hSize : f.size = (2 ^ r.size)) :
  mleEvalK f r = mleInnerProductFormK f r := by
  unfold mleEvalK
  simp [hSize, mleInnerProductFormK]

/-- Canonical closure of the package-level extension-field MLE identity surface. -/
theorem mleIdentityAssumptionK_holds : mleIdentityAssumptionK := by
  intro f r hSize
  exact mleEvalK_eq_innerProductForm_of_size hSize

/-- One multilinear folding layer over `KExt`. -/
def foldLayerK (vals : Array KExt) (ri : KExt) : Array KExt :=
  Array.ofFn (fun i : Fin (vals.size / 2) =>
    vals[2 * i.1]! * ((1 : KExt) - ri) + vals[2 * i.1 + 1]! * ri)

@[simp] theorem foldLayerK_size (vals : Array KExt) (ri : KExt) :
    (foldLayerK vals ri).size = vals.size / 2 := by
  simp [foldLayerK]

theorem foldLayerK_get
    (vals : Array KExt) (ri : KExt)
    (i : Nat) (hi : i < vals.size / 2) :
    (foldLayerK vals ri)[i]! =
      vals[2 * i]! * ((1 : KExt) - ri) + vals[2 * i + 1]! * ri := by
  simp [foldLayerK, hi]

/-- Executable compatibility evaluator: iterative multilinear folding across coordinates over `KExt`. -/
def mleByFoldingExecK (v r : Array KExt) : KExt :=
  if r.size = 0 then
    if v.isEmpty then
      0
    else
      v[0]!
  else
    let r0 := r[0]!
    let rTail := r.extract 1 r.size
    mleByFoldingExecK (foldLayerK v r0) rTail
termination_by r.size
decreasing_by
  have hPos : 0 < r.size := Nat.pos_of_ne_zero (by assumption)
  simpa using (Nat.sub_lt hPos (Nat.succ_pos 0))

/-- Theorem-facing folding surface over `KExt` (same executable evaluator). -/
def mleByFoldingK (v r : Array KExt) : KExt :=
  mleByFoldingExecK v r

theorem mleByFoldingK_step
    (v r : Array KExt)
    (hRNe : r.size ≠ 0) :
    mleByFoldingK v r =
      mleByFoldingK (foldLayerK v r[0]!) (r.extract 1 r.size) := by
  cases hSize : r.size with
  | zero =>
      exact (hRNe hSize).elim
  | succ _ =>
      have hZero : ¬ r.size = 0 := by
        simpa [hSize]
      rw [mleByFoldingK, mleByFoldingK]
      rw [mleByFoldingExecK]
      by_cases hR : r.size = 0
      · exact (hZero hR).elim
      · rw [if_neg hR]
        simpa [hSize]

theorem mleByFoldingK_empty
    (v : Array KExt)
    (hVNe : v.size ≠ 0) :
    mleByFoldingK v #[] = v[0]! := by
  have hNotEmpty : ¬ v.isEmpty := by
    intro hEmpty
    exact hVNe (by simpa [Array.isEmpty] using hEmpty)
  unfold mleByFoldingK mleByFoldingExecK
  simp [hNotEmpty]

private theorem eqPolyK_split_head
    (x y : Array KExt)
    (hSize : x.size = y.size)
    (hPos : 0 < x.size) :
    eqPolyK x y =
      eqTermK x[0]! y[0]! * eqPolyK (x.extract 1 y.size) (y.extract 1 y.size) := by
  let n := y.size - 1
  let xTail := x.extract 1 y.size
  let yTail := y.extract 1 y.size
  have hySucc : y.size = n + 1 := by
    unfold n
    exact (Nat.succ_pred_eq_of_pos (by simpa [hSize] using hPos)).symm
  have hxSucc : x.size = n + 1 := by
    calc
      x.size = y.size := hSize
      _ = n + 1 := hySucc
  have hxTailSize : xTail.size = n := by
    simp [xTail, hSize, hySucc]
  have hyTailSize : yTail.size = n := by
    simp [yTail, hySucc]

  unfold eqPolyK
  simp [hSize]
  have hHead := foldl_mul_head n (fun i => eqTermK x[i]! y[i]!)
  have hTailCongr :
      (List.range n).foldl (fun acc i => acc * eqTermK x[i + 1]! y[i + 1]!) 1
        =
      (List.range n).foldl (fun acc i => acc * eqTermK xTail[i]! yTail[i]!) 1 := by
    apply foldl_congr
    intro acc i hi
    have hiN : i < n := List.mem_range.mp hi
    have hiXT : i < xTail.size := by simpa [hxTailSize] using hiN
    have hiYT : i < yTail.size := by simpa [hyTailSize] using hiN
    have hiX : i + 1 < x.size := by
      simpa [hxSucc] using Nat.succ_lt_succ hiN
    have hiY : i + 1 < y.size := by
      simpa [hySucc] using Nat.succ_lt_succ hiN
    have hxElem : xTail[i] = x[1 + i] := by
      simpa [xTail, Nat.add_comm] using
        (Array.getElem_extract (xs := x) (start := 1) (stop := y.size) (i := i) hiXT)
    have hyElem : yTail[i] = y[1 + i] := by
      simpa [yTail, Nat.add_comm] using
        (Array.getElem_extract (xs := y) (start := 1) (stop := y.size) (i := i) hiYT)
    have hxBang : xTail[i]! = x[i + 1]! := by
      calc
        xTail[i]! = xTail[i] := by simp [hiXT]
        _ = x[i + 1] := by simpa [Nat.add_comm] using hxElem
        _ = x[i + 1]! := by simp [hiX]
    have hyBang : yTail[i]! = y[i + 1]! := by
      calc
        yTail[i]! = yTail[i] := by simp [hiYT]
        _ = y[i + 1] := by simpa [Nat.add_comm] using hyElem
        _ = y[i + 1]! := by simp [hiY]
    simp [hxBang, hyBang]
  have hTailEq :
      eqPolyK xTail yTail =
        (List.range n).foldl (fun acc i => acc * eqTermK xTail[i]! yTail[i]!) 1 := by
    unfold eqPolyK
    simp [hxTailSize, hyTailSize]
  have hMain :
      (List.range y.size).foldl (fun acc i => acc * eqTermK x[i]! y[i]!) 1
        = eqTermK x[0]! y[0]! * eqPolyK xTail yTail := by
    calc
      (List.range y.size).foldl (fun acc i => acc * eqTermK x[i]! y[i]!) 1
          = (List.range (n + 1)).foldl (fun acc i => acc * eqTermK x[i]! y[i]!) 1 := by
              simp [hySucc]
      _ = eqTermK x[0]! y[0]! * (List.range n).foldl (fun acc i => acc * eqTermK x[i + 1]! y[i + 1]!) 1 := hHead
      _ = eqTermK x[0]! y[0]! *
            (List.range n).foldl (fun acc i => acc * eqTermK xTail[i]! yTail[i]!) 1 := by
            rw [hTailCongr]
      _ = eqTermK x[0]! y[0]! * eqPolyK xTail yTail := by
            rw [hTailEq]
  calc
    (List.range y.size).foldl (fun acc i => acc * eqTermK x[i]! y[i]!) 1
        = eqTermK x[0]! y[0]! * eqPolyK xTail yTail := hMain
    _ = eqTermK x[0]! y[0]! *
          (List.range (y.size - 1)).foldl
            (fun acc i => acc * eqTermK (x.extract 1 y.size)[i]! (y.extract 1 y.size)[i]!) 1 := by
          simpa [xTail, yTail, n] using congrArg (fun t => eqTermK x[0]! y[0]! * t) hTailEq

private theorem eqPolyK_bits_even
    (k : Nat)
    (j : Nat)
    (r : Array KExt)
    (hSize : r.size = k + 1) :
    eqPolyK (bitsToKExtArray (k + 1) (2 * j)) r =
      ((1 : KExt) - r[0]!) * eqPolyK (bitsToKExtArray k j) (r.extract 1 r.size) := by
  have hXSize : (bitsToKExtArray (k + 1) (2 * j)).size = r.size := by
    simp [bitsToKExtArray, hSize]
  have hSplit :=
    eqPolyK_split_head (x := bitsToKExtArray (k + 1) (2 * j)) (y := r)
      hXSize (by simp [bitsToKExtArray])
  have hHead : eqTermK (bitsToKExtArray (k + 1) (2 * j))[0]! r[0]! = (1 : KExt) - r[0]! := by
    have hBit0 : (bitsToKExtArray (k + 1) (2 * j))[0]! = (0 : KExt) :=
      bitsToKExtArray_even_head k j
    calc
      eqTermK (bitsToKExtArray (k + 1) (2 * j))[0]! r[0]!
          = (0 : KExt) * r[0]! + (1 - (0 : KExt)) * (1 - r[0]!) := by simp [hBit0, eqTermK]
      _ = (1 : KExt) - r[0]! := by ring
  have hTail :
      (bitsToKExtArray (k + 1) (2 * j)).extract 1 r.size = bitsToKExtArray k j := by
    simpa [hSize] using bitsToKExtArray_even_extract k j
  calc
    eqPolyK (bitsToKExtArray (k + 1) (2 * j)) r
        = eqTermK (bitsToKExtArray (k + 1) (2 * j))[0]! r[0]! *
            eqPolyK ((bitsToKExtArray (k + 1) (2 * j)).extract 1 r.size) (r.extract 1 r.size) := hSplit
    _ = ((1 : KExt) - r[0]!) * eqPolyK ((bitsToKExtArray (k + 1) (2 * j)).extract 1 r.size) (r.extract 1 r.size) := by
          simp [hHead]
    _ = ((1 : KExt) - r[0]!) * eqPolyK (bitsToKExtArray k j) (r.extract 1 r.size) := by
          simp [hTail]

private theorem eqPolyK_bits_odd
    (k : Nat)
    (j : Nat)
    (r : Array KExt)
    (hSize : r.size = k + 1) :
    eqPolyK (bitsToKExtArray (k + 1) (2 * j + 1)) r =
      r[0]! * eqPolyK (bitsToKExtArray k j) (r.extract 1 r.size) := by
  have hXSize : (bitsToKExtArray (k + 1) (2 * j + 1)).size = r.size := by
    simp [bitsToKExtArray, hSize]
  have hSplit :=
    eqPolyK_split_head (x := bitsToKExtArray (k + 1) (2 * j + 1)) (y := r)
      hXSize (by simp [bitsToKExtArray])
  have hHead : eqTermK (bitsToKExtArray (k + 1) (2 * j + 1))[0]! r[0]! = r[0]! := by
    have hBit0 : (bitsToKExtArray (k + 1) (2 * j + 1))[0]! = (1 : KExt) :=
      bitsToKExtArray_odd_head k j
    calc
      eqTermK (bitsToKExtArray (k + 1) (2 * j + 1))[0]! r[0]!
          = (1 : KExt) * r[0]! + (1 - (1 : KExt)) * (1 - r[0]!) := by simp [hBit0, eqTermK]
      _ = r[0]! := by ring
  have hTail :
      (bitsToKExtArray (k + 1) (2 * j + 1)).extract 1 r.size = bitsToKExtArray k j := by
    simpa [hSize] using bitsToKExtArray_odd_extract k j
  calc
    eqPolyK (bitsToKExtArray (k + 1) (2 * j + 1)) r
        = eqTermK (bitsToKExtArray (k + 1) (2 * j + 1))[0]! r[0]! *
            eqPolyK ((bitsToKExtArray (k + 1) (2 * j + 1)).extract 1 r.size) (r.extract 1 r.size) := hSplit
    _ = r[0]! * eqPolyK ((bitsToKExtArray (k + 1) (2 * j + 1)).extract 1 r.size) (r.extract 1 r.size) := by
          simp [hHead]
    _ = r[0]! * eqPolyK (bitsToKExtArray k j) (r.extract 1 r.size) := by
          simp [hTail]

private theorem foldl_range_pair
    (n : Nat)
    (f : Nat → KExt) :
    (List.range (2 * n)).foldl (fun acc i => acc + f i) 0 =
      (List.range n).foldl (fun acc j => acc + (f (2 * j) + f (2 * j + 1))) 0 := by
  induction n with
  | zero =>
      simp
  | succ n ih =>
      calc
        (List.range (2 * (n + 1))).foldl (fun acc i => acc + f i) 0
            = (List.range (2 * n + 2)).foldl (fun acc i => acc + f i) 0 := by
                simp [Nat.mul_add, Nat.mul_one, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]
        _ = ((List.range (2 * n + 1)).foldl (fun acc i => acc + f i) 0) + f (2 * n + 1) := by
              simp [List.range_succ, List.foldl_append]
        _ = (((List.range (2 * n)).foldl (fun acc i => acc + f i) 0) + f (2 * n)) + f (2 * n + 1) := by
              simp [List.range_succ, List.foldl_append]
        _ = (((List.range n).foldl (fun acc j => acc + (f (2 * j) + f (2 * j + 1))) 0) + f (2 * n)) + f (2 * n + 1) := by
              simp [ih]
        _ = ((List.range n).foldl (fun acc j => acc + (f (2 * j) + f (2 * j + 1))) 0) + (f (2 * n) + f (2 * n + 1)) := by
              ring
        _ = (List.range (n + 1)).foldl (fun acc j => acc + (f (2 * j) + f (2 * j + 1))) 0 := by
              simp [List.range_succ, List.foldl_append]

private theorem mleInnerProductFormK_fold_step
    (k : Nat)
    (v r : Array KExt)
    (hRSize : r.size = k + 1)
    (hVSize : v.size = 2 ^ (k + 1)) :
    mleInnerProductFormK v r =
      mleInnerProductFormK (foldLayerK v r[0]!) (r.extract 1 r.size) := by
  have hPairs : v.size / 2 = 2 ^ k := by
    calc
      v.size / 2 = (2 ^ (k + 1)) / 2 := by simpa [hVSize]
      _ = (2 * 2 ^ k) / 2 := by simp [Nat.pow_succ]
      _ = 2 ^ k := by simp
  have hVTwo : v.size = 2 * (2 ^ k) := by
    simpa [Nat.pow_succ, Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm] using hVSize
  have hTailSize : (r.extract 1 r.size).size = k := by
    simp [hRSize]
  have hCongr :
      (List.range (2 ^ k)).foldl
        (fun acc j =>
          acc +
            (v[2 * j]! * eqPolyK (bitsToKExtArray (k + 1) (2 * j)) r +
             v[2 * j + 1]! * eqPolyK (bitsToKExtArray (k + 1) (2 * j + 1)) r))
        0
        =
      (List.range (2 ^ k)).foldl
        (fun acc j =>
          acc +
            (foldLayerK v r[0]!)[j]! *
              eqPolyK (bitsToKExtArray k j) (r.extract 1 r.size))
        0 := by
    apply foldl_congr
    intro acc j hjMem
    have hj : j < 2 ^ k := List.mem_range.mp hjMem
    have hjFold : j < v.size / 2 := by simpa [hPairs] using hj
    have hEven := eqPolyK_bits_even k j r hRSize
    have hOdd := eqPolyK_bits_odd k j r hRSize
    have hFoldGet :
        (foldLayerK v r[0]!)[j]! =
          v[2 * j]! * ((1 : KExt) - r[0]!) + v[2 * j + 1]! * r[0]! :=
      foldLayerK_get v r[0]! j hjFold
    have hCombine :
        v[2 * j]! * eqPolyK (bitsToKExtArray (k + 1) (2 * j)) r +
          v[2 * j + 1]! * eqPolyK (bitsToKExtArray (k + 1) (2 * j + 1)) r
        =
        (foldLayerK v r[0]!)[j]! * eqPolyK (bitsToKExtArray k j) (r.extract 1 r.size) := by
      rw [hEven, hOdd]
      let w : KExt := eqPolyK (bitsToKExtArray k j) (r.extract 1 r.size)
      calc
        v[2 * j]! * (((1 : KExt) - r[0]!) * w) + v[2 * j + 1]! * (r[0]! * w)
            = (v[2 * j]! * ((1 : KExt) - r[0]!)) * w + (v[2 * j + 1]! * r[0]!) * w := by
                ring
        _ = (v[2 * j]! * ((1 : KExt) - r[0]!) + v[2 * j + 1]! * r[0]!) * w := by
              ring
        _ = (foldLayerK v r[0]!)[j]! * w := by
              rw [hFoldGet]
        _ = (foldLayerK v r[0]!)[j]! * eqPolyK (bitsToKExtArray k j) (r.extract 1 r.size) := by
              rfl
    simp [hCombine]
  unfold mleInnerProductFormK
  calc
    (List.range v.size).foldl
        (fun acc i => acc + v[i]! * eqPolyK (bitsToKExtArray r.size i) r)
        0
        = (List.range (2 * (2 ^ k))).foldl
            (fun acc i => acc + v[i]! * eqPolyK (bitsToKExtArray r.size i) r)
            0 := by
              simp [hVTwo]
    _ = (List.range (2 ^ k)).foldl
          (fun acc j =>
            acc +
              (v[2 * j]! * eqPolyK (bitsToKExtArray r.size (2 * j)) r +
               v[2 * j + 1]! * eqPolyK (bitsToKExtArray r.size (2 * j + 1)) r))
          0 := foldl_range_pair (2 ^ k)
            (fun i => v[i]! * eqPolyK (bitsToKExtArray r.size i) r)
    _ = (List.range (2 ^ k)).foldl
          (fun acc j =>
            acc +
              (v[2 * j]! * eqPolyK (bitsToKExtArray (k + 1) (2 * j)) r +
               v[2 * j + 1]! * eqPolyK (bitsToKExtArray (k + 1) (2 * j + 1)) r))
          0 := by
            simp [hRSize]
    _ = (List.range (2 ^ k)).foldl
          (fun acc j =>
            acc + (foldLayerK v r[0]!)[j]! *
              eqPolyK (bitsToKExtArray k j) (r.extract 1 r.size))
          0 := hCongr
    _ = (List.range (foldLayerK v r[0]!).size).foldl
          (fun acc i =>
            acc + (foldLayerK v r[0]!)[i]! *
              eqPolyK (bitsToKExtArray (r.extract 1 r.size).size i) (r.extract 1 r.size))
          0 := by
            simp [foldLayerK_size, hPairs, hTailSize]

private theorem mleInnerProductFormK_eq_mleByFoldingK_of_size_aux :
    ∀ (k : Nat) (r v : Array KExt),
      r.size = k →
      v.size = 2 ^ k →
      mleInnerProductFormK v r = mleByFoldingK v r
  | 0, r, v, hRSize, hVSize => by
      have hVNe : v.size ≠ 0 := by
        intro hE
        have : (0 : Nat) = 1 := by simpa [hVSize] using hE
        exact Nat.zero_ne_one this
      have hInner :
          mleInnerProductFormK v r = v[0]! := by
        have hEqPoly0 : eqPolyK (bitsToKExtArray 0 0) r = (1 : KExt) := by
          have hBitsSize : (bitsToKExtArray 0 0).size = r.size := by
            simp [bitsToKExtArray, hRSize]
          unfold eqPolyK
          simp [hBitsSize, bitsToKExtArray, hRSize]
        calc
          (List.range v.size).foldl
              (fun acc i => acc + v[i]! * eqPolyK (bitsToKExtArray r.size i) r)
              0
              = (List.range 1).foldl
                  (fun acc i => acc + v[i]! * eqPolyK (bitsToKExtArray r.size i) r)
                  0 := by simp [hVSize]
          _ = (0 : KExt) + v[0]! * eqPolyK (bitsToKExtArray r.size 0) r := by simp
          _ = (0 : KExt) + v[0]! * eqPolyK (bitsToKExtArray 0 0) r := by simp [hRSize]
          _ = (0 : KExt) + v[0]! * (1 : KExt) := by simp [hEqPoly0]
          _ = v[0]! := by ring
      have hFold :
          mleByFoldingK v r = v[0]! := by
        have hNotEmpty : ¬ v.isEmpty := by
          intro hEmpty
          exact hVNe (by simpa [Array.isEmpty] using hEmpty)
        unfold mleByFoldingK mleByFoldingExecK
        simp [hRSize, hNotEmpty]
      exact hInner.trans hFold.symm
  | Nat.succ k, r, v, hRSize, hVSize => by
      have hInnerStep :
          mleInnerProductFormK v r =
            mleInnerProductFormK (foldLayerK v r[0]!) (r.extract 1 r.size) := by
        exact mleInnerProductFormK_fold_step k v r hRSize hVSize
      have hFoldStep :
          mleByFoldingK v r =
            mleByFoldingK (foldLayerK v r[0]!) (r.extract 1 r.size) := by
        have hRNe : r.size ≠ 0 := by
          simpa [hRSize] using (Nat.succ_ne_zero k)
        exact mleByFoldingK_step v r hRNe
      have hTailSize : (r.extract 1 r.size).size = k := by
        simp [hRSize]
      have hFoldSize : (foldLayerK v r[0]!).size = 2 ^ k := by
        simpa [foldLayerK_size, hVSize, Nat.pow_succ] using
          congrArg (fun t => t / 2) hVSize
      have hIH :
          mleInnerProductFormK (foldLayerK v r[0]!) (r.extract 1 r.size) =
            mleByFoldingK (foldLayerK v r[0]!) (r.extract 1 r.size) :=
        mleInnerProductFormK_eq_mleByFoldingK_of_size_aux
          k (r.extract 1 r.size) (foldLayerK v r[0]!) hTailSize hFoldSize
      calc
        mleInnerProductFormK v r
            = mleInnerProductFormK (foldLayerK v r[0]!) (r.extract 1 r.size) := hInnerStep
        _ = mleByFoldingK (foldLayerK v r[0]!) (r.extract 1 r.size) := hIH
        _ = mleByFoldingK v r := by simpa [hFoldStep] using hFoldStep.symm

/--
Size-guarded folding/inner-product identity target over `KExt`.

Paper-faithful closure: executable folding equals the inner-product MLE form
for valid extension-field table sizes.
-/
theorem mleInnerProductFormK_eq_mleByFoldingK_of_size
  {v r : Array KExt}
  (hSize : v.size = 2 ^ r.size) :
  mleInnerProductFormK v r = mleByFoldingK v r := by
  exact mleInnerProductFormK_eq_mleByFoldingK_of_size_aux r.size r v rfl hSize

/--
Pointwise linear combination `f + δ*g` over `KExt` at fixed size.

Using an explicit size equality keeps this definition total and proof-friendly.
-/
def linCombK (δ : KExt) (f g : Array KExt) (hfg : f.size = g.size) : Array KExt :=
  Array.ofFn (fun i : Fin f.size => f[i] + δ * g[Fin.cast hfg i])

@[simp] theorem linCombK_size
  (δ : KExt) (f g : Array KExt) (hfg : f.size = g.size) :
  (linCombK δ f g hfg).size = f.size := by
  simp [linCombK]

/--
Theorem-facing linearity boundary on the unguarded extension-field
inner-product-form MLE sum.
-/
def mleInnerProductLinearityAssumptionK : Prop :=
  ∀ (δ : KExt) (f g r : Array KExt) (hfg : f.size = g.size),
    mleInnerProductFormK (linCombK δ f g hfg) r =
      mleInnerProductFormK f r + δ * mleInnerProductFormK g r

private theorem k_lin_seed
    (accF accG fi gi eqi δ : KExt) :
    (accF + δ * accG) + (fi + δ * gi) * eqi =
      (accF + fi * eqi) + δ * (accG + gi * eqi) := by
  ring

private theorem fold_linearK
    (δ : KExt) (f g r : Array KExt) (hfg : f.size = g.size) :
    ∀ (l : List Nat) (accF accG : KExt),
      (∀ i, i ∈ l → i < f.size) →
      l.foldl (fun acc i => acc + (linCombK δ f g hfg)[i]! * eqPolyK (bitsToKExtArray r.size i) r)
        (accF + δ * accG)
      =
      l.foldl (fun acc i => acc + f[i]! * eqPolyK (bitsToKExtArray r.size i) r) accF
        +
      δ * l.foldl (fun acc i => acc + g[i]! * eqPolyK (bitsToKExtArray r.size i) r) accG := by
  intro l
  induction l with
  | nil =>
      intro accF accG _hIn
      simp
  | cons i tl ih =>
      intro accF accG hIn
      have hi : i < f.size := hIn i (by simp)
      have hgi : i < g.size := by simpa [hfg] using hi
      have hTail : ∀ j, j ∈ tl → j < f.size := by
        intro j hj
        exact hIn j (by simp [hj])
      have hlin : (linCombK δ f g hfg)[i]! = f[i]! + δ * g[i]! := by
        simp [linCombK, hi, hgi]
      have hseed :
        (accF + δ * accG) + (linCombK δ f g hfg)[i]! * eqPolyK (bitsToKExtArray r.size i) r
          =
        (accF + f[i]! * eqPolyK (bitsToKExtArray r.size i) r)
          +
        δ * (accG + g[i]! * eqPolyK (bitsToKExtArray r.size i) r) := by
        rw [hlin]
        exact k_lin_seed accF accG f[i]! g[i]! (eqPolyK (bitsToKExtArray r.size i) r) δ
      calc
        (i :: tl).foldl
            (fun acc j => acc + (linCombK δ f g hfg)[j]! * eqPolyK (bitsToKExtArray r.size j) r)
            (accF + δ * accG)
            =
          tl.foldl
            (fun acc j => acc + (linCombK δ f g hfg)[j]! * eqPolyK (bitsToKExtArray r.size j) r)
            ((accF + δ * accG) + (linCombK δ f g hfg)[i]! * eqPolyK (bitsToKExtArray r.size i) r) := by
              rfl
        _ =
          tl.foldl
            (fun acc j => acc + (linCombK δ f g hfg)[j]! * eqPolyK (bitsToKExtArray r.size j) r)
            ((accF + f[i]! * eqPolyK (bitsToKExtArray r.size i) r)
              + δ * (accG + g[i]! * eqPolyK (bitsToKExtArray r.size i) r)) := by
                rw [hseed]
        _ =
          tl.foldl (fun acc j => acc + f[j]! * eqPolyK (bitsToKExtArray r.size j) r)
              (accF + f[i]! * eqPolyK (bitsToKExtArray r.size i) r)
            +
          δ *
            tl.foldl (fun acc j => acc + g[j]! * eqPolyK (bitsToKExtArray r.size j) r)
              (accG + g[i]! * eqPolyK (bitsToKExtArray r.size i) r) := by
                exact ih _ _ hTail
        _ =
          (i :: tl).foldl (fun acc j => acc + f[j]! * eqPolyK (bitsToKExtArray r.size j) r) accF
            +
          δ *
            (i :: tl).foldl (fun acc j => acc + g[j]! * eqPolyK (bitsToKExtArray r.size j) r) accG := by
                rfl

/-- Canonical closure of the extension-field inner-product linearity package. -/
theorem mleInnerProductLinearityAssumptionK_holds :
  mleInnerProductLinearityAssumptionK := by
  intro δ f g r hfg
  unfold mleInnerProductFormK
  have hIn : ∀ i, i ∈ List.range f.size → i < f.size := by
    intro i hi
    exact List.mem_range.mp hi
  have hFold := fold_linearK δ f g r hfg (List.range f.size) 0 0 hIn
  simpa [hfg] using hFold

/--
Theorem-facing linearity boundary on guarded `mleEvalK`.
-/
def mleEvalLinearityAssumptionK : Prop :=
  ∀ (δ : KExt) (f g r : Array KExt) (hfg : f.size = g.size),
    f.size = 2 ^ r.size →
    mleEvalK (linCombK δ f g hfg) r =
      mleEvalK f r + δ * mleEvalK g r

theorem mleEvalK_linComb_of_assumption
  (hLin : mleEvalLinearityAssumptionK)
  (δ : KExt) (f g r : Array KExt) (hfg : f.size = g.size) (hSize : f.size = 2 ^ r.size) :
  mleEvalK (linCombK δ f g hfg) r =
    mleEvalK f r + δ * mleEvalK g r := by
  exact hLin δ f g r hfg hSize

/--
Guarded extension-field MLE linearity follows from:
1. valid-size evaluator/inner-product identity, and
2. inner-product-form linearity.
-/
theorem mleEvalK_linComb_of_assumptions
  (hMLE : mleIdentityAssumptionK)
  (hInnerLin : mleInnerProductLinearityAssumptionK)
  (δ : KExt) (f g r : Array KExt) (hfg : f.size = g.size) (hSize : f.size = 2 ^ r.size) :
  mleEvalK (linCombK δ f g hfg) r =
    mleEvalK f r + δ * mleEvalK g r := by
  have hSizeLin : (linCombK δ f g hfg).size = 2 ^ r.size := by
    simpa [linCombK_size] using hSize
  calc
    mleEvalK (linCombK δ f g hfg) r = mleInnerProductFormK (linCombK δ f g hfg) r :=
      hMLE _ _ hSizeLin
    _ = mleInnerProductFormK f r + δ * mleInnerProductFormK g r :=
      hInnerLin δ f g r hfg
    _ = mleEvalK f r + δ * mleInnerProductFormK g r := by
      rw [hMLE f r hSize]
    _ = mleEvalK f r + δ * mleEvalK g r := by
      rw [hMLE g r (by simpa [hfg] using hSize)]

/-- Canonical closure of the guarded extension-field MLE linearity package. -/
theorem mleEvalLinearityAssumptionK_holds :
  mleEvalLinearityAssumptionK := by
  intro δ f g r hfg hSize
  exact mleEvalK_linComb_of_assumptions
    mleIdentityAssumptionK_holds
    mleInnerProductLinearityAssumptionK_holds
    δ f g r hfg hSize

/--
Guarded evaluator/folding identity over `KExt`.

This combines the guarded `mleEvalK = mleInnerProductFormK` identity with the
folding/inner-product theorem above.
-/
theorem mleEvalK_eq_mleByFoldingK_of_size
  {v r : Array KExt}
  (hSize : v.size = 2 ^ r.size) :
  mleEvalK v r = mleByFoldingK v r := by
  calc
    mleEvalK v r = mleInnerProductFormK v r :=
      mleEvalK_eq_innerProductForm_of_size hSize
    _ = mleByFoldingK v r :=
      mleInnerProductFormK_eq_mleByFoldingK_of_size hSize

end SuperNeo
