import SuperNeo.EqPoly

/-!
Multilinear-extension scaffold.

This file keeps executable evaluators and exposes theorem-facing interfaces used
by protocol composition.
-/

namespace SuperNeo

open F

/-- Bit-vector embedding for an index mask (little-endian). -/
def bitsToFieldArray (width mask : Nat) : Array F :=
  Array.ofFn (fun i : Fin width =>
    if Nat.testBit mask i.1 then (1 : F) else 0)

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

private theorem bitsToFieldArray_even_head (width j : Nat) :
    (bitsToFieldArray (width + 1) (2 * j))[0]! = (0 : F) := by
  simp [bitsToFieldArray, Nat.testBit_zero]

private theorem bitsToFieldArray_odd_head (width j : Nat) :
    (bitsToFieldArray (width + 1) (2 * j + 1))[0]! = (1 : F) := by
  simp [bitsToFieldArray, Nat.testBit_zero]

private theorem bitsToFieldArray_even_tail
    (width j i : Nat)
    (hi : i < width) :
    (bitsToFieldArray (width + 1) (2 * j))[i + 1]! =
      (bitsToFieldArray width j)[i]! := by
  have hiSucc : i + 1 < width + 1 := Nat.succ_lt_succ hi
  simp [bitsToFieldArray, hi, hiSucc, testBit_even_succ]

private theorem bitsToFieldArray_odd_tail
    (width j i : Nat)
    (hi : i < width) :
    (bitsToFieldArray (width + 1) (2 * j + 1))[i + 1]! =
      (bitsToFieldArray width j)[i]! := by
  have hiSucc : i + 1 < width + 1 := Nat.succ_lt_succ hi
  simp [bitsToFieldArray, hi, hiSucc, testBit_odd_succ]

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

/-- Canonical closure of the package-level MLE identity surface. -/
theorem mleIdentityAssumption_holds : mleIdentityAssumption := by
  intro f r hSize
  exact mleEval_eq_innerProductForm_of_size hSize

/-!
Executable compatibility layer

These definitions keep historical check/golden-vector call-sites stable while the
theorem-facing API above remains the canonical MLE boundary surface.
-/

/-- Compatibility alias for single-coordinate basis weight `χ_r(j)`. -/
def chiWeight (r : Array F) (j : Nat) : F :=
  eqPoly (bitsToFieldArray r.size j) r

/-- Compatibility vector `rHat` with explicit output length `n`. -/
def rHat (r : Array F) (n : Nat) : Array F :=
  Array.ofFn (fun i : Fin n => chiWeight r i.1)

@[simp] theorem rHat_size (r : Array F) (n : Nat) :
    (rHat r n).size = n := by
  simp [rHat]

/-- Compatibility evaluator: inner-product MLE using `rHat`. -/
def mleByInnerProduct (v r : Array F) : F :=
  mleInnerProductForm v r

/--
One multilinear folding layer.

This is the executable half-step used by `mleByFolding`; it is also reused by
the SumCheck prefix-soundness development to relate honest residual tables to
the MLE oracle semantics.
-/
def foldLayer (vals : Array F) (ri : F) : Array F :=
  Array.ofFn (fun i : Fin (vals.size / 2) =>
    vals[2 * i.1]! * ((1 : F) - ri) + vals[2 * i.1 + 1]! * ri)

@[simp] theorem foldLayer_size (vals : Array F) (ri : F) :
    (foldLayer vals ri).size = vals.size / 2 := by
  simp [foldLayer]

theorem foldLayer_get
    (vals : Array F) (ri : F)
    (i : Nat) (hi : i < vals.size / 2) :
    (foldLayer vals ri)[i]! = vals[2 * i]! * ((1 : F) - ri) + vals[2 * i + 1]! * ri := by
  simp [foldLayer, hi]

/--
Executable compatibility evaluator: iterative multilinear folding across
coordinates.
-/
def mleByFoldingExec (v r : Array F) : F :=
  if r.size = 0 then
    if v.isEmpty then
      0
    else
      v[0]!
  else
    let r0 := r[0]!
    let rTail := r.extract 1 r.size
    mleByFoldingExec (foldLayer v r0) rTail
termination_by r.size
decreasing_by
  have hPos : 0 < r.size := Nat.pos_of_ne_zero (by assumption)
  simpa using (Nat.sub_lt hPos (Nat.succ_pos 0))

/-- Unfolding step for non-empty folding states. -/
theorem mleByFoldingExec_step
    (v r : Array F)
    (hRNe : r.size ≠ 0) :
    mleByFoldingExec v r =
      mleByFoldingExec (foldLayer v r[0]!) (r.extract 1 r.size) := by
  cases hSize : r.size with
  | zero =>
      exact (hRNe hSize).elim
  | succ n =>
      have hZero : ¬ r.size = 0 := by
        simpa [hSize]
      rw [mleByFoldingExec]
      by_cases hR : r.size = 0
      · exact (hZero hR).elim
      · rw [if_neg hR]
        simpa [hSize]

/-- Theorem-facing folding surface (same executable evaluator). -/
def mleByFolding (v r : Array F) : F :=
  mleByFoldingExec v r

theorem mleByFolding_step
    (v r : Array F)
    (hRNe : r.size ≠ 0) :
    mleByFolding v r =
      mleByFolding (foldLayer v r[0]!) (r.extract 1 r.size) := by
  simpa [mleByFolding] using mleByFoldingExec_step v r hRNe

theorem mleByFolding_empty
    (v : Array F)
    (hVNe : v.size ≠ 0) :
    mleByFolding v #[] = v[0]! := by
  have hNotEmpty : ¬ v.isEmpty := by
    intro hEmpty
    exact hVNe (by simpa [Array.isEmpty] using hEmpty)
  unfold mleByFolding mleByFoldingExec
  simp [hNotEmpty]

/-!
`mleByInnerProduct_eq_mleByFolding_of_size` is proved later in this file after
the algebraic helper lemmas used by the folding-to-sum derivation.
-/

/--
Compatibility boolean identity check between theorem-facing inner-product MLE
and executable folding MLE.
-/
def mleIdentity (v r : Array F) : Bool :=
  if v.size != 2 ^ r.size then
    false
  else
    decide (mleByInnerProduct v r = mleByFolding v r)

theorem mleIdentity_sound
  {v r : Array F}
  (hOk : mleIdentity v r = true) :
  v.size = 2 ^ r.size ∧ mleByInnerProduct v r = mleByFolding v r := by
  unfold mleIdentity at hOk
  by_cases hSize : v.size = 2 ^ r.size
  ·
    have hDec : decide (mleByInnerProduct v r = mleByFolding v r) = true := by
      simpa [hSize] using hOk
    exact ⟨hSize, decide_eq_true_eq.mp hDec⟩
  · simp [hSize] at hOk

theorem mleIdentity_complete
  {v r : Array F}
  (hProp : v.size = 2 ^ r.size ∧ mleByInnerProduct v r = mleByFolding v r) :
  mleIdentity v r = true := by
  rcases hProp with ⟨hSize, hEq⟩
  unfold mleIdentity
  simp [hSize, decide_eq_true hEq]

theorem mleIdentity_eq_true_iff
  {v r : Array F} :
  mleIdentity v r = true ↔
    v.size = 2 ^ r.size ∧ mleByInnerProduct v r = mleByFolding v r := by
  constructor
  · exact mleIdentity_sound
  · exact mleIdentity_complete

/-- Lightweight executable sanity check for generated-vector regression harnesses. -/
def mleSanity : Bool :=
  let v := #[F.ofNat 3, F.ofNat 5, F.ofNat 7, F.ofNat 9]
  let r := #[F.ofNat 2, F.ofNat 1]
  mleIdentity v r

/--
Theorem-facing Boolean-cube delta boundary for `eqPoly`.

This matches the paper-level selector behavior:
`eqPoly x y = 1` iff `x = y`, else `0`, on bit vectors of equal size.
-/
def eqPolyDeltaOnBitsAssumption : Prop :=
  ∀ x y : Array F,
    x.size = y.size →
    IsBitVec x →
    IsBitVec y →
    eqPoly x y = (if x = y then 1 else 0)

theorem eqPoly_eq_delta_of_isBitVec_of_assumption
  (hDelta : eqPolyDeltaOnBitsAssumption)
  {x y : Array F}
  (hSize : x.size = y.size)
  (hx : IsBitVec x)
  (hy : IsBitVec y) :
  eqPoly x y = (if x = y then 1 else 0) := by
  exact hDelta x y hSize hx hy

/--
Bridge lemma: if `EqPoly` selector behavior is available, then the MLE-local
Boolean-cube delta package also holds.
-/
theorem eqPolyDeltaOnBitsAssumption_of_eqPolyAssumption
  (hEq : eqPolyAssumption) :
  eqPolyDeltaOnBitsAssumption := by
  intro x y hSize hx hy
  exact hEq x y hSize hx hy

/--
Canonical closure of the MLE-local Boolean-cube delta package from the EqPoly
selector boundary.
-/
theorem eqPolyDeltaOnBitsAssumption_holds_of_eqPolyAssumption
  (hEq : eqPolyAssumption) :
  eqPolyDeltaOnBitsAssumption :=
  eqPolyDeltaOnBitsAssumption_of_eqPolyAssumption hEq

/-- χ(r) (a.k.a. `r̂`) basis-weight vector, indexed by Boolean-cube masks. -/
def chi (r : Array F) : Array F :=
  Array.ofFn (fun i : Fin (2 ^ r.size) =>
    eqPoly (bitsToFieldArray r.size i.1) r)

@[simp] theorem chi_size (r : Array F) :
    (chi r).size = 2 ^ r.size := by
  simp [chi]

/--
Dot-product surface on arrays.

Returns `0` when lengths mismatch.
-/
def dot (a b : Array F) : F :=
  if _h : a.size = b.size then
    (List.range a.size).foldl
      (fun acc i => acc + a[i]! * b[i]!)
      0
  else
    0

/-- MLE expressed as an inner product with `chi`. -/
def mleViaChiDot (f r : Array F) : F :=
  dot f (chi r)

/--
Theorem-facing boundary linking the executable inner-product-form sum to `dot f (chi r)`.
-/
def mleChiIdentityAssumption : Prop :=
  ∀ f r : Array F,
    f.size = (2 ^ r.size) →
    mleInnerProductForm f r = mleViaChiDot f r

theorem mleInnerProductForm_eq_mleViaChiDot_of_size_of_assumption
  (hChi : mleChiIdentityAssumption)
  {f r : Array F}
  (hSize : f.size = (2 ^ r.size)) :
  mleInnerProductForm f r = mleViaChiDot f r := by
  exact hChi f r hSize

/-- Direct (assumption-free) size-guarded identity between sum-form and chi-dot form. -/
theorem mleInnerProductForm_eq_mleViaChiDot_of_size
  {f r : Array F}
  (hSize : f.size = (2 ^ r.size)) :
  mleInnerProductForm f r = mleViaChiDot f r := by
  -- Helper: two `foldl` traversals over the same list are equal when their
  -- step functions agree on every visited element.
  have hFoldlCongr :
      ∀ (l : List Nat) (init : F)
        (step1 step2 : F → Nat → F),
        (∀ acc i, i ∈ l → step1 acc i = step2 acc i) →
        l.foldl step1 init = l.foldl step2 init := by
      intro l init step1 step2 hStep
      induction l generalizing init with
      | nil =>
          rfl
      | cons a tl ih =>
          have hHead : step1 init a = step2 init a := hStep init a (by simp)
          calc
            (a :: tl).foldl step1 init = tl.foldl step1 (step1 init a) := by rfl
            _ = tl.foldl step1 (step2 init a) := by rw [hHead]
            _ = tl.foldl step2 (step2 init a) := by
              apply ih
              intro acc i hiMem
              exact hStep acc i (by simp [hiMem])
            _ = (a :: tl).foldl step2 init := by rfl
  unfold mleInnerProductForm mleViaChiDot dot
  have hChiSize : (chi r).size = f.size := by
    simpa [chi_size] using hSize.symm
  simp [hChiSize]
  apply hFoldlCongr
  intro acc i hiMem
  have hiF : i < f.size := List.mem_range.mp hiMem
  have hi : i < 2 ^ r.size := by
    simpa [hSize] using hiF
  simp [chi, hi]

/-- Canonical closure of the package-level chi/dot identity surface. -/
theorem mleChiIdentityAssumption_holds : mleChiIdentityAssumption := by
  intro f r hSize
  exact mleInnerProductForm_eq_mleViaChiDot_of_size hSize

/--
Pointwise linear combination `f + δ*g` at fixed size.

Using an explicit size equality keeps this definition total and proof-friendly.
-/
def linComb (δ : F) (f g : Array F) (hfg : f.size = g.size) : Array F :=
  Array.ofFn (fun i : Fin f.size => f[i] + δ * g[Fin.cast hfg i])

@[simp] theorem linComb_size
  (δ : F) (f g : Array F) (hfg : f.size = g.size) :
  (linComb δ f g hfg).size = f.size := by
  simp [linComb]

/--
Theorem-facing linearity boundary on the unguarded inner-product-form MLE sum.
-/
def mleInnerProductLinearityAssumption : Prop :=
  ∀ (δ : F) (f g r : Array F) (hfg : f.size = g.size),
    mleInnerProductForm (linComb δ f g hfg) r =
      mleInnerProductForm f r + δ * mleInnerProductForm g r

private theorem f_add_assoc (a b c : F) : (a + b) + c = a + (b + c) := by
  apply Fin.ext
  change (((a.val + b.val) % Goldilocks.q + c.val) % Goldilocks.q) =
    ((a.val + ((b.val + c.val) % Goldilocks.q)) % Goldilocks.q)
  calc
    (((a.val + b.val) % Goldilocks.q + c.val) % Goldilocks.q)
        = ((a.val + b.val + c.val) % Goldilocks.q) := by
            simp [Nat.add_assoc, Nat.mod_add_mod]
    _ = ((a.val + (b.val + c.val)) % Goldilocks.q) := by
            simp [Nat.add_assoc]
    _ = ((a.val + ((b.val + c.val) % Goldilocks.q)) % Goldilocks.q) := by
            simp [Nat.add_mod_mod]

private theorem f_add_comm (a b : F) : a + b = b + a := by
  apply Fin.ext
  change ((a.val + b.val) % Goldilocks.q) = ((b.val + a.val) % Goldilocks.q)
  simp [Nat.add_comm]

private theorem f_mul_assoc (a b c : F) : (a * b) * c = a * (b * c) := by
  simpa using Fin.mul_assoc a b c

private theorem f_left_distrib (a b c : F) : (a + b) * c = a * c + b * c := by
  apply Fin.ext
  change ((((a.val + b.val) % Goldilocks.q) * c.val) % Goldilocks.q) =
    (((a.val * c.val) % Goldilocks.q + (b.val * c.val) % Goldilocks.q) % Goldilocks.q)
  calc
    ((((a.val + b.val) % Goldilocks.q) * c.val) % Goldilocks.q)
        = (((a.val + b.val) * c.val) % Goldilocks.q) := by
            simp [Nat.mod_mul_mod]
    _ = (((a.val * c.val) + (b.val * c.val)) % Goldilocks.q) := by
            simp [Nat.add_mul]
    _ = (((a.val * c.val) % Goldilocks.q + (b.val * c.val) % Goldilocks.q) % Goldilocks.q) := by
            simp [Nat.add_mod]

private theorem f_right_distrib (a b c : F) : a * (b + c) = a * b + a * c := by
  apply Fin.ext
  change ((a.val * ((b.val + c.val) % Goldilocks.q)) % Goldilocks.q) =
    (((a.val * b.val) % Goldilocks.q + (a.val * c.val) % Goldilocks.q) % Goldilocks.q)
  calc
    ((a.val * ((b.val + c.val) % Goldilocks.q)) % Goldilocks.q)
        = ((a.val * (b.val + c.val)) % Goldilocks.q) := by
            simp [Nat.mul_mod_mod]
    _ = (((a.val * b.val) + (a.val * c.val)) % Goldilocks.q) := by
            simp [Nat.mul_add]
    _ = (((a.val * b.val) % Goldilocks.q + (a.val * c.val) % Goldilocks.q) % Goldilocks.q) := by
            simp [Nat.add_mod]

private theorem f_mul_zero (a : F) : a * 0 = 0 := by
  apply Fin.ext
  change (a.val * 0) % Goldilocks.q = 0
  simp

private theorem f_zero_mul (a : F) : (0 : F) * a = 0 := by
  apply Fin.ext
  change (0 * a.val) % Goldilocks.q = 0
  simp

private theorem f_mul_one (a : F) : a * (1 : F) = a := by
  apply Fin.ext
  change (a.val * 1) % Goldilocks.q = a.val
  simpa using Nat.mod_eq_of_lt a.isLt

private theorem f_one_mul (a : F) : (1 : F) * a = a := by
  apply Fin.ext
  change (1 * a.val) % Goldilocks.q = a.val
  simpa using Nat.mod_eq_of_lt a.isLt

private theorem f_add_zero (a : F) : a + 0 = a := by
  apply Fin.ext
  change (a.val + 0) % Goldilocks.q = a.val
  simpa using (Nat.mod_eq_of_lt a.isLt)

private theorem f_zero_add (a : F) : 0 + a = a := by
  rw [f_add_comm]
  exact f_add_zero a

private theorem f_mul_comm (a b : F) : a * b = b * a := by
  simpa using Fin.mul_comm a b

private theorem foldl_mul_head
    (n : Nat)
    (t : Nat → F) :
    (List.range (n + 1)).foldl (fun acc i => acc * t i) 1
      = t 0 * (List.range n).foldl (fun acc i => acc * t (i + 1)) 1 := by
  induction n with
  | zero =>
      simp [f_one_mul, f_mul_one]
  | succ n ih =>
      calc
        (List.range (Nat.succ n + 1)).foldl (fun acc i => acc * t i) 1
            = ((List.range (n + 1)).foldl (fun acc i => acc * t i) 1) * t (n + 1) := by
                simp [List.range_succ, List.foldl_append]
        _ = (t 0 * (List.range n).foldl (fun acc i => acc * t (i + 1)) 1) * t (n + 1) := by
              simp [ih]
        _ = t 0 * ((List.range n).foldl (fun acc i => acc * t (i + 1)) 1 * t (n + 1)) := by
              rw [f_mul_assoc]
        _ = t 0 * (List.range (n + 1)).foldl (fun acc i => acc * t (i + 1)) 1 := by
              simp [List.range_succ, List.foldl_append]

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

private theorem bitsToFieldArray_even_extract
    (width j : Nat) :
    (bitsToFieldArray (width + 1) (2 * j)).extract 1 (width + 1) =
      bitsToFieldArray width j := by
  apply Array.ext
  · simp [bitsToFieldArray]
  · intro i hiL hiR
    have hBit : Nat.testBit (2 * j) (1 + i) = Nat.testBit j i := by
      simpa [Nat.add_comm] using testBit_even_succ j i
    simp [Array.getElem_extract, bitsToFieldArray, hBit]

private theorem bitsToFieldArray_odd_extract
    (width j : Nat) :
    (bitsToFieldArray (width + 1) (2 * j + 1)).extract 1 (width + 1) =
      bitsToFieldArray width j := by
  apply Array.ext
  · simp [bitsToFieldArray]
  · intro i hiL hiR
    have hBit : Nat.testBit (2 * j + 1) (1 + i) = Nat.testBit j i := by
      simpa [Nat.add_comm] using testBit_odd_succ j i
    simp [Array.getElem_extract, bitsToFieldArray, hBit]

private theorem eqPoly_split_head
    (x y : Array F)
    (hSize : x.size = y.size)
    (hPos : 0 < x.size) :
    eqPoly x y =
      eqTerm x[0]! y[0]! * eqPoly (x.extract 1 y.size) (y.extract 1 y.size) := by
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

  unfold eqPoly
  simp [hSize]
  have hHead := foldl_mul_head n (fun i => eqTerm x[i]! y[i]!)
  have hTailCongr :
      (List.range n).foldl (fun acc i => acc * eqTerm x[i + 1]! y[i + 1]!) 1
        =
      (List.range n).foldl (fun acc i => acc * eqTerm xTail[i]! yTail[i]!) 1 := by
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
      eqPoly xTail yTail =
        (List.range n).foldl (fun acc i => acc * eqTerm xTail[i]! yTail[i]!) 1 := by
    unfold eqPoly
    simp [hxTailSize, hyTailSize]
  have hMain :
      (List.range y.size).foldl (fun acc i => acc * eqTerm x[i]! y[i]!) 1
        = eqTerm x[0]! y[0]! * eqPoly xTail yTail := by
    calc
      (List.range y.size).foldl (fun acc i => acc * eqTerm x[i]! y[i]!) 1
          = (List.range (n + 1)).foldl (fun acc i => acc * eqTerm x[i]! y[i]!) 1 := by
              simp [hySucc]
      _ = eqTerm x[0]! y[0]! * (List.range n).foldl (fun acc i => acc * eqTerm x[i + 1]! y[i + 1]!) 1 := hHead
      _ = eqTerm x[0]! y[0]! *
            (List.range n).foldl (fun acc i => acc * eqTerm xTail[i]! yTail[i]!) 1 := by
            rw [hTailCongr]
      _ = eqTerm x[0]! y[0]! * eqPoly xTail yTail := by
            rw [hTailEq]
  calc
    (List.range y.size).foldl (fun acc i => acc * eqTerm x[i]! y[i]!) 1
        = eqTerm x[0]! y[0]! * eqPoly xTail yTail := hMain
    _ = eqTerm x[0]! y[0]! *
          (List.range (y.size - 1)).foldl
            (fun acc i => acc * eqTerm (x.extract 1 y.size)[i]! (y.extract 1 y.size)[i]!) 1 := by
          simpa [xTail, yTail, n] using congrArg (fun t => eqTerm x[0]! y[0]! * t) hTailEq

private theorem eqPoly_bits_even
    (k : Nat)
    (j : Nat)
    (r : Array F)
    (hSize : r.size = k + 1) :
    eqPoly (bitsToFieldArray (k + 1) (2 * j)) r =
      ((1 : F) - r[0]!) * eqPoly (bitsToFieldArray k j) (r.extract 1 r.size) := by
  have hXSize : (bitsToFieldArray (k + 1) (2 * j)).size = r.size := by
    simp [bitsToFieldArray, hSize]
  have hSplit :=
    eqPoly_split_head (x := bitsToFieldArray (k + 1) (2 * j)) (y := r)
      hXSize (by simp [bitsToFieldArray])
  have hHead : eqTerm (bitsToFieldArray (k + 1) (2 * j))[0]! r[0]! = (1 : F) - r[0]! := by
    have hBit0 : (bitsToFieldArray (k + 1) (2 * j))[0]! = (0 : F) :=
      bitsToFieldArray_even_head k j
    calc
      eqTerm (bitsToFieldArray (k + 1) (2 * j))[0]! r[0]!
          = (0 : F) * r[0]! + (1 : F) * ((1 : F) - r[0]!) := by simp [hBit0, eqTerm]
      _ = (0 : F) + (1 : F) * ((1 : F) - r[0]!) := by simp [f_zero_mul]
      _ = (1 : F) * ((1 : F) - r[0]!) := f_zero_add _
      _ = (1 : F) - r[0]! := f_one_mul ((1 : F) - r[0]!)
  have hTail :
      (bitsToFieldArray (k + 1) (2 * j)).extract 1 r.size = bitsToFieldArray k j := by
    simpa [hSize] using bitsToFieldArray_even_extract k j
  calc
    eqPoly (bitsToFieldArray (k + 1) (2 * j)) r
        = eqTerm (bitsToFieldArray (k + 1) (2 * j))[0]! r[0]! *
            eqPoly ((bitsToFieldArray (k + 1) (2 * j)).extract 1 r.size) (r.extract 1 r.size) := hSplit
    _ = ((1 : F) - r[0]!) * eqPoly ((bitsToFieldArray (k + 1) (2 * j)).extract 1 r.size) (r.extract 1 r.size) := by
          simp [hHead]
    _ = ((1 : F) - r[0]!) * eqPoly (bitsToFieldArray k j) (r.extract 1 r.size) := by
          simp [hTail]

private theorem eqPoly_bits_odd
    (k : Nat)
    (j : Nat)
    (r : Array F)
    (hSize : r.size = k + 1) :
    eqPoly (bitsToFieldArray (k + 1) (2 * j + 1)) r =
      r[0]! * eqPoly (bitsToFieldArray k j) (r.extract 1 r.size) := by
  have hXSize : (bitsToFieldArray (k + 1) (2 * j + 1)).size = r.size := by
    simp [bitsToFieldArray, hSize]
  have hSplit :=
    eqPoly_split_head (x := bitsToFieldArray (k + 1) (2 * j + 1)) (y := r)
      hXSize (by simp [bitsToFieldArray])
  have hHead : eqTerm (bitsToFieldArray (k + 1) (2 * j + 1))[0]! r[0]! = r[0]! := by
    have hBit0 : (bitsToFieldArray (k + 1) (2 * j + 1))[0]! = (1 : F) :=
      bitsToFieldArray_odd_head k j
    calc
      eqTerm (bitsToFieldArray (k + 1) (2 * j + 1))[0]! r[0]!
          = (1 : F) * r[0]! + (0 : F) * ((1 : F) - r[0]!) := by simp [hBit0, eqTerm]
      _ = (1 : F) * r[0]! + (0 : F) := by simp [f_zero_mul]
      _ = (1 : F) * r[0]! := f_add_zero _
      _ = r[0]! := f_one_mul (r[0]!)
  have hTail :
      (bitsToFieldArray (k + 1) (2 * j + 1)).extract 1 r.size = bitsToFieldArray k j := by
    simpa [hSize] using bitsToFieldArray_odd_extract k j
  calc
    eqPoly (bitsToFieldArray (k + 1) (2 * j + 1)) r
        = eqTerm (bitsToFieldArray (k + 1) (2 * j + 1))[0]! r[0]! *
            eqPoly ((bitsToFieldArray (k + 1) (2 * j + 1)).extract 1 r.size) (r.extract 1 r.size) := hSplit
    _ = r[0]! * eqPoly ((bitsToFieldArray (k + 1) (2 * j + 1)).extract 1 r.size) (r.extract 1 r.size) := by
          simp [hHead]
    _ = r[0]! * eqPoly (bitsToFieldArray k j) (r.extract 1 r.size) := by
          simp [hTail]

private theorem foldl_range_pair
    (n : Nat)
    (f : Nat → F) :
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
              rw [f_add_assoc]
        _ = (List.range (n + 1)).foldl (fun acc j => acc + (f (2 * j) + f (2 * j + 1))) 0 := by
              simp [List.range_succ, List.foldl_append]

private theorem mleInnerProductForm_fold_step
    (k : Nat)
    (v r : Array F)
    (hRSize : r.size = k + 1)
    (hVSize : v.size = 2 ^ (k + 1)) :
    mleInnerProductForm v r =
      mleInnerProductForm (foldLayer v r[0]!) (r.extract 1 r.size) := by
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
            (v[2 * j]! * eqPoly (bitsToFieldArray (k + 1) (2 * j)) r +
             v[2 * j + 1]! * eqPoly (bitsToFieldArray (k + 1) (2 * j + 1)) r))
        0
        =
      (List.range (2 ^ k)).foldl
        (fun acc j =>
          acc +
            (foldLayer v r[0]!)[j]! *
              eqPoly (bitsToFieldArray k j) (r.extract 1 r.size))
        0 := by
    apply foldl_congr
    intro acc j hjMem
    have hj : j < 2 ^ k := List.mem_range.mp hjMem
    have hjFold : j < v.size / 2 := by simpa [hPairs] using hj
    have hEven := eqPoly_bits_even k j r hRSize
    have hOdd := eqPoly_bits_odd k j r hRSize
    have hFoldGet :
        (foldLayer v r[0]!)[j]! =
          v[2 * j]! * ((1 : F) - r[0]!) + v[2 * j + 1]! * r[0]! :=
      foldLayer_get v r[0]! j hjFold
    have hCombine :
        v[2 * j]! * eqPoly (bitsToFieldArray (k + 1) (2 * j)) r +
          v[2 * j + 1]! * eqPoly (bitsToFieldArray (k + 1) (2 * j + 1)) r
        =
        (foldLayer v r[0]!)[j]! * eqPoly (bitsToFieldArray k j) (r.extract 1 r.size) := by
      rw [hEven, hOdd]
      let w : F := eqPoly (bitsToFieldArray k j) (r.extract 1 r.size)
      calc
        v[2 * j]! * (((1 : F) - r[0]!) * w) + v[2 * j + 1]! * (r[0]! * w)
            = (v[2 * j]! * ((1 : F) - r[0]!)) * w + (v[2 * j + 1]! * r[0]!) * w := by
                rw [f_mul_assoc, f_mul_assoc]
        _ = (v[2 * j]! * ((1 : F) - r[0]!) + v[2 * j + 1]! * r[0]!) * w := by
              rw [f_left_distrib]
        _ = (foldLayer v r[0]!)[j]! * w := by
              rw [hFoldGet]
        _ = (foldLayer v r[0]!)[j]! * eqPoly (bitsToFieldArray k j) (r.extract 1 r.size) := by
              rfl
    simp [hCombine]
  unfold mleInnerProductForm
  calc
    (List.range v.size).foldl
        (fun acc i => acc + v[i]! * eqPoly (bitsToFieldArray r.size i) r)
        0
        = (List.range (2 * (2 ^ k))).foldl
            (fun acc i => acc + v[i]! * eqPoly (bitsToFieldArray r.size i) r)
            0 := by
              simp [hVTwo]
    _ = (List.range (2 ^ k)).foldl
          (fun acc j =>
            acc +
              (v[2 * j]! * eqPoly (bitsToFieldArray r.size (2 * j)) r +
               v[2 * j + 1]! * eqPoly (bitsToFieldArray r.size (2 * j + 1)) r))
          0 := foldl_range_pair (2 ^ k)
            (fun i => v[i]! * eqPoly (bitsToFieldArray r.size i) r)
    _ = (List.range (2 ^ k)).foldl
          (fun acc j =>
            acc +
              (v[2 * j]! * eqPoly (bitsToFieldArray (k + 1) (2 * j)) r +
               v[2 * j + 1]! * eqPoly (bitsToFieldArray (k + 1) (2 * j + 1)) r))
          0 := by
            simp [hRSize]
    _ = (List.range (2 ^ k)).foldl
          (fun acc j =>
            acc + (foldLayer v r[0]!)[j]! *
              eqPoly (bitsToFieldArray k j) (r.extract 1 r.size))
          0 := hCongr
    _ = (List.range (foldLayer v r[0]!).size).foldl
          (fun acc i =>
            acc + (foldLayer v r[0]!)[i]! *
              eqPoly (bitsToFieldArray (r.extract 1 r.size).size i) (r.extract 1 r.size))
          0 := by
            simp [foldLayer_size, hPairs, hTailSize]

private theorem mleByInnerProduct_eq_mleByFolding_of_size_aux :
    ∀ (k : Nat) (r v : Array F),
      r.size = k →
      v.size = 2 ^ k →
      mleByInnerProduct v r = mleByFolding v r
  | 0, r, v, hRSize, hVSize => by
      have hVNonEmpty : ¬ v.isEmpty := by
        intro hE
        have hZero : v.size = 0 := by simpa [Array.isEmpty] using hE
        have : (0 : Nat) = 1 := by simpa [hVSize] using hZero
        exact Nat.zero_ne_one this
      have hInner :
          mleByInnerProduct v r = v[0]! := by
        unfold mleByInnerProduct mleInnerProductForm
        have hEqPoly0 : eqPoly (bitsToFieldArray 0 0) r = (1 : F) := by
          have hBitsSize : (bitsToFieldArray 0 0).size = r.size := by
            simp [bitsToFieldArray, hRSize]
          unfold eqPoly
          simp [hBitsSize, bitsToFieldArray, hRSize]
        calc
          (List.range v.size).foldl
              (fun acc i => acc + v[i]! * eqPoly (bitsToFieldArray r.size i) r)
              0
              = (List.range 1).foldl
                  (fun acc i => acc + v[i]! * eqPoly (bitsToFieldArray r.size i) r)
                  0 := by simp [hVSize]
          _ = (0 : F) + v[0]! * eqPoly (bitsToFieldArray r.size 0) r := by simp
          _ = (0 : F) + v[0]! * eqPoly (bitsToFieldArray 0 0) r := by simp [hRSize]
          _ = (0 : F) + v[0]! * (1 : F) := by simp [hEqPoly0]
          _ = (0 : F) + v[0]! := by simp [f_mul_one]
          _ = v[0]! := f_zero_add (v[0]!)
      have hFold :
          mleByFolding v r = v[0]! := by
        unfold mleByFolding mleByFoldingExec
        simp [hRSize, hVNonEmpty]
      exact hInner.trans hFold.symm
  | Nat.succ k, r, v, hRSize, hVSize => by
      have hInnerStep :
          mleByInnerProduct v r =
            mleByInnerProduct (foldLayer v r[0]!) (r.extract 1 r.size) := by
        unfold mleByInnerProduct
        exact mleInnerProductForm_fold_step k v r hRSize hVSize
      have hFoldStep :
          mleByFolding v r =
            mleByFolding (foldLayer v r[0]!) (r.extract 1 r.size) := by
        have hRNe : r.size ≠ 0 := by
          simpa [hRSize] using (Nat.succ_ne_zero k)
        have hStep :
            mleByFoldingExec v r =
              mleByFoldingExec (foldLayer v r[0]!) (r.extract 1 r.size) :=
          mleByFoldingExec_step v r hRNe
        simpa [mleByFolding] using hStep
      have hTailSize : (r.extract 1 r.size).size = k := by
        simp [hRSize]
      have hFoldSize : (foldLayer v r[0]!).size = 2 ^ k := by
        simpa [foldLayer_size, hVSize, Nat.pow_succ] using
          congrArg (fun t => t / 2) hVSize
      have hIH :
          mleByInnerProduct (foldLayer v r[0]!) (r.extract 1 r.size) =
            mleByFolding (foldLayer v r[0]!) (r.extract 1 r.size) :=
        mleByInnerProduct_eq_mleByFolding_of_size_aux
          k (r.extract 1 r.size) (foldLayer v r[0]!) hTailSize hFoldSize
      calc
        mleByInnerProduct v r
            = mleByInnerProduct (foldLayer v r[0]!) (r.extract 1 r.size) := hInnerStep
        _ = mleByFolding (foldLayer v r[0]!) (r.extract 1 r.size) := hIH
        _ = mleByFolding v r := by simpa [hFoldStep] using hFoldStep.symm

/--
Size-guarded folding/inner-product identity target.

Paper-faithful closure: executable folding equals the inner-product MLE form
for valid table sizes.
-/
theorem mleByInnerProduct_eq_mleByFolding_of_size
  {v r : Array F}
  (hSize : v.size = 2 ^ r.size) :
  mleByInnerProduct v r = mleByFolding v r := by
  exact mleByInnerProduct_eq_mleByFolding_of_size_aux r.size r v rfl hSize

private theorem f_lin_seed
    (accF accG fi gi eqi δ : F) :
    (accF + δ * accG) + (fi + δ * gi) * eqi =
      (accF + fi * eqi) + δ * (accG + gi * eqi) := by
  calc
    (accF + δ * accG) + (fi + δ * gi) * eqi
        = (accF + δ * accG) + (fi * eqi + (δ * gi) * eqi) := by
            rw [f_left_distrib]
    _ = (accF + δ * accG) + (fi * eqi + δ * (gi * eqi)) := by
            rw [f_mul_assoc]
    _ = accF + (δ * accG + (fi * eqi + δ * (gi * eqi))) := by
            rw [f_add_assoc]
    _ = accF + ((δ * accG + fi * eqi) + δ * (gi * eqi)) := by
            rw [f_add_assoc]
    _ = accF + ((fi * eqi + δ * accG) + δ * (gi * eqi)) := by
            rw [f_add_comm (δ * accG) (fi * eqi)]
    _ = accF + (fi * eqi + (δ * accG + δ * (gi * eqi))) := by
            have hInner :
                fi * eqi + δ * accG + δ * (gi * eqi) =
                  fi * eqi + (δ * accG + δ * (gi * eqi)) :=
              f_add_assoc (fi * eqi) (δ * accG) (δ * (gi * eqi))
            exact congrArg (fun t => accF + t) hInner
    _ = accF + (fi * eqi + δ * (accG + gi * eqi)) := by
            rw [f_right_distrib]
    _ = (accF + fi * eqi) + δ * (accG + gi * eqi) := by
            rw [← f_add_assoc]

private theorem fold_linear
    (δ : F) (f g r : Array F) (hfg : f.size = g.size) :
    ∀ (l : List Nat) (accF accG : F),
      (∀ i, i ∈ l → i < f.size) →
      l.foldl (fun acc i => acc + (linComb δ f g hfg)[i]! * eqPoly (bitsToFieldArray r.size i) r)
        (accF + δ * accG)
      =
      l.foldl (fun acc i => acc + f[i]! * eqPoly (bitsToFieldArray r.size i) r) accF
        +
      δ * l.foldl (fun acc i => acc + g[i]! * eqPoly (bitsToFieldArray r.size i) r) accG := by
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
      have hlin : (linComb δ f g hfg)[i]! = f[i]! + δ * g[i]! := by
        simp [linComb, hi, hgi]
      have hseed :
        (accF + δ * accG) + (linComb δ f g hfg)[i]! * eqPoly (bitsToFieldArray r.size i) r
          =
        (accF + f[i]! * eqPoly (bitsToFieldArray r.size i) r)
          +
        δ * (accG + g[i]! * eqPoly (bitsToFieldArray r.size i) r) := by
        rw [hlin]
        exact f_lin_seed accF accG f[i]! g[i]! (eqPoly (bitsToFieldArray r.size i) r) δ
      calc
        (i :: tl).foldl (fun acc j => acc + (linComb δ f g hfg)[j]! * eqPoly (bitsToFieldArray r.size j) r)
            (accF + δ * accG)
            = tl.foldl (fun acc j => acc + (linComb δ f g hfg)[j]! * eqPoly (bitsToFieldArray r.size j) r)
                ((accF + δ * accG) + (linComb δ f g hfg)[i]! * eqPoly (bitsToFieldArray r.size i) r) := by
                  rfl
        _ = tl.foldl (fun acc j => acc + (linComb δ f g hfg)[j]! * eqPoly (bitsToFieldArray r.size j) r)
                ((accF + f[i]! * eqPoly (bitsToFieldArray r.size i) r)
                  +
                 δ * (accG + g[i]! * eqPoly (bitsToFieldArray r.size i) r)) := by
                  rw [hseed]
        _ = tl.foldl (fun acc j => acc + f[j]! * eqPoly (bitsToFieldArray r.size j) r)
              (accF + f[i]! * eqPoly (bitsToFieldArray r.size i) r)
              +
            δ *
            tl.foldl (fun acc j => acc + g[j]! * eqPoly (bitsToFieldArray r.size j) r)
              (accG + g[i]! * eqPoly (bitsToFieldArray r.size i) r) := by
                exact ih _ _ hTail
        _ = (i :: tl).foldl (fun acc j => acc + f[j]! * eqPoly (bitsToFieldArray r.size j) r) accF
              +
            δ *
            (i :: tl).foldl (fun acc j => acc + g[j]! * eqPoly (bitsToFieldArray r.size j) r) accG := by
              rfl

/-- Canonical closure of the inner-product linearity package. -/
theorem mleInnerProductLinearityAssumption_holds :
  mleInnerProductLinearityAssumption := by
  intro δ f g r hfg
  unfold mleInnerProductForm
  have hIn : ∀ i, i ∈ List.range f.size → i < f.size := by
    intro i hi
    exact List.mem_range.mp hi
  have hFold := fold_linear δ f g r hfg (List.range f.size) 0 0 hIn
  simpa [hfg] using hFold

/--
Theorem-facing linearity boundary on guarded `mleEval`.
-/
def mleEvalLinearityAssumption : Prop :=
  ∀ (δ : F) (f g r : Array F) (hfg : f.size = g.size),
    f.size = 2 ^ r.size →
    mleEval (linComb δ f g hfg) r =
      mleEval f r + δ * mleEval g r

theorem mleEval_linComb_of_assumption
  (hLin : mleEvalLinearityAssumption)
  (δ : F) (f g r : Array F) (hfg : f.size = g.size) (hSize : f.size = 2 ^ r.size) :
  mleEval (linComb δ f g hfg) r =
    mleEval f r + δ * mleEval g r := by
  exact hLin δ f g r hfg hSize

/--
Derived guarded linearity from:
1) executable-vs-inner-product identity (`mleIdentityAssumption`), and
2) inner-product-form linearity boundary (`mleInnerProductLinearityAssumption`).
-/
theorem mleEval_linComb_of_assumptions
  (hMLE : mleIdentityAssumption)
  (hInnerLin : mleInnerProductLinearityAssumption)
  (δ : F) (f g r : Array F) (hfg : f.size = g.size) (hSize : f.size = 2 ^ r.size) :
  mleEval (linComb δ f g hfg) r =
    mleEval f r + δ * mleEval g r := by
  have hSizeLC : (linComb δ f g hfg).size = 2 ^ r.size := by
    simpa [linComb_size] using hSize
  have hSizeG : g.size = 2 ^ r.size := by
    calc
      g.size = f.size := by simpa using hfg.symm
      _ = 2 ^ r.size := hSize
  have hLC :
      mleEval (linComb δ f g hfg) r =
        mleInnerProductForm (linComb δ f g hfg) r :=
    hMLE (linComb δ f g hfg) r hSizeLC
  have hF : mleEval f r = mleInnerProductForm f r :=
    hMLE f r hSize
  have hG : mleEval g r = mleInnerProductForm g r :=
    hMLE g r hSizeG
  calc
    mleEval (linComb δ f g hfg) r
        = mleInnerProductForm (linComb δ f g hfg) r := hLC
    _ = mleInnerProductForm f r + δ * mleInnerProductForm g r :=
      hInnerLin δ f g r hfg
    _ = mleEval f r + δ * mleEval g r := by
      rw [← hF, ← hG]

/-- Package-level builder for guarded MLE linearity from the two core boundary surfaces. -/
theorem mleEvalLinearityAssumption_of_assumptions
  (hMLE : mleIdentityAssumption)
  (hInnerLin : mleInnerProductLinearityAssumption) :
  mleEvalLinearityAssumption := by
  intro δ f g r hfg hSize
  exact mleEval_linComb_of_assumptions hMLE hInnerLin δ f g r hfg hSize

/-- Canonical closure of guarded MLE linearity from closed identity and inner-linearity. -/
theorem mleEvalLinearityAssumption_holds :
  mleEvalLinearityAssumption :=
  mleEvalLinearityAssumption_of_assumptions
    mleIdentityAssumption_holds
    mleInnerProductLinearityAssumption_holds

end SuperNeo
