import SuperNeo.Field
import SuperNeo.Dimensions

/-! Ring R_q operations, coefficient specs, and quotient-spec uniqueness. -/


namespace SuperNeo

open F

def D : Nat := 54

theorem D_eq_d : D = d := rfl
theorem D_pos : 0 < D := by decide

abbrev Coeffs := Array F

private def addAt (arr : Array F) (idx : Nat) (delta : F) : Array F :=
  arr.set! idx (arr[idx]! + delta)

private def subAt (arr : Array F) (idx : Nat) (delta : F) : Array F :=
  arr.set! idx (arr[idx]! - delta)

@[simp] theorem addAt_size (arr : Array F) (idx : Nat) (delta : F) :
  (addAt arr idx delta).size = arr.size := by
  unfold addAt
  simp

@[simp] theorem subAt_size (arr : Array F) (idx : Nat) (delta : F) :
  (subAt arr idx delta).size = arr.size := by
  unfold subAt
  simp

theorem setAt_allCanonical
  (arr : Array F) (idx : Nat) (val : F)
  (hVal : F.Canonical val)
  (hArr : arr.all F.Canonical = true) :
  (arr.set! idx val).all F.Canonical = true := by
  apply (Array.all_eq_true).2
  intro j hj
  have hjArr : j < arr.size := by
    simpa [Array.set!_eq_setIfInBounds] using hj
  by_cases hidx : idx = j
  · subst hidx
    have hji : idx < (arr.set! idx val).size := by
      simpa using hj
    have hSet :
        (arr.set! idx val)[idx] = val := by
      simpa [Array.set!_eq_setIfInBounds] using
        (Array.getElem_setIfInBounds_self (xs := arr) (i := idx) (a := val) hji)
    rw [hSet]
    exact decide_eq_true hVal
  · have hSet :
      (arr.set! idx val)[j] = arr[j] := by
      simpa [Array.set!_eq_setIfInBounds] using
        (Array.getElem_setIfInBounds_ne (xs := arr) (i := idx) (a := val) (j := j) hjArr hidx)
    have hArrjDec : decide (F.Canonical arr[j]) = true := (Array.all_eq_true.mp hArr) j hjArr
    have hArrj : F.Canonical arr[j] := decide_eq_true_eq.mp hArrjDec
    rw [hSet]
    exact decide_eq_true hArrj

theorem addAt_allCanonical
  (arr : Array F) (idx : Nat) (delta : F)
  (hArr : arr.all F.Canonical = true) :
  (addAt arr idx delta).all F.Canonical = true := by
  unfold addAt
  exact setAt_allCanonical arr idx (arr[idx]! + delta) (F.canonical_add _ _) hArr

theorem subAt_allCanonical
  (arr : Array F) (idx : Nat) (delta : F)
  (hArr : arr.all F.Canonical = true) :
  (subAt arr idx delta).all F.Canonical = true := by
  unfold subAt
  exact setAt_allCanonical arr idx (arr[idx]! - delta) (F.canonical_sub _ _) hArr

theorem replicate_zero_allCanonical (n : Nat) :
  (Array.replicate n (0 : F)).all F.Canonical = true := by
  apply (Array.all_eq_true).2
  intro i hi
  have hZero : (Array.replicate n (0 : F))[i] = (0 : F) := by
    simp [Array.getElem_replicate]
  rw [hZero]
  exact decide_eq_true F.canonical_zero

/-- Constant-term extraction ct : R_q -> F_q. -/
def ct (a : Coeffs) : F :=
  if a.isEmpty then
    0
  else
    a[0]!

theorem ct_of_isEmpty {a : Coeffs} (h : a.isEmpty = true) : ct a = 0 := by
  simp [ct, h]

theorem ct_of_not_isEmpty {a : Coeffs} (h : a.isEmpty = false) : ct a = a[0]! := by
  simp [ct, h]

theorem ct_canonical_of_all
  {a : Coeffs}
  (hAll : a.all F.Canonical = true) :
  F.Canonical (ct a) := by
  unfold ct
  by_cases hEmpty : a.isEmpty = true
  · simpa [hEmpty] using F.canonical_zero
  · have hGet : F.Canonical (a[0]!) := by
      exact F.canonical_getElem!_of_all (arr := a) (hArr := by simpa using hAll) 0
    simpa [hEmpty] using hGet

private def schoolbookRaw (a b : Coeffs) : Array F :=
  Id.run do
    -- Explicitly normalize representatives to degree < D (pad/truncate).
    -- This matches the effective loop behavior but makes the contract explicit.
    let aD : Coeffs := Array.ofFn (fun i : Fin D => a[i.1]!)
    let bD : Coeffs := Array.ofFn (fun i : Fin D => b[i.1]!)
    let mut tmp := Array.replicate (2 * D - 1) (0 : F)
    for i in [0:D] do
      let ai := aD[i]!
      for j in [0:D] do
        tmp := addAt tmp (i + j) (ai * bD[j]!)
    return tmp

theorem schoolbookRaw_size (a b : Coeffs) :
  (schoolbookRaw a b).size = 2 * D - 1 := by
  unfold schoolbookRaw
  let aD : Coeffs := Array.ofFn (fun i : Fin D => a[i.1]!)
  let bD : Coeffs := Array.ofFn (fun i : Fin D => b[i.1]!)
  have hInner :
      ∀ (i : Nat) (inner : List Nat) (tmp : Array F),
        (List.foldl (fun acc j => addAt acc (i + j) (aD[i]! * bD[j]!)) tmp inner).size =
          tmp.size := by
    intro i inner tmp
    induction inner generalizing tmp with
    | nil =>
        simp
    | cons j js ih =>
        simp [List.foldl_cons, ih, addAt_size]
  have hOuter :
      ∀ (outer : List Nat) (tmp : Array F),
        (List.foldl
            (fun acc i =>
              List.foldl (fun acc' j => addAt acc' (i + j) (aD[i]! * bD[j]!)) acc (List.range' 0 D))
            tmp
            outer).size =
          tmp.size := by
    intro outer tmp
    induction outer generalizing tmp with
    | nil =>
        simp
    | cons i is ih =>
        simp [List.foldl_cons, hInner i (List.range' 0 D), ih]
  simpa [aD, bD] using hOuter (List.range' 0 D) (Array.replicate (2 * D - 1) (0 : F))

private def takeFirstD (arr : Array F) : Coeffs :=
  Array.ofFn (fun i : Fin D => arr[i.1]!)

theorem takeFirstD_size (arr : Array F) : (takeFirstD arr).size = D := by
  unfold takeFirstD
  simp

theorem takeFirstD_allCanonical
  (arr : Array F)
  (hAll : arr.all F.Canonical = true) :
  (takeFirstD arr).all F.Canonical = true := by
  apply (Array.all_eq_true).2
  intro i hi
  have hCanon : F.Canonical (arr[i]!) := by
    exact F.canonical_getElem!_of_all (arr := arr) (hArr := by simpa using hAll) i
  simpa [takeFirstD] using decide_eq_true hCanon

theorem takeFirstD_not_isEmpty (arr : Array F) : (takeFirstD arr).isEmpty = false := by
  simp [takeFirstD, D]

theorem ct_takeFirstD (arr : Array F) : ct (takeFirstD arr) = arr[0]! := by
  simp [ct, takeFirstD, D]

theorem takeFirstD_getElem!
  (arr : Array F) (k : Nat) (hk : k < D) :
  (takeFirstD arr)[k]! = arr[k]! := by
  simp [takeFirstD, hk]

theorem ct_takeFirstD_canonical
  (arr : Array F)
  (hAll : arr.all F.Canonical = true) :
  F.Canonical (ct (takeFirstD arr)) := by
  exact ct_canonical_of_all (takeFirstD_allCanonical arr hAll)

/-- Explicit coefficient accessor: out-of-bounds coefficients are zero. -/
def getCoeff (a : Coeffs) (i : Nat) : F :=
  if h : i < a.size then a[i] else 0

theorem getCoeff_eq_getElem
  {a : Coeffs} {i : Nat} (hi : i < a.size) :
  getCoeff a i = a[i] := by
  simp [getCoeff, hi]

theorem getCoeff_eq_zero_of_ge
  {a : Coeffs} {i : Nat} (hi : a.size ≤ i) :
  getCoeff a i = 0 := by
  have hNotLt : ¬ i < a.size := Nat.not_lt_of_ge hi
  simp [getCoeff, hNotLt]

theorem getCoeff_takeFirstD_eq_getElem!
  (arr : Array F) (k : Nat) (hk : k < D) :
  getCoeff (takeFirstD arr) k = arr[k]! := by
  have hSize : (takeFirstD arr).size = D := takeFirstD_size arr
  have hk' : k < (takeFirstD arr).size := by simpa [hSize] using hk
  calc
    getCoeff (takeFirstD arr) k = (takeFirstD arr)[k] := getCoeff_eq_getElem hk'
    _ = (takeFirstD arr)[k]! := by simp [hk']
    _ = arr[k]! := takeFirstD_getElem! arr k hk

theorem getElemBang_eq_getCoeff (a : Coeffs) (i : Nat) :
  a[i]! = getCoeff a i := by
  by_cases hi : i < a.size
  · simp [getCoeff, hi]
  · have hNotLt : ¬ i < a.size := hi
    simp [getCoeff, hNotLt, F.default_eq_zero]

/-- Reduce modulo Phi_81(X)=X^54 + X^27 + 1, matching Rust logic. -/
private def reducePhi81Coeff (coeffsIn : Array F) (k : Nat) : F :=
  let base := coeffsIn[k]!
  let with81 :=
    if k ≤ 25 then
      base + coeffsIn[k + 81]!
    else
      base
  let with54 :=
    if k ≤ 26 then
      with81 - coeffsIn[k + 54]!
    else
      with81
  if 27 ≤ k then
    with54 - coeffsIn[k + 27]!
  else
    with54

/--
Closed-form reduction for `Phi_81(X) = X^54 + X^27 + 1` over degrees `0..106`.
This is equivalent to the imperative Rust-style reduction pass but easier to reason with.
-/
private def reducePhi81 (coeffsIn : Array F) : Coeffs :=
  Array.ofFn (fun i : Fin D => reducePhi81Coeff coeffsIn i.1)

theorem reducePhi81_size
  (coeffsIn : Array F) :
  (reducePhi81 coeffsIn).size = D := by
  unfold reducePhi81
  simp

theorem reducePhi81_getElem!
  (coeffsIn : Array F) (k : Nat) (hk : k < D) :
  (reducePhi81 coeffsIn)[k]! = reducePhi81Coeff coeffsIn k := by
  simp [reducePhi81, hk]

theorem reducePhi81Coeff_of_le25
  {coeffsIn : Array F} {k : Nat}
  (h25 : k ≤ 25) :
  reducePhi81Coeff coeffsIn k =
    coeffsIn[k]! + coeffsIn[k + 81]! - coeffsIn[k + 54]! := by
  have h26 : k ≤ 26 := Nat.le_trans h25 (by decide)
  have hNot27 : ¬ 27 ≤ k := by
    have hklt27 : k < 27 := Nat.lt_of_le_of_lt h25 (by decide)
    exact Nat.not_le_of_lt hklt27
  simp [reducePhi81Coeff, h25, h26, hNot27]

theorem reducePhi81Coeff_of_eq26
  {coeffsIn : Array F} {k : Nat}
  (h26 : k = 26) :
  reducePhi81Coeff coeffsIn k =
    coeffsIn[k]! - coeffsIn[k + 54]! := by
  subst h26
  simp [reducePhi81Coeff]

theorem reducePhi81Coeff_of_ge27
  {coeffsIn : Array F} {k : Nat}
  (h27 : 27 ≤ k) :
  reducePhi81Coeff coeffsIn k =
    coeffsIn[k]! - coeffsIn[k + 27]! := by
  have hNot25 : ¬ k ≤ 25 := by
    have hlt : 25 < k := Nat.lt_of_lt_of_le (by decide) h27
    exact Nat.not_le_of_lt hlt
  have hNot26 : ¬ k ≤ 26 := by
    have hlt : 26 < k := Nat.lt_of_lt_of_le (by decide) h27
    exact Nat.not_le_of_lt hlt
  simp [reducePhi81Coeff, hNot25, hNot26, h27]

theorem reducePhi81Coeff_formula
  {coeffsIn : Array F} {k : Nat} (hk : k < D) :
  reducePhi81Coeff coeffsIn k =
    if _ : k ≤ 25 then
      coeffsIn[k]! + coeffsIn[k + 81]! - coeffsIn[k + 54]!
    else if _ : k = 26 then
      coeffsIn[k]! - coeffsIn[k + 54]!
    else
      coeffsIn[k]! - coeffsIn[k + 27]! := by
  by_cases h25 : k ≤ 25
  · simp [h25, reducePhi81Coeff_of_le25 (coeffsIn := coeffsIn) h25]
  · by_cases h26 : k = 26
    · subst h26
      simp [h25, reducePhi81Coeff]
    · have h26le : 26 ≤ k := Nat.succ_le_of_lt (Nat.lt_of_not_ge h25)
      have h26lt : 26 < k := Nat.lt_of_le_of_ne h26le (Ne.symm h26)
      have h27 : 27 ≤ k := Nat.succ_le_of_lt h26lt
      simp [h25, h26, reducePhi81Coeff_of_ge27 (coeffsIn := coeffsIn) h27]

theorem reducePhi81Coeff_canonical
  {coeffsIn : Array F} {k : Nat} :
  F.Canonical (reducePhi81Coeff coeffsIn k) := by
  by_cases h25 : k ≤ 25
  · rw [reducePhi81Coeff_of_le25 (coeffsIn := coeffsIn) h25]
    exact F.canonical_sub _ _
  · by_cases h26 : k = 26
    · rw [reducePhi81Coeff_of_eq26 (coeffsIn := coeffsIn) h26]
      exact F.canonical_sub _ _
    · have h26le : 26 ≤ k := Nat.succ_le_of_lt (Nat.lt_of_not_ge h25)
      have h26lt : 26 < k := Nat.lt_of_le_of_ne h26le (Ne.symm h26)
      have h27 : 27 ≤ k := Nat.succ_le_of_lt h26lt
      rw [reducePhi81Coeff_of_ge27 (coeffsIn := coeffsIn) h27]
      exact F.canonical_sub _ _

theorem reducePhi81_allCanonical
  (coeffsIn : Array F) :
  (reducePhi81 coeffsIn).all F.Canonical = true := by
  apply (Array.all_eq_true).2
  intro i hi
  have hGet :
      (reducePhi81 coeffsIn)[i] = reducePhi81Coeff coeffsIn i := by
    simpa [reducePhi81] using
      (Array.getElem_ofFn
        (f := fun j : Fin D => reducePhi81Coeff coeffsIn j.1)
        (i := i) hi)
  rw [hGet]
  exact decide_eq_true (reducePhi81Coeff_canonical (coeffsIn := coeffsIn) (k := i))

/-- Ring multiplication in R_q = F_q[X]/(X^54 + X^27 + 1). -/
def mulRq (a b : Coeffs) : Coeffs :=
  reducePhi81 (schoolbookRaw a b)

theorem mulRq_allCanonical (a b : Coeffs) :
  (mulRq a b).all F.Canonical = true := by
  unfold mulRq
  exact reducePhi81_allCanonical (schoolbookRaw a b)

theorem mulRq_coeff_canonical (a b : Coeffs) (k : Nat) :
  F.Canonical ((mulRq a b)[k]!) := by
  exact F.canonical_getElem!_of_all
    (arr := mulRq a b)
    (hArr := by simpa using mulRq_allCanonical a b)
    k

theorem mulRq_coeff_of_le25
  {a b : Coeffs} {k : Nat}
  (hk : k < D)
  (h25 : k ≤ 25) :
  (mulRq a b)[k]! =
    (schoolbookRaw a b)[k]! + (schoolbookRaw a b)[k + 81]! - (schoolbookRaw a b)[k + 54]! := by
  unfold mulRq
  rw [reducePhi81_getElem! (coeffsIn := schoolbookRaw a b) (k := k) hk]
  exact reducePhi81Coeff_of_le25 (coeffsIn := schoolbookRaw a b) h25

theorem mulRq_coeff_of_eq26
  {a b : Coeffs} {k : Nat}
  (hk : k < D)
  (h26 : k = 26) :
  (mulRq a b)[k]! =
    (schoolbookRaw a b)[k]! - (schoolbookRaw a b)[k + 54]! := by
  unfold mulRq
  rw [reducePhi81_getElem! (coeffsIn := schoolbookRaw a b) (k := k) hk]
  exact reducePhi81Coeff_of_eq26 (coeffsIn := schoolbookRaw a b) h26

theorem mulRq_coeff_of_ge27
  {a b : Coeffs} {k : Nat}
  (hk : k < D)
  (h27 : 27 ≤ k) :
  (mulRq a b)[k]! =
    (schoolbookRaw a b)[k]! - (schoolbookRaw a b)[k + 27]! := by
  unfold mulRq
  rw [reducePhi81_getElem! (coeffsIn := schoolbookRaw a b) (k := k) hk]
  exact reducePhi81Coeff_of_ge27 (coeffsIn := schoolbookRaw a b) h27

theorem mulRq_coeff_formula
  {a b : Coeffs} {k : Nat} (hk : k < D) :
  (mulRq a b)[k]! =
    if _ : k ≤ 25 then
      (schoolbookRaw a b)[k]! + (schoolbookRaw a b)[k + 81]! - (schoolbookRaw a b)[k + 54]!
    else if _ : k = 26 then
      (schoolbookRaw a b)[k]! - (schoolbookRaw a b)[k + 54]!
    else
      (schoolbookRaw a b)[k]! - (schoolbookRaw a b)[k + 27]! := by
  unfold mulRq
  rw [reducePhi81_getElem! (coeffsIn := schoolbookRaw a b) (k := k) hk]
  exact reducePhi81Coeff_formula (coeffsIn := schoolbookRaw a b) hk

def mulRqCoeffSpec (a b : Coeffs) (k : Nat) : F :=
  let raw := schoolbookRaw a b
  if k ≤ 25 then
    raw[k]! + raw[k + 81]! - raw[k + 54]!
  else if k = 26 then
    raw[k]! - raw[k + 54]!
  else
    raw[k]! - raw[k + 27]!

/-- Public array view of the unreduced schoolbook coefficients. -/
def mulRqRawCoeffs (a b : Coeffs) : Coeffs :=
  schoolbookRaw a b

theorem mulRqRawCoeffs_size (a b : Coeffs) :
  (mulRqRawCoeffs a b).size = 2 * D - 1 := by
  unfold mulRqRawCoeffs
  exact schoolbookRaw_size a b

/-- Public accessor for coefficients of the unreduced schoolbook product. -/
def mulRqRawCoeffSpec (a b : Coeffs) (k : Nat) : F :=
  (mulRqRawCoeffs a b)[k]!

theorem mulRqRawCoeffSpec_eq_rawCoeffs_getElemBang
  (a b : Coeffs) (k : Nat) :
  mulRqRawCoeffSpec a b k = (mulRqRawCoeffs a b)[k]! := by
  rfl

theorem mulRqRawCoeffSpec_eq_zero_of_ge
  {a b : Coeffs} {k : Nat}
  (hk : 2 * D - 1 ≤ k) :
  mulRqRawCoeffSpec a b k = 0 := by
  unfold mulRqRawCoeffSpec
  have hNotLt : ¬ k < (mulRqRawCoeffs a b).size :=
    Nat.not_lt_of_ge (by simpa [mulRqRawCoeffs_size] using hk)
  simp [hNotLt, F.default_eq_zero]

theorem mulRqCoeffSpec_of_le25
  {a b : Coeffs} {k : Nat}
  (h25 : k ≤ 25) :
  mulRqCoeffSpec a b k =
    (schoolbookRaw a b)[k]! + (schoolbookRaw a b)[k + 81]! - (schoolbookRaw a b)[k + 54]! := by
  unfold mulRqCoeffSpec
  simp [h25]

theorem mulRqCoeffSpec_of_le25_raw
  {a b : Coeffs} {k : Nat}
  (h25 : k ≤ 25) :
  mulRqCoeffSpec a b k =
    mulRqRawCoeffSpec a b k + mulRqRawCoeffSpec a b (k + 81) - mulRqRawCoeffSpec a b (k + 54) := by
  rw [mulRqCoeffSpec_of_le25 (a := a) (b := b) (k := k) h25]
  simp [mulRqRawCoeffSpec, mulRqRawCoeffs]

theorem mulRqCoeffSpec_of_eq26
  {a b : Coeffs} {k : Nat}
  (h26 : k = 26) :
  mulRqCoeffSpec a b k =
    (schoolbookRaw a b)[k]! - (schoolbookRaw a b)[k + 54]! := by
  unfold mulRqCoeffSpec
  simp [h26]

theorem mulRqCoeffSpec_of_eq26_raw
  {a b : Coeffs} {k : Nat}
  (h26 : k = 26) :
  mulRqCoeffSpec a b k =
    mulRqRawCoeffSpec a b k - mulRqRawCoeffSpec a b (k + 54) := by
  rw [mulRqCoeffSpec_of_eq26 (a := a) (b := b) (k := k) h26]
  simp [mulRqRawCoeffSpec, mulRqRawCoeffs]

theorem mulRqCoeffSpec_of_ge27
  {a b : Coeffs} {k : Nat}
  (h27 : 27 ≤ k) :
  mulRqCoeffSpec a b k =
    (schoolbookRaw a b)[k]! - (schoolbookRaw a b)[k + 27]! := by
  unfold mulRqCoeffSpec
  have h25 : ¬ k ≤ 25 := by
    have hk25 : 25 < k := Nat.lt_of_lt_of_le (by decide : 25 < 27) h27
    exact Nat.not_le_of_gt hk25
  have h26 : ¬ k = 26 := by
    intro hEq
    subst hEq
    exact (Nat.not_succ_le_self 26) h27
  simp [h25, h26]

theorem mulRqCoeffSpec_of_ge27_raw
  {a b : Coeffs} {k : Nat}
  (h27 : 27 ≤ k) :
  mulRqCoeffSpec a b k =
    mulRqRawCoeffSpec a b k - mulRqRawCoeffSpec a b (k + 27) := by
  rw [mulRqCoeffSpec_of_ge27 (a := a) (b := b) (k := k) h27]
  simp [mulRqRawCoeffSpec, mulRqRawCoeffs]

theorem reducePhi81Coeff_eq_mulRqCoeffSpec
  {a b : Coeffs} {k : Nat}
  (hk : k < D) :
  reducePhi81Coeff (schoolbookRaw a b) k = mulRqCoeffSpec a b k := by
  simpa [mulRqCoeffSpec] using
    (reducePhi81Coeff_formula (coeffsIn := schoolbookRaw a b) (k := k) hk)

theorem mulRq_coeff_spec
  {a b : Coeffs} {k : Nat} (hk : k < D) :
  (mulRq a b)[k]! = mulRqCoeffSpec a b k := by
  unfold mulRqCoeffSpec
  simpa using mulRq_coeff_formula (a := a) (b := b) (k := k) hk

theorem mulRqCoeffSpec_canonical
  (a b : Coeffs) (k : Nat) :
  F.Canonical (mulRqCoeffSpec a b k) := by
  by_cases h25 : k ≤ 25
  · rw [mulRqCoeffSpec_of_le25 (a := a) (b := b) (k := k) h25]
    exact F.canonical_sub _ _
  · by_cases h26 : k = 26
    · rw [mulRqCoeffSpec_of_eq26 (a := a) (b := b) (k := k) h26]
      exact F.canonical_sub _ _
    · have h26le : 26 ≤ k := Nat.succ_le_of_lt (Nat.lt_of_not_ge h25)
      have h26lt : 26 < k := Nat.lt_of_le_of_ne h26le (Ne.symm h26)
      have h27 : 27 ≤ k := Nat.succ_le_of_lt h26lt
      rw [mulRqCoeffSpec_of_ge27 (a := a) (b := b) (k := k) h27]
      exact F.canonical_sub _ _

theorem mulRq_eq_ofFn_coeffSpec
  (a b : Coeffs) :
  mulRq a b = Array.ofFn (fun i : Fin D => mulRqCoeffSpec a b i.1) := by
  unfold mulRq reducePhi81
  apply Array.ext
  · simp
  · intro i hi1 hi2
    have hk : i < D := by simpa using hi1
    have hCoeff : reducePhi81Coeff (schoolbookRaw a b) i = mulRqCoeffSpec a b i := by
      unfold mulRqCoeffSpec
      simpa using (reducePhi81Coeff_formula (coeffsIn := schoolbookRaw a b) (k := i) hk)
    simpa [Array.getElem_ofFn] using hCoeff

theorem mulRq_ct_formula
  (a b : Coeffs) :
  ct (mulRq a b) = mulRqCoeffSpec a b 0 := by
  have hNotEmpty : (mulRq a b).isEmpty = false := by
    by_cases hE : (mulRq a b).isEmpty = true
    · have hEq : (mulRq a b) = #[] := Array.isEmpty_iff.mp hE
      have hSz0 : (mulRq a b).size = 0 := by simpa [hEq]
      have hSzD : (mulRq a b).size = D := by
        simpa [mulRq] using (reducePhi81_size (schoolbookRaw a b))
      have hD0 : D = 0 := by simpa [hSzD] using hSz0
      exact False.elim ((Nat.ne_of_gt D_pos) hD0)
    · simp [hE]
  calc
    ct (mulRq a b) = (mulRq a b)[0]! := ct_of_not_isEmpty hNotEmpty
    _ = mulRqCoeffSpec a b 0 := mulRq_coeff_spec (a := a) (b := b) (k := 0) D_pos

theorem mulRq_ct_formula_explicit
  (a b : Coeffs) :
  ct (mulRq a b) = (schoolbookRaw a b)[0]! + (schoolbookRaw a b)[81]! - (schoolbookRaw a b)[54]! := by
  simpa [mulRqCoeffSpec] using mulRq_ct_formula a b

theorem mulRq_ct_formula_explicit_canonical
  (a b : Coeffs) :
  F.Canonical ((schoolbookRaw a b)[0]! + (schoolbookRaw a b)[81]! - (schoolbookRaw a b)[54]!) := by
  have hSpec : F.Canonical (mulRqCoeffSpec a b 0) := mulRqCoeffSpec_canonical a b 0
  simpa [mulRqCoeffSpec] using hSpec

def hasRingDegreeShape (a : Coeffs) : Prop := a.size = D

def ringMulShapeProp (a b : Coeffs) : Prop :=
  hasRingDegreeShape a ∧ hasRingDegreeShape b ∧ (mulRq a b).size = D

instance hasRingDegreeShape_decidable (a : Coeffs) : Decidable (hasRingDegreeShape a) := by
  unfold hasRingDegreeShape
  infer_instance

instance ringMulShapeProp_decidable (a b : Coeffs) : Decidable (ringMulShapeProp a b) := by
  unfold ringMulShapeProp
  infer_instance

def ringMulShapeCheck (a b : Coeffs) : Bool :=
  decide (ringMulShapeProp a b)

theorem ringMulShapeCheck_sound
  {a b : Coeffs}
  (hOk : ringMulShapeCheck a b = true) :
  ringMulShapeProp a b := by
  unfold ringMulShapeCheck at hOk
  exact decide_eq_true_eq.mp hOk

theorem ringMulShapeCheck_complete
  {a b : Coeffs}
  (hProp : ringMulShapeProp a b) :
  ringMulShapeCheck a b = true := by
  unfold ringMulShapeCheck
  exact decide_eq_true hProp

theorem mulRq_size (a b : Coeffs) : (mulRq a b).size = D := by
  unfold mulRq
  exact reducePhi81_size (schoolbookRaw a b)

theorem reducePhi81Coeff_eq_self_of_shape_allCanonical
  {a : Coeffs} {k : Nat}
  (hSize : a.size = D)
  (hAll : a.all F.Canonical = true)
  (_hk : k < D) :
  reducePhi81Coeff a k = a[k]! := by
  have hCanonK : F.Canonical (a[k]!) := by
    exact F.canonical_getElem!_of_all (arr := a) (hArr := by simpa using hAll) k
  by_cases h25 : k ≤ 25
  · rw [reducePhi81Coeff_of_le25 (coeffsIn := a) h25]
    have hOut81 : ¬ k + 81 < a.size := by
      have hk81 : D ≤ k + 81 := by
        have h81le : (81 : Nat) ≤ k + 81 := by
          simpa [Nat.add_comm] using (Nat.le_add_left 81 k)
        calc
          D = 54 := rfl
          _ ≤ 81 := by decide
          _ ≤ k + 81 := h81le
      exact Nat.not_lt_of_ge (by simpa [hSize] using hk81)
    have hOut54 : ¬ k + 54 < a.size := by
      have hk54 : D ≤ k + 54 := by
        simpa [D, Nat.add_comm] using (Nat.le_add_left 54 k)
      exact Nat.not_lt_of_ge (by simpa [hSize] using hk54)
    have h81zero : a[k + 81]! = (0 : F) := by
      calc
        a[k + 81]! = getCoeff a (k + 81) := getElemBang_eq_getCoeff a (k + 81)
        _ = 0 := by
          exact getCoeff_eq_zero_of_ge (a := a) (i := k + 81)
            (Nat.le_of_not_gt hOut81)
    have h54zero : a[k + 54]! = (0 : F) := by
      calc
        a[k + 54]! = getCoeff a (k + 54) := getElemBang_eq_getCoeff a (k + 54)
        _ = 0 := by
          exact getCoeff_eq_zero_of_ge (a := a) (i := k + 54)
            (Nat.le_of_not_gt hOut54)
    rw [h81zero, h54zero]
    have hAdd0 : a[k]! + (0 : F) = a[k]! := by
      simpa using (F.add_zero_of_canonical hCanonK)
    have hSub0 : a[k]! - (0 : F) = a[k]! := by
      simpa using (F.sub_zero_of_canonical hCanonK)
    calc
      a[k]! + 0 - 0 = a[k]! - 0 := by
        rw [hAdd0]
      _ = a[k]! := hSub0
  · by_cases h26 : k = 26
    · rw [reducePhi81Coeff_of_eq26 (coeffsIn := a) h26]
      have hOut54 : ¬ k + 54 < a.size := by
        have hk54 : D ≤ k + 54 := by
          simpa [D, Nat.add_comm] using (Nat.le_add_left 54 k)
        exact Nat.not_lt_of_ge (by simpa [hSize] using hk54)
      have h54zero : a[k + 54]! = (0 : F) := by
        calc
          a[k + 54]! = getCoeff a (k + 54) := getElemBang_eq_getCoeff a (k + 54)
          _ = 0 := by
            exact getCoeff_eq_zero_of_ge (a := a) (i := k + 54)
              (Nat.le_of_not_gt hOut54)
      rw [h54zero]
      simpa using (F.sub_zero_of_canonical hCanonK)
    · have h26le : 26 ≤ k := Nat.succ_le_of_lt (Nat.lt_of_not_ge h25)
      have h26lt : 26 < k := Nat.lt_of_le_of_ne h26le (Ne.symm h26)
      have h27 : 27 ≤ k := Nat.succ_le_of_lt h26lt
      rw [reducePhi81Coeff_of_ge27 (coeffsIn := a) h27]
      have hk27 : D ≤ k + 27 := by
        have h54le : (27 + 27 : Nat) ≤ k + 27 := Nat.add_le_add_right h27 27
        simpa [D, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using h54le
      have hOut27 : ¬ k + 27 < a.size := Nat.not_lt_of_ge (by simpa [hSize] using hk27)
      have h27zero : a[k + 27]! = (0 : F) := by
        calc
          a[k + 27]! = getCoeff a (k + 27) := getElemBang_eq_getCoeff a (k + 27)
          _ = 0 := by
            exact getCoeff_eq_zero_of_ge (a := a) (i := k + 27)
              (Nat.le_of_not_gt hOut27)
      rw [h27zero]
      simpa using (F.sub_zero_of_canonical hCanonK)

theorem reducePhi81_idempotent_of_shape_allCanonical
  {a : Coeffs}
  (hSize : a.size = D)
  (hAll : a.all F.Canonical = true) :
  reducePhi81 a = a := by
  apply Array.ext
  · calc
      (reducePhi81 a).size = D := reducePhi81_size a
      _ = a.size := hSize.symm
  · intro i hiR hiA
    have hk : i < D := by simpa [reducePhi81_size] using hiR
    have hCoeff : (reducePhi81 a)[i]! = a[i]! := by
      rw [reducePhi81_getElem! a i hk]
      exact reducePhi81Coeff_eq_self_of_shape_allCanonical hSize hAll hk
    simpa [hiR, hiA] using hCoeff

theorem reducePhi81_mulRq_idempotent (a b : Coeffs) :
  reducePhi81 (mulRq a b) = mulRq a b := by
  exact reducePhi81_idempotent_of_shape_allCanonical
    (hSize := mulRq_size a b)
    (hAll := mulRq_allCanonical a b)

/--
Quotient-level normal-form spec for multiplication in
`R_q = F_q[X]/(X^54 + X^27 + 1)`.
`c` is a valid product representative iff it has degree shape `D`
and each coefficient matches the closed-form reduction spec.
-/
def mulRqQuotientSpec (a b c : Coeffs) : Prop :=
  c.size = D ∧ ∀ k, k < D → c[k]! = mulRqCoeffSpec a b k

theorem mulRq_satisfies_quotientSpec (a b : Coeffs) :
  mulRqQuotientSpec a b (mulRq a b) := by
  refine ⟨mulRq_size a b, ?_⟩
  intro k hk
  exact mulRq_coeff_spec (a := a) (b := b) (k := k) hk

theorem mulRqQuotientSpec_of_eq
  {a b c : Coeffs}
  (hEq : c = mulRq a b) :
  mulRqQuotientSpec a b c := by
  simpa [hEq] using mulRq_satisfies_quotientSpec a b

theorem mulRq_eq_of_quotientSpec
  {a b c : Coeffs}
  (hSpec : mulRqQuotientSpec a b c) :
  c = mulRq a b := by
  apply Array.ext
  · calc
      c.size = D := hSpec.1
      _ = (mulRq a b).size := (mulRq_size a b).symm
  · intro i hiC hiM
    have hk : i < D := by simpa [hSpec.1] using hiC
    have hCoeff : c[i]! = (mulRq a b)[i]! := by
      have hc : c[i]! = mulRqCoeffSpec a b i := hSpec.2 i hk
      have hm : (mulRq a b)[i]! = mulRqCoeffSpec a b i := mulRq_coeff_spec hk
      exact Eq.trans hc hm.symm
    simpa [hiC, hiM] using hCoeff

theorem mulRq_quotientSpec_iff
  (a b c : Coeffs) :
  mulRqQuotientSpec a b c ↔ c = mulRq a b :=
  ⟨mulRq_eq_of_quotientSpec, mulRqQuotientSpec_of_eq⟩

/--
Shaped quotient multiplication spec:
input representatives are explicitly constrained to ring-degree shape.
-/
def mulRqQuotientSpecShaped (a b c : Coeffs) : Prop :=
  hasRingDegreeShape a ∧ hasRingDegreeShape b ∧ mulRqQuotientSpec a b c

theorem mulRq_satisfies_quotientSpecShaped
  {a b : Coeffs}
  (ha : hasRingDegreeShape a)
  (hb : hasRingDegreeShape b) :
  mulRqQuotientSpecShaped a b (mulRq a b) := by
  exact ⟨ha, hb, mulRq_satisfies_quotientSpec a b⟩

theorem mulRqQuotientSpecShaped_of_eq
  {a b c : Coeffs}
  (ha : hasRingDegreeShape a)
  (hb : hasRingDegreeShape b)
  (hEq : c = mulRq a b) :
  mulRqQuotientSpecShaped a b c := by
  exact ⟨ha, hb, mulRqQuotientSpec_of_eq hEq⟩

theorem mulRq_eq_of_quotientSpecShaped
  {a b c : Coeffs}
  (hSpec : mulRqQuotientSpecShaped a b c) :
  c = mulRq a b := by
  exact mulRq_eq_of_quotientSpec hSpec.2.2

theorem mulRq_quotientSpecShaped_iff
  {a b c : Coeffs}
  (ha : hasRingDegreeShape a)
  (hb : hasRingDegreeShape b) :
  mulRqQuotientSpecShaped a b c ↔ c = mulRq a b := by
  constructor
  · intro hSpec
    exact mulRq_eq_of_quotientSpecShaped hSpec
  · intro hEq
    exact mulRqQuotientSpecShaped_of_eq ha hb hEq

theorem mulRqQuotientSpecShaped_iff_shapes_and_eq
  {a b c : Coeffs} :
  mulRqQuotientSpecShaped a b c ↔
    hasRingDegreeShape a ∧ hasRingDegreeShape b ∧ c = mulRq a b := by
  constructor
  · intro hSpec
    exact ⟨hSpec.1, hSpec.2.1, mulRq_eq_of_quotientSpecShaped hSpec⟩
  · intro h
    rcases h with ⟨ha, hb, hEq⟩
    exact mulRqQuotientSpecShaped_of_eq ha hb hEq

theorem mulRqQuotientSpecShaped_unique
  {a b c₁ c₂ : Coeffs}
  (h₁ : mulRqQuotientSpecShaped a b c₁)
  (h₂ : mulRqQuotientSpecShaped a b c₂) :
  c₁ = c₂ := by
  calc
    c₁ = mulRq a b := mulRq_eq_of_quotientSpecShaped h₁
    _ = c₂ := (mulRq_eq_of_quotientSpecShaped h₂).symm

theorem mulRqQuotientSpec_iff_hasRingDegreeShape
  (a b c : Coeffs) :
  mulRqQuotientSpec a b c ↔
    hasRingDegreeShape c ∧ ∀ k, k < D → c[k]! = mulRqCoeffSpec a b k := by
  constructor
  · intro h
    exact ⟨h.1, h.2⟩
  · intro h
    exact ⟨h.1, h.2⟩

theorem mulRqQuotientSpec_iff_hasRingDegreeShape_and_fin_coeffSpec
  (a b c : Coeffs) :
  mulRqQuotientSpec a b c ↔
    hasRingDegreeShape c ∧ ∀ i : Fin D, c[i.1]! = mulRqCoeffSpec a b i.1 := by
  constructor
  · intro h
    refine ⟨h.1, ?_⟩
    intro i
    exact h.2 i.1 i.2
  · intro h
    refine ⟨h.1, ?_⟩
    intro k hk
    exact h.2 ⟨k, hk⟩

theorem mulRqQuotientSpec_unique
  {a b c₁ c₂ : Coeffs}
  (h₁ : mulRqQuotientSpec a b c₁)
  (h₂ : mulRqQuotientSpec a b c₂) :
  c₁ = c₂ := by
  calc
    c₁ = mulRq a b := mulRq_eq_of_quotientSpec h₁
    _ = c₂ := (mulRq_eq_of_quotientSpec h₂).symm

theorem mulRqQuotientSpec_coeff_eq
  {a b c₁ c₂ : Coeffs} {k : Nat}
  (h₁ : mulRqQuotientSpec a b c₁)
  (h₂ : mulRqQuotientSpec a b c₂)
  (hk : k < D) :
  c₁[k]! = c₂[k]! := by
  calc
    c₁[k]! = mulRqCoeffSpec a b k := h₁.2 k hk
    _ = c₂[k]! := (h₂.2 k hk).symm

theorem mulRqQuotientSpec_getElem_eq_mulRq_getElem
  {a b c : Coeffs}
  (hSpec : mulRqQuotientSpec a b c)
  (i : Fin D) :
  c[i.1]'(by simpa [hSpec.1] using i.2) =
    (mulRq a b)[i.1]'(by simpa [mulRq_size a b] using i.2) := by
  have hBang : c[i.1]! = (mulRq a b)[i.1]! := by
    calc
      c[i.1]! = mulRqCoeffSpec a b i.1 := hSpec.2 i.1 i.2
      _ = (mulRq a b)[i.1]! :=
        (mulRq_coeff_spec (a := a) (b := b) (k := i.1) i.2).symm
  simpa [hSpec.1, mulRq_size a b, i.2] using hBang

theorem mulRqQuotientSpec_getElem_eq_pair
  {a b c₁ c₂ : Coeffs}
  (h₁ : mulRqQuotientSpec a b c₁)
  (h₂ : mulRqQuotientSpec a b c₂)
  (i : Fin D) :
  c₁[i.1]'(by simpa [h₁.1] using i.2) =
    c₂[i.1]'(by simpa [h₂.1] using i.2) := by
  have hBang : c₁[i.1]! = c₂[i.1]! :=
    mulRqQuotientSpec_coeff_eq (h₁ := h₁) (h₂ := h₂) (hk := i.2)
  simpa [h₁.1, h₂.1, i.2] using hBang

theorem mulRqQuotientSpec_iff_size_and_mulRq_coeff
  (a b c : Coeffs) :
  mulRqQuotientSpec a b c ↔
    c.size = D ∧ ∀ k, k < D → c[k]! = (mulRq a b)[k]! := by
  constructor
  · intro hSpec
    refine ⟨hSpec.1, ?_⟩
    intro k hk
    calc
      c[k]! = mulRqCoeffSpec a b k := hSpec.2 k hk
      _ = (mulRq a b)[k]! := (mulRq_coeff_spec (a := a) (b := b) (k := k) hk).symm
  · intro h
    refine ⟨h.1, ?_⟩
    intro k hk
    calc
      c[k]! = (mulRq a b)[k]! := h.2 k hk
      _ = mulRqCoeffSpec a b k := mulRq_coeff_spec (a := a) (b := b) (k := k) hk

theorem mulRqQuotientSpec_iff_size_and_fin_mulRq_coeff
  (a b c : Coeffs) :
  mulRqQuotientSpec a b c ↔
    c.size = D ∧ ∀ i : Fin D, c[i.1]! = (mulRq a b)[i.1]! := by
  constructor
  · intro h
    refine ⟨h.1, ?_⟩
    intro i
    calc
      c[i.1]! = mulRqCoeffSpec a b i.1 := h.2 i.1 i.2
      _ = (mulRq a b)[i.1]! :=
        (mulRq_coeff_spec (a := a) (b := b) (k := i.1) i.2).symm
  · intro h
    refine ⟨h.1, ?_⟩
    intro k hk
    calc
      c[k]! = (mulRq a b)[k]! := h.2 ⟨k, hk⟩
      _ = mulRqCoeffSpec a b k := mulRq_coeff_spec (a := a) (b := b) (k := k) hk

theorem mulRq_eq_of_size_and_mulRq_coeff
  {a b c : Coeffs}
  (hSize : c.size = D)
  (hCoeff : ∀ k, k < D → c[k]! = (mulRq a b)[k]!) :
  c = mulRq a b := by
  have hSpec : mulRqQuotientSpec a b c :=
    (mulRqQuotientSpec_iff_size_and_mulRq_coeff a b c).2 ⟨hSize, hCoeff⟩
  exact mulRq_eq_of_quotientSpec hSpec

theorem mulRq_eq_of_size_and_fin_mulRq_coeff
  {a b c : Coeffs}
  (hSize : c.size = D)
  (hCoeff : ∀ i : Fin D, c[i.1]! = (mulRq a b)[i.1]!) :
  c = mulRq a b := by
  exact mulRq_eq_of_size_and_mulRq_coeff hSize
    (fun k hk => hCoeff ⟨k, hk⟩)

theorem mulRq_eq_of_hasRingDegreeShape_and_coeffSpec
  {a b c : Coeffs}
  (hShape : hasRingDegreeShape c)
  (hCoeff : ∀ k, k < D → c[k]! = mulRqCoeffSpec a b k) :
  c = mulRq a b := by
  exact mulRq_eq_of_quotientSpec ⟨hShape, hCoeff⟩

theorem mulRq_eq_of_hasRingDegreeShape_and_fin_coeffSpec
  {a b c : Coeffs}
  (hShape : hasRingDegreeShape c)
  (hCoeff : ∀ i : Fin D, c[i.1]! = mulRqCoeffSpec a b i.1) :
  c = mulRq a b := by
  exact mulRq_eq_of_hasRingDegreeShape_and_coeffSpec hShape
    (fun k hk => hCoeff ⟨k, hk⟩)

theorem mulRqQuotientSpec_coeff_zero
  {a b c : Coeffs}
  (hSpec : mulRqQuotientSpec a b c) :
  c[0]! = mulRqCoeffSpec a b 0 := by
  exact hSpec.2 0 D_pos

theorem mulRqQuotientSpec_allCanonical
  {a b c : Coeffs}
  (hSpec : mulRqQuotientSpec a b c) :
  c.all F.Canonical = true := by
  rcases hSpec with ⟨hSize, hCoeff⟩
  apply (Array.all_eq_true).2
  intro i hi
  have hiD : i < D := by simpa [hSize] using hi
  have hciBang : c[i]! = mulRqCoeffSpec a b i := hCoeff i hiD
  have hci : c[i] = mulRqCoeffSpec a b i := by
    simpa [hi] using hciBang
  rw [hci]
  exact decide_eq_true (mulRqCoeffSpec_canonical a b i)

theorem mulRqQuotientSpec_hasRingDegreeShape
  {a b c : Coeffs}
  (hSpec : mulRqQuotientSpec a b c) :
  hasRingDegreeShape c := by
  exact hSpec.1

theorem reducePhi81_idempotent_of_mulRqQuotientSpec
  {a b c : Coeffs}
  (hSpec : mulRqQuotientSpec a b c) :
  reducePhi81 c = c := by
  exact reducePhi81_idempotent_of_shape_allCanonical
    (hSize := hSpec.1)
    (hAll := mulRqQuotientSpec_allCanonical hSpec)

theorem reducePhi81_eq_mulRq_of_mulRqQuotientSpec
  {a b c : Coeffs}
  (hSpec : mulRqQuotientSpec a b c) :
  reducePhi81 c = mulRq a b := by
  calc
    reducePhi81 c = c := reducePhi81_idempotent_of_mulRqQuotientSpec hSpec
    _ = mulRq a b := mulRq_eq_of_quotientSpec hSpec

private theorem coeffs_not_isEmpty_of_sizeD
  {c : Coeffs}
  (hSize : c.size = D) :
  c.isEmpty = false := by
  by_cases hE : c.isEmpty = true
  · have hEq : c = #[] := Array.isEmpty_iff.mp hE
    have hSz0 : c.size = 0 := by simpa [hEq]
    have hD0 : D = 0 := by simpa [hSize] using hSz0
    exact False.elim ((Nat.ne_of_gt D_pos) hD0)
  · simp [hE]

theorem ct_eq_of_mulRqQuotientSpec
  {a b c : Coeffs}
  (hSpec : mulRqQuotientSpec a b c) :
  ct c = mulRqCoeffSpec a b 0 := by
  have hSize : c.size = D := hSpec.1
  have hNotEmpty : c.isEmpty = false := coeffs_not_isEmpty_of_sizeD hSize
  calc
    ct c = c[0]! := ct_of_not_isEmpty hNotEmpty
    _ = mulRqCoeffSpec a b 0 := mulRqQuotientSpec_coeff_zero hSpec

theorem ct_canonical_of_mulRqQuotientSpec
  {a b c : Coeffs}
  (hSpec : mulRqQuotientSpec a b c) :
  F.Canonical (ct c) := by
  exact ct_canonical_of_all (mulRqQuotientSpec_allCanonical hSpec)

theorem reducePhi81_coeff_of_mulRqQuotientSpec
  {a b c : Coeffs} {k : Nat}
  (hSpec : mulRqQuotientSpec a b c)
  (hk : k < D) :
  (reducePhi81 c)[k]! = mulRqCoeffSpec a b k := by
  rw [reducePhi81_getElem! c k hk]
  have hSize : c.size = D := hSpec.1
  have hAll : c.all F.Canonical = true := mulRqQuotientSpec_allCanonical hSpec
  have hSelf : reducePhi81Coeff c k = c[k]! :=
    reducePhi81Coeff_eq_self_of_shape_allCanonical hSize hAll hk
  rw [hSelf]
  exact hSpec.2 k hk

theorem mulRqQuotientSpec_coeff_eq_mulRq_coeff
  {a b c : Coeffs} {k : Nat}
  (hSpec : mulRqQuotientSpec a b c)
  (hk : k < D) :
  c[k]! = (mulRq a b)[k]! := by
  calc
    c[k]! = mulRqCoeffSpec a b k := hSpec.2 k hk
    _ = (mulRq a b)[k]! := (mulRq_coeff_spec (a := a) (b := b) (k := k) hk).symm

theorem ct_reducePhi81_eq_of_mulRqQuotientSpec
  {a b c : Coeffs}
  (hSpec : mulRqQuotientSpec a b c) :
  ct (reducePhi81 c) = mulRqCoeffSpec a b 0 := by
  have hNotEmpty : (reducePhi81 c).isEmpty = false := by
    have hSize : (reducePhi81 c).size = D := reducePhi81_size c
    exact coeffs_not_isEmpty_of_sizeD hSize
  calc
    ct (reducePhi81 c) = (reducePhi81 c)[0]! := ct_of_not_isEmpty hNotEmpty
    _ = mulRqCoeffSpec a b 0 :=
      reducePhi81_coeff_of_mulRqQuotientSpec (hSpec := hSpec) (hk := D_pos)

theorem ct_eq_ct_mulRq_of_mulRqQuotientSpec
  {a b c : Coeffs}
  (hSpec : mulRqQuotientSpec a b c) :
  ct c = ct (mulRq a b) := by
  calc
    ct c = mulRqCoeffSpec a b 0 := ct_eq_of_mulRqQuotientSpec hSpec
    _ = ct (mulRq a b) := (mulRq_ct_formula a b).symm

theorem ct_eq_mulRq_ct_formula_explicit_of_mulRqQuotientSpec
  {a b c : Coeffs}
  (hSpec : mulRqQuotientSpec a b c) :
  ct c = (schoolbookRaw a b)[0]! + (schoolbookRaw a b)[81]! - (schoolbookRaw a b)[54]! := by
  calc
    ct c = ct (mulRq a b) := ct_eq_ct_mulRq_of_mulRqQuotientSpec hSpec
    _ = (schoolbookRaw a b)[0]! + (schoolbookRaw a b)[81]! - (schoolbookRaw a b)[54]! :=
      mulRq_ct_formula_explicit a b

theorem mulRqQuotientSpec_not_isEmpty
  {a b c : Coeffs}
  (hSpec : mulRqQuotientSpec a b c) :
  c.isEmpty = false := by
  exact coeffs_not_isEmpty_of_sizeD hSpec.1

theorem ct_eq_of_mulRqQuotientSpec_pair
  {a b c₁ c₂ : Coeffs}
  (h₁ : mulRqQuotientSpec a b c₁)
  (h₂ : mulRqQuotientSpec a b c₂) :
  ct c₁ = ct c₂ := by
  calc
    ct c₁ = mulRqCoeffSpec a b 0 := ct_eq_of_mulRqQuotientSpec h₁
    _ = ct c₂ := (ct_eq_of_mulRqQuotientSpec h₂).symm

theorem reducePhi81_eq_of_mulRqQuotientSpec_pair
  {a b c₁ c₂ : Coeffs}
  (h₁ : mulRqQuotientSpec a b c₁)
  (h₂ : mulRqQuotientSpec a b c₂) :
  reducePhi81 c₁ = reducePhi81 c₂ := by
  rw [reducePhi81_idempotent_of_mulRqQuotientSpec h₁]
  rw [reducePhi81_idempotent_of_mulRqQuotientSpec h₂]
  exact mulRqQuotientSpec_unique h₁ h₂

theorem mulRq_not_isEmpty (a b : Coeffs) : (mulRq a b).isEmpty = false := by
  by_cases hE : (mulRq a b).isEmpty = true
  · have hEq : (mulRq a b) = #[] := Array.isEmpty_iff.mp hE
    have hSz0 : (mulRq a b).size = 0 := by simpa [hEq]
    have hSzD : (mulRq a b).size = D := mulRq_size a b
    have hD0 : D = 0 := by simpa [hSzD] using hSz0
    exact False.elim ((Nat.ne_of_gt D_pos) hD0)
  · simp [hE]

theorem ct_mulRq_canonical (a b : Coeffs) :
  F.Canonical (ct (mulRq a b)) := by
  exact ct_canonical_of_all (mulRq_allCanonical a b)

theorem hasRingDegreeShape_mulRq (a b : Coeffs) : hasRingDegreeShape (mulRq a b) := by
  unfold hasRingDegreeShape
  exact mulRq_size a b

theorem ringMulShape_of_shapes
  {a b : Coeffs}
  (ha : hasRingDegreeShape a)
  (hb : hasRingDegreeShape b) :
  ringMulShapeProp a b := by
  exact ⟨ha, hb, mulRq_size a b⟩

theorem ringMulShapeCheck_true_of_shapes
  {a b : Coeffs}
  (ha : hasRingDegreeShape a)
  (hb : hasRingDegreeShape b) :
  ringMulShapeCheck a b = true := by
  unfold ringMulShapeCheck
  exact decide_eq_true (ringMulShape_of_shapes ha hb)

/-- Canonical ring-element wrapper (`size = D`) for theorem-native interfaces. -/
structure Rq where
  coeffs : Coeffs
  shape : hasRingDegreeShape coeffs

/-- Canonical ring-element wrapper (`size = D` and canonical coefficients). -/
structure RqCanon where
  coeffs : Coeffs
  shape : hasRingDegreeShape coeffs
  canon : coeffs.all F.Canonical = true

def truncateToRing (a : Coeffs) : Rq :=
  { coeffs := takeFirstD a
    shape := takeFirstD_size a }

def truncateToRingCanon (a : Coeffs) (hAll : a.all F.Canonical = true) : RqCanon :=
  { coeffs := takeFirstD a
    shape := takeFirstD_size a
    canon := takeFirstD_allCanonical a hAll }

def Rq.ct (a : Rq) : F := SuperNeo.ct a.coeffs

def RqCanon.ct (a : RqCanon) : F := SuperNeo.ct a.coeffs

def Rq.mul (a b : Rq) : Rq :=
  { coeffs := mulRq a.coeffs b.coeffs
    shape := hasRingDegreeShape_mulRq a.coeffs b.coeffs }

def RqCanon.mul (a b : RqCanon) : RqCanon :=
  { coeffs := mulRq a.coeffs b.coeffs
    shape := hasRingDegreeShape_mulRq a.coeffs b.coeffs
    canon := mulRq_allCanonical a.coeffs b.coeffs }

def Rq.mulQuotientSpec (a b c : Rq) : Prop :=
  mulRqQuotientSpec a.coeffs b.coeffs c.coeffs

def Rq.mulQuotientSpecShaped (a b c : Rq) : Prop :=
  mulRqQuotientSpecShaped a.coeffs b.coeffs c.coeffs

theorem Rq.mul_satisfies_quotientSpec (a b : Rq) :
  Rq.mulQuotientSpec a b (Rq.mul a b) := by
  exact mulRq_satisfies_quotientSpec a.coeffs b.coeffs

theorem Rq.mul_satisfies_quotientSpecShaped (a b : Rq) :
  Rq.mulQuotientSpecShaped a b (Rq.mul a b) := by
  exact mulRq_satisfies_quotientSpecShaped a.shape b.shape

@[ext] theorem Rq.ext {a b : Rq} (hCoeffs : a.coeffs = b.coeffs) : a = b := by
  cases a
  cases b
  cases hCoeffs
  rfl

theorem Rq.mul_eq_of_quotientSpec
  {a b c : Rq}
  (hSpec : Rq.mulQuotientSpec a b c) :
  c = Rq.mul a b := by
  apply Rq.ext
  exact mulRq_eq_of_quotientSpec hSpec

theorem Rq.mul_eq_of_quotientSpecShaped
  {a b c : Rq}
  (hSpec : Rq.mulQuotientSpecShaped a b c) :
  c = Rq.mul a b := by
  apply Rq.ext
  exact mulRq_eq_of_quotientSpecShaped hSpec

theorem Rq.mulQuotientSpec_iff
  (a b c : Rq) :
  Rq.mulQuotientSpec a b c ↔ c = Rq.mul a b := by
  constructor
  · exact Rq.mul_eq_of_quotientSpec
  · intro hEq
    simpa [Rq.mulQuotientSpec, hEq] using (Rq.mul_satisfies_quotientSpec a b)

theorem Rq.mulQuotientSpecShaped_iff
  (a b c : Rq) :
  Rq.mulQuotientSpecShaped a b c ↔ c = Rq.mul a b := by
  constructor
  · exact Rq.mul_eq_of_quotientSpecShaped
  · intro hEq
    simpa [Rq.mulQuotientSpecShaped, hEq] using (Rq.mul_satisfies_quotientSpecShaped a b)

/-- Dot product over F_q^d. -/
def dot (a b : Coeffs) : F :=
  Id.run do
    let mut acc : F := 0
    for i in [0:D] do
      acc := acc + a[i]! * b[i]!
    return acc

/-- Apply SuperNeo bar transform for one D-sized block. -/
def superneoBarBlock (bar : Array (Array F)) (v : Coeffs) : Coeffs :=
  Id.run do
    let mut out := Array.replicate D (0 : F)
    for row in [0:D] do
      let mut acc : F := 0
      let barRow := bar[row]!
      for col in [0:D] do
        acc := acc + barRow[col]! * v[col]!
      out := out.set! row acc
    return out

theorem superneoBarBlock_size (bar : Array (Array F)) (v : Coeffs) :
  (superneoBarBlock bar v).size = D := by
  unfold superneoBarBlock
  have hOuter :
      ∀ (rows : List Nat) (out : Array F),
        (List.foldl
            (fun acc row =>
              acc.setIfInBounds row
                (List.foldl (fun acc' col => acc' + bar[row]![col]! * v[col]!) 0 (List.range' 0 D)))
            out
            rows).size =
          out.size := by
    intro rows out
    induction rows generalizing out with
    | nil =>
        simp
    | cons row rows ih =>
        simp [List.foldl_cons, ih]
  simpa using hOuter (List.range' 0 D) (Array.replicate D (0 : F))

theorem hasRingDegreeShape_superneoBarBlock (bar : Array (Array F)) (v : Coeffs) :
  hasRingDegreeShape (superneoBarBlock bar v) := by
  unfold hasRingDegreeShape
  exact superneoBarBlock_size bar v

end SuperNeo
