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

private theorem addAt_getElemBang_eq_of_ne
  (arr : Array F) (idx k : Nat) (delta : F)
  (hNe : idx ≠ k) :
  (addAt arr idx delta)[k]! = arr[k]! := by
  unfold addAt
  by_cases hk : k < arr.size
  · have hkSet : k < (arr.set! idx (arr[idx]! + delta)).size := by
      simpa [Array.set!_eq_setIfInBounds] using hk
    have hSet :
      (arr.set! idx (arr[idx]! + delta))[k]'hkSet = arr[k] := by
      simpa [Array.set!_eq_setIfInBounds] using
        (Array.getElem_setIfInBounds_ne
          (xs := arr) (i := idx) (a := arr[idx]! + delta) (j := k) hk hNe)
    simpa [hk, hkSet] using hSet
  · simp [hk, F.default_eq_zero]

private theorem addAt_getElemBang_eq_of_lt
  (arr : Array F) (idx : Nat) (delta : F)
  (hidx : idx < arr.size) :
  (addAt arr idx delta)[idx]! = arr[idx]! + delta := by
  unfold addAt
  have hidxSet : idx < (arr.set! idx (arr[idx]! + delta)).size := by
    simpa [Array.set!_eq_setIfInBounds] using hidx
  have hSet :
      (arr.set! idx (arr[idx]! + delta))[idx]'hidxSet = arr[idx]! + delta := by
    simpa [Array.set!_eq_setIfInBounds] using
      (Array.getElem_setIfInBounds_self
        (xs := arr) (i := idx) (a := arr[idx]! + delta) hidx)
  simpa [hidx, hidxSet] using hSet

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

theorem getCoeff_addAt_eq_of_ne
  {arr : Array F} {idx k : Nat} {delta : F}
  (hNe : idx ≠ k) :
  getCoeff (addAt arr idx delta) k = getCoeff arr k := by
  calc
    getCoeff (addAt arr idx delta) k = (addAt arr idx delta)[k]! := by
      symm
      exact getElemBang_eq_getCoeff (addAt arr idx delta) k
    _ = arr[k]! := addAt_getElemBang_eq_of_ne arr idx k delta hNe
    _ = getCoeff arr k := getElemBang_eq_getCoeff arr k

theorem getCoeff_addAt_eq_of_lt
  {arr : Array F} {idx : Nat} {delta : F}
  (hidx : idx < arr.size) :
  getCoeff (addAt arr idx delta) idx = getCoeff arr idx + delta := by
  calc
    getCoeff (addAt arr idx delta) idx = (addAt arr idx delta)[idx]! := by
      symm
      exact getElemBang_eq_getCoeff (addAt arr idx delta) idx
    _ = arr[idx]! + delta := addAt_getElemBang_eq_of_lt arr idx delta hidx
    _ = getCoeff arr idx + delta := by
      simp [getElemBang_eq_getCoeff]

theorem getCoeff_addAt_eq_ite_of_lt
  {arr : Array F} {idx k : Nat} {delta : F}
  (hidx : idx < arr.size) :
  getCoeff (addAt arr idx delta) k =
    if idx = k then getCoeff arr k + delta else getCoeff arr k := by
  by_cases hEq : idx = k
  · subst hEq
    simp [getCoeff_addAt_eq_of_lt, hidx]
  · simp [getCoeff_addAt_eq_of_ne, hEq]

theorem schoolbookRaw_index_lt
  {i j : Nat}
  (hi : i < D)
  (hj : j < D) :
  i + j < 2 * D - 1 := by
  have hi54 : i < 54 := by simpa [D] using hi
  have hj54 : j < 54 := by simpa [D] using hj
  have hi53 : i ≤ 53 := Nat.le_of_lt_succ (by simpa using hi54)
  have hsum : i + j ≤ 53 + j := Nat.add_le_add_right hi53 j
  have hlt : 53 + j < 53 + 54 := Nat.add_lt_add_left hj54 53
  have hfinal : i + j < 107 := Nat.lt_of_le_of_lt hsum (by simpa using hlt)
  simpa [D] using hfinal

/--
Characterize exactly when row `i` has a schoolbook contribution to output index `k`.
-/
theorem schoolbook_row_hits_index_iff
  {i k : Nat} :
  (∃ j, j < D ∧ i + j = k) ↔ i ≤ k ∧ k < i + D := by
  constructor
  · intro h
    rcases h with ⟨j, hj, hEq⟩
    constructor
    · calc
        i ≤ i + j := Nat.le_add_right i j
        _ = k := hEq
    · calc
        k = i + j := hEq.symm
        _ < i + D := Nat.add_lt_add_left hj i
  · intro h
    rcases h with ⟨hik, hkD⟩
    refine ⟨k - i, ?_, ?_⟩
    · have hLt : i + (k - i) < i + D := by
        simpa [Nat.add_sub_of_le hik] using hkD
      exact Nat.lt_of_add_lt_add_left hLt
    · exact Nat.add_sub_of_le hik

theorem schoolbook_row_hit_unique
  {i k j1 j2 : Nat}
  (h1 : i + j1 = k)
  (h2 : i + j2 = k) :
  j1 = j2 := by
  exact Nat.add_left_cancel (h1.trans h2.symm)

/--
Row `i` misses output index `k` exactly when `k` is outside `[i, i + D)`.
-/
theorem schoolbook_row_misses_index_iff
  {i k : Nat} :
  (∀ j, j < D → i + j ≠ k) ↔ (k < i ∨ i + D ≤ k) := by
  constructor
  · intro hMiss
    by_cases hki : k < i
    · exact Or.inl hki
    · have hik : i ≤ k := Nat.le_of_not_gt hki
      by_cases hkD : k < i + D
      · exfalso
        have hHit : ∃ j, j < D ∧ i + j = k :=
          (schoolbook_row_hits_index_iff (i := i) (k := k)).2 ⟨hik, hkD⟩
        rcases hHit with ⟨j, hj, hEq⟩
        exact (hMiss j hj hEq).elim
      · exact Or.inr (Nat.le_of_not_gt hkD)
  · intro hOutside j hj
    intro hEq
    rcases hOutside with hkLt | hkGe
    · have hLe : i ≤ i + j := Nat.le_add_right i j
      have hLt : i + j < i := by simpa [hEq] using hkLt
      exact (Nat.not_lt_of_ge hLe) hLt
    · have hLt : i + j < i + D := Nat.add_lt_add_left hj i
      have hGe : i + D ≤ i + j := by simpa [hEq] using hkGe
      exact (Nat.not_lt_of_ge hGe) hLt

theorem list_length_filter_le {α : Type} (p : α → Bool) (xs : List α) :
  (xs.filter p).length ≤ xs.length := by
  induction xs with
  | nil =>
      simp
  | cons x xs ih =>
      by_cases hp : p x
      · simp [List.filter, hp, ih, Nat.succ_le_succ_iff]
      · simp [List.filter, hp, Nat.le_trans ih (Nat.le_succ _)]

theorem list_foldl_eq_foldl_filter_of_step_id
  {α β : Type}
  (step : β → α → β)
  (p : α → Bool)
  (init : β)
  (xs : List α)
  (hId : ∀ acc x, p x = false → step acc x = acc) :
  List.foldl step init xs = List.foldl step init (xs.filter p) := by
  induction xs generalizing init with
  | nil =>
      simp
  | cons x xs ih =>
      by_cases hp : p x = true
      · simp [List.filter, hp, ih]
      · have hpFalse : p x = false := by
          cases hpx : p x with
          | false => rfl
          | true => exact (hp hpx).elim
        have hStep : step init x = init := hId init x hpFalse
        simp [List.filter, hpFalse, hStep, ih]

/--
For a fixed output index `k`, the number of candidate schoolbook positions
`i in [0, D)` that pass a contribution predicate is at most `D`.
-/
theorem schoolbook_candidate_count_le_D (k : Nat) :
  (List.filter (fun i => k - i < D) (List.range' 0 D)).length ≤ D := by
  have hFilter :
      (List.filter (fun i => k - i < D) (List.range' 0 D)).length ≤ (List.range' 0 D).length :=
    list_length_filter_le (p := fun i => k - i < D) (xs := List.range' 0 D)
  simpa using hFilter

theorem schoolbook_row_active_true_iff
  {i k : Nat} :
  decide (i ≤ k ∧ k < i + D) = true ↔ i ≤ k ∧ k < i + D := by
  simp

theorem schoolbook_row_active_false_of_outside
  {i k : Nat}
  (hOutside : k < i ∨ i + D ≤ k) :
  decide (i ≤ k ∧ k < i + D) = false := by
  apply decide_eq_false_iff_not.mpr
  intro hActive
  rcases hOutside with hkLt | hkGe
  · exact (Nat.not_lt_of_ge hActive.1) hkLt
  · exact (Nat.not_lt_of_ge hkGe) hActive.2

theorem schoolbook_active_row_count_le_D (k : Nat) :
  ((List.range' 0 D).filter (fun i => decide (i ≤ k ∧ k < i + D))).length ≤ D := by
  have hFilter :
      ((List.range' 0 D).filter (fun i => decide (i ≤ k ∧ k < i + D))).length
        ≤ (List.range' 0 D).length :=
    list_length_filter_le
      (p := fun i => decide (i ≤ k ∧ k < i + D))
      (xs := List.range' 0 D)
  simpa using hFilter

theorem schoolbook_active_rows_nil_of_ge
  {k : Nat}
  (hk : 2 * D - 1 ≤ k) :
  ((List.range' 0 D).filter (fun i => decide (i ≤ k ∧ k < i + D))) = [] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro i hiMem
  have hFilter := List.mem_filter.mp hiMem
  have hiRange : i ∈ List.range' 0 D := hFilter.1
  have hActive : i ≤ k ∧ k < i + D := by
    exact (schoolbook_row_active_true_iff (i := i) (k := k)).1 hFilter.2
  have hiLtD : i < D := by
    rcases (List.mem_range').1 hiRange with ⟨t, ht, hEq⟩
    have hEq' : i = t := by simpa using hEq
    simpa [hEq'] using ht
  have hi54 : i < 54 := by simpa [D] using hiLtD
  have hi53 : i ≤ 53 := Nat.le_of_lt_succ (by simpa using hi54)
  have hiPlusLe : i + D ≤ 2 * D - 1 := by
    have hLe : i + 54 ≤ 53 + 54 := Nat.add_le_add_right hi53 54
    simpa [D] using hLe
  have hGe : i + D ≤ k := Nat.le_trans hiPlusLe hk
  exact (Nat.not_lt_of_ge hGe) hActive.2

theorem getCoeff_addAt_schoolbook_index
  {arr : Array F} {i j : Nat} {delta : F}
  (hSize : arr.size = 2 * D - 1)
  (hi : i < D)
  (hj : j < D) :
  getCoeff (addAt arr (i + j) delta) (i + j) = getCoeff arr (i + j) + delta := by
  apply getCoeff_addAt_eq_of_lt
  simpa [hSize] using schoolbookRaw_index_lt (i := i) (j := j) hi hj

theorem getCoeff_addAt_schoolbook_index_eq_ite
  {arr : Array F} {i j k : Nat} {delta : F}
  (hSize : arr.size = 2 * D - 1)
  (hi : i < D)
  (hj : j < D) :
  getCoeff (addAt arr (i + j) delta) k =
    if i + j = k then getCoeff arr k + delta else getCoeff arr k := by
  apply getCoeff_addAt_eq_ite_of_lt
  simpa [hSize] using schoolbookRaw_index_lt (i := i) (j := j) hi hj

theorem getCoeff_addAt_schoolbook_index_of_ne
  {arr : Array F} {i j k : Nat} {delta : F}
  (hNe : i + j ≠ k) :
  getCoeff (addAt arr (i + j) delta) k = getCoeff arr k := by
  exact getCoeff_addAt_eq_of_ne (arr := arr) (idx := i + j) (k := k) (delta := delta) hNe

theorem getCoeff_replicate_zero (n k : Nat) :
  getCoeff (Array.replicate n (0 : F)) k = 0 := by
  by_cases hk : k < (Array.replicate n (0 : F)).size
  · simp [getCoeff, Array.getElem_replicate]
  · simp [getCoeff]

/--
For a fixed schoolbook row index `i`, folding all `j in [0, D)` only updates
output positions in the interval `[i, i + D)`.
Hence positions `k` outside that interval are unchanged.
-/
theorem getCoeff_schoolbook_row_fold_unchanged_of_outside
  {aD bD : Coeffs} {i k : Nat} {tmp : Array F}
  (hOutside : k < i ∨ i + D ≤ k) :
  getCoeff
      (List.foldl
        (fun acc j => addAt acc (i + j) (aD[i]! * bD[j]!))
        tmp
        (List.range' 0 D))
      k
    = getCoeff tmp k := by
  let step : Array F → Nat → Array F :=
    fun acc j => addAt acc (i + j) (aD[i]! * bD[j]!)
  have hStep :
      ∀ inner acc,
        (∀ j, j ∈ inner → j < D) →
        getCoeff (List.foldl step acc inner) k = getCoeff acc k := by
    intro inner
    induction inner with
    | nil =>
        intro acc _hIn
        simp [step]
    | cons j js ih =>
        intro acc hIn
        have hjD : j < D := hIn j (by simp)
        have hNe : i + j ≠ k := by
          intro hEq
          rcases hOutside with hkLt | hkGe
          · have hLe : i ≤ i + j := Nat.le_add_right i j
            have hLt : i + j < i := by simpa [hEq] using hkLt
            exact (Nat.not_lt_of_ge hLe) hLt
          · have hLt : i + j < i + D := Nat.add_lt_add_left hjD i
            have hGe : i + D ≤ i + j := by simpa [hEq] using hkGe
            exact (Nat.not_lt_of_ge hGe) hLt
        have hHead :
            getCoeff (step acc j) k = getCoeff acc k := by
          exact getCoeff_addAt_eq_of_ne
            (arr := acc) (idx := i + j) (k := k) (delta := aD[i]! * bD[j]!) hNe
        have hTail :
            getCoeff (List.foldl step (step acc j) js) k = getCoeff (step acc j) k := by
          apply ih
          intro j' hj'
          exact hIn j' (by simp [hj'])
        calc
          getCoeff (List.foldl step acc (j :: js)) k
              = getCoeff (List.foldl step (step acc j) js) k := by
                  simp [List.foldl_cons]
          _ = getCoeff (step acc j) k := hTail
          _ = getCoeff acc k := hHead
  have hRange : ∀ j, j ∈ List.range' 0 D → j < D := by
    intro j hj
    rcases (List.mem_range').1 hj with ⟨t, ht, hEq⟩
    have hEq' : j = t := by simpa using hEq
    simpa [hEq'] using ht
  simpa [step] using hStep (inner := List.range' 0 D) (acc := tmp) hRange

/--
For fixed row `i` and output index `k`, the schoolbook inner fold rewrites to a
scalar fold over contributions guarded by `i + j = k`.
-/
theorem getCoeff_schoolbook_row_fold_eq_scalar_fold
  {aD bD : Coeffs} {i k : Nat} {tmp : Array F}
  (hi : i < D)
  (hSize : tmp.size = 2 * D - 1) :
  getCoeff
      (List.foldl
        (fun acc j => addAt acc (i + j) (aD[i]! * bD[j]!))
        tmp
        (List.range' 0 D))
      k
    =
    List.foldl
      (fun acc j => if i + j = k then acc + aD[i]! * bD[j]! else acc)
      (getCoeff tmp k)
      (List.range' 0 D) := by
  let step : Array F → Nat → Array F :=
    fun acc j => addAt acc (i + j) (aD[i]! * bD[j]!)
  let scalarStep : F → Nat → F :=
    fun acc j => if i + j = k then acc + aD[i]! * bD[j]! else acc
  have hInner :
      ∀ inner acc,
        acc.size = 2 * D - 1 →
        (∀ j, j ∈ inner → j < D) →
        getCoeff (List.foldl step acc inner) k =
          List.foldl scalarStep (getCoeff acc k) inner := by
    intro inner
    induction inner with
    | nil =>
        intro acc _hSize _hIn
        simp [step, scalarStep]
    | cons j js ih =>
        intro acc hSizeAcc hIn
        have hjD : j < D := hIn j (by simp)
        have hHead :
            getCoeff (step acc j) k =
              scalarStep (getCoeff acc k) j := by
          have hIte :
              getCoeff (step acc j) k =
                if i + j = k then getCoeff acc k + aD[i]! * bD[j]! else getCoeff acc k := by
            simpa [step] using
              (getCoeff_addAt_schoolbook_index_eq_ite
                (arr := acc) (i := i) (j := j) (k := k) (delta := aD[i]! * bD[j]!)
                hSizeAcc hi hjD)
          simpa [scalarStep] using hIte
        have hSizeNext : (step acc j).size = 2 * D - 1 := by
          simpa [step, hSizeAcc] using
            (addAt_size (arr := acc) (idx := i + j) (delta := aD[i]! * bD[j]!))
        have hTail :
            getCoeff (List.foldl step (step acc j) js) k =
              List.foldl scalarStep (getCoeff (step acc j) k) js := by
          apply ih
          · exact hSizeNext
          · intro j' hj'
            exact hIn j' (by simp [hj'])
        calc
          getCoeff (List.foldl step acc (j :: js)) k
              = getCoeff (List.foldl step (step acc j) js) k := by
                  simp [List.foldl_cons]
          _ = List.foldl scalarStep (getCoeff (step acc j) k) js := hTail
          _ = List.foldl scalarStep (scalarStep (getCoeff acc k) j) js := by
                simp [hHead]
          _ = List.foldl scalarStep (getCoeff acc k) (j :: js) := by
                simp [List.foldl_cons]
  have hRange : ∀ j, j ∈ List.range' 0 D → j < D := by
    intro j hj
    rcases (List.mem_range').1 hj with ⟨t, ht, hEq⟩
    have hEq' : j = t := by simpa using hEq
    simpa [hEq'] using ht
  simpa [step, scalarStep] using
    hInner (inner := List.range' 0 D) (acc := tmp) hSize hRange

theorem schoolbook_row_fold_size
  {aD bD : Coeffs} {i : Nat} {tmp : Array F} :
  (List.foldl
      (fun acc j => addAt acc (i + j) (aD[i]! * bD[j]!))
      tmp
      (List.range' 0 D)).size = tmp.size := by
  have hInner :
      ∀ inner acc,
        (List.foldl
          (fun acc' j => addAt acc' (i + j) (aD[i]! * bD[j]!))
          acc
          inner).size = acc.size := by
    intro inner
    induction inner with
    | nil =>
        intro acc
        simp
    | cons j js ih =>
        intro acc
        simp [List.foldl_cons, ih, addAt_size]
  simpa using hInner (inner := List.range' 0 D) (acc := tmp)

/--
Scalarized form of the nested schoolbook folds for a fixed output index `k`.
This rewrites array mutation (`addAt`) into guarded scalar accumulation.
-/
theorem getCoeff_schoolbook_outer_fold_eq_scalar_fold
  {aD bD : Coeffs} {k : Nat} :
  getCoeff
      (List.foldl
        (fun acc i =>
          List.foldl (fun acc' j => addAt acc' (i + j) (aD[i]! * bD[j]!)) acc (List.range' 0 D))
        (Array.replicate (2 * D - 1) (0 : F))
        (List.range' 0 D))
      k
    =
    List.foldl
      (fun acc i =>
        List.foldl
          (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
          acc
          (List.range' 0 D))
      (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
      (List.range' 0 D) := by
  let outerStep : Array F → Nat → Array F :=
    fun acc i =>
      List.foldl (fun acc' j => addAt acc' (i + j) (aD[i]! * bD[j]!)) acc (List.range' 0 D)
  let outerScalar : F → Nat → F :=
    fun acc i =>
      List.foldl (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc') acc (List.range' 0 D)
  have hOuter :
      ∀ outer acc,
        acc.size = 2 * D - 1 →
        (∀ i, i ∈ outer → i < D) →
        getCoeff (List.foldl outerStep acc outer) k =
          List.foldl outerScalar (getCoeff acc k) outer := by
    intro outer
    induction outer with
    | nil =>
        intro acc _hSize _hIn
        simp [outerStep, outerScalar]
    | cons i is ih =>
        intro acc hSizeAcc hIn
        have hi : i < D := hIn i (by simp)
        have hHead :
            getCoeff (outerStep acc i) k = outerScalar (getCoeff acc k) i := by
          simpa [outerStep, outerScalar] using
            (getCoeff_schoolbook_row_fold_eq_scalar_fold
              (aD := aD) (bD := bD) (i := i) (k := k) (tmp := acc)
              hi hSizeAcc)
        have hSizeNext : (outerStep acc i).size = 2 * D - 1 := by
          calc
            (outerStep acc i).size = acc.size := by
              simp [outerStep, schoolbook_row_fold_size]
            _ = 2 * D - 1 := hSizeAcc
        have hTail :
            getCoeff (List.foldl outerStep (outerStep acc i) is) k =
              List.foldl outerScalar (getCoeff (outerStep acc i) k) is := by
          apply ih
          · exact hSizeNext
          · intro i' hi'
            exact hIn i' (by simp [hi'])
        calc
          getCoeff (List.foldl outerStep acc (i :: is)) k
              = getCoeff (List.foldl outerStep (outerStep acc i) is) k := by
                  simp [List.foldl_cons]
          _ = List.foldl outerScalar (getCoeff (outerStep acc i) k) is := hTail
          _ = List.foldl outerScalar (outerScalar (getCoeff acc k) i) is := by
                simp [hHead]
          _ = List.foldl outerScalar (getCoeff acc k) (i :: is) := by
                simp [List.foldl_cons]
  have hRange : ∀ i, i ∈ List.range' 0 D → i < D := by
    intro i hi
    rcases (List.mem_range').1 hi with ⟨t, ht, hEq⟩
    have hEq' : i = t := by simpa using hEq
    simpa [hEq'] using ht
  have hSizeInit : (Array.replicate (2 * D - 1) (0 : F)).size = 2 * D - 1 := by
    simp
  simpa [outerStep, outerScalar] using
    hOuter (outer := List.range' 0 D) (acc := Array.replicate (2 * D - 1) (0 : F)) hSizeInit hRange

theorem getCoeff_schoolbookRaw_eq_scalar_fold
  (a b : Coeffs) (k : Nat) :
  let aD : Coeffs := Array.ofFn (fun i : Fin D => a[i.1]!)
  let bD : Coeffs := Array.ofFn (fun i : Fin D => b[i.1]!)
  getCoeff (schoolbookRaw a b) k
    =
    List.foldl
      (fun acc i =>
        List.foldl
          (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
          acc
          (List.range' 0 D))
      (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
      (List.range' 0 D) := by
  unfold schoolbookRaw
  let aD : Coeffs := Array.ofFn (fun i : Fin D => a[i.1]!)
  let bD : Coeffs := Array.ofFn (fun i : Fin D => b[i.1]!)
  simpa [aD, bD] using
    (getCoeff_schoolbook_outer_fold_eq_scalar_fold (aD := aD) (bD := bD) (k := k))

theorem schoolbook_row_scalar_fold_eq_of_miss
  {aD bD : Coeffs} {i k : Nat} {acc : F}
  (hMiss : ∀ j, j < D → i + j ≠ k) :
  List.foldl
      (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
      acc
      (List.range' 0 D)
    = acc := by
  have hInner :
      ∀ inner acc0,
        (∀ j, j ∈ inner → j < D) →
        List.foldl
            (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
            acc0
            inner
          = acc0 := by
    intro inner
    induction inner with
    | nil =>
        intro acc0 _hIn
        simp
    | cons j js ih =>
        intro acc0 hIn
        have hjD : j < D := hIn j (by simp)
        have hNe : i + j ≠ k := hMiss j hjD
        have hTail :
            List.foldl
                (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
                acc0
                js
              = acc0 := by
          apply ih
          intro j' hj'
          exact hIn j' (by simp [hj'])
        calc
          List.foldl
              (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
              acc0
              (j :: js)
              =
              List.foldl
                (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
                (if i + j = k then acc0 + aD[i]! * bD[j]! else acc0)
                js := by
                  simp [List.foldl_cons]
          _ =
              List.foldl
                (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
                acc0
                js := by
                  simp [hNe]
          _ = acc0 := hTail
  have hRange : ∀ j, j ∈ List.range' 0 D → j < D := by
    intro j hj
    rcases (List.mem_range').1 hj with ⟨t, ht, hEq⟩
    have hEq' : j = t := by simpa using hEq
    simpa [hEq'] using ht
  simpa using hInner (inner := List.range' 0 D) (acc0 := acc) hRange

theorem schoolbook_row_scalar_fold_eq_of_outside
  {aD bD : Coeffs} {i k : Nat} {acc : F}
  (hOutside : k < i ∨ i + D ≤ k) :
  List.foldl
      (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
      acc
      (List.range' 0 D)
    = acc := by
  have hMiss : ∀ j, j < D → i + j ≠ k :=
    (schoolbook_row_misses_index_iff (i := i) (k := k)).2 hOutside
  exact schoolbook_row_scalar_fold_eq_of_miss (aD := aD) (bD := bD) (i := i) (k := k) (acc := acc) hMiss

theorem getCoeff_schoolbookRaw_eq_scalar_fold_filtered
  (a b : Coeffs) (k : Nat) :
  let aD : Coeffs := Array.ofFn (fun i : Fin D => a[i.1]!)
  let bD : Coeffs := Array.ofFn (fun i : Fin D => b[i.1]!)
  getCoeff (schoolbookRaw a b) k
    =
    List.foldl
      (fun acc i =>
        List.foldl
          (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
          acc
          (List.range' 0 D))
      (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
      ((List.range' 0 D).filter (fun i => decide (i ≤ k ∧ k < i + D))) := by
  let aD : Coeffs := Array.ofFn (fun i : Fin D => a[i.1]!)
  let bD : Coeffs := Array.ofFn (fun i : Fin D => b[i.1]!)
  let rowScalar : F → Nat → F :=
    fun acc i =>
      List.foldl
        (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
        acc
        (List.range' 0 D)
  let rowActive : Nat → Bool := fun i => decide (i ≤ k ∧ k < i + D)
  have hBase :
      getCoeff (schoolbookRaw a b) k
        =
        List.foldl rowScalar
          (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
          (List.range' 0 D) := by
    simpa [aD, bD, rowScalar] using
      (getCoeff_schoolbookRaw_eq_scalar_fold a b k)
  have hId :
      ∀ acc i, rowActive i = false → rowScalar acc i = acc := by
    intro acc i hInactive
    have hNotActive : ¬ (i ≤ k ∧ k < i + D) := by
      simpa [rowActive] using hInactive
    have hOutside : k < i ∨ i + D ≤ k := by
      by_cases hki : k < i
      · exact Or.inl hki
      · have hik : i ≤ k := Nat.le_of_not_gt hki
        have hkNot : ¬ k < i + D := by
          intro hkLt
          exact hNotActive ⟨hik, hkLt⟩
        exact Or.inr (Nat.le_of_not_gt hkNot)
    simpa [rowScalar] using
      (schoolbook_row_scalar_fold_eq_of_outside
        (aD := aD) (bD := bD) (i := i) (k := k) (acc := acc) hOutside)
  calc
    getCoeff (schoolbookRaw a b) k
        = List.foldl rowScalar
            (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
            (List.range' 0 D) := hBase
    _ = List.foldl rowScalar
          (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
          ((List.range' 0 D).filter rowActive) := by
          exact list_foldl_eq_foldl_filter_of_step_id
            (step := rowScalar)
            (p := rowActive)
            (init := getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
            (xs := List.range' 0 D)
            hId
    _ = List.foldl
          (fun acc i =>
            List.foldl
              (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
              acc
              (List.range' 0 D))
          (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
          ((List.range' 0 D).filter (fun i => decide (i ≤ k ∧ k < i + D))) := by
          simp [rowScalar, rowActive]

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

/-- Raw-coefficient accessor rewritten through explicit OOB-zero semantics. -/
theorem mulRqRawCoeffSpec_eq_getCoeff
  (a b : Coeffs) (k : Nat) :
  mulRqRawCoeffSpec a b k = getCoeff (mulRqRawCoeffs a b) k := by
  unfold mulRqRawCoeffSpec
  exact getElemBang_eq_getCoeff (mulRqRawCoeffs a b) k

theorem mulRqRawCoeffSpec_eq_scalar_fold
  (a b : Coeffs) (k : Nat) :
  let aD : Coeffs := Array.ofFn (fun i : Fin D => a[i.1]!)
  let bD : Coeffs := Array.ofFn (fun i : Fin D => b[i.1]!)
  mulRqRawCoeffSpec a b k
    =
    List.foldl
      (fun acc i =>
        List.foldl
          (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
          acc
          (List.range' 0 D))
      (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
      (List.range' 0 D) := by
  let aD : Coeffs := Array.ofFn (fun i : Fin D => a[i.1]!)
  let bD : Coeffs := Array.ofFn (fun i : Fin D => b[i.1]!)
  have hGet :
      mulRqRawCoeffSpec a b k = getCoeff (schoolbookRaw a b) k := by
    simpa [mulRqRawCoeffs] using (mulRqRawCoeffSpec_eq_getCoeff a b k)
  calc
    mulRqRawCoeffSpec a b k = getCoeff (schoolbookRaw a b) k := hGet
    _ = List.foldl
          (fun acc i =>
            List.foldl
              (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
              acc
              (List.range' 0 D))
          (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
          (List.range' 0 D) := by
            simpa [aD, bD] using (getCoeff_schoolbookRaw_eq_scalar_fold a b k)

theorem mulRqRawCoeffSpec_eq_scalar_fold_filtered
  (a b : Coeffs) (k : Nat) :
  let aD : Coeffs := Array.ofFn (fun i : Fin D => a[i.1]!)
  let bD : Coeffs := Array.ofFn (fun i : Fin D => b[i.1]!)
  mulRqRawCoeffSpec a b k
    =
    List.foldl
      (fun acc i =>
        List.foldl
          (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
          acc
          (List.range' 0 D))
      (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
      ((List.range' 0 D).filter (fun i => decide (i ≤ k ∧ k < i + D))) := by
  let aD : Coeffs := Array.ofFn (fun i : Fin D => a[i.1]!)
  let bD : Coeffs := Array.ofFn (fun i : Fin D => b[i.1]!)
  have hGet :
      mulRqRawCoeffSpec a b k = getCoeff (schoolbookRaw a b) k := by
    simpa [mulRqRawCoeffs] using (mulRqRawCoeffSpec_eq_getCoeff a b k)
  calc
    mulRqRawCoeffSpec a b k = getCoeff (schoolbookRaw a b) k := hGet
    _ = List.foldl
          (fun acc i =>
            List.foldl
              (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
              acc
              (List.range' 0 D))
          (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
          ((List.range' 0 D).filter (fun i => decide (i ≤ k ∧ k < i + D))) := by
            simpa [aD, bD] using (getCoeff_schoolbookRaw_eq_scalar_fold_filtered a b k)

theorem mulRqRawCoeffSpec_eq_getElem_of_lt
  {a b : Coeffs} {k : Nat}
  (hk : k < (mulRqRawCoeffs a b).size) :
  mulRqRawCoeffSpec a b k = (mulRqRawCoeffs a b)[k] := by
  calc
    mulRqRawCoeffSpec a b k = getCoeff (mulRqRawCoeffs a b) k :=
      mulRqRawCoeffSpec_eq_getCoeff a b k
    _ = (mulRqRawCoeffs a b)[k] := getCoeff_eq_getElem hk

theorem mulRqRawCoeffSpec_eq_zero_of_ge
  {a b : Coeffs} {k : Nat}
  (hk : 2 * D - 1 ≤ k) :
  mulRqRawCoeffSpec a b k = 0 := by
  unfold mulRqRawCoeffSpec
  have hNotLt : ¬ k < (mulRqRawCoeffs a b).size :=
    Nat.not_lt_of_ge (by simpa [mulRqRawCoeffs_size] using hk)
  simp [hNotLt, F.default_eq_zero]

theorem mulRqRawCoeffSpec_eq_zero_of_ge_via_scalar_fold
  {a b : Coeffs} {k : Nat}
  (hk : 2 * D - 1 ≤ k) :
  mulRqRawCoeffSpec a b k = 0 := by
  exact mulRqRawCoeffSpec_eq_zero_of_ge hk

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
