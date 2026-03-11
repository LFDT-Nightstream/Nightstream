import SuperNeo.Ring

namespace SuperNeo

open F
local instance : NeZero Goldilocks.q := ⟨Nat.ne_of_gt Goldilocks.q_pos⟩

private def basisVecNat (i : Nat) : Array F :=
  Array.ofFn (fun j : Fin d => if j.1 = i then (1 : F) else 0)

@[simp] private theorem basisVecNat_size (i : Nat) : (basisVecNat i).size = d := by
  simp [basisVecNat]

private theorem f_mul_zero (a : F) : a * 0 = 0 := by
  simpa using (Fin.mul_zero (n := Goldilocks.q) a)

private theorem f_add_zero (a : F) : a + 0 = a := by
  simpa using (Fin.add_zero (n := Goldilocks.q) a)

private def basisExpandPrefix (a : Array F) : Nat → Array F
  | 0 => Array.replicate d (0 : F)
  | n + 1 => vecAdd (basisExpandPrefix a n) (vecScale a[n]! (basisVecNat n))

private def basisExpand (a : Array F) : Array F :=
  basisExpandPrefix a d

private theorem basisExpandPrefix_size (a : Array F) (n : Nat) :
    (basisExpandPrefix a n).size = d := by
  induction n with
  | zero => simp [basisExpandPrefix, d]
  | succ n ih =>
      calc
        (basisExpandPrefix a (n + 1)).size
            = (vecAdd (basisExpandPrefix a n) (vecScale a[n]! (basisVecNat n))).size := by
                simp [basisExpandPrefix]
        _ = (basisExpandPrefix a n).size := by
              apply vecAdd_size_of_eq
              simp [ih, basisVecNat]
        _ = d := ih

private theorem coeffAt_basisExpandPrefix_of_lt
    (a : Array F)
    (j n : Nat)
    (hj : j < d) :
    coeffAt (basisExpandPrefix a n) j = if j < n then a[j]! else 0 := by
  induction n with
  | zero =>
      simp [basisExpandPrefix, coeffAt, hj]
  | succ n ih =>
      have hSize : (basisExpandPrefix a n).size = d := by simp [basisExpandPrefix_size]
      have hScaledSize : (vecScale a[n]! (basisVecNat n)).size = d := by
        simp [basisVecNat]
      calc
        coeffAt (basisExpandPrefix a (n + 1)) j
            = coeffAt (vecAdd (basisExpandPrefix a n) (vecScale a[n]! (basisVecNat n))) j := by
                simp [basisExpandPrefix]
        _ = coeffAt (basisExpandPrefix a n) j +
              coeffAt (vecScale a[n]! (basisVecNat n)) j := by
                exact coeffAt_vecAdd_of_size_d
                  (basisExpandPrefix a n)
                  (vecScale a[n]! (basisVecNat n))
                  hSize hScaledSize j hj
        _ = (if j < n then a[j]! else 0) +
              coeffAt (vecScale a[n]! (basisVecNat n)) j := by
                simp [ih]
        _ = (if j < n then a[j]! else 0) +
              a[n]! * coeffAt (basisVecNat n) j := by
                rw [coeffAt_vecScale_of_size_d (s := a[n]!) (x := basisVecNat n)
                  (hx := by simp [basisVecNat]) (k := j) hj]
        _ = if j < n + 1 then a[j]! else 0 := by
              by_cases hjn : j < n
              · have hne : j ≠ n := by omega
                have hlt : j < n + 1 := by omega
                calc
                  (if j < n then a[j]! else 0) + a[n]! * coeffAt (basisVecNat n) j
                      = a[j]! + a[n]! * 0 := by
                          simp [hjn, hne, basisVecNat, coeffAt, hj]
                  _ = a[j]! + 0 := by simp [f_mul_zero]
                  _ = a[j]! := by simp [f_add_zero]
                  _ = if j < n + 1 then a[j]! else 0 := by simp [hlt]
              · have hge : n ≤ j := Nat.le_of_not_gt hjn
                by_cases hEq : j = n
                · subst hEq
                  have hnot : ¬ j < j := by omega
                  have hlt : j < j + 1 := Nat.lt_succ_self j
                  simp [hnot, hlt, basisVecNat, coeffAt, hj,
                    Lean.Grind.Fin.mul_one]
                · have hnotlt : ¬ j < n + 1 := by omega
                  calc
                    (if j < n then a[j]! else 0) + a[n]! * coeffAt (basisVecNat n) j
                        = 0 + a[n]! * 0 := by
                            simp [hjn, hEq, basisVecNat, coeffAt, hj]
                    _ = 0 + 0 := by simp [f_mul_zero]
                    _ = 0 := by simp
                    _ = if j < n + 1 then a[j]! else 0 := by simp [hnotlt]

attribute [irreducible] basisExpandPrefix basisExpand

set_option maxRecDepth 4096 in
set_option maxHeartbeats 1200000 in
private theorem basisExpand_eq_of_size_d
    (a : Array F)
    (ha : a.size = d) :
    basisExpand a = a := by
  have hSize : (basisExpand a).size = d := by
    unfold basisExpand
    exact basisExpandPrefix_size a d
  apply Array.ext
  · exact hSize.trans ha.symm
  · intro j hjL hjR
    have hjExpand : j < (basisExpand a).size := by
      simpa [hSize] using hjL
    have hj : j < d := lt_of_lt_of_eq hjExpand hSize
    have hLGet : (basisExpand a)[j]'hjL = (basisExpand a).getD j 0 := by
      exact Array.getElem_eq_getD
        (xs := basisExpand a) (i := j) (h := hjL) (fallback := (0 : F))
    have hCoeff : coeffAt (basisExpand a) j = a[j]! := by
      have hCoeff0 := coeffAt_basisExpandPrefix_of_lt a j d hj
      rw [if_pos hj] at hCoeff0
      simpa [basisExpand] using hCoeff0
    have hRGet : a[j]'hjR = a.getD j 0 := by
      exact Array.getElem_eq_getD (xs := a) (i := j) (h := hjR) (fallback := (0 : F))
    have hRBang : a[j]! = a.getD j 0 := by
      simpa using (Array.getElem!_eq_getD (xs := a) (i := j))
    calc
      (basisExpand a)[j]'hjL = (basisExpand a).getD j 0 := hLGet
      _ = coeffAt (basisExpand a) j := by simp [basisExpand, coeffAt, hj]
      _ = a[j]! := hCoeff
      _ = a.getD j 0 := hRBang
      _ = a[j]'hjR := hRGet.symm

private theorem vecScale_zero_basis (i : Nat) :
    vecScale (0 : F) (basisVecNat i) = Array.replicate d (0 : F) := by
  apply Array.ext
  · simp [vecScale, basisVecNat, d]
  · intro j hjL hjR
    have hj : j < d := by simpa [vecScale, basisVecNat] using hjL
    have hCoeff : coeffAt (vecScale (0 : F) (basisVecNat i)) j = 0 := by
      rw [coeffAt_vecScale_of_size_d (s := (0 : F)) (x := basisVecNat i)
        (hx := by simp [basisVecNat]) (k := j) hj]
      simpa using (Lean.Grind.Fin.zero_mul (n := Goldilocks.q) (coeffAt (basisVecNat i) j))
    have hGet : (vecScale (0 : F) (basisVecNat i))[j]'hjL =
        coeffAt (vecScale (0 : F) (basisVecNat i)) j := by
      unfold coeffAt
      simp [hj, Array.getD]
    calc
      (vecScale (0 : F) (basisVecNat i))[j]'hjL = coeffAt (vecScale (0 : F) (basisVecNat i)) j := hGet
      _ = 0 := hCoeff
      _ = (Array.replicate d (0 : F))[j]'hjR := by simp

private theorem linear_eq_of_basis
    (K L : Array F → Array F)
    (hKSize : ∀ a, a.size = d → (K a).size = d)
    (hLSize : ∀ a, a.size = d → (L a).size = d)
    (hKAdd : ∀ x y, x.size = d → y.size = d → K (vecAdd x y) = vecAdd (K x) (K y))
    (hLAdd : ∀ x y, x.size = d → y.size = d → L (vecAdd x y) = vecAdd (L x) (L y))
    (hKScale : ∀ s x, x.size = d → K (vecScale s x) = vecScale s (K x))
    (hLScale : ∀ s x, x.size = d → L (vecScale s x) = vecScale s (L x))
    (hBasis : ∀ n, n < d → K (basisVecNat n) = L (basisVecNat n)) :
    ∀ a, a.size = d → K a = L a := by
  intro a ha
  have hPrefix : ∀ n, n ≤ d → K (basisExpandPrefix a n) = L (basisExpandPrefix a n) := by
    intro n hn
    induction n with
    | zero =>
        have h0d : 0 < d := by decide
        have hZeroK : K (basisExpandPrefix a 0) = K (vecScale (0 : F) (basisVecNat 0)) := by
          simp [basisExpandPrefix, vecScale_zero_basis]
        have hZeroL : L (basisExpandPrefix a 0) = L (vecScale (0 : F) (basisVecNat 0)) := by
          simp [basisExpandPrefix, vecScale_zero_basis]
        calc
          K (basisExpandPrefix a 0) = K (vecScale (0 : F) (basisVecNat 0)) := hZeroK
          _ = vecScale (0 : F) (K (basisVecNat 0)) := by
                exact hKScale (0 : F) (basisVecNat 0) (by simp [basisVecNat])
          _ = vecScale (0 : F) (L (basisVecNat 0)) := by
                simp [hBasis 0 h0d]
          _ = L (vecScale (0 : F) (basisVecNat 0)) := by
                symm
                exact hLScale (0 : F) (basisVecNat 0) (by simp [basisVecNat])
          _ = L (basisExpandPrefix a 0) := hZeroL.symm
    | succ n ih =>
        have hnlt : n < d := by omega
        have hPrevSize : (basisExpandPrefix a n).size = d := by simp [basisExpandPrefix_size]
        calc
          K (basisExpandPrefix a (n + 1))
              = K (vecAdd (basisExpandPrefix a n) (vecScale a[n]! (basisVecNat n))) := by
                  simp [basisExpandPrefix]
          _ = vecAdd (K (basisExpandPrefix a n)) (K (vecScale a[n]! (basisVecNat n))) := by
                exact hKAdd (basisExpandPrefix a n) (vecScale a[n]! (basisVecNat n)) hPrevSize (by simp [basisVecNat])
          _ = vecAdd (L (basisExpandPrefix a n)) (K (vecScale a[n]! (basisVecNat n))) := by
                simp [ih (by omega)]
          _ = vecAdd (L (basisExpandPrefix a n)) (vecScale a[n]! (K (basisVecNat n))) := by
                rw [hKScale]
                simp [basisVecNat]
          _ = vecAdd (L (basisExpandPrefix a n)) (vecScale a[n]! (L (basisVecNat n))) := by
                simp [hBasis n hnlt]
          _ = vecAdd (L (basisExpandPrefix a n)) (L (vecScale a[n]! (basisVecNat n))) := by
                rw [hLScale]
                simp [basisVecNat]
          _ = L (vecAdd (basisExpandPrefix a n) (vecScale a[n]! (basisVecNat n))) := by
                exact (hLAdd (basisExpandPrefix a n) (vecScale a[n]! (basisVecNat n)) hPrevSize (by simp [basisVecNat])).symm
          _ = L (basisExpandPrefix a (n + 1)) := by
                simp [basisExpandPrefix]
  calc
    K a = K (basisExpand a) := by rw [basisExpand_eq_of_size_d a ha]
    _ = L (basisExpand a) := by
          simpa [basisExpand] using hPrefix d (Nat.le_refl d)
    _ = L a := by rw [basisExpand_eq_of_size_d a ha]

private theorem bilinear_eq_of_basis
    (K L : Array F → Array F → Array F)
    (hKSize : ∀ a b, a.size = d → b.size = d → (K a b).size = d)
    (hLSize : ∀ a b, a.size = d → b.size = d → (L a b).size = d)
    (hKAddLeft : ∀ x y b, x.size = d → y.size = d → b.size = d → K (vecAdd x y) b = vecAdd (K x b) (K y b))
    (hLAddLeft : ∀ x y b, x.size = d → y.size = d → b.size = d → L (vecAdd x y) b = vecAdd (L x b) (L y b))
    (hKScaleLeft : ∀ s x b, x.size = d → b.size = d → K (vecScale s x) b = vecScale s (K x b))
    (hLScaleLeft : ∀ s x b, x.size = d → b.size = d → L (vecScale s x) b = vecScale s (L x b))
    (hKAddRight : ∀ a x y, a.size = d → x.size = d → y.size = d → K a (vecAdd x y) = vecAdd (K a x) (K a y))
    (hLAddRight : ∀ a x y, a.size = d → x.size = d → y.size = d → L a (vecAdd x y) = vecAdd (L a x) (L a y))
    (hKScaleRight : ∀ s a x, a.size = d → x.size = d → K a (vecScale s x) = vecScale s (K a x))
    (hLScaleRight : ∀ s a x, a.size = d → x.size = d → L a (vecScale s x) = vecScale s (L a x))
    (hBasis : ∀ i j, i < d → j < d → K (basisVecNat i) (basisVecNat j) = L (basisVecNat i) (basisVecNat j)) :
    ∀ a b, a.size = d → b.size = d → K a b = L a b := by
  intro a b ha hb
  have hLeftBasis : ∀ i, i < d → K (basisVecNat i) b = L (basisVecNat i) b := by
    intro i hi
    exact linear_eq_of_basis
      (K := fun x => K (basisVecNat i) x)
      (L := fun x => L (basisVecNat i) x)
      (hKSize := fun x hx => hKSize (basisVecNat i) x (by simp [basisVecNat]) hx)
      (hLSize := fun x hx => hLSize (basisVecNat i) x (by simp [basisVecNat]) hx)
      (hKAdd := fun x y hx hy => hKAddRight (basisVecNat i) x y (by simp [basisVecNat]) hx hy)
      (hLAdd := fun x y hx hy => hLAddRight (basisVecNat i) x y (by simp [basisVecNat]) hx hy)
      (hKScale := fun s x hx => hKScaleRight s (basisVecNat i) x (by simp [basisVecNat]) hx)
      (hLScale := fun s x hx => hLScaleRight s (basisVecNat i) x (by simp [basisVecNat]) hx)
      (hBasis := fun j hj => hBasis i j hi hj)
      b hb
  exact linear_eq_of_basis
    (K := fun x => K x b)
    (L := fun x => L x b)
    (hKSize := fun x hx => hKSize x b hx hb)
    (hLSize := fun x hx => hLSize x b hx hb)
    (hKAdd := fun x y hx hy => hKAddLeft x y b hx hy hb)
    (hLAdd := fun x y hx hy => hLAddLeft x y b hx hy hb)
    (hKScale := fun s x hx => hKScaleLeft s x b hx hb)
    (hLScale := fun s x hx => hLScaleLeft s x b hx hb)
    (hBasis := hLeftBasis)
    a ha

private def canonicalCoeffs (a : Coeffs) : Coeffs :=
  Array.ofFn (fun i : Fin d => coeffAt a i.1)

@[simp] private theorem canonicalCoeffs_size (a : Coeffs) : (canonicalCoeffs a).size = d := by
  simp [canonicalCoeffs]

private theorem coeffAt_canonicalCoeffs
    (a : Coeffs)
    (i : Nat)
    (hi : i < d) :
    coeffAt (canonicalCoeffs a) i = coeffAt a i := by
  unfold canonicalCoeffs coeffAt
  have hGet : (Array.ofFn (fun j : Fin d => coeffAt a j.1)).getD i 0 = coeffAt a i := by
    simp [Array.getD, hi]
  simpa [hi] using hGet

private theorem canonicalCoeffs_eq_of_size_d
    (a : Coeffs)
    (ha : a.size = d) :
    canonicalCoeffs a = a := by
  apply Array.ext
  · simp [canonicalCoeffs, ha]
  · intro i hiL hiR
    have hi : i < d := by simpa [canonicalCoeffs, ha] using hiL
    have hiA : i < a.size := by simpa [ha] using hiR
    simp [canonicalCoeffs, coeffAt, hi, ha, hiA]

private theorem get_eq_coeffAt_of_size_d
    (a : Coeffs)
    (ha : a.size = d)
    (i : Nat)
    (hi : i < d) :
    a[i]'(by simpa [ha] using hi) = coeffAt a i := by
  unfold coeffAt
  simp [hi, ha, Array.getD]

private theorem list_foldl_congr_mem
  {α β : Type}
  (f g : α → β → α)
  (init : α)
  (l : List β)
  (hfg : ∀ acc b, b ∈ l → f acc b = g acc b) :
  List.foldl f init l = List.foldl g init l := by
  induction l generalizing init with
  | nil =>
      simp
  | cons b bs ih =>
      have hHead : f init b = g init b := by
        exact hfg init b (by simp)
      calc
        List.foldl f init (b :: bs)
            = List.foldl f (f init b) bs := by
                rfl
        _ = List.foldl f (g init b) bs := by
              rw [hHead]
        _ = List.foldl g (g init b) bs := by
              apply ih
              intro acc b' hb'
              exact hfg acc b' (by simp [hb'])
        _ = List.foldl g init (b :: bs) := by
              rfl

private theorem rawConvCoeff_canonical_left
    (a b : Coeffs)
    (n : Nat) :
    rawConvCoeff (canonicalCoeffs a) b n = rawConvCoeff a b n := by
  unfold rawConvCoeff
  apply list_foldl_congr_mem
  intro acc j hjMem
  have hj : j < d := by simpa [List.mem_range] using hjMem
  by_cases hIn : j ≤ n ∧ n - j < d
  · simp [hIn, coeffAt_canonicalCoeffs, hj]
  · simp [hIn]

private theorem rawConvCoeff_canonical_right
    (a b : Coeffs)
    (n : Nat) :
    rawConvCoeff a (canonicalCoeffs b) n = rawConvCoeff a b n := by
  unfold rawConvCoeff
  apply list_foldl_congr_mem
  intro acc j hjMem
  by_cases hIn : j ≤ n ∧ n - j < d
  · simp [hIn, coeffAt_canonicalCoeffs, hIn.2]
  · simp [hIn]

private def monomialReduce (n : Nat) : Coeffs :=
  let r := n % 81
  if hLt : r < d then
    basisVecNat r
  else
    vecAdd
      (vecScale (-1 : F) (basisVecNat (r - 54)))
      (vecScale (-1 : F) (basisVecNat (r - 27)))

@[simp] private theorem monomialReduce_size (n : Nat) :
    (monomialReduce n).size = d := by
  unfold monomialReduce
  dsimp
  split
  · simp [basisVecNat]
  · calc
      (vecAdd
        (vecScale (-1 : F) (basisVecNat (n % 81 - 54)))
        (vecScale (-1 : F) (basisVecNat (n % 81 - 27)))).size
          = (vecScale (-1 : F) (basisVecNat (n % 81 - 54))).size := by
              exact vecAdd_size_of_eq (by simp [basisVecNat])
      _ = d := by simp [basisVecNat]

set_option maxRecDepth 4096 in
private theorem mulRqPhi_basis_basis :
    ∀ i j : Fin d,
      mulRqPhi (basisVecNat i.1) (basisVecNat j.1) =
        monomialReduce (i.1 + j.1) := by
  native_decide

private theorem monomialReduce_eq_of_modEq
    (m n : Nat)
    (hmod : m % 81 = n % 81) :
    monomialReduce m = monomialReduce n := by
  unfold monomialReduce
  simp [hmod]

set_option maxRecDepth 4096 in
private theorem mulRqPhi_basis_monomialReduce_mod :
    ∀ i : Fin d, ∀ r : Fin 81,
      mulRqPhi (basisVecNat i.1) (monomialReduce r.1) =
        monomialReduce (i.1 + r.1) := by
  native_decide

set_option maxRecDepth 4096 in
set_option maxHeartbeats 800000 in
private theorem mulRqPhi_canonical_left
    (a b : Coeffs) :
    mulRqPhi a b = mulRqPhi (canonicalCoeffs a) b := by
  have hcoeff :
      ∀ i : Nat, i < d →
        coeffAt (mulRq a b) i = coeffAt (mulRq (canonicalCoeffs a) b) i := by
    intro i hi
    have hLeftI := rawConvCoeff_canonical_left a b i
    by_cases hLt26 : i < 26
    · have hLeft54 := rawConvCoeff_canonical_left a b (i + 54)
      have hLeft81 := rawConvCoeff_canonical_left a b (i + 81)
      calc
        coeffAt (mulRq a b) i
            = rawConvCoeff a b i - rawConvCoeff a b (i + 54) + rawConvCoeff a b (i + 81) := by
                exact coeffAt_mulRq_lt26 a b i hi hLt26
        _ = rawConvCoeff (canonicalCoeffs a) b i
              - rawConvCoeff (canonicalCoeffs a) b (i + 54)
              + rawConvCoeff (canonicalCoeffs a) b (i + 81) := by
                rw [← hLeftI, ← hLeft54, ← hLeft81]
        _ = coeffAt (mulRq (canonicalCoeffs a) b) i := by
              symm
              exact coeffAt_mulRq_lt26 (canonicalCoeffs a) b i hi hLt26
    · by_cases hEq26 : i = 26
      · subst hEq26
        have hLeft80 := rawConvCoeff_canonical_left a b 80
        calc
          coeffAt (mulRq a b) 26
              = rawConvCoeff a b 26 - rawConvCoeff a b 80 := by
                  exact coeffAt_mulRq_eq26 a b 26 (by decide) rfl
          _ = rawConvCoeff (canonicalCoeffs a) b 26
                - rawConvCoeff (canonicalCoeffs a) b 80 := by
                  rw [← hLeftI, ← hLeft80]
          _ = coeffAt (mulRq (canonicalCoeffs a) b) 26 := by
                symm
                exact coeffAt_mulRq_eq26 (canonicalCoeffs a) b 26 (by decide) rfl
      · have hLeft27 := rawConvCoeff_canonical_left a b (i + 27)
        have hGt26 : 26 < i := by omega
        calc
          coeffAt (mulRq a b) i
              = rawConvCoeff a b i - rawConvCoeff a b (i + 27) := by
                  exact coeffAt_mulRq_gt26 a b i hi hGt26
          _ = rawConvCoeff (canonicalCoeffs a) b i
                - rawConvCoeff (canonicalCoeffs a) b (i + 27) := by
                  rw [← hLeftI, ← hLeft27]
          _ = coeffAt (mulRq (canonicalCoeffs a) b) i := by
                symm
                exact coeffAt_mulRq_gt26 (canonicalCoeffs a) b i hi hGt26
  calc
    mulRqPhi a b = canonicalCoeffs (mulRqPhi a b) := by
      symm
      exact canonicalCoeffs_eq_of_size_d (mulRqPhi a b) (mulRqPhi_size a b)
    _ = canonicalCoeffs (mulRqPhi (canonicalCoeffs a) b) := by
      apply Array.ext
      · simp [canonicalCoeffs]
      · intro i hiL hiR
        have hi : i < d := by simpa [canonicalCoeffs] using hiL
        simpa [canonicalCoeffs, coeffAt, hi] using hcoeff i hi
    _ = mulRqPhi (canonicalCoeffs a) b := by
      exact canonicalCoeffs_eq_of_size_d (mulRqPhi (canonicalCoeffs a) b) (mulRqPhi_size (canonicalCoeffs a) b)

set_option maxRecDepth 4096 in
set_option maxHeartbeats 800000 in
private theorem mulRqPhi_canonical_right
    (a b : Coeffs) :
    mulRqPhi a b = mulRqPhi a (canonicalCoeffs b) := by
  have hcoeff :
      ∀ i : Nat, i < d →
        coeffAt (mulRq a b) i = coeffAt (mulRq a (canonicalCoeffs b)) i := by
    intro i hi
    have hRightI := rawConvCoeff_canonical_right a b i
    by_cases hLt26 : i < 26
    · have hRight54 := rawConvCoeff_canonical_right a b (i + 54)
      have hRight81 := rawConvCoeff_canonical_right a b (i + 81)
      calc
        coeffAt (mulRq a b) i
            = rawConvCoeff a b i - rawConvCoeff a b (i + 54) + rawConvCoeff a b (i + 81) := by
                exact coeffAt_mulRq_lt26 a b i hi hLt26
        _ = rawConvCoeff a (canonicalCoeffs b) i
              - rawConvCoeff a (canonicalCoeffs b) (i + 54)
              + rawConvCoeff a (canonicalCoeffs b) (i + 81) := by
                rw [hRightI, hRight54, hRight81]
        _ = coeffAt (mulRq a (canonicalCoeffs b)) i := by
              symm
              exact coeffAt_mulRq_lt26 a (canonicalCoeffs b) i hi hLt26
    · by_cases hEq26 : i = 26
      · subst hEq26
        have hRight80 := rawConvCoeff_canonical_right a b 80
        calc
          coeffAt (mulRq a b) 26
              = rawConvCoeff a b 26 - rawConvCoeff a b 80 := by
                  exact coeffAt_mulRq_eq26 a b 26 (by decide) rfl
          _ = rawConvCoeff a (canonicalCoeffs b) 26
                - rawConvCoeff a (canonicalCoeffs b) 80 := by
                  rw [hRightI, hRight80]
          _ = coeffAt (mulRq a (canonicalCoeffs b)) 26 := by
                symm
                exact coeffAt_mulRq_eq26 a (canonicalCoeffs b) 26 (by decide) rfl
      · have hRight27 := rawConvCoeff_canonical_right a b (i + 27)
        have hGt26 : 26 < i := by omega
        calc
          coeffAt (mulRq a b) i
              = rawConvCoeff a b i - rawConvCoeff a b (i + 27) := by
                  exact coeffAt_mulRq_gt26 a b i hi hGt26
          _ = rawConvCoeff a (canonicalCoeffs b) i
                - rawConvCoeff a (canonicalCoeffs b) (i + 27) := by
                  rw [hRightI, hRight27]
          _ = coeffAt (mulRq a (canonicalCoeffs b)) i := by
                symm
                exact coeffAt_mulRq_gt26 a (canonicalCoeffs b) i hi hGt26
  calc
    mulRqPhi a b = canonicalCoeffs (mulRqPhi a b) := by
      symm
      exact canonicalCoeffs_eq_of_size_d (mulRqPhi a b) (mulRqPhi_size a b)
    _ = canonicalCoeffs (mulRqPhi a (canonicalCoeffs b)) := by
      apply Array.ext
      · simp [canonicalCoeffs]
      · intro i hiL hiR
        have hi : i < d := by simpa [canonicalCoeffs] using hiL
        simpa [canonicalCoeffs, coeffAt, hi] using hcoeff i hi
    _ = mulRqPhi a (canonicalCoeffs b) := by
      exact canonicalCoeffs_eq_of_size_d (mulRqPhi a (canonicalCoeffs b)) (mulRqPhi_size a (canonicalCoeffs b))

private theorem mulRqPhi_canonical
    (a b : Coeffs) :
    mulRqPhi a b = mulRqPhi (canonicalCoeffs a) (canonicalCoeffs b) := by
  calc
    mulRqPhi a b = mulRqPhi (canonicalCoeffs a) b := mulRqPhi_canonical_left a b
    _ = mulRqPhi (canonicalCoeffs a) (canonicalCoeffs b) :=
      mulRqPhi_canonical_right (canonicalCoeffs a) b

private theorem mulRqPhi_basis_monomialReduce
    (i : Nat)
    (hi : i < d) :
    ∀ n : Nat,
      mulRqPhi (basisVecNat i) (monomialReduce n) =
        monomialReduce (i + n) := by
  intro n
  let r := n % 81
  let rf : Fin 81 := ⟨r, Nat.mod_lt _ (by decide)⟩
  have hMono : monomialReduce n = monomialReduce rf.1 := by
    apply monomialReduce_eq_of_modEq
    simp [r, rf]
  calc
    mulRqPhi (basisVecNat i) (monomialReduce n)
        = mulRqPhi (basisVecNat i) (monomialReduce rf.1) := by rw [hMono]
    _ = monomialReduce (i + rf.1) := mulRqPhi_basis_monomialReduce_mod ⟨i, hi⟩ rf
    _ = monomialReduce (i + n) := by
          apply monomialReduce_eq_of_modEq
          simp [Nat.add_mod, r, rf]

private theorem mulRqPhi_leftActionComm_on_basis :
    ∀ i j k : Fin d,
      mulRqPhi (basisVecNat i.1) (mulRqPhi (basisVecNat j.1) (basisVecNat k.1)) =
        mulRqPhi (basisVecNat j.1) (mulRqPhi (basisVecNat i.1) (basisVecNat k.1)) := by
  intro i j k
  calc
    mulRqPhi (basisVecNat i.1) (mulRqPhi (basisVecNat j.1) (basisVecNat k.1))
        = mulRqPhi (basisVecNat i.1) (monomialReduce (j.1 + k.1)) := by
            rw [mulRqPhi_basis_basis j k]
    _ = monomialReduce (i.1 + (j.1 + k.1)) :=
          mulRqPhi_basis_monomialReduce i.1 i.2 (j.1 + k.1)
    _ = monomialReduce (j.1 + (i.1 + k.1)) := by
          congr 1
          omega
    _ = mulRqPhi (basisVecNat j.1) (monomialReduce (i.1 + k.1)) := by
          symm
          exact mulRqPhi_basis_monomialReduce j.1 j.2 (i.1 + k.1)
    _ = mulRqPhi (basisVecNat j.1) (mulRqPhi (basisVecNat i.1) (basisVecNat k.1)) := by
          rw [mulRqPhi_basis_basis i k]

private theorem mulRqPhi_leftActionComm_on_basis_right
    (n : Nat)
    (hn : n < d) :
    ∀ a b, a.size = d → b.size = d →
      mulRqPhi a (mulRqPhi b (basisVecNat n)) =
        mulRqPhi b (mulRqPhi a (basisVecNat n)) := by
  intro a b ha hb
  exact bilinear_eq_of_basis
    (K := fun x y => mulRqPhi x (mulRqPhi y (basisVecNat n)))
    (L := fun x y => mulRqPhi y (mulRqPhi x (basisVecNat n)))
    (hKSize := fun x y _ _ => mulRqPhi_size x (mulRqPhi y (basisVecNat n)))
    (hLSize := fun x y _ _ => mulRqPhi_size y (mulRqPhi x (basisVecNat n)))
    (hKAddLeft := fun x y z hx hy hz => by
      calc
        mulRqPhi (vecAdd x y) (mulRqPhi z (basisVecNat n))
            = vecAdd (mulRqPhi x (mulRqPhi z (basisVecNat n)))
                (mulRqPhi y (mulRqPhi z (basisVecNat n))) := by
                  exact mulRqPhi_vecAdd_left_of_size_d x y (mulRqPhi z (basisVecNat n)) hx hy
        _ = vecAdd
              ((fun x y => mulRqPhi x (mulRqPhi y (basisVecNat n))) x z)
              ((fun x y => mulRqPhi x (mulRqPhi y (basisVecNat n))) y z) := by
              rfl)
    (hLAddLeft := fun x y z hx hy hz => by
      calc
        mulRqPhi z (mulRqPhi (vecAdd x y) (basisVecNat n))
            = mulRqPhi z (vecAdd (mulRqPhi x (basisVecNat n)) (mulRqPhi y (basisVecNat n))) := by
                  rw [mulRqPhi_vecAdd_left_of_size_d x y (basisVecNat n) hx hy]
        _ = vecAdd (mulRqPhi z (mulRqPhi x (basisVecNat n)))
                (mulRqPhi z (mulRqPhi y (basisVecNat n))) := by
                  exact mulRqPhi_vecAdd_right_of_size_d z (mulRqPhi x (basisVecNat n)) (mulRqPhi y (basisVecNat n))
                    (mulRqPhi_size x (basisVecNat n)) (mulRqPhi_size y (basisVecNat n))
        _ = vecAdd
              ((fun x y => mulRqPhi y (mulRqPhi x (basisVecNat n))) x z)
              ((fun x y => mulRqPhi y (mulRqPhi x (basisVecNat n))) y z) := by
              rfl)
    (hKScaleLeft := fun s x z hx hz => by
      calc
        mulRqPhi (vecScale s x) (mulRqPhi z (basisVecNat n))
            = vecScale s (mulRqPhi x (mulRqPhi z (basisVecNat n))) := by
                  exact mulRqPhi_vecScale_left_of_size_d s x (mulRqPhi z (basisVecNat n)) hx
        _ = vecScale s ((fun x y => mulRqPhi x (mulRqPhi y (basisVecNat n))) x z) := by
              rfl)
    (hLScaleLeft := fun s x z hx hz => by
      calc
        mulRqPhi z (mulRqPhi (vecScale s x) (basisVecNat n))
            = mulRqPhi z (vecScale s (mulRqPhi x (basisVecNat n))) := by
                  rw [mulRqPhi_vecScale_left_of_size_d s x (basisVecNat n) hx]
        _ = vecScale s (mulRqPhi z (mulRqPhi x (basisVecNat n))) := by
                  exact mulRqPhi_vecScale_right_of_size_d s z (mulRqPhi x (basisVecNat n)) (mulRqPhi_size x (basisVecNat n))
        _ = vecScale s ((fun x y => mulRqPhi y (mulRqPhi x (basisVecNat n))) x z) := by
              rfl)
    (hKAddRight := fun x y z hx hy hz => by
      calc
        mulRqPhi x (mulRqPhi (vecAdd y z) (basisVecNat n))
            = mulRqPhi x (vecAdd (mulRqPhi y (basisVecNat n)) (mulRqPhi z (basisVecNat n))) := by
                  rw [mulRqPhi_vecAdd_left_of_size_d y z (basisVecNat n) hy hz]
        _ = vecAdd (mulRqPhi x (mulRqPhi y (basisVecNat n)))
                (mulRqPhi x (mulRqPhi z (basisVecNat n))) := by
                  exact mulRqPhi_vecAdd_right_of_size_d x (mulRqPhi y (basisVecNat n)) (mulRqPhi z (basisVecNat n))
                    (mulRqPhi_size y (basisVecNat n)) (mulRqPhi_size z (basisVecNat n))
        _ = vecAdd
              ((fun x y => mulRqPhi x (mulRqPhi y (basisVecNat n))) x y)
              ((fun x y => mulRqPhi x (mulRqPhi y (basisVecNat n))) x z) := by
              rfl)
    (hLAddRight := fun x y z hx hy hz => by
      calc
        mulRqPhi (vecAdd y z) (mulRqPhi x (basisVecNat n))
            = vecAdd (mulRqPhi y (mulRqPhi x (basisVecNat n)))
                (mulRqPhi z (mulRqPhi x (basisVecNat n))) := by
                  exact mulRqPhi_vecAdd_left_of_size_d y z (mulRqPhi x (basisVecNat n)) hy hz
        _ = vecAdd
              ((fun x y => mulRqPhi y (mulRqPhi x (basisVecNat n))) x y)
              ((fun x y => mulRqPhi y (mulRqPhi x (basisVecNat n))) x z) := by
              rfl)
    (hKScaleRight := fun s x z hx hz => by
      calc
        mulRqPhi x (mulRqPhi (vecScale s z) (basisVecNat n))
            = mulRqPhi x (vecScale s (mulRqPhi z (basisVecNat n))) := by
                  rw [mulRqPhi_vecScale_left_of_size_d s z (basisVecNat n) hz]
        _ = vecScale s (mulRqPhi x (mulRqPhi z (basisVecNat n))) := by
                  exact mulRqPhi_vecScale_right_of_size_d s x (mulRqPhi z (basisVecNat n)) (mulRqPhi_size z (basisVecNat n))
        _ = vecScale s ((fun x y => mulRqPhi x (mulRqPhi y (basisVecNat n))) x z) := by
              rfl)
    (hLScaleRight := fun s x z hx hz => by
      calc
        mulRqPhi (vecScale s z) (mulRqPhi x (basisVecNat n))
            = vecScale s (mulRqPhi z (mulRqPhi x (basisVecNat n))) := by
                  exact mulRqPhi_vecScale_left_of_size_d s z (mulRqPhi x (basisVecNat n)) hz
        _ = vecScale s ((fun x y => mulRqPhi y (mulRqPhi x (basisVecNat n))) x z) := by
              rfl)
    (hBasis := fun i j hi hj => by
      simpa using mulRqPhi_leftActionComm_on_basis ⟨i, hi⟩ ⟨j, hj⟩ ⟨n, hn⟩)
    a b ha hb

private theorem mulRqPhi_leftActionComm_canonical
    (a b c : Coeffs)
    (ha : a.size = d)
    (hb : b.size = d)
    (hc : c.size = d) :
    mulRqPhi a (mulRqPhi b c) = mulRqPhi b (mulRqPhi a c) := by
  exact linear_eq_of_basis
    (K := fun x => mulRqPhi a (mulRqPhi b x))
    (L := fun x => mulRqPhi b (mulRqPhi a x))
    (hKSize := fun x hx => mulRqPhi_size a (mulRqPhi b x))
    (hLSize := fun x hx => mulRqPhi_size b (mulRqPhi a x))
    (hKAdd := fun x y hx hy => by
      calc
        mulRqPhi a (mulRqPhi b (vecAdd x y))
            = mulRqPhi a (vecAdd (mulRqPhi b x) (mulRqPhi b y)) := by
                  rw [mulRqPhi_vecAdd_right_of_size_d b x y hx hy]
        _ = vecAdd (mulRqPhi a (mulRqPhi b x)) (mulRqPhi a (mulRqPhi b y)) := by
                  exact mulRqPhi_vecAdd_right_of_size_d a (mulRqPhi b x) (mulRqPhi b y)
                    (mulRqPhi_size b x) (mulRqPhi_size b y)
        _ = vecAdd
              ((fun x => mulRqPhi a (mulRqPhi b x)) x)
              ((fun x => mulRqPhi a (mulRqPhi b x)) y) := by
              rfl)
    (hLAdd := fun x y hx hy => by
      calc
        mulRqPhi b (mulRqPhi a (vecAdd x y))
            = mulRqPhi b (vecAdd (mulRqPhi a x) (mulRqPhi a y)) := by
                  rw [mulRqPhi_vecAdd_right_of_size_d a x y hx hy]
        _ = vecAdd (mulRqPhi b (mulRqPhi a x)) (mulRqPhi b (mulRqPhi a y)) := by
                  exact mulRqPhi_vecAdd_right_of_size_d b (mulRqPhi a x) (mulRqPhi a y)
                    (mulRqPhi_size a x) (mulRqPhi_size a y)
        _ = vecAdd
              ((fun x => mulRqPhi b (mulRqPhi a x)) x)
              ((fun x => mulRqPhi b (mulRqPhi a x)) y) := by
              rfl)
    (hKScale := fun s x hx => by
      calc
        mulRqPhi a (mulRqPhi b (vecScale s x))
            = mulRqPhi a (vecScale s (mulRqPhi b x)) := by
                  rw [mulRqPhi_vecScale_right_of_size_d s b x hx]
        _ = vecScale s (mulRqPhi a (mulRqPhi b x)) := by
                  exact mulRqPhi_vecScale_right_of_size_d s a (mulRqPhi b x) (mulRqPhi_size b x)
        _ = vecScale s ((fun x => mulRqPhi a (mulRqPhi b x)) x) := by
              rfl)
    (hLScale := fun s x hx => by
      calc
        mulRqPhi b (mulRqPhi a (vecScale s x))
            = mulRqPhi b (vecScale s (mulRqPhi a x)) := by
                  rw [mulRqPhi_vecScale_right_of_size_d s a x hx]
        _ = vecScale s (mulRqPhi b (mulRqPhi a x)) := by
                  exact mulRqPhi_vecScale_right_of_size_d s b (mulRqPhi a x) (mulRqPhi_size a x)
        _ = vecScale s ((fun x => mulRqPhi b (mulRqPhi a x)) x) := by
              rfl)
    (hBasis := fun n hn => mulRqPhi_leftActionComm_on_basis_right n hn a b ha hb)
    c hc

theorem mulRqPhi_leftActionComm
    (a b c : Coeffs) :
    mulRqPhi a (mulRqPhi b c) = mulRqPhi b (mulRqPhi a c) := by
  calc
    mulRqPhi a (mulRqPhi b c)
        = mulRqPhi (canonicalCoeffs a) (canonicalCoeffs (mulRqPhi b c)) := by
            simpa using mulRqPhi_canonical a (mulRqPhi b c)
    _ = mulRqPhi (canonicalCoeffs a) (mulRqPhi (canonicalCoeffs b) (canonicalCoeffs c)) := by
          simp [canonicalCoeffs_eq_of_size_d, mulRqPhi_size, mulRqPhi_canonical]
    _ = mulRqPhi (canonicalCoeffs b) (mulRqPhi (canonicalCoeffs a) (canonicalCoeffs c)) :=
          mulRqPhi_leftActionComm_canonical (canonicalCoeffs a) (canonicalCoeffs b) (canonicalCoeffs c)
            (canonicalCoeffs_size a) (canonicalCoeffs_size b) (canonicalCoeffs_size c)
    _ = mulRqPhi (canonicalCoeffs b) (canonicalCoeffs (mulRqPhi a c)) := by
          simp [canonicalCoeffs_eq_of_size_d, mulRqPhi_size, mulRqPhi_canonical]
    _ = mulRqPhi b (mulRqPhi a c) := by
          symm
          simpa using mulRqPhi_canonical b (mulRqPhi a c)

theorem mulRq_leftActionComm
    (a b c : Coeffs) :
    mulRq a (mulRq b c) = mulRq b (mulRq a c) := by
  simpa [mulRq] using mulRqPhi_leftActionComm a b c

end SuperNeo
