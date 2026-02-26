import SuperNeo.CoeffMaps

namespace SuperNeo

/-- Half-modulus threshold for centered representatives in F_q. -/
def halfQ : Nat := (q - 1) / 2

theorem halfQ_lt_q : halfQ < q := by
  unfold halfQ q
  decide

theorem q_sub_halfQ_succ_eq_halfQ : q - (halfQ + 1) = halfQ := by
  unfold q halfQ
  decide

/-- Centered absolute value for a canonical residue x mod q. -/
def centeredAbsNat (x : Nat) : Nat :=
  let xr := x % q
  if xr <= halfQ then
    xr
  else
    q - xr

/-- Infinity norm of a field element in centered representation. -/
def normInfF (a : F) : Nat := centeredAbsNat a.val

/-- Infinity norm of one ring element represented by its d coefficients. -/
def normInfCoeffs (a : Coeffs) : Nat :=
  a.foldl (fun m x => Nat.max m (normInfF x)) 0

/-- Infinity norm of a vector of ring elements (max over element norms). -/
def normInfCoeffMatrix (m : Array Coeffs) : Nat :=
  m.foldl (fun acc row => Nat.max acc (normInfCoeffs row)) 0

/--
Coefficientwise subtraction used by low-norm invertibility side conditions.
Returns `#[]` on mismatched sizes.
-/
def coeffSub (a b : Coeffs) : Coeffs :=
  if _h : a.size = b.size then
    Array.ofFn (fun i : Fin a.size => a[i.1]! - b[i.1]!)
  else
    #[]

theorem coeffSub_size_of_eq
  {a b : Coeffs}
  (hSize : a.size = b.size) :
  (coeffSub a b).size = a.size := by
  unfold coeffSub
  simp [hSize]

/-- Tiny sanity checks matching Definition 3 behavior. -/
def normSanity : Bool :=
  let x : F := F.ofNat 3
  let y : F := F.ofNat (q - 1)
  decide (normInfF x = 3 ∧ normInfF y = 1)

def normSanityProp : Prop :=
  let x : F := F.ofNat 3
  let y : F := F.ofNat (q - 1)
  normInfF x = 3 ∧ normInfF y = 1

theorem normSanity_sound (hOk : normSanity = true) : normSanityProp := by
  unfold normSanity at hOk
  simpa [normSanityProp] using (decide_eq_true_eq.mp hOk)

theorem normInfCoeffs_nonneg (a : Coeffs) : 0 <= normInfCoeffs a :=
  Nat.zero_le (normInfCoeffs a)

theorem normInfCoeffMatrix_nonneg (m : Array Coeffs) : 0 <= normInfCoeffMatrix m :=
  Nat.zero_le (normInfCoeffMatrix m)

theorem normInfCoeffs_le_of_entry_bound
  {a : Coeffs} {B : Nat}
  (hEntry : ∀ i (hi : i < a.size), normInfF (a[i]'hi) ≤ B) :
  normInfCoeffs a ≤ B := by
  unfold normInfCoeffs
  refine Array.foldl_induction (as := a) (motive := fun _ acc => acc ≤ B) ?_ ?_
  · exact Nat.zero_le B
  · intro i acc hAcc
    have hI : normInfF (a[i]) ≤ B := by
      simpa using hEntry i i.2
    exact (Nat.max_le).2 ⟨hAcc, hI⟩

theorem normInfCoeffMatrix_le_of_row_bound
  {m : Array Coeffs} {B : Nat}
  (hRow : ∀ i (hi : i < m.size), normInfCoeffs (m[i]'hi) ≤ B) :
  normInfCoeffMatrix m ≤ B := by
  unfold normInfCoeffMatrix
  refine Array.foldl_induction (as := m) (motive := fun _ acc => acc ≤ B) ?_ ?_
  · exact Nat.zero_le B
  · intro i acc hAcc
    have hI : normInfCoeffs (m[i]) ≤ B := by
      simpa using hRow i i.2
    exact (Nat.max_le).2 ⟨hAcc, hI⟩

theorem normInfCoeffs_le_of_hasRingDegreeShape_and_coeff_bound
  {a : Coeffs} {B : Nat}
  (hShape : hasRingDegreeShape a)
  (hCoeff : ∀ k, k < D → normInfF (a[k]!) ≤ B) :
  normInfCoeffs a ≤ B := by
  apply normInfCoeffs_le_of_entry_bound
  intro i hi
  have hSize : a.size = D := hShape
  have hiD : i < D := by
    simpa [hSize] using (show i < a.size from hi)
  have hIBang : normInfF (a[i]!) ≤ B := hCoeff i hiD
  simpa [hi] using hIBang

theorem normInfCoeffs_entry_le
  {a : Coeffs} {i : Nat}
  (hi : i < a.size) :
  normInfF (a[i]'hi) ≤ normInfCoeffs a := by
  unfold normInfCoeffs
  have hAll :
      ∀ t (ht : t < a.size), normInfF (a[t]!) ≤ a.foldl (fun m x => Nat.max m (normInfF x)) 0 := by
    exact Array.foldl_induction
      (as := a)
      (motive := fun j acc => ∀ t, t < j → normInfF (a[t]!) ≤ acc)
      (h0 := by
        intro t ht
        exact (Nat.not_lt_zero t ht).elim)
      (hf := by
        intro j acc hAcc t ht
        by_cases htj : t < j.1
        · exact Nat.le_trans (hAcc t htj) (Nat.le_max_left _ _)
        · have hle : t ≤ j.1 := Nat.le_of_lt_succ ht
          have hge : j.1 ≤ t := Nat.le_of_not_gt htj
          have hEq : t = j.1 := Nat.le_antisymm hle hge
          subst hEq
          simpa [j.2] using (Nat.le_max_right acc (normInfF (a[j])))
      )
  have hIBang : normInfF (a[i]!) ≤ a.foldl (fun m x => Nat.max m (normInfF x)) 0 := hAll i hi
  simpa [hi] using hIBang

theorem normInfF_ofNat_le_halfQ
  (x : Nat)
  (hx : x % q <= halfQ) :
  normInfF (F.ofNat x) = x % q := by
  unfold normInfF centeredAbsNat
  simp [F.ofNat, hx]

theorem normInfF_ofNat_gt_halfQ
  (x : Nat)
  (hx : halfQ < x % q) :
  normInfF (F.ofNat x) = q - (x % q) := by
  unfold normInfF centeredAbsNat
  have hNotLe : ¬x % q <= halfQ := Nat.not_le.mpr hx
  simp [F.ofNat, hNotLe]

theorem centeredAbsNat_le_q (x : Nat) : centeredAbsNat x ≤ q := by
  unfold centeredAbsNat
  by_cases h : x % q ≤ halfQ
  · have hmod : x % q < q := Nat.mod_lt _ q_pos
    have hle : x % q ≤ q := Nat.le_of_lt hmod
    exact by simpa [h] using hle
  · have hsub : q - (x % q) ≤ q := Nat.sub_le _ _
    simpa [h] using hsub

theorem centeredAbsNat_le_halfQ (x : Nat) : centeredAbsNat x ≤ halfQ := by
  unfold centeredAbsNat
  by_cases h : x % q ≤ halfQ
  · simpa [h] using h
  · have hx : halfQ < x % q := Nat.lt_of_not_ge h
    have hx' : halfQ + 1 ≤ x % q := Nat.succ_le_of_lt hx
    have hsub : q - (x % q) ≤ q - (halfQ + 1) := Nat.sub_le_sub_left hx' q
    simpa [h, q_sub_halfQ_succ_eq_halfQ] using hsub

theorem normInfF_le_q (a : F) : normInfF a ≤ q := by
  unfold normInfF
  exact centeredAbsNat_le_q a.val

theorem normInfF_le_halfQ (a : F) : normInfF a ≤ halfQ := by
  unfold normInfF
  exact centeredAbsNat_le_halfQ a.val

theorem normInfCoeffs_le_q (a : Coeffs) : normInfCoeffs a ≤ q := by
  exact normInfCoeffs_le_of_entry_bound (a := a) (B := q) (fun i hi => by
    simpa using normInfF_le_q (a[i]'hi))

theorem normInfCoeffs_le_halfQ (a : Coeffs) : normInfCoeffs a ≤ halfQ := by
  exact normInfCoeffs_le_of_entry_bound (a := a) (B := halfQ) (fun i hi => by
    simpa using normInfF_le_halfQ (a[i]'hi))

theorem normInfCoeffMatrix_le_q (m : Array Coeffs) : normInfCoeffMatrix m ≤ q := by
  exact normInfCoeffMatrix_le_of_row_bound (m := m) (B := q) (fun i hi => by
    simpa using normInfCoeffs_le_q (m[i]'hi))

theorem normInfCoeffMatrix_le_halfQ (m : Array Coeffs) : normInfCoeffMatrix m ≤ halfQ := by
  exact normInfCoeffMatrix_le_of_row_bound (m := m) (B := halfQ) (fun i hi => by
    simpa using normInfCoeffs_le_halfQ (m[i]'hi))

theorem normInfF_add_le_q (x y : F) : normInfF (x + y) ≤ q := by
  simpa using normInfF_le_q (x + y)

theorem normInfF_add_le_halfQ (x y : F) : normInfF (x + y) ≤ halfQ := by
  simpa using normInfF_le_halfQ (x + y)

theorem normInfF_sub_le_q (x y : F) : normInfF (x - y) ≤ q := by
  simpa using normInfF_le_q (x - y)

theorem normInfF_sub_le_halfQ (x y : F) : normInfF (x - y) ≤ halfQ := by
  simpa using normInfF_le_halfQ (x - y)

theorem normInfF_mul_le_q (x y : F) : normInfF (x * y) ≤ q := by
  simpa using normInfF_le_q (x * y)

theorem normInfF_mul_le_halfQ (x y : F) : normInfF (x * y) ≤ halfQ := by
  simpa using normInfF_le_halfQ (x * y)

theorem normInfCoeffs_vecAdd_le_q (a b : Coeffs) : normInfCoeffs (vecAdd a b) ≤ q := by
  simpa using normInfCoeffs_le_q (vecAdd a b)

theorem normInfCoeffs_vecAdd_le_halfQ (a b : Coeffs) : normInfCoeffs (vecAdd a b) ≤ halfQ := by
  simpa using normInfCoeffs_le_halfQ (vecAdd a b)

theorem normInfCoeffs_vecScale_le_q (s : F) (a : Coeffs) : normInfCoeffs (vecScale s a) ≤ q := by
  simpa using normInfCoeffs_le_q (vecScale s a)

theorem normInfCoeffs_vecScale_le_halfQ (s : F) (a : Coeffs) : normInfCoeffs (vecScale s a) ≤ halfQ := by
  simpa using normInfCoeffs_le_halfQ (vecScale s a)

theorem normInfCoeffs_mulRq_le_q (a b : Coeffs) : normInfCoeffs (mulRq a b) ≤ q := by
  simpa using normInfCoeffs_le_q (mulRq a b)

theorem normInfCoeffs_mulRq_le_halfQ (a b : Coeffs) : normInfCoeffs (mulRq a b) ≤ halfQ := by
  simpa using normInfCoeffs_le_halfQ (mulRq a b)

theorem normInfCoeffs_coeffSub_le_q (a b : Coeffs) : normInfCoeffs (coeffSub a b) ≤ q := by
  simpa using normInfCoeffs_le_q (coeffSub a b)

theorem normInfCoeffs_coeffSub_le_halfQ (a b : Coeffs) : normInfCoeffs (coeffSub a b) ≤ halfQ := by
  simpa using normInfCoeffs_le_halfQ (coeffSub a b)

theorem normInfCoeffs_superneoBarBlock_le_q
  (bar : Array (Array F)) (v : Coeffs) :
  normInfCoeffs (superneoBarBlock bar v) ≤ q := by
  simpa using normInfCoeffs_le_q (superneoBarBlock bar v)

theorem normInfCoeffs_superneoBarBlock_le_halfQ
  (bar : Array (Array F)) (v : Coeffs) :
  normInfCoeffs (superneoBarBlock bar v) ≤ halfQ := by
  simpa using normInfCoeffs_le_halfQ (superneoBarBlock bar v)

theorem normInfCoeffs_barLiftVec_le_q
  (bar : Array (Array F)) (v : Array F) :
  normInfCoeffs (barLiftVec bar v) ≤ q := by
  simpa using normInfCoeffs_le_q (barLiftVec bar v)

theorem normInfCoeffs_barLiftVec_le_halfQ
  (bar : Array (Array F)) (v : Array F) :
  normInfCoeffs (barLiftVec bar v) ≤ halfQ := by
  simpa using normInfCoeffs_le_halfQ (barLiftVec bar v)

theorem normInfCoeffMatrix_barLiftMatrix_le_q
  (bar : Array (Array F)) (m : Array (Array F)) :
  normInfCoeffMatrix (barLiftMatrix bar m) ≤ q := by
  simpa using normInfCoeffMatrix_le_q (barLiftMatrix bar m)

theorem normInfCoeffMatrix_barLiftMatrix_le_halfQ
  (bar : Array (Array F)) (m : Array (Array F)) :
  normInfCoeffMatrix (barLiftMatrix bar m) ≤ halfQ := by
  simpa using normInfCoeffMatrix_le_halfQ (barLiftMatrix bar m)

theorem normInfCoeffs_vecAdd_le_of_entry_bound
  {a b : Coeffs} {B : Nat}
  (hSize : a.size = b.size)
  (hEntry : ∀ i (hiA : i < a.size) (hiB : i < b.size),
      normInfF ((a[i]'hiA) + (b[i]'hiB)) ≤ B) :
  normInfCoeffs (vecAdd a b) ≤ B := by
  apply normInfCoeffs_le_of_entry_bound
  intro i hi
  have hiA : i < a.size := by
    simpa [vecAdd_size_of_eq hSize] using hi
  have hiB : i < b.size := by
    simpa [hSize] using hiA
  have hI : normInfF ((a[i]'hiA) + (b[i]'hiB)) ≤ B := hEntry i hiA hiB
  simpa [vecAdd, hSize, hiA, hiB] using hI

theorem normInfCoeffs_coeffSub_le_of_entry_bound
  {a b : Coeffs} {B : Nat}
  (hSize : a.size = b.size)
  (hEntry : ∀ i (hiA : i < a.size) (hiB : i < b.size),
      normInfF ((a[i]'hiA) - (b[i]'hiB)) ≤ B) :
  normInfCoeffs (coeffSub a b) ≤ B := by
  apply normInfCoeffs_le_of_entry_bound
  intro i hi
  have hiA : i < a.size := by
    simpa [coeffSub_size_of_eq hSize] using hi
  have hiB : i < b.size := by
    simpa [hSize] using hiA
  have hI : normInfF ((a[i]'hiA) - (b[i]'hiB)) ≤ B := hEntry i hiA hiB
  simpa [coeffSub, hSize, hiA, hiB] using hI

theorem normInfCoeffs_vecScale_le_of_entry_bound
  {s : F} {a : Coeffs} {B : Nat}
  (hEntry : ∀ i (hi : i < a.size), normInfF (s * (a[i]'hi)) ≤ B) :
  normInfCoeffs (vecScale s a) ≤ B := by
  apply normInfCoeffs_le_of_entry_bound
  intro i hi
  have hiA : i < a.size := by
    simpa [vecScale_size s a] using hi
  have hI : normInfF (s * (a[i]'hiA)) ≤ B := hEntry i hiA
  simpa [vecScale, hiA] using hI

theorem normInfCoeffs_mulRq_le_of_coeffSpec_bound
  {a b : Coeffs} {B : Nat}
  (hCoeff : ∀ k, k < D → normInfF (mulRqCoeffSpec a b k) ≤ B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  refine normInfCoeffs_le_of_hasRingDegreeShape_and_coeff_bound
    (hShape := hasRingDegreeShape_mulRq a b) ?_
  intro k hk
  have hK : (mulRq a b)[k]! = mulRqCoeffSpec a b k := mulRq_coeff_spec (a := a) (b := b) (k := k) hk
  simpa [hK] using hCoeff k hk

theorem normInfCoeffs_vecAdd_le_of_norm_bounds
  {a b : Coeffs} {BA BB B : Nat}
  (hSize : a.size = b.size)
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hAdd : ∀ x y, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x + y) ≤ B) :
  normInfCoeffs (vecAdd a b) ≤ B := by
  exact normInfCoeffs_vecAdd_le_of_entry_bound hSize (fun i hiA hiB => by
    exact hAdd (a[i]'hiA) (b[i]'hiB)
      (Nat.le_trans (normInfCoeffs_entry_le (a := a) hiA) hA)
      (Nat.le_trans (normInfCoeffs_entry_le (a := b) hiB) hB))

theorem normInfCoeffs_coeffSub_le_of_norm_bounds
  {a b : Coeffs} {BA BB B : Nat}
  (hSize : a.size = b.size)
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hSub : ∀ x y, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x - y) ≤ B) :
  normInfCoeffs (coeffSub a b) ≤ B := by
  exact normInfCoeffs_coeffSub_le_of_entry_bound hSize (fun i hiA hiB => by
    exact hSub (a[i]'hiA) (b[i]'hiB)
      (Nat.le_trans (normInfCoeffs_entry_le (a := a) hiA) hA)
      (Nat.le_trans (normInfCoeffs_entry_le (a := b) hiB) hB))

theorem normInfCoeffs_vecScale_le_of_norm_bounds
  {s : F} {a : Coeffs} {BS BA B : Nat}
  (hS : normInfF s ≤ BS)
  (hA : normInfCoeffs a ≤ BA)
  (hMul : ∀ x y, normInfF x ≤ BS → normInfF y ≤ BA → normInfF (x * y) ≤ B) :
  normInfCoeffs (vecScale s a) ≤ B := by
  exact normInfCoeffs_vecScale_le_of_entry_bound (s := s) (a := a) (B := B) (fun i hi => by
    exact hMul s (a[i]'hi) hS
      (Nat.le_trans (normInfCoeffs_entry_le (a := a) hi) hA))

theorem normInfCoeffs_mulRq_le_of_norm_bounds
  {a b : Coeffs} {BA BB B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hCoeffFromNorm :
    ∀ k, k < D →
      normInfCoeffs a ≤ BA →
      normInfCoeffs b ≤ BB →
      normInfF (mulRqCoeffSpec a b k) ≤ B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact normInfCoeffs_mulRq_le_of_coeffSpec_bound (a := a) (b := b) (B := B) (fun k hk => by
    exact hCoeffFromNorm k hk hA hB)

theorem normInfF_mulRqCoeffSpec_le_of_rawCoeffBound
  {a b : Coeffs} {k : Nat} {BRaw B : Nat}
  (hRaw : ∀ t, normInfF (mulRqRawCoeffSpec a b t) ≤ BRaw)
  (hAddSub : ∀ x y z, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF z ≤ BRaw → normInfF (x + y - z) ≤ B)
  (hSub : ∀ x y, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF (x - y) ≤ B) :
  normInfF (mulRqCoeffSpec a b k) ≤ B := by
  by_cases h25 : k ≤ 25
  · rw [mulRqCoeffSpec_of_le25_raw (a := a) (b := b) (k := k) h25]
    exact hAddSub
      (mulRqRawCoeffSpec a b k)
      (mulRqRawCoeffSpec a b (k + 81))
      (mulRqRawCoeffSpec a b (k + 54))
      (hRaw k)
      (hRaw (k + 81))
      (hRaw (k + 54))
  · by_cases h26 : k = 26
    · rw [mulRqCoeffSpec_of_eq26_raw (a := a) (b := b) (k := k) h26]
      exact hSub
        (mulRqRawCoeffSpec a b k)
        (mulRqRawCoeffSpec a b (k + 54))
        (hRaw k)
        (hRaw (k + 54))
    · have h26le : 26 ≤ k := Nat.succ_le_of_lt (Nat.lt_of_not_ge h25)
      have h26lt : 26 < k := Nat.lt_of_le_of_ne h26le (Ne.symm h26)
      have h27 : 27 ≤ k := Nat.succ_le_of_lt h26lt
      rw [mulRqCoeffSpec_of_ge27_raw (a := a) (b := b) (k := k) h27]
      exact hSub
        (mulRqRawCoeffSpec a b k)
        (mulRqRawCoeffSpec a b (k + 27))
        (hRaw k)
        (hRaw (k + 27))

theorem normInfCoeffs_mulRq_le_of_rawCoeffBound
  {a b : Coeffs} {BRaw B : Nat}
  (hRaw : ∀ t, normInfF (mulRqRawCoeffSpec a b t) ≤ BRaw)
  (hAddSub : ∀ x y z, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF z ≤ BRaw → normInfF (x + y - z) ≤ B)
  (hSub : ∀ x y, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF (x - y) ≤ B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact normInfCoeffs_mulRq_le_of_coeffSpec_bound (a := a) (b := b) (B := B) (fun k hk => by
    exact normInfF_mulRqCoeffSpec_le_of_rawCoeffBound (a := a) (b := b) (k := k)
      hRaw hAddSub hSub)

theorem normInfCoeffs_mulRq_le_of_norm_bounds_via_rawCoeff
  {a b : Coeffs} {BA BB BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawFromNorm :
    ∀ t, normInfCoeffs a ≤ BA → normInfCoeffs b ≤ BB → normInfF (mulRqRawCoeffSpec a b t) ≤ BRaw)
  (hAddSub : ∀ x y z, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF z ≤ BRaw → normInfF (x + y - z) ≤ B)
  (hSub : ∀ x y, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF (x - y) ≤ B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact normInfCoeffs_mulRq_le_of_rawCoeffBound (a := a) (b := b) (BRaw := BRaw) (B := B)
    (hRaw := fun t => hRawFromNorm t hA hB)
    hAddSub hSub

theorem normInfF_add_sub_le_halfQ
  (x y z : F) :
  normInfF (x + y - z) ≤ halfQ := by
  exact normInfF_sub_le_halfQ (x + y) z

/--
Concrete raw-coefficient fallback: in Goldilocks, every field element is centered-bounded
by `halfQ`, so raw schoolbook coefficients satisfy this bound without extra assumptions.
-/
theorem normInfF_mulRqRawCoeffSpec_le_halfQ
  (a b : Coeffs) (t : Nat) :
  normInfF (mulRqRawCoeffSpec a b t) ≤ halfQ := by
  simpa using normInfF_le_halfQ (mulRqRawCoeffSpec a b t)

/--
Assumption-free raw-coefficient bridge for multiplication bounds (coarse concrete bound).
This discharges the `hRawFromNorm`/`hAddSub`/`hSub` obligations internally at `halfQ`.
-/
theorem normInfCoeffs_mulRq_le_of_norm_bounds_via_rawCoeff_halfQ
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB) :
  normInfCoeffs (mulRq a b) ≤ halfQ := by
  exact normInfCoeffs_mulRq_le_of_norm_bounds_via_rawCoeff
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := halfQ) (B := halfQ)
    hA hB
    (hRawFromNorm := fun t _ _ => normInfF_mulRqRawCoeffSpec_le_halfQ a b t)
    (hAddSub := fun x y z _ _ _ => normInfF_add_sub_le_halfQ x y z)
    (hSub := fun x y _ _ => normInfF_sub_le_halfQ x y)

theorem normInfCoeffs_mulRq_le_of_norm_bounds_via_rawCoeff_halfQ_le
  {a b : Coeffs} {BA BB B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hHalfQ : halfQ ≤ B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact Nat.le_trans
    (normInfCoeffs_mulRq_le_of_norm_bounds_via_rawCoeff_halfQ hA hB)
    hHalfQ

/--
Challenge-coefficient predicate used by SuperNeo concrete parameters:
coefficients are centered in `{-2,-1,0,1,2}`.
-/
def IsChallengeCoeff (x : F) : Prop :=
  x = F.ofInt (-2) ∨ x = F.ofInt (-1) ∨ x = 0 ∨ x = 1 ∨ x = 2

instance isChallengeCoeff_decidable (x : F) : Decidable (IsChallengeCoeff x) := by
  unfold IsChallengeCoeff
  infer_instance

theorem normInfF_ofInt_neg_two : normInfF (F.ofInt (-2)) = 2 := by
  native_decide

theorem normInfF_ofInt_neg_one : normInfF (F.ofInt (-1)) = 1 := by
  native_decide

theorem normInfF_zero : normInfF (0 : F) = 0 := by
  native_decide

theorem normInfF_one : normInfF (1 : F) = 1 := by
  native_decide

theorem normInfF_two : normInfF (2 : F) = 2 := by
  native_decide

theorem normInfF_le_two_of_isChallengeCoeff
  {x : F}
  (hx : IsChallengeCoeff x) :
  normInfF x ≤ 2 := by
  rcases hx with hx | hx | hx | hx | hx
  · simpa [hx, normInfF_ofInt_neg_two]
  · simpa [hx, normInfF_ofInt_neg_one]
  · simpa [hx, normInfF_zero]
  · simpa [hx, normInfF_one]
  · simpa [hx, normInfF_two]

theorem normInfF_sub_le_four_of_isChallengeCoeff
  {x y : F}
  (hx : IsChallengeCoeff x)
  (hy : IsChallengeCoeff y) :
  normInfF (x - y) ≤ 4 := by
  rcases hx with hx | hx | hx | hx | hx <;>
  rcases hy with hy | hy | hy | hy | hy <;>
  subst x <;>
  subst y <;>
  native_decide

theorem normInfF_add_le_four_of_isChallengeCoeff
  {x y : F}
  (hx : IsChallengeCoeff x)
  (hy : IsChallengeCoeff y) :
  normInfF (x + y) ≤ 4 := by
  rcases hx with hx | hx | hx | hx | hx <;>
  rcases hy with hy | hy | hy | hy | hy <;>
  subst x <;>
  subst y <;>
  native_decide

theorem normInfF_mul_le_four_of_isChallengeCoeff
  {x y : F}
  (hx : IsChallengeCoeff x)
  (hy : IsChallengeCoeff y) :
  normInfF (x * y) ≤ 4 := by
  rcases hx with hx | hx | hx | hx | hx <;>
  rcases hy with hy | hy | hy | hy | hy <;>
  subst x <;>
  subst y <;>
  native_decide

def AllChallengeCoeffs (a : Coeffs) : Prop :=
  ∀ i (hi : i < a.size), IsChallengeCoeff (a[i]'hi)

instance allChallengeCoeffs_decidable (a : Coeffs) : Decidable (AllChallengeCoeffs a) := by
  unfold AllChallengeCoeffs
  infer_instance

theorem allChallengeCoeffs_of_all
  {a : Coeffs}
  (hAll : a.all (fun x => decide (IsChallengeCoeff x)) = true) :
  AllChallengeCoeffs a := by
  intro i hi
  exact decide_eq_true_eq.mp ((Array.all_eq_true.mp hAll) i hi)

theorem all_eq_true_of_allChallengeCoeffs
  {a : Coeffs}
  (hAll : AllChallengeCoeffs a) :
  a.all (fun x => decide (IsChallengeCoeff x)) = true := by
  apply (Array.all_eq_true).2
  intro i hi
  exact decide_eq_true (hAll i hi)

theorem normInfCoeffs_le_four_of_allChallenge
  {a : Coeffs}
  (hAll : AllChallengeCoeffs a) :
  normInfCoeffs a ≤ 4 := by
  exact normInfCoeffs_le_of_entry_bound (a := a) (B := 4) (fun i hi => by
    have hEntry : normInfF (a[i]'hi) ≤ 2 := normInfF_le_two_of_isChallengeCoeff (hAll i hi)
    exact Nat.le_trans hEntry (by decide))

theorem normInfCoeffMatrix_le_four_of_allChallenge
  {m : Array Coeffs}
  (hAll : ∀ i (hi : i < m.size), AllChallengeCoeffs (m[i]'hi)) :
  normInfCoeffMatrix m ≤ 4 := by
  exact normInfCoeffMatrix_le_of_row_bound (m := m) (B := 4) (fun i hi => by
    simpa using normInfCoeffs_le_four_of_allChallenge (hAll i hi))

theorem normInfCoeffs_le_four_of_allChallenge_sub
  {a b : Coeffs}
  (hSize : a.size = b.size)
  (hAllA : AllChallengeCoeffs a)
  (hAllB : AllChallengeCoeffs b) :
  normInfCoeffs (coeffSub a b) ≤ 4 := by
  apply normInfCoeffs_le_of_entry_bound
  intro i hi
  have hiA : i < a.size := by
    simpa [coeffSub_size_of_eq hSize] using hi
  have hiB : i < b.size := by
    simpa [hSize] using hiA
  have hA : IsChallengeCoeff (a[i]'hiA) := hAllA i hiA
  have hB : IsChallengeCoeff (b[i]'hiB) := hAllB i hiB
  have hSub : normInfF ((a[i]'hiA) - (b[i]'hiB)) ≤ 4 :=
    normInfF_sub_le_four_of_isChallengeCoeff hA hB
  simpa [coeffSub, hSize, hiA, hiB] using hSub

theorem normInfCoeffs_le_four_of_allChallenge_add
  {a b : Coeffs}
  (hSize : a.size = b.size)
  (hAllA : AllChallengeCoeffs a)
  (hAllB : AllChallengeCoeffs b) :
  normInfCoeffs (vecAdd a b) ≤ 4 := by
  apply normInfCoeffs_le_of_entry_bound
  intro i hi
  have hiA : i < a.size := by
    simpa [vecAdd_size_of_eq hSize] using hi
  have hiB : i < b.size := by
    simpa [hSize] using hiA
  have hA : IsChallengeCoeff (a[i]'hiA) := hAllA i hiA
  have hB : IsChallengeCoeff (b[i]'hiB) := hAllB i hiB
  have hAdd : normInfF ((a[i]'hiA) + (b[i]'hiB)) ≤ 4 :=
    normInfF_add_le_four_of_isChallengeCoeff hA hB
  simpa [vecAdd, hSize, hiA, hiB] using hAdd

theorem normInfCoeffs_le_four_of_allChallenge_scale
  {s : F} {a : Coeffs}
  (hS : IsChallengeCoeff s)
  (hAllA : AllChallengeCoeffs a) :
  normInfCoeffs (vecScale s a) ≤ 4 := by
  apply normInfCoeffs_le_of_entry_bound
  intro i hi
  have hiA : i < a.size := by
    simpa [vecScale_size s a] using hi
  have hA : IsChallengeCoeff (a[i]'hiA) := hAllA i hiA
  have hMul : normInfF (s * (a[i]'hiA)) ≤ 4 :=
    normInfF_mul_le_four_of_isChallengeCoeff hS hA
  simpa [vecScale, hiA] using hMul

theorem normInfF_le_of_halfQ_le
  {x : F} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfF x ≤ B := by
  exact Nat.le_trans (normInfF_le_halfQ x) hB

theorem normInfCoeffs_le_of_halfQ_le
  {a : Coeffs} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfCoeffs a ≤ B := by
  exact Nat.le_trans (normInfCoeffs_le_halfQ a) hB

theorem normInfCoeffMatrix_le_of_halfQ_le
  {m : Array Coeffs} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfCoeffMatrix m ≤ B := by
  exact Nat.le_trans (normInfCoeffMatrix_le_halfQ m) hB

theorem normInfF_add_le_of_halfQ_le
  {x y : F} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfF (x + y) ≤ B := by
  exact Nat.le_trans (normInfF_add_le_halfQ x y) hB

theorem normInfF_sub_le_of_halfQ_le
  {x y : F} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfF (x - y) ≤ B := by
  exact Nat.le_trans (normInfF_sub_le_halfQ x y) hB

theorem normInfF_mul_le_of_halfQ_le
  {x y : F} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfF (x * y) ≤ B := by
  exact Nat.le_trans (normInfF_mul_le_halfQ x y) hB

theorem normInfCoeffs_vecAdd_le_of_halfQ_le
  {a b : Coeffs} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfCoeffs (vecAdd a b) ≤ B := by
  exact Nat.le_trans (normInfCoeffs_vecAdd_le_halfQ a b) hB

theorem normInfCoeffs_vecScale_le_of_halfQ_le
  {s : F} {a : Coeffs} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfCoeffs (vecScale s a) ≤ B := by
  exact Nat.le_trans (normInfCoeffs_vecScale_le_halfQ s a) hB

theorem normInfCoeffs_mulRq_le_of_halfQ_le
  {a b : Coeffs} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact Nat.le_trans (normInfCoeffs_mulRq_le_halfQ a b) hB

theorem normInfCoeffs_coeffSub_le_of_halfQ_le
  {a b : Coeffs} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfCoeffs (coeffSub a b) ≤ B := by
  exact Nat.le_trans (normInfCoeffs_coeffSub_le_halfQ a b) hB

theorem normInfCoeffs_superneoBarBlock_le_of_halfQ_le
  {bar : Array (Array F)} {v : Coeffs} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfCoeffs (superneoBarBlock bar v) ≤ B := by
  exact Nat.le_trans (normInfCoeffs_superneoBarBlock_le_halfQ bar v) hB

theorem normInfCoeffs_barLiftVec_le_of_halfQ_le
  {bar : Array (Array F)} {v : Array F} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfCoeffs (barLiftVec bar v) ≤ B := by
  exact Nat.le_trans (normInfCoeffs_barLiftVec_le_halfQ bar v) hB

theorem normInfCoeffMatrix_barLiftMatrix_le_of_halfQ_le
  {bar : Array (Array F)} {m : Array (Array F)} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfCoeffMatrix (barLiftMatrix bar m) ≤ B := by
  exact Nat.le_trans (normInfCoeffMatrix_barLiftMatrix_le_halfQ bar m) hB

theorem normInfF_le_B_of_isChallengeCoeff
  {x : F} {B : Nat}
  (hB : 2 ≤ B)
  (hx : IsChallengeCoeff x) :
  normInfF x ≤ B := by
  exact Nat.le_trans (normInfF_le_two_of_isChallengeCoeff hx) hB

theorem normInfF_add_le_B_of_isChallengeCoeff
  {x y : F} {B : Nat}
  (hB : 4 ≤ B)
  (hx : IsChallengeCoeff x)
  (hy : IsChallengeCoeff y) :
  normInfF (x + y) ≤ B := by
  exact Nat.le_trans (normInfF_add_le_four_of_isChallengeCoeff hx hy) hB

theorem normInfF_sub_le_B_of_isChallengeCoeff
  {x y : F} {B : Nat}
  (hB : 4 ≤ B)
  (hx : IsChallengeCoeff x)
  (hy : IsChallengeCoeff y) :
  normInfF (x - y) ≤ B := by
  exact Nat.le_trans (normInfF_sub_le_four_of_isChallengeCoeff hx hy) hB

theorem normInfF_mul_le_B_of_isChallengeCoeff
  {x y : F} {B : Nat}
  (hB : 4 ≤ B)
  (hx : IsChallengeCoeff x)
  (hy : IsChallengeCoeff y) :
  normInfF (x * y) ≤ B := by
  exact Nat.le_trans (normInfF_mul_le_four_of_isChallengeCoeff hx hy) hB

theorem normInfCoeffs_le_B_of_allChallenge
  {a : Coeffs} {B : Nat}
  (hB : 4 ≤ B)
  (hAll : AllChallengeCoeffs a) :
  normInfCoeffs a ≤ B := by
  exact Nat.le_trans (normInfCoeffs_le_four_of_allChallenge hAll) hB

theorem normInfCoeffs_le_B_of_allChallenge_add
  {a b : Coeffs} {B : Nat}
  (hB : 4 ≤ B)
  (hSize : a.size = b.size)
  (hAllA : AllChallengeCoeffs a)
  (hAllB : AllChallengeCoeffs b) :
  normInfCoeffs (vecAdd a b) ≤ B := by
  exact Nat.le_trans (normInfCoeffs_le_four_of_allChallenge_add hSize hAllA hAllB) hB

theorem normInfCoeffs_le_B_of_allChallenge_sub
  {a b : Coeffs} {B : Nat}
  (hB : 4 ≤ B)
  (hSize : a.size = b.size)
  (hAllA : AllChallengeCoeffs a)
  (hAllB : AllChallengeCoeffs b) :
  normInfCoeffs (coeffSub a b) ≤ B := by
  exact Nat.le_trans (normInfCoeffs_le_four_of_allChallenge_sub hSize hAllA hAllB) hB

theorem normInfCoeffs_le_B_of_allChallenge_scale
  {s : F} {a : Coeffs} {B : Nat}
  (hB : 4 ≤ B)
  (hS : IsChallengeCoeff s)
  (hAllA : AllChallengeCoeffs a) :
  normInfCoeffs (vecScale s a) ≤ B := by
  exact Nat.le_trans (normInfCoeffs_le_four_of_allChallenge_scale hS hAllA) hB

end SuperNeo
