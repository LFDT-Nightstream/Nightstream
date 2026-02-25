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

end SuperNeo
