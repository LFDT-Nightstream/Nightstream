import SuperNeo.Goldilocks
import Init.GrindInstances.Ring.Fin

namespace SuperNeo

/-- Base field carrier for SuperNeo (`F_q`). -/
abbrev F : Type := Fin Goldilocks.q

instance : NeZero Goldilocks.q := ⟨Nat.ne_of_gt Goldilocks.q_pos⟩

namespace F

instance : Inhabited F := ⟨⟨0, Goldilocks.q_pos⟩⟩

def ofNat (n : Nat) : F :=
  ⟨n % Goldilocks.q, Nat.mod_lt _ Goldilocks.q_pos⟩

def zero : F := ofNat 0
def one : F := ofNat 1

instance : Zero F := ⟨zero⟩
instance : One F := ⟨one⟩

def pow (a : F) (n : Nat) : F :=
  Id.run do
    let mut acc : F := 1
    let mut base := a
    let mut exp := n
    while exp > 0 do
      if exp % 2 = 1 then
        acc := acc * base
      base := base * base
      exp := exp / 2
    return acc

def inv (a : F) : F :=
  if a.val = 0 then
    0
  else
    pow a (Goldilocks.q - 2)

/-- Canonical representative in `[0, q)`. -/
def canonicalRep (a : F) : Nat := a.val

/-- Canonicality predicate (always true for the `Fin` encoding). -/
def isCanonical (a : F) : Prop := canonicalRep a < Goldilocks.q

instance (a : F) : Decidable (isCanonical a) := by
  unfold isCanonical canonicalRep
  infer_instance

theorem canonical (a : F) : isCanonical a :=
  a.isLt

def canonicalCheck (a : F) : Bool :=
  decide (isCanonical a)

theorem canonicalCheck_true (a : F) : canonicalCheck a = true := by
  unfold canonicalCheck
  exact decide_eq_true (canonical a)

/-- Centered integer representative in `[-q/2, q/2]` shape. -/
def centeredRep (a : F) : Int :=
  if _h : a.val ≤ Goldilocks.halfQ then
    Int.ofNat a.val
  else
    Int.ofNat a.val - Int.ofNat Goldilocks.q

def centeredAbs (a : F) : Nat :=
  Int.natAbs (centeredRep a)

private def distToZero (x : Nat) : Nat :=
  if x ≤ Goldilocks.halfQ then x else Goldilocks.q - x

private theorem q_eq_two_mul_halfQ_add_one :
    Goldilocks.q = 2 * Goldilocks.halfQ + 1 := by
  decide

private theorem centeredAbs_eq_distToZero (a : F) :
    centeredAbs a = distToZero a.val := by
  unfold centeredAbs distToZero
  by_cases h : a.val ≤ Goldilocks.halfQ
  · simp [centeredRep, h]
  · have hqle : a.val ≤ Goldilocks.q := Nat.le_of_lt a.isLt
    have hsub : Int.ofNat a.val - Int.ofNat Goldilocks.q = - Int.ofNat (Goldilocks.q - a.val) := by
      calc
        Int.ofNat a.val - Int.ofNat Goldilocks.q = -(Int.ofNat Goldilocks.q - Int.ofNat a.val) := by omega
        _ = - Int.ofNat (Goldilocks.q - a.val) := by
          congr
          exact (Int.ofNat_sub hqle).symm
    calc
      (if h : a.val ≤ Goldilocks.halfQ then Int.ofNat a.val else Int.ofNat a.val - Int.ofNat Goldilocks.q).natAbs
          = (Int.ofNat a.val - Int.ofNat Goldilocks.q).natAbs := by simp [h]
      _ = (- Int.ofNat (Goldilocks.q - a.val)).natAbs := by rw [hsub]
      _ = Goldilocks.q - a.val := by simp
      _ = (if a.val ≤ Goldilocks.halfQ then a.val else Goldilocks.q - a.val) := by simp [h]

private theorem distToZero_le_self_of_lt_q {x : Nat} (hx : x < Goldilocks.q) :
    distToZero x ≤ x := by
  unfold distToZero
  by_cases h : x ≤ Goldilocks.halfQ
  · simp [h]
  · have hgt : Goldilocks.halfQ < x := Nat.lt_of_not_ge h
    have hqle2x : Goldilocks.q ≤ 2 * x := by
      rw [q_eq_two_mul_halfQ_add_one]
      omega
    have hqmx_le : Goldilocks.q - x ≤ x := by
      omega
    simp [h, hqmx_le]

private theorem distToZero_le_compl_of_lt_q {x : Nat} (hx : x < Goldilocks.q) :
    distToZero x ≤ Goldilocks.q - x := by
  unfold distToZero
  by_cases h : x ≤ Goldilocks.halfQ
  · have h2xle : 2 * x ≤ Goldilocks.q := by
      rw [q_eq_two_mul_halfQ_add_one]
      omega
    have hxle : x ≤ Goldilocks.q - x := by
      omega
    simp [h, hxle]
  · simp [h]

private theorem distToZero_sub_triangle
    (x y : Nat)
    (hx : x < Goldilocks.q)
    (hy : y < Goldilocks.q) :
    distToZero ((x + Goldilocks.q - y) % Goldilocks.q) ≤
      distToZero x + distToZero y := by
  by_cases hxy : y ≤ x
  · have hmod : ((x + Goldilocks.q - y) % Goldilocks.q) = x - y := by
      have hdecomp : x + Goldilocks.q - y = Goldilocks.q + (x - y) := by omega
      calc
        ((x + Goldilocks.q - y) % Goldilocks.q)
            = ((Goldilocks.q + (x - y)) % Goldilocks.q) := by simp [hdecomp]
        _ = (x - y) % Goldilocks.q := by simpa [Nat.add_comm] using (Nat.add_mod_right (x - y) Goldilocks.q)
        _ = x - y := by
          have hlt : x - y < Goldilocks.q := Nat.lt_of_le_of_lt (Nat.sub_le _ _) hx
          simp [Nat.mod_eq_of_lt hlt]
    by_cases hxh : x ≤ Goldilocks.halfQ
    · have hyh : y ≤ Goldilocks.halfQ := Nat.le_trans hxy hxh
      have hleft : distToZero (x - y) ≤ x - y :=
        distToZero_le_self_of_lt_q (Nat.lt_of_le_of_lt (Nat.sub_le _ _) hx)
      have hsuble : x - y ≤ x + y := by omega
      have hxDist : distToZero x = x := by simp [distToZero, hxh]
      have hyDist : distToZero y = y := by simp [distToZero, hyh]
      calc
        distToZero ((x + Goldilocks.q - y) % Goldilocks.q)
            = distToZero (x - y) := by simp [hmod]
        _ ≤ x - y := hleft
        _ ≤ x + y := hsuble
        _ = distToZero x + distToZero y := by simp [hxDist, hyDist]
    · by_cases hyh : y ≤ Goldilocks.halfQ
      · have hright : distToZero (x - y) ≤ Goldilocks.q - (x - y) := by
          apply distToZero_le_compl_of_lt_q
          exact Nat.lt_of_le_of_lt (Nat.sub_le _ _) hx
        have hxDist : distToZero x = Goldilocks.q - x := by simp [distToZero, hxh]
        have hyDist : distToZero y = y := by simp [distToZero, hyh]
        calc
          distToZero ((x + Goldilocks.q - y) % Goldilocks.q)
              = distToZero (x - y) := by simp [hmod]
          _ ≤ Goldilocks.q - (x - y) := hright
          _ = (Goldilocks.q - x) + y := by omega
          _ = distToZero x + distToZero y := by simp [hxDist, hyDist]
      · have hleft : distToZero (x - y) ≤ x - y :=
          distToZero_le_self_of_lt_q (Nat.lt_of_le_of_lt (Nat.sub_le _ _) hx)
        have hsum : x - y ≤ (Goldilocks.q - x) + (Goldilocks.q - y) := by
          omega
        have hxDist : distToZero x = Goldilocks.q - x := by simp [distToZero, hxh]
        have hyDist : distToZero y = Goldilocks.q - y := by simp [distToZero, hyh]
        calc
          distToZero ((x + Goldilocks.q - y) % Goldilocks.q)
              = distToZero (x - y) := by simp [hmod]
          _ ≤ x - y := hleft
          _ ≤ (Goldilocks.q - x) + (Goldilocks.q - y) := hsum
          _ = distToZero x + distToZero y := by simp [hxDist, hyDist]
  · have hxygt : x < y := Nat.lt_of_not_ge hxy
    have hmod : ((x + Goldilocks.q - y) % Goldilocks.q) = Goldilocks.q + x - y := by
      have hlt : x + Goldilocks.q - y < Goldilocks.q := by omega
      have hEq : x + Goldilocks.q - y = Goldilocks.q + x - y := by omega
      calc
        ((x + Goldilocks.q - y) % Goldilocks.q) = (x + Goldilocks.q - y) := by simp [Nat.mod_eq_of_lt hlt]
        _ = Goldilocks.q + x - y := hEq
    by_cases hxh : x ≤ Goldilocks.halfQ
    · by_cases hyh : y ≤ Goldilocks.halfQ
      · have hright : distToZero (Goldilocks.q + x - y) ≤ Goldilocks.q - (Goldilocks.q + x - y) := by
          apply distToZero_le_compl_of_lt_q
          have hlt : Goldilocks.q + x - y < Goldilocks.q := by omega
          exact hlt
        have hxDist : distToZero x = x := by simp [distToZero, hxh]
        have hyDist : distToZero y = y := by simp [distToZero, hyh]
        have hsum : Goldilocks.q - (Goldilocks.q + x - y) ≤ x + y := by
          omega
        calc
          distToZero ((x + Goldilocks.q - y) % Goldilocks.q)
              = distToZero (Goldilocks.q + x - y) := by simp [hmod]
          _ ≤ Goldilocks.q - (Goldilocks.q + x - y) := hright
          _ ≤ x + y := hsum
          _ = distToZero x + distToZero y := by simp [hxDist, hyDist]
      · have hleft : distToZero (Goldilocks.q + x - y) ≤ Goldilocks.q + x - y := by
          apply distToZero_le_self_of_lt_q
          have hlt : Goldilocks.q + x - y < Goldilocks.q := by omega
          exact hlt
        have hxDist : distToZero x = x := by simp [distToZero, hxh]
        have hyDist : distToZero y = Goldilocks.q - y := by simp [distToZero, hyh]
        calc
          distToZero ((x + Goldilocks.q - y) % Goldilocks.q)
              = distToZero (Goldilocks.q + x - y) := by simp [hmod]
          _ ≤ Goldilocks.q + x - y := hleft
          _ = x + (Goldilocks.q - y) := by omega
          _ = distToZero x + distToZero y := by simp [hxDist, hyDist]
    · have hyh : ¬ y ≤ Goldilocks.halfQ := by
        intro hyle
        exact hxh (Nat.le_trans (Nat.le_of_lt hxygt) hyle)
      have hright : distToZero (Goldilocks.q + x - y) ≤ Goldilocks.q - (Goldilocks.q + x - y) := by
        apply distToZero_le_compl_of_lt_q
        have hlt : Goldilocks.q + x - y < Goldilocks.q := by omega
        exact hlt
      have hsum : Goldilocks.q - (Goldilocks.q + x - y) ≤ (Goldilocks.q - x) + (Goldilocks.q - y) := by
        omega
      have hxDist : distToZero x = Goldilocks.q - x := by simp [distToZero, hxh]
      have hyDist : distToZero y = Goldilocks.q - y := by simp [distToZero, hyh]
      calc
        distToZero ((x + Goldilocks.q - y) % Goldilocks.q)
            = distToZero (Goldilocks.q + x - y) := by simp [hmod]
        _ ≤ Goldilocks.q - (Goldilocks.q + x - y) := hright
        _ ≤ (Goldilocks.q - x) + (Goldilocks.q - y) := hsum
        _ = distToZero x + distToZero y := by simp [hxDist, hyDist]

theorem centeredRep_eq_of_le_halfQ {a : F} (h : a.val ≤ Goldilocks.halfQ) :
    centeredRep a = Int.ofNat a.val := by
  simp [centeredRep, h]

theorem centeredRep_eq_sub_q_of_halfQ_lt {a : F} (h : Goldilocks.halfQ < a.val) :
    centeredRep a = Int.ofNat a.val - Int.ofNat Goldilocks.q := by
  have hNot : ¬ a.val ≤ Goldilocks.halfQ := Nat.not_le_of_lt h
  simp [centeredRep, hNot]

@[simp] theorem ofNat_zero : ofNat 0 = (0 : F) := rfl

@[simp] theorem ofNat_one : ofNat 1 = (1 : F) := rfl

@[simp] theorem val_zero : (0 : F).val = 0 := by
  change 0 % Goldilocks.q = 0
  exact Nat.mod_eq_of_lt Goldilocks.q_pos

@[simp] theorem val_one : (1 : F).val = 1 := by
  change 1 % Goldilocks.q = 1
  exact Nat.mod_eq_of_lt Goldilocks.q_gt_one

@[simp] theorem ofNat_val (a : F) : ofNat a.val = a := by
  apply Fin.ext
  simp [ofNat, Nat.mod_eq_of_lt a.isLt]

@[simp] theorem canonicalRep_ofNat (n : Nat) :
    canonicalRep (ofNat n) = n % Goldilocks.q := rfl

@[simp] theorem val_lt_q (a : F) : a.val < Goldilocks.q :=
  a.isLt

@[simp] theorem canonicalRep_eq_val (a : F) :
    canonicalRep a = a.val := rfl

theorem ofNat_canonicalRep (a : F) :
    ofNat (canonicalRep a) = a := by
  simp [canonicalRep]

theorem ofNat_val_eq_of_canonical
    {n : Nat}
    (h : n < Goldilocks.q) :
    (ofNat n).val = n := by
  simp [ofNat, Nat.mod_eq_of_lt h]

theorem canonicalRep_ofNat_eq_of_lt
    {n : Nat}
    (h : n < Goldilocks.q) :
    canonicalRep (ofNat n) = n := by
  simpa [canonicalRep] using ofNat_val_eq_of_canonical h

@[simp] theorem isCanonical_true (a : F) :
    isCanonical a := canonical a

theorem canonicalCheck_iff (a : F) :
    canonicalCheck a = true ↔ isCanonical a := by
  unfold canonicalCheck
  constructor
  · exact decide_eq_true_eq.mp
  · exact decide_eq_true

@[simp] theorem canonicalRep_zero : canonicalRep (0 : F) = 0 := by
  simp [canonicalRep]

@[simp] theorem canonicalRep_one : canonicalRep (1 : F) = 1 := by
  simp [canonicalRep]

theorem isCanonical_zero : isCanonical (0 : F) := by
  exact canonical (0 : F)

theorem isCanonical_one : isCanonical (1 : F) := by
  exact canonical (1 : F)

@[simp] theorem centeredRep_zero : centeredRep (0 : F) = 0 := by
  have h : ((0 : F).val) ≤ Goldilocks.halfQ := by
    simp
  simpa [h] using centeredRep_eq_of_le_halfQ h

@[simp] theorem centeredAbs_zero : centeredAbs (0 : F) = 0 := by
  simp [centeredAbs, centeredRep_zero]

theorem centeredAbs_eq_zero_iff (a : F) :
    centeredAbs a = 0 ↔ a = 0 := by
  constructor
  · intro hAbs
    apply Fin.ext
    by_cases hHalf : a.val ≤ Goldilocks.halfQ
    · have hRep := centeredRep_eq_of_le_halfQ hHalf
      have hNatAbs : (Int.ofNat a.val).natAbs = 0 := by
        simpa [centeredAbs, hRep] using hAbs
      have hVal : a.val = 0 := by
        simpa using hNatAbs
      simpa [hVal]
    · have hGt : Goldilocks.halfQ < a.val := Nat.lt_of_not_ge hHalf
      have hRep := centeredRep_eq_sub_q_of_halfQ_lt hGt
      have hNatAbs : (Int.ofNat a.val - Int.ofNat Goldilocks.q).natAbs = 0 := by
        simpa [centeredAbs, hRep] using hAbs
      have hInt : Int.ofNat a.val - Int.ofNat Goldilocks.q = 0 := by
        exact Int.natAbs_eq_zero.mp hNatAbs
      have hEqInt : Int.ofNat a.val = Int.ofNat Goldilocks.q := by
        omega
      have hVal : a.val = Goldilocks.q := by
        exact Int.ofNat.inj hEqInt
      exact False.elim (Nat.ne_of_lt a.isLt hVal)
  · intro hEq
    exact hEq ▸ centeredAbs_zero

@[simp] theorem centeredRep_one : centeredRep (1 : F) = 1 := by
  have h : ((1 : F).val) ≤ Goldilocks.halfQ := by
    exact Goldilocks.one_le_halfQ
  calc
    centeredRep (1 : F) = Int.ofNat ((1 : F).val) := centeredRep_eq_of_le_halfQ h
    _ = 1 := by simp

/--
Total case split for centered representatives: every canonical field value
falls in exactly one of the two centered-representation branches.
-/
theorem centeredRep_cases (a : F) :
    (a.val ≤ Goldilocks.halfQ ∧ centeredRep a = Int.ofNat a.val) ∨
    (Goldilocks.halfQ < a.val ∧ centeredRep a = Int.ofNat a.val - Int.ofNat Goldilocks.q) := by
  by_cases h : a.val ≤ Goldilocks.halfQ
  · exact Or.inl ⟨h, centeredRep_eq_of_le_halfQ h⟩
  · have hlt : Goldilocks.halfQ < a.val := Nat.lt_of_not_ge h
    exact Or.inr ⟨hlt, centeredRep_eq_sub_q_of_halfQ_lt hlt⟩

/-- Non-dependent branch-erased centered-representation form. -/
theorem centeredRep_cover (a : F) :
    centeredRep a = Int.ofNat a.val ∨
      centeredRep a = Int.ofNat a.val - Int.ofNat Goldilocks.q := by
  rcases centeredRep_cases a with hL | hR
  · exact Or.inl hL.2
  · exact Or.inr hR.2

@[simp] theorem val_add (a b : F) :
    (a + b).val = (a.val + b.val) % Goldilocks.q := rfl

@[simp] theorem val_sub (a b : F) :
    (a - b).val = (a.val + Goldilocks.q - b.val) % Goldilocks.q := by
  have hEq : Goldilocks.q + a.val - b.val = a.val + (Goldilocks.q - b.val) := by
    omega
  calc
    (a - b).val = (a.val + (Goldilocks.q - b.val)) % Goldilocks.q := by
      simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using (Fin.val_sub a b)
    _ = (Goldilocks.q + a.val - b.val) % Goldilocks.q := by
      simp [hEq]
    _ = (a.val + Goldilocks.q - b.val) % Goldilocks.q := by
      simp [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]

@[simp] theorem val_neg (a : F) :
    (-a).val = (Goldilocks.q - a.val) % Goldilocks.q := rfl

theorem centeredAbs_sub_le (a b : F) :
    centeredAbs (a - b) ≤ centeredAbs a + centeredAbs b := by
  have hSub :
      centeredAbs (a - b) =
        distToZero ((a.val + Goldilocks.q - b.val) % Goldilocks.q) := by
    rw [centeredAbs_eq_distToZero]
    simpa [val_sub]
  calc
    centeredAbs (a - b)
        = distToZero ((a.val + Goldilocks.q - b.val) % Goldilocks.q) := hSub
    _ ≤ distToZero a.val + distToZero b.val :=
          distToZero_sub_triangle a.val b.val a.isLt b.isLt
    _ = centeredAbs a + centeredAbs b := by
          simp [centeredAbs_eq_distToZero]

private theorem zero_sub_eq_neg (a : F) : (0 : F) - a = -a := by
  apply Fin.ext
  simp [val_sub, val_neg]

private theorem neg_neg_eq (a : F) : -(-a) = a := by
  apply Fin.ext
  by_cases h0 : a.val = 0
  · simp [h0, val_neg]
  · have hpos : 0 < a.val := Nat.pos_of_ne_zero h0
    have hqsub_lt : Goldilocks.q - a.val < Goldilocks.q := Nat.sub_lt Goldilocks.q_pos hpos
    calc
      (-(-a)).val
          = (Goldilocks.q - ((Goldilocks.q - a.val) % Goldilocks.q)) % Goldilocks.q := by
              simp [val_neg]
      _ = (Goldilocks.q - (Goldilocks.q - a.val)) % Goldilocks.q := by
            simp [Nat.mod_eq_of_lt hqsub_lt]
      _ = a.val % Goldilocks.q := by
            have hsub : Goldilocks.q - (Goldilocks.q - a.val) = a.val := by omega
            simp [hsub]
      _ = a.val := Nat.mod_eq_of_lt a.isLt

theorem centeredAbs_neg_eq (a : F) :
    centeredAbs (-a) = centeredAbs a := by
  apply Nat.le_antisymm
  · have h := centeredAbs_sub_le (0 : F) a
    simpa [zero_sub_eq_neg, centeredAbs_zero] using h
  · have h := centeredAbs_sub_le (0 : F) (-a)
    have h0 : (0 : F) - (-a) = a := by
      calc
        (0 : F) - (-a) = -(-a) := by simpa using zero_sub_eq_neg (-a)
        _ = a := neg_neg_eq a
    simpa [h0, centeredAbs_zero] using h

theorem centeredAbs_add_le (a b : F) :
    centeredAbs (a + b) ≤ centeredAbs a + centeredAbs b := by
  have h := centeredAbs_sub_le a (-b)
  have hSub :
      a - (-b) = a + b := by
    calc
      a - (-b) = a + -(-b) := by
        simpa using (Lean.Grind.Fin.sub_eq_add_neg (n := Goldilocks.q) a (-b))
      _ = a + b := by
            congr
            exact neg_neg_eq b
  simpa [hSub, centeredAbs_neg_eq] using h

@[simp] theorem val_mul (a b : F) :
    (a * b).val = (a.val * b.val) % Goldilocks.q := rfl

theorem canonicalRep_add (a b : F) :
    canonicalRep (a + b) = (a.val + b.val) % Goldilocks.q := by
  show (a + b).val = (a.val + b.val) % Goldilocks.q
  exact val_add a b

theorem canonicalRep_mul (a b : F) :
    canonicalRep (a * b) = (a.val * b.val) % Goldilocks.q := by
  show (a * b).val = (a.val * b.val) % Goldilocks.q
  exact val_mul a b

theorem canonicalRep_neg (a : F) :
    canonicalRep (-a) = (Goldilocks.q - a.val) % Goldilocks.q := by
  show (-a).val = (Goldilocks.q - a.val) % Goldilocks.q
  exact val_neg a

@[simp] theorem ofNat_add (m n : Nat) :
    ofNat (m + n) = ofNat m + ofNat n := by
  apply Fin.ext
  simp [ofNat, Nat.add_mod]

@[simp] theorem ofNat_succ (n : Nat) :
    ofNat (n + 1) = ofNat n + 1 := by
  simpa using (ofNat_add n 1)

theorem centeredAbs_mul_ofNat_le (n : Nat) (a : F) :
    centeredAbs (ofNat n * a) ≤ n * centeredAbs a := by
  induction n with
  | zero =>
      have hZero : centeredAbs (ofNat 0 * a) = 0 := by
        calc
          centeredAbs (ofNat 0 * a)
              = centeredAbs ((0 : F) * a) := by simp [ofNat]
          _ = centeredAbs (0 : F) := by
                simp [Lean.Grind.Fin.zero_mul (n := Goldilocks.q) a]
          _ = 0 := by
                simp [centeredAbs, centeredRep_zero]
      simpa [hZero]
  | succ n ih =>
      calc
        centeredAbs (ofNat (n + 1) * a)
            = centeredAbs ((ofNat n + 1) * a) := by simp [ofNat_succ]
        _ = centeredAbs (ofNat n * a + a) := by
              calc
                centeredAbs ((ofNat n + 1) * a)
                    = centeredAbs (a * (ofNat n + 1)) := by
                        simp [Lean.Grind.Fin.mul_comm]
                _ = centeredAbs (a * ofNat n + a * 1) := by
                      simpa using
                        congrArg centeredAbs
                          (Lean.Grind.Fin.left_distrib (n := Goldilocks.q) a (ofNat n) (1 : F))
                _ = centeredAbs (ofNat n * a + a) := by
                      have hMulComm : a * ofNat n = ofNat n * a := by
                        simpa using (Lean.Grind.Fin.mul_comm (n := Goldilocks.q) a (ofNat n))
                      have hMulOne : a * (1 : F) = a := by
                        simpa using (Lean.Grind.Fin.mul_one (n := Goldilocks.q) a)
                      simp [hMulComm, hMulOne]
        _ ≤ centeredAbs (ofNat n * a) + centeredAbs a := centeredAbs_add_le (ofNat n * a) a
        _ ≤ n * centeredAbs a + centeredAbs a := Nat.add_le_add_right ih _
        _ = (n + 1) * centeredAbs a := by
              simp [Nat.succ_mul, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc]

theorem exists_smallNat_or_neg_of_centeredAbs_le
    (B : Nat)
    (x : F)
    (hx : centeredAbs x ≤ B) :
    ∃ n : Nat, n ≤ B ∧ (x = ofNat n ∨ x = - ofNat n) := by
  rcases centeredRep_cases x with hL | hR
  · refine ⟨x.val, ?_, ?_⟩
    · have habs : centeredAbs x = x.val := by
        unfold centeredAbs
        simpa [hL.2]
      simpa [habs] using hx
    · exact Or.inl (by simpa [canonicalRep] using (ofNat_canonicalRep x).symm)
  · let n : Nat := Goldilocks.q - x.val
    have hqle : x.val ≤ Goldilocks.q := Nat.le_of_lt x.isLt
    have hrep : centeredRep x = - Int.ofNat n := by
      unfold n
      calc
        centeredRep x = Int.ofNat x.val - Int.ofNat Goldilocks.q := hR.2
        _ = - (Int.ofNat Goldilocks.q - Int.ofNat x.val) := by omega
        _ = - Int.ofNat (Goldilocks.q - x.val) := by
              congr
              exact (Int.ofNat_sub hqle).symm
    have hnleB : n ≤ B := by
      have hNatAbs : Int.natAbs (- Int.ofNat n) ≤ B := by
        simpa [centeredAbs, hrep]
          using hx
      simpa using hNatAbs
    have hxpos : 0 < x.val := Nat.lt_of_le_of_lt (Nat.zero_le _) hR.1
    have hnlt : n < Goldilocks.q := by
      unfold n
      exact Nat.sub_lt Goldilocks.q_pos hxpos
    have hxEq : x = - ofNat n := by
      apply Fin.ext
      have hmodN : n % Goldilocks.q = n := Nat.mod_eq_of_lt hnlt
      unfold n
      calc
        x.val = Goldilocks.q - (Goldilocks.q - x.val) := by omega
        _ = (Goldilocks.q - ((Goldilocks.q - x.val) % Goldilocks.q)) % Goldilocks.q := by
              have hlt : Goldilocks.q - (Goldilocks.q - x.val) < Goldilocks.q := by omega
              have hsublt : Goldilocks.q - x.val < Goldilocks.q :=
                Nat.sub_lt Goldilocks.q_pos hxpos
              simp [Nat.mod_eq_of_lt hsublt]
              symm
              exact Nat.mod_eq_of_lt hlt
        _ = (- ofNat (Goldilocks.q - x.val)).val := by
              simp [val_neg, ofNat, Nat.mod_eq_of_lt (Nat.sub_lt Goldilocks.q_pos hxpos)]
    exact ⟨n, hnleB, Or.inr hxEq⟩

theorem exists_smallNat_or_neg_of_centeredAbs_le_four
    (x : F) (hx : centeredAbs x ≤ 4) :
    ∃ n : Nat, n ≤ 4 ∧ (x = ofNat n ∨ x = - ofNat n) := by
  exact exists_smallNat_or_neg_of_centeredAbs_le 4 x hx

theorem centeredAbs_mul_le_of_centeredAbs_left_le_four
    (x y : F)
    (hx : centeredAbs x ≤ 4) :
    centeredAbs (x * y) ≤ 4 * centeredAbs y := by
  rcases exists_smallNat_or_neg_of_centeredAbs_le_four x hx with ⟨n, hn, hxEq | hxEq⟩
  · subst hxEq
    exact Nat.le_trans (centeredAbs_mul_ofNat_le n y) (Nat.mul_le_mul_right _ hn)
  · subst hxEq
    have hnegMul : (- ofNat n) * y = - (ofNat n * y) := by
      simpa using (Lean.Grind.Fin.neg_mul (n := Goldilocks.q) (ofNat n) y)
    calc
      centeredAbs ((- ofNat n) * y)
          = centeredAbs (-(ofNat n * y)) := by simp [hnegMul]
      _ = centeredAbs (ofNat n * y) := centeredAbs_neg_eq (ofNat n * y)
      _ ≤ n * centeredAbs y := centeredAbs_mul_ofNat_le n y
      _ ≤ 4 * centeredAbs y := Nat.mul_le_mul_right _ hn

private theorem eq_of_qmul_pos_lt2q (k : Nat)
    (hpos : 0 < Goldilocks.q * k)
    (hlt : Goldilocks.q * k < 2 * Goldilocks.q) : k = 1 := by
  have hkpos : 0 < k := by
    by_cases hk0 : k = 0
    · subst hk0
      simp at hpos
    · exact Nat.pos_of_ne_zero hk0
  have hklt2 : k < 2 := by
    have hmul : k * Goldilocks.q < 2 * Goldilocks.q := by
      simpa [Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using hlt
    exact Nat.lt_of_mul_lt_mul_right hmul
  have hkge1 : 1 ≤ k := Nat.succ_le_of_lt hkpos
  have hkle1 : k ≤ 1 := Nat.lt_succ_iff.mp hklt2
  exact Nat.le_antisymm hkle1 hkge1

/-- Field subtraction cancellation in `F_q`. -/
theorem sub_eq_zero_iff (a b : F) : a - b = 0 ↔ a = b := by
  constructor
  · intro h
    apply Fin.ext
    let x := a.val + Goldilocks.q - b.val
    have hmod : x % Goldilocks.q = 0 := by
      have hv : (a - b).val = 0 := by simpa [h] using (val_zero)
      simpa [x, val_sub] using hv
    have hdiv : Goldilocks.q ∣ x := Nat.dvd_of_mod_eq_zero hmod
    rcases hdiv with ⟨k, hk⟩
    have hlt2q : Goldilocks.q * k < 2 * Goldilocks.q := by
      have ha : a.val < Goldilocks.q := a.isLt
      have hb : b.val < Goldilocks.q := b.isLt
      have hxlt : x < 2 * Goldilocks.q := by
        dsimp [x]
        omega
      simpa [hk] using hxlt
    have hpos : 0 < Goldilocks.q * k := by
      have hb : b.val < Goldilocks.q := b.isLt
      have hxpos : 0 < x := by
        dsimp [x]
        omega
      simpa [hk] using hxpos
    have hk1 : k = 1 := eq_of_qmul_pos_lt2q k hpos hlt2q
    have hx : x = Goldilocks.q := by
      simpa [hk1] using hk
    dsimp [x] at hx
    omega
  · intro h
    subst h
    apply Fin.ext
    simp [val_sub]

end F

end SuperNeo
