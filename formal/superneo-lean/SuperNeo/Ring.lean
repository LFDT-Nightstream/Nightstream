import SuperNeo.Field
import SuperNeo.Dimensions
import Init.Data.List.Perm
import Init.Data.List.Range
import Init.Data.List.Nat.Range
import Init.Data.Nat.Div.Basic
import Init.Data.Nat.Lemmas
import Init.GrindInstances.Ring.Fin

namespace SuperNeo

open F
local instance : NeZero Goldilocks.q := ⟨Nat.ne_of_gt Goldilocks.q_pos⟩

/-- Coefficient vector representation for ring elements. -/
abbrev Coeffs := Array F

def D : Nat := d

/-- Canonical ring-degree shape predicate for coefficient vectors. -/
def hasRingDegreeShape (a : Coeffs) : Prop :=
  a.size = d

/-- Shape precondition bundle for ring multiplication inputs. -/
def ringMulShapeProp (a b : Coeffs) : Prop :=
  hasRingDegreeShape a ∧ hasRingDegreeShape b


def vecAdd (a b : Array F) : Array F :=
  if h : a.size = b.size then
    Array.ofFn (fun i : Fin a.size =>
      a[i.1]'i.2 + b[i.1]'(by simpa [h] using i.2))
  else
    #[]

def vecScale (s : F) (a : Array F) : Array F :=
  a.map (fun x => s * x)

def linComb2Vec (ρ1 ρ2 : F) (z1 z2 : Array F) : Array F :=
  vecAdd (vecScale ρ1 z1) (vecScale ρ2 z2)

def ct (a : Coeffs) : F :=
  a.getD 0 0

def coeffAt (a : Coeffs) (i : Nat) : F :=
  if i < d then a.getD i 0 else 0

/-- Dot product with a shape guard, used by bar-block matrix application. -/
def dotBySize (row v : Array F) : F :=
  if _h : row.size = v.size then
    (List.range row.size).foldl (fun acc i => acc + row[i]! * v[i]!) 0
  else
    0

/-! ### Cyclotomic multiplication helper (`Φ(X)=X^54 + X^27 + 1`) -/

/--
Raw (unreduced) convolution coefficient `p_n = Σ_{i+j=n} a_i b_j`,
implemented with guarded indexing over `[0, d)`.
-/
def rawConvCoeff (a b : Coeffs) (n : Nat) : F :=
  (List.range d).foldl
    (fun acc i =>
      if hIn : i ≤ n ∧ n - i < d then
        acc + coeffAt a i * coeffAt b (n - i)
      else
        acc)
    0

/--
Cyclotomic multiplication in `F[X]/(X^54 + X^27 + 1)`.

Reduction uses:
* `X^54 = -X^27 - 1`
* `X^81 = 1`
-/
def mulRqPhi (a b : Coeffs) : Coeffs :=
  Array.ofFn (fun i : Fin d =>
    let n := i.1
    let base := rawConvCoeff a b n
    if hLt26 : n < 26 then
      base - rawConvCoeff a b (n + 54) + rawConvCoeff a b (n + 81)
    else if hEq26 : n = 26 then
      base - rawConvCoeff a b 80
    else
      base - rawConvCoeff a b (n + 27))

@[simp] theorem mulRqPhi_size (a b : Coeffs) : (mulRqPhi a b).size = d := by
  simp [mulRqPhi]

/-- Canonical ring multiplication surface (paper-faithful cyclotomic quotient). -/
abbrev mulRq (a b : Coeffs) : Coeffs := mulRqPhi a b

@[simp] theorem mulRq_size (a b : Coeffs) : (mulRq a b).size = d := by
  simp [mulRq]

theorem coeffAt_mulRq_lt26
    (a b : Coeffs) (i : Nat)
    (hi : i < d) (hLt26 : i < 26) :
    coeffAt (mulRq a b) i =
      rawConvCoeff a b i - rawConvCoeff a b (i + 54) + rawConvCoeff a b (i + 81) := by
  unfold coeffAt mulRq
  have his : i < (mulRqPhi a b).size := by simpa [mulRqPhi_size] using hi
  simp [hi, his, Array.getD, mulRqPhi, hLt26]

theorem coeffAt_mulRq_eq26
    (a b : Coeffs) (i : Nat)
    (hi : i < d) (hEq26 : i = 26) :
    coeffAt (mulRq a b) i =
      rawConvCoeff a b i - rawConvCoeff a b 80 := by
  subst hEq26
  have hi26 : (26 : Nat) < d := by decide
  unfold coeffAt mulRq
  have his : (26 : Nat) < (mulRqPhi a b).size := by simpa [mulRqPhi_size] using hi26
  simp [hi26, his, Array.getD, mulRqPhi, d]

theorem coeffAt_mulRq_gt26
    (a b : Coeffs) (i : Nat)
    (hi : i < d) (hGt26 : 26 < i) :
    coeffAt (mulRq a b) i =
      rawConvCoeff a b i - rawConvCoeff a b (i + 27) := by
  have hNotLt : ¬ i < 26 := by omega
  have hNe : i ≠ 26 := by omega
  unfold coeffAt mulRq
  have his : i < (mulRqPhi a b).size := by simpa [mulRqPhi_size] using hi
  simp [hi, his, Array.getD, mulRqPhi, hNotLt, hNe]

/--
Apply one SuperNeo bar block transform: matrix-vector multiplication `bar * a`
when both shapes are `d`; otherwise return the input block unchanged.
-/
def superneoBarBlock (bar : Array (Array F)) (a : Array F) : Coeffs :=
  if hBar : bar.size = d then
    if hA : a.size = d then
      Array.ofFn (fun i : Fin d =>
        dotBySize (bar[i.1]'(by simpa [hBar] using i.2)) a)
    else
      a
  else
    a

def zeroRq : Coeffs :=
  Array.replicate d 0

def oneRq : Coeffs :=
  (Array.replicate d 0).set! 0 1

@[simp] theorem vecScale_size (s : F) (a : Array F) :
    (vecScale s a).size = a.size := by
  simp [vecScale]

theorem vecAdd_size_of_eq {a b : Array F} (h : a.size = b.size) :
    (vecAdd a b).size = a.size := by
  simp [vecAdd, h]

theorem vecAdd_size_of_ne {a b : Array F} (h : a.size ≠ b.size) :
    (vecAdd a b).size = 0 := by
  simp [vecAdd, h]

theorem linComb2Vec_size_of_eq
    {ρ1 ρ2 : F} {z1 z2 : Array F} (h : z1.size = z2.size) :
    (linComb2Vec ρ1 ρ2 z1 z2).size = z1.size := by
  unfold linComb2Vec
  simp [vecScale_size, vecAdd_size_of_eq, h]

private theorem map_rotate_range_eq
  (r : Nat)
  (hr : r < d) :
  List.map (fun j => (r + j) % d) (List.range d) =
    List.range' r (d - r) ++ List.range r := by
  have hle : r ≤ d := Nat.le_of_lt hr
  have hsplit :
      List.range d =
        List.range (d - r) ++ (List.range r).map ((d - r) + ·) := by
    have h := List.range_add (n := d - r) (m := r)
    simpa [Nat.sub_add_cancel hle] using h
  rw [hsplit, List.map_append]
  have hpart1 :
      List.map (fun j => (r + j) % d) (List.range (d - r)) =
        List.range' r (d - r) := by
    have hmap :
        List.map (fun j => (r + j) % d) (List.range (d - r)) =
          List.map (fun j => r + j) (List.range (d - r)) := by
      apply List.map_congr_left
      intro a ha
      have haLt : a < d - r := by simpa [List.mem_range] using ha
      have hlt : r + a < d := by omega
      simp [Nat.mod_eq_of_lt hlt]
    calc
      List.map (fun j => (r + j) % d) (List.range (d - r))
          = List.map (fun j => r + j) (List.range (d - r)) := hmap
      _ = List.range' r (d - r) := by
            symm
            simpa using (List.range'_eq_map_range (s := r) (n := d - r))
  have hpart2 :
      List.map (fun x => (r + ((d - r) + x)) % d) (List.range r) =
        List.range r := by
    calc
      List.map (fun x => (r + ((d - r) + x)) % d) (List.range r)
          = List.map (fun x => x) (List.range r) := by
            apply List.map_congr_left
            intro a ha
            have haLt : a < r := by simpa [List.mem_range] using ha
            have had : a < d := Nat.lt_trans haLt hr
            have hrd : r + ((d - r) + a) = d + a := by omega
            simp [hrd, Nat.mod_eq_of_lt had]
      _ = List.range r := by simp
  simpa [hpart1, hpart2]

private theorem perm_rotate_range
  (r : Nat)
  (hr : r < d) :
  List.Perm (List.map (fun j => (r + j) % d) (List.range d)) (List.range d) := by
  rw [map_rotate_range_eq r hr]
  have hle : r ≤ d := Nat.le_of_lt hr
  have hsplit : List.range d = List.range r ++ List.range' r (d - r) := by
    have h := List.range_add (n := r) (m := d - r)
    simpa [Nat.add_sub_cancel' hle, List.range'_eq_map_range] using h
  exact (List.perm_append_comm (l₁ := List.range' r (d-r)) (l₂ := List.range r)).trans
    (List.Perm.of_eq hsplit.symm)

private theorem perm_modSub_range
  (i : Nat) :
  List.Perm
    (List.map (fun j => (i + d - j) % d) (List.range d))
    (List.range d) := by
  let r : Nat := (i + 1) % d
  have hr : r < d := by
    dsimp [r]
    exact Nat.mod_lt _ d_pos
  have hsubToRev :
      List.map (fun j => (i + d - j) % d) (List.range d) =
        List.map (fun x => (r + x) % d) (List.reverse (List.range d)) := by
    have hmapSub :
        List.map (fun j => (i + d - j) % d) (List.range d) =
          List.map (fun j => (r + (d - 1 - j)) % d) (List.range d) := by
      apply List.map_congr_left
      intro a ha
      have haLt : a < d := by simpa [List.mem_range] using ha
      have hEq₁ : i + d - a = (i + 1) + (d - 1 - a) := by omega
      calc
        (i + d - a) % d
            = ((i + 1) + (d - 1 - a)) % d := by simp [hEq₁]
        _ = (((i + 1) % d) + (d - 1 - a)) % d := by
              simpa using (Nat.add_mod (i + 1) (d - 1 - a) d)
        _ = (r + (d - 1 - a)) % d := by
              simp [r]
    have hrev :
        List.map (fun j => d - 1 - j) (List.range d) = List.reverse (List.range d) := by
      simpa [List.range_eq_range', Nat.zero_add] using
        (List.reverse_range' (s := 0) (n := d)).symm
    calc
      List.map (fun j => (i + d - j) % d) (List.range d)
          = List.map (fun j => (r + (d - 1 - j)) % d) (List.range d) := hmapSub
      _ = List.map (fun x => (r + x) % d) (List.map (fun j => d - 1 - j) (List.range d)) := by
            simp [List.map_map]
      _ = List.map (fun x => (r + x) % d) (List.reverse (List.range d)) := by
            simp [hrev]
  have hPerm1 :
      List.Perm
        (List.map (fun x => (r + x) % d) (List.reverse (List.range d)))
        (List.map (fun x => (r + x) % d) (List.range d)) :=
    (List.reverse_perm (l := List.range d)).map _
  have hPerm2 :
      List.Perm (List.map (fun x => (r + x) % d) (List.range d)) (List.range d) :=
    perm_rotate_range r hr
  exact (List.Perm.of_eq hsubToRev).trans (hPerm1.trans hPerm2)

private theorem mod_sub_involutive
  (i j : Nat)
  (hj : j < d) :
  ((i + d - ((i + d - j) % d)) % d) = j := by
  let x := i + d - j
  let q := x / d
  let t := x % d
  have hx : t + d * q = x := by
    simp [x, q, t, Nat.mod_add_div]
  have hEq : i + d - t = j + d * q := by
    have : i + d - j = t + d * q := by
      simpa [x, q, t] using hx.symm
    omega
  have hEq2 : i + d - ((i + d - j) % d) = j + d * q := by
    simpa [x, t] using hEq
  calc
    ((i + d - ((i + d - j) % d)) % d)
        = ((j + d * q) % d) := by simp [hEq2]
    _ = j % d := by
          simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using
            (Nat.add_mul_mod_self_left j q d)
    _ = j := Nat.mod_eq_of_lt hj

private theorem list_foldl_congr_mem
  {α β : Type}
  (f g : α → β → α)
  (init : α)
  (l : List β)
  (hfg : ∀ acc b, b ∈ l → f acc b = g acc b) :
  List.foldl f init l = List.foldl g init l := by
  induction l generalizing init with
  | nil => simp
  | cons b bs ih =>
      have hHead : f init b = g init b := by
        exact hfg init b (by simp)
      calc
        List.foldl f init (b :: bs)
            = List.foldl f (f init b) bs := by rfl
        _ = List.foldl f (g init b) bs := by rw [hHead]
        _ = List.foldl g (g init b) bs := by
              apply ih
              intro acc b' hb'
              exact hfg acc b' (by simp [hb'])
        _ = List.foldl g init (b :: bs) := by rfl

/-! ### Cyclotomic constant-term expansion helpers (`Φ(X)=X^54 + X^27 + 1`) -/

/-- Constant-term extraction for `mulRqPhi` at index `0`. -/
theorem ct_mulRqPhi_eq_raw
    (x b : Coeffs) :
    ct (mulRqPhi x b) =
      rawConvCoeff x b 0 - rawConvCoeff x b 54 + rawConvCoeff x b 81 := by
  unfold ct
  have h0 : (0 : Nat) < d := d_pos
  simp [mulRqPhi, h0]

/-- Raw convolution at `n = 0` keeps only the `i = 0` term. -/
theorem rawConvCoeff_zero
    (x b : Coeffs) :
    rawConvCoeff x b 0 = coeffAt x 0 * coeffAt b 0 := by
  unfold rawConvCoeff
  have hrewrite :
      (fun acc i =>
        if hIn : i ≤ 0 ∧ 0 - i < d then
          acc + coeffAt x i * coeffAt b (0 - i)
        else
          acc)
      =
      (fun acc i =>
        if i = 0 then
          acc + coeffAt x i * coeffAt b 0
        else
          acc) := by
    funext acc i
    by_cases h0 : i = 0
    · subst h0
      have hIn : (0 ≤ 0 ∧ 0 - 0 < d) := by
        constructor
        · decide
        · simpa using d_pos
      simp [hIn]
    · have hInFalse : ¬ (i ≤ 0 ∧ 0 - i < d) := by
        intro h
        exact h0 (Nat.eq_zero_of_le_zero h.1)
      simp [h0, hInFalse]
  rw [hrewrite]
  let step : F → Nat → F := fun acc i =>
    if i = 0 then acc + coeffAt x i * coeffAt b 0 else acc
  have hTail : ∀ (l : List Nat) (init' : F), (l.map Nat.succ).foldl step init' = init' := by
    intro l
    induction l with
    | nil =>
        intro init'
        simp [step]
    | cons x' xs ih =>
        intro init'
        simp [step, ih]
  have hd : d = 53 + 1 := by decide
  rw [hd, List.range_succ_eq_map]
  simp [step, hTail]

/--
Raw convolution at `n = 54` keeps exactly `i = 1..53`
(the `i = 0` term is excluded by the `54 - i < d` guard).
-/
theorem rawConvCoeff_54
    (x b : Coeffs) :
    rawConvCoeff x b 54 =
      (List.range d).foldl
        (fun acc i =>
          if i = 0 then
            acc
          else
            acc + coeffAt x i * coeffAt b (54 - i))
        0 := by
  unfold rawConvCoeff
  apply list_foldl_congr_mem
  intro acc i hi
  have hiLt : i < d := by
    simpa [List.mem_range] using hi
  by_cases h0 : i = 0
  · subst i
    have hnot : ¬ (54 < d) := by
      simp [d]
    have hif :
        (if hIn : (0 ≤ 54 ∧ 54 - 0 < d) then
          acc + coeffAt x 0 * coeffAt b (54 - 0)
        else
          acc) = acc := by
      by_cases h : 54 < d
      · exact False.elim (hnot h)
      · simp [h]
    simpa using hif
  · have hiPos : 0 < i := Nat.pos_of_ne_zero h0
    have hIn : (i ≤ 54 ∧ 54 - i < d) := by
      constructor
      · have hiLe : i ≤ d := Nat.le_of_lt hiLt
        simpa [d] using hiLe
      · simpa [d] using (Nat.sub_lt (by decide : 0 < 54) hiPos)
    simp [h0, hIn]

/--
Raw convolution at `n = 81` keeps exactly `i = 28..53`
under the guard `81 - i < d`.
-/
theorem rawConvCoeff_81
    (x b : Coeffs) :
    rawConvCoeff x b 81 =
      (List.range d).foldl
        (fun acc i =>
          if i < 28 then
            acc
          else
            acc + coeffAt x i * coeffAt b (81 - i))
        0 := by
  unfold rawConvCoeff
  apply list_foldl_congr_mem
  intro acc i hi
  have hiLt : i < d := by
    simpa [List.mem_range] using hi
  by_cases hlt28 : i < 28
  · have hInFalse : ¬ (i ≤ 81 ∧ 81 - i < d) := by
      intro h
      have : ¬ (81 - i < d) := by
        have hd : d = 54 := rfl
        have h54le : 54 ≤ 81 - i := by
          omega
        simpa [hd] using (Nat.not_lt.mpr h54le)
      exact this h.2
    by_cases hIn : (i ≤ 81 ∧ 81 - i < d)
    · exact (False.elim (hInFalse hIn))
    · simp [hlt28, hIn]
  · have hiGe28 : 28 ≤ i := Nat.le_of_not_lt hlt28
    have hIn : (i ≤ 81 ∧ 81 - i < d) := by
      constructor
      · have hiLe : i ≤ d := Nat.le_of_lt hiLt
        have hd : d = 54 := rfl
        omega
      · have hd : d = 54 := rfl
        omega
    simp [hlt28, hIn]

/--
Expanded constant-term formula for cyclotomic multiplication:
`ct(mulRqPhi x b) = p0 - p54 + p81` with specialized raw-convolution forms.
-/
theorem ct_mulRqPhi_expanded
    (x b : Coeffs) :
    ct (mulRqPhi x b) =
      coeffAt x 0 * coeffAt b 0
        - ((List.range d).foldl
            (fun acc i =>
              if i = 0 then
                acc
              else
                acc + coeffAt x i * coeffAt b (54 - i))
            0)
        + ((List.range d).foldl
            (fun acc i =>
              if i < 28 then
                acc
              else
                acc + coeffAt x i * coeffAt b (81 - i))
            0) := by
  calc
    ct (mulRqPhi x b)
        = rawConvCoeff x b 0 - rawConvCoeff x b 54 + rawConvCoeff x b 81 := by
            exact ct_mulRqPhi_eq_raw x b
    _ = coeffAt x 0 * coeffAt b 0 - rawConvCoeff x b 54 + rawConvCoeff x b 81 := by
          rw [rawConvCoeff_zero x b]
    _ =
        coeffAt x 0 * coeffAt b 0
          - ((List.range d).foldl
              (fun acc i =>
                if i = 0 then
                  acc
                else
                  acc + coeffAt x i * coeffAt b (54 - i))
              0)
          + rawConvCoeff x b 81 := by
          rw [rawConvCoeff_54 x b]
    _ =
        coeffAt x 0 * coeffAt b 0
          - ((List.range d).foldl
              (fun acc i =>
                if i = 0 then
                  acc
                else
                  acc + coeffAt x i * coeffAt b (54 - i))
              0)
          + ((List.range d).foldl
              (fun acc i =>
                if i < 28 then
                  acc
                else
                  acc + coeffAt x i * coeffAt b (81 - i))
              0) := by
          rw [rawConvCoeff_81 x b]

private theorem f_right_distrib (a b c : F) : (a + b) * c = a * c + b * c := by
  calc
    (a + b) * c = c * (a + b) := by
      simpa using (Lean.Grind.Fin.mul_comm (n := Goldilocks.q) (a + b) c)
    _ = c * a + c * b := by
      simpa using (Lean.Grind.Fin.left_distrib (n := Goldilocks.q) c a b)
    _ = a * c + b * c := by
      simp [Lean.Grind.Fin.mul_comm]

private theorem f_mul_zero (a : F) : a * 0 = 0 := by
  calc
    a * 0 = 0 * a := by
      simpa using (Lean.Grind.Fin.mul_comm (n := Goldilocks.q) a 0)
    _ = 0 := by
      simpa using (Lean.Grind.Fin.zero_mul (n := Goldilocks.q) a)

private theorem one_mul_fin (x : F) : (1 : F) * x = x := by
  calc
    (1 : F) * x = x * (1 : F) := by
      simpa using (Lean.Grind.Fin.mul_comm (n := Goldilocks.q) (1 : F) x)
    _ = x := by
      simpa using (Lean.Grind.Fin.mul_one (n := Goldilocks.q) x)

private theorem neg_one_mul (x : F) : (- (1 : F)) * x = -x := by
  calc
    (- (1 : F)) * x = -((1 : F) * x) := by
      simpa using (Lean.Grind.Fin.neg_mul (n := Goldilocks.q) (1 : F) x)
    _ = -x := by
      simp [one_mul_fin]

private theorem neg_add_distrib (u v : F) : -(u + v) = (-u) + (-v) := by
  calc
    -(u + v) = (- (1 : F)) * (u + v) := by
      symm
      exact neg_one_mul (u + v)
    _ = ((- (1 : F)) * u) + ((- (1 : F)) * v) := by
      simpa using (Lean.Grind.Fin.left_distrib (n := Goldilocks.q) (-(1 : F)) u v)
    _ = (-u) + (-v) := by
      simp [neg_one_mul]

private theorem f_add_left_comm (a b c : F) : a + (b + c) = b + (a + c) := by
  calc
    a + (b + c) = (a + b) + c := by
      simpa using (Lean.Grind.Fin.add_assoc (n := Goldilocks.q) a b c).symm
    _ = (b + a) + c := by
      simpa using congrArg (fun t => t + c) (Lean.Grind.Fin.add_comm (n := Goldilocks.q) a b)
    _ = b + (a + c) := by
      simpa using (Lean.Grind.Fin.add_assoc (n := Goldilocks.q) b a c)

private theorem sum_sub_sum_add_sum_split (a b c d e f : F) :
    (a + b) - (c + d) + (e + f) = (a - c + e) + (b - d + f) := by
  calc
    (a + b) - (c + d) + (e + f)
        = (a + b) + (-(c + d)) + (e + f) := by
            simp [Lean.Grind.Fin.sub_eq_add_neg]
    _ = (a + b) + ((-c) + (-d)) + (e + f) := by
          simp [neg_add_distrib]
    _ = (a + ((-c) + e)) + (b + ((-d) + f)) := by
          simp [Lean.Grind.Fin.add_assoc, Lean.Grind.Fin.add_comm, f_add_left_comm]
    _ = (a - c + e) + (b - d + f) := by
          simp [Lean.Grind.Fin.sub_eq_add_neg, Lean.Grind.Fin.add_assoc]

private theorem mul_neg_right (s x : F) : s * (-x) = -(s * x) := by
  calc
    s * (-x) = (-x) * s := by
      simpa using (Lean.Grind.Fin.mul_comm (n := Goldilocks.q) s (-x))
    _ = -(x * s) := by
          simpa using (Lean.Grind.Fin.neg_mul (n := Goldilocks.q) x s)
    _ = -(s * x) := by
          simpa using congrArg Neg.neg (Lean.Grind.Fin.mul_comm (n := Goldilocks.q) x s)

private theorem scale_sub_add_factor (s a b c : F) :
    s * a - s * b + s * c = s * (a - b + c) := by
  calc
    s * a - s * b + s * c
        = s * a + (-(s * b)) + s * c := by
            simp [Lean.Grind.Fin.sub_eq_add_neg, Lean.Grind.Fin.add_assoc]
    _ = s * a + s * (-b) + s * c := by
          simp [mul_neg_right]
    _ = s * (a + (-b)) + s * c := by
          simpa [Lean.Grind.Fin.left_distrib, Lean.Grind.Fin.add_assoc, Lean.Grind.Fin.add_comm]
    _ = s * ((a + (-b)) + c) := by
          simpa [Lean.Grind.Fin.left_distrib, Lean.Grind.Fin.add_assoc]
    _ = s * (a - b + c) := by
          simp [Lean.Grind.Fin.sub_eq_add_neg]

private theorem sum_sub_sum_split (a b c d : F) :
    (a + b) - (c + d) = (a - c) + (b - d) := by
  have h :=
    sum_sub_sum_add_sum_split a b c d (0 : F) (0 : F)
  simpa using h

private theorem scale_sub_factor (s a b : F) :
    s * a - s * b = s * (a - b) := by
  have h := scale_sub_add_factor s a b (0 : F)
  calc
    s * a - s * b
        = s * a - s * b + s * (0 : F) := by
            simp [f_mul_zero]
    _ = s * (a - b + (0 : F)) := h
    _ = s * (a - b) := by
          simp

private theorem foldl_add_linearity_F
  (l : List Nat)
  (t1 t2 : Nat → F)
  (acc1 acc2 : F) :
  l.foldl (fun acc j => acc + (t1 j + t2 j)) (acc1 + acc2) =
    (l.foldl (fun acc j => acc + t1 j) acc1) +
      (l.foldl (fun acc j => acc + t2 j) acc2) := by
  induction l generalizing acc1 acc2 with
  | nil =>
      simp
  | cons j js ih =>
      have hInit :
          (acc1 + acc2) + (t1 j + t2 j) =
            (acc1 + t1 j) + (acc2 + t2 j) := by
        calc
          (acc1 + acc2) + (t1 j + t2 j)
              = acc1 + (acc2 + (t1 j + t2 j)) := by
                  simpa using (Lean.Grind.Fin.add_assoc (n := Goldilocks.q) acc1 acc2 (t1 j + t2 j))
          _ = acc1 + (t1 j + (acc2 + t2 j)) := by
                exact congrArg (fun t => acc1 + t) (f_add_left_comm acc2 (t1 j) (t2 j))
          _ = (acc1 + t1 j) + (acc2 + t2 j) := by
                simpa using (Lean.Grind.Fin.add_assoc (n := Goldilocks.q) acc1 (t1 j) (acc2 + t2 j)).symm
      calc
        (j :: js).foldl (fun acc j => acc + (t1 j + t2 j)) (acc1 + acc2)
            = js.foldl (fun acc j => acc + (t1 j + t2 j)) ((acc1 + acc2) + (t1 j + t2 j)) := by
                simp [List.foldl]
        _ = js.foldl (fun acc j => acc + (t1 j + t2 j)) ((acc1 + t1 j) + (acc2 + t2 j)) := by
              simp [hInit]
        _ = js.foldl (fun acc j => acc + t1 j) (acc1 + t1 j) +
              js.foldl (fun acc j => acc + t2 j) (acc2 + t2 j) := by
              simpa using ih (acc1 := acc1 + t1 j) (acc2 := acc2 + t2 j)
        _ = (j :: js).foldl (fun acc j => acc + t1 j) acc1 +
              (j :: js).foldl (fun acc j => acc + t2 j) acc2 := by
              simp [List.foldl]

private theorem foldl_mul_right_distrib_F
  (l : List Nat)
  (t : Nat → F)
  (acc c : F) :
  (l.foldl (fun a j => a + t j) acc) * c =
    l.foldl (fun a j => a + t j * c) (acc * c) := by
  induction l generalizing acc with
  | nil =>
      simp
  | cons j js ih =>
      calc
        ((j :: js).foldl (fun a k => a + t k) acc) * c
            = (js.foldl (fun a k => a + t k) (acc + t j)) * c := by
                simp [List.foldl]
        _ = js.foldl (fun a k => a + t k * c) ((acc + t j) * c) := by
              simpa using ih (acc := acc + t j)
        _ = js.foldl (fun a k => a + t k * c) (acc * c + t j * c) := by
              simp [f_right_distrib]
        _ = (j :: js).foldl (fun a k => a + t k * c) (acc * c) := by
              simp [List.foldl, Lean.Grind.Fin.add_assoc]

private theorem foldl_mul_left_distrib_F
  (l : List Nat)
  (t : Nat → F)
  (acc c : F) :
  c * (l.foldl (fun a j => a + t j) acc) =
    l.foldl (fun a j => a + c * t j) (c * acc) := by
  induction l generalizing acc with
  | nil =>
      simp
  | cons j js ih =>
      calc
        c * ((j :: js).foldl (fun a k => a + t k) acc)
            = c * (js.foldl (fun a k => a + t k) (acc + t j)) := by
                simp [List.foldl]
        _ = js.foldl (fun a k => a + c * t k) (c * (acc + t j)) := by
              simpa using ih (acc := acc + t j)
        _ = js.foldl (fun a k => a + c * t k) (c * acc + c * t j) := by
              simp [Lean.Grind.Fin.left_distrib]
        _ = (j :: js).foldl (fun a k => a + c * t k) (c * acc) := by
              simp [List.foldl, Lean.Grind.Fin.add_assoc]

private theorem foldl_add_from_init_F
  (l : List Nat)
  (t : Nat → F)
  (init : F) :
  l.foldl (fun acc j => acc + t j) init =
    init + l.foldl (fun acc j => acc + t j) 0 := by
  induction l generalizing init with
  | nil =>
      simp
  | cons j js ih =>
      calc
        (j :: js).foldl (fun acc k => acc + t k) init
            = js.foldl (fun acc k => acc + t k) (init + t j) := by
                simp [List.foldl]
        _ = (init + t j) + js.foldl (fun acc k => acc + t k) 0 := by
              simpa using ih (init := init + t j)
        _ = init + (t j + js.foldl (fun acc k => acc + t k) 0) := by
              simpa using (Lean.Grind.Fin.add_assoc (n := Goldilocks.q) init (t j)
                (js.foldl (fun acc k => acc + t k) 0))
        _ = init + js.foldl (fun acc k => acc + t k) (t j) := by
              have hInitJ := ih (init := t j)
              simpa [hInitJ]
        _ = init + (j :: js).foldl (fun acc k => acc + t k) 0 := by
              simp [List.foldl]

/-! ### Public fold-linearity helpers (theorem-facing reuse) -/

theorem foldl_add_linearity
  (l : List Nat)
  (t1 t2 : Nat → F)
  (acc1 acc2 : F) :
  l.foldl (fun acc j => acc + (t1 j + t2 j)) (acc1 + acc2) =
    (l.foldl (fun acc j => acc + t1 j) acc1) +
      (l.foldl (fun acc j => acc + t2 j) acc2) := by
  simpa using foldl_add_linearity_F l t1 t2 acc1 acc2

theorem foldl_mul_right_distrib
  (l : List Nat)
  (t : Nat → F)
  (acc c : F) :
  (l.foldl (fun a j => a + t j) acc) * c =
    l.foldl (fun a j => a + t j * c) (acc * c) := by
  simpa using foldl_mul_right_distrib_F l t acc c

theorem foldl_mul_left_distrib
  (l : List Nat)
  (t : Nat → F)
  (acc c : F) :
  c * (l.foldl (fun a j => a + t j) acc) =
    l.foldl (fun a j => a + c * t j) (c * acc) := by
  simpa using foldl_mul_left_distrib_F l t acc c

theorem foldl_add_from_init
  (l : List Nat)
  (t : Nat → F)
  (init : F) :
  l.foldl (fun acc j => acc + t j) init =
    init + l.foldl (fun acc j => acc + t j) 0 := by
  simpa using foldl_add_from_init_F l t init

private theorem foldl_keep_init
  (l : List Nat)
  (init : F) :
  l.foldl (fun acc _ => acc) init = init := by
  induction l generalizing init with
  | nil =>
      simp
  | cons j js ih =>
      simpa [List.foldl] using ih (init := init)

private theorem double_sum_swap
  (l1 l2 : List Nat)
  (f : Nat → Nat → F) :
  l1.foldl
      (fun acc j => acc + l2.foldl (fun acc2 k => acc2 + f j k) 0)
      0
    =
  l2.foldl
      (fun acc k => acc + l1.foldl (fun acc2 j => acc2 + f j k) 0)
      0 := by
  induction l1 with
  | nil =>
      induction l2 with
      | nil => simp
      | cons k ks ih2 =>
          simpa [List.foldl] using (foldl_keep_init ks 0).symm
  | cons j js ih =>
      let s1 : Nat → F := fun k => f j k
      let s2 : Nat → F := fun k => js.foldl (fun acc2 j' => acc2 + f j' k) 0
      have hDecomp :
          l2.foldl
              (fun acc k => acc + (j :: js).foldl (fun acc2 j' => acc2 + f j' k) 0)
              0
            =
          l2.foldl (fun acc k => acc + (s1 k + s2 k)) 0 := by
        apply list_foldl_congr_mem
        intro acc k hk
        have hInner :
            (j :: js).foldl (fun acc2 j' => acc2 + f j' k) 0 =
              f j k + js.foldl (fun acc2 j' => acc2 + f j' k) 0 := by
          have hInit := foldl_add_from_init_F
            (l := js)
            (t := fun j' => f j' k)
            (init := f j k)
          simpa [List.foldl] using hInit
        simpa [s1, s2, hInner]
      calc
        (j :: js).foldl
            (fun acc j' => acc + l2.foldl (fun acc2 k => acc2 + f j' k) 0)
            0
            =
          js.foldl
            (fun acc j' => acc + l2.foldl (fun acc2 k => acc2 + f j' k) 0)
            (l2.foldl (fun acc2 k => acc2 + f j k) 0) := by
              simp [List.foldl]
        _ =
          l2.foldl (fun acc2 k => acc2 + f j k) 0
            +
          js.foldl
            (fun acc j' => acc + l2.foldl (fun acc2 k => acc2 + f j' k) 0)
            0 := by
              simpa using foldl_add_from_init_F
                (l := js)
                (t := fun j' => l2.foldl (fun acc2 k => acc2 + f j' k) 0)
                (init := l2.foldl (fun acc2 k => acc2 + f j k) 0)
        _ =
          l2.foldl (fun acc2 k => acc2 + f j k) 0
            +
          l2.foldl
            (fun acc2 k => acc2 + js.foldl (fun acc3 j' => acc3 + f j' k) 0)
            0 := by
              simpa using congrArg (fun t => l2.foldl (fun acc2 k => acc2 + f j k) 0 + t) ih
        _ =
          l2.foldl (fun acc2 k => acc2 + s1 k) 0
            +
          l2.foldl (fun acc2 k => acc2 + s2 k) 0 := by
              simp [s1, s2]
        _ = l2.foldl (fun acc2 k => acc2 + (s1 k + s2 k)) 0 := by
              simpa using (foldl_add_linearity_F
                (l := l2)
                (t1 := s1)
                (t2 := s2)
                (acc1 := 0)
                (acc2 := 0)).symm
        _ =
          l2.foldl
            (fun acc k => acc + (j :: js).foldl (fun acc2 j' => acc2 + f j' k) 0)
            0 := by
              simpa [List.foldl] using hDecomp.symm

private theorem foldl_add_eq_of_perm
  (l1 l2 : List Nat)
  (hperm : l1.Perm l2)
  (f : Nat → F)
  (init : F) :
  l1.foldl (fun acc j => acc + f j) init =
    l2.foldl (fun acc j => acc + f j) init := by
  induction hperm generalizing init with
  | nil => simp
  | @cons x l1 l2 hperm ih =>
      simpa [List.foldl] using ih (init := init + f x)
  | @swap x y l =>
      have hInit : init + f y + f x = init + f x + f y := by
        calc
          init + f y + f x = init + (f y + f x) := by
            simpa using (Lean.Grind.Fin.add_assoc (n := Goldilocks.q) init (f y) (f x))
          _ = init + (f x + f y) := by
            simpa using congrArg (fun t => init + t)
              (Lean.Grind.Fin.add_comm (n := Goldilocks.q) (f y) (f x))
          _ = init + f x + f y := by
            simpa using (Lean.Grind.Fin.add_assoc (n := Goldilocks.q) init (f x) (f y)).symm
      simpa [List.foldl, hInit]
  | @trans l1 l2 l3 h12 h23 ih12 ih23 =>
      exact (ih12 init).trans (ih23 init)

private theorem foldl_range_rotate
  (r : Nat)
  (hr : r < d)
  (f : Nat → F) :
  (List.range d).foldl (fun acc j => acc + f j) 0 =
    (List.range d).foldl (fun acc j => acc + f ((r + j) % d)) 0 := by
  have hperm :
      (List.map (fun j => (r + j) % d) (List.range d)).Perm (List.range d) :=
    perm_rotate_range r hr
  have hPermFold :
      (List.map (fun j => (r + j) % d) (List.range d)).foldl
          (fun acc j => acc + f j)
          0
        =
      (List.range d).foldl (fun acc j => acc + f j) 0 := by
    exact foldl_add_eq_of_perm _ _ hperm f 0
  calc
    (List.range d).foldl (fun acc j => acc + f j) 0
        =
      (List.map (fun j => (r + j) % d) (List.range d)).foldl
          (fun acc j => acc + f j)
          0 := by
            simpa using hPermFold.symm
    _ =
      (List.range d).foldl (fun acc j => acc + f ((r + j) % d)) 0 := by
        simp [List.foldl_map]

private theorem mod_shift_sub
  (x j : Nat) (hj : j < d) :
  ((x % d) + d - j) % d = (x + d - j) % d := by
  by_cases h0 : j = 0
  · subst h0
    calc
      ((x % d) + d - 0) % d = ((x % d) + d) % d := by simp
      _ = x % d := by simpa using (Nat.add_mod_right (x % d) d)
      _ = (x + d) % d := by simpa using (Nat.add_mod_right x d).symm
      _ = (x + d - 0) % d := by simp
  · have hjPos : 0 < j := Nat.pos_of_ne_zero h0
    have hjLe : j ≤ d := Nat.le_of_lt hj
    have hdjLt : d - j < d := Nat.sub_lt d_pos hjPos
    calc
      ((x % d) + d - j) % d = ((x % d) + (d - j)) % d := by
        rw [Nat.add_sub_assoc hjLe]
      _ = (((x % d) % d) + ((d - j) % d)) % d := by
        simpa using (Nat.add_mod (x % d) (d - j) d)
      _ = ((x % d) + (d - j)) % d := by
        simp [Nat.mod_eq_of_lt hdjLt]
      _ = (x + (d - j)) % d := by
          have hAdd : (x + (d - j)) % d = ((x % d) + ((d - j) % d)) % d := by
            simpa using (Nat.add_mod x (d - j) d)
          have hDJ : (d - j) % d = d - j := Nat.mod_eq_of_lt hdjLt
          simpa [hDJ] using hAdd.symm
      _ = (x + d - j) % d := by
        rw [Nat.add_sub_assoc hjLe]

private theorem mod_add_sub_cancel_of_lt
  (r j : Nat)
  (hr : r < d)
  (hj : j < d) :
  (((r + j) % d) + d - r) % d = j := by
  by_cases hsum : r + j < d
  · have hmod : (r + j) % d = r + j := Nat.mod_eq_of_lt hsum
    have hinside : (r + j) + d - r = j + d := by omega
    calc
      (((r + j) % d) + d - r) % d = ((r + j) + d - r) % d := by simp [hmod]
      _ = (j + d) % d := by simp [hinside]
      _ = j % d := by simpa using (Nat.add_mod_right j d)
      _ = j := Nat.mod_eq_of_lt hj
  · have hsumge : d ≤ r + j := Nat.le_of_not_gt hsum
    have hlt : r + j - d < d := by omega
    have hmod1 : (r + j) % d = (r + j - d) % d := Nat.mod_eq_sub_mod hsumge
    have hmod2 : (r + j - d) % d = r + j - d := Nat.mod_eq_of_lt hlt
    have hmod : (r + j) % d = r + j - d := by simpa [hmod2] using hmod1
    have hinside : (r + j - d) + d - r = j := by omega
    calc
      (((r + j) % d) + d - r) % d = ((r + j - d) + d - r) % d := by simp [hmod]
      _ = j % d := by simp [hinside]
      _ = j := Nat.mod_eq_of_lt hj

private theorem mod_nested_sub_eq
  (i j k : Nat)
  (hj : j < d)
  (hk : k < d) :
  (((i + d - k) % d) + d - j) % d = (i + d - ((k + j) % d)) % d := by
  have hL : (((i + d - k) % d) + d - j) % d = (i + d - k + d - j) % d := by
    simpa using mod_shift_sub (x := i + d - k) j hj
  by_cases hsum : k + j < d
  · have hmod : (k + j) % d = k + j := Nat.mod_eq_of_lt hsum
    have hR1 : (i + d - ((k + j) % d)) % d = (i + d - (k + j)) % d := by
      simp [hmod]
    have hE : (i + d - k + d - j) % d = (i + d - (k + j) + d) % d := by
      have hInside1 : i + d - k + d - j = (i + d - (k + j)) + d := by
        omega
      simp [hInside1]
    calc
      (((i + d - k) % d) + d - j) % d = (i + d - k + d - j) % d := hL
      _ = (i + d - (k + j) + d) % d := hE
      _ = (i + d - (k + j)) % d := by
            simpa [Nat.add_comm] using (Nat.add_mod_right (i + d - (k + j)) d)
      _ = (i + d - ((k + j) % d)) % d := by
            simpa [hR1] using rfl
  · have hsumge : d ≤ k + j := Nat.le_of_not_gt hsum
    have hlt : k + j - d < d := by omega
    have hmod1 : (k + j) % d = (k + j - d) % d := Nat.mod_eq_sub_mod hsumge
    have hmod2 : (k + j - d) % d = k + j - d := Nat.mod_eq_of_lt hlt
    have hmod : (k + j) % d = k + j - d := by simpa [hmod2] using hmod1
    have hR2 : (i + d - ((k + j) % d)) % d = (i + d - (k + j - d)) % d := by
      simp [hmod]
    have hInside2 : i + d - (k + j - d) = i + d - k + d - j := by omega
    calc
      (((i + d - k) % d) + d - j) % d = (i + d - k + d - j) % d := hL
      _ = (i + d - (k + j - d)) % d := by simp [hInside2]
      _ = (i + d - ((k + j) % d)) % d := by simpa [hR2] using rfl

private theorem inner_reindex_sub
  (b c : Coeffs)
  (i k : Nat)
  (hk : k < d) :
  (List.range d).foldl
      (fun acc j =>
        acc + coeffAt b ((j + d - k) % d) * coeffAt c ((i + d - j) % d))
      0
    =
  (List.range d).foldl
      (fun acc j =>
        acc + coeffAt b j * coeffAt c ((i + d - ((k + j) % d)) % d))
      0 := by
  let f : Nat → F := fun x =>
    coeffAt b ((x + d - k) % d) * coeffAt c ((i + d - x) % d)
  have hRot :
      (List.range d).foldl (fun acc j => acc + f j) 0 =
        (List.range d).foldl (fun acc j => acc + f ((k + j) % d)) 0 :=
    foldl_range_rotate k hk f
  calc
    (List.range d).foldl
        (fun acc j =>
          acc + coeffAt b ((j + d - k) % d) * coeffAt c ((i + d - j) % d))
        0
        =
      (List.range d).foldl (fun acc j => acc + f j) 0 := by
          simp [f]
    _ =
      (List.range d).foldl (fun acc j => acc + f ((k + j) % d)) 0 := hRot
    _ =
      (List.range d).foldl
        (fun acc j =>
          acc + coeffAt b j * coeffAt c ((i + d - ((k + j) % d)) % d))
        0 := by
          apply list_foldl_congr_mem
          intro acc j hjMem
          have hj : j < d := by simpa [List.mem_range] using hjMem
          have hIdxB : (((k + j) % d) + d - k) % d = j :=
            mod_add_sub_cancel_of_lt k j hk hj
          simp [f, hIdxB]

theorem coeffAt_vecAdd_of_size_d
  (x y : Coeffs)
  (hx : x.size = d) (hy : y.size = d)
  (k : Nat) (hk : k < d) :
  coeffAt (vecAdd x y) k = coeffAt x k + coeffAt y k := by
  have hxy : x.size = y.size := by simpa [hx, hy]
  have hkx : k < x.size := by simpa [hx] using hk
  have hky : k < y.size := by simpa [hy] using hk
  unfold coeffAt
  simp [vecAdd, hxy, hk, Array.getD, hkx, hky, coeffAt, hx, hy]

theorem coeffAt_vecScale_of_size_d
  (s : F) (x : Coeffs)
  (hx : x.size = d)
  (k : Nat) (hk : k < d) :
  coeffAt (vecScale s x) k = s * coeffAt x k := by
  have hkx : k < x.size := by simpa [hx] using hk
  unfold coeffAt
  simp [vecScale, hk, Array.getD, hkx, hx]

private theorem rawConvCoeff_eq_fold_add_if
  (a b : Coeffs) (n : Nat) :
  rawConvCoeff a b n =
    (List.range d).foldl
      (fun acc i =>
        acc + (if hIn : i ≤ n ∧ n - i < d then
          coeffAt a i * coeffAt b (n - i)
        else
          0))
      0 := by
  unfold rawConvCoeff
  apply list_foldl_congr_mem
  intro acc i hiMem
  by_cases hIn : i ≤ n ∧ n - i < d
  · simp [hIn]
  · simp [hIn]

theorem rawConvCoeff_vecAdd_left_of_size_d
  (x y b : Coeffs)
  (hx : x.size = d) (hy : y.size = d)
  (n : Nat) :
  rawConvCoeff (vecAdd x y) b n = rawConvCoeff x b n + rawConvCoeff y b n := by
  unfold rawConvCoeff
  have hxy : x.size = y.size := by simpa [hx, hy]
  let t1 : Nat → F := fun i =>
    if hIn : i ≤ n ∧ n - i < d then
      coeffAt x i * coeffAt b (n - i)
    else
      0
  let t2 : Nat → F := fun i =>
    if hIn : i ≤ n ∧ n - i < d then
      coeffAt y i * coeffAt b (n - i)
    else
      0
  have hLeft :
      (List.range d).foldl
          (fun acc i =>
            if hIn : i ≤ n ∧ n - i < d then
              acc + coeffAt (vecAdd x y) i * coeffAt b (n - i)
            else
              acc)
          0
      =
      (List.range d).foldl (fun acc i => acc + (t1 i + t2 i)) 0 := by
    apply list_foldl_congr_mem
    intro acc i hiMem
    have hi : i < d := by simpa [List.mem_range] using hiMem
    by_cases hIn : i ≤ n ∧ n - i < d
    · have hCoeff :
          coeffAt (vecAdd x y) i = coeffAt x i + coeffAt y i := by
        exact coeffAt_vecAdd_of_size_d x y hx hy i hi
      have hDist :
          (coeffAt x i + coeffAt y i) * coeffAt b (n - i) =
            coeffAt x i * coeffAt b (n - i) + coeffAt y i * coeffAt b (n - i) := by
        exact f_right_distrib (coeffAt x i) (coeffAt y i) (coeffAt b (n - i))
      simp [hIn, t1, t2, hCoeff, hDist]
    · simp [hIn, t1, t2]
  have hFold :
      (List.range d).foldl (fun acc i => acc + (t1 i + t2 i)) 0 =
        (List.range d).foldl (fun acc i => acc + t1 i) 0 +
          (List.range d).foldl (fun acc i => acc + t2 i) 0 := by
    simpa using (foldl_add_linearity_F (l := List.range d) (t1 := t1) (t2 := t2) (acc1 := 0) (acc2 := 0))
  have hRight1 :
      (List.range d).foldl (fun acc i => acc + t1 i) 0 = rawConvCoeff x b n := by
    simpa [t1] using (rawConvCoeff_eq_fold_add_if x b n).symm
  have hRight2 :
      (List.range d).foldl (fun acc i => acc + t2 i) 0 = rawConvCoeff y b n := by
    simpa [t2] using (rawConvCoeff_eq_fold_add_if y b n).symm
  calc
    rawConvCoeff (vecAdd x y) b n
        =
      (List.range d).foldl
          (fun acc i =>
            if hIn : i ≤ n ∧ n - i < d then
              acc + coeffAt (vecAdd x y) i * coeffAt b (n - i)
            else
              acc)
          0 := by rfl
    _ = (List.range d).foldl (fun acc i => acc + (t1 i + t2 i)) 0 := hLeft
    _ = (List.range d).foldl (fun acc i => acc + t1 i) 0 +
          (List.range d).foldl (fun acc i => acc + t2 i) 0 := hFold
    _ = rawConvCoeff x b n + rawConvCoeff y b n := by
      simp [hRight1, hRight2]

theorem rawConvCoeff_vecAdd_right_of_size_d
  (a x y : Coeffs)
  (hx : x.size = d) (hy : y.size = d)
  (n : Nat) :
  rawConvCoeff a (vecAdd x y) n = rawConvCoeff a x n + rawConvCoeff a y n := by
  unfold rawConvCoeff
  have hxy : x.size = y.size := by simpa [hx, hy]
  let t1 : Nat → F := fun i =>
    if hIn : i ≤ n ∧ n - i < d then
      coeffAt a i * coeffAt x (n - i)
    else
      0
  let t2 : Nat → F := fun i =>
    if hIn : i ≤ n ∧ n - i < d then
      coeffAt a i * coeffAt y (n - i)
    else
      0
  have hLeft :
      (List.range d).foldl
          (fun acc i =>
            if hIn : i ≤ n ∧ n - i < d then
              acc + coeffAt a i * coeffAt (vecAdd x y) (n - i)
            else
              acc)
          0
      =
      (List.range d).foldl (fun acc i => acc + (t1 i + t2 i)) 0 := by
    apply list_foldl_congr_mem
    intro acc i hiMem
    have hi : i < d := by simpa [List.mem_range] using hiMem
    by_cases hIn : i ≤ n ∧ n - i < d
    · have hIdx : n - i < d := hIn.2
      have hCoeff :
          coeffAt (vecAdd x y) (n - i) = coeffAt x (n - i) + coeffAt y (n - i) := by
        exact coeffAt_vecAdd_of_size_d x y hx hy (n - i) hIdx
      have hDist :
          coeffAt a i * (coeffAt x (n - i) + coeffAt y (n - i)) =
            coeffAt a i * coeffAt x (n - i) + coeffAt a i * coeffAt y (n - i) := by
        simpa using
          (Lean.Grind.Fin.left_distrib (n := Goldilocks.q)
            (coeffAt a i) (coeffAt x (n - i)) (coeffAt y (n - i)))
      simp [hIn, t1, t2, hCoeff, hDist]
    · simp [hIn, t1, t2]
  have hFold :
      (List.range d).foldl (fun acc i => acc + (t1 i + t2 i)) 0 =
        (List.range d).foldl (fun acc i => acc + t1 i) 0 +
          (List.range d).foldl (fun acc i => acc + t2 i) 0 := by
    simpa using (foldl_add_linearity_F (l := List.range d) (t1 := t1) (t2 := t2) (acc1 := 0) (acc2 := 0))
  have hRight1 :
      (List.range d).foldl (fun acc i => acc + t1 i) 0 = rawConvCoeff a x n := by
    simpa [t1] using (rawConvCoeff_eq_fold_add_if a x n).symm
  have hRight2 :
      (List.range d).foldl (fun acc i => acc + t2 i) 0 = rawConvCoeff a y n := by
    simpa [t2] using (rawConvCoeff_eq_fold_add_if a y n).symm
  calc
    rawConvCoeff a (vecAdd x y) n
        =
      (List.range d).foldl
          (fun acc i =>
            if hIn : i ≤ n ∧ n - i < d then
              acc + coeffAt a i * coeffAt (vecAdd x y) (n - i)
            else
              acc)
          0 := by rfl
    _ = (List.range d).foldl (fun acc i => acc + (t1 i + t2 i)) 0 := hLeft
    _ = (List.range d).foldl (fun acc i => acc + t1 i) 0 +
          (List.range d).foldl (fun acc i => acc + t2 i) 0 := hFold
    _ = rawConvCoeff a x n + rawConvCoeff a y n := by
      simp [hRight1, hRight2]

theorem rawConvCoeff_vecScale_left_of_size_d
  (s : F) (x b : Coeffs)
  (hx : x.size = d)
  (n : Nat) :
  rawConvCoeff (vecScale s x) b n = s * rawConvCoeff x b n := by
  unfold rawConvCoeff
  let t : Nat → F := fun i =>
    if hIn : i ≤ n ∧ n - i < d then
      coeffAt x i * coeffAt b (n - i)
    else
      0
  have hLeft :
      (List.range d).foldl
          (fun acc i =>
            if hIn : i ≤ n ∧ n - i < d then
              acc + coeffAt (vecScale s x) i * coeffAt b (n - i)
            else
              acc)
          0
      =
      (List.range d).foldl (fun acc i => acc + s * t i) 0 := by
    apply list_foldl_congr_mem
    intro acc i hiMem
    have hi : i < d := by simpa [List.mem_range] using hiMem
    by_cases hIn : i ≤ n ∧ n - i < d
    · have hCoeff :
          coeffAt (vecScale s x) i = s * coeffAt x i := by
        exact coeffAt_vecScale_of_size_d s x hx i hi
      simp [hIn, t, hCoeff, Lean.Grind.Fin.mul_assoc]
    · have hti : t i = 0 := by simp [t, hIn]
      have h0 : s * (0 : F) = 0 := by simpa using f_mul_zero s
      simp [hIn, hti, h0]
  have hFold :
      (List.range d).foldl (fun acc i => acc + s * t i) 0 =
        s * ((List.range d).foldl (fun acc i => acc + t i) 0) := by
    have hTmp := (foldl_mul_left_distrib_F (l := List.range d) (t := t) (acc := 0) (c := s))
    simpa [f_mul_zero] using hTmp.symm
  have hBase :
      (List.range d).foldl (fun acc i => acc + t i) 0 = rawConvCoeff x b n := by
    simpa [t] using (rawConvCoeff_eq_fold_add_if x b n).symm
  calc
    rawConvCoeff (vecScale s x) b n
        =
      (List.range d).foldl
          (fun acc i =>
            if hIn : i ≤ n ∧ n - i < d then
              acc + coeffAt (vecScale s x) i * coeffAt b (n - i)
            else
              acc)
          0 := by rfl
    _ = (List.range d).foldl (fun acc i => acc + s * t i) 0 := hLeft
    _ = s * ((List.range d).foldl (fun acc i => acc + t i) 0) := hFold
    _ = s * rawConvCoeff x b n := by simp [hBase]

theorem rawConvCoeff_vecScale_right_of_size_d
  (s : F) (a x : Coeffs)
  (hx : x.size = d)
  (n : Nat) :
  rawConvCoeff a (vecScale s x) n = s * rawConvCoeff a x n := by
  unfold rawConvCoeff
  let t : Nat → F := fun i =>
    if hIn : i ≤ n ∧ n - i < d then
      coeffAt a i * coeffAt x (n - i)
    else
      0
  have hLeft :
      (List.range d).foldl
          (fun acc i =>
            if hIn : i ≤ n ∧ n - i < d then
              acc + coeffAt a i * coeffAt (vecScale s x) (n - i)
            else
              acc)
          0
      =
      (List.range d).foldl (fun acc i => acc + s * t i) 0 := by
    apply list_foldl_congr_mem
    intro acc i hiMem
    have hi : i < d := by simpa [List.mem_range] using hiMem
    by_cases hIn : i ≤ n ∧ n - i < d
    · have hCoeff :
          coeffAt (vecScale s x) (n - i) = s * coeffAt x (n - i) := by
        exact coeffAt_vecScale_of_size_d s x hx (n - i) hIn.2
      have hTerm :
          coeffAt a i * coeffAt (vecScale s x) (n - i) = s * t i := by
        calc
          coeffAt a i * coeffAt (vecScale s x) (n - i)
              = coeffAt a i * (s * coeffAt x (n - i)) := by simp [hCoeff]
          _ = (coeffAt a i * s) * coeffAt x (n - i) := by
                simpa using (Lean.Grind.Fin.mul_assoc (n := Goldilocks.q) (coeffAt a i) s (coeffAt x (n - i))).symm
          _ = (s * coeffAt a i) * coeffAt x (n - i) := by
                simpa using congrArg (fun t => t * coeffAt x (n - i))
                  (Lean.Grind.Fin.mul_comm (n := Goldilocks.q) (coeffAt a i) s)
          _ = s * (coeffAt a i * coeffAt x (n - i)) := by
                simpa using (Lean.Grind.Fin.mul_assoc (n := Goldilocks.q) s (coeffAt a i) (coeffAt x (n - i)))
          _ = s * t i := by simp [t, hIn]
      simpa [hIn] using congrArg (fun u => acc + u) hTerm
    · have hti : t i = 0 := by simp [t, hIn]
      have h0 : s * (0 : F) = 0 := by simpa using f_mul_zero s
      simp [hIn, hti, h0]
  have hFold :
      (List.range d).foldl (fun acc i => acc + s * t i) 0 =
        s * ((List.range d).foldl (fun acc i => acc + t i) 0) := by
    have hTmp := (foldl_mul_left_distrib_F (l := List.range d) (t := t) (acc := 0) (c := s))
    simpa [f_mul_zero] using hTmp.symm
  have hBase :
      (List.range d).foldl (fun acc i => acc + t i) 0 = rawConvCoeff a x n := by
    simpa [t] using (rawConvCoeff_eq_fold_add_if a x n).symm
  calc
    rawConvCoeff a (vecScale s x) n
        =
      (List.range d).foldl
          (fun acc i =>
            if hIn : i ≤ n ∧ n - i < d then
              acc + coeffAt a i * coeffAt (vecScale s x) (n - i)
            else
              acc)
          0 := by rfl
    _ = (List.range d).foldl (fun acc i => acc + s * t i) 0 := hLeft
    _ = s * ((List.range d).foldl (fun acc i => acc + t i) 0) := hFold
    _ = s * rawConvCoeff a x n := by simp [hBase]

theorem ct_mulRqPhi_vecAdd_left_of_size_d
  (x y b : Coeffs)
  (hx : x.size = d) (hy : y.size = d) :
  ct (mulRqPhi (vecAdd x y) b) =
    ct (mulRqPhi x b) + ct (mulRqPhi y b) := by
  have h0 := rawConvCoeff_vecAdd_left_of_size_d x y b hx hy 0
  have h54 := rawConvCoeff_vecAdd_left_of_size_d x y b hx hy 54
  have h81 := rawConvCoeff_vecAdd_left_of_size_d x y b hx hy 81
  calc
    ct (mulRqPhi (vecAdd x y) b)
        = rawConvCoeff (vecAdd x y) b 0
            - rawConvCoeff (vecAdd x y) b 54
            + rawConvCoeff (vecAdd x y) b 81 := by
            exact ct_mulRqPhi_eq_raw (vecAdd x y) b
    _ = (rawConvCoeff x b 0 + rawConvCoeff y b 0)
          - (rawConvCoeff x b 54 + rawConvCoeff y b 54)
          + (rawConvCoeff x b 81 + rawConvCoeff y b 81) := by
            simp [h0, h54, h81]
    _ = (rawConvCoeff x b 0 - rawConvCoeff x b 54 + rawConvCoeff x b 81) +
          (rawConvCoeff y b 0 - rawConvCoeff y b 54 + rawConvCoeff y b 81) := by
            exact sum_sub_sum_add_sum_split
              (rawConvCoeff x b 0) (rawConvCoeff y b 0)
              (rawConvCoeff x b 54) (rawConvCoeff y b 54)
              (rawConvCoeff x b 81) (rawConvCoeff y b 81)
    _ = ct (mulRqPhi x b) + ct (mulRqPhi y b) := by
          simp [ct_mulRqPhi_eq_raw]

theorem ct_mulRqPhi_vecAdd_right_of_size_d
  (a x y : Coeffs)
  (hx : x.size = d) (hy : y.size = d) :
  ct (mulRqPhi a (vecAdd x y)) =
    ct (mulRqPhi a x) + ct (mulRqPhi a y) := by
  have h0 := rawConvCoeff_vecAdd_right_of_size_d a x y hx hy 0
  have h54 := rawConvCoeff_vecAdd_right_of_size_d a x y hx hy 54
  have h81 := rawConvCoeff_vecAdd_right_of_size_d a x y hx hy 81
  calc
    ct (mulRqPhi a (vecAdd x y))
        = rawConvCoeff a (vecAdd x y) 0
            - rawConvCoeff a (vecAdd x y) 54
            + rawConvCoeff a (vecAdd x y) 81 := by
            exact ct_mulRqPhi_eq_raw a (vecAdd x y)
    _ = (rawConvCoeff a x 0 + rawConvCoeff a y 0)
          - (rawConvCoeff a x 54 + rawConvCoeff a y 54)
          + (rawConvCoeff a x 81 + rawConvCoeff a y 81) := by
            simp [h0, h54, h81]
    _ = (rawConvCoeff a x 0 - rawConvCoeff a x 54 + rawConvCoeff a x 81) +
          (rawConvCoeff a y 0 - rawConvCoeff a y 54 + rawConvCoeff a y 81) := by
            exact sum_sub_sum_add_sum_split
              (rawConvCoeff a x 0) (rawConvCoeff a y 0)
              (rawConvCoeff a x 54) (rawConvCoeff a y 54)
              (rawConvCoeff a x 81) (rawConvCoeff a y 81)
    _ = ct (mulRqPhi a x) + ct (mulRqPhi a y) := by
          simp [ct_mulRqPhi_eq_raw]

theorem ct_mulRqPhi_vecScale_left_of_size_d
  (s : F) (x b : Coeffs)
  (hx : x.size = d) :
  ct (mulRqPhi (vecScale s x) b) = s * ct (mulRqPhi x b) := by
  have h0 := rawConvCoeff_vecScale_left_of_size_d s x b hx 0
  have h54 := rawConvCoeff_vecScale_left_of_size_d s x b hx 54
  have h81 := rawConvCoeff_vecScale_left_of_size_d s x b hx 81
  calc
    ct (mulRqPhi (vecScale s x) b)
        = rawConvCoeff (vecScale s x) b 0
            - rawConvCoeff (vecScale s x) b 54
            + rawConvCoeff (vecScale s x) b 81 := by
            exact ct_mulRqPhi_eq_raw (vecScale s x) b
    _ = s * rawConvCoeff x b 0 - s * rawConvCoeff x b 54 + s * rawConvCoeff x b 81 := by
          simp [h0, h54, h81]
    _ = s * (rawConvCoeff x b 0 - rawConvCoeff x b 54 + rawConvCoeff x b 81) := by
          exact scale_sub_add_factor s (rawConvCoeff x b 0) (rawConvCoeff x b 54) (rawConvCoeff x b 81)
    _ = s * ct (mulRqPhi x b) := by
          simp [ct_mulRqPhi_eq_raw]

theorem ct_mulRqPhi_vecScale_right_of_size_d
  (s : F) (a x : Coeffs)
  (hx : x.size = d) :
  ct (mulRqPhi a (vecScale s x)) = s * ct (mulRqPhi a x) := by
  have h0 := rawConvCoeff_vecScale_right_of_size_d s a x hx 0
  have h54 := rawConvCoeff_vecScale_right_of_size_d s a x hx 54
  have h81 := rawConvCoeff_vecScale_right_of_size_d s a x hx 81
  calc
    ct (mulRqPhi a (vecScale s x))
        = rawConvCoeff a (vecScale s x) 0
            - rawConvCoeff a (vecScale s x) 54
            + rawConvCoeff a (vecScale s x) 81 := by
            exact ct_mulRqPhi_eq_raw a (vecScale s x)
    _ = s * rawConvCoeff a x 0 - s * rawConvCoeff a x 54 + s * rawConvCoeff a x 81 := by
          simp [h0, h54, h81]
    _ = s * (rawConvCoeff a x 0 - rawConvCoeff a x 54 + rawConvCoeff a x 81) := by
          exact scale_sub_add_factor s (rawConvCoeff a x 0) (rawConvCoeff a x 54) (rawConvCoeff a x 81)
    _ = s * ct (mulRqPhi a x) := by
          simp [ct_mulRqPhi_eq_raw]

theorem mulRqPhi_vecAdd_left_of_size_d
  (x y b : Coeffs)
  (hx : x.size = d) (hy : y.size = d) :
  mulRqPhi (vecAdd x y) b = vecAdd (mulRqPhi x b) (mulRqPhi y b) := by
  have hEq : (mulRqPhi x b).size = (mulRqPhi y b).size := by
    simp [mulRqPhi_size]
  have hsizeR : (vecAdd (mulRqPhi x b) (mulRqPhi y b)).size = d := by
    calc
      (vecAdd (mulRqPhi x b) (mulRqPhi y b)).size = (mulRqPhi x b).size := vecAdd_size_of_eq hEq
      _ = d := mulRqPhi_size x b
  apply Array.ext
  · exact (mulRqPhi_size (vecAdd x y) b).trans hsizeR.symm
  · intro i hiL hiR
    have hi : i < d := by simpa [mulRqPhi_size] using hiL
    have hiR' : i < d := by simpa [hsizeR] using hiR
    have h0 := rawConvCoeff_vecAdd_left_of_size_d x y b hx hy i
    have h54 := rawConvCoeff_vecAdd_left_of_size_d x y b hx hy (i + 54)
    have h80 := rawConvCoeff_vecAdd_left_of_size_d x y b hx hy 80
    have h81 := rawConvCoeff_vecAdd_left_of_size_d x y b hx hy (i + 81)
    have h27 := rawConvCoeff_vecAdd_left_of_size_d x y b hx hy (i + 27)
    have hVecCoeff :
        coeffAt (vecAdd (mulRqPhi x b) (mulRqPhi y b)) i =
          coeffAt (mulRqPhi x b) i + coeffAt (mulRqPhi y b) i := by
      exact coeffAt_vecAdd_of_size_d (mulRqPhi x b) (mulRqPhi y b)
        (mulRqPhi_size x b) (mulRqPhi_size y b) i hi
    by_cases hLt26 : i < 26
    · have hCoeff :
        coeffAt (mulRqPhi (vecAdd x y) b) i =
          coeffAt (vecAdd (mulRqPhi x b) (mulRqPhi y b)) i := by
        calc
          coeffAt (mulRqPhi (vecAdd x y) b) i
              = (rawConvCoeff x b i + rawConvCoeff y b i) -
                  (rawConvCoeff x b (i + 54) + rawConvCoeff y b (i + 54)) +
                  (rawConvCoeff x b (i + 81) + rawConvCoeff y b (i + 81)) := by
                    simp [coeffAt, hi, mulRqPhi, hLt26, h0, h54, h81]
          _ = (rawConvCoeff x b i - rawConvCoeff x b (i + 54) + rawConvCoeff x b (i + 81)) +
                (rawConvCoeff y b i - rawConvCoeff y b (i + 54) + rawConvCoeff y b (i + 81)) := by
                  exact sum_sub_sum_add_sum_split
                    (rawConvCoeff x b i) (rawConvCoeff y b i)
                    (rawConvCoeff x b (i + 54)) (rawConvCoeff y b (i + 54))
                    (rawConvCoeff x b (i + 81)) (rawConvCoeff y b (i + 81))
          _ = coeffAt (mulRqPhi x b) i + coeffAt (mulRqPhi y b) i := by
                simp [coeffAt, hi, mulRqPhi, hLt26]
          _ = coeffAt (vecAdd (mulRqPhi x b) (mulRqPhi y b)) i := by
                symm
                exact hVecCoeff
      simpa [coeffAt, hi, hiR', Array.getD, hiL, hiR, hsizeR] using hCoeff
    · by_cases hEq26 : i = 26
      · have hCoeff :
          coeffAt (mulRqPhi (vecAdd x y) b) i =
            coeffAt (vecAdd (mulRqPhi x b) (mulRqPhi y b)) i := by
          subst hEq26
          calc
            coeffAt (mulRqPhi (vecAdd x y) b) 26
                = (rawConvCoeff x b 26 + rawConvCoeff y b 26) -
                    (rawConvCoeff x b 80 + rawConvCoeff y b 80) := by
                      simp [coeffAt, mulRqPhi, d, hLt26, h0, h80]
            _ = (rawConvCoeff x b 26 - rawConvCoeff x b 80) +
                  (rawConvCoeff y b 26 - rawConvCoeff y b 80) := by
                    exact sum_sub_sum_split
                      (rawConvCoeff x b 26) (rawConvCoeff y b 26)
                      (rawConvCoeff x b 80) (rawConvCoeff y b 80)
            _ = coeffAt (mulRqPhi x b) 26 + coeffAt (mulRqPhi y b) 26 := by
                  simp [coeffAt, mulRqPhi, d, hLt26]
            _ = coeffAt (vecAdd (mulRqPhi x b) (mulRqPhi y b)) 26 := by
                  symm
                  exact hVecCoeff
        simpa [coeffAt, hi, hiR', Array.getD, hiL, hiR, hsizeR] using hCoeff
      · have hCoeff :
          coeffAt (mulRqPhi (vecAdd x y) b) i =
            coeffAt (vecAdd (mulRqPhi x b) (mulRqPhi y b)) i := by
          calc
            coeffAt (mulRqPhi (vecAdd x y) b) i
                = (rawConvCoeff x b i + rawConvCoeff y b i) -
                    (rawConvCoeff x b (i + 27) + rawConvCoeff y b (i + 27)) := by
                      simp [coeffAt, hi, mulRqPhi, hLt26, hEq26, h0, h27]
            _ = (rawConvCoeff x b i - rawConvCoeff x b (i + 27)) +
                  (rawConvCoeff y b i - rawConvCoeff y b (i + 27)) := by
                    exact sum_sub_sum_split
                      (rawConvCoeff x b i) (rawConvCoeff y b i)
                      (rawConvCoeff x b (i + 27)) (rawConvCoeff y b (i + 27))
            _ = coeffAt (mulRqPhi x b) i + coeffAt (mulRqPhi y b) i := by
                  simp [coeffAt, hi, mulRqPhi, hLt26, hEq26]
            _ = coeffAt (vecAdd (mulRqPhi x b) (mulRqPhi y b)) i := by
                  symm
                  exact hVecCoeff
        simpa [coeffAt, hi, hiR', Array.getD, hiL, hiR, hsizeR] using hCoeff

theorem mulRqPhi_vecAdd_right_of_size_d
  (a x y : Coeffs)
  (hx : x.size = d) (hy : y.size = d) :
  mulRqPhi a (vecAdd x y) = vecAdd (mulRqPhi a x) (mulRqPhi a y) := by
  have hEq : (mulRqPhi a x).size = (mulRqPhi a y).size := by
    simp [mulRqPhi_size]
  have hsizeR : (vecAdd (mulRqPhi a x) (mulRqPhi a y)).size = d := by
    calc
      (vecAdd (mulRqPhi a x) (mulRqPhi a y)).size = (mulRqPhi a x).size := vecAdd_size_of_eq hEq
      _ = d := mulRqPhi_size a x
  apply Array.ext
  · exact (mulRqPhi_size a (vecAdd x y)).trans hsizeR.symm
  · intro i hiL hiR
    have hi : i < d := by simpa [mulRqPhi_size] using hiL
    have hiR' : i < d := by simpa [hsizeR] using hiR
    have h0 := rawConvCoeff_vecAdd_right_of_size_d a x y hx hy i
    have h54 := rawConvCoeff_vecAdd_right_of_size_d a x y hx hy (i + 54)
    have h80 := rawConvCoeff_vecAdd_right_of_size_d a x y hx hy 80
    have h81 := rawConvCoeff_vecAdd_right_of_size_d a x y hx hy (i + 81)
    have h27 := rawConvCoeff_vecAdd_right_of_size_d a x y hx hy (i + 27)
    have hVecCoeff :
        coeffAt (vecAdd (mulRqPhi a x) (mulRqPhi a y)) i =
          coeffAt (mulRqPhi a x) i + coeffAt (mulRqPhi a y) i := by
      exact coeffAt_vecAdd_of_size_d (mulRqPhi a x) (mulRqPhi a y)
        (mulRqPhi_size a x) (mulRqPhi_size a y) i hi
    by_cases hLt26 : i < 26
    · have hCoeff :
        coeffAt (mulRqPhi a (vecAdd x y)) i =
          coeffAt (vecAdd (mulRqPhi a x) (mulRqPhi a y)) i := by
        calc
          coeffAt (mulRqPhi a (vecAdd x y)) i
              = (rawConvCoeff a x i + rawConvCoeff a y i) -
                  (rawConvCoeff a x (i + 54) + rawConvCoeff a y (i + 54)) +
                  (rawConvCoeff a x (i + 81) + rawConvCoeff a y (i + 81)) := by
                    simp [coeffAt, hi, mulRqPhi, hLt26, h0, h54, h81]
          _ = (rawConvCoeff a x i - rawConvCoeff a x (i + 54) + rawConvCoeff a x (i + 81)) +
                (rawConvCoeff a y i - rawConvCoeff a y (i + 54) + rawConvCoeff a y (i + 81)) := by
                  exact sum_sub_sum_add_sum_split
                    (rawConvCoeff a x i) (rawConvCoeff a y i)
                    (rawConvCoeff a x (i + 54)) (rawConvCoeff a y (i + 54))
                    (rawConvCoeff a x (i + 81)) (rawConvCoeff a y (i + 81))
          _ = coeffAt (mulRqPhi a x) i + coeffAt (mulRqPhi a y) i := by
                simp [coeffAt, hi, mulRqPhi, hLt26]
          _ = coeffAt (vecAdd (mulRqPhi a x) (mulRqPhi a y)) i := by
                symm
                exact hVecCoeff
      simpa [coeffAt, hi, hiR', Array.getD, hiL, hiR, hsizeR] using hCoeff
    · by_cases hEq26 : i = 26
      · have hCoeff :
          coeffAt (mulRqPhi a (vecAdd x y)) i =
            coeffAt (vecAdd (mulRqPhi a x) (mulRqPhi a y)) i := by
          subst hEq26
          calc
            coeffAt (mulRqPhi a (vecAdd x y)) 26
                = (rawConvCoeff a x 26 + rawConvCoeff a y 26) -
                    (rawConvCoeff a x 80 + rawConvCoeff a y 80) := by
                      simp [coeffAt, mulRqPhi, d, hLt26, h0, h80]
            _ = (rawConvCoeff a x 26 - rawConvCoeff a x 80) +
                  (rawConvCoeff a y 26 - rawConvCoeff a y 80) := by
                    exact sum_sub_sum_split
                      (rawConvCoeff a x 26) (rawConvCoeff a y 26)
                      (rawConvCoeff a x 80) (rawConvCoeff a y 80)
            _ = coeffAt (mulRqPhi a x) 26 + coeffAt (mulRqPhi a y) 26 := by
                  simp [coeffAt, mulRqPhi, d, hLt26]
            _ = coeffAt (vecAdd (mulRqPhi a x) (mulRqPhi a y)) 26 := by
                  symm
                  exact hVecCoeff
        simpa [coeffAt, hi, hiR', Array.getD, hiL, hiR, hsizeR] using hCoeff
      · have hCoeff :
          coeffAt (mulRqPhi a (vecAdd x y)) i =
            coeffAt (vecAdd (mulRqPhi a x) (mulRqPhi a y)) i := by
          calc
            coeffAt (mulRqPhi a (vecAdd x y)) i
                = (rawConvCoeff a x i + rawConvCoeff a y i) -
                    (rawConvCoeff a x (i + 27) + rawConvCoeff a y (i + 27)) := by
                      simp [coeffAt, hi, mulRqPhi, hLt26, hEq26, h0, h27]
            _ = (rawConvCoeff a x i - rawConvCoeff a x (i + 27)) +
                  (rawConvCoeff a y i - rawConvCoeff a y (i + 27)) := by
                    exact sum_sub_sum_split
                      (rawConvCoeff a x i) (rawConvCoeff a y i)
                      (rawConvCoeff a x (i + 27)) (rawConvCoeff a y (i + 27))
            _ = coeffAt (mulRqPhi a x) i + coeffAt (mulRqPhi a y) i := by
                  simp [coeffAt, hi, mulRqPhi, hLt26, hEq26]
            _ = coeffAt (vecAdd (mulRqPhi a x) (mulRqPhi a y)) i := by
                  symm
                  exact hVecCoeff
        simpa [coeffAt, hi, hiR', Array.getD, hiL, hiR, hsizeR] using hCoeff

theorem mulRqPhi_vecScale_left_of_size_d
  (s : F) (x b : Coeffs)
  (hx : x.size = d) :
  mulRqPhi (vecScale s x) b = vecScale s (mulRqPhi x b) := by
  apply Array.ext
  · simp [mulRqPhi, vecScale]
  · intro i hiL hiR
    have hi : i < d := by simpa [mulRqPhi] using hiL
    have h0 := rawConvCoeff_vecScale_left_of_size_d s x b hx i
    have h54 := rawConvCoeff_vecScale_left_of_size_d s x b hx (i + 54)
    have h80 := rawConvCoeff_vecScale_left_of_size_d s x b hx 80
    have h81 := rawConvCoeff_vecScale_left_of_size_d s x b hx (i + 81)
    have h27 := rawConvCoeff_vecScale_left_of_size_d s x b hx (i + 27)
    by_cases hLt26 : i < 26
    · have hCoeff :
          coeffAt (mulRqPhi (vecScale s x) b) i =
            coeffAt (vecScale s (mulRqPhi x b)) i := by
        calc
          coeffAt (mulRqPhi (vecScale s x) b) i
              = s * rawConvCoeff x b i - s * rawConvCoeff x b (i + 54) + s * rawConvCoeff x b (i + 81) := by
                  simp [coeffAt, hi, mulRqPhi, hLt26, h0, h54, h81]
          _ = s * (rawConvCoeff x b i - rawConvCoeff x b (i + 54) + rawConvCoeff x b (i + 81)) := by
                exact scale_sub_add_factor s (rawConvCoeff x b i) (rawConvCoeff x b (i + 54)) (rawConvCoeff x b (i + 81))
          _ = s * coeffAt (mulRqPhi x b) i := by
                simp [coeffAt, hi, mulRqPhi, hLt26]
          _ = coeffAt (vecScale s (mulRqPhi x b)) i := by
                symm
                exact coeffAt_vecScale_of_size_d s (mulRqPhi x b) (mulRqPhi_size x b) i hi
      simpa [coeffAt, hi, Array.getD, hiL, hiR] using hCoeff
    · by_cases hEq26 : i = 26
      · have hCoeff :
            coeffAt (mulRqPhi (vecScale s x) b) i =
              coeffAt (vecScale s (mulRqPhi x b)) i := by
          subst hEq26
          calc
            coeffAt (mulRqPhi (vecScale s x) b) 26
                = s * rawConvCoeff x b 26 - s * rawConvCoeff x b 80 := by
                    simp [coeffAt, mulRqPhi, d, hLt26, h0, h80]
            _ = s * (rawConvCoeff x b 26 - rawConvCoeff x b 80) := by
                  exact scale_sub_factor s (rawConvCoeff x b 26) (rawConvCoeff x b 80)
            _ = s * coeffAt (mulRqPhi x b) 26 := by
                  simp [coeffAt, mulRqPhi, d, hLt26]
            _ = coeffAt (vecScale s (mulRqPhi x b)) 26 := by
                  symm
                  exact coeffAt_vecScale_of_size_d s (mulRqPhi x b) (mulRqPhi_size x b) 26 (by simp [d])
        simpa [coeffAt, hi, Array.getD, hiL, hiR] using hCoeff
      · have hCoeff :
            coeffAt (mulRqPhi (vecScale s x) b) i =
              coeffAt (vecScale s (mulRqPhi x b)) i := by
          calc
            coeffAt (mulRqPhi (vecScale s x) b) i
                = s * rawConvCoeff x b i - s * rawConvCoeff x b (i + 27) := by
                    simp [coeffAt, hi, mulRqPhi, hLt26, hEq26, h0, h27]
            _ = s * (rawConvCoeff x b i - rawConvCoeff x b (i + 27)) := by
                  exact scale_sub_factor s (rawConvCoeff x b i) (rawConvCoeff x b (i + 27))
            _ = s * coeffAt (mulRqPhi x b) i := by
                  simp [coeffAt, hi, mulRqPhi, hLt26, hEq26]
            _ = coeffAt (vecScale s (mulRqPhi x b)) i := by
                  symm
                  exact coeffAt_vecScale_of_size_d s (mulRqPhi x b) (mulRqPhi_size x b) i hi
        simpa [coeffAt, hi, Array.getD, hiL, hiR] using hCoeff

theorem mulRqPhi_vecScale_right_of_size_d
  (s : F) (a x : Coeffs)
  (hx : x.size = d) :
  mulRqPhi a (vecScale s x) = vecScale s (mulRqPhi a x) := by
  apply Array.ext
  · simp [mulRqPhi, vecScale]
  · intro i hiL hiR
    have hi : i < d := by simpa [mulRqPhi] using hiL
    have h0 := rawConvCoeff_vecScale_right_of_size_d s a x hx i
    have h54 := rawConvCoeff_vecScale_right_of_size_d s a x hx (i + 54)
    have h80 := rawConvCoeff_vecScale_right_of_size_d s a x hx 80
    have h81 := rawConvCoeff_vecScale_right_of_size_d s a x hx (i + 81)
    have h27 := rawConvCoeff_vecScale_right_of_size_d s a x hx (i + 27)
    by_cases hLt26 : i < 26
    · have hCoeff :
          coeffAt (mulRqPhi a (vecScale s x)) i =
            coeffAt (vecScale s (mulRqPhi a x)) i := by
        calc
          coeffAt (mulRqPhi a (vecScale s x)) i
              = s * rawConvCoeff a x i - s * rawConvCoeff a x (i + 54) + s * rawConvCoeff a x (i + 81) := by
                  simp [coeffAt, hi, mulRqPhi, hLt26, h0, h54, h81]
          _ = s * (rawConvCoeff a x i - rawConvCoeff a x (i + 54) + rawConvCoeff a x (i + 81)) := by
                exact scale_sub_add_factor s (rawConvCoeff a x i) (rawConvCoeff a x (i + 54)) (rawConvCoeff a x (i + 81))
          _ = s * coeffAt (mulRqPhi a x) i := by
                simp [coeffAt, hi, mulRqPhi, hLt26]
          _ = coeffAt (vecScale s (mulRqPhi a x)) i := by
                symm
                exact coeffAt_vecScale_of_size_d s (mulRqPhi a x) (mulRqPhi_size a x) i hi
      simpa [coeffAt, hi, Array.getD, hiL, hiR] using hCoeff
    · by_cases hEq26 : i = 26
      · have hCoeff :
            coeffAt (mulRqPhi a (vecScale s x)) i =
              coeffAt (vecScale s (mulRqPhi a x)) i := by
          subst hEq26
          calc
            coeffAt (mulRqPhi a (vecScale s x)) 26
                = s * rawConvCoeff a x 26 - s * rawConvCoeff a x 80 := by
                    simp [coeffAt, mulRqPhi, d, hLt26, h0, h80]
            _ = s * (rawConvCoeff a x 26 - rawConvCoeff a x 80) := by
                  exact scale_sub_factor s (rawConvCoeff a x 26) (rawConvCoeff a x 80)
            _ = s * coeffAt (mulRqPhi a x) 26 := by
                  simp [coeffAt, mulRqPhi, d, hLt26]
            _ = coeffAt (vecScale s (mulRqPhi a x)) 26 := by
                  symm
                  exact coeffAt_vecScale_of_size_d s (mulRqPhi a x) (mulRqPhi_size a x) 26 (by simp [d])
        simpa [coeffAt, hi, Array.getD, hiL, hiR] using hCoeff
      · have hCoeff :
            coeffAt (mulRqPhi a (vecScale s x)) i =
              coeffAt (vecScale s (mulRqPhi a x)) i := by
          calc
            coeffAt (mulRqPhi a (vecScale s x)) i
                = s * rawConvCoeff a x i - s * rawConvCoeff a x (i + 27) := by
                    simp [coeffAt, hi, mulRqPhi, hLt26, hEq26, h0, h27]
            _ = s * (rawConvCoeff a x i - rawConvCoeff a x (i + 27)) := by
                  exact scale_sub_factor s (rawConvCoeff a x i) (rawConvCoeff a x (i + 27))
            _ = s * coeffAt (mulRqPhi a x) i := by
                  simp [coeffAt, hi, mulRqPhi, hLt26, hEq26]
            _ = coeffAt (vecScale s (mulRqPhi a x)) i := by
                  symm
                  exact coeffAt_vecScale_of_size_d s (mulRqPhi a x) (mulRqPhi_size a x) i hi
        simpa [coeffAt, hi, Array.getD, hiL, hiR] using hCoeff

theorem mulRq_vecAdd_left
  (x y b : Coeffs)
  (hx : x.size = d) (hy : y.size = d) :
  mulRq (vecAdd x y) b = vecAdd (mulRq x b) (mulRq y b) := by
  simpa [mulRq] using mulRqPhi_vecAdd_left_of_size_d x y b hx hy

theorem mulRq_vecAdd_right
  (a x y : Coeffs)
  (hx : x.size = d) (hy : y.size = d) :
  mulRq a (vecAdd x y) = vecAdd (mulRq a x) (mulRq a y) := by
  simpa [mulRq] using mulRqPhi_vecAdd_right_of_size_d a x y hx hy

theorem mulRq_vecScale_left
  (s : F) (x b : Coeffs)
  (hx : x.size = d) :
  mulRq (vecScale s x) b = vecScale s (mulRq x b) := by
  simpa [mulRq] using mulRqPhi_vecScale_left_of_size_d s x b hx

theorem mulRq_vecScale_right
  (s : F) (a x : Coeffs)
  (hx : x.size = d) :
  mulRq a (vecScale s x) = vecScale s (mulRq a x) := by
  simpa [mulRq] using mulRqPhi_vecScale_right_of_size_d s a x hx

/-- `dotBySize` is additive in the right input under canonical `d`-sized shapes. -/
theorem dotBySize_vecAdd_right_of_size_d_ring
    (row x y : Array F)
    (hRow : row.size = d)
    (hx : x.size = d)
    (hy : y.size = d) :
    dotBySize row (vecAdd x y) = dotBySize row x + dotBySize row y := by
  have hxy : x.size = y.size := by simpa [hx, hy]
  have hVec : (vecAdd x y).size = d := by
    simpa [hx] using (vecAdd_size_of_eq hxy)
  have hEq : row.size = (vecAdd x y).size := by simpa [hRow, hVec]
  calc
    dotBySize row (vecAdd x y)
        = (List.range d).foldl
            (fun acc i => acc + row[i]! * (vecAdd x y)[i]!)
            0 := by
              simp [dotBySize, hEq, hVec]
    _ =
      (List.range d).foldl (fun acc i => acc + row[i]! * x[i]!) 0 +
      (List.range d).foldl (fun acc i => acc + row[i]! * y[i]!) 0 := by
        let t1 : Nat → F := fun i => row[i]! * x[i]!
        let t2 : Nat → F := fun i => row[i]! * y[i]!
        have hFold :
            (List.range d).foldl (fun acc i => acc + (t1 i + t2 i)) (0 + 0) =
              (List.range d).foldl (fun acc i => acc + t1 i) 0 +
                (List.range d).foldl (fun acc i => acc + t2 i) 0 := by
          simpa using
            (foldl_add_linearity (l := List.range d) (t1 := t1) (t2 := t2) (acc1 := 0) (acc2 := 0))
        have hPointwise :
            (List.range d).foldl
                (fun acc i => acc + row[i]! * (vecAdd x y)[i]!)
                0
              =
            (List.range d).foldl (fun acc i => acc + (t1 i + t2 i)) 0 := by
          exact list_foldl_congr_mem
            (fun acc i => acc + row[i]! * (vecAdd x y)[i]!)
            (fun acc i => acc + (t1 i + t2 i))
            0
            (List.range d)
            (by
              intro acc i hiMem
              have hi : i < d := by simpa [List.mem_range] using hiMem
              have hVecIdx :
                  (vecAdd x y)[i]! = x[i]! + y[i]! := by
                have hCoeff :
                    coeffAt (vecAdd x y) i = coeffAt x i + coeffAt y i :=
                  coeffAt_vecAdd_of_size_d x y hx hy i hi
                have hGetD :
                    (vecAdd x y).getD i 0 = x.getD i 0 + y.getD i 0 := by
                  simpa [coeffAt, hi] using hCoeff
                have hL : (vecAdd x y)[i]! = (vecAdd x y).getD i 0 := by
                  simpa using (Array.getElem!_eq_getD (xs := (vecAdd x y)) (i := i))
                have hX : x[i]! = x.getD i 0 := by
                  simpa using (Array.getElem!_eq_getD (xs := x) (i := i))
                have hY : y[i]! = y.getD i 0 := by
                  simpa using (Array.getElem!_eq_getD (xs := y) (i := i))
                calc
                  (vecAdd x y)[i]!
                      = (vecAdd x y).getD i 0 := hL
                  _ = x.getD i 0 + y.getD i 0 := hGetD
                  _ = x[i]! + y[i]! := by simpa [hX, hY]
              have hDistrib :
                  row[i]! * (x[i]! + y[i]!) = t1 i + t2 i := by
                simp [t1, t2, Lean.Grind.Fin.left_distrib]
              simpa [hVecIdx] using congrArg (fun u => acc + u) hDistrib)
        calc
          (List.range d).foldl
              (fun acc i => acc + row[i]! * (vecAdd x y)[i]!)
              0
              = (List.range d).foldl (fun acc i => acc + (t1 i + t2 i)) 0 := hPointwise
          _ = (List.range d).foldl (fun acc i => acc + t1 i) 0 +
                (List.range d).foldl (fun acc i => acc + t2 i) 0 := by
                simpa using hFold
    _ = dotBySize row x + dotBySize row y := by
          simp [dotBySize, hRow, hx, hy]

/-- `dotBySize` is homogeneous in the right input under canonical `d`-sized shapes. -/
theorem dotBySize_vecScale_right_of_size_d_ring
    (row x : Array F)
    (s : F)
    (hRow : row.size = d)
    (hx : x.size = d) :
    dotBySize row (vecScale s x) = s * dotBySize row x := by
  have hVec : (vecScale s x).size = d := by
    simpa [hx] using (vecScale_size s x)
  have hEq : row.size = (vecScale s x).size := by simpa [hRow, hVec]
  calc
    dotBySize row (vecScale s x)
        = (List.range d).foldl
            (fun acc i => acc + row[i]! * (vecScale s x)[i]!)
            0 := by
              simp [dotBySize, hEq, hVec]
    _ =
      (List.range d).foldl (fun acc i => acc + s * (row[i]! * x[i]!)) 0 := by
        exact list_foldl_congr_mem
          (fun acc i => acc + row[i]! * (vecScale s x)[i]!)
          (fun acc i => acc + s * (row[i]! * x[i]!))
          0
          (List.range d)
          (by
            intro acc i hiMem
            have hi : i < d := by simpa [List.mem_range] using hiMem
            have hVS :
                (vecScale s x)[i]! = s * x[i]! := by
              have hGetD :
                  (vecScale s x).getD i 0 = s * x.getD i 0 := by
                simpa [coeffAt, hi] using (coeffAt_vecScale_of_size_d s x hx i hi)
              have hL : (vecScale s x)[i]! = (vecScale s x).getD i 0 := by
                simpa using (Array.getElem!_eq_getD (xs := (vecScale s x)) (i := i))
              have hR : x[i]! = x.getD i 0 := by
                simpa using (Array.getElem!_eq_getD (xs := x) (i := i))
              calc
                (vecScale s x)[i]! = (vecScale s x).getD i 0 := hL
                _ = s * x.getD i 0 := hGetD
                _ = s * x[i]! := by simpa [hR]
            have hTerm :
                row[i]! * (vecScale s x)[i]! = s * (row[i]! * x[i]!) := by
              calc
                row[i]! * (vecScale s x)[i]!
                    = row[i]! * (s * x[i]!) := by simp [hVS]
                _ = (row[i]! * s) * x[i]! := by
                      simpa using (Lean.Grind.Fin.mul_assoc (n := Goldilocks.q) (row[i]!) s (x[i]!)).symm
                _ = (s * row[i]!) * x[i]! := by
                      simpa using congrArg (fun t => t * x[i]!)
                        (Lean.Grind.Fin.mul_comm (n := Goldilocks.q) (row[i]!) s)
                _ = s * (row[i]! * x[i]!) := by
                      simpa using (Lean.Grind.Fin.mul_assoc (n := Goldilocks.q) s (row[i]!) (x[i]!))
            simpa [hTerm] using (rfl : acc + row[i]! * (vecScale s x)[i]! = acc + row[i]! * (vecScale s x)[i]!))
    _ = s * (List.range d).foldl (fun acc i => acc + row[i]! * x[i]!) 0 := by
          have hFold := foldl_mul_left_distrib
            (l := List.range d)
            (t := fun i => row[i]! * x[i]!)
            (acc := (0 : F))
            (c := s)
          simpa using hFold.symm
    _ = s * dotBySize row x := by
          simp [dotBySize, hRow, hx]

/--
`dotBySize` additivity in the right input for arbitrary row shape.

If `row` is not `d`-sized, both sides reduce through the guard to `0`.
-/
theorem dotBySize_vecAdd_right_of_size_d_anyRow_ring
    (row x y : Array F)
    (hx : x.size = d)
    (hy : y.size = d) :
    dotBySize row (vecAdd x y) = dotBySize row x + dotBySize row y := by
  by_cases hRow : row.size = d
  · exact dotBySize_vecAdd_right_of_size_d_ring row x y hRow hx hy
  · have hxy : x.size = y.size := by simpa [hx, hy]
    have hVec : (vecAdd x y).size = d := by
      simpa [hx] using (vecAdd_size_of_eq hxy)
    have hNeL : row.size ≠ (vecAdd x y).size := by
      simpa [hVec] using hRow
    have hNeX : row.size ≠ x.size := by
      simpa [hx] using hRow
    have hNeY : row.size ≠ y.size := by
      simpa [hy] using hRow
    simp [dotBySize, hNeL, hNeX, hNeY]

/--
`dotBySize` homogeneity in the right input for arbitrary row shape.

If `row` is not `d`-sized, both sides reduce through the guard to `0`.
-/
theorem dotBySize_vecScale_right_of_size_d_anyRow_ring
    (row x : Array F)
    (s : F)
    (hx : x.size = d) :
    dotBySize row (vecScale s x) = s * dotBySize row x := by
  by_cases hRow : row.size = d
  · exact dotBySize_vecScale_right_of_size_d_ring row x s hRow hx
  · have hVec : (vecScale s x).size = d := by
      simpa [hx] using (vecScale_size s x)
    have hNeL : row.size ≠ (vecScale s x).size := by
      simpa [hVec] using hRow
    have hNeX : row.size ≠ x.size := by
      simpa [hx] using hRow
    have hs0 : s * (0 : F) = 0 := by
      simpa using (Fin.mul_zero (n := Goldilocks.q) s)
    simpa [dotBySize, hNeL, hNeX, hs0]

/-- Canonical-size output shape of `superneoBarBlock` when input has size `d`. -/
theorem superneoBarBlock_size_of_size_d_ring
    (bar : Array (Array F))
    (a : Array F)
    (ha : a.size = d) :
    (superneoBarBlock bar a).size = d := by
  by_cases hBar : bar.size = d
  · simp [superneoBarBlock, hBar, ha]
  · simp [superneoBarBlock, hBar, ha]

/-- `superneoBarBlock` is additive on canonical `d`-sized inputs. -/
theorem superneoBarBlock_vecAdd_of_size_d_ring
    (bar : Array (Array F))
    (x y : Array F)
    (hx : x.size = d)
    (hy : y.size = d) :
    superneoBarBlock bar (vecAdd x y) =
      vecAdd (superneoBarBlock bar x) (superneoBarBlock bar y) := by
  have hxy : x.size = y.size := by simpa [hx, hy]
  have hVec : (vecAdd x y).size = d := by
    simpa [hx] using (vecAdd_size_of_eq hxy)
  by_cases hBar : bar.size = d
  · have hOutEq :
      (superneoBarBlock bar x).size =
        (superneoBarBlock bar y).size := by
      simp [superneoBarBlock, hBar, hx, hy]
    have hRSize :
        (vecAdd (superneoBarBlock bar x) (superneoBarBlock bar y)).size = d := by
      calc
        (vecAdd (superneoBarBlock bar x) (superneoBarBlock bar y)).size
            = (superneoBarBlock bar x).size := vecAdd_size_of_eq hOutEq
        _ = d := by simp [superneoBarBlock, hBar, hx]
    have hLSize :
        (superneoBarBlock bar (vecAdd x y)).size = d := by
      simp [superneoBarBlock, hBar, hVec]
    apply Array.ext
    · exact hLSize.trans hRSize.symm
    · intro i hiL hiR
      have hi : i < d := by
        simpa [superneoBarBlock, hBar, hVec] using hiL
      have hDot :
          dotBySize (bar[i]'(by simpa [hBar] using hi)) (vecAdd x y) =
            dotBySize (bar[i]'(by simpa [hBar] using hi)) x +
              dotBySize (bar[i]'(by simpa [hBar] using hi)) y := by
        exact dotBySize_vecAdd_right_of_size_d_anyRow_ring
          (row := bar[i]'(by simpa [hBar] using hi))
          (x := x) (y := y) (hx := hx) (hy := hy)
      simpa [superneoBarBlock, hBar, hx, hy, hVec, vecAdd, hxy, hi] using hDot
  · simp [superneoBarBlock, hBar]

/-- `superneoBarBlock` is homogeneous on canonical `d`-sized inputs. -/
theorem superneoBarBlock_vecScale_of_size_d_ring
    (bar : Array (Array F))
    (x : Array F)
    (s : F)
    (hx : x.size = d) :
    superneoBarBlock bar (vecScale s x) =
      vecScale s (superneoBarBlock bar x) := by
  have hVec : (vecScale s x).size = d := by
    simpa [hx] using (vecScale_size s x)
  by_cases hBar : bar.size = d
  · apply Array.ext
    · simp [superneoBarBlock, hBar, hx, hVec, vecScale]
    · intro i hiL hiR
      have hi : i < d := by
        simpa [superneoBarBlock, hBar, hx, hVec, vecScale] using hiL
      have hDot :
          dotBySize (bar[i]'(by simpa [hBar] using hi)) (vecScale s x) =
            s * dotBySize (bar[i]'(by simpa [hBar] using hi)) x := by
        exact dotBySize_vecScale_right_of_size_d_anyRow_ring
          (row := bar[i]'(by simpa [hBar] using hi))
          (x := x)
          (s := s)
          (hx := hx)
      simpa [superneoBarBlock, hBar, hx, hVec, vecScale, hi] using hDot
  · simp [superneoBarBlock, hBar]

theorem zeroRq_size : zeroRq.size = d := by
  simp [zeroRq, d]

theorem oneRq_size : oneRq.size = d := by
  simp [oneRq, d]

theorem hasRingDegreeShape_zeroRq : hasRingDegreeShape zeroRq := by
  simp [hasRingDegreeShape, zeroRq, d]

theorem hasRingDegreeShape_oneRq : hasRingDegreeShape oneRq := by
  simp [hasRingDegreeShape, oneRq, d]

theorem hasRingDegreeShape_mulRq (a b : Coeffs) : hasRingDegreeShape (mulRq a b) := by
  simp [hasRingDegreeShape, mulRq_size]

theorem ct_zeroRq : ct zeroRq = 0 := by
  simp [ct, zeroRq, d]

theorem ct_oneRq : ct oneRq = 1 := by
  simp [ct, oneRq, d]

theorem coeffAt_zeroRq (i : Nat) : coeffAt zeroRq i = 0 := by
  unfold coeffAt zeroRq
  by_cases hi : i < d
  · simp [Array.getD, hi]
  · simp [Array.getD, hi]

theorem ringMulShapeProp_of_shapes
    {a b : Coeffs}
    (ha : hasRingDegreeShape a)
    (hb : hasRingDegreeShape b) :
    ringMulShapeProp a b := by
  exact And.intro ha hb

theorem ringMulShapeProp_left
    {a b : Coeffs}
    (h : ringMulShapeProp a b) :
    hasRingDegreeShape a := h.1

theorem ringMulShapeProp_right
    {a b : Coeffs}
    (h : ringMulShapeProp a b) :
    hasRingDegreeShape b := h.2

@[simp] theorem linComb2Vec_def
    (ρ1 ρ2 : F) (z1 z2 : Array F) :
    linComb2Vec ρ1 ρ2 z1 z2 = vecAdd (vecScale ρ1 z1) (vecScale ρ2 z2) := rfl

def allCanonical (a : Coeffs) : Bool :=
  a.all F.canonicalCheck

end SuperNeo
