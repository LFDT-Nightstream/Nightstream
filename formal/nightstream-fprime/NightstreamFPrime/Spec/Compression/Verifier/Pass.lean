import NightstreamFPrime.Spec.Compression.Verifier.Game
import NightstreamFPrime.Spec.Compression.Verifier.Matrices

/-! Rust `weights::matrix_part`: the verifier's one pass over the rows of each relation
(`matrixPass`), and `MatrixSpec` for it (`matrixPass_spec`).

The verifier holds each relation's rows in factored form `M_m = L_m · G` (`Factored`, Rust
`FactoredRows`). `matrixPass` follows Rust step by step: it puts the claims in groups with one
relation and one `β` (`groups`), and for each group it computes the slot values from windows
(`slotValue`, Rust `slot_values`), the row weights from a low and a high table (`rowWeight`, Rust
`RowWeights`), and one weighted row for each row (`weightedRow`, Rust `weighted_row`). The model
names a relation by an index (`which`), where Rust compares the address of the rows. It reads the
tables of Rust `equality` as `eqAt`, and it does not model the parallel sums of Rust. -/

namespace NightstreamFPrime.Spec.Compression.Verifier

open Finset NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Algebra

/-- Matrix rows in factored form `M_m = L_m · G` (Rust `FactoredRows`): slot `s` holds the cells
`start s` to `start s + width s − 1`, and `G` recomposes it with the powers of `ratio`. Row `row`
of `L_m` is the list `entries m row` of its pairs (slot, coefficient). -/
structure Factored (F : Type) where
  rows : ℕ
  columns : ℕ
  entries : ℕ → ℕ → List (ℕ × F)
  start : ℕ → ℕ
  width : ℕ → ℕ
  ratio : F

namespace Factored

variable {F : Type} [Field F]

/-- Entry `(slot, coefficient)` at cell `x`: one geometric run (Rust `visit_rows`). -/
def run (R : Factored F) (e : ℕ × F) (x : ℕ) : F :=
  if R.start e.1 ≤ x ∧ x < R.start e.1 + R.width e.1 then e.2 * R.ratio ^ (x - R.start e.1) else 0

/-- The dense matrices `M_m = L_m · G`. -/
def matrices (R : Factored F) : Matrices F where
  rows := R.rows
  columns := R.columns
  entry m row x := ((R.entries m row).map fun e => R.run e x).sum

end Factored

section Groups

variable {K : Type} [CommRing K] {n : ℕ} {A B : Type} [DecidableEq A] (key : Fin n → A)

/-- Rust `matrix_part`: claim `c` joins the first group whose first claim has its key, or it
starts a new group. -/
def joinGroup (c : Fin n) (item : B) : List (Fin n × List B) → List (Fin n × List B)
  | [] => [(c, [item])]
  | g :: rest =>
      if key g.1 = key c then (g.1, g.2 ++ [item]) :: rest else g :: joinGroup c item rest

/-- The groups of the claims, in the order of Rust `matrix_part`. -/
def groups (item : Fin n → B) : List (Fin n × List B) :=
  (List.finRange n).foldl (fun acc c => joinGroup key c (item c) acc) []

variable (value : Fin n → B → K)

/-- The sum over the groups of the values of their items, each read with its group's first claim. -/
def groupSum (gs : List (Fin n × List B)) : K := (gs.map fun g => (g.2.map (value g.1)).sum).sum

variable {key value}

theorem groupSum_join (same : ∀ c c', key c = key c' → value c = value c') (c : Fin n)
    (item : B) : ∀ gs, groupSum value (joinGroup key c item gs) = groupSum value gs + value c item
  | [] => by simp [joinGroup, groupSum]
  | g :: rest => by
      simp only [joinGroup]
      split_ifs with equal
      · simp only [groupSum, List.map_cons, List.sum_cons, List.map_append, List.sum_append,
          List.map_nil, List.sum_nil, same _ _ equal]
        ring
      · have := groupSum_join same c item rest
        simp only [groupSum, List.map_cons, List.sum_cons] at this ⊢
        rw [this]
        ring

theorem groupSum_foldl (same : ∀ c c', key c = key c' → value c = value c') (item : Fin n → B) :
    ∀ (l : List (Fin n)) (gs : List (Fin n × List B)),
      groupSum value (l.foldl (fun acc c => joinGroup key c (item c) acc) gs) =
        groupSum value gs + (l.map fun c => value c (item c)).sum
  | [], gs => by simp
  | c :: l, gs => by
      rw [List.foldl_cons, groupSum_foldl same item l, groupSum_join same, List.map_cons,
        List.sum_cons]
      ring

/-- The groups give each claim once. -/
theorem groupSum_groups (same : ∀ c c', key c = key c' → value c = value c')
    (item : Fin n → B) : groupSum value (groups key item) = ∑ c, value c (item c) := by
  rw [groups, groupSum_foldl same, Fin.sum_univ_def]
  simp [groupSum]

end Groups

section Lemmas

variable {K : Type} [CommRing K]

/-- `eq(r, i)` is the product of `eq` of the first `k` coordinates at the low `k` bits of `i` and
`eq` of the others at the high bits. -/
theorem eqAt_split : ∀ (k : ℕ) (l : List K) (i : ℕ),
    eqAt (l.take k) (i % 2 ^ k) * eqAt (l.drop k) (i / 2 ^ k) = eqAt l i
  | 0, l, i => by simp [eqAt]
  | k + 1, [], i => by simp [eqAt]
  | k + 1, a :: rest, i => by
      have low : i % 2 ^ (k + 1) % 2 = i % 2 :=
        Nat.mod_mod_of_dvd i (dvd_pow_self 2 (Nat.succ_ne_zero k))
      have half : i % 2 ^ (k + 1) / 2 = i / 2 % 2 ^ k := by
        rw [pow_succ', Nat.mod_mul_right_div_self]
      have high : i / 2 ^ (k + 1) = i / 2 / 2 ^ k := by
        rw [pow_succ', Nat.div_div_eq_div_mul]
      simp only [List.take_succ_cons, List.drop_succ_cons, eqAt, low, half, high]
      rw [← eqAt_split k rest (i / 2)]
      ring

/-- The conjugate is its own adjoint: `Σ_h u(h) · conjugate(w)(h) = Σ_h conjugate(u)(h) · w(h)`. -/
theorem sum_mul_conjugate (D : ℕ) (u w : ℕ → K) :
    ∑ h ∈ range D, u h * conjugate D w h = ∑ h ∈ range D, conjugate D u h * w h := by
  cases D with
  | zero => simp
  | succ d =>
    rw [sum_range_succ', sum_range_succ']
    simp only [conjugate, Nat.add_one_ne_zero, ↓reduceIte]
    congr 1
    rw [← sum_range_reflect]
    refine sum_congr rfl fun h member => ?_
    have small := mem_range.mp member
    rw [show d - 1 - h + 1 = d - h by omega, show d + 1 - (d - h) = h + 1 by omega,
      show d + 1 - (h + 1) = d - h by omega]
    ring

theorem sum_list_mul {B : Type} (items : List B) (f : B → ℕ → K) (g : ℕ → K) (N : ℕ) :
    ∑ row ∈ range N, (items.map fun y => f y row).sum * g row =
      (items.map fun y => ∑ row ∈ range N, f y row * g row).sum := by
  induction items with
  | nil => simp
  | cons y ys ih => simp only [List.map_cons, List.sum_cons, add_mul, sum_add_distrib, ih]

end Lemmas

variable {F K : Type} [Field F] [Field K] [Algebra F K]

local notation "ι" => algebraMap F K

section Model

/-- Rust `slot_values`, table `windows[w][t][h]`: the sum over `k < w` with `⌊(h + k)/D⌋ = t` of
`form[(h + k) mod D] · ratio^k`, built one `k` at a time. -/
def window (D : ℕ) (ratio : F) (form : ℕ → K) : ℕ → ℕ → ℕ → K
  | 0, _, _ => 0
  | k + 1, t, h => window D ratio form k t h +
      if (h + k) / D = t then form ((h + k) % D) * ι (ratio ^ k) else 0

/-- Rust `slot_values`: `Γ = Σ_t eq(β, b + t) · windows[width][t][h]` for the slot of the cells
`start` to `start + width − 1`, with `(b, h) = divmod(start, D)`. A block from `blocks` has no
weight, because Rust zips with `blocks[b..]`. -/
def slotValue (D blocks : ℕ) (β : List K) (ratio : F) (form : ℕ → K) (start width : ℕ) : K :=
  ∑ t ∈ range ((start % D + width + D - 1) / D),
    (if start / D + t < blocks then eqAt β (start / D + t) else 0) *
      window D ratio form width t (start % D)

/-- Rust `RowWeights::at`: `Σ c · eq(r, row)` over the claims `(c, r)`. Each `eq(r, ·)` is a table
of the low half of `r` times a table of the high half, and `c` scales the high table. -/
def rowWeight (claims : List (K × List K)) (row : ℕ) : K :=
  (claims.map fun y =>
    eqAt (y.2.take (y.2.length / 2)) (row % 2 ^ (y.2.length / 2)) *
      (y.1 * eqAt (y.2.drop (y.2.length / 2)) (row / 2 ^ (y.2.length / 2)))).sum

/-- Rust `FactoredRows::weighted_row`: `Σ_m ω_m · Σ_entries coefficient · Γ(slot)`. -/
def weightedRow (R : Factored F) (row t : ℕ) (omega : ℕ → K) (slots : ℕ → K) : K :=
  ∑ m ∈ range t, omega m * ((R.entries m row).map fun e => ι e.2 * slots e.1).sum

/-- The weight of cell `x = D·j + h` in `slotValue`: `eq(β, j) · form(h)` for a block `j` below
`blocks`, and 0 for another cell. -/
def blockForm (D blocks : ℕ) (β : List K) (form : ℕ → K) (x : ℕ) : K :=
  if x / D < blocks then eqAt β (x / D) * form (x % D) else 0

variable [DecidableEq K] (P : Params F K) (rel : ℕ → Factored F) (which : Fin P.claims → ℕ)
  (ch : Challenges P) (ρ : List K)

/-- Rust `slot_values` for the relation and the `β` of claim `c`. -/
def slots (c : Fin P.claims) (s : ℕ) : K :=
  slotValue P.lanes (P.shape c).blocks (uses P ch ρ c).point (rel (which c)).ratio
    (conjugate P.lanes (laneForm P ch ρ)) ((rel (which c)).start s) ((rel (which c)).width s)

/-- Rust `weighted_row` of the relation of claim `c`, with the `Eval_A` powers `ω^(κ+1+m)`. -/
def rowValue (c : Fin P.claims) (row : ℕ) : K :=
  weightedRow (rel (which c)) row P.matrices (fun m => ch.omega ^ (P.rows + 1 + m))
    (slots P rel which ch ρ c)

/-- Rust `matrix_part` for the group of first claim `c` and the claims `(coefficient, r)`. -/
def groupValue (c : Fin P.claims) (claims : List (K × List K)) : K :=
  ∑ row ∈ range (rel (which c)).rows, rowWeight claims row * rowValue P rel which ch ρ c row

/-- Rust `matrix_part`: the groups of one relation and one `β`, in the order of the claims. -/
def matrixPass (st : Statement P) (ch : Challenges P) (ρ : List K) : K :=
  ((groups (fun c => (which c, (uses P ch ρ c).point))
      (fun c => ((uses P ch ρ c).coefficient, (st c).point))).map
    fun g => groupValue P rel which ch ρ g.1 g.2).sum

/-- The part of claim `(coefficient, r)` of `matrixPass`, read with the relation and the `β` of
claim `c`. -/
def claimPass (c : Fin P.claims) (y : K × List K) : K :=
  y.1 * ∑ row ∈ range (rel (which c)).rows, eqAt y.2 row * rowValue P rel which ch ρ c row

end Model

section Proofs

theorem window_eq (D : ℕ) (ratio : F) (form : ℕ → K) (w t h : ℕ) :
    window D ratio form w t h =
      ∑ k ∈ range w, if (h + k) / D = t then form ((h + k) % D) * ι (ratio ^ k) else 0 := by
  induction w with
  | zero => simp [window]
  | succ k ih => rw [window, ih, sum_range_succ]

/-- The windows give `Γ = Σ_{k < width} ratio^k · Φ(start + k)`. -/
theorem slotValue_eq {D : ℕ} (pos : 0 < D) (blocks : ℕ) (β : List K) (ratio : F) (form : ℕ → K)
    (start width : ℕ) :
    slotValue D blocks β ratio form start width =
      ∑ k ∈ range width, ι (ratio ^ k) * blockForm D blocks β form (start + k) := by
  unfold slotValue
  simp only [window_eq, mul_sum]
  rw [sum_comm]
  refine sum_congr rfl fun k member => ?_
  have small := mem_range.mp member
  have inside : (start % D + k) / D < (start % D + width + D - 1) / D := by
    have le := Nat.div_le_div_right (c := D)
      (show start % D + k + D ≤ start % D + width + D - 1 by omega)
    rw [Nat.add_div_right _ pos] at le
    omega
  simp only [mul_ite, mul_zero, sum_ite_eq, mem_range, inside, ↓reduceIte]
  have split : start + k = D * (start / D) + (start % D + k) := by
    have := Nat.div_add_mod start D
    omega
  have div : (start + k) / D = start / D + (start % D + k) / D := by
    rw [split, Nat.mul_add_div pos]
  have mod : (start + k) % D = (start % D + k) % D := by
    rw [split, Nat.mul_add_mod]
  unfold blockForm
  rw [div, mod]
  split_ifs <;> ring

theorem blockForm_zero {D blocks : ℕ} (pos : 0 < D) (β : List K) (form : ℕ → K) {x : ℕ}
    (large : D * blocks ≤ x) : blockForm D blocks β form x = 0 := by
  have above : blocks ≤ x / D := (Nat.le_div_iff_mul_le pos).mpr (by rw [Nat.mul_comm]; exact large)
  unfold blockForm
  split_ifs with below
  · omega
  · rfl

/-- The cells of one entry. -/
theorem sum_run (R : Factored F) (e : ℕ × F) (N : ℕ) (Φ : ℕ → K)
    (zero : ∀ x, N ≤ x → Φ x = 0) :
    ∑ x ∈ range N, ι (R.run e x) * Φ x =
      ι e.2 * ∑ k ∈ range (R.width e.1), ι (R.ratio ^ k) * Φ (R.start e.1 + k) := by
  have term : ∀ x, ι (R.run e x) * Φ x = if R.start e.1 ≤ x ∧ x < R.start e.1 + R.width e.1 then
      ι e.2 * (ι (R.ratio ^ (x - R.start e.1)) * Φ x) else 0 := fun x => by
    unfold Factored.run
    split_ifs <;> simp [map_mul, mul_assoc]
  have large : ∑ x ∈ range N, ι (R.run e x) * Φ x =
      ∑ x ∈ range (N + R.start e.1 + R.width e.1), ι (R.run e x) * Φ x :=
    sum_subset (range_subset_range.mpr (by omega)) fun x _ outside => by
      rw [zero x (by simpa using outside), mul_zero]
  have cells : (range (N + R.start e.1 + R.width e.1)).filter
      (fun x => R.start e.1 ≤ x ∧ x < R.start e.1 + R.width e.1) =
        Ico (R.start e.1) (R.start e.1 + R.width e.1) := by
    ext x
    simp only [mem_filter, mem_range, mem_Ico]
    omega
  rw [large, sum_congr rfl fun x _ => term x, ← sum_filter, cells, sum_Ico_eq_sum_range,
    Nat.add_sub_cancel_left, mul_sum]
  exact sum_congr rfl fun k _ => by rw [Nat.add_sub_cancel_left]

/-- The dense entries of a factored relation, entry by entry. -/
theorem sum_entry (R : Factored F) (m row N : ℕ) (Φ : ℕ → K) :
    ∑ x ∈ range N, ι (R.matrices.entry m row x) * Φ x =
      ((R.entries m row).map fun e => ∑ x ∈ range N, ι (R.run e x) * Φ x).sum := by
  simp only [Factored.matrices]
  generalize R.entries m row = l
  induction l with
  | nil => simp
  | cons e es ih =>
    simp only [List.map_cons, List.sum_cons, map_add, add_mul, sum_add_distrib, ih]

/-- The matrix weights of `Matrices.form` on the blocks of a claim are the rows of `M_m`, read
against the cell weights `eq(β, j) · conjugate(u)(h)`. -/
theorem sum_form (M : Matrices F) {D : ℕ} (pos : 0 < D) (n : ℕ) (β r : List K) (u : ℕ → K)
    (m : ℕ) :
    ∑ j ∈ range n, eqAt β j * ∑ h ∈ range D, u h * M.form D r m j h =
      ∑ row ∈ range M.rows, eqAt r row *
        ∑ x ∈ range (D * n), ι (M.entry m row x) * blockForm D n β (conjugate D u) x := by
  have cell : ∀ j h, h < D → j < n →
      blockForm D n β (conjugate D u) (D * j + h) = eqAt β j * conjugate D u h := fun j h lane _ => by
    have div : (D * j + h) / D = j := by
      rw [Nat.mul_add_div pos, Nat.div_eq_of_lt lane, add_zero]
    have mod : (D * j + h) % D = h := by
      rw [Nat.mul_add_mod, Nat.mod_eq_of_lt lane]
    have below : j < n := by omega
    simp only [blockForm, div, mod, below, ↓reduceIte]
  calc ∑ j ∈ range n, eqAt β j * ∑ h ∈ range D, u h * M.form D r m j h
      = ∑ j ∈ range n, ∑ h ∈ range D, ∑ row ∈ range M.rows,
          eqAt r row * (ι (M.entry m row (D * j + h)) * (eqAt β j * conjugate D u h)) := by
        refine sum_congr rfl fun j _ => ?_
        simp only [Matrices.form]
        rw [sum_mul_conjugate, mul_sum]
        refine sum_congr rfl fun h _ => ?_
        simp only [mul_sum]
        exact sum_congr rfl fun row _ => by ring
    _ = ∑ row ∈ range M.rows, ∑ j ∈ range n, ∑ h ∈ range D,
          eqAt r row * (ι (M.entry m row (D * j + h)) * (eqAt β j * conjugate D u h)) :=
        (sum_congr rfl fun j _ => sum_comm).trans sum_comm
    _ = _ := by
        refine sum_congr rfl fun row _ => ?_
        rw [sum_range_mul, mul_sum]
        refine sum_congr rfl fun j member => ?_
        rw [mul_sum]
        refine sum_congr rfl fun h lane => ?_
        rw [cell j h (mem_range.mp lane) (mem_range.mp member)]

theorem rowWeight_eq (claims : List (K × List K)) (row : ℕ) :
    rowWeight claims row = (claims.map fun y => y.1 * eqAt y.2 row).sum := by
  unfold rowWeight
  congr 1
  refine List.map_congr_left fun y _ => ?_
  rw [← eqAt_split (y.2.length / 2) y.2 row]
  ring

variable (P : Params F K) (rel : ℕ → Factored F) (which : Fin P.claims → ℕ)
  (ch : Challenges P) (ρ : List K)

theorem groupValue_eq (c : Fin P.claims) (claims : List (K × List K)) :
    groupValue P rel which ch ρ c claims = (claims.map (claimPass P rel which ch ρ c)).sum := by
  unfold groupValue
  simp only [rowWeight_eq]
  rw [sum_list_mul claims (fun y row => y.1 * eqAt y.2 row)]
  congr 1
  refine List.map_congr_left fun y _ => ?_
  rw [claimPass, mul_sum]
  exact sum_congr rfl fun row _ => by ring

/-- The part of one claim of `matrixPass` is its part of `matrixTrue`. -/
theorem claimPass_eq (st : Statement P) (c : Fin P.claims)
    (form : P.matrixForm c = (rel (which c)).matrices.form P.lanes) :
    claimPass P rel which ch ρ c ((uses P ch ρ c).coefficient, (st c).point) =
      (uses P ch ρ c).coefficient * ∑ m ∈ range P.matrices, ch.omega ^ (P.rows + 1 + m) *
        ∑ j ∈ range (P.shape c).blocks, eqAt (uses P ch ρ c).point j *
          ∑ h ∈ range P.lanes, laneForm P ch ρ h * P.matrixForm c (st c).point m j h := by
  have pos : 0 < P.lanes := pow_pos two_pos _
  have entries : ∀ m row, ∑ x ∈ range (P.lanes * (P.shape c).blocks),
      ι ((rel (which c)).matrices.entry m row x) * blockForm P.lanes (P.shape c).blocks
        (uses P ch ρ c).point (conjugate P.lanes (laneForm P ch ρ)) x =
      (((rel (which c)).entries m row).map fun e => ι e.2 * slots P rel which ch ρ c e.1).sum := by
    intro m row
    rw [sum_entry]
    congr 1
    refine List.map_congr_left fun e _ => ?_
    rw [sum_run _ _ _ _ fun x large => blockForm_zero pos _ _ large, slots, slotValue_eq pos]
  simp only [claimPass]
  rw [form]
  congr 1
  simp only [sum_form _ pos]
  simp only [entries, rowValue, weightedRow, mul_sum]
  rw [sum_comm]
  exact sum_congr rfl fun m _ => sum_congr rfl fun row _ => by ring

/-- `MatrixSpec` for the model of Rust `matrix_part`: when the matrix weights of each claim are
those of its relation's factored rows, and claims of one relation have the same blocks, the one
pass computes the matrix part of the honest weights. -/
theorem matrixPass_spec [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K]
    (form : ∀ c, P.matrixForm c = (rel (which c)).matrices.form P.lanes)
    (blocks : ∀ c c', which c = which c' → (P.shape c).blocks = (P.shape c').blocks)
    (columnVars zeros : ℕ) :
    MatrixSpec (⟨matrixPass P rel which, columnVars, zeros⟩ : VerifierData P) := by
  intro st ch ρ _
  have same : ∀ c c', (which c, (uses P ch ρ c).point) = (which c', (uses P ch ρ c').point) →
      claimPass P rel which ch ρ c = claimPass P rel which ch ρ c' := fun c c' equal => by
    obtain ⟨relation, point⟩ := Prod.mk.inj equal
    have slot_eq : slots P rel which ch ρ c = slots P rel which ch ρ c' := by
      funext s
      simp only [slots, relation, point, blocks c c' relation]
    funext y
    simp only [claimPass, rowValue, relation, slot_eq]
  show matrixPass P rel which st ch ρ = matrixTrue P st ch ρ
  unfold matrixPass
  simp only [groupValue_eq]
  change groupSum (claimPass P rel which ch ρ) (groups _ _) = _
  rw [groupSum_groups same, matrixTrue]
  exact sum_congr rfl fun c _ => claimPass_eq P rel which ch ρ st c (form c)

end Proofs

end NightstreamFPrime.Spec.Compression.Verifier
