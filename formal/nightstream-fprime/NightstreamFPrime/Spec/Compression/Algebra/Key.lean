import NightstreamFPrime.Spec.Compression.Algebra.Point

/-! The committed Ajtai key (Rust `key.rs`, ePrint 2026/1983 section 9). The key polynomial holds,
lowest variable first, the lane, the block and the row of a key coefficient: cell
`h + D·j + D·2^b·i` holds lane `h` of key element `(i, j)` for each row `i < κ`, and each cell of a
row from `κ` is zero (`keyTable`).

The Ajtai rows of the claims add to the public weights the value `keyPart`. It is the sum over the
cube of the weights `T(h, j) = u(h) · Σ_claims c · eq(β, j) · [j < blocks]` (`weightTable`, Rust
`weight_table`) times `F = Σ_{i < κ} ω^i · A(i, ·, ·)` (`folded`, Rust `fold`): `key_sum`. So the
key sum-check proves the exact value that the first sum-check needs, and no new batch is needed.
The sum-check ends at a point `(λ, β′)`, where `T` is Rust `weight_at` (`weightAt_spec`) and `F` is
`scale` times one value of the key polynomial at the row point (`folded_at`). -/

namespace NightstreamFPrime.Spec.Compression.Algebra

open Finset NightstreamFPrime.Spec.Compression.Lib

variable {K : Type} [Field K]

/-- The witness of one claim in the weights at `ρ` (Rust `weights::Use`): the coefficient `c`, the
point `β` of the blocks and the number of blocks. -/
structure Use (K : Type) where
  coefficient : K
  point : List K
  blocks : ℕ

/-- `Σ_claims c · eq(β, j) · [j < blocks]`. -/
def blockWeight {s : ℕ} (uses : Fin s → Use K) (j : ℕ) : K :=
  ∑ c, if j < (uses c).blocks then (uses c).coefficient * eqAt (uses c).point j else 0

/-- Rust `key::weight_table`: cell `D·j + h` has `u(h) · blockWeight j`. -/
def weightTable {s : ℕ} (D : ℕ) (u : ℕ → K) (uses : Fin s → Use K) : ℕ → K :=
  fun x => u (x % D) * blockWeight uses (x / D)

/-- Rust `key::fold`: lane `h` of block `j` of `Σ_{i < κ} ω^i · A(i, j)` at `D·j + h`. -/
def folded (D κ : ℕ) (A : ℕ → ℕ → ℕ → K) (ω : K) : ℕ → K :=
  fun x => ∑ i ∈ range κ, ω ^ i * A i (x / D) (x % D)

/-- The key polynomial of `2^b` blocks: cell `h + D·j + D·2^b·i`. -/
def keyTable (D b κ : ℕ) (A : ℕ → ℕ → ℕ → K) : ℕ → K :=
  fun x => if x / (D * 2 ^ b) < κ then A (x / (D * 2 ^ b)) (x / D % 2 ^ b) (x % D) else 0

/-- Rust `weights::key_part`: `Σ_claims c · Σ_{j < blocks} eq(β, j) · Σ_h u(h) · F(D·j + h)`. -/
def keyPart {s : ℕ} (D : ℕ) (u : ℕ → K) (uses : Fin s → Use K) (F : ℕ → K) : K :=
  ∑ c, (uses c).coefficient *
    ∑ j ∈ range (uses c).blocks, eqAt (uses c).point j * ∑ h ∈ range D, u h * F (D * j + h)

/-- Rust `key::weight_at`: `(Σ_h eq(λ, h) · u(h)) · Σ_claims c · prefixEquality(β, blocks, β′)` for
the point `(λ, β′)` with `d` lane coordinates. -/
def weightAt {s : ℕ} (D d : ℕ) (u : ℕ → K) (uses : Fin s → Use K) (point : List K) : K :=
  (∑ h ∈ range D, eqAt (point.take d) h * u h) *
    ∑ c, (uses c).coefficient * prefixEquality (uses c).point (uses c).blocks (point.drop d)

theorem sum_range_mul {M : Type} [AddCommMonoid M] (D : ℕ) (f : ℕ → M) :
    ∀ n : ℕ, ∑ x ∈ range (D * n), f x = ∑ j ∈ range n, ∑ h ∈ range D, f (D * j + h) := by
  intro n
  induction n with
  | zero => simp
  | succ n ih => rw [mul_add, mul_one, sum_range_add, ih, sum_range_succ]

/-- A sum over a cube of `2^b` indices of a term that is zero from `blocks` is a sum below
`blocks`. -/
theorem sum_range_ite {M : Type} [AddCommMonoid M] (blocks n : ℕ) (fits : blocks ≤ n)
    (g : ℕ → M) : ∑ j ∈ range n, (if j < blocks then g j else 0) = ∑ j ∈ range blocks, g j := by
  rw [← sum_range_add_sum_Ico _ fits, sum_eq_zero (s := Ico blocks n) fun j member => by
    rw [ite_eq_right (by have := (mem_Ico.mp member).1; omega)], add_zero]
  exact sum_congr rfl fun j member => ite_eq_left (mem_range.mp member)

/-- The key identity: the sum of the weights times `F` over the cube of `2^b` blocks is the part of
the public weights that the Ajtai key gives. -/
theorem key_sum {s : ℕ} (D b : ℕ) (u : ℕ → K) (uses : Fin s → Use K) (F : ℕ → K)
    (fits : ∀ c, (uses c).blocks ≤ 2 ^ b) (lanes : 0 < D) :
    ∑ x ∈ range (D * 2 ^ b), weightTable D u uses x * F x = keyPart D u uses F := by
  have cell (j h : ℕ) (lane : h < D) : weightTable D u uses (D * j + h) * F (D * j + h) =
      ∑ c, if j < (uses c).blocks then
        (uses c).coefficient * (eqAt (uses c).point j * (u h * F (D * j + h))) else 0 := by
    have low : (D * j + h) % D = h := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt lane]
    have high : (D * j + h) / D = j := by
      rw [Nat.add_comm, Nat.add_mul_div_left _ _ lanes, Nat.div_eq_of_lt lane, zero_add]
    simp only [weightTable, blockWeight, low, high, mul_sum, sum_mul]
    refine sum_congr rfl fun c _ => ?_
    split <;> ring
  rw [sum_range_mul, sum_congr rfl fun j _ => sum_congr rfl fun h member =>
    cell j h (mem_range.mp member)]
  rw [sum_congr rfl fun j _ => sum_comm, sum_comm, keyPart]
  refine sum_congr rfl fun c _ => ?_
  have inner (j : ℕ) : (∑ h ∈ range D, if j < (uses c).blocks then
      (uses c).coefficient * (eqAt (uses c).point j * (u h * F (D * j + h))) else 0) =
      if j < (uses c).blocks then ∑ h ∈ range D,
        (uses c).coefficient * (eqAt (uses c).point j * (u h * F (D * j + h))) else 0 := by
    split <;> simp
  rw [sum_congr rfl fun j _ => inner j, sum_range_ite _ _ (fits c), mul_sum]
  refine sum_congr rfl fun j _ => ?_
  rw [mul_sum, mul_sum]

/-- Rust `weight_at` is the multilinear extension of `weightTable` at the point `(λ, β′)`. -/
theorem weightAt_spec {s : ℕ} (d b : ℕ) (u : ℕ → K) (uses : Fin s → Use K) (point : List K)
    (length : point.length = d + b)
    (fits : ∀ c, (uses c).point.length ≤ b ∧ (uses c).blocks ≤ 2 ^ (uses c).point.length) :
    mle point (weightTable (2 ^ d) u uses) = weightAt (2 ^ d) d u uses point := by
  have split : point = point.take d ++ point.drop d := (List.take_append_drop _ _).symm
  have lane_length : (point.take d).length = d := by simp [length]
  have block_length : (point.drop d).length = b := by simp [length]
  have lanes : 0 < 2 ^ d := pow_pos two_pos d
  set L := ∑ h ∈ range (2 ^ d), eqAt (point.take d) h * u h with L_def
  have inner (j : ℕ) : mle (point.take d) (fun y => weightTable (2 ^ d) u uses (y + 2 ^ d * j)) =
      blockWeight uses j * L := by
    rw [mle_congr (point.take d) (u := fun y => blockWeight uses j * u y) fun y small => by
      rw [lane_length] at small
      simp only [weightTable]
      rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt small, Nat.add_mul_div_left _ _ lanes,
        Nat.div_eq_of_lt small, zero_add, mul_comm],
      mle_mul_left, mle_eq_cubeSum, cubeSum_eq_sum_range, lane_length]
  rw [weightAt, ← L_def]
  conv_lhs => rw [split, mle_append_split, lane_length]
  simp only [inner]
  rw [mle_eq_cubeSum, cubeSum_eq_sum_range, block_length, mul_sum]
  simp only [blockWeight, mul_sum, sum_mul]
  rw [sum_comm]
  refine sum_congr rfl fun c _ => ?_
  obtain ⟨short, small⟩ := fits c
  rw [prefixEquality_spec (uses c).point (point.drop d) (uses c).blocks
    (by rw [block_length]; exact short), min_eq_left small, mul_sum, mul_sum]
  rw [← sum_range_ite (uses c).blocks (2 ^ b)
    (small.trans (Nat.pow_le_pow_right two_pos short))]
  refine sum_congr rfl fun j _ => ?_
  split <;> ring

/-- `F` at a point is `scale` times the key polynomial at the point and the row point. -/
theorem folded_at (D b κ r : ℕ) (A : ℕ → ℕ → ℕ → K) (ω : K) (point : List K)
    (length : 2 ^ point.length = D * 2 ^ b) (rows : κ ≤ 2 ^ r)
    (regular : ∀ k < r, 1 + ω ^ 2 ^ k ≠ 0) :
    mle point (folded D κ A ω) =
      rowScale r ω * mle (point ++ rowPoint r ω) (keyTable D b κ A) := by
  have positive : 0 < D * 2 ^ b := by rw [← length]; exact pow_pos two_pos _
  have lanes : 0 < D := Nat.pos_of_mul_pos_right positive
  have row (i : ℕ) : mle point (fun x => keyTable D b κ A (x + 2 ^ point.length * i)) =
      if i < κ then mle point (fun x => A i (x / D) (x % D)) else 0 := by
    rw [length]
    split
    · rename_i inside
      refine mle_congr point fun x small => ?_
      rw [length] at small
      have block : x / D < 2 ^ b := Nat.div_lt_of_lt_mul small
      have row_index : (x + D * 2 ^ b * i) / (D * 2 ^ b) = i := by
        rw [Nat.add_mul_div_left _ _ positive, Nat.div_eq_of_lt small, zero_add]
      have block_index : (x + D * 2 ^ b * i) / D % 2 ^ b = x / D := by
        rw [show x + D * 2 ^ b * i = x + D * (2 ^ b * i) by ring, Nat.add_mul_div_left _ _ lanes,
          Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt block]
      have lane_index : (x + D * 2 ^ b * i) % D = x % D := by
        rw [show x + D * 2 ^ b * i = x + D * (2 ^ b * i) by ring, Nat.add_mul_mod_self_left]
      simp only [keyTable]
      rw [row_index, ite_eq_left inside, block_index, lane_index]
    · rename_i outside
      rw [← mle_const point 0]
      refine mle_congr point fun x small => ?_
      rw [length] at small
      simp only [keyTable]
      rw [Nat.add_mul_div_left _ _ positive, Nat.div_eq_of_lt small, zero_add, ite_eq_right outside]
  rw [mle_append_split]
  simp only [row]
  conv_rhs => rw [mle_eq_cubeSum, cubeSum_eq_sum_range, rowPoint_length, mul_sum]
  have powers : ∀ i ∈ range (2 ^ r), rowScale r ω * (eqAt (rowPoint r ω) i *
      (if i < κ then mle point (fun x => A i (x / D) (x % D)) else 0)) =
      if i < κ then ω ^ i * mle point (fun x => A i (x / D) (x % D)) else 0 := fun i member => by
    rw [← mul_assoc, ← rowPoint_spec r ω regular i (mem_range.mp member)]
    split <;> ring
  rw [sum_congr rfl powers, ← sum_range_add_sum_Ico _ rows,
    sum_eq_zero (s := Ico κ (2 ^ r)) fun i member => by
      rw [ite_eq_right (by have := (mem_Ico.mp member).1; omega)], add_zero]
  rw [sum_congr rfl fun i member => ite_eq_left (mem_range.mp member)]
  rw [show folded D κ A ω = fun x => ∑ i ∈ range κ, ω ^ i * A i (x / D) (x % D) from rfl,
    mle_finset_sum]
  exact sum_congr rfl fun i _ => mle_mul_left point _ _

end NightstreamFPrime.Spec.Compression.Algebra
