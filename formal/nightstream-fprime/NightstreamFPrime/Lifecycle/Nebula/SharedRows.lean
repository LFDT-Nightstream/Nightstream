import Mathlib.Algebra.Field.Basic
import Mathlib.Order.Basic

/-! Owns shared rows (SPEC.md section 7.6, "Shared columns"): `g` consecutive rows of one step
that share one block of columns, so that a gadget which at most one of them uses runs once. Rust
`nebula::R1csApplication` keeps each row of the one-row relation for each row of the group, with
the shared columns read from the group's block, and keeps a row that reads only shared columns
and the constant-one column once for the group. The WASM host-event absorb is the instance: the
block is the 12 permutation input lanes `W`, the permutation words and the 4 output lanes `O`; the
block rows `P` are the width-12 Poseidon2 permutation with its feed-forward; and each row `z` has
the gated rows `flag z · (W l − input z l) = 0` and `flag z · (after z l − O l) = 0`, with
`flag = ABSORB`, `input` the previous chain, the seven buffered words and the staged word, and
`after = COL_COMM_CHAIN_AFTER`. Before the group, each row held its own permutation
(`SharedAbsorb.OwnRows`).

Soundness (`SharedAbsorb.groupRows_sound`) needs no bound on the absorbing rows of a group: the
gated input rows make each absorbing row's input equal to `W`. Completeness
(`SharedAbsorb.groupRows_complete`) needs that the absorbing rows of a group have one input. That
holds when the slot cursor puts two absorbs at least eight rows apart
(`SharedAbsorb.absorbs_apart`, `SharedAbsorb.group_complete`). It does not own the width-12
permutation: `f` and its rows `P` are premises. -/

namespace NightstreamFPrime.Lifecycle.Nebula

variable {K : Type*} [Field K] {Row Aux : Type*} {n m : ℕ}

/-- The absorb gadget of a row: its flag, its input lanes, its chain lanes after the row, and its
other rows, which read neither the gadget's output nor its permutation words. -/
structure SharedAbsorb (K Row : Type*) (n m : ℕ) where
  flag : Row → K
  input : Row → Fin n → K
  after : Row → Fin m → K
  rest : Row → Prop

namespace SharedAbsorb

variable (s : SharedAbsorb K Row n m) (P : (Fin n → K) → Aux → (Fin m → K) → Prop)

/-- The rows of one row with its own permutation words `aux` and output `out`: the block rows
over its own input, the gated chain rows and its other rows. -/
def OwnRows (z : Row) (aux : Aux) (out : Fin m → K) : Prop :=
  P (s.input z) aux out ∧ (∀ l, s.flag z * (s.after z l - out l) = 0) ∧ s.rest z

/-- The rows of a group `rows` with one shared block `(W, aux, O)`: the block rows once, then for
each row its gated input rows, its gated chain rows and its other rows. -/
def GroupRows (rows : List Row) (W : Fin n → K) (aux : Aux) (O : Fin m → K) : Prop :=
  P W aux O ∧ ∀ z ∈ rows,
    (∀ l, s.flag z * (W l - s.input z l) = 0) ∧ (∀ l, s.flag z * (s.after z l - O l) = 0) ∧
      s.rest z

/-- The block rows read only the block, so one copy holds exactly when the copy of each row of
a nonempty group holds: the group's rows are the one-row rows of each row on its block. -/
theorem groupRows_iff_forall {rows : List Row} (nonempty : rows ≠ []) {W : Fin n → K} {aux : Aux}
    {O : Fin m → K} :
    s.GroupRows P rows W aux O ↔ ∀ z ∈ rows, s.GroupRows P [z] W aux O := by
  constructor
  · rintro ⟨block, each⟩ z member
    exact ⟨block, fun z' same => by rw [List.mem_singleton.mp same]; exact each z member⟩
  · intro each
    obtain ⟨z, member⟩ := List.exists_mem_of_ne_nil rows nonempty
    exact ⟨(each z member).1, fun z' member' => (each z' member').2 z' (List.mem_singleton_self _)⟩

variable {s P}

/-- A gated equality with a nonzero flag is an equality. -/
theorem eq_of_gated {flag : K} {u v : Fin n → K} (nonzero : flag ≠ 0)
    (gated : ∀ l, flag * (u l - v l) = 0) : u = v :=
  funext fun l => sub_eq_zero.mp ((mul_eq_zero.mp (gated l)).resolve_left nonzero)

/-- Soundness. Let the block rows give the output `f` of their input and have rows for each
input. In a group whose rows hold, each row holds its own rows with the output `f` of its own
input, and the block's output is the output of each absorbing row's own input. -/
theorem groupRows_sound {f : (Fin n → K) → Fin m → K}
    (sound : ∀ i aux o, P i aux o → o = f i) (complete : ∀ i, ∃ aux, P i aux (f i))
    {rows : List Row} {W : Fin n → K} {aux : Aux} {O : Fin m → K}
    (holds : s.GroupRows P rows W aux O) {z : Row} (member : z ∈ rows) :
    (∃ aux', s.OwnRows P z aux' (f (s.input z))) ∧ (s.flag z ≠ 0 → O = f (s.input z)) := by
  obtain ⟨block, each⟩ := holds
  obtain ⟨input, chain, rest⟩ := each z member
  have output : s.flag z ≠ 0 → O = f (s.input z) := fun nonzero => by
    rw [sound _ _ _ block, eq_of_gated nonzero input]
  refine ⟨?_, output⟩
  obtain ⟨aux', own⟩ := complete (s.input z)
  refine ⟨aux', own, fun l => ?_, rest⟩
  by_cases zero : s.flag z = 0
  · rw [zero, zero_mul]
  · rw [← output zero]
    exact chain l

/-- Completeness. Let each row of a group hold its own rows, and let the absorbing rows of the
group (nonzero flag) have one input. Then the group has a shared block whose rows hold. -/
theorem groupRows_complete {f : (Fin n → K) → Fin m → K}
    (sound : ∀ i aux o, P i aux o → o = f i) (complete : ∀ i, ∃ aux, P i aux (f i))
    {rows : List Row} (own : ∀ z ∈ rows, ∃ aux out, s.OwnRows P z aux out)
    (agree : ∀ z ∈ rows, ∀ z' ∈ rows, s.flag z ≠ 0 → s.flag z' ≠ 0 → s.input z = s.input z') :
    ∃ W aux O, s.GroupRows P rows W aux O := by
  by_cases absorbs : ∃ z₀ ∈ rows, s.flag z₀ ≠ 0
  · obtain ⟨z₀, member₀, nonzero₀⟩ := absorbs
    obtain ⟨aux, block⟩ := complete (s.input z₀)
    refine ⟨s.input z₀, aux, f (s.input z₀), block, fun z member => ?_⟩
    obtain ⟨aux', out, ownBlock, chain, rest⟩ := own z member
    by_cases zero : s.flag z = 0
    · exact ⟨fun l => by rw [zero, zero_mul], fun l => by rw [zero, zero_mul], rest⟩
    · have same := agree z member z₀ member₀ zero nonzero₀
      refine ⟨fun l => by rw [same, sub_self, mul_zero], fun l => ?_, rest⟩
      rw [← same, ← sound _ _ _ ownBlock]
      exact chain l
  · obtain ⟨aux, block⟩ := complete 0
    refine ⟨0, aux, f 0, block, fun z member => ?_⟩
    obtain ⟨_, _, _, _, rest⟩ := own z member
    have zero : s.flag z = 0 := Classical.byContradiction fun nonzero => absorbs ⟨z, member, nonzero⟩
    exact ⟨fun l => by rw [zero, zero_mul], fun l => by rw [zero, zero_mul], rest⟩

/-- A cursor that moves at most one up per row and is 0 at row `t` is at most `j` at row `t + j`. -/
theorem cursor_le {cursor : ℕ → ℕ} (moves : ∀ t, cursor (t + 1) ≤ cursor t + 1) {t : ℕ}
    (zero : cursor t = 0) (j : ℕ) : cursor (t + j) ≤ j := by
  induction j with
  | zero => simp [zero]
  | succ j step => exact (moves (t + j)).trans (Nat.succ_le_succ step)

/-- The slot cursor before each row moves at most one up per row; an absorb has cursor 7 before
it and cursor 0 after it. Then two absorbs are at least eight rows apart. -/
theorem absorbs_apart {cursor : ℕ → ℕ} {absorbs : ℕ → Prop}
    (moves : ∀ t, cursor (t + 1) ≤ cursor t + 1) (full : ∀ t, absorbs t → cursor t = 7)
    (clears : ∀ t, absorbs t → cursor (t + 1) = 0) {t t' : ℕ} (first : absorbs t)
    (second : absorbs t') (before : t < t') : t + 8 ≤ t' := by
  obtain ⟨j, rfl⟩ := Nat.exists_eq_add_of_lt before
  have bound := cursor_le moves (clears t first) j
  rw [show t + 1 + j = t + j + 1 by omega, full _ second] at bound
  omega

/-- So any eight consecutive rows hold at most one absorb. -/
theorem absorbs_unique {cursor : ℕ → ℕ} {absorbs : ℕ → Prop}
    (moves : ∀ t, cursor (t + 1) ≤ cursor t + 1) (full : ∀ t, absorbs t → cursor t = 7)
    (clears : ∀ t, absorbs t → cursor (t + 1) = 0) {start i i' : ℕ} (small : i < 8)
    (small' : i' < 8) (first : absorbs (start + i)) (second : absorbs (start + i')) : i = i' := by
  rcases lt_trichotomy i i' with less | same | more
  · have := absorbs_apart moves full clears first second (by omega)
    omega
  · exact same
  · have := absorbs_apart moves full clears second first (by omega)
    omega

/-- Completeness of the WASM groups. Let each row `trace t` hold its own rows, let the slot
cursor before each row move at most one up per row, and let a row with a nonzero flag have cursor
7 before it and cursor 0 after it. Then each group of at most eight consecutive rows has a shared
block whose rows hold. -/
theorem group_complete {f : (Fin n → K) → Fin m → K} (sound : ∀ i aux o, P i aux o → o = f i)
    (complete : ∀ i, ∃ aux, P i aux (f i)) {trace : ℕ → Row} {cursor : ℕ → ℕ}
    (own : ∀ t, ∃ aux out, s.OwnRows P (trace t) aux out)
    (moves : ∀ t, cursor (t + 1) ≤ cursor t + 1)
    (full : ∀ t, s.flag (trace t) ≠ 0 → cursor t = 7)
    (clears : ∀ t, s.flag (trace t) ≠ 0 → cursor (t + 1) = 0) (start g : ℕ) (small : g ≤ 8) :
    ∃ W aux O, s.GroupRows P ((List.range g).map fun i => trace (start + i)) W aux O := by
  refine groupRows_complete sound complete (fun z member => ?_)
    fun z member z' member' absorbs absorbs' => ?_
  · obtain ⟨i, -, rfl⟩ := List.mem_map.mp member
    exact own _
  · obtain ⟨i, inside, rfl⟩ := List.mem_map.mp member
    obtain ⟨i', inside', rfl⟩ := List.mem_map.mp member'
    rw [absorbs_unique (absorbs := fun t => s.flag (trace t) ≠ 0) moves full clears
      (by have := List.mem_range.mp inside; omega) (by have := List.mem_range.mp inside'; omega)
      absorbs absorbs']

end SharedAbsorb

end NightstreamFPrime.Lifecycle.Nebula
