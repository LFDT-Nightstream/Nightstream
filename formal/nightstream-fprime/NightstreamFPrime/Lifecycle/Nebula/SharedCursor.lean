import NightstreamFPrime.Lifecycle.Nebula.SharedRows
import Mathlib.Algebra.Field.ZMod
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.LinearCombination

namespace NightstreamFPrime.Lifecycle.Nebula.SharedAbsorb

private def bit (b : Bool) : ℕ := if b then 1 else 0

private theorem bit_le (b : Bool) : bit b ≤ 1 := by cases b <;> decide

private theorem sum_bits_nat {q k : ℕ} [NeZero q] (oneLt : 1 < q) (small : k < q) (p : Fin k → Bool)
    (gather : Bool) (row : ((∑ i, bit (p i) : ℕ) : ZMod q) = (bit gather : ℕ)) :
    ∑ i, bit (p i) = bit gather := by
  have bound : ∑ i, bit (p i) ≤ k := by
    calc
      _ ≤ ∑ _ : Fin k, 1 := Finset.sum_le_sum fun i _ => bit_le (p i)
      _ = k := by simp
  have one : bit gather < q := lt_of_le_of_lt (bit_le gather) oneLt
  have values := congrArg ZMod.val row
  simpa only [ZMod.val_natCast_of_lt (lt_of_le_of_lt bound small),
    ZMod.val_natCast_of_lt one] using values

private theorem active_of_bit {k : ℕ} (p : Fin k → Bool) (gather : Bool)
    (total : ∑ i, bit (p i) = bit gather) (i : Fin k) (active : p i = true) : gather = true := by
  have bound := Finset.single_le_sum (fun j (_ : j ∈ Finset.univ) => Nat.zero_le (bit (p j)))
    (Finset.mem_univ i)
  rw [total, active] at bound
  cases gather <;> simp_all [bit]

private theorem weighted_last (p : Fin 8 → Bool) (total : ∑ i, bit (p i) = 1)
    (last : p 7 = true) : ∑ i, i.val * bit (p i) = 7 := by
  have erase : ∑ i ∈ Finset.univ.erase (7 : Fin 8), bit (p i) = 0 := by
    have sum := Finset.sum_erase_add (s := Finset.univ) (f := fun i => bit (p i))
      (Finset.mem_univ (7 : Fin 8))
    have value : bit (p 7) = 1 := by simp [last, bit]
    rw [value, total] at sum
    omega
  have other (i : Fin 8) (different : i ≠ 7) : bit (p i) = 0 := by
    have bound := Finset.single_le_sum (fun j (_ : j ∈ Finset.univ.erase (7 : Fin 8)) =>
      Nat.zero_le (bit (p j))) (by simp [different] : i ∈ Finset.univ.erase (7 : Fin 8))
    rw [erase] at bound
    omega
  rw [Finset.sum_eq_single (7 : Fin 8)]
  · simp [bit, last]
  · intro i _ different
    simp [other i different]
  · simp

/-- SPEC.md section 7.6. The checked Boolean decoding, one-hot rows, cursor
transition, absorb row, and Enter exclusion of the WASM host-event gadget. -/
structure CursorRows (q k : ℕ) (enter : Fin k) where
  before : ZMod q
  after : ZMod q
  absorb : ZMod q
  gather : Bool
  advice : Bool
  positions : Fin 8 → Bool
  kinds : Fin k → Bool
  positions_row : ((∑ i, bit (positions i) : ℕ) : ZMod q) = (bit gather : ℕ)
  kinds_row : ((∑ i, bit (kinds i) : ℕ) : ZMod q) = (bit gather : ℕ)
  cursor_row : (bit gather : ZMod q) * (((∑ i, i.val * bit (positions i) : ℕ) : ZMod q) - before) = 0
  transition_row : (bit (kinds enter) : ZMod q) * (before + 1) =
    before + (bit gather : ZMod q) - 8 * (bit (positions 7) : ZMod q) - after
  absorb_row : (bit (positions 7) : ZMod q) * (1 - (bit advice : ZMod q)) = absorb
  enter_row : absorb * (bit (kinds enter) : ZMod q) = 0

variable {q k : ℕ} [NeZero q] {enter : Fin k}

private theorem last_before (large : 8 < q) (r : CursorRows q k enter)
    (last : r.positions 7 = true) : r.before = 7 ∧ r.gather = true := by
  have total := sum_bits_nat (by omega) large r.positions r.gather r.positions_row
  have gather := active_of_bit r.positions r.gather total 7 last
  have weighted := weighted_last r.positions (by simpa [gather, bit] using total) last
  have row := r.cursor_row
  rw [weighted] at row
  have row' : (7 : ZMod q) - r.before = 0 := by simpa [gather, bit] using row
  exact ⟨(sub_eq_zero.mp row').symm, gather⟩

/-- An absorbing row has cursor seven and resets its cursor to zero. -/
theorem checked_absorb_cursor (large : 8 < q) (r : CursorRows q k enter) (active : r.absorb ≠ 0) :
    r.before = 7 ∧ r.after = 0 := by
  have last : r.positions 7 = true := by
    cases h : r.positions 7
    · have := r.absorb_row
      simp [h, bit] at this
      exact False.elim (active this.symm)
    · rfl
  obtain ⟨before, gather⟩ := last_before large r last
  have noEnter : r.kinds enter = false := by
    cases h : r.kinds enter
    · rfl
    · have := r.enter_row
      simp [h, bit] at this
      exact False.elim (active this)
  have row := r.transition_row
  simp [noEnter, last, before, gather, bit] at row
  refine ⟨before, ?_⟩
  linear_combination row

/-- Below seven the checked cursor can increase by at most one. This does
not assume that an arbitrary field cursor already lies in the slot range. -/
theorem checked_cursor_advance (large : 8 < q) (kindSmall : k < q) (r : CursorRows q k enter)
    (small : r.before.val < 7) : r.after.val ≤ r.before.val + 1 := by
  have last : r.positions 7 = false := by
    cases h : r.positions 7
    · rfl
    · have before := (last_before large r h).1
      have value : r.before.val = 7 := by rw [before]; exact ZMod.val_natCast_of_lt (n := q) (a := 7) (by omega)
      omega
  have enterActive : r.kinds enter = true → r.gather = true :=
    active_of_bit r.kinds r.gather (sum_bits_nat (by omega) kindSmall r.kinds r.gather r.kinds_row) enter
  have row := r.transition_row
  cases gather : r.gather <;> cases enters : r.kinds enter
  · have after : r.after = r.before := by
      simp [gather, enters, last, bit] at row
      linear_combination row
    rw [after]
    omega
  · have := enterActive enters
    simp [gather] at this
  · have after : r.after = r.before + 1 := by
      simp [gather, enters, last, bit] at row
      linear_combination row
    rw [after]
    have bound := ZMod.val_add_le r.before (1 : ZMod q)
    have one : (1 : ZMod q).val = 1 := by
      simpa only [Nat.cast_one] using ZMod.val_natCast_of_lt (n := q) (a := 1) (by omega : 1 < q)
    simpa [one] using bound
  · have after : r.after = 0 := by
      simp [gather, enters, last, bit] at row
      linear_combination row
    simp [after]

private theorem bounded_cursor_le {cursor : ℕ → ℕ}
    (moves : ∀ t, cursor t < 7 → cursor (t + 1) ≤ cursor t + 1) {t : ℕ}
    (zero : cursor t = 0) (j : ℕ) (small : j ≤ 7) : cursor (t + j) ≤ j := by
  induction j with
  | zero => simp [zero]
  | succ j step =>
    have previous := step (by omega)
    exact (moves (t + j) (by omega)).trans (by omega)

/-- Checked cursor rows and their actual continuity links imply absorb spacing. -/
theorem checked_absorbs_apart (large : 8 < q) (kindSmall : k < q)
    (trace : ℕ → CursorRows q k enter)
    (link : ∀ t, (trace t).after = (trace (t + 1)).before) {t t' : ℕ}
    (first : (trace t).absorb ≠ 0) (second : (trace t').absorb ≠ 0) (before : t < t') :
    t + 8 ≤ t' := by
  by_contra close
  obtain ⟨j, rfl⟩ := Nat.exists_eq_add_of_lt before
  have zero : (trace (t + 1)).before.val = 0 := by
    rw [← link t, (checked_absorb_cursor large (trace t) first).2]
    simp
  have bound := bounded_cursor_le (cursor := fun t => (trace t).before.val)
    (fun t small => by rw [← link t]; exact checked_cursor_advance large kindSmall (trace t) small)
    zero j (by omega)
  rw [show t + 1 + j = t + j + 1 by omega,
    (checked_absorb_cursor large _ second).1] at bound
  have seven : (7 : ZMod q).val = 7 := ZMod.val_natCast_of_lt (n := q) (a := 7) (by omega)
  rw [seven] at bound
  omega

/-- Every group of at most eight rows has at most one absorbing row, including
any final short group of a step. -/
theorem checked_absorbs_unique (large : 8 < q) (kindSmall : k < q)
    (trace : ℕ → CursorRows q k enter)
    (link : ∀ t, (trace t).after = (trace (t + 1)).before) {start i j : ℕ}
    (small : i < 8) (small' : j < 8)
    (first : (trace (start + i)).absorb ≠ 0) (second : (trace (start + j)).absorb ≠ 0) : i = j := by
  rcases lt_trichotomy i j with less | same | more
  · have := checked_absorbs_apart large kindSmall trace link first second (by omega)
    omega
  · exact same
  · have := checked_absorbs_apart large kindSmall trace link second first (by omega)
    omega

/-- Shared-block completeness follows from the checked cursor rows and links,
with no global natural-cursor range premise. -/
theorem checked_group_complete [Fact q.Prime] {Row Aux : Type*} {n m : ℕ}
    (s : SharedAbsorb (ZMod q) Row n m)
    {P : (Fin n → ZMod q) → Aux → (Fin m → ZMod q) → Prop}
    {f : (Fin n → ZMod q) → Fin m → ZMod q}
    (sound : ∀ i aux o, P i aux o → o = f i) (complete : ∀ i, ∃ aux, P i aux (f i))
    (trace : ℕ → Row) (cursor : ℕ → CursorRows q k enter)
    (flag : ∀ t, s.flag (trace t) = (cursor t).absorb)
    (own : ∀ t, ∃ aux out, s.OwnRows P (trace t) aux out)
    (large : 8 < q) (kindSmall : k < q)
    (link : ∀ t, (cursor t).after = (cursor (t + 1)).before)
    (start g : ℕ) (small : g ≤ 8) :
    ∃ W aux O, s.GroupRows P ((List.range g).map fun i => trace (start + i)) W aux O := by
  refine groupRows_complete sound complete (fun z member => ?_)
    fun z member z' member' absorbs absorbs' => ?_
  · obtain ⟨i, -, rfl⟩ := List.mem_map.mp member
    exact own _
  · obtain ⟨i, inside, rfl⟩ := List.mem_map.mp member
    obtain ⟨j, inside', rfl⟩ := List.mem_map.mp member'
    rw [flag] at absorbs absorbs'
    rw [checked_absorbs_unique large kindSmall cursor link
      (by have := List.mem_range.mp inside; omega) (by have := List.mem_range.mp inside'; omega)
      absorbs absorbs']

end NightstreamFPrime.Lifecycle.Nebula.SharedAbsorb
