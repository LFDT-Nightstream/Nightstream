import NightstreamFPrime.Lifecycle.Nebula.SharedCursor

namespace NightstreamFPrime.Lifecycle.Nebula.SharedAbsorb

private theorem finite_absorbs_apart {q k g : ℕ} [NeZero q] {enter : Fin k}
    (large : 8 < q) (kindSmall : k < q) (trace : Fin g → CursorRows q k enter)
    (link : ∀ (i : Fin g) (next : i.val + 1 < g),
      (trace i).after = (trace ⟨i.val + 1, next⟩).before)
    (i j : Fin g) (first : (trace i).absorb ≠ 0) (second : (trace j).absorb ≠ 0)
    (less : i.val < j.val) : i.val + 8 ≤ j.val := by
  by_contra close
  have bounded : ∀ (n : ℕ) (inside : i.val + 1 + n < g), n ≤ 7 →
      (trace ⟨i.val + 1 + n, inside⟩).before.val ≤ n := by
    intro n
    induction n with
    | zero =>
      intro inside _
      have zero : (trace ⟨i.val + 1 + 0, inside⟩).before = 0 :=
        (link i (by simpa using inside)).symm.trans (checked_absorb_cursor large _ first).2
      simp [zero]
    | succ n previous =>
      intro inside short
      have prior : i.val + 1 + n < g := by omega
      have bound := previous prior (by omega)
      have step := checked_cursor_advance large kindSmall (trace ⟨i.val + 1 + n, prior⟩) (by omega)
      rw [link ⟨i.val + 1 + n, prior⟩ (by dsimp; omega)] at step
      exact step.trans (Nat.succ_le_succ bound)
  obtain ⟨d, distance⟩ := Nat.exists_eq_add_of_lt less
  have inside : i.val + 1 + d < g := by omega
  have bound := bounded d inside (by omega)
  have same : (⟨i.val + 1 + d, inside⟩ : Fin g) = j := by
    apply Fin.ext
    change i.val + 1 + d = j.val
    omega
  rw [same, (checked_absorb_cursor large _ second).1] at bound
  have seven : (7 : ZMod q).val = 7 := by
    simpa only [Nat.cast_ofNat] using (ZMod.val_natCast_of_lt (n := q) (a := 7) (by omega))
  rw [seven] at bound
  omega

/-- SPEC.md section 7.6. A finite checked group needs no cursor link after its
last row and contains at most one absorb. -/
theorem finite_absorbs_unique {q k g : ℕ} [NeZero q] {enter : Fin k}
    (large : 8 < q) (kindSmall : k < q) (small : g ≤ 8)
    (trace : Fin g → CursorRows q k enter)
    (link : ∀ (i : Fin g) (next : i.val + 1 < g),
      (trace i).after = (trace ⟨i.val + 1, next⟩).before)
    (i j : Fin g) (first : (trace i).absorb ≠ 0) (second : (trace j).absorb ≠ 0) : i = j := by
  rcases lt_trichotomy i.val j.val with less | same | more
  · have := finite_absorbs_apart large kindSmall trace link i j first second less
    omega
  · exact Fin.ext same
  · have := finite_absorbs_apart large kindSmall trace link j i second first more
    omega

/-- SPEC.md section 7.6. Completeness for the finite rows that Rust supplies.
No infinite execution or valid row witness outside this group is required. -/
theorem finite_group_complete {q k g : ℕ} [NeZero q] [Fact q.Prime] {enter : Fin k}
    {Row Aux : Type*} {n m : ℕ} (s : SharedAbsorb (ZMod q) Row n m)
    {P : (Fin n → ZMod q) → Aux → (Fin m → ZMod q) → Prop}
    {f : (Fin n → ZMod q) → Fin m → ZMod q}
    (sound : ∀ i aux o, P i aux o → o = f i) (complete : ∀ i, ∃ aux, P i aux (f i))
    (trace : Fin g → Row) (cursor : Fin g → CursorRows q k enter)
    (flag : ∀ i, s.flag (trace i) = (cursor i).absorb)
    (own : ∀ i, ∃ aux out, s.OwnRows P (trace i) aux out)
    (large : 8 < q) (kindSmall : k < q) (small : g ≤ 8)
    (link : ∀ (i : Fin g) (next : i.val + 1 < g),
      (cursor i).after = (cursor ⟨i.val + 1, next⟩).before) :
    ∃ W aux O, s.GroupRows P (List.ofFn trace) W aux O := by
  refine groupRows_complete sound complete (fun z member => ?_)
    fun z member z' member' absorbs absorbs' => ?_
  · obtain ⟨i, rfl⟩ := List.mem_ofFn.mp member
    exact own i
  · obtain ⟨i, rfl⟩ := List.mem_ofFn.mp member
    obtain ⟨j, rfl⟩ := List.mem_ofFn.mp member'
    have same := finite_absorbs_unique large kindSmall small cursor link i j
      (by simpa only [flag] using absorbs) (by simpa only [flag] using absorbs')
    rw [same]

end NightstreamFPrime.Lifecycle.Nebula.SharedAbsorb
