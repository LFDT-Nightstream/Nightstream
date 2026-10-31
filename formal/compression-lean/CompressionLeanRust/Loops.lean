import CompressionLeanRust.Backend
import CompressionLeanRust.Poseidon2

/-! Loop, index and list facts that the proofs of `hash.rs` share.

`loop_total` and `loop_sound` read a translated loop through an invariant `I n` whose measure `n`
falls at each step: the first shows that the loop returns a value in `Q`, the second that every value
it returns is in `Q`, also where a step can fail. `loop_forIn_spec` relates a loop that threads the
builder to the copy's `for` loop over a list. -/

namespace CompressionLeanRust

open Aeneas Aeneas.Std Result
open CompressionLean.Circuit

theorem loop_total {α β : Type} (body : α → Result (ControlFlow α β)) (I : ℕ → α → Prop) (Q : β → Prop)
    (h : ∀ n a, I n a → (∃ b, body a = ok (.done b) ∧ Q b) ∨ ∃ a' m, body a = ok (.cont a') ∧ m < n ∧ I m a') :
    ∀ n a, I n a → ∃ y, loop body a = ok y ∧ Q y := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro a hI
    rcases h n a hI with ⟨b, hb, hQ⟩ | ⟨a', m, ha, hm, hI'⟩
    · exact ⟨b, loop_done hb, hQ⟩
    · rw [loop_cont ha]
      exact ih m hm a' hI'

theorem loop_sound {α β : Type} (body : α → Result (ControlFlow α β)) (I : ℕ → α → Prop) (Q : β → Prop)
    (h : ∀ n a c, I n a → body a = ok c → (∃ b, c = .done b ∧ Q b) ∨ ∃ a' m, c = .cont a' ∧ m < n ∧ I m a') :
    ∀ n a y, I n a → loop body a = ok y → Q y := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro a y hI hl
    rw [loop] at hl
    obtain ⟨c, hc, hl⟩ := bind_eq_ok hl
    rcases h n a c hI hc with ⟨b, rfl, hQ⟩ | ⟨a', m, rfl, hm, hI'⟩
    · simp only [Result.ok.injEq] at hl
      exact hl ▸ hQ
    · exact ih m hm a' y hI' hl

/-- A translated loop that threads the builder, read as the copy's `for x in xs` with accumulator `r`:
`R` relates the loop state to the builder, the accumulator and the index `k` into `xs`. -/
theorem loop_forIn_spec {α γ ρ β : Type} (body : α → Result (ControlFlow α β)) (R : α → Builder → ρ → ℕ → Prop)
    (Q : β → ρ → Builder → Prop) (xs : List γ) (f : γ → ρ → Build ρ)
    (hstep : ∀ a s r k (h : k < xs.length), R a s r k →
      ∃ a', body a = ok (.cont a') ∧ R a' ((f xs[k] r).run s).2 ((f xs[k] r).run s).1 (k + 1))
    (hstop : ∀ a s r k, R a s r k → xs.length ≤ k → ∃ b, body a = ok (.done b) ∧ Q b r s) :
    ∀ a s r k, R a s r k → ∃ y, loop body a = ok y ∧
      (let p := (forIn (xs.drop k) r fun x r => ForInStep.yield <$> f x r).run s; Q y p.1 p.2) := by
  suffices h : ∀ n a s r k, R a s r k → xs.length - k = n → ∃ y, loop body a = ok y ∧
      (let p := (forIn (xs.drop k) r fun x r => ForInStep.yield <$> f x r).run s; Q y p.1 p.2) from
    fun a s r k hR => h _ a s r k hR rfl
  intro n
  induction n with
  | zero =>
    intro a s r k hR hk
    obtain ⟨b, hb, hQ⟩ := hstop a s r k hR (by omega)
    refine ⟨b, loop_done hb, ?_⟩
    rw [List.drop_eq_nil_of_le (by omega)]
    exact hQ
  | succ n ih =>
    intro a s r k hR hk
    have hlt : k < xs.length := by omega
    obtain ⟨a', hbody, hR'⟩ := hstep a s r k hlt hR
    obtain ⟨y, hy, hQ⟩ := ih _ _ _ _ hR' (by omega)
    refine ⟨y, by rw [loop_cont hbody]; exact hy, ?_⟩
    rw [List.drop_eq_getElem_cons hlt, List.forIn_cons]
    simp only [StateT.run_bind, StateT.run_map]
    exact hQ

/-! ## Results -/

theorem sound_of_eq {α β γ : Type} {f : α → Result β} {view : β → γ} {copy : α → γ}
    (eq : ∀ x, ∃ y, f x = ok y ∧ view y = copy x) : Sound f view copy := by
  intro x y h
  obtain ⟨y', h', hv⟩ := eq x
  rw [h', Result.ok.injEq] at h
  exact h ▸ hv

/-- `Sound` from an equation that holds where `P` does, and a proof that `f` returns only there. -/
theorem sound_of_eq_on {α β γ : Type} {f : α → Result β} {view : β → γ} {copy : α → γ} (P : α → Prop)
    (eq : ∀ x, P x → ∃ y, f x = ok y ∧ view y = copy x) (returns : ∀ x y, f x = ok y → P x) :
    Sound f view copy := by
  intro x y h
  obtain ⟨y', h', hv⟩ := eq x (returns x y h)
  rw [h', Result.ok.injEq] at h
  exact h ▸ hv

theorem complete_of_eq {α β γ : Type} {H : α → Prop} {f : α → Result β} {view : β → γ} {copy : α → γ}
    (sound : Sound f view copy) (w : α) (hw : H w) (eq : ∀ x, H x → ∃ y, f x = ok y ∧ view y = copy x) :
    Complete H f view copy :=
  complete_of sound ⟨w, hw⟩ fun x hx => let ⟨y, h, _⟩ := eq x hx; ⟨y, h⟩

/-! ## Indices and lists -/

theorem wrapping_add_le (i j : Usize) (h : i.val + j.val ≤ Usize.max) :
    (core.num.Usize.wrapping_add i j).val = i.val + j.val := by
  have : i.val + j.val < Usize.size := by scalar_tac
  simp [Nat.mod_eq_of_lt this]

theorem wrapping_sub_val (i j : Usize) (h : j.val ≤ i.val) :
    (core.num.Usize.wrapping_sub i j).val = i.val - j.val := by
  have hle : j.bv ≤ i.bv := by
    rw [BitVec.le_def]
    exact h
  have e := congrArg BitVec.toNat (core.num.Usize.wrapping_sub_bv_eq i j)
  rw [BitVec.toNat_sub_of_le hle] at e
  exact e

theorem slice_lt {α : Type} (x : Slice α) (i : Usize) : i < Slice.len x ↔ i.val < x.val.length := by
  rw [UScalar.lt_equiv, Slice.len_val, Slice.length]

theorem id_pure {α : Type} (x : α) : (pure x : Id α) = x := rfl

theorem take_drop_succ {α : Type} (l : List α) (i k : ℕ) (h : i + k < l.length) :
    (l.drop i).take (k + 1) = (l.drop i).take k ++ [l[i + k]] := by
  rw [List.take_add_one, List.getElem?_drop, List.getElem?_eq_getElem h]
  rfl

theorem take_min_length {α : Type} (l : List α) (n : ℕ) : l.take (min n l.length) = l.take n := by
  rw [← List.take_take]
  simp

/-- A translated `[F; 16]` whose lanes are `l`, as the copies read its lanes. -/
theorem lanesOf_eq (a : Aeneas.Std.Array Var 16#usize) (l : List Var) (h : a.val = l) :
    lanesOf a = fun lane => (_root_.Array.mk l)[lane]! := by
  funext lane
  simp [lanesOf, h, List.getD_eq_getElem?_getD]

end CompressionLeanRust
