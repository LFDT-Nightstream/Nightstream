import NightstreamFPrime.Spec.Compression.Akita.Extract.Decoders

/-! A coupling by counting, for the slot reading of a squeeze (Akita, ePrint 2026/1983, revision
of 2026-11-23, Section 8.2 `[KECCAK]` and Appendix E.1, "Representatives").

- `exists_coupling`: for `|X| = n·|Y|`, a reading `f` of `X` and a partial reading `g` of `Y`
  with `n·|{y | g y = some t}| ≤ |{x | f x = t}|` for each `t`, a map `φ : X → Y` exists whose
  fibres all have `n` elements and with `g (φ x) = some t → f x = t`. `exists_reply_coupling` is
  this statement for `X = Reply × Y` and a reading of the first component.
- `card_preimage`: under a map whose fibres have `s` elements, the preimage of a set has `s`
  times the elements of the set.
- `expect_fibres`: a map whose fibres have one size keeps the uniform mean. `expect_coupled` is
  this statement for a product of such maps over a finite key type. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita
open scoped BigOperators

/-- For `|X| = n·|Y|` and readings `f` of `X` and `g` of `Y` with at least `n` elements of `X`
for each element of `Y` with the same reading, a map from `X` to `Y` exists whose fibres all have
`n` elements and that keeps each reading of `g`. -/
theorem exists_coupling {X Y T : Type} [Fintype X] [Fintype Y] (f : X → T) (g : Y → Option T)
    (n : ℕ) (size : Fintype.card X = n * Fintype.card Y)
    (dominated : ∀ t, n * Nat.card {y // g y = some t} ≤ Nat.card {x // f x = t}) :
    ∃ φ : X → Y, (∀ y, Nat.card {x // φ x = y} = n) ∧ ∀ x t, g (φ x) = some t → f x = t := by
  classical
  have embeds (t : T) : Nonempty (({y // g y = some t} × Fin n) ↪ {x // f x = t}) :=
    Function.Embedding.nonempty_of_card_le (by
      rw [Fintype.card_prod, Fintype.card_fin, ← Nat.card_eq_fintype_card,
        ← Nat.card_eq_fintype_card]
      exact (Nat.mul_comm _ _).trans_le (dominated t))
  have into : ∀ t, ({y // g y = some t} × Fin n) ↪ {x // f x = t} := fun t => (embeds t).some
  have apart : ∀ (t t' : T) (a : {y // g y = some t} × Fin n)
      (a' : {y // g y = some t'} × Fin n), (into t a).1 = (into t' a').1 →
      a.1.1 = a'.1.1 ∧ a.2 = a'.2 := by
    intro t t' a a' same
    have key : t = t' := (into t a).2.symm.trans ((congrArg f same).trans (into t' a').2)
    subst key
    have equal := (into t).injective (Subtype.ext same)
    subst equal
    exact ⟨rfl, rfl⟩
  obtain ⟨e, e_injective, e_reads⟩ : ∃ e : {p : Y × Fin n // (g p.1).isSome} → X,
      Function.Injective e ∧ ∀ p t, g p.1.1 = some t → f (e p) = t :=
    ⟨fun p => (into ((g p.1.1).get p.2) (⟨p.1.1, (Option.some_get p.2).symm⟩, p.1.2)).1,
      fun p p' same => by
        obtain ⟨first, second⟩ := apart _ _ _ _ same
        exact Subtype.ext (Prod.ext first second),
      fun p t found => (into _ _).2.trans (Option.some.inj ((Option.some_get p.2).trans found))⟩
  have count : Nat.card {p : Y × Fin n // ¬ (g p.1).isSome} =
      Nat.card {x // x ∉ Set.range e} := by
    have left := Nat.card_congr (Equiv.sumCompl fun p : Y × Fin n => (g p.1).isSome = true)
    have right := Nat.card_congr (Equiv.sumCompl fun x => x ∈ Set.range e)
    have middle : Nat.card {p : Y × Fin n // (g p.1).isSome} =
        Nat.card {x // x ∈ Set.range e} := Nat.card_congr (Equiv.ofInjective e e_injective)
    rw [Nat.card_sum] at left right
    rw [Nat.card_prod, Nat.card_eq_fintype_card (α := Y), Nat.card_eq_fintype_card (α := Fin n),
      Fintype.card_fin] at left
    rw [Nat.card_eq_fintype_card (α := X), size] at right
    have swap := Nat.mul_comm n (Fintype.card Y)
    omega
  obtain ⟨c⟩ : Nonempty ({p : Y × Fin n // ¬ (g p.1).isSome} ≃ {x // x ∉ Set.range e}) :=
    Fintype.card_eq.mp (by
      rw [← Nat.card_eq_fintype_card, ← Nat.card_eq_fintype_card]
      exact count)
  have glued : ∀ p p' : Y × Fin n,
      (if h : (g p.1).isSome then e ⟨p, h⟩ else (c ⟨p, h⟩).1) =
        (if h : (g p'.1).isSome then e ⟨p', h⟩ else (c ⟨p', h⟩).1) → p = p' := by
    intro p p' same
    by_cases h : (g p.1).isSome <;> by_cases h' : (g p'.1).isSome
    · rw [dite_eq_left h, dite_eq_left h'] at same
      exact congrArg Subtype.val (e_injective same)
    · rw [dite_eq_left h, dite_eq_right h'] at same
      exact ((c ⟨p', h'⟩).2 (Set.mem_range.mpr ⟨⟨p, h⟩, same⟩)).elim
    · rw [dite_eq_right h, dite_eq_left h'] at same
      exact ((c ⟨p, h⟩).2 (Set.mem_range.mpr ⟨⟨p', h'⟩, same.symm⟩)).elim
    · rw [dite_eq_right h, dite_eq_right h'] at same
      exact congrArg Subtype.val (c.injective (Subtype.ext same))
  obtain ⟨E, E_reads⟩ : ∃ E : Y × Fin n ≃ X, ∀ p (h : (g p.1).isSome), E p = e ⟨p, h⟩ :=
    ⟨Equiv.ofBijective (fun p => if h : (g p.1).isSome then e ⟨p, h⟩ else (c ⟨p, h⟩).1)
      ((Fintype.bijective_iff_injective_and_card _).mpr ⟨glued, by
        rw [Fintype.card_prod, Fintype.card_fin, size, Nat.mul_comm]⟩),
      fun p h => dite_eq_left h⟩
  refine ⟨fun x => (E.symm x).1, fun y => ?_, fun x t found => ?_⟩
  · exact (Nat.card_congr
      { toFun := fun x => (E.symm x.1).2
        invFun := fun i => ⟨E (y, i), congrArg Prod.fst (E.symm_apply_apply (y, i))⟩
        left_inv := fun x => Subtype.ext
          ((congrArg E (Prod.ext x.2.symm rfl : (y, (E.symm x.1).2) = E.symm x.1)).trans
            (E.apply_symm_apply x.1))
        right_inv := fun i => congrArg Prod.snd (E.symm_apply_apply (y, i)) }).trans
      (Nat.card_eq_fintype_card.trans (Fintype.card_fin n))
  · have decoded : (g (E.symm x).1).isSome := Option.isSome_iff_exists.mpr ⟨t, found⟩
    have back : e ⟨E.symm x, decoded⟩ = x :=
      (E_reads _ decoded).symm.trans (E.apply_symm_apply x)
    exact (congrArg f back).symm.trans (e_reads ⟨E.symm x, decoded⟩ t found)

/-- `exists_coupling` for the pairs of an answer and a string, and a reading of the answer. -/
theorem exists_reply_coupling {Reply Y T : Type} [Fintype Reply] [Fintype Y] (f : Reply → T)
    (g : Y → Option T)
    (dominated : ∀ t, Nat.card {y // g y = some t} * Fintype.card Reply ≤
      Nat.card {r // f r = t} * Fintype.card Y) :
    ∃ φ : Reply × Y → Y, (∀ y, Nat.card {x // φ x = y} = Fintype.card Reply) ∧
      ∀ r u t, g (φ (r, u)) = some t → f r = t := by
  obtain ⟨φ, fibres, reads⟩ := exists_coupling (fun x : Reply × Y => f x.1) g
    (Fintype.card Reply) (Fintype.card_prod Reply Y) fun t => by
      have pairs : Nat.card {x : Reply × Y // f x.1 = t} =
          Nat.card {r // f r = t} * Fintype.card Y :=
        (Nat.card_congr (β := {r // f r = t} × Y)
          { toFun := fun x => (⟨x.1.1, x.2⟩, x.1.2)
            invFun := fun p => ⟨(p.1.1, p.2), p.1.2⟩
            left_inv := fun _ => rfl
            right_inv := fun _ => rfl }).trans
          (by rw [Nat.card_prod, Nat.card_eq_fintype_card (α := Y)])
      rw [pairs, Nat.mul_comm]
      exact dominated t
  exact ⟨φ, fibres, fun r u t => reads (r, u) t⟩

/-- Under a map whose fibres have `s` elements, the preimage of a set has `s` times the elements
of the set. -/
theorem card_preimage {U P : Type} [Fintype U] [Fintype P] (D : U → P) {s : ℕ}
    (fibres : ∀ p, Nat.card {u // D u = p} = s) (S : P → Prop) :
    Nat.card {u // S (D u)} = s * Nat.card {p // S p} := by
  classical
  rw [Nat.card_congr (β := Σ p : {p // S p}, {u // D u = p.1})
      { toFun := fun u => ⟨⟨D u.1, u.2⟩, ⟨u.1, rfl⟩⟩
        invFun := fun x => ⟨x.2.1, (congrArg S x.2.2).mpr x.1.2⟩
        left_inv := fun _ => rfl
        right_inv := by
          rintro ⟨⟨p, inside⟩, ⟨u, same⟩⟩
          obtain rfl : D u = p := same
          rfl },
    Nat.card_sigma, Finset.sum_congr rfl fun p _ => fibres p.1, Finset.sum_const,
    Finset.card_univ, smul_eq_mul, Nat.card_eq_fintype_card, Nat.mul_comm]

/-- A map whose fibres have one size keeps the uniform mean. -/
theorem expect_fibres {X Y : Type} [Fintype X] [Nonempty X] [Fintype Y] (φ : X → Y) {n : ℕ}
    (fibres : ∀ y, Nat.card {x // φ x = y} = n) (F : Y → ℝ) : 𝔼 x, F (φ x) = 𝔼 y, F y := by
  have : Nonempty {x // φ x = φ (Classical.arbitrary X)} := ⟨⟨_, rfl⟩⟩
  exact Extract.expect_read φ (fibres (φ (Classical.arbitrary X)) ▸ Nat.card_pos) fibres F

/-- A product over a finite key type of maps whose fibres have one size keeps the uniform
mean. -/
theorem expect_coupled {Key : Type} [Fintype Key] [DecidableEq Key] {X Y : Key → Type}
    [∀ k, Fintype (X k)] [∀ k, Nonempty (X k)] [∀ k, Fintype (Y k)] (φ : ∀ k, X k → Y k)
    {n : Key → ℕ} (fibres : ∀ k y, Nat.card {x // φ k x = y} = n k) (F : (∀ k, Y k) → ℝ) :
    𝔼 x : ∀ k, X k, F (fun k => φ k (x k)) = 𝔼 y, F y :=
  expect_fibres (fun (x : ∀ k, X k) k => φ k (x k)) (n := ∏ k, n k) (fun y =>
    (Nat.card_congr
      { toFun := fun x k => ⟨x.1 k, congrFun x.2 k⟩
        invFun := fun p => ⟨fun k => (p k).1, funext fun k => (p k).2⟩
        left_inv := fun _ => rfl
        right_inv := fun _ => rfl }).trans
      (Nat.card_pi.trans (Finset.prod_congr rfl fun k _ => fibres k (y k)))) F

end NightstreamFPrime.Spec.Compression.Akita.Replay
