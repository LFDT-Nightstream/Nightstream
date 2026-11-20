import NightstreamFPrime.Spec.Compression.Verifier.Model
import NightstreamFPrime.Spec.Compression.Akita.Extract.Staged

/-! Fair decoders of the challenges of the compression argument from the slots of an answer of
the oracle (Akita, ePrint 2026/1983, revision of 2026-10-15, Definition 10.16 (vii): the law of
the challenge of a stage; SPEC.md section 6).

- `Fair.map`: a fair decoder before a map whose fibres have one size is fair. `Fair.equiv` and
  `Fair.hom` are this statement for an equivalence and for a surjective additive map.
- `exists_fair_family`: an answer with a slot for each index gives a uniform family of slots.
- `exists_fair_base`: an answer with a slot of the extension field for each index gives a uniform
  family of elements of the base field.
- `exists_fair_challenges`: an answer with `4 + L + 2·s + span` slots gives uniform challenges
  `Verifier.Challenges`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.RandomOracle (mass)

attribute [local instance low] Classical.propDecidable

section Maps

variable {Reply : Type} [Fintype Reply] {C D : Type} [Fintype C] [Fintype D]

/-- A fair decoder before a map `f` is fair, when each fibre of `f` has `size` elements. -/
theorem Fair.map {decode : Reply → C} (fair : Fair decode) (f : C → D) {size : ℕ}
    (fibres : ∀ d, Nat.card {c // f c = d} = size) : Fair fun reply => f (decode reply) := by
  intro bad
  have fibre : ∀ d, (Finset.univ.filter fun c => f c = d).card = size := fun d => by
    rw [← fibres d, Nat.card_eq_fintype_card, Fintype.card_subtype]
  have count : ∀ part : Finset D,
      (Finset.univ.filter fun c => f c ∈ part).card = part.card * size := by
    intro part
    calc _ = ∑ d ∈ part, ((Finset.univ.filter fun c => f c ∈ part).filter fun c => f c = d).card :=
          Finset.card_eq_sum_card_fiberwise fun c member =>
            (Finset.mem_filter.mp (Finset.mem_coe.mp member)).2
      _ = ∑ _d ∈ part, size := Finset.sum_congr rfl fun d member => by
          rw [← fibre d]
          congr 1
          ext c
          simp only [Finset.mem_filter, Finset.mem_univ, true_and]
          exact ⟨fun both => both.2, fun same => ⟨same ▸ member, same⟩⟩
      _ = part.card * size := by rw [Finset.sum_const, smul_eq_mul]
  have total : Fintype.card C = Fintype.card D * size := by
    have all := count Finset.univ
    simpa using all
  have same : {reply | f (decode reply) ∈ bad} =
      {reply | decode reply ∈ Finset.univ.filter fun c => f c ∈ bad} := by
    ext reply
    simp
  rw [same]
  refine (fair _).trans ?_
  rw [count, total]
  push_cast
  rcases Nat.eq_zero_or_pos size with rfl | positive
  · simp only [Nat.cast_zero, mul_zero, div_zero]
    positivity
  · rw [mul_div_mul_right _ _ (Nat.cast_ne_zero.mpr positive.ne')]

/-- A fair decoder before an equivalence is fair. -/
theorem Fair.equiv {decode : Reply → C} (fair : Fair decode) (e : C ≃ D) :
    Fair fun reply => e (decode reply) :=
  fair.map e fun d => Nat.card_eq_fintype_card.trans (Fintype.card_eq_one_iff.mpr
    ⟨⟨e.symm d, e.apply_symm_apply d⟩, fun c =>
      Subtype.ext (e.injective (c.2.trans (e.apply_symm_apply d).symm))⟩)

/-- A fair decoder before a surjective additive map is fair: each fibre is a coset of the
kernel. -/
theorem Fair.hom [AddGroup C] [AddGroup D] {decode : Reply → C} (fair : Fair decode)
    (f : C →+ D) (onto : Function.Surjective f) : Fair fun reply => f (decode reply) :=
  fair.map f (size := Nat.card {c // f c = 0}) fun d => by
    obtain ⟨shift, rfl⟩ := onto d
    exact Nat.card_congr
      ⟨fun c => ⟨c.1 - shift, by rw [map_sub, c.2, sub_self]⟩,
        fun c => ⟨c.1 + shift, by rw [map_add, c.2, zero_add]⟩,
        fun c => Subtype.ext (sub_add_cancel _ _), fun c => Subtype.ext (add_sub_cancel_right _ _)⟩

end Maps

section Slots

variable {Coord Slot Index : Type} [Fintype Coord] [DecidableEq Coord] [Fintype Slot]
  [Nonempty Slot] [Fintype Index] [DecidableEq Index]

/-- An answer with a slot or more for each index gives a uniform family of slots. -/
theorem exists_fair_family (enough : Fintype.card Index ≤ Fintype.card Coord) :
    ∃ decode : (Coord → Slot) → Index → Slot, Fair decode := by
  obtain ⟨decode, fair⟩ := exists_fair_slots (Slot := Slot) enough
  exact ⟨_, fair.equiv (Equiv.arrowCongr (Fintype.equivFin Index).symm (Equiv.refl Slot))⟩

end Slots

section Base

variable {F E Coord Index : Type} [Field F] [Fintype F] [Field E] [Fintype E] [Algebra F E]
  [Fintype Coord] [DecidableEq Coord] [Fintype Index] [DecidableEq Index]

/-- An answer with a slot of the extension field `E` or more for each index gives a uniform
family of elements of the base field `F`: the image of each slot by a linear map `E → F` that is
the identity on `F`. -/
theorem exists_fair_base (enough : Fintype.card Index ≤ Fintype.card Coord) :
    ∃ decode : (Coord → E) → Index → F, Fair decode := by
  obtain ⟨decode, fair⟩ := exists_fair_family (Slot := E) enough
  obtain ⟨project, inverse⟩ := LinearMap.exists_leftInverse_of_injective (Algebra.linearMap F E)
    (LinearMap.ker_eq_bot.mpr (algebraMap F E).injective)
  let each : (Index → E) →+ (Index → F) :=
    { toFun := fun slots index => project (slots index)
      map_zero' := funext fun _ => map_zero project
      map_add' := fun _ _ => funext fun _ => map_add project _ _ }
  exact ⟨_, fair.hom each fun values => ⟨fun index => algebraMap F E (values index),
    funext fun index => LinearMap.congr_fun inverse (values index)⟩⟩

end Base

section Challenges

open NightstreamFPrime.Spec.Compression

variable {F E Coord : Type} [Field E] [Fintype E] [Fintype Coord] [DecidableEq Coord]
  (P : Verifier.Params F E)

/-- An answer with `4 + L + 2·s + span` slots or more gives uniform challenges `(μ, ω, δ, γ)`
and `(ψ, ξ, λ, β)` of the compression argument: each element is one slot. -/
theorem exists_fair_challenges
    (enough : 4 + (P.lookups + (P.claims + P.claims + P.spanVars)) ≤ Fintype.card Coord) :
    ∃ decode : (Coord → E) → Verifier.Challenges P, Fair decode := by
  obtain ⟨decode, fair⟩ := exists_fair_family (Slot := E)
    (Index := (Unit ⊕ (Unit ⊕ (Unit ⊕ Unit))) ⊕
      (Fin P.lookups ⊕ ((Fin P.claims ⊕ Fin P.claims) ⊕ Fin P.spanVars)))
    (by
      simp only [Fintype.card_sum, Fintype.card_unit, Fintype.card_fin]
      omega)
  exact ⟨_, fair.equiv ((Equiv.sumArrowEquivProdArrow _ _ _).trans (Equiv.prodCongr
    ((Equiv.sumArrowEquivProdArrow _ _ _).trans (Equiv.prodCongr (Equiv.funUnique Unit E)
      ((Equiv.sumArrowEquivProdArrow _ _ _).trans (Equiv.prodCongr (Equiv.funUnique Unit E)
        ((Equiv.sumArrowEquivProdArrow _ _ _).trans
          (Equiv.prodCongr (Equiv.funUnique Unit E) (Equiv.funUnique Unit E)))))))
    ((Equiv.sumArrowEquivProdArrow _ _ _).trans (Equiv.prodCongr (Equiv.refl _)
      ((Equiv.sumArrowEquivProdArrow _ _ _).trans
        (Equiv.prodCongr (Equiv.sumArrowEquivProdArrow _ _ _) (Equiv.refl _)))))))⟩

end Challenges

end NightstreamFPrime.Spec.Compression.Akita.Extract
