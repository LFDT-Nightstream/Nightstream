import NightstreamFPrime.Spec.Compression.Verifier.Model
import NightstreamFPrime.Spec.Compression.Akita.Extract.Staged

/-! Fair decoders of the challenges of the compression argument from the slots of an answer of
the oracle (Akita, ePrint 2026/1983, revision of 2026-11-23, Definition 10.16 (vii): the law of
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

section FieldSlots

open NightstreamFPrime.Spec.Compression

/-- A map whose fibres have one size `size > 0` keeps the uniform law. -/
theorem expect_read {R R' : Type} [Fintype R] [Fintype R'] (read : R → R') {size : ℕ}
    (positive : 0 < size) (fibres : ∀ y, Nat.card {x // read x = y} = size) (φ : R' → ℝ) :
    𝔼 x, φ (read x) = 𝔼 y, φ y := by
  have each (ψ : R' → ℝ) : ∑ x, ψ (read x) = size * ∑ y, ψ y := by
    rw [← Fintype.sum_fiberwise' read ψ, Finset.mul_sum]
    refine Finset.sum_congr rfl fun y _ => ?_
    rw [Finset.sum_const, Finset.card_univ, ← Nat.card_eq_fintype_card, fibres, nsmul_eq_mul]
  have cards := each fun _ => 1
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one] at cards
  rw [Finset.expect_eq_sum_div_card, Finset.expect_eq_sum_div_card, Finset.card_univ,
    Finset.card_univ, each φ, cards, mul_div_mul_left _ _ (Nat.cast_ne_zero.mpr positive.ne')]

/-- `expect_read` for the probability of a set. -/
theorem mass_read {R R' : Type} [Fintype R] [Fintype R'] (read : R → R') {size : ℕ}
    (positive : 0 < size) (fibres : ∀ y, Nat.card {x // read x = y} = size) (S : Set R') :
    mass {x | read x ∈ S} = mass S := by
  unfold mass
  exact expect_read read positive fibres fun y => if y ∈ S then 1 else 0

/-- A fair decoder after a map whose fibres have one size is fair. -/
theorem Fair.comp {R R' C : Type} [Fintype R] [Fintype R'] [Fintype C] {decode : R' → C}
    (fair : Fair decode) (read : R → R') {size : ℕ}
    (fibres : ∀ y, Nat.card {x // read x = y} = size) : Fair fun x => decode (read x) := by
  intro bad
  rcases Nat.eq_zero_or_pos size with rfl | positive
  · have : IsEmpty R := ⟨fun x => by
      have : Nonempty {y // read y = read x} := ⟨⟨x, rfl⟩⟩
      have := (Nat.card_pos (α := {y // read y = read x})).trans_eq (fibres (read x))
      omega⟩
    rw [show mass {x | decode (read x) ∈ bad} = 0 by simp [mass]]
    positivity
  · exact (le_of_eq (mass_read read positive fibres {y | decode y ∈ bad})).trans (fair bad)

/-- A map from the slots to `E` whose fibres have one size: a uniform slot gives a uniform
element of `E`. -/
class FieldSlot (Slot E : Type) [Fintype Slot] [Fintype E] : Prop where
  exists_field : ∃ (field : Slot → E) (size : ℕ), ∀ e, Nat.card {slot // field slot = e} = size

instance FieldSlot.self (E : Type) [Fintype E] : FieldSlot E E :=
  ⟨⟨id, 1, fun e => Nat.card_eq_one_iff_exists.mpr ⟨⟨e, rfl⟩, fun s => Subtype.ext s.2⟩⟩⟩

/-- A nonempty slot type with a field map gives a nonempty `E`. -/
theorem FieldSlot.nonempty (Slot E : Type) [Fintype Slot] [Nonempty Slot] [Fintype E]
    [FieldSlot Slot E] : Nonempty E := by
  obtain ⟨field, -⟩ := FieldSlot.exists_field (Slot := Slot) (E := E)
  exact ⟨field (Classical.arbitrary Slot)⟩

variable {Coord Slot E : Type} [Fintype Coord] [DecidableEq Coord] [Fintype Slot] [Nonempty Slot]
  [Fintype E] [FieldSlot Slot E]

omit [DecidableEq Coord] [Nonempty Slot] in
variable (Coord Slot E) in
/-- An answer with slots of `Slot` maps to an answer with slots of `E` by a map whose fibres
have one size. -/
theorem FieldSlot.exists_read :
    ∃ (read : (Coord → Slot) → Coord → E) (size : ℕ),
      ∀ y, Nat.card {x // read x = y} = size := by
  obtain ⟨field, size, fibres⟩ := FieldSlot.exists_field (Slot := Slot) (E := E)
  refine ⟨fun x c => field (x c), size ^ Fintype.card Coord, fun y => ?_⟩
  calc Nat.card {x : Coord → Slot // (fun c => field (x c)) = y}
      = Nat.card (∀ c, {s // field s = y c}) :=
        Nat.card_congr ((Equiv.subtypeEquivRight fun _ => funext_iff).trans
          (Equiv.subtypePiEquivPi (p := fun c s => field s = y c)))
    _ = ∏ c, Nat.card {s // field s = y c} := Nat.card_pi
    _ = size ^ Fintype.card Coord := by
        rw [Finset.prod_congr rfl fun c _ => fibres (y c), Finset.prod_const, Finset.card_univ]

/-- `exists_fair_slot` for the slot type `Slot`: a uniform element of `E`. -/
theorem exists_fair_field [Nonempty Coord] : ∃ decode : (Coord → Slot) → E, Fair decode := by
  have := FieldSlot.nonempty Slot E
  obtain ⟨decode, fair⟩ := exists_fair_slot (Coord := Coord) (Slot := E)
  obtain ⟨read, _, fibres⟩ := FieldSlot.exists_read Coord Slot E
  exact ⟨_, fair.comp read fibres⟩

/-- `exists_fair_slots` for the slot type `Slot`: a uniform vector of `count` elements of `E`. -/
theorem exists_fair_fields {count : ℕ} (enough : count ≤ Fintype.card Coord) :
    ∃ decode : (Coord → Slot) → Fin count → E, Fair decode := by
  have := FieldSlot.nonempty Slot E
  obtain ⟨decode, fair⟩ := exists_fair_slots (Coord := Coord) (Slot := E) enough
  obtain ⟨read, _, fibres⟩ := FieldSlot.exists_read Coord Slot E
  exact ⟨_, fair.comp read fibres⟩

/-- `exists_fair_family` for the slot type `Slot`: a uniform family of elements of `E`. -/
theorem exists_fair_field_family {Index : Type} [Fintype Index] [DecidableEq Index]
    (enough : Fintype.card Index ≤ Fintype.card Coord) :
    ∃ decode : (Coord → Slot) → Index → E, Fair decode := by
  have := FieldSlot.nonempty Slot E
  obtain ⟨decode, fair⟩ := exists_fair_family (Coord := Coord) (Slot := E) enough
  obtain ⟨read, _, fibres⟩ := FieldSlot.exists_read Coord Slot E
  exact ⟨_, fair.comp read fibres⟩

omit [Nonempty Slot] [Fintype E] [FieldSlot Slot E] in
/-- `exists_fair_base` for the slot type `Slot`: a uniform family of elements of the base
field `F`. -/
theorem exists_fair_field_base {F Index : Type} [Field F] [Fintype F] [Field E] [Fintype E]
    [Algebra F E] [FieldSlot Slot E] [Fintype Index] [DecidableEq Index]
    (enough : Fintype.card Index ≤ Fintype.card Coord) :
    ∃ decode : (Coord → Slot) → Index → F, Fair decode := by
  obtain ⟨decode, fair⟩ := exists_fair_base (F := F) (E := E) (Coord := Coord) enough
  obtain ⟨read, _, fibres⟩ := FieldSlot.exists_read Coord Slot E
  exact ⟨_, fair.comp read fibres⟩

omit [Nonempty Slot] [Fintype E] [FieldSlot Slot E] in
/-- `exists_fair_challenges` for the slot type `Slot`. -/
theorem exists_fair_field_challenges {F : Type} [Field E] [Fintype E] [FieldSlot Slot E]
    (P : Verifier.Params F E)
    (enough : 4 + (P.lookups + (P.claims + P.claims + P.spanVars)) ≤ Fintype.card Coord) :
    ∃ decode : (Coord → Slot) → Verifier.Challenges P, Fair decode := by
  obtain ⟨decode, fair⟩ := exists_fair_challenges (Coord := Coord) P enough
  obtain ⟨read, _, fibres⟩ := FieldSlot.exists_read Coord Slot E
  exact ⟨_, fair.comp read fibres⟩

end FieldSlots

end NightstreamFPrime.Spec.Compression.Akita.Extract
