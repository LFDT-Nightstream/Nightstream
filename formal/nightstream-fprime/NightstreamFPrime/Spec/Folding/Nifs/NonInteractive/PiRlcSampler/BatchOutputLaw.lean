import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Data.Finite.Sigma
import Mathlib.SetTheory.Cardinal.Finite
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-!
Owns the finite product comparison used by the ordered sampler batch.
Each input function has equal mass. The scalar event bound and both
normalizations are explicit; this theorem supplies no transcript law.
-/

namespace NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.BatchOutputLaw

open scoped BigOperators

attribute [local instance] Classical.propDecidable

private noncomputable def frequency {Source : Type*} (event : Source → Prop) : ℚ :=
  (Nat.card {source // event source} : ℚ) / Nat.card Source

private theorem frequency_congr {Source : Type*} {first second : Source → Prop}
    (same : ∀ source, first source ↔ second source) : frequency first = frequency second := by
  unfold frequency
  rw [Nat.card_congr (Equiv.subtypeEquivRight same)]

private theorem frequency_equiv {Left Right : Type*}
    (equivalence : Left ≃ Right) (event : Right → Prop) :
    frequency (fun left => event (equivalence left)) = frequency event := by
  unfold frequency
  rw [Nat.card_congr (Equiv.subtypeEquivOfSubtype (p := event) equivalence),
    Nat.card_congr equivalence]

private theorem product_frequency {Left Right : Type*} [Finite Left] [Fintype Right]
    (event : Left × Right → Prop) :
    frequency event =
      (∑ right : Right, frequency (fun left : Left => event (left, right))) /
        (Nat.card Right : ℚ) := by
  let split : {input : Left × Right // event input} ≃
      (Σ right : Right, {left : Left // event (left, right)}) :=
    { toFun := fun input => ⟨input.val.2, ⟨input.val.1, input.property⟩⟩
      invFun := fun input => ⟨(input.2.val, input.1), input.2.property⟩
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  have count := Nat.card_congr split
  rw [Nat.card_sigma] at count
  unfold frequency
  rw [count, Nat.card_prod, Nat.cast_mul, Nat.cast_sum, ← Finset.sum_div, div_div]

private theorem replace_left {Left Right Common Output : Type*}
    [Finite Left] [Finite Right] [Finite Common]
    (left : Left → Output) (right : Right → Output) (error : ℚ)
    (single : ∀ event : Output → Prop,
      |frequency (fun input => event (left input)) -
        frequency (fun input => event (right input))| ≤ error)
    (commonPositive : (0 : ℚ) < Nat.card Common) (event : Output × Common → Prop) :
    |frequency (fun input : Left × Common => event (left input.1, input.2)) -
      frequency (fun input : Right × Common => event (right input.1, input.2))| ≤ error := by
  letI : Fintype Common := Fintype.ofFinite _
  rw [product_frequency, product_frequency, ← sub_div, abs_div,
    abs_of_pos commonPositive, ← Finset.sum_sub_distrib]
  calc
    _ ≤ (∑ common : Common,
        |frequency (fun input : Left => event (left input, common)) -
          frequency (fun input : Right => event (right input, common))|) /
        (Nat.card Common : ℚ) :=
      div_le_div_of_nonneg_right (Finset.abs_sum_le_sum_abs _ _)
        (le_of_lt commonPositive)
    _ ≤ (∑ _common : Common, error) / (Nat.card Common : ℚ) :=
      div_le_div_of_nonneg_right
        (Finset.sum_le_sum (fun common _member => single (fun output => event (output, common))))
        (le_of_lt commonPositive)
    _ = error := by
      simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul,
        ← Nat.card_eq_fintype_card]
      field_simp [ne_of_gt commonPositive]

private theorem product_error {Left Right TailLeft TailRight Output TailOutput : Type*}
    [Finite Left] [Finite Right] [Finite TailLeft] [Finite TailRight]
    (left : Left → Output) (right : Right → Output)
    (tailLeft : TailLeft → TailOutput) (tailRight : TailRight → TailOutput)
    (error tailError : ℚ)
    (single : ∀ event : Output → Prop,
      |frequency (fun input => event (left input)) -
        frequency (fun input => event (right input))| ≤ error)
    (tail : ∀ event : TailOutput → Prop,
      |frequency (fun input => event (tailLeft input)) -
        frequency (fun input => event (tailRight input))| ≤ tailError)
    (rightPositive : (0 : ℚ) < Nat.card Right)
    (tailLeftPositive : (0 : ℚ) < Nat.card TailLeft)
    (event : Output × TailOutput → Prop) :
    |frequency (fun input : Left × TailLeft => event (left input.1, tailLeft input.2)) -
      frequency (fun input : Right × TailRight => event (right input.1, tailRight input.2))| ≤
      error + tailError := by
  have first := replace_left left right error single tailLeftPositive
    (fun input : Output × TailLeft => event (input.1, tailLeft input.2))
  have second :
      |frequency (fun input : Right × TailLeft => event (right input.1, tailLeft input.2)) -
        frequency (fun input : Right × TailRight => event (right input.1, tailRight input.2))| ≤
        tailError := by
    rw [← frequency_equiv (Equiv.prodComm TailLeft Right),
      ← frequency_equiv (Equiv.prodComm TailRight Right)]
    exact replace_left tailLeft tailRight tailError tail rightPositive
      (fun input : TailOutput × Right => event (right input.2, input.1))
  calc
    _ ≤ |frequency (fun input : Left × TailLeft => event (left input.1, tailLeft input.2)) -
          frequency (fun input : Right × TailLeft => event (right input.1, tailLeft input.2))| +
        |frequency (fun input : Right × TailLeft => event (right input.1, tailLeft input.2)) -
          frequency (fun input : Right × TailRight => event (right input.1, tailRight input.2))| :=
      abs_sub_le _ _ _
    _ ≤ error + tailError := add_le_add first second

private theorem cons_frequency {Source Output : Type*} (decode : Source → Output)
    (count : Nat) (event : (Fin (count + 1) → Output) → Prop) :
    frequency (fun input : Fin (count + 1) → Source => event (fun index => decode (input index))) =
      frequency (fun input : Source × (Fin count → Source) =>
        event (Fin.cons (decode input.1) (fun index => decode (input.2 index)))) := by
  let split := Fin.consEquiv (fun _ : Fin (count + 1) => Source)
  have mapped (input : Source × (Fin count → Source)) :
      (fun index => decode (split input index)) =
        Fin.cons (decode input.1) (fun index => decode (input.2 index)) := by
    funext index
    refine Fin.cases ?_ (fun _ => ?_) index <;> rfl
  calc
    _ = frequency (fun input : Source × (Fin count → Source) =>
        event (fun index => decode (split input index))) :=
      (frequency_equiv split _).symm
    _ = _ := frequency_congr (fun input => iff_of_eq (congrArg event (mapped input)))

/-- Independent repetition scales every scalar event bound by the number
of coordinates. The output type can contain failure, and a later event can
collect all coordinates or reject the whole batch. Both finite input
normalizations are positive premises, separate from the scalar comparison. -/
theorem independent_batch_event_error_le {Left Right Output : Type*}
    [Finite Left] [Finite Right]
    (left : Left → Output) (right : Right → Output) (error : ℚ)
    (leftPositive : (0 : ℚ) < Nat.card Left)
    (rightPositive : (0 : ℚ) < Nat.card Right)
    (single : ∀ event : Output → Prop,
      |(Nat.card {input : Left // event (left input)} : ℚ) / Nat.card Left -
        (Nat.card {input : Right // event (right input)} : ℚ) / Nat.card Right| ≤ error)
    (count : Nat) (event : (Fin count → Output) → Prop) :
    |(Nat.card {input : Fin count → Left // event (fun index => left (input index))} : ℚ) /
          Nat.card (Fin count → Left) -
      (Nat.card {input : Fin count → Right // event (fun index => right (input index))} : ℚ) /
          Nat.card (Fin count → Right)| ≤ (count : ℚ) * error := by
  change |frequency (fun input : Fin count → Left => event (fun index => left (input index))) -
    frequency (fun input : Fin count → Right => event (fun index => right (input index)))| ≤ _
  induction count with
  | zero =>
      let empty : (Fin 0 → Left) ≃ (Fin 0 → Right) :=
        { toFun := fun _ => Fin.elim0
          invFun := fun _ => Fin.elim0
          left_inv := fun _ => Subsingleton.elim _ _
          right_inv := fun _ => Subsingleton.elim _ _ }
      have same :
          frequency (fun input : Fin 0 → Left => event (fun index => left (input index))) =
            frequency (fun input : Fin 0 → Right => event (fun index => right (input index))) := by
        calc
          _ = frequency (fun _ : Fin 0 → Left => event Fin.elim0) :=
            frequency_congr (fun _ => iff_of_eq (congrArg event (Subsingleton.elim _ _)))
          _ = frequency (fun _ : Fin 0 → Right => event Fin.elim0) :=
            frequency_equiv empty (fun _ : Fin 0 → Right => event (Fin.elim0 : Fin 0 → Output))
          _ = _ := frequency_congr
            (fun _ => iff_of_eq (congrArg event (Subsingleton.elim _ _)))
      rw [same, sub_self, abs_zero, Nat.cast_zero, zero_mul]
  | succ count inductionHypothesis =>
      rw [cons_frequency, cons_frequency]
      have tailPositive : (0 : ℚ) < Nat.card (Fin count → Left) := by
        rw [Nat.card_fun, Nat.card_fin, Nat.cast_pow]
        exact pow_pos leftPositive _
      have compared := product_error left right
        (fun input : Fin count → Left => fun index => left (input index))
        (fun input : Fin count → Right => fun index => right (input index))
        error ((count : ℚ) * error) single inductionHypothesis rightPositive tailPositive
        (fun input : Output × (Fin count → Output) => event (Fin.cons input.1 input.2))
      calc
        _ ≤ error + (count : ℚ) * error := compared
        _ = ((count + 1 : Nat) : ℚ) * error := by rw [Nat.cast_add, Nat.cast_one]; ring

end NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.BatchOutputLaw
