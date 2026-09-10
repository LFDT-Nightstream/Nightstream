import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Data.Finite.Sum
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldBatchShortfall

/-!
Owns the 32-lane comparison between uniform Goldilocks fields and uniform
16-bit candidate windows. The finite coupling bounds every deterministic
decoder event, including abort. It assigns no output law to Poseidon2.
-/

namespace NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldOutputLaw

open ProductionAlphabet Sampling FieldPairLaw FieldShortfall

local instance : NeZero pairModulus := ⟨by decide⟩

private theorem event_card_le_add {Source : Type*} [Finite Source]
    (first second bad : Source → Prop)
    (covered : ∀ source, first source → second source ∨ bad source) :
    Nat.card {source // first source} ≤
      Nat.card {source // second source} + Nat.card {source // bad source} := by
  classical
  let selected : {source // first source} → {source // second source} ⊕ {source // bad source} :=
    fun source => if accepted : second source.val then
      Sum.inl ⟨source.val, accepted⟩
    else Sum.inr ⟨source.val, (covered source.val source.property).resolve_left accepted⟩
  let project : {source // second source} ⊕ {source // bad source} → Source :=
    Sum.elim Subtype.val Subtype.val
  have recovered (source : {source // first source}) : project (selected source) = source.val := by
    dsimp only [selected]
    split_ifs <;> rfl
  have injective : Function.Injective selected := by
    intro left right same
    exact Subtype.ext ((recovered left).symm.trans
      ((congrArg project same).trans (recovered right)))
  simpa only [Nat.card_sum] using Nat.card_le_card_of_injective selected injective

private theorem left_event_card {Left Right : Type*} (event : Left → Prop) :
    Nat.card {input : Left × Right // event input.1} =
      Nat.card {left : Left // event left} * Nat.card Right :=
  (Nat.card_congr (Equiv.prodSubtypeFstEquivSubtypeProd (p := event))).trans (Nat.card_prod _ _)

private def rightEventEquiv {Left Right : Type*} (event : Right → Prop) :
    {input : Left × Right // event input.2} ≃ Left × {right : Right // event right} where
  toFun input := (input.val.1, ⟨input.val.2, input.property⟩)
  invFun input := ⟨(input.1, input.2.val), input.2.property⟩
  left_inv _ := Subtype.ext rfl
  right_inv _ := rfl

private theorem coupled_right_card {Left Right : Type*}
    (coupling : Left × Right ≃ Left × Right) (event : Right → Prop) :
    Nat.card {input : Left × Right // event (coupling input).2} =
      Nat.card Left * Nat.card {right : Right // event right} :=
  (Nat.card_congr ((Equiv.subtypeEquivOfSubtype
    (p := fun input : Left × Right => event input.2) coupling).trans
      (rightEventEquiv event))).trans (Nat.card_prod _ _)

private theorem coupled_event_error {Left Right : Type*} [Finite Left] [Finite Right]
    (coupling : Left × Right ≃ Left × Right)
    (first : Left → Prop) (second : Right → Prop) (bad : Left × Right → Prop)
    (leftPositive : (0 : ℚ) < Nat.card Left) (rightPositive : (0 : ℚ) < Nat.card Right)
    (agree : ∀ input, ¬ bad input → (first input.1 ↔ second (coupling input).2)) :
    |(Nat.card {left // first left} : ℚ) / Nat.card Left -
        (Nat.card {right // second right} : ℚ) / Nat.card Right| ≤
      (Nat.card {input // bad input} : ℚ) / ((Nat.card Left : ℚ) * Nat.card Right) := by
  classical
  have forward := event_card_le_add
    (fun input : Left × Right => first input.1)
    (fun input : Left × Right => second (coupling input).2) bad (by
      intro input accepted
      by_cases failed : bad input
      · exact Or.inr failed
      · exact Or.inl ((agree input failed).mp accepted))
  have backward := event_card_le_add
    (fun input : Left × Right => second (coupling input).2)
    (fun input : Left × Right => first input.1) bad (by
      intro input accepted
      by_cases failed : bad input
      · exact Or.inr failed
      · exact Or.inl ((agree input failed).mpr accepted))
  rw [left_event_card, coupled_right_card] at forward backward
  have forwardQ :
      (Nat.card {left // first left} : ℚ) * Nat.card Right ≤
        (Nat.card Left : ℚ) * Nat.card {right // second right} + Nat.card {input // bad input} := by
    exact_mod_cast forward
  have backwardQ :
      (Nat.card Left : ℚ) * Nat.card {right // second right} ≤
        (Nat.card {left // first left} : ℚ) * Nat.card Right + Nat.card {input // bad input} := by
    exact_mod_cast backward
  have deviation :
      |(Nat.card {left // first left} : ℚ) * Nat.card Right -
        (Nat.card Left : ℚ) * Nat.card {right // second right}| ≤
          (Nat.card {input // bad input} : ℚ) := by
    rw [abs_le]
    constructor <;> linarith only [forwardQ, backwardQ]
  rw [div_sub_div _ _ (ne_of_gt leftPositive) (ne_of_gt rightPositive), abs_div,
    abs_of_pos (mul_pos leftPositive rightPositive)]
  exact div_le_div_of_nonneg_right deviation (le_of_lt (mul_pos leftPositive rightPositive))

private def LaneDisagreement (input : F × Fin pairModulus) : Prop :=
  low32 input.1 ≠ (laneCoupling input).2

private theorem lane_disagreement_card :
    Nat.card {input : F × Fin pairModulus // LaneDisagreement input} = pairModulus - 1 := by
  let exactEvent : {input : F × Fin pairModulus // LaneDisagreement input} ≃
      {input : F × Fin pairModulus //
        input.1.val = goldilocksModulus - 1 ∧ input.2.val ≠ 0} :=
    Equiv.subtypeEquivRight (fun input => lane_coupling_disagrees_iff input.1 input.2)
  have finalCard : Nat.card {value : F // value.val = goldilocksModulus - 1} = 1 := by
    apply Nat.card_eq_one_iff_exists.mpr
    refine ⟨⟨⟨goldilocksModulus - 1, by decide⟩, rfl⟩, ?_⟩
    intro value
    exact Subtype.ext (Fin.ext value.property)
  have zeroIff (value : Fin pairModulus) : value.val = 0 ↔ value = 0 := by
    constructor
    · exact fun same => Fin.ext same
    · exact fun same => congrArg Fin.val same
  have nonzeroCard : Nat.card {value : Fin pairModulus // value.val ≠ 0} = pairModulus - 1 := by
    calc
      _ = Nat.card {value : Fin pairModulus // value ≠ 0} :=
        Nat.card_congr (Equiv.subtypeEquivRight (fun value => not_congr (zeroIff value)))
      _ = Nat.card (Fin (pairModulus - 1)) :=
        (Nat.card_congr (finSuccAboveEquiv (n := pairModulus - 1) (0 : Fin pairModulus))).symm
      _ = pairModulus - 1 := Nat.card_fin _
  calc
    _ = Nat.card {input : F × Fin pairModulus //
        input.1.val = goldilocksModulus - 1 ∧ input.2.val ≠ 0} := Nat.card_congr exactEvent
    _ = Nat.card {value : F // value.val = goldilocksModulus - 1} *
        Nat.card {value : Fin pairModulus // value.val ≠ 0} :=
      (Nat.card_congr (Equiv.subtypeProdEquivProd
        (p := fun value : F => value.val = goldilocksModulus - 1)
        (q := fun value : Fin pairModulus => value.val ≠ 0))).trans (Nat.card_prod _ _)
    _ = pairModulus - 1 := by rw [finalCard, nonzeroCard, Nat.one_mul]

abbrev JointWindow := FieldWindow × PairWindow

def CoupledDisagreement (input : JointWindow) : Prop :=
  ∃ lane : Fin fieldLaneCount, LaneDisagreement (input.1 lane, input.2 lane)

/-- The 32-coordinate union bound retains the exact exceptional-pair count. -/
theorem scalar_coupling_disagreement_card_le :
    Nat.card {input : JointWindow // CoupledDisagreement input} ≤
      fieldLaneCount * (pairModulus - 1) *
        (goldilocksModulus * pairModulus) ^ (fieldLaneCount - 1) := by
  let split := Equiv.arrowProdEquivProdArrow (Fin fieldLaneCount)
    (fun _ => F) (fun _ => Fin pairModulus)
  have sameCount : Nat.card {input : JointWindow // CoupledDisagreement input} =
      Nat.card {lanes : Fin fieldLaneCount → F × Fin pairModulus //
        ∃ lane, LaneDisagreement (lanes lane)} :=
    (Nat.card_congr (Equiv.subtypeEquiv
      (p := fun lanes : Fin fieldLaneCount → F × Fin pairModulus =>
        ∃ lane, LaneDisagreement (lanes lane))
      (q := CoupledDisagreement) split (fun _ => Iff.rfl))).symm
  have counted := FieldBatchShortfall.finite_batch_event_card_le 31 LaneDisagreement
  rw [lane_disagreement_card, Nat.card_prod, Nat.card_fin, Nat.card_fin] at counted
  rw [sameCount]
  exact counted

private theorem pair_window_cardinality : Nat.card PairWindow = pairModulus ^ fieldLaneCount := by
  rw [Nat.card_fun, Nat.card_fin, Nat.card_fin]

private theorem disagreement_ratio_le :
    (Nat.card {input : JointWindow // CoupledDisagreement input} : ℚ) /
        ((Nat.card FieldWindow : ℚ) * Nat.card PairWindow) ≤
      (fieldLaneCount : ℚ) * pairDeviation := by
  have counted : (Nat.card {input : JointWindow // CoupledDisagreement input} : ℚ) ≤
      (fieldLaneCount : ℚ) * ((pairModulus - 1 : Nat) : ℚ) *
        ((goldilocksModulus : ℚ) * pairModulus) ^ (fieldLaneCount - 1) := by
    exact_mod_cast scalar_coupling_disagreement_card_le
  have fieldPositive : (0 : ℚ) < goldilocksModulus := by norm_num [goldilocksModulus]
  have pairPositive : (0 : ℚ) < pairModulus := by norm_num [pairModulus, chunkModulus]
  have nonzero : (goldilocksModulus : ℚ) * pairModulus ≠ 0 :=
    ne_of_gt (mul_pos fieldPositive pairPositive)
  have denominator :
      ((goldilocksModulus : ℚ) * pairModulus) ^ fieldLaneCount =
        ((goldilocksModulus : ℚ) * pairModulus) ^ (fieldLaneCount - 1) *
          ((goldilocksModulus : ℚ) * pairModulus) := by
    rw [fieldLaneCount_eq]
    exact pow_succ ((goldilocksModulus : ℚ) * pairModulus) 31
  rw [field_window_cardinality, pair_window_cardinality, Nat.cast_pow, Nat.cast_pow]
  calc
    _ ≤ (fieldLaneCount : ℚ) * ((pairModulus - 1 : Nat) : ℚ) *
        ((goldilocksModulus : ℚ) * pairModulus) ^ (fieldLaneCount - 1) /
          ((goldilocksModulus : ℚ) ^ fieldLaneCount * (pairModulus : ℚ) ^ fieldLaneCount) :=
      div_le_div_of_nonneg_right counted
        (le_of_lt (mul_pos (pow_pos fieldPositive _) (pow_pos pairPositive _)))
    _ = (fieldLaneCount : ℚ) * ((pairModulus - 1 : Nat) : ℚ) /
        ((goldilocksModulus : ℚ) * pairModulus) := by
      rw [← mul_pow, denominator,
        mul_comm (((goldilocksModulus : ℚ) * pairModulus) ^ (fieldLaneCount - 1))
          ((goldilocksModulus : ℚ) * pairModulus),
        mul_div_mul_right _ _ (pow_ne_zero _ nonzero)]
    _ = (fieldLaneCount : ℚ) * pairDeviation := by
      unfold pairDeviation
      rw [Nat.cast_sub (by decide : 1 ≤ pairModulus), Nat.cast_one,
        mul_comm (goldilocksModulus : ℚ) (pairModulus : ℚ), mul_div_assoc]

private theorem coupled_windows_equal (input : JointWindow)
    (same : ¬ CoupledDisagreement input) :
    fieldCandidates input.1 = pairWindowEquiv (windowCoupling input).2 := by
  apply congrArg pairWindowEquiv
  funext lane
  change low32 (input.1 lane) = (laneCoupling (input.1 lane, input.2 lane)).2
  by_contra different
  exact same ⟨lane, different⟩

noncomputable def fieldDecodedFrequency {Output : Type*}
    (decode : ShortfallBound.Window → Output) (event : Output → Prop) : ℚ :=
  (Nat.card {fields : FieldWindow // event (decode (fieldCandidates fields))} : ℚ) /
    (goldilocksModulus : ℚ) ^ fieldLaneCount

noncomputable def bitDecodedFrequency {Output : Type*}
    (decode : ShortfallBound.Window → Output) (event : Output → Prop) : ℚ :=
  (Nat.card {window : ShortfallBound.Window // event (decode window)} : ℚ) /
    (chunkModulus : ℚ) ^ candidateBound

/-- Every forward decoder preserves the comparison bound. `Output` may
contain abort, and the event may include or be exactly that abort value. -/
theorem decoded_event_frequency_error_le {Output : Type*}
    (decode : ShortfallBound.Window → Output) (event : Output → Prop) :
    |fieldDecodedFrequency decode event - bitDecodedFrequency decode event| ≤
      32 * pairDeviation := by
  have leftPositive : (0 : ℚ) < Nat.card FieldWindow := by
    rw [field_window_cardinality, Nat.cast_pow]
    exact pow_pos (by norm_num [goldilocksModulus]) _
  have rightPositive : (0 : ℚ) < Nat.card PairWindow := by
    rw [pair_window_cardinality, Nat.cast_pow]
    exact pow_pos (by norm_num [pairModulus, chunkModulus]) _
  have compared := coupled_event_error windowCoupling
    (fun fields => event (decode (fieldCandidates fields)))
    (fun pairs => event (decode (pairWindowEquiv pairs)))
    CoupledDisagreement leftPositive rightPositive (by
      intro input same
      exact iff_of_eq (congrArg (fun window => event (decode window))
        (coupled_windows_equal input same)))
  have rightCount :
      Nat.card {pairs : PairWindow // event (decode (pairWindowEquiv pairs))} =
        Nat.card {window : ShortfallBound.Window // event (decode window)} :=
    Nat.card_congr (Equiv.subtypeEquivOfSubtype
      (p := fun window : ShortfallBound.Window => event (decode window)) pairWindowEquiv)
  have rightCard : Nat.card PairWindow = chunkModulus ^ candidateBound :=
    (Nat.card_congr pairWindowEquiv).trans ShortfallBound.window_cardinality
  have bound := compared.trans disagreement_ratio_le
  simpa only [fieldDecodedFrequency, bitDecodedFrequency, field_window_cardinality,
    rightCount, rightCard, Nat.cast_pow, fieldLaneCount_eq] using bound

/-- The actual bounded coefficient decoder is compared with abort retained. -/
theorem boundedSample_event_frequency_error_le (event : Option (List Coefficient) → Prop) :
    let decode := fun window : ShortfallBound.Window =>
      FirstAccepted.boundedSample verifier coefficientCount (List.ofFn window)
    |fieldDecodedFrequency decode event - bitDecodedFrequency decode event| ≤
      32 * pairDeviation := by
  exact decoded_event_frequency_error_le _ event

end NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldOutputLaw
