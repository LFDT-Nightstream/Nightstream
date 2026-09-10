import Mathlib.Data.Finite.Sigma
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldShortfall
import NightstreamFPrime.Spec.Folding.Nifs.PaperProfile

/-!
Owns the finite union bound for the selected 17-scalar comparison batch.
Every scalar abort is a batch failure. The comparison gives equal mass to
all 544 field-coordinate functions and assigns no law to Poseidon2.
-/

namespace NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldBatchShortfall

open scoped BigOperators
open ProductionAlphabet Sampling FieldShortfall

def batchCount : Nat := PaperProfile.arity.total

theorem batchCount_eq : batchCount = 17 := PaperProfile.arity_total

abbrev FieldBatch := Fin batchCount → FieldWindow

def ScalarShortfall (fields : FieldWindow) : Prop :=
  FirstAccepted.Shortfall verifier coefficientCount (List.ofFn (fieldCandidates fields))

def BatchShortfall (batch : FieldBatch) : Prop :=
  ∃ index : Fin batchCount, ScalarShortfall (batch index)

private theorem exists_event_card_le {Source Index : Type*} [Finite Source] [Fintype Index]
    (event : Index → Source → Prop) :
    Nat.card {source : Source // ∃ index, event index source} ≤
      ∑ index : Index, Nat.card {source : Source // event index source} := by
  let selected : {source : Source // ∃ index, event index source} →
      (Σ index : Index, {source : Source // event index source}) := fun source =>
    ⟨Classical.choose source.property,
      ⟨source.val, Classical.choose_spec source.property⟩⟩
  have injective : Function.Injective selected := by
    intro first second same
    apply Subtype.ext
    exact congrArg
      (fun output : (Σ index : Index, {source : Source // event index source}) =>
        output.2.val) same
  simpa only [Nat.card_sigma] using Nat.card_le_card_of_injective selected injective

private theorem coordinate_event_card {Sample : Type*} [Finite Sample]
    (count : Nat) (event : Sample → Prop) (index : Fin (count + 1)) :
    Nat.card {batch : Fin (count + 1) → Sample // event (batch index)} =
      Nat.card {sample : Sample // event sample} * Nat.card Sample ^ count := by
  let split : {batch : Fin (count + 1) → Sample // event (batch index)} ≃
      {pair : Sample × ({other : Fin (count + 1) // other ≠ index} → Sample) //
        event pair.1} :=
    Equiv.subtypeEquiv
      (p := fun batch : Fin (count + 1) → Sample => event (batch index))
      (q := fun pair : Sample × ({other : Fin (count + 1) // other ≠ index} → Sample) =>
        event pair.1)
      (Equiv.funSplitAt index Sample) (fun _ => Iff.rfl)
  have remainingCard : Nat.card {other : Fin (count + 1) // other ≠ index} = count :=
    (Nat.card_congr (finSuccAboveEquiv index)).symm.trans (Nat.card_fin count)
  calc
    _ = Nat.card ({sample : Sample // event sample} ×
        ({other : Fin (count + 1) // other ≠ index} → Sample)) :=
      Nat.card_congr (split.trans (Equiv.prodSubtypeFstEquivSubtypeProd (p := event)))
    _ = Nat.card {sample : Sample // event sample} * Nat.card Sample ^ count := by
      rw [Nat.card_prod, Nat.card_fun, remainingCard]

/-- A finite union bound for repeated uniform samples, with one event per coordinate. -/
theorem finite_batch_event_card_le {Sample : Type*} [Finite Sample]
    (count : Nat) (event : Sample → Prop) :
    Nat.card {batch : Fin (count + 1) → Sample // ∃ index, event (batch index)} ≤
      (count + 1) * Nat.card {sample : Sample // event sample} * Nat.card Sample ^ count := by
  have covered := exists_event_card_le
    (fun index : Fin (count + 1) => fun batch : Fin (count + 1) → Sample => event (batch index))
  calc
    _ ≤ ∑ index : Fin (count + 1),
        Nat.card {batch : Fin (count + 1) → Sample // event (batch index)} := covered
    _ = ∑ _index : Fin (count + 1),
        Nat.card {sample : Sample // event sample} * Nat.card Sample ^ count := by
      apply Finset.sum_congr rfl
      intro index _member
      exact coordinate_event_card count event index
    _ = (count + 1) * Nat.card {sample : Sample // event sample} * Nat.card Sample ^ count := by
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
        Nat.nsmul_eq_mul, Nat.mul_assoc]

/-- The comparison batch has the product cardinality of the selected field windows. -/
theorem field_batch_cardinality :
    Nat.card FieldBatch = (goldilocksModulus ^ fieldLaneCount) ^ batchCount := by
  rw [Nat.card_fun, field_window_cardinality, Nat.card_fin]

/-- Count all failing batches by a failed coordinate and its sixteen free windows. -/
theorem batch_shortfall_card_le :
    Nat.card {batch : FieldBatch // BatchShortfall batch} ≤
      batchCount * Nat.card {fields : FieldWindow // ScalarShortfall fields} *
        (goldilocksModulus ^ fieldLaneCount) ^ (batchCount - 1) := by
  have count := finite_batch_event_card_le 16 ScalarShortfall
  rw [field_window_cardinality] at count
  exact count

/-- Failure frequency in the explicit uniform field-batch comparison, including abort. -/
noncomputable def iidFieldBatchShortfallProbability : ℚ :=
  (Nat.card {batch : FieldBatch // BatchShortfall batch} : ℚ) /
    ((goldilocksModulus : ℚ) ^ fieldLaneCount) ^ batchCount

/-- The finite union bound needs no exact product identity for successful batches. -/
theorem iid_field_batch_shortfall_probability_le_scalar :
    iidFieldBatchShortfallProbability ≤
      (batchCount : ℚ) * iidFieldShortfallProbability := by
  let choices : ℚ := (goldilocksModulus : ℚ) ^ fieldLaneCount
  let failures := Nat.card {fields : FieldWindow // ScalarShortfall fields}
  have counted : (Nat.card {batch : FieldBatch // BatchShortfall batch} : ℚ) ≤
      (batchCount : ℚ) * (failures : ℚ) * choices ^ (batchCount - 1) := by
    dsimp only [choices, failures]
    exact_mod_cast batch_shortfall_card_le
  have choicesPositive : 0 < choices := pow_pos (by norm_num [goldilocksModulus]) _
  have nonzero : choices ≠ 0 := ne_of_gt choicesPositive
  have denominator : choices ^ batchCount = choices ^ (batchCount - 1) * choices :=
    pow_succ _ _
  change (Nat.card {batch : FieldBatch // BatchShortfall batch} : ℚ) /
    choices ^ batchCount ≤ (batchCount : ℚ) * ((failures : ℚ) / choices)
  calc
    _ ≤ (batchCount : ℚ) * (failures : ℚ) * choices ^ (batchCount - 1) /
        choices ^ batchCount :=
      div_le_div_of_nonneg_right counted (le_of_lt (pow_pos choicesPositive _))
    _ = (batchCount : ℚ) * ((failures : ℚ) / choices) := by
      rw [denominator, mul_comm (choices ^ (batchCount - 1)) choices,
        mul_div_mul_right _ _ (pow_ne_zero _ nonzero), mul_div_assoc]

/-- The selected arity is seventeen; each scalar keeps the checked field-pair law. -/
theorem iid_field_batch_shortfall_probability_le :
    iidFieldBatchShortfallProbability ≤
      17 * (Nat.choose candidateBound 11 : ℚ) / (chunkModulus : ℚ) ^ 11 := by
  have scaled := mul_le_mul_of_nonneg_left
    iid_field_shortfall_probability_le (Nat.cast_nonneg batchCount : (0 : ℚ) ≤ batchCount)
  have bounded := iid_field_batch_shortfall_probability_le_scalar.trans scaled
  simpa only [batchCount_eq, mul_div_assoc] using bounded

end NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldBatchShortfall
