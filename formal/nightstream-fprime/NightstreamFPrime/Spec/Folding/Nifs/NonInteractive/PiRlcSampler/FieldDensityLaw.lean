import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.BitOutputLaw

/-!
Owns successful-event density for the existing 54-of-64 decoder on 32
independent uniform Goldilocks fields. Quotient/remainder coordinates give
a finite injection; the bit decoder supplies the exact successful law.
Aborts are retained outside the event. No law is assigned to Poseidon2.
-/

namespace NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldDensityLaw

open ProductionAlphabet Sampling FieldPairLaw FieldShortfall FieldOutputLaw

private def fieldQuotient (value : F) : Fin pairModulus :=
  ⟨value.val / pairModulus, by
    have positive : 0 < pairModulus := by decide
    have gridBound : goldilocksModulus ≤ pairModulus * pairModulus := by
      rw [goldilocks_decomposition]
      have predecessor : pairModulus - 1 + 1 = pairModulus :=
        Nat.sub_add_cancel (by decide)
      calc
        pairModulus * (pairModulus - 1) + 1 ≤
            pairModulus * (pairModulus - 1) + pairModulus :=
          Nat.add_le_add_left (by decide) _
        _ = pairModulus * (pairModulus - 1 + 1) := by
          rw [Nat.mul_add, Nat.mul_one]
        _ = pairModulus * pairModulus := by rw [predecessor]
    exact (Nat.div_lt_iff_lt_mul positive).mpr
      (value.isLt.trans_le gridBound)⟩

private theorem field_coordinates_injective :
    Function.Injective (fun value : F => (low32 value, fieldQuotient value)) := by
  intro left right same
  have residues := congrArg (fun pair : Fin pairModulus × Fin pairModulus => pair.1.val) same
  have quotients := congrArg (fun pair : Fin pairModulus × Fin pairModulus => pair.2.val) same
  change left.val % pairModulus = right.val % pairModulus at residues
  change left.val / pairModulus = right.val / pairModulus at quotients
  apply Fin.ext
  calc
    left.val = left.val % pairModulus + pairModulus * (left.val / pairModulus) :=
      (Nat.mod_add_div left.val pairModulus).symm
    _ = right.val % pairModulus + pairModulus * (right.val / pairModulus) := by
      rw [residues, quotients]
    _ = right.val := Nat.mod_add_div right.val pairModulus

private theorem field_event_card_le (event : ShortfallBound.Window → Prop) :
    Nat.card {fields : FieldWindow // event (fieldCandidates fields)} ≤
      Nat.card {window : ShortfallBound.Window // event window} * Nat.card PairWindow := by
  let encode : {fields : FieldWindow // event (fieldCandidates fields)} →
      {window : ShortfallBound.Window // event window} × PairWindow := fun fields =>
    (⟨fieldCandidates fields.val, fields.property⟩,
      fun lane => fieldQuotient (fields.val lane))
  have injective : Function.Injective encode := by
    intro first second same
    have windows := congrArg
      (fun result : {window : ShortfallBound.Window // event window} × PairWindow =>
        result.1.val) same
    have quotients := congrArg
      (fun result : {window : ShortfallBound.Window // event window} × PairWindow =>
        result.2) same
    change fieldCandidates first.val = fieldCandidates second.val at windows
    have residues := pairWindowEquiv.injective windows
    apply Subtype.ext
    funext lane
    apply field_coordinates_injective
    exact Prod.ext (congrFun residues lane) (congrFun quotients lane)
  simpa only [Nat.card_prod] using Nat.card_le_card_of_injective encode injective

private theorem density_identity (base modulus count : ℚ) (lanes : Nat)
    (baseNonzero : base ≠ 0) (modulusNonzero : modulus ≠ 0) :
    count * base ^ lanes / modulus ^ lanes =
      (base ^ 2 / modulus) ^ lanes * (count / base ^ lanes) := by
  rw [pow_two, div_pow, mul_pow]
  field_simp [pow_ne_zero lanes baseNonzero, pow_ne_zero lanes modulusNonzero] <;> ring

private theorem decoded_event_density_le {Output : Type*}
    (decode : ShortfallBound.Window → Output) (event : Output → Prop) :
    fieldDecodedFrequency decode event ≤
      ((pairModulus : ℚ) ^ 2 / goldilocksModulus) ^ fieldLaneCount *
        bitDecodedFrequency decode event := by
  have pairCard : Nat.card PairWindow = pairModulus ^ fieldLaneCount := by
    rw [Nat.card_fun, Nat.card_fin, Nat.card_fin]
  have bitCard : chunkModulus ^ candidateBound = pairModulus ^ fieldLaneCount :=
    ((Nat.card_congr pairWindowEquiv).trans ShortfallBound.window_cardinality).symm.trans pairCard
  have bitDenominator : (chunkModulus : ℚ) ^ candidateBound =
      (pairModulus : ℚ) ^ fieldLaneCount := by
    exact_mod_cast bitCard
  have counted := field_event_card_le (fun window => event (decode window))
  rw [pairCard] at counted
  have countedQ :
      (Nat.card {fields : FieldWindow // event (decode (fieldCandidates fields))} : ℚ) ≤
        (Nat.card {window : ShortfallBound.Window // event (decode window)} : ℚ) *
          (pairModulus : ℚ) ^ fieldLaneCount := by
    exact_mod_cast counted
  have modulusPositive : (0 : ℚ) < goldilocksModulus := by norm_num [goldilocksModulus]
  have pairPositive : (0 : ℚ) < pairModulus := by norm_num [pairModulus, chunkModulus]
  unfold fieldDecodedFrequency bitDecodedFrequency
  rw [bitDenominator]
  calc
    _ ≤ (Nat.card {window : ShortfallBound.Window // event (decode window)} : ℚ) *
        (pairModulus : ℚ) ^ fieldLaneCount / (goldilocksModulus : ℚ) ^ fieldLaneCount :=
      div_le_div_of_nonneg_right countedQ (le_of_lt (pow_pos modulusPositive _))
    _ = _ := density_identity _ _ _ _ (ne_of_gt pairPositive) (ne_of_gt modulusPositive)

/-- Every successful scalar event has a multiplicative density bound.
The original Option decoder is unchanged; the event must exclude abort. -/
theorem boundedSample_success_event_le (event : Option (List Coefficient) → Prop)
    (noAbort : ¬ event none) :
    let decode := fun window : ShortfallBound.Window =>
      FirstAccepted.boundedSample verifier coefficientCount (List.ofFn window)
    fieldDecodedFrequency decode event ≤
      ((pairModulus : ℚ) ^ 2 / goldilocksModulus) ^ fieldLaneCount *
        BitOutputLaw.uniformSomeFrequency event := by
  let decode := fun window : ShortfallBound.Window =>
    FirstAccepted.boundedSample verifier coefficientCount (List.ofFn window)
  change fieldDecodedFrequency decode event ≤ _
  have bitBound : bitDecodedFrequency decode event ≤ BitOutputLaw.uniformSomeFrequency event := by
    rw [BitOutputLaw.boundedSample_event_frequency_eq_mixture, if_neg noAbort, add_zero]
    have abortNonnegative : (0 : ℚ) ≤ ShortfallBound.iidBitShortfallProbability := by
      unfold ShortfallBound.iidBitShortfallProbability
      exact div_nonneg (Nat.cast_nonneg _) (le_of_lt (pow_pos (by norm_num [chunkModulus]) _))
    have eventNonnegative : (0 : ℚ) ≤ BitOutputLaw.uniformSomeFrequency event := by
      unfold BitOutputLaw.uniformSomeFrequency
      exact div_nonneg (Nat.cast_nonneg _) (le_of_lt (pow_pos (by norm_num [alphabetSize]) _))
    have productNonnegative := mul_nonneg abortNonnegative eventNonnegative
    nlinarith only [productNonnegative]
  exact (decoded_event_density_le decode event).trans
    (mul_le_mul_of_nonneg_left bitBound
      (pow_nonneg (div_nonneg (sq_nonneg _) (Nat.cast_nonneg _)) _))

end NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldDensityLaw
