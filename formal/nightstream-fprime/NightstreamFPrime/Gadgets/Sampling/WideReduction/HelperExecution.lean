import NightstreamFPrime.Gadgets.Sampling.WideReduction.HelperExpressions
import NightstreamFPrime.Gadgets.Sampling.WideReduction.HintSupport
import NightstreamFPrime.Gadgets.Sampling.WideReduction.HintCertificate

/-! The sequential helper program constructs the integer reference values
from the four input fields, using only the existing hint meanings. -/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction.HelperExecution

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler
open HintCertificate

theorem source_bits (interface : Interface) (base : Env) (start : Nat)
    (inputs : Assumptions interface start) :
    Valid (HelperValues.environment base start (drawOf interface base start)) start
      (HintProgram.sourceBitHints interface start) := by
  unfold HintProgram.sourceBitHints
  apply finRange_flatMap _ start fieldCount 64
  · intro lane
    simp
  · intro lane
    apply range_map
    · intro bit _
      exact Expr.VarsBelow.mono _ (inputs lane) (by omega)
    · intro bit below
      rw [show start + lane.val * 64 + bit = (start + lane.val * 64) + bit by rfl]
      rw [HelperValues.source_bit base start _ lane bit below]
      apply HintValues.bit_hint _ _ _ bit (drawOf interface base start lane).isLt
      rw [fieldOfNat_val]
      exact Expr.eval_eq_of_agree_below _ start _ base (inputs lane)
        (HelperValues.below base start _)

theorem limb (base : Env) (start : Nat) (draw : Draw) (position : Nat)
    (bound : position < 16) :
    Valid (HelperValues.environment base start draw) (HintProgram.limbStart start position)
      (HintProgram.limbHints start position) := by
  unfold HintProgram.limbHints
  refine ⟨⟨trivial, HintSupport.limbTerms_below start position⟩, ?_, ?_⟩
  · rw [HelperValues.accumulator base start draw position bound]
    have meaning : (Expr.const (fieldOfNat 5) * linearExpr (HintProgram.limbTerms start position)).eval
        (HelperValues.environment base start draw) =
        fieldOfNat (5 * LimbArithmetic.accumulator draw position) := by
      rw [Expr.eval_hmul, Expr.eval_const, linearExpr_eval, HelperExpressions.limb_terms _ _ _ _ bound,
        fieldOfNat_mul]
    rw [WitnessArithmetic.quotient_hint _ _ _
      (LimbArithmetic.accumulator_times_five_lt_field draw position) meaning]
    rw [Nat.mul_div_cancel_left _ (by decide : 0 < 5)]
  · apply range_map
    · intro bit _
      change HintProgram.limbStart start position < HintProgram.limbStart start position + 1 + bit
      omega
    · intro bit below
      rw [HelperValues.accumulator_bit base start draw position bit bound below]
      apply HintValues.bit_hint _ _ _ bit
        (lt_trans (LimbArithmetic.accumulator_bound draw position) (by decide))
      exact HelperValues.accumulator base start draw position bound

theorem division_pair (base : Env) (start : Nat) (draw : Draw) (round position : Nat)
    (roundBound : round < 54) (positionBound : position < 5) :
    Valid (HelperValues.environment base start draw) (HintProgram.divisionColumn start round position)
      [.quotientFive (HintProgram.divisionInput start round position),
        .remainderFive (HintProgram.divisionInput start round position)] := by
  have source := HintSupport.divisionInput_below start round position
  have meaning := HelperExpressions.division_input base start draw round position roundBound positionBound
  have bound := HintValues.division_source_bound ((drawIndex draw).val / 5 ^ round) (4 - position)
  have step := HintValues.division_step ((drawIndex draw).val / 5 ^ round) (4 - position)
  refine ⟨source, ?_, Expr.VarsBelow.mono _ source (by omega), ?_, trivial⟩
  · rw [(HelperValues.division base start draw round position roundBound positionBound).1,
      WitnessArithmetic.quotient_hint _ _ _ bound meaning, step.1]
    unfold HelperValues.quotient
    rw [pow_succ, ← Nat.div_div_eq_div_mul]
  · rw [(HelperValues.division base start draw round position roundBound positionBound).2,
      WitnessArithmetic.remainder_hint _ _ _ bound meaning, step.2]
    rfl

theorem division_round (base : Env) (start : Nat) (draw : Draw) (round : Nat)
    (bound : round < 54) :
    Valid (HelperValues.environment base start draw)
      (HintProgram.divisionStart start + round * HintProgram.divisionStride)
      (HintProgram.divisionHints start round) := by
  unfold HintProgram.divisionHints
  apply range_flatMap _ _ HintProgram.divisionLimbs 2
  · intro position _
    rfl
  · intro position below
    have atPosition := division_pair base start draw round position bound below
    simpa only [HintProgram.divisionColumn, Nat.mul_comm 2 position] using atPosition

private theorem limb_batches_length (start : Nat) :
    ((List.range HintProgram.limbCount).flatMap (HintProgram.limbHints start)).length =
      HintProgram.limbCount * HintProgram.limbStride := by
  simp only [List.length_flatMap, HintProgram.limbHints_length, List.map_const',
    List.length_range, List.sum_replicate, smul_eq_mul]

theorem certificate (interface : Interface) (base : Env) (start : Nat)
    (inputs : Assumptions interface start) :
    Valid (HelperValues.environment base start (drawOf interface base start)) start
      (HintProgram.helpers interface start) := by
  let target := HelperValues.environment base start (drawOf interface base start)
  have source := source_bits interface base start inputs
  have limbs : Valid target (start + HintProgram.sourceBitCount)
      ((List.range HintProgram.limbCount).flatMap (HintProgram.limbHints start)) := by
    apply range_flatMap _ _ _ HintProgram.limbStride
    · intro position _
      exact HintProgram.limbHints_length start position
    · intro position below
      exact limb base start (drawOf interface base start) position below
  have divisions : Valid target (HintProgram.divisionStart start)
      ((List.range digitCount).flatMap (HintProgram.divisionHints start)) := by
    apply range_flatMap _ _ _ HintProgram.divisionStride
    · intro round _
      exact HintProgram.divisionHints_length start round
    · intro round below
      exact division_round base start (drawOf interface base start) round below
  have first := append target start (HintProgram.sourceBitHints interface start)
    ((List.range HintProgram.limbCount).flatMap (HintProgram.limbHints start)) source
    (by rw [HintProgram.sourceBitHints_length]; exact limbs)
  have full := append target start _
    ((List.range digitCount).flatMap (HintProgram.divisionHints start)) first (by
      rw [List.length_append, HintProgram.sourceBitHints_length, limb_batches_length]
      simpa only [HintProgram.divisionStart, Nat.add_assoc] using divisions)
  simpa only [HintProgram.helpers, List.append_assoc] using full

/-- Exact execution of all 1,404 helper hints, for arbitrary four input fields. -/
theorem execute_reference (interface : Interface) (base : Env) (start : Nat)
    (inputs : Assumptions interface start) :
    HintProgram.helperEnv interface base start =
      HelperValues.environment base start (drawOf interface base start) := by
  have valid := certificate interface base start inputs
  have assigned := HintCertificate.execute _ base start _ valid (by
    intro index below
    exact (HelperValues.below base start _ index below).symm)
  funext index
  by_cases inside : index < start + HintProgram.helperCount
  · exact assigned index (by rwa [HintProgram.helpers_length])
  · change executeHints base start (HintProgram.helpers interface start) index = _
    rw [executeHints_agrees_above _ _ _ index (by rw [HintProgram.helpers_length]; omega)]
    exact (HelperValues.outside base start _ index (Or.inr (by omega))).symm

end NightstreamFPrime.Gadgets.Sampling.WideReduction.HelperExecution
