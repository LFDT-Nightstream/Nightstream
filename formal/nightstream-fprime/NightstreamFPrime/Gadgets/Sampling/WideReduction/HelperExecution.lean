import NightstreamFPrime.Gadgets.Sampling.WideReduction.HelperExpressions
import NightstreamFPrime.Gadgets.Sampling.WideReduction.HintSupport
import NightstreamFPrime.Circuit.Execution

/-! The sequential helper program constructs the integer reference values
from the five input fields, using only the existing hint meanings. -/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction.HelperExecution

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler

def Certified (target : Env) (column : Nat) (hint : Hint) : Prop :=
  hint.source.VarsBelow column ∧ hint.eval target = target column

theorem source_bits (interface : Interface) (base : Env) (start : Nat)
    (inputs : Assumptions interface start) :
    HintsEach (Certified (HelperValues.environment base start (drawOf interface base start))) start
      (HintProgram.sourceBitHints interface start) := by
  unfold HintProgram.sourceBitHints
  refine HintsEach.finRange_flatMap (stride := 64) (fun lane => by simp) fun lane =>
    HintsEach.range_map fun bit below => ⟨?_, ?_⟩
  · exact Expr.VarsBelow.mono _ (inputs lane) (by omega)
  · rw [HelperValues.source_bit base start _ lane bit below]
    apply HintValues.bit_hint _ _ _ bit (drawOf interface base start lane).isLt
    rw [fieldOfNat_val]
    exact Expr.eval_eq_of_agree_below _ start _ base (inputs lane)
      (HelperValues.below base start _)

theorem limb (base : Env) (start : Nat) (draw : Draw) (position : Nat)
    (bound : position < 20) :
    HintsEach (Certified (HelperValues.environment base start draw))
      (HintProgram.limbStart start position) (HintProgram.limbHints start position) := by
  unfold HintProgram.limbHints
  refine ⟨⟨⟨trivial, HintSupport.limbTerms_below start position⟩, ?_⟩,
    HintsEach.range_map fun bit below => ⟨?_, ?_⟩⟩
  · rw [HelperValues.accumulator base start draw position bound]
    have meaning : (Expr.const (fieldOfNat 5) * linearExpr (HintProgram.limbTerms start position)).eval
        (HelperValues.environment base start draw) =
        fieldOfNat (5 * LimbArithmetic.accumulator draw position) := by
      rw [Expr.eval_hmul, Expr.eval_const, linearExpr_eval, HelperExpressions.limb_terms _ _ _ _ bound,
        fieldOfNat_mul]
    rw [WitnessArithmetic.quotient_hint _ _ _
      (LimbArithmetic.accumulator_times_five_lt_field draw position) meaning]
    rw [Nat.mul_div_cancel_left _ (by decide : 0 < 5)]
  · change HintProgram.limbStart start position < HintProgram.limbStart start position + 1 + bit
    omega
  · rw [HelperValues.accumulator_bit base start draw position bit bound below]
    apply HintValues.bit_hint _ _ _ bit
      (lt_trans (LimbArithmetic.accumulator_bound draw position) (by decide))
    exact HelperValues.accumulator base start draw position bound

theorem division_pair (base : Env) (start : Nat) (draw : Draw) (round position : Nat)
    (roundBound : round < 64) (positionBound : position < 6) :
    HintsEach (Certified (HelperValues.environment base start draw))
      (HintProgram.divisionColumn start round position)
      [.quotientFive (HintProgram.divisionInput start round position),
        .remainderFive (HintProgram.divisionInput start round position)] := by
  have source := HintSupport.divisionInput_below start round position
  have meaning := HelperExpressions.division_input base start draw round position roundBound positionBound
  have bound := HintValues.division_source_bound ((drawIndex draw).val / 5 ^ round) (5 - position)
  have step := HintValues.division_step ((drawIndex draw).val / 5 ^ round) (5 - position)
  refine ⟨⟨source, ?_⟩, ⟨Expr.VarsBelow.mono _ source (by omega), ?_⟩, trivial⟩
  · rw [(HelperValues.division base start draw round position roundBound positionBound).1,
      WitnessArithmetic.quotient_hint _ _ _ bound meaning, step.1]
    unfold HelperValues.quotient
    rw [pow_succ, ← Nat.div_div_eq_div_mul]
  · rw [(HelperValues.division base start draw round position roundBound positionBound).2,
      WitnessArithmetic.remainder_hint _ _ _ bound meaning, step.2]
    rfl

theorem division_round (base : Env) (start : Nat) (draw : Draw) (round : Nat)
    (bound : round < 64) :
    HintsEach (Certified (HelperValues.environment base start draw))
      (HintProgram.divisionStart start + round * HintProgram.divisionStride)
      (HintProgram.divisionHints start round) := by
  unfold HintProgram.divisionHints
  refine HintsEach.range_flatMap (stride := 2) (fun position _ => rfl) fun position below => ?_
  have atPosition := division_pair base start draw round position bound below
  simpa only [HintProgram.divisionColumn, Nat.mul_comm 2 position] using atPosition

private theorem limb_batches_length (start : Nat) :
    ((List.range HintProgram.limbCount).flatMap (HintProgram.limbHints start)).length =
      HintProgram.limbCount * HintProgram.limbStride := by
  simp only [List.length_flatMap, HintProgram.limbHints_length, List.map_const',
    List.length_range, List.sum_replicate, smul_eq_mul]

theorem certificate (interface : Interface) (base : Env) (start : Nat)
    (inputs : Assumptions interface start) :
    HintsCausal start (HintProgram.helpers interface start) ∧
      HintsComputed (HelperValues.environment base start (drawOf interface base start)) start
        (HintProgram.helpers interface start) := by
  let target := HelperValues.environment base start (drawOf interface base start)
  have source := source_bits interface base start inputs
  have limbs : HintsEach (Certified target) (start + HintProgram.sourceBitCount)
      ((List.range HintProgram.limbCount).flatMap (HintProgram.limbHints start)) :=
    HintsEach.range_flatMap (fun position _ => HintProgram.limbHints_length start position)
      fun position below => limb base start (drawOf interface base start) position below
  have divisions : HintsEach (Certified target) (HintProgram.divisionStart start)
      ((List.range digitCount).flatMap (HintProgram.divisionHints start)) :=
    HintsEach.range_flatMap (fun round _ => HintProgram.divisionHints_length start round)
      fun round below => division_round base start (drawOf interface base start) round below
  have first := source.append (by rw [HintProgram.sourceBitHints_length]; exact limbs)
  have full := first.append (by
    rw [List.length_append, HintProgram.sourceBitHints_length, limb_batches_length]
    simpa only [HintProgram.divisionStart, Nat.add_assoc] using divisions)
  exact ⟨full.imp fun _ _ certified => certified.1, full.imp fun _ _ certified => certified.2.symm⟩

theorem present (interface : Interface) (env : Env) (start : Nat)
    (inputs : Assumptions interface start)
    (computed : HintsComputed env start (HintProgram.helpers interface start)) :
    HelperValues.Present env start (drawOf interface env start) := by
  have certified := certificate interface env start inputs
  intro index _ upper
  exact hintsComputed_unique certified.1 computed certified.2
    (fun earlier below => (HelperValues.below env start _ earlier below).symm) index
    (by rwa [HintProgram.helpers_length])

end NightstreamFPrime.Gadgets.Sampling.WideReduction.HelperExecution
