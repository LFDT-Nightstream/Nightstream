import NightstreamFPrime.Export.Stage1.HyperNovaHistoryLaw

/-!
Work accounting follows the operational reverse history, including source
aborts and later calls after invalid source returns. Each generated source
result corresponds to one actual kernel invocation. This first bound counts
those invocations. The declared orchestration allowance charges one initial
entry and one processed source return, including an absent result. It is
separate from each NIFS source clock. It does not charge payload decoding,
array copying or advice evaluation, and is not a machine-time bound.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaHistoryWork

open scoped BigOperators
open HyperNovaHistory

/-- Every generated history makes at most its advertised iteration count of
source calls. No accepted-terminal, valid-source, or no-collision premise is
used. In particular, abort calls and all false-mark paths are included. -/
theorem source_calls_le_iteration
    (source : Statement → Payload → PMF SourceResult)
    (statement : Statement) (proof : Envelope) (outcomes : List SourceResult)
    (supported : outcomes ∈ (HyperNovaHistoryLaw.results source statement proof).support) :
    outcomes.length ≤ statement.iteration := by
  generalize count : statement.iteration = remaining at *
  induction remaining using Nat.strong_induction_on generalizing statement proof outcomes with
  | h remaining induction =>
      rw [HyperNovaHistoryLaw.results_eq] at supported
      cases proof with
      | bottom =>
          have empty := (PMF.mem_support_pure_iff [] outcomes).mp supported
          simp only [empty, List.length_nil, Nat.zero_le]
      | recursive payload =>
          by_cases zero : statement.iteration = 0
          · simp only [if_pos zero] at supported
            have empty := (PMF.mem_support_pure_iff [] outcomes).mp supported
            simp only [empty, List.length_nil, Nat.zero_le]
          · simp only [if_neg zero] at supported
            by_cases counter : (decodedInput payload).iteration + 1 = statement.iteration
            · simp only [if_pos counter] at supported
              by_cases base : (decodedInput payload).iteration = 0
              · simp only [if_pos base] at supported
                have empty := (PMF.mem_support_pure_iff [] outcomes).mp supported
                simp only [empty, List.length_nil, Nat.zero_le]
              · simp only [if_neg base] at supported
                rcases (PMF.mem_support_bind_iff _ _ _).mp supported with
                  ⟨result, _produced, tailSupported⟩
                cases result with
                | none =>
                    have singleton := (PMF.mem_support_pure_iff [none] outcomes).mp tailSupported
                    simp only [singleton, List.length_cons, List.length_nil]
                    omega
                | some values =>
                    rcases (PMF.mem_support_map_iff _ _ _).mp tailSupported with
                      ⟨tail, previousSupported, same⟩
                    have previousCount : (predecessorStatement payload).iteration + 1 = remaining := by
                      simpa only [predecessorStatement] using counter.trans count
                    have previous := induction (predecessorStatement payload).iteration
                      (by omega) (predecessorStatement payload)
                      (.recursive (predecessorPayload payload values)) tail previousSupported rfl
                    rw [← same, List.length_cons]
                    omega
            · simp only [if_neg counter] at supported
              have empty := (PMF.mem_support_pure_iff [] outcomes).mp supported
              simp only [empty, List.length_nil, Nat.zero_le]

/-- Declared orchestration allowance: one initial entry and one processed
return per actual source invocation, including an abort return. -/
def controlAllowance (outcomes : List SourceResult) : Nat := outcomes.length + 1

/-- The declared orchestration allowance is integrable and bounded by the
symbolic initial depth plus one. It includes every generated path, without
restricting the experiment to accepted or successful histories. -/
theorem expected_control_allowance_le
    (source : Statement → Payload → PMF SourceResult)
    (initial : PMF (Statement × Envelope)) (depth : Nat)
    (depthBound : ∀ input ∈ initial.support, input.1.iteration ≤ depth) :
    let distribution := HyperNovaHistoryLaw.law source initial
    Summable (fun sample => (distribution sample).toReal * (controlAllowance sample.2.2 : ℝ)) ∧
      (∑' sample, (distribution sample).toReal * (controlAllowance sample.2.2 : ℝ)) ≤
        ((depth + 1 : Nat) : ℝ) := by
  dsimp only
  let distribution := HyperNovaHistoryLaw.law source initial
  have allowanceBound (sample : HyperNovaHistoryProbability.Sample)
      (supported : sample ∈ distribution.support) : controlAllowance sample.2.2 ≤ depth + 1 := by
    change sample ∈ (HyperNovaHistoryLaw.law source initial).support at supported
    rw [HyperNovaHistoryLaw.law] at supported
    rcases (PMF.mem_support_bind_iff _ _ _).mp supported with
      ⟨input, inputSupported, outputSupported⟩
    rcases (PMF.mem_support_map_iff _ _ _).mp outputSupported with
      ⟨outcomes, outcomesSupported, same⟩
    rw [← same]
    exact Nat.add_le_add_right
      ((source_calls_le_iteration source input.1 input.2 outcomes outcomesSupported).trans
        (depthBound input inputSupported)) 1
  have weightedBound (sample : HyperNovaHistoryProbability.Sample) :
      (distribution sample).toReal * (controlAllowance sample.2.2 : ℝ) ≤
        (distribution sample).toReal * ((depth + 1 : Nat) : ℝ) := by
    by_cases zero : distribution sample = 0
    · simp only [zero, ENNReal.toReal_zero, zero_mul, le_refl]
    · exact mul_le_mul_of_nonneg_left
        (Nat.cast_le.mpr (allowanceBound sample ((distribution.mem_support_iff sample).mpr zero)))
        ENNReal.toReal_nonneg
  have weights : Summable (fun sample => (distribution sample).toReal) :=
    ENNReal.summable_toReal distribution.tsum_coe_ne_top
  have ceiling := weights.mul_right ((depth + 1 : Nat) : ℝ)
  have summable : Summable (fun sample =>
      (distribution sample).toReal * (controlAllowance sample.2.2 : ℝ)) :=
    Summable.of_nonneg_of_le (fun _ => mul_nonneg ENNReal.toReal_nonneg (Nat.cast_nonneg _))
      weightedBound ceiling
  refine ⟨summable, ?_⟩
  calc
    _ ≤ ∑' sample, (distribution sample).toReal * ((depth + 1 : Nat) : ℝ) :=
      summable.tsum_le_tsum weightedBound ceiling
    _ = _ := by
      rw [tsum_mul_right, ← ENNReal.tsum_toReal_eq distribution.apply_ne_top,
        distribution.tsum_coe, ENNReal.toReal_one, one_mul]

end NightstreamFPrime.Export.Stage1.HyperNovaHistoryWork
