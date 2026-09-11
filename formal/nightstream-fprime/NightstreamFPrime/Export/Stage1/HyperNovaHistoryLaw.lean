import NightstreamFPrime.Export.Stage1.HyperNovaHistoryProbability

/-!
Owns the adaptive law of the source returns consumed by the selected reverse
history. Each call uses the exact current statement and decoded payload.
The walk stops at the base or an abort; its natural counter supplies the
recursion bound. The source-failure bound sums actual visited failure mass,
without an independent-call or conditional-success assumption.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaHistoryLaw

open scoped ENNReal
open HyperNovaHistory

attribute [local instance] Classical.propDecidable

variable (source : Statement → Payload → PMF SourceResult)

private noncomputable def resultsAt : Nat → Statement → Envelope → PMF (List SourceResult)
  | 0, _, _ => PMF.pure []
  | remaining + 1, statement, proof =>
      match proof with
      | .bottom => PMF.pure []
      | .recursive payload =>
          if statement.iteration = 0 then PMF.pure []
          else if (decodedInput payload).iteration + 1 = statement.iteration then
            if (decodedInput payload).iteration = 0 then PMF.pure []
            else (source statement payload).bind fun result =>
              match result with
              | none => PMF.pure [none]
              | some values =>
                  (resultsAt remaining (predecessorStatement payload)
                    (.recursive (predecessorPayload payload values))).map (some values :: ·)
          else PMF.pure []

/-- Draw source results in the order read by the actual reverse program.
The recursion counter is exactly the advertised iteration, not a separate
caller-supplied depth limit. The base makes no source call; aborts are retained. -/
noncomputable def results (statement : Statement) (proof : Envelope) : PMF (List SourceResult) :=
  resultsAt source statement.iteration statement proof

private noncomputable def sourceFailureBoundAt : Nat → Statement → Envelope → ℝ≥0∞
  | 0, _, _ => 0
  | remaining + 1, statement, proof =>
      match proof with
      | .bottom => 0
      | .recursive payload =>
          if statement.iteration = 0 then 0
          else if (decodedInput payload).iteration + 1 = statement.iteration then
            if (decodedInput payload).iteration = 0 then 0
            else
              (source statement payload).toOuterMeasure {result | ¬ SourceSucceeded payload result} +
                ∑' result, source statement payload result *
                  match result with
                  | none => 0
                  | some values => sourceFailureBoundAt remaining (predecessorStatement payload)
                      (.recursive (predecessorPayload payload values))
          else 0

/-- Sum the actual visited source-failure masses. Future calls are weighted
by the preceding return, and unvisited calls contribute zero. The private
structural counter is set to the statement's exact advertised iteration. -/
noncomputable def sourceFailureBound (statement : Statement) (proof : Envelope) : ℝ≥0∞ :=
  sourceFailureBoundAt source statement.iteration statement proof

private theorem event_le_one {Sample : Type*} (law : PMF Sample) (event : Set Sample) :
    law.toOuterMeasure event ≤ 1 := by
  rw [PMF.toOuterMeasure_apply, ← law.tsum_coe]
  exact ENNReal.tsum_le_tsum (fun _ => Set.indicator_apply_le (fun _ => le_rfl))

private theorem and_failure_le {Sample : Type*} (law : PMF Sample)
    (first : Prop) (later : Sample → Prop) :
    law.toOuterMeasure {sample | ¬ (first ∧ later sample)} ≤
      (if first then 0 else 1) + law.toOuterMeasure {sample | ¬ later sample} := by
  by_cases good : first
  · simp only [good, true_and, if_true, zero_add, le_refl]
  · simp only [good, false_and, not_false_eq_true, if_false]
    exact (event_le_one law _).trans (le_add_of_nonneg_right zero_le)

private theorem successful_cons (statement : Statement) (payload : Payload)
    (result : SourceResult) (tail : List SourceResult)
    (nonzero : statement.iteration ≠ 0)
    (counter : (decodedInput payload).iteration + 1 = statement.iteration)
    (positive : (decodedInput payload).iteration ≠ 0) :
    SuccessfulSources statement (.recursive payload) (result :: tail) ↔
      SourceSucceeded payload result ∧
        match result with
        | none => True
        | some values => SuccessfulSources (predecessorStatement payload)
            (.recursive (predecessorPayload payload values)) tail := by
  rw [SuccessfulSources.eq_def]
  simp only [if_neg nonzero, if_pos counter, if_neg positive]
  exact Iff.rfl

private theorem source_failure_probability_le_at (remaining : Nat)
    (statement : Statement) (proof : Envelope) (fits : statement.iteration ≤ remaining) :
    (resultsAt source remaining statement proof).toOuterMeasure
      {returns | ¬ SuccessfulSources statement proof returns} ≤
        sourceFailureBoundAt source remaining statement proof := by
  induction remaining generalizing statement proof with
  | zero =>
      have zero : statement.iteration = 0 := Nat.eq_zero_of_le_zero fits
      cases proof <;>
        simp only [resultsAt, sourceFailureBoundAt, PMF.toOuterMeasure_pure_apply,
          Set.mem_setOf_eq, SuccessfulSources, if_pos zero, not_true_eq_false, if_false, le_refl]
  | succ remaining induction =>
      cases proof with
      | bottom =>
          simp only [resultsAt, sourceFailureBoundAt, SuccessfulSources, PMF.toOuterMeasure_pure_apply,
            Set.mem_setOf_eq, not_true_eq_false, if_false, le_refl]
      | recursive payload =>
        by_cases nonzero : statement.iteration = 0
        · rw [resultsAt, sourceFailureBoundAt]
          simp only [if_pos nonzero, PMF.toOuterMeasure_pure_apply, Set.mem_setOf_eq,
            SuccessfulSources, if_pos nonzero, not_true_eq_false, if_false, le_refl]
        · by_cases counter : (decodedInput payload).iteration + 1 = statement.iteration
          · by_cases positive : (decodedInput payload).iteration = 0
            · rw [resultsAt, sourceFailureBoundAt]
              simp only [if_neg nonzero, if_pos counter, if_pos positive,
                PMF.toOuterMeasure_pure_apply, Set.mem_setOf_eq, SuccessfulSources,
                if_neg nonzero, if_pos counter, if_pos positive, not_true_eq_false, if_false, le_refl]
            · rw [resultsAt, sourceFailureBoundAt]
              simp only [if_neg nonzero, if_pos counter, if_neg positive]
              rw [PMF.toOuterMeasure_bind_apply]
              conv_rhs => rw [PMF.toOuterMeasure_apply]
              rw [← ENNReal.tsum_add]
              apply ENNReal.tsum_le_tsum
              intro result
              have head : ({result | ¬ SourceSucceeded payload result}.indicator
                  (source statement payload)) result =
                  source statement payload result * (if SourceSucceeded payload result then 0 else 1) := by
                by_cases good : SourceSucceeded payload result <;> simp [good]
              rw [head, ← mul_add]
              apply mul_le_mul_left'
              cases result with
              | none =>
                  simp only [PMF.toOuterMeasure_pure_apply, Set.mem_setOf_eq,
                    successful_cons statement payload none [] nonzero counter positive, and_true, add_zero]
                  by_cases good : SourceSucceeded payload none <;> simp [good]
              | some values =>
                  rw [PMF.toOuterMeasure_map_apply]
                  have event :
                      ((fun tail => some values :: tail) ⁻¹'
                        {returns | ¬ SuccessfulSources statement (.recursive payload) returns}) =
                      {tail | ¬ (SourceSucceeded payload (some values) ∧
                        SuccessfulSources (predecessorStatement payload)
                          (.recursive (predecessorPayload payload values)) tail)} := by
                    ext tail
                    exact not_congr (successful_cons statement payload (some values) tail nonzero counter positive)
                  rw [event]
                  refine (and_failure_le _ _ _).trans (add_le_add le_rfl ?_)
                  apply induction (predecessorStatement payload)
                    (.recursive (predecessorPayload payload values))
                  dsimp only [predecessorStatement]
                  omega
          · rw [resultsAt, sourceFailureBoundAt]
            simp only [if_neg nonzero, if_neg counter, PMF.toOuterMeasure_pure_apply,
              Set.mem_setOf_eq, SuccessfulSources, if_neg nonzero, if_neg counter,
              not_true_eq_false, if_false, le_refl]

/-- The generated walk's source-failure mass is at most the sum of its
visited source-failure masses. The source may depend on the entire current
opening. The exact iteration counter prevents a missing required source
entry, without an independence or source-success premise. -/
theorem source_failure_probability_le (statement : Statement) (proof : Envelope) :
    (results source statement proof).toOuterMeasure
      {returns | ¬ SuccessfulSources statement proof returns} ≤
        sourceFailureBound source statement proof :=
  source_failure_probability_le_at source statement.iteration statement proof le_rfl

/-- Retain the initial terminal opening and all source returns from its
adaptive reverse run. -/
noncomputable def law (initial : PMF (Statement × Envelope)) :
    PMF HyperNovaHistoryProbability.Sample :=
  initial.bind fun input =>
    (results source input.1 input.2).map (fun returns => (input.1, input.2, returns))

/-- Sampling the reverse run preserves the exact initial terminal law. -/
theorem initial_marginal (initial : PMF (Statement × Envelope)) :
    (law source initial).map (fun sample => (sample.1, sample.2.1)) = initial := by
  rw [law, PMF.map_bind]
  have forget (input : Statement × Envelope) :
      ((results source input.1 input.2).map (fun returns => (input.1, input.2, returns))).map
        (fun sample => (sample.1, sample.2.1)) = PMF.pure input := by
    rw [PMF.map_comp]
    exact PMF.map_const (results source input.1 input.2) input
  simp only [forget, PMF.bind_pure]

private theorem source_failure_mass_le (initial : PMF (Statement × Envelope)) :
    (law source initial).toOuterMeasure {sample | HyperNovaHistoryProbability.SourceFailure sample} ≤
      ∑' input, initial input * (if PerApplicationTerminal.Holds Poseidon2HashChainV1Package.application
        Poseidon2HashChainV1Package.fits Poseidon2HashChainV1Setup.productionSetup input.1 input.2
        then sourceFailureBound source input.1 input.2 else 0) := by
  rw [law, PMF.toOuterMeasure_bind_apply]
  apply ENNReal.tsum_le_tsum
  intro input
  apply mul_le_mul_left'
  by_cases accepted : PerApplicationTerminal.Holds Poseidon2HashChainV1Package.application
        Poseidon2HashChainV1Package.fits Poseidon2HashChainV1Setup.productionSetup input.1 input.2
  · rw [if_pos accepted, PMF.toOuterMeasure_map_apply]
    refine (MeasureTheory.measure_mono ?_).trans
      (source_failure_probability_le source input.1 input.2)
    intro returns failed
    exact failed.2
  · simp [accepted, PMF.toOuterMeasure_map_apply, HyperNovaHistoryProbability.SourceFailure,
      HyperNovaHistoryProbability.Accepted]

/-- Under the generated adaptive law, accepted initial mass is at most the
complete returned-history mass plus visited source-failure mass from accepted
initial openings and the
actual encountered state-hash failure mass. This requires no conditioning
of the source law, independent-call assumption, or supplied successful trace. -/
theorem accepted_probability_le (initial : PMF (Statement × Envelope)) :
    initial.toOuterMeasure {input |
      PerApplicationTerminal.Holds Poseidon2HashChainV1Package.application
        Poseidon2HashChainV1Package.fits Poseidon2HashChainV1Setup.productionSetup input.1 input.2} ≤
      (law source initial).toOuterMeasure {sample | HyperNovaHistoryProbability.AdviceReturned sample} +
        (∑' input, initial input * (if PerApplicationTerminal.Holds Poseidon2HashChainV1Package.application
        Poseidon2HashChainV1Package.fits Poseidon2HashChainV1Setup.productionSetup input.1 input.2
          then sourceFailureBound source input.1 input.2 else 0)) +
        (law source initial).toOuterMeasure {sample | HyperNovaHistoryProbability.StateHashFailure sample} := by
  have acceptedMass := congrArg
    (fun distribution : PMF (Statement × Envelope) => distribution.toOuterMeasure
      {input | PerApplicationTerminal.Holds Poseidon2HashChainV1Package.application
        Poseidon2HashChainV1Package.fits Poseidon2HashChainV1Setup.productionSetup input.1 input.2})
    (initial_marginal source initial)
  dsimp only at acceptedMass
  rw [PMF.toOuterMeasure_map_apply] at acceptedMass
  calc
    _ = (law source initial).toOuterMeasure
        {sample | HyperNovaHistoryProbability.Accepted sample} := acceptedMass.symm
    _ ≤ _ := (HyperNovaHistoryProbability.accepted_probability_le (law source initial)).trans
      (add_le_add (add_le_add le_rfl (source_failure_mass_le source initial)) le_rfl)

end NightstreamFPrime.Export.Stage1.HyperNovaHistoryLaw
