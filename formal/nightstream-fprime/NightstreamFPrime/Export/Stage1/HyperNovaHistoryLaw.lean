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

variable (application : Lifecycle.Stage1.Application.Program)
  (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
  (setup : PerApplicationCanonicalPackage.CommitmentSetup application)

variable (source : Statement → Payload application → PMF (SourceResult application))

private noncomputable def resultsAt : Nat → Statement → Envelope application → PMF (List (SourceResult application))
  | 0, _, _ => PMF.pure []
  | remaining + 1, statement, proof =>
      match proof with
      | .bottom => PMF.pure []
      | .recursive payload =>
          if statement.iteration = 0 then PMF.pure []
          else if (decodedInput application fits payload).iteration + 1 = statement.iteration then
            if (decodedInput application fits payload).iteration = 0 then PMF.pure []
            else (source statement payload).bind fun result =>
              match result with
              | none => PMF.pure [none]
              | some values =>
                  (resultsAt remaining (predecessorStatement application fits payload)
                    (.recursive (predecessorPayload application fits payload values))).map (some values :: ·)
          else PMF.pure []

/-- Draw source results in the order read by the actual reverse program.
The recursion counter is exactly the advertised iteration, not a separate
caller-supplied depth limit. The base makes no source call; aborts are retained. -/
noncomputable def results (statement : Statement) (proof : Envelope application) : PMF (List (SourceResult application)) :=
  resultsAt application fits source statement.iteration statement proof

/-- The public recursion equation uses the exact predecessor counter. The
private structural bound adds no call and changes no stopped or abort branch. -/
theorem results_eq (statement : Statement) (proof : Envelope application) :
    results application fits source statement proof =
      match proof with
      | .bottom => PMF.pure []
      | .recursive payload =>
          if statement.iteration = 0 then PMF.pure []
          else if (decodedInput application fits payload).iteration + 1 = statement.iteration then
            if (decodedInput application fits payload).iteration = 0 then PMF.pure []
            else (source statement payload).bind fun result =>
              match result with
              | none => PMF.pure [none]
              | some values =>
                  (results application fits source (predecessorStatement application fits payload)
                    (.recursive (predecessorPayload application fits payload values))).map (some values :: ·)
          else PMF.pure [] := by
  cases count : statement.iteration with
  | zero =>
      cases proof <;> simp only [results, count, resultsAt, if_true]
  | succ remaining =>
      cases proof with
      | bottom => simp only [results, count, resultsAt]
      | recursive payload =>
          simp only [results, count, resultsAt, Nat.succ_ne_zero, if_false]
          by_cases counter : (decodedInput application fits payload).iteration + 1 = remaining + 1
          · have previousCount : (predecessorStatement application fits payload).iteration = remaining := by
              dsimp only [predecessorStatement]
              omega
            simp only [if_pos counter, previousCount]
          · simp only [if_neg counter]

private noncomputable def sourceFailureBoundAt : Nat → Statement → Envelope application → ℝ≥0∞
  | 0, _, _ => 0
  | remaining + 1, statement, proof =>
      match proof with
      | .bottom => 0
      | .recursive payload =>
          if statement.iteration = 0 then 0
          else if (decodedInput application fits payload).iteration + 1 = statement.iteration then
            if (decodedInput application fits payload).iteration = 0 then 0
            else
              (source statement payload).toOuterMeasure {result | ¬ SourceSucceeded application fits setup payload result} +
                ∑' result, source statement payload result *
                  match result with
                  | none => 0
                  | some values => sourceFailureBoundAt remaining (predecessorStatement application fits payload)
                      (.recursive (predecessorPayload application fits payload values))
          else 0

/-- Sum the actual visited source-failure masses. Future calls are weighted
by the preceding return, and unvisited calls contribute zero. The private
structural counter is set to the statement's exact advertised iteration. -/
noncomputable def sourceFailureBound (statement : Statement) (proof : Envelope application) : ℝ≥0∞ :=
  sourceFailureBoundAt application fits setup source statement.iteration statement proof

private theorem event_le_one {Sample : Type*} (distribution : PMF Sample) (event : Set Sample) :
    distribution.toOuterMeasure event ≤ 1 := by
  rw [PMF.toOuterMeasure_apply, ← distribution.tsum_coe]
  exact ENNReal.tsum_le_tsum (fun _ => Set.indicator_apply_le (fun _ => le_rfl))

private theorem and_failure_le {Sample : Type*} (distribution : PMF Sample)
    (first : Prop) (later : Sample → Prop) :
    distribution.toOuterMeasure {sample | ¬ (first ∧ later sample)} ≤
      (if first then 0 else 1) + distribution.toOuterMeasure {sample | ¬ later sample} := by
  by_cases good : first
  · simp only [good, true_and, if_true, zero_add, le_refl]
  · simp only [good, false_and, not_false_eq_true, if_false]
    exact (event_le_one distribution _).trans (le_add_of_nonneg_right zero_le)

private theorem successful_cons (statement : Statement) (payload : Payload application)
    (result : SourceResult application) (tail : List (SourceResult application))
    (nonzero : statement.iteration ≠ 0)
    (counter : (decodedInput application fits payload).iteration + 1 = statement.iteration)
    (positive : (decodedInput application fits payload).iteration ≠ 0) :
    SuccessfulSources application fits setup statement (.recursive payload) (result :: tail) ↔
      SourceSucceeded application fits setup payload result ∧
        match result with
        | none => True
        | some values => SuccessfulSources application fits setup (predecessorStatement application fits payload)
            (.recursive (predecessorPayload application fits payload values)) tail := by
  rw [SuccessfulSources.eq_def]
  simp only [if_neg nonzero, if_pos counter, if_neg positive]
  exact Iff.rfl

private theorem source_failure_probability_le_at (remaining : Nat)
    (statement : Statement) (proof : Envelope application) (bounded : statement.iteration ≤ remaining) :
    (resultsAt application fits source remaining statement proof).toOuterMeasure
      {returns | ¬ SuccessfulSources application fits setup statement proof returns} ≤
        sourceFailureBoundAt application fits setup source remaining statement proof := by
  induction remaining generalizing statement proof with
  | zero =>
      have zero : statement.iteration = 0 := Nat.eq_zero_of_le_zero bounded
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
          simp only [PMF.toOuterMeasure_pure_apply, Set.mem_setOf_eq,
            SuccessfulSources, if_pos nonzero, not_true_eq_false, if_false, le_refl]
        · by_cases counter : (decodedInput application fits payload).iteration + 1 = statement.iteration
          · by_cases positive : (decodedInput application fits payload).iteration = 0
            · rw [resultsAt, sourceFailureBoundAt]
              simp only [
                PMF.toOuterMeasure_pure_apply, Set.mem_setOf_eq, SuccessfulSources,
                if_neg nonzero, if_pos counter, if_pos positive, not_true_eq_false, if_false, le_refl]
            · rw [resultsAt, sourceFailureBoundAt]
              simp only [if_neg nonzero, if_pos counter, if_neg positive]
              rw [PMF.toOuterMeasure_bind_apply]
              conv_rhs => rw [PMF.toOuterMeasure_apply]
              rw [← ENNReal.tsum_add]
              apply ENNReal.tsum_le_tsum
              intro result
              have head : ({result | ¬ SourceSucceeded application fits setup payload result}.indicator
                  (source statement payload)) result =
                  source statement payload result * (if (SourceSucceeded application fits setup) payload result then 0 else 1) := by
                by_cases good : SourceSucceeded application fits setup payload result <;> simp [good]
              rw [head, ← mul_add]
              apply mul_le_mul_right
              cases result with
              | none =>
                  simp only [PMF.toOuterMeasure_pure_apply, Set.mem_setOf_eq,
                    successful_cons application fits setup statement payload none [] nonzero counter positive, and_true, add_zero]
                  by_cases good : SourceSucceeded application fits setup payload none <;> simp [good]
              | some values =>
                  rw [PMF.toOuterMeasure_map_apply]
                  have event :
                      ((fun tail => some values :: tail) ⁻¹'
                        {returns | ¬ SuccessfulSources application fits setup statement (.recursive payload) returns}) =
                      {tail | ¬ (SourceSucceeded application fits setup payload (some values) ∧
                        SuccessfulSources application fits setup (predecessorStatement application fits payload)
                          (.recursive (predecessorPayload application fits payload values)) tail)} := by
                    ext tail
                    exact not_congr (successful_cons application fits setup statement payload (some values) tail nonzero counter positive)
                  rw [event]
                  refine (and_failure_le _ _ _).trans (add_le_add le_rfl ?_)
                  apply induction (predecessorStatement application fits payload)
                    (.recursive (predecessorPayload application fits payload values))
                  dsimp only [predecessorStatement]
                  omega
          · rw [resultsAt, sourceFailureBoundAt]
            simp only [PMF.toOuterMeasure_pure_apply,
              Set.mem_setOf_eq, SuccessfulSources, if_neg nonzero, if_neg counter,
              not_true_eq_false, if_false, le_refl]

/-- The generated walk's source-failure mass is at most the sum of its
visited source-failure masses. The source may depend on the entire current
opening. The exact iteration counter prevents a missing required source
entry, without an independence or source-success premise. -/
theorem source_failure_probability_le (statement : Statement) (proof : Envelope application) :
    (results application fits source statement proof).toOuterMeasure
      {returns | ¬ SuccessfulSources application fits setup statement proof returns} ≤
        sourceFailureBound application fits setup source statement proof :=
  source_failure_probability_le_at application fits setup source statement.iteration statement proof le_rfl

/-- Retain the initial terminal opening and all source returns from its
adaptive reverse run. -/
noncomputable def law (initial : PMF (Statement × Envelope application)) :
    PMF (HyperNovaHistoryProbability.Sample application) :=
  initial.bind fun input =>
    (results application fits source input.1 input.2).map (fun returns => (input.1, input.2, returns))

/-- Sampling the reverse run preserves the exact initial terminal law. -/
theorem initial_marginal (initial : PMF (Statement × Envelope application)) :
    (law application fits source initial).map (fun sample => (sample.1, sample.2.1)) = initial := by
  rw [law, PMF.map_bind]
  have forget (input : Statement × Envelope application) :
      ((results application fits source input.1 input.2).map (fun returns => (input.1, input.2, returns))).map
        (fun sample => (sample.1, sample.2.1)) = PMF.pure input := by
    rw [PMF.map_comp]
    exact PMF.map_const (results application fits source input.1 input.2) input
  simp only [forget, PMF.bind_pure]

private theorem source_failure_mass_le (initial : PMF (Statement × Envelope application)) :
    (law application fits source initial).toOuterMeasure {sample | HyperNovaHistoryProbability.SourceFailure application fits setup sample} ≤
      ∑' input, initial input * (if PerApplicationTerminal.Holds application fits setup input.1 input.2
        then (sourceFailureBound application fits setup) source input.1 input.2 else 0) := by
  rw [law, PMF.toOuterMeasure_bind_apply]
  apply ENNReal.tsum_le_tsum
  intro input
  apply mul_le_mul_right
  by_cases accepted : PerApplicationTerminal.Holds application fits setup input.1 input.2
  · rw [if_pos accepted, PMF.toOuterMeasure_map_apply]
    refine (MeasureTheory.measure_mono ?_).trans
      (source_failure_probability_le application fits setup source input.1 input.2)
    intro returns failed
    exact failed.2
  · simp [accepted, PMF.toOuterMeasure_map_apply, HyperNovaHistoryProbability.SourceFailure,
      HyperNovaHistoryProbability.Accepted]

/-- Under the generated adaptive law, accepted initial mass is at most the
complete returned-history mass plus visited source-failure mass from accepted
initial openings and the
actual encountered state-hash failure mass. This requires no conditioning
of the source law, independent-call assumption, or supplied successful trace. -/
theorem accepted_probability_le (initial : PMF (Statement × Envelope application)) :
    initial.toOuterMeasure {input |
      PerApplicationTerminal.Holds application fits setup input.1 input.2} ≤
      (law application fits source initial).toOuterMeasure {sample | HyperNovaHistoryProbability.AdviceReturned application fits sample} +
        (∑' input, initial input * (if PerApplicationTerminal.Holds application fits setup input.1 input.2
          then (sourceFailureBound application fits setup) source input.1 input.2 else 0)) +
        (law application fits source initial).toOuterMeasure {sample | HyperNovaHistoryProbability.StateHashFailure application fits setup sample} := by
  have acceptedMass := congrArg
    (fun distribution : PMF (Statement × Envelope application) => distribution.toOuterMeasure
      {input | PerApplicationTerminal.Holds application fits setup input.1 input.2})
    (initial_marginal application fits source initial)
  rw [PMF.toOuterMeasure_map_apply] at acceptedMass
  calc
    _ = (law application fits source initial).toOuterMeasure
        {sample | HyperNovaHistoryProbability.Accepted application fits setup sample} := acceptedMass.symm
    _ ≤ _ := (HyperNovaHistoryProbability.accepted_probability_le application fits setup (law application fits source initial)).trans
      (add_le_add (add_le_add le_rfl (source_failure_mass_le application fits setup source initial)) le_rfl)

end NightstreamFPrime.Export.Stage1.HyperNovaHistoryLaw
