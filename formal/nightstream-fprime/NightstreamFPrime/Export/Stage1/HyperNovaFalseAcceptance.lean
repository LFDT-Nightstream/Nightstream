import NightstreamFPrime.Export.Stage1.HyperNovaVisitedSecurity

/-!
False acceptance at the selected full-opening terminal boundary means that
the verifier accepts but no advice list has the advertised length and forward
application result. This is exactly the history conclusion in AdviceReturned.
It is distinct from bare NIFS Boolean acceptance and from extractor failure.

The bound is on the IVC adversary's original mixed law; no conditioning on
invalid inputs is used. It runs the reverse extractor of HyperNova Lemma 17
(`HyperNovaVisitedSecurity.reverseStages`) under HyperNova errata Assumption 1
(`HyperNovaVisitedSecurity.Assumption1`) and keeps each stage's
hash-collision mass explicit. Honest rejection is separate.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaFalseAcceptance

open scoped BigOperators ENNReal
open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open HyperNovaHistory (Statement Envelope Payload SourceResult)
open HyperNovaHistoryProbability (Sample Accepted AdviceReturned)
open HyperNovaVisitedLaw (Visit goodActive visitedLaw guardedDraw observedDraw pathVisit listSource)
open HyperNovaFirstFailure (MarkedHashCollision MarkedSourceFailure)
open HyperNovaVisitedSecurity (event_ne_top NifsAdversary NifsExtractor Assumption1 Closed IvcAdversary
  Stage ExtractionFails reverseStages reverseExtractor)
open Poseidon2HashChainV1Package (application fits)
open Poseidon2HashChainV1Setup (productionSetup)

attribute [local instance] Classical.propDecidable

/-- Acceptance with no history meeting the existing exact length and forward
evaluation contract. The initial law may mix valid and invalid statements. -/
def FalseAcceptance (input : Statement × Envelope) : Prop :=
  PerApplicationTerminal.Holds application fits productionSetup input.1 input.2 ∧
    ¬ ∃ advice : List AppWitness,
      advice.length = input.1.iteration ∧
      advice.foldl application.step input.1.z0 = input.1.zi

/-- Invalid accepted statements cannot have a returned valid history. This
uses the event definitions only, with no output-soundness premise. -/
theorem falseAcceptance_not_adviceReturned (sample : Sample)
    (invalid : FalseAcceptance (sample.1, sample.2.1)) :
    ¬ AdviceReturned sample := by
  rintro ⟨advice, _unused, _returned, length, evaluated⟩
  exact invalid.2 ⟨advice, length, evaluated⟩

private theorem marked_hash_mass
    (source : Statement → Payload → PMF SourceResult) (contexts : PMF Visit) :
    (contexts.bind (guardedDraw source)).toOuterMeasure
      {draw | MarkedHashCollision draw.1} =
      contexts.toOuterMeasure {visit | MarkedHashCollision visit} := by
  have marginal : (contexts.bind (guardedDraw source)).map Prod.fst = contexts := by
    rw [PMF.map_bind]
    have each (visit : Visit) : (guardedDraw source visit).map Prod.fst = PMF.pure visit := by
      by_cases good : goodActive visit
      · rw [guardedDraw, if_pos good, PMF.map_comp]
        exact PMF.map_const _ _
      · rw [guardedDraw, if_neg good, PMF.pure_map]
    simp_rw [each]
    exact PMF.bind_pure _
  calc
    _ = ((contexts.bind (guardedDraw source)).map Prod.fst).toOuterMeasure
        {visit | MarkedHashCollision visit} := (PMF.toOuterMeasure_map_apply _ _ _).symm
    _ = _ := congrArg
      (fun distribution => distribution.toOuterMeasure {visit | MarkedHashCollision visit})
      marginal

/-- False-acceptance mass is bounded by the first marked failures at the
visited contexts of any source law. -/
theorem false_acceptance_mass_le_first_failures
    (source : Statement → Payload → PMF SourceResult)
    (initial : PMF (Statement × Envelope)) (depth : Nat)
    (bounded : ∀ input ∈ initial.support, input.1.iteration ≤ depth) :
    initial.toOuterMeasure {input | FalseAcceptance input} ≤
      ∑ j : Fin depth,
        ((visitedLaw source initial j.val).toOuterMeasure {visit | MarkedHashCollision visit} +
          (((visitedLaw source initial j.val).bind (guardedDraw source)).toOuterMeasure
            {draw | MarkedSourceFailure draw})) := by
  let distribution := HyperNovaHistoryLaw.law source initial
  let event (j : Fin depth) : Set Sample :=
    {sample | MarkedHashCollision (observedDraw j.val sample).1 ∨
      MarkedSourceFailure (observedDraw j.val sample)}
  have inclusion : {sample | FalseAcceptance (sample.1, sample.2.1)} ∩
      distribution.support ⊆ ⋃ j : Fin depth, event j := by
    rintro sample ⟨invalid, supported⟩
    have supportedInitial : (sample.1, sample.2.1) ∈ initial.support := by
      rw [← HyperNovaHistoryLaw.initial_marginal source initial]
      exact (PMF.mem_support_map_iff _ _ _).mpr ⟨sample, supported, rfl⟩
    rcases HyperNovaFirstFailure.accepted_failure_exists_first depth sample
      (bounded (sample.1, sample.2.1) supportedInitial) invalid.1
      (falseAcceptance_not_adviceReturned sample invalid) with ⟨j, below, failure⟩
    exact Set.mem_iUnion.mpr ⟨⟨j, below⟩, failure⟩
  have initialMass : initial.toOuterMeasure {input | FalseAcceptance input} =
      distribution.toOuterMeasure {sample | FalseAcceptance (sample.1, sample.2.1)} := by
    rw [← HyperNovaHistoryLaw.initial_marginal source initial, PMF.toOuterMeasure_map_apply]
    rfl
  rw [initialMass]
  calc
    _ ≤ distribution.toOuterMeasure (⋃ j : Fin depth, event j) :=
      distribution.toOuterMeasure_mono inclusion
    _ ≤ ∑ j : Fin depth, distribution.toOuterMeasure (event j) :=
      MeasureTheory.measure_iUnion_fintype_le _ _
    _ ≤ _ := by
      apply Finset.sum_le_sum
      intro j _member
      have eventMass : distribution.toOuterMeasure (event j) =
          (((visitedLaw source initial j.val).bind (guardedDraw source)).toOuterMeasure
            {draw | MarkedHashCollision draw.1 ∨ MarkedSourceFailure draw}) := by
        rw [← HyperNovaVisitedLaw.visitedDraw_marginal source initial j.val,
          PMF.toOuterMeasure_map_apply]
        rfl
      rw [eventMass]
      have unionBound := MeasureTheory.measure_union_le
        (μ := ((visitedLaw source initial j.val).bind (guardedDraw source)).toOuterMeasure)
        {draw | MarkedHashCollision draw.1} {draw | MarkedSourceFailure draw}
      rw [marked_hash_mass] at unionBound
      exact unionBound

variable {Admitted : NifsAdversary → Prop} {StageAdmitted : Stage → Prop}
  {Efficient : (adversary : NifsAdversary) → NifsExtractor adversary → Prop}
  {error : NifsAdversary → ℝ}
  (assumption : Assumption1 Admitted Efficient error) (closed : Closed Admitted StageAdmitted Efficient)
  (adversary : IvcAdversary) (admitted : StageAdmitted (Stage.start adversary))

local notation "stages" => reverseStages assumption closed adversary admitted
local notation "extractors" => reverseExtractor assumption closed adversary admitted

private theorem per_tape (depth : Nat)
    (depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).1.iteration ≤ depth)
    (tape : (stages depth).1.Tape) (supported : tape ∈ (stages depth).1.tape.support) :
    (PMF.pure ((stages depth).1.input tape)).toOuterMeasure {input | FalseAcceptance input} ≤
      ∑ j : Fin depth,
        ((PMF.pure (pathVisit ((stages depth).1.input tape) ((stages depth).1.results tape) j.val)).toOuterMeasure
            {visit | MarkedHashCollision visit} +
          (PMF.pure (pathVisit ((stages depth).1.input tape) ((stages depth).1.results tape) j.val,
            if goodActive (pathVisit ((stages depth).1.input tape) ((stages depth).1.results tape) j.val)
            then ((stages depth).1.results tape).getD j.val none else none)).toOuterMeasure
            {draw | MarkedSourceFailure draw}) := by
  have length := HyperNovaVisitedSecurity.results_length assumption closed adversary admitted depth tape
  have first := false_acceptance_mass_le_first_failures
    (listSource ((stages depth).1.input tape).1.iteration ((stages depth).1.results tape))
    (PMF.pure ((stages depth).1.input tape)) depth (by
      intro input member
      rw [PMF.mem_support_pure_iff] at member
      rw [member]
      exact HyperNovaVisitedSecurity.input_bound assumption closed adversary admitted depth depthBound
        depth tape supported)
  refine first.trans (le_of_eq ?_)
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [HyperNovaVisitedLaw.visitedLaw_listSource _ _ _ (by omega), PMF.pure_bind,
    HyperNovaVisitedLaw.guardedDraw_listSource _ _ _ (by omega)]

/-- False-acceptance bound for the selected terminal verifier on the IVC
adversary's original mixed law under Assumption 1 (HyperNova Lemma 17). Each
stage of the reverse extractor contributes its marked hash-collision mass and
the Assumption 1 error of its NIFS adversary. This is not a bound on bare NIFS
Boolean acceptance. -/
theorem probability_bound (depth : Nat)
    (depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).1.iteration ≤ depth) :
    (adversary.tape.toOuterMeasure {tape | FalseAcceptance (adversary.output tape)}).toReal ≤
      ∑ j : Fin depth,
        (((stages j.val).1.tape.toOuterMeasure
            {tape | MarkedHashCollision ((stages j.val).1.visit tape)}).toReal +
          error (stages j.val).1.nifs) := by
  let law := (stages depth).1.tape
  let hash : Fin depth → (stages depth).1.Tape → ℝ≥0∞ := fun j tape =>
    (PMF.pure (pathVisit ((stages depth).1.input tape) ((stages depth).1.results tape) j.val)).toOuterMeasure
      {visit | MarkedHashCollision visit}
  let failure : Fin depth → (stages depth).1.Tape → ℝ≥0∞ := fun j tape =>
    (PMF.pure (pathVisit ((stages depth).1.input tape) ((stages depth).1.results tape) j.val,
      if goodActive (pathVisit ((stages depth).1.input tape) ((stages depth).1.results tape) j.val)
      then ((stages depth).1.results tape).getD j.val none else none)).toOuterMeasure
      {draw | MarkedSourceFailure draw}
  have bound : adversary.tape.toOuterMeasure {tape | FalseAcceptance (adversary.output tape)} ≤
      ∑ j : Fin depth,
        ((stages j.val).1.tape.toOuterMeasure {tape | MarkedHashCollision ((stages j.val).1.visit tape)} +
          (extractors j.val).law.toOuterMeasure {draw | ExtractionFails _ (extractors j.val) draw}) := by
    calc
      _ = ∑' tape, law tape * (PMF.pure ((stages depth).1.input tape)).toOuterMeasure
            {input | FalseAcceptance input} :=
          HyperNovaVisitedSecurity.start_event assumption closed adversary admitted depth
            {input | FalseAcceptance input}
      _ ≤ ∑' tape, law tape * ∑ j : Fin depth, (hash j tape + failure j tape) := by
          apply ENNReal.tsum_le_tsum
          intro tape
          by_cases zero : law tape = 0
          · simp only [zero, zero_mul, le_refl]
          · exact mul_le_mul_right (per_tape assumption closed adversary admitted depth depthBound tape
              ((PMF.mem_support_iff _ _).mpr zero)) _
      _ = ∑ j : Fin depth, (∑' tape, law tape * hash j tape + ∑' tape, law tape * failure j tape) := by
          simp only [mul_add, Finset.mul_sum]
          rw [Summable.tsum_finsetSum (fun _ _ => ENNReal.summable)]
          exact Finset.sum_congr rfl fun j _ => ENNReal.tsum_add
      _ ≤ _ := by
          refine Finset.sum_le_sum fun j _ => add_le_add (le_of_eq ?_) ?_
          · exact HyperNovaVisitedSecurity.hash_term_total assumption closed adversary admitted j.val depth
              j.isLt.le
          · exact HyperNovaVisitedSecurity.failure_term_total assumption closed adversary admitted j.val
              depth j.isLt
  have finiteTerm (j : Fin depth) :
      (stages j.val).1.tape.toOuterMeasure {tape | MarkedHashCollision ((stages j.val).1.visit tape)} +
        (extractors j.val).law.toOuterMeasure {draw | ExtractionFails _ (extractors j.val) draw} ≠ ∞ :=
    ENNReal.add_ne_top.mpr ⟨event_ne_top _ _, event_ne_top _ _⟩
  have finiteSum := ENNReal.sum_ne_top.mpr (fun j (_ : j ∈ Finset.univ) => finiteTerm j)
  have realBound := ENNReal.toReal_mono finiteSum bound
  rw [ENNReal.toReal_sum (fun j (_ : j ∈ Finset.univ) => finiteTerm j)] at realBound
  refine realBound.trans (Finset.sum_le_sum fun j _ => ?_)
  rw [ENNReal.toReal_add (event_ne_top _ _) (event_ne_top _ _)]
  exact add_le_add le_rfl
    (HyperNovaVisitedSecurity.reverseExtractor_error assumption closed adversary admitted j.val)

end NightstreamFPrime.Export.Stage1.HyperNovaFalseAcceptance
