import NightstreamFPrime.Export.Stage1.HyperNovaHistory
import NightstreamFPrime.Export.Stage1.HyperNovaVisitedAcceptance

/-!
An accepted reverse run fails only at its first marked hash collision or
first good-active source failure. Observation includes the base visit, where
no source call is made. Missing source entries are read as the absent result.
The finite union bound uses the exact unconditional visited laws; later
operational calls after a false mark add no security-failure event.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaFirstFailure

open scoped BigOperators ENNReal
open HyperNovaHistory
open HyperNovaVisitedLaw
open HyperNovaHistoryProbability (Sample Accepted AdviceReturned)
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle

variable (target : Wide.Target)

attribute [local instance] Classical.propDecidable

/-- The current existing state-hash collision before any preceding marked
failure. This event includes a recursive base visit with no source call. -/
def MarkedHashCollision (visit : Visit target) : Prop :=
  visit.2 = true ∧
    match visit.1 with
    | some (statement, .recursive payload) => target.Collision statement payload
    | _ => False

/-- Failure of the exact checked source-return event on the good active
branch. Aborts and missing entries fail this event's source check. -/
def MarkedSourceFailure (draw : Visit target × SourceResult target) : Prop :=
  goodActive target draw.1 ∧
    ¬ CheckedWitnessExtraction.SourceReturned (PiCCSStoredWitnessCheck.commit target.security)
        productionGlobalParams
      (PiCCSStoredWitnessCheck.statement target.security
          (HyperNovaGuardedSourceLaw.inputs target draw.1)) draw.2

private def FirstFailure (draw : Visit target × SourceResult target) : Prop :=
  MarkedHashCollision target draw.1 ∨ MarkedSourceFailure target draw

private theorem source_none (payload : target.Payload) : ¬ SourceSucceeded target payload none := by
  rintro ⟨values, impossible, _⟩
  cases impossible

private theorem checks_of_no_first_failure (depth : Nat)
    (statement : Statement) (proof : target.Envelope) (results : List (SourceResult target))
    (accepted : target.Holds statement proof)
    (bounded : statement.iteration ≤ depth)
    (noFailure : ∀ j < depth,
      ¬ FirstFailure target (readDraw target j (some (statement, proof), true) results)) :
    SuccessfulSources target statement proof results ∧ NoStateHashCollisions target statement proof
        results := by
  induction depth generalizing statement proof results with
  | zero =>
      cases proof with
      | bottom =>
          rw [SuccessfulSources.eq_def, NoStateHashCollisions.eq_def]
          exact ⟨trivial, trivial⟩
      | recursive payload =>
          have positive :=
              (Lifecycle.Stage1.Terminal.holdsFor_recursive_iff target.relation target.ajtai
              target.context target.program statement payload).mp accepted |>.2.2.1
          omega
  | succ depth induction =>
      cases proof with
      | bottom =>
          rw [SuccessfulSources.eq_def, NoStateHashCollisions.eq_def]
          exact ⟨trivial, trivial⟩
      | recursive payload =>
          have positive :=
              (Lifecycle.Stage1.Terminal.holdsFor_recursive_iff target.relation target.ajtai
              target.context target.program statement payload).mp accepted |>.2.2.1
          have nonzero : statement.iteration ≠ 0 := Nat.ne_of_gt positive
          have safe : ¬ target.Collision statement payload := by
            intro collision
            apply noFailure 0 (Nat.zero_lt_succ depth)
            exact Or.inl ⟨rfl, collision⟩
          have counter := HyperNovaHistory.counter_matches target statement payload accepted safe
          by_cases base : (target.decodedInput payload).iteration = 0
          · constructor
            · rw [SuccessfulSources.eq_def]
              simp only [if_neg nonzero, if_pos counter, if_pos base]
            · rw [NoStateHashCollisions.eq_def]
              simpa only [if_neg nonzero, if_pos counter, if_pos base, and_true] using safe
          · have active : ready target (some (statement, .recursive payload), true) :=
              ⟨nonzero, counter, base⟩
            have good : goodActive target (some (statement, .recursive payload), true) :=
              ⟨rfl, active, safe⟩
            have success : SourceSucceeded target payload (results.headD none) := by
              by_contra failed
              apply noFailure 0 (Nat.zero_lt_succ depth)
              apply Or.inr
              change (goodActive target) (some (statement, .recursive payload), true) ∧
                ¬ SourceSucceeded target payload
                  (if goodActive target (some (statement, .recursive payload), true)
                    then results.headD none else none)
              exact ⟨good, by simpa only [if_pos good] using failed⟩
            cases results with
            | nil => exact False.elim (source_none target payload success)
            | cons result tail =>
                cases result with
                | none => exact False.elim (source_none target payload success)
                | some values =>
                    have previousAccepted := (HyperNovaHistory.predecessor_accepted target
                      statement payload values (Nat.pos_of_ne_zero base) accepted success safe).2.2
                    have previousBound : (predecessorStatement target payload).iteration ≤ depth := by
                      dsimp only [predecessorStatement]
                      omega
                    have advanced : advance target (some (statement, .recursive payload), true)
                        (some values) =
                        (some (predecessorStatement target payload,
                          .recursive (predecessorPayload target payload values)), true) := by
                      simp only [advance, if_pos active, Bool.true_and]
                      exact Prod.ext rfl (decide_eq_true (And.intro safe success))
                    have previousNoFailure : ∀ j < depth,
                        ¬ FirstFailure target (readDraw target j
                          (some (predecessorStatement target payload,
                            .recursive (predecessorPayload target payload values)), true) tail) := by
                      intro j below
                      have next := noFailure (j + 1) (Nat.succ_lt_succ below)
                      simpa only [readDraw, if_pos active, advanced] using next
                    have previous := induction (predecessorStatement target payload)
                      (.recursive (predecessorPayload target payload values)) tail
                      previousAccepted previousBound previousNoFailure
                    constructor
                    · rw [SuccessfulSources.eq_def]
                      simpa only [if_neg nonzero, if_pos counter, if_neg base] using!
                        And.intro success previous.1
                    · rw [NoStateHashCollisions.eq_def]
                      simpa only [if_neg nonzero, if_pos counter, if_neg base] using
                        And.intro safe previous.2

/-- An accepted failed reverse run has a first marked hash or source failure
at an observation below the symbolic iteration bound. The statement holds
for every supplied source list, including missing entries and aborts. -/
theorem accepted_failure_exists_first (depth : Nat) (sample : Sample target)
    (bounded : sample.1.iteration ≤ depth) (accepted : Accepted target sample)
    (failed : ¬ AdviceReturned target sample) :
    ∃ j < depth,
      MarkedHashCollision target (observedDraw target j sample).1 ∨
        MarkedSourceFailure target (observedDraw target j sample) := by
  by_contra absent
  have initial : initialVisit target (sample.1, sample.2.1) =
      (some (sample.1, sample.2.1), true) := by
    exact Prod.ext rfl (decide_eq_true accepted)
  have noFailure : ∀ j < depth,
      ¬ FirstFailure target (readDraw target j (some (sample.1, sample.2.1), true) sample.2.2) := by
    intro j below failure
    apply absent
    refine ⟨j, below, ?_⟩
    simpa only [observedDraw, initial, FirstFailure] using failure
  have checked := checks_of_no_first_failure target depth sample.1 sample.2.1 sample.2.2
    accepted bounded noFailure
  exact failed (run_correct target sample.1 sample.2.1 sample.2.2 accepted checked.1 checked.2)

private theorem marked_hash_mass
    (source : Statement → target.Payload → PMF (SourceResult target)) (contexts : PMF (Visit target)) :
    (contexts.bind (guardedDraw target source)).toOuterMeasure
      {draw | MarkedHashCollision target draw.1} =
      contexts.toOuterMeasure {visit | MarkedHashCollision target visit} := by
  have marginal : (contexts.bind (guardedDraw target source)).map Prod.fst = contexts := by
    rw [PMF.map_bind]
    have each (visit : Visit target) : (guardedDraw target source visit).map Prod.fst = PMF.pure visit := by
      by_cases good : goodActive target visit
      · rw [guardedDraw, if_pos good, PMF.map_comp]
        exact PMF.map_const _ _
      · rw [guardedDraw, if_neg good, PMF.pure_map]
    simp_rw [each]
    exact PMF.bind_pure _
  calc
    _ = ((contexts.bind (guardedDraw target source)).map Prod.fst).toOuterMeasure
        {visit | MarkedHashCollision target visit} := (PMF.toOuterMeasure_map_apply _ _ _).symm
    _ = _ := congrArg
      (fun distribution : PMF (Visit target) => distribution.toOuterMeasure
          {visit | MarkedHashCollision target visit})
      marginal

private theorem observed_event_mass
    (source : Statement → target.Payload → PMF (SourceResult target))
    (initial : PMF (Statement × target.Envelope)) (step : Nat)
        (event : Set (Visit target × SourceResult target)) :
    (HyperNovaHistoryLaw.law target source initial).toOuterMeasure
      {sample | observedDraw target step sample ∈ event} =
      ((visitedLaw target source initial step).bind (guardedDraw target source)).toOuterMeasure event := by
  calc
    _ = ((HyperNovaHistoryLaw.law target source initial).map
        (observedDraw target step)).toOuterMeasure event :=
      (PMF.toOuterMeasure_map_apply _ _ _).symm
    _ = _ := congrArg (fun distribution => distribution.toOuterMeasure event)
      (visitedDraw_marginal target source initial step)

/-- Accepted initial mass is bounded by complete returned-history mass and
the first marked failure masses at the exact visited contexts. The only
scope bound is the symbolic initial iteration bound. No independent-call,
source-success, conditional-model or successful-trace premise is used. -/
theorem accepted_probability_le_first_failures
    (source : Statement → target.Payload → PMF (SourceResult target))
    (initial : PMF (Statement × target.Envelope)) (depth : Nat)
    (bounded : ∀ input ∈ initial.support, input.1.iteration ≤ depth) :
    initial.toOuterMeasure {input |
      target.Holds input.1 input.2} ≤
      (HyperNovaHistoryLaw.law target source initial).toOuterMeasure {sample | AdviceReturned target sample} +
        ∑ j : Fin depth,
          ((visitedLaw target source initial j.val).toOuterMeasure
              {visit | MarkedHashCollision target visit} +
            (((visitedLaw target source initial j.val).bind (guardedDraw target source)).toOuterMeasure
              {draw | MarkedSourceFailure target draw})) := by
  let distribution := HyperNovaHistoryLaw.law target source initial
  let event (j : Fin depth) : Set (Sample target) :=
    {sample | FirstFailure target (observedDraw target j.val sample)}
  have inclusion : {sample | Accepted target sample} ∩ distribution.support ⊆
      {sample | AdviceReturned target sample} ∪ ⋃ j : Fin depth, event j := by
    rintro sample ⟨accepted, supported⟩
    by_cases returned : AdviceReturned target sample
    · exact Or.inl returned
    · apply Or.inr
      have supportedInitial : (sample.1, sample.2.1) ∈ initial.support := by
        rw [← HyperNovaHistoryLaw.initial_marginal target source initial]
        exact (PMF.mem_support_map_iff _ _ _).mpr ⟨sample, supported, rfl⟩
      rcases (accepted_failure_exists_first target) depth sample
        (bounded (sample.1, sample.2.1) supportedInitial) accepted returned with ⟨j, below, failure⟩
      exact Set.mem_iUnion.mpr ⟨⟨j, below⟩, failure⟩
  have initialMass : initial.toOuterMeasure {input |
      target.Holds input.1 input.2} =
      distribution.toOuterMeasure {sample | Accepted target sample} := by
    rw [← HyperNovaHistoryLaw.initial_marginal target source initial, PMF.toOuterMeasure_map_apply]
    rfl
  rw [initialMass]
  calc
    _ ≤ distribution.toOuterMeasure
        ({sample | AdviceReturned target sample} ∪ ⋃ j : Fin depth, event j) :=
      distribution.toOuterMeasure_mono inclusion
    _ ≤ distribution.toOuterMeasure {sample | AdviceReturned target sample} +
        distribution.toOuterMeasure (⋃ j : Fin depth, event j) :=
      MeasureTheory.measure_union_le _ _
    _ ≤ distribution.toOuterMeasure {sample | AdviceReturned target sample} +
        ∑ j : Fin depth, distribution.toOuterMeasure (event j) :=
      add_le_add_right (MeasureTheory.measure_iUnion_fintype_le _ _) _
    _ ≤ _ := by
      apply add_le_add_right
      apply Finset.sum_le_sum
      intro j _member
      rw [show distribution.toOuterMeasure (event j) =
        (((visitedLaw target source initial j.val).bind (guardedDraw target source)).toOuterMeasure
          {draw | FirstFailure target draw}) from
          observed_event_mass target source initial j.val {draw | FirstFailure target draw}]
      have unionBound := MeasureTheory.measure_union_le
        (μ := ((visitedLaw target source initial j.val).bind (guardedDraw target source)).toOuterMeasure)
        {draw | MarkedHashCollision target draw.1} {draw | MarkedSourceFailure target draw}
      rw [marked_hash_mass] at unionBound
      exact unionBound

end NightstreamFPrime.Export.Stage1.HyperNovaFirstFailure
