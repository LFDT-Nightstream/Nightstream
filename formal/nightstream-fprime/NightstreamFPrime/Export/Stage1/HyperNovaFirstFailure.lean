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
open Poseidon2HashChainV1Package (application fits)
open Poseidon2HashChainV1Setup (productionSetup)

attribute [local instance] Classical.propDecidable

/-- The current existing state-hash collision before any preceding marked
failure. This event includes a recursive base visit with no source call. -/
def MarkedHashCollision (visit : Visit) : Prop :=
  visit.2 = true ∧
    match visit.1 with
    | some (statement, .recursive payload) => Collision statement payload
    | _ => False

/-- Failure of the exact checked source-return event on the good active
branch. Aborts and missing entries fail this event's source check. -/
def MarkedSourceFailure (draw : Visit × SourceResult) : Prop :=
  goodActive draw.1 ∧
    ¬ CheckedWitnessExtraction.SourceReturned PiCCSStoredWitnessCheck.commit productionGlobalParams
      (PiCCSStoredWitnessCheck.statement (HyperNovaGuardedSourceLaw.inputs draw.1)) draw.2

private def FirstFailure (draw : Visit × SourceResult) : Prop :=
  MarkedHashCollision draw.1 ∨ MarkedSourceFailure draw

private theorem source_none (payload : Payload) : ¬ SourceSucceeded payload none := by
  rintro ⟨values, impossible, _⟩
  cases impossible

private theorem checks_of_no_first_failure (depth : Nat)
    (statement : Statement) (proof : Envelope) (results : List SourceResult)
    (accepted : PerApplicationTerminal.Holds application fits productionSetup statement proof)
    (bounded : statement.iteration ≤ depth)
    (noFailure : ∀ j < depth,
      ¬ FirstFailure (readDraw j (some (statement, proof), true) results)) :
    SuccessfulSources statement proof results ∧ NoStateHashCollisions statement proof results := by
  induction depth generalizing statement proof results with
  | zero =>
      cases proof with
      | bottom =>
          rw [SuccessfulSources.eq_def, NoStateHashCollisions.eq_def]
          exact ⟨trivial, trivial⟩
      | recursive payload =>
          have positive := (PerApplicationTerminal.holds_recursive_iff
            application fits productionSetup statement payload).mp accepted |>.2.2.1
          omega
  | succ depth induction =>
      cases proof with
      | bottom =>
          rw [SuccessfulSources.eq_def, NoStateHashCollisions.eq_def]
          exact ⟨trivial, trivial⟩
      | recursive payload =>
          have positive := (PerApplicationTerminal.holds_recursive_iff
            application fits productionSetup statement payload).mp accepted |>.2.2.1
          have nonzero : statement.iteration ≠ 0 := Nat.ne_of_gt positive
          have safe : ¬ Collision statement payload := by
            intro collision
            apply noFailure 0 (Nat.zero_lt_succ depth)
            exact Or.inl ⟨rfl, collision⟩
          have counter := HyperNovaHistory.counter_matches statement payload accepted safe
          by_cases base : (decodedInput payload).iteration = 0
          · constructor
            · rw [SuccessfulSources.eq_def]
              simp only [if_neg nonzero, if_pos counter, if_pos base]
            · rw [NoStateHashCollisions.eq_def]
              simpa only [if_neg nonzero, if_pos counter, if_pos base, and_true] using safe
          · have active : ready (some (statement, .recursive payload), true) :=
              ⟨nonzero, counter, base⟩
            have good : goodActive (some (statement, .recursive payload), true) :=
              ⟨rfl, active, safe⟩
            have success : SourceSucceeded payload (results.headD none) := by
              by_contra failed
              apply noFailure 0 (Nat.zero_lt_succ depth)
              apply Or.inr
              change goodActive (some (statement, .recursive payload), true) ∧
                ¬ SourceSucceeded payload
                  (if goodActive (some (statement, .recursive payload), true)
                    then results.headD none else none)
              exact ⟨good, by simpa only [if_pos good] using failed⟩
            cases results with
            | nil => exact False.elim (source_none payload success)
            | cons result tail =>
                cases result with
                | none => exact False.elim (source_none payload success)
                | some values =>
                    have previousAccepted := (HyperNovaHistory.predecessor_accepted
                      statement payload values (Nat.pos_of_ne_zero base) accepted success safe).2.2
                    have previousBound : (predecessorStatement payload).iteration ≤ depth := by
                      dsimp only [predecessorStatement]
                      omega
                    have advanced : advance (some (statement, .recursive payload), true)
                        (some values) =
                        (some (predecessorStatement payload,
                          .recursive (predecessorPayload payload values)), true) := by
                      simp only [advance, if_pos active, Bool.true_and]
                      exact Prod.ext rfl (decide_eq_true (And.intro safe success))
                    have previousNoFailure : ∀ j < depth,
                        ¬ FirstFailure (readDraw j
                          (some (predecessorStatement payload,
                            .recursive (predecessorPayload payload values)), true) tail) := by
                      intro j below
                      have next := noFailure (j + 1) (Nat.succ_lt_succ below)
                      simpa only [readDraw, if_pos active, advanced] using next
                    have previous := induction (predecessorStatement payload)
                      (.recursive (predecessorPayload payload values)) tail
                      previousAccepted previousBound previousNoFailure
                    constructor
                    · rw [SuccessfulSources.eq_def]
                      simpa only [if_neg nonzero, if_pos counter, if_neg base] using
                        And.intro success previous.1
                    · rw [NoStateHashCollisions.eq_def]
                      simpa only [if_neg nonzero, if_pos counter, if_neg base] using
                        And.intro safe previous.2

/-- An accepted failed reverse run has a first marked hash or source failure
at an observation below the symbolic iteration bound. The statement holds
for every supplied source list, including missing entries and aborts. -/
theorem accepted_failure_exists_first (depth : Nat) (sample : Sample)
    (bounded : sample.1.iteration ≤ depth) (accepted : Accepted sample)
    (failed : ¬ AdviceReturned sample) :
    ∃ j < depth,
      MarkedHashCollision (observedDraw j sample).1 ∨
        MarkedSourceFailure (observedDraw j sample) := by
  by_contra absent
  have initial : initialVisit (sample.1, sample.2.1) =
      (some (sample.1, sample.2.1), true) := by
    exact Prod.ext rfl (decide_eq_true accepted)
  have noFailure : ∀ j < depth,
      ¬ FirstFailure (readDraw j (some (sample.1, sample.2.1), true) sample.2.2) := by
    intro j below failure
    apply absent
    refine ⟨j, below, ?_⟩
    simpa only [observedDraw, initial, FirstFailure] using failure
  have checked := checks_of_no_first_failure depth sample.1 sample.2.1 sample.2.2
    accepted bounded noFailure
  exact failed (run_correct sample.1 sample.2.1 sample.2.2 accepted checked.1 checked.2)

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
      (fun distribution : PMF Visit => distribution.toOuterMeasure {visit | MarkedHashCollision visit})
      marginal

private theorem observed_event_mass
    (source : Statement → Payload → PMF SourceResult)
    (initial : PMF (Statement × Envelope)) (step : Nat) (event : Set (Visit × SourceResult)) :
    (HyperNovaHistoryLaw.law source initial).toOuterMeasure
      {sample | observedDraw step sample ∈ event} =
      ((visitedLaw source initial step).bind (guardedDraw source)).toOuterMeasure event := by
  calc
    _ = ((HyperNovaHistoryLaw.law source initial).map (observedDraw step)).toOuterMeasure event :=
      (PMF.toOuterMeasure_map_apply _ _ _).symm
    _ = _ := congrArg (fun distribution => distribution.toOuterMeasure event)
      (visitedDraw_marginal source initial step)

/-- Accepted initial mass is bounded by complete returned-history mass and
the first marked failure masses at the exact visited contexts. The only
scope bound is the symbolic initial iteration bound. No independent-call,
source-success, conditional-model or successful-trace premise is used. -/
theorem accepted_probability_le_first_failures
    (source : Statement → Payload → PMF SourceResult)
    (initial : PMF (Statement × Envelope)) (depth : Nat)
    (bounded : ∀ input ∈ initial.support, input.1.iteration ≤ depth) :
    initial.toOuterMeasure {input |
      PerApplicationTerminal.Holds application fits productionSetup input.1 input.2} ≤
      (HyperNovaHistoryLaw.law source initial).toOuterMeasure {sample | AdviceReturned sample} +
        ∑ j : Fin depth,
          ((visitedLaw source initial j.val).toOuterMeasure {visit | MarkedHashCollision visit} +
            (((visitedLaw source initial j.val).bind (guardedDraw source)).toOuterMeasure
              {draw | MarkedSourceFailure draw})) := by
  let distribution := HyperNovaHistoryLaw.law source initial
  let event (j : Fin depth) : Set Sample :=
    {sample | FirstFailure (observedDraw j.val sample)}
  have inclusion : {sample | Accepted sample} ∩ distribution.support ⊆
      {sample | AdviceReturned sample} ∪ ⋃ j : Fin depth, event j := by
    rintro sample ⟨accepted, supported⟩
    by_cases returned : AdviceReturned sample
    · exact Or.inl returned
    · apply Or.inr
      have supportedInitial : (sample.1, sample.2.1) ∈ initial.support := by
        rw [← HyperNovaHistoryLaw.initial_marginal source initial]
        exact (PMF.mem_support_map_iff _ _ _).mpr ⟨sample, supported, rfl⟩
      rcases accepted_failure_exists_first depth sample
        (bounded (sample.1, sample.2.1) supportedInitial) accepted returned with ⟨j, below, failure⟩
      exact Set.mem_iUnion.mpr ⟨⟨j, below⟩, failure⟩
  have initialMass : initial.toOuterMeasure {input |
      PerApplicationTerminal.Holds application fits productionSetup input.1 input.2} =
      distribution.toOuterMeasure {sample | Accepted sample} := by
    rw [← HyperNovaHistoryLaw.initial_marginal source initial, PMF.toOuterMeasure_map_apply]
    rfl
  rw [initialMass]
  calc
    _ ≤ distribution.toOuterMeasure
        ({sample | AdviceReturned sample} ∪ ⋃ j : Fin depth, event j) :=
      distribution.toOuterMeasure_mono inclusion
    _ ≤ distribution.toOuterMeasure {sample | AdviceReturned sample} +
        distribution.toOuterMeasure (⋃ j : Fin depth, event j) :=
      MeasureTheory.measure_union_le _ _
    _ ≤ distribution.toOuterMeasure {sample | AdviceReturned sample} +
        ∑ j : Fin depth, distribution.toOuterMeasure (event j) :=
      add_le_add_right (MeasureTheory.measure_iUnion_fintype_le _ _) _
    _ ≤ _ := by
      apply add_le_add_right
      apply Finset.sum_le_sum
      intro j _member
      rw [show distribution.toOuterMeasure (event j) =
        (((visitedLaw source initial j.val).bind (guardedDraw source)).toOuterMeasure
          {draw | FirstFailure draw}) from observed_event_mass source initial j.val {draw | FirstFailure draw}]
      have unionBound := MeasureTheory.measure_union_le
        (μ := ((visitedLaw source initial j.val).bind (guardedDraw source)).toOuterMeasure)
        {draw | MarkedHashCollision draw.1} {draw | MarkedSourceFailure draw}
      rw [marked_hash_mass] at unionBound
      exact unionBound

end NightstreamFPrime.Export.Stage1.HyperNovaFirstFailure
