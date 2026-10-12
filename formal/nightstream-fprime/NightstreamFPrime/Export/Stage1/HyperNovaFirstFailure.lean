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
variable (application : Lifecycle.Stage1.Application.Program)
  (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
  (setup : PerApplicationCanonicalPackage.CommitmentSetup application)

attribute [local instance] Classical.propDecidable

/-- The current existing state-hash collision before any preceding marked
failure. This event includes a recursive base visit with no source call. -/
def MarkedHashCollision (visit : Visit application) : Prop :=
  visit.2 = true ∧
    match visit.1 with
    | some (statement, .recursive payload) => Collision application fits setup statement payload
    | _ => False

/-- Failure of the exact checked source-return event on the good active
branch. Aborts and missing entries fail this event's source check. -/
def MarkedSourceFailure (draw : Visit application × SourceResult application) : Prop :=
  goodActive application fits setup draw.1 ∧
    ¬ CheckedWitnessExtraction.SourceReturned (PiCCSStoredWitnessCheck.commit application setup) productionGlobalParams
      (PiCCSStoredWitnessCheck.statement application fits (HyperNovaGuardedSourceLaw.inputs application fits draw.1)) draw.2

private def FirstFailure (draw : Visit application × SourceResult application) : Prop :=
  MarkedHashCollision application fits setup draw.1 ∨ MarkedSourceFailure application fits setup draw

private theorem source_none (payload : Payload application) : ¬ SourceSucceeded application fits setup payload none := by
  rintro ⟨values, impossible, _⟩
  cases impossible

private theorem checks_of_no_first_failure (depth : Nat)
    (statement : Statement) (proof : Envelope application) (results : List (SourceResult application))
    (accepted : PerApplicationTerminal.Holds application fits setup statement proof)
    (bounded : statement.iteration ≤ depth)
    (noFailure : ∀ j < depth,
      ¬ FirstFailure application fits setup (readDraw application fits setup j (some (statement, proof), true) results)) :
    SuccessfulSources application fits setup statement proof results ∧ NoStateHashCollisions application fits setup statement proof results := by
  induction depth generalizing statement proof results with
  | zero =>
      cases proof with
      | bottom =>
          rw [SuccessfulSources.eq_def, NoStateHashCollisions.eq_def]
          exact ⟨trivial, trivial⟩
      | recursive payload =>
          have positive := (PerApplicationTerminal.holds_recursive_iff
            application fits setup statement payload).mp accepted |>.2.2.2.1
          omega
  | succ depth induction =>
      cases proof with
      | bottom =>
          rw [SuccessfulSources.eq_def, NoStateHashCollisions.eq_def]
          exact ⟨trivial, trivial⟩
      | recursive payload =>
          have positive := (PerApplicationTerminal.holds_recursive_iff
            application fits setup statement payload).mp accepted |>.2.2.2.1
          have nonzero : statement.iteration ≠ 0 := Nat.ne_of_gt positive
          have safe : ¬ Collision application fits setup statement payload := by
            intro collision
            apply noFailure 0 (Nat.zero_lt_succ depth)
            exact Or.inl ⟨rfl, collision⟩
          have counter := HyperNovaHistory.counter_matches application fits setup statement payload accepted safe
          by_cases base : (decodedInput application fits payload).iteration = 0
          · constructor
            · rw [SuccessfulSources.eq_def]
              simp only [if_neg nonzero, if_pos counter, if_pos base]
            · rw [NoStateHashCollisions.eq_def]
              simpa only [if_neg nonzero, if_pos counter, if_pos base, and_true] using safe
          · have active : ready application fits (some (statement, .recursive payload), true) :=
              ⟨nonzero, counter, base⟩
            have good : goodActive application fits setup (some (statement, .recursive payload), true) :=
              ⟨rfl, active, safe⟩
            have success : SourceSucceeded application fits setup payload (results.headD none) := by
              by_contra failed
              apply noFailure 0 (Nat.zero_lt_succ depth)
              apply Or.inr
              change (goodActive application fits setup) (some (statement, .recursive payload), true) ∧
                ¬ SourceSucceeded application fits setup payload
                  (if (goodActive application fits setup) (some (statement, .recursive payload), true)
                    then results.headD none else none)
              exact ⟨good, by simpa only [if_pos good] using failed⟩
            cases results with
            | nil => exact False.elim (source_none application fits setup payload success)
            | cons result tail =>
                cases result with
                | none => exact False.elim (source_none application fits setup payload success)
                | some values =>
                    have previousAccepted := (HyperNovaHistory.predecessor_accepted application fits setup
                      statement payload values (Nat.pos_of_ne_zero base) accepted success safe).2.2
                    have previousBound : (predecessorStatement application fits payload).iteration ≤ depth := by
                      dsimp only [predecessorStatement]
                      omega
                    have advanced : advance application fits setup (some (statement, .recursive payload), true)
                        (some values) =
                        (some (predecessorStatement application fits payload,
                          .recursive (predecessorPayload application fits payload values)), true) := by
                      simp only [advance, if_pos active, Bool.true_and]
                      exact Prod.ext rfl (decide_eq_true (And.intro safe success))
                    have previousNoFailure : ∀ j < depth,
                        ¬ FirstFailure application fits setup (readDraw application fits setup j
                          (some (predecessorStatement application fits payload,
                            .recursive (predecessorPayload application fits payload values)), true) tail) := by
                      intro j below
                      have next := noFailure (j + 1) (Nat.succ_lt_succ below)
                      simpa only [readDraw, if_pos active, advanced] using next
                    have previous := induction (predecessorStatement application fits payload)
                      (.recursive (predecessorPayload application fits payload values)) tail
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
theorem accepted_failure_exists_first (depth : Nat) (sample : Sample application)
    (bounded : sample.1.iteration ≤ depth) (accepted : Accepted application fits setup sample)
    (failed : ¬ AdviceReturned application fits sample) :
    ∃ j < depth,
      MarkedHashCollision application fits setup (observedDraw application fits setup j sample).1 ∨
        MarkedSourceFailure application fits setup (observedDraw application fits setup j sample) := by
  by_contra absent
  have initial : initialVisit application fits setup (sample.1, sample.2.1) =
      (some (sample.1, sample.2.1), true) := by
    exact Prod.ext rfl (decide_eq_true accepted)
  have noFailure : ∀ j < depth,
      ¬ FirstFailure application fits setup (readDraw application fits setup j (some (sample.1, sample.2.1), true) sample.2.2) := by
    intro j below failure
    apply absent
    refine ⟨j, below, ?_⟩
    simpa only [observedDraw, initial, FirstFailure] using failure
  have checked := checks_of_no_first_failure application fits setup depth sample.1 sample.2.1 sample.2.2
    accepted bounded noFailure
  exact failed (run_correct application fits setup sample.1 sample.2.1 sample.2.2 accepted checked.1 checked.2)

private theorem marked_hash_mass
    (source : Statement → Payload application → PMF (SourceResult application)) (contexts : PMF (Visit application)) :
    (contexts.bind (guardedDraw application fits setup source)).toOuterMeasure
      {draw | MarkedHashCollision application fits setup draw.1} =
      contexts.toOuterMeasure {visit | MarkedHashCollision application fits setup visit} := by
  have marginal : (contexts.bind (guardedDraw application fits setup source)).map Prod.fst = contexts := by
    rw [PMF.map_bind]
    have each (visit : Visit application) : (guardedDraw application fits setup source visit).map Prod.fst = PMF.pure visit := by
      by_cases good : goodActive application fits setup visit
      · rw [guardedDraw, if_pos good, PMF.map_comp]
        exact PMF.map_const _ _
      · rw [guardedDraw, if_neg good, PMF.pure_map]
    simp_rw [each]
    exact PMF.bind_pure _
  calc
    _ = ((contexts.bind (guardedDraw application fits setup source)).map Prod.fst).toOuterMeasure
        {visit | MarkedHashCollision application fits setup visit} := (PMF.toOuterMeasure_map_apply _ _ _).symm
    _ = _ := congrArg
      (fun distribution : PMF (Visit application) => distribution.toOuterMeasure {visit | MarkedHashCollision application fits setup visit})
      marginal

private theorem observed_event_mass
    (source : Statement → Payload application → PMF (SourceResult application))
    (initial : PMF (Statement × Envelope application)) (step : Nat) (event : Set (Visit application × SourceResult application)) :
    (HyperNovaHistoryLaw.law application fits source initial).toOuterMeasure
      {sample | observedDraw application fits setup step sample ∈ event} =
      ((visitedLaw application fits setup source initial step).bind (guardedDraw application fits setup source)).toOuterMeasure event := by
  calc
    _ = ((HyperNovaHistoryLaw.law application fits source initial).map (observedDraw application fits setup step)).toOuterMeasure event :=
      (PMF.toOuterMeasure_map_apply _ _ _).symm
    _ = _ := congrArg (fun distribution => distribution.toOuterMeasure event)
      (visitedDraw_marginal application fits setup source initial step)

/-- The history law's mass where the terminal verifier accepts and no
complete history returns is bounded by the first marked failure masses at the
exact visited contexts. The only scope bound is the symbolic initial iteration
bound. No independent-call, source-success, conditional-model or
successful-trace premise is used. -/
theorem unreturned_acceptance_le_first_failures
    (source : Statement → Payload application → PMF (SourceResult application))
    (initial : PMF (Statement × Envelope application)) (depth : Nat)
    (bounded : ∀ input ∈ initial.support, input.1.iteration ≤ depth) :
    (HyperNovaHistoryLaw.law application fits source initial).toOuterMeasure
        {sample | Accepted application fits setup sample ∧ ¬ AdviceReturned application fits sample} ≤
      ∑ j : Fin depth,
        ((visitedLaw application fits setup source initial j.val).toOuterMeasure {visit | MarkedHashCollision application fits setup visit} +
          (((visitedLaw application fits setup source initial j.val).bind (guardedDraw application fits setup source)).toOuterMeasure
            {draw | MarkedSourceFailure application fits setup draw})) := by
  let distribution := HyperNovaHistoryLaw.law application fits source initial
  let event (j : Fin depth) : Set (Sample application) :=
    {sample | FirstFailure application fits setup (observedDraw application fits setup j.val sample)}
  have inclusion : {sample | Accepted application fits setup sample ∧ ¬ AdviceReturned application fits sample} ∩
      distribution.support ⊆ ⋃ j : Fin depth, event j := by
    rintro sample ⟨⟨accepted, returned⟩, supported⟩
    have supportedInitial : (sample.1, sample.2.1) ∈ initial.support := by
      rw [← HyperNovaHistoryLaw.initial_marginal application fits source initial]
      exact (PMF.mem_support_map_iff _ _ _).mpr ⟨sample, supported, rfl⟩
    rcases (accepted_failure_exists_first application fits setup) depth sample
      (bounded (sample.1, sample.2.1) supportedInitial) accepted returned with ⟨j, below, failure⟩
    exact Set.mem_iUnion.mpr ⟨⟨j, below⟩, failure⟩
  calc
    _ ≤ distribution.toOuterMeasure (⋃ j : Fin depth, event j) :=
      distribution.toOuterMeasure_mono inclusion
    _ ≤ ∑ j : Fin depth, distribution.toOuterMeasure (event j) :=
      MeasureTheory.measure_iUnion_fintype_le _ _
    _ ≤ _ := by
      apply Finset.sum_le_sum
      intro j _member
      rw [show distribution.toOuterMeasure (event j) =
        (((visitedLaw application fits setup source initial j.val).bind (guardedDraw application fits setup source)).toOuterMeasure
          {draw | FirstFailure application fits setup draw}) from (observed_event_mass application fits setup) source initial j.val {draw | FirstFailure application fits setup draw}]
      have unionBound := MeasureTheory.measure_union_le
        (μ := ((visitedLaw application fits setup source initial j.val).bind (guardedDraw application fits setup source)).toOuterMeasure)
        {draw | MarkedHashCollision application fits setup draw.1} {draw | MarkedSourceFailure application fits setup draw}
      rw [marked_hash_mass] at unionBound
      exact unionBound

end NightstreamFPrime.Export.Stage1.HyperNovaFirstFailure
