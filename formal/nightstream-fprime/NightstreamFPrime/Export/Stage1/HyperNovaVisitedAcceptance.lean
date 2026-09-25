import NightstreamFPrime.Export.Stage1.HyperNovaHistory
import NightstreamFPrime.Export.Stage1.HyperNovaGuardedSourceLaw

/-!
The analytical good-prefix mark is justified by actual terminal membership.
Every marked transition uses the checked source return and the existing
predecessor theorem. The guarded real experiment therefore succeeds exactly
on its good active visits, under the original unconditional visited law.
No Fiat--Shamir transfer, source-validity assumption, or conditioning is added.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaVisitedAcceptance

open scoped BigOperators ENNReal
open HyperNovaHistory
open HyperNovaVisitedLaw
open HyperNovaGuardedSourceLaw (inputs realOutput realLaw)
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.Nifs

variable (target : Wide.Target)

attribute [local instance] Classical.propDecidable

private def MarkedAccepted (visit : Visit target) : Prop :=
  visit.2 = true →
    match visit.1 with
    | none => False
    | some (statement, proof) =>
        target.Holds statement proof

private theorem stopped_accepted : MarkedAccepted target (stopped target) := by
  simp only [MarkedAccepted, stopped, Bool.false_eq_true, false_implies]

private theorem initial_accepted (input : Statement × target.Envelope) :
    MarkedAccepted target (initialVisit target input) := by
  intro marked
  exact of_decide_eq_true marked

private theorem advance_accepted (visit : Visit target) (result : SourceResult target)
    (accepted : MarkedAccepted target visit) : MarkedAccepted target (advance target visit result) := by
  by_cases active : ready target visit
  · rcases visit with ⟨current, mark⟩
    cases current with
    | none => exact False.elim active
    | some input =>
        rcases input with ⟨statement, proof⟩
        cases proof with
        | bottom => exact False.elim active
        | recursive payload =>
            cases result with
            | none =>
                simpa only [advance, if_pos active] using (stopped_accepted target)
            | some values =>
                simp only [advance, if_pos active, MarkedAccepted]
                intro marked
                have marks : mark = true ∧
                    decide (¬ target.Collision statement payload ∧
                      SourceSucceeded target payload (some values)) = true := by
                  simpa only [Bool.and_eq_true] using marked
                have events : ¬ target.Collision statement payload ∧
                    SourceSucceeded target payload (some values) := of_decide_eq_true marks.2
                have positive : 0 < (target.decodedInput payload).iteration :=
                  Nat.pos_of_ne_zero active.2.2
                exact (HyperNovaHistory.predecessor_accepted target statement payload values
                  positive (accepted marks.1) events.2 events.1).2.2
  · simpa only [advance, if_neg active] using (stopped_accepted target)

private theorem readDraw_accepted (steps : Nat) (visit : Visit target)
    (results : List (SourceResult target)) (accepted : MarkedAccepted target visit) :
    MarkedAccepted target (readDraw target steps visit results).1 := by
  induction steps generalizing visit results with
  | zero => exact accepted
  | succ steps induction =>
      by_cases active : ready target visit
      · cases results with
        | nil =>
            simpa only [readDraw, if_pos active] using
              induction (stopped target) [] (stopped_accepted target)
        | cons result tail =>
            simpa only [readDraw, if_pos active] using
              induction (advance target visit result) tail (advance_accepted target visit result accepted)
      · simpa only [readDraw, if_neg active] using
          induction (stopped target) results (stopped_accepted target)

private theorem guardedDraw_context
    (source : Statement → target.Payload → PMF (SourceResult target)) (visit : Visit target) :
    (guardedDraw target source visit).map Prod.fst = PMF.pure visit := by
  by_cases good : goodActive target visit
  · rw [guardedDraw, if_pos good, PMF.map_comp]
    exact PMF.map_const _ _
  · rw [guardedDraw, if_neg good, PMF.pure_map]

private theorem visited_context_marginal
    (source : Statement → target.Payload → PMF (SourceResult target))
    (initial : PMF (Statement × target.Envelope)) (steps : Nat) :
    (HyperNovaHistoryLaw.law target source initial).map (fun sample => (observedDraw target steps sample).1) =
      visitedLaw target source initial steps := by
  calc
    _ = ((HyperNovaHistoryLaw.law target source initial).map (observedDraw target steps)).map Prod.fst :=
      (PMF.map_comp _ _ _).symm
    _ = ((visitedLaw target source initial steps).bind (guardedDraw target source)).map Prod.fst :=
      congrArg (fun distribution => distribution.map Prod.fst)
        (visitedDraw_marginal target source initial steps)
    _ = _ := by
      rw [PMF.map_bind]
      simp_rw [guardedDraw_context]
      exact PMF.bind_pure _

private theorem supported_accepted
    (source : Statement → target.Payload → PMF (SourceResult target))
    (initial : PMF (Statement × target.Envelope)) (steps : Nat) (visit : Visit target)
    (supported : visit ∈ (visitedLaw target source initial steps).support) :
    MarkedAccepted target visit := by
  rw [← visited_context_marginal target source initial steps] at supported
  rcases (PMF.mem_support_map_iff _ _ _).mp supported with ⟨sample, _produced, same⟩
  rw [← same]
  exact readDraw_accepted target steps (initialVisit target (sample.1, sample.2.1)) sample.2.2
    (initial_accepted target (sample.1, sample.2.1))

/-- Every marked context in the actual visited law contains an accepted
selected terminal opening. Initial acceptance and every predecessor's source
membership are derived from the mark construction; none is a new premise. -/
theorem marked_accepted
    (source : Statement → target.Payload → PMF (SourceResult target))
    (initial : PMF (Statement × target.Envelope)) (steps : Nat) (visit : Visit target)
    (supported : visit ∈ (visitedLaw target source initial steps).support)
    (marked : visit.2 = true) :
    ∃ statement proof, visit.1 = some (statement, proof) ∧
      target.Holds statement proof := by
  have accepted := supported_accepted target source initial steps visit supported marked
  cases current : visit.1 with
  | none => exact False.elim (by simpa only [current] using accepted)
  | some input =>
      exact ⟨input.1, input.2, rfl, by simpa only [current] using accepted⟩

/-- At every supported visited context, the guarded real verifier event is
exactly the good active mark. This includes the original stopped and abort
mass, with no renormalization or accepted-context law as input. -/
theorem realSuccess_iff_goodActive
    (source : Statement → target.Payload → PMF (SourceResult target))
    (initial : PMF (Statement × target.Envelope)) (steps : Nat) (visit : Visit target)
    (supported : visit ∈ (visitedLaw target source initial steps).support) :
    WideFiatShamir.RealSuccess target.relation target.ajtai
      (target.security.running (inputs target visit)) (target.security.fresh (inputs target visit))
      (realOutput target visit) ↔ goodActive target visit := by
  by_cases good : goodActive target visit
  · refine ⟨fun _ => good, fun _ => ?_⟩
    rcases (marked_accepted target) source initial steps visit supported good.1 with
      ⟨statement, proof, current, accepted⟩
    cases proof with
    | bottom =>
        have impossible := good.2.1
        simp only [ready, current] at impossible
    | recursive payload =>
        have active : statement.iteration ≠ 0 ∧
            (target.decodedInput payload).iteration + 1 = statement.iteration ∧
            (target.decodedInput payload).iteration ≠ 0 := by
          simpa only [ready, current] using good.2.1
        have safe : ¬ target.Collision statement payload := by
          have safe := good.2.2
          change ¬ (match visit.1 with
            | some (statement, .recursive payload) => target.Collision statement payload
            | _ => False) at safe
          simpa only [current] using safe
        have outputEq : realOutput target visit = some (HyperNovaRealInput.output target payload) := by
          simp only [realOutput, if_pos good, current]
        rw [outputEq]
        simpa only [inputs, current] using
          (HyperNovaRealInput.realSuccess_of_terminal target) statement payload accepted safe
            (Nat.pos_of_ne_zero active.2.2)
  · rw [HyperNovaGuardedSourceLaw.realOutput_off target visit good]
    exact iff_of_false id good

private theorem realSuccessProbability_eq_event
    (distribution : PMF (Visit target × Option (WideFiatShamir.RealOutput target.relation))) :
    WideFiatShamir.realSuccessProbability target.relation target.ajtai
      (fun visit => target.security.running (inputs target visit))
      (fun visit => target.security.fresh (inputs target visit)) distribution =
      (distribution.toOuterMeasure {sample |
        WideFiatShamir.RealSuccess target.relation target.ajtai
          (target.security.running (inputs target sample.1)) (target.security.fresh (inputs target sample.1))
          sample.2}).toReal := by
  unfold WideFiatShamir.realSuccessProbability
  rw [PMF.toOuterMeasure_apply, ENNReal.tsum_toReal_eq (fun sample => by
    by_cases success : WideFiatShamir.RealSuccess target.relation target.ajtai
        (target.security.running ((inputs target) sample.1))
            (target.security.fresh ((inputs target) sample.1)) sample.2
    · simpa only [Set.indicator, Set.mem_setOf_eq, if_pos success] using
        distribution.apply_ne_top sample
    · simp only [Set.indicator, Set.mem_setOf_eq, if_neg success]
      exact ENNReal.zero_ne_top)]
  apply tsum_congr
  intro sample
  by_cases success : WideFiatShamir.RealSuccess target.relation target.ajtai
      (target.security.running (inputs target sample.1))
          (target.security.fresh (inputs target sample.1)) sample.2
  · simp only [Set.indicator, Set.mem_setOf_eq, if_pos success]
  · simp only [Set.indicator, Set.mem_setOf_eq, if_neg success, ENNReal.toReal_zero]

/-- The real success probability used by the approved FS transfer is the exact
mass of good active visits in this actual history law. The source kernel is
arbitrary; success of its earlier returns is checked by the analytical mark. -/
theorem realSuccessProbability_eq_goodActive
    (source : Statement → target.Payload → PMF (SourceResult target))
    (initial : PMF (Statement × target.Envelope)) (steps : Nat) :
    WideFiatShamir.realSuccessProbability target.relation target.ajtai
      (fun visit => target.security.running (inputs target visit))
      (fun visit => target.security.fresh (inputs target visit))
      (realLaw target (visitedLaw target source initial steps)) =
      ((visitedLaw target source initial steps).toOuterMeasure {visit | goodActive target visit}).toReal := by
  rw [realSuccessProbability_eq_event, realLaw, PMF.toOuterMeasure_map_apply]
  apply congrArg ENNReal.toReal
  apply PMF.toOuterMeasure_apply_eq_of_inter_support_eq
  ext visit
  constructor
  · rintro ⟨success, supported⟩
    exact ⟨(realSuccess_iff_goodActive target source initial steps visit supported).mp success, supported⟩
  · rintro ⟨good, supported⟩
    exact ⟨(realSuccess_iff_goodActive target source initial steps visit supported).mpr good, supported⟩

end NightstreamFPrime.Export.Stage1.HyperNovaVisitedAcceptance
