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
open HyperNovaGuardedSourceLaw (inputs realOutput)
open NightstreamFPrime.Lifecycle
variable (application : Lifecycle.Stage1.Application.Program)
  (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
  (setup : PerApplicationCanonicalPackage.CommitmentSetup application)

attribute [local instance] Classical.propDecidable

private def MarkedAccepted (visit : Visit application) : Prop :=
  visit.2 = true →
    match visit.1 with
    | none => False
    | some (statement, proof) =>
        PerApplicationTerminal.Holds application fits setup statement proof

private theorem stopped_accepted : MarkedAccepted application fits setup (stopped application) := by
  simp only [MarkedAccepted, stopped, Bool.false_eq_true, false_implies]

private theorem initial_accepted (input : Statement × Envelope application) :
    MarkedAccepted application fits setup (initialVisit application fits setup input) := by
  intro marked
  exact of_decide_eq_true marked

private theorem advance_accepted (visit : Visit application) (result : SourceResult application)
    (accepted : MarkedAccepted application fits setup visit) : MarkedAccepted application fits setup (advance application fits setup visit result) := by
  by_cases active : ready application fits visit
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
                simpa only [advance, if_pos active] using (stopped_accepted application fits setup)
            | some values =>
                simp only [advance, if_pos active, MarkedAccepted]
                intro marked
                have marks : mark = true ∧
                    decide (¬ Collision application fits setup statement payload ∧
                      SourceSucceeded application fits setup payload (some values)) = true := by
                  simpa only [Bool.and_eq_true] using marked
                have events : ¬ Collision application fits setup statement payload ∧
                    SourceSucceeded application fits setup payload (some values) := of_decide_eq_true marks.2
                have positive : 0 < (decodedInput application fits payload).iteration :=
                  Nat.pos_of_ne_zero active.2.2
                exact (HyperNovaHistory.predecessor_accepted application fits setup statement payload values
                  positive (accepted marks.1) events.2 events.1).2.2
  · simpa only [advance, if_neg active] using (stopped_accepted application fits setup)

private theorem readDraw_accepted (steps : Nat) (visit : Visit application)
    (results : List (SourceResult application)) (accepted : MarkedAccepted application fits setup visit) :
    MarkedAccepted application fits setup (readDraw application fits setup steps visit results).1 := by
  induction steps generalizing visit results with
  | zero => exact accepted
  | succ steps induction =>
      by_cases active : ready application fits visit
      · cases results with
        | nil =>
            simpa only [readDraw, if_pos active] using
              induction (stopped application) [] (stopped_accepted application fits setup)
        | cons result tail =>
            simpa only [readDraw, if_pos active] using
              induction (advance application fits setup visit result) tail (advance_accepted application fits setup visit result accepted)
      · simpa only [readDraw, if_neg active] using
          induction (stopped application) results (stopped_accepted application fits setup)

private theorem guardedDraw_context
    (source : Statement → Payload application → PMF (SourceResult application)) (visit : Visit application) :
    (guardedDraw application fits setup source visit).map Prod.fst = PMF.pure visit := by
  by_cases good : goodActive application fits setup visit
  · rw [guardedDraw, if_pos good, PMF.map_comp]
    exact PMF.map_const _ _
  · rw [guardedDraw, if_neg good, PMF.pure_map]

private theorem visited_context_marginal
    (source : Statement → Payload application → PMF (SourceResult application))
    (initial : PMF (Statement × Envelope application)) (steps : Nat) :
    (HyperNovaHistoryLaw.law application fits source initial).map (fun sample => (observedDraw application fits setup steps sample).1) =
      visitedLaw application fits setup source initial steps := by
  calc
    _ = ((HyperNovaHistoryLaw.law application fits source initial).map (observedDraw application fits setup steps)).map Prod.fst :=
      (PMF.map_comp _ _ _).symm
    _ = ((visitedLaw application fits setup source initial steps).bind (guardedDraw application fits setup source)).map Prod.fst :=
      congrArg (fun distribution => distribution.map Prod.fst)
        (visitedDraw_marginal application fits setup source initial steps)
    _ = _ := by
      rw [PMF.map_bind]
      simp_rw [guardedDraw_context]
      exact PMF.bind_pure _

private theorem supported_accepted
    (source : Statement → Payload application → PMF (SourceResult application))
    (initial : PMF (Statement × Envelope application)) (steps : Nat) (visit : Visit application)
    (supported : visit ∈ (visitedLaw application fits setup source initial steps).support) :
    MarkedAccepted application fits setup visit := by
  rw [← visited_context_marginal application fits setup source initial steps] at supported
  rcases (PMF.mem_support_map_iff _ _ _).mp supported with ⟨sample, _produced, same⟩
  rw [← same]
  exact (readDraw_accepted application fits setup) steps (initialVisit application fits setup (sample.1, sample.2.1)) sample.2.2
    (initial_accepted application fits setup (sample.1, sample.2.1))

/-- Every marked context in the actual visited law contains an accepted
selected terminal opening. Initial acceptance and every predecessor's source
membership are derived from the mark construction; none is a new premise. -/
theorem marked_accepted
    (source : Statement → Payload application → PMF (SourceResult application))
    (initial : PMF (Statement × Envelope application)) (steps : Nat) (visit : Visit application)
    (supported : visit ∈ (visitedLaw application fits setup source initial steps).support)
    (marked : visit.2 = true) :
    ∃ statement proof, visit.1 = some (statement, proof) ∧
      PerApplicationTerminal.Holds application fits setup statement proof := by
  have accepted := supported_accepted application fits setup source initial steps visit supported marked
  cases current : visit.1 with
  | none => exact False.elim (by simp only [current] at accepted)
  | some input =>
      exact ⟨input.1, input.2, rfl, by simpa only [current] using accepted⟩

/-- A good active supported visit holds an accepted, collision-free, non-base
recursive terminal. -/
private theorem good_terminal
    (source : Statement → Payload application → PMF (SourceResult application))
    (initial : PMF (Statement × Envelope application)) (steps : Nat) (visit : Visit application)
    (supported : visit ∈ (visitedLaw application fits setup source initial steps).support)
    (good : goodActive application fits setup visit) :
    ∃ statement payload, visit.1 = some (statement, .recursive payload) ∧
      PerApplicationTerminal.Holds application fits setup statement
        (.recursive payload) ∧
      ¬ Collision application fits setup statement payload ∧ 0 < (decodedInput application fits payload).iteration := by
  rcases (marked_accepted application fits setup) source initial steps visit supported good.1 with
    ⟨statement, proof, current, accepted⟩
  cases proof with
  | bottom =>
      have impossible := good.2.1
      simp only [ready, current] at impossible
  | recursive payload =>
      have active : statement.iteration ≠ 0 ∧
          (decodedInput application fits payload).iteration + 1 = statement.iteration ∧
          (decodedInput application fits payload).iteration ≠ 0 := by
        simpa only [ready, current] using good.2.1
      have safe : ¬ Collision application fits setup statement payload := by
        have safe := good.2.2
        change ¬ (match visit.1 with
          | some (statement, .recursive payload) => Collision application fits setup statement payload
          | _ => False) at safe
        simpa only [current] using safe
      exact ⟨statement, payload, current, accepted, safe, Nat.pos_of_ne_zero active.2.2⟩

/-- At every supported visited context, the guarded real verifier event is
exactly the good active mark. This includes the original stopped and abort
mass, with no renormalization or accepted-context law as input. -/
theorem realSuccess_iff_goodActive
    (source : Statement → Payload application → PMF (SourceResult application))
    (initial : PMF (Statement × Envelope application)) (steps : Nat) (visit : Visit application)
    (supported : visit ∈ (visitedLaw application fits setup source initial steps).support) :
    NifsRealSuccess.RealSuccess (PerApplicationFixedPoint.relation application fits) (PerApplicationCanonicalPackage.commitmentKey setup)
      (PerApplicationCanonicalPackage.verifierContextDigest fits setup)
      (PiCCSInputCheck.running (inputs application fits visit)) (PiCCSInputCheck.fresh (inputs application fits visit))
      (realOutput application fits setup visit) ↔ goodActive application fits setup visit := by
  by_cases good : goodActive application fits setup visit
  · refine ⟨fun _ => good, fun _ => ?_⟩
    rcases (good_terminal application fits setup) source initial steps visit supported good with
      ⟨statement, payload, current, accepted, safe, positive⟩
    have outputEq : realOutput application fits setup visit = some (HyperNovaRealInput.output application fits setup payload) := by
      simp only [realOutput, if_pos good, current]
    rw [outputEq]
    simpa only [inputs, current] using
      HyperNovaRealInput.realSuccess_of_terminal application fits setup statement payload accepted safe positive
  · rw [HyperNovaGuardedSourceLaw.realOutput_off application fits setup visit good]
    exact iff_of_false id good

end NightstreamFPrime.Export.Stage1.HyperNovaVisitedAcceptance
