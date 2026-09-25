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
open Poseidon2HashChainV1Package (application fits)
open Poseidon2HashChainV1Setup (productionSetup productionAjtaiKey)

attribute [local instance] Classical.propDecidable

private def MarkedAccepted (visit : Visit) : Prop :=
  visit.2 = true →
    match visit.1 with
    | none => False
    | some (statement, proof) =>
        PerApplicationTerminal.Holds application fits productionSetup statement proof

private theorem stopped_accepted : MarkedAccepted stopped := by
  simp only [MarkedAccepted, stopped, Bool.false_eq_true, false_implies]

private theorem initial_accepted (input : Statement × Envelope) :
    MarkedAccepted (initialVisit input) := by
  intro marked
  exact of_decide_eq_true marked

private theorem advance_accepted (visit : Visit) (result : SourceResult)
    (accepted : MarkedAccepted visit) : MarkedAccepted (advance visit result) := by
  by_cases active : ready visit
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
                simpa only [advance, if_pos active] using stopped_accepted
            | some values =>
                simp only [advance, if_pos active, MarkedAccepted]
                intro marked
                have marks : mark = true ∧
                    decide (¬ Collision statement payload ∧
                      SourceSucceeded payload (some values)) = true := by
                  simpa only [Bool.and_eq_true] using marked
                have events : ¬ Collision statement payload ∧
                    SourceSucceeded payload (some values) := of_decide_eq_true marks.2
                have positive : 0 < (decodedInput payload).iteration :=
                  Nat.pos_of_ne_zero active.2.2
                exact (HyperNovaHistory.predecessor_accepted statement payload values
                  positive (accepted marks.1) events.2 events.1).2.2
  · simpa only [advance, if_neg active] using stopped_accepted

private theorem readDraw_accepted (steps : Nat) (visit : Visit)
    (results : List SourceResult) (accepted : MarkedAccepted visit) :
    MarkedAccepted (readDraw steps visit results).1 := by
  induction steps generalizing visit results with
  | zero => exact accepted
  | succ steps induction =>
      by_cases active : ready visit
      · cases results with
        | nil =>
            simpa only [readDraw, if_pos active] using
              induction stopped [] stopped_accepted
        | cons result tail =>
            simpa only [readDraw, if_pos active] using
              induction (advance visit result) tail (advance_accepted visit result accepted)
      · simpa only [readDraw, if_neg active] using
          induction stopped results stopped_accepted

private theorem guardedDraw_context
    (source : Statement → Payload → PMF SourceResult) (visit : Visit) :
    (guardedDraw source visit).map Prod.fst = PMF.pure visit := by
  by_cases good : goodActive visit
  · rw [guardedDraw, if_pos good, PMF.map_comp]
    exact PMF.map_const _ _
  · rw [guardedDraw, if_neg good, PMF.pure_map]

private theorem visited_context_marginal
    (source : Statement → Payload → PMF SourceResult)
    (initial : PMF (Statement × Envelope)) (steps : Nat) :
    (HyperNovaHistoryLaw.law source initial).map (fun sample => (observedDraw steps sample).1) =
      visitedLaw source initial steps := by
  calc
    _ = ((HyperNovaHistoryLaw.law source initial).map (observedDraw steps)).map Prod.fst :=
      (PMF.map_comp _ _ _).symm
    _ = ((visitedLaw source initial steps).bind (guardedDraw source)).map Prod.fst :=
      congrArg (fun distribution => distribution.map Prod.fst)
        (visitedDraw_marginal source initial steps)
    _ = _ := by
      rw [PMF.map_bind]
      simp_rw [guardedDraw_context]
      exact PMF.bind_pure _

private theorem supported_accepted
    (source : Statement → Payload → PMF SourceResult)
    (initial : PMF (Statement × Envelope)) (steps : Nat) (visit : Visit)
    (supported : visit ∈ (visitedLaw source initial steps).support) :
    MarkedAccepted visit := by
  rw [← visited_context_marginal source initial steps] at supported
  rcases (PMF.mem_support_map_iff _ _ _).mp supported with ⟨sample, _produced, same⟩
  rw [← same]
  exact readDraw_accepted steps (initialVisit (sample.1, sample.2.1)) sample.2.2
    (initial_accepted (sample.1, sample.2.1))

/-- Every marked context in the actual visited law contains an accepted
selected terminal opening. Initial acceptance and every predecessor's source
membership are derived from the mark construction; none is a new premise. -/
theorem marked_accepted
    (source : Statement → Payload → PMF SourceResult)
    (initial : PMF (Statement × Envelope)) (steps : Nat) (visit : Visit)
    (supported : visit ∈ (visitedLaw source initial steps).support)
    (marked : visit.2 = true) :
    ∃ statement proof, visit.1 = some (statement, proof) ∧
      PerApplicationTerminal.Holds application fits productionSetup statement proof := by
  have accepted := supported_accepted source initial steps visit supported marked
  cases current : visit.1 with
  | none => exact False.elim (by simpa only [current] using accepted)
  | some input =>
      exact ⟨input.1, input.2, rfl, by simpa only [current] using accepted⟩

/-- At every supported visited context, the guarded real verifier event is
exactly the good active mark. This includes the original stopped and abort
mass, with no renormalization or accepted-context law as input. -/
theorem realSuccess_iff_goodActive
    (source : Statement → Payload → PMF SourceResult)
    (initial : PMF (Statement × Envelope)) (steps : Nat) (visit : Visit)
    (supported : visit ∈ (visitedLaw source initial steps).support) :
    FiatShamirTransfer.RealSuccess PiDECInputCheck.relation productionAjtaiKey
      (PiCCSInputCheck.running (inputs visit)) (PiCCSInputCheck.fresh (inputs visit))
      (realOutput visit) ↔ goodActive visit := by
  by_cases good : goodActive visit
  · refine ⟨fun _ => good, fun _ => ?_⟩
    rcases marked_accepted source initial steps visit supported good.1 with
      ⟨statement, proof, current, accepted⟩
    cases proof with
    | bottom =>
        have impossible := good.2.1
        simp only [ready, current] at impossible
    | recursive payload =>
        have active : statement.iteration ≠ 0 ∧
            (decodedInput payload).iteration + 1 = statement.iteration ∧
            (decodedInput payload).iteration ≠ 0 := by
          simpa only [ready, current] using good.2.1
        have safe : ¬ Collision statement payload := by
          have safe := good.2.2
          change ¬ (match visit.1 with
            | some (statement, .recursive payload) => Collision statement payload
            | _ => False) at safe
          simpa only [current] using safe
        have outputEq : realOutput visit = some (HyperNovaRealInput.output payload) := by
          simp only [realOutput, if_pos good, current]
        rw [outputEq]
        simpa only [inputs, current] using
          HyperNovaRealInput.realSuccess_of_terminal statement payload accepted safe
            (Nat.pos_of_ne_zero active.2.2)
  · rw [HyperNovaGuardedSourceLaw.realOutput_off visit good]
    exact iff_of_false id good

private theorem realSuccessProbability_eq_event
    (distribution : PMF (Visit × Option (FiatShamirTransfer.RealOutput PiDECInputCheck.relation))) :
    FiatShamirTransfer.realSuccessProbability PiDECInputCheck.relation productionAjtaiKey
      (fun visit => PiCCSInputCheck.running (inputs visit))
      (fun visit => PiCCSInputCheck.fresh (inputs visit)) distribution =
      (distribution.toOuterMeasure {sample |
        FiatShamirTransfer.RealSuccess PiDECInputCheck.relation productionAjtaiKey
          (PiCCSInputCheck.running (inputs sample.1)) (PiCCSInputCheck.fresh (inputs sample.1))
          sample.2}).toReal := by
  unfold FiatShamirTransfer.realSuccessProbability
  rw [PMF.toOuterMeasure_apply, ENNReal.tsum_toReal_eq (fun sample => by
    by_cases success : FiatShamirTransfer.RealSuccess PiDECInputCheck.relation productionAjtaiKey
        (PiCCSInputCheck.running (inputs sample.1)) (PiCCSInputCheck.fresh (inputs sample.1)) sample.2
    · simpa only [Set.indicator, Set.mem_setOf_eq, if_pos success] using
        distribution.apply_ne_top sample
    · simp only [Set.indicator, Set.mem_setOf_eq, if_neg success]
      exact ENNReal.zero_ne_top)]
  apply tsum_congr
  intro sample
  by_cases success : FiatShamirTransfer.RealSuccess PiDECInputCheck.relation productionAjtaiKey
      (PiCCSInputCheck.running (inputs sample.1)) (PiCCSInputCheck.fresh (inputs sample.1)) sample.2
  · simp only [Set.indicator, Set.mem_setOf_eq, if_pos success]
  · simp only [Set.indicator, Set.mem_setOf_eq, if_neg success, ENNReal.toReal_zero]

/-- The real success probability used by the approved FS transfer is the exact
mass of good active visits in this actual history law. The source kernel is
arbitrary; success of its earlier returns is checked by the analytical mark. -/
theorem realSuccessProbability_eq_goodActive
    (source : Statement → Payload → PMF SourceResult)
    (initial : PMF (Statement × Envelope)) (steps : Nat) :
    FiatShamirTransfer.realSuccessProbability PiDECInputCheck.relation productionAjtaiKey
      (fun visit => PiCCSInputCheck.running (inputs visit))
      (fun visit => PiCCSInputCheck.fresh (inputs visit))
      (realLaw (visitedLaw source initial steps)) =
      ((visitedLaw source initial steps).toOuterMeasure {visit | goodActive visit}).toReal := by
  rw [realSuccessProbability_eq_event, realLaw, PMF.toOuterMeasure_map_apply]
  apply congrArg ENNReal.toReal
  apply PMF.toOuterMeasure_apply_eq_of_inter_support_eq
  ext visit
  constructor
  · rintro ⟨success, supported⟩
    exact ⟨(realSuccess_iff_goodActive source initial steps visit supported).mp success, supported⟩
  · rintro ⟨good, supported⟩
    exact ⟨(realSuccess_iff_goodActive source initial steps visit supported).mpr good, supported⟩

end NightstreamFPrime.Export.Stage1.HyperNovaVisitedAcceptance
