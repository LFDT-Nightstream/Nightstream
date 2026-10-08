import NightstreamFPrime.Export.Stage1.HyperNovaHistoryLaw

/-!
Owns the joint law of a visited reverse-history context and its source
result. The mark records initial terminal acceptance and the exact preceding
source-success and no-collision events. These marks are analytical; this
module claims no efficient sampler or adversary translation.

Operationally active states draw their actual source kernel even when the
mark is false. Only the reported guarded draw is masked. A stopped state is
reported once and then becomes the absorbing `none` state.

`listSource` reads a fixed result list. With it, the visited law is the single
context on that list's path (`visitedLaw_listSource`); the reverse extractor
of `HyperNovaVisitedSecurity` uses this once for each of its tapes.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaVisitedLaw

open HyperNovaHistory
open Poseidon2HashChainV1Package (application fits)
open Poseidon2HashChainV1Setup (productionSetup)

attribute [local instance] Classical.propDecidable

/-- The current existing terminal pair and its analytical good-prefix mark.
There is no protocol message or additional witness representation. -/
abbrev Visit := Option (Statement × Envelope) × Bool

/-- Every stopped or aborted visit reaches this absorbing state. -/
def stopped : Visit := (none, false)

/-- The initial mark is the actual selected terminal-acceptance predicate. -/
noncomputable def initialVisit (input : Statement × Envelope) : Visit :=
  (some input, decide (PerApplicationTerminal.Holds application fits productionSetup input.1 input.2))

/-- Exactly the source-call guard in the selected reverse program. -/
def ready (visit : Visit) : Prop :=
  match visit.1 with
  | some (statement, .recursive payload) =>
      statement.iteration ≠ 0 ∧
        (decodedInput payload).iteration + 1 = statement.iteration ∧
        (decodedInput payload).iteration ≠ 0
  | _ => False

private def collisionAt (visit : Visit) : Prop :=
  match visit.1 with
  | some (statement, .recursive payload) => Collision statement payload
  | _ => False

/-- The first-failure source event is observed only before any prior failure
and when this visit has no current state-hash collision. -/
def goodActive (visit : Visit) : Prop :=
  visit.2 = true ∧ ready visit ∧ ¬ collisionAt visit

/-- Advance using the actual source value regardless of the prior mark.
Only the analytical mark tests source membership and the current collision. -/
noncomputable def advance (visit : Visit) (result : SourceResult) : Visit :=
  if ready visit then
    match visit.1 with
    | some (statement, .recursive payload) =>
        match result with
        | none => stopped
        | some values =>
            (some (predecessorStatement payload, .recursive (predecessorPayload payload values)),
              visit.2 && decide (¬ Collision statement payload ∧ SourceSucceeded payload result))
    | _ => stopped
  else stopped

variable (source : Statement → Payload → PMF SourceResult)

/-- The actual source draw, including draws on false-mark paths. Inactive
states make no source call and have the unique absent result. -/
noncomputable def draw (visit : Visit) : PMF SourceResult :=
  if ready visit then
    match visit.1 with
    | some (statement, .recursive payload) => source statement payload
    | _ => PMF.pure none
  else PMF.pure none

/-- One transition of the actual visited-state law. It does not filter by
source validity, terminal acceptance, or collision resistance. -/
noncomputable def transition (visit : Visit) : PMF Visit :=
  (draw source visit).map (advance visit)

private noncomputable def after : Nat → Visit → PMF Visit
  | 0, visit => PMF.pure visit
  | steps + 1, visit => (transition source visit).bind (after steps)

/-- The normalized law after the given number of visited transitions.
Stopped and abort mass is retained; `steps` is an observation index. -/
noncomputable def visitedLaw (initial : PMF (Statement × Envelope)) (steps : Nat) : PMF Visit :=
  initial.bind fun input => after source steps (initialVisit input)

/-- Mask only the draw reported to the marked NIFS experiment. This does not
change the actual transition or remove stopped contexts from the law. -/
noncomputable def guardedDraw (visit : Visit) : PMF (Visit × SourceResult) :=
  if goodActive visit then (draw source visit).map (fun result => (visit, result))
  else PMF.pure (visit, none)

/-- Read a visited context from the existing actual result list. Every head
is used for advancement before the next observation, even on false-mark
paths. An inactive visit leaves the source list unread and then stops. -/
noncomputable def readDraw : Nat → Visit → List SourceResult → Visit × SourceResult
  | 0, visit, results => (visit, if goodActive visit then results.headD none else none)
  | steps + 1, visit, results =>
      if ready visit then
        match results with
        | [] => readDraw steps stopped []
        | result :: tail => readDraw steps (advance visit result) tail
      else readDraw steps stopped results

/-- The observed guarded draw is a direct map of the already-generated
history sample. No successful witness or conditional sample is selected. -/
noncomputable def observedDraw (steps : Nat) (sample : HyperNovaHistoryProbability.Sample) :
    Visit × SourceResult :=
  readDraw steps (initialVisit (sample.1, sample.2.1)) sample.2.2

private noncomputable def resultLaw (visit : Visit) : PMF (List SourceResult) :=
  match visit.1 with
  | none => PMF.pure []
  | some (statement, proof) => HyperNovaHistoryLaw.results source statement proof

private theorem not_good_of_not_ready (visit : Visit) (inactive : ¬ ready visit) :
    ¬ goodActive visit := fun good => inactive good.2.1

private theorem stopped_not_ready : ¬ ready stopped := by
  exact id

private theorem transition_inactive (visit : Visit) (inactive : ¬ ready visit) :
    transition source visit = PMF.pure stopped := by
  simp only [transition, draw, PMF.pure_map, advance, if_neg inactive]

private theorem after_stopped (steps : Nat) : after source steps stopped = PMF.pure stopped := by
  induction steps with
  | zero => rfl
  | succ steps induction =>
      rw [after, transition_inactive source stopped stopped_not_ready, PMF.pure_bind, induction]

private theorem readDraw_stopped (steps : Nat) (results : List SourceResult) :
    readDraw steps stopped results = (stopped, none) := by
  induction steps with
  | zero =>
      simp only [readDraw, if_neg (not_good_of_not_ready stopped stopped_not_ready)]
  | succ steps induction =>
      simp only [readDraw, if_neg stopped_not_ready, induction]

private theorem resultLaw_inactive (visit : Visit) (inactive : ¬ ready visit) :
    resultLaw source visit = PMF.pure [] := by
  rcases visit with ⟨current, mark⟩
  cases current with
  | none => rfl
  | some input =>
      rcases input with ⟨statement, proof⟩
      cases proof with
      | bottom =>
          exact HyperNovaHistoryLaw.results_eq source statement .bottom
      | recursive payload =>
          change HyperNovaHistoryLaw.results source statement (.recursive payload) = PMF.pure []
          rw [HyperNovaHistoryLaw.results_eq]
          by_cases zero : statement.iteration = 0
          · simp only [if_pos zero]
          · by_cases counter : (decodedInput payload).iteration + 1 = statement.iteration
            · by_cases base : (decodedInput payload).iteration = 0
              · simp only [if_neg zero, if_pos counter, if_pos base]
              · exact False.elim (inactive ⟨zero, counter, base⟩)
            · simp only [if_neg zero, if_neg counter]

private theorem resultLaw_active (visit : Visit) (active : ready visit) :
    resultLaw source visit =
      (draw source visit).bind fun result =>
        (resultLaw source (advance visit result)).map (result :: ·) := by
  rcases visit with ⟨current, mark⟩
  cases current with
  | none => exact False.elim active
  | some input =>
      rcases input with ⟨statement, proof⟩
      cases proof with
      | bottom => exact False.elim active
      | recursive payload =>
          have conditions := active
          rcases conditions with ⟨nonzero, counter, positive⟩
          change HyperNovaHistoryLaw.results source statement (.recursive payload) = _
          rw [HyperNovaHistoryLaw.results_eq]
          simp only [if_neg nonzero, if_pos counter, if_neg positive, draw, if_pos active]
          apply congrArg (PMF.bind (source statement payload))
          funext result
          cases result with
          | none => simp only [advance, if_pos active, resultLaw, stopped, PMF.pure_map]
          | some values => simp only [advance, if_pos active, resultLaw]

private theorem readDraw_marginal (steps : Nat) (visit : Visit) :
    (resultLaw source visit).map (readDraw steps visit) =
      (after source steps visit).bind (guardedDraw source) := by
  induction steps generalizing visit with
  | zero =>
      simp only [after, PMF.pure_bind]
      by_cases active : ready visit
      · rw [resultLaw_active source visit active, PMF.map_bind]
        have head (result : SourceResult) :
            ((resultLaw source (advance visit result)).map (result :: ·)).map (readDraw 0 visit) =
              PMF.pure (visit, if goodActive visit then result else none) := by
          rw [PMF.map_comp]
          simpa only [Function.comp_def, readDraw, List.headD_cons, Function.const] using!
            (PMF.map_const (resultLaw source (advance visit result))
              (visit, if goodActive visit then result else none))
        simp only [head]
        by_cases good : goodActive visit
        · simp only [guardedDraw, if_pos good]
          rfl
        · simp only [guardedDraw, if_neg good, PMF.bind_const]
      · rw [resultLaw_inactive source visit active, PMF.pure_map]
        simp only [readDraw, guardedDraw, if_neg (not_good_of_not_ready visit active)]
  | succ steps induction =>
      by_cases active : ready visit
      · rw [resultLaw_active source visit active, PMF.map_bind]
        conv_rhs => rw [after, transition, PMF.bind_map, PMF.bind_bind]
        apply congrArg (PMF.bind (draw source visit))
        funext result
        rw [PMF.map_comp]
        have reader : readDraw (steps + 1) visit ∘ (result :: ·) =
            readDraw steps (advance visit result) := by
          funext tail
          simp only [Function.comp_def, readDraw, if_pos active]
        rw [reader]
        exact induction (advance visit result)
      · rw [resultLaw_inactive source visit active, PMF.pure_map]
        rw [after, transition_inactive source visit active, PMF.pure_bind,
          after_stopped source steps, PMF.pure_bind]
        simp only [readDraw, if_neg active, readDraw_stopped, guardedDraw,
          if_neg (not_good_of_not_ready stopped stopped_not_ready)]

/-- The guarded draw at every observation index has exactly the joint law
obtained from the actual visited contexts. False-mark paths still generate
their actual source returns, while stopped and abort mass remains present.
There is no acceptance, source-success, model, or event-equality premise. -/
theorem visitedDraw_marginal (initial : PMF (Statement × Envelope)) (steps : Nat) :
    (HyperNovaHistoryLaw.law source initial).map (observedDraw steps) =
      (visitedLaw source initial steps).bind (guardedDraw source) := by
  rw [HyperNovaHistoryLaw.law, PMF.map_bind, visitedLaw, PMF.bind_bind]
  apply congrArg (PMF.bind initial)
  funext input
  rw [PMF.map_comp]
  exact readDraw_marginal source steps (initialVisit input)

private theorem after_succ (steps : Nat) (visit : Visit) :
    after source (steps + 1) visit = (after source steps visit).bind (transition source) := by
  induction steps generalizing visit with
  | zero => simp only [after, PMF.bind_pure, PMF.pure_bind]
  | succ steps induction =>
      calc after source (steps + 1 + 1) visit
          = (transition source visit).bind (after source (steps + 1)) := rfl
        _ = (transition source visit).bind
              (fun next => (after source steps next).bind (transition source)) := by
            congr 1
            funext next
            exact induction next
        _ = ((transition source visit).bind (after source steps)).bind (transition source) :=
            (PMF.bind_bind _ _ _).symm
        _ = (after source (steps + 1) visit).bind (transition source) := rfl

/-- One more observation index applies one more actual transition. -/
theorem visitedLaw_succ (initial : PMF (Statement × Envelope)) (steps : Nat) :
    visitedLaw source initial (steps + 1) =
      (visitedLaw source initial steps).bind (transition source) := by
  simp only [visitedLaw, after_succ, PMF.bind_bind]

/-! ## A source that reads a fixed result list -/

/-- The context after advancing through the first `steps` entries of a fixed
result list from the initial visit. -/
noncomputable def pathVisit (input : Statement × Envelope) (results : List SourceResult)
    (steps : Nat) : Visit :=
  (results.take steps).foldl advance (initialVisit input)

/-- The deterministic source that reads a fixed result list: a statement `j`
iterations below `top` reads entry `j`. Along the reverse path from an
initial statement at iteration `top`, visit `j` reads entry `j`. -/
noncomputable def listSource (top : Nat) (results : List SourceResult) :
    Statement → Payload → PMF SourceResult :=
  fun statement _ => PMF.pure (results.getD (top - statement.iteration) none)

private theorem pathVisit_succ (input : Statement × Envelope) (results : List SourceResult)
    (steps : Nat) (within : steps < results.length) :
    pathVisit input results (steps + 1) =
      advance (pathVisit input results steps) (results.getD steps none) := by
  simp only [pathVisit, List.take_add_one, List.getElem?_eq_getElem within, Option.toList_some,
    List.foldl_append, List.foldl_cons, List.foldl_nil, List.getD_eq_getElem _ _ within]

private theorem ready_cases (visit : Visit) (active : ready visit) :
    ∃ statement payload, visit.1 = some (statement, .recursive payload) := by
  rcases visit with ⟨current, mark⟩
  cases current with
  | none => exact False.elim active
  | some input =>
      rcases input with ⟨statement, proof⟩
      cases proof with
      | bottom => exact False.elim active
      | recursive payload => exact ⟨statement, payload, rfl⟩

/-- Every present context on the path is exactly `steps` iterations below
the initial statement. -/
private theorem pathVisit_iteration (input : Statement × Envelope) (results : List SourceResult)
    (steps : Nat) (within : steps ≤ results.length) (statement : Statement) (proof : Envelope)
    (current : (pathVisit input results steps).1 = some (statement, proof)) :
    statement.iteration + steps = input.1.iteration := by
  induction steps generalizing statement proof with
  | zero =>
      have same : some input = some (statement, proof) := current
      cases Option.some.inj same
      rfl
  | succ steps induction =>
      rw [pathVisit_succ input results steps within] at current
      by_cases active : ready (pathVisit input results steps)
      · obtain ⟨previous, payload, previousCurrent⟩ := ready_cases _ active
        have conditions := active
        simp only [ready, previousCurrent] at conditions
        have previousIteration := induction (Nat.le_of_succ_le within) previous (.recursive payload)
          previousCurrent
        cases result : results.getD steps none with
        | none =>
            simp only [advance, if_pos active, previousCurrent, result, stopped] at current
            cases current
        | some values =>
            simp only [advance, if_pos active, previousCurrent, result, Option.some.injEq,
              Prod.mk.injEq] at current
            rw [← current.1]
            simp only [predecessorStatement]
            omega
      · simp only [advance, if_neg active, stopped] at current
        cases current

private theorem draw_listSource (input : Statement × Envelope) (results : List SourceResult)
    (steps : Nat) (within : steps < results.length) (active : ready (pathVisit input results steps)) :
    draw (listSource input.1.iteration results) (pathVisit input results steps) =
      PMF.pure (results.getD steps none) := by
  obtain ⟨statement, payload, current⟩ := ready_cases _ active
  have iteration := pathVisit_iteration input results steps within.le statement
    (.recursive payload) current
  have conditions := active
  simp only [ready, current] at conditions
  have index : input.1.iteration - statement.iteration = steps := by omega
  simp only [draw, if_pos active, current, listSource, index]

/-- Along the fixed list, each actual transition reads the next entry. -/
private theorem transition_listSource (input : Statement × Envelope) (results : List SourceResult)
    (steps : Nat) (within : steps < results.length) :
    transition (listSource input.1.iteration results) (pathVisit input results steps) =
      PMF.pure (pathVisit input results (steps + 1)) := by
  rw [pathVisit_succ input results steps within]
  by_cases active : ready (pathVisit input results steps)
  · rw [transition, draw_listSource input results steps within active, PMF.pure_map]
  · rw [transition_inactive _ _ active]
    simp only [advance, if_neg active]

/-- With the fixed-list source, the visited law at every index inside the
list is the single context on the list's path. -/
theorem visitedLaw_listSource (input : Statement × Envelope) (results : List SourceResult)
    (steps : Nat) (within : steps ≤ results.length) :
    visitedLaw (listSource input.1.iteration results) (PMF.pure input) steps =
      PMF.pure (pathVisit input results steps) := by
  induction steps with
  | zero => simp only [visitedLaw, after, PMF.pure_bind, pathVisit, List.take_zero, List.foldl_nil]
  | succ steps induction =>
      rw [visitedLaw_succ, induction (Nat.le_of_succ_le within), PMF.pure_bind,
        transition_listSource input results steps within]

/-- With the fixed-list source, the reported draw at a list index is that
entry on the good active branch and absent otherwise. -/
theorem guardedDraw_listSource (input : Statement × Envelope) (results : List SourceResult)
    (steps : Nat) (within : steps < results.length) :
    guardedDraw (listSource input.1.iteration results) (pathVisit input results steps) =
      PMF.pure (pathVisit input results steps,
        if goodActive (pathVisit input results steps) then results.getD steps none else none) := by
  by_cases good : goodActive (pathVisit input results steps)
  · rw [guardedDraw, if_pos good, draw_listSource input results steps within good.2.1, PMF.pure_map,
      if_pos good]
  · rw [guardedDraw, if_neg good, if_neg good]

end NightstreamFPrime.Export.Stage1.HyperNovaVisitedLaw
