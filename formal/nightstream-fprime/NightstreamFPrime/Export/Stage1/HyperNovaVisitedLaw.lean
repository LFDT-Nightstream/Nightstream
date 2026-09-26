import NightstreamFPrime.Export.Stage1.HyperNovaHistoryLaw

/-!
Owns the joint law of a visited reverse-history context and its source
result. The mark records initial terminal acceptance and the exact preceding
source-success and no-collision events. These marks are analytical; this
module claims no efficient sampler or adversary translation.

Operationally active states draw their actual source kernel even when the
mark is false. Only the reported guarded draw is masked. A stopped state is
reported once and then becomes the absorbing `none` state.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaVisitedLaw

open HyperNovaHistory

variable (target : Wide.Target)

attribute [local instance] Classical.propDecidable

/-- The current existing terminal pair and its analytical good-prefix mark.
There is no protocol message or additional witness representation. -/
abbrev Visit := Option (Statement × target.Envelope) × Bool

/-- Every stopped or aborted visit reaches this absorbing state. -/
def stopped : Visit target := (none, false)

/-- The initial mark is the actual selected terminal-acceptance predicate. -/
noncomputable def initialVisit (input : Statement × target.Envelope) : Visit target :=
  (some input, decide (target.Holds input.1 input.2))

/-- Exactly the source-call guard in the selected reverse program. -/
def ready (visit : Visit target) : Prop :=
  match visit.1 with
  | some (statement, .recursive payload) =>
      statement.iteration ≠ 0 ∧
        (target.decodedInput payload).iteration + 1 = statement.iteration ∧
        (target.decodedInput payload).iteration ≠ 0
  | _ => False

private def collisionAt (visit : Visit target) : Prop :=
  match visit.1 with
  | some (statement, .recursive payload) => target.Collision statement payload
  | _ => False

/-- The first-failure source event is observed only before any prior failure
and when this visit has no current state-hash collision. -/
def goodActive (visit : Visit target) : Prop :=
  visit.2 = true ∧ ready target visit ∧ ¬ collisionAt target visit

/-- Advance using the actual source value regardless of the prior mark.
Only the analytical mark tests source membership and the current collision. -/
noncomputable def advance (visit : Visit target) (result : SourceResult target) : Visit target :=
  if ready target visit then
    match visit.1 with
    | some (statement, .recursive payload) =>
        match result with
        | none => stopped target
        | some values =>
            (some (predecessorStatement target payload, .recursive
                (predecessorPayload target payload values)),
              visit.2 && decide
                  (¬ target.Collision statement payload ∧ SourceSucceeded target payload result))
    | _ => stopped target
  else stopped target

variable (source : Statement → target.Payload → PMF (SourceResult target))

/-- The actual source draw, including draws on false-mark paths. Inactive
states make no source call and have the unique absent result. -/
noncomputable def draw (visit : Visit target) : PMF (SourceResult target) :=
  if ready target visit then
    match visit.1 with
    | some (statement, .recursive payload) => source statement payload
    | _ => PMF.pure none
  else PMF.pure none

/-- One transition of the actual visited-state law. It does not filter by
source validity, terminal acceptance, or collision resistance. -/
noncomputable def transition (visit : Visit target) : PMF (Visit target) :=
  (draw target source visit).map (advance target visit)

private noncomputable def after : Nat → Visit target → PMF (Visit target)
  | 0, visit => PMF.pure visit
  | steps + 1, visit => (transition target source visit).bind (after steps)

/-- The normalized law after the given number of visited transitions.
Stopped and abort mass is retained; `steps` is an observation index. -/
noncomputable def visitedLaw (initial : PMF (Statement × target.Envelope)) (steps : Nat) : PMF
    (Visit target) :=
  initial.bind fun input => after target source steps (initialVisit target input)

private theorem after_succ_bind (steps : Nat) (visit : Visit target) :
    after target source (steps + 1) visit = (after target source steps visit).bind
        (transition target source) := by
  induction steps generalizing visit with
  | zero => simp only [after, PMF.pure_bind, PMF.bind_pure]
  | succ steps induction =>
      change (transition target source visit).bind (after target source (steps + 1)) =
        ((transition target source visit).bind (after target source steps)).bind (transition target source)
      rw [PMF.bind_bind]
      apply congrArg (PMF.bind (transition target source visit))
      funext next
      exact induction next

/-- The first visited law is the original initial law with its actual
acceptance mark. No terminal opening is filtered or replaced. -/
theorem visitedLaw_zero (initial : PMF (Statement × target.Envelope)) :
    visitedLaw target source initial 0 = initial.map (initialVisit target) := rfl

/-- Each observation follows one more actual operational transition, with
all false-mark, inactive and abort mass retained. -/
theorem visitedLaw_succ (initial : PMF (Statement × target.Envelope)) (steps : Nat) :
    visitedLaw target source initial (steps + 1) =
      (visitedLaw target source initial steps).bind (transition target source) := by
  simp only [visitedLaw, after_succ_bind, PMF.bind_bind]

/-- Mask only the draw reported to the marked NIFS experiment. This does not
change the actual transition or remove stopped contexts from the law. -/
noncomputable def guardedDraw (visit : Visit target) : PMF (Visit target × SourceResult target) :=
  if goodActive target visit then (draw target source visit).map (fun result => (visit, result))
  else PMF.pure (visit, none)

/-- Read a visited context from the existing actual result list. Every head
is used for advancement before the next observation, even on false-mark
paths. An inactive visit leaves the source list unread and then stops. -/
noncomputable def readDraw : Nat → Visit target → List (SourceResult target) → Visit target ×
    SourceResult target
  | 0, visit, results => (visit, if goodActive target visit then results.headD none else none)
  | steps + 1, visit, results =>
      if ready target visit then
        match results with
        | [] => readDraw steps (stopped target) []
        | result :: tail => readDraw steps (advance target visit result) tail
      else readDraw steps (stopped target) results

/-- The observed guarded draw is a direct map of the already-generated
history sample. No successful witness or conditional sample is selected. -/
noncomputable def observedDraw (steps : Nat) (sample : HyperNovaHistoryProbability.Sample target) :
    Visit target × SourceResult target :=
  readDraw target steps (initialVisit target (sample.1, sample.2.1)) sample.2.2

private noncomputable def resultLaw (visit : Visit target) : PMF (List (SourceResult target)) :=
  match visit.1 with
  | none => PMF.pure []
  | some (statement, proof) => HyperNovaHistoryLaw.results target source statement proof

private theorem not_good_of_not_ready (visit : Visit target) (inactive : ¬ ready target visit) :
    ¬ goodActive target visit := fun good => inactive good.2.1

private theorem stopped_not_ready : ¬ ready target (stopped target) := by
  exact id

private theorem transition_inactive (visit : Visit target) (inactive : ¬ ready target visit) :
    transition target source visit = PMF.pure (stopped target) := by
  simp only [transition, draw, if_neg inactive, PMF.pure_map, advance, if_neg inactive]

private theorem after_stopped (steps : Nat) : after target source steps (stopped target) = PMF.pure
    (stopped target) := by
  induction steps with
  | zero => rfl
  | succ steps induction =>
      rw [after, transition_inactive target source (stopped target) (stopped_not_ready target),
          PMF.pure_bind, induction]

private theorem readDraw_stopped (steps : Nat) (results : List (SourceResult target)) :
    readDraw target steps (stopped target) results = (stopped target, none) := by
  induction steps with
  | zero =>
      simp only [readDraw, if_neg (not_good_of_not_ready target (stopped target) (stopped_not_ready target))]
  | succ steps induction =>
      simp only [readDraw, if_neg (stopped_not_ready target), induction]

private theorem resultLaw_inactive (visit : Visit target) (inactive : ¬ ready target visit) :
    resultLaw target source visit = PMF.pure [] := by
  rcases visit with ⟨current, mark⟩
  cases current with
  | none => rfl
  | some input =>
      rcases input with ⟨statement, proof⟩
      cases proof with
      | bottom =>
          exact HyperNovaHistoryLaw.results_eq target source statement .bottom
      | recursive payload =>
          change (HyperNovaHistoryLaw.results target) source statement (.recursive payload) = PMF.pure []
          rw [HyperNovaHistoryLaw.results_eq]
          by_cases zero : statement.iteration = 0
          · simp only [if_pos zero]
          · by_cases counter : (target.decodedInput payload).iteration + 1 = statement.iteration
            · by_cases base : (target.decodedInput payload).iteration = 0
              · simp only [if_neg zero, if_pos counter, if_pos base]
              · exact False.elim (inactive ⟨zero, counter, base⟩)
            · simp only [if_neg zero, if_neg counter]

private theorem resultLaw_active (visit : Visit target) (active : ready target visit) :
    resultLaw target source visit =
      (draw target source visit).bind fun result =>
        (resultLaw target source (advance target visit result)).map (result :: ·) := by
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
          change (HyperNovaHistoryLaw.results target) source statement (.recursive payload) = _
          rw [HyperNovaHistoryLaw.results_eq]
          simp only [if_neg nonzero, if_pos counter, if_neg positive, draw, if_pos active]
          apply congrArg (PMF.bind (source statement payload))
          funext result
          cases result with
          | none => simp only [advance, if_pos active, resultLaw, stopped, PMF.pure_map]
          | some values => simp only [advance, if_pos active, resultLaw]

private theorem readDraw_marginal (steps : Nat) (visit : Visit target) :
    (resultLaw target source visit).map (readDraw target steps visit) =
      (after target source steps visit).bind (guardedDraw target source) := by
  induction steps generalizing visit with
  | zero =>
      simp only [after, PMF.pure_bind]
      by_cases active : ready target visit
      · rw [resultLaw_active target source visit active, PMF.map_bind]
        have head (result : SourceResult target) :
            ((resultLaw target source (advance target visit result)).map (result :: ·)).map
                (readDraw target 0 visit) =
              PMF.pure (visit, if goodActive target visit then result else none) := by
          rw [PMF.map_comp]
          simpa only [Function.comp_def, readDraw, List.headD_cons, Function.const] using!
            (PMF.map_const (resultLaw target source (advance target visit result))
              (visit, if goodActive target visit then result else none))
        simp only [head]
        by_cases good : goodActive target visit
        · simp only [guardedDraw, if_pos good]
          rfl
        · simp only [guardedDraw, if_neg good, PMF.bind_const]
      · rw [resultLaw_inactive target source visit active, PMF.pure_map]
        simp only [readDraw, guardedDraw, if_neg (not_good_of_not_ready target visit active)]
  | succ steps induction =>
      by_cases active : ready target visit
      · rw [resultLaw_active target source visit active, PMF.map_bind]
        conv_rhs => rw [after, transition, PMF.bind_map, PMF.bind_bind]
        apply congrArg (PMF.bind (draw target source visit))
        funext result
        rw [PMF.map_comp]
        have reader : readDraw target (steps + 1) visit ∘ (result :: ·) =
            readDraw target steps (advance target visit result) := by
          funext tail
          simp only [Function.comp_def, readDraw, if_pos active]
        rw [reader]
        exact induction (advance target visit result)
      · rw [resultLaw_inactive target source visit active, PMF.pure_map]
        rw [after, transition_inactive target source visit active, PMF.pure_bind,
          after_stopped target source steps, PMF.pure_bind]
        simp only [readDraw, if_neg active, readDraw_stopped, guardedDraw,
          if_neg (not_good_of_not_ready target (stopped target) (stopped_not_ready target))]

/-- The guarded draw at every observation index has exactly the joint law
obtained from the actual visited contexts. False-mark paths still generate
their actual source returns, while stopped and abort mass remains present.
There is no acceptance, source-success, model, or event-equality premise. -/
theorem visitedDraw_marginal (initial : PMF (Statement × target.Envelope)) (steps : Nat) :
    (HyperNovaHistoryLaw.law target source initial).map (observedDraw target steps) =
      (visitedLaw target source initial steps).bind (guardedDraw target source) := by
  rw [HyperNovaHistoryLaw.law, PMF.map_bind, visitedLaw, PMF.bind_bind]
  apply congrArg (PMF.bind initial)
  funext input
  rw [PMF.map_comp]
  exact readDraw_marginal target source steps (initialVisit target input)

end NightstreamFPrime.Export.Stage1.HyperNovaVisitedLaw
