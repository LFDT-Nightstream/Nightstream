import NightstreamFPrime.Export.Stage1.Wide.TerminalSecurity
import NightstreamFPrime.Export.Stage1.HyperNovaInput
import NightstreamFPrime.Export.Stage1.HyperNovaSource

/-!
Owns the selected deterministic reverse history from HyperNova Appendix H.3.
The caller supplies the actual NIFS source results in reverse chronological
order. The walk decodes each current fresh opening, constructs its exact
predecessor from the supplied source value, and collects application advice.
The final base step consumes no source result.

Correctness requires the existing source-success event and absence of the
existing state-hash collision at the steps actually visited. This is neither
a probabilistic extractor construction nor a runtime bound.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaHistory

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open Wide (Target)

abbrev Statement := TerminalStatement AppState

variable (target : Target)

abbrev SourceValues := WitnessProjection.SourceWitness productionShape
  (PiCCSStoredWitnessCheck.carrier target.security)
abbrev SourceResult := Option (SourceValues target)

/-- The exact prior claims and local proof supplied to the source checker. -/
def sourceInput (payload : target.Payload) : PiCCSInputCheck.Input :=
  HyperNovaInput.ofClaims ((target.decodedInput payload).running functionIndex)
    (target.decodedInput payload).fresh (target.decodedInput payload).nifsProof

/-- The prior public state advertised by the decoded local step. -/
def predecessorStatement (payload : target.Payload) : Statement :=
  let input := target.decodedInput payload
  { iteration := input.iteration, z0 := input.z0, zi := input.zi }

/-- Reconstruct the prior terminal opening from the actual source return. -/
def predecessorPayload (payload : target.Payload) (values : SourceValues target) : target.Payload :=
  let input := target.decodedInput payload
  { running := input.running
    runningWitness := fun _ => HyperNovaSource.runningWitness target.security values
    fresh := input.fresh
    freshWitness := HyperNovaSource.freshWitness target.security (sourceInput target payload) values
    pc := input.priorPc }

/-- The selected source checker succeeds for the exact decoded prior input. -/
def SourceSucceeded (payload : target.Payload) (result : SourceResult target) : Prop :=
  CheckedWitnessExtraction.SourceReturned (PiCCSStoredWitnessCheck.commit target.security)
    productionGlobalParams (PiCCSStoredWitnessCheck.statement target.security (sourceInput target payload))
    result

/-- Explicit failures of the supplied reverse walk. A failed source check is
represented by the actual `none` result supplied by its caller. -/
inductive Failure where
  | unexpectedEnvelope
  | initialStateMismatch
  | counterMismatch
  | missingSource
  | sourceAbort
  deriving DecidableEq, Repr

private def walk (statement : Statement) (proof : target.Envelope)
    (results : List (SourceResult target)) (suffix : List AppWitness) :
    Except Failure (List AppWitness × List (SourceResult target)) :=
  match proof with
  | .bottom =>
      if statement.iteration = 0 then
        if statement.zi = statement.z0 then .ok (suffix, results)
        else .error .initialStateMismatch
      else .error .unexpectedEnvelope
  | .recursive payload =>
      if statement.iteration = 0 then .error .unexpectedEnvelope
      else
        let input := target.decodedInput payload
        if input.iteration + 1 = statement.iteration then
          if input.iteration = 0 then .ok (input.witness :: suffix, results)
          else
            match results with
            | [] => .error .missingSource
            | none :: _ => .error .sourceAbort
            | some values :: tail =>
                walk (predecessorStatement target payload)
                  (.recursive (predecessorPayload target payload values)) tail
                  (input.witness :: suffix)
        else .error .counterMismatch
termination_by results.length
decreasing_by simp_wf

/-- Recover forward-ordered advice from the actual returned source values.
The remaining list contains exactly the values this walk did not consume. -/
def run (statement : Statement) (proof : target.Envelope) (results : List (SourceResult target)) :
    Except Failure (List AppWitness × List (SourceResult target)) :=
  walk target statement proof results []

/-- Exact source-return success at every source read reached by `run`.
The base branch and paths that stop before a source read add no source event.
An absent required list entry fails this condition. -/
def SuccessfulSources (statement : Statement) (proof : target.Envelope)
    (results : List (SourceResult target)) : Prop :=
  match proof with
  | .bottom => True
  | .recursive payload =>
      if statement.iteration = 0 then True
      else
        let input := target.decodedInput payload
        if input.iteration + 1 = statement.iteration then
          if input.iteration = 0 then True
          else
            match results with
            | [] => False
            | result :: tail =>
                SourceSucceeded target payload result ∧
                  match result with
                  | none => True
                  | some values => SuccessfulSources (predecessorStatement target payload)
                      (.recursive (predecessorPayload target payload values)) tail
        else True
termination_by results.length
decreasing_by simp_wf

/-- No existing state-hash collision occurs at any decoded recursive payload
visited by the walk. This is a condition on this supplied finite history,
not a global collision-resistance axiom. -/
def NoStateHashCollisions (statement : Statement) (proof : target.Envelope)
    (results : List (SourceResult target)) : Prop :=
  match proof with
  | .bottom => True
  | .recursive payload =>
      if statement.iteration = 0 then True
      else
        ¬ target.Collision statement payload ∧
          let input := target.decodedInput payload
          if input.iteration + 1 = statement.iteration then
            if input.iteration = 0 then True
            else
              match results with
              | some values :: tail => NoStateHashCollisions (predecessorStatement target payload)
                  (.recursive (predecessorPayload target payload values)) tail
              | _ => True
          else True
termination_by results.length
decreasing_by simp_wf

private theorem source_memberships (payload : target.Payload) (values : SourceValues target)
    (success : SourceSucceeded target payload (some values)) :
    Lifecycle.TerminalHolds target.relation target.ajtai
      ((target.decodedInput payload).running functionIndex)
      (HyperNovaSource.runningWitness target.security values) (target.decodedInput payload).fresh
      (HyperNovaSource.freshWitness target.security (sourceInput target payload) values) := by
  rcases (HyperNovaSource.sourceReturned_iff_terminalHolds target.security
    (sourceInput target payload) (some values)).mp success with ⟨returned, same, memberships⟩
  have equal : returned = values := (Option.some.inj same).symm
  subst returned
  simp only [SecurityInstance.running, SecurityInstance.fresh, sourceInput,
    HyperNovaInput.running_ofClaims, HyperNovaInput.fresh_ofClaims] at memberships
  exact memberships

private theorem source_none (payload : target.Payload) : ¬ SourceSucceeded target payload none := by
  rintro ⟨values, impossible, _⟩
  cases impossible

/-- Accepted terminal membership without the existing state-hash collision
identifies the decoded predecessor counter with the public successor counter. -/
theorem counter_matches (statement : Statement) (payload : target.Payload)
    (accepted : target.Holds statement (.recursive payload))
    (safe : ¬ target.Collision statement payload) :
    (target.decodedInput payload).iteration + 1 = statement.iteration := by
  rcases target.terminal_implies_matchingStepOrCollision statement payload accepted with
    ⟨_step, same⟩ | collision
  · exact congrArg (fun preimage => preimage.iteration) same
  · exact False.elim (safe collision)

/-- A checked source return reconstructs the accepted predecessor of a non-base
terminal opening when the existing current state-hash collision does not occur.
The initial state and application transition are preserved exactly. -/
theorem predecessor_accepted (statement : Statement) (payload : target.Payload)
    (values : SourceValues target) (positive : 0 < (target.decodedInput payload).iteration)
    (accepted : target.Holds statement (.recursive payload))
    (source : SourceSucceeded target payload (some values))
    (safe : ¬ target.Collision statement payload) :
    (target.decodedInput payload).z0 = statement.z0 ∧
      statement.zi = target.program.step (target.decodedInput payload).zi
          (target.decodedInput payload).witness ∧
      target.Holds (predecessorStatement target payload)
        (.recursive (predecessorPayload target payload values)) := by
  have memberships := source_memberships target payload values source
  rcases target.terminal_implies_predecessorOrCollision statement payload
      (HyperNovaSource.runningWitness target.security values)
      (HyperNovaSource.freshWitness target.security (sourceInput target payload) values) accepted
      (fun _ => memberships) with ⟨_counter, initial, transition, previous⟩ | collision
  · refine ⟨initial, transition, ?_⟩
    simp only [if_neg (Nat.ne_of_gt positive)] at previous
    exact previous
  · exact False.elim (safe collision)

private theorem foldl_singleton_append {State Advice : Type}
    (step : State → Advice → State) (initial : State) (earlier : List Advice) (last : Advice) :
    (earlier ++ [last]).foldl step initial = step (earlier.foldl step initial) last := by
  rw [List.foldl_append]
  rfl

private theorem walk_correct (statement : Statement) (proof : target.Envelope)
    (results : List (SourceResult target)) (suffix : List AppWitness)
    (accepted : target.Holds statement proof)
    (sources : SuccessfulSources target statement proof results)
    (safe : NoStateHashCollisions target statement proof results) :
    ∃ earlier unused,
      walk target statement proof results suffix = .ok (earlier ++ suffix, unused) ∧
      earlier.length = statement.iteration ∧
      earlier.foldl target.program.step statement.z0 = statement.zi := by
  cases proof with
  | bottom =>
      rcases (Lifecycle.Stage1.Terminal.holdsFor_bottom_iff target.relation target.ajtai
        target.context target.program statement).mp accepted with ⟨_valid, zero, initial⟩
      refine ⟨[], results, ?_, ?_, ?_⟩
      · rw [walk]
        simp only [if_pos zero, if_pos initial, List.nil_append]
      · exact zero.symm
      · exact initial.symm
  | recursive payload =>
      have terminalChecks := (Lifecycle.Stage1.Terminal.holdsFor_recursive_iff target.relation
        target.ajtai target.context target.program statement payload).mp accepted
      rcases terminalChecks with ⟨_valid, _pcValid, positive, _public, _running, _fresh⟩
      have notZero : statement.iteration ≠ 0 := Nat.ne_of_gt positive
      have currentSafe : ¬ target.Collision statement payload := by
        have unfolded := safe
        rw [NoStateHashCollisions.eq_def] at unfolded
        simp only [if_neg notZero] at unfolded
        exact unfolded.1
      have counter := counter_matches target statement payload accepted currentSafe
      by_cases zero : (target.decodedInput payload).iteration = 0
      · have first : statement.iteration = 1 := by omega
        rcases target.terminal_one_implies_baseOrCollision statement payload first accepted with
          ⟨_bottom, transition⟩ | collision
        · refine ⟨[(target.decodedInput payload).witness], results, ?_, ?_, ?_⟩
          · rw [walk]
            simp only [if_neg notZero, if_pos counter, if_pos zero,
              List.singleton_append]
          · simpa only [List.length_singleton] using first.symm
          · exact transition.symm
        · exact False.elim (currentSafe collision)
      · have previousPositive : 0 < (target.decodedInput payload).iteration := Nat.pos_of_ne_zero zero
        cases resultsEq : results with
        | nil =>
            rw [SuccessfulSources.eq_def] at sources
            have impossible : False := by
              simpa only [if_neg notZero, if_pos counter, if_neg zero, resultsEq] using sources
            exact False.elim impossible
        | cons result tail =>
            have sourceAndTail : SourceSucceeded target payload result ∧
                (match result with
                | none => True
                | some values => SuccessfulSources target (predecessorStatement target payload)
                    (.recursive (predecessorPayload target payload values)) tail) := by
              rw [SuccessfulSources.eq_def] at sources
              cases result <;>
                simpa only [if_neg notZero, if_pos counter, if_neg zero, resultsEq] using sources
            cases result with
            | none => exact False.elim (source_none target payload sourceAndTail.1)
            | some values =>
                have tailSafe : NoStateHashCollisions target (predecessorStatement target payload)
                    (.recursive (predecessorPayload target payload values)) tail := by
                  have conjunction : ¬ target.Collision statement payload ∧
                      NoStateHashCollisions target (predecessorStatement target payload)
                        (.recursive (predecessorPayload target payload values)) tail := by
                    rw [NoStateHashCollisions.eq_def] at safe
                    simpa only [if_neg notZero, if_pos counter, if_neg zero, resultsEq] using safe
                  exact conjunction.2
                rcases predecessor_accepted target statement payload values previousPositive
                    accepted sourceAndTail.1 currentSafe with ⟨initial, transition, previous⟩
                rcases walk_correct (predecessorStatement target payload)
                    (.recursive (predecessorPayload target payload values)) tail
                    ((target.decodedInput payload).witness :: suffix)
                    previous sourceAndTail.2 tailSafe with
                  ⟨earlier, unused, executed, length, reached⟩
                refine ⟨earlier ++ [(target.decodedInput payload).witness], unused, ?_, ?_, ?_⟩
                · rw [walk]
                  simpa only [if_neg notZero, if_pos counter, if_neg zero,
                    resultsEq, List.append_assoc, List.singleton_append] using executed
                · rw [List.length_append, List.length_singleton]
                  dsimp only [predecessorStatement] at length
                  exact (congrArg (fun count => count + 1) length).trans counter
                · rw [foldl_singleton_append, ← initial]
                  exact (congrArg
                      (fun state => target.program.step state (target.decodedInput payload).witness)
                    reached).trans transition.symm
termination_by results.length
decreasing_by
  simp_wf
  rw [resultsEq]
  exact Nat.lt_succ_self _

/-- Every successful supplied source return, together with absence of the
encountered state-hash collisions, yields exactly the advertised number of
application witnesses. Executing them in forward order reaches the public
terminal state. No existence or efficiency assumption about an extractor is
added; the source results are the explicit inputs consumed by `run`. -/
theorem run_correct (statement : Statement) (proof : target.Envelope) (results : List (SourceResult target))
    (accepted : target.Holds statement proof)
    (sources : SuccessfulSources target statement proof results)
    (safe : NoStateHashCollisions target statement proof results) :
    ∃ advice unused,
      run target statement proof results = .ok (advice, unused) ∧
      advice.length = statement.iteration ∧
      advice.foldl target.program.step statement.z0 = statement.zi := by
  rcases walk_correct target statement proof results [] accepted sources safe with
    ⟨advice, unused, executed, length, reached⟩
  exact ⟨advice, unused, by simpa only [run, List.append_nil] using executed, length, reached⟩

end NightstreamFPrime.Export.Stage1.HyperNovaHistory
