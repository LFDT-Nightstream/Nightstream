import NightstreamFPrime.Export.Stage1.HyperNovaPredecessor
import NightstreamFPrime.Export.Stage1.HyperNovaInput
import NightstreamFPrime.Export.Stage1.HyperNovaSource
import NightstreamFPrime.Export.Stage1.PerApplicationTerminal

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
open Poseidon2HashChainV1Package (application fits)
open Poseidon2HashChainV1Setup (productionSetup productionAjtaiKey)

abbrev Statement := TerminalStatement AppState
abbrev Envelope := PerApplicationTerminal.ProofEnvelope application
abbrev SourceValues := WitnessProjection.SourceWitness productionShape
  PiCCSStoredWitnessCheck.carrier
abbrev SourceResult := Option SourceValues

private abbrev Payload := ActualContextSecurity.TerminalPayload application

private abbrev Accepted (statement : Statement) (proof : Envelope) : Prop :=
  PerApplicationTerminal.Holds application fits productionSetup statement proof

private def decodedInput (payload : Payload) :=
  let assignment := ProductionRelation.Plan.logicalAssignment payload.freshWitness
  ActualStep.input application fits assignment
    (ActualStep.decodedFresh application assignment)
    (ActualPiDECMessages.proof application fits assignment)

private def sourceInput (payload : Payload) : PiCCSInputCheck.Input :=
  let input := decodedInput payload
  HyperNovaInput.ofClaims (input.running functionIndex) input.fresh input.nifsProof

private def predecessorStatement (payload : Payload) : Statement :=
  let input := decodedInput payload
  { iteration := input.iteration, z0 := input.z0, zi := input.zi }

private def predecessorPayload (payload : Payload) (values : SourceValues) : Payload :=
  let input := decodedInput payload
  { running := input.running
    runningWitness := fun _ => HyperNovaSource.runningWitness values
    fresh := input.fresh
    freshWitness := HyperNovaSource.freshWitness (sourceInput payload) values
    pc := input.priorPc }

private def Collision (statement : Statement) (payload : Payload) : Prop :=
  PiCCSSecurity.StateHashCollision
    (ActualContextSecurity.decodedNext application
      (ProductionRelation.Plan.logicalAssignment payload.freshWitness))
    (ActualContextSecurity.terminalPreimage application fits productionSetup statement payload)

private def SourceSucceeded (payload : Payload) (result : SourceResult) : Prop :=
  CheckedWitnessExtraction.SourceReturned PiCCSStoredWitnessCheck.commit
    productionGlobalParams (PiCCSStoredWitnessCheck.statement (sourceInput payload)) result

/-- Explicit failures of the supplied reverse walk. A failed source check is
represented by the actual `none` result supplied by its caller. -/
inductive Failure where
  | unexpectedEnvelope
  | initialStateMismatch
  | counterMismatch
  | missingSource
  | sourceAbort
  deriving DecidableEq, Repr

private def walk (statement : Statement) (proof : Envelope)
    (results : List SourceResult) (suffix : List AppWitness) :
    Except Failure (List AppWitness × List SourceResult) :=
  match proof with
  | .bottom =>
      if statement.iteration = 0 then
        if statement.zi = statement.z0 then .ok (suffix, results)
        else .error .initialStateMismatch
      else .error .unexpectedEnvelope
  | .recursive payload =>
      if statement.iteration = 0 then .error .unexpectedEnvelope
      else
        let input := decodedInput payload
        if input.iteration + 1 = statement.iteration then
          if input.iteration = 0 then .ok (input.witness :: suffix, results)
          else
            match results with
            | [] => .error .missingSource
            | none :: _ => .error .sourceAbort
            | some values :: tail =>
                walk (predecessorStatement payload)
                  (.recursive (predecessorPayload payload values)) tail
                  (input.witness :: suffix)
        else .error .counterMismatch
termination_by results.length
decreasing_by simp_wf

/-- Recover forward-ordered advice from the actual returned source values.
The remaining list contains exactly the values this walk did not consume. -/
def run (statement : Statement) (proof : Envelope) (results : List SourceResult) :
    Except Failure (List AppWitness × List SourceResult) :=
  walk statement proof results []

/-- Exact source-return success at every source read reached by `run`.
The base branch and paths that stop before a source read add no source event.
An absent required list entry fails this condition. -/
def SuccessfulSources (statement : Statement) (proof : Envelope)
    (results : List SourceResult) : Prop :=
  match proof with
  | .bottom => True
  | .recursive payload =>
      if statement.iteration = 0 then True
      else
        let input := decodedInput payload
        if input.iteration + 1 = statement.iteration then
          if input.iteration = 0 then True
          else
            match results with
            | [] => False
            | result :: tail =>
                SourceSucceeded payload result ∧
                  match result with
                  | none => True
                  | some values => SuccessfulSources (predecessorStatement payload)
                      (.recursive (predecessorPayload payload values)) tail
        else True
termination_by results.length
decreasing_by simp_wf

/-- No existing state-hash collision occurs at any decoded recursive payload
visited by the walk. This is a condition on this supplied finite history,
not a global collision-resistance axiom. -/
def NoStateHashCollisions (statement : Statement) (proof : Envelope)
    (results : List SourceResult) : Prop :=
  match proof with
  | .bottom => True
  | .recursive payload =>
      if statement.iteration = 0 then True
      else
        ¬ Collision statement payload ∧
          let input := decodedInput payload
          if input.iteration + 1 = statement.iteration then
            if input.iteration = 0 then True
            else
              match results with
              | some values :: tail => NoStateHashCollisions (predecessorStatement payload)
                  (.recursive (predecessorPayload payload values)) tail
              | _ => True
          else True
termination_by results.length
decreasing_by simp_wf

private theorem source_memberships (payload : Payload) (values : SourceValues)
    (success : SourceSucceeded payload (some values)) :
    Lifecycle.TerminalHolds (PerApplicationFixedPoint.relation application fits)
      (PerApplicationCanonicalPackage.commitmentKey productionSetup)
      ((decodedInput payload).running functionIndex) (HyperNovaSource.runningWitness values)
      (decodedInput payload).fresh (HyperNovaSource.freshWitness (sourceInput payload) values) := by
  rcases (HyperNovaSource.sourceReturned_iff_terminalHolds
    (sourceInput payload) (some values)).mp success with ⟨returned, same, memberships⟩
  have equal : returned = values := (Option.some.inj same).symm
  subst returned
  simpa only [PiDECInputCheck.relation_eq_selected, sourceInput,
    HyperNovaInput.running_ofClaims, HyperNovaInput.fresh_ofClaims] using memberships

private theorem source_none (payload : Payload) : ¬ SourceSucceeded payload none := by
  rintro ⟨values, impossible, _⟩
  cases impossible

private theorem counter_matches (statement : Statement) (payload : Payload)
    (accepted : Accepted statement (.recursive payload))
    (safe : ¬ Collision statement payload) :
    (decodedInput payload).iteration + 1 = statement.iteration := by
  rcases ActualContextSecurity.terminal_implies_matchingStepOrCollision
      application fits productionSetup statement payload accepted with ⟨_step, same⟩ | collision
  · have counter := congrArg (fun preimage => preimage.iteration) same
    dsimp only [ActualContextSecurity.decodedNext, ActualHashSlots.nextPreimage,
      StateDecoder.preimage, ActualContextSecurity.terminalPreimage] at counter
    dsimp only [decodedInput, ActualStep.input]
    exact counter
  · exact False.elim (safe collision)

private theorem predecessor_accepted (statement : Statement) (payload : Payload)
    (values : SourceValues) (positive : 0 < (decodedInput payload).iteration)
    (accepted : Accepted statement (.recursive payload))
    (source : SourceSucceeded payload (some values))
    (safe : ¬ Collision statement payload) :
    (decodedInput payload).z0 = statement.z0 ∧
      statement.zi = application.step (decodedInput payload).zi (decodedInput payload).witness ∧
      Accepted (predecessorStatement payload) (.recursive (predecessorPayload payload values)) := by
  have memberships := source_memberships payload values source
  rcases HyperNovaPredecessor.terminal_implies_predecessorOrCollision application fits
      productionSetup statement payload (HyperNovaSource.runningWitness values)
      (HyperNovaSource.freshWitness (sourceInput payload) values) accepted
      (fun _ => memberships) with ⟨_counter, initial, transition, previous⟩ | collision
  · refine ⟨initial, transition, ?_⟩
    dsimp only [decodedInput] at positive
    simp only [if_neg (Nat.ne_of_gt positive)] at previous
    dsimp only [Accepted, PerApplicationTerminal.Holds, predecessorStatement,
      predecessorPayload, decodedInput]
    exact previous
  · exact False.elim (safe collision)

private theorem foldl_singleton_append {State Advice : Type}
    (step : State → Advice → State) (initial : State) (earlier : List Advice) (last : Advice) :
    (earlier ++ [last]).foldl step initial = step (earlier.foldl step initial) last := by
  rw [List.foldl_append]
  rfl

private theorem walk_correct (statement : Statement) (proof : Envelope)
    (results : List SourceResult) (suffix : List AppWitness)
    (accepted : Accepted statement proof)
    (sources : SuccessfulSources statement proof results)
    (safe : NoStateHashCollisions statement proof results) :
    ∃ earlier unused,
      walk statement proof results suffix = .ok (earlier ++ suffix, unused) ∧
      earlier.length = statement.iteration ∧
      earlier.foldl application.step statement.z0 = statement.zi := by
  cases proof with
  | bottom =>
      rcases (PerApplicationTerminal.holds_bottom_iff application fits productionSetup statement).mp
        accepted with ⟨_valid, zero, initial⟩
      refine ⟨[], results, ?_, ?_, ?_⟩
      · rw [walk]
        simp only [if_pos zero, if_pos initial, List.nil_append]
      · exact zero.symm
      · exact initial.symm
  | recursive payload =>
      have terminalChecks := (PerApplicationTerminal.holds_recursive_iff
        application fits productionSetup statement payload).mp accepted
      rcases terminalChecks with ⟨_valid, _pcValid, positive, _public, _running, _fresh⟩
      have notZero : statement.iteration ≠ 0 := Nat.ne_of_gt positive
      have currentSafe : ¬ Collision statement payload := by
        have unfolded := safe
        rw [NoStateHashCollisions.eq_def] at unfolded
        simp only [if_neg notZero] at unfolded
        exact unfolded.1
      have counter := counter_matches statement payload accepted currentSafe
      by_cases zero : (decodedInput payload).iteration = 0
      · have first : statement.iteration = 1 := by omega
        rcases HyperNovaPredecessor.terminal_one_implies_baseOrCollision
            application fits productionSetup statement payload first accepted with
          ⟨_bottom, transition⟩ | collision
        · refine ⟨[(decodedInput payload).witness], results, ?_, ?_, ?_⟩
          · rw [walk]
            simp only [if_neg notZero, if_pos counter, if_pos zero,
              List.singleton_append]
          · simpa only [List.length_singleton] using first.symm
          · exact transition.symm
        · exact False.elim (currentSafe collision)
      · have previousPositive : 0 < (decodedInput payload).iteration := Nat.pos_of_ne_zero zero
        cases resultsEq : results with
        | nil =>
            rw [SuccessfulSources.eq_def] at sources
            have impossible : False := by
              simpa only [if_neg notZero, if_pos counter, if_neg zero, resultsEq] using sources
            exact False.elim impossible
        | cons result tail =>
            have sourceAndTail : SourceSucceeded payload result ∧
                (match result with
                | none => True
                | some values => SuccessfulSources (predecessorStatement payload)
                    (.recursive (predecessorPayload payload values)) tail) := by
              rw [SuccessfulSources.eq_def] at sources
              cases result <;>
                simpa only [if_neg notZero, if_pos counter, if_neg zero, resultsEq] using sources
            cases result with
            | none => exact False.elim (source_none payload sourceAndTail.1)
            | some values =>
                have tailSafe : NoStateHashCollisions (predecessorStatement payload)
                    (.recursive (predecessorPayload payload values)) tail := by
                  have conjunction : ¬ Collision statement payload ∧
                      NoStateHashCollisions (predecessorStatement payload)
                        (.recursive (predecessorPayload payload values)) tail := by
                    rw [NoStateHashCollisions.eq_def] at safe
                    simpa only [if_neg notZero, if_pos counter, if_neg zero, resultsEq] using safe
                  exact conjunction.2
                rcases predecessor_accepted statement payload values previousPositive
                    accepted sourceAndTail.1 currentSafe with ⟨initial, transition, previous⟩
                rcases walk_correct (predecessorStatement payload)
                    (.recursive (predecessorPayload payload values)) tail
                    ((decodedInput payload).witness :: suffix)
                    previous sourceAndTail.2 tailSafe with
                  ⟨earlier, unused, executed, length, reached⟩
                refine ⟨earlier ++ [(decodedInput payload).witness], unused, ?_, ?_, ?_⟩
                · rw [walk]
                  simpa only [if_neg notZero, if_pos counter, if_neg zero,
                    resultsEq, List.append_assoc, List.singleton_append] using executed
                · rw [List.length_append, List.length_singleton]
                  dsimp only [predecessorStatement] at length
                  exact (congrArg (fun count => count + 1) length).trans counter
                · rw [foldl_singleton_append, ← initial]
                  exact (congrArg (fun state => application.step state (decodedInput payload).witness)
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
theorem run_correct (statement : Statement) (proof : Envelope) (results : List SourceResult)
    (accepted : PerApplicationTerminal.Holds application fits productionSetup statement proof)
    (sources : SuccessfulSources statement proof results)
    (safe : NoStateHashCollisions statement proof results) :
    ∃ advice unused,
      run statement proof results = .ok (advice, unused) ∧
      advice.length = statement.iteration ∧
      advice.foldl application.step statement.z0 = statement.zi := by
  rcases walk_correct statement proof results [] accepted sources safe with
    ⟨advice, unused, executed, length, reached⟩
  exact ⟨advice, unused, by simpa only [run, List.append_nil] using executed, length, reached⟩

end NightstreamFPrime.Export.Stage1.HyperNovaHistory
