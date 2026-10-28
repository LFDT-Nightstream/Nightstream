import NightstreamFPrime.Export.Stage1.HyperNovaPredecessor
import NightstreamFPrime.Export.Stage1.HyperNovaInput
import NightstreamFPrime.Export.Stage1.HyperNovaSource
import NightstreamFPrime.Export.Stage1.PerApplicationTerminal

/-!
Owns the selected deterministic reverse history from HyperNova Appendix H.3.
The caller supplies the actual NIFS source results in reverse chronological
order.

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
variable (application : Lifecycle.Stage1.Application.Program)
  (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
  (setup : PerApplicationCanonicalPackage.CommitmentSetup application)

abbrev Statement := TerminalStatement AppState
abbrev Envelope := PerApplicationTerminal.ProofEnvelope application
abbrev SourceValues := WitnessProjection.SourceWitness productionShape
  (PiCCSStoredWitnessCheck.carrier application)
abbrev SourceResult := Option (SourceValues application)

abbrev Payload := ActualContextSecurity.TerminalPayload application

private abbrev Accepted (statement : Statement) (proof : Envelope application) : Prop :=
  PerApplicationTerminal.Holds application fits setup statement proof

def decodedInput (payload : Payload application) :=
  let assignment := ProductionRelation.Plan.logicalAssignment payload.freshWitness
  ActualStep.input application fits assignment
    (ActualStep.decodedFresh application assignment)
    (ActualPiDECMessages.proof application fits assignment)

def sourceInput (payload : Payload application) : PiCCSInputCheck.Input :=
  let input := decodedInput application fits payload
  HyperNovaInput.ofClaims (input.running functionIndex) input.fresh input.nifsProof

def predecessorStatement (payload : Payload application) : Statement :=
  let input := decodedInput application fits payload
  { iteration := input.iteration, z0 := input.z0, zi := input.zi }

def predecessorPayload (payload : Payload application) (values : SourceValues application) : Payload application :=
  let input := decodedInput application fits payload
  { running := input.running
    runningWitness := fun _ => HyperNovaSource.runningWitness application values
    fresh := input.fresh
    freshWitness := HyperNovaSource.freshWitness application (sourceInput application fits payload) values
    pc := input.priorPc }

def Collision (statement : Statement) (payload : Payload application) : Prop :=
  PiCCSSecurity.StateHashCollision
    (ActualContextSecurity.decodedNext application
      (ProductionRelation.Plan.logicalAssignment payload.freshWitness))
    (ActualContextSecurity.terminalPreimage application fits setup statement payload)

def SourceSucceeded (payload : Payload application) (result : (SourceResult application)) : Prop :=
  CheckedWitnessExtraction.SourceReturned (PiCCSStoredWitnessCheck.commit application setup)
    productionGlobalParams (PiCCSStoredWitnessCheck.statement application fits (sourceInput application fits payload)) result

inductive Failure where
  | unexpectedEnvelope
  | initialStateMismatch
  | counterMismatch
  | missingSource
  | sourceAbort
  deriving DecidableEq, Repr

/-- Each advice word list satisfies the application's validity predicate at the
state that the earlier advice reaches from `initial`. -/
def ValidAdvice (initial : AppState) : List AppWitness → Prop
  | [] => True
  | advice :: rest =>
      application.valid initial advice ∧ ValidAdvice (application.step initial advice) rest

theorem validAdvice_append_singleton (initial : AppState) (earlier : List AppWitness)
    (last : AppWitness) :
    ValidAdvice application initial (earlier ++ [last]) ↔
      ValidAdvice application initial earlier ∧
        application.valid (earlier.foldl application.step initial) last := by
  induction earlier generalizing initial with
  | nil => simp only [List.nil_append, ValidAdvice, and_true, true_and, List.foldl_nil]
  | cons advice rest induction =>
      simp only [List.cons_append, ValidAdvice, List.foldl_cons, induction, and_assoc]

private def walk (statement : Statement) (proof : Envelope application)
    (results : List (SourceResult application)) (suffix : List AppWitness) :
    Except Failure (List AppWitness × List (SourceResult application)) :=
  match proof with
  | .bottom =>
      if statement.iteration = 0 then
        if statement.zi = statement.z0 then .ok (suffix, results)
        else .error .initialStateMismatch
      else .error .unexpectedEnvelope
  | .recursive payload =>
      if statement.iteration = 0 then .error .unexpectedEnvelope
      else
        let input := decodedInput application fits payload
        if input.iteration + 1 = statement.iteration then
          if input.iteration = 0 then .ok (input.witness :: suffix, results)
          else
            match results with
            | [] => .error .missingSource
            | none :: _ => .error .sourceAbort
            | some values :: tail =>
                walk (predecessorStatement application fits payload)
                  (.recursive (predecessorPayload application fits payload values)) tail
                  (input.witness :: suffix)
        else .error .counterMismatch
termination_by results.length
decreasing_by simp_wf

def run (statement : Statement) (proof : Envelope application) (results : List (SourceResult application)) :
    Except Failure (List AppWitness × List (SourceResult application)) :=
  walk application fits statement proof results []

private theorem walk_witness_length (statement : Statement) (proof : Envelope application)
    (results : List (SourceResult application)) (suffix : List AppWitness)
    (sized : ∀ witness ∈ suffix, witness.length = application.witnessWordCount)
    {advice : List AppWitness} {unused : List (SourceResult application)}
    (returned : walk application fits statement proof results suffix = .ok (advice, unused)) :
    ∀ witness ∈ advice, witness.length = application.witnessWordCount := by
  have extended (payload : Payload application) :
      ∀ witness ∈ (decodedInput application fits payload).witness :: suffix,
        witness.length = application.witnessWordCount := by
    intro witness member
    rcases List.mem_cons.mp member with rfl | member
    · exact List.length_ofFn
    · exact sized witness member
  cases proof with
  | bottom =>
      rw [walk] at returned
      split_ifs at returned
      cases returned
      exact sized
  | recursive payload =>
      rw [walk] at returned
      dsimp only at returned
      split_ifs at returned
      · cases returned
        exact extended payload
      · cases resultsEq : results with
        | nil => simp [resultsEq] at returned
        | cons head tail =>
            cases head with
            | none => simp [resultsEq] at returned
            | some values =>
                simp only [resultsEq] at returned
                exact walk_witness_length _ _ tail _ (extended payload) returned
termination_by results.length
decreasing_by
  rw [resultsEq]
  exact Nat.lt_succ_self _

theorem run_witness_length (statement : Statement) (proof : Envelope application)
    (results : List (SourceResult application)) {advice : List AppWitness}
    {unused : List (SourceResult application)}
    (returned : run application fits statement proof results = .ok (advice, unused)) :
    ∀ witness ∈ advice, witness.length = application.witnessWordCount :=
  walk_witness_length application fits statement proof results [] (by simp) returned

def SuccessfulSources (statement : Statement) (proof : Envelope application)
    (results : List (SourceResult application)) : Prop :=
  match proof with
  | .bottom => True
  | .recursive payload =>
      if statement.iteration = 0 then True
      else
        let input := decodedInput application fits payload
        if input.iteration + 1 = statement.iteration then
          if input.iteration = 0 then True
          else
            match results with
            | [] => False
            | result :: tail =>
                SourceSucceeded application fits setup payload result ∧
                  match result with
                  | none => True
                  | some values => SuccessfulSources (predecessorStatement application fits payload)
                      (.recursive (predecessorPayload application fits payload values)) tail
        else True
termination_by results.length
decreasing_by simp_wf

def NoStateHashCollisions (statement : Statement) (proof : Envelope application)
    (results : List (SourceResult application)) : Prop :=
  match proof with
  | .bottom => True
  | .recursive payload =>
      if statement.iteration = 0 then True
      else
        ¬ Collision application fits setup statement payload ∧
          let input := decodedInput application fits payload
          if input.iteration + 1 = statement.iteration then
            if input.iteration = 0 then True
            else
              match results with
              | some values :: tail => NoStateHashCollisions (predecessorStatement application fits payload)
                  (.recursive (predecessorPayload application fits payload values)) tail
              | _ => True
          else True
termination_by results.length
decreasing_by simp_wf

private theorem source_memberships (payload : Payload application) (values : SourceValues application)
    (success : SourceSucceeded application fits setup payload (some values)) :
    Lifecycle.TerminalHolds (PerApplicationFixedPoint.relation application fits)
      (PerApplicationCanonicalPackage.commitmentKey setup)
      ((decodedInput application fits payload).running functionIndex) (HyperNovaSource.runningWitness application values)
      (decodedInput application fits payload).fresh (HyperNovaSource.freshWitness application (sourceInput application fits payload) values) := by
  rcases (HyperNovaSource.sourceReturned_iff_terminalHolds application fits setup
    (sourceInput application fits payload) (some values)).mp success with ⟨returned, same, memberships⟩
  have equal : returned = values := (Option.some.inj same).symm
  subst returned
  simpa only [sourceInput,
    HyperNovaInput.running_ofClaims, HyperNovaInput.fresh_ofClaims] using! memberships

private theorem source_none (payload : Payload application) : ¬ SourceSucceeded application fits setup payload none := by
  rintro ⟨values, impossible, _⟩
  cases impossible

theorem counter_matches (statement : Statement) (payload : Payload application)
    (accepted : PerApplicationTerminal.Holds application fits setup
      statement (.recursive payload))
    (safe : ¬ Collision application fits setup statement payload) :
    (decodedInput application fits payload).iteration + 1 = statement.iteration := by
  rcases ActualContextSecurity.terminal_implies_matchingStepOrCollision
      application fits setup statement payload accepted with ⟨_step, same⟩ | collision
  · have counter := congrArg (fun preimage => preimage.iteration) same
    dsimp only [ActualContextSecurity.decodedNext, ActualHashSlots.nextPreimage,
      StateDecoder.preimage, ActualContextSecurity.terminalPreimage] at counter
    dsimp only [decodedInput, ActualStep.input]
    exact counter
  · exact False.elim (safe collision)

theorem predecessor_accepted (statement : Statement) (payload : Payload application)
    (values : SourceValues application) (positive : 0 < (decodedInput application fits payload).iteration)
    (accepted : PerApplicationTerminal.Holds application fits setup
      statement (.recursive payload))
    (source : SourceSucceeded application fits setup payload (some values))
    (safe : ¬ Collision application fits setup statement payload) :
    (decodedInput application fits payload).z0 = statement.z0 ∧
      statement.zi = application.step (decodedInput application fits payload).zi (decodedInput application fits payload).witness ∧
      PerApplicationTerminal.Holds application fits setup
        (predecessorStatement application fits payload) (.recursive (predecessorPayload application fits payload values)) := by
  have memberships := source_memberships application fits setup payload values source
  rcases HyperNovaPredecessor.terminal_implies_predecessorOrCollision application fits
      setup statement payload (HyperNovaSource.runningWitness application values)
      (HyperNovaSource.freshWitness application (sourceInput application fits payload) values) accepted
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

private theorem walk_correct (statement : Statement) (proof : Envelope application)
    (results : List (SourceResult application)) (suffix : List AppWitness)
    (accepted : Accepted application fits setup statement proof)
    (sources : SuccessfulSources application fits setup statement proof results)
    (safe : NoStateHashCollisions application fits setup statement proof results) :
    ∃ earlier unused,
      walk application fits statement proof results suffix = .ok (earlier ++ suffix, unused) ∧
      earlier.length = statement.iteration ∧
      earlier.foldl application.step statement.z0 = statement.zi ∧
      ValidAdvice application statement.z0 earlier := by
  cases proof with
  | bottom =>
      rcases (PerApplicationTerminal.holds_bottom_iff application fits setup statement).mp
        accepted with ⟨_valid, zero, initial⟩
      refine ⟨[], results, ?_, ?_, ?_, trivial⟩
      · rw [walk]
        simp only [if_pos zero, if_pos initial, List.nil_append]
      · exact zero.symm
      · exact initial.symm
  | recursive payload =>
      have terminalChecks := (PerApplicationTerminal.holds_recursive_iff
        application fits setup statement payload).mp accepted
      rcases terminalChecks with ⟨_valid, _canonical, _pcValid, positive, _public, _running, _fresh⟩
      have notZero : statement.iteration ≠ 0 := Nat.ne_of_gt positive
      have currentSafe : ¬ Collision application fits setup statement payload := by
        have unfolded := safe
        rw [NoStateHashCollisions.eq_def] at unfolded
        simp only [if_neg notZero] at unfolded
        exact unfolded.1
      have counter := counter_matches application fits setup statement payload accepted currentSafe
      by_cases zero : (decodedInput application fits payload).iteration = 0
      · have first : statement.iteration = 1 := by omega
        rcases HyperNovaPredecessor.terminal_one_implies_baseOrCollision
            application fits setup statement payload first accepted with
          ⟨_bottom, prior, transition⟩ | collision
        · refine ⟨[(decodedInput application fits payload).witness], results, ?_, ?_, ?_, ?_⟩
          · rw [walk]
            simp only [if_neg notZero, if_pos counter, if_pos zero,
              List.singleton_append]
          · simpa only [List.length_singleton] using first.symm
          · exact transition.symm
          · refine ⟨?_, trivial⟩
            rw [← prior]
            exact ActualContextSecurity.terminal_implies_valid application fits setup
              statement payload accepted
        · exact False.elim (currentSafe collision)
      · have previousPositive : 0 < (decodedInput application fits payload).iteration := Nat.pos_of_ne_zero zero
        cases resultsEq : results with
        | nil =>
            rw [SuccessfulSources.eq_def] at sources
            have impossible : False := by
              simp only [if_neg notZero, if_pos counter, if_neg zero, resultsEq] at sources
            exact False.elim impossible
        | cons result tail =>
            have sourceAndTail : SourceSucceeded application fits setup payload result ∧
                (match result with
                | none => True
                | some values => SuccessfulSources application fits setup (predecessorStatement application fits payload)
                    (.recursive (predecessorPayload application fits payload values)) tail) := by
              rw [SuccessfulSources.eq_def] at sources
              cases result <;>
                simpa only [if_neg notZero, if_pos counter, if_neg zero, resultsEq] using sources
            cases result with
            | none => exact False.elim (source_none application fits setup payload sourceAndTail.1)
            | some values =>
                have tailSafe : NoStateHashCollisions application fits setup (predecessorStatement application fits payload)
                    (.recursive (predecessorPayload application fits payload values)) tail := by
                  have conjunction : ¬ Collision application fits setup statement payload ∧
                      NoStateHashCollisions application fits setup (predecessorStatement application fits payload)
                        (.recursive (predecessorPayload application fits payload values)) tail := by
                    rw [NoStateHashCollisions.eq_def] at safe
                    simpa only [if_neg notZero, if_pos counter, if_neg zero, resultsEq] using safe
                  exact conjunction.2
                rcases predecessor_accepted application fits setup statement payload values previousPositive
                    accepted sourceAndTail.1 currentSafe with ⟨initial, transition, previous⟩
                rcases walk_correct (predecessorStatement application fits payload)
                    (.recursive (predecessorPayload application fits payload values)) tail
                    ((decodedInput application fits payload).witness :: suffix)
                    previous sourceAndTail.2 tailSafe with
                  ⟨earlier, unused, executed, length, reached, valid⟩
                refine ⟨earlier ++ [(decodedInput application fits payload).witness], unused,
                  ?_, ?_, ?_, ?_⟩
                · rw [walk]
                  simpa only [if_neg notZero, if_pos counter, if_neg zero,
                    resultsEq, List.append_assoc, List.singleton_append] using executed
                · rw [List.length_append, List.length_singleton]
                  dsimp only [predecessorStatement] at length
                  exact (congrArg (fun count => count + 1) length).trans counter
                · rw [foldl_singleton_append, ← initial]
                  exact (congrArg (fun state => application.step state (decodedInput application fits payload).witness)
                    reached).trans transition.symm
                · rw [validAdvice_append_singleton, ← initial]
                  refine ⟨valid, ?_⟩
                  change application.valid (earlier.foldl application.step
                    (predecessorStatement application fits payload).z0) _
                  rw [reached]
                  exact ActualContextSecurity.terminal_implies_valid application fits setup
                    statement payload accepted
termination_by results.length
decreasing_by
  rw [resultsEq]
  exact Nat.lt_succ_self _

theorem run_correct (statement : Statement) (proof : Envelope application) (results : List (SourceResult application))
    (accepted : PerApplicationTerminal.Holds application fits setup statement proof)
    (sources : SuccessfulSources application fits setup statement proof results)
    (safe : NoStateHashCollisions application fits setup statement proof results) :
    ∃ advice unused,
      run application fits statement proof results = .ok (advice, unused) ∧
      advice.length = statement.iteration ∧
      advice.foldl application.step statement.z0 = statement.zi ∧
      ValidAdvice application statement.z0 advice := by
  rcases walk_correct application fits setup statement proof results [] accepted sources safe with
    ⟨advice, unused, executed, length, reached, valid⟩
  exact ⟨advice, unused, by simpa only [run, List.append_nil] using executed, length, reached, valid⟩

end NightstreamFPrime.Export.Stage1.HyperNovaHistory
