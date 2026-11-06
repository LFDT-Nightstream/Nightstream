import NightstreamFPrime.Export.OptionB.IvcSecurity
import NightstreamFPrime.Lifecycle.OptionB.FinishStep

/-! The finish step as the first stage of option B's reverse walk (CW-10.1).

The final verifier's claim is a last fold: a refresh step input whose finish checks hold
(`Lifecycle.OptionB.FinishStep.FinishHolds`) and whose two next slots hold under their circuits'
relations. The finish step has no application step and no next state. So its input is a terminal
of HyperNova Construction 2: the statement is the input's iteration and states
(`finishStatement`), and the proof holds the input's slots, its fresh instance and its prior
program counter (`finishPayload`). The finish checks give the terminal's state hash, and the slot
that the fold does not touch keeps its next claim. The refresh fold consumed the selected slot's
input claim and the fresh instance, so the terminal relation needs witnesses of both. The fold is
a walk attempt (`finishAttempt`) that succeeds (`attempt_succeeds`). An answer that extracts it
gives both witnesses, so `Terminal.HoldsFor` accepts the terminal (`holdsFor_of_extracted`), and
its visit with every slot at `B` is relaxed (`finish_relaxed`, through `IvcWalk.initial_relaxed`).

So for a last fold whose statement no valid advice reaches, each answer either fails to extract
the finish attempt or makes the terminal an IVC false acceptance (`finish_covered`). With the
walk's answers, a collision or a failed stage follows at the finish stage or at one of the
terminal's first `iteration` visits (`finish_walk_covered`, through `IvcWalk.covered`). For an
input that the refresh step's words decode (`StepRelation.refreshInput`), the prior state is well
formed (`refresh_wellFormed`) and the statement is valid (`refresh_valid`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.OptionB.FinishWalk

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.HyperNova.ReverseStages (visitAfter StageFails)
open NightstreamFPrime.Spec.Folding.Schedule (productionLightFoldCap)
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Export.OptionB.StepRelation
open NightstreamFPrime.Export.OptionB.TerminalBridge
open NightstreamFPrime.Export.OptionB.IvcWalk
open NightstreamFPrime.Export.OptionB.IvcSecurity (Output FalseAcceptance)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The finish step's input as a terminal statement: its iteration and both states. -/
def finishStatement (input : StepInput logicalWidth publicFits) : TerminalStatement AppState :=
  ⟨input.iteration, input.z0, input.zi⟩

/-- The finish fold: the refresh circuit folds the fresh instance into the slot that the prior
program counter selects, under that slot's key. -/
def finishFold (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits) :
    StepFold logicalWidth publicFits :=
  let slot := circuitOf input.priorPc
  { circuit := 1
    key := circuitKeys slot
    input := (input.running slot).1
    fresh := input.fresh
    proof := input.nifsProof
    output := output.runningNext slot }

/-- The finish stage's attempt: the finish fold, linked to the input's prior state, with the
selected next slot's witness at bound `B`. -/
def finishAttempt (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (witnesses : Fin 2 → Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    WalkAttempt logicalWidth publicFits :=
  ⟨⟨priorOf circuitKeys input, input.running (circuitOf input.priorPc),
      finishFold circuitKeys input output⟩,
    productionGlobalParams.bigB, witnesses (circuitOf input.priorPc)⟩

/-- The terminal proof of the finish step's input: the input's slots and fresh instance, the
answer's witnesses for the selected slot and the fresh instance, the next slots' witness for the
other slot, and the prior program counter. -/
def finishPayload (input : StepInput logicalWidth publicFits)
    (witnesses : Fin 2 → Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
    (answer : Answer logicalWidth publicFits) : Payload logicalWidth publicFits :=
  ⟨input.running, Function.update witnesses (circuitOf input.priorPc) answer.1, input.fresh,
    answer.2, input.priorPc⟩

/-- The IVC output of the finish stage: the input's statement and its terminal proof. -/
def finishOutput (input : StepInput logicalWidth publicFits)
    (witnesses : Fin 2 → Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
    (answer : Answer logicalWidth publicFits) : Output logicalWidth publicFits :=
  (finishStatement input, .recursive (finishPayload input witnesses answer))

section Link

variable (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
  {input : StepInput logicalWidth publicFits} {output : StepOutput logicalWidth publicFits}
  {witnesses : Fin 2 → Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)}

/-- The finish fold is an accepted refresh fold of the selected slot, linked to the input's prior
state, whose next claim holds at `B`. -/
theorem attempt_succeeds (cap : Nat) (well : (toState (priorOf circuitKeys input)).WellFormed)
    (finish : FinishStep.FinishHolds cap ajtai circuitKeys 1 input output)
    (slots : ∀ slot, CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
      (toClaim ((circuitKeys slot).system ajtai) (output.runningNext slot).1) (witnesses slot)) :
    WalkSucceeds ajtai (finishAttempt circuitKeys input output witnesses) := by
  obtain ⟨priorPcValid, freshPublic, folded, depthZero, _⟩ := finish
  have slotEq := circuitOf_selected priorPcValid
  rw [← slotEq] at folded depthZero
  refine ⟨⟨freshPublic, ⟨priorPcValid, congrArg input.running slotEq.symm⟩, well⟩, rfl, ?_,
    slots _, Nat.le_add_right _ _⟩
  exact (accepts_iff productionLightFoldCap ajtai 1 _ _ _ _ _).2
    ⟨fun light => by simp [BoundedDepth.circuitStep] at light, folded, depthZero⟩

/-- An answer that extracts the finish attempt makes the finish step's input an accepted
terminal: the answer opens the selected slot and the fresh instance, and the other slot keeps its
next claim. -/
theorem holdsFor_of_extracted (cap : Nat)
    (valid : Terminal.StatementValid (finishStatement input)) (positive : 0 < input.iteration)
    (finish : FinishStep.FinishHolds cap ajtai circuitKeys 1 input output)
    (slots : ∀ slot, CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
      (toClaim ((circuitKeys slot).system ajtai) (output.runningNext slot).1) (witnesses slot))
    {answer : Answer logicalWidth publicFits}
    (extracted : WalkExtracted ajtai (finishAttempt circuitKeys input output witnesses) answer)
    (terminalCap : Nat) (step : AppState → AppWitness → AppState)
    (control : AppState → AppWitness → Fin 2) :
    Terminal.HoldsFor terminalCap ajtai circuitKeys step control (finishStatement input)
      (.recursive (finishPayload input witnesses answer)) := by
  obtain ⟨priorPcValid, freshPublic, _, _, others⟩ := finish
  have slotEq := circuitOf_selected priorPcValid
  rw [← slotEq] at others
  obtain ⟨claimHolds, freshHolds⟩ := extracted
  change FreshHolds ajtai (circuitKeys (circuitOf input.priorPc)) input.fresh answer.2 at freshHolds
  rw [slotEq] at freshHolds
  refine (Terminal.holdsFor_recursive_iff terminalCap ajtai circuitKeys step control _ _).2
    ⟨valid, priorPcValid, positive, freshPublic, fun slot => ?_, freshHolds⟩
  by_cases same : slot = circuitOf input.priorPc
  · subst same
    show CE.Holds _ _ _ (Function.update witnesses (circuitOf input.priorPc) answer.1
      (circuitOf input.priorPc))
    rw [Function.update_self]
    exact claimHolds
  · show CE.Holds _ _ (toClaim ((circuitKeys slot).system ajtai) (input.running slot).1)
      (Function.update witnesses (circuitOf input.priorPc) answer.1 slot)
    rw [Function.update_of_ne same, ← others slot same]
    exact slots slot

/-- The finish output starts the walk: its visit, with every slot at bound `B`, is relaxed. -/
theorem finish_relaxed (cap : Nat)
    (valid : Terminal.StatementValid (finishStatement input)) (positive : 0 < input.iteration)
    (finish : FinishStep.FinishHolds cap ajtai circuitKeys 1 input output)
    (slots : ∀ slot, CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
      (toClaim ((circuitKeys slot).system ajtai) (output.runningNext slot).1) (witnesses slot))
    {answer : Answer logicalWidth publicFits}
    (extracted : WalkExtracted ajtai (finishAttempt circuitKeys input output witnesses) answer) :
    Relaxed ajtai circuitKeys (initial (finishOutput input witnesses answer)) :=
  initial_relaxed (step := fun state _ => state) ajtai circuitKeys (fun _ _ => 0) cap _
    (holdsFor_of_extracted ajtai circuitKeys cap valid positive finish slots extracted cap _ _)

/-- The deterministic link from a last fold to the IVC bound: if no valid advice reaches the
finish step's statement, an answer either fails to extract the succeeding finish attempt or makes
the finish output an IVC false acceptance (`IvcSecurity.FalseAcceptance`). -/
theorem finish_covered {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {step : AppState → AppWitness → AppState}
    {cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : Exports relation ajtai circuitKeys step cubeFits)
    (control : AppState → AppWitness → Fin 2) (terminalCap cap : Nat)
    (well : (toState (priorOf circuitKeys input)).WellFormed)
    (valid : Terminal.StatementValid (finishStatement input)) (positive : 0 < input.iteration)
    (finish : FinishStep.FinishHolds cap ajtai circuitKeys 1 input output)
    (slots : ∀ slot, CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
      (toClaim ((circuitKeys slot).system ajtai) (output.runningNext slot).1) (witnesses slot))
    (unreached : ¬ Reaches exports.program (finishStatement input))
    (answer : Answer logicalWidth publicFits) :
    (WalkSucceeds ajtai (finishAttempt circuitKeys input output witnesses) ∧
        ¬ WalkExtracted ajtai (finishAttempt circuitKeys input output witnesses) answer) ∨
      FalseAcceptance ajtai circuitKeys exports control terminalCap
        (finishOutput input witnesses answer) := by
  by_cases extracted : WalkExtracted ajtai (finishAttempt circuitKeys input output witnesses) answer
  · exact Or.inr ⟨holdsFor_of_extracted ajtai circuitKeys cap valid positive finish slots extracted
      terminalCap step control, unreached⟩
  · exact Or.inl ⟨attempt_succeeds ajtai circuitKeys cap well finish slots, extracted⟩

/-- The finish stage followed by the reverse walk: if no valid advice reaches the finish step's
statement, the finish answer fails to extract, or the walk from the finish output has a collision
or a failed stage at one of its first `iteration` visits. -/
theorem finish_walk_covered {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {step : AppState → AppWitness → AppState}
    {cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : Exports relation ajtai circuitKeys step cubeFits)
    (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (control : AppState → AppWitness → Fin 2) (terminalCap cap : Nat)
    (well : (toState (priorOf circuitKeys input)).WellFormed)
    (valid : Terminal.StatementValid (finishStatement input)) (positive : 0 < input.iteration)
    (finish : FinishStep.FinishHolds cap ajtai circuitKeys 1 input output)
    (slots : ∀ slot, CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
      (toClaim ((circuitKeys slot).system ajtai) (output.runningNext slot).1) (witnesses slot))
    (unreached : ¬ Reaches exports.program (finishStatement input))
    (answer : Answer logicalWidth publicFits) (answers : List (Answer logicalWidth publicFits))
    (depthBound : input.iteration ≤ answers.length) :
    (WalkSucceeds ajtai (finishAttempt circuitKeys input output witnesses) ∧
        ¬ WalkExtracted ajtai (finishAttempt circuitKeys input output witnesses) answer) ∨
      ∃ j < answers.length,
        WalkCollision ajtai circuitKeys exports
            (visitAfter (back ajtai circuitKeys exports)
              (initial (finishOutput input witnesses answer)) (answers.take j)) ∨
          StageFails (IvcWalk.attemptOf ajtai circuitKeys exports) (back ajtai circuitKeys exports)
            (WalkSucceeds ajtai) (WalkExtracted ajtai) (initial (finishOutput input witnesses answer))
            answers j := by
  rcases finish_covered ajtai circuitKeys exports control terminalCap cap well valid positive finish
    slots unreached answer with failed | ⟨accepted, unreached'⟩
  · exact Or.inl failed
  · exact Or.inr (covered ajtai circuitKeys exports keysShaped control terminalCap
      (finishOutput input witnesses answer) answers depthBound accepted
      fun leads => unreached' (reaches_of_leads _ _ _ leads
        (trail_lengths ajtai circuitKeys exports _ _)))

end Link

/-- The prior state of an input that the refresh step's words decode is well formed when both
key digests have four words. -/
theorem refresh_wellFormed (program : Stage1.Application.Program)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (keysShaped : ∀ key, (circuitKeys key).digest.length = 4) (env : Circuit.Env) :
    (toState (priorOf circuitKeys (refreshInput (logicalWidth := logicalWidth)
      (publicFits := publicFits) program env))).WellFormed :=
  have represents := refreshRepresents program ajtai circuitKeys 1 env
  ⟨keysShaped, List.length_ofFn, List.length_ofFn, represents.iterationBound,
    represents.priorPcBound, represents.priorDepth⟩

/-- The statement of an input that the refresh step's words decode is valid. -/
theorem refresh_valid (program : Stage1.Application.Program) (env : Circuit.Env) :
    Terminal.StatementValid (finishStatement (refreshInput (logicalWidth := logicalWidth)
      (publicFits := publicFits) program env)) :=
  ⟨Fin.isLt _, List.length_ofFn, List.length_ofFn⟩

end NightstreamFPrime.Export.OptionB.FinishWalk
