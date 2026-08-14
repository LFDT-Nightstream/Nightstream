import Nightstream.Protocol.Nebula.CheckedStepBatch

/-! Regression and omitted-link countermodel for checked-step batching. -/

set_option autoImplicit false

namespace tests.NebulaCheckedStepBatch

open Nightstream.Protocol.Nebula.CheckedStepBatch
open Nightstream.Protocol.Nebula.Ports

def inactiveStep : CheckedStep where
  rows := fun _ => NormalizedRow.inactive

def twoSteps : Batch 2 where
  steps := fun _ => inactiveStep

example : twoSteps.rowList.length = 6 := by decide

example : twoSteps.accesses = [] := by
  simp [Batch.accesses, Batch.stepList, twoSteps, inactiveStep,
    CheckedStep.accesses, CheckedStep.physicalPorts,
    NormalizedRow.inactive, compactPayloads]

def preservesState (before : Bool) (_step : CheckedStep) (after : Bool) : Prop :=
  before = after

/-- An endpoints-only batching rule would accept this pair. The required
adjacent-state links reject it because both steps preserve state. -/
theorem omitted_intermediate_links_are_unsound :
    True /\
      ¬ Sequential preservesState twoSteps false true := by
  constructor
  · trivial
  · intro sequential
    rcases sequential with ⟨witness⟩
    have first : witness.states 0 = witness.states 1 := by
      simpa [preservesState] using witness.step (0 : Fin 2)
    have second : witness.states 1 = witness.states 2 := by
      simpa [preservesState] using witness.step (1 : Fin 2)
    have impossible : false = true := by
      calc
        false = witness.states 0 := witness.initial.symm
        _ = witness.states 1 := first
        _ = witness.states 2 := second
        _ = true := witness.final
    cases impossible

#check Batch.rowList_flatMap_accesses
#check Witness.step_exact

end tests.NebulaCheckedStepBatch
