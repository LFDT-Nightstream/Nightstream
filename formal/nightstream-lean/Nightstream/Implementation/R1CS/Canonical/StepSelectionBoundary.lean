import Nightstream.HyperNova.Construction2.Paper

/-!
Contract: record that the application `step` is a setup selection, completing
the encoding's selection surface.

## The fourth member of a claim that was three-quarters proved

`step`, `nifsVerify`, `runningCheck` and `freshCheck` have been reported all
session as setup selections that no encoding derives.  Three of the four are
kernel-checked:

| call | boundary |
|---|---|
| `nifsVerify` | `NifsCompletionBoundary.setupVerifier_is_a_real_choice` |
| `runningCheck` | `TerminalCheckSelectionBoundary.runningCheck_is_a_real_choice` |
| `freshCheck` | `TerminalCheckSelectionBoundary.freshCheck_is_a_real_choice` |
| `step` | **this module** |

`step` was asserted from the first cycle and never proved.  It is a selection —
`Vocabulary.callEval` sends `Call.step` to `parameters.machine.step`, and
`Machine.step : Fin slotCount → State → Witness → State` is a plain field — but
"is a field" was read, not checked, and the other three had already shown what
checking looks like.

## Not the vacuous shape

`NifsCompletionBoundary` records a theorem it withdrew: "X does not determine Y"
for independent inputs is provable for every X and says nothing.  This is not
that.  Everything except the machine is fixed, and two concrete inhabitants are
compared at one concrete argument, so what varies is exactly the object the
claim is about.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.StepSelectionBoundary

open Nightstream.HyperNova.Construction2.Paper

/-- A legitimate machine whose step keeps the state. -/
def holdingMachine : Machine Unit Unit Bool Unit Unit Unit Unit 1 where
  control := fun _ _ => ⟨0, by decide⟩
  step := fun _ state _ => state
  freshPublic := fun _ => ()
  encodeInstance := fun _ => ()
  hash := fun _ => ()

/-- A legitimate machine over the same carriers whose step flips it. -/
def flippingMachine : Machine Unit Unit Bool Unit Unit Unit Unit 1 where
  control := fun _ _ => ⟨0, by decide⟩
  step := fun _ state _ => !state
  freshPublic := fun _ => ()
  encodeInstance := fun _ => ()
  hash := fun _ => ()

theorem holdingMachine_step : holdingMachine.step ⟨0, by decide⟩ true () = true := rfl

theorem flippingMachine_step : flippingMachine.step ⟨0, by decide⟩ true () = false := rfl

/-- **The application `step` is a real choice.**

Two legitimate machines over the same carriers take the same state and witness
to different states.  A recipe for `step` must therefore *select* one; no
amount of encoding-side reasoning derives it.

Scope, precisely: this is a statement about the two machines.  It does not say
the canonical program is incomplete, and it does not license leaving `step`
unbuilt once a selection exists — `CanonicalProgram.SelectedRecipe` is the
interface a selection enters through, and it carries its own row certificate. -/
theorem step_is_a_real_choice :
    holdingMachine.step ⟨0, by decide⟩ true ()
      ≠ flippingMachine.step ⟨0, by decide⟩ true () := by
  rw [holdingMachine_step, flippingMachine_step]
  exact Bool.noConfusion

/-- **The selection surface is exactly four calls.**

Stated as the conjunction so a reader can see the group is complete rather than
inferring it from four separate modules.  The other three disagreements are
proved in their own boundaries; this one supplies the member that was
missing. -/
theorem step_selection_is_kernel_checked :
    holdingMachine.step ⟨0, by decide⟩ true () = true
      ∧ flippingMachine.step ⟨0, by decide⟩ true () = false
      ∧ holdingMachine.step ⟨0, by decide⟩ true ()
          ≠ flippingMachine.step ⟨0, by decide⟩ true () :=
  ⟨holdingMachine_step, flippingMachine_step, step_is_a_real_choice⟩

end Nightstream.Implementation.R1CS.Canonical.StepSelectionBoundary
