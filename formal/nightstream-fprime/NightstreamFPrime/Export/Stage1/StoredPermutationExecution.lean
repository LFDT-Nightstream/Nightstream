import NightstreamFPrime.Export.Stage1.StoredWitnessExecution
import NightstreamFPrime.Export.Pilot

/-!
Materialize one canonical permutation locally, then copy its witness cells.
The recipe list and all expression arithmetic retain their existing owners.
This proves equality with the existing completion procedure; it does not
assert row satisfaction when input cells overlap the destination interval.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.StoredPermutationExecution

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Export.Package
open StoredWitnessExecution (asEnv)

/-- Snapshot the eight input combinations into the canonical 600-cell local
storage. All remaining cells begin at zero, as in the reference procedure. -/
def localInput (invocation : PermutationInvocation) (values : Array F) : Array F :=
  Array.ofFn fun column : Fin 600 =>
    if column.val < 8 then
      (invocationInputCombination invocation column.val).toR1CS.eval (asEnv values)
    else 0

private theorem localInput_size (invocation : PermutationInvocation) (values : Array F) :
    (localInput invocation values).size = 600 := by
  exact Array.size_ofFn

private theorem localInput_env (invocation : PermutationInvocation) (values : Array F) :
    asEnv (localInput invocation values) = fun column =>
      if column < 8 then
        (invocationInputCombination invocation column).toR1CS.eval (asEnv values)
      else 0 := by
  funext column
  by_cases live : column < 600
  · simp only [asEnv, localInput, Array.getElem?_ofFn, dif_pos live, Option.getD_some]
  · have outsideInputs : ¬ column < 8 := by omega
    simp only [asEnv, localInput, Array.getElem?_ofFn, dif_neg live,
      Option.getD_none, if_neg outsideInputs]

private theorem canonicalRecipes_length : (PilotData.canonicalRecipes ()).length = 592 :=
  Permutation.compile_schedule_recipe_count 8 PilotData.canonicalState

/-- The completed local state is an Array value. Each canonical recipe is
executed once, with reads served from the current stored state. -/
def localCompleted (invocation : PermutationInvocation) (values : Array F) : Array F :=
  StoredWitnessExecution.executeRecipes (localInput invocation values) 8
    (PilotData.canonicalRecipes ())

private theorem localCompleted_env (invocation : PermutationInvocation) (values : Array F) :
    asEnv (localCompleted invocation values) =
      NightstreamFPrime.Circuit.executeRecipes
        (fun column => if column < 8 then
          (invocationInputCombination invocation column).toR1CS.eval (asEnv values)
        else 0) 8 (PilotData.canonicalRecipes ()) := by
  have fits : 8 + (PilotData.canonicalRecipes ()).length ≤
      (localInput invocation values).size := by
    simpa only [canonicalRecipes_length, localInput_size] using (Nat.le_refl 600)
  rw [localCompleted, StoredWitnessExecution.executeRecipes_eq
    (localInput invocation values) 8 (PilotData.canonicalRecipes ()) fits, localInput_env]

/-- Copy all 592 completed witness values with the existing stored executor.
The let binding retains one completed array before any global write occurs. -/
def execute (invocation : PermutationInvocation) (values : Array F) : Array F :=
  let completed := localCompleted invocation values
  let constants := List.ofFn fun index : Fin 592 =>
    Expr.const (completed[8 + index.val]?.getD 0)
  StoredWitnessExecution.executeRecipes values invocation.witnessStart constants

theorem execute_size (invocation : PermutationInvocation) (values : Array F) :
    (execute invocation values).size = values.size := by
  unfold execute
  exact StoredWitnessExecution.executeRecipes_size values invocation.witnessStart _

/-- Every global environment coordinate equals the existing permutation
completion. Both procedures read the original input snapshot; only the
complete destination interval must fit the supplied global array. -/
theorem execute_eq (invocation : PermutationInvocation) (values : Array F)
    (fits : invocation.witnessStart + 592 ≤ values.size) :
    asEnv (execute invocation values) =
      Pilot.completePermutationInvocationEnv invocation (asEnv values) := by
  let constants := List.ofFn fun index : Fin 592 =>
    Expr.const ((localCompleted invocation values)[8 + index.val]?.getD 0)
  have copyFits : invocation.witnessStart + constants.length ≤ values.size := by
    simpa only [constants, List.length_ofFn] using fits
  change asEnv (StoredWitnessExecution.executeRecipes values invocation.witnessStart constants) = _
  rw [StoredWitnessExecution.executeRecipes_eq values invocation.witnessStart constants copyFits]
  change NightstreamFPrime.Circuit.executeRecipes (asEnv values) invocation.witnessStart
      (List.ofFn fun index : Fin 592 =>
        Expr.const (asEnv (localCompleted invocation values) (8 + index.val))) =
    NightstreamFPrime.Circuit.executeRecipes (asEnv values) invocation.witnessStart
      (List.ofFn fun index : Fin 592 => Expr.const
        (NightstreamFPrime.Circuit.executeRecipes
          (fun column => if column < 8 then
            (invocationInputCombination invocation column).toR1CS.eval (asEnv values)
          else 0) 8 (PilotData.canonicalRecipes ()) (8 + index.val)))
  rw [localCompleted_env]

end NightstreamFPrime.Export.Stage1.StoredPermutationExecution
