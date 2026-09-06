import NightstreamFPrime.Export.Pilot
import NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedSlots

/-!
Owns the physical final-layer contract of a canonical Poseidon2 invocation.
Its output is the external linear layer of the same invocation's final
retained S-box values. This contract does not select a protocol or layout.
-/

namespace NightstreamFPrime.Export.PermutationOutput

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Package

private def finalSboxes : Layer.EState :=
  fun lane => .var (563 + 4 * lane.val)

private def finalPrefix : List Permutation.Step := Permutation.schedule.dropLast

private theorem schedule_split :
    Permutation.schedule = finalPrefix ++ [.terminalFullRound 3] := by
  rfl

private theorem finalPrefix_start : 8 + Permutation.scheduleSize finalPrefix = 560 := by
  rfl

private theorem finalPrefix_output :
    (Permutation.compile 8 PilotData.canonicalState finalPrefix).output =
      Permutation.freshState 552 := by
  funext lane
  rfl

private theorem compile_suffix_rows (env : Env) (start : Nat) (state : Layer.EState)
    (initial remaining : List Permutation.Step)
    (rows : ConstraintsHold env (recipeConstraints start
      (Permutation.compile start state (initial ++ remaining)).recipes)) :
    ConstraintsHold env
      (recipeConstraints (start + Permutation.scheduleSize initial)
        (Permutation.compile (start + Permutation.scheduleSize initial)
          (Permutation.compile start state initial).output remaining).recipes) := by
  induction initial generalizing start state with
  | nil => simpa only [List.nil_append, Permutation.scheduleSize,
      List.map_nil, List.sum_nil, Nat.add_zero, Permutation.compile] using rows
  | cons step rest inductionHypothesis =>
      simp only [List.cons_append, Permutation.compile] at rows
      rw [Permutation.recipeConstraints_append] at rows
      have restRows := (Permutation.constraintsHold_append env _ _).mp rows |>.2
      rw [Permutation.stepRecipes_length] at restRows
      have selected := inductionHypothesis (start + Permutation.stepSize step)
        (Permutation.stepOutput start step) restRows
      simpa only [Permutation.compile, Permutation.scheduleSize, List.map_cons,
        List.sum_cons, Nat.add_assoc] using selected

private theorem canonical_finalLayer_rows (env : Env)
    (rows : ConstraintsHold env (PilotData.canonicalConstraints ())) :
    ConstraintsHold env
      (recipeConstraints 592 (List.ofFn (Layer.externalE finalSboxes))) := by
  have suffixRows := compile_suffix_rows env 8 PilotData.canonicalState
    finalPrefix [.terminalFullRound 3] (by
      simpa only [PilotData.canonicalConstraints, PilotData.canonicalRecipes,
        schedule_split] using rows)
  have lastRows : ConstraintsHold env
      (recipeConstraints 560 (Permutation.stepRecipes 560 (.terminalFullRound 3)
        (Permutation.freshState 552))) := by
    rw [finalPrefix_start, finalPrefix_output] at suffixRows
    change ConstraintsHold env
      (recipeConstraints 560
        (Permutation.stepRecipes 560 (.terminalFullRound 3)
          (Permutation.freshState 552) ++ [])) at suffixRows
    simpa only [List.append_nil] using suffixRows
  rw [Permutation.stepRecipes, Permutation.recipeConstraints_append] at lastRows
  have externalRows := (Permutation.constraintsHold_append env _ _).mp lastRows |>.2
  simp only [Permutation.compileSboxes_recipes_length, Permutation.fullInputs,
    List.length_ofFn] at externalRows
  have sboxesEq : Permutation.fullSboxState 560 Spec.Poseidon2.terminalConstants 3
      (Permutation.freshState 552) = finalSboxes := by
    funext lane
    fin_cases lane <;> rfl
  simpa only [sboxesEq] using externalRows

private theorem finalSbox_local (lane : Fin 8) :
    (PoseidonRetainedSlots.localOutput (PoseidonRetainedSlots.finalRow lane)).val =
      555 + 4 * lane.val := by
  fin_cases lane <;> rfl

/-- The eight accepted final-layer rows identify output words with the
external layer of the eight final S-box words in the same invocation. -/
theorem invocation_finalLayer (invocation : PermutationInvocation) (env : Env)
    (rows : PermutationInvocationHolds
      (PilotData.circuitPackage ()) invocation env) :
    (fun lane : Fin 8 => env (invocation.witnessStart + 584 + lane.val)) =
      Layer.externalF (fun lane => env
        (invocation.witnessStart +
          (PoseidonRetainedSlots.localOutput (PoseidonRetainedSlots.finalRow lane)).val)) := by
  let localEnv := Pilot.canonicalInvocationEnv invocation env
  have logical := Pilot.canonicalPermutationInvocation_implies_constraints
    invocation env rows
  have finalRows : ConstraintsHold localEnv
      (recipeConstraints 592 (List.ofFn (Layer.externalE finalSboxes))) := by
    exact canonical_finalLayer_rows localEnv logical
  have finalState := Permutation.stateRows_sound localEnv 592
    (Layer.externalE finalSboxes) finalRows
  have external : Layer.evalState localEnv (Layer.externalE finalSboxes) =
      Layer.externalF (Layer.evalState localEnv finalSboxes) := by
    funext lane
    exact Layer.eval_externalE localEnv finalSboxes lane
  rw [external] at finalState
  have output : Layer.evalState localEnv (Permutation.freshState 592) =
      fun lane : Fin 8 => env (invocation.witnessStart + 584 + lane.val) := by
    funext lane
    change Pilot.canonicalInvocationEnv invocation env (592 + lane.val) = _
    have indexEq : 592 + lane.val = 8 + (584 + lane.val) := by omega
    rw [indexEq, Pilot.canonicalInvocationEnv_local]
    congr 1
    omega
  have sboxes : Layer.evalState localEnv finalSboxes =
      fun lane : Fin 8 => env
        (invocation.witnessStart +
          (PoseidonRetainedSlots.localOutput (PoseidonRetainedSlots.finalRow lane)).val) := by
    funext lane
    change Pilot.canonicalInvocationEnv invocation env (563 + 4 * lane.val) = _
    have indexEq : 563 + 4 * lane.val = 8 + (555 + 4 * lane.val) := by omega
    rw [indexEq, Pilot.canonicalInvocationEnv_local, finalSbox_local]
  rw [output, sboxes] at finalState
  exact finalState

end NightstreamFPrime.Export.PermutationOutput
