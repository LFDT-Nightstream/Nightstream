import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingVariableHashRecipeConstantSound

/-!
Contract: the three exact terminal advance-chain row blocks recompute each
updated `D_seen` digest from its prior digest and commitment-leaf digest.

It does not own the non-hash advance rows, close rows, or the typed terminal
transition.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerAdvanceChainRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingVariableHashRecipeConstantSound
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer

abbrev DigestValues := Fin 4 → Nat

def inputValues (lane : Fin 3) (assignment : Nat → Nat) : List Nat :=
  (advanceChainLink lane).recipe.inputColumns.map assignment

def computedDigest
    (lane : Fin 3) (assignment : Nat → Nat) : DigestValues :=
  fun output => runValueRounds (advanceChainLink lane).recipe.trace.rounds
    (inputValues lane assignment) (fun _ => 0) output.val

def assignedDigest
    (lane : Fin 3) (assignment : Nat → Nat) : DigestValues :=
  fun output => assignment
    ((advanceChainLink lane).recipe.outputColumns.getD output.val 0)

structure Sound
    (lane : Fin 3) (assignment : Nat → Nat) : Prop where
  constants :
    (advanceChainLink lane).recipe.constantColumns.map assignment =
      (advanceChainLink lane).recipe.constantValues
  hash : assignedDigest lane assignment = computedDigest lane assignment

theorem rows_sound
    (lane : Fin 3)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (advanceChainLink lane).Satisfied assignment) :
    Sound lane assignment := by
  change Satisfies
    (constantRows (advanceChainLink lane).recipe ++
      (advanceChainLink lane).recipe.trace.rows) assignment at satisfied
  have constantsSatisfied :
      Satisfies (constantRows (advanceChainLink lane).recipe) assignment := by
    intro row member
    exact satisfied row (List.mem_append_left _ member)
  have traceSatisfied :
      Satisfies (advanceChainLink lane).recipe.trace.rows assignment := by
    intro row member
    exact satisfied row (List.mem_append_right _ member)
  refine {
    constants := constantRows_values (advanceChainLink lane).recipe assignment
      canonical one (advance_chain_constants_canonical lane) constantsSatisfied
    hash := ?_ }
  funext output
  exact ownedTrace_values_sound (advance_chain_trace_ownedValid lane)
    canonical one traceSatisfied output.val output.isLt

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerAdvanceChainRowSound
