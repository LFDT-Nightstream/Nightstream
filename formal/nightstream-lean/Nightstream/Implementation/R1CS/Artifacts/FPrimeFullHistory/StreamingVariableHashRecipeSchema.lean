import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge

/-!
Contract: variable-length structural recipe for one Rust Poseidon2 hash.

The exact input-column length derives the absorb-round count. Each absorb
round owns only the available input fields, so a final partial chunk has fewer
definition rows and a shorter allocation stride.

Does not own concrete artifact values, input authority, or collision
resistance.

Assurance tier: model-level.

Emits constraints: no. It describes emitted constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2Sponge

def rate : Nat := 4
def permutationRows : Nat := 600

structure VariableHashRecipe where
  constantValues : List Nat
  constantStartColumn : Nat
  localColumns : List Nat
  payloadColumns : List Nat
  /-- Exact absorb order. Constants can be interleaved with external inputs. -/
  orderedInputColumns : List Nat
  outputColumns : List Nat
deriving DecidableEq, Repr

def VariableHashRecipe.constantColumns
    (recipe : VariableHashRecipe) : List Nat :=
  List.range' recipe.constantStartColumn recipe.constantValues.length

def VariableHashRecipe.inputColumns
    (recipe : VariableHashRecipe) : List Nat :=
  recipe.orderedInputColumns

def VariableHashRecipe.absorbRounds
    (recipe : VariableHashRecipe) : Nat :=
  (recipe.inputColumns.length + (rate - 1)) / rate

def VariableHashRecipe.zeroColumn (recipe : VariableHashRecipe) : Nat :=
  recipe.constantStartColumn + recipe.constantValues.length

def VariableHashRecipe.chunkColumns
    (recipe : VariableHashRecipe) (round : Nat) : List Nat :=
  (recipe.inputColumns.drop (rate * round)).take rate

def VariableHashRecipe.definitionCount
    (recipe : VariableHashRecipe) (round : Nat) : Nat :=
  if round < recipe.absorbRounds then
    (recipe.chunkColumns round).length
  else
    1

def VariableHashRecipe.allocatedBefore
    (recipe : VariableHashRecipe) (round : Nat) : Nat :=
  ((List.range round).map fun index =>
    recipe.definitionCount index + permutationRows).sum

def VariableHashRecipe.roundColumnStart
    (recipe : VariableHashRecipe) (round : Nat) : Nat :=
  recipe.zeroColumn + 1 + recipe.allocatedBefore round

def VariableHashRecipe.callFirstAllocatedColumn
    (recipe : VariableHashRecipe) (round : Nat) : Nat :=
  recipe.roundColumnStart round + recipe.definitionCount round

def VariableHashRecipe.callOutputColumns
    (recipe : VariableHashRecipe) (round : Nat) : List Nat :=
  List.range' (recipe.callFirstAllocatedColumn round + 592) 8

def VariableHashRecipe.stateBeforeColumns
    (recipe : VariableHashRecipe) (round : Nat) : List Nat :=
  if round = 0 then List.replicate 8 recipe.zeroColumn
  else recipe.callOutputColumns (round - 1)

def VariableHashRecipe.callInputColumns
    (recipe : VariableHashRecipe) (round : Nat) : List Nat :=
  if round < recipe.absorbRounds then
    List.range' (recipe.roundColumnStart round)
        (recipe.definitionCount round) ++
      (recipe.stateBeforeColumns round).drop (recipe.definitionCount round)
  else
    recipe.roundColumnStart round ::
      (recipe.stateBeforeColumns round).drop 1

def VariableHashRecipe.call
    (recipe : VariableHashRecipe) (round : Nat) : Poseidon2Call.Call where
  rowStart := recipe.definitionCount round
  rowEnd := recipe.definitionCount round + permutationRows
  inputColumns := recipe.callInputColumns round
  firstAllocatedColumn := recipe.callFirstAllocatedColumn round

def VariableHashRecipe.absorbRound
    (recipe : VariableHashRecipe) (round : Nat) : Round where
  kind := .absorb (recipe.chunkColumns round)
  stateBeforeColumns := recipe.stateBeforeColumns round
  permutationInputColumns := recipe.callInputColumns round
  permutationOutputColumns := recipe.callOutputColumns round
  definingRows := List.range (recipe.definitionCount round)
  call := recipe.call round

def VariableHashRecipe.padRound (recipe : VariableHashRecipe) : Round where
  kind := .pad
  stateBeforeColumns := recipe.stateBeforeColumns recipe.absorbRounds
  permutationInputColumns := recipe.callInputColumns recipe.absorbRounds
  permutationOutputColumns := recipe.callOutputColumns recipe.absorbRounds
  definingRows := [0]
  call := recipe.call recipe.absorbRounds

def VariableHashRecipe.rounds (recipe : VariableHashRecipe) : List Round :=
  (List.range recipe.absorbRounds).map recipe.absorbRound ++
    [recipe.padRound]

def VariableHashRecipe.trace (recipe : VariableHashRecipe) : Trace where
  inputColumns := recipe.inputColumns
  zeroColumn := recipe.zeroColumn
  zeroRow := 0
  rounds := recipe.rounds
  outputColumns := recipe.outputColumns

def constantRows (recipe : VariableHashRecipe) : List Row :=
  (recipe.constantColumns.zip recipe.constantValues).map fun entry =>
    builderLinearRow entry.1 [(0, entry.2)]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact
