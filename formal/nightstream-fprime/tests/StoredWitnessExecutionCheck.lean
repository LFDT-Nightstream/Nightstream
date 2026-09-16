import NightstreamFPrime.Export.Stage1.StoredPermutationExecution
import NightstreamFPrime.Export.Codec

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Stage1

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok result => pure result
  | .error error => throw (IO.userError error)

private def field (value : Lean.Json) : Except String F := do
  let word ← value.getNat?
  if canonical : word < goldilocksModulus then pure ⟨word, canonical⟩
  else throw "noncanonical caller field"

private def run (mode callerPath : String) (output : System.FilePath) : IO UInt32 := do
  unless mode == "reference" || mode == "stored" || mode == "permutation" do
    throw (IO.userError "expected reference, stored or permutation mode")
  unless !(← output.pathExists) do throw (IO.userError "output already exists")
  let fields ← checked ((← checked (Lean.Json.parse (← IO.FS.readFile callerPath))).getArr?)
  unless fields.size == 5 do throw (IO.userError "expected a complete caller packet")
  let caller ← checked (fields[2]!.getArr?)
  unless 8 ≤ caller.size do throw (IO.userError "caller has fewer than eight input words")
  let inputs ← (caller.extract 0 8).mapM fun value => checked (field value)
  let initial := Array.ofFn fun column : Fin 600 => inputs[column.val]?.getD 0
  let recipes ← IO.wait (Task.spawn fun _ => PilotData.canonicalRecipes ())
  let invocation : Package.PermutationInvocation := {
    phase := 0, rowStart := 0, witnessStart := 8
    inputs := List.ofFn fun lane : Fin 8 =>
      { constant := 0, terms := [⟨lane.val, 1⟩] } }
  let started ← IO.monoNanosNow
  let values ← IO.wait (Task.spawn fun _ =>
    if mode == "reference" then
      Array.ofFn fun column : Fin 600 =>
        NightstreamFPrime.Circuit.executeRecipes
          (StoredWitnessExecution.asEnv initial) 8 recipes column.val
    else if mode == "stored" then StoredWitnessExecution.executeRecipes initial 8 recipes
    else StoredPermutationExecution.execute invocation initial)
  let elapsed := (← IO.monoNanosNow) - started
  unless values.size == 600 do throw (IO.userError "wrong completed local width")
  IO.FS.writeFile output
    ((Codec.Value.array (values.toList.map fun value => .atom value.val)).render ++ "\n")
  IO.println (Lean.Json.mkObj [("event", .str "stored_witness_local_case"),
    ("mode", .str mode), ("field_words", Lean.toJson values.size),
    ("recipes", Lean.toJson recipes.length), ("compute_ns", Lean.toJson elapsed)]).compress
  return 0

def main (arguments : List String) : IO UInt32 :=
  match arguments with
  | [mode, caller, output] => run mode caller output
  | _ => do
      IO.eprintln "usage: checkStoredWitnessExecution <reference|stored|permutation> <caller> <new-output>"
      return 2
