import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.RecursiveStepFixture
import NightstreamFPrime.Export.Stage1.Wide.BaseStepFixture

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Spec

private def decodeWords (value : Lean.Json) : Except String (List F) := do
  let values ← value.getArr?
  unless values.size == 4 do
    throw "recursive fixture: expected four state or message words"
  values.toList.mapM fun value => do
    let word ← value.getNat?
    if canonical : word < goldilocksModulus then
      pure ⟨word, canonical⟩
    else
      throw "recursive fixture: noncanonical Goldilocks word"

private def parseState (text : String) :
    Except String (Nat × List F × List F × List F) := do
  let values ← (← Lean.Json.parse text).getArr?
  match values.toList with
  | [iteration, z0, current, message] =>
      let iteration ← iteration.getNat?
      let z0 ← decodeWords z0
      let current ← decodeWords current
      let message ← decodeWords message
      let words := fun values : List F => Value.array (values.map fun word => .atom word.val)
      let canonical := (Value.array [.atom iteration, words z0, words current, words message]).render
      unless text == canonical || text == canonical ++ "\n" do
        throw "recursive fixture: expected canonical numeric JSON with an optional final newline"
      pure (iteration, z0, current, message)
  | _ => throw "recursive fixture: expected [iteration,z0,current,message]"

private def run (w0 w1 w2 w3 inputPath childrenPath outputPath : String)
    (statePath : Option String) :
    IO UInt32 := do
  match NightstreamFPrime.Export.ParityEmitter.parseVerifierKey w0 w1 w2 w3 with
  | .error error => IO.eprintln error; pure 2
  | .ok context =>
      let inputText ← IO.FS.readFile inputPath
      let childrenText ← IO.FS.readFile childrenPath
      match NightstreamFPrime.Export.Stage1.PiCCSInputCheck.parse inputText,
          NightstreamFPrime.Export.Stage1.PiCCSInputCheck.parseRunning childrenText with
      | .ok input, .ok children =>
          match statePath with
          | none =>
              NightstreamFPrime.Export.ParityEmitter.runIO "emitted_recursive_step_fixture"
                (NightstreamFPrime.Export.Stage1.RecursiveStepFixture.valueIOWith
                  (fun state => some (NightstreamFPrime.Export.Stage1.Wide.BaseStepFixture.batch state)) context input children)
                [outputPath]
          | some path =>
              match parseState (← IO.FS.readFile path) with
              | .error error => IO.eprintln error; pure 2
              | .ok (iteration, z0, current, message) =>
                  NightstreamFPrime.Export.ParityEmitter.runIO "emitted_recursive_step_fixture"
                    (NightstreamFPrime.Export.Stage1.RecursiveStepFixture.valueFromStateIOWith
                      (fun state => some (NightstreamFPrime.Export.Stage1.Wide.BaseStepFixture.batch state))
                      context iteration z0 current message input children) [outputPath]
      | .error error, _ | _, .error error => IO.eprintln error; pure 2

def main (arguments : List String) : IO UInt32 :=
  match arguments with
  | [w0, w1, w2, w3, inputPath, childrenPath, outputPath] =>
      run w0 w1 w2 w3 inputPath childrenPath outputPath none
  | ["--", w0, w1, w2, w3, inputPath, childrenPath, outputPath] =>
      run w0 w1 w2 w3 inputPath childrenPath outputPath none
  | [w0, w1, w2, w3, inputPath, childrenPath, statePath, outputPath] =>
      run w0 w1 w2 w3 inputPath childrenPath outputPath (some statePath)
  | ["--", w0, w1, w2, w3, inputPath, childrenPath, statePath, outputPath] =>
      run w0 w1 w2 w3 inputPath childrenPath outputPath (some statePath)
  | _ => do
      IO.eprintln "usage: emitRecursiveStepFixture <context[4]> <PiCCS-input> <child-running> [<prior-state-message>] <output>"
      pure 2
