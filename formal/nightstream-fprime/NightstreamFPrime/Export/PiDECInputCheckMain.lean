import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.PiDECInputCheck

private def run (p0 p1 p2 p3 inputPath messagesPath outputPath : String) : IO UInt32 := do
  let inputText ← IO.FS.readFile inputPath
  let messagesText ← IO.FS.readFile messagesPath
  let parsed := do
    let identity ← NightstreamFPrime.Export.ParityEmitter.parseVerifierKey p0 p1 p2 p3
    let input ← NightstreamFPrime.Export.Stage1.PiCCSInputCheck.parse inputText
    let messages ← NightstreamFPrime.Export.Stage1.PiCCSInputCheck.parseRunning messagesText
    pure (identity, input, messages)
  match parsed with
  | .error error =>
      IO.eprintln error
      pure 2
  | .ok (identity, input, messages) =>
      NightstreamFPrime.Export.ParityEmitter.runIO "checked_pi_dec_input"
        (NightstreamFPrime.Export.Stage1.PiDECInputCheck.checkValueIO input messages identity)
        [outputPath]

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Stage1

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok result => pure result
  | .error error => throw (IO.userError error)

private def decodeExtension (value : Lean.Json) : Except String K := do
  let words ← PiCCSInputCheck.decodeVector 2 PiCCSInputCheck.decodeField value
  return ⟨words.get 0, words.get 1⟩

/-- Read complete independently computed commitments and evaluations. Public
inputs are filled later by the verifier-owned children of the derived parent. -/
private def decodeReplay (commitmentText evaluationText : String) :
    Except String PiDECInputCheck.Messages := do
  let commitmentFields ← (← Lean.Json.parse commitmentText).getArr?
  let rows ← match commitmentFields.toList with
    | [schema, blocks, first, finish, values] => do
        let expected := Poseidon2HashChainV1Setup.messageColumns
        unless (← schema.getNat?) == 1 && (← blocks.getNat?) == expected &&
            (← first.getNat?) == 0 && (← finish.getNat?) == expected do
          throw "expected a complete selected Lean commitment result"
        PiCCSInputCheck.decodeVector 22
          (PiCCSInputCheck.decodeVector 16
            (PiCCSInputCheck.decodeVector 54 PiCCSInputCheck.decodeField)) values
    | _ => throw "expected five complete commitment fields"
  let evaluationFields ← (← Lean.Json.parse evaluationText).getArr?
  match evaluationFields.toList with
  | [schema, blocks, point, pad, matrix] => do
      unless (← schema.getNat?) == 1 &&
          (← blocks.getNat?) == Poseidon2HashChainV1Setup.messageColumns do
        throw "expected a complete selected Lean evaluation result"
      let point ← PiCCSInputCheck.decodeVector 28 decodeExtension point
      let pad ← PiCCSInputCheck.decodeVector 16
        (PiCCSInputCheck.decodeVector 54 decodeExtension) pad
      let matrix ← PiCCSInputCheck.decodeVector 16
        (PiCCSInputCheck.decodeVector 14
          (PiCCSInputCheck.decodeVector 54 decodeExtension)) matrix
      return {
        point := point
        commitments := Vector.ofFn fun child : Fin 16 =>
          Vector.ofFn fun coordinate : Fin 1188 =>
            ((rows.get ⟨coordinate.val / 54, by have := coordinate.isLt; omega⟩).get child).get
              ⟨coordinate.val % 54, Nat.mod_lt _ (by decide)⟩
        publicInputs := Vector.replicate 16 (Vector.replicate 270 0)
        evalK := pad
        evalA := matrix }
  | _ => throw "expected five complete evaluation fields"

private def fromReplay (p0 p1 p2 p3 : String)
    (inputPath commitmentsPath evaluationsPath childrenPath outputPath : System.FilePath) :
    IO UInt32 := do
  let outputs ← [childrenPath, outputPath].mapM fun path => do
    let some name := path.fileName | throw (IO.userError "expected a final output file name")
    return (← IO.FS.realPath (path.parent.getD ".")) / name
  unless outputs.eraseDups.length == outputs.length do
    throw (IO.userError "duplicate PiDEC replay output path")
  for path in outputs do
    unless !(← path.pathExists) do throw (IO.userError "output already exists")
  let identity ← checked (NightstreamFPrime.Export.ParityEmitter.parseVerifierKey p0 p1 p2 p3)
  let input ← checked (PiCCSInputCheck.parse (← IO.FS.readFile inputPath))
  let supplied ← checked (decodeReplay (← IO.FS.readFile commitmentsPath)
    (← IO.FS.readFile evaluationsPath))
  let previous ← PiRLCInputCheck.checkIO input identity
  let some parent := previous.parent
    | throw (IO.userError "PiCCS/PiRLC rejected or returned no parent")
  unless PiDECInputCheck.parentBounded parent do
    throw (IO.userError "PiDEC parent public input is out of range")
  unless decide (supplied.point.toList = parent.point.coordinates) do
    throw (IO.userError "Lean evaluation point differs from the derived parent")
  let messages : PiDECInputCheck.Messages := { supplied with
    publicInputs := Vector.ofFn fun child => Vector.ofFn fun coordinate =>
      (PiDECInputCheck.children parent supplied child).publicInput coordinate }
  unless PiDECInputCheck.accepted parent messages do
    throw (IO.userError "independent PiDEC replay rejected")
  let children := PiDECInputCheck.runningValue parent messages
  let result := Value.array (previous.fields ++
    [PiDECInputCheck.inputValue parent messages, PiDECInputCheck.resultValue parent messages])
  for path in outputs do
    unless !(← path.pathExists) do throw (IO.userError "output already exists")
  for (path, value) in outputs.zip [children, result] do
    IO.FS.writeFile path (value.render ++ "\n")
  IO.println "independent_pidec_replay_complete"
  return 0

def main (arguments : List String) : IO UInt32 :=
  match arguments with
  | ["from-replay", p0, p1, p2, p3, inputPath, commitmentsPath, evaluationsPath,
      childrenPath, outputPath] =>
      fromReplay p0 p1 p2 p3 inputPath commitmentsPath evaluationsPath childrenPath outputPath
  | ["--", "from-replay", p0, p1, p2, p3, inputPath, commitmentsPath, evaluationsPath,
      childrenPath, outputPath] =>
      fromReplay p0 p1 p2 p3 inputPath commitmentsPath evaluationsPath childrenPath outputPath
  | [p0, p1, p2, p3, inputPath, messagesPath, outputPath] =>
      run p0 p1 p2 p3 inputPath messagesPath outputPath
  | ["--", p0, p1, p2, p3, inputPath, messagesPath, outputPath] =>
      run p0 p1 p2 p3 inputPath messagesPath outputPath
  | _ => do
      IO.eprintln "usage: checkPiDECInput <package[4]> <PiCCS-input> <children> <output>"
      IO.eprintln "       checkPiDECInput from-replay <package[4]> <PiCCS-input> <Lean-commitments> <Lean-evaluations> <new-children> <new-output>"
      pure 2
