import CompressionLean.Ports.OptionB.Program
import CompressionLeanExe.Parse
import CompressionLeanExe.Digest
import CompressionLeanExe.Key

/-! Option B's final program against Rust (CW-6.5). The Rust test `option_b_final_program_matches_lean`
logs `FinalProgram::run` on the finish package, the state keys and both slots' layer-1 keys, one
instruction per `Backend` call; this check decodes the finish package from its artifact
(`StepPackage.Package.format`), builds `OptionB.finalProgram` and requires the same instructions.
It prints the run digest that the Rust pin `OPTION_B_PROGRAM_DIGEST` records. -/

namespace CompressionLeanExe.OptionBCheck

open CompressionLean CompressionLean.Shrink CompressionLean.Circuit CompressionLeanExe.Parse
open CompressionLeanExe.Digest
open CompressionLean.Replay (Word)
open NightstreamFPrime.Export.Codec (Value)
open NightstreamFPrime.Export.OptionB.StepPackage (Package)

def digitsEnd (data : ByteArray) (pos : ℕ) : ℕ :=
  if h : pos < data.size then if 48 ≤ data[pos] ∧ data[pos] ≤ 57 then digitsEnd data (pos + 1) else pos
  else pos
termination_by data.size - pos

mutual

/-- One JSON value of the package grammar: a natural number or an array of values. -/
partial def value (data : ByteArray) (start : ℕ) : Except String (Value × ℕ) := do
  let pos := skipSpaces data start
  if pos < data.size ∧ data[pos]! == 91 then
    items data (skipSpaces data (pos + 1)) #[]
  else
    let stop := digitsEnd data pos
    if stop == pos then throw s!"expected a number or an array at byte {pos}"
    pure (.atom ((data.extract pos stop).foldl (fun acc b => 10 * acc + (b - 48).toNat) 0), stop)

partial def items (data : ByteArray) (pos : ℕ) (out : Array Value) : Except String (Value × ℕ) := do
  if pos < data.size ∧ data[pos]! == 93 then return (.array out.toList, pos + 1)
  let (item, next) ← value data pos
  let next := skipSpaces data next
  if next < data.size ∧ data[next]! == 44 then items data (skipSpaces data (next + 1)) (out.push item)
  else if next < data.size ∧ data[next]! == 93 then return (.array (out.push item).toList, next + 1)
  else throw s!"expected , or ] at byte {next}"

end

def digest (instrs : Array Instr) : List ℕ :=
  let run := Circuit.lower (F := Gl) instrs []
  (runDigest run (run.calls.flatMap Call.computedRows).toArray).toList.map Word.toNat

def check (finish : Package) (keys : Fin 2 → Layer1Key) : Parser String := do
  keyword "keys"
  let stateKeys ← many (← nat) glWord
  let rust ← program
  let lean := (OptionB.finalProgram finish stateKeys keys).program
  samePrograms lean rust
  let upToLayer1 : Build Unit := do
    let (_, first) ← OptionB.finishPrefix finish stateKeys
    let _ ← OptionB.seed (fun k => first + k) (OptionB.bitsStart finish.application.1) 0
  pure s!"match: all {lean.size} instructions of option B's final program, \
    {(OptionB.finishRows finish).length} finish rows, {finish.blocks.length} blocks; digest {digest lean}; \
    prefix digest {digest upToLayer1.program}"

end CompressionLeanExe.OptionBCheck

open CompressionLeanExe.OptionBCheck in
def main (args : List String) : IO UInt32 := do
  let started ← IO.monoMsNow
  let [path, finishPath, lightPath, refreshPath] := args
    | IO.eprintln "usage: optionBCheck <export.txt> <finish.json> <light-layer1.json> <refresh-layer1.json>"
      return 2
  let mut keys : Array CompressionLean.Circuit.Layer1Key := #[]
  for keyPath in [lightPath, refreshPath] do
    match ← CompressionLeanExe.Parse.readJson CompressionLean.Circuit.Layer1Key keyPath with
    | .ok key =>
        unless ← CompressionLeanExe.Key.printFacts key do return 1
        keys := keys.push key
    | .error message => IO.eprintln s!"{keyPath}: {message}"; return 2
  let package ← match value (← IO.FS.readBinFile finishPath) 0 with
    | .ok (json, _) =>
        match NightstreamFPrime.Export.OptionB.StepPackage.Package.format.decode json with
        | .ok package => pure package
        | .error message => IO.eprintln s!"{finishPath}: {message}"; return 2
    | .error message => IO.eprintln s!"{finishPath}: {message}"; return 2
  IO.println s!"finish package decoded ({(← IO.monoMsNow) - started} ms)"
  let [light, refresh] := keys.toList | return 2
  match (check package ![light, refresh] |>.run 0).run (← IO.FS.readBinFile path) with
  | .ok (message, _) => IO.println s!"{message} ({(← IO.monoMsNow) - started} ms)"; return 0
  | .error message => IO.println s!"mismatch: {message}"; return 1
