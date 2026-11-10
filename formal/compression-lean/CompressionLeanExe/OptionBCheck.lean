import CompressionLean.Lowering.Compiled
import CompressionLean.Spec.OptionBRuns
import CompressionLeanExe.Parse
import CompressionLeanExe.Digest
import CompressionLeanExe.Key
import CompressionLeanExe.PackageJson

/-! Option B's final program against Rust (CW-6.5). The Rust test `option_b_final_program_matches_lean`
logs `FinalProgram::run` on the finish package, the state keys and both slots' layer-1 keys, one
instruction per `Backend` call; this check decodes the finish package from its artifact
(`StepPackage.Package.format`), builds `OptionB.finalProgram` and requires the same instructions.
It prints the run digest that the Rust pin `OPTION_B_PROGRAM_DIGEST` records. -/

attribute [local instance high] CompressionLean.Lowering.Compiled.glField

namespace CompressionLeanExe.OptionBCheck

open CompressionLean CompressionLean.Shrink CompressionLean.Circuit CompressionLeanExe.Parse
open CompressionLeanExe.Digest
open CompressionLean.Replay (Word)
open NightstreamFPrime.Export.Codec (Value)
open NightstreamFPrime.Export.OptionB.StepPackage (Package)

def digest (instrs : Array Instr) : List ℕ :=
  let run := Circuit.lower (F := Gl) instrs []
  (runDigest run (run.calls.flatMap Call.computedRows).toArray).toList.map Word.toNat

def stateKeys (finish : Package) : Parser (List Word) := do
  keyword "keys"
  let stateKeys ← many (← nat) glWord
  unless stateKeys.length = 8 do throw s!"{stateKeys.length} state key words, not 8"
  unless Spec.OptionB.fits finish do throw "the finish package does not fit its columns (`fits`)"
  pure stateKeys

/-- Rust's program against `lean`, which a task builds while this reads Rust's. The digest of the
program up to layer 1 runs in a second task while this computes the whole program's. -/
def check (finish : Package) (stateKeys : List Word) (lean : Task (Array Instr)) : Parser String := do
  let rust ← program
  let lean := lean.get
  samePrograms lean rust
  let upToLayer1 : Build Unit := do
    let (_, column) ← OptionB.finishPrefix finish stateKeys
    let _ ← OptionB.seed column (OptionB.bitsStart finish.application.1) 0
  let prefixDigest := Task.spawn fun _ => digest upToLayer1.program
  let wholeDigest := digest lean
  pure s!"finish package fits; match: all {lean.size} instructions of option B's final program, \
    {Lowering.Compiled.finishRowCount finish} finish rows, {finish.blocks.length} blocks; digest {wholeDigest}; \
    prefix digest {prefixDigest.get}"

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
        unless ← CompressionLeanExe.Key.printWithin key do return 1
        keys := keys.push key
    | .error message => IO.eprintln s!"{keyPath}: {message}"; return 2
  let package ← match ← CompressionLeanExe.PackageJson.readPackage finishPath with
    | .ok package => pure package
    | .error message => IO.eprintln s!"{finishPath}: {message}"; return 2
  IO.println s!"finish package decoded ({(← IO.monoMsNow) - started} ms)"
  let [light, refresh] := keys.toList | return 2
  let data ← IO.FS.readBinFile path
  let (words, start) ← match (stateKeys package |>.run 0).run data with
    | .ok read => pure read
    | .error message => IO.println s!"mismatch: {message}"; return 1
  let lean := Task.spawn fun _ => (CompressionLean.OptionB.finalProgram package words ![light, refresh]).program
  match (check package words lean |>.run start).run data with
  | .ok (message, _) => IO.println s!"{message} ({(← IO.monoMsNow) - started} ms)"; return 0
  | .error message => IO.println s!"mismatch: {message}"; return 1
