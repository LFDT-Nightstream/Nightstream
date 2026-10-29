import CompressionLean.Ports.Terminal.Program
import CompressionLeanExe.Parse
import CompressionLean.Ports.Circuit.Verifier
import CompressionLean.Sim.Claim
import CompressionLean.Checks.Templates
import CompressionLeanExe.Digest
import CompressionLeanExe.Key

namespace CompressionLeanExe.TerminalCheck

open CompressionLean.Spec
open CompressionLean CompressionLean.Shrink CompressionLean.Circuit CompressionLeanExe.Parse
open CompressionLean.Terminal.Package CompressionLean.Terminal.Program CompressionLeanExe.Digest
open CompressionLean.Replay (Word)

def coordinates {n : ℕ} (a : Fin n → Var) : List Var := List.ofFn a

def handles : Parser (List Var) := do many (← nat) nat

def flatPairs (list : Array (Var × Var)) : List Var := list.toList.flatMap fun (re, im) => [re, im]

def digest (instrs : Array Instr) : List ℕ :=
  let run := lower (F := Gl) instrs []
  (runDigest run (run.calls.flatMap Call.computedRows).toArray).toList.map Word.toNat

def check (package : Loaded) (key : Option Layer1Key) : Parser String := do
  keyword "context"
  let context := (← many (← nat) glWord).toArray
  let rust ← program
  if let some key := key then
    let lean := (finalProgram package context fun claim state => verifyWith key state claim.view).program
    samePrograms lean rust
    return s!"match: all {lean.size} instructions of the final program, {package.system.size} package rows; \
      digest {digest lean}"
  let ((claim, state), built) :=
    (finalProgram package context fun claim state => pure (claim, state)).run {}
  let lean := built.instrs
  samePrograms lean rust
  keyword "commitment"
  expect "commitment" (claim.commitment.toList.flatMap Array.toList) (← handles)
  keyword "public"
  expect "public input" (claim.publicInput.toList.flatMap Array.toList) (← handles)
  keyword "point"
  expect "point" (flatPairs claim.point) (← handles)
  keyword "eval_k"
  expect "Eval_K" (flatPairs claim.evalK) (← handles)
  keyword "eval_a"
  expect "Eval_A" (claim.evalA.toList.flatMap flatPairs) (← handles)
  keyword "state"
  expect "state" (coordinates state) (← handles)
  pure s!"match: {lean.size} instructions, {package.system.size} rows, {package.sources.size} columns, \
    {package.computed.size} computed; digest {digest lean}"

end CompressionLeanExe.TerminalCheck

open CompressionLeanExe.TerminalCheck in
def main (args : List String) : IO UInt32 := do
  let started ← IO.monoMsNow
  let (path, key) ← match args with
    | [path] => pure (path, none)
    | [path, keyPath] =>
        match ← CompressionLeanExe.Parse.readJson CompressionLean.Circuit.Layer1Key keyPath with
        | .ok key =>
            unless ← CompressionLeanExe.Key.printFacts key do return 1
            CompressionLeanExe.Key.printWidth key
            pure (path, some key)
        | .error message => IO.eprintln s!"{keyPath}: {message}"; return 2
    | _ => IO.eprintln "usage: terminalCheck <export.txt> [<layer1.json>]"; return 2
  let circuit := CompressionLean.Terminal.circuitPackage ()
  unless CompressionLean.Spec.templatesNamed circuit do
    IO.println "a compact row invocation of the Lean package names no template"; return 1
  IO.println "package templates hold"
  match CompressionLean.Terminal.Package.load circuit CompressionLean.Terminal.outputs with
  | .error message => IO.println s!"the Lean package does not load: {message}"; return 1
  | .ok package =>
      match (check package key |>.run 0).run (← IO.FS.readBinFile path) with
      | .ok (message, _) => IO.println s!"{message} ({(← IO.monoMsNow) - started} ms)"; return 0
      | .error message => IO.println s!"mismatch: {message}"; return 1
