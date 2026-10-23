import CompressionLean.Circuit.Terminal
import CompressionLean.Circuit.Parse
import CompressionLean.Circuit.Verifier
import CompressionLean.Circuit.Digest
import CompressionLean.Final.KeyFacts

/-!
The `terminalCheck` executable. It reads an export of `nightstream`'s
`tests/lifecycle_native/terminal_ir.rs`: the context words, the IR that Rust
logs for `FinalProgram::run`, and for a run that stops at `verify_with` the
claim and state handles that `Package::apply` hands to layer 1. It loads the
Lean terminal package (`Package.load`), after it checks that every compact
row invocation names a template (`templatesNamed`), and writes
`finalProgram`. Without a
layer-1 key the layer-1 step returns its arguments, and the check requires
the same instructions, one by one, and the same handles. With the key file
of `neo-spartan`'s `lowering.rs` the step is `verifyWith`, the key must have
`KeyFacts` (`keyFailure`), and the check requires every instruction of the
final program. It prints `match`, the counts and the run digest of Lean's
program (`runDigest`), or the first difference and exits with 1.
-/

namespace CompressionLean.TerminalCheck

open CompressionLean.Shrink CompressionLean.Circuit CompressionLean.Circuit.Parse
open CompressionLean.Circuit.Package CompressionLean.Circuit.Terminal CompressionLean.Circuit.Digest
open CompressionLean.Replay (Gl)

instance : BEq (R1csRow Gl) := ⟨fun x y => x.a == y.a && x.b == y.b && x.c == y.c⟩

def coordinates {n : ℕ} (a : Fin n → Var) : List Var := List.ofFn a

/-- The first index where two arrays differ, if any. -/
def firstDifference {α : Type} [BEq α] (x y : Array α) : Option ℕ :=
  (List.range (max x.size y.size)).find? fun k => !(x[k]? == y[k]?)

/-- Why two instructions differ, or `none` when they are equal. -/
def difference : Instr → Instr → Option String
  | .const a, .const b => if a == b then none else s!"constants {a.toNat}, {b.toNat}"
  | .priv, .priv | .pub, .pub => none
  | .linear p, .linear q =>
      if p == q then none else s!"linear {p.map (·.1)}, {q.map (·.1)}"
  | .mul a b, .mul c d => if a == c && b == d then none else s!"mul {a} {b}, {c} {d}"
  | .extMul a b, .extMul c d =>
      if coordinates a == coordinates c && coordinates b == coordinates d then none else "ext_mul"
  | .extInv a, .extInv b => if coordinates a == coordinates b then none else "ext_inverse"
  | .assertZero a, .assertZero b => if a == b then none else s!"assert_zero {a}, {b}"
  | .bits a n, .bits b m => if a == b && n == m then none else s!"bits {a} {n}, {b} {m}"
  | .permute a, .permute b => if coordinates a == coordinates b then none else "permute"
  | .hint n, .hint m => if n == m then none else s!"hint {n}, {m}"
  | .r1cs rows columns, .r1cs rows' columns' =>
      match firstDifference columns.toArray columns'.toArray with
      | some k => s!"r1cs column {k}: {columns[k]?}, {columns'[k]?}"
      | none => (firstDifference rows.toArray rows'.toArray).map fun k =>
          let side := fun (r : Option (R1csRow Gl)) =>
            r.map fun r => [r.a, r.b, r.c].map fun s => s.map fun t => (t.1, t.2.toNat)
          s!"r1cs row {k}: {side rows[k]?}, {side rows'[k]?}"
  | _, _ => "opcodes"

def handles : Parser (List Var) := do many (← nat) nat

def flatPairs (list : Array (Var × Var)) : List Var := list.toList.flatMap fun (re, im) => [re, im]

/-- Lean's instructions against Rust's, one by one. -/
def sameInstrs (lean rust : Array Instr) : Parser Unit := do
  for k in [0:max lean.size rust.size] do
    match lean[k]?, rust[k]? with
    | some l, some r =>
        if let some why := difference l r then throw s!"instruction {k}: {why} (Lean, Rust)"
    | _, _ => throw s!"instructions: Lean {lean.size}, Rust {rust.size}"

/-- The terminal claim as the verifier's statement. -/
def claimView (claim : ClaimVars) : ClaimView :=
  { commitment := claim.commitment, publicInput := claim.publicInput, point := claim.point,
    evalK := claim.evalK, evalA := claim.evalA }

/-- The run digest of a program: its lowering's counts and rows. The rows do
not depend on the statement values. -/
def digest (instrs : Array Instr) : List ℕ :=
  let run := lower (F := G) instrs []
  (runDigest run (run.calls.flatMap Call.computedRows).toArray).toList.map Gl.toNat

/-- Compare `finalProgram` on `package` with the export: up to layer 1, or
with `verifyWith` for `key`. -/
def check (package : Loaded) (key : Option Layer1Input) : Parser String := do
  keyword "context"
  let context := (← many (← nat) glWord).toArray
  let rust ← program
  if let some key := key then
    let lean := (finalProgram package context fun claim state => verifyWith key state (claimView claim)).program
    sameInstrs lean rust
    return s!"match: all {lean.size} instructions of the final program, {package.system.size} package rows; \
      digest {digest lean}"
  let ((claim, state), built) :=
    (finalProgram package context fun claim state => pure (claim, state)).run {}
  let lean := built.instrs
  sameInstrs lean rust
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

end CompressionLean.TerminalCheck

open CompressionLean.TerminalCheck in
def main (args : List String) : IO UInt32 := do
  let started ← IO.monoMsNow
  let (path, key) ← match args with
    | [path] => pure (path, none)
    | [path, keyPath] =>
        match Lean.Json.parse (← IO.FS.readFile keyPath) >>= Lean.fromJson? (α := CompressionLean.Circuit.Layer1Input) with
        | .ok key =>
            match CompressionLean.Final.keyFailure key with
            | some fact => IO.println s!"key fact fails: {fact}"; return 1
            | none => IO.println "key facts hold"; pure (path, some key)
        | .error message => IO.eprintln s!"{keyPath}: {message}"; return 2
    | _ => IO.eprintln "usage: terminalCheck <export.txt> [<layer1.json>]"; return 2
  let circuit := CompressionLean.Terminal.circuitPackage ()
  unless CompressionLean.Circuit.Package.templatesNamed circuit do
    IO.println "a compact row invocation of the Lean package names no template"; return 1
  IO.println "package templates hold"
  match CompressionLean.Circuit.Package.load circuit CompressionLean.Terminal.outputs with
  | .error message => IO.println s!"the Lean package does not load: {message}"; return 1
  | .ok package =>
      match (check package key |>.run 0).run (← IO.FS.readBinFile path) with
      | .ok (message, _) => IO.println s!"{message} ({(← IO.monoMsNow) - started} ms)"; return 0
      | .error message => IO.println s!"mismatch: {message}"; return 1
