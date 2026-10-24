import CompressionLean.Ports.Terminal.Package
import CompressionLean.Ports.Replay.Data
import NightstreamFPrime.Lifecycle.XOut

/-!
Owns the final program up to layer 1 as a builder program: the wrapper of
`FinalProgram::run` (`crates/nightstream/src/lifecycle/terminal/mod.rs:54-80`)
and `Package::apply` (`terminal/package.rs:179-237`). The layer-1 verifier
(`verify_with`, `crates/neo-spartan/src/lib.rs:692-703`) is a parameter: it
receives the claim and the transcript state that the package's output
columns hold.

- Statement: nine `public` words, the iteration, `z0` and the current state.
- Iteration: a hint for its inverse, and the product equals one.
- Package: one hint for the computed columns, then one value per system
  column (`column`), then the rows over those values. The prior state
  preimage starts with the state domain chunk and ends with the context and
  the statement words, so those columns are constants and statement words.
  The public context words are constants. Every other caller input is a
  proof word.
-/

namespace CompressionLean.Terminal.Program

open CompressionLean.Circuit CompressionLean.Terminal.Package
open CompressionLean.Replay (Word D)
open NightstreamFPrime.Layout

/-- `ClaimWords` over handles (`lib.rs:776-787`); a `K` value is
`(re, im)`. -/
structure ClaimVars where
  /-- `κ` rows of `D` lanes. -/
  commitment : Array (Array Var)
  /-- One ring element per public block. -/
  publicInput : Array (Array Var)
  point : Array (Var × Var)
  /-- `D` values. -/
  evalK : Array (Var × Var)
  /-- Per matrix, `D` values. -/
  evalA : Array (Array (Var × Var))

/-- A state preimage ends with the verifier context (4 words), then the
statement (`inputs.rs:39-43`). -/
def statementWords : ℕ := 9

def contextStart : ℕ := PilotProduction.stateHashWords - statementWords - 4

/-- The public inputs: the prior public input (270 words), the output
digest (4), the context (`PUBLIC_CONTEXT_START`, `package.rs:40`). -/
def publicContextStart : ℕ := 270 + 4

/-- The value of a system column (`apply`, `package.rs:191-212`). Hint
output `k` is value `computed + k`. -/
def column (statement : Array Var) (context : Array Word) (computed : Var) : Source → Build Var
  | .one => Build.const 1
  | .priv i =>
      if i < NightstreamFPrime.Lifecycle.stateDomainChunk.length then
        Build.const (Word.ofF (NightstreamFPrime.Lifecycle.stateDomainChunk.getD i 0))
      else if contextStart ≤ i ∧ i < contextStart + 4 then
        Build.const (context.getD (i - contextStart) 0)
      else if contextStart + 4 ≤ i ∧ i < PilotProduction.stateHashWords then
        pure (statement.getD (i - contextStart - 4) 0)
      else Build.priv
  | .pub i =>
      if publicContextStart ≤ i then Build.const (context.getD (i - publicContextStart) 0)
      else Build.priv
  | .computed k => pure (computed + k)

/-- Consecutive runs of `n` entries. -/
def chunks {α : Type} (n : ℕ) (list : Array α) : Array (Array α) :=
  (Array.range (list.size / n)).map fun k => list.extract (n * k) (n * k + n)

def pairs (list : Array Var) : Array (Var × Var) :=
  (chunks 2 list).map fun pair => (pair.getD 0 0, pair.getD 1 0)

/-- `Package::apply`: the rows over the statement and the context, then the
claim and the transcript state in the output columns. -/
def apply (package : Loaded) (statement : Array Var) (context : Array Word) :
    Build (ClaimVars × (Fin 16 → Var)) := do
  let computed ← Build.emit (.hint package.computed.size)
  let columns ← package.sources.mapM (column statement context computed)
  Build.r1cs package.system.toList columns.toList
  let read := fun (list : Array ℕ) => list.map fun k => columns.getD k 0
  let outputs := package.outputs
  let claim : ClaimVars := {
    commitment := chunks D (read outputs.commitment)
    publicInput := chunks D (read outputs.publicInput)
    point := pairs (read outputs.point)
    evalK := pairs (read outputs.evalK)
    evalA := (chunks (2 * D) (read outputs.evalA)).map pairs }
  pure (claim, fun lane => columns.getD (outputs.state.getD lane 0) 0)

/-- `FinalProgram::run`, with `layer1` for `verify_with`. -/
def finalProgram {α : Type} (package : Loaded) (context : Array Word)
    (layer1 : ClaimVars → (Fin 16 → Var) → Build α) : Build α := do
  let statement ← (Array.range statementWords).mapM fun _ => Build.pub
  -- An active proof has a positive iteration.
  let inverse ← Build.emit (.hint 1)
  let product ← Build.mul (statement.getD 0 0) inverse
  let one ← Build.const 1
  Build.assertEqual product one
  let (claim, state) ← apply package statement context
  layer1 claim state

end CompressionLean.Terminal.Program
