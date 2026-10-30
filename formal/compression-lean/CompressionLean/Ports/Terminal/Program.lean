import CompressionLean.Ports.Terminal.Package
import CompressionLean.Ports.Replay.Data
import NightstreamFPrime.Lifecycle.XOut

namespace CompressionLean.Terminal.Program

open CompressionLean.Circuit CompressionLean.Terminal.Package
open CompressionLean.Replay (Word D)
open NightstreamFPrime.Layout

structure ClaimVars where
  commitment : Array (Array Var)
  publicInput : Array (Array Var)
  point : Array (Var × Var)
  evalK : Array (Var × Var)
  evalA : Array (Array (Var × Var))

def statementWords : ℕ := 9

def contextStart : ℕ := PilotProduction.stateHashWords - statementWords - 4

def publicContextStart : ℕ := 270 + 4

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

def chunks {α : Type} (n : ℕ) (list : Array α) : Array (Array α) :=
  (Array.range (list.size / n)).map fun k => list.extract (n * k) (n * k + n)

def pairs (list : Array Var) : Array (Var × Var) :=
  (chunks 2 list).map fun pair => (pair.getD 0 0, pair.getD 1 0)

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

def finalProgram {α : Type} (package : Loaded) (context : Array Word)
    (layer1 : ClaimVars → (Fin 16 → Var) → Build α) : Build α := do
  let statement ← (Array.range statementWords).mapM fun _ => Build.pub
  let inverse ← Build.emit (.hint 1)
  let product ← Build.mul (statement.getD 0 0) inverse
  let one ← Build.const 1
  Build.assertEqual product one
  let (claim, state) ← apply package statement context
  layer1 claim state

end CompressionLean.Terminal.Program
