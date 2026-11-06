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

def chunks {α : Type} (n : ℕ) (list : Array α) : Array (Array α) :=
  (Array.range (list.size / n)).map fun k => list.extract (n * k) (n * k + n)

def pairs (list : Array Var) : Array (Var × Var) :=
  (chunks 2 list).map fun pair => (pair.getD 0 0, pair.getD 1 0)

end CompressionLean.Terminal.Program
