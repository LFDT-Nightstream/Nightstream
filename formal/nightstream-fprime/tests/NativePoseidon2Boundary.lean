import NightstreamFPrime.Export.NativePoseidon2RoundCore

/-! Runs the compiled native partial round at the edge values `1` and `q - 1` and compares
each lane with `Poseidon2.internalLayer`. `State64.partialRound64_denote` proves the equality for
every state. -/

namespace NightstreamFPrime.Tests.NativePoseidon2Boundary

open NightstreamFPrime.Spec
open NightstreamFPrime.Export.NativePoseidon2

/-- `q - 1` for `q = 2^64 - 59`. -/
private def top : UInt64 := 0xffffffffffffffc4

private def allTop : State64 where
  x0 := top
  x1 := top
  x2 := top
  x3 := top
  x4 := top
  x5 := top
  x6 := top
  x7 := top
  x8 := top
  x9 := top
  x10 := top
  x11 := top
  x12 := top
  x13 := top
  x14 := top
  x15 := top
  canonical := by decide

private def alternating : State64 where
  x0 := 1
  x1 := top
  x2 := 1
  x3 := top
  x4 := 1
  x5 := top
  x6 := 1
  x7 := top
  x8 := 1
  x9 := top
  x10 := 1
  x11 := top
  x12 := 1
  x13 := top
  x14 := 1
  x15 := top
  canonical := by decide

private def expected (state : State64) (constant : UInt64) : Poseidon2.State :=
  Poseidon2.internalLayer
    (Poseidon2.sbox (state.x0.denote + constant.denote) :: state.denote.drop 1)

private def regression : IO Unit := do
  for (name, state) in [("q - 1 in each lane", allTop), ("1 and q - 1", alternating)] do
    for constant in [0, 1, top] do
      unless (State64.partialRound64 state constant).denote == expected state constant do
        throw (IO.userError s!"the native partial round differs from the specification: {name}")

#eval regression

end NightstreamFPrime.Tests.NativePoseidon2Boundary
