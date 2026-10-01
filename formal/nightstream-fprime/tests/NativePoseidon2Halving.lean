import NightstreamFPrime.Export.NativePoseidon2RoundCore

/-! Regression checks for the fixed-half lanes in the native partial round. -/

namespace NightstreamFPrime.Tests.NativePoseidon2Halving

open NightstreamFPrime.Export.NativePoseidon2

private def oddAtX3 : State64 where
  x0 := 0
  x1 := 0
  x2 := 0
  x3 := 0xfffffffeffffffff
  x4 := 0
  x5 := 0
  x6 := 2
  x7 := 0
  x8 := 0
  x9 := 0
  x10 := 0
  x11 := 0
  x12 := 0
  x13 := 0
  x14 := 0
  x15 := 0
  canonical := by decide

private def oddAtX6 : State64 where
  x0 := 0
  x1 := 0
  x2 := 0
  x3 := 2
  x4 := 0
  x5 := 0
  x6 := 0xfffffffeffffffff
  x7 := 0
  x8 := 0
  x9 := 0
  x10 := 0
  x11 := 0
  x12 := 0
  x13 := 0
  x14 := 0
  x15 := 0
  canonical := by decide

private def regression : IO Unit := do
  let first := State64.partialRound64 oddAtX3 0
  unless first.x3 == 0xffffffff00000000 &&
      first.x6 == 0xffffffff00000000 do
    throw (IO.userError "fixed-half regression at the odd x3 boundary")
  let second := State64.partialRound64 oddAtX6 0
  unless second.x3 == 1 && second.x6 == 1 do
    throw (IO.userError "fixed-half regression at the odd x6 boundary")

#eval regression

end NightstreamFPrime.Tests.NativePoseidon2Halving
