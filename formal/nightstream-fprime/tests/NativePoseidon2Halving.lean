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
  x5 := 2
  x6 := 0
  x7 := 0
  canonical := by decide

private def oddAtX5 : State64 where
  x0 := 0
  x1 := 0
  x2 := 0
  x3 := 2
  x4 := 0
  x5 := 0xfffffffeffffffff
  x6 := 0
  x7 := 0
  canonical := by decide

private def regression : IO Unit := do
  let first := State64.partialRound64 oddAtX3 0
  unless first.x3 == 0xffffffff00000000 &&
      first.x5 == 0xffffffff00000000 do
    throw (IO.userError "fixed-half regression at the odd x3 boundary")
  let second := State64.partialRound64 oddAtX5 0
  unless second.x3 == 1 && second.x5 == 1 do
    throw (IO.userError "fixed-half regression at the odd x5 boundary")

#eval regression

end NightstreamFPrime.Tests.NativePoseidon2Halving
