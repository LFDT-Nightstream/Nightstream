import SuperNeo.Field

namespace SuperNeo

open F
open Goldilocks

private def u64Mod : Nat := 2 ^ (64 : Nat)

private def lcgMul : Nat := 6364136223846793005
private def lcgAdd : Nat := 1442695040888963407

private def lcgStep (x : Nat) : Nat :=
  (x * lcgMul + lcgAdd) % u64Mod

private def genWords (seed count : Nat) : Array Nat :=
  Id.run do
    let mut out : Array Nat := #[]
    let mut x : Nat := seed % u64Mod
    for _ in [0:count] do
      x := lcgStep x
      out := out.push x
    return out

private def genPairs (seed count : Nat) : Array (Nat × Nat) :=
  let ws := genWords seed (count * 2)
  Id.run do
    let mut out : Array (Nat × Nat) := #[]
    for i in [0:count] do
      out := out.push (ws[2 * i]!, ws[2 * i + 1]!)
    return out

private def canonicalNonZero (x : Nat) : Nat :=
  let v := x % q
  if v = 0 then 1 else v

private def addLine (a b : Nat) : String :=
  let out := ((F.ofNat a) + (F.ofNat b)).val
  s!"add,{a},{b},{out}"

private def subLine (a b : Nat) : String :=
  let out := ((F.ofNat a) - (F.ofNat b)).val
  s!"sub,{a},{b},{out}"

private def mulLine (a b : Nat) : String :=
  let out := ((F.ofNat a) * (F.ofNat b)).val
  s!"mul,{a},{b},{out}"

private def negLine (a : Nat) : String :=
  let out := (-(F.ofNat a)).val
  s!"neg,{a},{out}"

private def invLine (a : Nat) : String :=
  let x := canonicalNonZero a
  let out := (F.inv (F.ofNat x)).val
  s!"inv,{x},{out}"

def goldilocksGoldenLines : Array String :=
  let addCases := genPairs 0x0123456789ABCDEF 128
  let subCases := genPairs 0x89ABCDEF01234567 128
  let mulCases := genPairs 0xF00DFACECAFEBEEF 128
  let unaryCases := genWords 0x0DDC0FFEE1234567 128
  Id.run do
    let mut lines : Array String := #[]
    lines := lines.push "# superneo_goldilocks_v1"
    lines := lines.push s!"modulus,{q}"
    for (a, b) in addCases do
      lines := lines.push (addLine a b)
    for (a, b) in subCases do
      lines := lines.push (subLine a b)
    for (a, b) in mulCases do
      lines := lines.push (mulLine a b)
    for a in unaryCases do
      lines := lines.push (negLine a)
    for a in unaryCases do
      lines := lines.push (invLine a)
    return lines

def emitGoldilocksGolden : IO Unit := do
  for line in goldilocksGoldenLines do
    IO.println line

def goldilocksGoldenMain : IO UInt32 := do
  emitGoldilocksGolden
  pure 0

end SuperNeo

def main : IO UInt32 :=
  SuperNeo.goldilocksGoldenMain
