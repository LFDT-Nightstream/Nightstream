import SuperNeo.Ring

namespace SuperNeo

open F

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

private def genCoeffVecs (seed count : Nat) : Array Coeffs :=
  let ws := genWords seed (count * D)
  Id.run do
    let mut out : Array Coeffs := #[]
    for i in [0:count] do
      let start := i * D
      let stop := start + D
      out := out.push (F.ofNatArray (ws.extract start stop))
    return out

private def coeffVals (a : Coeffs) : Array Nat :=
  a.map (fun x => x.val)

private def natArrayToCsvField (xs : Array Nat) : String :=
  String.intercalate ":" (xs.toList.map (fun x => toString x))

private def mulLine (idx : Nat) (a b : Coeffs) : String :=
  let c := mulRq a b
  s!"mul,{idx},{natArrayToCsvField (coeffVals a)},{natArrayToCsvField (coeffVals b)},{natArrayToCsvField (coeffVals c)},{(ct c).val}"

def ringGoldenCaseCount : Nat := 128

def ringGoldenLines : Array String :=
  let as := genCoeffVecs 0x123456789ABCDEF0 ringGoldenCaseCount
  let bs := genCoeffVecs 0x0FEDCBA987654321 ringGoldenCaseCount
  Id.run do
    let mut lines : Array String := #[]
    lines := lines.push "# superneo_ring_v1"
    lines := lines.push s!"modulus,{q}"
    lines := lines.push s!"d,{D}"
    lines := lines.push s!"cases,{ringGoldenCaseCount}"
    for i in [0:ringGoldenCaseCount] do
      lines := lines.push (mulLine i as[i]! bs[i]!)
    return lines

def emitRingGolden : IO Unit := do
  for line in ringGoldenLines do
    IO.println line

def ringGoldenMain : IO UInt32 := do
  emitRingGolden
  pure 0

end SuperNeo
