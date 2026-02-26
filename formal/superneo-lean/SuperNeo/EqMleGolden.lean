import SuperNeo.MLE

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

private def seedFor (base idx : Nat) : Nat :=
  base + idx * 0x9E3779B97F4A7C15

private def natArrayToCsvField (xs : Array Nat) : String :=
  String.intercalate ":" (xs.toList.map (fun x => toString x))

private def fVals (xs : Array F) : Array Nat :=
  xs.map (fun x => x.val)

private def fArrayToCsvField (xs : Array F) : String :=
  natArrayToCsvField (fVals xs)

private def genFVec (seed width : Nat) : Array F :=
  F.ofNatArray (genWords seed width)

private def genBoolVec (seed width : Nat) : Array F :=
  let ws := genWords seed width
  Id.run do
    let mut out := Array.replicate width (0 : F)
    for i in [0:width] do
      let w := ws[i]!
      let shift := (i + 7) % 29
      let bit := (w / (2 ^ shift)) % 2
      out := out.set! i (F.ofNat bit)
    return out

private def sumF (xs : Array F) : F :=
  xs.foldl (fun acc x => acc + x) 0

def eqGoldenCaseCount : Nat := 64
def eqGoldenWidth : Nat := 8
def mleGoldenCaseCount : Nat := 40

private def eqLine (idx : Nat) (isBoolCase : Bool) (x y : Array F) : String :=
  let expectedEq := eqPoly x y
  let expectedIndicator : Nat :=
    if isBoolCase then
      if x = y then 1 else 0
    else
      0
  let boolTag : Nat := if isBoolCase then 1 else 0
  s!"eq,{idx},{x.size},{boolTag},{fArrayToCsvField x},{fArrayToCsvField y},{expectedEq.val},{expectedIndicator}"

private def mleLine (idx : Nat) : String :=
  let ell := 2 + (genWords (seedFor 0xA55AA55AA55AA55A idx) 1)[0]! % 5
  let n := 2 ^ ell
  let v := genFVec (seedFor 0x0123456789ABCDEF idx) n
  let r := genFVec (seedFor 0xF00DFACECAFEBEEF idx) ell
  let expectedInner := mleByInnerProduct v r
  let expectedFold := mleByFolding v r
  let chi := rHat r n
  let expectedChiSum := sumF chi
  let probeIdx := (genWords (seedFor 0x0DDC0FFEE1234567 idx) 1)[0]! % n
  let expectedProbeWeight := chiWeight r probeIdx
  s!"mle,{idx},{ell},{fArrayToCsvField v},{fArrayToCsvField r},{expectedInner.val},{expectedFold.val},{expectedChiSum.val},{probeIdx},{expectedProbeWeight.val}"

def eqMleGoldenLines : Array String :=
  Id.run do
    let mut lines : Array String := #[]
    lines := lines.push "# superneo_eq_mle_v1"
    lines := lines.push s!"modulus,{q}"
    lines := lines.push s!"eq_cases,{eqGoldenCaseCount}"
    lines := lines.push s!"mle_cases,{mleGoldenCaseCount}"
    for i in [0:eqGoldenCaseCount] do
      let isBoolCase := i < eqGoldenCaseCount / 2
      let x :=
        if isBoolCase then
          genBoolVec (seedFor 0x1111222233334444 i) eqGoldenWidth
        else
          genFVec (seedFor 0x5555666677778888 i) eqGoldenWidth
      let y :=
        if isBoolCase then
          if i % 4 = 0 then
            x
          else
            genBoolVec (seedFor 0x9999AAAABBBBCCCC i) eqGoldenWidth
        else
          genFVec (seedFor 0xDDDDEEEEFFFF0001 i) eqGoldenWidth
      lines := lines.push (eqLine i isBoolCase x y)
    for i in [0:mleGoldenCaseCount] do
      lines := lines.push (mleLine i)
    return lines

def emitEqMleGolden : IO Unit := do
  for line in eqMleGoldenLines do
    IO.println line

def eqMleGoldenMain : IO UInt32 := do
  emitEqMleGolden
  pure 0

end SuperNeo
