import SuperNeo.EvalHom
import SuperNeo.Generated.Vectors

namespace SuperNeo

open F
open SuperNeo.Generated

private def barMatrix : Array (Array F) := barMatrixU64.map F.ofNatArray

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

private def fVals (xs : Array F) : Array Nat :=
  xs.map (fun x => x.val)

private def natArrayToCsvField (xs : Array Nat) : String :=
  String.intercalate ":" (xs.toList.map (fun x => toString x))

private def fArrayToCsvField (xs : Array F) : String :=
  natArrayToCsvField (fVals xs)

private def fMatrixToCsvField (m : Array (Array F)) : String :=
  String.intercalate ";" (m.toList.map fArrayToCsvField)

private def genFVec (seed width : Nat) : Array F :=
  F.ofNatArray (genWords seed width)

private def genMatrix (seed rows cols : Nat) : Array (Array F) :=
  Id.run do
    let mut out : Array (Array F) := #[]
    for i in [0:rows] do
      out := out.push (genFVec (seedFor seed i) cols)
    return out

private def dotF (a b : Array F) : F :=
  if a.size != b.size then
    0
  else
    Id.run do
      let mut acc : F := 0
      for i in [0:a.size] do
        acc := acc + a[i]! * b[i]!
      return acc

def p13EvalLinkGoldenCaseCount : Nat := 20
def p14EvalHomGoldenCaseCount : Nat := 20

private def linkLine (idx : Nat) (m : Array (Array F)) (z r : Array F) : String :=
  let ys := barMzRing barMatrix m z
  let weights := rHat r ys.size
  let y := evalRingVec ys weights
  let coeffSide := evalCoeffRows (coeffRowsOfRingVec ys) weights
  let ctY := ct y
  let ctSide := (dotF (ctRow ys) weights)
  let identity : Nat := if evalLinkForMatrix barMatrix m z r then 1 else 0
  s!"link,{idx},{m.size},{z.size},{fMatrixToCsvField m},{fArrayToCsvField z},{fArrayToCsvField r},{fArrayToCsvField y},{fArrayToCsvField coeffSide},{ctY.val},{ctSide.val},{identity}"

private def homLine
  (idx : Nat)
  (m : Array (Array F))
  (z1 z2 r : Array F)
  (ρ1 ρ2 : F) : String :=
  let y1 := evalBarMzAt barMatrix m z1 r
  let y2 := evalBarMzAt barMatrix m z2 r
  let yLin := vecAdd (vecScale ρ1 y1) (vecScale ρ2 y2)
  let zStar := linComb2Vec ρ1 ρ2 z1 z2
  let yDirect := evalBarMzAt barMatrix m zStar r
  let ctLin := ct yLin
  let ctFormula := ρ1 * ct y1 + ρ2 * ct y2
  let identity : Nat := if evalHom2 barMatrix m z1 z2 r ρ1 ρ2 then 1 else 0
  s!"hom,{idx},{m.size},{z1.size},{fMatrixToCsvField m},{fArrayToCsvField z1},{fArrayToCsvField z2},{fArrayToCsvField r},{ρ1.val},{ρ2.val},{fArrayToCsvField y1},{fArrayToCsvField y2},{fArrayToCsvField yLin},{fArrayToCsvField yDirect},{ctLin.val},{ctFormula.val},{identity}"

def p13p14GoldenLines : Array String :=
  Id.run do
    let mut lines : Array String := #[]
    lines := lines.push "# superneo_p13_p14_v1"
    lines := lines.push s!"modulus,{q}"
    lines := lines.push s!"d,{D}"
    lines := lines.push s!"eval_link_cases,{p13EvalLinkGoldenCaseCount}"
    lines := lines.push s!"eval_hom_cases,{p14EvalHomGoldenCaseCount}"

    for i in [0:p13EvalLinkGoldenCaseCount] do
      let rows := 3 + (i % 3)
      let nBlocks := 2 + (i % 2)
      let cols := nBlocks * d
      let m := genMatrix (seedFor 0xAAAABBBBCCCC1111 i) rows cols
      let z := genFVec (seedFor 0xDDDDEEEEFFFF2222 i) cols
      let rLen := 2 + (i % 2)
      let r := genFVec (seedFor 0x1234123412341234 i) rLen
      lines := lines.push (linkLine i m z r)

    for i in [0:p14EvalHomGoldenCaseCount] do
      let rows := 4 + (i % 2)
      let nBlocks := 2 + (i % 2)
      let cols := nBlocks * d
      let m := genMatrix (seedFor 0x0F0F0F0FABCD1234 i) rows cols
      let z1 := genFVec (seedFor 0x1111222233334444 i) cols
      let z2 := genFVec (seedFor 0x5555666677778888 i) cols
      let rLen := 2 + (i % 3)
      let r := genFVec (seedFor 0x9999AAAABBBBCCCC i) rLen
      let ρ1w := (genWords (seedFor 0xCAFEBABE00001111 i) 1)[0]!
      let ρ2w := (genWords (seedFor 0xDEADBEEF22223333 i) 1)[0]!
      let ρ1 := F.ofNat ρ1w
      let ρ2 := F.ofNat ρ2w
      lines := lines.push (homLine i m z1 z2 r ρ1 ρ2)

    return lines

def emitP13P14Golden : IO Unit := do
  for line in p13p14GoldenLines do
    IO.println line

def p13p14GoldenMain : IO UInt32 := do
  emitP13P14Golden
  pure 0

end SuperNeo
