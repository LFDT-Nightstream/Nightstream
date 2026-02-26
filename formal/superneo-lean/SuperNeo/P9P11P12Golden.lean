import SuperNeo.Embedding
import SuperNeo.BarLift
import SuperNeo.MatrixTransform
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

private def fBlocksToCsvField (bs : Array (Array F)) : String :=
  String.intercalate "|" (bs.toList.map fArrayToCsvField)

private def genFVec (seed width : Nat) : Array F :=
  F.ofNatArray (genWords seed width)

private def genMatrix (seed rows cols : Nat) : Array (Array F) :=
  Id.run do
    let mut out : Array (Array F) := #[]
    for i in [0:rows] do
      out := out.push (genFVec (seedFor seed i) cols)
    return out

def p9EmbedGoldenCaseCount : Nat := 24
def p11BarLiftGoldenCaseCount : Nat := 24
def p12MatrixGoldenCaseCount : Nat := 16

private def embedLine (idx : Nat) (input : Array F) : String :=
  let blocks := embedVec input
  let roundTrip : Nat := if unembedVec blocks = input then 1 else 0
  s!"embed,{idx},{fArrayToCsvField input},{fBlocksToCsvField blocks},{roundTrip}"

private def barLiftLine (idx : Nat) (v w : Array F) (scalar : F) : String :=
  let liftV := barLiftVec barMatrix v
  let liftW := barLiftVec barMatrix w
  let liftAdd := barLiftVec barMatrix (vecAdd v w)
  let liftScale := barLiftVec barMatrix (vecScale scalar v)
  s!"barlift,{idx},{fArrayToCsvField v},{fArrayToCsvField w},{scalar.val},{fArrayToCsvField liftV},{fArrayToCsvField liftW},{fArrayToCsvField liftAdd},{fArrayToCsvField liftScale}"

private def matrixLine (idx : Nat) (m : Array (Array F)) (z : Array F) : String :=
  let mz := matrixVecDirect m z
  let ctBar := matrixVecCtBar barMatrix m z
  let identity : Nat := if mz = ctBar then 1 else 0
  s!"matrix,{idx},{m.size},{z.size},{fMatrixToCsvField m},{fArrayToCsvField z},{fArrayToCsvField mz},{fArrayToCsvField ctBar},{identity}"

def p9p11p12GoldenLines : Array String :=
  Id.run do
    let mut lines : Array String := #[]
    lines := lines.push "# superneo_p9_p11_p12_v1"
    lines := lines.push s!"modulus,{q}"
    lines := lines.push s!"d,{D}"
    lines := lines.push s!"embed_cases,{p9EmbedGoldenCaseCount}"
    lines := lines.push s!"barlift_cases,{p11BarLiftGoldenCaseCount}"
    lines := lines.push s!"matrix_cases,{p12MatrixGoldenCaseCount}"

    for i in [0:p9EmbedGoldenCaseCount] do
      let nBlocks := 1 + (i % 3)
      let width := nBlocks * d
      let input := genFVec (seedFor 0x13579BDF2468ACE0 i) width
      lines := lines.push (embedLine i input)

    for i in [0:p11BarLiftGoldenCaseCount] do
      let nBlocks := 2 + (i % 2)
      let width := nBlocks * d
      let v := genFVec (seedFor 0xAAAABBBBCCCCDDDD i) width
      let w := genFVec (seedFor 0xDDDDAAAACCCCBBBB i) width
      let scalarWord := (genWords (seedFor 0x0123456789ABCDEF i) 1)[0]!
      let scalar := F.ofNat scalarWord
      lines := lines.push (barLiftLine i v w scalar)

    for i in [0:p12MatrixGoldenCaseCount] do
      let rows := 2 + (i % 3)
      let nBlocks := 2 + (i % 2)
      let cols := nBlocks * d
      let m := genMatrix (seedFor 0xCAFEBABE11112222 i) rows cols
      let z := genFVec (seedFor 0x9999888877776666 i) cols
      lines := lines.push (matrixLine i m z)

    return lines

def emitP9P11P12Golden : IO Unit := do
  for line in p9p11p12GoldenLines do
    IO.println line

def p9p11p12GoldenMain : IO UInt32 := do
  emitP9P11P12Golden
  pure 0

end SuperNeo
