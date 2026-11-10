import CompressionLeanExe.Parse
import NightstreamFPrime.Export.OptionB.StepPackage
import NightstreamFPrime.Lifecycle.Nebula.FactorWords
import NightstreamFPrime.Spec
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.GoldilocksCausal
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetConvention
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork
import NightstreamFPrime.Spec.Nebula

/-! An option-B step package from its artifact. The package grammar is natural numbers and arrays
(`StepPackage.Package.format`), nested deeper than a general JSON reader allows. -/

namespace CompressionLeanExe.PackageJson

open CompressionLeanExe.Parse
open NightstreamFPrime.Export.Codec (Value)
open NightstreamFPrime.Export.OptionB.StepPackage (Package)

def digitsEnd (data : ByteArray) (pos : ℕ) : ℕ :=
  if h : pos < data.size then if 48 ≤ data[pos] ∧ data[pos] ≤ 57 then digitsEnd data (pos + 1) else pos
  else pos
termination_by data.size - pos

mutual

/-- One JSON value of the package grammar: a natural number or an array of values. -/
partial def value (data : ByteArray) (start : ℕ) : Except String (Value × ℕ) := do
  let pos := skipSpaces data start
  if pos < data.size ∧ data[pos]! == 91 then
    items data (skipSpaces data (pos + 1)) #[]
  else
    let stop := digitsEnd data pos
    if stop == pos then throw s!"expected a number or an array at byte {pos}"
    pure (.atom ((data.extract pos stop).foldl (fun acc b => 10 * acc + (b - 48).toNat) 0), stop)

partial def items (data : ByteArray) (pos : ℕ) (out : Array Value) : Except String (Value × ℕ) := do
  if pos < data.size ∧ data[pos]! == 93 then return (.array out.toList, pos + 1)
  let (item, next) ← value data pos
  let next := skipSpaces data next
  if next < data.size ∧ data[next]! == 44 then items data (skipSpaces data (next + 1)) (out.push item)
  else if next < data.size ∧ data[next]! == 93 then return (.array (out.push item).toList, next + 1)
  else throw s!"expected , or ] at byte {next}"

end

def readPackage (path : System.FilePath) : IO (Except String Package) := do
  let data ← IO.FS.readBinFile path
  pure do
    let (json, _) ← value data 0
    Package.format.decode json

end CompressionLeanExe.PackageJson
