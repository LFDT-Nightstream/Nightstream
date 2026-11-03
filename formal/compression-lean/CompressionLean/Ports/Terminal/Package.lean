import CompressionLean.Ports.Circuit.Ir
import CompressionLean.Fields
import NightstreamFPrime.Export.PackageExpansion

namespace CompressionLean.Terminal.Package

open CompressionLean.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open CompressionLean.Shrink (R1csRow)
open CompressionLean.Replay (Word)

def collect : List (ℕ × NightstreamFPrime.Spec.F) → List (ℕ × NightstreamFPrime.Spec.F)
  | (c, x) :: (d, y) :: rest =>
      if c = d then collect ((c, x + y) :: rest) else (c, x) :: collect ((d, y) :: rest)
  | terms => terms
termination_by terms => terms.length

def side (constantColumn : ℕ) (combination : R1CS.LinearCombination) : List (ℕ × Word) :=
  let terms := ((constantColumn, combination.constant) :: combination.terms).filter (·.2 ≠ 0)
  (collect (terms.mergeSort fun x y => x.1 ≤ y.1)).filterMap fun (c, x) =>
    if x = 0 then none else some (c, Word.ofF x)

def row (constantColumn : ℕ) (r : R1CS.Row) : R1csRow Word :=
  ⟨side constantColumn r.a, side constantColumn r.b, side constantColumn r.c⟩

end CompressionLean.Terminal.Package
