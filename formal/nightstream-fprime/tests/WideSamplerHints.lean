import NightstreamFPrime.Gadgets.Sampling.WideReduction.Program
import NightstreamFPrime.Gadgets.Sampling.WideReduction.HintExecution
import NightstreamFPrime.Gadgets.Sampling.WideReduction.Completeness
import NightstreamFPrime.Layout.PiRlcWideSampler.Rows

namespace NightstreamFPrime.Tests.WideSamplerHints

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler

private def interface : WideReduction.Interface where
  source := fun lane _ => .var lane.val

/-- Independent entry count for one standalone block: four field-encoded
inputs, the proved 937-coordinate retained layout, and one constant column. -/
private def combinationEntries (combination : Layout.R1CS.LinearCombination) : IO Nat := do
  let coreOffset := WideReduction.Program.coreOffset 4
  let mut coefficients : Array F := Array.replicate (1 + 4 * 41 + 937) 0
  coefficients := coefficients.set! 0 combination.constant
  for (source, coefficient) in combination.terms do
    let (start, width) ← if source < 4 then
      pure (1 + source * 41, 41)
    else if coreOffset ≤ source && source < coreOffset + WideReduction.privateCount then do
      let relative := source - coreOffset
      if relative < 264 then
        let lane := relative / 66
        let index := relative % 66
        if index < 64 then pure (1 + 4 * 41 + lane * 146 + index, 1)
        else pure (1 + 4 * 41 + lane * 146 + 64 + (index - 64) * 41, 41)
      else pure (1 + 4 * 41 + 4 * 146 + relative - 264, 1)
    else
      throw (IO.userError s!"matrix reads an unretained source column: {source}")
    let mut weight : F := 1
    for index in List.range width do
      unless start + index < coefficients.size do
        throw (IO.userError "retained coordinate is out of range")
      coefficients := coefficients.set! (start + index)
        (coefficients[start + index]! + coefficient * weight)
      weight := weight * 3
  pure ((coefficients.filter (fun coefficient => coefficient != 0)).size)

private def checkMatrixCost : IO Unit := do
  let ops := WideReduction.Program.operations interface 4
  let some rows := Layout.PiRlcWideSampler.Rows.compile? (WideReduction.Program.coreOffset 4)
      (flatConstraints ops)
    | throw (IO.userError "wide sampler contains an unsupported direct row")
  let mut nonzeros := 0
  for row in rows do
    let a ← combinationEntries row.a
    let b ← combinationEntries row.b
    let c ← combinationEntries row.c
    nonzeros := nonzeros + 1 + a + b + c
  unless rows.length = 681 do
    throw (IO.userError "direct compiler changed the logical row count")
  IO.println s!"standalone sampler matrix: rows={rows.length}, retained_coordinates={Layout.PiRlcWideSampler.coordinateCount}, nonzeros={nonzeros}; includes four field-encoded inputs"

private def check (integer : Nat) : IO Unit := do
  let draw : Draw := drawIndex.symm ⟨integer % drawCount, Nat.mod_lt _ (by decide)⟩
  let env : Env := fun index => if below : index < 4 then draw ⟨index, below⟩ else 0
  let helperOffset := 4
  let gadgetOffset := helperOffset + WideReduction.HintProgram.helperCount
  let hints := WideReduction.HintProgram.resultHints helperOffset gadgetOffset
  let ops := WideReduction.Program.operations interface helperOffset
  let mut values : Array F := #[]
  for batch in witnesses ops do
    unless batch.start = helperOffset + values.size do
      throw (IO.userError "non-contiguous witness batch")
    for recipe in batch.recipes do
      values := values.push (recipe.eval (WideReduction.HintExecution.read env helperOffset values))
    for hint in batch.hints do
      values := values.push (hint.eval (WideReduction.HintExecution.read env helperOffset values))
  let completed := WideReduction.HintExecution.read env helperOffset values
  let rows := flatConstraints ops
  unless hints.length = 353 && rows.length = 681 do
    throw (IO.userError "wrong hint or row count")
  for (row, index) in rows.zipIdx do
    unless row.eval completed = 0 do
      throw (IO.userError s!"wide sampler row {index} fails for X={integer}")
  for digit in List.range 54 do
    let actual := (WideReduction.digitBit gadgetOffset digit 0).eval completed +
      2 * (WideReduction.digitBit gadgetOffset digit 1).eval completed +
      4 * (WideReduction.digitBit gadgetOffset digit 2).eval completed
    let expected := ((drawIndex draw).val % scalarCount / 5 ^ digit) % 5
    unless actual.val = expected do
      throw (IO.userError s!"wide sampler digit {digit} differs for X={integer}")

def run : IO Unit := do
  checkMatrixCost
  let cases := [0, 1, scalarCount - 1, scalarCount, scalarCount + 1,
    goldilocksModulus - 1, goldilocksModulus, goldilocksModulus + 1,
    drawCount - scalarCount, drawCount - 2, drawCount - 1]
  for integer in cases do check integer
  IO.println s!"wide sampler hint controls passed: cases={cases.length}, helpers={WideReduction.HintProgram.helperCount}, checked_rows_per_case=681"

end NightstreamFPrime.Tests.WideSamplerHints

def main : IO Unit := NightstreamFPrime.Tests.WideSamplerHints.run
