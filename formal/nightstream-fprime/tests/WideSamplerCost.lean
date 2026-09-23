import Lean
import Std.Data.HashMap
import NightstreamFPrime.Layout.PiRlcWideSampler.BatchPlan

/-! Emit the candidate's exact sparse input forms and normalized range-row
costs. The first input is a previous Poseidon endpoint, as in production.
The Python experiment normalizes each subsequent Poseidon linear layer. -/

namespace NightstreamFPrime.Tests.WideSamplerCost

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation PiRlcWideSampler

private def columns : Nat := 329 + 135813

private def inputBlock : LowNormBlock.Block 8 where
  kind := .field
  slotCount := 8
  source := id

private def interface : BatchPlan.Interface columns where
  oneColumn := ⟨0, by decide⟩
  initialState := SparseLayer.external (fun lane => inputBlock.form 1 (by decide) lane)
  start := 329
  fits := by rw [BatchPlan.coordinateCount_eq]; decide

private def wire {width : Nat} (form : SparseForm width) : Lean.Json :=
  Lean.toJson (form.entries.map fun entry => (entry.column.val, entry.coefficient.val))

private def nonzeros {width : Nat} (form : SparseForm width) : Nat :=
  let coefficients := form.entries.foldl (fun (result : Std.HashMap Nat F) entry =>
    result.insert entry.column.val (result.getD entry.column.val 0 + entry.coefficient)) {}
  coefficients.toArray.foldl (fun count entry => if entry.2 = 0 then count else count + 1) 0

def run : IO Unit := do
  let some compiled := RangePlan.compile?
    | throw (IO.userError "range compiler rejected the candidate")
  let mut rangeCounts := Array.replicate 14 0
  for source in List.finRange 17 do
    let input := BatchPlan.rangeInputs compiled interface source
    for row in compiled.rows.attach do
      let bounded := compiled.bounded row.val row.property
      let forms := SourceCompiler.compileRow (input.sourceMap ⟨0, by change 0 < compiled.rows.length; rw [compiled.rowCount]; decide⟩)
        input.oneColumn row.val bounded
      for port in List.finRange 13 do
        rangeCounts := rangeCounts.modify port.val (· + nonzeros (forms.meaningfulForm port))
  let inputs := List.ofFn fun invocation : Fin 34 =>
    Lean.Json.arr (Array.ofFn fun lane : Fin 8 => wire ((BatchPlan.poseidonInterface interface).input invocation lane))
  IO.println (Lean.Json.mkObj [
    ("rows", Lean.toJson (BatchPlan.plan compiled interface).rowCount),
    ("coordinates", Lean.toJson BatchPlan.coordinateCount),
    ("range_nonzeros", Lean.toJson rangeCounts),
    ("poseidon_inputs", Lean.toJson inputs),
    ("poseidon_retained_start", Lean.toJson interface.start),
    ("initial_constants", Lean.toJson Poseidon2.initialConstants),
    ("internal_constants", Lean.toJson Poseidon2.internalConstants),
    ("terminal_constants", Lean.toJson Poseidon2.terminalConstants),
    ("internal_diagonal", Lean.toJson Poseidon2.internalDiagonal)]).compress

end NightstreamFPrime.Tests.WideSamplerCost

def main : IO Unit := NightstreamFPrime.Tests.WideSamplerCost.run
