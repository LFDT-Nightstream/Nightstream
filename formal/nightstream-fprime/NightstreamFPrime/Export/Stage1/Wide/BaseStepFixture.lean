import NightstreamFPrime.Export.Stage1.BaseStepFixture
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript
import NightstreamFPrime.Layout.Stage1.Wide.SourceOrder
import NightstreamFPrime.Layout.Stage1.Wide.PiRLCStarts

/-! Base caller fixture for the exact wide transcript. The supplied context
is fixture input; production binding must check it against the selected key. -/

namespace NightstreamFPrime.Export.Stage1.Wide.BaseStepFixture

open NightstreamFPrime.Spec NightstreamFPrime.Lifecycle
open Spec.Folding.PiCCS.PaperJoint

private def table {α : Type} {rows columns : Nat} (source : Fin rows → Fin columns → α) : Array (Array α) :=
  Array.ofFn fun row => Array.ofFn (source row)

private def read {α : Type} {rows columns : Nat} (source : Fin rows → Fin columns → α)
    (values : Array (Array α)) (stored : values = table source) (row : Fin rows) (column : Fin columns) : α :=
  let valuesRow := values[row.val]'(by simpa only [stored, table, Array.size_ofFn] using row.isLt)
  valuesRow[column.val]'(by simpa only [valuesRow, stored, table, Array.getElem_ofFn, Array.size_ofFn] using column.isLt)

private theorem read_eq {α : Type} {rows columns : Nat} (source : Fin rows → Fin columns → α)
    (values : Array (Array α)) (stored : values = table source) (row : Fin rows) (column : Fin columns) :
    read source values stored row column = source row column := by
  subst values
  simp only [read, table, Array.getElem_ofFn]

def batch (state : Transcript.State) : Transcript.PiRlcSampler.Batch productionShape.sourceCount :=
  let source := fun source : Fin productionShape.sourceCount =>
    Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.challengeAt state source.val
  let values := table source
  {
    challenges := read source values rfl
    finalState := Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.stateAt state productionShape.sourceCount }

theorem batch_challenges (state : Transcript.State) (source : Fin productionShape.sourceCount)
    (lane : Fin ringDegree) :
    (batch state).challenges source lane =
      Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.challengeAt state source.val lane := by
  simp only [batch, read_eq]

def valueIO (context : VerifierContext.Digest4) : IO Codec.Value := do
  let fixture ← Stage1.BaseStepFixture.valueIOWith (fun state => some (batch state)) context
  let helpers := (List.range 17).map fun source => Codec.Value.array [
    .atom (Layout.Stage1.Wide.SourceOrder.column (Layout.Stage1.Wide.PiRLCStarts.rangeLogicalStart source)),
    .atom Gadgets.Sampling.WideReduction.HintProgram.helperCount]
  let bridge := Codec.Value.array [
    .atom (Layout.Stage1.Wide.SourceOrder.column (Layout.Stage1.Wide.PiRLCStarts.challengeWordStart 0)),
    .atom PiRLC.Wide.DigitWords.count]
  let checkedBit := Layout.Stage1.Wide.SourceOrder.column
    (Gadgets.Sampling.WideReduction.quotientStart (Gadgets.Sampling.WideReduction.Program.coreOffset
      (Layout.Stage1.Wide.PiRLCStarts.rangeLogicalStart 0)))
  return .array [.atom 1, fixture, .array helpers, bridge, .atom checkedBit]

end NightstreamFPrime.Export.Stage1.Wide.BaseStepFixture
