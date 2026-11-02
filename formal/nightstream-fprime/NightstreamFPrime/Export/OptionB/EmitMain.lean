import NightstreamFPrime.Export.OptionB.StepRelation
import NightstreamFPrime.Export.Generic.Pieces
import NightstreamFPrime.Export.Stage1.Rows
import NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1

/-! The option-B emitter (CW-5.5), first part: the row census of the light and the refresh step
circuits for today's fixture application (`Poseidon2HashChainV1.program`). For each circuit it
prints the constraints, the unpacked R1CS rows of `LogicalPlan.rows`, the largest column, the
Poseidon2 blocks that `Pieces.split` recognizes, and the rows after packing each block into 150
CCS rows. -/

namespace NightstreamFPrime.Export.OptionB.EmitMain

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout.OptionB
open NightstreamFPrime.Export.Generic

/-- The census width; the step circuits' rows do not depend on it. -/
def width : Nat := 2 ^ 24

theorem publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth width := by
  decide

theorem cubeFits : Phi81CarrierLayout.carrierWidth width ≤ 2 ^ cubeVariables := by
  decide

/-- The relation argument of the step circuits; their rows do not read its matrices. -/
def relation : ProductionKey.LogicalRelation width publicFits :=
  ⟨fun _ _ _ => 0, cubeFits⟩

def program : Stage1.Application.Program := Stage1.Poseidon2HashChainV1.program

def census (name : String) (ops : List Op) (offset : Nat) (digestColumn : Fin 4 → Nat) :
    IO Unit := do
  let start ← IO.monoMsNow
  let constraints := flatConstraints ops ++ LogicalPlan.decodeConstraints digestColumn
  let firstFresh := offset + localLength ops
  let lowered := Stage1.Rows.lowerConstraintsTR constraints firstFresh
  IO.println s!"{name}: {constraints.length} constraints, {localLength ops} locals from {offset}"
  IO.println s!"{name}: {lowered.rows.length} unpacked rows, next column {lowered.next}, \
    largest column {LogicalPlan.rowsMax lowered.rows} ({(← IO.monoMsNow) - start} ms)"
  let pieces := Pieces.split (PilotData.canonicalConstraints ()) constraints
  let permutations := (pieces.filter fun | .permutation .. => true | _ => false).length
  let generic := Pieces.genericConstraints pieces
  let genericRows := (Stage1.Rows.lowerConstraintsTR generic firstFresh).rows.length
  IO.println s!"{name}: {permutations} Poseidon2 blocks, {generic.length} other constraints, \
    {genericRows} other rows, {genericRows + 150 * permutations} packed rows \
    ({(← IO.monoMsNow) - start} ms)"

def main : IO Unit := do
  let base := StepRelation.base
  let words := program.witnessWordCount
  let lightOffset := LightAssembler.sharedEnd base words
  census "light" (Circuit.ops (OptionB.LightStepLayout.main program relation 0
      (LightAssembler.interface (logicalWidth := width) (publicFits := publicFits) base words))
      lightOffset) lightOffset (StepRelation.lightDigestColumn program)
  let refreshOffset := RefreshAssembler.sharedEnd base words
  census "refresh" (Circuit.ops (OptionB.RefreshStepLayout.main program relation 1
      (RefreshAssembler.interface (logicalWidth := width) (publicFits := publicFits) base words))
      refreshOffset) refreshOffset (StepRelation.refreshDigestColumn program)

end NightstreamFPrime.Export.OptionB.EmitMain

def main : IO Unit := NightstreamFPrime.Export.OptionB.EmitMain.main
