import NightstreamFPrime.Export.Stage1.Wide.PhysicalRelabel
import NightstreamFPrime.Export.Stage1.Wide.MatrixProgram
import NightstreamFPrime.Layout.MatrixProgram.SourceComposition

/-! Rebind ordinary matrix blocks to the wide physical row archive. Embedded
range templates and retained-coordinate maps keep their existing operands. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PhysicalMatrixSource

open NightstreamFPrime.Layout NightstreamFPrime.Layout.MatrixProgram

def inverseRanges (extraColumns : Nat) : List SourceProjectionRange :=
  (PhysicalRelabel.columnRanges.take 3).map fun range =>
    {
      packageStart := range.sourceStart
      sourceStart := range.packageStart
      count := range.count + if range.packageStart =
        Layout.Stage1.Spartan.sourceToSpartan Layout.Stage1.PiRLCStarts.commitmentFreshStart
        then extraColumns else 0 }

def inverse (extraColumns : Nat) : SourceProjection := .mapped (inverseRanges extraColumns)

theorem inverseRanges_eq (extraColumns : Nat) :
    inverseRanges extraColumns =
      [⟨0, 0, 19512839⟩, ⟨19568242, 19776407, 52326⟩,
        ⟨19646884, 20572364, 8212655 + extraColumns⟩] := by
  have old := Layout.Stage1.PiRLCStarts.childLogicalStarts_eq
  have current := Layout.Stage1.Wide.PiRLCStarts.childLogicalStarts_eq
  simp only [List.cons.injEq] at old current
  rcases old with ⟨_, oldCommit, _, _, _, oldOutput, _⟩
  rcases current with ⟨_, newCommit, _, _, _, _, _⟩
  unfold inverseRanges
  rw [PhysicalRelabel.columnRanges, oldCommit, oldOutput, newCommit, Layout.Stage1.PiRLCStarts.phaseLogicalStart_eq,
    Layout.Stage1.PiRLCStarts.commitmentFreshStart_eq,
    Layout.Stage1.Wide.PiRLCStarts.commitmentFreshStart_eq,
    Layout.Stage1.Spartan.spartanColumnCount_eq]
  norm_num [Layout.Stage1.Wide.SourceOrder.column, Layout.Stage1.Wide.SourceOrder.relocate,
    Layout.Stage1.Spartan.sourceToSpartan, Layout.Stage1.Spartan.pilotSourceColumnCount,
    Layout.Stage1.Spartan.proofInputSourceStart, Layout.Stage1.Spartan.piCcsPhaseOffset,
    Layout.Stage1.Spartan.piCcsLocalStart, Layout.Stage1.Spartan.privateColumnCount_eq,
    Layout.Stage1.Wide.SourceOrder.privateColumns_eq]

theorem inverse_unique (extraColumns : Nat) : (inverse extraColumns).Unique := by
  intro source
  simp only [inverseRanges_eq,
    SourceProjectionRange.column?, List.filterMap_cons, List.filterMap_nil]
  split_ifs <;> simp_all <;> omega

theorem inverse_compose_column (extraColumns : Nat) (target : SourceProjection) (source : Nat) :
    ((inverse extraColumns).compose target).column? source =
      ((inverse extraColumns).column? source).bind target.column? :=
  SourceProjection.compose_column _ _ (inverse_unique extraColumns) source

def schedule : IndexSchedule → Except String IndexSchedule
  | .indexTable indices => return .indexTable (← indices.mapM PhysicalRelabel.row)
  | .rangeList ranges => do
    let moved ← ranges.mapM fun range => do
      let first ← PhysicalRelabel.row range.start
      unless 0 < range.count do throw "empty source row range"
      let last ← PhysicalRelabel.row (range.start + range.count - 1)
      unless first + range.count = last + 1 do throw "source row range crosses the removed sampler"
      return { range with start := first }
    return .rangeList moved

def block (extraColumns : Nat) : Block → Except String Block
  | .ordinary value => do
    return .ordinary { value with
      rows := ← schedule value.rows
      projection := (inverse extraColumns).compose value.projection }
  | .mapped width projection inner => return .mapped width projection (← block extraColumns inner)
  | other => .ok other

def program (application : RetainedLayout.Program) (compiled : PiRlcWideSampler.RangePlan.Compiled) :
    Except String Layout.MatrixProgram.Program := do
  let extra := PerApplicationPackage.directAddedPrivateColumnCount application
  return ⟨← (MatrixProgram.program application compiled).blocks.mapM (block extra)⟩

end NightstreamFPrime.Export.Stage1.Wide.PhysicalMatrixSource
