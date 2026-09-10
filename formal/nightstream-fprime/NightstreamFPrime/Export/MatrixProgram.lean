import NightstreamFPrime.Layout.MatrixProgram
import NightstreamFPrime.Export.Codec

/-! Canonical codecs for the shared Layout MatrixProgram types. -/

namespace NightstreamFPrime.Layout.MatrixProgram

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

def WireEntry.format : Format WireEntry where
  encode := fun entry => .array [.atom entry.column, .atom entry.coefficient]
  decode
    | .array [.atom column, .atom coefficient] =>
        .ok ⟨column, coefficient⟩
    | _ => .error "invalid matrix sparse entry"
  decode_encode := by
    intro entry
    cases entry
    rfl

def WireForm.format : Format WireForm where
  encode := fun form => (list WireEntry.format).encode form.entries
  decode := fun value => do
    pure ⟨← (list WireEntry.format).decode value⟩
  decode_encode := by
    intro form
    cases form
    simp [Format.decode_encode]

def retainedKindFormat : Format LowNormSlot.Kind where
  encode
    | .bit => .atom 0
    | .centered => .atom 1
    | .field => .atom 2
  decode
    | .atom 0 => .ok .bit
    | .atom 1 => .ok .centered
    | .atom 2 => .ok .field
    | _ => .error "invalid retained slot kind"
  decode_encode := by
    intro kind
    cases kind <;> rfl

def RetainedBlock.format : Format RetainedBlock where
  encode := fun block => .array [
    retainedKindFormat.encode block.kind,
    .atom block.slotCount,
    .atom block.start]
  decode
    | .array [kind, .atom slotCount, .atom start] => do
      pure ⟨← retainedKindFormat.decode kind, slotCount, start⟩
    | _ => .error "invalid retained matrix block"
  decode_encode := by
    intro block
    cases block
    simp [retainedKindFormat.decode_encode]

def SourceRange.format : Format SourceRange where
  encode := fun range => .array [
    .atom range.sourceStart,
    .atom range.sourceCount,
    RetainedBlock.format.encode range.retained,
    .atom range.slotStart]
  decode
    | .array [.atom sourceStart, .atom sourceCount, retained,
        .atom slotStart] => do
      pure ⟨sourceStart, sourceCount,
        ← RetainedBlock.format.decode retained, slotStart⟩
    | _ => .error "invalid matrix source range"
  decode_encode := by
    intro range
    cases range
    simp [RetainedBlock.format.decode_encode]

def SourceGridMode.format : Format SourceGridMode where
  encode
    | .direct => .atom 0
    | .external8 => .atom 1
  decode
    | .atom 0 => .ok .direct
    | .atom 1 => .ok .external8
    | _ => .error "invalid matrix source grid mode"
  decode_encode := by
    intro mode
    cases mode <;> rfl

def SourceGrid.format : Format SourceGrid where
  encode := fun grid => .array [
    .atom grid.sourceStart,
    .atom grid.majorCount,
    .atom grid.majorSourceStride,
    .atom grid.minorCount,
    .atom grid.minorSourceStride,
    .atom grid.runCount,
    RetainedBlock.format.encode grid.retained,
    SourceGridMode.format.encode grid.mode,
    .atom grid.slotStart,
    .atom grid.majorSlotStride,
    .atom grid.minorSlotStride]
  decode
    | .array [.atom sourceStart, .atom majorCount,
        .atom majorSourceStride, .atom minorCount,
        .atom minorSourceStride, .atom runCount, retained,
        mode, .atom slotStart, .atom majorSlotStride,
        .atom minorSlotStride] => do
      pure {
        sourceStart
        majorCount
        majorSourceStride
        minorCount
        minorSourceStride
        runCount
        retained := ← RetainedBlock.format.decode retained
        mode := ← SourceGridMode.format.decode mode
        slotStart
        majorSlotStride
        minorSlotStride }
    | _ => .error "invalid matrix source grid"
  decode_encode := by
    intro grid
    cases grid
    simp [RetainedBlock.format.decode_encode,
      SourceGridMode.format.decode_encode]
    rfl

def SourceSubstitution.format : Format SourceSubstitution where
  encode := fun substitution => .array [
    (list SourceRange.format).encode substitution.ranges,
    (list SourceGrid.format).encode substitution.grids]
  decode
    | .array [ranges, grids] => do
      pure ⟨← (list SourceRange.format).decode ranges,
        ← (list SourceGrid.format).decode grids⟩
    | _ => .error "invalid matrix source substitution"
  decode_encode := by
    rintro ⟨ranges, grids⟩
    simp only
    rw [(list SourceRange.format).decode_encode,
      (list SourceGrid.format).decode_encode]
    rfl

def IndexRange.format : Format IndexRange where
  encode := fun range => .array [.atom range.start, .atom range.count]
  decode
    | .array [.atom start, .atom count] => .ok ⟨start, count⟩
    | _ => .error "invalid matrix index range"
  decode_encode := by
    intro range
    cases range
    rfl

def IndexSchedule.format : Format IndexSchedule where
  encode
    | .rangeList ranges => .array [
        .atom 0, (list IndexRange.format).encode ranges]
    | .indexTable indices => .array [
        .atom 1, (list nat).encode indices.toList]
  decode
    | .array [.atom 0, ranges] => do
        pure (.rangeList (← (list IndexRange.format).decode ranges))
    | .array [.atom 1, indices] => do
        pure (.indexTable (← (list nat).decode indices).toArray)
    | _ => .error "invalid matrix index schedule"
  decode_encode := by
    intro schedule
    cases schedule <;> simp [Format.decode_encode]

end NightstreamFPrime.Layout.MatrixProgram
