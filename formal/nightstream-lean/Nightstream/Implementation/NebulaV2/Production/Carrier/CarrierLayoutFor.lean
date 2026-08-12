import Nightstream.Implementation.NebulaV2.FPrime.Claim.FieldNativeCarrierAlias
import Nightstream.Implementation.NebulaV2.Production.Memory.BatchCarrierBridge

/-!
Contract: exact physical full-claim carrier at the generated
augmented-relation exponent.

The running window has `83160 + 2 * rowVariables` Goldilocks coordinates.
The same `rowVariables` indexes the typed running value and its zero-copy NIFS
alias. All windows are consecutive and non-overlapping.

`Placed` states only physical values. It does not assume NIFS acceptance,
memory execution, F-prime continuity, terminal acceptance, or soundness.

Assurance tier: exponent-indexed physical carrier schema.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductionFullClaimCarrierLayoutFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.MemoryClaimCounterRows
open Nightstream.Implementation.NebulaV2.ProductionMemoryBatchCarrierBridge
open Nightstream.Implementation.NebulaV2.ProductionMemoryCheckedBatchRows
open Nightstream.Implementation.NebulaV2.ProductionMemorySuffixCarrier
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

def ccsPublicOffset : Nat := 0
def bundleOffset : Nat := 540
def runningOffset : Nat := 4428

def memoryCounterOffset (rowVariables : Nat) : Nat :=
  runningOffset + runningFieldCoordinatesFor rowVariables

def memoryNativeOffset (candidate : Id) (rowVariables : Nat) : Nat :=
  memoryCounterOffset rowVariables + checkedStepsPerFreshClaim candidate * 116

def endOffset (candidate : Id) (rowVariables : Nat) : Nat :=
  memoryNativeOffset candidate rowVariables +
    checkedStepsPerFreshClaim candidate * 76

theorem section_offsets_exact (rowVariables : Nat) :
    bundleOffset = ccsPublicOffset + 540 /\
      runningOffset = bundleOffset + 3888 /\
      memoryCounterOffset rowVariables =
        runningOffset + runningFieldCoordinatesFor rowVariables := by
  simp [ccsPublicOffset, bundleOffset,
    runningOffset, memoryCounterOffset]

theorem endOffset_exact (candidate : Id) (rowVariables : Nat) :
    endOffset candidate rowVariables =
      fieldNativeEnvelopeCoordinatesFor candidate rowVariables := by
  unfold endOffset memoryNativeOffset memoryCounterOffset
    fieldNativeEnvelopeCoordinatesFor narrowCoordinates fixedNarrowCoordinates
    memoryFieldCoordinates bundleFieldCoordinates runningOffset
  rw [memorySuffixCoordinate_split_exact.1,
    memorySuffixCoordinate_split_exact.2.1]
  omega

/-- One consecutive carrier window. Column zero remains the constant-one
wire. -/
structure Layout (candidate : Id) (rowVariables : Nat) where
  start : Nat
  startPositive : 0 < start
deriving Repr

def Layout.ccsPublicColumn {candidate rowVariables}
    (layout : Layout candidate rowVariables) (index : Fin 540) : Nat :=
  layout.start + ccsPublicOffset + index.val

def Layout.bundleColumn {candidate rowVariables}
    (layout : Layout candidate rowVariables)
    (index : Fin bundleFieldCoordinates) : Nat :=
  layout.start + bundleOffset + index.val

def Layout.runningColumn {candidate rowVariables}
    (layout : Layout candidate rowVariables)
    (index : Fin (runningFieldCoordinatesFor rowVariables)) : Nat :=
  layout.start + runningOffset + index.val

def Layout.memoryCounterStepStart {candidate rowVariables}
    (layout : Layout candidate rowVariables)
    (step : Fin (checkedStepsPerFreshClaim candidate)) : Nat :=
  layout.start + memoryCounterOffset rowVariables + step.val * 116

def Layout.memoryCounterColumn {candidate rowVariables}
    (layout : Layout candidate rowVariables)
    (step : Fin (checkedStepsPerFreshClaim candidate))
    (counter : Counter) (digit : Nat) : Nat :=
  layout.memoryCounterStepStart step + counter.bitOffset + digit

def Layout.memoryNativeColumn {candidate rowVariables}
    (layout : Layout candidate rowVariables)
    (step : Fin (checkedStepsPerFreshClaim candidate))
    (slot : MemoryClaimFieldRows.Slot) : Nat :=
  layout.start + memoryNativeOffset candidate rowVariables +
    step.val * 76 + slot.position

def Layout.memoryCounterDigits {candidate rowVariables}
    (layout : Layout candidate rowVariables) (assignment : Nat -> Nat)
    (step : Fin (checkedStepsPerFreshClaim candidate))
    (counter : Counter) : List Nat :=
  (List.range counter.width).map fun digit =>
    assignment (layout.memoryCounterColumn step counter digit)

theorem counter_intervals_exact :
    (Counter.all.flatMap fun counter =>
      (List.range counter.width).map fun digit =>
        counter.bitOffset + digit) = List.range 116 := by
  decide

/-- Exact semantic placement of every carrier section. -/
structure Placed
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    (layout : Layout candidate rowVariables) (assignment : Nat -> Nat)
    (value : ProductionFieldNativeFullClaim.Value candidate fullShape) : Prop where
  ccsPublic : forall index : Fin 540,
    assignment (layout.ccsPublicColumn index) =
      value.ccsPublic.val.get
        ⟨index.val, by rw [value.ccsPublic.property.1]; exact index.isLt⟩
  bundle : forall index : Fin bundleFieldCoordinates,
    assignment (layout.bundleColumn index) =
      (ProductionFieldNativeFullClaim.bundleFields
        value.commitmentBundle).get
        ⟨index.val, by
          rw [ProductionFieldNativeFullClaim.bundleFields_length]
          exact index.isLt⟩
  running : forall index : Fin (runningFieldCoordinatesFor rowVariables),
    assignment (layout.runningColumn index) =
      (ProductionFieldNativeFullClaim.runningNativeValues
        value.recursiveState).get
        ⟨index.val, by
          have lengthExact :=
            ProductionFieldNativeFullClaim.runningFields_lengthFor
              contract.toShape value.recursiveState
          have nativeLength :
              (ProductionFieldNativeFullClaim.runningNativeValues
                value.recursiveState).length =
                runningFieldCoordinatesFor rowVariables := by
            rw [ProductionFieldNativeFullClaim.runningNativeValues,
              List.length_map, lengthExact]
            simp [runningFieldCoordinatesFor,
              ProductNifsCodec.runningFieldCountFor,
              contract.rowVariablesExact]
          rw [nativeLength]
          exact index.isLt⟩
  memoryCounter : forall step counter,
    layout.memoryCounterDigits assignment step counter =
      WasmStateCodec.encodeWord counter.width
        (counter.claimValue (claimAt value.memory step))
  memoryNative : forall step slot,
    assignment (layout.memoryNativeColumn step slot) =
      (claimAt value.memory step).fieldValue slot.tag

structure CheckedMemoryAliases
    {candidate : Id} {rowVariables : Nat}
    (carrier : Layout candidate rowVariables)
    (checked : ProductionMemoryCheckedBatchRows.Layout candidate) : Prop where
  counterStart : forall step,
    (checked.steps step).claim.counterBitStart =
      carrier.memoryCounterStepStart step
  nativeField : forall step slot,
    (checked.steps step).claim.nativeFieldColumn slot =
      carrier.memoryNativeColumn step slot

theorem checkedMemoryPlacement
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    {contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape}
    {carrier : Layout candidate rowVariables}
    {checked : ProductionMemoryCheckedBatchRows.Layout candidate}
    {assignment : Nat -> Nat}
    {value : ProductionFieldNativeFullClaim.Value candidate fullShape}
    (placed : Placed contract carrier assignment value)
    (aliases : CheckedMemoryAliases carrier checked) :
    ProductionMemoryBatchCarrierBridge.Placement
      checked assignment value.memory := by
  constructor
  · intro step counter
    simpa [ProductionMemoryClaimRows.Layout.counters,
      MemoryClaimCounterRows.Layout.word,
      BoundedWordRows.Layout.digits,
      BoundedWordRows.Layout.bitColumn,
      Layout.memoryCounterDigits, Layout.memoryCounterColumn,
      aliases.counterStart step] using placed.memoryCounter step counter
  · intro step slot
    rw [aliases.nativeField step slot]
    exact placed.memoryNative step slot

structure NifsAliases
    {candidate : Id} {rowVariables : Nat}
    (carrier : Layout candidate rowVariables) where
  nifsRunningColumn : Fin (runningFieldCoordinatesFor rowVariables) -> Nat
  nifsBundleColumn : Fin bundleFieldCoordinates -> Nat
  running : forall coordinate,
    nifsRunningColumn coordinate = carrier.runningColumn coordinate
  bundle : forall coordinate,
    nifsBundleColumn coordinate = carrier.bundleColumn coordinate

def NifsAliases.toAliasContract
    {candidate : Id} {rowVariables : Nat}
    {carrier : Layout candidate rowVariables}
    (aliases : NifsAliases carrier) : AliasContractFor rowVariables where
  runningCarrierColumn := carrier.runningColumn
  nifsRunningColumn := aliases.nifsRunningColumn
  runningColumnsEqual := aliases.running
  bundleCarrierColumn := carrier.bundleColumn
  nifsBundleColumn := aliases.nifsBundleColumn
  bundleColumnsEqual := aliases.bundle

theorem nifsRunningValues_eq_carrier
    {candidate : Id} {rowVariables : Nat}
    {carrier : Layout candidate rowVariables}
    (aliases : NifsAliases carrier) (assignment : Nat -> Nat) :
    FieldNativeCarrierAlias.nifsRunningValuesFor
        aliases.toAliasContract assignment =
      FieldNativeCarrierAlias.runningCarrierValuesFor
        aliases.toAliasContract assignment :=
  FieldNativeCarrierAlias.runningValuesFor_eq aliases.toAliasContract assignment

theorem nifsBundleValues_eq_carrier
    {candidate : Id} {rowVariables : Nat}
    {carrier : Layout candidate rowVariables}
    (aliases : NifsAliases carrier) (assignment : Nat -> Nat) :
    FieldNativeCarrierAlias.nifsBundleValuesFor
        aliases.toAliasContract assignment =
      FieldNativeCarrierAlias.bundleCarrierValuesFor
        aliases.toAliasContract assignment :=
  FieldNativeCarrierAlias.bundleValuesFor_eq aliases.toAliasContract assignment

theorem memoryNativeColumn_lt_end
    {candidate : Id} {rowVariables : Nat}
    (layout : Layout candidate rowVariables)
    (step : Fin (checkedStepsPerFreshClaim candidate))
    (slot : MemoryClaimFieldRows.Slot) :
    layout.memoryNativeColumn step slot <
      layout.start + endOffset candidate rowVariables := by
  have slotBound : slot.position < 76 := by
    rw [MemoryClaimFieldRows.Slot.position]
    simpa [MemoryClaimFieldRows.Slot.all_length_exact] using
      List.idxOf_lt_length_of_mem slot.mem_all
  simp only [Layout.memoryNativeColumn, endOffset, memoryNativeOffset]
  have stepBound := step.isLt
  omega

end Nightstream.Implementation.NebulaV2.ProductionFullClaimCarrierLayoutFor
