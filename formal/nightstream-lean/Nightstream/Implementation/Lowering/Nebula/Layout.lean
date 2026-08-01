import Nightstream.Implementation.Lowering.Nebula.StepPolynomial

/-!
Lean-owned physical layout arithmetic for the stackless Nebula step relation.

Assurance tier: model-level.

Owns: parameter validity, all public and witness column offsets, ring-column
alignment, and closed row/column counts for the selected stackless relation.

Does not own: a WASM program, memory-port selection, matrix coefficients,
witness values, transcript challenges, Rust geometry, or a security result.

This first physical profile is stackless because the 42-times-6 benchmark
uses linear memory only. Stack namespaces require extra selectors and pointer
rows and are not silently represented by this layout.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.Nebula.Layout

open Nightstream.SuperNeo.Concrete

def valueBits : Nat := 32
def timestampBits : Nat := 44
def segmentIndexBits : Nat := 16
def stepIndexBits : Nat := 16
def extensionLimbBits : Nat := 64
def extensionBits : Nat := 2 * extensionLimbBits
def cellBits : Nat := valueBits + timestampBits

theorem cellBits_exact : cellBits = 76 := by
  rfl

/-- Stackless public input width:
`segment, step, ts_in, ts_out, gamma[2], h_in[4], h_out[4]`. -/
def publicInputBits : Nat :=
  segmentIndexBits + stepIndexBits + 2 * timestampBits +
    2 * extensionBits + 8 * extensionBits

theorem publicInputBits_exact : publicInputBits = 1400 := by
  rfl

/-- Stackless physical parameters. `r` and `mu` are binary address widths. -/
structure Params where
  r : Nat
  mu : Nat
  operationSlots : Nat
  scanSlots : Nat
  segmentLimit : Nat
deriving DecidableEq, Repr

def Params.romCells (params : Params) : Nat := 2 ^ params.r
def Params.ramCells (params : Params) : Nat := 2 ^ params.mu
def Params.scannedCells (params : Params) : Nat :=
  params.romCells + params.ramCells

/-- Every arithmetic precondition consumed by the layout. -/
structure Params.Valid (params : Params) : Prop where
  operationSlotsPositive : 0 < params.operationSlots
  scanSlotsPositive : 0 < params.scanSlots
  segmentLimitPositive : 0 < params.segmentLimit
  romFitsRamAddress : params.r ≤ params.mu
  addressFitsPacking : timestampBits + params.mu + 1 ≤ 62
  exactScanCover : params.scannedCells % params.scanSlots = 0
  stepCounterFits : params.scannedCells / params.scanSlots ≤ 2 ^ stepIndexBits
  segmentCounterFits : params.segmentLimit ≤ 2 ^ segmentIndexBits
  timestampFits :
    params.segmentLimit * (params.scannedCells / params.scanSlots) *
      params.operationSlots < 2 ^ timestampBits

/-- Round a bit width up to a whole Phi81 ring column. -/
def alignToRing (width : Nat) : Nat :=
  ((width + ringDegree - 1) / ringDegree) * ringDegree

def Params.addressBits (params : Params) : Nat := params.mu

/-- `pad, is_write, ram, addr, v_r, v_w, read_timestamp`. -/
def Params.operationBits (params : Params) : Nat :=
  3 + params.addressBits + 2 * valueBits + timestampBits

def Params.operationLaneBits (params : Params) : Nat :=
  alignToRing (params.operationSlots * params.operationBits)

def Params.scanLaneBits (params : Params) : Nat :=
  alignToRing (params.scanSlots * cellBits)

/-- Bit width of the running active-operation counter. -/
def Params.countBits (params : Params) : Nat :=
  Nat.log2 params.operationSlots + 1

/-- Column zero is constant one. Public input bits start at column one. -/
def Params.publicEnd (_params : Params) : Nat := 1 + publicInputBits

def Params.operationLane (params : Params) : Nat :=
  alignToRing params.publicEnd

def Params.initialScanLane (params : Params) : Nat :=
  params.operationLane + params.operationLaneBits

def Params.finalScanLane (params : Params) : Nat :=
  params.initialScanLane + params.scanLaneBits

def Params.auxiliaryStart (params : Params) : Nat :=
  params.finalScanLane + params.scanLaneBits

/-- Per-operation deterministic auxiliary witness:
`diff`, `count`, `h_rs`, and `h_ws`. -/
def Params.operationAuxiliaryBits (params : Params) : Nat :=
  timestampBits + params.countBits + 2 * extensionBits

/-- Per-scan-slot deterministic auxiliary witness: `h_is` and `h_fs`. -/
def scanAuxiliaryBits : Nat := 2 * extensionBits

/-- Exact logical assignment width, including constant one and public input. -/
def Params.columnCount (params : Params) : Nat :=
  params.auxiliaryStart +
    params.operationSlots * params.operationAuxiliaryBits +
    params.scanSlots * scanAuxiliaryBits

def Params.witnessColumns (params : Params) : Nat :=
  params.columnCount - params.publicEnd

/-- First bit of operation slot `slot`. -/
def Params.operationSlot (params : Params) (slot : Nat) : Nat :=
  params.operationLane + slot * params.operationBits

/-- First auxiliary bit of operation slot `slot`. -/
def Params.operationAuxiliary (params : Params) (slot : Nat) : Nat :=
  params.auxiliaryStart + slot * params.operationAuxiliaryBits

def Params.operationDiff (params : Params) (slot : Nat) : Nat :=
  params.operationAuxiliary slot

def Params.operationCount (params : Params) (slot : Nat) : Nat :=
  params.operationDiff slot + timestampBits

def Params.operationReadProduct (params : Params) (slot : Nat) : Nat :=
  params.operationCount slot + params.countBits

def Params.operationWriteProduct (params : Params) (slot : Nat) : Nat :=
  params.operationReadProduct slot + extensionBits

/-- Scan auxiliaries follow every operation auxiliary. -/
def Params.scanAuxiliaryStart (params : Params) : Nat :=
  params.auxiliaryStart +
    params.operationSlots * params.operationAuxiliaryBits

def Params.initialScanProduct (params : Params) (slot : Nat) : Nat :=
  params.scanAuxiliaryStart + slot * scanAuxiliaryBits

def Params.finalScanProduct (params : Params) (slot : Nat) : Nat :=
  params.initialScanProduct slot + extensionBits

/-- Columns forced to zero by alignment. -/
def Params.fillerColumns (params : Params) : List Nat :=
  List.range' params.publicEnd
      (params.operationLane - params.publicEnd) ++
    List.range'
      (params.operationLane + params.operationSlots * params.operationBits)
      (params.initialScanLane -
        (params.operationLane + params.operationSlots * params.operationBits)) ++
    List.range'
      (params.initialScanLane + params.scanSlots * cellBits)
      (params.finalScanLane -
        (params.initialScanLane + params.scanSlots * cellBits)) ++
    List.range'
      (params.finalScanLane + params.scanSlots * cellBits)
      (params.auxiliaryStart -
        (params.finalScanLane + params.scanSlots * cellBits))

def Params.operationBitRows (params : Params) : Nat :=
  params.operationBits + timestampBits + params.countBits +
    2 * extensionBits

/-- E1--E9 rows for one stackless operation slot. -/
def Params.rowsPerOperation (params : Params) : Nat :=
  params.operationBitRows +
    4 +                         -- E2--E5
    (params.addressBits - params.r) + -- E6
    6 +                         -- E7
    4                           -- E8/E9, two K components each

/-- S1--S3 rows for one scan position across IS and FS. -/
def rowsPerScanSlot : Nat :=
  2 * (cellBits + extensionBits + 2)

/-- Boundary rows: timestamp plus four two-component products. -/
def boundaryRows : Nat := 1 + 4 * 2

def Params.rowCount (params : Params) : Nat :=
  params.fillerColumns.length +
    params.operationSlots * params.rowsPerOperation +
    params.scanSlots * rowsPerScanSlot + boundaryRows

/-! ## Selected reduced 42-times-6 memory profile -/

/-- One memory access per application batch. The 1,024/1,024 memory geometry
and 1,024 scan slots are the selected reduced benchmark profile. -/
def wasm42x6 : Params where
  r := 10
  mu := 10
  operationSlots := 1
  scanSlots := 1024
  segmentLimit := 16

theorem wasm42x6_valid : wasm42x6.Valid := by
  constructor <;> decide

theorem wasm42x6_operationSlots : wasm42x6.operationSlots = 1 := by
  rfl

theorem wasm42x6_scanSlots : wasm42x6.scanSlots = 1024 := by
  rfl

theorem wasm42x6_stepsPerSegment :
    wasm42x6.scannedCells / wasm42x6.scanSlots = 2 := by
  decide

theorem wasm42x6_operationBits : wasm42x6.operationBits = 121 := by
  decide

theorem wasm42x6_countBits : wasm42x6.countBits = 1 := by
  decide

theorem wasm42x6_fillerColumns : wasm42x6.fillerColumns.length = 132 := by
  decide

theorem wasm42x6_publicColumns : wasm42x6.publicEnd = 1401 := by
  decide

theorem wasm42x6_operationLane : wasm42x6.operationLane = 1404 := by
  decide

theorem wasm42x6_initialScanLane : wasm42x6.initialScanLane = 1566 := by
  decide

theorem wasm42x6_finalScanLane : wasm42x6.finalScanLane = 79434 := by
  decide

theorem wasm42x6_auxiliaryStart : wasm42x6.auxiliaryStart = 157302 := by
  decide

theorem wasm42x6_scanAuxiliaryStart :
    wasm42x6.scanAuxiliaryStart = 157603 := by
  decide

theorem wasm42x6_columnCount : wasm42x6.columnCount = 419747 := by
  decide

theorem wasm42x6_witnessColumns : wasm42x6.witnessColumns = 418346 := by
  decide

theorem wasm42x6_rowsPerOperation : wasm42x6.rowsPerOperation = 436 := by
  decide

theorem wasm42x6_rowCount : wasm42x6.rowCount = 422465 := by
  decide

end Nightstream.Implementation.Lowering.Nebula.Layout
