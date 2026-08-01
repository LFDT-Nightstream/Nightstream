import Nightstream.Implementation.Lowering.Nebula.Rows

/-!
Lean-owned stackless Nebula `S_mem` CCS compiler.

Assurance tier: model-level.

Owns: the exact E1--E9 operation rows, S1--S3 scan rows, alignment pins,
boundary links, row order, and the derived row count. Every row is an exact
fifteen-matrix image consumed by `StepPolynomial.polynomial`.

Does not own: WASM application-port bindings, the values assigned to these
columns, Fiat--Shamir challenge timing, folding, R1CS terminal lowering,
Rust serialization, or a security reduction.

This compiler intentionally supports only the stackless relation selected by
the current 42-times-6 benchmark. It does not omit stack checks from a profile
that declares stacks.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.Nebula.Compiler

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Nebula
open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.Rows
open Nightstream.Implementation.Lowering.Nebula.Rows.LinearCombination

private abbrev Lin := Rows.LinearCombination

def fieldOfNat (value : Nat) : F :=
  ⟨value % goldilocksModulus, Nat.mod_lt _ (by decide)⟩

def xColumn (offset : Nat) : Nat := 1 + offset

namespace XOffset

def segment : Nat := 0
def step : Nat := segment + segmentIndexBits
def timestampIn : Nat := step + stepIndexBits
def timestampOut : Nat := timestampIn + timestampBits
def gamma : Nat := timestampOut + timestampBits
def productsIn : Nat := gamma + 2 * extensionBits
def productsOut : Nat := productsIn + 4 * extensionBits

theorem end_exact : productsOut + 4 * extensionBits = publicInputBits := by
  rfl

end XOffset

def one : Lin := constant 1

def publicWord (offset width : Nat) : Lin :=
  word (xColumn offset) width

def gammaWord (challenge component : Nat) : Lin :=
  publicWord
    (XOffset.gamma + challenge * extensionBits +
      component * extensionLimbBits)
    extensionLimbBits

def productInputWord (product component : Nat) : Lin :=
  publicWord
    (XOffset.productsIn + product * extensionBits +
      component * extensionLimbBits)
    extensionLimbBits

def productOutputWord (product component : Nat) : Lin :=
  publicWord
    (XOffset.productsOut + product * extensionBits +
      component * extensionLimbBits)
    extensionLimbBits

def id (family : Family) (slot component ordinal : Nat) : RowId :=
  { family := family
    slot := slot
    component := component
    ordinal := ordinal }

def bitRows (family : Family) (slot ordinalBase start width : Nat) :
    List Row :=
  (List.range width).map fun offset =>
    bitRow (id family slot 0 (ordinalBase + offset)) (start + offset)

def fillerRows (params : Params) : List Row :=
  params.fillerColumns.map fun column =>
    linearRow (id .filler column 0 0) (bit column) zero

def operationLaneBitRows (params : Params) (slot : Nat) : List Row :=
  let operation := bitRows .operationBit slot 0
    (params.operationSlot slot) params.operationBits
  let diff := bitRows .operationBit slot params.operationBits
    (params.operationDiff slot) timestampBits
  let count := bitRows .operationBit slot
    (params.operationBits + timestampBits)
    (params.operationCount slot) params.countBits
  let readProduct := bitRows .operationBit slot
    (params.operationBits + timestampBits + params.countBits)
    (params.operationReadProduct slot) extensionBits
  let writeProduct := bitRows .operationBit slot
    (params.operationBits + timestampBits + params.countBits + extensionBits)
    (params.operationWriteProduct slot) extensionBits
  operation ++ diff ++ count ++ readProduct ++ writeProduct

def operationPad (params : Params) (slot : Nat) : Lin :=
  bit (params.operationSlot slot)

def operationIsWrite (params : Params) (slot : Nat) : Lin :=
  bit (params.operationSlot slot + 1)

def operationRam (params : Params) (slot : Nat) : Lin :=
  bit (params.operationSlot slot + 2)

def operationAddress (params : Params) (slot : Nat) : Lin :=
  word (params.operationSlot slot + 3) params.addressBits

def operationReadValue (params : Params) (slot : Nat) : Lin :=
  word (params.operationSlot slot + 3 + params.addressBits) valueBits

def operationWriteValue (params : Params) (slot : Nat) : Lin :=
  word (params.operationSlot slot + 3 + params.addressBits + valueBits)
    valueBits

def operationReadTimestamp (params : Params) (slot : Nat) : Lin :=
  word (params.operationSlot slot + 3 + params.addressBits + 2 * valueBits)
    timestampBits

def operationCountWord (params : Params) (slot : Nat) : Lin :=
  word (params.operationCount slot) params.countBits

def operationDiffWord (params : Params) (slot : Nat) : Lin :=
  word (params.operationDiff slot) timestampBits

def operationReadProductWord (params : Params) (slot component : Nat) : Lin :=
  word (params.operationReadProduct slot + component * extensionLimbBits)
    extensionLimbBits

def operationWriteProductWord (params : Params) (slot component : Nat) : Lin :=
  word (params.operationWriteProduct slot + component * extensionLimbBits)
    extensionLimbBits

def previousOperationProduct
    (params : Params) (slot product component : Nat) : Lin :=
  if slot = 0 then productInputWord product component
  else if product = 0 then operationReadProductWord params (slot - 1) component
  else operationWriteProductWord params (slot - 1) component

def operationGlobalIndex (params : Params) (slot : Nat) : Lin :=
  add (operationAddress params slot)
    (scale (fieldOfNat params.romCells) (operationRam params slot))

def timestampIn : Lin := publicWord XOffset.timestampIn timestampBits

def operationWriteTimestamp (params : Params) (slot : Nat) : Lin :=
  add timestampIn (operationCountWord params slot)

def operationFingerprintPrefix
    (params : Params) (slot : Nat) (write : Bool) : Lin :=
  let timestamp := if write then operationWriteTimestamp params slot
    else operationReadTimestamp params slot
  sub (sub (gammaWord 1 0) timestamp)
    (scale (fieldTwoPower timestampBits)
      (operationGlobalIndex params slot))

def extensionRows
    (family : Family) (slot : Nat)
    (output0 output1 previous0 previous1 pad active fingerprint0
      fingerprint1 value0 value1 value : Lin) : List Row :=
  [ extensionUpdateRow (id family slot 0 0)
      output0 previous0 (scale 7 previous1) pad active
      fingerprint0 fingerprint1 value0 value1 value
  , extensionUpdateRow (id family slot 1 0)
      output1 previous1 previous0 pad active
      fingerprint0 fingerprint1 value0 value1 value
  ]

def operationProductRows
    (params : Params) (slot : Nat) (write : Bool) : List Row :=
  let product := if write then 1 else 0
  let output0 := if write then operationWriteProductWord params slot 0
    else operationReadProductWord params slot 0
  let output1 := if write then operationWriteProductWord params slot 1
    else operationReadProductWord params slot 1
  let previous0 := previousOperationProduct params slot product 0
  let previous1 := previousOperationProduct params slot product 1
  let pad := operationPad params slot
  let active := sub one pad
  let fingerprintPrefix := operationFingerprintPrefix params slot write
  let value := if write then operationWriteValue params slot
    else operationReadValue params slot
  extensionRows (if write then .writeProduct else .readProduct) slot
    output0 output1 previous0 previous1 pad active fingerprintPrefix
    (gammaWord 1 1)
    (gammaWord 0 0) (gammaWord 0 1) value

def operationCoreRows (params : Params) (slot : Nat) : List Row :=
  let pad := operationPad params slot
  let isWrite := operationIsWrite params slot
  let ram := operationRam params slot
  let address := operationAddress params slot
  let readValue := operationReadValue params slot
  let writeValue := operationWriteValue params slot
  let readTimestamp := operationReadTimestamp params slot
  let count := operationCountWord params slot
  let previousCount := if slot = 0 then zero
    else operationCountWord params (slot - 1)
  let notPad := sub one pad
  let rom := sub one ram
  let writeTimestamp := operationWriteTimestamp params slot
  let fixed :=
    [ linearRow (id .operationCount slot 0 0) count
        (add previousCount notPad)
    , productRow (id .readWrite slot 0 0) (sub one isWrite)
        (sub writeValue readValue)
    , productRow (id .timestampOrder slot 0 0) notPad
        (sub (sub (sub writeTimestamp readTimestamp) one)
          (operationDiffWord params slot))
    , productRow (id .romWrite slot 0 0) isWrite rom
    ]
  let rangeRows :=
    (List.range (params.addressBits - params.r)).map fun offset =>
      productRow (id .romRange slot 0 offset) rom
        (bit (params.operationSlot slot + 3 + params.r + offset))
  let paddingFields := [isWrite, ram, address, readValue, writeValue,
    readTimestamp]
  let paddingRows := paddingFields.mapIdx fun ordinal field =>
    productRow (id .padding slot 0 ordinal) pad field
  fixed ++ rangeRows ++ paddingRows ++
    operationProductRows params slot false ++
    operationProductRows params slot true

def operationRows (params : Params) (slot : Nat) : List Row :=
  operationLaneBitRows params slot ++ operationCoreRows params slot

def scanCellStart (params : Params) (final : Bool) (slot : Nat) : Nat :=
  (if final then params.finalScanLane else params.initialScanLane) +
    slot * cellBits

def scanValue (params : Params) (final : Bool) (slot : Nat) : Lin :=
  word (scanCellStart params final slot) valueBits

def scanTimestamp (params : Params) (final : Bool) (slot : Nat) : Lin :=
  word (scanCellStart params final slot + valueBits) timestampBits

def scanProductStart (params : Params) (final : Bool) (slot : Nat) : Nat :=
  if final then params.finalScanProduct slot
  else params.initialScanProduct slot

def scanProductWord
    (params : Params) (final : Bool) (slot component : Nat) : Lin :=
  word (scanProductStart params final slot + component * extensionLimbBits)
    extensionLimbBits

def previousScanProduct
    (params : Params) (final : Bool) (slot component : Nat) : Lin :=
  if slot = 0 then productInputWord (if final then 3 else 2) component
  else scanProductWord params final (slot - 1) component

def scanGlobalIndex (params : Params) (slot : Nat) : Lin :=
  add (scale (fieldOfNat params.scanSlots)
      (publicWord XOffset.step stepIndexBits))
    (constant (fieldOfNat slot))

def scanRowsForLane (params : Params) (final : Bool) (slot : Nat) :
    List Row :=
  let familyBit := if final then Family.finalScanBit else .initialScanBit
  let familyProduct := if final then Family.finalScanProduct
    else .initialScanProduct
  let cellStart := scanCellStart params final slot
  let productStart := scanProductStart params final slot
  let laneBits := bitRows familyBit slot 0 cellStart cellBits
  let productBits := bitRows familyBit slot cellBits productStart extensionBits
  let fingerprintPrefix :=
    sub (sub (gammaWord 1 0) (scanTimestamp params final slot))
    (scale (fieldTwoPower timestampBits) (scanGlobalIndex params slot))
  laneBits ++ productBits ++
    extensionRows familyProduct slot
      (scanProductWord params final slot 0)
      (scanProductWord params final slot 1)
      (previousScanProduct params final slot 0)
      (previousScanProduct params final slot 1)
      zero one fingerprintPrefix (gammaWord 1 1)
      (gammaWord 0 0) (gammaWord 0 1)
      (scanValue params final slot)

def scanRows (params : Params) (slot : Nat) : List Row :=
  scanRowsForLane params false slot ++ scanRowsForLane params true slot

def lastOperationSlot (params : Params) : Nat :=
  params.operationSlots - 1

def lastScanSlot (params : Params) : Nat := params.scanSlots - 1

def boundaryRows (params : Params) : List Row :=
  let timestamp :=
    linearRow (id .boundaryTimestamp 0 0 0)
      (publicWord XOffset.timestampOut timestampBits)
      (add timestampIn
        (operationCountWord params (lastOperationSlot params)))
  let products :=
    (List.range 4).flatMap fun product =>
      (List.range 2).map fun component =>
        let source :=
          if product = 0 then
            operationReadProductWord params (lastOperationSlot params) component
          else if product = 1 then
            operationWriteProductWord params (lastOperationSlot params) component
          else if product = 2 then
            scanProductWord params false (lastScanSlot params) component
          else
            scanProductWord params true (lastScanSlot params) component
        linearRow (id .boundaryProduct product component 0)
          (productOutputWord product component) source
  timestamp :: products

/-- Exact semantic row order before physical positions are attached. -/
def rawRows (params : Params) : List Row :=
  fillerRows params ++
    (List.range params.operationSlots).flatMap (operationRows params) ++
    (List.range params.scanSlots).flatMap (scanRows params) ++
    boundaryRows params

/-- Attach the global physical row position without changing any matrix
image. -/
def numberRowsFrom : Nat -> List Row -> List Row
  | _, [] => []
  | position, row :: rest =>
      row.withPosition position :: numberRowsFrom (position + 1) rest

/-- Exact emitted row order: alignment, operation slots, scan slots, then
public-output boundary links. Every row carries its unique physical position. -/
def rows (params : Params) : List Row :=
  numberRowsFrom 0 (rawRows params)

theorem bitRows_length (family : Family) (slot ordinalBase start width : Nat) :
    (bitRows family slot ordinalBase start width).length = width := by
  simp [bitRows]

theorem fillerRows_length (params : Params) :
    (fillerRows params).length = params.fillerColumns.length := by
  simp [fillerRows]

theorem operationLaneBitRows_length (params : Params) (slot : Nat) :
    (operationLaneBitRows params slot).length = params.operationBitRows := by
  simp [operationLaneBitRows, Params.operationBitRows, bitRows_length]
  omega

theorem extensionRows_length (family : Family) (slot : Nat)
    (output0 output1 previous0 previous1 pad active fingerprint0
      fingerprint1 value0 value1 value : Lin) :
    (extensionRows family slot output0 output1 previous0 previous1 pad active
      fingerprint0 fingerprint1 value0 value1 value).length = 2 := by
  rfl

theorem operationProductRows_length
    (params : Params) (slot : Nat) (write : Bool) :
    (operationProductRows params slot write).length = 2 := by
  simp [operationProductRows, extensionRows]

theorem operationCoreRows_length (params : Params) (slot : Nat) :
    (operationCoreRows params slot).length =
      4 + (params.addressBits - params.r) + 6 + 4 := by
  simp [operationCoreRows, operationProductRows_length]
  omega

theorem operationRows_length (params : Params) (slot : Nat) :
    (operationRows params slot).length = params.rowsPerOperation := by
  rw [show operationRows params slot =
    operationLaneBitRows params slot ++ operationCoreRows params slot from rfl,
    List.length_append, operationLaneBitRows_length,
    operationCoreRows_length]
  unfold Params.rowsPerOperation
  omega

theorem scanRowsForLane_length
    (params : Params) (final : Bool) (slot : Nat) :
    (scanRowsForLane params final slot).length =
      cellBits + extensionBits + 2 := by
  simp [scanRowsForLane, extensionRows, bitRows_length]
  omega

theorem scanRows_length (params : Params) (slot : Nat) :
    (scanRows params slot).length = rowsPerScanSlot := by
  simp [scanRows, scanRowsForLane_length, rowsPerScanSlot]
  omega

theorem boundaryRows_length (params : Params) :
    (boundaryRows params).length = Layout.boundaryRows := by
  simp [boundaryRows, Layout.boundaryRows]
  decide

theorem numberRowsFrom_length (position : Nat) (source : List Row) :
    (numberRowsFrom position source).length = source.length := by
  induction source generalizing position with
  | nil => rfl
  | cons row rest inductionHypothesis =>
      simp [numberRowsFrom, inductionHypothesis]

theorem numberRowsFrom_positions (position : Nat) (source : List Row) :
    (numberRowsFrom position source).map (fun row => row.id.position) =
      List.range' position source.length := by
  induction source generalizing position with
  | nil => rfl
  | cons row rest inductionHypothesis =>
      simp [numberRowsFrom, inductionHypothesis, List.range'_succ]

theorem rows_positions (params : Params) :
    (rows params).map (fun row => row.id.position) =
      List.range (rawRows params).length := by
  simpa [rows, List.range_eq_range'] using
    numberRowsFrom_positions 0 (rawRows params)

private theorem nodup_of_map_nodup
    {alpha beta : Type} (items : List alpha) (project : alpha -> beta)
    (mapped : (items.map project).Nodup) : items.Nodup := by
  induction items with
  | nil => exact List.nodup_nil
  | cons head tail inductionHypothesis =>
      rw [List.map_cons, List.nodup_cons] at mapped
      rw [List.nodup_cons]
      constructor
      · intro member
        exact mapped.1 (List.mem_map.mpr ⟨_, member, rfl⟩)
      · exact inductionHypothesis mapped.2

theorem rows_ids_nodup (params : Params) :
    ((rows params).map Row.id).Nodup := by
  have positions :
      (((rows params).map Row.id).map RowId.position).Nodup := by
    simpa [List.map_map, Function.comp_def, rows_positions] using
      (List.nodup_range (n := (rawRows params).length))
  exact nodup_of_map_nodup ((rows params).map Row.id) RowId.position positions

private theorem flatMap_range_length
    (count width : Nat) (items : Nat -> List Row)
    (exact : ∀ index, index < count -> (items index).length = width) :
    ((List.range count).flatMap items).length = count * width := by
  induction count with
  | zero => simp
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.flatMap_append, List.length_append]
      simp only [List.flatMap_singleton]
      rw [inductionHypothesis]
      · rw [exact count (Nat.lt_succ_self count)]
        rw [Nat.succ_mul]
      · intro index bound
        exact exact index (Nat.lt_succ_of_lt bound)

theorem rows_length (params : Params) :
    (rows params).length = params.rowCount := by
  have operations :
      ((List.range params.operationSlots).flatMap
          (operationRows params)).length =
        params.operationSlots * params.rowsPerOperation :=
    flatMap_range_length params.operationSlots params.rowsPerOperation
      (operationRows params) (fun index _ => operationRows_length params index)
  have scans :
      ((List.range params.scanSlots).flatMap (scanRows params)).length =
        params.scanSlots * rowsPerScanSlot :=
    flatMap_range_length params.scanSlots rowsPerScanSlot
      (scanRows params) (fun index _ => scanRows_length params index)
  rw [show (rows params).length = (rawRows params).length by
    exact numberRowsFrom_length 0 (rawRows params)]
  unfold rawRows Params.rowCount
  rw [List.length_append, List.length_append, List.length_append,
    fillerRows_length, operations, scans, boundaryRows_length]

theorem wasm42x6_rows_length :
    (rows wasm42x6).length = 422465 := by
  rw [rows_length, wasm42x6_rowCount]

end Nightstream.Implementation.Lowering.Nebula.Compiler
