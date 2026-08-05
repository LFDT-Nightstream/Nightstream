import Nightstream.Implementation.Lowering.Nebula.TerminalR1cs

/-!
Contract: whole-program terminal R1CS for the Lean-owned Nebula compiler.

Assurance tier: model-level.

Owns: flattening of per-source-row lowerings, exact row and auxiliary-column
census, source-program soundness, and the shape proof for the selected
stackless compiler.

Does not own: assignment placement beside F-prime, terminal Ajtai checks,
Spartan, WHIR, JSON, Rust, or a security reduction.

Emits constraints: the concatenation of the per-row programs from
`Nebula.TerminalR1cs`.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Lowering.Nebula.TerminalR1csProgram

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Nebula
open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.TerminalR1cs

def rows (source : List Rows.Row) : List TerminalR1cs.Row :=
  source.flatMap lowerRow

def columns (source : List Rows.Row) : List Column :=
  source.flatMap auxiliaryColumns

def extensionCount : List Rows.Row -> Nat
  | [] => 0
  | row :: rest =>
      (if familyKind row.id.family = .extension then 1 else 0) +
        extensionCount rest

def WellShaped (source : List Rows.Row) : Prop :=
  ∀ row, row ∈ source -> Shape row

@[simp] theorem familyKind_bitRow (id : Rows.RowId) (column : Nat) :
    familyKind (Rows.bitRow id column).id.family = familyKind id.family :=
  rfl

@[simp] theorem familyKind_productRow (id : Rows.RowId)
    (left right : Rows.LinearCombination) :
    familyKind (Rows.productRow id left right).id.family =
      familyKind id.family :=
  rfl

@[simp] theorem familyKind_linearRow (id : Rows.RowId)
    (left right : Rows.LinearCombination) :
    familyKind (Rows.linearRow id left right).id.family =
      familyKind id.family :=
  rfl

@[simp] theorem familyKind_extensionUpdateRow (id : Rows.RowId)
    (output a b pad active fingerprintA fingerprintB valueA valueB value :
      Rows.LinearCombination) :
    familyKind
        (Rows.extensionUpdateRow id output a b pad active fingerprintA
          fingerprintB valueA valueB value).id.family =
      familyKind id.family :=
  rfl

theorem shape_withPosition {row : Rows.Row} (shape : Shape row)
    (position : Nat) : Shape (row.withPosition position) := by
  cases shape with
  | bit id column kind =>
      change Shape
        (Rows.bitRow { id with position := position } column)
      exact .bit _ _ kind
  | product id left right kind =>
      change Shape
        (Rows.productRow { id with position := position } left right)
      exact .product _ _ _ kind
  | linear id left right kind =>
      change Shape
        (Rows.linearRow { id with position := position } left right)
      exact .linear _ _ _ kind
  | extension id output extensionA extensionB pad active fingerprintA
      fingerprintB valueA valueB value kind =>
      change Shape
        (Rows.extensionUpdateRow { id with position := position } output
          extensionA extensionB pad active fingerprintA fingerprintB valueA
          valueB value)
      exact .extension _ _ _ _ _ _ _ _ _ _ _ kind

@[simp] theorem wellShaped_append_iff (left right : List Rows.Row) :
    WellShaped (left ++ right) ↔ WellShaped left ∧ WellShaped right := by
  constructor
  · intro shaped
    exact
      ⟨fun row member => shaped row (List.mem_append_left right member),
       fun row member => shaped row (List.mem_append_right left member)⟩
  · rintro ⟨leftShaped, rightShaped⟩ row member
    rw [List.mem_append] at member
    exact member.elim (leftShaped row) (rightShaped row)

theorem numberRowsFrom_wellShaped (position : Nat) (source : List Rows.Row)
    (shaped : WellShaped source) :
    WellShaped (Compiler.numberRowsFrom position source) := by
  induction source generalizing position with
  | nil => intro row member; nomatch member
  | cons head tail inductionHypothesis =>
      intro row member
      simp only [Compiler.numberRowsFrom, List.mem_cons] at member
      rcases member with rfl | member
      · exact shape_withPosition (shaped head (by simp)) position
      · exact inductionHypothesis (position + 1)
          (fun item itemMember => shaped item (by simp [itemMember])) row member

@[simp] theorem extensionCount_append (left right : List Rows.Row) :
    extensionCount (left ++ right) =
      extensionCount left + extensionCount right := by
  induction left with
  | nil => simp [extensionCount]
  | cons row rest inductionHypothesis =>
      simp [extensionCount, inductionHypothesis, Nat.add_assoc]

theorem rows_length (source : List Rows.Row) :
    (rows source).length = source.length + 5 * extensionCount source := by
  induction source with
  | nil => rfl
  | cons row rest inductionHypothesis =>
      rw [show rows (row :: rest) = lowerRow row ++ rows rest from rfl,
        List.length_append, inductionHypothesis]
      cases kind : familyKind row.id.family <;>
        simp [lowerRow, extensionCount, kind] <;> omega

theorem columns_length (source : List Rows.Row) :
    (columns source).length = 5 * extensionCount source := by
  induction source with
  | nil => rfl
  | cons row rest inductionHypothesis =>
      rw [show columns (row :: rest) =
        auxiliaryColumns row ++ columns rest from rfl,
        List.length_append, inductionHypothesis]
      cases kind : familyKind row.id.family <;>
        simp [auxiliaryColumns, extensionCount, kind] <;> omega

theorem sound (source : List Rows.Row) (assignment : Column -> F)
    (constantOne : assignment (.source 0) = 1)
    (wellShaped : WellShaped source)
    (satisfied : TerminalR1cs.Satisfies (rows source) assignment) :
    Rows.Satisfies source (fun column => assignment (.source column)) := by
  intro row member
  apply lowerRow_sound row assignment constantOne (wellShaped row member)
  rw [TerminalR1cs.satisfies_iff_forall] at satisfied ⊢
  intro lowered loweredMember
  apply satisfied lowered
  rw [rows, List.mem_flatMap]
  exact ⟨row, member, loweredMember⟩

theorem numberRowsFrom_extensionCount (position : Nat)
    (source : List Rows.Row) :
    extensionCount (Compiler.numberRowsFrom position source) =
      extensionCount source := by
  induction source generalizing position with
  | nil => rfl
  | cons row rest inductionHypothesis =>
      change
        (if familyKind row.id.family = .extension then 1 else 0) +
            extensionCount (Compiler.numberRowsFrom (position + 1) rest) =
          (if familyKind row.id.family = .extension then 1 else 0) +
            extensionCount rest
      rw [inductionHypothesis]

private theorem flatMap_range_extensionCount
    (count width : Nat) (items : Nat -> List Rows.Row)
    (exact : ∀ index, index < count -> extensionCount (items index) = width) :
    extensionCount ((List.range count).flatMap items) = count * width := by
  induction count with
  | zero => simp [extensionCount]
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.flatMap_append, extensionCount_append]
      simp only [List.flatMap_singleton]
      rw [inductionHypothesis]
      · rw [exact count (Nat.lt_succ_self count), Nat.succ_mul]
      · intro index bound
        exact exact index (Nat.lt_succ_of_lt bound)

private theorem extensionCount_map_zero
    {alpha : Type} (items : List alpha) (make : alpha -> Rows.Row)
    (nonextension : ∀ item, familyKind (make item).id.family ≠ .extension) :
    extensionCount (items.map make) = 0 := by
  induction items with
  | nil => rfl
  | cons item rest inductionHypothesis =>
      simp [extensionCount, nonextension item, inductionHypothesis]

theorem bitRows_extensionCount (family : Rows.Family)
    (slot ordinalBase start width : Nat)
    (kind : familyKind family = .bit) :
    extensionCount
      (Compiler.bitRows family slot ordinalBase start width) = 0 := by
  apply extensionCount_map_zero
  intro offset
  rw [familyKind_bitRow]
  change familyKind family ≠ .extension
  rw [kind]
  decide

theorem fillerRows_extensionCount (params : Params) :
    extensionCount (Compiler.fillerRows params) = 0 := by
  apply extensionCount_map_zero
  intro column
  rw [familyKind_linearRow]
  simp [Compiler.id, familyKind]

theorem extensionRows_extensionCount (family : Rows.Family) (slot : Nat)
    (output0 output1 previous0 previous1 pad active fingerprint0
      fingerprint1 value0 value1 value : Rows.LinearCombination)
    (kind : familyKind family = .extension) :
    extensionCount
      (Compiler.extensionRows family slot output0 output1 previous0 previous1
        pad active fingerprint0 fingerprint1 value0 value1 value) = 2 := by
  simp [Compiler.extensionRows, extensionCount, Compiler.id, kind]

theorem operationLaneBitRows_extensionCount (params : Params) (slot : Nat) :
    extensionCount (Compiler.operationLaneBitRows params slot) = 0 := by
  unfold Compiler.operationLaneBitRows
  rw [extensionCount_append, extensionCount_append, extensionCount_append,
    extensionCount_append]
  rw [bitRows_extensionCount .operationBit _ _ _ _ rfl,
    bitRows_extensionCount .operationBit _ _ _ _ rfl,
    bitRows_extensionCount .operationBit _ _ _ _ rfl,
    bitRows_extensionCount .operationBit _ _ _ _ rfl,
    bitRows_extensionCount .operationBit _ _ _ _ rfl]

theorem operationProductRows_extensionCount
    (params : Params) (slot : Nat) (write : Bool) :
    extensionCount (Compiler.operationProductRows params slot write) = 2 := by
  cases write <;>
    simp only [Compiler.operationProductRows, Bool.false_eq_true, if_false,
      if_true] <;>
    apply extensionRows_extensionCount <;>
    rfl

theorem operationCoreRows_extensionCount (params : Params) (slot : Nat) :
    extensionCount (Compiler.operationCoreRows params slot) = 4 := by
  let pad := Compiler.operationPad params slot
  let isWrite := Compiler.operationIsWrite params slot
  let ram := Compiler.operationRam params slot
  let address := Compiler.operationAddress params slot
  let readValue := Compiler.operationReadValue params slot
  let writeValue := Compiler.operationWriteValue params slot
  let readTimestamp := Compiler.operationReadTimestamp params slot
  let count := Compiler.operationCountWord params slot
  let previousCount := if slot = 0 then Rows.LinearCombination.zero
    else Compiler.operationCountWord params (slot - 1)
  let notPad := Rows.LinearCombination.sub Compiler.one pad
  let rom := Rows.LinearCombination.sub Compiler.one ram
  let writeTimestamp := Compiler.operationWriteTimestamp params slot
  let fixed : List Rows.Row :=
    [ Rows.linearRow (Compiler.id .operationCount slot 0 0) count
        (Rows.LinearCombination.add previousCount notPad)
    , Rows.productRow (Compiler.id .readWrite slot 0 0)
        (Rows.LinearCombination.sub Compiler.one isWrite)
        (Rows.LinearCombination.sub writeValue readValue)
    , Rows.productRow (Compiler.id .timestampOrder slot 0 0) notPad
        (Rows.LinearCombination.sub
          (Rows.LinearCombination.sub
            (Rows.LinearCombination.sub writeTimestamp readTimestamp)
            Compiler.one)
          (Compiler.operationDiffWord params slot))
    , Rows.productRow (Compiler.id .romWrite slot 0 0) isWrite rom ]
  let rangeRows :=
    (List.range (params.addressBits - params.r)).map fun offset =>
      Rows.productRow (Compiler.id .romRange slot 0 offset) rom
        (Rows.LinearCombination.bit
          (params.operationSlot slot + 3 + params.r + offset))
  let paddingFields :=
    [isWrite, ram, address, readValue, writeValue, readTimestamp]
  let paddingRows := paddingFields.mapIdx fun ordinal field =>
    Rows.productRow (Compiler.id .padding slot 0 ordinal) pad field
  have fixedZero : extensionCount fixed = 0 := by
    rfl
  have rangeZero : extensionCount rangeRows = 0 := by
    apply extensionCount_map_zero
    intro offset
    rw [familyKind_productRow]
    simp [Compiler.id, familyKind]
  have paddingZero : extensionCount paddingRows = 0 := by
    rfl
  change extensionCount
    (fixed ++ rangeRows ++ paddingRows ++
      Compiler.operationProductRows params slot false ++
      Compiler.operationProductRows params slot true) = 4
  simp [extensionCount_append, fixedZero, rangeZero, paddingZero,
    operationProductRows_extensionCount]

theorem operationRows_extensionCount (params : Params) (slot : Nat) :
    extensionCount (Compiler.operationRows params slot) = 4 := by
  rw [show Compiler.operationRows params slot =
    Compiler.operationLaneBitRows params slot ++
      Compiler.operationCoreRows params slot from rfl,
    extensionCount_append, operationLaneBitRows_extensionCount,
    operationCoreRows_extensionCount]

theorem scanRowsForLane_extensionCount
    (params : Params) (final : Bool) (slot : Nat) :
    extensionCount (Compiler.scanRowsForLane params final slot) = 2 := by
  cases final <;>
    simp only [Compiler.scanRowsForLane, Bool.false_eq_true, if_false,
      if_true, extensionCount_append] <;>
    rw [bitRows_extensionCount _ _ _ _ _ rfl,
      bitRows_extensionCount _ _ _ _ _ rfl,
      extensionRows_extensionCount _ _ _ _ _ _ _ _ _ _ _ _ _ rfl]

theorem scanRows_extensionCount (params : Params) (slot : Nat) :
    extensionCount (Compiler.scanRows params slot) = 4 := by
  simp [Compiler.scanRows, extensionCount_append,
    scanRowsForLane_extensionCount]

theorem boundaryRows_extensionCount (params : Params) :
    extensionCount (Compiler.boundaryRows params) = 0 := by
  rfl

theorem compilerRows_extensionCount (params : Params) :
    extensionCount (Compiler.rows params) =
      4 * params.operationSlots + 4 * params.scanSlots := by
  have operations :
      extensionCount
        ((List.range params.operationSlots).flatMap
          (Compiler.operationRows params)) = params.operationSlots * 4 :=
    flatMap_range_extensionCount params.operationSlots 4
      (Compiler.operationRows params)
      (fun index _ => operationRows_extensionCount params index)
  have scans :
      extensionCount
        ((List.range params.scanSlots).flatMap
          (Compiler.scanRows params)) = params.scanSlots * 4 :=
    flatMap_range_extensionCount params.scanSlots 4
      (Compiler.scanRows params)
      (fun index _ => scanRows_extensionCount params index)
  rw [Compiler.rows, numberRowsFrom_extensionCount]
  unfold Compiler.rawRows
  rw [extensionCount_append, extensionCount_append, extensionCount_append,
    fillerRows_extensionCount, operations, scans,
    boundaryRows_extensionCount]
  omega

theorem wasm42x6_extensionCount :
    extensionCount (Compiler.rows wasm42x6) = 4100 := by
  rw [compilerRows_extensionCount]
  decide

theorem wasm42x6_rows_length :
    (rows (Compiler.rows wasm42x6)).length = 442965 := by
  rw [rows_length, Compiler.wasm42x6_rows_length,
    wasm42x6_extensionCount]

theorem wasm42x6_columns_length :
    (columns (Compiler.rows wasm42x6)).length = 20500 := by
  rw [columns_length, wasm42x6_extensionCount]

end Nightstream.Implementation.Lowering.Nebula.TerminalR1csProgram
