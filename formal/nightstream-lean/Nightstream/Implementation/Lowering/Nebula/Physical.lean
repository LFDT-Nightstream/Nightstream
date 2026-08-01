import Nightstream.Implementation.Lowering.Nebula.Compiler

/-!
Physical support and allocation proof for the Lean-owned Nebula compiler.

Assurance tier: model-level.

Owns: sparse-term support bounds, complete column allocation, positional row
ownership, and the exact public/witness split for the selected 42-times-6
profile.

Does not own: witness values, WASM port binding, transcript challenges, Rust,
or a security reduction.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.Nebula.Physical

open Nightstream.Implementation.Lowering.Nebula
open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.Rows
open Nightstream.Implementation.Lowering.Nebula.Compiler

namespace LinearCombination

def Bounded (limit : Nat) (combination : Rows.LinearCombination) : Prop :=
  ∀ term, term ∈ combination -> term.column < limit

theorem bounded_zero (limit : Nat) : Bounded limit .zero := by
  simp [Bounded, Rows.LinearCombination.zero]

theorem bounded_constant {limit : Nat} (positive : 0 < limit)
    (coefficient : Nightstream.SuperNeo.Concrete.F) :
    Bounded limit (.constant coefficient) := by
  intro term member
  simp [Rows.LinearCombination.constant] at member
  subst term
  exact positive

theorem bounded_bit {limit column : Nat} (bound : column < limit) :
    Bounded limit (.bit column) := by
  intro term member
  simp [Rows.LinearCombination.bit] at member
  subst term
  exact bound

theorem bounded_wordScaled {limit start width : Nat}
    (bound : start + width ≤ limit)
    (coefficient : Nightstream.SuperNeo.Concrete.F) :
    Bounded limit (.wordScaled start width coefficient) := by
  intro term member
  rcases List.mem_map.1 member with ⟨offset, offsetMember, rfl⟩
  have offsetBound : offset < width := List.mem_range.1 offsetMember
  change start + offset < limit
  omega

theorem bounded_word {limit start width : Nat}
    (bound : start + width ≤ limit) :
    Bounded limit (.word start width) :=
  bounded_wordScaled bound 1

theorem bounded_add {limit : Nat} {left right : Rows.LinearCombination}
    (leftBound : Bounded limit left) (rightBound : Bounded limit right) :
    Bounded limit (.add left right) := by
  intro term member
  rcases List.mem_append.1 member with inLeft | inRight
  · exact leftBound term inLeft
  · exact rightBound term inRight

theorem bounded_scale {limit : Nat} {combination : Rows.LinearCombination}
    (bound : Bounded limit combination)
    (coefficient : Nightstream.SuperNeo.Concrete.F) :
    Bounded limit (.scale coefficient combination) := by
  intro term member
  rcases List.mem_map.1 member with ⟨source, sourceMember, rfl⟩
  exact bound source sourceMember

theorem bounded_neg {limit : Nat} {combination : Rows.LinearCombination}
    (bound : Bounded limit combination) :
    Bounded limit (.neg combination) :=
  bounded_scale bound (-1)

theorem bounded_sub {limit : Nat} {left right : Rows.LinearCombination}
    (leftBound : Bounded limit left) (rightBound : Bounded limit right) :
    Bounded limit (.sub left right) :=
  bounded_add leftBound (bounded_neg rightBound)

end LinearCombination

def ImagesBounded (limit : Nat) (images : Images) : Prop :=
  ∀ role, LinearCombination.Bounded limit (images.at role)

def RowBounded (limit : Nat) (row : Row) : Prop :=
  ImagesBounded limit row.images

def RowsBounded (limit : Nat) (program : List Row) : Prop :=
  ∀ row, row ∈ program -> RowBounded limit row

namespace Row

theorem bounded_bitRow {limit column : Nat} (identifier : RowId)
    (bound : column < limit) :
    RowBounded limit (bitRow identifier column) := by
  unfold RowBounded ImagesBounded
  intro role
  cases role <;> simp only [bitRow, Images.at]
  · exact LinearCombination.bounded_bit bound
  all_goals exact LinearCombination.bounded_zero limit

theorem bounded_productRow {limit : Nat} (identifier : RowId)
    {left right : Rows.LinearCombination}
    (leftBound : LinearCombination.Bounded limit left)
    (rightBound : LinearCombination.Bounded limit right) :
    RowBounded limit (productRow identifier left right) := by
  unfold RowBounded ImagesBounded
  intro role
  cases role <;> simp only [productRow, Images.at]
  all_goals first
    | exact leftBound
    | exact rightBound
    | exact LinearCombination.bounded_zero limit

theorem bounded_linearRow {limit : Nat} (identifier : RowId)
    {left right : Rows.LinearCombination}
    (leftBound : LinearCombination.Bounded limit left)
    (rightBound : LinearCombination.Bounded limit right) :
    RowBounded limit (linearRow identifier left right) := by
  unfold RowBounded ImagesBounded
  intro role
  cases role <;> simp only [linearRow, Images.at]
  all_goals first
    | exact leftBound
    | exact rightBound
    | exact LinearCombination.bounded_zero limit

theorem bounded_extensionUpdateRow {limit : Nat} (identifier : RowId)
    {output a b pad active fingerprintA fingerprintB valueA valueB value :
      Rows.LinearCombination}
    (outputBound : LinearCombination.Bounded limit output)
    (aBound : LinearCombination.Bounded limit a)
    (bBound : LinearCombination.Bounded limit b)
    (padBound : LinearCombination.Bounded limit pad)
    (activeBound : LinearCombination.Bounded limit active)
    (fingerprintABound : LinearCombination.Bounded limit fingerprintA)
    (fingerprintBBound : LinearCombination.Bounded limit fingerprintB)
    (valueABound : LinearCombination.Bounded limit valueA)
    (valueBBound : LinearCombination.Bounded limit valueB)
    (valueBound : LinearCombination.Bounded limit value) :
    RowBounded limit
      (extensionUpdateRow identifier output a b pad active fingerprintA
        fingerprintB valueA valueB value) := by
  unfold RowBounded ImagesBounded
  intro role
  cases role <;> simp only [extensionUpdateRow, Images.at]
  all_goals first
    | exact outputBound
    | exact aBound
    | exact bBound
    | exact padBound
    | exact activeBound
    | exact fingerprintABound
    | exact fingerprintBBound
    | exact valueABound
    | exact valueBBound
    | exact valueBound
    | exact LinearCombination.bounded_zero limit

@[simp] theorem bounded_withPosition {limit position : Nat} {row : Row} :
    RowBounded limit (row.withPosition position) ↔ RowBounded limit row :=
  Iff.rfl

end Row

theorem rowsBounded_append {limit : Nat} {left right : List Row}
    (leftBound : RowsBounded limit left)
    (rightBound : RowsBounded limit right) :
    RowsBounded limit (left ++ right) := by
  intro row member
  rcases List.mem_append.1 member with inLeft | inRight
  · exact leftBound row inLeft
  · exact rightBound row inRight

theorem rowsBounded_nil (limit : Nat) : RowsBounded limit [] := by
  simp [RowsBounded]

theorem rowsBounded_cons {limit : Nat} {head : Row} {tail : List Row}
    (headBound : RowBounded limit head)
    (tailBound : RowsBounded limit tail) :
    RowsBounded limit (head :: tail) := by
  intro row member
  rcases List.mem_cons.1 member with rfl | inTail
  · exact headBound
  · exact tailBound row inTail

theorem rowsBounded_map_withPosition {limit position : Nat}
    {source : List Row} (bound : RowsBounded limit source) :
    RowsBounded limit (Compiler.numberRowsFrom position source) := by
  induction source generalizing position with
  | nil => simp [RowsBounded, Compiler.numberRowsFrom]
  | cons head tail inductionHypothesis =>
      intro row member
      simp only [Compiler.numberRowsFrom, List.mem_cons] at member
      rcases member with rfl | inTail
      · exact (Row.bounded_withPosition).2 (bound head (by simp))
      · exact inductionHypothesis
          (fun item itemMember => bound item (by simp [itemMember])) row inTail

/-! ## Selected 42-times-6 support -/

private abbrev selectedLimit : Nat := wasm42x6.columnCount

private theorem selectedLimit_positive : 0 < selectedLimit := by
  rw [show selectedLimit = 419747 from wasm42x6_columnCount]
  decide

private theorem selected_operation_slot_zero (slot : Nat)
    (bound : slot < wasm42x6.operationSlots) : slot = 0 := by
  change slot < 1 at bound
  omega

private theorem selected_publicWord_bounded (offset width : Nat)
    (bound : offset + width ≤ publicInputBits) :
    LinearCombination.Bounded selectedLimit (publicWord offset width) := by
  apply LinearCombination.bounded_word
  rw [show selectedLimit = 419747 from wasm42x6_columnCount]
  unfold xColumn
  rw [show publicInputBits = 1400 from publicInputBits_exact] at bound
  omega

private theorem selected_one_bounded :
    LinearCombination.Bounded selectedLimit Compiler.one :=
  LinearCombination.bounded_constant selectedLimit_positive 1

private theorem selected_operation_bit_bounded (slot offset : Nat)
    (slotBound : slot < wasm42x6.operationSlots)
    (offsetBound : offset < wasm42x6.operationBits) :
    LinearCombination.Bounded selectedLimit
      (.bit (wasm42x6.operationSlot slot + offset)) := by
  have slotZero := selected_operation_slot_zero slot slotBound
  subst slot
  apply LinearCombination.bounded_bit
  change 1404 + offset < 419747
  change offset < 121 at offsetBound
  omega

private theorem selected_operation_word_bounded
    (slot offset width : Nat)
    (slotBound : slot < wasm42x6.operationSlots)
    (bound : offset + width ≤ wasm42x6.operationBits) :
    LinearCombination.Bounded selectedLimit
      (.word (wasm42x6.operationSlot slot + offset) width) := by
  have slotZero := selected_operation_slot_zero slot slotBound
  subst slot
  apply LinearCombination.bounded_word
  change 1404 + offset + width ≤ 419747
  change offset + width ≤ 121 at bound
  omega

private theorem selected_operation_aux_word_bounded
    (slot offset width : Nat)
    (slotBound : slot < wasm42x6.operationSlots)
    (bound : offset + width ≤ wasm42x6.operationAuxiliaryBits) :
    LinearCombination.Bounded selectedLimit
      (.word (wasm42x6.operationAuxiliary slot + offset) width) := by
  have slotZero := selected_operation_slot_zero slot slotBound
  subst slot
  apply LinearCombination.bounded_word
  change 157302 + offset + width ≤ 419747
  change offset + width ≤ 301 at bound
  omega

private theorem selected_scan_lane_word_bounded
    (final : Bool) (slot offset width : Nat)
    (slotBound : slot < wasm42x6.scanSlots)
    (bound : offset + width ≤ cellBits) :
    LinearCombination.Bounded selectedLimit
      (.word (scanCellStart wasm42x6 final slot + offset) width) := by
  apply LinearCombination.bounded_word
  cases final with
  | false =>
    change 1566 + slot * 76 + offset + width ≤ 419747
    change slot < 1024 at slotBound
    change offset + width ≤ 76 at bound
    omega
  | true =>
    change 79434 + slot * 76 + offset + width ≤ 419747
    change slot < 1024 at slotBound
    change offset + width ≤ 76 at bound
    omega

private theorem selected_scan_aux_word_bounded
    (final : Bool) (slot component : Nat)
    (slotBound : slot < wasm42x6.scanSlots)
    (componentBound : component < 2) :
    LinearCombination.Bounded selectedLimit
      (.word (scanProductStart wasm42x6 final slot +
        component * extensionLimbBits) extensionLimbBits) := by
  apply LinearCombination.bounded_word
  cases final with
  | false =>
    change 157603 + slot * 256 + component * 64 + 64 ≤ 419747
    change slot < 1024 at slotBound
    change component < 2 at componentBound
    omega
  | true =>
    simp only [scanProductStart, ↓reduceIte, Params.finalScanProduct,
      Params.initialScanProduct, scanAuxiliaryBits, extensionBits,
      extensionLimbBits]
    rw [wasm42x6_scanAuxiliaryStart]
    change 157603 + slot * 256 + 128 + component * 64 + 64 ≤ 419747
    change slot < 1024 at slotBound
    change component < 2 at componentBound
    omega

private theorem selected_gammaWord_bounded (challenge component : Nat)
    (challengeBound : challenge < 2) (componentBound : component < 2) :
    LinearCombination.Bounded selectedLimit
      (gammaWord challenge component) := by
  apply selected_publicWord_bounded
  change 120 + challenge * 128 + component * 64 + 64 ≤ 1400
  omega

private theorem selected_productInputWord_bounded (product component : Nat)
    (productBound : product < 4) (componentBound : component < 2) :
    LinearCombination.Bounded selectedLimit
      (productInputWord product component) := by
  apply selected_publicWord_bounded
  change 376 + product * 128 + component * 64 + 64 ≤ 1400
  omega

private theorem selected_productOutputWord_bounded (product component : Nat)
    (productBound : product < 4) (componentBound : component < 2) :
    LinearCombination.Bounded selectedLimit
      (productOutputWord product component) := by
  apply selected_publicWord_bounded
  change 888 + product * 128 + component * 64 + 64 ≤ 1400
  omega

private theorem selected_bitRows_bounded
    (family : Family) (slot ordinalBase start width : Nat)
    (bound : start + width ≤ selectedLimit) :
    RowsBounded selectedLimit
      (bitRows family slot ordinalBase start width) := by
  intro row member
  rcases List.mem_map.1 member with ⟨offset, offsetMember, rfl⟩
  apply Row.bounded_bitRow
  have offsetBound := List.mem_range.1 offsetMember
  omega

private theorem selected_extensionRows_bounded
    (family : Family) (slot : Nat)
    (output0 output1 previous0 previous1 pad active fingerprint0
      fingerprint1 value0 value1 value : Rows.LinearCombination)
    (output0Bound : LinearCombination.Bounded selectedLimit output0)
    (output1Bound : LinearCombination.Bounded selectedLimit output1)
    (previous0Bound : LinearCombination.Bounded selectedLimit previous0)
    (previous1Bound : LinearCombination.Bounded selectedLimit previous1)
    (padBound : LinearCombination.Bounded selectedLimit pad)
    (activeBound : LinearCombination.Bounded selectedLimit active)
    (fingerprint0Bound : LinearCombination.Bounded selectedLimit fingerprint0)
    (fingerprint1Bound : LinearCombination.Bounded selectedLimit fingerprint1)
    (value0Bound : LinearCombination.Bounded selectedLimit value0)
    (value1Bound : LinearCombination.Bounded selectedLimit value1)
    (valueBound : LinearCombination.Bounded selectedLimit value) :
    RowsBounded selectedLimit
      (extensionRows family slot output0 output1 previous0 previous1 pad
        active fingerprint0 fingerprint1 value0 value1 value) := by
  intro row member
  simp only [extensionRows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact Row.bounded_extensionUpdateRow _ output0Bound previous0Bound
      (LinearCombination.bounded_scale previous1Bound 7) padBound activeBound
      fingerprint0Bound fingerprint1Bound value0Bound value1Bound valueBound
  · exact Row.bounded_extensionUpdateRow _ output1Bound previous1Bound
      previous0Bound padBound activeBound fingerprint0Bound fingerprint1Bound
      value0Bound value1Bound valueBound

private theorem selected_operationPad_bounded (slot : Nat)
    (slotBound : slot < wasm42x6.operationSlots) :
    LinearCombination.Bounded selectedLimit (operationPad wasm42x6 slot) := by
  exact selected_operation_bit_bounded slot 0 slotBound (by decide)

private theorem selected_operationIsWrite_bounded (slot : Nat)
    (slotBound : slot < wasm42x6.operationSlots) :
    LinearCombination.Bounded selectedLimit
      (operationIsWrite wasm42x6 slot) := by
  exact selected_operation_bit_bounded slot 1 slotBound (by decide)

private theorem selected_operationRam_bounded (slot : Nat)
    (slotBound : slot < wasm42x6.operationSlots) :
    LinearCombination.Bounded selectedLimit (operationRam wasm42x6 slot) := by
  exact selected_operation_bit_bounded slot 2 slotBound (by decide)

private theorem selected_operationAddress_bounded (slot : Nat)
    (slotBound : slot < wasm42x6.operationSlots) :
    LinearCombination.Bounded selectedLimit
      (operationAddress wasm42x6 slot) := by
  exact selected_operation_word_bounded slot 3 10 slotBound (by decide)

private theorem selected_operationReadValue_bounded (slot : Nat)
    (slotBound : slot < wasm42x6.operationSlots) :
    LinearCombination.Bounded selectedLimit
      (operationReadValue wasm42x6 slot) := by
  exact selected_operation_word_bounded slot 13 32 slotBound (by decide)

private theorem selected_operationWriteValue_bounded (slot : Nat)
    (slotBound : slot < wasm42x6.operationSlots) :
    LinearCombination.Bounded selectedLimit
      (operationWriteValue wasm42x6 slot) := by
  exact selected_operation_word_bounded slot 45 32 slotBound (by decide)

private theorem selected_operationReadTimestamp_bounded (slot : Nat)
    (slotBound : slot < wasm42x6.operationSlots) :
    LinearCombination.Bounded selectedLimit
      (operationReadTimestamp wasm42x6 slot) := by
  exact selected_operation_word_bounded slot 77 44 slotBound (by decide)

private theorem selected_operationDiffWord_bounded (slot : Nat)
    (slotBound : slot < wasm42x6.operationSlots) :
    LinearCombination.Bounded selectedLimit
      (operationDiffWord wasm42x6 slot) := by
  exact selected_operation_aux_word_bounded slot 0 44 slotBound (by decide)

private theorem selected_operationCountWord_bounded (slot : Nat)
    (slotBound : slot < wasm42x6.operationSlots) :
    LinearCombination.Bounded selectedLimit
      (operationCountWord wasm42x6 slot) := by
  exact selected_operation_aux_word_bounded slot 44 1 slotBound (by decide)

private theorem selected_operationReadProductWord_bounded
    (slot component : Nat) (slotBound : slot < wasm42x6.operationSlots)
    (componentBound : component < 2) :
    LinearCombination.Bounded selectedLimit
      (operationReadProductWord wasm42x6 slot component) := by
  have widthBound : 45 + component * 64 + 64 ≤
      wasm42x6.operationAuxiliaryBits := by
    change 45 + component * 64 + 64 ≤ 301
    omega
  simpa [operationReadProductWord, Params.operationReadProduct,
    Params.operationCount, Params.operationDiff,
    extensionLimbBits, timestampBits, wasm42x6_countBits, Nat.add_assoc] using
      selected_operation_aux_word_bounded slot (45 + component * 64) 64
        slotBound widthBound

private theorem selected_operationWriteProductWord_bounded
    (slot component : Nat) (slotBound : slot < wasm42x6.operationSlots)
    (componentBound : component < 2) :
    LinearCombination.Bounded selectedLimit
      (operationWriteProductWord wasm42x6 slot component) := by
  have widthBound : 173 + component * 64 + 64 ≤
      wasm42x6.operationAuxiliaryBits := by
    change 173 + component * 64 + 64 ≤ 301
    omega
  simpa [operationWriteProductWord, Params.operationWriteProduct,
    Params.operationReadProduct, Params.operationCount, Params.operationDiff,
    extensionBits, extensionLimbBits, timestampBits,
    wasm42x6_countBits, Nat.add_assoc] using
      selected_operation_aux_word_bounded slot (173 + component * 64) 64
        slotBound widthBound

private theorem selected_timestampIn_bounded :
    LinearCombination.Bounded selectedLimit timestampIn := by
  apply selected_publicWord_bounded
  decide

private theorem selected_operationWriteTimestamp_bounded (slot : Nat)
    (slotBound : slot < wasm42x6.operationSlots) :
    LinearCombination.Bounded selectedLimit
      (operationWriteTimestamp wasm42x6 slot) :=
  LinearCombination.bounded_add selected_timestampIn_bounded
    (selected_operationCountWord_bounded slot slotBound)

private theorem selected_operationGlobalIndex_bounded (slot : Nat)
    (slotBound : slot < wasm42x6.operationSlots) :
    LinearCombination.Bounded selectedLimit
      (operationGlobalIndex wasm42x6 slot) :=
  LinearCombination.bounded_add
    (selected_operationAddress_bounded slot slotBound)
    (LinearCombination.bounded_scale
      (selected_operationRam_bounded slot slotBound)
      (fieldOfNat wasm42x6.romCells))

private theorem selected_operationFingerprintPrefix_bounded
    (slot : Nat) (write : Bool)
    (slotBound : slot < wasm42x6.operationSlots) :
    LinearCombination.Bounded selectedLimit
      (operationFingerprintPrefix wasm42x6 slot write) := by
  apply LinearCombination.bounded_sub
  · apply LinearCombination.bounded_sub
    · exact selected_gammaWord_bounded 1 0 (by decide) (by decide)
    · cases write
      · exact selected_operationReadTimestamp_bounded slot slotBound
      · exact selected_operationWriteTimestamp_bounded slot slotBound
  · exact LinearCombination.bounded_scale
      (selected_operationGlobalIndex_bounded slot slotBound)
      (Rows.LinearCombination.fieldTwoPower timestampBits)

private theorem selected_previousOperationProduct_bounded
    (slot product component : Nat)
    (slotBound : slot < wasm42x6.operationSlots)
    (productBound : product < 2) (componentBound : component < 2) :
    LinearCombination.Bounded selectedLimit
      (previousOperationProduct wasm42x6 slot product component) := by
  have slotZero := selected_operation_slot_zero slot slotBound
  subst slot
  simp only [previousOperationProduct, ↓reduceIte]
  exact selected_productInputWord_bounded product component
    (Nat.lt_trans productBound (by decide)) componentBound

private theorem selected_operationProductRows_bounded
    (slot : Nat) (write : Bool)
    (slotBound : slot < wasm42x6.operationSlots) :
    RowsBounded selectedLimit
      (operationProductRows wasm42x6 slot write) := by
  have read0 := selected_operationReadProductWord_bounded slot 0 slotBound
    (by decide)
  have read1 := selected_operationReadProductWord_bounded slot 1 slotBound
    (by decide)
  have write0 := selected_operationWriteProductWord_bounded slot 0 slotBound
    (by decide)
  have write1 := selected_operationWriteProductWord_bounded slot 1 slotBound
    (by decide)
  have previousRead0 := selected_previousOperationProduct_bounded slot 0 0
    slotBound (by decide) (by decide)
  have previousRead1 := selected_previousOperationProduct_bounded slot 0 1
    slotBound (by decide) (by decide)
  have previousWrite0 := selected_previousOperationProduct_bounded slot 1 0
    slotBound (by decide) (by decide)
  have previousWrite1 := selected_previousOperationProduct_bounded slot 1 1
    slotBound (by decide) (by decide)
  have padBound := selected_operationPad_bounded slot slotBound
  have activeBound := LinearCombination.bounded_sub selected_one_bounded padBound
  have fingerprint0Bound :=
    selected_operationFingerprintPrefix_bounded slot write slotBound
  have fingerprint1Bound := selected_gammaWord_bounded 1 1
    (by decide) (by decide)
  have value0Bound := selected_gammaWord_bounded 0 0
    (by decide) (by decide)
  have value1Bound := selected_gammaWord_bounded 0 1
    (by decide) (by decide)
  have readValueBound := selected_operationReadValue_bounded slot slotBound
  have writeValueBound := selected_operationWriteValue_bounded slot slotBound
  cases write
  · simpa [operationProductRows] using
      selected_extensionRows_bounded .readProduct slot
        (operationReadProductWord wasm42x6 slot 0)
        (operationReadProductWord wasm42x6 slot 1)
        (previousOperationProduct wasm42x6 slot 0 0)
        (previousOperationProduct wasm42x6 slot 0 1)
        (operationPad wasm42x6 slot)
        (Rows.LinearCombination.sub one (operationPad wasm42x6 slot))
        (operationFingerprintPrefix wasm42x6 slot false)
        (gammaWord 1 1) (gammaWord 0 0) (gammaWord 0 1)
        (operationReadValue wasm42x6 slot)
        read0 read1 previousRead0 previousRead1 padBound activeBound
        fingerprint0Bound fingerprint1Bound value0Bound value1Bound
        readValueBound
  · simpa [operationProductRows] using
      selected_extensionRows_bounded .writeProduct slot
        (operationWriteProductWord wasm42x6 slot 0)
        (operationWriteProductWord wasm42x6 slot 1)
        (previousOperationProduct wasm42x6 slot 1 0)
        (previousOperationProduct wasm42x6 slot 1 1)
        (operationPad wasm42x6 slot)
        (Rows.LinearCombination.sub one (operationPad wasm42x6 slot))
        (operationFingerprintPrefix wasm42x6 slot true)
        (gammaWord 1 1) (gammaWord 0 0) (gammaWord 0 1)
        (operationWriteValue wasm42x6 slot)
        write0 write1 previousWrite0 previousWrite1 padBound activeBound
        fingerprint0Bound fingerprint1Bound value0Bound value1Bound
        writeValueBound

private theorem selected_operationLaneBitRows_bounded
    (slot : Nat) (slotBound : slot < wasm42x6.operationSlots) :
    RowsBounded selectedLimit
      (operationLaneBitRows wasm42x6 slot) := by
  have slotZero := selected_operation_slot_zero slot slotBound
  subst slot
  have operationBound := selected_bitRows_bounded
    .operationBit 0 0 (wasm42x6.operationSlot 0)
      wasm42x6.operationBits (by
    change 1404 + 121 ≤ 419747
    decide)
  have diffBound := selected_bitRows_bounded
    .operationBit 0 wasm42x6.operationBits
      (wasm42x6.operationDiff 0) timestampBits (by
    change 157302 + 44 ≤ 419747
    decide)
  have countBound := selected_bitRows_bounded
    .operationBit 0 (wasm42x6.operationBits + timestampBits)
      (wasm42x6.operationCount 0) wasm42x6.countBits (by
    change 157346 + 1 ≤ 419747
    decide)
  have readBound := selected_bitRows_bounded
    .operationBit 0
      (wasm42x6.operationBits + timestampBits + wasm42x6.countBits)
      (wasm42x6.operationReadProduct 0) extensionBits (by
    change 157347 + 128 ≤ 419747
    decide)
  have writeBound := selected_bitRows_bounded
    .operationBit 0
      (wasm42x6.operationBits + timestampBits + wasm42x6.countBits +
        extensionBits)
      (wasm42x6.operationWriteProduct 0) extensionBits (by
    change 157475 + 128 ≤ 419747
    decide)
  simpa [operationLaneBitRows] using
    rowsBounded_append
      (rowsBounded_append
        (rowsBounded_append
          (rowsBounded_append operationBound diffBound) countBound)
        readBound)
      writeBound

private theorem selected_operationCoreRows_bounded
    (slot : Nat) (slotBound : slot < wasm42x6.operationSlots) :
    RowsBounded selectedLimit (operationCoreRows wasm42x6 slot) := by
  have slotZero := selected_operation_slot_zero slot slotBound
  subst slot
  have padBound := selected_operationPad_bounded 0 (by decide)
  have writeBound := selected_operationIsWrite_bounded 0 (by decide)
  have ramBound := selected_operationRam_bounded 0 (by decide)
  have addressBound := selected_operationAddress_bounded 0 (by decide)
  have readValueBound := selected_operationReadValue_bounded 0 (by decide)
  have writeValueBound := selected_operationWriteValue_bounded 0 (by decide)
  have readTimestampBound :=
    selected_operationReadTimestamp_bounded 0 (by decide)
  have countBound := selected_operationCountWord_bounded 0 (by decide)
  have diffBound := selected_operationDiffWord_bounded 0 (by decide)
  have notPadBound := LinearCombination.bounded_sub selected_one_bounded padBound
  have romBound := LinearCombination.bounded_sub selected_one_bounded ramBound
  have timestampBound :=
    selected_operationWriteTimestamp_bounded 0 (by decide)
  have fixedBound : RowsBounded selectedLimit
      [ linearRow (id .operationCount 0 0 0)
          (operationCountWord wasm42x6 0)
          (Rows.LinearCombination.add Rows.LinearCombination.zero
            (Rows.LinearCombination.sub one (operationPad wasm42x6 0)))
      , productRow (id .readWrite 0 0 0)
          (Rows.LinearCombination.sub one (operationIsWrite wasm42x6 0))
          (Rows.LinearCombination.sub (operationWriteValue wasm42x6 0)
            (operationReadValue wasm42x6 0))
      , productRow (id .timestampOrder 0 0 0)
          (Rows.LinearCombination.sub one (operationPad wasm42x6 0))
          (Rows.LinearCombination.sub
            (Rows.LinearCombination.sub
              (Rows.LinearCombination.sub
                (operationWriteTimestamp wasm42x6 0)
                (operationReadTimestamp wasm42x6 0)) one)
            (operationDiffWord wasm42x6 0))
      , productRow (id .romWrite 0 0 0)
          (operationIsWrite wasm42x6 0)
          (Rows.LinearCombination.sub one (operationRam wasm42x6 0))
      ] := by
    apply rowsBounded_cons
    · exact Row.bounded_linearRow _ countBound
        (LinearCombination.bounded_add
          (LinearCombination.bounded_zero selectedLimit) notPadBound)
    apply rowsBounded_cons
    · exact Row.bounded_productRow _
        (LinearCombination.bounded_sub selected_one_bounded writeBound)
        (LinearCombination.bounded_sub writeValueBound readValueBound)
    apply rowsBounded_cons
    · exact Row.bounded_productRow _ notPadBound
        (LinearCombination.bounded_sub
          (LinearCombination.bounded_sub
            (LinearCombination.bounded_sub timestampBound readTimestampBound)
            selected_one_bounded) diffBound)
    apply rowsBounded_cons
    · exact Row.bounded_productRow _ writeBound romBound
    · exact rowsBounded_nil selectedLimit
  have paddingBound : RowsBounded selectedLimit
      [ productRow (id .padding 0 0 0) (operationPad wasm42x6 0)
          (operationIsWrite wasm42x6 0)
      , productRow (id .padding 0 0 1) (operationPad wasm42x6 0)
          (operationRam wasm42x6 0)
      , productRow (id .padding 0 0 2) (operationPad wasm42x6 0)
          (operationAddress wasm42x6 0)
      , productRow (id .padding 0 0 3) (operationPad wasm42x6 0)
          (operationReadValue wasm42x6 0)
      , productRow (id .padding 0 0 4) (operationPad wasm42x6 0)
          (operationWriteValue wasm42x6 0)
      , productRow (id .padding 0 0 5) (operationPad wasm42x6 0)
          (operationReadTimestamp wasm42x6 0)
      ] := by
    apply rowsBounded_cons (Row.bounded_productRow _ padBound writeBound)
    apply rowsBounded_cons (Row.bounded_productRow _ padBound ramBound)
    apply rowsBounded_cons (Row.bounded_productRow _ padBound addressBound)
    apply rowsBounded_cons (Row.bounded_productRow _ padBound readValueBound)
    apply rowsBounded_cons (Row.bounded_productRow _ padBound writeValueBound)
    apply rowsBounded_cons
      (Row.bounded_productRow _ padBound readTimestampBound)
    exact rowsBounded_nil selectedLimit
  have readProductBound :=
    selected_operationProductRows_bounded 0 false (by decide)
  have writeProductBound :=
    selected_operationProductRows_bounded 0 true (by decide)
  simpa [operationCoreRows] using
    rowsBounded_append fixedBound
      (rowsBounded_append (rowsBounded_nil selectedLimit)
        (rowsBounded_append paddingBound
          (rowsBounded_append readProductBound writeProductBound)))

private theorem selected_operationRows_bounded
    (slot : Nat) (slotBound : slot < wasm42x6.operationSlots) :
    RowsBounded selectedLimit (operationRows wasm42x6 slot) :=
  rowsBounded_append
    (selected_operationLaneBitRows_bounded slot slotBound)
    (selected_operationCoreRows_bounded slot slotBound)

private theorem selected_scanValue_bounded
    (final : Bool) (slot : Nat) (slotBound : slot < wasm42x6.scanSlots) :
    LinearCombination.Bounded selectedLimit (scanValue wasm42x6 final slot) :=
  selected_scan_lane_word_bounded final slot 0 valueBits slotBound (by
    decide)

private theorem selected_scanTimestamp_bounded
    (final : Bool) (slot : Nat) (slotBound : slot < wasm42x6.scanSlots) :
    LinearCombination.Bounded selectedLimit
      (scanTimestamp wasm42x6 final slot) :=
  selected_scan_lane_word_bounded final slot valueBits timestampBits
    slotBound (by decide)

private theorem selected_scanProductWord_bounded
    (final : Bool) (slot component : Nat)
    (slotBound : slot < wasm42x6.scanSlots)
    (componentBound : component < 2) :
    LinearCombination.Bounded selectedLimit
      (scanProductWord wasm42x6 final slot component) :=
  selected_scan_aux_word_bounded final slot component slotBound componentBound

private theorem selected_previousScanProduct_bounded
    (final : Bool) (slot component : Nat)
    (slotBound : slot < wasm42x6.scanSlots)
    (componentBound : component < 2) :
    LinearCombination.Bounded selectedLimit
      (previousScanProduct wasm42x6 final slot component) := by
  by_cases first : slot = 0
  · subst slot
    simp only [previousScanProduct, ↓reduceIte]
    cases final
    · exact selected_productInputWord_bounded 2 component
        (by decide) componentBound
    · exact selected_productInputWord_bounded 3 component
        (by decide) componentBound
  · have previousBound : slot - 1 < wasm42x6.scanSlots := by omega
    simp only [previousScanProduct, first, ↓reduceIte]
    exact selected_scanProductWord_bounded final (slot - 1) component
      previousBound componentBound

private theorem selected_scanGlobalIndex_bounded
    (slot : Nat) (_slotBound : slot < wasm42x6.scanSlots) :
    LinearCombination.Bounded selectedLimit
      (scanGlobalIndex wasm42x6 slot) := by
  apply LinearCombination.bounded_add
  · exact LinearCombination.bounded_scale
      (selected_publicWord_bounded XOffset.step stepIndexBits (by decide))
      (fieldOfNat wasm42x6.scanSlots)
  · exact LinearCombination.bounded_constant selectedLimit_positive
      (fieldOfNat slot)

private theorem selected_scanRowsForLane_bounded
    (final : Bool) (slot : Nat) (slotBound : slot < wasm42x6.scanSlots) :
    RowsBounded selectedLimit (scanRowsForLane wasm42x6 final slot) := by
  have laneBitsBound : RowsBounded selectedLimit
      (bitRows (if final then .finalScanBit else .initialScanBit) slot 0
        (scanCellStart wasm42x6 final slot) cellBits) := by
    apply selected_bitRows_bounded
    cases final with
    | false =>
      change 1566 + slot * 76 + 76 ≤ 419747
      change slot < 1024 at slotBound
      omega
    | true =>
      change 79434 + slot * 76 + 76 ≤ 419747
      change slot < 1024 at slotBound
      omega
  have productBitsBound : RowsBounded selectedLimit
      (bitRows (if final then .finalScanBit else .initialScanBit) slot cellBits
        (scanProductStart wasm42x6 final slot) extensionBits) := by
    apply selected_bitRows_bounded
    cases final with
    | false =>
      change 157603 + slot * 256 + 128 ≤ 419747
      change slot < 1024 at slotBound
      omega
    | true =>
      simp only [scanProductStart, ↓reduceIte, Params.finalScanProduct,
        Params.initialScanProduct, scanAuxiliaryBits, extensionBits]
      rw [wasm42x6_scanAuxiliaryStart]
      change 157603 + slot * 256 + 128 + 128 ≤ 419747
      change slot < 1024 at slotBound
      omega
  have output0Bound := selected_scanProductWord_bounded final slot 0
    slotBound (by decide)
  have output1Bound := selected_scanProductWord_bounded final slot 1
    slotBound (by decide)
  have previous0Bound := selected_previousScanProduct_bounded final slot 0
    slotBound (by decide)
  have previous1Bound := selected_previousScanProduct_bounded final slot 1
    slotBound (by decide)
  have fingerprint0Bound := LinearCombination.bounded_sub
    (LinearCombination.bounded_sub
      (selected_gammaWord_bounded 1 0 (by decide) (by decide))
      (selected_scanTimestamp_bounded final slot slotBound))
    (LinearCombination.bounded_scale
      (selected_scanGlobalIndex_bounded slot slotBound)
      (Rows.LinearCombination.fieldTwoPower timestampBits))
  have extensionBound := selected_extensionRows_bounded
    (if final then .finalScanProduct else .initialScanProduct) slot
    (scanProductWord wasm42x6 final slot 0)
    (scanProductWord wasm42x6 final slot 1)
    (previousScanProduct wasm42x6 final slot 0)
    (previousScanProduct wasm42x6 final slot 1)
    Rows.LinearCombination.zero one
    (Rows.LinearCombination.sub
      (Rows.LinearCombination.sub (gammaWord 1 0)
        (scanTimestamp wasm42x6 final slot))
      (Rows.LinearCombination.scale
        (Rows.LinearCombination.fieldTwoPower timestampBits)
        (scanGlobalIndex wasm42x6 slot)))
    (gammaWord 1 1) (gammaWord 0 0) (gammaWord 0 1)
    (scanValue wasm42x6 final slot)
    output0Bound output1Bound previous0Bound previous1Bound
    (LinearCombination.bounded_zero selectedLimit) selected_one_bounded
    fingerprint0Bound
    (selected_gammaWord_bounded 1 1 (by decide) (by decide))
    (selected_gammaWord_bounded 0 0 (by decide) (by decide))
    (selected_gammaWord_bounded 0 1 (by decide) (by decide))
    (selected_scanValue_bounded final slot slotBound)
  simpa [scanRowsForLane] using
    rowsBounded_append (rowsBounded_append laneBitsBound productBitsBound)
      extensionBound

private theorem selected_scanRows_bounded
    (slot : Nat) (slotBound : slot < wasm42x6.scanSlots) :
    RowsBounded selectedLimit (scanRows wasm42x6 slot) :=
  rowsBounded_append
    (selected_scanRowsForLane_bounded false slot slotBound)
    (selected_scanRowsForLane_bounded true slot slotBound)

private theorem selected_fillerColumn_bounded
    (column : Nat) (member : column ∈ wasm42x6.fillerColumns) :
    column < selectedLimit := by
  unfold Params.fillerColumns at member
  rw [List.mem_append] at member
  rcases member with firstThree | fourth
  · rw [List.mem_append] at firstThree
    rcases firstThree with firstTwo | third
    · rw [List.mem_append] at firstTwo
      rcases firstTwo with first | second
      · rcases List.mem_range'.mp first with ⟨offset, offsetBound, columnExact⟩
        simp only [wasm42x6_publicColumns, wasm42x6_operationLane] at offsetBound
        simp only [wasm42x6_publicColumns] at columnExact
        rw [show selectedLimit = 419747 from wasm42x6_columnCount]
        omega
      · rcases List.mem_range'.mp second with ⟨offset, offsetBound, columnExact⟩
        simp only [wasm42x6_operationLane, wasm42x6_initialScanLane,
          wasm42x6_operationBits, wasm42x6_operationSlots] at offsetBound
        simp only [wasm42x6_operationLane, wasm42x6_operationBits,
          wasm42x6_operationSlots] at columnExact
        rw [show selectedLimit = 419747 from wasm42x6_columnCount]
        omega
    · rcases List.mem_range'.mp third with ⟨offset, offsetBound, columnExact⟩
      simp only [wasm42x6_initialScanLane, wasm42x6_finalScanLane,
        wasm42x6_scanSlots, cellBits_exact] at offsetBound
      simp only [wasm42x6_initialScanLane, wasm42x6_scanSlots,
        cellBits_exact] at columnExact
      rw [show selectedLimit = 419747 from wasm42x6_columnCount]
      omega
  · rcases List.mem_range'.mp fourth with ⟨offset, offsetBound, columnExact⟩
    simp only [wasm42x6_finalScanLane, wasm42x6_auxiliaryStart,
      wasm42x6_scanSlots, cellBits_exact] at offsetBound
    simp only [wasm42x6_finalScanLane, wasm42x6_scanSlots,
      cellBits_exact] at columnExact
    rw [show selectedLimit = 419747 from wasm42x6_columnCount]
    omega

private theorem selected_fillerRows_bounded :
    RowsBounded selectedLimit (fillerRows wasm42x6) := by
  intro row member
  rcases List.mem_map.1 member with ⟨column, columnMember, rfl⟩
  exact Row.bounded_linearRow _
    (LinearCombination.bounded_bit
      (selected_fillerColumn_bounded column columnMember))
    (LinearCombination.bounded_zero selectedLimit)

private theorem selected_boundaryRows_bounded :
    RowsBounded selectedLimit (Compiler.boundaryRows wasm42x6) := by
  have timestampBound : RowBounded selectedLimit
      (linearRow (id .boundaryTimestamp 0 0 0)
        (publicWord XOffset.timestampOut timestampBits)
        (Rows.LinearCombination.add timestampIn
          (operationCountWord wasm42x6 (lastOperationSlot wasm42x6)))) := by
    apply Row.bounded_linearRow
    · exact selected_publicWord_bounded XOffset.timestampOut timestampBits
        (by decide)
    · exact LinearCombination.bounded_add selected_timestampIn_bounded
        (selected_operationCountWord_bounded 0 (by decide))
  intro row member
  simp only [Compiler.boundaryRows, List.mem_cons] at member
  rcases member with rfl | productMember
  · exact timestampBound
  · rcases List.mem_flatMap.1 productMember with
      ⟨product, productMember, componentMember⟩
    have productBound : product < 4 := List.mem_range.1 productMember
    rcases List.mem_map.1 componentMember with
      ⟨component, componentInRange, rowExact⟩
    subst row
    have componentBound : component < 2 :=
      List.mem_range.1 componentInRange
    apply Row.bounded_linearRow
    · exact selected_productOutputWord_bounded product component
        productBound componentBound
    · have productCases :
          product = 0 ∨ product = 1 ∨ product = 2 ∨ product = 3 := by
        omega
      rcases productCases with rfl | rfl | rfl | rfl
      · simp only [lastOperationSlot, ↓reduceIte]
        exact selected_operationReadProductWord_bounded 0 component
          (by decide) componentBound
      · simp only [lastOperationSlot, ↓reduceIte]
        exact selected_operationWriteProductWord_bounded 0 component
          (by decide) componentBound
      · simp only [lastOperationSlot, lastScanSlot, ↓reduceIte]
        exact selected_scanProductWord_bounded false 1023 component
          (by decide) componentBound
      · simp only [lastOperationSlot, lastScanSlot]
        exact selected_scanProductWord_bounded true 1023 component
          (by decide) componentBound

private theorem selected_flatMap_operationRows_bounded :
    RowsBounded selectedLimit
      ((List.range wasm42x6.operationSlots).flatMap
        (operationRows wasm42x6)) := by
  intro row member
  rcases List.mem_flatMap.1 member with
    ⟨slot, slotMember, rowMember⟩
  exact selected_operationRows_bounded slot
    (List.mem_range.1 slotMember) row rowMember

private theorem selected_flatMap_scanRows_bounded :
    RowsBounded selectedLimit
      ((List.range wasm42x6.scanSlots).flatMap (scanRows wasm42x6)) := by
  intro row member
  rcases List.mem_flatMap.1 member with
    ⟨slot, slotMember, rowMember⟩
  exact selected_scanRows_bounded slot
    (List.mem_range.1 slotMember) row rowMember

theorem wasm42x6_rawRows_bounded :
    RowsBounded wasm42x6.columnCount (rawRows wasm42x6) := by
  exact rowsBounded_append
    (rowsBounded_append
      (rowsBounded_append selected_fillerRows_bounded
        selected_flatMap_operationRows_bounded)
      selected_flatMap_scanRows_bounded)
    selected_boundaryRows_bounded

/-- Every sparse term in the emitted selected program is inside the exact
Lean-owned column allocation. -/
theorem wasm42x6_rows_bounded :
    RowsBounded wasm42x6.columnCount (rows wasm42x6) :=
  rowsBounded_map_withPosition wasm42x6_rawRows_bounded

def allocatedColumns (params : Params) : List Nat :=
  List.range params.columnCount

theorem allocatedColumns_nodup (params : Params) :
    (allocatedColumns params).Nodup :=
  List.nodup_range

theorem wasm42x6_every_term_allocated
    (row : Row) (rowMember : row ∈ rows wasm42x6)
    (role : StepPolynomial.Role) (term : Rows.Term)
    (termMember : term ∈ row.images.at role) :
    term.column ∈ allocatedColumns wasm42x6 := by
  rw [allocatedColumns, List.mem_range]
  exact wasm42x6_rows_bounded row rowMember role term termMember

theorem wasm42x6_allocatedColumns_length :
    (allocatedColumns wasm42x6).length = 419747 := by
  simp [allocatedColumns, wasm42x6_columnCount]

theorem wasm42x6_publicColumnCount : wasm42x6.publicEnd = 1401 :=
  wasm42x6_publicColumns

theorem wasm42x6_witnessColumnCount :
    wasm42x6.columnCount - wasm42x6.publicEnd = 418346 :=
  wasm42x6_witnessColumns

end Nightstream.Implementation.Lowering.Nebula.Physical
