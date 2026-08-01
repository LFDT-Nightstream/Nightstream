import Nightstream.Implementation.Lowering.Nebula.ProductSemantics

/-!
Declarative semantics for every row emitted by the stackless Nebula compiler.

Assurance tier: model-level.

Owns: the non-product E1--E7 checks, scan bitness, filler-zero rules,
boundary links, and their composition with the four product recurrences.

Does not own: Nat decoding, WASM port binding, an honest witness constructor,
segment composition, terminal balance, Rust, or a collision probability bound.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.Nebula.ConstraintSemantics

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Nebula
open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.Rows
open Nightstream.Implementation.Lowering.Nebula.Compiler
open Nightstream.Implementation.Lowering.Nebula.ProductSemantics
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier

private abbrev Lin := Rows.LinearCombination

def IsBit (assignment : Nat → F) (column : Nat) : Prop :=
  assignment column * assignment column + -assignment column = 0

def LinearEqual (assignment : Nat → F) (left right : Lin) : Prop :=
  Rows.LinearCombination.eval assignment left =
    Rows.LinearCombination.eval assignment right

def ProductZero (assignment : Nat → F) (left right : Lin) : Prop :=
  Rows.LinearCombination.eval assignment left *
    Rows.LinearCombination.eval assignment right = 0

structure OperationAccepted
    (assignment : Nat → F) (params : Params) (slot : Nat) : Prop where
  laneBits : ∀ offset, offset < params.operationBits →
    IsBit assignment (params.operationSlot slot + offset)
  diffBits : ∀ offset, offset < timestampBits →
    IsBit assignment (params.operationDiff slot + offset)
  countBits : ∀ offset, offset < params.countBits →
    IsBit assignment (params.operationCount slot + offset)
  readProductBits : ∀ offset, offset < extensionBits →
    IsBit assignment (params.operationReadProduct slot + offset)
  writeProductBits : ∀ offset, offset < extensionBits →
    IsBit assignment (params.operationWriteProduct slot + offset)
  countExact :
    LinearEqual assignment (operationCountWord params slot)
      (Rows.LinearCombination.add
        (if slot = 0 then Rows.LinearCombination.zero
          else operationCountWord params (slot - 1))
        (Rows.LinearCombination.sub one (operationPad params slot)))
  readWritesBack :
    ProductZero assignment
      (Rows.LinearCombination.sub one (operationIsWrite params slot))
      (Rows.LinearCombination.sub (operationWriteValue params slot)
        (operationReadValue params slot))
  timestampOrdered :
    ProductZero assignment
      (Rows.LinearCombination.sub one (operationPad params slot))
      (Rows.LinearCombination.sub
        (Rows.LinearCombination.sub
          (Rows.LinearCombination.sub
            (operationWriteTimestamp params slot)
            (operationReadTimestamp params slot)) one)
        (operationDiffWord params slot))
  romReadOnly :
    ProductZero assignment (operationIsWrite params slot)
      (Rows.LinearCombination.sub one (operationRam params slot))
  romAddressRange : ∀ offset,
    offset < params.addressBits - params.r →
      ProductZero assignment
        (Rows.LinearCombination.sub one (operationRam params slot))
        (Rows.LinearCombination.bit
          (params.operationSlot slot + 3 + params.r + offset))
  padIsWriteZero :
    ProductZero assignment (operationPad params slot)
      (operationIsWrite params slot)
  padRamZero :
    ProductZero assignment (operationPad params slot)
      (operationRam params slot)
  padAddressZero :
    ProductZero assignment (operationPad params slot)
      (operationAddress params slot)
  padReadValueZero :
    ProductZero assignment (operationPad params slot)
      (operationReadValue params slot)
  padWriteValueZero :
    ProductZero assignment (operationPad params slot)
      (operationWriteValue params slot)
  padReadTimestampZero :
    ProductZero assignment (operationPad params slot)
      (operationReadTimestamp params slot)
  readProductExact :
    operationProduct assignment params slot false =
      K.mul (previousOperationProductValue assignment params slot false)
        (operationGate assignment params slot false)
  writeProductExact :
    operationProduct assignment params slot true =
      K.mul (previousOperationProductValue assignment params slot true)
        (operationGate assignment params slot true)

structure ScanAccepted
    (assignment : Nat → F) (params : Params) (slot : Nat) : Prop where
  initialCellBits : ∀ offset, offset < cellBits →
    IsBit assignment (scanCellStart params false slot + offset)
  initialProductBits : ∀ offset, offset < extensionBits →
    IsBit assignment (params.initialScanProduct slot + offset)
  finalCellBits : ∀ offset, offset < cellBits →
    IsBit assignment (scanCellStart params true slot + offset)
  finalProductBits : ∀ offset, offset < extensionBits →
    IsBit assignment (params.finalScanProduct slot + offset)
  initialProductExact :
    scanProduct assignment params false slot =
      K.mul (previousScanProductValue assignment params false slot)
        (scanFactor assignment params false slot)
  finalProductExact :
    scanProduct assignment params true slot =
      K.mul (previousScanProductValue assignment params true slot)
        (scanFactor assignment params true slot)

structure BoundaryAccepted
    (assignment : Nat → F) (params : Params) : Prop where
  timestampExact :
    LinearEqual assignment
      (publicWord XOffset.timestampOut timestampBits)
      (Rows.LinearCombination.add timestampIn
        (operationCountWord params (lastOperationSlot params)))
  product0 : outputProduct assignment 0 =
    boundaryProductValue assignment params 0
  product1 : outputProduct assignment 1 =
    boundaryProductValue assignment params 1
  product2 : outputProduct assignment 2 =
    boundaryProductValue assignment params 2
  product3 : outputProduct assignment 3 =
    boundaryProductValue assignment params 3

structure Accepted (assignment : Nat → F) (params : Params) : Prop where
  constantWire : assignment 0 = 1
  fillerZero : ∀ column, column ∈ params.fillerColumns →
    assignment column = 0
  operations : ∀ slot, slot < params.operationSlots →
    OperationAccepted assignment params slot
  scans : ∀ slot, slot < params.scanSlots →
    ScanAccepted assignment params slot
  boundary : BoundaryAccepted assignment params

private theorem linear_sound
    (identifier : RowId) (left right : Lin) (assignment : Nat → F)
    (holds : (linearRow identifier left right).Holds assignment) :
    LinearEqual assignment left right := by
  rw [linearRow_holds_iff] at holds
  exact Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp
    (by simpa only [Fin.sub_eq_add_neg] using holds)

private theorem product_sound
    (identifier : RowId) (left right : Lin) (assignment : Nat → F)
    (holds : (productRow identifier left right).Holds assignment) :
    ProductZero assignment left right := by
  exact (productRow_holds_iff identifier left right assignment).mp holds

private theorem linear_honest
    (identifier : RowId) (left right : Lin) (assignment : Nat → F)
    (equal : LinearEqual assignment left right) :
    (linearRow identifier left right).Holds assignment := by
  rw [linearRow_holds_iff]
  unfold LinearEqual at equal
  rw [equal, ← Fin.sub_eq_add_neg]
  exact Fin.sub_self

private theorem product_honest
    (identifier : RowId) (left right : Lin) (assignment : Nat → F)
    (zero : ProductZero assignment left right) :
    (productRow identifier left right).Holds assignment := by
  exact (productRow_holds_iff identifier left right assignment).mpr zero

private theorem satisfies_append_iff
    (left right : List Row) (assignment : Nat → F) :
    Satisfies (left ++ right) assignment ↔
      Satisfies left assignment ∧ Satisfies right assignment := by
  constructor
  · intro satisfied
    constructor
    · intro row member
      exact satisfied row (List.mem_append_left right member)
    · intro row member
      exact satisfied row (List.mem_append_right left member)
  · rintro ⟨leftSatisfied, rightSatisfied⟩ row member
    rcases List.mem_append.mp member with inLeft | inRight
    · exact leftSatisfied row inLeft
    · exact rightSatisfied row inRight

private theorem satisfies_append
    {left right : List Row} {assignment : Nat → F}
    (leftSatisfied : Satisfies left assignment)
    (rightSatisfied : Satisfies right assignment) :
    Satisfies (left ++ right) assignment :=
  (satisfies_append_iff left right assignment).mpr
    ⟨leftSatisfied, rightSatisfied⟩

private theorem bitRows_sound
    (family : Family) (slot ordinalBase start width : Nat)
    (assignment : Nat → F)
    (satisfied : Satisfies (bitRows family slot ordinalBase start width)
      assignment) :
    ∀ offset, offset < width → IsBit assignment (start + offset) := by
  intro offset offsetBound
  exact (bitRow_holds_iff
      (id family slot 0 (ordinalBase + offset)) (start + offset)
      assignment).mp
    (satisfied _ (by
      exact List.mem_map.mpr
        ⟨offset, List.mem_range.mpr offsetBound, rfl⟩))

private theorem bitRows_honest
    (family : Family) (slot ordinalBase start width : Nat)
    (assignment : Nat → F)
    (bits : ∀ offset, offset < width →
      IsBit assignment (start + offset)) :
    Satisfies (bitRows family slot ordinalBase start width) assignment := by
  intro row member
  rcases List.mem_map.mp member with ⟨offset, offsetMember, rfl⟩
  exact (bitRow_holds_iff
    (id family slot 0 (ordinalBase + offset)) (start + offset)
    assignment).mpr (bits offset (List.mem_range.mp offsetMember))

private theorem operationLaneBits_satisfied
    (assignment : Nat → F) (params : Params) (slot : Nat)
    (satisfied : Satisfies (operationRows params slot) assignment) :
    Satisfies (operationLaneBitRows params slot) assignment := by
  intro row member
  exact satisfied row (List.mem_append_left _ member)

private theorem operationCoreSatisfied
    (assignment : Nat → F) (params : Params) (slot : Nat)
    (satisfied : Satisfies (operationRows params slot) assignment) :
    Satisfies (operationCoreRows params slot) assignment := by
  intro row member
  exact satisfied row (List.mem_append_right _ member)

private theorem operationBitGroups
    (assignment : Nat → F) (params : Params) (slot : Nat)
    (satisfied : Satisfies (operationRows params slot) assignment) :
    (∀ offset, offset < params.operationBits →
      IsBit assignment (params.operationSlot slot + offset)) ∧
    (∀ offset, offset < timestampBits →
      IsBit assignment (params.operationDiff slot + offset)) ∧
    (∀ offset, offset < params.countBits →
      IsBit assignment (params.operationCount slot + offset)) ∧
    (∀ offset, offset < extensionBits →
      IsBit assignment (params.operationReadProduct slot + offset)) ∧
    (∀ offset, offset < extensionBits →
      IsBit assignment (params.operationWriteProduct slot + offset)) := by
  have laneSatisfied := operationLaneBits_satisfied assignment params slot
    satisfied
  unfold operationLaneBitRows at laneSatisfied
  repeat' apply And.intro
  · apply bitRows_sound .operationBit slot 0 _ _ assignment
    intro row member
    exact laneSatisfied row (by simp [member])
  · apply bitRows_sound .operationBit slot params.operationBits _ _ assignment
    intro row member
    exact laneSatisfied row (by simp [member])
  · apply bitRows_sound .operationBit slot
      (params.operationBits + timestampBits) _ _ assignment
    intro row member
    exact laneSatisfied row (by simp [member])
  · apply bitRows_sound .operationBit slot
      (params.operationBits + timestampBits + params.countBits) _ _ assignment
    intro row member
    exact laneSatisfied row (by simp [member])
  · apply bitRows_sound .operationBit slot
      (params.operationBits + timestampBits + params.countBits +
        extensionBits) _ _ assignment
    intro row member
    exact laneSatisfied row (by simp [member])

private theorem coreRowMember
    (params : Params) (slot : Nat) (row : Row)
    (member : row ∈ operationCoreRows params slot) :
    row ∈ operationCoreRows params slot := member

theorem operationAccepted_of_rows
    (assignment : Nat → F) (params : Params)
    (allSatisfied : Satisfies (rows params) assignment)
    (slot : Nat) (slotBound : slot < params.operationSlots) :
    OperationAccepted assignment params slot := by
  have satisfied := operationRows_satisfied_of_rows assignment params
    allSatisfied slot slotBound
  have core := operationCoreSatisfied assignment params slot satisfied
  obtain ⟨laneBits, diffBits, countBits, readProductBits,
      writeProductBits⟩ := operationBitGroups assignment params slot satisfied
  let pad := operationPad params slot
  let isWrite := operationIsWrite params slot
  let ram := operationRam params slot
  let address := operationAddress params slot
  let readValue := operationReadValue params slot
  let writeValue := operationWriteValue params slot
  let readTimestamp := operationReadTimestamp params slot
  let count := operationCountWord params slot
  let previousCount := if slot = 0 then Rows.LinearCombination.zero
    else operationCountWord params (slot - 1)
  let notPad := Rows.LinearCombination.sub one pad
  let rom := Rows.LinearCombination.sub one ram
  let writeTimestamp := operationWriteTimestamp params slot
  refine {
    laneBits := laneBits
    diffBits := diffBits
    countBits := countBits
    readProductBits := readProductBits
    writeProductBits := writeProductBits
    countExact := ?_
    readWritesBack := ?_
    timestampOrdered := ?_
    romReadOnly := ?_
    romAddressRange := ?_
    padIsWriteZero := ?_
    padRamZero := ?_
    padAddressZero := ?_
    padReadValueZero := ?_
    padWriteValueZero := ?_
    padReadTimestampZero := ?_
    readProductExact := operationProduct_sound_of_rows assignment params
      allSatisfied slot slotBound false
    writeProductExact := operationProduct_sound_of_rows assignment params
      allSatisfied slot slotBound true }
  · apply linear_sound (id .operationCount slot 0 0) _ _ assignment
    apply core
    simp [operationCoreRows]
  · apply product_sound (id .readWrite slot 0 0) _ _ assignment
    apply core
    simp [operationCoreRows]
  · apply product_sound (id .timestampOrder slot 0 0) _ _ assignment
    apply core
    simp [operationCoreRows]
  · apply product_sound (id .romWrite slot 0 0) _ _ assignment
    apply core
    simp [operationCoreRows]
  · intro offset offsetBound
    apply product_sound (id .romRange slot 0 offset) _ _ assignment
    apply core
    unfold operationCoreRows
    apply List.mem_append_left
    apply List.mem_append_left
    apply List.mem_append_left
    apply List.mem_append_right
    exact List.mem_map.mpr
      ⟨offset, List.mem_range.mpr offsetBound, rfl⟩
  · apply product_sound (id .padding slot 0 0) _ _ assignment
    apply core
    simp [operationCoreRows]
  · apply product_sound (id .padding slot 0 1) _ _ assignment
    apply core
    simp [operationCoreRows]
  · apply product_sound (id .padding slot 0 2) _ _ assignment
    apply core
    simp [operationCoreRows]
  · apply product_sound (id .padding slot 0 3) _ _ assignment
    apply core
    simp [operationCoreRows]
  · apply product_sound (id .padding slot 0 4) _ _ assignment
    apply core
    simp [operationCoreRows]
  · apply product_sound (id .padding slot 0 5) _ _ assignment
    apply core
    simp [operationCoreRows]

private theorem operationLaneBitRows_honest
    (assignment : Nat → F) (params : Params) (slot : Nat)
    (accepted : OperationAccepted assignment params slot) :
    Satisfies (operationLaneBitRows params slot) assignment := by
  have operation := bitRows_honest .operationBit slot 0
    (params.operationSlot slot) params.operationBits assignment
    accepted.laneBits
  have diff := bitRows_honest .operationBit slot params.operationBits
    (params.operationDiff slot) timestampBits assignment accepted.diffBits
  have count := bitRows_honest .operationBit slot
    (params.operationBits + timestampBits)
    (params.operationCount slot) params.countBits assignment
    accepted.countBits
  have readProduct := bitRows_honest .operationBit slot
    (params.operationBits + timestampBits + params.countBits)
    (params.operationReadProduct slot) extensionBits assignment
    accepted.readProductBits
  have writeProduct := bitRows_honest .operationBit slot
    (params.operationBits + timestampBits + params.countBits +
      extensionBits)
    (params.operationWriteProduct slot) extensionBits assignment
    accepted.writeProductBits
  simpa [operationLaneBitRows] using
    satisfies_append
      (satisfies_append
        (satisfies_append (satisfies_append operation diff) count)
        readProduct)
      writeProduct

private theorem operationCoreRows_honest
    (assignment : Nat → F) (params : Params) (slot : Nat)
    (accepted : OperationAccepted assignment params slot) :
    Satisfies (operationCoreRows params slot) assignment := by
  let pad := operationPad params slot
  let isWrite := operationIsWrite params slot
  let ram := operationRam params slot
  let address := operationAddress params slot
  let readValue := operationReadValue params slot
  let writeValue := operationWriteValue params slot
  let readTimestamp := operationReadTimestamp params slot
  let count := operationCountWord params slot
  let previousCount := if slot = 0 then Rows.LinearCombination.zero
    else operationCountWord params (slot - 1)
  let notPad := Rows.LinearCombination.sub one pad
  let rom := Rows.LinearCombination.sub one ram
  let writeTimestamp := operationWriteTimestamp params slot
  have fixed : Satisfies
      [ linearRow (id .operationCount slot 0 0) count
          (Rows.LinearCombination.add previousCount notPad)
      , productRow (id .readWrite slot 0 0)
          (Rows.LinearCombination.sub one isWrite)
          (Rows.LinearCombination.sub writeValue readValue)
      , productRow (id .timestampOrder slot 0 0) notPad
          (Rows.LinearCombination.sub
            (Rows.LinearCombination.sub
              (Rows.LinearCombination.sub writeTimestamp readTimestamp) one)
            (operationDiffWord params slot))
      , productRow (id .romWrite slot 0 0) isWrite rom ] assignment := by
    intro row member
    simp at member
    rcases member with rfl | rfl | rfl | rfl
    · exact linear_honest _ _ _ assignment accepted.countExact
    · exact product_honest _ _ _ assignment accepted.readWritesBack
    · exact product_honest _ _ _ assignment accepted.timestampOrdered
    · exact product_honest _ _ _ assignment accepted.romReadOnly
  have ranges : Satisfies
      ((List.range (params.addressBits - params.r)).map fun offset =>
        productRow (id .romRange slot 0 offset) rom
          (Rows.LinearCombination.bit
            (params.operationSlot slot + 3 + params.r + offset)))
      assignment := by
    intro row member
    rcases List.mem_map.mp member with ⟨offset, offsetMember, rfl⟩
    exact product_honest _ _ _ assignment
      (accepted.romAddressRange offset (List.mem_range.mp offsetMember))
  have padding : Satisfies
      [ productRow (id .padding slot 0 0) pad isWrite
      , productRow (id .padding slot 0 1) pad ram
      , productRow (id .padding slot 0 2) pad address
      , productRow (id .padding slot 0 3) pad readValue
      , productRow (id .padding slot 0 4) pad writeValue
      , productRow (id .padding slot 0 5) pad readTimestamp ] assignment := by
    intro row member
    simp at member
    rcases member with rfl | rfl | rfl | rfl | rfl | rfl
    · exact product_honest _ _ _ assignment accepted.padIsWriteZero
    · exact product_honest _ _ _ assignment accepted.padRamZero
    · exact product_honest _ _ _ assignment accepted.padAddressZero
    · exact product_honest _ _ _ assignment accepted.padReadValueZero
    · exact product_honest _ _ _ assignment accepted.padWriteValueZero
    · exact product_honest _ _ _ assignment accepted.padReadTimestampZero
  have readProduct := operationProductRows_honest assignment params slot false
    accepted.readProductExact
  have writeProduct := operationProductRows_honest assignment params slot true
    accepted.writeProductExact
  simpa [operationCoreRows] using
    satisfies_append
      (satisfies_append
        (satisfies_append (satisfies_append fixed ranges) padding)
        readProduct)
      writeProduct

theorem operationRows_honest
    (assignment : Nat → F) (params : Params) (slot : Nat)
    (accepted : OperationAccepted assignment params slot) :
    Satisfies (operationRows params slot) assignment := by
  exact satisfies_append
    (operationLaneBitRows_honest assignment params slot accepted)
    (operationCoreRows_honest assignment params slot accepted)

private theorem scanLaneSatisfied
    (assignment : Nat → F) (params : Params) (slot : Nat)
    (final : Bool) (satisfied : Satisfies (scanRows params slot) assignment) :
    Satisfies (scanRowsForLane params final slot) assignment := by
  intro row member
  cases final
  · exact satisfied row (List.mem_append_left _ member)
  · exact satisfied row (List.mem_append_right _ member)

private theorem scanBitGroups
    (assignment : Nat → F) (params : Params) (slot : Nat)
    (final : Bool)
    (satisfied : Satisfies (scanRowsForLane params final slot) assignment) :
    (∀ offset, offset < cellBits →
      IsBit assignment (scanCellStart params final slot + offset)) ∧
    (∀ offset, offset < extensionBits →
      IsBit assignment (scanProductStart params final slot + offset)) := by
  constructor
  · apply bitRows_sound (if final then .finalScanBit else .initialScanBit)
      slot 0 _ _ assignment
    intro row member
    exact satisfied row (by simp [scanRowsForLane, member])
  · apply bitRows_sound (if final then .finalScanBit else .initialScanBit)
      slot cellBits _ _ assignment
    intro row member
    exact satisfied row (by simp [scanRowsForLane, member])

theorem scanAccepted_of_rows
    (assignment : Nat → F) (params : Params)
    (constantWire : assignment 0 = 1)
    (allSatisfied : Satisfies (rows params) assignment)
    (slot : Nat) (slotBound : slot < params.scanSlots) :
    ScanAccepted assignment params slot := by
  have satisfied := scanRows_satisfied_of_rows assignment params allSatisfied
    slot slotBound
  have initialSatisfied := scanLaneSatisfied assignment params slot false
    satisfied
  have finalSatisfied := scanLaneSatisfied assignment params slot true
    satisfied
  obtain ⟨initialCellBits, initialProductBits⟩ :=
    scanBitGroups assignment params slot false initialSatisfied
  obtain ⟨finalCellBits, finalProductBits⟩ :=
    scanBitGroups assignment params slot true finalSatisfied
  exact {
    initialCellBits := initialCellBits
    initialProductBits := by simpa [scanProductStart] using initialProductBits
    finalCellBits := finalCellBits
    finalProductBits := by simpa [scanProductStart] using finalProductBits
    initialProductExact := scanProduct_sound_of_rows assignment params
      constantWire allSatisfied false slot slotBound
    finalProductExact := scanProduct_sound_of_rows assignment params
      constantWire allSatisfied true slot slotBound }

private theorem scanRowsForLane_honest
    (assignment : Nat → F) (params : Params) (slot : Nat)
    (final : Bool) (constantWire : assignment 0 = 1)
    (cellBitsExact : ∀ offset, offset < cellBits →
      IsBit assignment (scanCellStart params final slot + offset))
    (productBitsExact : ∀ offset, offset < extensionBits →
      IsBit assignment (scanProductStart params final slot + offset))
    (productExact :
      scanProduct assignment params final slot =
        K.mul (previousScanProductValue assignment params final slot)
          (scanFactor assignment params final slot)) :
    Satisfies (scanRowsForLane params final slot) assignment := by
  have cellRows := bitRows_honest
    (if final then .finalScanBit else .initialScanBit) slot 0
    (scanCellStart params final slot) cellBits assignment cellBitsExact
  have productRows := bitRows_honest
    (if final then .finalScanBit else .initialScanBit) slot cellBits
    (scanProductStart params final slot) extensionBits assignment
    productBitsExact
  have extension := scanProductRows_honest assignment params final slot
    constantWire productExact
  simpa [scanRowsForLane] using
    satisfies_append (satisfies_append cellRows productRows) extension

theorem scanRows_honest
    (assignment : Nat → F) (params : Params) (slot : Nat)
    (constantWire : assignment 0 = 1)
    (accepted : ScanAccepted assignment params slot) :
    Satisfies (scanRows params slot) assignment := by
  exact satisfies_append
    (scanRowsForLane_honest assignment params slot false constantWire
      accepted.initialCellBits
      (by simpa [scanProductStart] using accepted.initialProductBits)
      accepted.initialProductExact)
    (scanRowsForLane_honest assignment params slot true constantWire
      accepted.finalCellBits
      (by simpa [scanProductStart] using accepted.finalProductBits)
      accepted.finalProductExact)

theorem boundaryAccepted_of_rows
    (assignment : Nat → F) (params : Params)
    (allSatisfied : Satisfies (rows params) assignment) :
    BoundaryAccepted assignment params := by
  have satisfied := boundaryRows_satisfied_of_rows assignment params
    allSatisfied
  refine {
    timestampExact := ?_
    product0 := boundaryProduct_sound assignment params allSatisfied 0
      (by decide)
    product1 := boundaryProduct_sound assignment params allSatisfied 1
      (by decide)
    product2 := boundaryProduct_sound assignment params allSatisfied 2
      (by decide)
    product3 := boundaryProduct_sound assignment params allSatisfied 3
      (by decide) }
  apply linear_sound (id .boundaryTimestamp 0 0 0) _ _ assignment
  apply satisfied
  simp [Compiler.boundaryRows]

private theorem boundaryComponent_honest
    (assignment : Nat → F) (params : Params)
    (accepted : BoundaryAccepted assignment params)
    (product component : Nat)
    (productBound : product < 4) (componentBound : component < 2) :
    LinearEqual assignment (productOutputWord product component)
      (boundarySource params product component) := by
  have productEqual :
      outputProduct assignment product =
        boundaryProductValue assignment params product := by
    have cases : product = 0 ∨ product = 1 ∨ product = 2 ∨ product = 3 := by
      omega
    rcases cases with rfl | rfl | rfl | rfl
    · exact accepted.product0
    · exact accepted.product1
    · exact accepted.product2
    · exact accepted.product3
  have components : component = 0 ∨ component = 1 := by
    omega
  rcases components with rfl | rfl
  · simpa [LinearEqual, outputProduct, boundaryProductValue, evaluatePair]
      using congrArg K.c0 productEqual
  · simpa [LinearEqual, outputProduct, boundaryProductValue, evaluatePair]
      using congrArg K.c1 productEqual

theorem boundaryRows_honest
    (assignment : Nat → F) (params : Params)
    (accepted : BoundaryAccepted assignment params) :
    Satisfies (Compiler.boundaryRows params) assignment := by
  let timestamp :=
    linearRow (id .boundaryTimestamp 0 0 0)
      (publicWord XOffset.timestampOut timestampBits)
      (Rows.LinearCombination.add timestampIn
        (operationCountWord params (lastOperationSlot params)))
  let products :=
    (List.range 4).flatMap fun product =>
      (List.range 2).map fun component =>
        linearRow (id .boundaryProduct product component 0)
          (productOutputWord product component)
          (boundarySource params product component)
  have timestampSatisfied : timestamp.Holds assignment :=
    linear_honest _ _ _ assignment accepted.timestampExact
  have productsSatisfied : Satisfies products assignment := by
    intro row member
    rcases List.mem_flatMap.mp member with
      ⟨product, productMember, componentRowsMember⟩
    rcases List.mem_map.mp componentRowsMember with
      ⟨component, componentMember, rfl⟩
    exact linear_honest _ _ _ assignment
      (boundaryComponent_honest assignment params accepted product component
        (List.mem_range.mp productMember)
        (List.mem_range.mp componentMember))
  intro row member
  change row ∈ timestamp :: products at member
  rcases List.mem_cons.mp member with rfl | inProducts
  · exact timestampSatisfied
  · exact productsSatisfied row inProducts

private theorem fillerZero_of_rows
    (assignment : Nat → F) (params : Params)
    (allSatisfied : Satisfies (rows params) assignment) :
    ∀ column, column ∈ params.fillerColumns → assignment column = 0 := by
  have rawSatisfied := rawRows_satisfied_of_rows assignment params allSatisfied
  intro column columnMember
  have holds :
      (linearRow (id .filler column 0 0)
        (Rows.LinearCombination.bit column)
        Rows.LinearCombination.zero).Holds assignment := by
    apply rawSatisfied
    unfold rawRows
    apply List.mem_append_left
    apply List.mem_append_left
    apply List.mem_append_left
    exact List.mem_map.mpr ⟨column, columnMember, rfl⟩
  have equal := linear_sound _ _ _ assignment holds
  simpa [LinearEqual] using equal

private theorem fillerRows_honest
    (assignment : Nat → F) (params : Params)
    (fillerZero : ∀ column, column ∈ params.fillerColumns →
      assignment column = 0) :
    Satisfies (fillerRows params) assignment := by
  intro row member
  rcases List.mem_map.mp member with ⟨column, columnMember, rfl⟩
  apply linear_honest
  unfold LinearEqual
  simpa using fillerZero column columnMember

private theorem satisfies_numberRowsFrom
    (position : Nat) (source : List Row) (assignment : Nat → F)
    (satisfied : Satisfies source assignment) :
    Satisfies (numberRowsFrom position source) assignment := by
  induction source generalizing position with
  | nil => simp [Satisfies, numberRowsFrom]
  | cons head tail inductionHypothesis =>
      have headHolds := satisfied head List.mem_cons_self
      have tailSatisfied : Satisfies tail assignment := by
        intro row member
        exact satisfied row (List.mem_cons_of_mem head member)
      intro row member
      simp only [numberRowsFrom, List.mem_cons] at member
      rcases member with rfl | tailMember
      · exact (Row.withPosition_holds_iff position head assignment).mpr
          headHolds
      · exact inductionHypothesis (position + 1) tailSatisfied row tailMember

/-- Every declaratively accepted stackless Nebula assignment satisfies the
exact numbered rows. This is row-level honest completeness; construction of
such an assignment from a memory trace is owned by the witness layer. -/
theorem rows_honest_of_accepted
    (assignment : Nat → F) (params : Params)
    (accepted : Accepted assignment params) :
    Satisfies (rows params) assignment := by
  have fillers := fillerRows_honest assignment params accepted.fillerZero
  have operations : Satisfies
      ((List.range params.operationSlots).flatMap (operationRows params))
      assignment := by
    intro row member
    rcases List.mem_flatMap.mp member with
      ⟨slot, slotMember, rowMember⟩
    exact operationRows_honest assignment params slot
      (accepted.operations slot (List.mem_range.mp slotMember))
      row rowMember
  have scans : Satisfies
      ((List.range params.scanSlots).flatMap (scanRows params))
      assignment := by
    intro row member
    rcases List.mem_flatMap.mp member with
      ⟨slot, slotMember, rowMember⟩
    exact scanRows_honest assignment params slot accepted.constantWire
      (accepted.scans slot (List.mem_range.mp slotMember))
      row rowMember
  have boundary := boundaryRows_honest assignment params accepted.boundary
  have raw : Satisfies (rawRows params) assignment := by
    unfold rawRows
    exact satisfies_append
      (satisfies_append (satisfies_append fillers operations) scans)
      boundary
  exact satisfies_numberRowsFrom 0 (rawRows params) assignment raw

/-- Satisfaction of the complete selected compiler implies every declarative
stackless Nebula constraint, including all four public products. -/
theorem accepted_of_rows
    (assignment : Nat → F) (params : Params)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows params) assignment) :
    Accepted assignment params := {
  constantWire := constantWire
  fillerZero := fillerZero_of_rows assignment params satisfied
  operations := operationAccepted_of_rows assignment params satisfied
  scans := scanAccepted_of_rows assignment params constantWire satisfied
  boundary := boundaryAccepted_of_rows assignment params satisfied
}

/-- Exact execution correspondence for the Lean-owned compiler. The constant
wire is an explicit verifier-owned condition, not an emitted row. -/
theorem satisfies_iff_accepted
    (assignment : Nat → F) (params : Params) :
    assignment 0 = 1 ∧ Satisfies (rows params) assignment ↔
      Accepted assignment params := by
  constructor
  · rintro ⟨constantWire, satisfied⟩
    exact accepted_of_rows assignment params constantWire satisfied
  · intro accepted
    exact ⟨accepted.constantWire,
      rows_honest_of_accepted assignment params accepted⟩

end Nightstream.Implementation.Lowering.Nebula.ConstraintSemantics
