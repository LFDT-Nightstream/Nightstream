import Nightstream.Implementation.Lowering.Nebula.TerminalR1csProgram

/-!
Contract: shape certificate for every row emitted by the Lean-owned stackless
Nebula compiler.

Assurance tier: model-level.

Owns: constructor provenance for the exact compiler row list. This prevents a
family tag from selecting a terminal lowering that does not match the sparse
row that the compiler emitted.

Does not own: row satisfaction, terminal assignments, Spartan, WHIR, JSON, or
Rust.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Lowering.Nebula.TerminalR1csCompilerShape

open Nightstream.Implementation.Lowering.Nebula
open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.TerminalR1cs
open Nightstream.Implementation.Lowering.Nebula.TerminalR1csProgram

theorem bitRows_wellShaped (family : Rows.Family)
    (slot ordinalBase start width : Nat)
    (kind : familyKind family = .bit) :
    WellShaped (Compiler.bitRows family slot ordinalBase start width) := by
  intro row member
  rw [Compiler.bitRows, List.mem_map] at member
  rcases member with ⟨offset, _, rfl⟩
  exact .bit _ _ kind

theorem fillerRows_wellShaped (params : Params) :
    WellShaped (Compiler.fillerRows params) := by
  intro row member
  rw [Compiler.fillerRows, List.mem_map] at member
  rcases member with ⟨column, _, rfl⟩
  exact .linear _ _ _ rfl

theorem extensionRows_wellShaped (family : Rows.Family) (slot : Nat)
    (output0 output1 previous0 previous1 pad active fingerprint0
      fingerprint1 value0 value1 value : Rows.LinearCombination)
    (kind : familyKind family = .extension) :
    WellShaped
      (Compiler.extensionRows family slot output0 output1 previous0 previous1
        pad active fingerprint0 fingerprint1 value0 value1 value) := by
  intro row member
  simp only [Compiler.extensionRows, List.mem_cons] at member
  rcases member with rfl | rfl | impossible
  · exact .extension _ _ _ _ _ _ _ _ _ _ _ kind
  · exact .extension _ _ _ _ _ _ _ _ _ _ _ kind
  · nomatch impossible

theorem operationLaneBitRows_wellShaped (params : Params) (slot : Nat) :
    WellShaped (Compiler.operationLaneBitRows params slot) := by
  unfold Compiler.operationLaneBitRows
  simp only [wellShaped_append_iff]
  exact
    ⟨⟨⟨⟨bitRows_wellShaped _ _ _ _ _ rfl,
          bitRows_wellShaped _ _ _ _ _ rfl⟩,
        bitRows_wellShaped _ _ _ _ _ rfl⟩,
      bitRows_wellShaped _ _ _ _ _ rfl⟩,
     bitRows_wellShaped _ _ _ _ _ rfl⟩

theorem operationProductRows_wellShaped
    (params : Params) (slot : Nat) (write : Bool) :
    WellShaped (Compiler.operationProductRows params slot write) := by
  cases write <;>
    simp only [Compiler.operationProductRows, Bool.false_eq_true, if_false,
      if_true] <;>
    apply extensionRows_wellShaped <;>
    rfl

theorem operationCoreRows_wellShaped (params : Params) (slot : Nat) :
    WellShaped (Compiler.operationCoreRows params slot) := by
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
  have fixedShaped : WellShaped fixed := by
    intro row member
    simp only [fixed, List.mem_cons] at member
    rcases member with rfl | rfl | rfl | rfl | impossible
    · exact .linear _ _ _ rfl
    · exact .product _ _ _ rfl
    · exact .product _ _ _ rfl
    · exact .product _ _ _ rfl
    · nomatch impossible
  have rangeShaped : WellShaped rangeRows := by
    intro row member
    simp only [rangeRows, List.mem_map] at member
    rcases member with ⟨offset, _, rfl⟩
    exact .product _ _ _ rfl
  have paddingShaped : WellShaped paddingRows := by
    have paddingRowsEq : paddingRows =
        [ Rows.productRow (Compiler.id .padding slot 0 0) pad isWrite
        , Rows.productRow (Compiler.id .padding slot 0 1) pad ram
        , Rows.productRow (Compiler.id .padding slot 0 2) pad address
        , Rows.productRow (Compiler.id .padding slot 0 3) pad readValue
        , Rows.productRow (Compiler.id .padding slot 0 4) pad writeValue
        , Rows.productRow (Compiler.id .padding slot 0 5) pad readTimestamp ] :=
      rfl
    intro row member
    rw [paddingRowsEq] at member
    simp only [List.mem_cons] at member
    rcases member with rfl | rfl | rfl | rfl | rfl | rfl | impossible
    · exact .product _ _ _ rfl
    · exact .product _ _ _ rfl
    · exact .product _ _ _ rfl
    · exact .product _ _ _ rfl
    · exact .product _ _ _ rfl
    · exact .product _ _ _ rfl
    · nomatch impossible
  change WellShaped
    (fixed ++ rangeRows ++ paddingRows ++
      Compiler.operationProductRows params slot false ++
      Compiler.operationProductRows params slot true)
  simp only [wellShaped_append_iff]
  exact
    ⟨⟨⟨⟨fixedShaped, rangeShaped⟩, paddingShaped⟩,
        operationProductRows_wellShaped params slot false⟩,
      operationProductRows_wellShaped params slot true⟩

theorem operationRows_wellShaped (params : Params) (slot : Nat) :
    WellShaped (Compiler.operationRows params slot) := by
  rw [show Compiler.operationRows params slot =
    Compiler.operationLaneBitRows params slot ++
      Compiler.operationCoreRows params slot from rfl,
    wellShaped_append_iff]
  exact ⟨operationLaneBitRows_wellShaped params slot,
    operationCoreRows_wellShaped params slot⟩

theorem scanRowsForLane_wellShaped
    (params : Params) (final : Bool) (slot : Nat) :
    WellShaped (Compiler.scanRowsForLane params final slot) := by
  cases final <;>
    simp only [Compiler.scanRowsForLane, Bool.false_eq_true, if_false,
      if_true, wellShaped_append_iff] <;>
    exact
      ⟨⟨bitRows_wellShaped _ _ _ _ _ rfl,
          bitRows_wellShaped _ _ _ _ _ rfl⟩,
        extensionRows_wellShaped _ _ _ _ _ _ _ _ _ _ _ _ _ rfl⟩

theorem scanRows_wellShaped (params : Params) (slot : Nat) :
    WellShaped (Compiler.scanRows params slot) := by
  rw [show Compiler.scanRows params slot =
    Compiler.scanRowsForLane params false slot ++
      Compiler.scanRowsForLane params true slot from rfl,
    wellShaped_append_iff]
  exact ⟨scanRowsForLane_wellShaped params false slot,
    scanRowsForLane_wellShaped params true slot⟩

theorem boundaryRows_wellShaped (params : Params) :
    WellShaped (Compiler.boundaryRows params) := by
  intro row member
  unfold Compiler.boundaryRows at member
  simp only [List.mem_cons] at member
  rcases member with rfl | member
  · exact .linear _ _ _ rfl
  · rw [List.mem_flatMap] at member
    rcases member with ⟨product, _, member⟩
    rw [List.mem_map] at member
    rcases member with ⟨component, _, rfl⟩
    exact .linear _ _ _ rfl

private theorem flatMap_range_wellShaped
    (count : Nat) (items : Nat -> List Rows.Row)
    (shaped : ∀ index, index < count -> WellShaped (items index)) :
    WellShaped ((List.range count).flatMap items) := by
  intro row member
  rw [List.mem_flatMap] at member
  rcases member with ⟨index, indexMember, rowMember⟩
  exact shaped index (List.mem_range.mp indexMember) row rowMember

theorem rawRows_wellShaped (params : Params) :
    WellShaped (Compiler.rawRows params) := by
  rw [show Compiler.rawRows params =
    Compiler.fillerRows params ++
      (List.range params.operationSlots).flatMap
        (Compiler.operationRows params) ++
      (List.range params.scanSlots).flatMap (Compiler.scanRows params) ++
      Compiler.boundaryRows params from rfl]
  simp only [wellShaped_append_iff]
  exact
    ⟨⟨⟨fillerRows_wellShaped params,
          flatMap_range_wellShaped params.operationSlots
            (Compiler.operationRows params)
            (fun index _ => operationRows_wellShaped params index)⟩,
        flatMap_range_wellShaped params.scanSlots (Compiler.scanRows params)
          (fun index _ => scanRows_wellShaped params index)⟩,
      boundaryRows_wellShaped params⟩

theorem compilerRows_wellShaped (params : Params) :
    WellShaped (Compiler.rows params) := by
  rw [Compiler.rows]
  exact numberRowsFrom_wellShaped 0 (Compiler.rawRows params)
    (rawRows_wellShaped params)

theorem wasm42x6_rows_wellShaped :
    WellShaped (Compiler.rows wasm42x6) :=
  compilerRows_wellShaped wasm42x6

end Nightstream.Implementation.Lowering.Nebula.TerminalR1csCompilerShape
