import Nightstream.Implementation.Lowering.Nebula.Physical

set_option autoImplicit false

namespace tests.NebulaCompiler

open Nightstream.Implementation.Lowering.Nebula
open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.Rows
open Nightstream.Implementation.Lowering.Nebula.Compiler
open Nightstream.Implementation.Lowering.Nebula.Physical

theorem selected_layout_exact :
    wasm42x6.publicEnd = 1401 ∧
      wasm42x6.columnCount = 419747 ∧
      wasm42x6.witnessColumns = 418346 := by
  exact ⟨wasm42x6_publicColumns,
    wasm42x6_columnCount, wasm42x6_witnessColumns⟩

theorem selected_schedule_exact :
    (fillerRows wasm42x6).length = 132 ∧
      (operationRows wasm42x6 0).length = 436 ∧
      (scanRows wasm42x6 0).length = 412 ∧
      (boundaryRows wasm42x6).length = 9 := by
  constructor
  · simpa using wasm42x6_fillerColumns
  constructor
  · rw [operationRows_length, wasm42x6_rowsPerOperation]
  constructor
  · rw [scanRows_length]
    rfl
  · rw [boundaryRows_length]
    rfl

theorem selected_program_exact :
    (rows wasm42x6).length = 422465 :=
  wasm42x6_rows_length

theorem selected_allocation_exact :
    (allocatedColumns wasm42x6).length = 419747 ∧
      wasm42x6.publicEnd = 1401 ∧
      wasm42x6.columnCount - wasm42x6.publicEnd = 418346 := by
  exact ⟨wasm42x6_allocatedColumns_length,
    wasm42x6_publicColumnCount, wasm42x6_witnessColumnCount⟩

/-- The first column beyond the selected allocation is rejected. -/
theorem out_of_range_column_rejected :
    wasm42x6.columnCount ∉ allocatedColumns wasm42x6 := by
  simp [allocatedColumns]

/-- The stackless profile emits no ROM high-bit range rows because the ROM
and RAM address fields both use ten bits. -/
theorem selected_rom_range_is_empty :
    ((operationCoreRows wasm42x6 0).filter fun row =>
      decide (row.id.family = .romRange)).length = 0 := by
  decide

/-- A coefficient mutation in one emitted bit equation is rejected. -/
theorem mutated_lane_bit_rejected :
    ¬ (bitRow (id .operationBit 0 0 0)
      wasm42x6.operationLane).Holds
        (fun column => if column = wasm42x6.operationLane then 2 else 0) := by
  rw [bitRow_holds_iff]
  decide

end tests.NebulaCompiler
