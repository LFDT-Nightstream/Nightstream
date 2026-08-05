import Nightstream.Implementation.Lowering.Nebula.TerminalR1csProgram

/-!
Contract: exact terminal-R1CS lowering size for the selected Nebula rows.

Assurance tier: model-level.

Owns: the count of degree-four extension rows and the exact one-row versus
six-row lowering census. An extension row uses five product auxiliaries.

Does not own: the physical lowering rows, column identities, a terminal
statement, Spartan, WHIR, Rust, or a security reduction.

Emits constraints: none. The physical lowering module consumes these counts.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Lowering.Nebula.TerminalR1csCost

open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.TerminalR1csProgram

/-- Two extension components for each of read/write and initial/final scan. -/
def extensionRows (params : Params) : Nat :=
  4 * params.operationSlots + 4 * params.scanSlots

/-- Every non-extension Nebula row lowers to one R1CS row. Each extension
row lowers to six, which adds five rows over its source occurrence. -/
def relationRows (params : Params) : Nat :=
  params.rowCount + 5 * extensionRows params

/-- Each extension row needs exactly five product-result columns. -/
def auxiliaryColumns (params : Params) : Nat :=
  5 * extensionRows params

theorem extensionRows_eq_emitted (params : Params) :
    extensionRows params =
      TerminalR1csProgram.extensionCount (Compiler.rows params) := by
  rw [TerminalR1csProgram.compilerRows_extensionCount]
  rfl

theorem relationRows_eq_emitted (params : Params) :
    relationRows params =
      (TerminalR1csProgram.rows (Compiler.rows params)).length := by
  rw [TerminalR1csProgram.rows_length, Compiler.rows_length,
    ← extensionRows_eq_emitted]
  rfl

theorem auxiliaryColumns_eq_emitted (params : Params) :
    auxiliaryColumns params =
      (TerminalR1csProgram.columns (Compiler.rows params)).length := by
  rw [TerminalR1csProgram.columns_length, ← extensionRows_eq_emitted]
  rfl

theorem wasm42x6_extensionRows :
    extensionRows wasm42x6 = 4100 := by
  decide

theorem wasm42x6_relationRows :
    relationRows wasm42x6 = 442965 := by
  decide

theorem wasm42x6_auxiliaryColumns :
    auxiliaryColumns wasm42x6 = 20500 := by
  decide

end Nightstream.Implementation.Lowering.Nebula.TerminalR1csCost
