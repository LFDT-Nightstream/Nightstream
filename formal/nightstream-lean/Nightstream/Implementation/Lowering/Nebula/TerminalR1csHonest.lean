import Nightstream.Implementation.Lowering.Nebula.TerminalR1csCompilerShape

/-!
Contract: honest whole-program terminal assignment for the Lean-owned Nebula
compiler.

Assurance tier: model-level.

Owns: one global assignment whose five auxiliary values at each physical row
position satisfy the exact terminal R1CS lowering whenever the source Nebula
program is satisfied.

Does not own: construction of the source memory trace, terminal Ajtai checks,
Spartan, WHIR, JSON, or Rust.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Lowering.Nebula.TerminalR1csHonest

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Nebula
open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.TerminalR1cs
open Nightstream.Implementation.Lowering.Nebula.TerminalR1csProgram
open Nightstream.Implementation.Lowering.Nebula.TerminalR1csCompilerShape

def Positioned (sourceRows : List Rows.Row) : Prop :=
  sourceRows.map (fun row => row.id.position) = List.range sourceRows.length

def completeRows (sourceRows : List Rows.Row) (source : Nat -> F) : Column -> F
  | .source column => source column
  | .auxiliary position kind =>
      match sourceRows[position]? with
      | some row => auxiliaryValue row source kind
      | none => 0

@[simp] theorem completeRows_source (sourceRows : List Rows.Row)
    (source : Nat -> F) (column : Nat) :
    completeRows sourceRows source (.source column) = source column :=
  rfl

theorem positioned_lookup (sourceRows : List Rows.Row) (row : Rows.Row)
    (positioned : Positioned sourceRows) (member : row ∈ sourceRows) :
    sourceRows[row.id.position]? = some row := by
  rcases List.mem_iff_getElem.mp member with ⟨index, bound, rowAt⟩
  have rowLookup : sourceRows[index]? = some row :=
    List.getElem?_eq_some_iff.mpr ⟨bound, rowAt⟩
  have positionAt := congrArg (fun items => items[index]?) positioned
  have positionEq : row.id.position = index := by
    calc
      row.id.position = sourceRows[index].id.position :=
        congrArg (fun item => item.id.position) rowAt.symm
      _ = index := by
        simpa [List.getElem?_map, rowLookup, bound] using positionAt
  rw [positionEq]
  exact List.getElem?_eq_some_iff.mpr ⟨bound, rowAt⟩

@[simp] theorem completeRows_auxiliary (sourceRows : List Rows.Row)
    (source : Nat -> F) (row : Rows.Row) (positioned : Positioned sourceRows)
    (member : row ∈ sourceRows) (kind : Auxiliary) :
    completeRows sourceRows source (.auxiliary row.id.position kind) =
      auxiliaryValue row source kind := by
  simp [completeRows, positioned_lookup sourceRows row positioned member]

theorem lowerRow_completeRows (sourceRows : List Rows.Row) (row : Rows.Row)
    (source : Nat -> F) (constantOne : source 0 = 1)
    (positioned : Positioned sourceRows) (member : row ∈ sourceRows)
    (shape : Shape row)
    (holds : row.Holds source) :
    TerminalR1cs.Satisfies (lowerRow row)
      (completeRows sourceRows source) := by
  cases shape with
  | bit id column kind =>
      rw [show lowerRow (Rows.bitRow id column) =
        bitRows (Rows.bitRow id column) by
          simp [lowerRow, Rows.bitRow, kind]]
      simpa [bitRows, Satisfies, Row.Holds, complete] using
        bitRows_complete id column source constantOne holds
  | product id left right kind =>
      rw [show lowerRow (Rows.productRow id left right) =
        productRows (Rows.productRow id left right) by
          simp [lowerRow, Rows.productRow, kind]]
      simpa [productRows, Satisfies, Row.Holds, complete] using
        productRows_complete id left right source holds
  | linear id left right kind =>
      rw [show lowerRow (Rows.linearRow id left right) =
        linearRows (Rows.linearRow id left right) by
          simp [lowerRow, Rows.linearRow, kind]]
      simpa [linearRows, Satisfies, Row.Holds, complete] using
        linearRows_complete id left right source constantOne holds
  | extension id output extensionA extensionB pad active fingerprintA
      fingerprintB valueA valueB value kind =>
      rw [show lowerRow
        (Rows.extensionUpdateRow id output extensionA extensionB pad active
          fingerprintA fingerprintB valueA valueB value) =
        extensionRows
          (Rows.extensionUpdateRow id output extensionA extensionB pad active
            fingerprintA fingerprintB valueA valueB value) by
          simp [lowerRow, Rows.extensionUpdateRow, kind]]
      have localProof := extensionRows_complete id output extensionA extensionB pad
        active fingerprintA fingerprintB valueA valueB value source holds
      simpa [extensionRows, Satisfies, Row.Holds, complete, auxiliary,
        LinearCombination.eval_source, LinearCombination.eval_singleton,
        LinearCombination.eval_sub, LinearCombination.eval_add,
        completeRows_auxiliary sourceRows _ _ positioned member] using localProof

theorem compilerRows_positioned (params : Params) :
    Positioned (Compiler.rows params) := by
  rw [Positioned, Compiler.rows_positions, Compiler.rows,
    Compiler.numberRowsFrom_length]

theorem completeCompiler_satisfies (params : Params) (source : Nat -> F)
    (constantOne : source 0 = 1)
    (sourceSatisfied : Rows.Satisfies (Compiler.rows params) source) :
    TerminalR1cs.Satisfies (rows (Compiler.rows params))
      (completeRows (Compiler.rows params) source) := by
  rw [TerminalR1cs.satisfies_iff_forall]
  intro lowered loweredMember
  rw [rows, List.mem_flatMap] at loweredMember
  rcases loweredMember with ⟨row, member, loweredMember⟩
  have localProof := lowerRow_completeRows (Compiler.rows params) row source
    constantOne (compilerRows_positioned params) member
    (compilerRows_wellShaped params row member)
    (sourceSatisfied row member)
  rw [TerminalR1cs.satisfies_iff_forall] at localProof
  exact localProof lowered loweredMember

theorem wasm42x6_complete (source : Nat -> F)
    (constantOne : source 0 = 1)
    (sourceSatisfied : Rows.Satisfies (Compiler.rows wasm42x6) source) :
    TerminalR1cs.Satisfies (rows (Compiler.rows wasm42x6))
      (completeRows (Compiler.rows wasm42x6) source) :=
  completeCompiler_satisfies wasm42x6 source constantOne sourceSatisfied

end Nightstream.Implementation.Lowering.Nebula.TerminalR1csHonest
