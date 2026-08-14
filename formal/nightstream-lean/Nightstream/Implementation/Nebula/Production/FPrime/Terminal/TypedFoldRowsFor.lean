import Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.FoldManifestFor

/-!
Contract: typed embedding of the complete numeric terminal-fold manifest.

Every numeric row is translated through the fixed injective `numericColumn`
map. The exact generic bridge proves that typed row satisfaction is equivalent
to numeric row satisfaction on the canonical numeric view of the same field
assignment. No second assignment or agreement proposition exists.

This module does not add terminal opening or CE-core rows. Those rows use the
same typed assignment through `ProductionPaperTerminalOpeningRowsFor.Family`.

Assurance tier: exponent-indexed terminal row implementation.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Nebula.ProductionPaperTerminalTypedFoldRowsFor

open Nightstream.Implementation.Nebula
open Nightstream.Protocol.Nebula.ProductionProfileCandidates

/-- Physical ownership of the exact translated terminal-fold row list. -/
structure Frame (candidate : Id) (rowVariables : Nat) where
  program : ProductionPaperTerminalFoldManifestFor.Program candidate rowVariables
  owner : Nightstream.Implementation.Lowering.Goldilocks.PhysicalOwner
  firstOrdinal : Nat

def Frame.numericRows
    {candidate : Id} {rowVariables : Nat}
    (frame : Frame candidate rowVariables) :
    List Nightstream.Implementation.R1CS.Row :=
  frame.program.rows

def Frame.rows
    {candidate : Id} {rowVariables : Nat}
    (frame : Frame candidate rowVariables) :
    List Nightstream.Implementation.Lowering.Goldilocks.OwnedRow :=
  Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.ownedRowsFrom
    frame.owner frame.firstOrdinal
    TerminalBundleOpeningRows.Layout.numericColumn frame.numericRows

theorem Frame.rows_length_exact
    {candidate : Id} {rowVariables : Nat}
    (frame : Frame candidate rowVariables) :
    frame.rows.length =
      ProductionPaperTerminalFoldManifestFor.rowCount candidate rowVariables := by
  rw [Frame.rows,
    Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.ownedRowsFrom_length]
  exact frame.program.rows_length_exact

/-- The complete typed list has exactly the semantics of the complete numeric
terminal-fold manifest on the same assignment. -/
theorem Frame.rows_satisfied_iff
    {candidate : Id} {rowVariables : Nat}
    (frame : Frame candidate rowVariables)
    (assignment : Nightstream.Implementation.Lowering.Goldilocks.ColumnId ->
      Nightstream.SuperNeo.Concrete.F) :
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies frame.rows
        assignment <->
      Nightstream.Implementation.R1CS.Satisfies frame.numericRows
        (ProductionPaperTerminalOpeningRowsFor.pulledAssignment assignment) := by
  exact
    Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.ownedRowsFrom_satisfies_iff
      frame.owner frame.firstOrdinal
      TerminalBundleOpeningRows.Layout.numericColumn frame.numericRows assignment

end Nightstream.Implementation.Nebula.ProductionPaperTerminalTypedFoldRowsFor
