import NightstreamFPrime.Export.MatrixProgram.Program
import NightstreamFPrime.Export.Stage1.PerApplicationSourceProjection
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryMatrixSubstitution

/-!
Owns the compact ordinary matrix block for the 220,881 PiRLC sampler rows.
The Lean-authored row schedule selects the exact physical compiled row, and
the four-grid substitution reconstructs its direct source forms.

This module does not close PiRLC status or select a final package identity.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open PiRLCSamplerOrdinaryMatrixSchedule
open PiRLCSamplerOrdinaryMatrixSubstitution
open PiRLCSamplerOrdinaryRetainedGeometry

abbrev Program := Lifecycle.Stage1.Application.Program

def block {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : Ordinary.Block where
  rows := rowSchedule
  oneColumn := (oneColumn geometry).val
  substitution := substitution program
  projection := PerApplicationSourceProjection.base program

@[simp] theorem block_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (block geometry).rowCount = 220881 := by
  exact rowSchedule_count

def matrixProgram {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : MatrixProgram.Program where
  blocks := [.ordinary (block geometry)]

@[simp] theorem matrixProgram_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (matrixProgram geometry).rowCount = 220881 := by
  rw [show matrixProgram geometry =
      MatrixProgram.Program.mk [.ordinary (block geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_rowCount]
  exact block_rowCount geometry

end NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryMatrixProgram
