import NightstreamFPrime.Export.MatrixProgram.Program
import NightstreamFPrime.Export.Stage1.PiCCSPoseidonPlan
import NightstreamFPrime.Export.Stage1.PiCCSPayloadMatrix

/-!
Owns the compact matrix program for the complete PiCCS Poseidon2 block. Lean
supplies the parent affine payload words and the retained S-boxes.

This module does not select PiCCS actions or close package conformance.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSPoseidonMatrixProgram

open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

abbrev Program := Lifecycle.Stage1.Application.Program

def previousRule (program : Program) : PoseidonInput.Rule where
  region := ⟨1, 947, 0, 16⟩
  term := .external
    (RetainedBlock.ofSemantic (PiCCSPoseidonPlan.retainedBlock program)
      (PiCCSPoseidonPlan.retainedStart program)) 134 150

def payloadRule (program : Program) : PoseidonInput.Rule where
  region := ⟨0, 948, 0, 12⟩
  term := .affine (PiCCSPayloadMatrix.table ())
    (PiCCSOrdinaryMatrixProgram.substitution program) 12

def inputProgram (program : Program) : PoseidonInput.Program where
  rules := [previousRule program, payloadRule program]

def poseidonBlock {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    Poseidon.Block :=
  Poseidon.Block.ofSemantic (PiCCSPoseidonPlan.schedule program)
    (PiCCSPoseidonPlan.retainedStart program)
    (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) (inputProgram program)

/-- The PiCCS permutation rows. -/
def matrixProgram {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    MatrixProgram.Program where
  blocks := [.poseidon (poseidonBlock geometry)]

@[simp] theorem poseidonBlock_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    (poseidonBlock geometry).rowCount = 142200 := by
  calc
    (poseidonBlock geometry).rowCount =
        PiCCSPoseidonPlan.invocationCount * 150 := by
      exact Poseidon.Block.ofSemantic_rowCount
        (PiCCSPoseidonPlan.schedule program)
        (PiCCSPoseidonPlan.retainedStart program)
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) (inputProgram program)
    _ = 142200 := by
      norm_num [PiCCSPoseidonPlan.invocationCount_eq]

@[simp] theorem matrixProgram_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    (matrixProgram geometry).rowCount = 142200 := by
  rw [show matrixProgram geometry = MatrixProgram.Program.mk
      [.poseidon (poseidonBlock geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_rowCount]
  exact poseidonBlock_rowCount geometry

end NightstreamFPrime.Export.Stage1.PiCCSPoseidonMatrixProgram
