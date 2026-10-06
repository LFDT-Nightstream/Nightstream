import NightstreamFPrime.Export.MatrixProgram.Program
import NightstreamFPrime.Export.Stage1.PilotPoseidonPlan

/-!
Owns the compact wire input programs and Poseidon2 matrix blocks for the two
pilot state-hash chains. Each chain uses the same four-rule shape and its own
Lean-owned retained blocks.

This module does not select later transcript families or package order.
-/

namespace NightstreamFPrime.Export.Stage1.PilotPoseidonMatrixProgram

open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

abbrev Program := Lifecycle.Stage1.Application.Program

def previousRule {sourceWidth : Nat}
    (schedule : PoseidonRetainedFamily.Schedule sourceWidth 3109)
    (retainedStart : Nat) : PoseidonInput.Rule where
  region := ⟨1, 3108, 0, 16⟩
  term := .external
    (RetainedBlock.ofSemantic schedule.block retainedStart) 134 150

def previousProgram {sourceWidth : Nat}
    (schedule : PoseidonRetainedFamily.Schedule sourceWidth 3109)
    (retainedStart : Nat) : PoseidonInput.Program where
  rules := [previousRule schedule retainedStart]

def fullInputRule {sourceWidth : Nat}
    (inputBlock : LowNormBlock.Block sourceWidth) (inputStart : Nat) :
    PoseidonInput.Rule where
  region := ⟨0, 3107, 0, 12⟩
  term := .retained (RetainedBlock.ofSemantic inputBlock inputStart) 0 12 1

def tailInputRule {sourceWidth : Nat}
    (inputBlock : LowNormBlock.Block sourceWidth) (inputStart : Nat) :
    PoseidonInput.Rule where
  region := ⟨3107, 1, 0, 11⟩
  term := .retained (RetainedBlock.ofSemantic inputBlock inputStart)
    37284 0 1

def paddingRule : PoseidonInput.Rule where
  region := ⟨3108, 1, 0, 1⟩
  term := .constant 1

def chainInputProgram {poseidonSourceWidth inputSourceWidth : Nat}
    (schedule : PoseidonRetainedFamily.Schedule poseidonSourceWidth 3109)
    (retainedStart : Nat) (inputBlock : LowNormBlock.Block inputSourceWidth)
    (inputStart : Nat) : PoseidonInput.Program where
  rules := [previousRule schedule retainedStart,
    fullInputRule inputBlock inputStart,
    tailInputRule inputBlock inputStart, paddingRule]

def priorInputProgram (program : Program) : PoseidonInput.Program :=
  chainInputProgram (PilotPoseidonPlan.priorSchedule program)
    (PiRLCRetainedGeometry.priorPoseidonStart program)
    (PiRLCPoseidonGeometry.priorInputBlock program)
    (PiRLCPoseidonGeometry.priorInputStart program)

def outputInputProgram (program : Program) : PoseidonInput.Program :=
  chainInputProgram (PilotPoseidonPlan.outputSchedule program)
    (PiRLCRetainedGeometry.outputPoseidonStart program)
    (PiRLCPoseidonGeometry.outputInputBlock program)
    (PiRLCPoseidonGeometry.outputInputStart program)

def priorBlock {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth) :
    Poseidon.Block :=
  Poseidon.Block.ofSemantic (PilotPoseidonPlan.priorSchedule program)
    (PiRLCRetainedGeometry.priorPoseidonStart program)
    (PiRLCPoseidonGeometry.oneColumn geometry) (priorInputProgram program)

def outputBlock {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth) :
    Poseidon.Block :=
  Poseidon.Block.ofSemantic (PilotPoseidonPlan.outputSchedule program)
    (PiRLCRetainedGeometry.outputPoseidonStart program)
    (PiRLCPoseidonGeometry.oneColumn geometry) (outputInputProgram program)

@[simp] theorem priorBlock_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth) :
    (priorBlock geometry).rowCount = 466350 := by
  calc
    (priorBlock geometry).rowCount = 3109 * 150 := by
      exact Poseidon.Block.ofSemantic_rowCount
        (PilotPoseidonPlan.priorSchedule program)
        (PiRLCRetainedGeometry.priorPoseidonStart program)
        (PiRLCPoseidonGeometry.oneColumn geometry) (priorInputProgram program)
    _ = 466350 := by norm_num

@[simp] theorem outputBlock_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth) :
    (outputBlock geometry).rowCount = 466350 := by
  calc
    (outputBlock geometry).rowCount = 3109 * 150 := by
      exact Poseidon.Block.ofSemantic_rowCount
        (PilotPoseidonPlan.outputSchedule program)
        (PiRLCRetainedGeometry.outputPoseidonStart program)
        (PiRLCPoseidonGeometry.oneColumn geometry) (outputInputProgram program)
    _ = 466350 := by norm_num

/-- The exact Pilot Poseidon row order: prior-state hash, then output-state
hash. -/
def matrixProgram {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth) :
    MatrixProgram.Program :=
  (MatrixProgram.Program.mk [.poseidon (priorBlock geometry)]).append
    (MatrixProgram.Program.mk [.poseidon (outputBlock geometry)])

@[simp] theorem matrixProgram_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth) :
    (matrixProgram geometry).rowCount = 932700 := by
  rw [matrixProgram, MatrixProgram.Program.append_rowCount]
  simp only [MatrixProgram.Program.singleton_rowCount]
  change (priorBlock geometry).rowCount + (outputBlock geometry).rowCount = _
  rw [priorBlock_rowCount, outputBlock_rowCount]

end NightstreamFPrime.Export.Stage1.PilotPoseidonMatrixProgram
