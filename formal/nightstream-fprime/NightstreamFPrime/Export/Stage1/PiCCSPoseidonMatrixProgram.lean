import NightstreamFPrime.Export.MatrixProgram.Program
import NightstreamFPrime.Export.Stage1.PiCCSPoseidonPlan

/-!
Owns the compact matrix program for the complete PiCCS Poseidon2 block. Lean
supplies one action tag per invocation, the retained payload and S-box blocks,
and the squeeze-binding pin rows.

This module does not select PiCCS actions or close package conformance.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSPoseidonMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

abbrev Program := Lifecycle.Stage1.Application.Program

def invocationTag : PoseidonActionSchedule.Kind → PoseidonInput.InvocationTag
  | .absorb _ => .absorb
  | .squeezeFirst _ => .squeezeFirst
  | .squeezeSecond => .squeezeSecond

def tagAt (invocation : Fin PiCCSPoseidonPlan.invocationCount) :
    PoseidonInput.InvocationTag :=
  invocationTag (PiCCSActionPayloadBlock.kindAt invocation)

def tags : PoseidonInput.TagTable :=
  PoseidonInput.TagTable.ofSemantic tagAt

def previousRule (program : Program) : PoseidonInput.Rule where
  region := ⟨1, 7549, 0, 8⟩
  term := .external
    (RetainedBlock.ofSemantic (PiCCSPoseidonPlan.retainedBlock program)
      (PiCCSPoseidonPlan.retainedStart program)) 78 86

def payloadRule (program : Program) : PoseidonInput.Rule where
  region := ⟨0, 7550, 0, 4⟩
  term := .taggedRetained
    (RetainedBlock.ofSemantic (PiCCSActionPayloadBlock.block program)
      (PiCCSActionPayloadBlock.payloadStart program))
    tags .absorb 0 4 1

def inputProgram (program : Program) : PoseidonInput.Program where
  rules := [previousRule program, payloadRule program]

def poseidonBlock {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    Poseidon.Block :=
  Poseidon.Block.ofSemantic (PiCCSPoseidonPlan.schedule program)
    (PiCCSPoseidonPlan.retainedStart program)
    (PiCCSPoseidonPlan.oneColumn geometry) (inputProgram program)

def bindingBlock {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) : Pin.Block :=
  Pin.Block.ofSemantic (PiCCSPoseidonPlan.bindingInterface geometry)

/-- PiCCS permutation rows precede the squeeze-binding rows. -/
def matrixProgram {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    MatrixProgram.Program where
  blocks := [.poseidon (poseidonBlock geometry), .pin (bindingBlock geometry)]

@[simp] theorem poseidonBlock_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    (poseidonBlock geometry).rowCount = 709700 := by
  calc
    (poseidonBlock geometry).rowCount =
        PiCCSPoseidonPlan.invocationCount * 94 := by
      exact Poseidon.Block.ofSemantic_rowCount
        (PiCCSPoseidonPlan.schedule program)
        (PiCCSPoseidonPlan.retainedStart program)
        (PiCCSPoseidonPlan.oneColumn geometry) (inputProgram program)
    _ = 709700 := by
      norm_num [PiCCSPoseidonPlan.invocationCount_eq]

@[simp] theorem bindingBlock_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    (bindingBlock geometry).rowCount = 15100 := by
  calc
    (bindingBlock geometry).rowCount = PiCCSPoseidonPlan.bindingRowCount := by
      exact Pin.Block.ofSemantic_rowCount
        (PiCCSPoseidonPlan.bindingInterface geometry)
    _ = 15100 := by
      norm_num [PiCCSPoseidonPlan.bindingRowCount,
        PiCCSPoseidonPlan.invocationCount_eq]

@[simp] theorem matrixProgram_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    (matrixProgram geometry).rowCount = 724800 := by
  rw [show matrixProgram geometry = MatrixProgram.Program.mk
      [.poseidon (poseidonBlock geometry), .pin (bindingBlock geometry)] by
    rfl]
  rw [MatrixProgram.Program.two_rowCount]
  change (poseidonBlock geometry).rowCount +
    (bindingBlock geometry).rowCount = 724800
  rw [poseidonBlock_rowCount, bindingBlock_rowCount]

end NightstreamFPrime.Export.Stage1.PiCCSPoseidonMatrixProgram
