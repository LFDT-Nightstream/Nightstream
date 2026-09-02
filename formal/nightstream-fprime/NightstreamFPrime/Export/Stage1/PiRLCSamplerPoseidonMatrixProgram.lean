import NightstreamFPrime.Export.MatrixProgram.Program
import NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonPlan

/-!
Owns the compact matrix program for the 153 PiRLC sampler Poseidon2
invocations. The package carries the cross-family previous-state wires and
one optional constant per invocation lane.

This module does not own sampler digit or selector rows.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec

abbrev Program := Lifecycle.Stage1.Application.Program

def piCcsFinalSlotBase : Nat :=
  (PiCCSPoseidonPlan.invocationCount - 1) * 86 + 78

@[simp] theorem piCcsFinalSlotBase_eq : piCcsFinalSlotBase = 653936 := by
  norm_num [piCcsFinalSlotBase, PiCCSPoseidonPlan.invocationCount_eq]

def constantAt
    (index : Fin (PiRLCSamplerPoseidonPlan.invocationCount * 8)) : Option F :=
  let decoded : Fin PiRLCSamplerPoseidonPlan.invocationCount × Fin 8 :=
    Fin.decodeProd index
  let descriptor := PiRLCSamplerPoseidonPlan.descriptor decoded.1
  if descriptor.2.val = 0 then
    some (PiRLCSamplerPoseidonPlan.entryWord descriptor.1 decoded.2)
  else
    none

def constants : PoseidonInput.OptionalConstantTable :=
  PoseidonInput.OptionalConstantTable.ofSemantic constantAt

def piCcsPreviousRule (program : Program) : PoseidonInput.Rule where
  region := ⟨0, 1, 0, 8⟩
  term := .external
    (RetainedBlock.ofSemantic (PiCCSPoseidonPlan.retainedBlock program)
      (PiCCSPoseidonPlan.retainedStart program)) piCcsFinalSlotBase 0

def samplerPreviousRule (program : Program) : PoseidonInput.Rule where
  region := ⟨1, 152, 0, 8⟩
  term := .external
    (RetainedBlock.ofSemantic (PiRLCSamplerPoseidonPlan.retainedBlock program)
      (PiRLCSamplerPoseidonPlan.retainedStart program)) 78 86

def entryRule : PoseidonInput.Rule where
  region := ⟨0, 153, 0, 8⟩
  term := .optionalConstant constants 8

def inputProgram (program : Program) : PoseidonInput.Program where
  rules := [piCcsPreviousRule program, samplerPreviousRule program, entryRule]

def block {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    Poseidon.Block :=
  Poseidon.Block.ofSemantic (PiRLCSamplerPoseidonPlan.schedule program)
    (PiRLCSamplerPoseidonPlan.retainedStart program)
    (PiRLCSamplerPoseidonPlan.oneColumn geometry) (inputProgram program)

def matrixProgram {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    MatrixProgram.Program where
  blocks := [.poseidon (block geometry)]

@[simp] theorem block_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    (block geometry).rowCount = 14382 := by
  calc
    (block geometry).rowCount =
        PiRLCSamplerPoseidonPlan.invocationCount * 94 := by
      exact Poseidon.Block.ofSemantic_rowCount
        (PiRLCSamplerPoseidonPlan.schedule program)
        (PiRLCSamplerPoseidonPlan.retainedStart program)
        (PiRLCSamplerPoseidonPlan.oneColumn geometry) (inputProgram program)
    _ = 14382 := by
      norm_num [PiRLCSamplerPoseidonPlan.invocationCount_eq]

@[simp] theorem matrixProgram_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    (matrixProgram geometry).rowCount = 14382 := by
  rw [show matrixProgram geometry =
      MatrixProgram.Program.mk [.poseidon (block geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_rowCount]
  exact block_rowCount geometry

end NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonMatrixProgram
