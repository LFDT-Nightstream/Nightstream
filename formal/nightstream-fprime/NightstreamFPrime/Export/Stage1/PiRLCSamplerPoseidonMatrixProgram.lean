import NightstreamFPrime.Export.MatrixProgram.Program
import NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonPlan

/-!
Owns the compact matrix program for the 34 PiRLC sampler Poseidon2
invocations. The package carries the cross-family previous-state wires and
one optional constant per invocation lane.

This module does not own sampler reduction or coefficient-word rows.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonMatrixProgram

open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec

abbrev Program := Lifecycle.Stage1.Application.Program

def piCcsFinalSlotBase : Nat :=
  (PiCCSPoseidonPlan.invocationCount - 1) * 150 + 134

@[simp] theorem piCcsFinalSlotBase_eq : piCcsFinalSlotBase = 397634 := by
  norm_num [piCcsFinalSlotBase, PiCCSPoseidonPlan.invocationCount_eq]

def constantAt
    (index : Fin (PiRLCSamplerPoseidonPlan.invocationCount * 16)) : Option F :=
  let decoded : Fin PiRLCSamplerPoseidonPlan.invocationCount × Fin 16 :=
    Fin.decodeProd index
  let descriptor := PiRLCSamplerPoseidonPlan.descriptor decoded.1
  if descriptor.2.val = 0 then
    some (PiRLCSamplerPoseidonPlan.entryWord descriptor.1 decoded.2)
  else
    none

def constants : PoseidonInput.OptionalConstantTable :=
  PoseidonInput.OptionalConstantTable.ofSemantic constantAt

def piCcsPreviousRule (program : Program) : PoseidonInput.Rule where
  region := ⟨0, 1, 0, 16⟩
  term := .external
    (RetainedBlock.ofSemantic (PiCCSPoseidonPlan.retainedBlock program)
      (PiCCSPoseidonPlan.retainedStart program)) piCcsFinalSlotBase 0

def samplerPreviousRule (program : Program) : PoseidonInput.Rule where
  region := ⟨1, 33, 0, 16⟩
  term := .external
    (RetainedBlock.ofSemantic (PiRLCSamplerPoseidonPlan.retainedBlock program)
      (PiRLCSamplerPoseidonPlan.retainedStart program)) 134 150

def entryRule : PoseidonInput.Rule where
  region := ⟨0, 34, 0, 16⟩
  term := .optionalConstant constants 16

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
    (block geometry).rowCount = 5100 := by
  calc
    (block geometry).rowCount =
        PiRLCSamplerPoseidonPlan.invocationCount * 150 := by
      exact Poseidon.Block.ofSemantic_rowCount
        (PiRLCSamplerPoseidonPlan.schedule program)
        (PiRLCSamplerPoseidonPlan.retainedStart program)
        (PiRLCSamplerPoseidonPlan.oneColumn geometry) (inputProgram program)
    _ = 5100 := by
      norm_num [PiRLCSamplerPoseidonPlan.invocationCount_eq]

@[simp] theorem matrixProgram_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    (matrixProgram geometry).rowCount = 5100 := by
  rw [show matrixProgram geometry =
      MatrixProgram.Program.mk [.poseidon (block geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_rowCount]
  exact block_rowCount geometry

end NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonMatrixProgram
