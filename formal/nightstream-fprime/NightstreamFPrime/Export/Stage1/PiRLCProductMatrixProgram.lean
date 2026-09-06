import NightstreamFPrime.Export.MatrixProgram.Program
import NightstreamFPrime.Export.Stage1.PiRLCRetainedInputs
import NightstreamFPrime.Export.Stage1.PiRLCValueMatrixProgram

/-!
Owns the compact matrix-program block for the four canonical PiRLC Phi81
product families. Family order and every retained-block operand are Lean
data. A consumer does not select family bounds, row order, or columns.

This module does not assemble later PiRLC or Stage 1 blocks.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCProductMatrixProgram

open NightstreamFPrime.Export
open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

abbrev ProductFamily := MatrixProgram.Phi81Product.Family

def commitmentFamily : ProductFamily :=
  { sourceCount := 17, blockCount := 22, cellCount := 1 }

def publicInputFamily : ProductFamily :=
  { sourceCount := 17, blockCount := 5, cellCount := 1 }

def evalKFamily : ProductFamily :=
  { sourceCount := 17, blockCount := 1, cellCount := 2 }

def evalAFamily : ProductFamily :=
  { sourceCount := 17, blockCount := 14, cellCount := 2 }

/-- Exact SuperNeo family order: commitment, public input, Eval_K, Eval_A. -/
def families : List ProductFamily :=
  [commitmentFamily, publicInputFamily, evalKFamily, evalAFamily]

@[simp] theorem families_invocationCount :
    MatrixProgram.Phi81Product.invocationCount families = 52326 := by
  norm_num [families, MatrixProgram.Phi81Product.invocationCount,
    commitmentFamily, publicInputFamily, evalKFamily, evalAFamily,
    MatrixProgram.Phi81Product.Family.invocationCount,
    MatrixProgram.Phi81Product.Family.privateCount,
    CombinationStep.privateCount, ringDegree]

/-- Final First54 value slot for source zero. -/
def challengeSlotStart : Nat := 63 * 54

/-- Distance between final First54 value rows of adjacent sources. -/
def challengeSourceStride : Nat := 64 * 54

def prefixGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    PiRLCRetainedGeometry.Geometry program logicalWidth :=
  PiCCSPoseidonPlan.prefixGeometry <|
    RunningTransitionRetainedGeometry.poseidonGeometry <|
      PiCCSOrdinaryRetainedGeometry.prefixGeometry geometry

def inputs {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    PiRLCProductPlan.Inputs program logicalWidth :=
  PiRLCRetainedInputs.productInputs (PiRLCValueWiring.form geometry)
    (prefixGeometry geometry)

/-- One compact product-family block over the canonical retained assignment. -/
def block {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    MatrixProgram.Phi81Product.Block where
  families := families
  oneColumn := (PiRLCRetainedGeometry.oneColumn (prefixGeometry geometry)).val
  challenge := MatrixProgram.RetainedBlock.ofSemantic
    (PiRLCFirst54RetainedBlocks.valueBlock program)
    (PiRLCRetainedGeometry.valueStart program)
  challengeSlotStart := challengeSlotStart
  challengeSourceStride := challengeSourceStride
  input := PiRLCValueMatrixProgram.substitution program
  output := MatrixProgram.RetainedBlock.ofSemantic
    (PiRLCRetainedGeometry.productOutputBlock program)
    (PiRLCRetainedGeometry.productOutputStart program)
  group := MatrixProgram.RetainedBlock.ofSemantic
    (PiRLCRetainedGeometry.productGroupBlock program)
    (PiRLCRetainedGeometry.productGroupStart program)

@[simp] theorem block_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    (block geometry).rowCount = 1779084 := by
  norm_num [block, MatrixProgram.Phi81Product.Block.rowCount,
    MatrixProgram.Phi81Product.Block.invocationCount]

def matrixProgram {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    MatrixProgram.Program where
  blocks := [.phi81Product (block geometry)]

@[simp] theorem matrixProgram_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    (matrixProgram geometry).rowCount = 1779084 := by
  rw [show matrixProgram geometry =
      MatrixProgram.Program.mk [.phi81Product (block geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_rowCount]
  exact block_rowCount geometry

end NightstreamFPrime.Export.Stage1.PiRLCProductMatrixProgram
