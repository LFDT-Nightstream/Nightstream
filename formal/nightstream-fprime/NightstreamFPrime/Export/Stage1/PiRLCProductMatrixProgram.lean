import NightstreamFPrime.Export.MatrixProgram.Program
import NightstreamFPrime.Export.Stage1.PiRLCRetainedInputs
import NightstreamFPrime.Export.Stage1.PiRLCValueMatrixProgram

namespace NightstreamFPrime.Export.Stage1.PiRLCProductMatrixProgram

open NightstreamFPrime.Export
open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle.PiRLC.v1_2
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

abbrev ProductFamily := MatrixProgram.RingProduct.Family

def commitmentFamily : ProductFamily :=
  { sourceCount := 17, blockCount := 18, cellCount := 1 }

def publicInputFamily : ProductFamily :=
  { sourceCount := 17, blockCount := 5, cellCount := 1 }

def evalKFamily : ProductFamily :=
  { sourceCount := 17, blockCount := 1, cellCount := 2 }

def evalAFamily : ProductFamily :=
  { sourceCount := 17, blockCount := productionProfile.ccsMatrices, cellCount := 2 }

def families : List ProductFamily :=
  [commitmentFamily, publicInputFamily, evalKFamily, evalAFamily]

@[simp] theorem families_invocationCount :
    MatrixProgram.RingProduct.invocationCount families = 35904 := by
  norm_num [families, MatrixProgram.RingProduct.invocationCount,
    commitmentFamily, publicInputFamily, evalKFamily, evalAFamily,
    MatrixProgram.RingProduct.Family.invocationCount,
    MatrixProgram.RingProduct.Family.privateCount,
    CombinationStep.privateCount, ringDegree, productionProfile]

@[simp] theorem families_ringCount :
    MatrixProgram.RingProduct.ringCount families = 561 := by
  norm_num [families, MatrixProgram.RingProduct.ringCount,
    MatrixProgram.RingProduct.Family.ringCount,
    commitmentFamily, publicInputFamily, evalKFamily, evalAFamily, productionProfile]

def challengeSlotStart : Nat := 0

def challengeSourceStride : Nat := 64

theorem prefixGeometry {program : Lifecycle.Stage1.Application.Program}
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

def block {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    MatrixProgram.RingProduct.Block where
  families := families
  oneColumn := (PiRLCRetainedGeometry.oneColumn (prefixGeometry geometry)).val
  challenge := MatrixProgram.RetainedBlock.ofSemantic
    (PiRLCRetainedGeometry.challengeBlock program)
    (PiRLCRetainedGeometry.challengeStart program)
  challengeSlotStart := challengeSlotStart
  challengeSourceStride := challengeSourceStride
  input := PiRLCValueMatrixProgram.substitution program
  output := MatrixProgram.RetainedBlock.ofSemantic
    (PiRLCRetainedGeometry.productOutputBlock program)
    (PiRLCRetainedGeometry.productOutputStart program)
  quotient := MatrixProgram.RetainedBlock.ofSemantic
    (PiRLCRetainedGeometry.productGroupBlock program)
    (PiRLCRetainedGeometry.productGroupStart program)

@[simp] theorem block_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    (block geometry).rowCount = 71808 := by
  norm_num [block, MatrixProgram.RingProduct.Block.rowCount]

def matrixProgram {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    MatrixProgram.Program where
  blocks := [.ringProduct (block geometry)]

@[simp] theorem matrixProgram_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    (matrixProgram geometry).rowCount = 71808 := by
  rw [show matrixProgram geometry =
      MatrixProgram.Program.mk [.ringProduct (block geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_rowCount]
  exact block_rowCount geometry

end NightstreamFPrime.Export.Stage1.PiRLCProductMatrixProgram
