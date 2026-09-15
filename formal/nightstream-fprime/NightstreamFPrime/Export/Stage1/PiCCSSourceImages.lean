import NightstreamFPrime.Export.Stage1.PiDECMatrixNumericRows
import NightstreamFPrime.Export.Stage1.PiDECParentSparseRead
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1MatrixRows

/-! Original-assignment reads for PiCCS source images. The caller shares the
existing compact program and source accessor. No digit split is performed.
Preservation proofs are in PiCCSSourceImagesPreservation, outside the runner. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSSourceImages

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace

abbrev logicalWidth :=
  PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application
abbrev publicFits :=
  PerApplicationFixedPoint.publicFits Poseidon2HashChainV1Package.application
abbrev shape := PaperAlgebra.FullShape logicalWidth publicFits
abbrev blockCount := Phi81ColumnLayout.blockCount shape.carrierWidth

/-- Read the original complete block, including all retained tail lanes. -/
def blockAt (assignment : Phi81Relation.Assignment shape) (block : Nat) : RingF :=
  if live : block < blockCount then
    CarrierAction.assignmentBlock assignment ⟨block, live⟩
  else ringFZero

def kernelRead {columns : Nat} (assignment : Phi81Relation.Assignment shape)
    (output : Fin ringDegree) (column : Fin columns) : F :=
  CarrierAction.kernelImage
    ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩
    (blockAt assignment (column.val / ringDegree)) output

/-- Reuse the exact prepared basis forms on the original source block.
The caller prepares tables once and shares them across source/row reads. -/
def preparedRead {columns : Nat}
    (tables : FixedArray (FixedArray (SparseForm ringDegree) ringDegree) ringDegree)
    (assignment : Phi81Relation.Assignment shape) (output : Fin ringDegree)
    (column : Fin columns) : F :=
  ((tables.get ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩).get output).evalSparse
    (blockAt assignment (column.val / ringDegree))

/-- The original scalar assignment read for the fresh CCS matrices. -/
def plainRead (assignment : Phi81Relation.Assignment shape)
    (column : Fin logicalWidth) : F :=
  assignment (Phi81CarrierLayout.embedLogical column)

/-- Only rows beyond the complete program are zero. An active loader failure
remains none; the selected preservation theorem discharges successful loads. -/
def rowValues? (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row) (read : Fin logicalWidth → F)
    (vertex : BooleanVertex Lifecycle.cubeVariables) : Option (Vector F Spec.ProductionRelation.matrixCount) :=
  let index := NumericBooleanDomain.index vertex
  if index < program.rowCount then
    PiDECMatrixNumericRows.row? program sourceRow read index
  else some (Vector.replicate Spec.ProductionRelation.matrixCount 0)

def matrixImage? (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row) (assignment : Phi81Relation.Assignment shape)
    (output : Fin ringDegree) (vertex : BooleanVertex Lifecycle.cubeVariables) :
    Option (Vector F Spec.ProductionRelation.matrixCount) :=
  rowValues? program sourceRow (kernelRead assignment output) vertex

def freshMatrixImage? (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row) (assignment : Phi81Relation.Assignment shape)
    (vertex : BooleanVertex Lifecycle.cubeVariables) :
    Option (Vector F Spec.ProductionRelation.matrixCount) :=
  rowValues? program sourceRow (plainRead assignment) vertex

def padImage (layout : UnifiedSources.ColumnLayout Lifecycle.cubeVariables shape.carrierWidth)
    (assignment : Phi81Relation.Assignment shape) (output : Fin ringDegree)
    (vertex : BooleanVertex Lifecycle.cubeVariables) : F :=
  (match layout.toColumn? vertex with
    | some column => SparseForm.singleton column 1
    | none => SparseForm.empty).evalSparse (kernelRead assignment output)

def assignmentValue (layout : UnifiedSources.ColumnLayout Lifecycle.cubeVariables shape.carrierWidth)
    (assignment : Phi81Relation.Assignment shape)
    (vertex : BooleanVertex Lifecycle.cubeVariables) : F :=
  layout.paddedValue 0 assignment vertex

/-- The prepared basis read at the layout's Pad column, or its exact zero suffix. -/
def preparedPadImage
    (tables : FixedArray (FixedArray (SparseForm ringDegree) ringDegree) ringDegree)
    (layout : ColumnLayout Lifecycle.cubeVariables shape.carrierWidth)
    (assignment : Phi81Relation.Assignment shape) (output : Fin ringDegree)
    (vertex : BooleanVertex Lifecycle.cubeVariables) : F :=
  match layout.toColumn? vertex with
  | some column => preparedRead tables assignment output column
  | none => 0

/-- Assemble the four first-round image families from the original sources.
All matrix calls use the numeric interpreter; any failed call returns none. -/
def images? (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row)
    (tables : FixedArray (FixedArray (SparseForm ringDegree) ringDegree) ringDegree)
    (layout : UnifiedSources.ColumnLayout Lifecycle.cubeVariables shape.carrierWidth)
    (assignments : Fin productionShape.sourceCount → Phi81Relation.Assignment shape)
    (vertex : BooleanVertex Lifecycle.cubeVariables) :
    Option (ProtocolPolynomial.OutputMessage K productionShape) := do
  let fresh ← freshMatrixImage? program sourceRow
    (assignments (freshSourceIndex ⟨0, by decide⟩)) vertex
  let matrices ← Vector.ofFnM fun source : Fin productionShape.runningCount =>
    Vector.ofFnM fun output : Fin ringDegree =>
      rowValues? program sourceRow
        (preparedRead tables (assignments (runningSourceIndex source)) output) vertex
  let pad := Vector.ofFn fun source : Fin productionShape.runningCount =>
    Vector.ofFn fun output : Fin ringDegree =>
      preparedPadImage tables layout (assignments (runningSourceIndex source)) output vertex
  let norm := Vector.ofFn fun source : Fin productionShape.sourceCount =>
    assignmentValue layout (assignments source) vertex
  return {
    freshMatrixImage := fun _ matrix => K.embed (fresh.get matrix)
    sourceAssignment := fun source => K.embed (norm.get source)
    padImage := fun coordinate =>
      K.embed ((pad.get coordinate.running).get coordinate.coefficient)
    matrixImage := fun coordinate => K.embed
      (((matrices.get coordinate.running).get coordinate.coefficient).get coordinate.matrix) }

end NightstreamFPrime.Export.Stage1.PiCCSSourceImages
