import NightstreamFPrime.Export.Stage1.PiCCSCarriedRead
import NightstreamFPrime.Export.Stage1.PiCCSLinearRows
import NightstreamFPrime.Export.Stage1.PiCCSSourceImages

/-! Compute original-source PiCCS endpoints with the linear carried terms
aggregated before matrix evaluation. The endpoint message carries the fresh
matrix images and all original norm inputs. Its unused carried fields are
zero; the two returned scalars contain the complete carried sums. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSAggregatedImages

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle

/-- Combine every original running source at the selected complete block. -/
def combinedBlock (powers : Nat → K)
    (assignments : Fin productionShape.sourceCount →
      Phi81Relation.Assignment PiCCSSourceImages.shape)
    (block : Nat) : Vector K ringDegree :=
  PiCCSCarriedRead.combine
    (fun source : Fin productionShape.runningCount => powers source.val)
    (fun source => PiCCSSourceImages.blockAt
      (assignments (runningSourceIndex source)) block)

/-- Prepare the two exact output-weight families; the matrix slot weight
is applied after row evaluation and the global shift remains in the pair. -/
def prepare
    (forms : FixedArray (FixedArray (ProductionRelation.SparseForm ringDegree) ringDegree) ringDegree)
    (powers : Nat → K) :
    FixedArray (Vector K ringDegree) ringDegree × FixedArray (Vector K ringDegree) ringDegree :=
  (PiCCSCarriedRead.prepare forms
      (fun output => powers (productionShape.runningCount * output.val)),
   PiCCSCarriedRead.prepare forms
      (fun output => powers
        (productionShape.runningCount * productionShape.matrixCount * output.val)))

/-- Preserve the fresh and norm fields without evaluating the carried
families. The pair kernel receives the carried totals separately. -/
def nonlinearMessage
    (layout : ColumnLayout cubeVariables PiCCSSourceImages.shape.carrierWidth)
    (assignments : Fin productionShape.sourceCount →
      Phi81Relation.Assignment PiCCSSourceImages.shape)
    (vertex : BooleanVertex cubeVariables)
    (fresh : Vector F Spec.ProductionRelation.matrixCount) :
    ProtocolPolynomial.OutputMessage K productionShape where
  freshMatrixImage := fun _ matrix => K.embed (fresh.get matrix)
  sourceAssignment := fun source =>
    K.embed (PiCCSSourceImages.assignmentValue layout (assignments source) vertex)
  padImage := fun _ => K.zero
  matrixImage := fun _ => K.zero

/-- Assemble an endpoint from its computed rows. A caller may reuse the
stored numeric invocation that supplies these same row values. -/
def fromRows
    (layout : ColumnLayout cubeVariables PiCCSSourceImages.shape.carrierWidth)
    (assignments : Fin productionShape.sourceCount →
      Phi81Relation.Assignment PiCCSSourceImages.shape)
    (padBasis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree) (powers : Nat → K)
    (vertex : BooleanVertex cubeVariables)
    (fresh : Vector F Spec.ProductionRelation.matrixCount)
    (matrices : Vector K Spec.ProductionRelation.matrixCount) :
    ProtocolPolynomial.OutputMessage K productionShape × K × K :=
  let pad := match layout.toColumn? vertex with
    | some column => PiCCSCarriedRead.read padBasis blocks column
    | none => K.zero
  let matrix := sumMap extensionOps (canonicalFinIndices productionShape.matrixCount)
    (fun slot => extensionOps.mul (powers (productionShape.runningCount * slot.val))
      (matrices.get slot))
  (nonlinearMessage layout assignments vertex fresh, pad, matrix)

/-- Compute one endpoint from original assignments. Active row failures
remain failures. The exact padded suffix contributes zero matrix values. -/
def endpoint? (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row)
    (layout : ColumnLayout cubeVariables PiCCSSourceImages.shape.carrierWidth)
    (assignments : Fin productionShape.sourceCount →
      Phi81Relation.Assignment PiCCSSourceImages.shape)
    (padBasis matrixBasis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree) (powers : Nat → K)
    (vertex : BooleanVertex cubeVariables) :
    Option (ProtocolPolynomial.OutputMessage K productionShape × K × K) := do
  let fresh ← PiCCSSourceImages.freshMatrixImage? program sourceRow
    (assignments (freshSourceIndex ⟨0, by decide⟩)) vertex
  let index := NumericBooleanDomain.index vertex
  let matrices ← if index < program.rowCount then
      PiCCSLinearRows.row? program (columns := PiCCSSourceImages.logicalWidth)
        sourceRow (PiCCSCarriedRead.read matrixBasis blocks) index
    else some (Vector.replicate Spec.ProductionRelation.matrixCount K.zero)
  return fromRows layout assignments padBasis blocks powers vertex fresh matrices

end NightstreamFPrime.Export.Stage1.PiCCSAggregatedImages
