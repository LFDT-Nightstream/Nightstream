import NightstreamFPrime.Export.Stage1.PiDECMatrixNumericRows
import NightstreamFPrime.Export.Stage1.PiDECEvaluationBlockSupport

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECMatrixBatchBoundary

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Layout.ProductionRelation

theorem row?_kernel_value (program : MatrixProgram.Program) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row)
    (children : Nat → Vector StoredRing productionGlobalParams.k)
    (ordinal : Nat) (child : Fin productionGlobalParams.k)
    (port : Fin matrixCount) (output : Fin ringDegree) :
    let read : Fin columns → F := fun column =>
      CarrierAction.kernelImage
        ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩
        ((children (column.val / ringDegree)).get child).get output
    (PiDECMatrixNumericRows.row? program sourceRow read ordinal).map
        (fun values => values.get port) =
      (program.row? columns sourceRow ordinal).map (fun forms =>
        ((PiDECEvaluationBlockSupport.kernel
          (match meaningfulPort? port with
            | some meaningful => forms meaningful
            | none => SparseForm.empty) children).get child).get output) := by
  dsimp only
  rw [PiDECMatrixNumericRows.row?_value]
  cases program.row? columns sourceRow ordinal with
  | none => rfl
  | some forms =>
      exact congrArg some
        (PiDECEvaluationBlockSupport.kernel_eq_evalSparse _ children child output).symm

theorem storedInvocation_row (block : Poseidon.Block) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → F)
    (invocation : Fin block.invocationCount) (row : Fin 150) :
    ((PiDECPoseidonNumericBlock.loadInvocation? block columns invocation).map
      (PiDECPoseidonNumericRows.stored read)).map
        (fun rows => Vector.ofFn (rows.get row).get) =
      PiDECMatrixNumericRows.blockRow? (.poseidon block) sourceRow read
        (Fin.encodeProd (invocation, row)).val := by
  simp only [PiDECMatrixNumericRows.blockRow?, PiDECPoseidonNumericBlock.row?,
    PiDECPoseidonNumericBlock.loadRow?_encodeProd, Option.map_map, Function.comp_def]

end NightstreamFPrime.Export.Stage1.PiDECMatrixBatchBoundary
