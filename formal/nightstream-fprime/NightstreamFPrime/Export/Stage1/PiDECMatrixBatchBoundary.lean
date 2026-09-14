import NightstreamFPrime.Export.Stage1.PiDECMatrixNumericRows
import NightstreamFPrime.Export.Stage1.PiDECEvaluationBlockSupport

/-!
Two local batch links: numeric matrix rows equal the existing complete
Phi81 sparse kernel, and an existing stored Poseidon invocation supplies
each of its numeric block rows. No source or witness validity is assumed.
-/

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

/-- The numeric interface and existing sparse block kernel compute the same
coefficient for every child and every matrix port, with the same optional
failure. The complete child block function is arbitrary. -/
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

/-- Load one existing interface and compute all 94 port records once.
Every stored row equals the original numeric block row at the existing
Fin product index, including rejection of the invocation interface. -/
theorem storedInvocation_row (block : Poseidon.Block) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row) (read : Fin columns → F)
    (invocation : Fin block.invocationCount) (row : Fin 94) :
    ((PiDECPoseidonNumericBlock.loadInvocation? block columns invocation).map
      (PiDECPoseidonNumericRows.stored read)).map
        (fun rows => Vector.ofFn (rows.get row).get) =
      PiDECMatrixNumericRows.blockRow? (.poseidon block) sourceRow read
        (Fin.encodeProd (invocation, row)).val := by
  simp only [PiDECMatrixNumericRows.blockRow?, PiDECPoseidonNumericBlock.row?,
    PiDECPoseidonNumericBlock.loadRow?_encodeProd, Option.map_map, Function.comp_def]

end NightstreamFPrime.Export.Stage1.PiDECMatrixBatchBoundary
