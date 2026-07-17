import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.RowAction

namespace Tests.FPrimeFullHistorySelectiveCcsRowAction

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Interpreter
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Rows

private def booleanRelation : InterpretedRelation 1 1 where
  matrices := fun role _row _column => booleanPoint 1 1 role.index

private def booleanAssignment : Assignment F 1 := fun _ => 1

private def onlyRow : Fin 1 := ⟨0, by decide⟩

private theorem booleanImages :
    rowPoint booleanRelation booleanAssignment onlyRow =
      booleanPoint 1 1 := by
  funext port
  have matrixAtValue :
      booleanRelation.matrixAt port onlyRow (0 : Fin 1) =
        booleanPoint 1 1 port := by
    change booleanPoint 1 1 (Role.ofIndex port).index =
      booleanPoint 1 1 port
    rw [Role.index_ofIndex]
  simp [rowPoint, matrixImageAt, matrixVectorAt,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.canonicalFinIndices,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps,
    booleanAssignment, matrixAtValue, Fin.mul_one]

example : residualAt booleanRelation booleanAssignment onlyRow = 0 := by
  rw [residualAt_booleanPoint booleanRelation booleanAssignment onlyRow 1 1
    booleanImages]
  decide

example (port : Fin 13) :
    matrixImageAt booleanRelation booleanAssignment onlyRow port =
      matrixVectorAt
        Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps
        (Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.RowPadding.padRows
          (booleanRelation.matrixAt port))
        booleanAssignment
        (Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.RowPadding.numericRowVertex
          (rowVariables := 1) (by decide) onlyRow) := by
  exact matrixImageAt_eq_paddedMatrixVectorAt
    booleanRelation booleanAssignment (rowVariables := 1) (by decide) onlyRow port

end Tests.FPrimeFullHistorySelectiveCcsRowAction
