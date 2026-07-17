import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact

namespace NightstreamTests.FPrimeFullHistorySelectiveCcsArtifact

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Schema
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Interpreter
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports

private def validCsc : CscPayload where
  colPtr := [0, 0, 1, 2]
  rowIdx := [1, 0]
  vals := [7, 9]

private def compactMatrix : CompactMatrix where
  csc := validCsc
  seededBlocks := []
  geometricRuns := []

private def rawBundle : Bundle where
  rows := 2
  columns := 3
  matrices := List.replicate 13 compactMatrix

private def validatedBundle : ValidatedBundle where
  raw := rawBundle
  valid := by
    constructor
    · decide
    · decide
    · native_decide
    · intro matrix membership
      have matrixEq : matrix = compactMatrix := by
        simpa [rawBundle] using membership
      subst matrix
      constructor
      · native_decide
      · simp [compactMatrix]
      · simp [compactMatrix]

example :
    (ValidatedBundle.interpretRelation validatedBundle).matrices Role.bit
      ⟨1, by decide⟩ ⟨1, by decide⟩ = 7 := by
  native_decide

example :
    (ValidatedBundle.interpretRelation validatedBundle).matrices Role.evalTailRight
      ⟨0, by decide⟩ ⟨2, by decide⟩ = 9 := by
  native_decide

example :
    (ValidatedBundle.interpretRelation validatedBundle).matrices Role.evalTailRight
      ⟨1, by decide⟩ ⟨2, by decide⟩ = 0 := by
  native_decide

private def nonmonotonePointers : CscPayload where
  colPtr := [0, 2, 1]
  rowIdx := [0, 1]
  vals := [7, 9]

example : ¬ nonmonotonePointers.Valid 2 2 := by
  native_decide

private def noncanonicalRowOrder : CscPayload where
  colPtr := [0, 2]
  rowIdx := [1, 0]
  vals := [7, 9]

example : ¬ noncanonicalRowOrder.Valid 2 1 := by
  native_decide

private def explicitZero : CscPayload where
  colPtr := [0, 1]
  rowIdx := [0]
  vals := [0]

example : ¬ explicitZero.Valid 1 1 := by
  native_decide

private def rowIndexOutsideU32 : CscPayload where
  colPtr := [0, 1]
  rowIdx := [2 ^ 32]
  vals := [1]

example : ¬ rowIndexOutsideU32.Valid (2 ^ 32 + 1) 1 := by
  native_decide

private def wrongPointerLength : CscPayload where
  colPtr := [0]
  rowIdx := []
  vals := []

example : ¬ wrongPointerLength.Valid 1 1 := by
  native_decide

private def nonzeroPointerHead : CscPayload where
  colPtr := [1, 1]
  rowIdx := [0]
  vals := [1]

example : ¬ nonzeroPointerHead.Valid 1 1 := by
  native_decide

private def terminalPointerMismatch : CscPayload where
  colPtr := [0, 0]
  rowIdx := [0]
  vals := [1]

example : ¬ terminalPointerMismatch.Valid 1 1 := by
  native_decide

private def parallelArrayMismatch : CscPayload where
  colPtr := [0, 1]
  rowIdx := []
  vals := [1]

example : ¬ parallelArrayMismatch.Valid 1 1 := by
  native_decide

private def duplicateRowIndex : CscPayload where
  colPtr := [0, 2]
  rowIdx := [0, 0]
  vals := [1, 2]

example : ¬ duplicateRowIndex.Valid 1 1 := by
  native_decide

private def rowOutsideMatrix : CscPayload where
  colPtr := [0, 1]
  rowIdx := [1]
  vals := [1]

example : ¬ rowOutsideMatrix.Valid 1 1 := by
  native_decide

private def pointerOutsideU32 : CscPayload where
  colPtr := [0, 2 ^ 32]
  rowIdx := []
  vals := []

example : ¬ pointerOutsideU32.Valid 1 1 := by
  native_decide

example :
    (ValidatedBundle.interpretRelation validatedBundle).constraintPolynomial =
      gatePolynomial := by
  rfl

private def geometricRun : GeometricRowRun where
  row := 0
  columnStart := 1
  length := 3
  initial := 2
  ratio := 3

example :
    GeometricRowRun.valueAt geometricRun (⟨0, by decide⟩ : Fin 1)
      (⟨3, by decide⟩ : Fin 4) = 18 := by
  native_decide

end NightstreamTests.FPrimeFullHistorySelectiveCcsArtifact
