import NightstreamFPrime.Export.Stage1.ApplicationDirectPlan
import NightstreamFPrime.Export.Stage1.ApplicationOrdinaryMatrixProgram
import NightstreamFPrime.Export.Stage1.ApplicationPoseidonMatrixExecution

/-! Select the executable matrix program with the same checked application certificate. -/

namespace NightstreamFPrime.Export.Stage1.ApplicationMatrixProgram

open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open ApplicationRetainedGeometry

def matrixProgram {application : Stage1.Application.Program} {columns : Nat}
    (geometry : Geometry application columns) : MatrixProgram.Program :=
  match selected : application.compactHashChain with
  | none => ApplicationOrdinaryMatrixProgram.matrixProgram (ordinaryGeometry geometry selected)
  | some certificate => ApplicationPoseidonMatrixProgram.matrixProgram
      (poseidonGeometry geometry certificate selected)

theorem matrixProgram_none {application : Stage1.Application.Program} {columns : Nat}
    (geometry : Geometry application columns) (selected : application.compactHashChain = none) :
    matrixProgram geometry = ApplicationOrdinaryMatrixProgram.matrixProgram
      (ordinaryGeometry geometry selected) := by
  unfold matrixProgram
  split
  · rfl
  · rename_i certificate found
    rw [selected] at found
    cases found

theorem matrixProgram_some {application : Stage1.Application.Program} {columns : Nat}
    (geometry : Geometry application columns)
    (certificate : ApplicationPoseidonRetainedBlock.Certificate application)
    (selected : application.compactHashChain = some certificate) :
    matrixProgram geometry = ApplicationPoseidonMatrixProgram.matrixProgram
      (poseidonGeometry geometry certificate selected) := by
  unfold matrixProgram
  split
  · rename_i found
    rw [selected] at found
    cases found
  · rename_i candidate found
    have same : certificate = candidate := Option.some.inj (selected.symm.trans found)
    subst candidate
    rfl

@[simp] theorem matrixProgram_rowCount {application : Stage1.Application.Program} {columns : Nat}
    (geometry : Geometry application columns) :
    (matrixProgram geometry).rowCount = ApplicationDirectPlan.rowCount application := by
  cases selected : application.compactHashChain with
  | none => rw [matrixProgram_none geometry selected, ApplicationOrdinaryMatrixProgram.matrixProgram_rowCount,
      ApplicationDirectPlan.rowCount, selected]
  | some certificate => rw [matrixProgram_some geometry certificate selected,
      ApplicationPoseidonMatrixProgram.matrixProgram_rowCount, ApplicationDirectPlan.rowCount, selected]

end NightstreamFPrime.Export.Stage1.ApplicationMatrixProgram
