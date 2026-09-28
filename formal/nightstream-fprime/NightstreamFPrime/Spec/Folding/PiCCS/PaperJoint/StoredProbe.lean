import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork

/-!
Stored PiCCS output coefficients with the existing coins and raw certificate.
Erasure preserves every certificate coefficient, including malformed widths.
The producing call owns array creation. Reads count record projection, array
lookups, and result construction in the existing named-operation model.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

open StrongReduction
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

structure StoredProbe (shape : Shape) where
  coins : PublicCoins K shape
  certificate : SumCheck.Finite.Certificate K
  pad : Vector (Vector K shape.coefficientCount) shape.sourceCount
  matrix : Vector (Vector (Vector K shape.coefficientCount) shape.matrixCount) shape.sourceCount

namespace StoredProbe

variable {shape : Shape}

/-- Erase only output storage. Coins and raw messages are the same values. -/
def view (probe : StoredProbe shape) : Probe K shape where
  coins := probe.coins
  response := {
    rounds := probe.certificate
    fullOutput := {
      padCoordinate := fun source coefficient => (probe.pad.get source).get coefficient
      matrixCoordinate := fun source matrix coefficient =>
        ((probe.matrix.get source).get matrix).get coefficient } }

/-- One record projection, two array lookups, and the returned result. -/
def padRead (probe : StoredProbe shape) (source : Fin shape.sourceCount)
    (coefficient : Fin shape.coefficientCount) : Result K :=
  let family : Result (Vector (Vector K shape.coefficientCount) shape.sourceCount) := ⟨probe.pad, 1⟩
  let row : Result (Vector K shape.coefficientCount) := ⟨family.value.get source, 1⟩
  let value : Result K := ⟨row.value.get coefficient, 1⟩
  ⟨value.value, family.work + row.work + value.work + 1⟩

/-- One record projection, three array lookups, and the returned result. -/
def matrixRead (probe : StoredProbe shape) (source : Fin shape.sourceCount)
    (matrix : Fin shape.matrixCount) (coefficient : Fin shape.coefficientCount) : Result K :=
  let family : Result (Vector (Vector (Vector K shape.coefficientCount) shape.matrixCount) shape.sourceCount) :=
    ⟨probe.matrix, 1⟩
  let sourceRows : Result (Vector (Vector K shape.coefficientCount) shape.matrixCount) :=
    ⟨family.value.get source, 1⟩
  let row : Result (Vector K shape.coefficientCount) := ⟨sourceRows.value.get matrix, 1⟩
  let value : Result K := ⟨row.value.get coefficient, 1⟩
  ⟨value.value, family.work + sourceRows.work + row.work + value.work + 1⟩

theorem padRead_value (probe : StoredProbe shape) (source : Fin shape.sourceCount)
    (coefficient : Fin shape.coefficientCount) :
    (padRead probe source coefficient).value = probe.view.response.fullOutput.padCoordinate source coefficient := rfl

theorem padRead_work (probe : StoredProbe shape) (source : Fin shape.sourceCount)
    (coefficient : Fin shape.coefficientCount) : (padRead probe source coefficient).work = 4 := rfl

theorem matrixRead_value (probe : StoredProbe shape) (source : Fin shape.sourceCount)
    (matrix : Fin shape.matrixCount) (coefficient : Fin shape.coefficientCount) :
    (matrixRead probe source matrix coefficient).value =
      probe.view.response.fullOutput.matrixCoordinate source matrix coefficient := rfl

theorem matrixRead_work (probe : StoredProbe shape) (source : Fin shape.sourceCount)
    (matrix : Fin shape.matrixCount) (coefficient : Fin shape.coefficientCount) :
    (matrixRead probe source matrix coefficient).work = 5 := rfl

end StoredProbe

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
