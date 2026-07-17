import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Types

/-!
Source-to-verifier projection for production-shaped Split-NC `Pi_CCS`.

Owns: erasure of the rich independent matrix/assignment source model into the
raw verifier-visible `PublicInput`, preserving the three public families
exactly.

Does not own: the raw carrier definitions, FE/NC polynomials, output claims,
SumCheck, transcript execution, Rust, R1CS, rows, costs, or row removal.

Emits constraints: no.

Authority boundary: this module is the only semantic source dependency of the
projection. Core verifier carriers and transcript interfaces must import
`Verifier.Types` without importing this module.

| Stage path | Source field | Verifier field | Mathematical obligation |
|---|---|---|---|
| `nifs.pi_ccs.verify.input.structure` | `Data.constraintPolynomial` | `PublicInput.constraintPolynomial` | exact dataflow |
| `nifs.pi_ccs.verify.input.prior_point` | `Data.priorPoint` | `PublicInput.priorPoint` | exact dataflow |
| `nifs.pi_ccs.verify.input.running` | `Data.claimedCoefficient` | `PublicInput.claimedYRing` | canonical running/matrix/lane reindexing |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources

namespace PublicInput

/-- Erase every hidden matrix and assignment from the executable verifier
surface. The running carried coefficients are reindexed into their explicit
source/matrix/lane product without changing their values. -/
def ofSources
    {shape : SemanticShape}
    (data : Data shape) : PublicInput shape where
  constraintPolynomial := data.constraintPolynomial
  priorPoint := data.priorPoint
  claimedYRing := fun running matrix lane =>
    data.claimedCoefficient {
      running := running
      matrix := matrix
      coefficient := lane
    }

@[simp] theorem ofSources_claimedYRing
    {shape : SemanticShape}
    (data : Data shape)
    (running : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    (ofSources data).claimedYRing running matrix lane =
      data.claimedCoefficient {
        running := running
        matrix := matrix
        coefficient := lane
      } := by
  rfl

/-- Rich semantic sources with the same public structure and prior claims
project to the same executable input, regardless of all matrices and
assignments. -/
theorem ofSources_eq
    {shape : SemanticShape}
    (left right : Data shape)
    (constraintPolynomial :
      left.constraintPolynomial = right.constraintPolynomial)
    (priorPoint : left.priorPoint = right.priorPoint)
    (claimedCoefficient :
      left.claimedCoefficient = right.claimedCoefficient) :
    ofSources left = ofSources right := by
  apply PublicInput.ext
  · exact constraintPolynomial
  · exact priorPoint
  · funext running matrix lane
    exact congrFun claimedCoefficient {
      running := running
      matrix := matrix
      coefficient := lane
    }

end PublicInput

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
