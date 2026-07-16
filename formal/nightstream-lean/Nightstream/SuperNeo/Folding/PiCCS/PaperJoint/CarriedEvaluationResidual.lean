import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanHypercubeSum
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.TargetPolynomial

/-!
Concrete carried-evaluation residuals for the paper-level joint `Pi_CCS` model.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: construction of the carried `Eval(X, C)` coefficient family.
Constraint family: one prior-CE evaluation equation per typed
`(running source, matrix, ring coefficient)` coordinate.

Owns: explicit running assignments, coefficient-expanded Boolean matrices,
derived matrix-image tables, explicit equality-weighted MLEs at the prior
point, claimed-minus-derived residuals, and single/batch truth equivalence.

Does not own: the concrete base-to-extension lift, proof that coefficient-
expanded matrices refine cyclotomic ring multiplication, external row/bit
serialization, the target exponent repair, placement inside signed joint `Q`,
SumCheck, Fiat--Shamir, Rust, R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: the claimed coefficient is public statement data. The
comparison value is not prover-supplied: it is derived from explicit matrices
and assignments through the shared finite dot product, lifted to the extension
carrier, and evaluated by the explicit `sum_x eq(x,r) * value(x)` theorem.
The lift itself is an explicit later-instantiation boundary and carries no
unproved homomorphism claim here.

| Code owner | Paper object | Derived mathematical value | Proven result |
|---|---|---|---|
| `EvaluationData` | running `z_i`, prior `r`, claimed `cf(y_i,j)_l` | typed public/secret data | exact `k*t*d` coordinate family |
| `imageCoefficientAt` | `cf((M_j z_i)(x))_l` | shared finite matrix-vector row, then explicit lift | no evaluator supplied |
| `computedCoefficient` | `sum_x eq(x,r) * cf((M_j z_i)(x))_l` | explicit canonical hypercube sum | equals recursive table MLE |
| `residual` | Equation (9) orientation | claimed minus computed | zero iff the evaluation equation holds |
| `allResidualsZero_iff_allClaimsHold` | Lemma 7 Item 3 | every carried coordinate | unconditional relative to explicit algebra/lift data |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CarriedEvaluationResidual

universe uBase uExtension

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open PaperLinearAlgebra

/-- Explicit paper-level data for every carried evaluation coordinate. Base
assignments and matrices are kept separate from the extension-field point and
claimed coefficients. -/
structure EvaluationData
    (Base : Type uBase)
    (Extension : Type uExtension)
    (shape : Shape)
    (columns : Nat) where
  priorPoint : CubePoint Extension shape.cubeVariables
  assignments : Fin shape.runningCount -> Assignment Base columns
  coefficientMatrices :
    Fin shape.matrixCount -> Fin shape.coefficientCount ->
      BooleanMatrix Base shape.cubeVariables columns
  claimedCoefficient : CarriedCoordinate shape -> Extension

/-- The carried target polynomial consumes exactly the public coefficients in
`EvaluationData`; there is no second caller-selected target family. -/
def EvaluationData.targetCoefficients
    {Base : Type uBase}
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns) :
    TargetPolynomial.CarriedTargetCoefficients Extension shape where
  coefficient := data.claimedCoefficient

/-- The coefficient of `(M_j z_i)(x)` selected by one typed carried
coordinate, derived from explicit matrix and assignment data. -/
def imageCoefficientAt
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (liftCoefficient : Base -> Extension)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns)
    (coordinate : CarriedCoordinate shape)
    (vertex : BooleanVertex shape.cubeVariables) : Extension :=
  liftCoefficient <|
    matrixVectorAt baseOps
      (data.coefficientMatrices coordinate.matrix coordinate.coefficient)
      (data.assignments coordinate.running)
      vertex

/-- Canonical Boolean table for one carried matrix-image coefficient. -/
def imageTable
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (liftCoefficient : Base -> Extension)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns)
    (coordinate : CarriedCoordinate shape) :
    BooleanTable Extension shape.cubeVariables :=
  BooleanTable.tabulate <|
    imageCoefficientAt baseOps liftCoefficient data coordinate

/-- The prior evaluation computed by the explicit equality-weighted hypercube
sum, not by a caller-provided evaluator. -/
def computedCoefficient
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (extensionOps : InterpolationOps Extension)
    (liftCoefficient : Base -> Extension)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns)
    (coordinate : CarriedCoordinate shape) : Extension :=
  (imageTable baseOps liftCoefficient data coordinate).equalityWeightedSum
    extensionOps data.priorPoint

/-- The recursive table MLE computes exactly the explicit paper hypercube sum
used above. -/
theorem imageTable_evaluate_eq_computedCoefficient
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (extensionOps : InterpolationOps Extension)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (liftCoefficient : Base -> Extension)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns)
    (coordinate : CarriedCoordinate shape) :
    (imageTable baseOps liftCoefficient data coordinate).evaluate
        extensionOps data.priorPoint =
      computedCoefficient baseOps extensionOps liftCoefficient data coordinate := by
  exact BooleanTable.evaluate_eq_equalityWeightedSum
    extensionOps extensionLaws _ _

/-- The independently stated paper evaluation equation for one coordinate. -/
def EvaluationClaimHolds
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (extensionOps : InterpolationOps Extension)
    (liftCoefficient : Base -> Extension)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns)
    (coordinate : CarriedCoordinate shape) : Prop :=
  data.claimedCoefficient coordinate =
    computedCoefficient baseOps extensionOps liftCoefficient data coordinate

/-- Candidate signed orientation from the paper audit: target coefficient
minus the independently computed matrix-image evaluation. -/
def residual
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (extensionOps : InterpolationOps Extension)
    (liftCoefficient : Base -> Extension)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns)
    (coordinate : CarriedCoordinate shape) : Extension :=
  extensionOps.sub
    (data.claimedCoefficient coordinate)
    (computedCoefficient baseOps extensionOps liftCoefficient data coordinate)

private theorem sub_eq_zero_iff
    {Extension : Type uExtension}
    (ops : InterpolationOps Extension)
    (laws : InterpolationEvaluationLaws ops)
    (left right : Extension) :
    ops.sub left right = ops.zero ↔ left = right := by
  constructor
  · intro zero
    change ops.add left (ops.neg right) = ops.zero at zero
    have negAdd : ops.add (ops.neg right) right = ops.zero := by
      rw [laws.add_comm]
      exact laws.add_neg right
    calc
      left = ops.add left ops.zero := (laws.add_zero left).symm
      _ = ops.add left (ops.add (ops.neg right) right) := by
        rw [negAdd]
      _ = ops.add (ops.add left (ops.neg right)) right :=
        (laws.add_assoc left (ops.neg right) right).symm
      _ = ops.add ops.zero right := by
        rw [zero]
      _ = right := laws.zero_add right
  · intro equal
    subst right
    exact laws.add_neg left

/-- One carried residual is zero exactly when its independently stated
evaluation equation holds. -/
theorem residual_eq_zero_iff_evaluationClaimHolds
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (extensionOps : InterpolationOps Extension)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (liftCoefficient : Base -> Extension)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns)
    (coordinate : CarriedCoordinate shape) :
    residual baseOps extensionOps liftCoefficient data coordinate =
        extensionOps.zero ↔
      EvaluationClaimHolds baseOps extensionOps liftCoefficient data coordinate := by
  exact sub_eq_zero_iff extensionOps extensionLaws _ _

/-- Every carried prior-evaluation equation is true. -/
def AllClaimsHold
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (extensionOps : InterpolationOps Extension)
    (liftCoefficient : Base -> Extension)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns) : Prop :=
  ∀ coordinate,
    EvaluationClaimHolds baseOps extensionOps liftCoefficient data coordinate

/-- The complete typed residual family is zero iff every independently stated
carried evaluation equation holds. -/
theorem allResidualsZero_iff_allClaimsHold
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (extensionOps : InterpolationOps Extension)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (liftCoefficient : Base -> Extension)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns) :
    (∀ coordinate,
        residual baseOps extensionOps liftCoefficient data coordinate =
          extensionOps.zero) ↔
      AllClaimsHold baseOps extensionOps liftCoefficient data := by
  constructor
  · intro allZero coordinate
    exact (residual_eq_zero_iff_evaluationClaimHolds
      baseOps extensionOps extensionLaws liftCoefficient data coordinate).mp
        (allZero coordinate)
  · intro allHold coordinate
    exact (residual_eq_zero_iff_evaluationClaimHolds
      baseOps extensionOps extensionLaws liftCoefficient data coordinate).mpr
        (allHold coordinate)

/-- Residuals serialized in the one canonical carried-coordinate order. -/
def orderedResiduals
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (extensionOps : InterpolationOps Extension)
    (liftCoefficient : Base -> Extension)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns) : List Extension :=
  (canonicalCarriedCoordinates shape).map <|
    residual baseOps extensionOps liftCoefficient data

/-- Canonical serialization cannot omit or insert a carried residual. -/
theorem orderedResiduals_length
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (extensionOps : InterpolationOps Extension)
    (liftCoefficient : Base -> Extension)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns) :
    (orderedResiduals baseOps extensionOps liftCoefficient data).length =
      shape.carriedEvaluationCount := by
  simp [orderedResiduals, canonicalCarriedCoordinates_length]

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CarriedEvaluationResidual
