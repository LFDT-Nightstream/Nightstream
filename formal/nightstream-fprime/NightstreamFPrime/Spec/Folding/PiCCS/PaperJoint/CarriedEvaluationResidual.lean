import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanHypercubeSum
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetPolynomial

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/CarriedEvaluationResidual.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

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
serialization, target placement inside signed joint `Q`,
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
| zero assignment | `z_i = 0` | every image-table leaf and its explicit hypercube sum are zero | `imageCoefficientAt_eq_zero_of_assignment_zero`, `computedCoefficient_eq_zero_of_assignment_zero` |
| `residual` | Equation (9) orientation | claimed minus computed | zero iff the evaluation equation holds |
| `allResidualsZero_iff_allClaimsHold` | Lemma 7 Item 3 | every carried coordinate | unconditional relative to explicit algebra/lift data |
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CarriedEvaluationResidual

universe uBase uExtension

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
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

/-- A canonical zero running assignment makes every coefficient-expanded
matrix-image leaf zero. No verifier polynomial or implementation evaluator is
involved: this is the shared finite matrix-vector definition itself. -/
theorem imageCoefficientAt_eq_zero_of_assignment_zero
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (baseLaws : InterpolationEvaluationLaws baseOps)
    (extensionOps : InterpolationOps Extension)
    (liftCoefficient : Base -> Extension)
    (liftZero : liftCoefficient baseOps.zero = extensionOps.zero)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns)
    (coordinate : CarriedCoordinate shape)
    (assignmentZero :
      data.assignments coordinate.running = fun _ => baseOps.zero)
    (vertex : BooleanVertex shape.cubeVariables) :
    imageCoefficientAt baseOps liftCoefficient data coordinate vertex =
      extensionOps.zero := by
  unfold imageCoefficientAt
  rw [assignmentZero,
    PaperLinearAlgebra.matrixVectorAt_zero baseOps baseLaws]
  exact liftZero

/-- The explicit equality-weighted prior evaluation of a zero running
assignment is zero. This closes the paper/source semantic fact directly,
without routing through a verifier-owned `yRing` evaluator. -/
theorem computedCoefficient_eq_zero_of_assignment_zero
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (baseLaws : InterpolationEvaluationLaws baseOps)
    (extensionOps : InterpolationOps Extension)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (liftCoefficient : Base -> Extension)
    (liftZero : liftCoefficient baseOps.zero = extensionOps.zero)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns)
    (coordinate : CarriedCoordinate shape)
    (assignmentZero :
      data.assignments coordinate.running = fun _ => baseOps.zero) :
    computedCoefficient baseOps extensionOps liftCoefficient data coordinate =
      extensionOps.zero := by
  unfold computedCoefficient BooleanTable.equalityWeightedSum
  change FiniteSumAlgebra.sumMap extensionOps
      (BooleanVertex.all shape.cubeVariables)
      (fun vertex =>
        extensionOps.mul
          (vertex.equalityWeight extensionOps data.priorPoint)
          ((imageTable baseOps liftCoefficient data coordinate).valueAt vertex)) =
    extensionOps.zero
  calc
    _ = FiniteSumAlgebra.sumMap extensionOps
        (BooleanVertex.all shape.cubeVariables)
        (fun _ => extensionOps.zero) := by
      apply FiniteSumAlgebra.sumMap_congr
      intro vertex _
      rw [show
        (imageTable baseOps liftCoefficient data coordinate).valueAt vertex =
            imageCoefficientAt baseOps liftCoefficient data coordinate vertex by
          exact BooleanTable.valueAt_tabulate _ _]
      rw [imageCoefficientAt_eq_zero_of_assignment_zero
        baseOps baseLaws extensionOps liftCoefficient liftZero data coordinate
        assignmentZero vertex]
      exact extensionLaws.mul_zero _
    _ = extensionOps.zero :=
      FiniteSumAlgebra.sumMap_zero extensionOps extensionLaws _

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

/-- Selected signed orientation from the corrected paper: target coefficient
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
  exact FiniteSumAlgebra.sub_eq_zero_iff extensionOps extensionLaws _ _

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

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CarriedEvaluationResidual
