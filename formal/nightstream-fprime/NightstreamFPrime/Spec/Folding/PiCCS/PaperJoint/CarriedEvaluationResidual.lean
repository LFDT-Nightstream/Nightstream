import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanHypercubeSum
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetPolynomial

/-!
Concrete v1.2 matrix-evaluation residuals for `Eval_A`.

Protocol: SuperNeo v1.2 `Pi_CCS` (Section 7.3 / Appendix B.2).
Phase: construction of the matrix `Eval_A(X, C)` coefficient family.
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
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixEvaluationResidual

universe uBase uExtension

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PaperLinearAlgebra

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
  claimedCoefficient : MatrixCoordinate shape -> Extension

def imageCoefficientAt
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (liftCoefficient : Base -> Extension)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns)
    (coordinate : MatrixCoordinate shape)
    (vertex : BooleanVertex shape.cubeVariables) : Extension :=
  liftCoefficient <|
    matrixVectorAt baseOps
      (data.coefficientMatrices coordinate.matrix coordinate.coefficient)
      (data.assignments coordinate.running)
      vertex

def imageTable
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (liftCoefficient : Base -> Extension)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns)
    (coordinate : MatrixCoordinate shape) :
    BooleanTable Extension shape.cubeVariables :=
  BooleanTable.tabulate <|
    imageCoefficientAt baseOps liftCoefficient data coordinate

def computedCoefficient
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (extensionOps : InterpolationOps Extension)
    (liftCoefficient : Base -> Extension)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns)
    (coordinate : MatrixCoordinate shape) : Extension :=
  (imageTable baseOps liftCoefficient data coordinate).equalityWeightedSum
    extensionOps data.priorPoint

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
    (coordinate : MatrixCoordinate shape)
    (assignmentZero :
      data.assignments coordinate.running = fun _ => baseOps.zero)
    (vertex : BooleanVertex shape.cubeVariables) :
    imageCoefficientAt baseOps liftCoefficient data coordinate vertex =
      extensionOps.zero := by
  unfold imageCoefficientAt
  rw [assignmentZero,
    PaperLinearAlgebra.matrixVectorAt_zero baseOps baseLaws]
  exact liftZero

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
    (coordinate : MatrixCoordinate shape)
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
    (coordinate : MatrixCoordinate shape) :
    (imageTable baseOps liftCoefficient data coordinate).evaluate
        extensionOps data.priorPoint =
      computedCoefficient baseOps extensionOps liftCoefficient data coordinate := by
  exact BooleanTable.evaluate_eq_equalityWeightedSum
    extensionOps extensionLaws _ _

def EvaluationClaimHolds
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (extensionOps : InterpolationOps Extension)
    (liftCoefficient : Base -> Extension)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns)
    (coordinate : MatrixCoordinate shape) : Prop :=
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
    (coordinate : MatrixCoordinate shape) : Extension :=
  extensionOps.sub
    (data.claimedCoefficient coordinate)
    (computedCoefficient baseOps extensionOps liftCoefficient data coordinate)

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
    (coordinate : MatrixCoordinate shape) :
    residual baseOps extensionOps liftCoefficient data coordinate =
        extensionOps.zero ↔
      EvaluationClaimHolds baseOps extensionOps liftCoefficient data coordinate := by
  exact FiniteSumAlgebra.sub_eq_zero_iff extensionOps extensionLaws _ _

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

def orderedResiduals
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (extensionOps : InterpolationOps Extension)
    (liftCoefficient : Base -> Extension)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns) : List Extension :=
  (canonicalMatrixCoordinates shape).map <|
    residual baseOps extensionOps liftCoefficient data

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
      shape.matrixEvaluationCount := by
  simp [orderedResiduals, canonicalMatrixCoordinates_length]

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixEvaluationResidual

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PadEvaluationResidual

universe uBase uExtension

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PaperLinearAlgebra

structure EvaluationData
    (Base : Type uBase)
    (Extension : Type uExtension)
    (shape : Shape)
    (columns : Nat) where
  priorPoint : CubePoint Extension shape.cubeVariables
  assignments : Fin shape.runningCount -> Assignment Base columns
  coefficientMatrices : Fin shape.coefficientCount ->
    BooleanMatrix Base shape.cubeVariables columns
  claimedCoefficient : PadCoordinate shape -> Extension

def imageCoefficientAt
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (liftCoefficient : Base -> Extension)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns)
    (coordinate : PadCoordinate shape)
    (vertex : BooleanVertex shape.cubeVariables) : Extension :=
  liftCoefficient <|
    matrixVectorAt baseOps
      (data.coefficientMatrices coordinate.coefficient)
      (data.assignments coordinate.running)
      vertex

def imageTable
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (liftCoefficient : Base -> Extension)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns)
    (coordinate : PadCoordinate shape) :
    BooleanTable Extension shape.cubeVariables :=
  BooleanTable.tabulate <|
    imageCoefficientAt baseOps liftCoefficient data coordinate

def computedCoefficient
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (extensionOps : InterpolationOps Extension)
    (liftCoefficient : Base -> Extension)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns)
    (coordinate : PadCoordinate shape) : Extension :=
  (imageTable baseOps liftCoefficient data coordinate).equalityWeightedSum
    extensionOps data.priorPoint

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
    (coordinate : PadCoordinate shape) :
    (imageTable baseOps liftCoefficient data coordinate).evaluate
        extensionOps data.priorPoint =
      computedCoefficient baseOps extensionOps liftCoefficient data coordinate := by
  exact BooleanTable.evaluate_eq_equalityWeightedSum
    extensionOps extensionLaws _ _

def EvaluationClaimHolds
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (extensionOps : InterpolationOps Extension)
    (liftCoefficient : Base -> Extension)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns)
    (coordinate : PadCoordinate shape) : Prop :=
  data.claimedCoefficient coordinate =
    computedCoefficient baseOps extensionOps liftCoefficient data coordinate

def residual
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (extensionOps : InterpolationOps Extension)
    (liftCoefficient : Base -> Extension)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns)
    (coordinate : PadCoordinate shape) : Extension :=
  extensionOps.sub
    (data.claimedCoefficient coordinate)
    (computedCoefficient baseOps extensionOps liftCoefficient data coordinate)

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
    (coordinate : PadCoordinate shape) :
    residual baseOps extensionOps liftCoefficient data coordinate =
        extensionOps.zero ↔
      EvaluationClaimHolds baseOps extensionOps liftCoefficient data coordinate := by
  exact FiniteSumAlgebra.sub_eq_zero_iff extensionOps extensionLaws _ _

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

def orderedResiduals
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (extensionOps : InterpolationOps Extension)
    (liftCoefficient : Base -> Extension)
    {shape : Shape}
    {columns : Nat}
    (data : EvaluationData Base Extension shape columns) : List Extension :=
  (canonicalPadCoordinates shape).map <|
    residual baseOps extensionOps liftCoefficient data

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
      shape.padEvaluationCount := by
  simp [orderedResiduals, canonicalPadCoordinates_length]

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PadEvaluationResidual
