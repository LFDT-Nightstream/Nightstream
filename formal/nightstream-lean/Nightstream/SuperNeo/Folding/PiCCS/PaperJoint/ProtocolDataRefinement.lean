import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConstraintPolynomialLift.Evaluation

/-!
Refinement from one authoritative `Pi_CCS` source family to the actual
off-cube protocol polynomial.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: source-image construction before Fiat--Shamir and SumCheck.
Constraint family: semantic construction and carrier placement only; this
file emits no rows.

Owns: the norm-aware extension of the shared sparse-polynomial lift contract;
specialization of the neutral evaluation theorem; proof that Boolean-table
construction commutes with the shared structural lift;
construction of `ProtocolPolynomial.Data` from the
single authoritative source owner; and exact equality of the resulting
Boolean restriction with the independently derived joint residual object.

Does not own: the generic sparse-polynomial evaluation proof; an instantiation
of the lift for the production extension/ring,
proof that coefficient-expanded carried matrices derive from the CCS
structure matrices, output-message serialization, Fiat--Shamir, SumCheck
degree bounds, Rust, R1CS, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: callers provide the algebraic carrier operations and a
lift satisfying explicit mathematical laws. They cannot provide image tables,
residual tables, a `ProtocolPolynomial.Data`, a `JointData`, or an equality
between those objects. Every such object and the equality theorem are derived
here from `UnifiedSources.UnifiedInputs`.

| Protocol | Phase | Family | Derived object / theorem |
|---|---|---|---|
| `Pi_CCS` | carrier placement | zero, one, add, mul, strict norm | `ProtocolLift` |
| `Pi_CCS` | CCS syntax | sparse monomials and polynomial | imported `ConstraintPolynomialLift.liftConstraintPolynomial` |
| `Pi_CCS` | CCS semantics | specialize the neutral lift/evaluation theorem | `evaluatePolynomial_lift` |
| `Pi_CCS` | Boolean source images | canonical low/high table | `liftTable_tabulate` |
| `Pi_CCS` | actual polynomial input | all source-image families | `toProtocolData` |
| assurance | CCS branch | actual Boolean restriction = independent CCS residual | `ccsTable_eq` |
| assurance | norm branch | actual Boolean restriction = independent cubic residual | `normTable_eq` |
| assurance | joint closure | all five typed families agree | `toProtocolData_toJointData_eq` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolDataRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open PaperLinearAlgebra
open ConstraintPolynomialLift

universe uExtension

/-- Exact algebraic assumptions needed to place the independently defined
base-field data in the active protocol carrier. The strict-norm law is named
separately because the base residual uses concrete centered-representative
semantics rather than a caller-selected arithmetic expression. -/
structure ProtocolLift
    {Extension : Type uExtension}
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    extends ConcreteJointData.ZeroReflectingLift
      baseOps extensionOps lift where
  map_one : lift baseOps.one = extensionOps.one
  map_add : forall left right,
    lift (baseOps.add left right) =
      extensionOps.add (lift left) (lift right)
  map_mul : forall left right,
    lift (baseOps.mul left right) =
      extensionOps.mul (lift left) (lift right)
  map_strictNorm : forall value,
    ProtocolPolynomial.strictNormResidual extensionOps (lift value) =
      lift (NormRange.cubicResidual value)

namespace ProtocolLift

/-- Zero preservation is derived from zero reflection; it is not an
additional caller-selected law. -/
theorem map_zero
    {Extension : Type uExtension}
    {baseOps : InterpolationOps F}
    {extensionOps : InterpolationOps Extension}
    {lift : F -> Extension}
    (laws : ProtocolLift baseOps extensionOps lift) :
    lift baseOps.zero = extensionOps.zero :=
  (laws.zero_iff baseOps.zero).mpr rfl

/-- Forget the norm and zero-reflection obligations when only sparse CCS
evaluation is being refined. -/
def toConstraintEvaluationLaws
    {Extension : Type uExtension}
    {baseOps : InterpolationOps F}
    {extensionOps : InterpolationOps Extension}
    {lift : F -> Extension}
    (laws : ProtocolLift baseOps extensionOps lift) :
    ConstraintPolynomialLift.Evaluation.EvaluationLaws
      baseOps extensionOps lift where
  map_zero := laws.map_zero
  map_one := laws.map_one
  map_add := laws.map_add
  map_mul := laws.map_mul

end ProtocolLift

/-- Evaluation of one explicit sparse monomial commutes with the lift. -/
theorem evaluateMonomial_lift
    {Extension : Type uExtension}
    {matrixCount : Nat}
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    (laws : ProtocolLift baseOps extensionOps lift)
    (monomial : CCSResidualTable.Monomial F matrixCount)
    (point : Fin matrixCount -> F) :
    CCSResidualTable.evaluateMonomial extensionOps
        (liftMonomial lift monomial) (fun index => lift (point index)) =
      lift (CCSResidualTable.evaluateMonomial baseOps monomial point) := by
  exact ConstraintPolynomialLift.Evaluation.evaluateMonomial_lift
    baseOps extensionOps lift laws.toConstraintEvaluationLaws monomial point

/-- Evaluation of the whole explicit sparse CCS polynomial commutes with the
lift. This theorem is derived from syntax and algebraic laws, not supplied as
a refinement callback. -/
theorem evaluatePolynomial_lift
    {Extension : Type uExtension}
    {matrixCount : Nat}
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    (laws : ProtocolLift baseOps extensionOps lift)
    (polynomial : CCSResidualTable.ConstraintPolynomial F matrixCount)
    (point : Fin matrixCount -> F) :
    CCSResidualTable.evaluatePolynomial extensionOps
        (liftConstraintPolynomial lift polynomial)
        (fun index => lift (point index)) =
      lift (CCSResidualTable.evaluatePolynomial baseOps polynomial point) := by
  exact ConstraintPolynomialLift.Evaluation.evaluatePolynomial_lift
    baseOps extensionOps lift laws.toConstraintEvaluationLaws polynomial point

/-- Structural lifting and canonical low/high tabulation commute exactly. -/
theorem liftTable_tabulate
    {Extension : Type uExtension}
    (lift : F -> Extension)
    {variables : Nat}
    (values : BooleanVertex variables -> F) :
    ConcreteJointData.liftTable lift (BooleanTable.tabulate values) =
      BooleanTable.tabulate (fun vertex => lift (values vertex)) := by
  induction variables with
  | zero => rfl
  | succ variables inductionHypothesis =>
      simp only [BooleanTable.tabulate, ConcreteJointData.liftTable]
      congr 1
      · exact inductionHypothesis _
      · exact inductionHypothesis _

/-- Construct every actual-protocol source image from the sole authoritative
assignment family. No residual table or off-cube evaluator is an input. -/
def toProtocolData
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (baseOps : InterpolationOps F)
    (lift : F -> Extension)
    (data : UnifiedSources.UnifiedInputs Extension shape columns) :
    ProtocolPolynomial.Data Extension shape where
  constraintPolynomial :=
    liftConstraintPolynomial lift data.system.constraintPolynomial
  freshMatrixImages := fun source matrix => BooleanTable.tabulate fun vertex =>
    lift (CCSResidualTable.matrixImagesAt baseOps data.system
      (data.assignments (UnifiedSources.freshSourceIndex source))
      vertex matrix)
  sourceAssignments := fun source => BooleanTable.tabulate fun vertex =>
    lift (data.layout.paddedValue 0 (data.assignments source) vertex)
  priorPoint := data.priorPoint
  carriedImages := fun coordinate =>
    CarriedEvaluationResidual.imageTable baseOps lift data.carriedData coordinate
  claimedCoefficient := data.claimedCoefficient

/-- The CCS branch of the actual protocol's Boolean restriction is exactly
the lifted independently constructed CCS residual table. -/
theorem ccsTable_eq
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    (laws : ProtocolLift baseOps extensionOps lift)
    (data : UnifiedSources.UnifiedInputs Extension shape columns)
    (source : Fin shape.freshCount) :
    ((toProtocolData baseOps lift data).toJointData
        extensionOps).ccs source =
      (data.toIndependentInputs.toJointData baseOps lift).ccs source := by
  change
    BooleanTable.tabulate (fun vertex =>
      CCSResidualTable.evaluatePolynomial extensionOps
        (liftConstraintPolynomial lift data.system.constraintPolynomial)
        (fun matrix =>
          (BooleanTable.tabulate (fun row =>
            lift (CCSResidualTable.matrixImagesAt baseOps data.system
              (data.assignments (UnifiedSources.freshSourceIndex source))
              row matrix))).valueAt vertex)) =
      ConcreteJointData.liftTable lift (BooleanTable.tabulate (fun vertex =>
        CCSResidualTable.evaluatePolynomial baseOps
          data.system.constraintPolynomial
          (CCSResidualTable.matrixImagesAt baseOps data.system
            (data.assignments (UnifiedSources.freshSourceIndex source))
            vertex)))
  rw [liftTable_tabulate]
  apply congrArg BooleanTable.tabulate
  funext vertex
  simp only [BooleanTable.valueAt_tabulate]
  exact evaluatePolynomial_lift baseOps extensionOps lift laws
    data.system.constraintPolynomial
    (CCSResidualTable.matrixImagesAt baseOps data.system
      (data.assignments (UnifiedSources.freshSourceIndex source)) vertex)

/-- The norm branch of the actual protocol's Boolean restriction is exactly
the lifted independently constructed cubic residual table. -/
theorem normTable_eq
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    (laws : ProtocolLift baseOps extensionOps lift)
    (data : UnifiedSources.UnifiedInputs Extension shape columns)
    (source : Fin shape.sourceCount) :
    ((toProtocolData baseOps lift data).toJointData
        extensionOps).norm source =
      (data.toIndependentInputs.toJointData baseOps lift).norm source := by
  change
    BooleanTable.tabulate (fun vertex =>
      ProtocolPolynomial.strictNormResidual extensionOps
        ((BooleanTable.tabulate (fun row =>
          lift (data.layout.paddedValue 0
            (data.assignments source) row))).valueAt vertex)) =
      ConcreteJointData.liftTable lift (BooleanTable.tabulate (fun vertex =>
        NormRange.cubicResidual
          (data.layout.paddedValue 0
            (data.assignments source) vertex)))
  rw [liftTable_tabulate]
  apply congrArg BooleanTable.tabulate
  funext vertex
  simp only [BooleanTable.valueAt_tabulate]
  exact laws.map_strictNorm _

/-- Exact closure: restricting the derived actual protocol polynomial to the
Boolean cube produces the independently derived joint residual object for all
five typed families. No equality premise is accepted from a caller. -/
theorem toProtocolData_toJointData_eq
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    (laws : ProtocolLift baseOps extensionOps lift)
    (data : UnifiedSources.UnifiedInputs Extension shape columns) :
    (toProtocolData baseOps lift data).toJointData extensionOps =
      data.toIndependentInputs.toJointData baseOps lift := by
  apply SignedJointIdentity.JointData.ext
  · funext source
    exact ccsTable_eq baseOps extensionOps lift laws data source
  · funext source
    exact normTable_eq baseOps extensionOps lift laws data source
  · rfl
  · rfl
  · rfl

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolDataRefinement
