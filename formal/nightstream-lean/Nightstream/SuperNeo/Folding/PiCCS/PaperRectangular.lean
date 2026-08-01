import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedJointIdentity

/-!
The corrected SuperNeo `Pi_CCS` polynomial split across rectangular domains.

Protocol: SuperNeo Section 7.3 / Appendix D.4, with one declared extension:
the fresh CCS and carried-evaluation terms use the row cube, while the norm
terms use the column cube. The split adds a second SumCheck but does not add a
coefficient/lane axis or change any gamma exponent.

Owns: the two pointwise polynomials, their initial claims, their Boolean sums,
and the square-domain theorem that reconstructs the paper's joint polynomial.

Does not own: concrete matrices or witnesses, transcript encoding, Rust,
Poseidon2, SumCheck messages, R1CS, or constraint counts.

The absolute gamma layout is inherited from `PaperJoint.Shape`:

* fresh CCS source `i` uses exponent `i`;
* norm source `i` uses exponent `K + i`;
* carried coordinate `(i,j,l)` uses exponent `2K+k+I(i,j,l)`.
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperRectangular

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteSumAlgebra

universe uField

/-- Source tables for the exact rectangular decomposition. Row-side tables
use `shape.cubeVariables`; norm tables use the independent column cube. -/
structure Data
    (Field : Type uField)
    (shape : Shape)
    (columnVariables : Nat) where
  ccs : Fin shape.freshCount -> BooleanTable Field shape.cubeVariables
  norm : Fin shape.sourceCount -> BooleanTable Field columnVariables
  priorPoint : CubePoint Field shape.cubeVariables
  carriedImage :
    CarriedCoordinate shape -> BooleanTable Field shape.cubeVariables
  claimedCoefficient : CarriedCoordinate shape -> Field

/-- Paper `F` on one Boolean row. -/
def ccsAt
    {Field : Type uField}
    {shape : Shape}
    {columnVariables : Nat}
    (ops : InterpolationOps Field)
    (data : Data Field shape columnVariables)
    (gamma : Field)
    (row : BooleanVertex shape.cubeVariables) : Field :=
  SignedJointIdentity.sumMap ops
    (canonicalFinIndices shape.freshCount) fun source =>
      SignedJointIdentity.gammaTerm ops gamma source.val
        ((data.ccs source).valueAt row)

/-- Unshifted paper `NC` on one Boolean column. -/
def normAt
    {Field : Type uField}
    {shape : Shape}
    {columnVariables : Nat}
    (ops : InterpolationOps Field)
    (data : Data Field shape columnVariables)
    (gamma : Field)
    (column : BooleanVertex columnVariables) : Field :=
  SignedJointIdentity.sumMap ops
    (canonicalFinIndices shape.sourceCount) fun source =>
      SignedJointIdentity.gammaTerm ops gamma source.val
        ((data.norm source).valueAt column)

/-- Unshifted paper `Eval` on one Boolean row. The local carried-coordinate
exponent remains explicit; no coefficient coordinate becomes a cube axis. -/
def carriedAt
    {Field : Type uField}
    {shape : Shape}
    {columnVariables : Nat}
    (ops : InterpolationOps Field)
    (data : Data Field shape columnVariables)
    (gamma : Field)
    (row : BooleanVertex shape.cubeVariables) : Field :=
  ops.mul (row.equalityWeight ops data.priorPoint) <|
    SignedJointIdentity.sumMap ops
      (canonicalCarriedCoordinates shape) fun coordinate =>
        SignedJointIdentity.gammaTerm ops gamma
          coordinate.localGammaExponent
          ((data.carriedImage coordinate).valueAt row)

/-- Row-domain polynomial. It contains exactly the paper's fresh CCS and
carried-evaluation terms. -/
def feAt
    {Field : Type uField}
    {shape : Shape}
    {columnVariables : Nat}
    (ops : InterpolationOps Field)
    (data : Data Field shape columnVariables)
    (alphaRow : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (row : BooleanVertex shape.cubeVariables) : Field :=
  ops.add
    (ops.mul (row.equalityWeight ops alphaRow)
      (ccsAt ops data gamma row))
    (SignedJointIdentity.gammaTerm ops gamma
      shape.carriedEvaluationOffset (carriedAt ops data gamma row))

/-- Column-domain polynomial. The outer `gamma^K` is the paper's norm-block
shift, so norm source `i` has the absolute exponent `K+i`. -/
def ncAt
    {Field : Type uField}
    {shape : Shape}
    {columnVariables : Nat}
    (ops : InterpolationOps Field)
    (data : Data Field shape columnVariables)
    (alphaColumn : CubePoint Field columnVariables)
    (gamma : Field)
    (column : BooleanVertex columnVariables) : Field :=
  ops.mul (column.equalityWeight ops alphaColumn) <|
    SignedJointIdentity.gammaTerm ops gamma shape.freshCount
      (normAt ops data gamma column)

/-- Boolean sum claimed by the FE SumCheck. -/
def summedFe
    {Field : Type uField}
    {shape : Shape}
    {columnVariables : Nat}
    (ops : InterpolationOps Field)
    (data : Data Field shape columnVariables)
    (alphaRow : CubePoint Field shape.cubeVariables)
    (gamma : Field) : Field :=
  SignedJointIdentity.sumMap ops
    (BooleanVertex.all shape.cubeVariables) fun row =>
      feAt ops data alphaRow gamma row

/-- Boolean sum claimed by the NC SumCheck. -/
def summedNc
    {Field : Type uField}
    {shape : Shape}
    {columnVariables : Nat}
    (ops : InterpolationOps Field)
    (data : Data Field shape columnVariables)
    (alphaColumn : CubePoint Field columnVariables)
    (gamma : Field) : Field :=
  SignedJointIdentity.sumMap ops
    (BooleanVertex.all columnVariables) fun column =>
      ncAt ops data alphaColumn gamma column

/-- Corrected paper target. Only the FE SumCheck claims this value. -/
def feInitial
    {Field : Type uField}
    {shape : Shape}
    {columnVariables : Nat}
    (ops : InterpolationOps Field)
    (data : Data Field shape columnVariables)
    (gamma : Field) : Field :=
  TargetPolynomial.evaluateShifted ops.toOps
    { coefficient := data.claimedCoefficient } gamma

/-- The independent norm SumCheck has the zero claim. -/
def ncInitial
    {Field : Type uField}
    (ops : InterpolationOps Field) : Field :=
  ops.zero

/-- Exact two-SumCheck truth statement. -/
def Holds
    {Field : Type uField}
    {shape : Shape}
    {columnVariables : Nat}
    (ops : InterpolationOps Field)
    (data : Data Field shape columnVariables)
    (alphaRow : CubePoint Field shape.cubeVariables)
    (alphaColumn : CubePoint Field columnVariables)
    (gamma : Field) : Prop :=
  feInitial ops data gamma = summedFe ops data alphaRow gamma /\
    ncInitial ops = summedNc ops data alphaColumn gamma

namespace Square

/-- On a square domain, the rectangular sources instantiate the paper's
single joint source object without any regrouping or reindexing. -/
def toJointData
    {Field : Type uField}
    {shape : Shape}
    (data : Data Field shape shape.cubeVariables) :
    SignedJointIdentity.JointData Field shape where
  ccs := data.ccs
  norm := data.norm
  priorPoint := data.priorPoint
  carriedImage := data.carriedImage
  claimedCoefficient := data.claimedCoefficient

/-- Pointwise square-domain decomposition of the paper polynomial. This is
the exact boundary for the permitted rectangular change. -/
theorem joint_qAt_eq_fe_add_nc
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : Data Field shape shape.cubeVariables)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (vertex : BooleanVertex shape.cubeVariables) :
    SignedJointIdentity.qAt ops (toJointData data) alpha gamma vertex =
      ops.add
        (feAt ops data alpha gamma vertex)
        (ncAt ops data alpha gamma vertex) := by
  let equality := vertex.equalityWeight ops alpha
  let ccs := ccsAt ops data gamma vertex
  let norm := SignedJointIdentity.gammaTerm ops gamma shape.freshCount
    (normAt ops data gamma vertex)
  let carried := SignedJointIdentity.gammaTerm ops gamma
    shape.carriedEvaluationOffset (carriedAt ops data gamma vertex)
  change ops.add (ops.mul equality (ops.add ccs norm)) carried =
    ops.add (ops.add (ops.mul equality ccs) carried)
      (ops.mul equality norm)
  rw [laws.left_distrib]
  calc
    ops.add
        (ops.add (ops.mul equality ccs) (ops.mul equality norm))
        carried =
      ops.add (ops.mul equality ccs)
        (ops.add (ops.mul equality norm) carried) :=
          laws.add_assoc _ _ _
    _ = ops.add (ops.mul equality ccs)
        (ops.add carried (ops.mul equality norm)) := by
          rw [laws.add_comm (ops.mul equality norm) carried]
    _ = ops.add (ops.add (ops.mul equality ccs) carried)
        (ops.mul equality norm) :=
          (laws.add_assoc _ _ _).symm

/-- Summing the pointwise identity reconstructs the complete paper joint
sum. In particular, the coefficient order and all absolute gamma slots are
unchanged. -/
theorem joint_summedQ_eq_summedFe_add_summedNc
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : Data Field shape shape.cubeVariables)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field) :
    SignedJointIdentity.summedQ ops (toJointData data) alpha gamma =
      ops.add
        (summedFe ops data alpha gamma)
        (summedNc ops data alpha gamma) := by
  unfold SignedJointIdentity.summedQ summedFe summedNc
  have pointwise :
      (fun vertex =>
        SignedJointIdentity.qAt ops (toJointData data) alpha gamma vertex) =
      (fun vertex =>
        ops.add (feAt ops data alpha gamma vertex)
          (ncAt ops data alpha gamma vertex)) := by
    funext vertex
    exact joint_qAt_eq_fe_add_nc ops laws data alpha gamma vertex
  rw [pointwise]
  have split := FiniteSumAlgebra.sumMap_add ops laws
    (BooleanVertex.all shape.cubeVariables)
    (fun vertex => feAt ops data alpha gamma vertex)
    (fun vertex => ncAt ops data alpha gamma vertex)
  simpa only [SignedJointIdentity.sumMap, FiniteSumAlgebra.sumMap] using split

/-- The rectangular FE target is definitionally the corrected absolute target
of the paper joint object. -/
theorem feInitial_eq_joint_target
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : Data Field shape shape.cubeVariables)
    (gamma : Field) :
    feInitial ops data gamma =
      SignedJointIdentity.targetAbsolute ops (toJointData data) gamma := by
  rfl

/-- If both rectangular claims hold on a square domain, then the paper's one
joint claimed-sum equation holds. -/
theorem holds_implies_joint_claim
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : Data Field shape shape.cubeVariables)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (holds : Holds ops data alpha alpha gamma) :
    SignedJointIdentity.targetAbsolute ops (toJointData data) gamma =
      SignedJointIdentity.summedQ ops (toJointData data) alpha gamma := by
  rw [joint_summedQ_eq_summedFe_add_summedNc ops laws data alpha gamma]
  rw [<- feInitial_eq_joint_target ops data gamma]
  calc
    feInitial ops data gamma = summedFe ops data alpha gamma := holds.1
    _ = ops.add (summedFe ops data alpha gamma) ops.zero :=
      (laws.add_zero _).symm
    _ = ops.add (summedFe ops data alpha gamma)
        (summedNc ops data alpha gamma) := by
      rw [<- holds.2]
      rfl

end Square

end Nightstream.SuperNeo.Folding.PiCCS.PaperRectangular
