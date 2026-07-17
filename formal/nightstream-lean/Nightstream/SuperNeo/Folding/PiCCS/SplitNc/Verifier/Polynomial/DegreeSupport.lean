import Nightstream.SuperNeo.SumCheck.FixedPolynomial
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanReproduction
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckTruthPath

/-!
Shared one-variable degree support for Split-NC FE and NC polynomials.

Owns: the concrete `K` polynomial laws, exact Boolean-MLE coordinate slices,
equality-polynomial coordinate slices, finite same-degree weighted sums, and
the typed coordinate-splice constructor used by both protocol phases.

Does not own: an FE or NC degree bound, any protocol polynomial, SumCheck
acceptance, transcript derivation, Rust, R1CS, emitted rows, or costs.

Emits constraints: no.

Authority boundary: every representation is constructed from explicit
coefficients and evaluated by the verifier-visible Horner evaluator. This
module supplies algebraic closure only; phase modules remain responsible for
showing that their independently defined protocol polynomial has that form.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.degree.coordinate` | replace exactly one cube coordinate | computed | `cubeSlice` |
| `nifs.pi_ccs.degree.mle` | one coordinate of a Boolean-table MLE is affine | derived | `evaluateCoordinates_affine` |
| `nifs.pi_ccs.degree.selector` | one coordinate of `eq(point,beta)` is affine | derived | `pointEqualityCoordinates_affine` |
| `nifs.pi_ccs.degree.selector.right` | the same bound when the exposed coordinate is on the right | derived | `pointEqualityCoordinates_right_affine` |
| `nifs.pi_ccs.degree.sum` | finite weighted sums preserve a fixed degree | derived | `polynomial_sum_exists` |
| `nifs.pi_ccs.degree.strict_range` | strict-`b = 2` maps an affine slice to the exact cubic | derived | `strictRangeOfAffine`, `evaluate_strictRangeOfAffine` |
| `nifs.pi_ccs.degree.suffix` | Boolean suffix summation preserves a fixed degree | derived | `sumCompletions_represents` |
| `nifs.pi_ccs.degree.message` | a typed representation projects to exactly `degree + 1` verifier-visible coefficients | computed | `Represents.message_shape` |
| `nifs.pi_ccs.degree.closure` | widen, scale, add, and multiply representations | derived | `Represents.widen`, `Represents.scale`, `Represents.add`, `Represents.mul` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.DegreeSupport

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev ops := ConcreteCarrier.extensionOps
abbrev laws := ConcreteCarrier.extensionLaws
abbrev Polynomial := SumCheck.Finite.FixedPolynomial K

/-- A scalar function has a verifier-visible coefficient representation at
the declared degree. The highest coefficient may be zero. -/
def Represents (degree : Nat) (function : K -> K) : Prop :=
  exists polynomial : Polynomial degree, forall point,
    polynomial.evaluate ops.toOps point = function point

/-- The concrete extension-field laws in the form expected by the shared
fixed-width coefficient carrier. -/
def polynomialLaws : SumCheck.Finite.FixedPolynomial.Laws ops.toOps where
  add_assoc := laws.add_assoc
  add_comm := laws.add_comm
  zero_add := laws.zero_add
  add_zero := laws.add_zero
  mul_assoc := laws.mul_assoc
  mul_comm := laws.mul_comm
  mul_zero := laws.mul_zero
  left_distrib := laws.left_distrib
  right_distrib := laws.right_distrib

namespace Represents

/-- Project a typed degree representation to the raw finite-verifier message
without changing its constant-first coefficients or evaluation. -/
theorem message_shape
    {degree : Nat}
    {function : K -> K}
    (represented : Represents degree function) :
    exists message : SumCheck.Finite.Message K,
      message.coefficients.length = degree + 1 ∧
      message.degreeUpperBound = degree ∧
      forall point, message.evaluate ops.toOps point = function point := by
  rcases represented with ⟨polynomial, represents⟩
  exact ⟨polynomial.toMessage,
    polynomial.toMessage_coefficients_length,
    polynomial.toMessage_degreeUpperBound,
    represents⟩

/-- Append only high zero slots to inhabit a larger declared degree. -/
theorem widen
    {degree target : Nat}
    {function : K -> K}
    (degree_le_target : degree <= target)
    (represented : Represents degree function) :
    Represents target function := by
  rcases represented with ⟨polynomial, represents⟩
  refine ⟨SumCheck.Finite.FixedPolynomial.widen
    ops.toOps degree_le_target polynomial, ?_⟩
  intro point
  rw [SumCheck.Finite.FixedPolynomial.evaluate_widen
    ops.toOps polynomialLaws, represents]

/-- Scalar multiplication preserves a declared degree. -/
theorem scale
    {degree : Nat}
    {function : K -> K}
    (scalar : K)
    (represented : Represents degree function) :
    Represents degree fun point => ops.mul scalar (function point) := by
  rcases represented with ⟨polynomial, represents⟩
  refine ⟨SumCheck.Finite.FixedPolynomial.scale ops.toOps scalar polynomial, ?_⟩
  intro point
  rw [SumCheck.Finite.FixedPolynomial.evaluate_scale
    ops.toOps polynomialLaws, represents]

/-- Pointwise addition preserves a shared declared degree. -/
theorem add
    {degree : Nat}
    {left right : K -> K}
    (leftRepresented : Represents degree left)
    (rightRepresented : Represents degree right) :
    Represents degree fun point => ops.add (left point) (right point) := by
  rcases leftRepresented with ⟨leftPolynomial, leftRepresents⟩
  rcases rightRepresented with ⟨rightPolynomial, rightRepresents⟩
  refine ⟨SumCheck.Finite.FixedPolynomial.add
    ops.toOps leftPolynomial rightPolynomial, ?_⟩
  intro point
  rw [SumCheck.Finite.FixedPolynomial.evaluate_add
    ops.toOps polynomialLaws, leftRepresents, rightRepresents]

/-- Pointwise multiplication adds the two declared degree ceilings. -/
theorem mul
    {leftDegree rightDegree : Nat}
    {left right : K -> K}
    (leftRepresented : Represents leftDegree left)
    (rightRepresented : Represents rightDegree right) :
    Represents (leftDegree + rightDegree) fun point =>
      ops.mul (left point) (right point) := by
  rcases leftRepresented with ⟨leftPolynomial, leftRepresents⟩
  rcases rightRepresented with ⟨rightPolynomial, rightRepresents⟩
  refine ⟨SumCheck.Finite.FixedPolynomial.mul
    ops.toOps leftPolynomial rightPolynomial, ?_⟩
  intro point
  rw [SumCheck.Finite.FixedPolynomial.evaluate_mul
    ops.toOps polynomialLaws, leftRepresents, rightRepresents]

end Represents

private def negOne : K := ops.neg ops.one

/-- Coefficient-wise subtraction at one fixed declared degree. -/
def subtract
    {degree : Nat}
    (left right : Polynomial degree) : Polynomial degree :=
  SumCheck.Finite.FixedPolynomial.add ops.toOps left
    (SumCheck.Finite.FixedPolynomial.scale ops.toOps negOne right)

@[simp] theorem evaluate_subtract
    {degree : Nat}
    (left right : Polynomial degree)
    (point : K) :
    (subtract left right).evaluate ops.toOps point =
      ops.sub (left.evaluate ops.toOps point)
        (right.evaluate ops.toOps point) := by
  rw [subtract,
    SumCheck.Finite.FixedPolynomial.evaluate_add ops.toOps polynomialLaws,
    SumCheck.Finite.FixedPolynomial.evaluate_scale ops.toOps polynomialLaws]
  unfold negOne
  rw [laws.neg_mul, laws.one_mul]
  rfl

/-- A constant represented at declared degree one. -/
def affineConstant (value : K) : Polynomial 1 :=
  SumCheck.Finite.FixedPolynomial.affine value ops.zero

@[simp] theorem evaluate_affineConstant (value point : K) :
    (affineConstant value).evaluate ops.toOps point = value := by
  rw [affineConstant,
    SumCheck.Finite.FixedPolynomial.evaluate_affine ops.toOps polynomialLaws,
    laws.mul_zero, laws.add_zero]

/-- Strict-`b = 2` range checking applied to an affine slice, represented by
the exact cubic `(z + 1) * z * (z - 1)`. -/
def strictRangeOfAffine (value : Polynomial 1) : Polynomial 3 :=
  let plusOne := SumCheck.Finite.FixedPolynomial.add ops.toOps value
    (affineConstant ops.one)
  let minusOne := subtract value (affineConstant ops.one)
  SumCheck.Finite.FixedPolynomial.mul ops.toOps
    (SumCheck.Finite.FixedPolynomial.mul ops.toOps plusOne value) minusOne

/-- Evaluation of the shared strict-range cubic preserves the exact
verifier-visible operation order. -/
theorem evaluate_strictRangeOfAffine
    (value : Polynomial 1)
    (point : K) :
    (strictRangeOfAffine value).evaluate ops.toOps point =
      ops.mul
        (ops.mul (ops.add (value.evaluate ops.toOps point) ops.one)
          (value.evaluate ops.toOps point))
        (ops.sub (value.evaluate ops.toOps point) ops.one) := by
  unfold strictRangeOfAffine
  rw [SumCheck.Finite.FixedPolynomial.evaluate_mul ops.toOps polynomialLaws,
    SumCheck.Finite.FixedPolynomial.evaluate_mul ops.toOps polynomialLaws,
    SumCheck.Finite.FixedPolynomial.evaluate_add ops.toOps polynomialLaws,
    evaluate_subtract, evaluate_affineConstant]

/-- Replace exactly one coordinate between a fixed prefix and suffix. -/
def cubeSlice
    {variables : Nat}
    (before after : List K)
    (length : before.length + 1 + after.length = variables)
    (point : K) : CubePoint K variables where
  coordinates := before ++ point :: after
  dimension := by simp; omega

/-- A finite weighted sum of same-degree functions has a same-degree
coefficient representation. -/
theorem polynomial_sum_exists
    {Index : Type}
    {degree : Nat}
    (indices : List Index)
    (weight : Index -> K)
    (value : Index -> K -> K)
    (represented : forall index, index ∈ indices ->
      exists polynomial : Polynomial degree, forall point,
        polynomial.evaluate ops.toOps point = value index point) :
    exists polynomial : Polynomial degree, forall point,
      polynomial.evaluate ops.toOps point =
        FiniteSumAlgebra.sumMap ops indices fun index =>
          ops.mul (weight index) (value index point) := by
  induction indices with
  | nil =>
      refine ⟨SumCheck.Finite.FixedPolynomial.zero ops.toOps degree, ?_⟩
      intro point
      exact SumCheck.Finite.FixedPolynomial.evaluate_zero
        ops.toOps polynomialLaws degree point
  | cons index indices inductionHypothesis =>
      rcases represented index (by simp) with
        ⟨headPolynomial, headRepresents⟩
      rcases inductionHypothesis (by
        intro tail tailMember
        exact represented tail (by simp [tailMember])) with
        ⟨tailPolynomial, tailRepresents⟩
      refine ⟨SumCheck.Finite.FixedPolynomial.add ops.toOps
        (SumCheck.Finite.FixedPolynomial.scale ops.toOps
          (weight index) headPolynomial)
        tailPolynomial, ?_⟩
      intro point
      rw [SumCheck.Finite.FixedPolynomial.evaluate_add
          ops.toOps polynomialLaws,
        SumCheck.Finite.FixedPolynomial.evaluate_scale
          ops.toOps polynomialLaws,
        headRepresents, tailRepresents]
      rfl

/-- Summing an explicit Boolean suffix preserves any fixed per-variable
degree representation. -/
theorem sumCompletions_represents
    {degree : Nat}
    (polynomial : List K -> K)
    (fixed : List K)
    (remaining : Nat)
    (represented : forall vertex : BooleanVertex remaining,
      Represents degree fun point =>
        polynomial
          ((fixed ++ [point]) ++
            SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex)) :
    Represents degree fun point =>
      SumCheck.Finite.HypercubeTruth.sumCompletions ops.toOps polynomial
        (fixed ++ [point]) remaining := by
  have summed := polynomial_sum_exists
    (BooleanVertex.all remaining)
    (fun _ => ops.one)
    (fun vertex point =>
      polynomial
        ((fixed ++ [point]) ++
          SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex))
    (by
      intro vertex _
      exact represented vertex)
  rcases summed with ⟨sumPolynomial, sumRepresents⟩
  refine ⟨sumPolynomial, ?_⟩
  intro point
  rw [sumRepresents]
  change _ = SumCheck.Finite.HypercubeTruth.sumCompletions ops.toOps
    polynomial (fixed ++ [point]) remaining
  rw [SumCheckTruthPath.sumCompletions_eq_vertexSum ops laws]
  unfold FiniteSumAlgebra.sumMap
  congr 1
  apply List.map_congr_left
  intro vertex _
  rw [laws.one_mul]

/-- Every coordinate slice of an explicit Boolean-table MLE is affine. -/
theorem evaluateCoordinates_affine
    {variables : Nat}
    (table : BooleanTable K variables)
    (before after : List K)
    (length : before.length + 1 + after.length = variables) :
    exists polynomial : Polynomial 1, forall point,
      polynomial.evaluate ops.toOps point =
        table.evaluateCoordinates ops (before ++ point :: after) := by
  induction table generalizing before after with
  | leaf => simp at length
  | @branch variables low high lowInduction highInduction =>
      cases before with
      | nil =>
          exact ⟨SumCheck.Finite.FixedPolynomial.affine
            (low.evaluateCoordinates ops after)
            (ops.sub
              (high.evaluateCoordinates ops after)
              (low.evaluateCoordinates ops after)), by
                intro point
                rw [SumCheck.Finite.FixedPolynomial.evaluate_affine
                  ops.toOps polynomialLaws]
                rfl⟩
      | cons head before =>
          have tailLength : before.length + 1 + after.length = variables := by
            simp only [List.length_cons] at length
            omega
          rcases lowInduction before after tailLength with
            ⟨lowPolynomial, lowRepresents⟩
          rcases highInduction before after tailLength with
            ⟨highPolynomial, highRepresents⟩
          refine ⟨SumCheck.Finite.FixedPolynomial.add ops.toOps lowPolynomial
            (SumCheck.Finite.FixedPolynomial.scale ops.toOps head
              (subtract highPolynomial lowPolynomial)), ?_⟩
          intro point
          rw [SumCheck.Finite.FixedPolynomial.evaluate_add
              ops.toOps polynomialLaws,
            SumCheck.Finite.FixedPolynomial.evaluate_scale
              ops.toOps polynomialLaws,
            evaluate_subtract, lowRepresents, highRepresents]
          rfl

private theorem equalityFactor_eq_affine (point beta : K) :
    SumCheckTruthPath.equalityFactor ops point beta =
      ops.add (ops.sub ops.one beta)
        (ops.mul point (ops.sub beta (ops.sub ops.one beta))) := by
  calc
    SumCheckTruthPath.equalityFactor ops point beta =
        ops.add
          (ops.mul (ops.sub ops.one point) (ops.sub ops.one beta))
          (ops.mul point beta) := by
      rfl
    _ = ops.add
        (ops.add (ops.sub ops.one beta)
          (ops.neg (ops.mul point (ops.sub ops.one beta))))
        (ops.mul point beta) := by
      unfold InterpolationOps.sub
      rw [laws.right_distrib ops.one (ops.neg point)
        (ops.add ops.one (ops.neg beta))]
      rw [laws.one_mul, laws.neg_mul]
    _ = ops.add (ops.sub ops.one beta)
        (ops.add (ops.mul point beta)
          (ops.neg (ops.mul point (ops.sub ops.one beta)))) := by
      letI : Std.Associative ops.add := ⟨laws.add_assoc⟩
      letI : Std.Commutative ops.add := ⟨laws.add_comm⟩
      ac_rfl
    _ = ops.add (ops.sub ops.one beta)
        (ops.mul point (ops.sub beta (ops.sub ops.one beta))) := by
      congr 1
      exact (FiniteSumAlgebra.mul_sub ops laws point beta
        (ops.sub ops.one beta)).symm

private theorem equalityFactor_comm (left right : K) :
    SumCheckTruthPath.equalityFactor ops left right =
      SumCheckTruthPath.equalityFactor ops right left := by
  unfold SumCheckTruthPath.equalityFactor
  rw [laws.mul_comm (ops.sub ops.one left) (ops.sub ops.one right),
    laws.mul_comm left right]

/-- The coordinate-list equality polynomial is symmetric on equal-length
lists. -/
theorem pointEqualityCoordinates_comm
    (left right : List K)
    (sameLength : left.length = right.length) :
    SumCheckTruthPath.pointEqualityCoordinates ops left right =
      SumCheckTruthPath.pointEqualityCoordinates ops right left := by
  induction left generalizing right with
  | nil =>
      have rightEmpty : right = [] := List.eq_nil_of_length_eq_zero sameLength.symm
      subst right
      rfl
  | cons left lefts inductionHypothesis =>
      cases right with
      | nil => simp at sameLength
      | cons right rights =>
          have tailLength : lefts.length = rights.length := by
            simpa using sameLength
          simp only [SumCheckTruthPath.pointEqualityCoordinates]
          rw [equalityFactor_comm, inductionHypothesis rights tailLength]

/-- Every coordinate slice of an equality polynomial is affine. -/
theorem pointEqualityCoordinates_affine
    (before after beta : List K)
    (length : before.length + 1 + after.length = beta.length) :
    exists polynomial : Polynomial 1, forall point,
      polynomial.evaluate ops.toOps point =
        SumCheckTruthPath.pointEqualityCoordinates ops
          (before ++ point :: after) beta := by
  induction beta generalizing before after with
  | nil => simp at length
  | cons betaHead betaTail inductionHypothesis =>
      cases before with
      | nil =>
          let tailEquality :=
            SumCheckTruthPath.pointEqualityCoordinates ops after betaTail
          let oneMinusBeta := ops.sub ops.one betaHead
          let factor := SumCheck.Finite.FixedPolynomial.affine oneMinusBeta
            (ops.sub betaHead oneMinusBeta)
          refine ⟨SumCheck.Finite.FixedPolynomial.scale
            ops.toOps tailEquality factor, ?_⟩
          intro point
          rw [SumCheck.Finite.FixedPolynomial.evaluate_scale
              ops.toOps polynomialLaws,
            SumCheck.Finite.FixedPolynomial.evaluate_affine
              ops.toOps polynomialLaws]
          change ops.mul tailEquality
              (ops.add oneMinusBeta
                (ops.mul point (ops.sub betaHead oneMinusBeta))) =
            ops.mul (SumCheckTruthPath.equalityFactor ops point betaHead)
              tailEquality
          rw [laws.mul_comm tailEquality, equalityFactor_eq_affine]
      | cons head before =>
          have tailLength : before.length + 1 + after.length =
              betaTail.length := by
            simp only [List.length_cons] at length
            omega
          rcases inductionHypothesis before after tailLength with
            ⟨tailPolynomial, tailRepresents⟩
          refine ⟨SumCheck.Finite.FixedPolynomial.scale ops.toOps
            (SumCheckTruthPath.equalityFactor ops head betaHead)
            tailPolynomial, ?_⟩
          intro point
          rw [SumCheck.Finite.FixedPolynomial.evaluate_scale
              ops.toOps polynomialLaws,
            tailRepresents]
          rfl

/-- Every coordinate slice is also affine when the varying point is the
right argument of the symmetric equality polynomial. -/
theorem pointEqualityCoordinates_right_affine
    (beta before after : List K)
    (length : before.length + 1 + after.length = beta.length) :
    exists polynomial : Polynomial 1, forall point,
      polynomial.evaluate ops.toOps point =
        SumCheckTruthPath.pointEqualityCoordinates ops beta
          (before ++ point :: after) := by
  rcases pointEqualityCoordinates_affine before after beta length with
    ⟨polynomial, represents⟩
  refine ⟨polynomial, ?_⟩
  intro point
  rw [represents]
  exact pointEqualityCoordinates_comm
    (before ++ point :: after) beta (by
      simp only [List.length_append, List.length_cons]
      omega)

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.DegreeSupport
