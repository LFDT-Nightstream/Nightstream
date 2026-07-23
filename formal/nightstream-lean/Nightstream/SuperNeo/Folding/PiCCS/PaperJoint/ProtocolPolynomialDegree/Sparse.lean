import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialDegree.Support

/-!
Sparse-CCS substitution for the paper `Pi_CCS` one-variable degree proof.

Owns: explicit fixed-polynomial powers of affine matrix-image slices, sparse
monomial substitution, and the syntax-derived equality-gated CCS ceiling.

Does not own: a concrete matrix source, protocol aggregation, SumCheck,
probability, Fiat--Shamir, Rust, R1CS, artifacts, or costs. Declared CCS degree
metadata is never used; the ceiling comes only from explicit monomial syntax.
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialDegree.Sparse

open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open ProtocolPolynomialDegree.Support

universe uField

private def affinePower
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (polynomial : Polynomial Field 1)
    (exponent : Nat) : Polynomial Field exponent :=
  SumCheck.Finite.FixedPolynomial.widen ops.toOps (by omega)
    (SumCheck.Finite.FixedPolynomial.power ops.toOps polynomial exponent)

private theorem evaluate_affinePower
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    (polynomial : Polynomial Field 1)
    (exponent : Nat)
    (point : Field) :
    (affinePower ops polynomial exponent).evaluate ops.toOps point =
      CCSResidualTable.pow ops
        (polynomial.evaluate ops.toOps point) exponent := by
  rw [affinePower,
    SumCheck.Finite.FixedPolynomial.evaluate_widen
      ops.toOps (polynomialLaws laws),
    SumCheck.Finite.FixedPolynomial.evaluate_power
      ops.toOps (polynomialLaws laws)]
  induction exponent with
  | zero => rfl
  | succ exponent inductionHypothesis =>
      simp only [SumCheck.Finite.FixedPolynomial.valuePower,
        CCSResidualTable.pow]
      rw [inductionHypothesis]

private def productPowers
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {Index : Type}
    (exponents : Index -> Nat)
    (polynomials : Index -> Polynomial Field 1) :
    (indices : List Index) ->
      Polynomial Field ((indices.map exponents).sum)
  | [] => SumCheck.Finite.FixedPolynomial.constant ops.one
  | index :: indices =>
      SumCheck.Finite.FixedPolynomial.mul ops.toOps
        (affinePower ops (polynomials index) (exponents index))
        (productPowers ops exponents polynomials indices)

private theorem evaluate_productPowers
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    {Index : Type}
    (exponents : Index -> Nat)
    (polynomials : Index -> Polynomial Field 1)
    (indices : List Index)
    (point : Field) :
    (productPowers ops exponents polynomials indices).evaluate ops.toOps point =
      indices.foldr
        (fun index total =>
          ops.mul
            (CCSResidualTable.pow ops
              ((polynomials index).evaluate ops.toOps point)
              (exponents index))
            total)
        ops.one := by
  induction indices with
  | nil =>
      exact SumCheck.Finite.FixedPolynomial.evaluate_constant
        ops.toOps (polynomialLaws laws) ops.one point
  | cons index indices inductionHypothesis =>
      calc
        (productPowers ops exponents polynomials
            (index :: indices)).evaluate ops.toOps point =
          ops.mul
            ((affinePower ops (polynomials index)
              (exponents index)).evaluate ops.toOps point)
            ((productPowers ops exponents polynomials indices).evaluate
              ops.toOps point) :=
          SumCheck.Finite.FixedPolynomial.evaluate_mul
            ops.toOps (polynomialLaws laws) _ _ point
        _ = ops.mul
            (CCSResidualTable.pow ops
              ((polynomials index).evaluate ops.toOps point)
              (exponents index))
            (indices.foldr
              (fun next total =>
                ops.mul
                  (CCSResidualTable.pow ops
                    ((polynomials next).evaluate ops.toOps point)
                    (exponents next))
                  total)
              ops.one) := by
          rw [evaluate_affinePower laws, inductionHypothesis]
        _ = (index :: indices).foldr
            (fun next total =>
              ops.mul
                (CCSResidualTable.pow ops
                  ((polynomials next).evaluate ops.toOps point)
                  (exponents next))
                total)
            ops.one := rfl

private theorem foldl_mul_eq_mul_foldr
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    {Index : Type}
    (indices : List Index)
    (factor : Index -> Field)
    (accumulated : Field) :
    indices.foldl (fun value index => ops.mul value (factor index))
        accumulated =
      ops.mul accumulated
        (indices.foldr (fun index total => ops.mul (factor index) total)
          ops.one) := by
  induction indices generalizing accumulated with
  | nil => exact (laws.mul_one accumulated).symm
  | cons index indices inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis]
      exact laws.mul_assoc _ _ _

private def monomialPolynomial
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {matrixCount : Nat}
    (monomial : CCSResidualTable.Monomial Field matrixCount)
    (matrixPolynomials : Fin matrixCount -> Polynomial Field 1) :
    Polynomial Field monomial.totalDegree :=
  SumCheck.Finite.FixedPolynomial.scale ops.toOps monomial.coefficient
    (productPowers ops monomial.exponents matrixPolynomials
      (canonicalFinIndices matrixCount))

private theorem evaluate_monomialPolynomial
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    {matrixCount : Nat}
    (monomial : CCSResidualTable.Monomial Field matrixCount)
    (matrixPolynomials : Fin matrixCount -> Polynomial Field 1)
    (point : Field) :
    (monomialPolynomial ops monomial matrixPolynomials).evaluate
        ops.toOps point =
      CCSResidualTable.evaluateMonomial ops monomial fun matrix =>
        (matrixPolynomials matrix).evaluate ops.toOps point := by
  calc
    (monomialPolynomial ops monomial matrixPolynomials).evaluate
        ops.toOps point =
      ops.mul monomial.coefficient
        ((productPowers ops monomial.exponents matrixPolynomials
          (canonicalFinIndices matrixCount)).evaluate ops.toOps point) :=
      SumCheck.Finite.FixedPolynomial.evaluate_scale
        ops.toOps (polynomialLaws laws) _ _ point
    _ = ops.mul monomial.coefficient
        ((canonicalFinIndices matrixCount).foldr
          (fun matrix total =>
            ops.mul
              (CCSResidualTable.pow ops
                ((matrixPolynomials matrix).evaluate ops.toOps point)
                (monomial.exponents matrix))
              total)
          ops.one) := by
      exact congrArg (ops.mul monomial.coefficient)
        (evaluate_productPowers laws monomial.exponents matrixPolynomials
          (canonicalFinIndices matrixCount) point)
    _ = CCSResidualTable.evaluateMonomial ops monomial fun matrix =>
        (matrixPolynomials matrix).evaluate ops.toOps point := by
      unfold CCSResidualTable.evaluateMonomial
      exact (foldl_mul_eq_mul_foldr laws
        (canonicalFinIndices matrixCount)
        (fun matrix => CCSResidualTable.pow ops
          ((matrixPolynomials matrix).evaluate ops.toOps point)
          (monomial.exponents matrix))
        monomial.coefficient).symm

/-- Substituting affine matrix images into one explicit sparse monomial has
exactly the monomial's syntax-derived total degree. -/
theorem monomial_represents
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    {matrixCount : Nat}
    (monomial : CCSResidualTable.Monomial Field matrixCount)
    (matrixImage : Fin matrixCount -> Field -> Field)
    (matrixAffine : forall matrix,
      Represents ops 1 (matrixImage matrix)) :
    Represents ops monomial.totalDegree fun point =>
      CCSResidualTable.evaluateMonomial ops monomial fun matrix =>
        matrixImage matrix point := by
  let matrixPolynomial : Fin matrixCount -> Polynomial Field 1 :=
    fun matrix => Classical.choose (matrixAffine matrix)
  have matrixRepresents : forall matrix point,
      (matrixPolynomial matrix).evaluate ops.toOps point =
        matrixImage matrix point := by
    intro matrix point
    exact Classical.choose_spec (matrixAffine matrix) point
  refine ⟨monomialPolynomial ops monomial matrixPolynomial, ?_⟩
  intro point
  rw [evaluate_monomialPolynomial laws]
  congr 1
  funext matrix
  exact matrixRepresents matrix point

/-- Multiplying the explicit sparse CCS polynomial by one affine equality
selector fits the exact maximum `totalDegree + 1` computed from its terms. -/
theorem equalityGated_represents
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    {matrixCount : Nat}
    (polynomial : CCSResidualTable.ConstraintPolynomial Field matrixCount)
    (selector : Field -> Field)
    (selectorAffine : Represents ops 1 selector)
    (matrixImage : Fin matrixCount -> Field -> Field)
    (matrixAffine : forall matrix,
      Represents ops 1 (matrixImage matrix)) :
    Represents ops polynomial.canonicalEqualityGatedDegreeBound fun point =>
      ops.mul (selector point)
        (CCSResidualTable.evaluatePolynomial ops polynomial fun matrix =>
          matrixImage matrix point) := by
  rcases weightedSum laws
    polynomial.terms
    (fun _ => ops.one)
    (fun monomial point =>
      ops.mul (selector point)
        (CCSResidualTable.evaluateMonomial ops monomial fun matrix =>
          matrixImage matrix point))
    (by
      intro monomial member
      have multiplied := Represents.mul laws selectorAffine
        (monomial_represents laws monomial matrixImage matrixAffine)
      have atTermDegree :
          Represents ops (monomial.totalDegree + 1) fun point =>
            ops.mul (selector point)
              (CCSResidualTable.evaluateMonomial ops monomial fun matrix =>
                matrixImage matrix point) := by
        simpa [Nat.add_comm] using multiplied
      exact Represents.widen laws
        (polynomial.term_totalDegree_succ_le_canonicalEqualityGatedDegreeBound
          monomial member)
        atTermDegree) with
    ⟨sumPolynomial, sumRepresents⟩
  refine ⟨sumPolynomial, ?_⟩
  intro point
  change sumPolynomial.evaluate ops.toOps point =
    ops.mul (selector point)
      (CCSResidualTable.evaluatePolynomial ops polynomial fun matrix =>
        matrixImage matrix point)
  rw [sumRepresents]
  rw [CCSResidualTable.evaluatePolynomial_eq_sumMap ops laws]
  calc
    FiniteSumAlgebra.sumMap ops polynomial.terms
        (fun monomial =>
          ops.mul ops.one
            (ops.mul (selector point)
              (CCSResidualTable.evaluateMonomial ops monomial fun matrix =>
                matrixImage matrix point))) =
      FiniteSumAlgebra.sumMap ops polynomial.terms
        (fun monomial =>
          ops.mul (selector point)
            (CCSResidualTable.evaluateMonomial ops monomial fun matrix =>
              matrixImage matrix point)) := by
        apply FiniteSumAlgebra.sumMap_congr
        intro monomial _
        rw [laws.one_mul]
    _ = ops.mul (selector point)
        (FiniteSumAlgebra.sumMap ops polynomial.terms fun monomial =>
          CCSResidualTable.evaluateMonomial ops monomial fun matrix =>
            matrixImage matrix point) :=
      FiniteSumAlgebra.sumMap_mul_left ops laws _ _ _

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialDegree.Sparse
