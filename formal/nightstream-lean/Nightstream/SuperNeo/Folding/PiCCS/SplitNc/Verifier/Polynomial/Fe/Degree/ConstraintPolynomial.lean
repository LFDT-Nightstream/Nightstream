import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.Degree.Source

/-!
Sparse CCS substitution for the Split-NC FE row-degree proof.

Owns: exact natural powers of affine matrix-image slices, finite products of
those powers, one sparse monomial representation, and the complete
equality-gated sparse-polynomial representation at the syntax-derived degree
ceiling.

Does not own: source MLE affinity, fresh-source gamma compression, carried CE
terms, complete FE row/lane composition, SumCheck rounds, transcripts, Rust,
R1CS, rows, or costs.

Emits constraints: no.

Authority boundary: degree comes only from each explicit exponent vector and
term membership in the sparse polynomial. Declared degree metadata and Rust's
current shared `d_sc` are never consulted.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.fe.degree.ccs.power` | an affine matrix image raised to exponent `e` has degree `e` | derived | `affinePower`, `evaluate_affinePower` |
| `nifs.pi_ccs.fe.degree.ccs.product` | multiplying explicit matrix powers adds their degrees | derived | `productPowers`, `evaluate_productPowers` |
| `nifs.pi_ccs.fe.degree.ccs.monomial` | affine substitution has the monomial's explicit total degree | derived | `monomial_row_represents` |
| `nifs.pi_ccs.fe.degree.ccs.equality_gated` | `eq * f` fits the maximum explicit `totalDegree + 1` | derived | `equalityGated_row_represents` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.Degree.ConstraintPolynomial

set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.DegreeSupport
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.Degree.Source

private def affinePower
    (polynomial : Polynomial 1)
    (exponent : Nat) : Polynomial exponent :=
  SumCheck.Finite.FixedPolynomial.widen ops.toOps (by omega)
    (SumCheck.Finite.FixedPolynomial.power ops.toOps polynomial exponent)

private theorem evaluate_affinePower
    (polynomial : Polynomial 1)
    (exponent : Nat)
    (point : K) :
    (affinePower polynomial exponent).evaluate ops.toOps point =
      CCSResidualTable.pow ops
        (polynomial.evaluate ops.toOps point) exponent := by
  rw [affinePower,
    SumCheck.Finite.FixedPolynomial.evaluate_widen ops.toOps polynomialLaws,
    SumCheck.Finite.FixedPolynomial.evaluate_power ops.toOps polynomialLaws]
  induction exponent with
  | zero => rfl
  | succ exponent inductionHypothesis =>
      simp only [SumCheck.Finite.FixedPolynomial.valuePower,
        CCSResidualTable.pow]
      rw [inductionHypothesis]

private def productPowers
    {Index : Type}
    (exponents : Index -> Nat)
    (polynomials : Index -> Polynomial 1) :
    (indices : List Index) ->
      Polynomial ((indices.map exponents).sum)
  | [] => SumCheck.Finite.FixedPolynomial.constant ops.one
  | index :: indices =>
      SumCheck.Finite.FixedPolynomial.mul ops.toOps
        (affinePower (polynomials index) (exponents index))
        (productPowers exponents polynomials indices)

private theorem evaluate_productPowers
    {Index : Type}
    (exponents : Index -> Nat)
    (polynomials : Index -> Polynomial 1)
    (indices : List Index)
    (point : K) :
    (productPowers exponents polynomials indices).evaluate ops.toOps point =
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
        ops.toOps polynomialLaws ops.one point
  | cons index indices inductionHypothesis =>
      calc
        (productPowers exponents polynomials (index :: indices)).evaluate
            ops.toOps point =
          ops.mul
            ((affinePower (polynomials index)
              (exponents index)).evaluate ops.toOps point)
            ((productPowers exponents polynomials indices).evaluate
              ops.toOps point) :=
          SumCheck.Finite.FixedPolynomial.evaluate_mul
            ops.toOps polynomialLaws _ _ point
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
          rw [evaluate_affinePower, inductionHypothesis]
        _ = (index :: indices).foldr
            (fun next total =>
              ops.mul
                (CCSResidualTable.pow ops
                  ((polynomials next).evaluate ops.toOps point)
                  (exponents next))
                total)
            ops.one := rfl

private theorem foldl_mul_eq_mul_foldr
    {Index : Type}
    (indices : List Index)
    (factor : Index -> K)
    (accumulated : K) :
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
    {matrixCount : Nat}
    (monomial : CCSResidualTable.Monomial K matrixCount)
    (matrixPolynomials : Fin matrixCount -> Polynomial 1) :
    Polynomial monomial.totalDegree :=
  SumCheck.Finite.FixedPolynomial.scale ops.toOps monomial.coefficient
    (productPowers monomial.exponents matrixPolynomials
      (canonicalFinIndices matrixCount))

private theorem evaluate_monomialPolynomial
    {matrixCount : Nat}
    (monomial : CCSResidualTable.Monomial K matrixCount)
    (matrixPolynomials : Fin matrixCount -> Polynomial 1)
    (point : K) :
    (monomialPolynomial monomial matrixPolynomials).evaluate ops.toOps point =
      CCSResidualTable.evaluateMonomial ops monomial fun matrix =>
        (matrixPolynomials matrix).evaluate ops.toOps point := by
  calc
    (monomialPolynomial monomial matrixPolynomials).evaluate ops.toOps point =
        ops.mul monomial.coefficient
          ((productPowers monomial.exponents matrixPolynomials
            (canonicalFinIndices matrixCount)).evaluate ops.toOps point) :=
      SumCheck.Finite.FixedPolynomial.evaluate_scale
        ops.toOps polynomialLaws _ _ point
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
        (evaluate_productPowers monomial.exponents matrixPolynomials
          (canonicalFinIndices matrixCount) point)
    _ = CCSResidualTable.evaluateMonomial ops monomial fun matrix =>
        (matrixPolynomials matrix).evaluate ops.toOps point := by
      unfold CCSResidualTable.evaluateMonomial
      exact (foldl_mul_eq_mul_foldr
        (canonicalFinIndices matrixCount)
        (fun matrix => CCSResidualTable.pow ops
          ((matrixPolynomials matrix).evaluate ops.toOps point)
          (monomial.exponents matrix))
        monomial.coefficient).symm

/-- Substituting affine row images into one explicit sparse monomial produces
a coefficient representation at exactly its syntax-derived total degree. -/
theorem monomial_row_represents
    {shape : SemanticShape}
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (monomial : CCSResidualTable.Monomial K shape.matrixCount)
    (lane : Fin ringDegree)
    (before after : List K)
    (length : before.length + 1 + after.length = shape.rowVariables) :
    Represents monomial.totalDegree fun point =>
      CCSResidualTable.evaluateMonomial ops monomial fun matrix =>
        sourceYRingAt data (cubeSlice before after length point)
          source matrix lane := by
  let matrixPolynomial : Fin shape.matrixCount -> Polynomial 1 :=
    fun matrix => Classical.choose
      (sourceYRingAt_row_affine data source matrix lane before after length)
  have matrixRepresents : forall matrix point,
      (matrixPolynomial matrix).evaluate ops.toOps point =
        sourceYRingAt data (cubeSlice before after length point)
          source matrix lane := by
    intro matrix point
    exact Classical.choose_spec
      (sourceYRingAt_row_affine data source matrix lane before after length)
      point
  refine ⟨monomialPolynomial monomial matrixPolynomial, ?_⟩
  intro point
  rw [evaluate_monomialPolynomial]
  congr 1
  funext matrix
  exact matrixRepresents matrix point

/-- Multiplying an explicit sparse CCS polynomial by one affine row selector
fits exactly the canonical maximum `totalDegree + 1` derived from its terms.
The declared CCS degree metadata is absent from the statement and proof. -/
theorem equalityGated_row_represents
    {shape : SemanticShape}
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (polynomial : CCSResidualTable.ConstraintPolynomial K shape.matrixCount)
    (lane : Fin ringDegree)
    (selector : K -> K)
    (selectorRepresents : Represents 1 selector)
    (before after : List K)
    (length : before.length + 1 + after.length = shape.rowVariables) :
    Represents polynomial.canonicalEqualityGatedDegreeBound fun point =>
      ops.mul (selector point)
        (CCSResidualTable.evaluatePolynomial ops polynomial fun matrix =>
          sourceYRingAt data (cubeSlice before after length point)
            source matrix lane) := by
  rcases polynomial_sum_exists
    polynomial.terms
    (fun _ => ops.one)
    (fun monomial point =>
      ops.mul (selector point)
        (CCSResidualTable.evaluateMonomial ops monomial fun matrix =>
          sourceYRingAt data (cubeSlice before after length point)
            source matrix lane))
    (by
      intro monomial member
      have multiplied := Represents.mul selectorRepresents
        (monomial_row_represents data source monomial lane
          before after length)
      have atTermDegree : Represents (monomial.totalDegree + 1) fun point =>
          ops.mul (selector point)
            (CCSResidualTable.evaluateMonomial ops monomial fun matrix =>
              sourceYRingAt data (cubeSlice before after length point)
                source matrix lane) := by
        simpa [Nat.add_comm] using multiplied
      exact Represents.widen
        (polynomial.term_totalDegree_succ_le_canonicalEqualityGatedDegreeBound
          monomial member)
        atTermDegree) with
    ⟨sumPolynomial, sumRepresents⟩
  refine ⟨sumPolynomial, ?_⟩
  intro point
  change sumPolynomial.evaluate ops.toOps point =
    ops.mul (selector point)
      (CCSResidualTable.evaluatePolynomial ops polynomial fun matrix =>
        sourceYRingAt data (cubeSlice before after length point)
          source matrix lane)
  rw [sumRepresents]
  rw [CCSResidualTable.evaluatePolynomial_eq_sumMap ops laws]
  calc
    FiniteSumAlgebra.sumMap ops polynomial.terms
        (fun monomial =>
          ops.mul ops.one
            (ops.mul (selector point)
              (CCSResidualTable.evaluateMonomial ops monomial fun matrix =>
                sourceYRingAt data (cubeSlice before after length point)
                  source matrix lane))) =
      FiniteSumAlgebra.sumMap ops polynomial.terms
        (fun monomial =>
          ops.mul (selector point)
            (CCSResidualTable.evaluateMonomial ops monomial fun matrix =>
              sourceYRingAt data (cubeSlice before after length point)
                source matrix lane)) := by
        apply FiniteSumAlgebra.sumMap_congr
        intro monomial _
        rw [laws.one_mul]
    _ = ops.mul (selector point)
        (FiniteSumAlgebra.sumMap ops polynomial.terms fun monomial =>
          CCSResidualTable.evaluateMonomial ops monomial fun matrix =>
            sourceYRingAt data (cubeSlice before after length point)
              source matrix lane) :=
      FiniteSumAlgebra.sumMap_mul_left ops laws _ _ _

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.Degree.ConstraintPolynomial
