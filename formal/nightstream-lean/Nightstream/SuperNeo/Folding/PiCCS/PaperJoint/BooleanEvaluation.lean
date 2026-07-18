import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanTable
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Sampling

/-!
Evaluation semantics for the canonical Boolean-table coefficient transform.

Owns: an independently recursive multilinear evaluation of an explicit
Boolean table and its equality with evaluation of `toAlphaPolynomial` against
the verifier-owned canonical alpha basis.

Does not own: Boolean-leaf lookup, alignment of this low/high order with an
external paper or production bit serialization, CCS or norm residual
construction, SumCheck, sampling, root counting, Fiat--Shamir, Rust, R1CS, or
constraint counts.

Emits constraints: no.

Authority boundary: the table and evaluation point are explicit finite data.
The polynomial coefficients and basis are derived by `BooleanTable`; no
caller-supplied evaluator or polynomial identity is assumed.

| Mathematical object | Independent definition | Proven relation |
|---|---|---|
| table MLE | recurse as `low + x * (high - low)` | equals canonical polynomial evaluation |
| canonical polynomial | `toAlphaPolynomial` on the squarefree alpha basis | no external evaluation oracle |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

open Nightstream.SuperNeo.SumCheck

universe uField

/-- Explicit algebraic laws used by the evaluation proof. They are stated over
the same operations that construct and evaluate the polynomial. -/
structure InterpolationEvaluationLaws
    {Field : Type uField}
    (ops : InterpolationOps Field) : Prop where
  add_assoc : forall left middle right,
    ops.add (ops.add left middle) right =
      ops.add left (ops.add middle right)
  add_comm : forall left right, ops.add left right = ops.add right left
  zero_add : forall value, ops.add ops.zero value = value
  add_zero : forall value, ops.add value ops.zero = value
  mul_assoc : forall left middle right,
    ops.mul (ops.mul left middle) right =
      ops.mul left (ops.mul middle right)
  mul_comm : forall left right, ops.mul left right = ops.mul right left
  one_mul : forall value, ops.mul ops.one value = value
  mul_one : forall value, ops.mul value ops.one = value
  mul_zero : forall value, ops.mul value ops.zero = ops.zero
  left_distrib : forall left middle right,
    ops.mul left (ops.add middle right) =
      ops.add (ops.mul left middle) (ops.mul left right)
  right_distrib : forall left middle right,
    ops.mul (ops.add left middle) right =
      ops.add (ops.mul left right) (ops.mul middle right)
  add_neg : forall value, ops.add value (ops.neg value) = ops.zero
  neg_add : forall left right,
    ops.neg (ops.add left right) =
      ops.add (ops.neg left) (ops.neg right)
  neg_mul : forall left right,
    ops.mul (ops.neg left) right = ops.neg (ops.mul left right)

namespace BooleanTable

private def evaluateTerms
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (point : CubePoint Field shape.cubeVariables) :
    List Field -> List (AlphaMonomial shape) -> Field
  | [], [] => ops.zero
  | coefficient :: coefficients, monomial :: monomials =>
      ops.add
        (ops.mul coefficient (monomial.evaluate ops.toOps point))
        (evaluateTerms ops point coefficients monomials)
  | _, _ => ops.zero

private theorem alphaPolynomial_evaluate_eq_evaluateTerms
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : InterpolationOps Field)
    (polynomial : AlphaPolynomial Field basis)
    (point : CubePoint Field shape.cubeVariables) :
    polynomial.evaluate ops.toOps point =
      evaluateTerms ops point polynomial.coefficients basis.monomials := by
  unfold AlphaPolynomial.evaluate
  generalize polynomial.coefficients = coefficients
  generalize basis.monomials = monomials
  induction coefficients generalizing monomials with
  | nil => cases monomials <;> rfl
  | cons coefficient coefficients inductionHypothesis =>
      cases monomials with
      | nil => rfl
      | cons monomial monomials =>
          change ops.add
              (ops.mul coefficient (monomial.evaluate ops.toOps point)) _ =
            ops.add
              (ops.mul coefficient (monomial.evaluate ops.toOps point))
              (evaluateTerms ops point coefficients monomials)
          congr 1
          exact inductionHypothesis monomials

/-- Independent recursive multilinear extension of a Boolean table. Mismatch
branches only make the raw function total; a `CubePoint` rules them out. -/
def evaluateCoordinates
    {Field : Type uField}
    (ops : InterpolationOps Field) :
    {variables : Nat} -> BooleanTable Field variables -> List Field -> Field
  | 0, .leaf value, [] => value
  | _ + 1, .branch low high, coordinate :: coordinates =>
      ops.add
        (low.evaluateCoordinates ops coordinates)
        (ops.mul coordinate
          (ops.sub
            (high.evaluateCoordinates ops coordinates)
            (low.evaluateCoordinates ops coordinates)))
  | _, _, _ => ops.zero

/-- Evaluate the independent table MLE at a dimension-checked point. -/
def evaluate
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {variables : Nat}
    (table : BooleanTable Field variables)
    (point : CubePoint Field variables) : Field :=
  table.evaluateCoordinates ops point.coordinates

private def evaluationShape (template : Shape) (variables : Nat) : Shape :=
  { template with cubeVariables := variables }

private def prependMonomial
    (headExponent : Nat)
    {template : Shape}
    {variables : Nat}
    (monomial : AlphaMonomial (evaluationShape template variables)) :
    AlphaMonomial (evaluationShape template (variables + 1)) where
  exponents := headExponent :: monomial.exponents
  arity := by simp [evaluationShape, monomial.arity]

private def prependPoint
    {Field : Type uField}
    {variables : Nat}
    (coordinate : Field)
    (point : CubePoint Field variables) : CubePoint Field (variables + 1) where
  coordinates := coordinate :: point.coordinates
  dimension := by simp [point.dimension]

private theorem pmap_prependExponent
    (template : Shape)
    (headExponent variables : Nat)
    (vectors : List (List Nat))
    (vectorArities : forall vector, vector ∈ vectors ->
      vector.length = variables)
    (prependedArities : forall vector,
      vector ∈ vectors.map (fun exponents => headExponent :: exponents) ->
        vector.length = variables + 1) :
    List.pmap
        (fun exponents arity =>
          ({ exponents := exponents
             arity := arity } :
            AlphaMonomial (evaluationShape template (variables + 1))))
        (vectors.map (fun exponents => headExponent :: exponents))
        prependedArities =
      (List.pmap
        (fun exponents arity =>
          ({ exponents := exponents
             arity := arity } : AlphaMonomial (evaluationShape template variables)))
        vectors vectorArities).map (prependMonomial headExponent) := by
  induction vectors with
  | nil => rfl
  | cons vector vectors inductionHypothesis =>
      simp only [List.map_cons, List.pmap_cons]
      congr 1
      apply inductionHypothesis

private theorem canonicalAlphaMonomials_succ
    (template : Shape) (variables : Nat) :
    canonicalAlphaMonomials (evaluationShape template (variables + 1)) =
      (canonicalAlphaMonomials (evaluationShape template variables)).map
          (prependMonomial 0) ++
      (canonicalAlphaMonomials (evaluationShape template variables)).map
          (prependMonomial 1) := by
  unfold canonicalAlphaMonomials
  simp only [evaluationShape, canonicalExponentVectors, List.pmap_append]
  congr 1
  · apply pmap_prependExponent template
  · apply pmap_prependExponent template

private theorem prependMonomial_zero_evaluate
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {template : Shape}
    {variables : Nat}
    (coordinate : Field)
    (point : CubePoint Field variables)
    (monomial : AlphaMonomial (evaluationShape template variables)) :
    (prependMonomial 0 monomial).evaluate ops.toOps
        (prependPoint coordinate point) =
      monomial.evaluate ops.toOps point := by
  unfold AlphaMonomial.evaluate prependMonomial prependPoint
  change ops.mul ops.one _ = _
  exact laws.one_mul _

private theorem prependMonomial_one_evaluate
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {template : Shape}
    {variables : Nat}
    (coordinate : Field)
    (point : CubePoint Field variables)
    (monomial : AlphaMonomial (evaluationShape template variables)) :
    (prependMonomial 1 monomial).evaluate ops.toOps
        (prependPoint coordinate point) =
      ops.mul coordinate (monomial.evaluate ops.toOps point) := by
  unfold AlphaMonomial.evaluate prependMonomial prependPoint
  change ops.mul (ops.mul coordinate ops.one) _ = ops.mul coordinate _
  rw [laws.mul_one]

private theorem evaluateTerms_prepend_zero
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {template : Shape}
    {variables : Nat}
    (coordinate : Field)
    (point : CubePoint Field variables)
    (coefficients : List Field)
    (monomials : List (AlphaMonomial (evaluationShape template variables))) :
    evaluateTerms ops (prependPoint coordinate point) coefficients
        (monomials.map (prependMonomial 0)) =
      evaluateTerms ops point coefficients monomials := by
  induction coefficients generalizing monomials with
  | nil => cases monomials <;> rfl
  | cons coefficient coefficients inductionHypothesis =>
      cases monomials with
      | nil => rfl
      | cons monomial monomials =>
          simp only [List.map_cons, evaluateTerms,
            prependMonomial_zero_evaluate ops laws]
          rw [inductionHypothesis monomials]

private theorem evaluateTerms_prepend_one
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {template : Shape}
    {variables : Nat}
    (coordinate : Field)
    (point : CubePoint Field variables)
    (coefficients : List Field)
    (monomials : List (AlphaMonomial (evaluationShape template variables))) :
    evaluateTerms ops (prependPoint coordinate point) coefficients
        (monomials.map (prependMonomial 1)) =
      ops.mul coordinate (evaluateTerms ops point coefficients monomials) := by
  induction coefficients generalizing monomials with
  | nil =>
      cases monomials <;> simp [evaluateTerms, laws.mul_zero]
  | cons coefficient coefficients inductionHypothesis =>
      cases monomials with
      | nil => simp [evaluateTerms, laws.mul_zero]
      | cons monomial monomials =>
          simp only [List.map_cons, evaluateTerms,
            prependMonomial_one_evaluate ops laws,
            inductionHypothesis monomials, laws.left_distrib]
          have commuteHead :
              ops.mul coefficient
                  (ops.mul coordinate (monomial.evaluate ops.toOps point)) =
                ops.mul coordinate
                  (ops.mul coefficient (monomial.evaluate ops.toOps point)) := by
            rw [← laws.mul_assoc, laws.mul_comm coefficient coordinate,
              laws.mul_assoc]
          rw [commuteHead]

private theorem evaluateTerms_append
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (point : CubePoint Field shape.cubeVariables)
    (leftCoefficients rightCoefficients : List Field)
    (leftMonomials rightMonomials : List (AlphaMonomial shape))
    (leftLength : leftCoefficients.length = leftMonomials.length) :
    evaluateTerms ops point (leftCoefficients ++ rightCoefficients)
        (leftMonomials ++ rightMonomials) =
      ops.add
        (evaluateTerms ops point leftCoefficients leftMonomials)
        (evaluateTerms ops point rightCoefficients rightMonomials) := by
  induction leftCoefficients generalizing leftMonomials with
  | nil =>
      cases leftMonomials with
      | nil => simp [evaluateTerms, laws.zero_add]
      | cons monomial monomials => simp at leftLength
  | cons coefficient coefficients inductionHypothesis =>
      cases leftMonomials with
      | nil => simp at leftLength
      | cons monomial monomials =>
          simp only [List.length_cons, Nat.succ.injEq] at leftLength
          simp only [List.cons_append, evaluateTerms]
          rw [inductionHypothesis monomials leftLength]
          exact (laws.add_assoc _ _ _).symm

private theorem neg_zero
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) :
    ops.neg ops.zero = ops.zero := by
  have inverse := laws.add_neg ops.zero
  simpa only [laws.zero_add] using inverse

private theorem add_sub_add_sub
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (left right suffixLeft suffixRight : Field) :
    ops.add (ops.sub left right) (ops.sub suffixLeft suffixRight) =
      ops.sub (ops.add left suffixLeft) (ops.add right suffixRight) := by
  unfold InterpolationOps.sub
  rw [laws.neg_add]
  calc
    ops.add (ops.add left (ops.neg right))
        (ops.add suffixLeft (ops.neg suffixRight)) =
      ops.add left
        (ops.add (ops.neg right)
          (ops.add suffixLeft (ops.neg suffixRight))) :=
        laws.add_assoc _ _ _
    _ = ops.add left
        (ops.add suffixLeft
          (ops.add (ops.neg right) (ops.neg suffixRight))) := by
      congr 1
      calc
        ops.add (ops.neg right)
            (ops.add suffixLeft (ops.neg suffixRight)) =
          ops.add (ops.add (ops.neg right) suffixLeft)
            (ops.neg suffixRight) :=
          (laws.add_assoc _ _ _).symm
        _ = ops.add (ops.add suffixLeft (ops.neg right))
            (ops.neg suffixRight) := by
          rw [laws.add_comm (ops.neg right) suffixLeft]
        _ = ops.add suffixLeft
            (ops.add (ops.neg right) (ops.neg suffixRight)) :=
          laws.add_assoc _ _ _
    _ = ops.add (ops.add left suffixLeft)
        (ops.add (ops.neg right) (ops.neg suffixRight)) :=
      (laws.add_assoc _ _ _).symm

private theorem sub_mul
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (left right multiplier : Field) :
    ops.mul (ops.sub left right) multiplier =
      ops.sub (ops.mul left multiplier) (ops.mul right multiplier) := by
  unfold InterpolationOps.sub
  rw [laws.right_distrib, laws.neg_mul]

private theorem evaluateTerms_zipWith_sub
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (point : CubePoint Field shape.cubeVariables)
    (highCoefficients lowCoefficients : List Field)
    (monomials : List (AlphaMonomial shape))
    (highLength : highCoefficients.length = monomials.length)
    (lowLength : lowCoefficients.length = monomials.length) :
    evaluateTerms ops point
        (List.zipWith ops.sub highCoefficients lowCoefficients) monomials =
      ops.sub
        (evaluateTerms ops point highCoefficients monomials)
        (evaluateTerms ops point lowCoefficients monomials) := by
  induction monomials generalizing highCoefficients lowCoefficients with
  | nil =>
      have highEmpty := List.eq_nil_of_length_eq_zero highLength
      have lowEmpty := List.eq_nil_of_length_eq_zero lowLength
      subst highCoefficients
      subst lowCoefficients
      simp [evaluateTerms, InterpolationOps.sub, neg_zero ops laws,
        laws.add_zero]
  | cons monomial monomials inductionHypothesis =>
      cases highCoefficients with
      | nil => simp at highLength
      | cons high highCoefficients =>
          cases lowCoefficients with
          | nil => simp at lowLength
          | cons low lowCoefficients =>
              simp only [List.length_cons, Nat.succ.injEq] at highLength lowLength
              simp only [List.zipWith, evaluateTerms]
              rw [inductionHypothesis highCoefficients lowCoefficients
                highLength lowLength]
              rw [sub_mul ops laws]
              exact add_sub_add_sub ops laws _ _ _ _

private theorem evaluation_probe
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (template : Shape)
    {variables : Nat}
    (table : BooleanTable Field variables)
    (point : CubePoint Field variables) :
    (table.toAlphaPolynomial (shape := evaluationShape template variables) ops).evaluate
      ops.toOps point = table.evaluate ops point := by
  induction table with
  | leaf value =>
      rw [alphaPolynomial_evaluate_eq_evaluateTerms]
      rcases point with ⟨coordinates, dimension⟩
      have coordinatesEmpty : coordinates = [] :=
        List.eq_nil_of_length_eq_zero dimension
      subst coordinates
      simp only [toAlphaPolynomial, evaluateTerms,
        canonicalAlphaBasis, canonicalAlphaMonomials,
        canonicalExponentVectors, interpolateCoefficients,
        AlphaMonomial.evaluate, evaluate, evaluateCoordinates,
        evaluationShape, List.pmap_cons, List.pmap_nil]
      change ops.add (ops.mul value ops.one) ops.zero = value
      rw [laws.mul_one, laws.add_zero]
  | @branch tailVariables low high lowInduction highInduction =>
      rw [alphaPolynomial_evaluate_eq_evaluateTerms]
      rcases point with ⟨coordinates, dimension⟩
      cases coordinates with
      | nil => simp at dimension
      | cons coordinate coordinates =>
          have tailDimension : coordinates.length = tailVariables :=
            Nat.succ.inj dimension
          let tailPoint : CubePoint Field tailVariables :=
            ⟨coordinates, tailDimension⟩
          simp only [toAlphaPolynomial, canonicalAlphaBasis,
            evaluate, evaluateCoordinates]
          rw [interpolateCoefficients.eq_def,
            canonicalAlphaMonomials_succ template]
          change evaluateTerms ops (prependPoint coordinate tailPoint)
              (interpolateCoefficients ops low ++
                List.zipWith ops.sub
                  (interpolateCoefficients ops high)
                  (interpolateCoefficients ops low))
              ((canonicalAlphaMonomials
                  (evaluationShape template tailVariables)).map
                  (prependMonomial 0) ++
                (canonicalAlphaMonomials
                  (evaluationShape template tailVariables)).map
                  (prependMonomial 1)) = _
          have lowLength :
              (interpolateCoefficients ops low).length =
                (canonicalAlphaMonomials
                  (evaluationShape template tailVariables)).length := by
            rw [interpolateCoefficients_length,
              canonicalAlphaMonomials_length]
            rfl
          have highLength :
              (interpolateCoefficients ops high).length =
                (canonicalAlphaMonomials
                  (evaluationShape template tailVariables)).length := by
            rw [interpolateCoefficients_length,
              canonicalAlphaMonomials_length]
            rfl
          have firstBlockLength :
              (interpolateCoefficients ops low).length =
                ((canonicalAlphaMonomials
                  (evaluationShape template tailVariables)).map
                  (prependMonomial 0)).length := by
            simpa using lowLength
          rw [evaluateTerms_append ops laws _ _ _ _ _ firstBlockLength]
          rw [evaluateTerms_prepend_zero ops laws]
          rw [evaluateTerms_prepend_one ops laws]
          rw [evaluateTerms_zipWith_sub ops laws _ _ _ _ highLength lowLength]
          have lowEvaluation :
              evaluateTerms ops tailPoint
                  (interpolateCoefficients ops low)
                  (canonicalAlphaMonomials
                    (evaluationShape template tailVariables)) =
                evaluateCoordinates ops low coordinates := by
            have result := lowInduction tailPoint
            rw [alphaPolynomial_evaluate_eq_evaluateTerms] at result
            simpa only [toAlphaPolynomial, canonicalAlphaBasis,
              evaluate] using result
          have highEvaluation :
              evaluateTerms ops tailPoint
                  (interpolateCoefficients ops high)
                  (canonicalAlphaMonomials
                    (evaluationShape template tailVariables)) =
                evaluateCoordinates ops high coordinates := by
            have result := highInduction tailPoint
            rw [alphaPolynomial_evaluate_eq_evaluateTerms] at result
            simpa only [toAlphaPolynomial, canonicalAlphaBasis,
              evaluate] using result
          rw [lowEvaluation, highEvaluation]

/-- Evaluating the canonical polynomial derived from a Boolean table equals
the independently recursive multilinear extension at every dimension-checked
point. This is a model-level algebraic theorem; it makes no implementation or
constraint claim. -/
theorem toAlphaPolynomial_evaluate_eq_evaluate
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {shape : Shape}
    (table : BooleanTable Field shape.cubeVariables)
    (point : CubePoint Field shape.cubeVariables) :
    (table.toAlphaPolynomial ops).evaluate ops.toOps point =
      table.evaluate ops point := by
  simpa only [evaluationShape] using
    (evaluation_probe ops laws shape table point)

end BooleanTable

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
