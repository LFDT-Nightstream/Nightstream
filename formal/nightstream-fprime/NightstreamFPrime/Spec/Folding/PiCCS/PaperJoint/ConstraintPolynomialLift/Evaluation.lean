import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConstraintPolynomialLift

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/ConstraintPolynomialLift/Evaluation.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Evaluation refinement for structurally lifted sparse CCS polynomials.

Protocol: shared CCS syntax infrastructure.
Phase: carrier placement followed by polynomial evaluation.
Constraint family: none; this file emits no rows.

Owns: the minimal zero/one/add/mul preservation contract and proofs that
monomial and polynomial evaluation commute with coefficient lifting.

Does not own: a concrete carrier embedding, zero reflection, norm semantics,
protocol source data, SumCheck, Rust, R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: the lifted syntax comes only from
`ConstraintPolynomialLift`; callers provide four explicit homomorphism laws,
not an evaluator or a claimed equality between evaluations.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.ccs.lift.algebra` | lift preserves zero, one, addition, multiplication | security boundary | `EvaluationLaws` |
| `pi_ccs.ccs.lift.monomial_eval` | lifted monomial evaluation equals lifted base evaluation | derived | `evaluateMonomial_lift` |
| `pi_ccs.ccs.lift.polynomial_eval` | lifted sparse polynomial evaluation equals lifted base evaluation | derived | `evaluatePolynomial_lift` |
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConstraintPolynomialLift.Evaluation

universe uBase uExtension

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open ConstraintPolynomialLift

/-- Exactly the algebraic preservation laws used by sparse CCS evaluation. -/
structure EvaluationLaws
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (extensionOps : InterpolationOps Extension)
    (lift : Base -> Extension) : Prop where
  map_zero : lift baseOps.zero = extensionOps.zero
  map_one : lift baseOps.one = extensionOps.one
  map_add : forall left right,
    lift (baseOps.add left right) =
      extensionOps.add (lift left) (lift right)
  map_mul : forall left right,
    lift (baseOps.mul left right) =
      extensionOps.mul (lift left) (lift right)

private theorem pow_lift
    {Base : Type uBase}
    {Extension : Type uExtension}
    (baseOps : InterpolationOps Base)
    (extensionOps : InterpolationOps Extension)
    (lift : Base -> Extension)
    (laws : EvaluationLaws baseOps extensionOps lift)
    (value : Base)
    (exponent : Nat) :
    CCSResidualTable.pow extensionOps (lift value) exponent =
      lift (CCSResidualTable.pow baseOps value exponent) := by
  induction exponent with
  | zero => exact laws.map_one.symm
  | succ exponent inductionHypothesis =>
      simp only [CCSResidualTable.pow]
      rw [inductionHypothesis]
      exact (laws.map_mul _ _).symm

private theorem monomialFold_lift
    {Base : Type uBase}
    {Extension : Type uExtension}
    {matrixCount : Nat}
    (baseOps : InterpolationOps Base)
    (extensionOps : InterpolationOps Extension)
    (lift : Base -> Extension)
    (laws : EvaluationLaws baseOps extensionOps lift)
    (monomial : CCSResidualTable.Monomial Base matrixCount)
    (point : Fin matrixCount -> Base)
    (indices : List (Fin matrixCount))
    (accumulated : Base) :
    indices.foldl
        (fun value index =>
          extensionOps.mul value
            (CCSResidualTable.pow extensionOps (lift (point index))
              (monomial.exponents index)))
        (lift accumulated) =
      lift (indices.foldl
        (fun value index =>
          baseOps.mul value
            (CCSResidualTable.pow baseOps (point index)
              (monomial.exponents index)))
        accumulated) := by
  induction indices generalizing accumulated with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      simp only [List.foldl_cons]
      rw [pow_lift baseOps extensionOps lift laws]
      rw [<- laws.map_mul]
      exact inductionHypothesis _

/-- Evaluation of one explicit sparse monomial commutes with lifting. -/
theorem evaluateMonomial_lift
    {Base : Type uBase}
    {Extension : Type uExtension}
    {matrixCount : Nat}
    (baseOps : InterpolationOps Base)
    (extensionOps : InterpolationOps Extension)
    (lift : Base -> Extension)
    (laws : EvaluationLaws baseOps extensionOps lift)
    (monomial : CCSResidualTable.Monomial Base matrixCount)
    (point : Fin matrixCount -> Base) :
    CCSResidualTable.evaluateMonomial extensionOps
        (liftMonomial lift monomial) (fun index => lift (point index)) =
      lift (CCSResidualTable.evaluateMonomial baseOps monomial point) := by
  unfold CCSResidualTable.evaluateMonomial liftMonomial
  exact monomialFold_lift baseOps extensionOps lift laws monomial point
    (canonicalFinIndices matrixCount) monomial.coefficient

private theorem polynomialFold_lift
    {Base : Type uBase}
    {Extension : Type uExtension}
    {matrixCount : Nat}
    (baseOps : InterpolationOps Base)
    (extensionOps : InterpolationOps Extension)
    (lift : Base -> Extension)
    (laws : EvaluationLaws baseOps extensionOps lift)
    (point : Fin matrixCount -> Base)
    (terms : List (CCSResidualTable.Monomial Base matrixCount))
    (accumulated : Base) :
    (terms.map (liftMonomial lift)).foldl
        (fun value monomial =>
          extensionOps.add value
            (CCSResidualTable.evaluateMonomial extensionOps monomial
              (fun index => lift (point index))))
        (lift accumulated) =
      lift (terms.foldl
        (fun value monomial =>
          baseOps.add value
            (CCSResidualTable.evaluateMonomial baseOps monomial point))
        accumulated) := by
  induction terms generalizing accumulated with
  | nil => rfl
  | cons monomial terms inductionHypothesis =>
      simp only [List.map_cons, List.foldl_cons]
      rw [evaluateMonomial_lift baseOps extensionOps lift laws]
      rw [<- laws.map_add]
      exact inductionHypothesis _

/-- Evaluation of a complete explicit sparse CCS polynomial commutes with
lifting. This follows from syntax and the four algebraic laws; no protocol
truth predicate is an input. -/
theorem evaluatePolynomial_lift
    {Base : Type uBase}
    {Extension : Type uExtension}
    {matrixCount : Nat}
    (baseOps : InterpolationOps Base)
    (extensionOps : InterpolationOps Extension)
    (lift : Base -> Extension)
    (laws : EvaluationLaws baseOps extensionOps lift)
    (polynomial : CCSResidualTable.ConstraintPolynomial Base matrixCount)
    (point : Fin matrixCount -> Base) :
    CCSResidualTable.evaluatePolynomial extensionOps
        (liftConstraintPolynomial lift polynomial)
        (fun index => lift (point index)) =
      lift (CCSResidualTable.evaluatePolynomial baseOps polynomial point) := by
  unfold CCSResidualTable.evaluatePolynomial liftConstraintPolynomial
  rw [<- laws.map_zero]
  exact polynomialFold_lift baseOps extensionOps lift laws point
    polynomial.terms baseOps.zero

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConstraintPolynomialLift.Evaluation
