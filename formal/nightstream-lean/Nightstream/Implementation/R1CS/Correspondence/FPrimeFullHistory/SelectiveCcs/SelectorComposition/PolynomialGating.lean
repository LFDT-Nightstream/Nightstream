import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial.Components
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.Semantics

/-!
Contract: generic selector factorization of the exact 66-term selective CCS
polynomial.

Owns: the two selector-port classes, selector normalization of an arbitrary
13-port row image, and one homogeneity theorem for each class. The theorems
cover every arm-local physical row without naming its arithmetic family.

Does not own: final matrix support, emitted-run ownership, selector columns,
branch residual semantics, constant-one connectivity, or row removal.

Emits constraints: no.

| Gate port | Required row image | Exact result | Rust families covered |
|---|---|---|---|
| general | `G = weight`, `E = 0` | `evaluate point = weight * evaluate (ungate general point)` | arm-domain, retained, Poseidon2, centered, canonical |
| evaluation | `G = 0`, `E = weight` | `evaluate point = weight * evaluate (ungate evaluation point)` | polynomial evaluation, product sum |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.PolynomialGating

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics

set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

/-- Which of the two linearly disjoint selector ports owns a row. -/
inductive GatePort where
  | general
  | evaluation
deriving DecidableEq, Repr

/-- Replace the active selector image with one and the inactive selector image
with zero, preserving every arithmetic port. -/
def ungate (gate : GatePort) (point : Fin 13 → F) : Fin 13 → F :=
  fun port =>
    match gate with
    | .general =>
        if port = Role.generalSelector.index then 1
        else if port = Role.evalSelector.index then 0
        else point port
    | .evaluation =>
        if port = Role.generalSelector.index then 0
        else if port = Role.evalSelector.index then 1
        else point port

@[simp] theorem ungate_general_generalSelector (point : Fin 13 → F) :
    ungate .general point Role.generalSelector.index = 1 := by
  simp [ungate]

@[simp] theorem ungate_general_evalSelector (point : Fin 13 → F) :
    ungate .general point Role.evalSelector.index = 0 := by
  simp [ungate]

@[simp] theorem ungate_evaluation_generalSelector (point : Fin 13 → F) :
    ungate .evaluation point Role.generalSelector.index = 0 := by
  simp [ungate]

@[simp] theorem ungate_evaluation_evalSelector (point : Fin 13 → F) :
    ungate .evaluation point Role.evalSelector.index = 1 := by
  simp [ungate]

private theorem fmul_add (left middle right : F) :
    left * (middle + right) = left * middle + left * right :=
  Lean.Grind.Fin.left_distrib _ _ _

private theorem fmul_neg (left right : F) :
    left * -right = -(left * right) := by
  calc
    left * -right = (-right) * left := Fin.mul_comm _ _
    _ = -(right * left) := Lean.Grind.Fin.neg_mul _ _
    _ = -(left * right) := congrArg Neg.neg (Fin.mul_comm _ _)

private theorem fmul_assoc (left middle right : F) :
    (left * middle) * right = left * (middle * right) :=
  Fin.mul_assoc _ _ _

private theorem fmul_comm (left right : F) : left * right = right * left :=
  Fin.mul_comm _ _

local instance : Std.Associative (fun (left right : F) => left * right) :=
  ⟨fmul_assoc⟩

local instance : Std.Commutative (fun (left right : F) => left * right) :=
  ⟨fmul_comm⟩

private def CanonicallyGated (term : Monomial F 13) : Prop :=
  term.exponents Role.generalSelector.index = 1 ∧
    term.exponents Role.evalSelector.index = 0

private instance (term : Monomial F 13) :
    Decidable (CanonicallyGated term) := by
  unfold CanonicallyGated
  infer_instance

private theorem every_canonical_term_gated :
    canonicalTerms.all (fun term => decide (CanonicallyGated term)) = true := by
  decide

private theorem canonical_term_gated
    (term : Monomial F 13) (member : term ∈ canonicalTerms) :
    CanonicallyGated term :=
  of_decide_eq_true
    ((List.all_eq_true.mp every_canonical_term_gated) term member)

private theorem evaluateMonomial_general_gated
    (term : Monomial F 13) (shape : CanonicallyGated term)
    (point : Fin 13 → F) (weight : F)
    (general : point Role.generalSelector.index = weight) :
    evaluateMonomial baseOps term point =
      weight * evaluateMonomial baseOps term (ungate .general point) := by
  rcases shape with ⟨generalExponent, evaluationExponent⟩
  change term.exponents 1 = 1 at generalExponent
  change term.exponents 7 = 0 at evaluationExponent
  change point 1 = weight at general
  simp [evaluateMonomial,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.canonicalFinIndices,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.pow,
    baseOps, ungate, Role.index, generalExponent, evaluationExponent,
    general, Fin.one_mul, Fin.mul_one]
  ac_rfl

private theorem evaluateTerms_general_gated
    (items : List (Monomial F 13))
    (shape : ∀ term ∈ items, CanonicallyGated term)
    (point : Fin 13 → F) (weight : F)
    (general : point Role.generalSelector.index = weight) :
    evaluateTerms point items =
      weight * evaluateTerms (ungate .general point) items := by
  induction items with
  | nil =>
      simp [evaluateTerms, Fin.mul_zero]
  | cons head tail inductionHypothesis =>
      have headShape : CanonicallyGated head :=
        shape head (by simp)
      have tailShape :
          ∀ term ∈ tail, CanonicallyGated term := by
        intro term member
        exact shape term (by simp [member])
      calc
        evaluateTerms point (head :: tail) =
            evaluateMonomial baseOps head point +
              evaluateTerms point tail := rfl
        _ = weight *
              evaluateMonomial baseOps head (ungate .general point) +
            weight *
              evaluateTerms (ungate .general point) tail := by
                rw [evaluateMonomial_general_gated
                  head headShape point weight general,
                  inductionHypothesis tailShape]
        _ = weight *
              (evaluateMonomial baseOps head (ungate .general point) +
                evaluateTerms (ungate .general point) tail) :=
                  (fmul_add _ _ _).symm
        _ = weight *
              evaluateTerms (ungate .general point) (head :: tail) := rfl

private theorem canonicalResidual_general_gated
    (point : Fin 13 → F) (weight : F)
    (general : point Role.generalSelector.index = weight) :
    canonicalResidual point =
      weight * canonicalResidual (ungate .general point) := by
  unfold canonicalResidual
  exact evaluateTerms_general_gated canonicalTerms
    canonical_term_gated point weight general

private theorem canonicalResidual_zero_of_general_zero
    (point : Fin 13 → F)
    (general : point Role.generalSelector.index = 0) :
    canonicalResidual point = 0 := by
  rw [canonicalResidual_general_gated point 0 general]
  exact Lean.Grind.Fin.zero_mul _

/-- Every general-gated monomial contains `G` exactly once and no `E`.
Consequently an arbitrary general row is its selector weight times the same
row with `G=1`. -/
theorem evaluate_general_gated
    (point : Fin 13 → F) (weight : F)
    (general : point Role.generalSelector.index = weight)
    (evaluation : point Role.evalSelector.index = 0) :
    evaluate point = weight * evaluate (ungate .general point) := by
  have canonical :=
    canonicalResidual_general_gated point weight general
  rw [evaluate_eq_combinedResidual, evaluate_eq_combinedResidual]
  simp only [combinedResidual]
  rw [canonical]
  generalize canonicalResidual (ungate .general point) = canonicalValue
  simp only [booleanResidual, productResidual,
    sboxResidual, centeredResidual, evaluationResidual,
    ungate_general_generalSelector, ungate_general_evalSelector,
    general, evaluation, Fin.zero_mul, Fin.mul_zero, Fin.add_zero,
    Fin.one_mul, fmul_add, fmul_neg]
  ac_rfl

/-- Every evaluation-gated monomial contains `E` exactly once and no `G`.
Consequently an arbitrary evaluation row is its selector weight times the same
row with `E=1`. -/
theorem evaluate_evaluation_gated
    (point : Fin 13 → F) (weight : F)
    (general : point Role.generalSelector.index = 0)
    (evaluation : point Role.evalSelector.index = weight) :
    evaluate point = weight * evaluate (ungate .evaluation point) := by
  have canonicalPoint :=
    canonicalResidual_zero_of_general_zero point general
  have canonicalUngated :=
    canonicalResidual_zero_of_general_zero (ungate .evaluation point)
      (ungate_evaluation_generalSelector point)
  rw [evaluate_eq_combinedResidual, evaluate_eq_combinedResidual]
  simp only [combinedResidual]
  rw [canonicalPoint, canonicalUngated]
  simp only [booleanResidual, productResidual,
    sboxResidual, centeredResidual, evaluationResidual,
    ungate_evaluation_generalSelector, ungate_evaluation_evalSelector,
    general, evaluation, Fin.zero_mul, Fin.mul_zero, Fin.zero_add,
    Fin.add_zero, Fin.one_mul, fmul_add, fmul_neg]
  ac_rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.PolynomialGating
