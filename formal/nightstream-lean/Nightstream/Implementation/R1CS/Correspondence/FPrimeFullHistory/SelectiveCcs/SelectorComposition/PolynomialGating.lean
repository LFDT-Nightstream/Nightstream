import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial.Components
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.Semantics

/-!
Contract: generic selector factorization of the exact 27-term selective CCS
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
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics

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

/-- Every general-gated monomial contains `G` exactly once and no `E`.
Consequently an arbitrary general row is its selector weight times the same
row with `G=1`. -/
theorem evaluate_general_gated
    (point : Fin 13 → F) (weight : F)
    (general : point Role.generalSelector.index = weight)
    (evaluation : point Role.evalSelector.index = 0) :
    evaluate point = weight * evaluate (ungate .general point) := by
  rw [evaluate_eq_combinedResidual, evaluate_eq_combinedResidual]
  simp only [combinedResidual, booleanResidual, productResidual,
    sboxResidual, centeredResidual, evaluationResidual, canonicalResidual,
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
  rw [evaluate_eq_combinedResidual, evaluate_eq_combinedResidual]
  simp only [combinedResidual, booleanResidual, productResidual,
    sboxResidual, centeredResidual, evaluationResidual, canonicalResidual,
    ungate_evaluation_generalSelector, ungate_evaluation_evalSelector,
    general, evaluation, Fin.zero_mul, Fin.mul_zero, Fin.zero_add,
    Fin.add_zero, Fin.one_mul, fmul_add, fmul_neg]
  ac_rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.PolynomialGating
