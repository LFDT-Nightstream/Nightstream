import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Polynomial.Rows

/-!
Contract: model-level algebra for the selective compiler's grouped-product
rewrite.

Assurance tier: model-level.

Owns: the exact five-product evaluation-row equation, soundness of a finite
chain of grouped-product accumulator rows, and completeness of the canonical
prefix assignment for such a chain.

Does not own: decoding a Rust artifact, equality with a Rust-emitted row,
coverage of every production rewrite, source-relation semantics, or permission
to remove a production row or coordinate.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

namespace Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.GroupedProduct

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial

/-- One unscaled product consumed by a grouped evaluation row. A Rust factor
coefficient is part of either factor before it reaches this algebraic layer. -/
structure Product where
  left : F
  right : F
deriving DecidableEq, Repr

namespace Product

def value (product : Product) : F :=
  product.left * product.right

end Product

/-- Ordered sum of the products in one compiler group. -/
def productSum : List Product -> F
  | [] => 0
  | product :: tail => product.value + productSum tail

/-- Inputs of one grouped-product accumulator step, without its output. -/
structure StepSpec where
  base : F
  products : List Product
deriving DecidableEq, Repr

namespace StepSpec

def contribution (step : StepSpec) : F :=
  step.base + productSum step.products

end StepSpec

/-- One emitted accumulator step. -/
structure Step extends StepSpec where
  output : F
deriving DecidableEq, Repr

namespace Step

def holds (previous : F) (step : Step) : Prop :=
  step.output = previous + step.toStepSpec.contribution

end Step

/-- Every accumulator output is the authority for the next row. -/
def ChainHolds : F -> List Step -> Prop
  | _, [] => True
  | previous, step :: tail =>
      step.holds previous /\ ChainHolds step.output tail

/-- Output carried by the final row, or the input when the chain is empty. -/
def finalValue : F -> List Step -> F
  | previous, [] => previous
  | _, step :: tail => finalValue step.output tail

/-- Sum of all base terms and products consumed by the chain. -/
def totalContribution : List Step -> F
  | [] => 0
  | step :: tail =>
      step.toStepSpec.contribution + totalContribution tail

/-- A satisfying finite chain cannot lose or add a product contribution. -/
theorem chainHolds_sound
    (initial : F) (steps : List Step)
    (holds : ChainHolds initial steps) :
    finalValue initial steps = initial + totalContribution steps := by
  induction steps generalizing initial with
  | nil =>
      simp [finalValue, totalContribution]
  | cons step tail inductionHypothesis =>
      have headHolds : step.holds initial := holds.1
      have tailHolds : ChainHolds step.output tail := holds.2
      calc
        finalValue initial (step :: tail) =
            step.output + totalContribution tail :=
          inductionHypothesis step.output tailHolds
        _ = (initial + step.toStepSpec.contribution) +
              totalContribution tail := by
          rw [headHolds]
        _ = initial +
              (step.toStepSpec.contribution + totalContribution tail) :=
          Lean.Grind.Fin.add_assoc _ _ _
        _ = initial + totalContribution (step :: tail) := rfl

/-- Canonical output for one grouped-product accumulator step. -/
def compileStep (previous : F) (spec : StepSpec) : Step where
  toStepSpec := spec
  output := previous + spec.contribution

/-- Canonical prefix assignment for a complete grouped-product chain. -/
def compile : F -> List StepSpec -> List Step
  | _, [] => []
  | previous, spec :: tail =>
      let step := compileStep previous spec
      step :: compile step.output tail

/-- The canonical prefix assignment satisfies every grouped-product row. -/
theorem compile_chainHolds (initial : F) (specs : List StepSpec) :
    ChainHolds initial (compile initial specs) := by
  induction specs generalizing initial with
  | nil =>
      simp [compile, ChainHolds]
  | cons spec tail inductionHypothesis =>
      change
        (compileStep initial spec).holds initial /\
          ChainHolds (compileStep initial spec).output
            (compile (compileStep initial spec).output tail)
      exact ⟨rfl, inductionHypothesis (compileStep initial spec).output⟩

/-- Five product slots in the exact order used by the selective polynomial. -/
def fiveProductSum
    (left0 right0 left1 right1 left2 right2 left3 right3 left4 right4 : F) : F :=
  left0 * right0 + left1 * right1 + left2 * right2 +
    left3 * right3 + left4 * right4

/-- The exact evaluation-row residual is the five-product sum minus the C-port
image. The C-port image is `output - base - previous` in a compiler rewrite. -/
theorem evaluationRow_residual
    (base previous : F)
    (left0 right0 left1 right1 left2 right2 left3 right3 left4 right4 : F)
    (output : F) :
    Polynomial.Semantics.evaluate
        (Polynomial.Rows.evaluationPoint 1
          left0 right0 left1 right1 left2 right2 left3 right3 left4 right4
          (output - base - previous)) =
      fiveProductSum left0 right0 left1 right1 left2 right2 left3 right3
          left4 right4 -
        (output - base - previous) := by
  rw [Polynomial.Rows.evaluate_evaluationPoint]
  simp [Polynomial.Components.evaluationResidual,
    Polynomial.Rows.evaluationPoint, Polynomial.Rows.sparsePoint,
    Polynomial.Ports.Role.index, fiveProductSum,
    Fin.one_mul, Fin.mul_one, Fin.sub_eq_add_neg,
    Lean.Grind.AddCommGroup.neg_add,
    Lean.Grind.AddCommGroup.neg_neg]
  letI : Std.Associative (fun (left right : F) => left + right) :=
    ⟨Lean.Grind.Fin.add_assoc⟩
  letI : Std.Commutative (fun (left right : F) => left + right) :=
    ⟨Lean.Grind.Fin.add_comm⟩
  ac_rfl

private theorem restoreDifference
    (output base previous : F) :
    previous + (base + (output - base - previous)) = output := by
  simp only [Fin.sub_eq_add_neg]
  have cancelBase : base + -base = 0 := by
    rw [Lean.Grind.Fin.add_comm, Lean.Grind.Fin.neg_add_cancel]
  have cancelPrevious : previous + -previous = 0 := by
    rw [Lean.Grind.Fin.add_comm, Lean.Grind.Fin.neg_add_cancel]
  letI : Std.Associative (fun (left right : F) => left + right) :=
    ⟨Lean.Grind.Fin.add_assoc⟩
  letI : Std.Commutative (fun (left right : F) => left + right) :=
    ⟨Lean.Grind.Fin.add_comm⟩
  calc
    previous + (base + ((output + -base) + -previous)) =
        output + (base + -base) + (previous + -previous) := by
      ac_rfl
    _ = output := by
      rw [cancelBase, cancelPrevious, Fin.add_zero, Fin.add_zero]

private theorem removePrefix
    (previous base value : F) :
    (previous + (base + value)) - base - previous = value := by
  simp only [Fin.sub_eq_add_neg]
  have cancelBase : base + -base = 0 := by
    rw [Lean.Grind.Fin.add_comm, Lean.Grind.Fin.neg_add_cancel]
  have cancelPrevious : previous + -previous = 0 := by
    rw [Lean.Grind.Fin.add_comm, Lean.Grind.Fin.neg_add_cancel]
  letI : Std.Associative (fun (left right : F) => left + right) :=
    ⟨Lean.Grind.Fin.add_assoc⟩
  letI : Std.Commutative (fun (left right : F) => left + right) :=
    ⟨Lean.Grind.Fin.add_comm⟩
  calc
    ((previous + (base + value)) + -base) + -previous =
        value + (base + -base) + (previous + -previous) := by
      ac_rfl
    _ = value := by
      rw [cancelBase, cancelPrevious, Fin.add_zero, Fin.add_zero]

/-- One selected evaluation row holds exactly when its accumulator output is
the previous value plus the row's base and five products. -/
theorem evaluationRow_zero_iff_stepHolds
    (base previous : F)
    (left0 right0 left1 right1 left2 right2 left3 right3 left4 right4 : F)
    (output : F) :
    Polynomial.Semantics.evaluate
        (Polynomial.Rows.evaluationPoint 1
          left0 right0 left1 right1 left2 right2 left3 right3 left4 right4
          (output - base - previous)) = 0 ↔
      output = previous +
        (base + fiveProductSum left0 right0 left1 right1 left2 right2 left3
          right3 left4 right4) := by
  rw [evaluationRow_residual]
  rw [Lean.Grind.AddCommGroup.sub_eq_zero_iff]
  constructor
  · intro same
    rw [same]
    exact (restoreDifference output base previous).symm
  · intro same
    rw [same]
    exact (removePrefix previous base _).symm

/-- Selector-one specialization used by a decoded final evaluation row. -/
theorem evaluationPoint_zero_iff_fiveProduct
    (left0 right0 left1 right1 left2 right2 left3 right3 left4 right4 : F)
    (output : F) :
    Polynomial.Semantics.evaluate
        (Polynomial.Rows.evaluationPoint 1
          left0 right0 left1 right1 left2 right2 left3 right3 left4 right4
          output) = 0 ↔
      output = fiveProductSum left0 right0 left1 right1 left2 right2 left3
        right3 left4 right4 := by
  have outputEq : output - (0 : F) - (0 : F) = output := by
    simp only [Fin.sub_eq_add_neg, Lean.Grind.AddCommGroup.neg_zero,
      Fin.add_zero]
  have residualEq :
      Polynomial.Semantics.evaluate
          (Polynomial.Rows.evaluationPoint 1
            left0 right0 left1 right1 left2 right2 left3 right3 left4 right4
            output) =
        fiveProductSum left0 right0 left1 right1 left2 right2 left3 right3
          left4 right4 - output := by
    calc
      Polynomial.Semantics.evaluate
          (Polynomial.Rows.evaluationPoint 1
            left0 right0 left1 right1 left2 right2 left3 right3 left4 right4
            output) =
          Polynomial.Semantics.evaluate
            (Polynomial.Rows.evaluationPoint 1
              left0 right0 left1 right1 left2 right2 left3 right3 left4 right4
              (output - (0 : F) - (0 : F))) := by
            rw [outputEq]
      _ = fiveProductSum left0 right0 left1 right1 left2 right2 left3 right3
            left4 right4 - (output - (0 : F) - (0 : F)) :=
          evaluationRow_residual 0 0 left0 right0 left1 right1 left2 right2
            left3 right3 left4 right4 output
      _ = fiveProductSum left0 right0 left1 right1 left2 right2 left3 right3
            left4 right4 - output := by
          rw [outputEq]
  rw [residualEq]
  rw [Lean.Grind.AddCommGroup.sub_eq_zero_iff]
  constructor <;> intro same <;> exact same.symm

end Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.GroupedProduct
