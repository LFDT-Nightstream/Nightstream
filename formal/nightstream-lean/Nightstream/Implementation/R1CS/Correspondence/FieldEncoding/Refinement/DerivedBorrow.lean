import Nightstream.Implementation.R1CS.Core.Program
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.DerivedNegative

/-!
Contract: explicit polynomial model for the borrow equations obtained after
eliminating shifted-ternary negative-indicator witness columns.

Owns: a small polynomial AST, total-degree accounting, the exact substitution
`n = d(d - 1) / 2`, translation of every sparse borrow row, semantic equality
with `DerivedBorrowHolds`, and the finite 41-equation degree census.

Does not own: a CCS custom-gate encoding, strict-R1CS product-wire lowering,
generated Rust emission, or equality with a production emitted artifact.

Emits constraints: no. The equations are a model-level candidate schedule.

Authority boundary: negative indicators are reconstructed from digit columns.
The degree theorem assumes only this explicit substitution; digit membership in
the centered alphabet still comes from the separately checked `b = 2` norm.

| Branch | Mathematical obligation | Result | Tier |
|---|---|---|---|
| polynomial syntax | constants, source columns, sums, products | `Polynomial` | model-level |
| negative substitution | `n = d(d - 1) / 2` | `eval_derivedNegativePolynomial` | model-level |
| borrow translation | translated equation iff old row after reconstruction | `derivedBorrowEquation_holds_iff` | model-level |
| degree census | all 41 translated equations have total degree at most three | `derivedBorrowEquation_degree_le_three` | model-level |
-/

namespace Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.CenteredTernaryDerivedNegative

set_option maxRecDepth 262144

/-- Minimal polynomial syntax needed to make the substituted borrow degree a
kernel-checked statement instead of a prose estimate. -/
inductive Polynomial where
  | constant (value : Nat)
  | variable (column : Nat)
  | add (left right : Polynomial)
  | mul (left right : Polynomial)
deriving DecidableEq, Repr

namespace Polynomial

/-- Evaluation in canonical Goldilocks residues. Reducing at every AST node is
semantically equivalent to reducing only at the root. -/
def eval (assignment : Nat → Nat) : Polynomial → Nat
  | .constant value => value % goldilocksP
  | .variable column => assignment column % goldilocksP
  | .add left right =>
      (eval assignment left + eval assignment right) % goldilocksP
  | .mul left right =>
      (eval assignment left * eval assignment right) % goldilocksP

/-- Syntactic total degree. Constants, including zero, have degree zero. -/
def degree : Polynomial → Nat
  | .constant _ => 0
  | .variable _ => 1
  | .add left right => max left.degree right.degree
  | .mul left right => left.degree + right.degree

end Polynomial

/-- A polynomial equality. Keeping two sides avoids hiding field subtraction
inside a special AST node and mirrors one R1CS row exactly. -/
structure Equation where
  left : Polynomial
  right : Polynomial
deriving DecidableEq, Repr

namespace Equation

def degree (equation : Equation) : Nat :=
  max equation.left.degree equation.right.degree

def Holds (assignment : Nat → Nat) (equation : Equation) : Prop :=
  equation.left.eval assignment = equation.right.eval assignment

instance (assignment : Nat → Nat) (equation : Equation) :
    Decidable (equation.Holds assignment) := by
  unfold Holds
  infer_instance

end Equation

/-- Explicit quadratic polynomial for the derived negative indicator at one
digit column. `p - 1` is field subtraction by one. -/
def derivedNegativePolynomial (digitColumn : Nat) : Polynomial :=
  .mul
    (.mul (.variable digitColumn)
      (.add (.variable digitColumn) (.constant (goldilocksP - 1))))
    (.constant inverseTwo)

theorem derivedNegativePolynomial_degree (digitColumn : Nat) :
    (derivedNegativePolynomial digitColumn).degree = 2 := by
  rfl

theorem eval_derivedNegativePolynomial
    (assignment : Nat → Nat) (digitColumn : Nat) :
    (derivedNegativePolynomial digitColumn).eval assignment =
      derivedNegative (assignment digitColumn) := by
  simp only [derivedNegativePolynomial, Polynomial.eval, derivedNegative,
    fieldPred]
  simp [Nat.add_mod, Nat.mul_mod]

/-- Source-column expression after eliminating exactly the old negative range
`[99, 140)`. Every other source column remains an ordinary variable. -/
def sourcePolynomial (column : Nat) : Polynomial :=
  if 99 ≤ column ∧ column < 140 then
    derivedNegativePolynomial (column - 41)
  else
    .variable column

theorem eval_sourcePolynomial (assignment : Nat → Nat) (column : Nat) :
    (sourcePolynomial column).eval assignment =
      materializeNegatives assignment column % goldilocksP := by
  by_cases interval : 99 ≤ column ∧ column < 140
  · simp [sourcePolynomial, materializeNegatives, interval,
      eval_derivedNegativePolynomial,
      Nat.mod_eq_of_lt (derivedNegative_lt _)]
  · simp [sourcePolynomial, materializeNegatives, interval,
      Polynomial.eval]

/-- Sparse linear combination translated through `sourcePolynomial`. -/
def linearCombination : List (Nat × Nat) → Polynomial
  | [] => .constant 0
  | term :: tail =>
      .add (.mul (.constant term.2) (sourcePolynomial term.1))
        (linearCombination tail)

theorem eval_linearCombination_raw
    (assignment : Nat → Nat) (terms : List (Nat × Nat)) :
    (linearCombination terms).eval assignment =
      Program.rawLcEval (materializeNegatives assignment) terms %
        goldilocksP := by
  induction terms with
  | nil => simp [linearCombination, Polynomial.eval, Program.rawLcEval]
  | cons term tail inductionHypothesis =>
      simp only [linearCombination, Polynomial.eval, Program.rawLcEval]
      rw [eval_sourcePolynomial, inductionHypothesis]
      simp [Nat.add_mod, Nat.mul_mod]

theorem eval_linearCombination
    (assignment : Nat → Nat) (terms : List (Nat × Nat)) :
    (linearCombination terms).eval assignment =
      lcEval (materializeNegatives assignment) terms := by
  rw [Program.lcEval_eq_raw_mod]
  exact eval_linearCombination_raw assignment terms

/-- Direct polynomial translation of one R1CS row after negative-column
substitution. -/
def rowEquation (row : Row) : Equation where
  left := .mul (linearCombination row.a) (linearCombination row.b)
  right := linearCombination row.c

theorem rowEquation_holds_iff
    (assignment : Nat → Nat) (row : Row) :
    (rowEquation row).Holds assignment ↔
      RowHolds (materializeNegatives assignment) row := by
  simp only [Equation.Holds, rowEquation, Polynomial.eval,
    eval_linearCombination, RowHolds]

/-- Explicit substituted polynomial equation for one canonical borrow step. -/
def derivedBorrowEquation (index : Nat) : Equation :=
  rowEquation (borrowRow index)

theorem derivedBorrowEquation_holds_iff
    (assignment : Nat → Nat) (index : Nat) :
    (derivedBorrowEquation index).Holds assignment ↔
      DerivedBorrowHolds assignment index := by
  exact rowEquation_holds_iff assignment (borrowRow index)

/-- Exact finite schedule corresponding to `DerivedAccepts`. -/
def derivedBorrowEquations : List Equation :=
  (List.range digitCount).map derivedBorrowEquation

theorem derivedBorrowEquations_length :
    derivedBorrowEquations.length = 41 := by
  native_decide

/-- Kernel evaluation of the concrete 41-row schedule: every substituted
borrow identity has syntactic total degree at most three. -/
theorem derivedBorrowEquations_degree_le_three :
    ∀ equation ∈ derivedBorrowEquations, equation.degree ≤ 3 := by
  native_decide

theorem derivedBorrowEquation_degree_le_three
    {index : Nat} (indexLt : index < digitCount) :
    (derivedBorrowEquation index).degree ≤ 3 := by
  exact derivedBorrowEquations_degree_le_three _
    (List.mem_map.mpr ⟨index, List.mem_range.mpr indexLt, rfl⟩)

/-- The bound is attained by the concrete schedule; it is not merely a loose
upper bound inherited from the AST constructors. -/
def maximumDerivedBorrowDegree : Nat :=
  derivedBorrowEquations.foldl (fun maximum equation =>
    max maximum equation.degree) 0

theorem maximumDerivedBorrowDegree_eq_three :
    maximumDerivedBorrowDegree = 3 := by
  native_decide

theorem derivedAccepts_iff_polynomial_schedule
    (assignment : Nat → Nat) :
    DerivedAccepts assignment ↔
      ∀ equation ∈ derivedBorrowEquations, equation.Holds assignment := by
  constructor
  · intro accepts equation member
    rcases List.mem_map.mp member with ⟨index, indexMember, rfl⟩
    exact (derivedBorrowEquation_holds_iff assignment index).2
      (accepts index (List.mem_range.mp indexMember))
  · intro equations index indexLt
    apply (derivedBorrowEquation_holds_iff assignment index).1
    exact equations (derivedBorrowEquation index)
      (List.mem_map.mpr ⟨index, List.mem_range.mpr indexLt, rfl⟩)

end Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow
