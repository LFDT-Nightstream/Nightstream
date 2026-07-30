import Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport
import Nightstream.Implementation.R1CS.Canonical.KSparsePolynomialHonest

/-!
Contract: constructive completeness for a consecutive batch of evaluations of
one typed sparse polynomial at independently carried points.

Owns the left-to-right witness composition and exact prefix preservation.
Protocol modules remain responsible for proving that their job enumeration is
positional and for identifying the selected polynomial.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KSparsePolynomialSequentialHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev ConcreteK := Nightstream.SuperNeo.Concrete.K

def blockWidth
    {matrixCount : Nat}
    (polynomial :
      CCSResidualTable.ConstraintPolynomial ConcreteK matrixCount) : Nat :=
  3 * KSparsePolynomial.totalDegreeSum polynomial.terms

def inputAt
    {Job : Type} {matrixCount : Nat}
    (polynomial :
      CCSResidualTable.ConstraintPolynomial ConcreteK matrixCount)
    (point : Job → Fin matrixCount → Carried)
    (base offset : Nat) (job : Job) :
    KSparsePolynomial.Input matrixCount where
  polynomial := polynomial
  point := point job
  frameBase := base + offset * blockWidth polynomial

private theorem inputAt_end
    {Job : Type} {matrixCount : Nat}
    (polynomial :
      CCSResidualTable.ConstraintPolynomial ConcreteK matrixCount)
    (point : Job → Fin matrixCount → Carried)
    (base offset : Nat) (job : Job) :
    (inputAt polynomial point base offset job).frameBase +
        3 * KSparsePolynomial.totalDegreeSum
          (inputAt polynomial point base offset job).polynomial.terms =
      base + (offset + 1) * blockWidth polynomial := by
  unfold inputAt blockWidth
  rw [Nat.add_mul, Nat.one_mul, Nat.add_assoc]

def rowsFrom
    {Job : Type} {matrixCount : Nat}
    (polynomial :
      CCSResidualTable.ConstraintPolynomial ConcreteK matrixCount)
    (point : Job → Fin matrixCount → Carried)
    (base : Nat) : List Job → Nat → List Row
  | [], _ => []
  | job :: rest, offset =>
      KSparsePolynomial.rows (inputAt polynomial point base offset job) ++
        rowsFrom polynomial point base rest (offset + 1)

def witnessFrom
    {Job : Type} {matrixCount : Nat}
    (assignment : Nat → Nat)
    (polynomial :
      CCSResidualTable.ConstraintPolynomial ConcreteK matrixCount)
    (point : Job → Fin matrixCount → Carried)
    (base : Nat) : List Job → Nat → Nat → Nat
  | [], _ => assignment
  | job :: rest, offset =>
      witnessFrom
        (KSparsePolynomialHonest.witness
          (inputAt polynomial point base offset job) assignment)
        polynomial point base rest (offset + 1)

theorem witnessFrom_off_before
    {Job : Type} {matrixCount : Nat}
    (assignment : Nat → Nat)
    (polynomial :
      CCSResidualTable.ConstraintPolynomial ConcreteK matrixCount)
    (point : Job → Fin matrixCount → Carried) :
    ∀ (jobs : List Job) (base offset column : Nat),
      column < base + offset * blockWidth polynomial →
      witnessFrom assignment polynomial point base jobs offset column =
        assignment column
  | [], _, _, _, _ => rfl
  | job :: rest, base, offset, column, below => by
      rw [witnessFrom,
        witnessFrom_off_before
          (KSparsePolynomialHonest.witness
            (inputAt polynomial point base offset job) assignment)
          polynomial point rest base (offset + 1) column (by
            exact Nat.lt_of_lt_of_le below
              (Nat.add_le_add_left
                (Nat.mul_le_mul_right (blockWidth polynomial)
                  (Nat.le_succ offset))
                base)),
        KSparsePolynomialHonest.witness_off_block
          (inputAt polynomial point base offset job) assignment column
          (by simpa [inputAt] using below)]

private theorem rowsBelow_append
    {left right : List Row} {boundary : Nat}
    (leftBelow : RowsBelow left boundary)
    (rightBelow : RowsBelow right boundary) :
    RowsBelow (left ++ right) boundary := by
  intro row member column mentioned
  exact (List.mem_append.1 member).elim
    (fun inLeft => leftBelow row inLeft column mentioned)
    (fun inRight => rightBelow row inRight column mentioned)

theorem rowsFrom_below_end
    {Job : Type} {matrixCount base : Nat}
    (polynomial :
      CCSResidualTable.ConstraintPolynomial ConcreteK matrixCount)
    (point : Job → Fin matrixCount → Carried)
    (basePositive : 0 < base)
    (pointsBelow :
      ∀ job index, CarriedBelow (point job index) base) :
    ∀ (jobs : List Job) (offset : Nat),
      RowsBelow (rowsFrom polynomial point base jobs offset)
        (base + (offset + jobs.length) * blockWidth polynomial)
  | [], offset => by
      intro row member
      simp [rowsFrom] at member
  | job :: rest, offset => by
      have currentPositive :
          0 < base + offset * blockWidth polynomial := by
        exact Nat.lt_of_lt_of_le basePositive
          (Nat.le_add_right base _)
      have headBelow :
          RowsBelow
            (KSparsePolynomial.rows
              (inputAt polynomial point base offset job))
            (base + (offset + 1) * blockWidth polynomial) := by
        apply sparsePolynomial_rows_below
            (boundary :=
              base + (offset + 1) * blockWidth polynomial)
            (inputAt polynomial point base offset job)
            (Nat.lt_of_lt_of_le basePositive (Nat.le_add_right base _))
        · intro index
          exact carried_mono (pointsBelow job index)
            (Nat.le_add_right base _)
        · exact Nat.le_of_eq
            (inputAt_end polynomial point base offset job)
      have tailBelow :=
        rowsFrom_below_end polynomial point basePositive pointsBelow
          rest (offset + 1)
      unfold rowsFrom
      apply rowsBelow_append
      · intro row member column mentioned
        exact Nat.lt_of_lt_of_le
          (headBelow row member column mentioned)
          (by
            simp only [List.length_cons]
            apply Nat.add_le_add_left
            apply Nat.mul_le_mul_right
            omega)
      · simpa only [List.length_cons, Nat.add_assoc,
          Nat.add_left_comm, Nat.add_comm] using tailBelow

private theorem satisfies_append
    {left right : List Row} {assignment : Nat → Nat}
    (leftSatisfied : Satisfies left assignment)
    (rightSatisfied : Satisfies right assignment) :
    Satisfies (left ++ right) assignment := by
  intro row member
  exact (List.mem_append.1 member).elim
    (leftSatisfied row) (rightSatisfied row)

theorem rowsFrom_honest
    {Job : Type} {matrixCount base : Nat}
    (assignment : Nat → Nat)
    (polynomial :
      CCSResidualTable.ConstraintPolynomial ConcreteK matrixCount)
    (point : Job → Fin matrixCount → Carried)
    (basePositive : 0 < base)
    (pointsBelow :
      ∀ job index, CarriedBelow (point job index) base) :
    ∀ (jobs : List Job) (offset : Nat),
      Satisfies (rowsFrom polynomial point base jobs offset)
        (witnessFrom assignment polynomial point base jobs offset)
  | [], _ => by
      intro row member
      simp [rowsFrom] at member
  | job :: rest, offset => by
      let currentInput := inputAt polynomial point base offset job
      let headWitness :=
        KSparsePolynomialHonest.witness currentInput assignment
      let finalWitness :=
        witnessFrom headWitness polynomial point base rest (offset + 1)
      have currentPositive : 0 < currentInput.frameBase := by
        unfold currentInput inputAt
        exact Nat.lt_of_lt_of_le basePositive
          (Nat.le_add_right base _)
      have pointAtCurrent :
          ∀ index,
            CarriedBelow (currentInput.point index)
              currentInput.frameBase := by
        intro index
        exact carried_mono (pointsBelow job index) (by
          unfold currentInput inputAt
          exact Nat.le_add_right base _)
      have headSatisfied :
          Satisfies (KSparsePolynomial.rows currentInput) headWitness :=
        KSparsePolynomialHonest.rows_honest currentInput assignment
          currentPositive pointAtCurrent
      have headPreserved :
          Satisfies (KSparsePolynomial.rows currentInput) finalWitness := by
        apply KHornerSupport.satisfies_extend _ headWitness finalWitness
        · intro row member column mentioned
          symm
          apply witnessFrom_off_before
          have bounded :
              column <
                base + (offset + 1) * blockWidth polynomial := by
            have currentLeEnd :
                currentInput.frameBase ≤
                  base + (offset + 1) * blockWidth polynomial := by
              exact Nat.le_trans
                (Nat.le_add_right currentInput.frameBase _)
                (Nat.le_of_eq (by
                  simpa [currentInput] using
                    inputAt_end polynomial point base offset job))
            apply sparsePolynomial_rows_below
                (boundary :=
                  base + (offset + 1) * blockWidth polynomial)
                currentInput
                (Nat.lt_of_lt_of_le basePositive
                  (Nat.le_add_right base _))
            · intro index
              exact carried_mono (pointAtCurrent index) currentLeEnd
            · exact Nat.le_of_eq (by
                simpa [currentInput] using
                  inputAt_end polynomial point base offset job)
            · exact member
            · exact mentioned
          exact bounded
        · exact headSatisfied
      have tailSatisfied :
          Satisfies (rowsFrom polynomial point base rest (offset + 1))
            finalWitness :=
        rowsFrom_honest headWitness polynomial point basePositive
          pointsBelow rest (offset + 1)
      simpa [rowsFrom, witnessFrom, currentInput, headWitness, finalWitness]
        using satisfies_append headPreserved tailSatisfied

/-- A positional job certificate rewrites the sequential allocator to the
protocol emitter whose base is written through `position`. -/
theorem rowsFrom_eq_flatMap
    {Job : Type} {matrixCount : Nat}
    (jobs : List Job)
    (polynomial :
      CCSResidualTable.ConstraintPolynomial ConcreteK matrixCount)
    (point : Job → Fin matrixCount → Carried)
    (position : Job → Nat)
    (base offset : Nat)
    (positions : jobs.map position = List.range' offset jobs.length) :
    rowsFrom polynomial point base jobs offset =
      jobs.flatMap fun job =>
        KSparsePolynomial.rows
          (inputAt polynomial point base (position job) job) := by
  induction jobs generalizing offset with
  | nil => rfl
  | cons job rest inductionHypothesis =>
      simp only [List.map_cons, List.length_cons, List.range'_succ,
        List.cons.injEq] at positions
      rw [rowsFrom, List.flatMap_cons, positions.1,
        inductionHypothesis (offset + 1) positions.2]

end Nightstream.Implementation.R1CS.Canonical.KSparsePolynomialSequentialHonest
