import Nightstream.Implementation.R1CS.Canonical.KBooleanMleHonest

/-!
Contract: honest completeness for a consecutive sequence of independent
Boolean-MLE programs.

Owns: the exact left-to-right witness composition for equal-width MLE blocks,
preservation of completed prefixes, and the bridge from positional job lists
to emitters whose block base is written through an explicit ordinal.

Does not own any protocol-specific job enumeration.  Consumers must prove
that their typed enumeration has consecutive ordinals.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KBooleanMleSequentialHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KBooleanMleSupport
open Nightstream.Implementation.R1CS.Canonical.KHornerSupport
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

def blockWidth (variables : Nat) : Nat :=
  3 * KBooleanMle.frameCount variables

def rowsFrom
    {Job : Type} {variables : Nat}
    (jobs : List Job)
    (table : Job → BooleanTable KMul.Carried variables)
    (coordinates : Job → List KMul.Carried)
    (base offset : Nat) : List Row :=
  match jobs with
  | [] => []
  | job :: rest =>
      KBooleanMle.rows
          (KFrames.frameAt
            (base + blockWidth variables * offset))
          (table job) (coordinates job) 0 ++
        rowsFrom rest table coordinates base (offset + 1)

def witnessFrom
    {Job : Type} {variables : Nat}
    (assignment : Nat → Nat)
    (jobs : List Job)
    (table : Job → BooleanTable KMul.Carried variables)
    (coordinates : Job → List KMul.Carried)
    (base offset : Nat) : Nat → Nat :=
  match jobs with
  | [] => assignment
  | job :: rest =>
      let head :=
        KBooleanMleHonest.witness assignment
          (base + blockWidth variables * offset)
          (table job) (coordinates job) 0
      witnessFrom head rest table coordinates base (offset + 1)

theorem witnessFrom_off_before
    {Job : Type} {variables : Nat}
    (assignment : Nat → Nat)
    (table : Job → BooleanTable KMul.Carried variables)
    (coordinates : Job → List KMul.Carried) :
    ∀ (jobs : List Job) (base offset column : Nat),
      column < base + blockWidth variables * offset →
      witnessFrom assignment jobs table coordinates base offset column =
        assignment column
  | [], _, _, _, _ => rfl
  | job :: rest, base, offset, column, below => by
      rw [witnessFrom,
        witnessFrom_off_before
          (KBooleanMleHonest.witness assignment
            (base + blockWidth variables * offset)
          (table job) (coordinates job) 0)
          table coordinates rest base (offset + 1) column (by
            unfold blockWidth at below ⊢
            rw [Nat.mul_succ]
            omega),
        KBooleanMleHonest.witness_off_before assignment
          (base + blockWidth variables * offset)
          (table job) (coordinates job) 0 column (by simpa using below)]

theorem tableBelow_mono
    {variables : Nat}
    (table : BooleanTable KMul.Carried variables) :
    ∀ {lower upper : Nat},
      TableBelowBase table lower → lower ≤ upper →
      TableBelowBase table upper := by
  induction table with
  | leaf value =>
      intro lower upper below ordered
      exact carriedBelow_mono below ordered
  | branch left right leftInduction rightInduction =>
      intro lower upper below ordered
      exact ⟨leftInduction below.1 ordered, rightInduction below.2 ordered⟩

theorem coordinatesBelow_mono
    (coordinates : List KMul.Carried)
    {lower upper : Nat}
    (below : CoordinatesBelowBase coordinates lower)
    (ordered : lower ≤ upper) :
    CoordinatesBelowBase coordinates upper :=
  fun coordinate member =>
    carriedBelow_mono (below coordinate member) ordered

private theorem satisfies_append
    {left right : List Row} {assignment : Nat → Nat}
    (leftSatisfied : Satisfies left assignment)
    (rightSatisfied : Satisfies right assignment) :
    Satisfies (left ++ right) assignment := by
  intro row member
  exact (List.mem_append.1 member).elim
    (leftSatisfied row) (rightSatisfied row)

theorem rowsFrom_honest
    {Job : Type} {variables base : Nat}
    (assignment : Nat → Nat)
    (table : Job → BooleanTable KMul.Carried variables)
    (coordinates : Job → List KMul.Carried)
    (basePositive : 0 < base)
    (tablesBelow : ∀ job, TableBelowBase (table job) base)
    (coordinatesBelow :
      ∀ job, CoordinatesBelowBase (coordinates job) base) :
    ∀ (jobs : List Job) (offset : Nat),
      Satisfies (rowsFrom jobs table coordinates base offset)
        (witnessFrom assignment jobs table coordinates base offset)
  | [], _ => by
      intro row member
      simp [rowsFrom] at member
  | job :: rest, offset => by
      let currentBase := base + blockWidth variables * offset
      let headWitness :=
        KBooleanMleHonest.witness assignment currentBase
          (table job) (coordinates job) 0
      let finalWitness :=
        witnessFrom headWitness rest table coordinates base (offset + 1)
      have currentPositive : 0 < currentBase := by
        unfold currentBase
        omega
      have tableAtCurrent :
          TableBelowBase (table job) currentBase :=
        tableBelow_mono (table job) (tablesBelow job) (by
          unfold currentBase
          omega)
      have coordinatesAtCurrent :
          CoordinatesBelowBase (coordinates job) currentBase :=
        coordinatesBelow_mono (coordinates job) (coordinatesBelow job) (by
          unfold currentBase
          omega)
      have headSatisfied :
          Satisfies
            (KBooleanMle.rows (KFrames.frameAt currentBase)
              (table job) (coordinates job) 0)
            headWitness :=
        KBooleanMleHonest.witness_satisfies_from_base
          assignment currentBase (table job) (coordinates job)
          tableAtCurrent coordinatesAtCurrent
      have headPreserved :
          Satisfies
            (KBooleanMle.rows (KFrames.frameAt currentBase)
              (table job) (coordinates job) 0)
            finalWitness := by
        apply satisfies_extend _ headWitness finalWitness
        · intro row member column mentioned
          symm
          apply witnessFrom_off_before
          have bounded :=
            KBooleanMleSupport.rows_below currentBase
              (table job) (coordinates job) 0
              tableAtCurrent coordinatesAtCurrent
              row member column mentioned
          unfold currentBase at bounded
          unfold blockWidth at bounded ⊢
          rw [Nat.mul_succ]
          omega
        · exact headSatisfied
      have tailSatisfied :
          Satisfies
            (rowsFrom rest table coordinates base (offset + 1))
            finalWitness :=
        rowsFrom_honest headWitness table coordinates basePositive
          tablesBelow coordinatesBelow rest (offset + 1)
      simpa [rowsFrom, witnessFrom, currentBase, headWitness, finalWitness]
        using satisfies_append headPreserved tailSatisfied

/-- Every row in a sequential batch lies below the exact end of the batch's
consecutive frame allocation. -/
theorem rowsFrom_below_end
    {Job : Type} {variables base : Nat}
    (table : Job → BooleanTable KMul.Carried variables)
    (coordinates : Job → List KMul.Carried)
    (tablesBelow : ∀ job, TableBelowBase (table job) base)
    (coordinatesBelow :
      ∀ job, CoordinatesBelowBase (coordinates job) base) :
    ∀ (jobs : List Job) (offset : Nat)
      (row : Row), row ∈ rowsFrom jobs table coordinates base offset →
      ∀ column,
        (Mentions row.a column ∨ Mentions row.b column ∨
          Mentions row.c column) →
        column <
          base + blockWidth variables * (offset + jobs.length)
  | [], _, _, member, _, _ => by
      simp [rowsFrom] at member
  | job :: rest, offset, row, member, column, mentioned => by
      simp only [rowsFrom, List.mem_append] at member
      rcases member with inHead | inTail
      · have tableAtCurrent :
            TableBelowBase (table job)
              (base + blockWidth variables * offset) :=
          tableBelow_mono (table job) (tablesBelow job) (by
            omega)
        have coordinatesAtCurrent :
            CoordinatesBelowBase (coordinates job)
              (base + blockWidth variables * offset) :=
          coordinatesBelow_mono (coordinates job)
            (coordinatesBelow job) (by
              omega)
        have bounded :=
          KBooleanMleSupport.rows_below
            (base + blockWidth variables * offset)
            (table job) (coordinates job) 0
            tableAtCurrent coordinatesAtCurrent
            row inHead column mentioned
        simp only [List.length_cons]
        have blockEnd :
            column <
              base + blockWidth variables * (offset + 1) := by
          unfold blockWidth at bounded ⊢
          rw [Nat.mul_succ]
          omega
        exact Nat.lt_of_lt_of_le blockEnd (by
          apply Nat.add_le_add_left
          apply Nat.mul_le_mul_left
          omega)
      · have bounded :=
          rowsFrom_below_end table coordinates tablesBelow coordinatesBelow
            rest (offset + 1) row inTail column mentioned
        simp only [List.length_cons]
        have same :
            offset + 1 + rest.length = offset + (rest.length + 1) := by
          omega
        simpa [same] using bounded

/-- A positional certificate turns the sequential emitter into the
protocol-specific `flatMap` whose base is written through `position`. -/
theorem rowsFrom_eq_flatMap
    {Job : Type} {variables : Nat}
    (jobs : List Job)
    (table : Job → BooleanTable KMul.Carried variables)
    (coordinates : Job → List KMul.Carried)
    (position : Job → Nat)
    (base offset : Nat)
    (positions :
      jobs.map position = List.range' offset jobs.length) :
    rowsFrom jobs table coordinates base offset =
      jobs.flatMap fun job =>
        KBooleanMle.rows
          (KFrames.frameAt
            (base + blockWidth variables * position job))
          (table job) (coordinates job) 0 := by
  induction jobs generalizing offset with
  | nil => rfl
  | cons job rest inductionHypothesis =>
      simp only [List.map_cons, List.length_cons, List.range'_succ,
        List.cons.injEq] at positions
      rw [rowsFrom, List.flatMap_cons, positions.1,
        inductionHypothesis (offset + 1) positions.2]

end Nightstream.Implementation.R1CS.Canonical.KBooleanMleSequentialHonest
