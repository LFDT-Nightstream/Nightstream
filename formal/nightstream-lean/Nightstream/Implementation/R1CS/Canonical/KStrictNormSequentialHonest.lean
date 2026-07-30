import Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport
import Nightstream.Implementation.R1CS.Canonical.KStrictNormHonest

/-!
Contract: honest completeness for consecutive strict-norm programs.

Owns the left-to-right witness composition for equal six-column
`KStrictNorm` blocks and preservation of completed prefixes.  Protocol
consumers provide the job order and prove that their physical emitter uses
the same consecutive ordinals.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KStrictNormSequentialHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport
open Nightstream.Implementation.R1CS.Canonical.KMul

def inputAt
    {Job : Type}
    (value : Job → Carried) (base offset : Nat) (job : Job) :
    KStrictNorm.Input where
  value := value job
  frameBase := base + 6 * offset

def rowsFrom
    {Job : Type}
    (jobs : List Job) (value : Job → Carried)
    (base offset : Nat) : List Row :=
  match jobs with
  | [] => []
  | job :: rest =>
      KStrictNorm.rows (inputAt value base offset job) ++
        rowsFrom rest value base (offset + 1)

def witnessFrom
    {Job : Type}
    (assignment : Nat → Nat)
    (jobs : List Job) (value : Job → Carried)
    (base offset : Nat) : Nat → Nat :=
  match jobs with
  | [] => assignment
  | job :: rest =>
      let head :=
        KStrictNormHonest.honestAssignment
          (inputAt value base offset job) assignment
      witnessFrom head rest value base (offset + 1)

theorem witnessFrom_off_before
    {Job : Type}
    (assignment : Nat → Nat) (value : Job → Carried) :
    ∀ (jobs : List Job) (base offset column : Nat),
      column < base + 6 * offset →
      witnessFrom assignment jobs value base offset column =
        assignment column
  | [], _, _, _, _ => rfl
  | job :: rest, base, offset, column, below => by
      rw [witnessFrom,
        witnessFrom_off_before
          (KStrictNormHonest.honestAssignment
            (inputAt value base offset job) assignment)
          value rest base (offset + 1) column (by omega),
        KStrictNormHonest.honestAssignment_preserves_below
          (inputAt value base offset job) assignment column (by
            simpa [inputAt] using below)]

private theorem satisfies_append
    {left right : List Row} {assignment : Nat → Nat}
    (leftSatisfied : Satisfies left assignment)
    (rightSatisfied : Satisfies right assignment) :
    Satisfies (left ++ right) assignment := by
  intro row member
  exact (List.mem_append.1 member).elim
    (leftSatisfied row) (rightSatisfied row)

theorem rowsFrom_honest
    {Job : Type} {base : Nat}
    (assignment : Nat → Nat) (value : Job → Carried)
    (basePositive : 0 < base)
    (valuesBelow : ∀ job, CarriedBelow (value job) base) :
    ∀ (jobs : List Job) (offset : Nat),
      Satisfies (rowsFrom jobs value base offset)
        (witnessFrom assignment jobs value base offset)
  | [], _ => by
      intro row member
      simp [rowsFrom] at member
  | job :: rest, offset => by
      let currentInput := inputAt value base offset job
      let headWitness :=
        KStrictNormHonest.honestAssignment currentInput assignment
      let finalWitness :=
        witnessFrom headWitness rest value base (offset + 1)
      have currentPositive : 0 < currentInput.frameBase := by
        simp only [currentInput, inputAt]
        omega
      have currentValueBelow :
          CarriedBelow currentInput.value currentInput.frameBase := by
        simpa only [currentInput, inputAt] using
          carried_mono (valuesBelow job) (by omega)
      have headSatisfied :
          Satisfies (KStrictNorm.rows currentInput) headWitness :=
        KStrictNormHonest.rows_honest currentInput assignment
          currentPositive currentValueBelow
      have headPreserved :
          Satisfies (KStrictNorm.rows currentInput) finalWitness := by
        apply KHornerSupport.satisfies_extend _ headWitness finalWitness
        · intro row member column mentioned
          symm
          apply witnessFrom_off_before
          exact strictNorm_rows_below currentInput
            (base + 6 * (offset + 1)) (by omega)
            (carried_mono currentValueBelow (by
              simp only [currentInput, inputAt]
              omega))
            (by
              simp only [currentInput, inputAt]
              omega)
            row member column mentioned
        · exact headSatisfied
      have tailSatisfied :
          Satisfies (rowsFrom rest value base (offset + 1)) finalWitness :=
        rowsFrom_honest headWitness value basePositive valuesBelow
          rest (offset + 1)
      simpa [rowsFrom, witnessFrom, currentInput, headWitness, finalWitness]
        using satisfies_append headPreserved tailSatisfied

theorem rowsFrom_below_end
    {Job : Type} {base : Nat}
    (value : Job → Carried)
    (basePositive : 0 < base)
    (valuesBelow : ∀ job, CarriedBelow (value job) base) :
    ∀ (jobs : List Job) (offset : Nat)
      (row : Row), row ∈ rowsFrom jobs value base offset →
      ∀ column,
        (LinCombNormal.Mentions row.a column ∨
          LinCombNormal.Mentions row.b column ∨
          LinCombNormal.Mentions row.c column) →
        column < base + 6 * (offset + jobs.length)
  | [], _, _, member, _, _ => by
      simp [rowsFrom] at member
  | job :: rest, offset, row, member, column, mentioned => by
      simp only [rowsFrom, List.mem_append] at member
      rcases member with inHead | inTail
      · have currentValueBelow :
            CarriedBelow (value job) (base + 6 * offset) :=
          carried_mono (valuesBelow job) (by omega)
        have bounded :=
          strictNorm_rows_below
            (inputAt value base offset job)
            (base + 6 * (offset + 1)) (by omega)
            (carried_mono currentValueBelow (by omega))
            (by simp only [inputAt]; omega)
            row inHead column mentioned
        simp only [List.length_cons]
        exact Nat.lt_of_lt_of_le bounded (by omega)
      · have bounded :=
          rowsFrom_below_end value basePositive valuesBelow rest
            (offset + 1) row inTail column mentioned
        simp only [List.length_cons]
        exact Nat.lt_of_lt_of_le bounded (by omega)

theorem rowsFrom_eq_flatMap
    {Job : Type}
    (jobs : List Job) (value : Job → Carried)
    (position : Job → Nat) (base offset : Nat)
    (positions :
      jobs.map position = List.range' offset jobs.length) :
    rowsFrom jobs value base offset =
      jobs.flatMap fun job =>
        KStrictNorm.rows (inputAt value base (position job) job) := by
  induction jobs generalizing offset with
  | nil => rfl
  | cons job rest inductionHypothesis =>
      simp only [List.map_cons, List.length_cons, List.range'_succ,
        List.cons.injEq] at positions
      rw [rowsFrom, List.flatMap_cons, positions.1,
        inductionHypothesis (offset + 1) positions.2]

end Nightstream.Implementation.R1CS.Canonical.KStrictNormSequentialHonest
