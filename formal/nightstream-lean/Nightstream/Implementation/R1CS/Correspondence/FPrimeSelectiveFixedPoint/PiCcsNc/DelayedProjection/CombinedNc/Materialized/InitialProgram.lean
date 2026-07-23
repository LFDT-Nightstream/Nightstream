import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Semantics

/-!
Exact straight-line semantics of the production combined-NC claimed-initial
program.

Owns: the 53 seven-row Horner steps over the 54 pending-parent coefficients,
the final five-row multiplication by `batchWeight`, and the kernel theorem
that satisfaction computes the advertised quadratic-extension value.

Does not own: generated-row equality, selective-compiler refinement,
transcript sampling, pending-parent authority, commitment binding, costs, or
row removal.

Emits constraints: 376 rows: `53 * 7` Horner rows and one five-row
quadratic-extension multiplication.
-/

/-!
| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.initial_program` | State and prove the initial claimed-sum source program used by the combined-NC chain. | derived |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.InitialProgram

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Semantics

abbrev rawBoundary : RawBoundaryMap := Metadata.boundary

def pendingColumns : List KColumns :=
  rawBoundary.pendingParentYZcolColumns.map rawKColumnsToColumns

def producerBetaColumns : KColumns :=
  rawKColumnsToColumns rawBoundary.producerBetaColumns

def batchWeightColumns : KColumns :=
  rawKColumnsToColumns rawBoundary.batchWeightColumns

def claimedInitialColumns : KColumns :=
  rawKColumnsToColumns rawBoundary.claimedInitialColumns

/-- The first fresh column is recovered from the final claimed-output column
and the exact `53 * 7 + 5` allocation schedule. The row artifact separately
checks every sparse coefficient, so this arithmetic is not authority. -/
def firstAllocatedColumn : Nat :=
  claimedInitialColumns.c0 - (7 * (activeLaneCount - 1) + 3)

private def addDefinitions
    (product coefficient output : KColumns) : List Definition :=
  [⟨output.c0, .linear [(product.c0, 1), (coefficient.c0, 1)]⟩,
   ⟨output.c1, .linear [(product.c1, 1), (coefficient.c1, 1)]⟩]

private def multiplicationAt
    (base : Nat) (accumulator beta : KColumns) : KMulTrace :=
  KMulTrace.ofColumns accumulator beta ⟨base + 3, base + 4⟩

private def nextAccumulator (base : Nat) : KColumns :=
  ⟨base + 5, base + 6⟩

/-- Exact Rust allocation order: multiply the current accumulator by beta,
then allocate the two limb-wise sums with the next descending coefficient. -/
def hornerDefinitionsFrom (base : Nat) (accumulator beta : KColumns) :
    List KColumns → List Definition
  | [] => []
  | coefficient :: remaining =>
      let multiplication := multiplicationAt base accumulator beta
      let next := nextAccumulator base
      multiplication.definitions ++
        (addDefinitions multiplication.output coefficient next ++
          hornerDefinitionsFrom (base + 7) next beta remaining)

def hornerOutputFrom (base : Nat) (accumulator beta : KColumns) :
    List KColumns → KColumns
  | [] => accumulator
  | _ :: remaining =>
      hornerOutputFrom (base + 7) (nextAccumulator base) beta remaining

def hornerDefinitions : List Definition :=
  match pendingColumns.reverse with
  | [] => []
  | highest :: remaining =>
      hornerDefinitionsFrom firstAllocatedColumn highest producerBetaColumns
        remaining

def hornerOutput : KColumns :=
  match pendingColumns.reverse with
  | [] => default
  | highest :: remaining =>
      hornerOutputFrom firstAllocatedColumn highest producerBetaColumns
        remaining

def finalMultiplication : KMulTrace :=
  KMulTrace.ofColumns batchWeightColumns hornerOutput claimedInitialColumns

def definitions : List Definition :=
  hornerDefinitions ++ finalMultiplication.definitions

def rows : List Row := definitions.map Definition.builderRow

private theorem hornerDefinitionsFrom_length
    (base : Nat) (accumulator beta : KColumns)
    (coefficients : List KColumns) :
    (hornerDefinitionsFrom base accumulator beta coefficients).length =
      7 * coefficients.length := by
  induction coefficients generalizing base accumulator with
  | nil => simp [hornerDefinitionsFrom]
  | cons coefficient remaining inductionHypothesis =>
      simp [hornerDefinitionsFrom, multiplicationAt, KMulTrace.definitions,
        addDefinitions, inductionHypothesis]
      omega

/-- Executable value recurrence matching the program's descending-coefficient
Horner walk. -/
def hornerFold (point accumulator : K) : List K → K
  | [] => accumulator
  | coefficient :: remaining =>
      hornerFold point (K.add coefficient (K.mul accumulator point)) remaining

private theorem hornerFold_append (point accumulator : K)
    (left right : List K) :
    hornerFold point accumulator (left ++ right) =
      hornerFold point (hornerFold point accumulator left) right := by
  induction left generalizing accumulator with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.cons_append, hornerFold]
      exact inductionHypothesis _

/-- The descending accumulator recurrence is the same constant-first Horner
machine used by the independent projection semantics. -/
private theorem eval_eq_hornerFold_reverse
    (coefficients : List K) (point : K) :
    Nightstream.SuperNeo.ProjectionCheck.eval K.ops coefficients point =
      hornerFold point K.zero coefficients.reverse := by
  induction coefficients with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      change K.add head
          (K.mul point
            (Nightstream.SuperNeo.ProjectionCheck.eval K.ops tail point)) = _
      rw [List.reverse_cons, hornerFold_append, ← inductionHypothesis]
      simp only [hornerFold, K.zero_mul, K.add_zero]
      rw [K.mul_comm]

private theorem multiplicationAt_layout
    (base : Nat) (accumulator beta : KColumns) :
    (multiplicationAt base accumulator beta).SumLayoutValid := by
  simp [multiplicationAt, KMulTrace.ofColumns, KMulTrace.SumLayoutValid,
    KTerms.ofColumns]

private theorem multiplicationDefinitions_canonical (trace : KMulTrace) :
    ∀ definition ∈ trace.definitions, definition.Canonical := by
  intro definition member
  simp [KMulTrace.definitions] at member
  rcases member with rfl | rfl | rfl | rfl | rfl <;>
    simp [Definition.Canonical, CanonicalTerms, goldilocksP]

private theorem addDefinitions_canonical
    (product coefficient output : KColumns) :
    ∀ definition ∈ addDefinitions product coefficient output,
      definition.Canonical := by
  intro definition member
  simp [addDefinitions] at member
  rcases member with rfl | rfl <;>
    simp [Definition.Canonical, CanonicalTerms, goldilocksP]

private theorem hornerDefinitionsFrom_canonical
    (base : Nat) (accumulator beta : KColumns) (remaining : List KColumns) :
    ∀ definition ∈ hornerDefinitionsFrom base accumulator beta remaining,
      definition.Canonical := by
  induction remaining generalizing base accumulator with
  | nil => simp [hornerDefinitionsFrom]
  | cons coefficient remaining inductionHypothesis =>
      intro definition member
      simp only [hornerDefinitionsFrom] at member
      rcases List.mem_append.mp member with multiplicationMember | restMember
      · exact multiplicationDefinitions_canonical _ definition
          multiplicationMember
      · rcases List.mem_append.mp restMember with addMember | tailMember
        · exact addDefinitions_canonical _ _ _ definition addMember
        · exact inductionHypothesis (base := base + 7)
            (accumulator := nextAccumulator base) definition tailMember

private theorem hornerDefinitions_canonical :
    ∀ definition ∈ hornerDefinitions, definition.Canonical := by
  unfold hornerDefinitions
  split
  · simp
  · exact hornerDefinitionsFrom_canonical _ _ _ _

theorem definitions_canonical :
    ∀ definition ∈ definitions, definition.Canonical := by
  intro definition member
  rcases List.mem_append.mp member with hornerMember | finalMember
  · exact hornerDefinitions_canonical definition hornerMember
  · exact multiplicationDefinitions_canonical finalMultiplication definition
      finalMember

private theorem addOutput_sound
    {assignment : Nat → Nat}
    (product coefficient output : KColumns)
    (definitionsHold :
      DefinitionsHold assignment
        (addDefinitions product coefficient output)) :
    output.value assignment =
      K.add (product.value assignment) (coefficient.value assignment) := by
  have low := definitionsHold
    ⟨output.c0, .linear [(product.c0, 1), (coefficient.c0, 1)]⟩
    (by simp [addDefinitions])
  have high := definitionsHold
    ⟨output.c1, .linear [(product.c1, 1), (coefficient.c1, 1)]⟩
    (by simp [addDefinitions])
  simp only [KColumns.value, K.add, K.mk.injEq]
  constructor
  · apply Fin.ext
    simpa [Definition.Holds, Rhs.eval, KColumns.value, K.add, baseAt,
      residue, lcEval, Fin.val_add] using
        congrArg (fun value => value % goldilocksP) low
  · apply Fin.ext
    simpa [Definition.Holds, Rhs.eval, KColumns.value, K.add, baseAt,
      residue, lcEval, Fin.val_add] using
        congrArg (fun value => value % goldilocksP) high

private theorem hornerDefinitionsFrom_sound
    {assignment : Nat → Nat}
    (base : Nat) (accumulator beta : KColumns) (remaining : List KColumns)
    (definitionsHold :
      DefinitionsHold assignment
        (hornerDefinitionsFrom base accumulator beta remaining)) :
    (hornerOutputFrom base accumulator beta remaining).value assignment =
      hornerFold (beta.value assignment) (accumulator.value assignment)
        (remaining.map fun coefficient => coefficient.value assignment) := by
  induction remaining generalizing base accumulator with
  | nil => rfl
  | cons coefficient remaining inductionHypothesis =>
      let multiplication := multiplicationAt base accumulator beta
      let next := nextAccumulator base
      change DefinitionsHold assignment
        (multiplication.definitions ++
          (addDefinitions multiplication.output coefficient next ++
            hornerDefinitionsFrom (base + 7) next beta remaining)) at definitionsHold
      have multiplicationHolds :
          DefinitionsHold assignment multiplication.definitions := by
        intro definition member
        exact definitionsHold definition (List.mem_append_left _ member)
      have additionHolds : DefinitionsHold assignment
          (addDefinitions multiplication.output coefficient next) := by
        intro definition member
        exact definitionsHold definition
          (List.mem_append_right multiplication.definitions
            (List.mem_append_left _ member))
      have tailHolds : DefinitionsHold assignment
          (hornerDefinitionsFrom (base + 7) next beta remaining) := by
        intro definition member
        exact definitionsHold definition
          (List.mem_append_right multiplication.definitions
            (List.mem_append_right _ member))
      have multiplied := multiplication.sound assignment
        (multiplicationAt_layout base accumulator beta) multiplicationHolds
      have multiplied' : multiplication.output.value assignment =
          K.mul (accumulator.value assignment) (beta.value assignment) := by
        simpa [multiplication, multiplicationAt, KMulTrace.ofColumns,
          KTerms.ofColumns_value] using multiplied
      have added := addOutput_sound multiplication.output coefficient next
        additionHolds
      have tail := inductionHypothesis (base := base + 7)
        (accumulator := next) tailHolds
      simp only [hornerOutputFrom, List.map_cons, hornerFold]
      rw [tail, added, multiplied']
      rw [K.add_comm]

/-! This certificate evaluates exactly 54 proof-free `RawKColumns` records.
It does not inspect rows, assignments, decoded structures, or proof objects. -/
set_option maxRecDepth 100000 in
theorem pendingColumns_length : pendingColumns.length = activeLaneCount := by
  native_decide

theorem definition_count : definitions.length = 376 := by
  cases reversed : pendingColumns.reverse with
  | nil =>
      have lengthEq := congrArg List.length reversed
      simp [pendingColumns_length, activeLaneCount] at lengthEq
  | cons highest remaining =>
      have lengthEq := congrArg List.length reversed
      simp only [List.length_reverse, List.length_cons,
        pendingColumns_length] at lengthEq
      simp [activeLaneCount] at lengthEq
      simp only [definitions, hornerDefinitions, reversed,
        List.length_append, hornerDefinitionsFrom_length]
      have finalLength : finalMultiplication.definitions.length = 5 := rfl
      rw [finalLength]
      omega

/-- Exact straight-line soundness. Every premise is either ordinary R1CS
satisfaction or canonical field representation; the claimed formula is not a
premise. -/
theorem sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    claimedInitialColumns.value assignment =
      K.mul (batchWeightColumns.value assignment)
        (Nightstream.SuperNeo.ProjectionCheck.eval K.ops
          (pendingColumns.map fun coefficient => coefficient.value assignment)
          (producerBetaColumns.value assignment)) := by
  have definitionsHold : DefinitionsHold assignment definitions :=
    builderDefinitions_sound canonical constantOne definitions_canonical
      satisfies
  have hornerHolds : DefinitionsHold assignment hornerDefinitions := by
    intro definition member
    exact definitionsHold definition (List.mem_append_left _ member)
  have finalHolds :
      DefinitionsHold assignment finalMultiplication.definitions := by
    intro definition member
    exact definitionsHold definition
      (List.mem_append_right hornerDefinitions member)
  have finalValue := finalMultiplication.sound assignment
    (by simp [finalMultiplication, KMulTrace.ofColumns,
      KMulTrace.SumLayoutValid, KTerms.ofColumns]) finalHolds
  cases reversed : pendingColumns.reverse with
  | nil =>
      have impossible := congrArg List.length reversed
      simp [pendingColumns_length, activeLaneCount] at impossible
  | cons highest remaining =>
      have hornerValue := hornerDefinitionsFrom_sound
        (assignment := assignment) firstAllocatedColumn highest
        producerBetaColumns remaining (by
          simpa [hornerDefinitions, reversed] using hornerHolds)
      have reverseValues :
          (pendingColumns.map fun coefficient =>
              coefficient.value assignment).reverse =
            highest.value assignment ::
              remaining.map fun coefficient => coefficient.value assignment := by
        simpa using congrArg
          (List.map fun coefficient => coefficient.value assignment) reversed
      have evaluated := eval_eq_hornerFold_reverse
        (pendingColumns.map fun coefficient => coefficient.value assignment)
        (producerBetaColumns.value assignment)
      rw [reverseValues] at evaluated
      simp only [hornerFold, K.zero_mul, K.add_zero] at evaluated
      have exactHorner : hornerOutput.value assignment =
          hornerFold (producerBetaColumns.value assignment)
            (highest.value assignment)
            (remaining.map fun coefficient => coefficient.value assignment) := by
        simpa [hornerOutput, reversed] using hornerValue
      calc
        claimedInitialColumns.value assignment =
            K.mul (batchWeightColumns.value assignment)
              (hornerOutput.value assignment) := by
          change claimedInitialColumns.value assignment =
              K.mul ((KTerms.ofColumns batchWeightColumns).value assignment)
                ((KTerms.ofColumns hornerOutput).value assignment) at finalValue
          simpa only [KTerms.ofColumns_value] using finalValue
        _ = K.mul (batchWeightColumns.value assignment)
              (hornerFold (producerBetaColumns.value assignment)
                (highest.value assignment)
                (remaining.map fun coefficient =>
                  coefficient.value assignment)) := by rw [exactHorner]
        _ = K.mul (batchWeightColumns.value assignment)
              (Nightstream.SuperNeo.ProjectionCheck.eval K.ops
                (pendingColumns.map fun coefficient => coefficient.value assignment)
                (producerBetaColumns.value assignment)) := by rw [evaluated]

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.InitialProgram
