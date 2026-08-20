import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonFinalRowBridge
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.LeanCompiler.StableRows

/-!
Contract: exact finite relation slice constructor for one PiRLC Poseidon2
leaf block.

Assurance tier: model-level matrix-action constructor.

Owns: expansion of compact ports to absolute production columns, their
canonical finite dot products, and `FinalRowSliceExact` for the relation
slice built from any supplied emitted block.

Does not own: generated-block identity, the complete production relation,
relation satisfaction, lifecycle semantics, or permission to remove rows.

Emits constraints: no new rows. It interprets one existing emitted block.
-/

set_option autoImplicit false
set_option compiler.extract_closed false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafFinalSlice

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.SelectiveCcs.LeanCompiler
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Interpreter
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel.Wire
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallRowProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallFamily
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonFinalRowBridge

abbrev Combination (columns : Nat) :=
  DirectRows.LinearCombination columns

def valueAt {columns : Nat}
    (assignment : Fin columns -> F) (column : Nat) : F :=
  if bounded : column < columns then
    assignment ⟨column, bounded⟩
  else
    0

def explicitValueAt {columns : Nat} (site : CallSite)
    (assignment : Fin columns -> F) : ExplicitColumn -> F
  | .one => valueAt assignment 0
  | .selector => valueAt assignment (selectorColumn site.kind)

def digitValueAt {columns : Nat} (site : CallSite)
    (assignment : Fin columns -> F)
    (slot : Slot) (digit : Fin 41) : F :=
  match digitColumn site slot digit with
  | some column => valueAt assignment column
  | none => 0

def explicitActionAt {columns : Nat} (site : CallSite)
    (assignment : Fin columns -> F)
    (terms : List ExplicitTerm) : F :=
  sum (terms.map fun term =>
    term.coefficient * explicitValueAt site assignment term.column)

def geometricActionAt {columns : Nat} (site : CallSite)
    (assignment : Fin columns -> F)
    (run : GeometricRun) : F :=
  sum (List.ofFn fun digit : Fin 41 =>
    geometricCoefficient run.initial run.ratio digit.val *
      digitValueAt site assignment run.slot digit)

def portActionAt {columns : Nat} (site : CallSite)
    (assignment : Fin columns -> F) (port : Port) : F :=
  explicitActionAt site assignment port.explicit +
    sum (port.geometric.map fun run =>
      geometricActionAt site assignment run)

noncomputable def combinationSum {columns : Nat} :
    List (Combination columns) -> Combination columns
  | [] => fun _ => 0
  | head :: tail => fun column => head column + combinationSum tail column

theorem combinationSum_eval {columns : Nat}
    (combinations : List (Combination columns))
    (assignment : Fin columns -> F) :
    DirectRows.LinearCombination.eval (combinationSum combinations) assignment =
      sum (combinations.map fun combination =>
        DirectRows.LinearCombination.eval combination assignment) := by
  induction combinations with
  | nil =>
      exact StableRows.eval_zero assignment
  | cons head tail inductionHypothesis =>
      simp only [combinationSum, List.map_cons, sum]
      rw [StableRows.eval_pointwise_add, inductionHypothesis]

noncomputable def natCombination {columns : Nat}
    (column : Nat) (coefficient : F) : Combination columns :=
  if bounded : column < columns then
    StableRows.single ⟨column, bounded⟩ coefficient
  else
    fun _ => 0

theorem natCombination_eval {columns : Nat}
    (column : Nat) (coefficient : F)
    (assignment : Fin columns -> F) :
    DirectRows.LinearCombination.eval
        (natCombination column coefficient) assignment =
      coefficient * valueAt assignment column := by
  by_cases bounded : column < columns
  · simp [natCombination, bounded, valueAt, StableRows.eval_single]
  · simp [natCombination, bounded, valueAt, StableRows.eval_zero]

def explicitColumn (site : CallSite) : ExplicitColumn -> Nat
  | .one => 0
  | .selector => selectorColumn site.kind

noncomputable def explicitTermCombination {columns : Nat}
    (site : CallSite) (term : ExplicitTerm) : Combination columns :=
  natCombination (explicitColumn site term.column) term.coefficient

theorem explicitTermCombination_eval {columns : Nat}
    (site : CallSite) (term : ExplicitTerm)
    (assignment : Fin columns -> F) :
    DirectRows.LinearCombination.eval
        (explicitTermCombination site term) assignment =
      term.coefficient *
        explicitValueAt site assignment term.column := by
  rw [explicitTermCombination, natCombination_eval]
  cases term.column <;> rfl

noncomputable def digitCombination {columns : Nat}
    (site : CallSite) (run : GeometricRun)
    (digit : Fin 41) : Combination columns :=
  match digitColumn site run.slot digit with
  | some column =>
      natCombination column
        (geometricCoefficient run.initial run.ratio digit.val)
  | none => fun _ => 0

theorem digitCombination_eval {columns : Nat}
    (site : CallSite) (run : GeometricRun) (digit : Fin 41)
    (assignment : Fin columns -> F) :
    DirectRows.LinearCombination.eval
        (digitCombination site run digit) assignment =
      geometricCoefficient run.initial run.ratio digit.val *
        digitValueAt site assignment run.slot digit := by
  cases columnExact : digitColumn site run.slot digit with
  | none =>
      simp [digitCombination, digitValueAt, columnExact,
        StableRows.eval_zero]
  | some column =>
      simp [digitCombination, digitValueAt, columnExact,
        natCombination_eval]

noncomputable def runCombination {columns : Nat}
    (site : CallSite) (run : GeometricRun) : Combination columns :=
  combinationSum
    (List.ofFn fun digit : Fin 41 => digitCombination site run digit)

theorem runCombination_eval {columns : Nat}
    (site : CallSite) (run : GeometricRun)
    (assignment : Fin columns -> F) :
    DirectRows.LinearCombination.eval
        (runCombination site run) assignment =
      geometricActionAt site assignment run := by
  rw [runCombination, combinationSum_eval]
  unfold geometricActionAt
  rw [List.map_ofFn]
  apply congrArg sum
  apply List.ext_get
  · simp
  · intro index leftBounded rightBounded
    simp only [List.length_ofFn] at leftBounded rightBounded
    simpa only [List.get_ofFn, Function.comp_apply, Fin.cast_mk] using
      digitCombination_eval site run ⟨index, leftBounded⟩ assignment

noncomputable def explicitCombination {columns : Nat}
    (site : CallSite) (terms : List ExplicitTerm) : Combination columns :=
  combinationSum (terms.map (explicitTermCombination site))

theorem explicitCombination_eval {columns : Nat}
    (site : CallSite) (terms : List ExplicitTerm)
    (assignment : Fin columns -> F) :
    DirectRows.LinearCombination.eval
        (explicitCombination site terms) assignment =
      explicitActionAt site assignment terms := by
  induction terms with
  | nil =>
      simp [explicitCombination, explicitActionAt,
        combinationSum, StableRows.eval_zero, sum]
  | cons head tail inductionHypothesis =>
      simp only [explicitCombination, List.map_cons, combinationSum,
        explicitActionAt, sum]
      rw [StableRows.eval_pointwise_add, explicitTermCombination_eval]
      exact congrArg
        (fun value =>
          head.coefficient *
              explicitValueAt site assignment head.column + value)
        (by
          simpa [explicitCombination, explicitActionAt] using
            inductionHypothesis)

noncomputable def geometricCombination {columns : Nat}
    (site : CallSite) (runs : List GeometricRun) : Combination columns :=
  combinationSum (runs.map (runCombination site))

theorem geometricCombination_eval {columns : Nat}
    (site : CallSite) (runs : List GeometricRun)
    (assignment : Fin columns -> F) :
    DirectRows.LinearCombination.eval
        (geometricCombination site runs) assignment =
      sum (runs.map fun run =>
        geometricActionAt site assignment run) := by
  induction runs with
  | nil =>
      simp [geometricCombination, combinationSum, StableRows.eval_zero, sum]
  | cons head tail inductionHypothesis =>
      simp only [geometricCombination, List.map_cons, combinationSum, sum]
      rw [StableRows.eval_pointwise_add, runCombination_eval]
      exact congrArg
        (fun value => geometricActionAt site assignment head + value)
        (by
          simpa [geometricCombination] using inductionHypothesis)

noncomputable def portCombination {columns : Nat}
    (site : CallSite) (port : Port) : Combination columns :=
  fun column =>
    explicitCombination site port.explicit column +
      geometricCombination site port.geometric column

theorem portCombination_eval {columns : Nat}
    (site : CallSite) (port : Port)
    (assignment : Fin columns -> F) :
    DirectRows.LinearCombination.eval
        (portCombination site port) assignment =
      portActionAt site assignment port := by
  unfold portCombination
  rw [StableRows.eval_pointwise_add, explicitCombination_eval,
    geometricCombination_eval]
  rfl

theorem portActionAt_production
    (site : CallSite) (assignment : Fin productionFinalColumns -> F)
    (port : Port) :
    portActionAt site assignment port =
      absolutePortAction site assignment port := by
  rfl

/-- A finite relation whose only active rows are one emitted block exists, and
its matrix action is exact on every assignment. The witness stays inside this
proof so Lean does not compile a closed scan of all production columns. -/
theorem blockRelation_exists (block : EmittedBlock) :
    ∃ relation : InterpretedRelation
        (block.finalRowStart + block.rows.length) productionFinalColumns,
      ∀ assignment : Fin productionFinalColumns -> F,
        FinalRowSliceExact block relation assignment := by
  let relation : InterpretedRelation
      (block.finalRowStart + block.rows.length) productionFinalColumns :=
    { matrices := fun role row column =>
        if block.finalRowStart <= row.val then
          if bounded : row.val - block.finalRowStart < block.rows.length then
            portCombination block.site
              ((block.rows.get
                ⟨row.val - block.finalRowStart, bounded⟩).port role.index)
              column
          else
            0
        else
          0 }
  have matrixRow (offset : Fin block.rows.length) (port : Fin 13) :
      relation.matrixAt port
          (finalRowIndex block (by omega) offset) =
        portCombination block.site ((block.rows.get offset).port port) := by
    funext column
    change
      (if block.finalRowStart <= block.finalRowStart + offset.val then
        if bounded :
            block.finalRowStart + offset.val - block.finalRowStart <
              block.rows.length then
          portCombination block.site
            ((block.rows.get
              ⟨block.finalRowStart + offset.val - block.finalRowStart,
                bounded⟩).port (Role.ofIndex port).index) column
        else
          0
      else
        0) =
        portCombination block.site ((block.rows.get offset).port port) column
    rw [if_pos (by omega)]
    rw [dif_pos (by omega)]
    simp only [Nat.add_sub_cancel_left]
    rw [Role.index_ofIndex]
  have pointExact
      (assignment : Fin productionFinalColumns -> F)
      (offset : Fin block.rows.length) :
      rowPoint relation assignment
          (finalRowIndex block (by omega) offset) =
        absolutePoint block.site assignment (block.rows.get offset) := by
    funext port
    unfold rowPoint matrixImageAt
    change
      DirectRows.LinearCombination.eval
          (relation.matrixAt port
            (finalRowIndex block (by omega) offset)) assignment =
        absolutePortAction block.site assignment
          ((block.rows.get offset).port port)
    rw [matrixRow, portCombination_eval, portActionAt_production]
  refine ⟨relation, ?_⟩
  intro assignment
  exact
    { rowsFit := by omega
      pointExact := pointExact assignment }

/-- Opaque choice of the exact one-block relation. Consumers use
`blockRelation_exact`; they do not unfold the production-width witness. -/
noncomputable def blockRelation (block : EmittedBlock) :
    InterpretedRelation
      (block.finalRowStart + block.rows.length) productionFinalColumns :=
  Classical.choose (blockRelation_exists block)

theorem blockRelation_exact
    (block : EmittedBlock)
    (assignment : Fin productionFinalColumns -> F) :
    FinalRowSliceExact block (blockRelation block) assignment :=
  (Classical.choose_spec (blockRelation_exists block)) assignment

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafFinalSlice
