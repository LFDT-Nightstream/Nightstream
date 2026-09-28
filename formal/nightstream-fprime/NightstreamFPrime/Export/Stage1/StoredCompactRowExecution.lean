import NightstreamFPrime.Export.Stage1.CompactRowExecution
import NightstreamFPrime.Export.Stage1.StoredWitnessExecution

/-!
Array storage for the existing compact-row arithmetic core. The output recipe
reads the original snapshot; each row computes A * B before its optional write
and checks C afterward. Refinement requires the same physical write bounds and
local-output bounds as the guarded runner. No template success is asserted.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.StoredCompactRowExecution

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package
open StoredWitnessExecution (asEnv write write_size asEnv_write)

/-- The enclosing executor checks the complete local interval. Each row also
checks its declared local index before writing. -/
def step (inputColumn : Nat → Nat) (localStart localCount : Nat)
    (values : Array F) (row : CompactTemplateRow) : Option (Array F) :=
  let product :=
    (CompactRows.instantiateCombination inputColumn localStart row.a).eval (asEnv values) *
      (CompactRows.instantiateCombination inputColumn localStart row.b).eval (asEnv values)
  match row.outputLocal with
  | none =>
      if product =
          (CompactRows.instantiateCombination inputColumn localStart row.c).eval (asEnv values)
      then some values else none
  | some localIndex =>
      if localIndex < localCount then
        let after := write values (localStart + localIndex) product
        if product =
            (CompactRows.instantiateCombination inputColumn localStart row.c).eval (asEnv after)
        then some after else none
      else none

/-- Process exactly the supplied row order and stop at the first rejection. -/
def run (inputColumn : Nat → Nat) (localStart localCount : Nat) :
    Array F → List CompactTemplateRow → Option (Array F)
  | values, [] => some values
  | values, row :: rest =>
      (step inputColumn localStart localCount values row).bind fun after =>
        run inputColumn localStart localCount after rest

/-- Check both physical write regions before reading the output recipe. The
per-row local-count guard remains in step. -/
def execute (inputColumn : Nat → Nat) (localStart : Nat)
    (template : CompactRowTemplate) (values : Array F) : Option (Array F) :=
  if inputColumn template.outputInput < values.size ∧
      localStart + template.localColumnCount ≤ values.size then
    let output := template.outputRecipe.eval (fun input => asEnv values (inputColumn input))
    let seeded := write values (inputColumn template.outputInput) output
    run inputColumn localStart template.localColumnCount seeded template.rows
  else none

private theorem step_size (inputColumn : Nat → Nat) (localStart localCount : Nat)
    (values : Array F) (row : CompactTemplateRow) (after : Array F)
    (success : step inputColumn localStart localCount values row = some after) :
    after.size = values.size := by
  unfold step at success
  cases selected : row.outputLocal with
  | none =>
      simp only [selected] at success
      split_ifs at success
      · have same := Option.some.inj success
        rw [← same]
  | some localIndex =>
      simp only [selected] at success
      split_ifs at success
      · have same := Option.some.inj success
        rw [← same, write_size]

/-- Stored and functional row checks agree exactly, including rejection.
The local interval and the row's declared output must both be valid. -/
theorem step_eq (inputColumn : Nat → Nat) (localStart localCount : Nat)
    (values : Array F) (row : CompactTemplateRow)
    (fits : localStart + localCount ≤ values.size)
    (localBound : ∀ index, row.outputLocal = some index → index < localCount) :
    (step inputColumn localStart localCount values row).map asEnv =
      CompactRowExecution.step inputColumn localStart (asEnv values) row := by
  cases selected : row.outputLocal with
  | none =>
      simp only [step, CompactRowExecution.step, selected]
      split_ifs <;> rfl
  | some localIndex =>
      have inside : localIndex < localCount := localBound localIndex selected
      have targetBound : localStart + localIndex < values.size := by omega
      simp only [step, CompactRowExecution.step, selected, if_pos inside,
        asEnv_write values (localStart + localIndex) _ targetBound]
      split_ifs <;>
        simp only [Option.map_some, Option.map_none,
          asEnv_write values (localStart + localIndex) _ targetBound]

/-- Every row uses the same current environment. Successful writes preserve
the array size, so the original interval bound is available to every row. -/
theorem run_eq (inputColumn : Nat → Nat) (localStart localCount : Nat)
    (values : Array F) (rows : List CompactTemplateRow)
    (fits : localStart + localCount ≤ values.size)
    (localBounds : ∀ row ∈ rows, ∀ index,
      row.outputLocal = some index → index < localCount) :
    (run inputColumn localStart localCount values rows).map asEnv =
      CompactRowExecution.run inputColumn localStart (asEnv values) rows := by
  induction rows generalizing values with
  | nil => rfl
  | cons row rest inductionHypothesis =>
      have rowBound : ∀ index, row.outputLocal = some index → index < localCount :=
        localBounds row (by simp)
      have restBounds : ∀ item ∈ rest, ∀ index,
          item.outputLocal = some index → index < localCount := by
        intro item member
        exact localBounds item (by simp [member])
      have firstEq := step_eq inputColumn localStart localCount values row fits rowBound
      cases first : step inputColumn localStart localCount values row with
      | none =>
          have pureNone :
              CompactRowExecution.step inputColumn localStart (asEnv values) row = none := by
            simpa only [first, Option.map_none] using firstEq.symm
          simp [run, CompactRowExecution.run, first, pureNone]
      | some after =>
          have sameSize := step_size inputColumn localStart localCount values row after first
          have afterFits : localStart + localCount ≤ after.size := by
            rw [sameSize]
            exact fits
          have pureSome :
              CompactRowExecution.step inputColumn localStart (asEnv values) row =
                some (asEnv after) := by
            simpa only [first, Option.map_some] using firstEq.symm
          simpa only [run, CompactRowExecution.run, first, pureSome, Option.bind_some] using
            inductionHypothesis after afterFits restBounds

/-- The guarded stored executor refines the existing pure executor exactly.
No input-read bound or template-satisfaction premise is needed. Local-output
bounds are required because the pure core intentionally has no local-count guard. -/
theorem execute_eq (inputColumn : Nat → Nat) (localStart : Nat)
    (template : CompactRowTemplate) (values : Array F)
    (outputFits : inputColumn template.outputInput < values.size)
    (localFits : localStart + template.localColumnCount ≤ values.size)
    (localBounds : ∀ row ∈ template.rows, ∀ index,
      row.outputLocal = some index → index < template.localColumnCount) :
    (execute inputColumn localStart template values).map asEnv =
      CompactRowExecution.execute inputColumn localStart template (asEnv values) := by
  let output := template.outputRecipe.eval (fun input => asEnv values (inputColumn input))
  let seeded := write values (inputColumn template.outputInput) output
  have seededFits : localStart + template.localColumnCount ≤ seeded.size := by
    simpa only [seeded, write_size] using localFits
  have seededEnv : asEnv seeded =
      Env.set (asEnv values) (inputColumn template.outputInput) output :=
    asEnv_write values (inputColumn template.outputInput) output outputFits
  change (if inputColumn template.outputInput < values.size ∧
      localStart + template.localColumnCount ≤ values.size then
      run inputColumn localStart template.localColumnCount seeded template.rows else none).map asEnv = _
  rw [if_pos ⟨outputFits, localFits⟩,
    run_eq inputColumn localStart template.localColumnCount seeded template.rows seededFits localBounds,
    seededEnv]
  rfl

end NightstreamFPrime.Export.Stage1.StoredCompactRowExecution
