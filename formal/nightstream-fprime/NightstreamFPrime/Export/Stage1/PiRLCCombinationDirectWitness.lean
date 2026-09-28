import NightstreamFPrime.Export.Stage1.CompactOutputExecution
import NightstreamFPrime.Export.Stage1.CompactRowRelocation
import NightstreamFPrime.Export.Stage1.PiRLCCompactRecipeScope
import NightstreamFPrime.Export.Stage1.PiRLCCombinationScratchGeometry

/-!
Owns direct PiRLC combination output writes and reconstruction of their R1CS
scratch. The output recipe is unchanged. Full execution and the direct write
agree outside the invocation's scratch interval. Consumer custody is separate.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiRLCCombinationDirectWitness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open PiRLCCombinationTemplates

def constraint (firstSource : Bool) (lane : Fin ringDegree) : Expr :=
  Expr.var outputInput - outputRecipe firstSource lane

def scratchCount (firstSource : Bool) (lane : Fin ringDegree) : Nat :=
  R1CS.mulCount (constraint firstSource lane)

/-- Read the original input snapshot and write only the required output. -/
def directOutput (inputColumn : Nat → Nat) (firstSource : Bool)
    (lane : Fin ringDegree) (env : Env) : Env :=
  Env.set env (inputColumn outputInput)
    ((outputRecipe firstSource lane).eval (fun input => env (inputColumn input)))

/-- Reconstruct only the local rows, starting from an already written output. -/
def extendScratch (inputColumn : Nat → Nat) (localStart : Nat)
    (firstSource : Bool) (lane : Fin ringDegree) (env : Env) : Option Env :=
  CompactRowExecution.run inputColumn localStart env (template firstSource lane).rows

/-- Erasure does not change addresses or values outside the selected interval. -/
def eraseScratch (localStart count : Nat) (env : Env) : Env :=
  fun column => if localStart ≤ column ∧ column < localStart + count then 0 else env column

theorem directOutput_recurrence (inputColumn : Nat → Nat) (firstSource : Bool)
    (lane : Fin ringDegree) (env : Env) :
    directOutput inputColumn firstSource lane env (inputColumn outputInput) =
      (if firstSource then 0 else env (inputColumn priorInput)) +
        ringFMul
          (fun current => env (inputColumn (challengeInputStart + current.val)) - 2)
          (fun current => env (inputColumn (valueInputStart + current.val))) lane := by
  unfold directOutput
  rw [Env.set_self]
  unfold outputRecipe
  rw [Expr.eval_hadd, CombinationStep.mulExpr_eval]
  have challengeEq :
      CombinationStep.evalRing (fun input => env (inputColumn input)) challenge =
        fun current => env (inputColumn (challengeInputStart + current.val)) - 2 := by
    funext current
    simp only [CombinationStep.evalRing, challenge, Expr.eval_sub, Expr.eval]
    rfl
  rw [challengeEq]
  cases firstSource <;> rfl

theorem directOutput_agreesOutside (inputColumn : Nat → Nat)
    (firstSource : Bool) (lane : Fin ringDegree) (env : Env) :
    AgreesOutside env (directOutput inputColumn firstSource lane env)
      (inputColumn outputInput) 1 := by
  intro column outside
  apply Env.set_of_ne
  rcases outside with before | beyond <;> omega

/-- The existing full executor is exactly the direct write followed by scratch reconstruction. -/
theorem fullExecute_eq_extendScratch (inputColumn : Nat → Nat) (localStart : Nat)
    (firstSource : Bool) (lane : Fin ringDegree) (env : Env) :
    CompactRowExecution.execute inputColumn localStart (template firstSource lane) env =
      extendScratch inputColumn localStart firstSource lane
        (directOutput inputColumn firstSource lane env) := by
  rfl

private theorem rowLocalBound (firstSource : Bool) (lane : Fin ringDegree)
    (row : CompactTemplateRow) (member : row ∈ (template firstSource lane).rows)
    (index : Nat) (found : row.outputLocal = some index) :
    index < scratchCount firstSource lane := by
  change row ∈ ((R1CS.lowerGenericConstraint (constraint firstSource lane)
    inputCount).rows.map (CompactRows.abstractRow inputCount)) at member
  rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
  apply CompactRowRelocation.outputLocal_bound inputCount
    (scratchCount firstSource lane) source.c _ index found
  exact (R1CS.lowerGenericConstraint_rows_varsBelow
    (constraint firstSource lane) inputCount
    (constraint_varsBelow firstSource lane) source sourceMember).2.2

private theorem step_agreesOutside (inputColumn : Nat → Nat) (localStart count : Nat)
    (row : CompactTemplateRow) (env after : Env)
    (localBound : ∀ index, row.outputLocal = some index → index < count)
    (success : CompactRowExecution.step inputColumn localStart env row = some after) :
    AgreesOutside env after localStart count := by
  unfold CompactRowExecution.step at success
  cases selected : row.outputLocal with
  | none =>
      simp only [selected] at success
      split_ifs at success with checked
      · cases Option.some.inj success
        intro column _
        rfl
  | some index =>
      simp only [selected] at success
      split_ifs at success with checked
      · have bounded := localBound index selected
        cases Option.some.inj success
        intro column outside
        apply Env.set_of_ne
        rcases outside with before | beyond <;> omega

private theorem run_agreesOutside (inputColumn : Nat → Nat) (localStart count : Nat)
    (rows : List CompactTemplateRow) (env after : Env)
    (localBounds : ∀ row ∈ rows, ∀ index,
      row.outputLocal = some index → index < count)
    (success : CompactRowExecution.run inputColumn localStart env rows = some after) :
    AgreesOutside env after localStart count := by
  induction rows generalizing env with
  | nil =>
      cases Option.some.inj success
      intro column _
      rfl
  | cons row rest inductionHypothesis =>
      cases first : CompactRowExecution.step inputColumn localStart env row with
      | none => simp [CompactRowExecution.run, first] at success
      | some middle =>
          have tail : CompactRowExecution.run inputColumn localStart middle rest =
              some after := by
            simpa only [CompactRowExecution.run, first, Option.bind_some] using success
          have firstAgrees := step_agreesOutside inputColumn localStart count row env middle
            (localBounds row (by simp)) first
          have restAgrees := inductionHypothesis middle
            (fun item member => localBounds item (by simp [member])) tail
          intro column outside
          exact (restAgrees column outside).trans (firstAgrees column outside)

/-- This physical-coordinate statement needs no equality inside scratch. -/
theorem fullExecute_agrees_directOutput (inputColumn : Nat → Nat) (localStart : Nat)
    (firstSource : Bool) (lane : Fin ringDegree) (env after : Env)
    (success : CompactRowExecution.execute inputColumn localStart
      (template firstSource lane) env = some after) :
    AgreesOutside (directOutput inputColumn firstSource lane env) after
      localStart (scratchCount firstSource lane) := by
  rw [fullExecute_eq_extendScratch] at success
  exact run_agreesOutside inputColumn localStart (scratchCount firstSource lane)
    (template firstSource lane).rows _ after (rowLocalBound firstSource lane) success

theorem eraseScratch_agreesOutside (localStart count : Nat) (env : Env) :
    AgreesOutside env (eraseScratch localStart count env) localStart count := by
  intro column outside
  unfold eraseScratch
  rw [if_neg]
  rcases outside with before | beyond <;> omega

/-- Erasure is the same whether the local rows have been executed or omitted. -/
theorem erase_fullExecute (inputColumn : Nat → Nat) (localStart : Nat)
    (firstSource : Bool) (lane : Fin ringDegree) (env after : Env)
    (success : CompactRowExecution.execute inputColumn localStart
      (template firstSource lane) env = some after) :
    eraseScratch localStart (scratchCount firstSource lane) after =
      eraseScratch localStart (scratchCount firstSource lane)
        (directOutput inputColumn firstSource lane env) := by
  have agrees := fullExecute_agrees_directOutput inputColumn localStart
    firstSource lane env after success
  funext column
  unfold eraseScratch
  split_ifs with inside
  · rfl
  · exact agrees column (by omega)

/-- Normalized execution succeeds for every input snapshot. All layout and
recipe-scope premises are discharged by the canonical PiRLC template. -/
theorem normalized_fullExecute (firstSource : Bool) (lane : Fin ringDegree) (env : Env) :
    CompactRowExecution.execute id inputCount (template firstSource lane) env =
      some (R1CS.executeExpression (directOutput id firstSource lane env)
        (constraint firstSource lane) inputCount) := by
  exact CompactOutputExecution.execute_compactTemplate inputCount outputInput
    (outputRecipe firstSource lane) env (by decide)
    (PiRLCCompactRecipeScope.combination_outputRecipe firstSource lane)

/-- The executable extension recovers a full normalized witness from the
direct output. It does not require an input-validity or row-satisfaction premise. -/
theorem extendScratch_directOutput (firstSource : Bool) (lane : Fin ringDegree) (env : Env) :
    extendScratch id inputCount firstSource lane (directOutput id firstSource lane env) =
      some (R1CS.executeExpression (directOutput id firstSource lane env)
        (constraint firstSource lane) inputCount) := by
  rw [← fullExecute_eq_extendScratch]
  exact normalized_fullExecute firstSource lane env

theorem directOutput_constraint (firstSource : Bool) (lane : Fin ringDegree) (env : Env) :
    (constraint firstSource lane).eval (directOutput id firstSource lane env) = 0 := by
  exact CompactOutputExecution.logical_after_output env outputInput
    (outputRecipe firstSource lane)
    (PiRLCCompactRecipeScope.combination_outputRecipe firstSource lane)

/-- Canonical relocation preserves success. These are only input alias and
interval conditions; there is no premise about field values or satisfied rows. -/
theorem relocated_fullExecute_succeeds (inputColumn : Nat → Nat) (localStart : Nat)
    (firstSource : Bool) (lane : Fin ringDegree) (env : Env)
    (localBound : inputCount ≤ localStart)
    (inputsOutside : ∀ input, input < inputCount →
      inputColumn input < localStart ∨
        localStart + scratchCount firstSource lane ≤ inputColumn input)
    (outputDistinct : ∀ input, input < outputInput →
      inputColumn input ≠ inputColumn outputInput) :
    ∃ after, CompactRowExecution.execute inputColumn localStart
      (template firstSource lane) env = some after := by
  let seeded := directOutput inputColumn firstSource lane env
  let pulled := CompactRowRelocation.pullback inputCount localStart inputColumn seeded
  let inputEnv : Env := fun input => env (inputColumn input)
  let value := (outputRecipe firstSource lane).eval inputEnv
  let normalized := Env.set inputEnv outputInput value
  have scope := constraint_varsBelow firstSource lane
  have inputEq : ∀ input, input < inputCount → pulled input = normalized input := by
    intro input below
    change seeded (CompactRows.relocate inputCount (localStart - inputCount)
      inputColumn input) = normalized input
    rw [CompactRows.relocate_input inputCount (localStart - inputCount)
      inputColumn input below]
    by_cases same : input = outputInput
    · subst input
      simp only [seeded, directOutput, normalized, Env.set_self]
      rfl
    · have earlier : input < outputInput := by
        norm_num only [inputCount, outputInput] at below same ⊢
        omega
      simp only [seeded, directOutput, normalized, Env.set, same,
        outputDistinct input earlier, if_false, inputEnv]
  have logical : (constraint firstSource lane).eval pulled = 0 := by
    rw [(constraint firstSource lane).eval_eq_of_agree_below
      inputCount pulled normalized scope inputEq]
    exact CompactOutputExecution.logical_after_output inputEnv outputInput
      (outputRecipe firstSource lane)
      (PiRLCCompactRecipeScope.combination_outputRecipe firstSource lane)
  have relocated :
      (CompactRowExecution.execute inputColumn localStart
        (template firstSource lane) env).map
          (CompactRowRelocation.pullback inputCount localStart inputColumn) =
        some (R1CS.executeExpression pulled (constraint firstSource lane) inputCount) := by
    change (CompactRowExecution.run inputColumn localStart seeded
      ((R1CS.lowerGenericConstraint (constraint firstSource lane) inputCount).rows.map
        (CompactRows.abstractRow inputCount))).map _ = _
    rw [CompactRowRelocation.run_relocate inputCount localStart
      (scratchCount firstSource lane) inputColumn seeded
      (R1CS.lowerGenericConstraint (constraint firstSource lane) inputCount).rows
      localBound inputsOutside
      (fun row member => (R1CS.lowerGenericConstraint_rows_varsBelow
        (constraint firstSource lane) inputCount scope row member).2.2)]
    exact CompactRowExecution.run_lowerGenericConstraint inputCount pulled
      (constraint firstSource lane) inputCount (Nat.le_refl _) scope logical
  cases result : CompactRowExecution.execute inputColumn localStart
      (template firstSource lane) env with
  | none => simp only [result, Option.map_none] at relocated; contradiction
  | some after => exact ⟨after, rfl⟩

open PiRLCCombinationScratchGeometry (scratchStart scratchEnd)

/-- The canonical invocation writes only its required accumulator value. -/
def directInvocation (descriptor : PiRLCProductSchedule.Descriptor) (env : Env) : Env :=
  directOutput (PiRLCCombinationScratchGeometry.inputColumn descriptor)
    (PiRLCCombinationInvocations.firstSource descriptor.source.val) descriptor.lane env

/-- Reference execution keeps the same canonical invocation and its local rows. -/
def fullInvocation (descriptor : PiRLCProductSchedule.Descriptor) (env : Env) : Option Env :=
  CompactRowExecution.execute (PiRLCCombinationScratchGeometry.inputColumn descriptor)
    descriptor.compactInvocation.localStart
    (template (PiRLCCombinationInvocations.firstSource descriptor.source.val)
      descriptor.lane) env

theorem fullInvocation_succeeds (descriptor : PiRLCProductSchedule.Descriptor) (env : Env) :
    ∃ after, fullInvocation descriptor env = some after := by
  exact relocated_fullExecute_succeeds _ _ _ _ env
    (PiRLCCombinationScratchGeometry.localStart_ge descriptor)
    (PiRLCCombinationScratchGeometry.inputs_outside_local descriptor)
    (PiRLCCombinationScratchGeometry.output_distinct descriptor)

theorem fullInvocation_agrees_directInvocation
    (descriptor : PiRLCProductSchedule.Descriptor) (env after : Env)
    (success : fullInvocation descriptor env = some after) :
    ∀ column, column < scratchStart ∨ scratchEnd ≤ column →
      after column = directInvocation descriptor env column := by
  have agrees := fullExecute_agrees_directOutput
    (PiRLCCombinationScratchGeometry.inputColumn descriptor)
    descriptor.compactInvocation.localStart
    (PiRLCCombinationInvocations.firstSource descriptor.source.val)
    descriptor.lane env after success
  have contained := PiRLCCombinationScratchGeometry.scratch_contained descriptor
  intro column outside
  apply agrees column
  change scratchStart ≤ descriptor.compactInvocation.localStart ∧
    descriptor.compactInvocation.localStart +
      scratchCount (PiRLCCombinationInvocations.firstSource descriptor.source.val)
        descriptor.lane ≤ scratchEnd at contained
  rcases outside with before | beyond
  · exact Or.inl (Nat.lt_of_lt_of_le before contained.1)
  · exact Or.inr (Nat.le_trans contained.2 beyond)

private theorem directInvocation_congr
    (descriptor : PiRLCProductSchedule.Descriptor) (direct full : Env)
    (agree : ∀ column, column < scratchStart ∨ scratchEnd ≤ column →
      full column = direct column) :
    ∀ column, column < scratchStart ∨ scratchEnd ≤ column →
      directInvocation descriptor full column = directInvocation descriptor direct column := by
  let inputs := PiRLCCombinationScratchGeometry.inputColumn descriptor
  let first := PiRLCCombinationInvocations.firstSource descriptor.source.val
  have sameValue : (outputRecipe first descriptor.lane).eval
      (fun input => full (inputs input)) =
        (outputRecipe first descriptor.lane).eval (fun input => direct (inputs input)) := by
    apply Expr.eval_eq_of_agree_below _ outputInput _ _
      (PiRLCCompactRecipeScope.combination_outputRecipe first descriptor.lane)
    intro input before
    exact agree (inputs input)
      (PiRLCCombinationScratchGeometry.inputs_outside_scratch descriptor input
        (Nat.lt_trans before (by decide : outputInput < inputCount)))
  intro column outside
  change Env.set full (inputs outputInput) _ column =
    Env.set direct (inputs outputInput) _ column
  simp only [Env.set]
  split_ifs
  · exact sameValue
  · exact agree column outside

/-- This fold evaluates only existing output recipes. It does not construct
an invocation list or evaluate any R1CS scratch row. -/
def directPrefix (count : Nat) (bounded : count ≤ PiRLCProductSchedule.invocationCount)
    (env : Env) : Env :=
  Nat.fold count (fun index inside current =>
    directInvocation
      (PiRLCProductSchedule.descriptor ⟨index, Nat.lt_of_lt_of_le inside bounded⟩) current) env

def fullPrefix (count : Nat) (bounded : count ≤ PiRLCProductSchedule.invocationCount)
    (env : Env) : Option Env :=
  Nat.fold count (fun index inside current =>
    current.bind (fullInvocation
      (PiRLCProductSchedule.descriptor ⟨index, Nat.lt_of_lt_of_le inside bounded⟩))) (some env)

/-- Prefix induction uses the symbolic invocation count. No field validity,
full-execution success, or scratch equality is supplied by the caller. -/
theorem fullPrefix_agrees_directPrefix (count : Nat)
    (bounded : count ≤ PiRLCProductSchedule.invocationCount) (env : Env) :
    ∃ after, fullPrefix count bounded env = some after ∧
      ∀ column, column < scratchStart ∨ scratchEnd ≤ column →
        after column = directPrefix count bounded env column := by
  induction count with
  | zero => exact ⟨env, rfl, fun _ _ => rfl⟩
  | succ count inductionHypothesis =>
      have previousBound : count ≤ PiRLCProductSchedule.invocationCount := by omega
      rcases inductionHypothesis previousBound with ⟨previous, previousEq, previousAgrees⟩
      let descriptor := PiRLCProductSchedule.descriptor
        ⟨count, Nat.lt_of_lt_of_le (Nat.lt_succ_self count) bounded⟩
      rcases fullInvocation_succeeds descriptor previous with ⟨after, afterEq⟩
      refine ⟨after, ?_, ?_⟩
      · change (fullPrefix count previousBound env).bind
          (fullInvocation descriptor) = some after
        rw [previousEq, Option.bind_some]
        exact afterEq
      · intro column outside
        change after column =
          directInvocation descriptor (directPrefix count previousBound env) column
        exact (fullInvocation_agrees_directInvocation descriptor previous after afterEq
          column outside).trans
            (directInvocation_congr descriptor (directPrefix count previousBound env)
              previous previousAgrees column outside)

def directPhase (env : Env) : Env :=
  directPrefix PiRLCProductSchedule.invocationCount (Nat.le_refl _) env

def fullPhase (env : Env) : Option Env :=
  fullPrefix PiRLCProductSchedule.invocationCount (Nat.le_refl _) env

/-- All 52,326 canonical invocations preserve every non-scratch value when
the direct constructor replaces their full row execution. -/
theorem fullPhase_agrees_directPhase (env : Env) :
    ∃ after, fullPhase env = some after ∧
      ∀ column, column < scratchStart ∨ scratchEnd ≤ column →
        after column = directPhase env column := by
  exact fullPrefix_agrees_directPrefix PiRLCProductSchedule.invocationCount
    (Nat.le_refl _) env

end NightstreamFPrime.Export.Stage1.PiRLCCombinationDirectWitness
