import NightstreamFPrime.Export.PilotData
import NightstreamFPrime.Export.RowSemantics
import NightstreamFPrime.Layout.PilotSpartan
import NightstreamFPrime.Layout.Poseidon2

/-!
Owns the proofs that connect the executable pilot package in `PilotData` to
the production lifecycle layout. It owns no second package, row list, or
schedule.
-/

namespace NightstreamFPrime.Export.Pilot

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper

private theorem stateHashWords_match :
    PilotValues.stateHashWords = PilotProduction.stateHashWords := by
  rfl

private theorem firstPublicStart_match :
    PilotValues.firstPublicStart = PilotSpartan.firstPublicStart := by
  rfl

private theorem fieldValue_val (value : F) : fieldValue value.val = value := by
  apply Fin.ext
  simp [fieldValue, Spec.Poseidon2.ofNat, Nat.mod_eq_of_lt value.isLt]

@[simp] private theorem fieldValue_zero : fieldValue 0 = (0 : F) := rfl

@[simp] private theorem fieldValue_one : fieldValue 1 = (1 : F) := rfl

@[simp] private theorem fieldValue_neg_one :
    fieldValue (-1 : F).val = (-1 : F) :=
  fieldValue_val (-1 : F)

private def canonicalTemplateEnv (value : ColumnRef → F) : Env :=
  fun column => (PilotData.columnRef column).eval value

private theorem canonicalOutputLocalIndex (lane : Fin 8) :
    592 + lane.val - 8 = 584 + lane.val := by
  omega

private theorem templateCombination_eval
    (combination : R1CS.LinearCombination) (value : ColumnRef → F) :
    (PilotData.templateCombination combination).eval value =
      combination.eval (canonicalTemplateEnv value) := by
  unfold PilotData.templateCombination TemplateCombination.eval
  rw [fieldValue_val]
  unfold R1CS.LinearCombination.eval
  congr 1
  rw [List.map_map]
  apply congrArg List.sum
  apply List.map_congr_left
  intro term member
  simp [PilotData.templateTerm, canonicalTemplateEnv,
    fieldValue_val]

private theorem templateRow_holds (output : Nat) (row : R1CS.Row)
    (value : ColumnRef → F) :
    (⟨output, PilotData.templateCombination row.a,
        PilotData.templateCombination row.b,
        PilotData.templateCombination row.c⟩ : TemplateRow).Holds value ↔
      row.Holds (canonicalTemplateEnv value) := by
  unfold TemplateRow.Holds R1CS.Row.Holds
  rw [templateCombination_eval, templateCombination_eval,
    templateCombination_eval]

private theorem templateRowsFrom_hold (output : Nat) (rows : List R1CS.Row)
    (value : ColumnRef → F) :
    (∀ row ∈ PilotData.templateRowsFrom output rows, row.Holds value) ↔
      R1CS.RowsHold (canonicalTemplateEnv value) rows := by
  induction rows generalizing output with
  | nil => simp [PilotData.templateRowsFrom, R1CS.RowsHold]
  | cons row rest ih =>
      constructor
      · intro holds
        intro current member
        rcases List.mem_cons.mp member with equals | member
        · subst current
          exact (templateRow_holds output row value).mp
            (holds _ (by simp [PilotData.templateRowsFrom]))
        · exact (ih (output + 1)).mp (by
            intro template templateMember
            exact holds template (by
              simp [PilotData.templateRowsFrom, templateMember]))
            current member
      · intro holds template member
        simp only [PilotData.templateRowsFrom, List.mem_cons] at member
        rcases member with equals | member
        · subst template
          exact (templateRow_holds output row value).mpr
            (holds row (by simp))
        · exact (ih (output + 1)).mpr
            (fun current currentMember => holds current (by
              simp [currentMember])) template member

/-- The emitted 592-row template has exactly the compiler-produced canonical
permutation semantics at every package invocation. -/
theorem canonicalTemplateInvocation_iff (chain : HashChain)
    (invocation : Nat) (env : Env) :
    TemplateInvocationHolds (PilotData.circuitPackage ()) chain invocation env ↔
      R1CS.RowsHold
        (canonicalTemplateEnv (fun column =>
          (instantiateColumn (PilotData.circuitPackage ()) chain invocation
            column).eval env))
        (PilotData.canonicalRows ()) := by
  let value : ColumnRef → F := fun column =>
    (instantiateColumn (PilotData.circuitPackage ()) chain invocation
      column).eval env
  constructor
  · intro holds
    apply (templateRowsFrom_hold 0 (PilotData.canonicalRows ()) value).mp
    intro row member
    exact (instantiateRow_holds (PilotData.circuitPackage ()) chain invocation
      row env).mp (holds row (by
        simpa [PilotData.circuitPackage, PilotData.permutationTemplate,
          PilotData.templateRows] using member))
  · intro holds row member
    apply (instantiateRow_holds (PilotData.circuitPackage ()) chain invocation
      row env).mpr
    apply (templateRowsFrom_hold 0 (PilotData.canonicalRows ()) value).mpr holds
    simpa [PilotData.circuitPackage, PilotData.permutationTemplate,
      PilotData.templateRows] using member

/-- Every satisfying emitted template invocation is the exact proved
Poseidon2 permutation between its package-defined inputs and local outputs. -/
theorem canonicalTemplateInvocation_sound (chain : HashChain)
    (invocation : Nat) (env : Env)
    (holds : TemplateInvocationHolds (PilotData.circuitPackage ()) chain
      invocation env) :
    (fun lane : Fin 8 => env
      (invocationLocalStart (PilotData.circuitPackage ()) chain invocation +
        (PilotData.circuitPackage ()).permutation.outputLocalStart +
        lane.val)) =
      Permutation.runF Permutation.schedule (fun lane : Fin 8 =>
        (invocationInput (PilotData.circuitPackage ()) chain invocation
          lane.val).eval env) := by
  let value : ColumnRef → F := fun column =>
    (instantiateColumn (PilotData.circuitPackage ()) chain invocation
      column).eval env
  let canonicalEnv := canonicalTemplateEnv value
  have physical : R1CS.RowsHold canonicalEnv
      (PilotData.canonicalRows ()) :=
    (canonicalTemplateInvocation_iff chain invocation env).mp holds
  have logical : ConstraintsHold canonicalEnv
      (PilotData.canonicalConstraints ()) :=
    R1CS.lowerConstraints_sound canonicalEnv
      (PilotData.canonicalConstraints ()) 600 physical
  have sound := Permutation.compile_sound canonicalEnv 8
    PilotData.canonicalState Permutation.schedule logical
  rw [Poseidon2.compile_schedule_output_eq] at sound
  have outputBoundary :
      Layer.evalState canonicalEnv (Permutation.freshState (8 + 584)) =
        (fun lane : Fin 8 => env
          (invocationLocalStart (PilotData.circuitPackage ()) chain invocation +
            (PilotData.circuitPackage ()).permutation.outputLocalStart +
            lane.val)) := by
    funext lane
    have refEq : PilotData.columnRef (592 + lane.val) =
        .local (584 + lane.val) := by
      unfold PilotData.columnRef
      rw [if_neg (by omega)]
      rw [canonicalOutputLocalIndex lane]
    change (PilotData.columnRef (592 + lane.val)).eval value = _
    rw [refEq]
    simp [ColumnRef.eval, value, instantiateColumn, invocationLocalStart,
      PilotData.circuitPackage, PilotData.permutationTemplate, Nat.add_assoc]
  have inputBoundary :
      Layer.evalState canonicalEnv PilotData.canonicalState =
        (fun lane : Fin 8 =>
          (invocationInput (PilotData.circuitPackage ()) chain invocation
            lane.val).eval env) := by
    funext lane
    have refEq : PilotData.columnRef lane.val = .input lane.val := by
      unfold PilotData.columnRef
      rw [if_pos lane.isLt]
    change (PilotData.columnRef lane.val).eval value = _
    rw [refEq]
    rfl
  exact outputBoundary.symm.trans
    (sound.trans (congrArg (Permutation.runF Permutation.schedule)
      inputBoundary))

private theorem canonicalPermutationInvocation_iff
    (invocation : PermutationInvocation) (env : Env) :
    PermutationInvocationHolds (PilotData.circuitPackage ()) invocation env ↔
      R1CS.RowsHold
        (canonicalTemplateEnv (fun column =>
          (instantiateInvocationColumn invocation column).eval env))
        (PilotData.canonicalRows ()) := by
  let value : ColumnRef → F := fun column =>
    (instantiateInvocationColumn invocation column).eval env
  constructor
  · intro holds
    apply (templateRowsFrom_hold 0 (PilotData.canonicalRows ()) value).mp
    intro row member
    exact (instantiateInvocationRow_holds invocation row env).mp
      (holds row (by
        simpa [PilotData.circuitPackage, PilotData.permutationTemplate,
          PilotData.templateRows] using member))
  · intro holds row member
    apply (instantiateInvocationRow_holds invocation row env).mpr
    apply (templateRowsFrom_hold 0 (PilotData.canonicalRows ()) value).mpr holds
    simpa [PilotData.circuitPackage, PilotData.permutationTemplate,
      PilotData.templateRows] using member

/-- The local environment selected by one explicit package invocation.
Columns `0..7` are the invocation inputs. Columns `8..599` are its 592
canonical Poseidon2 witness values. -/
def canonicalInvocationEnv (invocation : PermutationInvocation)
    (env : Env) : Env :=
  canonicalTemplateEnv fun column =>
    (instantiateInvocationColumn invocation column).eval env

/-- Constructive completeness of one explicit canonical Poseidon2
invocation. The premise is the fixed-size logical permutation constraint
list under the invocation's exact column map. -/
theorem canonicalPermutationInvocation_complete
    (invocation : PermutationInvocation) (env : Env)
    (logical : ConstraintsHold (canonicalInvocationEnv invocation env)
      (PilotData.canonicalConstraints ())) :
    PermutationInvocationHolds
      (PilotData.circuitPackage ()) invocation env := by
  apply (canonicalPermutationInvocation_iff invocation env).mpr
  change R1CS.RowsHold (canonicalInvocationEnv invocation env)
    (PilotData.canonicalRows ())
  unfold PilotData.canonicalRows
  apply R1CS.lowerConstraints_complete_of_noFresh
  · apply R1CS.recipeConstraints_noFresh
    apply NightstreamFPrime.Layout.Poseidon2.compile_schedule_direct
    intro lane
    exact R1CS.isAffine_var _
  · exact logical

private def canonicalInvocationInputEnv
    (invocation : PermutationInvocation) (env : Env) : Env :=
  fun column =>
    if column < 8 then
      (invocationInputCombination invocation column).toR1CS.eval env
    else
      0

private def canonicalInvocationLocalCompleted
    (invocation : PermutationInvocation) (env : Env) : Env :=
  executeRecipes (canonicalInvocationInputEnv invocation env) 8
    (PilotData.canonicalRecipes ())

private def canonicalInvocationWitnessRecipes
    (invocation : PermutationInvocation) (env : Env) : List Expr :=
  List.ofFn fun index : Fin 592 =>
    Expr.const (canonicalInvocationLocalCompleted invocation env
      (8 + index.val))

def completePermutationInvocationEnv
    (invocation : PermutationInvocation) (env : Env) : Env :=
  executeRecipes env invocation.witnessStart
    (canonicalInvocationWitnessRecipes invocation env)

private theorem executeConstantRecipes_at
    {count : Nat} (values : Fin count → F) (env : Env) (start : Nat)
    (index : Fin count) :
    executeRecipes env start
        (List.ofFn fun current : Fin count => Expr.const (values current))
        (start + index.val) =
      values index := by
  induction count generalizing env start with
  | zero => exact Fin.elim0 index
  | succ count inductionHypothesis =>
      refine Fin.cases ?_ (fun tail => ?_) index
      · rw [List.ofFn_succ, executeRecipes]
        have agrees := executeRecipes_agrees_below
          (Env.set env start (values 0)) (start + 1)
          (List.ofFn fun current : Fin count =>
            Expr.const (values current.succ))
          start (by omega)
        simpa using agrees.trans (Env.set_self env start (values 0))
      · rw [List.ofFn_succ, executeRecipes]
        have address : start + (Fin.succ tail).val =
            (start + 1) + tail.val := by
          rw [Fin.val_succ]
          omega
        rw [address]
        exact inductionHypothesis (fun current => values current.succ)
          (Env.set env start (values 0)) (start + 1) tail

private theorem linearCombination_eval_eq_of_termsBelow
    (combination : R1CS.LinearCombination) (before after : Env)
    (bound : Nat)
    (termsBelow : ∀ term ∈ combination.terms, term.1 < bound)
    (agrees : ∀ index, index < bound → after index = before index) :
    combination.eval after = combination.eval before := by
  unfold R1CS.LinearCombination.eval
  congr 1
  apply congrArg List.sum
  apply List.map_congr_left
  intro term member
  rw [agrees term.1 (termsBelow term member)]

private theorem linearCombination_eval_eq_of_termAgreement
    (combination : R1CS.LinearCombination) (before after : Env)
    (agrees : ∀ term ∈ combination.terms,
      after term.1 = before term.1) :
    combination.eval after = combination.eval before := by
  unfold R1CS.LinearCombination.eval
  congr 1
  apply congrArg List.sum
  apply List.map_congr_left
  intro term member
  rw [agrees term member]

private theorem canonicalInvocationLocalCompleted_holds
    (invocation : PermutationInvocation) (env : Env) :
    ConstraintsHold (canonicalInvocationLocalCompleted invocation env)
      (PilotData.canonicalConstraints ()) := by
  apply executeRecipes_holds_recipeConstraints
  apply Permutation.compile_schedule_causal
  intro lane
  exact lane.isLt

/-- Honest execution of one explicit invocation fills exactly its 592 local
Poseidon2 witness values and satisfies every emitted template row. Invocation
inputs must come from columns outside the local witness interval. -/
theorem completePermutationInvocation
    (invocation : PermutationInvocation) (env : Env)
    (inputsOutside : ∀ lane : Fin 8,
      ∀ term ∈ (invocationInputCombination invocation lane.val).toR1CS.terms,
        term.1 < invocation.witnessStart ∨
          invocation.witnessStart + 592 ≤ term.1) :
    AgreesOutside env (completePermutationInvocationEnv invocation env)
        invocation.witnessStart 592 ∧
      PermutationInvocationHolds (PilotData.circuitPackage ()) invocation
        (completePermutationInvocationEnv invocation env) := by
  let completed := completePermutationInvocationEnv invocation env
  let localEnv := canonicalInvocationLocalCompleted invocation env
  have completedOutside : AgreesOutside env completed
      invocation.witnessStart 592 := by
    have agrees := executeRecipes_agreesOutside env invocation.witnessStart
      (canonicalInvocationWitnessRecipes invocation env)
    have recipeLength :
        (canonicalInvocationWitnessRecipes invocation env).length = 592 := by
      unfold canonicalInvocationWitnessRecipes
      exact List.length_ofFn
    rw [recipeLength] at agrees
    exact agrees
  have localAgrees : ∀ index, index < 8 →
      localEnv index = canonicalInvocationInputEnv invocation env index := by
    intro index below
    exact executeRecipes_agrees_below
      (canonicalInvocationInputEnv invocation env) 8
      (PilotData.canonicalRecipes ()) index below
  have mappedAgrees : ∀ column, column < 600 →
      canonicalInvocationEnv invocation completed column = localEnv column := by
    intro column below
    by_cases input : column < 8
    · have inputStable :
        (invocationInputCombination invocation column).toR1CS.eval completed =
          (invocationInputCombination invocation column).toR1CS.eval env := by
        apply linearCombination_eval_eq_of_termAgreement
        intro term member
        exact completedOutside term.1
          (inputsOutside ⟨column, input⟩ term member)
      rw [localAgrees column input]
      unfold canonicalInvocationEnv canonicalTemplateEnv
      rw [show PilotData.columnRef column = .input column by
        simp [PilotData.columnRef, input]]
      simp only [ColumnRef.eval, instantiateInvocationColumn]
      unfold canonicalInvocationInputEnv
      rw [if_pos input]
      exact inputStable
    · have localIndex : column - 8 < 592 := by omega
      have written := executeConstantRecipes_at
        (fun index : Fin 592 =>
          canonicalInvocationLocalCompleted invocation env (8 + index.val))
        env invocation.witnessStart ⟨column - 8, localIndex⟩
      unfold canonicalInvocationEnv canonicalTemplateEnv
      rw [show PilotData.columnRef column = .local (column - 8) by
        simp [PilotData.columnRef, input]]
      simp only [ColumnRef.eval, instantiateInvocationColumn,
        R1CS.LinearCombination.eval_ofVar]
      change completed (invocation.witnessStart + (column - 8)) =
        localEnv column
      change completed (invocation.witnessStart + (column - 8)) =
        canonicalInvocationLocalCompleted invocation env
          (8 + (column - 8)) at written
      rw [show 8 + (column - 8) = column by omega] at written
      exact written
  have canonicalScope : ∀ expression ∈ PilotData.canonicalConstraints (),
      expression.VarsBelow 600 := by
    intro expression member
    have scope := recipeConstraints_varsBelow_of_causal 8
      (PilotData.canonicalRecipes ())
      (Permutation.compile_schedule_causal 8 PilotData.canonicalState (by
        intro lane
        exact lane.isLt)) expression member
    have recipeLength : (PilotData.canonicalRecipes ()).length = 592 := by
      exact Permutation.compile_schedule_recipe_count 8
        PilotData.canonicalState
    rw [recipeLength] at scope
    norm_num at scope
    exact scope
  have mappedLogical : ConstraintsHold
      (canonicalInvocationEnv invocation completed)
      (PilotData.canonicalConstraints ()) :=
    constraintsHold_of_agree_below localEnv
      (canonicalInvocationEnv invocation completed)
      (PilotData.canonicalConstraints ()) 600 canonicalScope mappedAgrees
      (canonicalInvocationLocalCompleted_holds invocation env)
  constructor
  · exact completedOutside
  · exact canonicalPermutationInvocation_complete invocation completed
      mappedLogical

/-- An invocation remains satisfied when its eight input values and its exact
592-cell local witness interval are unchanged. -/
theorem permutationInvocationHolds_of_agreement
    (invocation : PermutationInvocation) (before after : Env)
    (inputsAgree : ∀ lane : Fin 8,
      (invocationInputCombination invocation lane.val).toR1CS.eval after =
        (invocationInputCombination invocation lane.val).toR1CS.eval before)
    (localsAgree : ∀ index, index < 592 →
      after (invocation.witnessStart + index) =
        before (invocation.witnessStart + index))
    (holds : PermutationInvocationHolds
      (PilotData.circuitPackage ()) invocation before) :
    PermutationInvocationHolds
      (PilotData.circuitPackage ()) invocation after := by
  have mappedAgrees : ∀ column, column < 600 →
      canonicalInvocationEnv invocation after column =
        canonicalInvocationEnv invocation before column := by
    intro column below
    by_cases input : column < 8
    · unfold canonicalInvocationEnv canonicalTemplateEnv
      rw [show PilotData.columnRef column = .input column by
        simp [PilotData.columnRef, input]]
      simp only [ColumnRef.eval, instantiateInvocationColumn]
      exact inputsAgree ⟨column, input⟩
    · unfold canonicalInvocationEnv canonicalTemplateEnv
      rw [show PilotData.columnRef column = .local (column - 8) by
        simp [PilotData.columnRef, input]]
      simp only [ColumnRef.eval, instantiateInvocationColumn,
        R1CS.LinearCombination.eval_ofVar]
      exact localsAgree (column - 8) (by omega)
  have canonicalScope : ∀ expression ∈ PilotData.canonicalConstraints (),
      expression.VarsBelow 600 := by
    intro expression member
    have scope := recipeConstraints_varsBelow_of_causal 8
      (PilotData.canonicalRecipes ())
      (Permutation.compile_schedule_causal 8 PilotData.canonicalState (by
        intro lane
        exact lane.isLt)) expression member
    have recipeLength : (PilotData.canonicalRecipes ()).length = 592 := by
      exact Permutation.compile_schedule_recipe_count 8
        PilotData.canonicalState
    rw [recipeLength] at scope
    norm_num at scope
    exact scope
  have beforeRows : R1CS.RowsHold
      (canonicalInvocationEnv invocation before)
      (PilotData.canonicalRows ()) :=
    (canonicalPermutationInvocation_iff invocation before).mp holds
  have beforeLogical : ConstraintsHold
      (canonicalInvocationEnv invocation before)
      (PilotData.canonicalConstraints ()) := by
    unfold PilotData.canonicalRows at beforeRows
    exact R1CS.lowerConstraints_sound _ _ 600 beforeRows
  have afterLogical : ConstraintsHold
      (canonicalInvocationEnv invocation after)
      (PilotData.canonicalConstraints ()) :=
    constraintsHold_of_agree_below
      (canonicalInvocationEnv invocation before)
      (canonicalInvocationEnv invocation after)
      (PilotData.canonicalConstraints ()) 600 canonicalScope mappedAgrees
      beforeLogical
  exact canonicalPermutationInvocation_complete invocation after afterLogical

/-- An invocation remains satisfied after a disjoint witness interval changes.
The caller proves separation for every sparse input term and local cell. -/
theorem permutationInvocationHolds_of_agreesOutside
    (invocation : PermutationInvocation) (before after : Env)
    (start length : Nat)
    (inputsOutside : ∀ lane : Fin 8,
      ∀ term ∈ (invocationInputCombination invocation lane.val).toR1CS.terms,
        term.1 < start ∨ start + length ≤ term.1)
    (localsOutside : ∀ index, index < 592 →
      invocation.witnessStart + index < start ∨
        start + length ≤ invocation.witnessStart + index)
    (agrees : AgreesOutside before after start length)
    (holds : PermutationInvocationHolds
      (PilotData.circuitPackage ()) invocation before) :
    PermutationInvocationHolds
      (PilotData.circuitPackage ()) invocation after := by
  apply permutationInvocationHolds_of_agreement invocation before after
  · intro lane
    apply linearCombination_eval_eq_of_termAgreement
    intro term member
    exact agrees term.1 (inputsOutside lane term member)
  · intro index below
    exact agrees (invocation.witnessStart + index)
      (localsOutside index below)
  · exact holds

/-- A completed invocation remains satisfied after a later witness program
changes only columns at or above `bound`. -/
theorem permutationInvocationHolds_of_agree_below
    (invocation : PermutationInvocation) (before after : Env) (bound : Nat)
    (inputsBelow : ∀ lane : Fin 8,
      ∀ term ∈ (invocationInputCombination invocation lane.val).toR1CS.terms,
        term.1 < bound)
    (localsBelow : invocation.witnessStart + 592 ≤ bound)
    (agrees : ∀ index, index < bound → after index = before index)
    (holds : PermutationInvocationHolds
      (PilotData.circuitPackage ()) invocation before) :
    PermutationInvocationHolds
      (PilotData.circuitPackage ()) invocation after := by
  apply permutationInvocationHolds_of_agreement invocation before after
  · intro lane
    apply linearCombination_eval_eq_of_termsBelow
    · exact inputsBelow lane
    · exact agrees
  · intro index below
    apply agrees
    omega
  · exact holds

/-- Every satisfying explicit package invocation is the exact proved
Poseidon2 permutation between its eight sparse inputs and local outputs. -/
theorem canonicalPermutationInvocation_sound
    (invocation : PermutationInvocation) (env : Env)
    (holds : PermutationInvocationHolds
      (PilotData.circuitPackage ()) invocation env) :
    (fun lane : Fin 8 => env
      (invocation.witnessStart +
        (PilotData.circuitPackage ()).permutation.outputLocalStart +
        lane.val)) =
      Permutation.runF Permutation.schedule (fun lane : Fin 8 =>
        (invocationInputCombination invocation lane.val).toR1CS.eval env) := by
  let value : ColumnRef → F := fun column =>
    (instantiateInvocationColumn invocation column).eval env
  let canonicalEnv := canonicalTemplateEnv value
  have physical : R1CS.RowsHold canonicalEnv
      (PilotData.canonicalRows ()) :=
    (canonicalPermutationInvocation_iff invocation env).mp holds
  have logical : ConstraintsHold canonicalEnv
      (PilotData.canonicalConstraints ()) :=
    R1CS.lowerConstraints_sound canonicalEnv
      (PilotData.canonicalConstraints ()) 600 physical
  have sound := Permutation.compile_sound canonicalEnv 8
    PilotData.canonicalState Permutation.schedule logical
  rw [Poseidon2.compile_schedule_output_eq] at sound
  have outputBoundary :
      Layer.evalState canonicalEnv (Permutation.freshState (8 + 584)) =
        (fun lane : Fin 8 => env
          (invocation.witnessStart +
            (PilotData.circuitPackage ()).permutation.outputLocalStart +
            lane.val)) := by
    funext lane
    have refEq : PilotData.columnRef (592 + lane.val) =
        .local (584 + lane.val) := by
      unfold PilotData.columnRef
      rw [if_neg (by omega)]
      rw [canonicalOutputLocalIndex lane]
    change (PilotData.columnRef (592 + lane.val)).eval value = _
    rw [refEq]
    simp [ColumnRef.eval, value, instantiateInvocationColumn,
      PilotData.circuitPackage, PilotData.permutationTemplate, Nat.add_assoc]
  have inputBoundary :
      Layer.evalState canonicalEnv PilotData.canonicalState =
        (fun lane : Fin 8 =>
          (invocationInputCombination invocation lane.val).toR1CS.eval env) := by
    funext lane
    have refEq : PilotData.columnRef lane.val = .input lane.val := by
      unfold PilotData.columnRef
      rw [if_pos lane.isLt]
    change (PilotData.columnRef lane.val).eval value = _
    rw [refEq]
    rfl
  exact outputBoundary.symm.trans
    (sound.trans (congrArg (Permutation.runF Permutation.schedule)
      inputBoundary))

def chainOutputState (chain : HashChain) (invocation : Nat)
    (env : Env) : Layer.FState :=
  fun lane => env
    (invocationLocalStart (PilotData.circuitPackage ()) chain invocation +
      (PilotData.circuitPackage ()).permutation.outputLocalStart + lane.val)

def chainCarriedState (chain : HashChain) (count : Nat)
    (env : Env) : Layer.FState :=
  if count = 0 then Hash.zeroF else chainOutputState chain (count - 1) env

def chainBlockState (chain : HashChain) (invocation : Nat)
    (env : Env) : Layer.FState :=
  fun lane =>
    let inputOffset :=
      invocation * (PilotData.circuitPackage ()).poseidon.rate + lane.val
    if lane.val < (PilotData.circuitPackage ()).poseidon.rate ∧
        inputOffset < chain.inputLength then
      env (chain.inputStart + inputOffset)
    else
      0

def chainInputValues (chain : HashChain) (env : Env) : List F :=
  List.ofFn fun index : Fin chain.inputLength =>
    env (chain.inputStart + index.val)

def chainBlockList (chain : HashChain) (invocation : Nat)
    (env : Env) : List F :=
  (chainInputValues chain env).drop
      (invocation * (PilotData.circuitPackage ()).poseidon.rate) |>.take
    (PilotData.circuitPackage ()).poseidon.rate

def chainChunks (chain : HashChain) (env : Env) : List (List F) :=
  (List.range chain.absorbCount).map fun invocation =>
    chainBlockList chain invocation env

private theorem chainBlockList_getD (chain : HashChain) (invocation : Nat)
    (env : Env) (lane : Fin 8) :
    (chainBlockList chain invocation env).getD lane.val 0 =
      chainBlockState chain invocation env lane := by
  rw [List.getD_eq_getElem?_getD]
  by_cases inRate : lane.val < Poseidon2.rate <;>
    by_cases inInput : invocation * Poseidon2.rate + lane.val <
      chain.inputLength <;>
    simp [chainBlockList, chainInputValues, chainBlockState,
      List.getElem?_drop,
      PilotData.circuitPackage, PilotData.poseidonSchedule,
      inRate, inInput]

def chainAbsorbed (chain : HashChain) (env : Env) : Nat → Layer.FState
  | 0 => Hash.zeroF
  | count + 1 =>
      Permutation.runF Permutation.schedule (fun lane =>
        chainAbsorbed chain env count lane +
          chainBlockState chain count env lane)

private theorem absorbManyF_append_single (state : Layer.FState)
    (blocks : List (List F)) (block : List F) :
    Hash.absorbManyF state (blocks ++ [block]) =
      Permutation.runF Permutation.schedule
        (Hash.absorbF (Hash.absorbManyF state blocks) block) := by
  induction blocks generalizing state with
  | nil => rfl
  | cons first rest ih =>
      simp only [List.cons_append, Hash.absorbManyF]
      exact ih _

/-- The package absorb recurrence is the standard logical sponge recurrence
over the package-derived input chunks. -/
theorem chainAbsorbed_eq_absorbManyF (chain : HashChain) (env : Env) :
    ∀ count, count ≤ chain.absorbCount →
      chainAbsorbed chain env count =
        Hash.absorbManyF Hash.zeroF ((chainChunks chain env).take count) := by
  intro count
  induction count with
  | zero =>
      intro bound
      rfl
  | succ count ih =>
      intro bound
      have chunkBound : count < (chainChunks chain env).length := by
        simp [chainChunks]
        omega
      have chunkEq : (chainChunks chain env)[count] =
          chainBlockList chain count env := by
        simp [chainChunks]
      calc
        chainAbsorbed chain env (count + 1) =
            Permutation.runF Permutation.schedule (fun lane =>
              chainAbsorbed chain env count lane +
                chainBlockState chain count env lane) := rfl
        _ = Permutation.runF Permutation.schedule
              (Hash.absorbF (chainAbsorbed chain env count)
                (chainBlockList chain count env)) := by
          congr 1
          funext lane
          unfold Hash.absorbF
          rw [chainBlockList_getD]
        _ = Permutation.runF Permutation.schedule
              (Hash.absorbF
                (Hash.absorbManyF Hash.zeroF
                  ((chainChunks chain env).take count))
                (chainBlockList chain count env)) := by
          rw [ih (by omega)]
        _ = Hash.absorbManyF Hash.zeroF
              ((chainChunks chain env).take (count + 1)) := by
          rw [List.take_succ_eq_append_getElem chunkBound,
            absorbManyF_append_single, chunkEq]

private theorem chainChunks_eq_inputChunks (chain : HashChain) (env : Env)
    (countEq : chain.absorbCount =
      (chain.inputLength + Poseidon2.rate - 1) / Poseidon2.rate) :
    chainChunks chain env = Hash.inputChunks (chainInputValues chain env) := by
  unfold chainChunks Hash.inputChunks chainBlockList
  simp only [chainInputValues, List.length_ofFn]
  rw [countEq]
  rfl

private theorem invocationInput_absorb (chain : HashChain)
    (invocation : Nat) (env : Env)
    (beforeFinal : invocation < chain.absorbCount) :
    (fun lane : Fin 8 =>
      (invocationInput (PilotData.circuitPackage ()) chain invocation
        lane.val).eval env) =
      (fun lane => chainCarriedState chain invocation env lane +
        chainBlockState chain invocation env lane) := by
  funext lane
  by_cases zero : invocation = 0
  · subst invocation
    by_cases absorbed : lane.val < Poseidon2.rate ∧
        lane.val < chain.inputLength <;>
      simp [invocationInput, chainCarriedState, chainBlockState,
        PilotData.circuitPackage, PilotData.poseidonSchedule,
        beforeFinal, absorbed, Hash.zeroF]
  · by_cases absorbed : lane.val < Poseidon2.rate ∧
        invocation * Poseidon2.rate + lane.val < chain.inputLength <;>
      simp [invocationInput, chainCarriedState, chainBlockState,
        PilotData.circuitPackage, PilotData.poseidonSchedule,
        PilotData.permutationTemplate, chainOutputState,
        invocationLocalStart, beforeFinal, absorbed, zero,
        Nat.add_assoc]

/-- All satisfying absorb invocations produce the exact package-defined
sponge recurrence. The proof is structural in the invocation count. -/
theorem canonicalChainAbsorptions_sound (chain : HashChain) (env : Env)
    (holds : HashChainHolds (PilotData.circuitPackage ()) chain env) :
    ∀ count, count ≤ chain.absorbCount →
      chainCarriedState chain count env = chainAbsorbed chain env count := by
  intro count
  induction count with
  | zero =>
      intro bound
      simp [chainCarriedState, chainAbsorbed]
  | succ count ih =>
      intro bound
      have beforeFinal : count < chain.absorbCount := by omega
      have invocationSound := canonicalTemplateInvocation_sound chain count env
        (holds count (by omega))
      change chainOutputState chain count env =
        Permutation.runF Permutation.schedule (fun lane : Fin 8 =>
          (invocationInput (PilotData.circuitPackage ()) chain count
            lane.val).eval env) at invocationSound
      calc
        chainCarriedState chain (count + 1) env =
            chainOutputState chain count env := by
          simp [chainCarriedState]
        _ = Permutation.runF Permutation.schedule (fun lane : Fin 8 =>
              (invocationInput (PilotData.circuitPackage ()) chain count
                lane.val).eval env) := invocationSound
        _ = Permutation.runF Permutation.schedule (fun lane =>
              chainCarriedState chain count env lane +
                chainBlockState chain count env lane) := by
          rw [invocationInput_absorb chain count env beforeFinal]
        _ = Permutation.runF Permutation.schedule (fun lane =>
              chainAbsorbed chain env count lane +
                chainBlockState chain count env lane) := by
          rw [ih (by omega)]
        _ = chainAbsorbed chain env (count + 1) := rfl

/-- The last package invocation is exactly the sponge padding permutation. -/
theorem canonicalChainFinal_sound (chain : HashChain) (env : Env)
    (holds : HashChainHolds (PilotData.circuitPackage ()) chain env) :
    chainOutputState chain chain.absorbCount env =
      Permutation.runF Permutation.schedule
        (Hash.padF (chainAbsorbed chain env chain.absorbCount)) := by
  have invocationSound := canonicalTemplateInvocation_sound chain
    chain.absorbCount env (holds chain.absorbCount (by omega))
  change chainOutputState chain chain.absorbCount env =
    Permutation.runF Permutation.schedule (fun lane : Fin 8 =>
      (invocationInput (PilotData.circuitPackage ()) chain chain.absorbCount
        lane.val).eval env) at invocationSound
  have finalInput :
      (fun lane : Fin 8 =>
        (invocationInput (PilotData.circuitPackage ()) chain chain.absorbCount
          lane.val).eval env) =
        Hash.padF (chainCarriedState chain chain.absorbCount env) := by
    funext lane
    by_cases empty : chain.absorbCount = 0
    · by_cases zeroLane : lane = 0 <;>
        simp [invocationInput, Hash.padF, chainCarriedState,
          empty, Hash.zeroF, zeroLane]
    · by_cases zeroLane : lane.val = 0
      · have laneEq : lane = 0 := Fin.ext zeroLane
        subst lane
        simp [invocationInput, Hash.padF, chainCarriedState,
          PilotData.circuitPackage, PilotData.poseidonSchedule,
          PilotData.permutationTemplate, chainOutputState,
          invocationLocalStart, empty]
      · simp [invocationInput, Hash.padF, chainCarriedState,
          PilotData.circuitPackage, PilotData.poseidonSchedule,
          PilotData.permutationTemplate, chainOutputState,
          invocationLocalStart, empty, zeroLane]
  have absorbed := canonicalChainAbsorptions_sound chain env holds
    chain.absorbCount (by omega)
  calc
    chainOutputState chain chain.absorbCount env =
        Permutation.runF Permutation.schedule (fun lane : Fin 8 =>
          (invocationInput (PilotData.circuitPackage ()) chain
            chain.absorbCount lane.val).eval env) := invocationSound
    _ = Permutation.runF Permutation.schedule
        (Hash.padF (chainCarriedState chain chain.absorbCount env)) := by
      rw [finalInput]
    _ = Permutation.runF Permutation.schedule
        (Hash.padF (chainAbsorbed chain env chain.absorbCount)) := by
      rw [absorbed]

/-- A well-sized package chain ends in the standard logical Poseidon2 sponge
state over its exact input segment. -/
theorem canonicalChainFinal_eq_hashF (chain : HashChain) (env : Env)
    (countEq : chain.absorbCount =
      (chain.inputLength + Poseidon2.rate - 1) / Poseidon2.rate)
    (holds : HashChainHolds (PilotData.circuitPackage ()) chain env) :
    chainOutputState chain chain.absorbCount env =
      Permutation.runF Permutation.schedule
        (Hash.padF (Hash.absorbManyF Hash.zeroF
          (Hash.inputChunks (chainInputValues chain env)))) := by
  have finalState := canonicalChainFinal_sound chain env holds
  have absorbed := chainAbsorbed_eq_absorbManyF chain env
    chain.absorbCount (by omega)
  have chunksLength : (chainChunks chain env).length = chain.absorbCount := by
    simp [chainChunks]
  have absorbedStandard :
      chainAbsorbed chain env chain.absorbCount =
        Hash.absorbManyF Hash.zeroF
          (Hash.inputChunks (chainInputValues chain env)) := by
    calc
      chainAbsorbed chain env chain.absorbCount =
          Hash.absorbManyF Hash.zeroF
            ((chainChunks chain env).take chain.absorbCount) := absorbed
      _ = Hash.absorbManyF Hash.zeroF (chainChunks chain env) := by
        rw [← chunksLength, List.take_length]
      _ = Hash.absorbManyF Hash.zeroF
          (Hash.inputChunks (chainInputValues chain env)) := by
        rw [chainChunks_eq_inputChunks chain env countEq]
  rw [absorbedStandard] at finalState
  exact finalState

/-- The first four final package lanes are the exact executable Poseidon2
digest of the chain's input segment. -/
theorem canonicalChainDigest_eq_hash (chain : HashChain) (env : Env)
    (countEq : chain.absorbCount =
      (chain.inputLength + Poseidon2.rate - 1) / Poseidon2.rate)
    (holds : HashChainHolds (PilotData.circuitPackage ()) chain env) :
    List.ofFn (fun lane : Fin 4 =>
      chainOutputState chain chain.absorbCount env
        ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩) =
      Spec.Poseidon2.hash (chainInputValues chain env) := by
  have finalState := canonicalChainFinal_eq_hashF chain env countEq holds
  calc
    List.ofFn (fun lane : Fin 4 =>
        chainOutputState chain chain.absorbCount env
          ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩) =
        List.ofFn (Hash.digestF
          (chainOutputState chain chain.absorbCount env)) := rfl
    _ = List.ofFn (Hash.hashF (chainInputValues chain env)) := by
      rw [finalState]
      rfl
    _ = Spec.Poseidon2.hash (chainInputValues chain env) :=
      Hash.hashF_eq_reference _

private theorem digestRow_sound (chain : HashChain) (lane : Fin 4)
    (env : Env) (holds : (PilotData.digestRow chain lane).Holds env) :
    env (chain.digestStart + lane.val) =
      chainOutputState chain chain.absorbCount env
        ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩ := by
  have equation := holds
  simp [SparseRow.Holds, SparseCombination.eval, PilotData.digestRow,
    PilotData.oneCombination, PilotData.zeroCombination,
    fieldValue_neg_one] at equation
  have difference :
      env (chain.witnessStart + chain.absorbCount * 592 + 584 + lane.val) -
        env (chain.digestStart + lane.val) = 0 := by
    simpa [sub_eq_add_neg] using equation
  calc
    env (chain.digestStart + lane.val) =
        env (chain.witnessStart + chain.absorbCount * 592 +
          584 + lane.val) := (sub_eq_zero.mp difference).symm
    _ = chainOutputState chain chain.absorbCount env
        ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩ := by
      simp [chainOutputState, invocationLocalStart,
        PilotData.circuitPackage, PilotData.permutationTemplate,
        Nat.add_assoc]

private theorem digestRow_mem (chain : HashChain) (lane : Fin 4) :
    PilotData.digestRow chain lane ∈ PilotData.digestRows chain := by
  rw [PilotData.digestRows, List.mem_ofFn']
  exact ⟨lane, rfl⟩

/-- The 58 emitted assertion rows bind both hash digests and the complete
54-cell prior public input marker/tail layout. -/
theorem canonicalAssertions_sound (env : Env)
    (holds : AssertionsHold (PilotData.circuitPackage ()) env) :
    env PilotSpartan.firstPublicStart = 1 ∧
      (∀ lane : Fin 4, env (PilotData.priorChain.digestStart + lane.val) =
        chainOutputState PilotData.priorChain
          PilotData.priorChain.absorbCount env
          ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩) ∧
      (∀ lane : Fin 49,
        env (PilotSpartan.firstPublicStart + 5 + lane.val) = 0) ∧
      (∀ lane : Fin 4, env (PilotData.outputChain.digestStart + lane.val) =
        chainOutputState PilotData.outputChain
          PilotData.outputChain.absorbCount env
          ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩) := by
  unfold AssertionsHold at holds
  have priorDigest (lane : Fin 4) :
      (PilotData.digestRow PilotData.priorChain lane).Holds env := by
    apply holds _
    change PilotData.digestRow PilotData.priorChain lane ∈
      PilotData.assertionRows ()
    apply List.mem_append_left
    apply List.mem_append_left
    exact digestRow_mem PilotData.priorChain lane
  have outputDigest (lane : Fin 4) :
      (PilotData.digestRow PilotData.outputChain lane).Holds env := by
    apply holds _
    change PilotData.digestRow PilotData.outputChain lane ∈
      PilotData.assertionRows ()
    apply List.mem_append_right
    exact digestRow_mem PilotData.outputChain lane
  have marker : PilotData.markerBindingRow.Holds env := by
    apply holds _
    change PilotData.markerBindingRow ∈ PilotData.assertionRows ()
    apply List.mem_append_left
    apply List.mem_append_right
    simp [PilotData.bindingRows]
  have tail (lane : Fin 49) :
      (⟨PilotData.priorBindingRowStart + 1 + lane.val,
        ⟨0, [⟨PilotSpartan.firstPublicStart + 5 + lane.val, 1⟩]⟩,
        PilotData.oneCombination,
        PilotData.zeroCombination⟩ : SparseRow).Holds env := by
    apply holds _
    change (⟨PilotData.priorBindingRowStart + 1 + lane.val,
      ⟨0, [⟨PilotSpartan.firstPublicStart + 5 + lane.val, 1⟩]⟩,
      PilotData.oneCombination,
      PilotData.zeroCombination⟩ : SparseRow) ∈ PilotData.assertionRows ()
    apply List.mem_append_left
    apply List.mem_append_right
    apply List.mem_cons.mpr
    apply Or.inr
    rw [PilotData.tailBindingRows, List.mem_ofFn']
    exact ⟨lane, rfl⟩
  refine ⟨?_, ?_, ?_, ?_⟩
  · have markerEquation := marker
    simp [SparseRow.Holds, SparseCombination.eval,
      PilotData.markerBindingRow, PilotData.oneCombination,
      PilotData.zeroCombination, fieldValue_neg_one] at markerEquation
    have difference : env PilotSpartan.firstPublicStart - 1 = 0 := by
      simpa [sub_eq_add_neg, add_comm] using markerEquation
    exact sub_eq_zero.mp difference
  · intro lane
    exact digestRow_sound PilotData.priorChain lane env (priorDigest lane)
  · intro lane
    simpa [SparseRow.Holds, SparseCombination.eval,
      PilotData.oneCombination, PilotData.zeroCombination] using tail lane
  · intro lane
    exact digestRow_sound PilotData.outputChain lane env (outputDigest lane)

private theorem priorChain_count_eq :
    PilotData.priorChain.absorbCount =
      (PilotData.priorChain.inputLength + Poseidon2.rate - 1) /
        Poseidon2.rate := by
  rfl

private theorem outputChain_count_eq :
    PilotData.outputChain.absorbCount =
      (PilotData.outputChain.inputLength + Poseidon2.rate - 1) /
        Poseidon2.rate := by
  rfl

/-- The canonical package rows enforce both reference hash outputs and the
complete prior public-input marker/tail binding. -/
theorem canonicalPackage_hashes (env : Env)
    (holds : (PilotData.circuitPackage ()).RowsHold env) :
    env PilotSpartan.firstPublicStart = 1 ∧
      (∀ lane : Fin 49,
        env (PilotSpartan.firstPublicStart + 5 + lane.val) = 0) ∧
      List.ofFn (fun lane : Fin 4 =>
        env (PilotData.priorChain.digestStart + lane.val)) =
          Spec.Poseidon2.hash
            (chainInputValues PilotData.priorChain env) ∧
      List.ofFn (fun lane : Fin 4 =>
        env (PilotData.outputChain.digestStart + lane.val)) =
          Spec.Poseidon2.hash
            (chainInputValues PilotData.outputChain env) := by
  rcases holds with ⟨chains, _invocations, _instructions, assertions⟩
  have priorHolds : HashChainHolds (PilotData.circuitPackage ())
      PilotData.priorChain env := by
    apply chains _
    simp [PilotData.circuitPackage]
  have outputHolds : HashChainHolds (PilotData.circuitPackage ())
      PilotData.outputChain env := by
    apply chains _
    simp [PilotData.circuitPackage]
  have assertionFacts := canonicalAssertions_sound env assertions
  have priorHash := canonicalChainDigest_eq_hash PilotData.priorChain env
    priorChain_count_eq priorHolds
  have outputHash := canonicalChainDigest_eq_hash PilotData.outputChain env
    outputChain_count_eq outputHolds
  refine ⟨assertionFacts.1, assertionFacts.2.2.1, ?_, ?_⟩
  · calc
      List.ofFn (fun lane : Fin 4 =>
          env (PilotData.priorChain.digestStart + lane.val)) =
          List.ofFn (fun lane : Fin 4 =>
            chainOutputState PilotData.priorChain
              PilotData.priorChain.absorbCount env
              ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩) := by
        congr 1
        funext lane
        exact assertionFacts.2.1 lane
      _ = Spec.Poseidon2.hash
          (chainInputValues PilotData.priorChain env) := priorHash
  · calc
      List.ofFn (fun lane : Fin 4 =>
          env (PilotData.outputChain.digestStart + lane.val)) =
          List.ofFn (fun lane : Fin 4 =>
            chainOutputState PilotData.outputChain
              PilotData.outputChain.absorbCount env
              ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩) := by
        congr 1
        funext lane
        exact assertionFacts.2.2.2 lane
      _ = Spec.Poseidon2.hash
          (chainInputValues PilotData.outputChain env) := outputHash

private theorem priorInputValues_eq (env : Env) :
    chainInputValues PilotData.priorChain env =
      Hash.evalList (PilotSpartan.pullback env)
        (PilotProduction.priorPreimage PilotProduction.witnessOffset) := by
  simp only [chainInputValues, PilotData.priorChain, Hash.evalList,
    PilotProduction.priorPreimage, PilotProduction.variableExprs,
    List.map_ofFn]
  apply congrArg List.ofFn
  funext index
  change env (0 + index.val) =
    PilotSpartan.pullback env
      (PilotProduction.priorPreimageStart + index.val)
  unfold PilotSpartan.pullback PilotSpartan.sourceToSpartan
  rw [if_pos (by
    have indexBound : index.val < PilotProduction.stateHashWords := by
      rw [← stateHashWords_match]
      exact index.isLt
    unfold PilotProduction.priorPreimageStart PilotSpartan.priorPublicStart
    omega)]
  rfl

private theorem sourceToSpartan_outputPreimage
    (index : Fin PilotProduction.stateHashWords) :
    PilotSpartan.sourceToSpartan
        (PilotProduction.outputPreimageStart + index.val) =
      PilotSpartan.secondPrivateStart + index.val := by
  have indexBound : index.val < 42475 := by
    calc
      index.val < PilotProduction.stateHashWords := index.isLt
      _ = 42475 := PilotProduction.stateHashWords_eq
  unfold PilotSpartan.sourceToSpartan
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals norm_num [PilotProduction.outputPreimageStart,
    PilotProduction.priorPublicInputStart,
    PilotProduction.priorPreimageStart, PilotProduction.stateHashWords,
    PilotProduction.digestWords, PriorStateHash.publicWidth,
    Spec.ringDegree, PaperAlgebra.publicRingColumns,
    PilotSpartan.priorPublicStart, PilotSpartan.outputPreimageStart,
    PilotSpartan.outputDigestStart, PilotSpartan.witnessStart,
    PilotSpartan.secondPrivateStart, PilotSpartan.secondPublicStart,
    PilotSpartan.witnessPrivateStart] at * <;> omega

private theorem outputInputValues_eq (env : Env) :
    chainInputValues PilotData.outputChain env =
      Hash.evalList (PilotSpartan.pullback env)
        (PilotProduction.outputPreimage
          (Lifecycle.Pilot.outputOffset PilotProduction.interface
            PilotProduction.witnessOffset)) := by
  simp only [chainInputValues, PilotData.outputChain, Hash.evalList,
    PilotProduction.outputPreimage, PilotProduction.variableExprs,
    List.map_ofFn]
  apply congrArg List.ofFn
  funext index
  change env (PilotSpartan.secondPrivateStart + index.val) =
    env (PilotSpartan.sourceToSpartan
      (PilotProduction.outputPreimageStart + index.val))
  rw [sourceToSpartan_outputPreimage]

private theorem sourceToSpartan_priorPublic
    (column : Fin PriorStateHash.publicWidth) :
    PilotSpartan.sourceToSpartan
        (PilotProduction.priorPublicInputStart + column.val) =
      PilotSpartan.firstPublicStart + column.val := by
  have columnBound : column.val < 54 := by
    rw [← PriorStateHash.publicWidth_eq]
    exact column.isLt
  unfold PilotSpartan.sourceToSpartan
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals norm_num [PilotProduction.priorPublicInputStart,
    PilotProduction.priorPreimageStart, PilotProduction.stateHashWords,
    PilotProduction.digestWords, PriorStateHash.publicWidth,
    Spec.ringDegree, PaperAlgebra.publicRingColumns,
    PilotSpartan.priorPublicStart, PilotSpartan.outputPreimageStart,
    PilotSpartan.outputDigestStart, PilotSpartan.witnessStart,
    PilotSpartan.firstPublicStart, PilotSpartan.secondPrivateStart,
    PilotSpartan.secondPublicStart, PilotSpartan.witnessPrivateStart] at * <;>
    omega

private theorem priorPublicInput_eval (env : Env)
    (column : Fin PriorStateHash.publicWidth) :
    (PilotProduction.priorInterface.publicInput
      PilotProduction.witnessOffset column).eval
        (PilotSpartan.pullback env) =
      env (PilotSpartan.firstPublicStart + column.val) := by
  rw [PilotProduction.priorInterface_publicInput_apply]
  unfold PilotProduction.priorPublicInput PilotSpartan.pullback
  change env (PilotSpartan.sourceToSpartan
    (PilotProduction.priorPublicInputStart + column.val)) = _
  rw [sourceToSpartan_priorPublic]

private theorem sourceToSpartan_outputDigest (lane : Fin 4) :
    PilotSpartan.sourceToSpartan
        (PilotProduction.outputDigestStart + lane.val) =
      PilotSpartan.secondPublicStart + lane.val := by
  unfold PilotSpartan.sourceToSpartan
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals norm_num [PilotProduction.outputDigestStart,
    PilotProduction.outputPreimageStart,
    PilotProduction.priorPublicInputStart,
    PilotProduction.priorPreimageStart, PilotProduction.stateHashWords,
    PilotProduction.digestWords, PriorStateHash.publicWidth,
    Spec.ringDegree, PaperAlgebra.publicRingColumns,
    PilotSpartan.priorPublicStart, PilotSpartan.outputPreimageStart,
    PilotSpartan.outputDigestStart, PilotSpartan.witnessStart,
    PilotSpartan.firstPublicStart, PilotSpartan.secondPrivateStart,
    PilotSpartan.secondPublicStart, PilotSpartan.witnessPrivateStart] at * <;>
    omega

private theorem outputDigestValues_eq (env : Env) :
    List.ofFn (fun lane : Fin 4 =>
      (PilotProduction.outputInterface.digest
        (Lifecycle.Pilot.outputOffset PilotProduction.interface
          PilotProduction.witnessOffset) lane).eval
        (PilotSpartan.pullback env)) =
      List.ofFn (fun lane : Fin 4 =>
        env (PilotData.outputChain.digestStart + lane.val)) := by
  apply congrArg List.ofFn
  funext lane
  change env (PilotSpartan.sourceToSpartan
      (PilotProduction.outputDigestStart + lane.val)) =
    env (PilotData.outputChain.digestStart + lane.val)
  rw [sourceToSpartan_outputDigest]
  rfl

private theorem priorColumn_cases
    (column : Fin PriorStateHash.publicWidth) :
    column = PriorStateHash.markerIndex ∨
      (∃ lane : Fin 4, column = PriorStateHash.digestIndex lane) ∨
      ∃ lane : Fin 49, column = PriorStateHash.tailIndex lane := by
  have columnBound : column.val < 54 := by
    rw [← PriorStateHash.publicWidth_eq]
    exact column.isLt
  by_cases isMarker : column.val = 0
  · apply Or.inl
    apply Fin.ext
    simpa [PriorStateHash.markerIndex] using isMarker
  · apply Or.inr
    by_cases isDigest : column.val < 5
    · apply Or.inl
      let lane : Fin 4 := ⟨column.val - 1, by omega⟩
      refine ⟨lane, ?_⟩
      apply Fin.ext
      change column.val = lane.val + 1
      dsimp [lane]
      omega
    · apply Or.inr
      let lane : Fin 49 := ⟨column.val - 5, by omega⟩
      refine ⟨lane, ?_⟩
      apply Fin.ext
      change column.val = lane.val + 5
      dsimp [lane]
      omega

/-- The four package-enforced hash facts imply both logical pilot builder
specifications after the proved Spartan column pullback. -/
theorem hashFacts_imply_spec (env : Env)
    (facts :
      env PilotSpartan.firstPublicStart = 1 ∧
      (∀ lane : Fin 49,
        env (PilotSpartan.firstPublicStart + 5 + lane.val) = 0) ∧
      List.ofFn (fun lane : Fin 4 =>
        env (PilotData.priorChain.digestStart + lane.val)) =
          Spec.Poseidon2.hash
            (chainInputValues PilotData.priorChain env) ∧
      List.ofFn (fun lane : Fin 4 =>
        env (PilotData.outputChain.digestStart + lane.val)) =
          Spec.Poseidon2.hash
            (chainInputValues PilotData.outputChain env)) :
    Lifecycle.Pilot.SpecHolds PilotProduction.interface
      PilotProduction.witnessOffset (PilotSpartan.pullback env) := by
  constructor
  · rw [PilotProduction.interface_prior]
    unfold PriorStateHash.SpecHolds
    funext column
    rw [priorPublicInput_eval]
    rw [PilotProduction.priorInterface_preimage_apply]
    rw [← priorInputValues_eq]
    rw [← facts.2.2.1]
    rcases priorColumn_cases column with marker | digest | tail
    · subst column
      simpa [PilotSpartan.firstPublicStart] using facts.1
    · rcases digest with ⟨lane, rfl⟩
      rw [PriorStateHash.encodedHash_digest]
      apply congrArg env
      rw [← firstPublicStart_match]
      norm_num [PilotSpartan.firstPublicStart, PilotData.priorChain,
        PriorStateHash.digestIndex]
      omega
    · rcases tail with ⟨lane, rfl⟩
      rw [PriorStateHash.encodedHash_tail]
      calc
        env (PilotSpartan.firstPublicStart +
            (PriorStateHash.tailIndex lane).val) =
            env (PilotSpartan.firstPublicStart + 5 + lane.val) := by
          apply congrArg env
          norm_num [PilotSpartan.firstPublicStart,
            PriorStateHash.tailIndex]
          omega
        _ = 0 := facts.2.1 lane
  · rw [PilotProduction.interface_output]
    unfold OutputHash.SpecHolds Formal.SpecHolds
    rw [OutputHash.hashInterface_input]
    simp only [OutputHash.hashInterface_expected]
    rw [PilotProduction.outputInterface_preimage_apply]
    rw [← outputInputValues_eq]
    rw [outputDigestValues_eq]
    exact facts.2.2.2

/-- Satisfaction of the canonical emitted rows implies both logical pilot
builder specifications after the proved Spartan column pullback. -/
theorem canonicalPackage_implies_spec (env : Env)
    (holds : (PilotData.circuitPackage ()).RowsHold env) :
    Lifecycle.Pilot.SpecHolds PilotProduction.interface
      PilotProduction.witnessOffset (PilotSpartan.pullback env) := by
  exact hashFacts_imply_spec env (canonicalPackage_hashes env holds)

/-- The four package-enforced hash facts, together with the fixed protocol ABI
values below the witness boundary, imply the two recursive lifecycle slots. -/
theorem hashFacts_imply_recursive_hash_slots
    {logicalWidth : Nat}
    {publicFits : Spec.ringDegree * PaperAlgebra.publicRingColumns <=
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (F : AppState → AppWitness → AppState)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      slotCount)
    (priorFixed : PilotProduction.FixedPreimage
      (priorHashPreimage (setup relation ajtai vk) input))
    (outputFixed : PilotProduction.FixedPreimage
      (nextHashPreimage (setup relation ajtai vk) input output))
    (digestFixed : output.x.length = PilotProduction.digestWords)
    (env : Env)
    (agrees : PilotProduction.AgreesBelow (PilotSpartan.pullback env)
      (PilotProduction.protocolEnv
        (priorHashPreimage (setup relation ajtai vk) input)
        ((machine publicFits F).freshPublic input.fresh)
        (nextHashPreimage (setup relation ajtai vk) input output)
        output.x priorFixed outputFixed digestFixed)
      PilotProduction.witnessOffset)
    (facts :
      env PilotSpartan.firstPublicStart = 1 ∧
      (∀ lane : Fin 49,
        env (PilotSpartan.firstPublicStart + 5 + lane.val) = 0) ∧
      List.ofFn (fun lane : Fin 4 =>
        env (PilotData.priorChain.digestStart + lane.val)) =
          Spec.Poseidon2.hash
            (chainInputValues PilotData.priorChain env) ∧
      List.ofFn (fun lane : Fin 4 =>
        env (PilotData.outputChain.digestStart + lane.val)) =
          Spec.Poseidon2.hash
            (chainInputValues PilotData.outputChain env)) :
    (machine publicFits F).freshPublic input.fresh =
        (machine publicFits F).encodeInstance
          ((machine publicFits F).hash
            (priorHashPreimage (setup relation ajtai vk) input)) ∧
      OutputHolds (setup relation ajtai vk) (machine publicFits F)
        input output := by
  have specification := hashFacts_imply_spec env facts
  have represented := PilotProduction.protocolEnv_represents_of_agreesBelow
    (priorHashPreimage (setup relation ajtai vk) input)
    ((machine publicFits F).freshPublic input.fresh)
    (nextHashPreimage (setup relation ajtai vk) input output)
    output.x priorFixed outputFixed digestFixed
    (PilotSpartan.pullback env) agrees
  exact Lifecycle.Pilot.builders_imply_hash_slots
    PilotProduction.interface PilotProduction.witnessOffset
    (PilotSpartan.pullback env) relation ajtai vk F input output specification
    represented.1 represented.2.1 represented.2.2.1 represented.2.2.2

/-- Canonical package rows, together with the fixed protocol ABI values below
the witness boundary, imply the two recursive lifecycle relation slots. -/
theorem canonicalPackage_implies_recursive_hash_slots
    {logicalWidth : Nat}
    {publicFits : Spec.ringDegree * PaperAlgebra.publicRingColumns <=
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (F : AppState → AppWitness → AppState)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      slotCount)
    (priorFixed : PilotProduction.FixedPreimage
      (priorHashPreimage (setup relation ajtai vk) input))
    (outputFixed : PilotProduction.FixedPreimage
      (nextHashPreimage (setup relation ajtai vk) input output))
    (digestFixed : output.x.length = PilotProduction.digestWords)
    (env : Env)
    (agrees : PilotProduction.AgreesBelow (PilotSpartan.pullback env)
      (PilotProduction.protocolEnv
        (priorHashPreimage (setup relation ajtai vk) input)
        ((machine publicFits F).freshPublic input.fresh)
        (nextHashPreimage (setup relation ajtai vk) input output)
        output.x priorFixed outputFixed digestFixed)
      PilotProduction.witnessOffset)
    (holds : (PilotData.circuitPackage ()).RowsHold env) :
    (machine publicFits F).freshPublic input.fresh =
        (machine publicFits F).encodeInstance
          ((machine publicFits F).hash
            (priorHashPreimage (setup relation ajtai vk) input)) ∧
      OutputHolds (setup relation ajtai vk) (machine publicFits F)
        input output := by
  exact hashFacts_imply_recursive_hash_slots relation ajtai vk F input output
    priorFixed outputFixed digestFixed env agrees
      (canonicalPackage_hashes env holds)

theorem canonicalState_affine :
    Poseidon2.StateAffine PilotData.canonicalState := by
  intro lane
  exact R1CS.isAffine_var lane.val

theorem canonicalRows_length :
    (PilotData.canonicalRows ()).length = 592 := by
  calc
    (PilotData.canonicalRows ()).length =
        R1CS.totalRowCount (PilotData.canonicalConstraints ()) := by
      exact R1CS.lowerConstraints_rows_length
        (PilotData.canonicalConstraints ()) 600
    _ = (PilotData.canonicalRecipes ()).length := by
      exact R1CS.recipeConstraints_totalRowCount 8
        (PilotData.canonicalRecipes ())
        (Poseidon2.compile_schedule_direct 8 PilotData.canonicalState
          canonicalState_affine)
    _ = 592 := by
      exact Permutation.compile_schedule_recipe_count 8
        PilotData.canonicalState

theorem templateRowsFrom_length (output : Nat) (rows : List R1CS.Row) :
    (PilotData.templateRowsFrom output rows).length = rows.length := by
  induction rows generalizing output with
  | nil => rfl
  | cons row rest ih =>
      simp [PilotData.templateRowsFrom, ih]

theorem templateRows_length :
    (PilotData.templateRows ()).length = 592 := by
  rw [PilotData.templateRows, templateRowsFrom_length,
    canonicalRows_length]

@[simp] theorem digestRows_length (chain : HashChain) :
    (PilotData.digestRows chain).length = 4 := by
  simp [PilotData.digestRows]

theorem tailBindingRows_length :
    PilotData.tailBindingRows.length = 49 := by
  simp [PilotData.tailBindingRows]

theorem bindingRows_length :
    (PilotData.bindingRows ()).length = 50 := by
  simp [PilotData.bindingRows, tailBindingRows_length]

theorem assertionRows_length :
    (PilotData.assertionRows ()).length = 58 := by
  simp [PilotData.assertionRows, bindingRows_length]

theorem circuitPackage_decode_encode :
    CircuitPackage.format.decode
      (CircuitPackage.format.encode (PilotData.circuitPackage ())) =
        .ok (PilotData.circuitPackage ()) :=
  Package.decode_encode (PilotData.circuitPackage ())

theorem circuitPackage_template_rows :
    (PilotData.circuitPackage ()).permutation.rows.length = 592 :=
  templateRows_length

theorem circuitPackage_assertion_rows :
    (PilotData.circuitPackage ()).assertionRows.length = 58 :=
  assertionRows_length

theorem circuitPackage_row_coverage :
    PilotData.priorChain.witnessLength +
      PilotData.outputChain.witnessLength +
      (PilotData.circuitPackage ()).assertionRows.length =
        (PilotData.circuitPackage ()).layout.rowCount := by
  rw [circuitPackage_assertion_rows]
  rfl

/-- The executable package uses the exact proved pilot row and Spartan column
counts. No Rust-selected layout value enters this statement. -/
theorem circuitPackage_layout_matches :
    let layout := (PilotData.circuitPackage ()).layout
    layout.rowCount =
        Layout.Pilot.physicalRowCount PilotProduction.interface
          PilotProduction.witnessOffset ∧
      layout.privateColumnCount = PilotSpartan.privateColumnCount ∧
      layout.constantColumn = PilotSpartan.constantColumn ∧
      layout.publicColumnCount = PilotSpartan.publicColumnCount ∧
      layout.totalColumnCount = PilotSpartan.spartanColumnCount := by
  dsimp [PilotData.circuitPackage, PilotData.physicalLayout]
  exact ⟨PilotProduction.physicalRowCount_eq.symm, rfl, rfl, rfl, rfl⟩

theorem artifact_package :
    (PilotData.artifact ()).package = PilotData.circuitPackage () := by
  rfl

end NightstreamFPrime.Export.Pilot
