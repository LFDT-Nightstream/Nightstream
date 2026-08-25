import NightstreamFPrime.Gadgets.Poseidon2.Layer

/-!
Owns the logical Poseidon2 permutation schedule. The builder emits one
straight-line recipe for each of eight lanes at every layer. Its proofs connect
the recipe rows to the executable reference permutation and show that the
canonical witness program is causal. Physical rows and columns belong to
`Layout/`.
-/

namespace NightstreamFPrime.Gadgets.Poseidon2.Permutation

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit

abbrev EState := Layer.EState
abbrev FState := Layer.FState

/-- One fixed step of the production Poseidon2 permutation. -/
inductive Step where
  | initialLayer
  | initialFullRound (round : Nat)
  | partialRound (round : Nat)
  | terminalFullRound (round : Nat)
deriving Repr, DecidableEq

def applyE : Step → EState → EState
  | .initialLayer => Layer.externalE
  | .initialFullRound round => Layer.fullE Spec.Poseidon2.initialConstants round
  | .partialRound round => Layer.partialE round
  | .terminalFullRound round =>
      Layer.fullE Spec.Poseidon2.terminalConstants round

def applyF : Step → FState → FState
  | .initialLayer => Layer.externalF
  | .initialFullRound round => Layer.fullF Spec.Poseidon2.initialConstants round
  | .partialRound round => Layer.partialF round
  | .terminalFullRound round =>
      Layer.fullF Spec.Poseidon2.terminalConstants round

def applyReference : Step → Spec.Poseidon2.State → Spec.Poseidon2.State
  | .initialLayer => Spec.Poseidon2.externalLayer
  | .initialFullRound round =>
      Spec.Poseidon2.fullRound Spec.Poseidon2.initialConstants round
  | .partialRound round => Spec.Poseidon2.partialRound round
  | .terminalFullRound round =>
      Spec.Poseidon2.fullRound Spec.Poseidon2.terminalConstants round

/-- The unique production permutation schedule. -/
def schedule : List Step :=
  [Step.initialLayer] ++
    (List.range Spec.Poseidon2.halfFullRounds).map Step.initialFullRound ++
    (List.range Spec.Poseidon2.partialRounds).map Step.partialRound ++
    (List.range Spec.Poseidon2.halfFullRounds).map Step.terminalFullRound

def runF : List Step → FState → FState
  | [], state => state
  | step :: rest, state => runF rest (applyF step state)

def runReference : List Step → Spec.Poseidon2.State → Spec.Poseidon2.State
  | [], state => state
  | step :: rest, state => runReference rest (applyReference step state)

@[simp] theorem eval_applyE (env : Env) (step : Step) (state : EState) :
    Layer.evalState env (applyE step state) =
      applyF step (Layer.evalState env state) := by
  funext lane
  cases step <;> simp [applyE, applyF, Layer.evalState]

theorem applyF_eq_reference (step : Step) (state : FState) :
    List.ofFn (applyF step state) =
      applyReference step (List.ofFn state) := by
  cases step with
  | initialLayer => exact Layer.externalF_eq_reference state
  | initialFullRound round =>
      exact Layer.fullF_eq_reference Spec.Poseidon2.initialConstants round state
  | partialRound round => exact Layer.partialF_eq_reference round state
  | terminalFullRound round =>
      exact Layer.fullF_eq_reference Spec.Poseidon2.terminalConstants round state

theorem varsBelow_getE (state : EState) {bound : Nat}
    (hstate : ∀ lane, (state lane).VarsBelow bound) (index : Nat) :
    (Layer.getE state index).VarsBelow bound := by
  unfold Layer.getE
  split
  · exact hstate _
  · trivial

theorem varsBelow_sboxE {value : Expr} {bound : Nat}
    (hvalue : value.VarsBelow bound) :
    (Layer.sboxE value).VarsBelow bound := by
  simp [Layer.sboxE, Expr.VarsBelow, hvalue]

theorem varsBelow_mat4E (state : EState) {bound : Nat}
    (hstate : ∀ lane, (state lane).VarsBelow bound) (base lane : Nat) :
    (Layer.mat4E state base lane).VarsBelow bound := by
  rcases lane with _ | _ | _ | lane <;>
    simp [Layer.mat4E, Expr.VarsBelow, varsBelow_getE state hstate]

theorem varsBelow_externalE (state : EState) {bound : Nat}
    (hstate : ∀ lane, (state lane).VarsBelow bound) (lane : Fin 8) :
    (Layer.externalE state lane).VarsBelow bound := by
  simp [Layer.externalE, Layer.blockE, Expr.VarsBelow,
    varsBelow_mat4E state hstate]

theorem varsBelow_sumE (state : EState) {bound : Nat}
    (hstate : ∀ lane, (state lane).VarsBelow bound) :
    (Layer.sumE state).VarsBelow bound := by
  simp [Layer.sumE, Expr.VarsBelow, varsBelow_getE state hstate]

theorem varsBelow_internalE (state : EState) {bound : Nat}
    (hstate : ∀ lane, (state lane).VarsBelow bound) (lane : Fin 8) :
    (Layer.internalE state lane).VarsBelow bound := by
  simp [Layer.internalE, Expr.VarsBelow, hstate, varsBelow_sumE state hstate]

theorem varsBelow_fullE (rows : List (List Nat)) (round : Nat)
    (state : EState) {bound : Nat}
    (hstate : ∀ lane, (state lane).VarsBelow bound) (lane : Fin 8) :
    (Layer.fullE rows round state lane).VarsBelow bound := by
  apply varsBelow_externalE
  intro index
  apply varsBelow_sboxE
  simp [Expr.VarsBelow, hstate]

theorem varsBelow_partialE (round : Nat) (state : EState) {bound : Nat}
    (hstate : ∀ lane, (state lane).VarsBelow bound) (lane : Fin 8) :
    (Layer.partialE round state lane).VarsBelow bound := by
  apply varsBelow_internalE
  intro index
  by_cases hzero : index.val = 0
  · simp [hzero, Expr.VarsBelow, hstate, varsBelow_sboxE]
  · simp [hzero, hstate]

theorem applyE_varsBelow (step : Step) (state : EState) {bound : Nat}
    (hstate : ∀ lane, (state lane).VarsBelow bound) (lane : Fin 8) :
    (applyE step state lane).VarsBelow bound := by
  cases step with
  | initialLayer => exact varsBelow_externalE state hstate lane
  | initialFullRound round =>
      exact varsBelow_fullE Spec.Poseidon2.initialConstants round state hstate lane
  | partialRound round => exact varsBelow_partialE round state hstate lane
  | terminalFullRound round =>
      exact varsBelow_fullE Spec.Poseidon2.terminalConstants round state hstate lane

/-- Variables allocated for one eight-lane state. -/
def freshState (start : Nat) : EState :=
  fun lane => Expr.var (start + lane.val)

theorem freshState_varsBelow (start : Nat) (lane : Fin 8) :
    (freshState start lane).VarsBelow (start + 8) := by
  simpa [freshState, Expr.VarsBelow] using
    Nat.add_lt_add_left lane.isLt start

theorem recipeConstraints_append (start : Nat) (first second : List Expr) :
    recipeConstraints start (first ++ second) =
      recipeConstraints start first ++
        recipeConstraints (start + first.length) second := by
  induction first generalizing start with
  | nil => simp [recipeConstraints]
  | cons recipe rest ih =>
      simp only [List.cons_append, recipeConstraints, List.length_cons,
        List.cons_append, List.cons.injEq, true_and]
      rw [ih]
      congr 2
      omega

theorem constraintsHold_append (env : Env) (first second : List Expr) :
    ConstraintsHold env (first ++ second) ↔
      ConstraintsHold env first ∧ ConstraintsHold env second := by
  constructor
  · intro holds
    constructor
    · intro expression member
      exact holds expression (List.mem_append_left second member)
    · intro expression member
      exact holds expression (List.mem_append_right first member)
  · rintro ⟨holdsFirst, holdsSecond⟩ expression member
    rcases List.mem_append.mp member with member | member
    · exact holdsFirst expression member
    · exact holdsSecond expression member

theorem recipesCausal_append_causal (start : Nat) (first second : List Expr)
    (hfirst : RecipesCausal start first)
    (hsecond : RecipesCausal (start + first.length) second) :
    RecipesCausal start (first ++ second) := by
  induction first generalizing start with
  | nil => simpa using hsecond
  | cons recipe rest ih =>
      constructor
      · exact hfirst.1
      · apply ih (start := start + 1) hfirst.2
        convert hsecond using 1 <;> simp only [List.length_cons] <;> omega

/-- Four shared multiplication recipes for `x⁷`: `x²`, `x⁴`, `x⁶`,
then `x⁷`. -/
def sboxRecipes (start : Nat) (value : Expr) : List Expr :=
  [value * value,
   Expr.var start * Expr.var start,
   Expr.var (start + 1) * Expr.var start,
   Expr.var (start + 2) * value]

def sboxOutput (start : Nat) : Expr := Expr.var (start + 3)

@[simp] theorem sboxRecipes_length (start : Nat) (value : Expr) :
    (sboxRecipes start value).length = 4 := by
  rfl

theorem sboxRecipes_causal (start : Nat) (value : Expr)
    (hvalue : value.VarsBelow start) :
    RecipesCausal start (sboxRecipes start value) := by
  simp only [sboxRecipes, RecipesCausal, Expr.VarsBelow]
  refine ⟨⟨hvalue, hvalue⟩, ?_⟩
  refine ⟨⟨by omega, by omega⟩, ?_⟩
  refine ⟨⟨by omega, by omega⟩, ?_⟩
  exact ⟨⟨by omega, Expr.VarsBelow.mono value hvalue (by omega)⟩, trivial⟩

theorem sboxRows_sound (env : Env) (start : Nat) (value : Expr)
    (rows : ConstraintsHold env
      (recipeConstraints start (sboxRecipes start value))) :
    (sboxOutput start).eval env = Layer.sboxF (value.eval env) := by
  have row0 := rows (Expr.var start - value * value) (by
    simp [sboxRecipes, recipeConstraints])
  have row1 := rows
    (Expr.var (start + 1) - Expr.var start * Expr.var start) (by
      simp [sboxRecipes, recipeConstraints])
  have row2 := rows
    (Expr.var (start + 2) - Expr.var (start + 1) * Expr.var start) (by
      simp [sboxRecipes, recipeConstraints])
  have row3 := rows
    (Expr.var (start + 3) - Expr.var (start + 2) * value) (by
      simp [sboxRecipes, recipeConstraints])
  have value0 : env start = value.eval env * value.eval env :=
    sub_eq_zero.mp (by simpa only [Expr.eval_sub, Expr.eval_var,
      Expr.eval_hmul] using row0)
  have value1 : env (start + 1) = env start * env start :=
    sub_eq_zero.mp (by simpa only [Expr.eval_sub, Expr.eval_var,
      Expr.eval_hmul] using row1)
  have value2 : env (start + 2) = env (start + 1) * env start :=
    sub_eq_zero.mp (by simpa only [Expr.eval_sub, Expr.eval_var,
      Expr.eval_hmul] using row2)
  have value3 : env (start + 3) = env (start + 2) * value.eval env :=
    sub_eq_zero.mp (by simpa only [Expr.eval_sub, Expr.eval_var,
      Expr.eval_hmul] using row3)
  simp only [sboxOutput, Expr.eval_var]
  rw [value3, value2, value1, value0]
  rfl

/-- Shared S-box staging for a finite input list. -/
structure SboxProgram where
  recipes : List Expr
  outputs : List Expr

def compileSboxes : Nat → List Expr → SboxProgram
  | _, [] => ⟨[], []⟩
  | start, value :: rest =>
      let tail := compileSboxes (start + 4) rest
      ⟨sboxRecipes start value ++ tail.recipes,
        sboxOutput start :: tail.outputs⟩

@[simp] theorem compileSboxes_recipes_length (start : Nat)
    (values : List Expr) :
    (compileSboxes start values).recipes.length = values.length * 4 := by
  induction values generalizing start with
  | nil => rfl
  | cons value rest ih =>
      simp [compileSboxes, ih]
      omega

@[simp] theorem compileSboxes_outputs_length (start : Nat)
    (values : List Expr) :
    (compileSboxes start values).outputs.length = values.length := by
  induction values generalizing start with
  | nil => rfl
  | cons value rest ih => simp [compileSboxes, ih]

theorem compileSboxes_causal (start : Nat) (values : List Expr)
    (hvalues : ∀ value ∈ values, value.VarsBelow start) :
    RecipesCausal start (compileSboxes start values).recipes := by
  induction values generalizing start with
  | nil => trivial
  | cons value rest ih =>
      apply recipesCausal_append_causal
      · exact sboxRecipes_causal start value (hvalues value (by simp))
      · simpa using ih (start + 4) (by
          intro current member
          exact Expr.VarsBelow.mono current
            (hvalues current (by simp [member])) (by omega))

theorem compileSboxes_outputs_below (start : Nat) (values : List Expr)
    (output : Expr) (member : output ∈ (compileSboxes start values).outputs) :
    output.VarsBelow (start + (compileSboxes start values).recipes.length) := by
  induction values generalizing start with
  | nil => simp [compileSboxes] at member
  | cons value rest ih =>
      simp only [compileSboxes, List.mem_cons] at member
      rcases member with rfl | member
      · simp [sboxOutput, Expr.VarsBelow]
        omega
      · have tail := ih (start + 4) member
        convert tail using 1 <;>
          simp only [compileSboxes, List.length_append, sboxRecipes_length,
            compileSboxes_recipes_length] <;> omega

theorem compileSboxes_sound (env : Env) (start : Nat) (values : List Expr)
    (rows : ConstraintsHold env
      (recipeConstraints start (compileSboxes start values).recipes)) :
    (compileSboxes start values).outputs.map (Expr.eval env) =
      values.map (fun value => Layer.sboxF (value.eval env)) := by
  induction values generalizing start with
  | nil => rfl
  | cons value rest ih =>
      have splitRows :
          ConstraintsHold env (recipeConstraints start (sboxRecipes start value)) ∧
          ConstraintsHold env (recipeConstraints (start + 4)
            (compileSboxes (start + 4) rest).recipes) := by
        rw [compileSboxes, recipeConstraints_append] at rows
        have separated := (constraintsHold_append env _ _).mp rows
        simpa using separated
      have head := sboxRows_sound env start value splitRows.1
      have tail := ih (start + 4) splitRows.2
      simp [compileSboxes, head, tail]

/-- One canonical straight-line permutation program. -/
structure Program where
  recipes : List Expr
  output : EState

def fullInputs (rows : List (List Nat)) (round : Nat)
    (state : EState) : List Expr :=
  List.ofFn fun lane =>
    state lane + Expr.const (Spec.Poseidon2.constantAt rows round lane.val)

def fullSboxState (start : Nat) (rows : List (List Nat)) (round : Nat)
    (state : EState) : EState :=
  fun lane => (compileSboxes start (fullInputs rows round state)).outputs.getD
    lane.val 0

def partialInput (round : Nat) (state : EState) : Expr :=
  state 0 + Expr.const (Spec.Poseidon2.ofNat
    (Spec.Poseidon2.internalConstants.getD round 0))

def partialSboxState (start : Nat) (round : Nat) (state : EState) : EState :=
  fun lane => if lane.val = 0 then
    (compileSboxes start [partialInput round state]).outputs.getD 0 0
  else state lane

def stepSize : Step → Nat
  | .initialLayer => 8
  | .initialFullRound _ | .terminalFullRound _ => 40
  | .partialRound _ => 12

def stepRecipes (start : Nat) : Step → EState → List Expr
  | .initialLayer, state => List.ofFn (Layer.externalE state)
  | .initialFullRound round, state =>
      let sboxes := compileSboxes start
        (fullInputs Spec.Poseidon2.initialConstants round state)
      sboxes.recipes ++ List.ofFn
        (Layer.externalE (fullSboxState start
          Spec.Poseidon2.initialConstants round state))
  | .terminalFullRound round, state =>
      let sboxes := compileSboxes start
        (fullInputs Spec.Poseidon2.terminalConstants round state)
      sboxes.recipes ++ List.ofFn
        (Layer.externalE (fullSboxState start
          Spec.Poseidon2.terminalConstants round state))
  | .partialRound round, state =>
      let sboxes := compileSboxes start [partialInput round state]
      sboxes.recipes ++ List.ofFn
        (Layer.internalE (partialSboxState start round state))

def stepOutput (start : Nat) : Step → EState
  | .initialLayer => freshState start
  | .initialFullRound _ | .terminalFullRound _ => freshState (start + 32)
  | .partialRound _ => freshState (start + 4)

@[simp] theorem stepRecipes_length (start : Nat) (step : Step) (state : EState) :
    (stepRecipes start step state).length = stepSize step := by
  cases step <;> simp [stepRecipes, stepSize, fullInputs]

def compile (start : Nat) (state : EState) : List Step → Program
  | [] => ⟨[], state⟩
  | step :: rest =>
      let recipes := stepRecipes start step state
      let nextState := stepOutput start step
      let tail := compile (start + stepSize step) nextState rest
      ⟨recipes ++ tail.recipes, tail.output⟩

/-- Executable projection of the fixed production schedule's output lanes.
The complete compiler still owns and checks all 592 internal recipes. -/
def scheduleOutput (start : Nat) : EState := freshState (start + 584)

theorem scheduleOutput_eq_compile (start : Nat) (state : EState) :
    scheduleOutput start = (compile start state schedule).output := by
  funext lane
  rfl

def scheduleSize (steps : List Step) : Nat :=
  (steps.map stepSize).sum

@[simp] theorem compile_recipes_length (start : Nat) (state : EState)
    (steps : List Step) :
    (compile start state steps).recipes.length = scheduleSize steps := by
  induction steps generalizing start state with
  | nil => rfl
  | cons step rest ih =>
      simp [compile, scheduleSize, ih]

theorem stateRows_sound (env : Env) (start : Nat) (recipes : EState)
    (rows : ConstraintsHold env
      (recipeConstraints start (List.ofFn recipes))) :
    Layer.evalState env (freshState start) = Layer.evalState env recipes := by
  have rowMember (lane : Fin 8) :
      Expr.var (start + lane.val) - recipes lane ∈
        recipeConstraints start (List.ofFn recipes) := by
    fin_cases lane <;> simp [recipeConstraints, List.ofFn_succ]
  funext lane
  have equation := rows _ (rowMember lane)
  change env (start + lane.val) = (recipes lane).eval env
  exact sub_eq_zero.mp (by
    simpa only [Expr.eval_sub, Expr.eval_var] using equation)

theorem fullSboxState_sound (env : Env) (start : Nat)
    (rows : List (List Nat)) (round : Nat) (state : EState)
    (sboxRows : ConstraintsHold env (recipeConstraints start
      (compileSboxes start (fullInputs rows round state)).recipes)) :
    Layer.evalState env (fullSboxState start rows round state) =
      (fun lane => Layer.sboxF
        ((state lane + Expr.const
          (Spec.Poseidon2.constantAt rows round lane.val)).eval env)) := by
  have all := compileSboxes_sound env start (fullInputs rows round state) sboxRows
  funext lane
  have selected := congrArg (fun values : List F => values.getD lane.val 0) all
  fin_cases lane <;>
    simpa [fullSboxState, fullInputs, Layer.evalState, List.ofFn_succ] using selected

theorem partialSboxState_sound (env : Env) (start round : Nat) (state : EState)
    (sboxRows : ConstraintsHold env (recipeConstraints start
      (compileSboxes start [partialInput round state]).recipes)) :
    Layer.evalState env (partialSboxState start round state) =
      (fun lane => if lane.val = 0 then
        Layer.sboxF ((partialInput round state).eval env)
      else (state lane).eval env) := by
  have all := compileSboxes_sound env start [partialInput round state] sboxRows
  funext lane
  by_cases hzero : lane.val = 0
  · have selected := congrArg (fun values : List F => values.getD 0 0) all
    simp [partialSboxState, Layer.evalState, hzero] at selected ⊢
    exact selected
  · simp [partialSboxState, Layer.evalState, hzero]

theorem stepRecipes_causal (start : Nat) (step : Step) (state : EState)
    (hstate : ∀ lane, (state lane).VarsBelow start) :
    RecipesCausal start (stepRecipes start step state) := by
  cases step with
  | initialLayer =>
      exact recipesCausal_of_all_below start _ (by
        simp only [stepRecipes]
        rw [List.forall_mem_ofFn_iff]
        exact fun lane => varsBelow_externalE state hstate lane)
  | initialFullRound round =>
      have inputsBelow : ∀ value ∈ fullInputs
          Spec.Poseidon2.initialConstants round state, value.VarsBelow start := by
        unfold fullInputs
        rw [List.forall_mem_ofFn_iff]
        intro lane
        simp [fullInputs, Expr.VarsBelow, hstate]
      have sboxes := compileSboxes_causal start _ inputsBelow
      apply recipesCausal_append
      · exact sboxes
      · intro expression member
        rw [List.mem_ofFn'] at member
        rcases member with ⟨lane, rfl⟩
        apply varsBelow_externalE
        intro current
        unfold fullSboxState
        apply compileSboxes_outputs_below start _
        rw [List.getD_eq_get (compileSboxes start
          (fullInputs Spec.Poseidon2.initialConstants round state)).outputs 0
          ⟨current.val, by simp [fullInputs]⟩]
        exact List.get_mem _ _
  | terminalFullRound round =>
      have inputsBelow : ∀ value ∈ fullInputs
          Spec.Poseidon2.terminalConstants round state, value.VarsBelow start := by
        unfold fullInputs
        rw [List.forall_mem_ofFn_iff]
        intro lane
        simp [fullInputs, Expr.VarsBelow, hstate]
      have sboxes := compileSboxes_causal start _ inputsBelow
      apply recipesCausal_append
      · exact sboxes
      · intro expression member
        rw [List.mem_ofFn'] at member
        rcases member with ⟨lane, rfl⟩
        apply varsBelow_externalE
        intro current
        unfold fullSboxState
        apply compileSboxes_outputs_below start _
        rw [List.getD_eq_get (compileSboxes start
          (fullInputs Spec.Poseidon2.terminalConstants round state)).outputs 0
          ⟨current.val, by simp [fullInputs]⟩]
        exact List.get_mem _ _
  | partialRound round =>
      have inputBelow : (partialInput round state).VarsBelow start := by
        simp [partialInput, Expr.VarsBelow, hstate]
      have sboxes := compileSboxes_causal start [partialInput round state] (by
        intro value member
        simp only [List.mem_singleton] at member
        subst value
        exact inputBelow)
      apply recipesCausal_append
      · exact sboxes
      · intro expression member
        rw [List.mem_ofFn'] at member
        rcases member with ⟨lane, rfl⟩
        apply varsBelow_internalE
        intro current
        by_cases hzero : current.val = 0
        · simp [partialSboxState, hzero]
          apply compileSboxes_outputs_below start [partialInput round state]
          simp [compileSboxes]
        · simp only [partialSboxState, hzero, if_false]
          exact Expr.VarsBelow.mono (state current) (hstate current) (by omega)

theorem compile_causal (start : Nat) (state : EState) (steps : List Step)
    (hstate : ∀ lane, (state lane).VarsBelow start) :
    RecipesCausal start (compile start state steps).recipes := by
  induction steps generalizing start state with
  | nil => trivial
  | cons step rest ih =>
      apply recipesCausal_append_causal
      · exact stepRecipes_causal start step state hstate
      · have hnext : ∀ lane,
            (stepOutput start step lane).VarsBelow (start + stepSize step) := by
          cases step <;> intro lane <;>
            simp [stepOutput, stepSize, freshState, Expr.VarsBelow] <;> omega
        simpa [stepRecipes_length] using
          ih (start + stepSize step) (stepOutput start step) hnext

theorem compile_output_varsBelow (start : Nat) (state : EState)
    (steps : List Step)
    (hstate : ∀ lane, (state lane).VarsBelow start) (lane : Fin 8) :
    ((compile start state steps).output lane).VarsBelow
      (start + (compile start state steps).recipes.length) := by
  induction steps generalizing start state with
  | nil => simpa [compile] using hstate lane
  | cons step rest ih =>
      have hnext : ∀ current,
          (stepOutput start step current).VarsBelow (start + stepSize step) := by
        cases step <;> intro current <;>
          simp [stepOutput, stepSize, freshState, Expr.VarsBelow] <;> omega
      have tail := ih (start + stepSize step) (stepOutput start step) hnext
      convert tail using 1 <;>
        simp only [compile, List.length_append, stepRecipes_length,
          compile_recipes_length] <;> omega

theorem stepRows_sound (env : Env) (start : Nat) (step : Step)
    (state : EState)
    (rows : ConstraintsHold env
      (recipeConstraints start (stepRecipes start step state))) :
    Layer.evalState env (stepOutput start step) =
      applyF step (Layer.evalState env state) := by
  cases step with
  | initialLayer =>
      calc
        Layer.evalState env (freshState start) =
            Layer.evalState env (Layer.externalE state) :=
          stateRows_sound env start _ rows
        _ = Layer.externalF (Layer.evalState env state) := by
          funext lane
          exact Layer.eval_externalE env state lane
  | initialFullRound round =>
      let inputs := fullInputs Spec.Poseidon2.initialConstants round state
      let sboxes := compileSboxes start inputs
      have splitRows :
          ConstraintsHold env (recipeConstraints start sboxes.recipes) ∧
          ConstraintsHold env (recipeConstraints (start + 32)
            (List.ofFn (Layer.externalE (fullSboxState start
              Spec.Poseidon2.initialConstants round state)))) := by
        rw [stepRecipes, recipeConstraints_append] at rows
        have separated := (constraintsHold_append env _ _).mp rows
        simpa [inputs, sboxes] using separated
      have staged := fullSboxState_sound env start
        Spec.Poseidon2.initialConstants round state splitRows.1
      calc
        Layer.evalState env (freshState (start + 32)) =
            Layer.evalState env (Layer.externalE (fullSboxState start
              Spec.Poseidon2.initialConstants round state)) :=
          stateRows_sound env (start + 32) _ splitRows.2
        _ = Layer.externalF (Layer.evalState env (fullSboxState start
              Spec.Poseidon2.initialConstants round state)) := by
          funext lane
          exact Layer.eval_externalE env _ lane
        _ = Layer.fullF Spec.Poseidon2.initialConstants round
              (Layer.evalState env state) := by
          unfold Layer.fullF
          rw [staged]
          congr 1
  | terminalFullRound round =>
      let inputs := fullInputs Spec.Poseidon2.terminalConstants round state
      let sboxes := compileSboxes start inputs
      have splitRows :
          ConstraintsHold env (recipeConstraints start sboxes.recipes) ∧
          ConstraintsHold env (recipeConstraints (start + 32)
            (List.ofFn (Layer.externalE (fullSboxState start
              Spec.Poseidon2.terminalConstants round state)))) := by
        rw [stepRecipes, recipeConstraints_append] at rows
        have separated := (constraintsHold_append env _ _).mp rows
        simpa [inputs, sboxes] using separated
      have staged := fullSboxState_sound env start
        Spec.Poseidon2.terminalConstants round state splitRows.1
      calc
        Layer.evalState env (freshState (start + 32)) =
            Layer.evalState env (Layer.externalE (fullSboxState start
              Spec.Poseidon2.terminalConstants round state)) :=
          stateRows_sound env (start + 32) _ splitRows.2
        _ = Layer.externalF (Layer.evalState env (fullSboxState start
              Spec.Poseidon2.terminalConstants round state)) := by
          funext lane
          exact Layer.eval_externalE env _ lane
        _ = Layer.fullF Spec.Poseidon2.terminalConstants round
              (Layer.evalState env state) := by
          unfold Layer.fullF
          rw [staged]
          congr 1
  | partialRound round =>
      let sboxes := compileSboxes start [partialInput round state]
      have splitRows :
          ConstraintsHold env (recipeConstraints start sboxes.recipes) ∧
          ConstraintsHold env (recipeConstraints (start + 4)
            (List.ofFn (Layer.internalE
              (partialSboxState start round state)))) := by
        rw [stepRecipes, recipeConstraints_append] at rows
        have separated := (constraintsHold_append env _ _).mp rows
        simpa [sboxes] using separated
      have staged := partialSboxState_sound env start round state splitRows.1
      calc
        Layer.evalState env (freshState (start + 4)) =
            Layer.evalState env (Layer.internalE
              (partialSboxState start round state)) :=
          stateRows_sound env (start + 4) _ splitRows.2
        _ = Layer.internalF (Layer.evalState env
              (partialSboxState start round state)) := by
          funext lane
          exact Layer.eval_internalE env _ lane
        _ = Layer.partialF round (Layer.evalState env state) := by
          unfold Layer.partialF
          rw [staged]
          congr 1
          funext lane
          by_cases hzero : lane.val = 0
          · have laneEq : lane = 0 := Fin.ext hzero
            subst lane
            simp [partialInput, Layer.evalState]
          · simp [Layer.evalState, hzero]

theorem compile_sound (env : Env) (start : Nat) (state : EState)
    (steps : List Step)
    (hrows : ConstraintsHold env
      (recipeConstraints start (compile start state steps).recipes)) :
    Layer.evalState env (compile start state steps).output =
      runF steps (Layer.evalState env state) := by
  induction steps generalizing start state with
  | nil => rfl
  | cons step rest ih =>
      have splitRows :
          ConstraintsHold env (recipeConstraints start
            (stepRecipes start step state)) ∧
          ConstraintsHold env (recipeConstraints (start + stepSize step)
            (compile (start + stepSize step) (stepOutput start step) rest).recipes) := by
        rw [compile, recipeConstraints_append] at hrows
        have separated := (constraintsHold_append env _ _).mp hrows
        simpa using separated
      have headSound := stepRows_sound env start step state splitRows.1
      have tailSound := ih (start + stepSize step) (stepOutput start step) splitRows.2
      simpa [compile, runF, headSound] using tailSound

theorem runF_eq_reference (steps : List Step) (state : FState) :
    List.ofFn (runF steps state) =
      runReference steps (List.ofFn state) := by
  induction steps generalizing state with
  | nil => rfl
  | cons step rest ih =>
      rw [runF, runReference, ih, applyF_eq_reference]

theorem runReference_append (first second : List Step)
    (state : Spec.Poseidon2.State) :
    runReference (first ++ second) state =
      runReference second (runReference first state) := by
  induction first generalizing state with
  | nil => rfl
  | cons step rest ih =>
      simp only [List.cons_append, runReference]
      exact ih (applyReference step state)

theorem runReference_initialRounds (rounds : List Nat)
    (state : Spec.Poseidon2.State) :
    runReference (rounds.map Step.initialFullRound) state =
      rounds.foldl (fun current round =>
        Spec.Poseidon2.fullRound Spec.Poseidon2.initialConstants round current)
        state := by
  induction rounds generalizing state with
  | nil => rfl
  | cons round rest ih =>
      simp only [List.map_cons, runReference, List.foldl_cons, applyReference]
      exact ih _

theorem runReference_partialRounds (rounds : List Nat)
    (state : Spec.Poseidon2.State) :
    runReference (rounds.map Step.partialRound) state =
      rounds.foldl (fun current round =>
        Spec.Poseidon2.partialRound round current) state := by
  induction rounds generalizing state with
  | nil => rfl
  | cons round rest ih =>
      simp only [List.map_cons, runReference, List.foldl_cons, applyReference]
      exact ih _

theorem runReference_terminalRounds (rounds : List Nat)
    (state : Spec.Poseidon2.State) :
    runReference (rounds.map Step.terminalFullRound) state =
      rounds.foldl (fun current round =>
        Spec.Poseidon2.fullRound Spec.Poseidon2.terminalConstants round current)
        state := by
  induction rounds generalizing state with
  | nil => rfl
  | cons round rest ih =>
      simp only [List.map_cons, runReference, List.foldl_cons, applyReference]
      exact ih _

theorem runReference_schedule (state : Spec.Poseidon2.State) :
    runReference schedule state = Spec.Poseidon2.permute state := by
  rw [show schedule =
      [Step.initialLayer] ++
        (List.range Spec.Poseidon2.halfFullRounds).map Step.initialFullRound ++
        (List.range Spec.Poseidon2.partialRounds).map Step.partialRound ++
        (List.range Spec.Poseidon2.halfFullRounds).map Step.terminalFullRound
      by rfl]
  repeat' rw [runReference_append]
  rw [runReference_initialRounds, runReference_partialRounds,
    runReference_terminalRounds]
  rfl

/-- Every environment that satisfies the generated permutation rows carries
the exact executable Poseidon2 output in the program's final variables. -/
theorem compile_schedule_sound (env : Env) (start : Nat) (state : EState)
    (hrows : ConstraintsHold env
      (recipeConstraints start (compile start state schedule).recipes)) :
    List.ofFn (Layer.evalState env (compile start state schedule).output) =
      Spec.Poseidon2.permute (List.ofFn (Layer.evalState env state)) := by
  calc
    List.ofFn (Layer.evalState env (compile start state schedule).output) =
        List.ofFn (runF schedule (Layer.evalState env state)) :=
      congrArg List.ofFn (compile_sound env start state schedule hrows)
    _ = runReference schedule (List.ofFn (Layer.evalState env state)) :=
      runF_eq_reference schedule (Layer.evalState env state)
    _ = Spec.Poseidon2.permute (List.ofFn (Layer.evalState env state)) :=
      runReference_schedule _

/-- Honest witness execution is causal for every input state stored before the
permutation's local range. -/
theorem compile_schedule_causal (start : Nat) (state : EState)
    (hstate : ∀ lane, (state lane).VarsBelow start) :
    RecipesCausal start (compile start state schedule).recipes :=
  compile_causal start state schedule hstate

/-- The staged logical permutation owns 592 shared-intermediate recipes. -/
theorem compile_schedule_recipe_count (start : Nat) (state : EState) :
    (compile start state schedule).recipes.length = 592 := by
  rw [compile_recipes_length]
  rfl

end NightstreamFPrime.Gadgets.Poseidon2.Permutation
