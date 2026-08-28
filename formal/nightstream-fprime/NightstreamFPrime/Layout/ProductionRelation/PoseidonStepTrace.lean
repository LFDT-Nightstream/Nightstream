import NightstreamFPrime.Gadgets.Poseidon2.Permutation

/-!
Owns the exact fixed-step trace selected by the production Poseidon2 rewrite.
Each step exposes its S-box inputs and outputs plus its eight final affine
output recipes. The trace is definitionally derived from the canonical
permutation builder.

This module proves source-to-trace soundness and exact replacement row counts.
It does not construct final sparse low-norm forms.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.PoseidonStepTrace

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2

abbrev Step := Permutation.Step
abbrev EState := Permutation.EState

/-- S-box inputs owned by one fixed permutation step. -/
def sboxInputs : Step → EState → List Expr
  | .initialLayer, _ => []
  | .initialFullRound round, state =>
      Permutation.fullInputs Poseidon2.initialConstants round state
  | .partialRound round, state => [Permutation.partialInput round state]
  | .terminalFullRound round, state =>
      Permutation.fullInputs Poseidon2.terminalConstants round state

def sboxProgram (start : Nat) (step : Step) (state : EState) :
    Permutation.SboxProgram :=
  Permutation.compileSboxes start (sboxInputs step state)

/-- Eight affine output recipes after the S-box stage. -/
def outputExpressions (start : Nat) : Step → EState → EState
  | .initialLayer, state => Layer.externalE state
  | .initialFullRound round, state =>
      Layer.externalE (Permutation.fullSboxState start
        Poseidon2.initialConstants round state)
  | .partialRound round, state =>
      Layer.internalE (Permutation.partialSboxState start round state)
  | .terminalFullRound round, state =>
      Layer.externalE (Permutation.fullSboxState start
        Poseidon2.terminalConstants round state)

/-- First source column of the eight final output recipes. -/
def outputStart (start : Nat) : Step → Nat
  | .initialLayer => start
  | .initialFullRound _ | .terminalFullRound _ => start + 32
  | .partialRound _ => start + 4

theorem stepRecipes_eq (start : Nat) (step : Step) (state : EState) :
    Permutation.stepRecipes start step state =
      (sboxProgram start step state).recipes ++
        List.ofFn (outputExpressions start step state) := by
  cases step <;> rfl

theorem stepOutput_eq (start : Nat) (step : Step) :
    Permutation.stepOutput start step =
      Permutation.freshState (outputStart start step) := by
  cases step <;> rfl

/-- Direct selective rows emitted for one source step. -/
def directRowCount (step : Step) : Nat :=
  match step with
  | .initialLayer => 8
  | .initialFullRound _ | .terminalFullRound _ => 16
  | .partialRound _ => 9

@[simp] theorem directRowCount_initialLayer :
    directRowCount .initialLayer = 8 := by
  rfl

@[simp] theorem directRowCount_initialFullRound (round : Nat) :
    directRowCount (.initialFullRound round) = 16 := by
  rfl

@[simp] theorem directRowCount_partialRound (round : Nat) :
    directRowCount (.partialRound round) = 9 := by
  rfl

@[simp] theorem directRowCount_terminalFullRound (round : Nat) :
    directRowCount (.terminalFullRound round) = 16 := by
  rfl

/-- The complete fixed production permutation rewrites to exactly 334
selective rows, instead of the 592 source recipe rows. -/
theorem schedule_directRowCount :
    (Permutation.schedule.map directRowCount).sum = 334 := by
  rfl

/-- Exact trace equations retained from one source step. -/
def Holds (env : Env) (start : Nat) (step : Step) (state : EState) : Prop :=
  (sboxProgram start step state).outputs.map (Expr.eval env) =
      (sboxInputs step state).map (fun input => Layer.sboxF (input.eval env)) ∧
    Layer.evalState env (Permutation.stepOutput start step) =
      Layer.evalState env (outputExpressions start step state)

/-- Every source recipe row of one fixed step implies the exact direct trace
selected by the production rewrite. -/
theorem rows_imply_holds (env : Env) (start : Nat) (step : Step)
    (state : EState)
    (rows : ConstraintsHold env
      (recipeConstraints start (Permutation.stepRecipes start step state))) :
    Holds env start step state := by
  cases step with
  | initialLayer =>
      constructor
      · rfl
      · simpa [Holds, stepOutput_eq, outputExpressions] using
          Permutation.stateRows_sound env start (Layer.externalE state) rows
  | initialFullRound round =>
      have splitRows :
          ConstraintsHold env (recipeConstraints start
            (sboxProgram start (.initialFullRound round) state).recipes) ∧
          ConstraintsHold env (recipeConstraints (start + 32)
            (List.ofFn (outputExpressions start
              (.initialFullRound round) state))) := by
        rw [stepRecipes_eq, Permutation.recipeConstraints_append] at rows
        simpa using (constraintsHold_append env _ _).mp rows
      constructor
      · exact Permutation.compileSboxes_sound env start _ splitRows.1
      · rw [stepOutput_eq]
        exact Permutation.stateRows_sound env (start + 32) _ splitRows.2
  | partialRound round =>
      have splitRows :
          ConstraintsHold env (recipeConstraints start
            (sboxProgram start (.partialRound round) state).recipes) ∧
          ConstraintsHold env (recipeConstraints (start + 4)
            (List.ofFn (outputExpressions start (.partialRound round) state))) := by
        rw [stepRecipes_eq, Permutation.recipeConstraints_append] at rows
        simpa using (constraintsHold_append env _ _).mp rows
      constructor
      · exact Permutation.compileSboxes_sound env start _ splitRows.1
      · rw [stepOutput_eq]
        exact Permutation.stateRows_sound env (start + 4) _ splitRows.2
  | terminalFullRound round =>
      have splitRows :
          ConstraintsHold env (recipeConstraints start
            (sboxProgram start (.terminalFullRound round) state).recipes) ∧
          ConstraintsHold env (recipeConstraints (start + 32)
            (List.ofFn (outputExpressions start
              (.terminalFullRound round) state))) := by
        rw [stepRecipes_eq, Permutation.recipeConstraints_append] at rows
        simpa using (constraintsHold_append env _ _).mp rows
      constructor
      · exact Permutation.compileSboxes_sound env start _ splitRows.1
      · rw [stepOutput_eq]
        exact Permutation.stateRows_sound env (start + 32) _ splitRows.2

private theorem fullSboxState_of_holds (env : Env) (start : Nat)
    (rows : List (List Nat)) (round : Nat) (state : EState)
    (holds :
      (Permutation.compileSboxes start
          (Permutation.fullInputs rows round state)).outputs.map (Expr.eval env) =
        (Permutation.fullInputs rows round state).map
          (fun input => Layer.sboxF (input.eval env))) :
    Layer.evalState env (Permutation.fullSboxState start rows round state) =
      fun lane => Layer.sboxF
        ((state lane + Expr.const
          (Poseidon2.constantAt rows round lane.val)).eval env) := by
  funext lane
  have selected := congrArg (fun values : List F => values.getD lane.val 0) holds
  fin_cases lane <;>
    simpa [Permutation.fullSboxState, Permutation.fullInputs,
      Layer.evalState, List.ofFn_succ] using selected

private theorem partialSboxState_of_holds (env : Env) (start round : Nat)
    (state : EState)
    (holds :
      (Permutation.compileSboxes start
          [Permutation.partialInput round state]).outputs.map (Expr.eval env) =
        [Permutation.partialInput round state].map
          (fun input => Layer.sboxF (input.eval env))) :
    Layer.evalState env (Permutation.partialSboxState start round state) =
      fun lane => if lane.val = 0 then
        Layer.sboxF ((Permutation.partialInput round state).eval env)
      else (state lane).eval env := by
  funext lane
  by_cases zero : lane.val = 0
  · have selected := congrArg (fun values : List F => values.getD 0 0) holds
    simp [Permutation.partialSboxState, Layer.evalState, zero] at selected ⊢
    exact selected
  · simp [Permutation.partialSboxState, Layer.evalState, zero]

/-- The retained direct trace alone implies the exact mathematical Poseidon2
step. Removed `x²`, `x⁴`, and `x⁶` source columns are not assumptions. -/
theorem holds_implies_step_sound (env : Env) (start : Nat) (step : Step)
    (state : EState) (holds : Holds env start step state) :
    Layer.evalState env (Permutation.stepOutput start step) =
      Permutation.applyF step (Layer.evalState env state) := by
  rcases holds with ⟨sboxes, output⟩
  cases step with
  | initialLayer =>
      calc
        Layer.evalState env
            (Permutation.stepOutput start .initialLayer) =
            Layer.evalState env (outputExpressions start .initialLayer state) :=
          output
        _ = Layer.externalF (Layer.evalState env state) := by
          funext lane
          exact Layer.eval_externalE env state lane
  | initialFullRound round =>
      have staged := fullSboxState_of_holds env start
        Poseidon2.initialConstants round state sboxes
      calc
        Layer.evalState env
            (Permutation.stepOutput start (.initialFullRound round)) =
            Layer.evalState env
              (outputExpressions start (.initialFullRound round) state) :=
          output
        _ = Layer.externalF (Layer.evalState env
              (Permutation.fullSboxState start
                Poseidon2.initialConstants round state)) := by
          funext lane
          exact Layer.eval_externalE env _ lane
        _ = Layer.fullF Poseidon2.initialConstants round
              (Layer.evalState env state) := by
          unfold Layer.fullF
          rw [staged]
          congr 1
  | partialRound round =>
      have staged := partialSboxState_of_holds env start round state sboxes
      calc
        Layer.evalState env
            (Permutation.stepOutput start (.partialRound round)) =
            Layer.evalState env
              (outputExpressions start (.partialRound round) state) :=
          output
        _ = Layer.internalF (Layer.evalState env
              (Permutation.partialSboxState start round state)) := by
          funext lane
          exact Layer.eval_internalE env _ lane
        _ = Layer.partialF round (Layer.evalState env state) := by
          unfold Layer.partialF
          rw [staged]
          congr 1
          funext lane
          by_cases zero : lane.val = 0
          · have laneZero : lane = 0 := Fin.ext zero
            subst lane
            simp [Permutation.partialInput, Layer.evalState]
          · simp [Layer.evalState, zero]
  | terminalFullRound round =>
      have staged := fullSboxState_of_holds env start
        Poseidon2.terminalConstants round state sboxes
      calc
        Layer.evalState env
            (Permutation.stepOutput start (.terminalFullRound round)) =
            Layer.evalState env
              (outputExpressions start (.terminalFullRound round) state) :=
          output
        _ = Layer.externalF (Layer.evalState env
              (Permutation.fullSboxState start
                Poseidon2.terminalConstants round state)) := by
          funext lane
          exact Layer.eval_externalE env _ lane
        _ = Layer.fullF Poseidon2.terminalConstants round
              (Layer.evalState env state) := by
          unfold Layer.fullF
          rw [staged]
          congr 1

/-- Canonical honest source environment for one fixed step. -/
def honestEnv (env : Env) (start : Nat) (step : Step) (state : EState) : Env :=
  executeRecipes env start (Permutation.stepRecipes start step state)

/-- Honest trace generation preserves every caller-owned input below the
step's local range. -/
theorem honestEnv_agrees_below (env : Env) (start : Nat) (step : Step)
    (state : EState) (index : Nat) (below : index < start) :
    honestEnv env start step state index = env index :=
  executeRecipes_agrees_below env start
    (Permutation.stepRecipes start step state) index below

/-- Canonical causal execution satisfies the retained direct trace. -/
theorem honest_holds (env : Env) (start : Nat) (step : Step) (state : EState)
    (stateBelow : ∀ lane, (state lane).VarsBelow start) :
    Holds (honestEnv env start step state) start step state := by
  apply rows_imply_holds
  exact executeRecipes_holds_recipeConstraints env start
    (Permutation.stepRecipes start step state)
    (Permutation.stepRecipes_causal start step state stateBelow)

/-- Honest completeness of one direct step trace, with exact input-prefix
preservation. -/
theorem exists_holds (env : Env) (start : Nat) (step : Step) (state : EState)
    (stateBelow : ∀ lane, (state lane).VarsBelow start) :
    ∃ completed,
      (∀ index, index < start → completed index = env index) ∧
        Holds completed start step state := by
  exact ⟨honestEnv env start step state,
    honestEnv_agrees_below env start step state,
    honest_holds env start step state stateBelow⟩

end NightstreamFPrime.Layout.ProductionRelation.PoseidonStepTrace
