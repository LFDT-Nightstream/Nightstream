import NightstreamFPrime.Export.Stage1.Invocations

/-!
Owns value-level transport for the fixed Poseidon2 straight-line compiler.
The proof is structural in the step list. It relates two relocated compiler
runs when their input states and local witness intervals have equal values.
-/

namespace NightstreamFPrime.Export.Stage1.PermutationCompilerTransport

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Stage1.Invocations
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1

private theorem getD_eval (env : Env) (values : List Expr) (index : Nat) :
    (values.getD index 0).eval env =
      (values.map (Expr.eval env)).getD index 0 := by
  have evalZero : (0 : Expr).eval env = (0 : NightstreamFPrime.Spec.F) := by
    rfl
  rw [← evalZero]
  exact (List.getD_map (n := index) values (0 : Expr) (Expr.eval env)).symm

private theorem compileSboxes_eval_eq (leftEnv rightEnv : Env) :
    ∀ (leftStart rightStart : Nat) (leftValues rightValues : List Expr),
      leftValues.map (Expr.eval leftEnv) =
          rightValues.map (Expr.eval rightEnv) →
      (∀ index, index < leftValues.length * 4 →
        leftEnv (leftStart + index) = rightEnv (rightStart + index)) →
      (Permutation.compileSboxes leftStart leftValues).recipes.map
          (Expr.eval leftEnv) =
        (Permutation.compileSboxes rightStart rightValues).recipes.map
          (Expr.eval rightEnv) ∧
      (Permutation.compileSboxes leftStart leftValues).outputs.map
          (Expr.eval leftEnv) =
        (Permutation.compileSboxes rightStart rightValues).outputs.map
          (Expr.eval rightEnv) := by
  intro leftStart rightStart leftValues
  induction leftValues generalizing leftStart rightStart with
  | nil =>
      intro rightValues valuesEq _
      cases rightValues with
      | nil => exact ⟨rfl, rfl⟩
      | cons value rest => simp at valuesEq
  | cons leftValue leftRest inductionHypothesis =>
      intro rightValues valuesEq localsEq
      cases rightValues with
      | nil => simp at valuesEq
      | cons rightValue rightRest =>
          simp only [List.map_cons, List.cons.injEq] at valuesEq
          have local0 := localsEq 0 (by
            simp only [List.length_cons]
            omega)
          have local1 := localsEq 1 (by
            simp only [List.length_cons]
            omega)
          have local2 := localsEq 2 (by
            simp only [List.length_cons]
            omega)
          have local3 := localsEq 3 (by
            simp only [List.length_cons]
            omega)
          have local0' : leftEnv leftStart = rightEnv rightStart := by
            simpa using local0
          have tailLocals : ∀ index, index < leftRest.length * 4 →
              leftEnv (leftStart + 4 + index) =
                rightEnv (rightStart + 4 + index) := by
            intro index bounded
            have shifted := localsEq (4 + index) (by
              simp only [List.length_cons]
              omega)
            simpa [Nat.add_assoc] using shifted
          have tail := inductionHypothesis (leftStart + 4) (rightStart + 4)
            rightRest valuesEq.2 tailLocals
          have headRecipes :
              (Permutation.sboxRecipes leftStart leftValue).map
                  (Expr.eval leftEnv) =
                (Permutation.sboxRecipes rightStart rightValue).map
                  (Expr.eval rightEnv) := by
            simp [Permutation.sboxRecipes, valuesEq.1, local0', local1, local2]
          constructor
          · change
              ((Permutation.sboxRecipes leftStart leftValue ++
                (Permutation.compileSboxes (leftStart + 4) leftRest).recipes).map
                  (Expr.eval leftEnv)) =
              ((Permutation.sboxRecipes rightStart rightValue ++
                (Permutation.compileSboxes (rightStart + 4) rightRest).recipes).map
                  (Expr.eval rightEnv))
            simp only [List.map_append]
            rw [headRecipes, tail.1]
          · change
              ((Permutation.sboxOutput leftStart ::
                (Permutation.compileSboxes (leftStart + 4) leftRest).outputs).map
                  (Expr.eval leftEnv)) =
              ((Permutation.sboxOutput rightStart ::
                (Permutation.compileSboxes (rightStart + 4) rightRest).outputs).map
                  (Expr.eval rightEnv))
            simp only [List.map_cons, List.cons.injEq]
            constructor
            · simpa [Permutation.sboxOutput] using local3
            · exact tail.2

private theorem fullStepRecipes_eval_eq (leftEnv rightEnv : Env)
    (leftStart rightStart : Nat) (rows : List (List Nat)) (round : Nat)
    (leftState rightState : Permutation.EState)
    (stateEq : Layer.evalState leftEnv leftState =
      Layer.evalState rightEnv rightState)
    (localsEq : ∀ index, index < 40 →
      leftEnv (leftStart + index) = rightEnv (rightStart + index)) :
    (let leftSboxes := Permutation.compileSboxes leftStart
        (Permutation.fullInputs rows round leftState)
     let rightSboxes := Permutation.compileSboxes rightStart
        (Permutation.fullInputs rows round rightState)
     (leftSboxes.recipes ++ List.ofFn
        (Layer.externalE
          (Permutation.fullSboxState leftStart rows round leftState))).map
          (Expr.eval leftEnv) =
       (rightSboxes.recipes ++ List.ofFn
        (Layer.externalE
          (Permutation.fullSboxState rightStart rows round rightState))).map
          (Expr.eval rightEnv)) := by
  have inputsEq :
      (Permutation.fullInputs rows round leftState).map (Expr.eval leftEnv) =
        (Permutation.fullInputs rows round rightState).map
          (Expr.eval rightEnv) := by
    unfold Permutation.fullInputs
    simp only [List.map_ofFn]
    apply congrArg List.ofFn
    funext lane
    have laneEq := congrFun stateEq lane
    change (leftState lane).eval leftEnv =
      (rightState lane).eval rightEnv at laneEq
    change (leftState lane).eval leftEnv + _ =
      (rightState lane).eval rightEnv + _
    rw [laneEq]
    rfl
  have sboxes := compileSboxes_eval_eq leftEnv rightEnv leftStart rightStart
    (Permutation.fullInputs rows round leftState)
    (Permutation.fullInputs rows round rightState) inputsEq (by
      intro index bounded
      exact localsEq index (by
        simp only [Permutation.fullInputs, List.length_ofFn] at bounded
        omega))
  have sboxStateEq :
      Layer.evalState leftEnv
          (Permutation.fullSboxState leftStart rows round leftState) =
        Layer.evalState rightEnv
          (Permutation.fullSboxState rightStart rows round rightState) := by
    funext lane
    unfold Layer.evalState Permutation.fullSboxState
    rw [getD_eval, getD_eval, sboxes.2]
  simp only [List.map_append]
  rw [sboxes.1]
  congr 1
  simp only [List.map_ofFn]
  apply congrArg List.ofFn
  funext lane
  simp [sboxStateEq]

private theorem partialStepRecipes_eval_eq (leftEnv rightEnv : Env)
    (leftStart rightStart round : Nat)
    (leftState rightState : Permutation.EState)
    (stateEq : Layer.evalState leftEnv leftState =
      Layer.evalState rightEnv rightState)
    (localsEq : ∀ index, index < 12 →
      leftEnv (leftStart + index) = rightEnv (rightStart + index)) :
    (let leftSboxes := Permutation.compileSboxes leftStart
        [Permutation.partialInput round leftState]
     let rightSboxes := Permutation.compileSboxes rightStart
        [Permutation.partialInput round rightState]
     (leftSboxes.recipes ++ List.ofFn
        (Layer.internalE
          (Permutation.partialSboxState leftStart round leftState))).map
          (Expr.eval leftEnv) =
       (rightSboxes.recipes ++ List.ofFn
        (Layer.internalE
          (Permutation.partialSboxState rightStart round rightState))).map
          (Expr.eval rightEnv)) := by
  have inputEq :
      [Permutation.partialInput round leftState].map (Expr.eval leftEnv) =
        [Permutation.partialInput round rightState].map
          (Expr.eval rightEnv) := by
    simp only [List.map_singleton, List.cons.injEq]
    constructor
    · have laneEq := congrFun stateEq (0 : Fin 8)
      change (leftState 0).eval leftEnv =
        (rightState 0).eval rightEnv at laneEq
      change (leftState 0).eval leftEnv + _ =
        (rightState 0).eval rightEnv + _
      rw [laneEq]
      rfl
    · trivial
  have sboxes := compileSboxes_eval_eq leftEnv rightEnv leftStart rightStart
    [Permutation.partialInput round leftState]
    [Permutation.partialInput round rightState] inputEq (by
      intro index bounded
      exact localsEq index (by simp at bounded ⊢; omega))
  have sboxStateEq :
      Layer.evalState leftEnv
          (Permutation.partialSboxState leftStart round leftState) =
        Layer.evalState rightEnv
          (Permutation.partialSboxState rightStart round rightState) := by
    funext lane
    unfold Layer.evalState Permutation.partialSboxState
    by_cases zero : lane.val = 0
    · simp only [zero, if_true]
      rw [getD_eval, getD_eval, sboxes.2]
    · simp only [zero, if_false]
      exact congrFun stateEq lane
  simp only [List.map_append]
  rw [sboxes.1]
  congr 1
  simp only [List.map_ofFn]
  apply congrArg List.ofFn
  funext lane
  simp [sboxStateEq]

/-- One compiler step has equal recipe values after relocation when its input
state values and complete local interval values agree. -/
theorem stepRecipes_eval_eq (leftEnv rightEnv : Env)
    (leftStart rightStart : Nat) (step : Permutation.Step)
    (leftState rightState : Permutation.EState)
    (stateEq : Layer.evalState leftEnv leftState =
      Layer.evalState rightEnv rightState)
    (localsEq : ∀ index, index < Permutation.stepSize step →
      leftEnv (leftStart + index) = rightEnv (rightStart + index)) :
    (Permutation.stepRecipes leftStart step leftState).map
        (Expr.eval leftEnv) =
      (Permutation.stepRecipes rightStart step rightState).map
        (Expr.eval rightEnv) := by
  cases step with
  | initialLayer =>
      simp only [Permutation.stepRecipes, List.map_ofFn]
      apply congrArg List.ofFn
      funext lane
      simp [stateEq]
  | initialFullRound round =>
      exact fullStepRecipes_eval_eq leftEnv rightEnv leftStart rightStart
        NightstreamFPrime.Spec.Poseidon2.initialConstants round leftState
        rightState stateEq (by
          intro index bounded
          exact localsEq index (by
            simpa [Permutation.stepSize] using bounded))
  | partialRound round =>
      exact partialStepRecipes_eval_eq leftEnv rightEnv leftStart rightStart
        round leftState rightState stateEq (by
          intro index bounded
          exact localsEq index (by
            simpa [Permutation.stepSize] using bounded))
  | terminalFullRound round =>
      exact fullStepRecipes_eval_eq leftEnv rightEnv leftStart rightStart
        NightstreamFPrime.Spec.Poseidon2.terminalConstants round leftState
        rightState stateEq (by
          intro index bounded
          exact localsEq index (by
            simpa [Permutation.stepSize] using bounded))

/-- Complete compiler recipe values are invariant under a relocation that
preserves the input-state values and every local witness value. -/
theorem compileRecipes_eval_eq (leftEnv rightEnv : Env)
    (leftStart rightStart : Nat) (leftState rightState : Permutation.EState)
    (steps : List Permutation.Step)
    (stateEq : Layer.evalState leftEnv leftState =
      Layer.evalState rightEnv rightState)
    (localsEq : ∀ index, index < Permutation.scheduleSize steps →
      leftEnv (leftStart + index) = rightEnv (rightStart + index)) :
    (Permutation.compile leftStart leftState steps).recipes.map
        (Expr.eval leftEnv) =
      (Permutation.compile rightStart rightState steps).recipes.map
        (Expr.eval rightEnv) := by
  induction steps generalizing leftStart rightStart leftState rightState with
  | nil => rfl
  | cons step rest inductionHypothesis =>
      have headEq := stepRecipes_eval_eq leftEnv rightEnv leftStart rightStart
        step leftState rightState stateEq (by
          intro index bounded
          exact localsEq index (by
            simp only [Permutation.scheduleSize, List.map_cons, List.sum_cons]
            omega))
      have nextStateEq :
          Layer.evalState leftEnv (Permutation.stepOutput leftStart step) =
            Layer.evalState rightEnv
              (Permutation.stepOutput rightStart step) := by
        funext lane
        cases step with
        | initialLayer =>
            simpa [Permutation.stepOutput, Permutation.freshState,
              Layer.evalState] using localsEq lane.val (by
                simp only [Permutation.scheduleSize, List.map_cons,
                  List.sum_cons, Permutation.stepSize]
                omega)
        | initialFullRound round =>
            simpa [Permutation.stepOutput, Permutation.freshState,
              Layer.evalState, Nat.add_assoc] using
              localsEq (32 + lane.val) (by
                simp only [Permutation.scheduleSize, List.map_cons,
                  List.sum_cons, Permutation.stepSize]
                omega)
        | partialRound round =>
            simpa [Permutation.stepOutput, Permutation.freshState,
              Layer.evalState, Nat.add_assoc] using
              localsEq (4 + lane.val) (by
                simp only [Permutation.scheduleSize, List.map_cons,
                  List.sum_cons, Permutation.stepSize]
                omega)
        | terminalFullRound round =>
            simpa [Permutation.stepOutput, Permutation.freshState,
              Layer.evalState, Nat.add_assoc] using
              localsEq (32 + lane.val) (by
                simp only [Permutation.scheduleSize, List.map_cons,
                  List.sum_cons, Permutation.stepSize]
                omega)
      have tailLocals : ∀ index,
          index < Permutation.scheduleSize rest →
          leftEnv (leftStart + Permutation.stepSize step + index) =
            rightEnv (rightStart + Permutation.stepSize step + index) := by
        intro index bounded
        change index < (rest.map Permutation.stepSize).sum at bounded
        have shifted := localsEq (Permutation.stepSize step + index) (by
          simp only [Permutation.scheduleSize, List.map_cons, List.sum_cons]
          omega)
        simpa [Nat.add_assoc] using shifted
      have tailEq := inductionHypothesis
        (leftStart + Permutation.stepSize step)
        (rightStart + Permutation.stepSize step)
        (Permutation.stepOutput leftStart step)
        (Permutation.stepOutput rightStart step) nextStateEq tailLocals
      simp only [Permutation.compile, List.map_append]
      rw [headEq, tailEq]

/-- Held recipe constraints transport across any relocation that preserves
the input-state values and every local witness value. -/
theorem compileConstraintsHold_of_transport (leftEnv rightEnv : Env)
    (leftStart rightStart : Nat) (leftState rightState : Permutation.EState)
    (steps : List Permutation.Step)
    (stateEq : Layer.evalState leftEnv leftState =
      Layer.evalState rightEnv rightState)
    (localsEq : ∀ index, index < Permutation.scheduleSize steps →
      leftEnv (leftStart + index) = rightEnv (rightStart + index))
    (rightHolds : ConstraintsHold rightEnv
      (recipeConstraints rightStart
        (Permutation.compile rightStart rightState steps).recipes)) :
    ConstraintsHold leftEnv
      (recipeConstraints leftStart
        (Permutation.compile leftStart leftState steps).recipes) := by
  apply recipeConstraints_hold_of_values
  intro index leftBound
  have rightBound : index <
      (Permutation.compile rightStart rightState steps).recipes.length := by
    simpa only [Permutation.compile_recipes_length] using leftBound
  have localEq := localsEq index (by
    simpa only [Permutation.compile_recipes_length] using leftBound)
  have rightValue := recipeConstraints_value rightEnv rightStart
    (Permutation.compile rightStart rightState steps).recipes rightHolds
    index rightBound
  have recipesEq := compileRecipes_eval_eq leftEnv rightEnv leftStart
    rightStart leftState rightState steps stateEq localsEq
  have selected := congrArg
    (fun values : List NightstreamFPrime.Spec.F => values.getD index 0)
    recipesEq
  change
    ((Permutation.compile leftStart leftState steps).recipes.map
      (Expr.eval leftEnv)).getD index 0 =
    ((Permutation.compile rightStart rightState steps).recipes.map
      (Expr.eval rightEnv)).getD index 0 at selected
  rw [← getD_eval leftEnv
      (Permutation.compile leftStart leftState steps).recipes index,
    ← getD_eval rightEnv
      (Permutation.compile rightStart rightState steps).recipes index,
    List.getD_eq_get
      (Permutation.compile leftStart leftState steps).recipes (0 : Expr)
      ⟨index, leftBound⟩,
    List.getD_eq_get
      (Permutation.compile rightStart rightState steps).recipes (0 : Expr)
      ⟨index, rightBound⟩] at selected
  calc
    leftEnv (leftStart + index) = rightEnv (rightStart + index) := localEq
    _ = ((Permutation.compile rightStart rightState steps).recipes.get
          ⟨index, rightBound⟩).eval rightEnv := rightValue
    _ = ((Permutation.compile leftStart leftState steps).recipes.get
          ⟨index, leftBound⟩).eval leftEnv := selected.symm

/-- Exact source-layout constraints for one affine Poseidon2 compiler run
construct the corresponding canonical compact invocation in final columns. -/
theorem invocation_complete_of_sourceConstraints
    (phase rowStart witnessStart : Nat) (state : Permutation.EState)
    (env : Env)
    (witnessLocal : Spartan.piCcsPhaseOffset ≤ witnessStart)
    (stateAffine : NightstreamFPrime.Layout.Poseidon2.StateAffine state)
    (sourceHolds : ConstraintsHold (Spartan.pullback env)
      (recipeConstraints witnessStart
        (Permutation.compile witnessStart state Permutation.schedule).recipes)) :
    PermutationInvocationHolds (PilotData.circuitPackage ())
      (invocation phase rowStart witnessStart state) env := by
  let current := invocation phase rowStart witnessStart state
  apply NightstreamFPrime.Export.Pilot.canonicalPermutationInvocation_complete
  unfold PilotData.canonicalConstraints PilotData.canonicalRecipes
  apply compileConstraintsHold_of_transport
    (NightstreamFPrime.Export.Pilot.canonicalInvocationEnv current env)
    (Spartan.pullback env) 8 witnessStart PilotData.canonicalState state
    Permutation.schedule
  · funext lane
    change NightstreamFPrime.Export.Pilot.canonicalInvocationEnv current env
        lane.val =
      (state lane).eval (Spartan.pullback env)
    rw [NightstreamFPrime.Export.Pilot.canonicalInvocationEnv_input]
    have selected : invocationInputCombination current lane.val =
        inputCombination (state lane) := by
      change (List.ofFn (fun selected : Fin 8 =>
        inputCombination (state selected))).getD lane.val
          zeroSparseCombination = inputCombination (state lane)
      exact NightstreamFPrime.Lifecycle.PriorStateHash.ofFn_getD
        (fun selected : Fin 8 => inputCombination (state selected)) lane
        zeroSparseCombination
    rw [selected]
    exact inputCombination_eval (stateAffine lane) env
  · intro index _
    rw [NightstreamFPrime.Export.Pilot.canonicalInvocationEnv_local]
    change env (Spartan.sourceToSpartan witnessStart + index) =
      env (Spartan.sourceToSpartan (witnessStart + index))
    rw [Spartan.sourceToSpartan_add_of_piCcsLocal witnessStart index
      witnessLocal]
  · exact sourceHolds

end NightstreamFPrime.Export.Stage1.PermutationCompilerTransport
