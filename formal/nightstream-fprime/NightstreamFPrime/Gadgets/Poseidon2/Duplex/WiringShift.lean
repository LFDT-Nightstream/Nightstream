import NightstreamFPrime.Gadgets.Poseidon2.Duplex.WiringSupport

/-!
Proves uniform column relocation for the recipe-free Poseidon2 Duplex wiring.

This module changes no compiler or transcript definition. It states only that
moving the start column and every incoming-state variable by one fixed delta
moves every exposed sample and final-state variable by that same delta.
-/

namespace NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.WiringShift

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec

def expression (delta : Nat) : Expr → Expr
  | .var index => .var (index + delta)
  | .const value => .const value
  | .add left right => .add (expression delta left) (expression delta right)
  | .mul left right => .mul (expression delta left) (expression delta right)

def quadratic (delta : Nat) (value : KExpr) : KExpr :=
  ⟨expression delta value.c0, expression delta value.c1⟩

def state (delta : Nat) (value : EState) : EState :=
  fun lane => expression delta (value lane)

@[simp] theorem expression_eval (delta : Nat) (value : Expr) (env : Env) :
    (expression delta value).eval env = value.eval (fun index => env (index + delta)) := by
  induction value with
  | var index => rfl
  | const value => rfl
  | add left right leftHypothesis rightHypothesis =>
      simp [expression, leftHypothesis, rightHypothesis]
  | mul left right leftHypothesis rightHypothesis =>
      simp [expression, leftHypothesis, rightHypothesis]

theorem expression_eval_eq_of_shift_agreement
    (delta : Nat) (value : Expr) (allowed : Nat → Prop)
    (left right : Env) (support : value.VarsSatisfy allowed)
    (agrees : ∀ index, allowed index → right (index + delta) = left index) :
    (expression delta value).eval right = value.eval left := by
  rw [expression_eval]
  exact value.eval_eq_of_agree_satisfy allowed
    (fun index => right (index + delta)) left support agrees

@[simp] theorem quadratic_eval (delta : Nat) (value : KExpr) (env : Env) :
    (quadratic delta value).eval env =
      value.eval (fun index => env (index + delta)) := by
  cases value
  simp [quadratic, KExpr.eval]

theorem quadratic_eval_eq_of_shift_agreement
    (delta : Nat) (value : KExpr) (allowed : Nat → Prop)
    (left right : Env) (support : KSupported value allowed)
    (agrees : ∀ index, allowed index → right (index + delta) = left index) :
    (quadratic delta value).eval right = value.eval left := by
  rw [quadratic_eval]
  exact congrArg₂ K.mk
    (value.c0.eval_eq_of_agree_satisfy allowed
      (fun index => right (index + delta)) left support.1 agrees)
    (value.c1.eval_eq_of_agree_satisfy allowed
      (fun index => right (index + delta)) left support.2 agrees)

theorem state_eval_eq_of_shift_agreement
    (delta : Nat) (value : EState) (allowed : Nat → Prop)
    (left right : Env) (support : StateSupported value allowed)
    (agrees : ∀ index, allowed index → right (index + delta) = left index)
    (lane : Fin Spec.Poseidon2.width) :
    (state delta value lane).eval right = (value lane).eval left := by
  simp only [state, expression_eval]
  exact (value lane).eval_eq_of_agree_satisfy allowed
    (fun index => right (index + delta)) left (support lane) agrees

theorem state_scheduleOutput (delta start : Nat) :
    state delta (Permutation.scheduleOutput start) =
      Permutation.scheduleOutput (start + delta) := by
  funext lane
  simp only [state, Permutation.scheduleOutput, Permutation.freshState,
    expression]
  congr 1
  omega

inductive SameShape : List Action → List Action → Prop
  | nil : SameShape [] []
  | absorb (leftInput rightInput : List Expr)
      (chunkCount : (Hash.inputChunks leftInput).length =
        (Hash.inputChunks rightInput).length)
      (leftTail rightTail : List Action)
      (tail : SameShape leftTail rightTail) :
      SameShape (.absorb leftInput :: leftTail)
        (.absorb rightInput :: rightTail)
  | squeeze (leftExpected rightExpected : KExpr)
      (leftTail rightTail : List Action)
      (tail : SameShape leftTail rightTail) :
      SameShape (.squeezeK leftExpected :: leftTail)
        (.squeezeK rightExpected :: rightTail)

theorem SameShape.refl (actions : List Action) : SameShape actions actions := by
  induction actions with
  | nil => exact .nil
  | cons action actions inductionHypothesis =>
      cases action with
      | absorb input =>
          exact .absorb input input rfl actions actions inductionHypothesis
      | squeezeK expected =>
          exact .squeeze expected expected actions actions inductionHypothesis

theorem SameShape.append {leftPrefix rightPrefix leftTail rightTail : List Action}
    (headShape : SameShape leftPrefix rightPrefix)
    (tail : SameShape leftTail rightTail) :
    SameShape (leftPrefix ++ leftTail) (rightPrefix ++ rightTail) := by
  induction headShape generalizing leftTail rightTail with
  | nil => exact tail
  | absorb leftInput rightInput chunkCount leftRest rightRest _ hypothesis =>
      exact .absorb leftInput rightInput chunkCount _ _ (hypothesis tail)
  | squeeze leftExpected rightExpected leftRest rightRest _ hypothesis =>
      exact .squeeze leftExpected rightExpected _ _ (hypothesis tail)

theorem SameShape.flatMap
    {Index : Type} (indices : List Index)
    (left right : Index → List Action)
    (each : ∀ index ∈ indices, SameShape (left index) (right index)) :
    SameShape (indices.flatMap left) (indices.flatMap right) := by
  induction indices with
  | nil => exact .nil
  | cons index indices inductionHypothesis =>
      rw [List.flatMap_cons, List.flatMap_cons]
      exact (each index (by simp)).append
        (inductionHypothesis (fun current member =>
          each current (by simp [member])))

theorem inputChunks_length_eq_of_length_eq
    {left right : List Expr} (sameLength : left.length = right.length) :
    (Hash.inputChunks left).length = (Hash.inputChunks right).length := by
  simp [Hash.inputChunks, sameLength]

theorem SameShape.absorb_of_length {left right : List Expr}
    (sameLength : left.length = right.length) :
    SameShape [.absorb left] [.absorb right] :=
  .absorb left right (inputChunks_length_eq_of_length_eq sameLength) [] [] .nil

private theorem compileAbsorbWiring_next_shift
    (delta start : Nat) (initial : EState)
    (leftBlocks rightBlocks : List (List Expr))
    (sameLength : leftBlocks.length = rightBlocks.length) :
    (compileAbsorbWiring (start + delta) (state delta initial) rightBlocks).next =
      (compileAbsorbWiring start initial leftBlocks).next + delta := by
  rw [compileAbsorbWiring_next, compileAbsorbWiring_next]
  rw [← sameLength]
  omega

private theorem compileAbsorbWiring_output_shift
    (delta start : Nat) (initial : EState)
    (leftBlocks rightBlocks : List (List Expr))
    (sameLength : leftBlocks.length = rightBlocks.length) :
    (compileAbsorbWiring (start + delta) (state delta initial) rightBlocks).output =
      state delta (compileAbsorbWiring start initial leftBlocks).output := by
  cases leftBlocks with
  | nil =>
      have rightNil : rightBlocks = [] := by
        simpa using sameLength.symm
      subst rightBlocks
      rfl
  | cons leftBlock leftRest =>
      cases rightBlocks with
      | nil => simp at sameLength
      | cons rightBlock rightRest =>
        have restLength : leftRest.length = rightRest.length := by
          simpa using sameLength
        calc
          (compileAbsorbWiring (start + delta) (state delta initial)
              (rightBlock :: rightRest)).output =
              Permutation.scheduleOutput
                (start + delta + rightRest.length * 592) :=
            compileAbsorbWiring_output_cons _ _ _ _
          _ = Permutation.scheduleOutput
              (start + leftRest.length * 592 + delta) := by
            congr 1
            rw [← restLength]
            omega
          _ = state delta
              (Permutation.scheduleOutput
                (start + leftRest.length * 592)) :=
            (state_scheduleOutput delta _).symm
          _ = state delta
              (compileAbsorbWiring start initial
                (leftBlock :: leftRest)).output := by
            rw [compileAbsorbWiring_output_cons]

/-- Recipe-free Duplex wiring reads only absorb chunk counts and squeeze
positions. Under the same shape, a uniform start and state shift relocates
every exposed sample and final-state variable by the same delta. -/
theorem compileWiring_shift_of_sameShape
    (delta start : Nat) (initial : EState)
    {leftActions rightActions : List Action}
    (shape : SameShape leftActions rightActions) :
    (compileWiring (start + delta) (state delta initial) rightActions).samples =
        (compileWiring start initial leftActions).samples.map (quadratic delta) ∧
      (compileWiring (start + delta) (state delta initial) rightActions).output =
        state delta (compileWiring start initial leftActions).output := by
  induction shape generalizing start initial with
  | nil => exact ⟨rfl, rfl⟩
  | absorb leftInput rightInput chunkCount leftTail rightTail tail
      inductionHypothesis =>
          let leftAbsorbed := compileAbsorbWiring start initial
            (Hash.inputChunks leftInput)
          let rightAbsorbed := compileAbsorbWiring (start + delta)
            (state delta initial) (Hash.inputChunks rightInput)
          have nextEq : rightAbsorbed.next = leftAbsorbed.next + delta :=
            compileAbsorbWiring_next_shift delta start initial
              (Hash.inputChunks leftInput) (Hash.inputChunks rightInput)
              chunkCount
          have outputEq : rightAbsorbed.output =
              state delta leftAbsorbed.output :=
            compileAbsorbWiring_output_shift delta start initial
              (Hash.inputChunks leftInput) (Hash.inputChunks rightInput)
              chunkCount
          have tailResult := inductionHypothesis leftAbsorbed.next
            leftAbsorbed.output
          change
            (compileWiring rightAbsorbed.next rightAbsorbed.output
                rightTail).samples =
                (compileWiring leftAbsorbed.next leftAbsorbed.output
                  leftTail).samples.map (quadratic delta) ∧
              (compileWiring rightAbsorbed.next rightAbsorbed.output
                  rightTail).output =
                state delta
                  (compileWiring leftAbsorbed.next leftAbsorbed.output
                    leftTail).output
          rw [nextEq, outputEq]
          exact tailResult
  | squeeze leftExpected rightExpected leftTail rightTail tail
      inductionHypothesis =>
          have tailResult := inductionHypothesis (start + 1184)
            (Permutation.scheduleOutput (start + 592))
          change
            (⟨state delta initial 0,
                Permutation.scheduleOutput (start + delta) 0⟩ : KExpr) ::
                  (compileWiring (start + delta + 1184)
                    (Permutation.scheduleOutput (start + delta + 592))
                    rightTail).samples =
                ((⟨initial 0, Permutation.scheduleOutput start 0⟩ : KExpr) ::
                  (compileWiring (start + 1184)
                    (Permutation.scheduleOutput (start + 592))
                    leftTail).samples).map (quadratic delta) ∧
              (compileWiring (start + delta + 1184)
                  (Permutation.scheduleOutput (start + delta + 592))
                  rightTail).output =
                state delta
                  (compileWiring (start + 1184)
                    (Permutation.scheduleOutput (start + 592))
                    leftTail).output
          have firstEq :
              (⟨state delta initial 0,
                  Permutation.scheduleOutput (start + delta) 0⟩ : KExpr) =
                quadratic delta
                  (⟨initial 0, Permutation.scheduleOutput start 0⟩ : KExpr) := by
            exact congrArg₂ KExpr.mk rfl
              (congrFun (state_scheduleOutput delta start).symm 0)
          have firstStateEq :
              Permutation.scheduleOutput (start + delta + 592) =
                state delta (Permutation.scheduleOutput (start + 592)) := by
            rw [state_scheduleOutput]
            congr 1
            omega
          rw [firstEq, firstStateEq]
          simpa [List.map_cons, Nat.add_assoc, Nat.add_comm,
            Nat.add_left_comm] using tailResult

/-- Recipe-free Duplex wiring commutes with one uniform column shift. -/
theorem compileWiring_shift
    (delta start : Nat) (initial : EState) (actions : List Action) :
    (compileWiring (start + delta) (state delta initial) actions).samples =
        (compileWiring start initial actions).samples.map (quadratic delta) ∧
      (compileWiring (start + delta) (state delta initial) actions).output =
        state delta (compileWiring start initial actions).output :=
  compileWiring_shift_of_sameShape delta start initial (SameShape.refl actions)

end NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.WiringShift
