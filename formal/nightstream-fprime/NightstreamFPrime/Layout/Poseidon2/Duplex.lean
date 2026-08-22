import NightstreamFPrime.Layout.Poseidon2
import NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal

/-!
Owns the physical recipe boundary for the reusable Poseidon2 Duplex compiler.
It proves direct R1CS lowering from affine symbolic action inputs. Protocol
labels, action schedules, and public bindings remain lifecycle-owned.
-/

namespace NightstreamFPrime.Layout.Poseidon2.Duplex

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex

/-- Both base-field coordinates of one symbolic extension-field value are
affine. -/
def KExprAffine (value : KExpr) : Prop :=
  R1CS.IsAffine value.c0 ∧ R1CS.IsAffine value.c1

/-- Symbolic inputs read by one Duplex action are affine. -/
def ActionAffine : Formal.Action → Prop
  | .absorb input => ListAffine input
  | .squeezeK expected => KExprAffine expected

/-- Every action input in one Duplex schedule is affine. -/
def ActionsAffine (actions : List Formal.Action) : Prop :=
  ∀ action ∈ actions, ActionAffine action

theorem ActionsAffine.append {first second : List Formal.Action}
    (firstAffine : ActionsAffine first)
    (secondAffine : ActionsAffine second) :
    ActionsAffine (first ++ second) := by
  intro action member
  rcases List.mem_append.mp member with member | member
  · exact firstAffine action member
  · exact secondAffine action member

theorem ActionsAffine.cons {action : Formal.Action}
    {actions : List Formal.Action}
    (headAffine : ActionAffine action)
    (tailAffine : ActionsAffine actions) :
    ActionsAffine (action :: actions) := by
  intro current member
  rcases List.mem_cons.mp member with rfl | member
  · exact headAffine
  · exact tailAffine current member

theorem squeeze_recipes_direct (start : Nat) (state : Layer.EState)
    (stateAffine : StateAffine state) :
    R1CS.RecipesDirect start (Squeeze.compile start state).recipes := by
  rw [Squeeze.compile_recipes_eq]
  apply R1CS.recipesDirect_append
  · exact compile_schedule_direct start state stateAffine
  · exact compile_schedule_direct
      (start + (Squeeze.firstPermutation start state).recipes.length)
      (Squeeze.firstPermutation start state).output
      (compile_schedule_output_affine start state stateAffine)

theorem squeeze_output_affine (start : Nat) (state : Layer.EState)
    (stateAffine : StateAffine state) :
    StateAffine (Squeeze.compile start state).output := by
  intro lane
  rw [Squeeze.compile_output_apply]
  exact compile_schedule_output_affine
    (start + (Squeeze.firstPermutation start state).recipes.length)
    (Squeeze.firstPermutation start state).output
    (compile_schedule_output_affine start state stateAffine) lane

theorem squeeze_sample_affine (start : Nat) (state : Layer.EState)
    (stateAffine : StateAffine state) :
    KExprAffine (Squeeze.compile start state).sample := by
  rw [Squeeze.compile_sample_eq]
  exact ⟨stateAffine 0,
    compile_schedule_output_affine start state stateAffine 0⟩

/-- Every Duplex witness recipe lowers to one direct R1CS row. This theorem
is structural in the action list and does not evaluate an emitted schedule. -/
theorem compile_recipes_direct (start : Nat) (state : Layer.EState)
    (actions : List Formal.Action)
    (stateAffine : StateAffine state)
    (actionsAffine : ActionsAffine actions) :
    R1CS.RecipesDirect start (Formal.compile start state actions).recipes := by
  induction actions generalizing start state with
  | nil => trivial
  | cons action actions inductionHypothesis =>
      have headAffine : ActionAffine action :=
        actionsAffine action (by simp)
      have tailAffine : ActionsAffine actions := by
        intro current member
        exact actionsAffine current (by simp [member])
      cases action with
      | absorb input =>
          let absorbed := Hash.compileAbsorptions start state
            (Hash.inputChunks input)
          change R1CS.RecipesDirect start
            (absorbed.recipes ++
              (Formal.compile (start + absorbed.recipes.length)
                absorbed.output actions).recipes)
          apply R1CS.recipesDirect_append
          · exact compileAbsorptions_direct start state
              (Hash.inputChunks input) stateAffine
              (inputChunks_affine input headAffine)
          · exact inductionHypothesis
              (start := start + absorbed.recipes.length)
              (state := absorbed.output)
              (compileAbsorptions_output_affine start state
                (Hash.inputChunks input) stateAffine
                (inputChunks_affine input headAffine))
              tailAffine
      | squeezeK expected =>
          let squeezed := Squeeze.compile start state
          change R1CS.RecipesDirect start
            (squeezed.recipes ++
              (Formal.compile (start + squeezed.recipes.length)
                squeezed.output actions).recipes)
          apply R1CS.recipesDirect_append
          · exact squeeze_recipes_direct start state stateAffine
          · exact inductionHypothesis
              (start := start + squeezed.recipes.length)
              (state := squeezed.output)
              (squeeze_output_affine start state stateAffine)
              tailAffine

/-- Every verifier-derived Duplex sample is an affine pair of existing
symbolic variables. Expected samples do not alter this output list. -/
theorem compile_samples_affine (start : Nat) (state : Layer.EState)
    (actions : List Formal.Action)
    (stateAffine : StateAffine state)
    (actionsAffine : ActionsAffine actions) :
    ∀ sample ∈ (Formal.compile start state actions).samples,
      KExprAffine sample := by
  induction actions generalizing start state with
  | nil => simp [Formal.compile]
  | cons action actions inductionHypothesis =>
      have headAffine : ActionAffine action :=
        actionsAffine action (by simp)
      have tailAffine : ActionsAffine actions := by
        intro current member
        exact actionsAffine current (by simp [member])
      cases action with
      | absorb input =>
          let absorbed := Hash.compileAbsorptions start state
            (Hash.inputChunks input)
          exact inductionHypothesis
            (start := start + absorbed.recipes.length)
            (state := absorbed.output)
            (compileAbsorptions_output_affine start state
              (Hash.inputChunks input) stateAffine
              (inputChunks_affine input headAffine))
            tailAffine
      | squeezeK expected =>
          let squeezed := Squeeze.compile start state
          intro sample member
          change sample ∈ squeezed.sample ::
            (Formal.compile (start + squeezed.recipes.length)
              squeezed.output actions).samples at member
          rcases List.mem_cons.mp member with rfl | member
          · exact squeeze_sample_affine start state stateAffine
          · exact inductionHypothesis
              (start := start + squeezed.recipes.length)
              (state := squeezed.output)
              (squeeze_output_affine start state stateAffine)
              tailAffine sample member

/-- The final Duplex state is affine when the initial state and action inputs
are affine. -/
theorem compile_output_affine (start : Nat) (state : Layer.EState)
    (actions : List Formal.Action)
    (stateAffine : StateAffine state)
    (actionsAffine : ActionsAffine actions) :
    StateAffine (Formal.compile start state actions).output := by
  induction actions generalizing start state with
  | nil => simpa [Formal.compile] using stateAffine
  | cons action actions inductionHypothesis =>
      have headAffine : ActionAffine action :=
        actionsAffine action (by simp)
      have tailAffine : ActionsAffine actions := by
        intro current member
        exact actionsAffine current (by simp [member])
      cases action with
      | absorb input =>
          let absorbed := Hash.compileAbsorptions start state
            (Hash.inputChunks input)
          exact inductionHypothesis
            (start := start + absorbed.recipes.length)
            (state := absorbed.output)
            (compileAbsorptions_output_affine start state
              (Hash.inputChunks input) stateAffine
              (inputChunks_affine input headAffine))
            tailAffine
      | squeezeK expected =>
          let squeezed := Squeeze.compile start state
          exact inductionHypothesis
            (start := start + squeezed.recipes.length)
            (state := squeezed.output)
            (squeeze_output_affine start state stateAffine)
            tailAffine

end NightstreamFPrime.Layout.Poseidon2.Duplex
