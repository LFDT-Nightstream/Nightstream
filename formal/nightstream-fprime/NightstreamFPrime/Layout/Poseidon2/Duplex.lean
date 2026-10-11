import NightstreamFPrime.Layout.Poseidon2
import NightstreamFPrime.Layout.Polynomial.Horner
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
open NightstreamFPrime.Layout.Polynomial.Horner

/-- Both base-field coordinates of one symbolic extension-field value are
affine. -/
def KExprAffine (value : KExpr) : Prop :=
  R1CS.IsAffine value.c0 ∧ R1CS.IsAffine value.c1

/-- Symbolic inputs read by one Duplex action are affine. -/
def ActionAffine : Formal.Action → Prop
  | .absorb input => ListAffine input
  | .readK _ expected => KExprAffine expected

/-- Every action input in one Duplex schedule is affine. -/
def ActionsAffine (actions : List Formal.Action) : Prop :=
  ∀ action ∈ actions, ActionAffine action

/-- A transcript state is one exact permutation-owned block of eight fresh
variables. -/
def StateFresh (state : Layer.EState) : Prop :=
  ∃ start, state = Permutation.freshState start

theorem StateFresh.affine {state : Layer.EState}
    (fresh : StateFresh state) : StateAffine state := by
  rcases fresh with ⟨start, rfl⟩
  exact freshState_affine start

private theorem compileAbsorptions_output_fresh_of_nonempty
    (start : Nat) (state : Layer.EState) (blocks : List (List Expr))
    (nonempty : blocks ≠ []) :
    StateFresh (Hash.compileAbsorptions start state blocks).output := by
  induction blocks generalizing start state with
  | nil => exact (nonempty rfl).elim
  | cons block blocks inductionHypothesis =>
      cases blocks with
      | nil =>
          refine ⟨start + 1080, ?_⟩
          simp [Hash.compileAbsorptions, compile_schedule_output_eq]
      | cons next rest =>
          change StateFresh
            (Hash.compileAbsorptions (start + 1096)
              (Permutation.compile start (Hash.absorbE state block)
                Permutation.schedule).output (next :: rest)).output
          exact inductionHypothesis (start + 1096)
            (Permutation.compile start (Hash.absorbE state block)
              Permutation.schedule).output (by simp)

theorem compileAbsorptions_output_fresh
    (start : Nat) (state : Layer.EState) (blocks : List (List Expr))
    (stateFresh : StateFresh state) :
    StateFresh (Hash.compileAbsorptions start state blocks).output := by
  cases blocks with
  | nil => simpa [Hash.compileAbsorptions] using stateFresh
  | cons block rest =>
      exact compileAbsorptions_output_fresh_of_nonempty start state
        (block :: rest) (by simp)

theorem read_sample_linear (state : Layer.EState) (pair : Read.Pair)
    (stateFresh : StateFresh state) :
    KExprLinear (Read.sample state pair) := by
  rcases stateFresh with ⟨stateStart, rfl⟩
  refine ⟨rfl, rfl, ?_, ?_⟩
  · simp [Read.sample, Permutation.freshState, Nonconstant]
  · simp [Read.sample, Permutation.freshState, Nonconstant]

/-- Every action preserves a fresh state once the initial state is fresh. -/
theorem compile_output_fresh (start : Nat) (state : Layer.EState)
    (actions : List Formal.Action) (stateFresh : StateFresh state) :
    StateFresh (Formal.compile start state actions).output := by
  induction actions generalizing start state with
  | nil => simpa [Formal.compile] using stateFresh
  | cons action actions inductionHypothesis =>
      cases action with
      | absorb input =>
          let absorbed := Hash.compileAbsorptions start state
            (Hash.inputChunks input)
          change StateFresh
            (Formal.compile (start + absorbed.recipes.length)
              absorbed.output actions).output
          exact inductionHypothesis _ _
            (compileAbsorptions_output_fresh start state
              (Hash.inputChunks input) stateFresh)
      | readK pair expected =>
          exact inductionHypothesis start state stateFresh

/-- Every read sample is a nonconstant linear pair once the initial state is
fresh. -/
theorem compile_samples_linear (start : Nat) (state : Layer.EState)
    (actions : List Formal.Action) (stateFresh : StateFresh state) :
    ∀ sample ∈ (Formal.compile start state actions).samples,
      KExprLinear sample := by
  induction actions generalizing start state with
  | nil => simp [Formal.compile]
  | cons action actions inductionHypothesis =>
      cases action with
      | absorb input =>
          let absorbed := Hash.compileAbsorptions start state
            (Hash.inputChunks input)
          exact inductionHypothesis _ _
            (compileAbsorptions_output_fresh start state
              (Hash.inputChunks input) stateFresh)
      | readK pair expected =>
          intro sample member
          change sample ∈ Read.sample state pair ::
            (Formal.compile start state actions).samples at member
          rcases List.mem_cons.mp member with rfl | member
          · exact read_sample_linear state pair stateFresh
          · exact inductionHypothesis start state stateFresh sample member

/-- A leading nonempty absorb establishes the fresh-state invariant for the
remainder of the program. -/
theorem compile_output_fresh_of_head_absorb
    (start : Nat) (state : Layer.EState) (input : List Expr)
    (actions : List Formal.Action)
    (chunksNonempty : Hash.inputChunks input ≠ []) :
    StateFresh
      (Formal.compile start state (.absorb input :: actions)).output := by
  let absorbed := Hash.compileAbsorptions start state (Hash.inputChunks input)
  change StateFresh
    (Formal.compile (start + absorbed.recipes.length)
      absorbed.output actions).output
  apply compile_output_fresh
  exact compileAbsorptions_output_fresh_of_nonempty start state
    (Hash.inputChunks input) chunksNonempty

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

theorem read_sample_affine (state : Layer.EState) (pair : Read.Pair)
    (stateAffine : StateAffine state) :
    KExprAffine (Read.sample state pair) :=
  ⟨stateAffine (Read.lowLane pair), stateAffine (Read.highLane pair)⟩

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
      | readK pair expected =>
          exact inductionHypothesis (start := start) (state := state)
            stateAffine tailAffine

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
      | readK pair expected =>
          intro sample member
          change sample ∈ Read.sample state pair ::
            (Formal.compile start state actions).samples at member
          rcases List.mem_cons.mp member with rfl | member
          · exact read_sample_affine state pair stateAffine
          · exact inductionHypothesis (start := start) (state := state)
              stateAffine tailAffine sample member

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
      | readK pair expected =>
          exact inductionHypothesis (start := start) (state := state)
            stateAffine tailAffine

end NightstreamFPrime.Layout.Poseidon2.Duplex
