import NightstreamFPrime.Gadgets.Poseidon2.Duplex.Absorb
import NightstreamFPrime.Gadgets.Poseidon2.Duplex.Squeeze

/-!
Owns the proof-carrying Poseidon2 duplex trace. A trace contains only absorb
blocks and quadratic-extension squeezes. Protocol labels and serialization
remain lifecycle-owned data supplied as absorb expressions.
-/

namespace NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2

abbrev EState := Layer.EState
abbrev FState := Layer.FState

inductive Action where
  | absorb (input : List Expr)
  | squeezeK (expected : KExpr)

inductive ValueAction where
  | absorb (input : List F)
  | squeezeK (expected : K)

/-- Exact symbolic recipe cost of one duplex action. Each absorb chunk runs
one 592-recipe Poseidon2 permutation. A quadratic-extension squeeze runs two
such permutations. -/
def Action.recipeCount : Action → Nat
  | .absorb input => (Hash.inputChunks input).length * 592
  | .squeezeK _ => 1184

/-- Exact symbolic recipe cost of a complete duplex action trace. -/
def recipeCount (actions : List Action) : Nat :=
  (actions.map Action.recipeCount).sum

/-- Exact number of non-final-state assertions contributed by one action. -/
def Action.assertionCount : Action → Nat
  | .absorb _ => 0
  | .squeezeK _ => 2

/-- Exact number of non-final-state assertions in an action trace. -/
def assertionCount (actions : List Action) : Nat :=
  (actions.map Action.assertionCount).sum

@[simp] theorem recipeCount_append (left right : List Action) :
    recipeCount (left ++ right) = recipeCount left + recipeCount right := by
  simp [recipeCount, List.sum_append]

@[simp] theorem assertionCount_append (left right : List Action) :
    assertionCount (left ++ right) =
      assertionCount left + assertionCount right := by
  simp [assertionCount, List.sum_append]

theorem recipeCount_flatMap_constant
    {Index : Type}
    (indices : List Index)
    (group : Index → List Action)
    (cost : Nat)
    (each : ∀ index, index ∈ indices → recipeCount (group index) = cost) :
    recipeCount (indices.flatMap group) = indices.length * cost := by
  induction indices with
  | nil => simp [recipeCount]
  | cons head tail inductionHypothesis =>
      have headCost : recipeCount (group head) = cost :=
        each head (by simp)
      have tailCost : ∀ index, index ∈ tail →
          recipeCount (group index) = cost := by
        intro index member
        exact each index (by simp [member])
      rw [List.flatMap_cons, recipeCount_append, headCost,
        inductionHypothesis tailCost]
      simp [Nat.succ_mul, Nat.add_comm]

def Action.eval (env : Env) : Action → ValueAction
  | .absorb input => .absorb (Hash.evalList env input)
  | .squeezeK expected => .squeezeK (expected.eval env)

/-- The operation schedule without caller-provided squeeze outputs. -/
inductive ActionShape where
  | absorb (input : List Expr)
  | squeezeK
deriving DecidableEq

def Action.shape : Action → ActionShape
  | .absorb input => .absorb input
  | .squeezeK _ => .squeezeK

def expectedSamples : List Action → List KExpr
  | [] => []
  | .absorb _ :: actions => expectedSamples actions
  | .squeezeK expected :: actions => expected :: expectedSamples actions

/-- Deterministic duplex trace semantics. -/
def TraceHolds : Spec.Poseidon2.State → List ValueAction →
    Spec.Poseidon2.State → Prop
  | state, [], final => state = final
  | state, .absorb input :: actions, final =>
      TraceHolds (Absorb.reference state input) actions final
  | state, .squeezeK expected :: actions, final =>
      expected = Squeeze.referenceSample state ∧
        TraceHolds (Squeeze.referenceState state) actions final

structure Program where
  recipes : List Expr
  assertions : List Expr
  samples : List KExpr
  output : EState

def compile : Nat → EState → List Action → Program
  | _, state, [] => ⟨[], [], [], state⟩
  | start, state, .absorb input :: actions =>
      let absorbed := Hash.compileAbsorptions start state
        (Hash.inputChunks input)
      let tail := compile (start + absorbed.recipes.length)
        absorbed.output actions
      ⟨absorbed.recipes ++ tail.recipes, tail.assertions, tail.samples,
        tail.output⟩
  | start, state, .squeezeK expected :: actions =>
      let squeezed := Squeeze.compile start state
      let tail := compile (start + squeezed.recipes.length)
        squeezed.output actions
      ⟨squeezed.recipes ++ tail.recipes,
        KExpr.equalities squeezed.sample expected ++ tail.assertions,
        squeezed.sample :: tail.samples, tail.output⟩

/-- Recipe-free absorb projection used only to recover symbolic wiring. -/
structure AbsorbWiring where
  next : Nat
  output : EState

def compileAbsorbWiring : Nat → EState → List (List Expr) → AbsorbWiring
  | start, state, [] => ⟨start, state⟩
  | start, _state, _block :: blocks =>
      compileAbsorbWiring (start + 592)
        (Permutation.scheduleOutput start) blocks

theorem compileAbsorbWiring_next (start : Nat) (state : EState)
    (blocks : List (List Expr)) :
    (compileAbsorbWiring start state blocks).next =
      start + blocks.length * 592 := by
  induction blocks generalizing start state with
  | nil => rfl
  | cons block blocks inductionHypothesis =>
      simp only [compileAbsorbWiring]
      rw [inductionHypothesis]
      simp only [List.length_cons]
      omega

theorem compileAbsorbWiring_output (start : Nat) (state : EState)
    (blocks : List (List Expr)) :
    (compileAbsorbWiring start state blocks).output =
      (Hash.compileAbsorptions start state blocks).output := by
  induction blocks generalizing start state with
  | nil => rfl
  | cons block blocks inductionHypothesis =>
      simp only [compileAbsorbWiring, Hash.compileAbsorptions]
      rw [inductionHypothesis, Permutation.scheduleOutput_eq_compile]

/-- Recipe-free projection of the Duplex compiler's wiring outputs. -/
structure Wiring where
  next : Nat
  samples : List KExpr
  output : EState

def compileWiring : Nat → EState → List Action → Wiring
  | start, state, [] => ⟨start, [], state⟩
  | start, state, .absorb input :: actions =>
      let absorbed := compileAbsorbWiring start state (Hash.inputChunks input)
      let tail := compileWiring absorbed.next absorbed.output actions
      ⟨tail.next, tail.samples, tail.output⟩
  | start, state, .squeezeK _expected :: actions =>
      let first := Permutation.scheduleOutput start
      let tail := compileWiring (start + 1184)
        (Permutation.scheduleOutput (start + 592)) actions
      ⟨tail.next, ⟨state 0, first 0⟩ :: tail.samples, tail.output⟩

theorem compileWiring_next (start : Nat) (state : EState)
    (actions : List Action) :
    (compileWiring start state actions).next = start + recipeCount actions := by
  induction actions generalizing start state with
  | nil => rfl
  | cons action actions inductionHypothesis =>
      cases action with
      | absorb input =>
          simp only [compileWiring]
          rw [inductionHypothesis, compileAbsorbWiring_next]
          simp [recipeCount, Action.recipeCount]
          omega
      | squeezeK expected =>
          simp only [compileWiring]
          rw [inductionHypothesis]
          simp [recipeCount, Action.recipeCount]
          omega

theorem wiringSample_eq_squeeze (start : Nat) (state : EState) :
    (⟨state 0, Permutation.scheduleOutput start 0⟩ : KExpr) =
      (Squeeze.compile start state).sample := by
  rw [Squeeze.compile_sample_eq]
  congr 1

theorem wiringOutput_eq_squeeze (start : Nat) (state : EState) :
    Permutation.scheduleOutput (start + 592) =
      (Squeeze.compile start state).output := by
  funext lane
  rw [Squeeze.compile_output_apply]
  unfold Squeeze.secondPermutation
  rw [Squeeze.first_recipes_length]
  exact congrFun (Permutation.scheduleOutput_eq_compile (start + 592)
    (Squeeze.firstPermutation start state).output) lane

/-- Wiring projection agrees with the full compiler on every externally used
sample and on the final symbolic state. -/
theorem compileWiring_matches (start : Nat) (state : EState)
    (actions : List Action) :
    (compileWiring start state actions).samples =
        (compile start state actions).samples ∧
      (compileWiring start state actions).output =
        (compile start state actions).output := by
  induction actions generalizing start state with
  | nil => exact ⟨rfl, rfl⟩
  | cons action actions inductionHypothesis =>
      cases action with
      | absorb input =>
          let blocks := Hash.inputChunks input
          let projected := compileAbsorbWiring start state blocks
          let absorbed := Hash.compileAbsorptions start state blocks
          change
            (compileWiring projected.next projected.output actions).samples =
                (compile (start + absorbed.recipes.length)
                  absorbed.output actions).samples ∧
              (compileWiring projected.next projected.output actions).output =
                (compile (start + absorbed.recipes.length)
                  absorbed.output actions).output
          have nextEq : projected.next =
              start + absorbed.recipes.length := by
            rw [compileAbsorbWiring_next,
              Hash.compileAbsorptions_recipes_length]
          have outputEq : projected.output = absorbed.output :=
            compileAbsorbWiring_output start state blocks
          rw [nextEq, outputEq]
          exact inductionHypothesis _ _
      | squeezeK expected =>
          let squeezed := Squeeze.compile start state
          change
            (⟨state 0, Permutation.scheduleOutput start 0⟩ : KExpr) ::
                  (compileWiring (start + 1184)
                    (Permutation.scheduleOutput (start + 592))
                    actions).samples =
                squeezed.sample ::
                  (compile (start + squeezed.recipes.length)
                    squeezed.output actions).samples ∧
              (compileWiring (start + 1184)
                    (Permutation.scheduleOutput (start + 592))
                    actions).output =
                (compile (start + squeezed.recipes.length)
                  squeezed.output actions).output
          have tailMatches := inductionHypothesis (start + 1184)
            squeezed.output
          constructor
          · rw [wiringSample_eq_squeeze start state,
              wiringOutput_eq_squeeze start state,
              Squeeze.compile_recipes_length start state]
            exact congrArg (List.cons squeezed.sample) tailMatches.1
          · rw [wiringOutput_eq_squeeze start state,
              Squeeze.compile_recipes_length start state]
            exact tailMatches.2

/-- Lazy-state absorb projection. A nonempty absorb replaces the incoming
state before it can be forced. -/
def compileAbsorbWiringLazy : Nat → (Unit → EState) →
    List (List Expr) → AbsorbWiring
  | start, delayed, [] => ⟨start, delayed ()⟩
  | start, _delayed, _block :: blocks =>
      compileAbsorbWiringLazy (start + 592)
        (fun _ => Permutation.scheduleOutput start) blocks

theorem compileAbsorbWiringLazy_eq (start : Nat)
    (delayed : Unit → EState) (state : EState)
    (blocks : List (List Expr)) (stateEq : delayed () = state) :
    compileAbsorbWiringLazy start delayed blocks =
      compileAbsorbWiring start state blocks := by
  induction blocks generalizing start delayed state with
  | nil => simp [compileAbsorbWiringLazy, compileAbsorbWiring, stateEq]
  | cons block blocks inductionHypothesis =>
      simp only [compileAbsorbWiringLazy, compileAbsorbWiring]
      exact inductionHypothesis _ _ _ rfl

/-- Lazy-state form of `compileWiring`. It is an execution device only; the
agreement theorem below keeps `compileWiring` as the proof meaning. -/
def compileWiringLazy : Nat → (Unit → EState) → List Action → Wiring
  | start, delayed, [] => ⟨start, [], delayed ()⟩
  | start, delayed, .absorb input :: actions =>
      let absorbed := compileAbsorbWiringLazy start delayed
        (Hash.inputChunks input)
      let tail := compileWiringLazy absorbed.next
        (fun _ => absorbed.output) actions
      ⟨tail.next, tail.samples, tail.output⟩
  | start, delayed, .squeezeK _expected :: actions =>
      let state := delayed ()
      let first := Permutation.scheduleOutput start
      let tail := compileWiringLazy (start + 1184)
        (fun _ => Permutation.scheduleOutput (start + 592)) actions
      ⟨tail.next, ⟨state 0, first 0⟩ :: tail.samples, tail.output⟩

theorem compileWiringLazy_eq (start : Nat) (delayed : Unit → EState)
    (state : EState) (actions : List Action) (stateEq : delayed () = state) :
    compileWiringLazy start delayed actions =
      compileWiring start state actions := by
  induction actions generalizing start delayed state with
  | nil => simp [compileWiringLazy, compileWiring, stateEq]
  | cons action actions inductionHypothesis =>
      cases action with
      | absorb input =>
          have absorbedEq :
              compileAbsorbWiringLazy start delayed (Hash.inputChunks input) =
                compileAbsorbWiring start state (Hash.inputChunks input) :=
            compileAbsorbWiringLazy_eq start delayed state
              (Hash.inputChunks input) stateEq
          simp only [compileWiringLazy, compileWiring]
          rw [absorbedEq]
          let absorbed := compileAbsorbWiring start state
            (Hash.inputChunks input)
          rw [inductionHypothesis absorbed.next
            (fun _ => absorbed.output) absorbed.output rfl]
      | squeezeK expected =>
          simp only [compileWiringLazy, compileWiring]
          rw [stateEq]
          rw [inductionHypothesis (start + 1184)
            (fun _ => Permutation.scheduleOutput (start + 592))
            (Permutation.scheduleOutput (start + 592)) rfl]

/-- Samples are exposed in action order. Absorptions add no sample and every
quadratic squeeze adds exactly one. -/
@[simp] theorem compile_samples_length (start : Nat) (state : EState)
    (actions : List Action) :
    (compile start state actions).samples.length =
      (actions.filterMap fun action => match action with
        | .absorb _ => none
        | .squeezeK _ => some ()).length := by
  induction actions generalizing start state with
  | nil => rfl
  | cons action actions inductionHypothesis =>
      cases action with
      | absorb input =>
          simp [compile, inductionHypothesis]
      | squeezeK expected =>
          simp [compile, inductionHypothesis]

/-- Expected squeeze values affect only assertion expressions. They cannot
change witness recipes, computed samples, or the final state. -/
theorem compile_shape_eq (start : Nat) (state : EState)
    (left right : List Action)
    (same : left.map Action.shape = right.map Action.shape) :
    (compile start state left).recipes = (compile start state right).recipes ∧
      (compile start state left).samples =
        (compile start state right).samples ∧
      (compile start state left).output =
        (compile start state right).output := by
  induction left generalizing right start state with
  | nil =>
      cases right with
      | nil => exact ⟨rfl, rfl, rfl⟩
      | cons action actions => simp at same
  | cons leftAction leftActions inductionHypothesis =>
      cases right with
      | nil => simp at same
      | cons rightAction rightActions =>
          simp only [List.map_cons, List.cons.injEq] at same
          rcases same with ⟨headSame, tailSame⟩
          cases leftAction <;> cases rightAction
          case absorb.absorb leftInput rightInput =>
            simp only [Action.shape, ActionShape.absorb.injEq] at headSame
            subst rightInput
            let absorbed := Hash.compileAbsorptions start state
              (Hash.inputChunks leftInput)
            have tailResult := inductionHypothesis
              (start := start + absorbed.recipes.length)
              (state := absorbed.output) rightActions tailSame
            change
              absorbed.recipes ++
                    (compile (start + absorbed.recipes.length)
                      absorbed.output leftActions).recipes =
                  absorbed.recipes ++
                    (compile (start + absorbed.recipes.length)
                      absorbed.output rightActions).recipes ∧
                (compile (start + absorbed.recipes.length)
                    absorbed.output leftActions).samples =
                  (compile (start + absorbed.recipes.length)
                    absorbed.output rightActions).samples ∧
                (compile (start + absorbed.recipes.length)
                    absorbed.output leftActions).output =
                  (compile (start + absorbed.recipes.length)
                    absorbed.output rightActions).output
            exact ⟨congrArg (fun recipes => absorbed.recipes ++ recipes)
              tailResult.1, tailResult.2⟩
          case absorb.squeezeK => simp [Action.shape] at headSame
          case squeezeK.absorb => simp [Action.shape] at headSame
          case squeezeK.squeezeK leftExpected rightExpected =>
            let squeezed := Squeeze.compile start state
            have tailResult := inductionHypothesis
              (start := start + squeezed.recipes.length)
              (state := squeezed.output) rightActions tailSame
            simpa [compile, squeezed, tailResult.1, tailResult.2.1,
              tailResult.2.2]

/-- Assertion rows bind exactly the ordered expected values to the ordered
computed squeeze samples. -/
theorem compile_assertions_hold_iff (env : Env) (start : Nat)
    (state : EState) (actions : List Action) :
    ConstraintsHold env (compile start state actions).assertions ↔
      (compile start state actions).samples.map (KExpr.eval env) =
        (expectedSamples actions).map (KExpr.eval env) := by
  induction actions generalizing start state with
  | nil => simp [compile, expectedSamples, ConstraintsHold]
  | cons action actions inductionHypothesis =>
      cases action with
      | absorb input =>
          simpa [compile, expectedSamples] using
            inductionHypothesis
              (start := start +
                (Hash.compileAbsorptions start state
                  (Hash.inputChunks input)).recipes.length)
              (state := (Hash.compileAbsorptions start state
                (Hash.inputChunks input)).output)
      | squeezeK expected =>
          let squeezed := Squeeze.compile start state
          rw [show (compile start state
              (.squeezeK expected :: actions)).assertions =
              KExpr.equalities squeezed.sample expected ++
                (compile (start + squeezed.recipes.length)
                  squeezed.output actions).assertions by rfl]
          rw [Permutation.constraintsHold_append,
            KExpr.equalities_hold_iff,
            inductionHypothesis (start := start + squeezed.recipes.length)
              (state := squeezed.output)]
          simp [compile, expectedSamples, squeezed]

/-- Compilation emits exactly two assertions per quadratic squeeze and no
assertions for absorbs. This proof is structural in the action list. -/
@[simp] theorem compile_assertions_length (start : Nat) (state : EState)
    (actions : List Action) :
    (compile start state actions).assertions.length =
      assertionCount actions := by
  induction actions generalizing start state with
  | nil => rfl
  | cons action actions inductionHypothesis =>
      cases action with
      | absorb input =>
          simp [compile, assertionCount, Action.assertionCount,
            inductionHypothesis]
      | squeezeK expected =>
          simp [compile, assertionCount, Action.assertionCount,
            KExpr.equalities, inductionHypothesis]
          omega

/-- Duplex compilation allocates exactly the declared structural footprint.
The proof depends on the action list, not on emitted rows or values. -/
@[simp] theorem compile_recipes_length (start : Nat) (state : EState)
    (actions : List Action) :
    (compile start state actions).recipes.length = recipeCount actions := by
  induction actions generalizing start state with
  | nil => rfl
  | cons action actions inductionHypothesis =>
      cases action with
      | absorb input =>
          simp [compile, recipeCount, Action.recipeCount,
            inductionHypothesis]
      | squeezeK expected =>
          simp [compile, recipeCount, Action.recipeCount,
            inductionHypothesis, Squeeze.compile_recipes_length]

def Action.Below (bound : Nat) : Action → Prop
  | .absorb input =>
      ∀ expression ∈ input, expression.VarsBelow bound
  | .squeezeK expected =>
      expected.c0.VarsBelow bound ∧ expected.c1.VarsBelow bound

def ActionsBelow (bound : Nat) (actions : List Action) : Prop :=
  ∀ action ∈ actions, action.Below bound

theorem Action.below_mono {lower upper : Nat} (action : Action)
    (below : action.Below lower) (le : lower ≤ upper) :
    action.Below upper := by
  cases action with
  | absorb input =>
      intro expression member
      exact Expr.VarsBelow.mono expression (below expression member) le
  | squeezeK expected =>
      exact ⟨Expr.VarsBelow.mono _ below.1 le,
        Expr.VarsBelow.mono _ below.2 le⟩

theorem actionsBelow_mono {lower upper : Nat} {actions : List Action}
    (below : ActionsBelow lower actions) (le : lower ≤ upper) :
    ActionsBelow upper actions := by
  intro action member
  exact action.below_mono (below action member) le

theorem compile_causal (start : Nat) (state : EState)
    (actions : List Action)
    (stateBelow : ∀ lane, (state lane).VarsBelow start)
    (actionsBelow : ActionsBelow start actions) :
    RecipesCausal start (compile start state actions).recipes := by
  induction actions generalizing start state with
  | nil => trivial
  | cons action actions inductionHypothesis =>
      have headBelow := actionsBelow action (by simp)
      have tailBelow : ActionsBelow start actions := by
        intro current member
        exact actionsBelow current (by simp [member])
      cases action with
      | absorb input =>
          let absorbed := Hash.compileAbsorptions start state
            (Hash.inputChunks input)
          have headCausal := Absorb.compile_causal start state input
            stateBelow headBelow
          have outputBelow : ∀ lane,
              (absorbed.output lane).VarsBelow
                (start + absorbed.recipes.length) := by
            intro lane
            exact Absorb.compile_output_below start state input stateBelow
              headBelow lane
          have tailCausal := inductionHypothesis
            (start := start + absorbed.recipes.length)
            (state := absorbed.output) outputBelow
            (actionsBelow_mono tailBelow (by omega))
          change RecipesCausal start
            (absorbed.recipes ++
              (compile (start + absorbed.recipes.length)
                absorbed.output actions).recipes)
          exact Permutation.recipesCausal_append_causal start _ _
            headCausal tailCausal
      | squeezeK expected =>
          let squeezed := Squeeze.compile start state
          have headCausal := Squeeze.compile_causal start state stateBelow
          have outputBelow : ∀ lane,
              (squeezed.output lane).VarsBelow
                (start + squeezed.recipes.length) := by
            intro lane
            exact Squeeze.compile_output_below start state stateBelow lane
          have tailCausal := inductionHypothesis
            (start := start + squeezed.recipes.length)
            (state := squeezed.output) outputBelow
            (actionsBelow_mono tailBelow (by omega))
          change RecipesCausal start
            (squeezed.recipes ++
              (compile (start + squeezed.recipes.length)
                squeezed.output actions).recipes)
          exact Permutation.recipesCausal_append_causal start _ _
            headCausal tailCausal

theorem splitRecipeRows (env : Env) (start : Nat)
    (first second : List Expr)
    (rows : ConstraintsHold env
      (recipeConstraints start (first ++ second))) :
    ConstraintsHold env (recipeConstraints start first) ∧
      ConstraintsHold env
        (recipeConstraints (start + first.length) second) := by
  rw [Permutation.recipeConstraints_append] at rows
  exact (Permutation.constraintsHold_append env _ _).mp rows

theorem compile_sound (env : Env) (start : Nat) (state : EState)
    (actions : List Action)
    (recipeRows : ConstraintsHold env
      (recipeConstraints start (compile start state actions).recipes))
    (assertionRows : ConstraintsHold env
      (compile start state actions).assertions) :
    TraceHolds (List.ofFn (Layer.evalState env state))
      (actions.map (Action.eval env))
      (List.ofFn (Layer.evalState env
        (compile start state actions).output)) := by
  induction actions generalizing start state with
  | nil => rfl
  | cons action actions inductionHypothesis =>
      cases action with
      | absorb input =>
          let absorbed := Hash.compileAbsorptions start state
            (Hash.inputChunks input)
          let tail := compile (start + absorbed.recipes.length)
            absorbed.output actions
          have split := splitRecipeRows env start absorbed.recipes
            tail.recipes (by simpa [compile, absorbed, tail] using recipeRows)
          have absorbedSound := Absorb.compile_sound env start state input
            split.1
          have tailSound := inductionHypothesis
            (start := start + absorbed.recipes.length)
            (state := absorbed.output) split.2 (by
              simpa [compile, absorbed, tail] using assertionRows)
          simp only [List.map_cons, Action.eval, TraceHolds]
          rw [← absorbedSound]
          exact tailSound
      | squeezeK expected =>
          let squeezed := Squeeze.compile start state
          let tail := compile (start + squeezed.recipes.length)
            squeezed.output actions
          have split := splitRecipeRows env start squeezed.recipes
            tail.recipes (by simpa [compile, squeezed, tail] using recipeRows)
          have splitAssertions :=
            (Permutation.constraintsHold_append env _ _).mp (by
              simpa [compile, squeezed, tail] using assertionRows)
          have squeezedSound := Squeeze.compile_sound env start state split.1
          have sampleEquals :=
            (KExpr.equalities_hold_iff env squeezed.sample expected).mp
              splitAssertions.1
          have tailSound := inductionHypothesis
            (start := start + squeezed.recipes.length)
            (state := squeezed.output) split.2 splitAssertions.2
          simp only [List.map_cons, Action.eval, TraceHolds]
          refine ⟨sampleEquals.symm.trans squeezedSound.1, ?_⟩
          rw [← squeezedSound.2]
          exact tailSound

theorem compile_complete (env : Env) (start : Nat) (state : EState)
    (actions : List Action) (final : Spec.Poseidon2.State)
    (recipeRows : ConstraintsHold env
      (recipeConstraints start (compile start state actions).recipes))
    (trace : TraceHolds (List.ofFn (Layer.evalState env state))
      (actions.map (Action.eval env)) final) :
    ConstraintsHold env (compile start state actions).assertions ∧
      List.ofFn (Layer.evalState env
        (compile start state actions).output) = final := by
  induction actions generalizing start state with
  | nil => exact ⟨by simp [compile, ConstraintsHold], trace⟩
  | cons action actions inductionHypothesis =>
      cases action with
      | absorb input =>
          let absorbed := Hash.compileAbsorptions start state
            (Hash.inputChunks input)
          let tail := compile (start + absorbed.recipes.length)
            absorbed.output actions
          have split := splitRecipeRows env start absorbed.recipes
            tail.recipes (by simpa [compile, absorbed, tail] using recipeRows)
          have absorbedSound := Absorb.compile_sound env start state input
            split.1
          have tailTrace : TraceHolds
              (List.ofFn (Layer.evalState env absorbed.output))
              (actions.map (Action.eval env)) final := by
            rw [absorbedSound]
            simpa [Action.eval, TraceHolds] using trace
          have tailComplete := inductionHypothesis
            (start := start + absorbed.recipes.length)
            (state := absorbed.output) split.2 tailTrace
          simpa [compile, absorbed, tail] using tailComplete
      | squeezeK expected =>
          let squeezed := Squeeze.compile start state
          let tail := compile (start + squeezed.recipes.length)
            squeezed.output actions
          have split := splitRecipeRows env start squeezed.recipes
            tail.recipes (by simpa [compile, squeezed, tail] using recipeRows)
          have squeezedSound := Squeeze.compile_sound env start state split.1
          have expectedReference : expected.eval env =
              Squeeze.referenceSample
                (List.ofFn (Layer.evalState env state)) := by
            simpa [Action.eval, TraceHolds] using trace.1
          have sampleEquals : squeezed.sample.eval env = expected.eval env :=
            squeezedSound.1.trans expectedReference.symm
          have tailTrace : TraceHolds
              (List.ofFn (Layer.evalState env squeezed.output))
              (actions.map (Action.eval env)) final := by
            rw [squeezedSound.2]
            simpa [Action.eval, TraceHolds] using trace.2
          have tailComplete := inductionHypothesis
            (start := start + squeezed.recipes.length)
            (state := squeezed.output) split.2 tailTrace
          constructor
          · apply (Permutation.constraintsHold_append env _ _).mpr
            exact ⟨
              (KExpr.equalities_hold_iff env squeezed.sample expected).mpr
                sampleEquals,
              tailComplete.1⟩
          · exact tailComplete.2

def stateEqualities (left right : EState) : List Expr :=
  List.ofFn fun lane => left lane - right lane

@[simp] theorem stateEqualities_length (left right : EState) :
    (stateEqualities left right).length = 8 := by
  simp [stateEqualities]

theorem stateEqualities_hold_iff (env : Env) (left right : EState) :
    ConstraintsHold env (stateEqualities left right) ↔
      List.ofFn (Layer.evalState env left) =
        List.ofFn (Layer.evalState env right) := by
  constructor
  · intro rows
    apply congrArg List.ofFn
    funext lane
    have row := rows (left lane - right lane) (by
      rw [stateEqualities, List.mem_ofFn']
      exact Set.mem_range_self lane)
    exact sub_eq_zero.mp (by simpa using row)
  · intro equals expression member
    rw [stateEqualities, List.mem_ofFn'] at member
    rcases member with ⟨lane, rfl⟩
    have laneEquals := congrArg
      (fun values : List F => values.getD lane.val 0) equals
    change (left lane - right lane).eval env = 0
    have coordinate : (left lane).eval env = (right lane).eval env := by
      fin_cases lane <;>
        simpa [Layer.evalState, List.ofFn_succ] using laneEquals
    simpa using sub_eq_zero.mpr coordinate

theorem stateEqualities_varsBelow (left right : EState) (bound : Nat)
    (leftBelow : ∀ lane, (left lane).VarsBelow bound)
    (rightBelow : ∀ lane, (right lane).VarsBelow bound) :
    ∀ expression ∈ stateEqualities left right,
      expression.VarsBelow bound := by
  intro expression member
  rw [stateEqualities, List.mem_ofFn'] at member
  rcases member with ⟨lane, rfl⟩
  exact Expr.VarsBelow.sub _ _ bound (leftBelow lane) (rightBelow lane)

structure Interface where
  initial : Nat → EState
  actions : Nat → List Action
  final : Nat → EState

def Assumptions (interface : Interface) (offset : Nat) (_env : Env) : Prop :=
  (∀ lane, (interface.initial offset lane).VarsBelow offset) ∧
    ActionsBelow offset (interface.actions offset) ∧
    ∀ lane, (interface.final offset lane).VarsBelow offset

def SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  TraceHolds
    (List.ofFn (Layer.evalState env (interface.initial offset)))
    ((interface.actions offset).map (Action.eval env))
    (List.ofFn (Layer.evalState env (interface.final offset)))

def allAssertions (interface : Interface) (offset : Nat) : List Expr :=
  let program := compile offset (interface.initial offset)
    (interface.actions offset)
  program.assertions ++ stateEqualities program.output (interface.final offset)

theorem allAssertions_length (interface : Interface) (offset : Nat) :
    (allAssertions interface offset).length =
      assertionCount (interface.actions offset) + 8 := by
  unfold allAssertions
  rw [List.length_append, compile_assertions_length, stateEqualities_length]

def opsAt (interface : Interface) (offset : Nat) : List Op :=
  let program := compile offset (interface.initial offset)
    (interface.actions offset)
  Op.witness (WitnessBatch.arithmetic offset program.recipes) ::
    (allAssertions interface offset).map Op.assertZero

def main (interface : Interface) : Circuit Unit := fun offset =>
  let program := compile offset (interface.initial offset)
    (interface.actions offset)
  ((), offset + program.recipes.length, opsAt interface offset)

@[simp] theorem main_ops (interface : Interface) (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := rfl

theorem witness_mem (interface : Interface) (offset : Nat) :
    Op.witness
      (WitnessBatch.arithmetic offset
        (compile offset (interface.initial offset)
          (interface.actions offset)).recipes) ∈ opsAt interface offset := by
  simp [opsAt]

theorem assertion_mem (interface : Interface) (offset : Nat)
    (expression : Expr) (member : expression ∈ allAssertions interface offset) :
    Op.assertZero expression ∈ opsAt interface offset := by
  unfold opsAt
  simp [member]

theorem assertions_localLength (expressions : List Expr) :
    localLength (expressions.map Op.assertZero) = 0 := by
  simp [localLength, Function.comp_def, Op.localLength]

theorem opsAt_localLength (interface : Interface) (offset : Nat) :
    localLength (opsAt interface offset) =
      (compile offset (interface.initial offset)
        (interface.actions offset)).recipes.length := by
  unfold opsAt
  change
    (compile offset (interface.initial offset)
      (interface.actions offset)).recipes.length +
        localLength ((allAssertions interface offset).map Op.assertZero) = _
  rw [assertions_localLength]
  omega

theorem flatConstraints_opsAt (interface : Interface) (offset : Nat) :
    flatConstraints (opsAt interface offset) =
      recipeConstraints offset
          (compile offset (interface.initial offset)
            (interface.actions offset)).recipes ++
        allAssertions interface offset := by
  have flatten : List.flatMap Op.flatConstraints
      ((allAssertions interface offset).map Op.assertZero) =
        allAssertions interface offset := by
    induction allAssertions interface offset with
    | nil => rfl
    | cons expression expressions inductionHypothesis =>
        simp only [List.map_cons, List.flatMap_cons, Op.flatConstraints,
          List.singleton_append]
        rw [inductionHypothesis]
  unfold opsAt flatConstraints
  dsimp only
  simp only [List.flatMap_cons, Op.flatConstraints]
  rw [flatten]
  simp

/-- Exact number of parent-visible operations without unfolding a schedule. -/
theorem operations_length (interface : Interface) (offset : Nat) :
    (opsAt interface offset).length =
      1 + (assertionCount (interface.actions offset) + 8) := by
  unfold opsAt
  rw [List.length_cons, List.length_map, allAssertions_length]
  omega

/-- Exact flattened row count without evaluating the emitted recipes. -/
theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (opsAt interface offset)).length =
      recipeCount (interface.actions offset) +
        (assertionCount (interface.actions offset) + 8) := by
  rw [flatConstraints_opsAt, List.length_append, recipeConstraints_length,
    allAssertions_length, compile_recipes_length]

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (_assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    SpecHolds interface offset env := by
  let program := compile offset (interface.initial offset)
    (interface.actions offset)
  have recipeRows : ConstraintsHold env
      (recipeConstraints offset program.recipes) :=
    rows (Op.witness (WitnessBatch.arithmetic offset program.recipes)) (by
      rw [main_ops]
      simpa [program] using witness_mem interface offset)
  have assertionRows : ConstraintsHold env
      (allAssertions interface offset) := by
    intro expression member
    exact rows (Op.assertZero expression) (by
      rw [main_ops]
      exact assertion_mem interface offset expression member)
  have splitAssertions :=
    (Permutation.constraintsHold_append env _ _).mp (by
      simpa [allAssertions, program] using assertionRows)
  have trace := compile_sound env offset (interface.initial offset)
    (interface.actions offset) recipeRows splitAssertions.1
  have finalEquals := (stateEqualities_hold_iff env program.output
    (interface.final offset)).mp splitAssertions.2
  unfold SpecHolds
  rw [← finalEquals]
  exact trace

theorem action_eval_preserved (before after : Env) (bound : Nat)
    (action : Action) (below : action.Below bound)
    (agrees : ∀ index, index < bound → after index = before index) :
    action.eval after = action.eval before := by
  cases action with
  | absorb input =>
      apply congrArg ValueAction.absorb
      unfold Hash.evalList
      apply List.map_congr_left
      intro expression member
      exact expression.eval_eq_of_agree_below bound after before
        (below expression member) agrees
  | squeezeK expected =>
      apply congrArg ValueAction.squeezeK
      exact congrArg₂ K.mk
        (expected.c0.eval_eq_of_agree_below bound after before below.1 agrees)
        (expected.c1.eval_eq_of_agree_below bound after before below.2 agrees)

/-- The semantic trace is stable when every external input wire is
unchanged. -/
theorem specHolds_of_agree_below (interface : Interface) (offset : Nat)
    (before after : Env) (assumptions : Assumptions interface offset before)
    (agrees : ∀ index, index < offset → after index = before index)
    (specification : SpecHolds interface offset before) :
    SpecHolds interface offset after := by
  have initialEval :
      List.ofFn (Layer.evalState after (interface.initial offset)) =
        List.ofFn (Layer.evalState before (interface.initial offset)) := by
    apply congrArg List.ofFn
    funext lane
    exact (interface.initial offset lane).eval_eq_of_agree_below offset
      after before (assumptions.1 lane) agrees
  have actionsEval :
      (interface.actions offset).map (Action.eval after) =
        (interface.actions offset).map (Action.eval before) := by
    apply List.map_congr_left
    intro action member
    exact action_eval_preserved before after offset action
      (assumptions.2.1 action member) agrees
  have finalEval :
      List.ofFn (Layer.evalState after (interface.final offset)) =
        List.ofFn (Layer.evalState before (interface.final offset)) := by
    apply congrArg List.ofFn
    funext lane
    exact (interface.final offset lane).eval_eq_of_agree_below offset
      after before (assumptions.2.2 lane) agrees
  unfold SpecHolds at specification ⊢
  rw [initialEval, actionsEval, finalEval]
  exact specification

/-- Compilation output and non-final assertions use only the external
prefix and the completed local recipe interval. The proof is structural in
the action list. -/
theorem compile_scope (start : Nat) (state : EState) (actions : List Action)
    (stateBelow : ∀ lane, (state lane).VarsBelow start)
    (actionsBelow : ActionsBelow start actions) :
    (∀ lane, ((compile start state actions).output lane).VarsBelow
      (start + (compile start state actions).recipes.length)) ∧
    (∀ expression ∈ (compile start state actions).assertions,
      expression.VarsBelow
        (start + (compile start state actions).recipes.length)) := by
  induction actions generalizing start state with
  | nil =>
      exact ⟨by simpa [compile] using stateBelow, by simp [compile]⟩
  | cons action actions inductionHypothesis =>
      have headBelow := actionsBelow action (by simp)
      have tailBelow : ActionsBelow start actions := by
        intro current member
        exact actionsBelow current (by simp [member])
      cases action with
      | absorb input =>
          let absorbed := Hash.compileAbsorptions start state
            (Hash.inputChunks input)
          have outputBelow : ∀ lane,
              (absorbed.output lane).VarsBelow
                (start + absorbed.recipes.length) := by
            intro lane
            exact Absorb.compile_output_below start state input stateBelow
              headBelow lane
          have tailScope := inductionHypothesis
            (start := start + absorbed.recipes.length)
            (state := absorbed.output) outputBelow
            (actionsBelow_mono tailBelow (by omega))
          simpa [compile, absorbed, List.length_append, Nat.add_assoc] using
            tailScope
      | squeezeK expected =>
          let squeezed := Squeeze.compile start state
          have outputBelow : ∀ lane,
              (squeezed.output lane).VarsBelow
                (start + squeezed.recipes.length) := by
            intro lane
            exact Squeeze.compile_output_below start state stateBelow lane
          have tailScope := inductionHypothesis
            (start := start + squeezed.recipes.length)
            (state := squeezed.output) outputBelow
            (actionsBelow_mono tailBelow (by omega))
          constructor
          · simpa [compile, squeezed, List.length_append, Nat.add_assoc] using
              tailScope.1
          · intro expression member
            change expression ∈
                KExpr.equalities squeezed.sample expected ++
                  (compile (start + squeezed.recipes.length)
                    squeezed.output actions).assertions at member
            rcases List.mem_append.mp member with headMember | tailMember
            · apply KExpr.equalities_varsBelow squeezed.sample expected
                  (start +
                    (squeezed.recipes ++
                      (compile (start + squeezed.recipes.length)
                        squeezed.output actions).recipes).length)
              · have sampleBelow : squeezed.sample.VarsBelow
                    (start + squeezed.recipes.length) := by
                  simpa [squeezed] using
                    Squeeze.compile_sample_below start state stateBelow
                constructor <;> apply Expr.VarsBelow.mono _
                · exact sampleBelow.1
                · simp only [List.length_append]
                  omega
                · exact sampleBelow.2
                · simp only [List.length_append]
                  omega
              · constructor <;> apply Expr.VarsBelow.mono _
                · exact headBelow.1
                · simp only [List.length_append]
                  omega
                · exact headBelow.2
                · simp only [List.length_append]
                  omega
              · exact headMember
            · have below := tailScope.2 expression tailMember
              simpa [compile, squeezed, List.length_append, Nat.add_assoc]
                using below

/-- Every computed squeeze sample lies in the causal recipe interval. -/
theorem compile_samples_scope (start : Nat) (state : EState)
    (actions : List Action)
    (stateBelow : ∀ lane, (state lane).VarsBelow start)
    (actionsBelow : ActionsBelow start actions) :
    ∀ sample ∈ (compile start state actions).samples,
      sample.VarsBelow (start + (compile start state actions).recipes.length) := by
  induction actions generalizing start state with
  | nil => simp [compile]
  | cons action actions inductionHypothesis =>
      have headBelow := actionsBelow action (by simp)
      have tailBelow : ActionsBelow start actions := by
        intro current member
        exact actionsBelow current (by simp [member])
      cases action with
      | absorb input =>
          let absorbed := Hash.compileAbsorptions start state
            (Hash.inputChunks input)
          have outputBelow : ∀ lane,
              (absorbed.output lane).VarsBelow
                (start + absorbed.recipes.length) := by
            intro lane
            exact Absorb.compile_output_below start state input stateBelow
              headBelow lane
          have tailScope := inductionHypothesis
            (start := start + absorbed.recipes.length)
            (state := absorbed.output) outputBelow
            (actionsBelow_mono tailBelow (by omega))
          simpa [compile, absorbed, List.length_append, Nat.add_assoc] using
            tailScope
      | squeezeK expected =>
          let squeezed := Squeeze.compile start state
          have outputBelow : ∀ lane,
              (squeezed.output lane).VarsBelow
                (start + squeezed.recipes.length) := by
            intro lane
            exact Squeeze.compile_output_below start state stateBelow lane
          have tailScope := inductionHypothesis
            (start := start + squeezed.recipes.length)
            (state := squeezed.output) outputBelow
            (actionsBelow_mono tailBelow (by omega))
          intro sample member
          change sample ∈ squeezed.sample ::
            (compile (start + squeezed.recipes.length)
              squeezed.output actions).samples at member
          simp only [List.mem_cons] at member
          rcases member with rfl | member
          · have sampleBelow :=
              Squeeze.compile_sample_below start state stateBelow
            exact KExpr.varsBelow_mono squeezed.sample sampleBelow (by
              simp only [compile, List.length_append]
              omega)
          · have below := tailScope sample member
            simpa [compile, squeezed, List.length_append, Nat.add_assoc]
              using below

/-- Every flattened Duplex row is scoped to the call's completed local
interval. -/
theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (main interface) offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (main interface) offset)) := by
  let program := compile offset (interface.initial offset)
    (interface.actions offset)
  have scope := compile_scope offset (interface.initial offset)
    (interface.actions offset) assumptions.1 assumptions.2.1
  have recipeScope := recipeConstraints_varsBelow_of_causal offset
    program.recipes (compile_causal offset (interface.initial offset)
      (interface.actions offset) assumptions.1 assumptions.2.1)
  have finalBelow : ∀ lane,
      (interface.final offset lane).VarsBelow
        (offset + program.recipes.length) := by
    intro lane
    exact Expr.VarsBelow.mono _ (assumptions.2.2 lane) (by omega)
  have finalScope := stateEqualities_varsBelow program.output
    (interface.final offset) (offset + program.recipes.length) scope.1
      finalBelow
  rw [show Circuit.ops (main interface) offset =
      opsAt interface offset by rfl, flatConstraints_opsAt,
    opsAt_localLength]
  intro expression member
  rcases List.mem_append.mp member with recipeMember | assertionMember
  · simpa [program] using recipeScope expression recipeMember
  · unfold allAssertions at assertionMember
    dsimp only at assertionMember
    rcases List.mem_append.mp assertionMember with compileMember | finalMember
    · have below := scope.2 expression compileMember
      simpa [program, opsAt_localLength] using below
    · have below := finalScope expression finalMember
      simpa [program, opsAt_localLength] using below

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  let program := compile offset (interface.initial offset)
    (interface.actions offset)
  let completed := executeRecipes env offset program.recipes
  have causal := compile_causal offset (interface.initial offset)
    (interface.actions offset) assumptions.1 assumptions.2.1
  have recipeRows : ConstraintsHold completed
      (recipeConstraints offset program.recipes) :=
    executeRecipes_holds_recipeConstraints env offset program.recipes causal
  have agreesBelow : ∀ index, index < offset → completed index = env index :=
    executeRecipes_agrees_below env offset program.recipes
  have initialEval :
      List.ofFn (Layer.evalState completed (interface.initial offset)) =
        List.ofFn (Layer.evalState env (interface.initial offset)) := by
    apply congrArg List.ofFn
    funext lane
    exact (interface.initial offset lane).eval_eq_of_agree_below offset
      completed env (assumptions.1 lane) agreesBelow
  have actionsEval :
      (interface.actions offset).map (Action.eval completed) =
        (interface.actions offset).map (Action.eval env) := by
    apply List.map_congr_left
    intro action member
    exact action_eval_preserved env completed offset action
      (assumptions.2.1 action member) agreesBelow
  have finalEval :
      List.ofFn (Layer.evalState completed (interface.final offset)) =
        List.ofFn (Layer.evalState env (interface.final offset)) := by
    apply congrArg List.ofFn
    funext lane
    exact (interface.final offset lane).eval_eq_of_agree_below offset
      completed env (assumptions.2.2 lane) agreesBelow
  have completedSpec : TraceHolds
      (List.ofFn (Layer.evalState completed (interface.initial offset)))
      ((interface.actions offset).map (Action.eval completed))
      (List.ofFn (Layer.evalState completed (interface.final offset))) := by
    rw [initialEval, actionsEval, finalEval]
    exact specification
  have traceComplete := compile_complete completed offset
    (interface.initial offset) (interface.actions offset)
    (List.ofFn (Layer.evalState completed (interface.final offset)))
    recipeRows completedSpec
  have finalRows := (stateEqualities_hold_iff completed program.output
    (interface.final offset)).mpr traceComplete.2
  have allRows : ConstraintsHold completed
      (recipeConstraints offset program.recipes ++
        allAssertions interface offset) := by
    apply (Permutation.constraintsHold_append completed _ _).mpr
    refine ⟨recipeRows, ?_⟩
    apply (Permutation.constraintsHold_append completed _ _).mpr
    simpa [allAssertions, program] using ⟨traceComplete.1, finalRows⟩
  refine ⟨completed, ?_, ?_⟩
  · have agrees := executeRecipes_agreesOutside env offset program.recipes
    rw [main_ops, opsAt_localLength]
    exact agrees
  · change ConstraintsHold completed
      (flatConstraints (opsAt interface offset))
    rw [flatConstraints_opsAt]
    exact allRows

def circuit (interface : Interface) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  soundness := soundness interface
  completeness := completeness interface

/-! ## Child-owned final-state variant -/

namespace Owned

/-!
Obligation: execute one Duplex action schedule and expose its compiled final
state directly to a parent circuit.

The interface has no external final-state wire. Squeeze expectations remain
authoritative action inputs and are still constrained by compiler assertions.
-/

structure Interface where
  initial : Nat → EState
  actions : Nat → List Action

def program (interface : Interface) (offset : Nat) : Program :=
  compile offset (interface.initial offset) (interface.actions offset)

def output (interface : Interface) (offset : Nat) : EState :=
  (program interface offset).output

def Assumptions (interface : Interface) (offset : Nat) (_env : Env) : Prop :=
  (∀ lane, (interface.initial offset lane).VarsBelow offset) ∧
    ActionsBelow offset (interface.actions offset)

def SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  TraceHolds
    (List.ofFn (Layer.evalState env (interface.initial offset)))
    ((interface.actions offset).map (Action.eval env))
    (List.ofFn (Layer.evalState env (output interface offset)))

def allAssertions (interface : Interface) (offset : Nat) : List Expr :=
  (program interface offset).assertions

def opsAt (interface : Interface) (offset : Nat) : List Op :=
  Op.witness (WitnessBatch.arithmetic offset
    (program interface offset).recipes) ::
    (allAssertions interface offset).map Op.assertZero

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), offset + (program interface offset).recipes.length,
    opsAt interface offset)

@[simp] theorem main_ops (interface : Interface) (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := by
  rfl

private theorem flatConstraints_assertions (expressions : List Expr) :
    flatConstraints (expressions.map Op.assertZero) = expressions := by
  induction expressions with
  | nil => rfl
  | cons expression rest inductionHypothesis =>
      change [expression] ++ flatConstraints (rest.map Op.assertZero) =
        expression :: rest
      rw [inductionHypothesis]
      rfl

theorem flatConstraints_opsAt (interface : Interface) (offset : Nat) :
    flatConstraints (opsAt interface offset) =
      recipeConstraints offset (program interface offset).recipes ++
        allAssertions interface offset := by
  unfold opsAt flatConstraints
  simp only [List.flatMap_cons, Op.flatConstraints]
  rw [show List.flatMap Op.flatConstraints
      ((allAssertions interface offset).map Op.assertZero) =
        allAssertions interface offset by
      simpa [flatConstraints] using
        flatConstraints_assertions (allAssertions interface offset)]
  simp

theorem opsAt_localLength (interface : Interface) (offset : Nat) :
    localLength (opsAt interface offset) =
      (program interface offset).recipes.length := by
  unfold opsAt
  change (program interface offset).recipes.length +
    localLength ((allAssertions interface offset).map Op.assertZero) = _
  rw [assertions_localLength]
  omega

theorem operations_length (interface : Interface) (offset : Nat) :
    (opsAt interface offset).length =
      1 + assertionCount (interface.actions offset) := by
  unfold opsAt allAssertions program
  rw [List.length_cons, List.length_map, compile_assertions_length]
  omega

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (opsAt interface offset)).length =
      recipeCount (interface.actions offset) +
        assertionCount (interface.actions offset) := by
  rw [flatConstraints_opsAt, List.length_append,
    recipeConstraints_length]
  unfold allAssertions program
  rw [compile_recipes_length, compile_assertions_length]

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (_assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    SpecHolds interface offset env := by
  have recipeRows : ConstraintsHold env
      (recipeConstraints offset (program interface offset).recipes) :=
    rows (Op.witness (WitnessBatch.arithmetic offset
      (program interface offset).recipes)) (by
      rw [main_ops]
      simp [opsAt])
  have assertionRows : ConstraintsHold env
      (allAssertions interface offset) := by
    intro expression member
    exact rows (Op.assertZero expression) (by
      rw [main_ops]
      simp [opsAt, member])
  exact compile_sound env offset (interface.initial offset)
    (interface.actions offset) recipeRows (by
      simpa [allAssertions, program] using assertionRows)

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  let compiled := program interface offset
  let completed := executeRecipes env offset compiled.recipes
  have causal := compile_causal offset (interface.initial offset)
    (interface.actions offset) assumptions.1 assumptions.2
  have recipeRows : ConstraintsHold completed
      (recipeConstraints offset compiled.recipes) :=
    executeRecipes_holds_recipeConstraints env offset compiled.recipes causal
  have agreesBelow : ∀ index, index < offset → completed index = env index :=
    executeRecipes_agrees_below env offset compiled.recipes
  have initialEval :
      List.ofFn (Layer.evalState completed (interface.initial offset)) =
        List.ofFn (Layer.evalState env (interface.initial offset)) := by
    apply congrArg List.ofFn
    funext lane
    exact (interface.initial offset lane).eval_eq_of_agree_below offset
      completed env (assumptions.1 lane) agreesBelow
  have actionsEval :
      (interface.actions offset).map (Action.eval completed) =
        (interface.actions offset).map (Action.eval env) := by
    apply List.map_congr_left
    intro action member
    exact action_eval_preserved env completed offset action
      (assumptions.2 action member) agreesBelow
  have completedSpec : TraceHolds
      (List.ofFn (Layer.evalState completed (interface.initial offset)))
      ((interface.actions offset).map (Action.eval completed))
      (List.ofFn (Layer.evalState env (output interface offset))) := by
    rw [initialEval, actionsEval]
    exact specification
  have traceComplete := compile_complete completed offset
    (interface.initial offset) (interface.actions offset)
    (List.ofFn (Layer.evalState env (output interface offset))) recipeRows
      completedSpec
  refine ⟨completed, ?_, ?_⟩
  · have agrees := executeRecipes_agreesOutside env offset compiled.recipes
    rw [main_ops, opsAt_localLength]
    exact agrees
  · change ConstraintsHold completed
      (flatConstraints (opsAt interface offset))
    rw [flatConstraints_opsAt]
    exact (Permutation.constraintsHold_append completed _ _).mpr
      ⟨by simpa [compiled, program] using recipeRows,
        by simpa [allAssertions, compiled, program] using traceComplete.1⟩

/-- Absorb-only schedules have no assertion rows. Honest execution therefore
constructs the owned final state without a semantic premise. -/
theorem build_of_no_assertions (interface : Interface) (env : Env)
    (offset : Nat) (assumptions : Assumptions interface offset env)
    (none : allAssertions interface offset = []) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  let compiled := program interface offset
  let completed := executeRecipes env offset compiled.recipes
  have causal := compile_causal offset (interface.initial offset)
    (interface.actions offset) assumptions.1 assumptions.2
  have recipeRows : ConstraintsHold completed
      (recipeConstraints offset compiled.recipes) :=
    executeRecipes_holds_recipeConstraints env offset compiled.recipes causal
  refine ⟨completed, ?_, ?_⟩
  · have agrees := executeRecipes_agreesOutside env offset compiled.recipes
    rw [main_ops, opsAt_localLength]
    exact agrees
  · change ConstraintsHold completed
      (flatConstraints (opsAt interface offset))
    rw [flatConstraints_opsAt, none, List.append_nil]
    simpa [compiled, program] using recipeRows

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (main interface) offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (main interface) offset)) := by
  have scope := compile_scope offset (interface.initial offset)
    (interface.actions offset) assumptions.1 assumptions.2
  have recipeScope := recipeConstraints_varsBelow_of_causal offset
    (program interface offset).recipes
    (compile_causal offset (interface.initial offset)
      (interface.actions offset) assumptions.1 assumptions.2)
  rw [main_ops, flatConstraints_opsAt, opsAt_localLength]
  intro expression member
  rcases List.mem_append.mp member with recipeMember | assertionMember
  · exact recipeScope expression recipeMember
  · exact scope.2 expression (by
      simpa [allAssertions, program] using assertionMember)

theorem output_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ lane, (output interface offset lane).VarsBelow
      (offset + localLength (Circuit.ops (main interface) offset)) := by
  have scope := (compile_scope offset (interface.initial offset)
    (interface.actions offset) assumptions.1 assumptions.2).1
  rw [main_ops, opsAt_localLength]
  simpa [output, program] using scope

/-- The owned semantic trace is stable when the complete local interval is
unchanged. The wider bound is necessary because the owned output contains
compiled recipe variables. -/
theorem specHolds_of_agree_below (interface : Interface) (offset : Nat)
    (before after : Env) (assumptions : Assumptions interface offset before)
    (agrees : ∀ index,
      index < offset + localLength (Circuit.ops (main interface) offset) →
        after index = before index)
    (specification : SpecHolds interface offset before) :
    SpecHolds interface offset after := by
  let bound := offset + localLength (Circuit.ops (main interface) offset)
  have offsetLe : offset ≤ bound := by
    simp [bound]
  have initialEval :
      List.ofFn (Layer.evalState after (interface.initial offset)) =
        List.ofFn (Layer.evalState before (interface.initial offset)) := by
    apply congrArg List.ofFn
    funext lane
    exact (interface.initial offset lane).eval_eq_of_agree_below bound
      after before
      (Expr.VarsBelow.mono _ (assumptions.1 lane) offsetLe)
      (by simpa [bound] using agrees)
  have actionsEval :
      (interface.actions offset).map (Action.eval after) =
        (interface.actions offset).map (Action.eval before) := by
    apply List.map_congr_left
    intro action member
    exact action_eval_preserved before after offset action
      (assumptions.2 action member)
      (fun index below => agrees index (lt_of_lt_of_le below offsetLe))
  have finalEval :
      List.ofFn (Layer.evalState after (output interface offset)) =
        List.ofFn (Layer.evalState before (output interface offset)) := by
    apply congrArg List.ofFn
    funext lane
    exact (output interface offset lane).eval_eq_of_agree_below bound
      after before
      (by simpa [bound] using
        output_varsBelow interface offset before assumptions lane)
      (by simpa [bound] using agrees)
  unfold SpecHolds at specification ⊢
  rw [initialEval, actionsEval, finalEval]
  exact specification

def circuit (interface : Interface) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  soundness := soundness interface
  completeness := completeness interface

end Owned

end NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal
