import NightstreamFPrime.Export.Stage1.Rows
import NightstreamFPrime.Layout.Stage1.Spartan
import NightstreamFPrime.Export.Pilot
import NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal
import NightstreamFPrime.Layout.Poseidon2.Duplex

/-!
Owns the compact Poseidon2 invocation schedule for Stage 1 transcript leaves.

The compiler follows the authoritative Duplex action list. It does not infer
transcript order from Rust or from expanded R1CS rows.
-/

namespace NightstreamFPrime.Export.Stage1.Invocations

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Stage1.Rows
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1

abbrev EState := Layer.EState
abbrev Action := Formal.Action

/-- Executable affine lowering. The zero fallback is unreachable for the
proved transcript action inputs. -/
def affineCombination (expression : Expr) : R1CS.LinearCombination :=
  match R1CS.lowerAffine expression with
  | some lowered => lowered.combination
  | none => R1CS.LinearCombination.zero

theorem affineCombination_eq
    {expression : Expr} {lowered : R1CS.AffineResult expression}
    (result : R1CS.lowerAffine expression = some lowered) :
    affineCombination expression = lowered.combination := by
  simp [affineCombination, result]

/-- Convert one source-layout affine expression to its Spartan combination. -/
def inputCombination (expression : Expr) : SparseCombination :=
  sparseCombination
    (Spartan.remapCombination (affineCombination expression))

theorem inputCombination_eval {expression : Expr}
    (affine : R1CS.IsAffine expression) (env : Env) :
    (inputCombination expression).toR1CS.eval env =
      expression.eval (Spartan.pullback env) := by
  rcases affine with ⟨lowered, loweredEq⟩
  unfold inputCombination
  rw [sparseCombination_toR1CS, Spartan.remapCombination_eval,
    affineCombination_eq loweredEq, lowered.sound]

/-- Every term of a lowered PiCCS-local invocation input maps either before
that invocation or into the stable Spartan public suffix. -/
theorem inputCombination_termsOutside (expression : Expr)
    (start ceiling : Nat)
    (startLocal : Spartan.piCcsPhaseOffset ≤ start)
    (ceilingPrivate : ceiling ≤ Spartan.privateColumnCount)
    (affine : R1CS.IsAffine expression)
    (scope : expression.VarsBelow start) :
    ∀ term ∈ (inputCombination expression).toR1CS.terms,
      term.1 < Spartan.sourceToSpartan start ∨ ceiling ≤ term.1 := by
  rcases affine with ⟨lowered, loweredEq⟩
  have sourceScope := R1CS.lowerAffine_varsBelow expression start scope
    lowered loweredEq
  intro term member
  rw [inputCombination, sparseCombination_toR1CS,
    affineCombination_eq loweredEq] at member
  unfold Spartan.remapCombination at member
  simp only [List.mem_map] at member
  rcases member with ⟨sourceTerm, sourceMember, rfl⟩
  rcases Spartan.sourceToSpartan_before_piCcsLocal sourceTerm.1 start
      startLocal (sourceScope sourceTerm sourceMember) with
    mappedBefore | mappedPublic
  · exact Or.inl mappedBefore
  · exact Or.inr (by omega)

def invocationInputs (state : EState) : List SparseCombination :=
  List.ofFn fun lane : Fin 8 => inputCombination (state lane)

@[simp] theorem invocationInputs_length (state : EState) :
    (invocationInputs state).length = 8 := by
  simp [invocationInputs]

def invocation (phase rowStart witnessStart : Nat)
    (state : EState) : PermutationInvocation where
  phase := phase
  rowStart := rowStart
  witnessStart := Spartan.sourceToSpartan witnessStart
  inputs := invocationInputs state

@[simp] theorem invocation_witnessStart
    (phase rowStart witnessStart : Nat) (state : EState) :
    (invocation phase rowStart witnessStart state).witnessStart =
      Spartan.sourceToSpartan witnessStart := by
  rfl

/-- Every sparse input of one invocation is either fixed before its local
witness interval or belongs to the stable suffix after the complete schedule. -/
def InvocationInputsOutside (ceiling : Nat)
    (invocation : PermutationInvocation) : Prop :=
  ∀ lane : Fin 8,
    ∀ term ∈ (invocationInputCombination invocation lane.val).toR1CS.terms,
      term.1 < invocation.witnessStart ∨ ceiling ≤ term.1

theorem invocation_inputsOutside (phase rowStart witnessStart ceiling : Nat)
    (state : EState)
    (witnessLocal : Spartan.piCcsPhaseOffset ≤ witnessStart)
    (ceilingPrivate : ceiling ≤ Spartan.privateColumnCount)
    (stateAffine : Poseidon2.StateAffine state)
    (stateBelow : ∀ lane, (state lane).VarsBelow witnessStart) :
    InvocationInputsOutside ceiling
      (invocation phase rowStart witnessStart state) := by
  intro lane term member
  have selected : invocationInputCombination
      (invocation phase rowStart witnessStart state) lane.val =
        inputCombination (state lane) := by
    change (List.ofFn (fun current : Fin 8 =>
      inputCombination (state current))).getD lane.val
        zeroSparseCombination = inputCombination (state lane)
    exact NightstreamFPrime.Lifecycle.PriorStateHash.ofFn_getD
      (fun current : Fin 8 => inputCombination (state current)) lane
      zeroSparseCombination
  rw [selected] at member
  exact inputCombination_termsOutside (state lane) witnessStart ceiling
    witnessLocal ceilingPrivate (stateAffine lane) (stateBelow lane) term member

/-- The fixed production permutation writes its final eight lanes in the last
eight cells of its 592-cell witness interval. Package tracing needs this state,
not the 592 internal recipe expressions. -/
def permutationOutput (witnessStart : Nat) : EState :=
  Permutation.freshState (witnessStart + 584)

theorem permutationOutput_affine (witnessStart : Nat) :
    Poseidon2.StateAffine (permutationOutput witnessStart) := by
  intro lane
  simp [permutationOutput, Permutation.freshState]

theorem permutationOutput_varsBelow (witnessStart : Nat) :
    ∀ lane, (permutationOutput witnessStart lane).VarsBelow
      (witnessStart + 592) := by
  intro lane
  simpa [permutationOutput, Nat.add_assoc] using
    Permutation.freshState_varsBelow (witnessStart + 584) lane

/-- The compact output slice is exactly the output of the authoritative
Poseidon2 circuit compiler. -/
theorem permutationOutput_eq_compile (witnessStart : Nat) (state : EState) :
    permutationOutput witnessStart =
      (Permutation.compile witnessStart state Permutation.schedule).output := by
  funext lane
  rfl

/-- One satisfying compact invocation is the exact symbolic Poseidon2
transition at its source-layout witness interval. -/
theorem invocation_sound (phase rowStart witnessStart : Nat)
    (state : EState) (env : Env)
    (witnessLocal : Spartan.piCcsPhaseOffset ≤ witnessStart)
    (stateAffine : Poseidon2.StateAffine state)
    (holds : PermutationInvocationHolds (PilotData.circuitPackage ())
      (invocation phase rowStart witnessStart state) env) :
    Layer.evalState (Spartan.pullback env) (permutationOutput witnessStart) =
      Permutation.runF Permutation.schedule
        (Layer.evalState (Spartan.pullback env) state) := by
  have canonical :=
    NightstreamFPrime.Export.Pilot.canonicalPermutationInvocation_sound
      (invocation phase rowStart witnessStart state) env holds
  have outputBoundary :
      (fun lane : Fin 8 => env
        ((invocation phase rowStart witnessStart state).witnessStart +
          (PilotData.circuitPackage ()).permutation.outputLocalStart +
          lane.val)) =
        Layer.evalState (Spartan.pullback env)
          (permutationOutput witnessStart) := by
    funext lane
    change env (Spartan.sourceToSpartan witnessStart + 584 + lane.val) =
      env (Spartan.sourceToSpartan (witnessStart + 584 + lane.val))
    apply congrArg env
    calc
      Spartan.sourceToSpartan witnessStart + 584 + lane.val =
          Spartan.sourceToSpartan witnessStart + (584 + lane.val) := by omega
      _ = Spartan.sourceToSpartan (witnessStart + (584 + lane.val)) :=
        (Spartan.sourceToSpartan_add_of_piCcsLocal witnessStart
          (584 + lane.val) witnessLocal).symm
      _ = Spartan.sourceToSpartan (witnessStart + 584 + lane.val) := by
        congr 1
        omega
  have inputBoundary :
      (fun lane : Fin 8 =>
        (invocationInputCombination
          (invocation phase rowStart witnessStart state) lane.val).toR1CS.eval
            env) =
        Layer.evalState (Spartan.pullback env) state := by
    funext lane
    have selected : invocationInputCombination
        (invocation phase rowStart witnessStart state) lane.val =
          inputCombination (state lane) := by
      change (List.ofFn (fun current : Fin 8 =>
        inputCombination (state current))).getD lane.val
          zeroSparseCombination = inputCombination (state lane)
      exact NightstreamFPrime.Lifecycle.PriorStateHash.ofFn_getD
        (fun current : Fin 8 => inputCombination (state current)) lane
        zeroSparseCombination
    rw [selected]
    exact inputCombination_eval (stateAffine lane) env
  exact outputBoundary.symm.trans
    (canonical.trans (congrArg (Permutation.runF Permutation.schedule)
      inputBoundary))

structure Trace where
  rowNext : Nat
  witnessNext : Nat
  state : EState
  invocations : List PermutationInvocation

def compileBlocks (phase : Nat) : Nat → Nat → EState →
    List (List Expr) → Trace
  | rowStart, witnessStart, state, [] =>
      ⟨rowStart, witnessStart, state, []⟩
  | rowStart, witnessStart, state, block :: blocks =>
      let input := Hash.absorbE state block
      let tail := compileBlocks phase (rowStart + 592)
        (witnessStart + 592) (permutationOutput witnessStart) blocks
      ⟨tail.rowNext, tail.witnessNext, tail.state,
        invocation phase rowStart witnessStart input :: tail.invocations⟩

/-- A held compact absorb trace has the exact reference block-fold
semantics. The proof follows the block list, not a materialized row list. -/
theorem compileBlocks_sound (phase rowStart witnessStart : Nat)
    (state : EState) (blocks : List (List Expr)) (env : Env)
    (witnessLocal : Spartan.piCcsPhaseOffset ≤ witnessStart)
    (stateAffine : Poseidon2.StateAffine state)
    (blocksAffine : Poseidon2.BlocksAffine blocks)
    (holds : ∀ current ∈
      (compileBlocks phase rowStart witnessStart state blocks).invocations,
      PermutationInvocationHolds (PilotData.circuitPackage ()) current env) :
    Layer.evalState (Spartan.pullback env)
        (compileBlocks phase rowStart witnessStart state blocks).state =
      Hash.absorbManyF (Layer.evalState (Spartan.pullback env) state)
        (blocks.map (Hash.evalList (Spartan.pullback env))) := by
  induction blocks generalizing rowStart witnessStart state with
  | nil => rfl
  | cons block blocks inductionHypothesis =>
      have blockAffine : Poseidon2.ListAffine block :=
        blocksAffine block (by simp)
      have restAffine : Poseidon2.BlocksAffine blocks := by
        intro current member
        exact blocksAffine current (by simp [member])
      have absorbedAffine : Poseidon2.StateAffine (Hash.absorbE state block) :=
        Poseidon2.absorbE_affine state block stateAffine blockAffine
      have headHolds : PermutationInvocationHolds
          (PilotData.circuitPackage ())
          (invocation phase rowStart witnessStart
            (Hash.absorbE state block)) env :=
        holds _ (by simp [compileBlocks])
      have headSound := invocation_sound phase rowStart witnessStart
        (Hash.absorbE state block) env witnessLocal absorbedAffine headHolds
      have tailHolds : ∀ current ∈
          (compileBlocks phase (rowStart + 592) (witnessStart + 592)
            (permutationOutput witnessStart) blocks).invocations,
          PermutationInvocationHolds (PilotData.circuitPackage ()) current env := by
        intro current member
        exact holds current (by simp [compileBlocks, member])
      have outputAffine : Poseidon2.StateAffine
          (permutationOutput witnessStart) := by
        intro lane
        simp [permutationOutput, Permutation.freshState]
      have tailSound := inductionHypothesis (rowStart + 592)
        (witnessStart + 592) (permutationOutput witnessStart) (by omega)
        outputAffine restAffine tailHolds
      change Layer.evalState (Spartan.pullback env)
          (compileBlocks phase (rowStart + 592) (witnessStart + 592)
            (permutationOutput witnessStart) blocks).state = _
      rw [tailSound, headSound, Hash.eval_absorbE]
      rfl

def compileActions (phase : Nat) : Nat → Nat → EState →
    List Action → Trace
  | rowStart, witnessStart, state, [] =>
      ⟨rowStart, witnessStart, state, []⟩
  | rowStart, witnessStart, state, .absorb input :: actions =>
      let absorbed := compileBlocks phase rowStart witnessStart state
        (Hash.inputChunks input)
      let tail := compileActions phase absorbed.rowNext absorbed.witnessNext
        absorbed.state actions
      ⟨tail.rowNext, tail.witnessNext, tail.state,
        absorbed.invocations ++ tail.invocations⟩
  | rowStart, witnessStart, state, .squeezeK _expected :: actions =>
      let tail := compileActions phase (rowStart + 1184)
        (witnessStart + 1184) (permutationOutput (witnessStart + 592)) actions
      ⟨tail.rowNext, tail.witnessNext, tail.state,
        invocation phase rowStart witnessStart state ::
        invocation phase (rowStart + 592) (witnessStart + 592)
            (permutationOutput witnessStart) :: tail.invocations⟩

theorem compileBlocks_witnessNext
    (phase rowStart witnessStart : Nat) (state : EState)
    (blocks : List (List Expr)) :
    (compileBlocks phase rowStart witnessStart state blocks).witnessNext =
      witnessStart + blocks.length * 592 := by
  induction blocks generalizing rowStart witnessStart state with
  | nil => rfl
  | cons block blocks inductionHypothesis =>
      simp only [compileBlocks]
      rw [inductionHypothesis]
      simp only [List.length_cons]
      omega

theorem compileBlocks_state_eq
    (phase rowStart witnessStart : Nat) (state : EState)
    (blocks : List (List Expr)) :
    (compileBlocks phase rowStart witnessStart state blocks).state =
      (Hash.compileAbsorptions witnessStart state blocks).output := by
  induction blocks generalizing rowStart witnessStart state with
  | nil => rfl
  | cons block blocks inductionHypothesis =>
      simp only [compileBlocks, Hash.compileAbsorptions]
      rw [inductionHypothesis]
      rw [permutationOutput_eq_compile]

theorem squeezeOutput_eq_compile (witnessStart : Nat) (state : EState) :
    permutationOutput (witnessStart + 592) =
      (Squeeze.compile witnessStart state).output := by
  funext lane
  rw [Squeeze.compile_output_apply]
  unfold Squeeze.secondPermutation
  rw [Squeeze.first_recipes_length]
  exact congrFun
    (permutationOutput_eq_compile (witnessStart + 592)
      (Squeeze.firstPermutation witnessStart state).output) lane

/-- Two held compact invocations implement one exact quadratic-extension
squeeze, including its sampled value and outgoing state. -/
theorem squeeze_sound (phase rowStart witnessStart : Nat)
    (state : EState) (expected : KExpr) (env : Env)
    (witnessLocal : Spartan.piCcsPhaseOffset ≤ witnessStart)
    (stateAffine : Poseidon2.StateAffine state)
    (expectedEq : expected = (Squeeze.compile witnessStart state).sample)
    (firstHolds : PermutationInvocationHolds (PilotData.circuitPackage ())
      (invocation phase rowStart witnessStart state) env)
    (secondHolds : PermutationInvocationHolds (PilotData.circuitPackage ())
      (invocation phase (rowStart + 592) (witnessStart + 592)
        (permutationOutput witnessStart)) env) :
    expected.eval (Spartan.pullback env) =
        Squeeze.referenceSample
          (List.ofFn (Layer.evalState (Spartan.pullback env) state)) ∧
      List.ofFn (Layer.evalState (Spartan.pullback env)
        (permutationOutput (witnessStart + 592))) =
        Squeeze.referenceState
          (List.ofFn (Layer.evalState (Spartan.pullback env) state)) := by
  have firstSound := invocation_sound phase rowStart witnessStart state env
    witnessLocal stateAffine firstHolds
  have firstAffine : Poseidon2.StateAffine
      (permutationOutput witnessStart) := by
    intro lane
    simp [permutationOutput, Permutation.freshState]
  have secondSound := invocation_sound phase (rowStart + 592)
    (witnessStart + 592) (permutationOutput witnessStart) env (by omega)
    firstAffine secondHolds
  have firstList :
      List.ofFn (Layer.evalState (Spartan.pullback env)
        (permutationOutput witnessStart)) =
        Spec.Poseidon2.permute
          (List.ofFn (Layer.evalState (Spartan.pullback env) state)) := by
    calc
      _ = List.ofFn (Permutation.runF Permutation.schedule
          (Layer.evalState (Spartan.pullback env) state)) :=
        congrArg List.ofFn firstSound
      _ = _ := by
        rw [Permutation.runF_eq_reference,
          Permutation.runReference_schedule]
  have secondList :
      List.ofFn (Layer.evalState (Spartan.pullback env)
        (permutationOutput (witnessStart + 592))) =
        Spec.Poseidon2.permute
          (List.ofFn (Layer.evalState (Spartan.pullback env)
            (permutationOutput witnessStart))) := by
    calc
      _ = List.ofFn (Permutation.runF Permutation.schedule
          (Layer.evalState (Spartan.pullback env)
            (permutationOutput witnessStart))) :=
        congrArg List.ofFn secondSound
      _ = _ := by
        rw [Permutation.runF_eq_reference,
          Permutation.runReference_schedule]
  constructor
  · rw [expectedEq, Squeeze.compile_sample_eq]
    have firstLane := congrArg (fun values : List F => values.getD 0 0)
      firstList
    unfold KExpr.eval Squeeze.referenceSample
    apply congrArg₂ K.mk
    · simp [Layer.evalState, List.ofFn_succ]
    · rw [← permutationOutput_eq_compile witnessStart state]
      simpa [Layer.evalState, List.ofFn_succ] using firstLane
  · unfold Squeeze.referenceState
    rw [secondList, firstList]

/-- Compact package tracing and the authoritative Duplex compiler finish in
the same symbolic state. The proof is structural in the action list. -/
theorem compileActions_state_eq (phase rowStart witnessStart : Nat)
    (state : EState) (actions : List Action) :
    (compileActions phase rowStart witnessStart state actions).state =
      (Formal.compile witnessStart state actions).output := by
  induction actions generalizing rowStart witnessStart state with
  | nil => rfl
  | cons action actions inductionHypothesis =>
      cases action with
      | absorb input =>
          let blocks := Hash.inputChunks input
          let traced := compileBlocks phase rowStart witnessStart state blocks
          let absorbed := Hash.compileAbsorptions witnessStart state blocks
          change
            (compileActions phase traced.rowNext traced.witnessNext
              traced.state actions).state =
              (Formal.compile (witnessStart + absorbed.recipes.length)
                absorbed.output actions).output
          have witnessNext : traced.witnessNext =
              witnessStart + absorbed.recipes.length := by
            rw [compileBlocks_witnessNext,
              Hash.compileAbsorptions_recipes_length]
          have stateNext : traced.state = absorbed.output :=
            compileBlocks_state_eq phase rowStart witnessStart state blocks
          rw [witnessNext, stateNext]
          exact inductionHypothesis _ _ _
      | squeezeK expected =>
          let squeezed := Squeeze.compile witnessStart state
          change
            (compileActions phase (rowStart + 1184) (witnessStart + 1184)
              (permutationOutput (witnessStart + 592)) actions).state =
              (Formal.compile (witnessStart + squeezed.recipes.length)
                squeezed.output actions).output
          rw [Squeeze.compile_recipes_length, squeezeOutput_eq_compile]
          exact inductionHypothesis _ _ _

theorem expectedSamples_eq_samples_of_assertionCount_zero
    (start : Nat) (state : EState) (actions : List Action)
    (none : Formal.assertionCount actions = 0) :
    Formal.expectedSamples actions = (Formal.compile start state actions).samples := by
  induction actions generalizing start state with
  | nil => rfl
  | cons action actions inductionHypothesis =>
      cases action with
      | absorb input =>
          have tailNone : Formal.assertionCount actions = 0 := by
            simpa [Formal.assertionCount, Formal.Action.assertionCount] using none
          simpa [Formal.expectedSamples, Formal.compile] using
            inductionHypothesis
              (start + (Hash.compileAbsorptions start state
                (Hash.inputChunks input)).recipes.length)
              (Hash.compileAbsorptions start state
                (Hash.inputChunks input)).output tailNone
      | squeezeK expected =>
          simp [Formal.assertionCount, Formal.Action.assertionCount] at none

/-- Held compact invocations imply the exact Duplex trace semantics when the
leaf wires every expected squeeze value to the compiler-owned sample list. -/
theorem compileActions_traceHolds (phase rowStart witnessStart : Nat)
    (state : EState) (actions : List Action) (env : Env)
    (witnessLocal : Spartan.piCcsPhaseOffset ≤ witnessStart)
    (stateAffine : Poseidon2.StateAffine state)
    (actionsAffine :
      NightstreamFPrime.Layout.Poseidon2.Duplex.ActionsAffine actions)
    (expectedSamples : Formal.expectedSamples actions =
      (Formal.compile witnessStart state actions).samples)
    (holds : ∀ current ∈
      (compileActions phase rowStart witnessStart state actions).invocations,
      PermutationInvocationHolds (PilotData.circuitPackage ()) current env) :
    Formal.TraceHolds
      (List.ofFn (Layer.evalState (Spartan.pullback env) state))
      (actions.map (Formal.Action.eval (Spartan.pullback env)))
      (List.ofFn (Layer.evalState (Spartan.pullback env)
        (compileActions phase rowStart witnessStart state actions).state)) := by
  induction actions generalizing rowStart witnessStart state with
  | nil => rfl
  | cons action actions inductionHypothesis =>
      have headAffine :
          NightstreamFPrime.Layout.Poseidon2.Duplex.ActionAffine action :=
        actionsAffine action (by simp)
      have tailAffine :
          NightstreamFPrime.Layout.Poseidon2.Duplex.ActionsAffine actions := by
        intro current member
        exact actionsAffine current (by simp [member])
      cases action with
      | absorb input =>
          let blocks := Hash.inputChunks input
          let absorbed := compileBlocks phase rowStart witnessStart state blocks
          let symbolic := Hash.compileAbsorptions witnessStart state blocks
          have blocksAffine : Poseidon2.BlocksAffine blocks :=
            Poseidon2.inputChunks_affine input headAffine
          have absorbedHolds : ∀ current ∈ absorbed.invocations,
              PermutationInvocationHolds (PilotData.circuitPackage ())
                current env := by
            intro current member
            apply holds current
            simp only [compileActions, List.mem_append]
            exact Or.inl member
          have absorbedSound := compileBlocks_sound phase rowStart
            witnessStart state blocks env witnessLocal stateAffine blocksAffine
            absorbedHolds
          have absorbedStateEq : absorbed.state = symbolic.output :=
            compileBlocks_state_eq phase rowStart witnessStart state blocks
          have absorbedAffine : Poseidon2.StateAffine absorbed.state := by
            rw [absorbedStateEq]
            exact Poseidon2.compileAbsorptions_output_affine witnessStart state
              blocks stateAffine blocksAffine
          have witnessNextEq : absorbed.witnessNext =
              witnessStart + symbolic.recipes.length := by
            rw [compileBlocks_witnessNext,
              Hash.compileAbsorptions_recipes_length]
          have tailExpected : Formal.expectedSamples actions =
              (Formal.compile absorbed.witnessNext absorbed.state actions).samples := by
            rw [witnessNextEq, absorbedStateEq]
            simpa [Formal.expectedSamples, Formal.compile, symbolic, blocks]
              using expectedSamples
          have tailHolds : ∀ current ∈
              (compileActions phase absorbed.rowNext absorbed.witnessNext
                absorbed.state actions).invocations,
              PermutationInvocationHolds (PilotData.circuitPackage ())
                current env := by
            intro current member
            apply holds current
            simp only [compileActions, List.mem_append]
            exact Or.inr member
          have tailSound := inductionHypothesis absorbed.rowNext
            absorbed.witnessNext absorbed.state (by omega) absorbedAffine
            tailAffine tailExpected tailHolds
          have absorbedReference :
              List.ofFn (Layer.evalState (Spartan.pullback env)
                absorbed.state) =
                Absorb.reference
                  (List.ofFn (Layer.evalState (Spartan.pullback env) state))
                  (Hash.evalList (Spartan.pullback env) input) := by
            calc
              _ = List.ofFn (Hash.absorbManyF
                    (Layer.evalState (Spartan.pullback env) state)
                    (blocks.map
                      (Hash.evalList (Spartan.pullback env)))) :=
                congrArg List.ofFn absorbedSound
              _ = (blocks.map
                    (Hash.evalList (Spartan.pullback env))).foldl
                    Spec.Poseidon2.absorbBlock
                    (List.ofFn
                      (Layer.evalState (Spartan.pullback env) state)) :=
                Hash.absorbManyF_eq_reference _ _
              _ = Absorb.reference
                    (List.ofFn
                      (Layer.evalState (Spartan.pullback env) state))
                    (Hash.evalList (Spartan.pullback env) input) := by
                unfold blocks Absorb.reference
                rw [Hash.inputChunks_eval]
          simp only [List.map_cons, Formal.Action.eval, Formal.TraceHolds]
          rw [← absorbedReference]
          exact tailSound
      | squeezeK expected =>
          let squeezed := Squeeze.compile witnessStart state
          have expectedParts : expected = squeezed.sample ∧
              Formal.expectedSamples actions =
                (Formal.compile (witnessStart + squeezed.recipes.length)
                  squeezed.output actions).samples := by
            simpa [Formal.expectedSamples, Formal.compile, squeezed]
              using List.cons.inj expectedSamples
          have firstHolds : PermutationInvocationHolds
              (PilotData.circuitPackage ())
              (invocation phase rowStart witnessStart state) env :=
            holds _ (by simp [compileActions])
          have secondHolds : PermutationInvocationHolds
              (PilotData.circuitPackage ())
              (invocation phase (rowStart + 592) (witnessStart + 592)
                (permutationOutput witnessStart)) env :=
            holds _ (by simp [compileActions])
          have squeezedSound := squeeze_sound phase rowStart witnessStart
            state expected env witnessLocal stateAffine expectedParts.1
            firstHolds secondHolds
          have outputAffine : Poseidon2.StateAffine
              (permutationOutput (witnessStart + 592)) := by
            intro lane
            simp [permutationOutput, Permutation.freshState]
          have outputEq : permutationOutput (witnessStart + 592) =
              squeezed.output := squeezeOutput_eq_compile witnessStart state
          have tailExpected : Formal.expectedSamples actions =
              (Formal.compile (witnessStart + 1184)
                (permutationOutput (witnessStart + 592)) actions).samples := by
            rw [outputEq, ← Squeeze.compile_recipes_length witnessStart state]
            exact expectedParts.2
          have tailHolds : ∀ current ∈
              (compileActions phase (rowStart + 1184)
                (witnessStart + 1184)
                (permutationOutput (witnessStart + 592)) actions).invocations,
              PermutationInvocationHolds (PilotData.circuitPackage ())
                current env := by
            intro current member
            exact holds current (by simp [compileActions, member])
          have tailSound := inductionHypothesis (rowStart + 1184)
            (witnessStart + 1184)
            (permutationOutput (witnessStart + 592)) (by omega) outputAffine
            tailAffine tailExpected tailHolds
          simp only [List.map_cons, Formal.Action.eval, Formal.TraceHolds]
          refine ⟨squeezedSound.1, ?_⟩
          rw [← squeezedSound.2]
          exact tailSound

/-- Squeeze expectations own assertion rows only. Equal action shapes produce
the same compact permutation trace. -/
theorem compileActions_eq_of_shapes (phase rowStart witnessStart : Nat)
    (state : EState) (left right : List Action)
    (same : left.map Formal.Action.shape =
      right.map Formal.Action.shape) :
    compileActions phase rowStart witnessStart state left =
      compileActions phase rowStart witnessStart state right := by
  induction left generalizing right rowStart witnessStart state with
  | nil =>
      cases right with
      | nil => rfl
      | cons action actions => simp at same
  | cons leftAction leftActions inductionHypothesis =>
      cases right with
      | nil => simp at same
      | cons rightAction rightActions =>
          simp only [List.map_cons, List.cons.injEq] at same
          rcases same with ⟨headSame, tailSame⟩
          cases leftAction <;> cases rightAction
          case absorb.absorb leftInput rightInput =>
            simp only [Formal.Action.shape,
              Formal.ActionShape.absorb.injEq] at headSame
            subst rightInput
            simp only [compileActions]
            rw [inductionHypothesis _ _ _ _ tailSame]
          case absorb.squeezeK => simp [Formal.Action.shape] at headSame
          case squeezeK.absorb => simp [Formal.Action.shape] at headSame
          case squeezeK.squeezeK leftExpected rightExpected =>
            simp only [compileActions]
            rw [inductionHypothesis _ _ _ _ tailSame]

def Action.invocationCount : Action → Nat
  | .absorb input => (Hash.inputChunks input).length
  | .squeezeK _ => 2

def invocationCount (actions : List Action) : Nat :=
  (actions.map Action.invocationCount).sum

/-- Exact affine premise read by the compact invocation compiler. Squeeze
expectations are assertion data and do not occur in permutation inputs. -/
def actionShapeInputsAffine : Formal.ActionShape → Prop
  | .absorb input => Poseidon2.ListAffine input
  | .squeezeK => True

def ActionsInvocationInputsAffine (actions : List Action) : Prop :=
  ∀ shape ∈ actions.map Formal.Action.shape,
    actionShapeInputsAffine shape

/-- Exact source-bound premise read by the compact invocation compiler. -/
def actionShapeInputsBelow
    (bound : Nat) : Formal.ActionShape → Prop
  | .absorb input => ∀ expression ∈ input, expression.VarsBelow bound
  | .squeezeK => True

def ActionsInvocationInputsBelow (bound : Nat)
    (actions : List Action) : Prop :=
  ∀ shape ∈ actions.map Formal.Action.shape,
    actionShapeInputsBelow bound shape

theorem actionsInvocationInputsAffine_of_actionsAffine
    (actions : List Action)
    (strong : NightstreamFPrime.Layout.Poseidon2.Duplex.ActionsAffine
      actions) :
    ActionsInvocationInputsAffine actions := by
  intro shape member
  rw [List.mem_map] at member
  rcases member with ⟨action, actionMember, rfl⟩
  cases action with
  | absorb input => exact strong (.absorb input) actionMember
  | squeezeK expected => trivial

theorem actionsInvocationInputsBelow_of_actionsBelow
    (bound : Nat) (actions : List Action)
    (strong : Formal.ActionsBelow bound actions) :
    ActionsInvocationInputsBelow bound actions := by
  intro shape member
  rw [List.mem_map] at member
  rcases member with ⟨action, actionMember, rfl⟩
  cases action with
  | absorb input => exact strong (.absorb input) actionMember
  | squeezeK expected => trivial

theorem actionsInvocationInputsAffine_of_shapes
    (left right : List Action)
    (same : left.map Formal.Action.shape =
      right.map Formal.Action.shape)
    (leftAffine : ActionsInvocationInputsAffine left) :
    ActionsInvocationInputsAffine right := by
  intro shape member
  apply leftAffine shape
  rw [same]
  exact member

theorem actionsInvocationInputsBelow_of_shapes
    (bound : Nat) (left right : List Action)
    (same : left.map Formal.Action.shape =
      right.map Formal.Action.shape)
    (leftBelow : ActionsInvocationInputsBelow bound left) :
    ActionsInvocationInputsBelow bound right := by
  intro shape member
  apply leftBelow shape
  rw [same]
  exact member

theorem compileBlocks_invocations_length
    (phase rowStart witnessStart : Nat) (state : EState)
    (blocks : List (List Expr)) :
    (compileBlocks phase rowStart witnessStart state blocks).invocations.length =
      blocks.length := by
  induction blocks generalizing rowStart witnessStart state with
  | nil => rfl
  | cons block blocks inductionHypothesis =>
      simp [compileBlocks, inductionHypothesis]

theorem compileActions_invocations_length
    (phase rowStart witnessStart : Nat) (state : EState)
    (actions : List Action) :
    (compileActions phase rowStart witnessStart state actions).invocations.length =
      invocationCount actions := by
  induction actions generalizing rowStart witnessStart state with
  | nil => rfl
  | cons action actions inductionHypothesis =>
      cases action with
      | absorb input =>
          simp [compileActions, invocationCount,
            Action.invocationCount, compileBlocks_invocations_length,
            inductionHypothesis]
      | squeezeK expected =>
          simp [compileActions, invocationCount, Action.invocationCount,
            inductionHypothesis]
          omega

theorem invocationCount_eq_of_shapes (left right : List Action)
    (same : left.map Formal.Action.shape =
      right.map Formal.Action.shape) :
    invocationCount left = invocationCount right := by
  have traces := compileActions_eq_of_shapes 0 0 0 Hash.zeroE left right same
  have lengths := congrArg (fun trace : Trace => trace.invocations.length) traces
  simpa only [compileActions_invocations_length] using lengths

theorem recipeCount_eq_invocationCount_mul (actions : List Action) :
    Formal.recipeCount actions = invocationCount actions * 592 := by
  induction actions with
  | nil => rfl
  | cons action actions inductionHypothesis =>
      change Formal.Action.recipeCount action + Formal.recipeCount actions =
        (Action.invocationCount action + invocationCount actions) * 592
      rw [inductionHypothesis, Nat.add_mul]
      cases action with
      | absorb input =>
          simp only [Formal.Action.recipeCount, Action.invocationCount]
      | squeezeK expected =>
          norm_num [Formal.Action.recipeCount, Action.invocationCount]

theorem compileActions_witnessNext
    (phase rowStart witnessStart : Nat) (state : EState)
    (actions : List Action) :
    (compileActions phase rowStart witnessStart state actions).witnessNext =
      witnessStart + invocationCount actions * 592 := by
  induction actions generalizing rowStart witnessStart state with
  | nil => rfl
  | cons action actions inductionHypothesis =>
      cases action with
      | absorb input =>
          simp only [compileActions]
          rw [inductionHypothesis, compileBlocks_witnessNext]
          simp only [invocationCount, Action.invocationCount, List.map_cons,
            List.sum_cons]
          omega
      | squeezeK expected =>
          simp only [compileActions]
          rw [inductionHypothesis]
          simp only [invocationCount, Action.invocationCount, List.map_cons,
            List.sum_cons]
          omega

theorem compileBlocks_invocation_inputs
    (phase rowStart witnessStart : Nat) (state : EState)
    (blocks : List (List Expr))
    (current : PermutationInvocation)
    (member : current ∈
      (compileBlocks phase rowStart witnessStart state blocks).invocations) :
    current.inputs.length = 8 := by
  induction blocks generalizing rowStart witnessStart state with
  | nil => simp [compileBlocks] at member
  | cons block blocks inductionHypothesis =>
      simp only [compileBlocks, List.mem_cons] at member
      rcases member with rfl | member
      · exact invocationInputs_length _
      · exact inductionHypothesis _ _ _ member

theorem compileActions_invocation_inputs
    (phase rowStart witnessStart : Nat) (state : EState)
    (actions : List Action)
    (current : PermutationInvocation)
    (member : current ∈
      (compileActions phase rowStart witnessStart state actions).invocations) :
    current.inputs.length = 8 := by
  induction actions generalizing rowStart witnessStart state with
  | nil => simp [compileActions] at member
  | cons action actions inductionHypothesis =>
      cases action with
      | absorb input =>
          simp only [compileActions, List.mem_append] at member
          rcases member with member | member
          · exact compileBlocks_invocation_inputs _ _ _ _ _ _ member
          · exact inductionHypothesis _ _ _ member
      | squeezeK expected =>
          simp only [compileActions, List.mem_cons] at member
          rcases member with rfl | rfl | member
          · exact invocationInputs_length _
          · exact invocationInputs_length _
          · exact inductionHypothesis _ _ _ member

/-! ## Constructive package invocation execution -/

/-- One ordered invocation list inside `[bound, ceiling)`. The stable suffix
permits relocated Spartan public inputs after every private witness interval. -/
def ScheduleWithin : Nat → Nat → List PermutationInvocation → Prop
  | _, _, [] => True
  | bound, ceiling, invocation :: rest =>
      bound ≤ invocation.witnessStart ∧
        invocation.witnessStart + 592 ≤ ceiling ∧
          InvocationInputsOutside ceiling invocation ∧
            InvocationInputsOutside Spartan.privateColumnCount invocation ∧
              ScheduleWithin (invocation.witnessStart + 592) ceiling rest

theorem ScheduleWithin.cons
    {bound ceiling : Nat} {invocation : PermutationInvocation}
    {rest : List PermutationInvocation}
    (startsAfter : bound ≤ invocation.witnessStart)
    (endsBefore : invocation.witnessStart + 592 ≤ ceiling)
    (inputs : InvocationInputsOutside ceiling invocation)
    (stableInputs : InvocationInputsOutside Spartan.privateColumnCount
      invocation)
    (restSchedule : ScheduleWithin (invocation.witnessStart + 592)
      ceiling rest) :
    ScheduleWithin bound ceiling (invocation :: rest) :=
  ⟨startsAfter, endsBefore, inputs, stableInputs, restSchedule⟩

def InvocationsBefore (bound : Nat)
    (invocations : List PermutationInvocation) : Prop :=
  ∀ invocation ∈ invocations, invocation.witnessStart + 592 ≤ bound

theorem InvocationsBefore.mono
    {lower upper : Nat} {invocations : List PermutationInvocation}
    (before : InvocationsBefore lower invocations) (le : lower ≤ upper) :
    InvocationsBefore upper invocations := by
  intro invocation member
  exact Nat.le_trans (before invocation member) le

theorem InvocationsBefore.append
    {bound : Nat} {left right : List PermutationInvocation}
    (leftBefore : InvocationsBefore bound left)
    (rightBefore : InvocationsBefore bound right) :
    InvocationsBefore bound (left ++ right) := by
  intro invocation member
  rw [List.mem_append] at member
  rcases member with member | member
  · exact leftBefore invocation member
  · exact rightBefore invocation member

theorem ScheduleWithin.start_mono
    {lower bound ceiling : Nat} {invocations : List PermutationInvocation}
    (lowerLe : lower ≤ bound)
    (schedule : ScheduleWithin bound ceiling invocations) :
    ScheduleWithin lower ceiling invocations := by
  cases invocations with
  | nil => trivial
  | cons head rest =>
      rcases schedule with
        ⟨startsAfter, endsBefore, inputs, stableInputs, restSchedule⟩
      exact ⟨Nat.le_trans lowerLe startsAfter, endsBefore, inputs, stableInputs,
        restSchedule⟩

theorem ScheduleWithin.append
    {bound middle ceiling : Nat}
    {first second : List PermutationInvocation}
    (firstSchedule : ScheduleWithin bound ceiling first)
    (firstBefore : InvocationsBefore middle first)
    (boundLeMiddle : bound ≤ middle)
    (secondSchedule : ScheduleWithin middle ceiling second) :
    ScheduleWithin bound ceiling (first ++ second) := by
  induction first generalizing bound with
  | nil =>
      exact ScheduleWithin.start_mono boundLeMiddle secondSchedule
  | cons head rest inductionHypothesis =>
      rcases firstSchedule with
        ⟨startsAfter, endsBefore, inputs, stableInputs, restSchedule⟩
      refine ⟨startsAfter, endsBefore, inputs, stableInputs, ?_⟩
      apply inductionHypothesis restSchedule
      · intro invocation member
        exact firstBefore invocation (by simp [member])
      · exact firstBefore head (by simp)

theorem compileBlocks_scheduleWithin
    (phase rowStart witnessStart ceiling : Nat) (state : EState)
    (blocks : List (List Expr))
    (witnessLocal : Spartan.piCcsPhaseOffset ≤ witnessStart)
    (ceilingPrivate : ceiling ≤ Spartan.privateColumnCount)
    (endWithin : Spartan.sourceToSpartan
      (witnessStart + blocks.length * 592) ≤ ceiling)
    (stateAffine : Poseidon2.StateAffine state)
    (stateBelow : ∀ lane, (state lane).VarsBelow witnessStart)
    (blocksAffine : Poseidon2.BlocksAffine blocks)
    (blocksBelow : Hash.BlocksBelow witnessStart blocks) :
    ScheduleWithin (Spartan.sourceToSpartan witnessStart) ceiling
        (compileBlocks phase rowStart witnessStart state blocks).invocations ∧
      InvocationsBefore (Spartan.sourceToSpartan
        (witnessStart + blocks.length * 592))
        (compileBlocks phase rowStart witnessStart state blocks).invocations := by
  induction blocks generalizing rowStart witnessStart state with
  | nil =>
      exact ⟨trivial, by simp [compileBlocks, InvocationsBefore]⟩
  | cons block blocks inductionHypothesis =>
      have blockAffine : Poseidon2.ListAffine block :=
        blocksAffine block (by simp)
      have restAffine : Poseidon2.BlocksAffine blocks := by
        intro current member
        exact blocksAffine current (by simp [member])
      have blockBelow : ∀ expression ∈ block,
          expression.VarsBelow witnessStart :=
        blocksBelow block (by simp)
      have restBelow : Hash.BlocksBelow witnessStart blocks := by
        intro current member
        exact blocksBelow current (by simp [member])
      have absorbedAffine : Poseidon2.StateAffine
          (Hash.absorbE state block) :=
        Poseidon2.absorbE_affine state block stateAffine blockAffine
      have absorbedBelow : ∀ lane,
          (Hash.absorbE state block lane).VarsBelow witnessStart :=
        Hash.absorbE_varsBelow state block stateBelow blockBelow
      have headInputs := invocation_inputsOutside phase rowStart witnessStart
        ceiling (Hash.absorbE state block) witnessLocal ceilingPrivate
        absorbedAffine absorbedBelow
      have headStableInputs := invocation_inputsOutside phase rowStart
        witnessStart Spartan.privateColumnCount (Hash.absorbE state block)
        witnessLocal (by exact le_rfl) absorbedAffine absorbedBelow
      have outputAffine : Poseidon2.StateAffine
          (permutationOutput witnessStart) := by
        intro lane
        simp [permutationOutput, Permutation.freshState]
      have outputBelow : ∀ lane,
          (permutationOutput witnessStart lane).VarsBelow
            (witnessStart + 592) := by
        intro lane
        simpa [permutationOutput, Nat.add_assoc] using
          Permutation.freshState_varsBelow (witnessStart + 584) lane
      have sourceEndEq : (witnessStart + 592) + blocks.length * 592 =
          witnessStart + (block :: blocks).length * 592 := by
        simp
        omega
      have restEndWithin : Spartan.sourceToSpartan
          ((witnessStart + 592) + blocks.length * 592) ≤ ceiling := by
        rw [sourceEndEq]
        exact endWithin
      have widenedRestBelow : Hash.BlocksBelow (witnessStart + 592) blocks :=
        Hash.blocksBelow_mono restBelow (by omega)
      rcases inductionHypothesis (rowStart := rowStart + 592)
          (witnessStart := witnessStart + 592)
          (state := permutationOutput witnessStart) (by omega)
          restEndWithin outputAffine outputBelow restAffine widenedRestBelow with
        ⟨restSchedule, restBefore⟩
      have nextMap := Spartan.sourceToSpartan_add_of_piCcsLocal witnessStart
        592 witnessLocal
      have finalMap := Spartan.sourceToSpartan_add_of_piCcsLocal
        (witnessStart + 592) (blocks.length * 592) (by omega)
      have mappedEndEq := congrArg Spartan.sourceToSpartan sourceEndEq
      constructor
      · refine ⟨le_rfl, ?_, headInputs, headStableInputs, ?_⟩
        · change Spartan.sourceToSpartan witnessStart + 592 ≤ ceiling
          calc
            _ = Spartan.sourceToSpartan (witnessStart + 592) := nextMap.symm
            _ ≤ Spartan.sourceToSpartan
                ((witnessStart + 592) + blocks.length * 592) := by
              rw [finalMap]
              omega
            _ ≤ ceiling := restEndWithin
        · change ScheduleWithin
            (Spartan.sourceToSpartan witnessStart + 592) ceiling _
          rw [← nextMap]
          exact restSchedule
      · intro current member
        simp only [compileBlocks, List.mem_cons] at member
        rcases member with rfl | member
        · change Spartan.sourceToSpartan witnessStart + 592 ≤ _
          calc
            _ = Spartan.sourceToSpartan (witnessStart + 592) := nextMap.symm
            _ ≤ Spartan.sourceToSpartan
                ((witnessStart + 592) + blocks.length * 592) := by
              rw [finalMap]
              omega
            _ = _ := mappedEndEq
        · calc
            current.witnessStart + 592 ≤ Spartan.sourceToSpartan
                ((witnessStart + 592) + blocks.length * 592) :=
              restBefore current member
            _ = _ := mappedEndEq

theorem compileActions_scheduleWithin
    (phase rowStart witnessStart ceiling : Nat) (state : EState)
    (actions : List Action)
    (witnessLocal : Spartan.piCcsPhaseOffset ≤ witnessStart)
    (ceilingPrivate : ceiling ≤ Spartan.privateColumnCount)
    (endWithin : Spartan.sourceToSpartan
      (witnessStart + invocationCount actions * 592) ≤ ceiling)
    (stateAffine : Poseidon2.StateAffine state)
    (stateBelow : ∀ lane, (state lane).VarsBelow witnessStart)
    (actionsAffine : ActionsInvocationInputsAffine actions)
    (actionsBelow : ActionsInvocationInputsBelow witnessStart actions) :
    ScheduleWithin (Spartan.sourceToSpartan witnessStart) ceiling
        (compileActions phase rowStart witnessStart state actions).invocations ∧
      InvocationsBefore (Spartan.sourceToSpartan
        (witnessStart + invocationCount actions * 592))
        (compileActions phase rowStart witnessStart state actions).invocations := by
  induction actions generalizing rowStart witnessStart state with
  | nil =>
      exact ⟨trivial, by simp [compileActions, InvocationsBefore]⟩
  | cons action actions inductionHypothesis =>
      have headAffine := actionsAffine action.shape (by simp)
      have tailAffine : ActionsInvocationInputsAffine actions := by
        intro shape member
        exact actionsAffine shape (by simp [member])
      have headBelow := actionsBelow action.shape (by simp)
      have tailBelow : ActionsInvocationInputsBelow witnessStart actions := by
        intro shape member
        exact actionsBelow shape (by simp [member])
      cases action with
      | absorb input =>
          let blocks := Hash.inputChunks input
          let absorbed := compileBlocks phase rowStart witnessStart state blocks
          let tail := compileActions phase absorbed.rowNext
            absorbed.witnessNext absorbed.state actions
          have blocksAffine : Poseidon2.BlocksAffine blocks :=
            Poseidon2.inputChunks_affine input headAffine
          have blocksBelow : Hash.BlocksBelow witnessStart blocks :=
            Hash.inputChunks_below input witnessStart headBelow
          have absorbedWitnessNext : absorbed.witnessNext =
              witnessStart + blocks.length * 592 := by
            exact compileBlocks_witnessNext phase rowStart witnessStart state
              blocks
          have totalSourceEq : absorbed.witnessNext +
              invocationCount actions * 592 =
              witnessStart +
                invocationCount (.absorb input :: actions) * 592 := by
            rw [absorbedWitnessNext]
            simp only [invocationCount, Action.invocationCount,
              List.map_cons, List.sum_cons]
            change witnessStart + blocks.length * 592 +
                invocationCount actions * 592 = _
            dsimp [blocks]
            unfold invocationCount
            omega
          have tailEndWithin : Spartan.sourceToSpartan
              (absorbed.witnessNext + invocationCount actions * 592) ≤
                ceiling := by
            rw [totalSourceEq]
            exact endWithin
          have absorbedLocal : Spartan.piCcsPhaseOffset ≤
              absorbed.witnessNext := by
            rw [absorbedWitnessNext]
            omega
          have tailMap := Spartan.sourceToSpartan_add_of_piCcsLocal
            absorbed.witnessNext (invocationCount actions * 592)
            absorbedLocal
          have absorbedEndWithin : Spartan.sourceToSpartan
              absorbed.witnessNext ≤ ceiling := by
            rw [tailMap] at tailEndWithin
            omega
          have blockEndWithin : Spartan.sourceToSpartan
              (witnessStart + blocks.length * 592) ≤ ceiling := by
            rw [← absorbedWitnessNext]
            exact absorbedEndWithin
          have absorbedStateAffine : Poseidon2.StateAffine absorbed.state := by
            rw [compileBlocks_state_eq]
            exact Poseidon2.compileAbsorptions_output_affine witnessStart state
              blocks stateAffine blocksAffine
          have absorbedStateBelow : ∀ lane,
              (absorbed.state lane).VarsBelow absorbed.witnessNext := by
            intro lane
            rw [compileBlocks_state_eq, absorbedWitnessNext]
            have scope := Hash.compileAbsorptions_output_varsBelow
              witnessStart state blocks stateBelow blocksBelow lane
            rw [Hash.compileAbsorptions_recipes_length] at scope
            exact scope
          have widenedTailBelow : ActionsInvocationInputsBelow
              absorbed.witnessNext actions := by
            intro shape member
            have below := tailBelow shape member
            cases shape with
            | squeezeK => trivial
            | absorb input =>
                intro expression expressionMember
                exact Expr.VarsBelow.mono expression
                  (below expression expressionMember) (by
                    rw [absorbedWitnessNext]
                    omega)
          rcases compileBlocks_scheduleWithin phase rowStart witnessStart
              ceiling state blocks witnessLocal ceilingPrivate blockEndWithin
              stateAffine stateBelow blocksAffine blocksBelow with
            ⟨blockSchedule, blockBefore⟩
          rcases inductionHypothesis (rowStart := absorbed.rowNext)
              (witnessStart := absorbed.witnessNext)
              (state := absorbed.state) absorbedLocal tailEndWithin
              absorbedStateAffine absorbedStateBelow tailAffine
              widenedTailBelow with ⟨tailSchedule, tailBefore⟩
          have blockBefore' : InvocationsBefore
              (Spartan.sourceToSpartan absorbed.witnessNext)
              absorbed.invocations := by
            rw [absorbedWitnessNext]
            exact blockBefore
          have mappedStartToAbsorbed : Spartan.sourceToSpartan witnessStart ≤
              Spartan.sourceToSpartan absorbed.witnessNext := by
            rw [absorbedWitnessNext]
            have mapped := Spartan.sourceToSpartan_add_of_piCcsLocal
              witnessStart (blocks.length * 592) witnessLocal
            rw [mapped]
            omega
          have schedule := ScheduleWithin.append blockSchedule blockBefore'
            mappedStartToAbsorbed tailSchedule
          have mappedTotalEq := congrArg Spartan.sourceToSpartan totalSourceEq
          change ScheduleWithin (Spartan.sourceToSpartan witnessStart) ceiling
              (absorbed.invocations ++ tail.invocations) ∧
            InvocationsBefore (Spartan.sourceToSpartan
              (witnessStart +
                invocationCount (.absorb input :: actions) * 592))
              (absorbed.invocations ++ tail.invocations)
          constructor
          · exact schedule
          · intro current member
            rw [List.mem_append] at member
            rcases member with blockMember | tailMember
            · calc
                current.witnessStart + 592 ≤
                    Spartan.sourceToSpartan absorbed.witnessNext :=
                  blockBefore' current blockMember
                _ ≤ Spartan.sourceToSpartan
                    (absorbed.witnessNext +
                      invocationCount actions * 592) := by
                  rw [tailMap]
                  omega
                _ = _ := mappedTotalEq
            · calc
                current.witnessStart + 592 ≤ Spartan.sourceToSpartan
                    (absorbed.witnessNext +
                      invocationCount actions * 592) :=
                  tailBefore current tailMember
                _ = _ := mappedTotalEq
      | squeezeK expected =>
          let firstInvocation := invocation phase rowStart witnessStart state
          let secondInvocation := invocation phase (rowStart + 592)
            (witnessStart + 592) (permutationOutput witnessStart)
          let tail := compileActions phase (rowStart + 1184)
            (witnessStart + 1184)
            (permutationOutput (witnessStart + 592)) actions
          have totalSourceEq : witnessStart + 1184 +
              invocationCount actions * 592 =
              witnessStart +
                invocationCount (.squeezeK expected :: actions) * 592 := by
            simp only [invocationCount, Action.invocationCount,
              List.map_cons, List.sum_cons]
            omega
          have tailEndWithin : Spartan.sourceToSpartan
              (witnessStart + 1184 + invocationCount actions * 592) ≤
                ceiling := by
            rw [totalSourceEq]
            exact endWithin
          have tailStateAffine := permutationOutput_affine
            (witnessStart + 592)
          have tailStateBelow := permutationOutput_varsBelow
            (witnessStart + 592)
          have widenedTailBelow : ActionsInvocationInputsBelow
              (witnessStart + 1184) actions := by
            intro shape member
            have below := tailBelow shape member
            cases shape with
            | squeezeK => trivial
            | absorb input =>
                intro expression expressionMember
                exact Expr.VarsBelow.mono expression
                  (below expression expressionMember) (by omega)
          rcases inductionHypothesis (rowStart := rowStart + 1184)
              (witnessStart := witnessStart + 1184)
              (state := permutationOutput (witnessStart + 592)) (by omega)
              tailEndWithin tailStateAffine (by simpa [Nat.add_assoc] using
                tailStateBelow) tailAffine widenedTailBelow with
            ⟨tailSchedule, tailBefore⟩
          have firstInputs : InvocationInputsOutside ceiling firstInvocation := by
            dsimp [firstInvocation]
            exact invocation_inputsOutside phase rowStart witnessStart ceiling
              state witnessLocal ceilingPrivate stateAffine stateBelow
          have firstStableInputs : InvocationInputsOutside
              Spartan.privateColumnCount firstInvocation := by
            dsimp [firstInvocation]
            exact invocation_inputsOutside phase rowStart witnessStart
              Spartan.privateColumnCount state witnessLocal (by exact le_rfl)
              stateAffine stateBelow
          have secondInputs :
              InvocationInputsOutside ceiling secondInvocation := by
            dsimp [secondInvocation]
            exact invocation_inputsOutside phase (rowStart + 592)
              (witnessStart + 592) ceiling (permutationOutput witnessStart)
              (by omega) ceilingPrivate (permutationOutput_affine witnessStart)
              (permutationOutput_varsBelow witnessStart)
          have secondStableInputs : InvocationInputsOutside
              Spartan.privateColumnCount secondInvocation := by
            dsimp [secondInvocation]
            exact invocation_inputsOutside phase (rowStart + 592)
              (witnessStart + 592) Spartan.privateColumnCount
              (permutationOutput witnessStart) (by omega) (by exact le_rfl)
              (permutationOutput_affine witnessStart)
              (permutationOutput_varsBelow witnessStart)
          have mapFirst := Spartan.sourceToSpartan_add_of_piCcsLocal
            witnessStart 592 witnessLocal
          have mapSecond := Spartan.sourceToSpartan_add_of_piCcsLocal
            (witnessStart + 592) 592 (by omega)
          have mapTail := Spartan.sourceToSpartan_add_of_piCcsLocal
            (witnessStart + 1184) (invocationCount actions * 592) (by omega)
          have tailStartEq : (witnessStart + 592) + 592 =
              witnessStart + 1184 := by omega
          have mappedTotalEq := congrArg Spartan.sourceToSpartan totalSourceEq
          have firstEndWithin : Spartan.sourceToSpartan witnessStart + 592 ≤
              ceiling := by
            calc
              _ = Spartan.sourceToSpartan (witnessStart + 592) :=
                mapFirst.symm
              _ ≤ Spartan.sourceToSpartan (witnessStart + 1184) :=
                Spartan.sourceToSpartan_lt_of_piCcsLocal
                  (witnessStart + 592) (witnessStart + 1184) (by omega)
                  (by omega) |>.le
              _ ≤ Spartan.sourceToSpartan
                  (witnessStart + 1184 +
                    invocationCount actions * 592) := by
                rw [mapTail]
                omega
              _ ≤ ceiling := tailEndWithin
          have secondEndWithin :
              Spartan.sourceToSpartan (witnessStart + 592) + 592 ≤
                ceiling := by
            calc
              _ = Spartan.sourceToSpartan ((witnessStart + 592) + 592) :=
                mapSecond.symm
              _ = Spartan.sourceToSpartan (witnessStart + 1184) := by
                rw [tailStartEq]
              _ ≤ Spartan.sourceToSpartan
                  (witnessStart + 1184 +
                    invocationCount actions * 592) := by
                rw [mapTail]
                omega
              _ ≤ ceiling := tailEndWithin
          have firstStarts : Spartan.sourceToSpartan witnessStart ≤
              firstInvocation.witnessStart := by
            simp only [firstInvocation, invocation_witnessStart]
            exact le_rfl
          have firstEnds : firstInvocation.witnessStart + 592 ≤ ceiling := by
            simpa only [firstInvocation, invocation_witnessStart] using
              firstEndWithin
          have secondEnds : secondInvocation.witnessStart + 592 ≤
              ceiling := by
            simpa only [secondInvocation, invocation_witnessStart] using
              secondEndWithin
          have secondStarts :
              firstInvocation.witnessStart + 592 ≤
                secondInvocation.witnessStart := by
            simpa only [firstInvocation, secondInvocation,
              invocation_witnessStart] using
              Nat.le_of_eq mapFirst.symm
          have tailStarts :
              secondInvocation.witnessStart + 592 =
                Spartan.sourceToSpartan (witnessStart + 1184) := by
            simpa only [secondInvocation, invocation_witnessStart] using
              (calc
                Spartan.sourceToSpartan (witnessStart + 592) + 592 =
                    Spartan.sourceToSpartan ((witnessStart + 592) + 592) :=
                  mapSecond.symm
                _ = Spartan.sourceToSpartan (witnessStart + 1184) := by
                  rw [tailStartEq])
          have firstBeforeFinal :
              firstInvocation.witnessStart + 592 ≤
                Spartan.sourceToSpartan
                  (witnessStart +
                    invocationCount (.squeezeK expected :: actions) * 592) := by
            simpa only [firstInvocation, invocation_witnessStart] using
              (calc
                Spartan.sourceToSpartan witnessStart + 592 =
                    Spartan.sourceToSpartan (witnessStart + 592) :=
                  mapFirst.symm
                _ ≤ Spartan.sourceToSpartan
                    (witnessStart + 1184 +
                      invocationCount actions * 592) := by
                  rw [mapTail]
                  have localMap :=
                    Spartan.sourceToSpartan_add_of_piCcsLocal
                      (witnessStart + 592) 592 (by omega)
                  rw [localMap]
                  omega
                _ = _ := mappedTotalEq)
          have secondBeforeFinal :
              secondInvocation.witnessStart + 592 ≤
                Spartan.sourceToSpartan
                  (witnessStart +
                    invocationCount (.squeezeK expected :: actions) * 592) := by
            simpa only [secondInvocation, invocation_witnessStart] using
              (calc
                Spartan.sourceToSpartan (witnessStart + 592) + 592 ≤
                    Spartan.sourceToSpartan
                      (witnessStart + 1184 +
                        invocationCount actions * 592) := by
                  rw [← mapSecond, tailStartEq, mapTail]
                  omega
                _ = _ := mappedTotalEq)
          have secondSchedule : ScheduleWithin
              (firstInvocation.witnessStart + 592) ceiling
              (secondInvocation :: tail.invocations) := by
            have tailScheduleAt : ScheduleWithin
                (secondInvocation.witnessStart + 592)
                ceiling tail.invocations := by
              rw [tailStarts]
              exact tailSchedule
            exact ScheduleWithin.cons
              (invocation := secondInvocation)
              (rest := tail.invocations) secondStarts secondEnds
              secondInputs secondStableInputs tailScheduleAt
          change ScheduleWithin (Spartan.sourceToSpartan witnessStart) ceiling
              (firstInvocation :: secondInvocation :: tail.invocations) ∧
            InvocationsBefore (Spartan.sourceToSpartan
              (witnessStart +
                invocationCount (.squeezeK expected :: actions) * 592))
              (firstInvocation :: secondInvocation :: tail.invocations)
          constructor
          · exact ScheduleWithin.cons (invocation := firstInvocation)
              (rest := secondInvocation :: tail.invocations) firstStarts
              firstEnds firstInputs firstStableInputs secondSchedule
          · intro current member
            rcases List.mem_cons.mp member with firstMember | member
            · rw [firstMember]
              exact firstBeforeFinal
            · rcases List.mem_cons.mp member with secondMember | member
              · rw [secondMember]
                exact secondBeforeFinal
              · calc
                  current.witnessStart + 592 ≤ Spartan.sourceToSpartan
                      (witnessStart + 1184 +
                        invocationCount actions * 592) :=
                    tailBefore current member
                  _ = _ := mappedTotalEq

/-- Two environments agree at every column not owned by an invocation in the
given list. -/
def AgreesOutsideInvocations (before after : Env)
    (invocations : List PermutationInvocation) : Prop :=
  ∀ index,
    (∀ invocation ∈ invocations,
      index < invocation.witnessStart ∨
        invocation.witnessStart + 592 ≤ index) →
      after index = before index

/-- Honest execution of an ordered invocation schedule has one completed
environment that preserves both sides of the full private schedule interval
and satisfies every emitted template invocation. The proof follows the list,
not its production length. -/
theorem completeInvocations
    (env : Env) (bound ceiling : Nat)
    (invocations : List PermutationInvocation)
    (schedule : ScheduleWithin bound ceiling invocations) :
    ∃ completed,
      AgreesOutside env completed bound (ceiling - bound) ∧
        AgreesOutsideInvocations env completed invocations ∧
          ∀ invocation ∈ invocations,
            PermutationInvocationHolds (PilotData.circuitPackage ()) invocation
              completed := by
  induction invocations generalizing env bound with
  | nil =>
      exact ⟨env, fun _ _ => rfl, fun _ _ => rfl, by simp⟩
  | cons head rest inductionHypothesis =>
      rcases schedule with
        ⟨startsAfter, headEndsBefore, inputsOutside, _stableInputs,
          restSchedule⟩
      have headComplete :=
        NightstreamFPrime.Export.Pilot.completePermutationInvocation
          head env (by
            intro lane term member
            rcases inputsOutside lane term member with before | after
            · exact Or.inl before
            · exact Or.inr (by omega))
      let headEnv :=
        NightstreamFPrime.Export.Pilot.completePermutationInvocationEnv
          head env
      rcases inductionHypothesis headEnv (head.witnessStart + 592)
        restSchedule with
          ⟨completed, restAgrees, restExact, restHolds⟩
      have restSpan : head.witnessStart + 592 +
          (ceiling - (head.witnessStart + 592)) = ceiling := by
        omega
      have headHolds : PermutationInvocationHolds
          (PilotData.circuitPackage ()) head completed := by
        apply
          NightstreamFPrime.Export.Pilot.permutationInvocationHolds_of_agreesOutside
            head headEnv completed (head.witnessStart + 592)
              (ceiling - (head.witnessStart + 592))
        · intro lane term member
          rcases inputsOutside lane term member with before | after
          · exact Or.inl (by omega)
          · exact Or.inr (by omega)
        · intro index below
          exact Or.inl (by omega)
        · exact restAgrees
        · exact headComplete.2
      have totalSpan : bound + (ceiling - bound) = ceiling := by
        omega
      refine ⟨completed, ?_, ?_, ?_⟩
      · intro index outside
        have location : index < bound ∨ ceiling ≤ index := by
          rcases outside with before | after
          · exact Or.inl before
          · exact Or.inr (by omega)
        have outsideRest : index < head.witnessStart + 592 ∨
            head.witnessStart + 592 +
                (ceiling - (head.witnessStart + 592)) ≤ index := by
          rcases location with before | after
          · exact Or.inl (by omega)
          · exact Or.inr (by omega)
        have outsideHead : index < head.witnessStart ∨
            head.witnessStart + 592 ≤ index := by
          rcases location with before | after
          · exact Or.inl (by omega)
          · exact Or.inr (by omega)
        calc
          completed index = headEnv index := restAgrees index outsideRest
          _ = env index := headComplete.1 index outsideHead
      · intro index outsideAll
        have outsideHead := outsideAll head (by simp)
        have outsideRest : ∀ invocation ∈ rest,
            index < invocation.witnessStart ∨
              invocation.witnessStart + 592 ≤ index := by
          intro invocation member
          exact outsideAll invocation (by simp [member])
        calc
          completed index = headEnv index := restExact index outsideRest
          _ = env index := headComplete.1 index outsideHead
      · intro invocation member
        rcases List.mem_cons.mp member with rfl | member
        · exact headHolds
        · exact restHolds invocation member

end NightstreamFPrime.Export.Stage1.Invocations
