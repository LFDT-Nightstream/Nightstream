import NightstreamFPrime.Export.Stage1.PoseidonActionSemantics
import NightstreamFPrime.Export.Stage1.InvocationLastOutput

/-!
Owns the indexed input-value law for the existing compact Duplex compiler.
Block/action induction identifies each actual invocation and its predecessor.
The final law reads the stored preceding output and the actual affine payload.
No row satisfaction or alternative transcript representation is assumed.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.InvocationInputLaw

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open Invocations

private theorem compileBlocks_rowNext
    (phase rowStart witnessStart : Nat) (state : EState)
    (blocks : List (List Expr)) :
    (compileBlocks phase rowStart witnessStart state blocks).rowNext =
      rowStart + blocks.length * 592 := by
  induction blocks generalizing rowStart witnessStart state with
  | nil => rfl
  | cons block blocks inductionHypothesis =>
      simp only [compileBlocks]
      rw [inductionHypothesis]
      simp only [List.length_cons]
      omega

private theorem compileBlocks_last_or_initial
    (phase rowStart witnessStart : Nat) (state : EState)
    (blocks : List (List Expr)) :
    (compileBlocks phase rowStart witnessStart state blocks).state =
      if blocks.length = 0 then state
      else permutationOutput (witnessStart + (blocks.length - 1) * 592) := by
  by_cases empty : blocks = []
  · subst blocks
    rfl
  · have nonzero : blocks.length ≠ 0 := by
      intro lengthZero
      exact empty (List.length_eq_zero_iff.mp lengthZero)
    rw [if_neg nonzero]
    exact InvocationLastOutput.compileBlocks_state_last phase rowStart witnessStart state blocks empty

private theorem previous_shift
    (initial headState : EState) (witnessStart headCount index : Nat)
    (head : headState = if headCount = 0 then initial
      else permutationOutput (witnessStart + (headCount - 1) * 592)) :
    (if index = 0 then headState
      else permutationOutput (witnessStart + headCount * 592 + (index - 1) * 592)) =
    (if headCount + index = 0 then initial
      else permutationOutput (witnessStart + (headCount + index - 1) * 592)) := by
  by_cases first : index = 0
  · subst index
    simpa using head
  · rw [if_neg first, if_neg (by omega)]
    apply congrArg permutationOutput
    omega

private theorem compileBlocks_at
    (phase rowStart witnessStart : Nat) (state : EState)
    (blocks : List (List Expr)) (index : Nat) :
    (compileBlocks phase rowStart witnessStart state blocks).invocations[index]? =
      (blocks[index]?).map (fun block =>
        invocation phase (rowStart + index * 592) (witnessStart + index * 592)
          (Hash.absorbE
            (if index = 0 then state
              else permutationOutput (witnessStart + (index - 1) * 592)) block)) := by
  induction blocks generalizing rowStart witnessStart state index with
  | nil => simp only [compileBlocks, List.getElem?_nil, Option.map_none]
  | cons block blocks inductionHypothesis =>
      cases index with
      | zero => simp [compileBlocks]
      | succ index =>
          have rowEq : rowStart + 592 + index * 592 = rowStart + (index + 1) * 592 := by omega
          have witnessEq : witnessStart + 592 + index * 592 =
              witnessStart + (index + 1) * 592 := by omega
          have previousEq :
              (if index = 0 then permutationOutput witnessStart
                else permutationOutput (witnessStart + 592 + (index - 1) * 592)) =
              permutationOutput (witnessStart + index * 592) := by
            by_cases first : index = 0
            · subst index
              simp
            · rw [if_neg first]
              have address : witnessStart + 592 + (index - 1) * 592 =
                  witnessStart + index * 592 := by
                obtain ⟨prior, rfl⟩ := Nat.exists_eq_succ_of_ne_zero first
                simpa only [Nat.succ_sub_one, Nat.succ_mul, Nat.add_assoc] using
                  (Nat.add_right_comm witnessStart 592 (prior * 592))
              exact congrArg permutationOutput address
          change (compileBlocks phase (rowStart + 592) (witnessStart + 592)
              (permutationOutput witnessStart) blocks).invocations[index]? =
            (blocks[index]?).map (fun current =>
              invocation phase (rowStart + (index + 1) * 592)
                (witnessStart + (index + 1) * 592)
                (Hash.absorbE
                  (if index + 1 = 0 then state
                    else permutationOutput (witnessStart + (index + 1 - 1) * 592)) current))
          rw [inductionHypothesis]
          simp only [rowEq, witnessEq, previousEq, Nat.succ_ne_zero, if_false,
            Nat.add_sub_cancel]

private theorem compileActions_at
    (phase rowStart witnessStart : Nat) (state : EState)
    (actions : List Action) (index : Nat) :
    (compileActions phase rowStart witnessStart state actions).invocations[index]? =
      ((PoseidonActionSchedule.kinds actions)[index]?).map (fun kind =>
        invocation phase (rowStart + index * 592) (witnessStart + index * 592)
          (let previous := if index = 0 then state
            else permutationOutput (witnessStart + (index - 1) * 592)
           match kind with
           | .absorb block => Hash.absorbE previous block
           | .squeezeFirst _ | .squeezeSecond => previous)) := by
  induction actions generalizing rowStart witnessStart state index with
  | nil => simp [compileActions, PoseidonActionSchedule.kinds]
  | cons action actions inductionHypothesis =>
      cases action with
      | absorb input =>
          let blocks := Hash.inputChunks input
          let absorbed := compileBlocks phase rowStart witnessStart state blocks
          change (absorbed.invocations ++
              (compileActions phase absorbed.rowNext absorbed.witnessNext absorbed.state actions).invocations)[index]? =
            ((blocks.map PoseidonActionSchedule.Kind.absorb ++
              PoseidonActionSchedule.kinds actions)[index]?).map _
          by_cases inHead : index < blocks.length
          · have headLength : absorbed.invocations.length = blocks.length :=
              compileBlocks_invocations_length phase rowStart witnessStart state blocks
            rw [List.getElem?_append_left (by simpa only [headLength] using inHead),
              List.getElem?_append_left (by simpa only [List.length_map] using inHead),
              List.getElem?_map, compileBlocks_at]
            simp only [Option.map_map, Function.comp_def]
          · have headLength : absorbed.invocations.length = blocks.length :=
              compileBlocks_invocations_length phase rowStart witnessStart state blocks
            rw [List.getElem?_append_right (by simpa only [headLength] using Nat.le_of_not_gt inHead),
              List.getElem?_append_right (by simpa only [List.length_map] using Nat.le_of_not_gt inHead),
              headLength, List.length_map, inductionHypothesis]
            have rowNext : absorbed.rowNext = rowStart + blocks.length * 592 :=
              compileBlocks_rowNext phase rowStart witnessStart state blocks
            have witnessNext : absorbed.witnessNext = witnessStart + blocks.length * 592 :=
              compileBlocks_witnessNext phase rowStart witnessStart state blocks
            rw [rowNext, witnessNext]
            have indexEq : blocks.length + (index - blocks.length) = index := by omega
            have previousEq := previous_shift state absorbed.state witnessStart blocks.length
              (index - blocks.length)
              (compileBlocks_last_or_initial phase rowStart witnessStart state blocks)
            rw [indexEq] at previousEq
            have rowEq : rowStart + blocks.length * 592 + (index - blocks.length) * 592 =
                rowStart + index * 592 := by omega
            have witnessEq : witnessStart + blocks.length * 592 + (index - blocks.length) * 592 =
                witnessStart + index * 592 := by omega
            simp only [rowEq, witnessEq, previousEq]
      | squeezeK expected =>
          cases index with
          | zero => simp [compileActions, PoseidonActionSchedule.kinds,
              PoseidonActionSchedule.actionKinds]
          | succ index =>
              cases index with
              | zero => simp [compileActions, PoseidonActionSchedule.kinds,
                  PoseidonActionSchedule.actionKinds]
              | succ index =>
                  have previousEq :
                      (if index = 0 then permutationOutput (witnessStart + 592)
                        else permutationOutput (witnessStart + 1184 + (index - 1) * 592)) =
                      permutationOutput (witnessStart + (index + 1) * 592) := by
                    by_cases first : index = 0
                    · subst index
                      simp
                    · rw [if_neg first]
                      apply congrArg permutationOutput
                      omega
                  have rowEq : rowStart + 1184 + index * 592 =
                      rowStart + (index + 1 + 1) * 592 := by omega
                  have witnessEq : witnessStart + 1184 + index * 592 =
                      witnessStart + (index + 1 + 1) * 592 := by omega
                  simp only [compileActions, PoseidonActionSchedule.kinds,
                    List.flatMap_cons, PoseidonActionSchedule.actionKinds,
                    List.cons_append, List.nil_append, List.getElem?_cons_succ]
                  rw [inductionHypothesis]
                  simp only [rowEq, witnessEq, previousEq, Nat.succ_ne_zero,
                    if_false, Nat.add_sub_cancel, PoseidonActionSchedule.kinds]

private theorem compiled_invocation
    (phase rowStart witnessStart : Nat) (state : EState)
    (actions : List Action) (index : Fin (invocationCount actions)) :
    (compileActions phase rowStart witnessStart state actions).invocations.get
        (Fin.cast (compileActions_invocations_length phase rowStart witnessStart state actions).symm index) =
      invocation phase (rowStart + index.val * 592) (witnessStart + index.val * 592)
        (let previous := if index.val = 0 then state
          else permutationOutput (witnessStart + (index.val - 1) * 592)
         match PoseidonActionSchedule.kindAt actions index with
         | .absorb block => Hash.absorbE previous block
         | .squeezeFirst _ | .squeezeSecond => previous) := by
  have selected := compileActions_at phase rowStart witnessStart state actions index.val
  rw [← PoseidonActionSchedule.kindAt_materializes, List.getElem?_ofFn,
    dif_pos index.isLt, Option.map_some] at selected
  have bounded : index.val <
      (compileActions phase rowStart witnessStart state actions).invocations.length := by
    rw [compileActions_invocations_length]
    exact index.isLt
  rw [List.getElem?_eq_getElem bounded] at selected
  exact Option.some.inj selected

private theorem selectedBlock_affine
    (actions : List Action) (affine : ActionsInvocationInputsAffine actions)
    (index : Fin (invocationCount actions)) (block : List Expr)
    (found : PoseidonActionSchedule.kindAt actions index = .absorb block) :
    Poseidon2.ListAffine block := by
  have member : PoseidonActionSchedule.Kind.absorb block ∈ PoseidonActionSchedule.kinds actions := by
    rw [← PoseidonActionSchedule.kindAt_materializes]
    exact List.mem_ofFn.mpr ⟨index, found⟩
  rw [PoseidonActionSchedule.kinds, List.mem_flatMap] at member
  obtain ⟨action, actionMember, kindMember⟩ := member
  cases action with
  | absorb input =>
      simp only [PoseidonActionSchedule.actionKinds, List.mem_map] at kindMember
      obtain ⟨selected, selectedMember, same⟩ := kindMember
      have blockEq : selected = block := PoseidonActionSchedule.Kind.absorb.inj same
      subst selected
      have inputAffine : Poseidon2.ListAffine input :=
        affine (.absorb input) (List.mem_map.mpr ⟨.absorb input, actionMember, rfl⟩)
      exact Poseidon2.inputChunks_affine input inputAffine block selectedMember
  | squeezeK expected =>
      simp [PoseidonActionSchedule.actionKinds] at kindMember

private theorem invocation_input
    (phase rowStart witnessStart : Nat) (state : EState) (lane : Fin 8) :
    invocationInputCombination (invocation phase rowStart witnessStart state) lane.val =
      inputCombination (state lane) := by
  exact Lifecycle.PriorStateHash.ofFn_getD
    (fun current : Fin 8 => inputCombination (state current)) lane zeroSparseCombination

/-- Each actual indexed invocation reads the preceding stored output. An
absorb adds its actual payload lane; both squeeze permutations use that
preceding state unchanged. The only premises are the compiler's existing
affine-input contract and its local source interval, never row validity. -/
theorem compileActions_input_eval
    (phase rowStart witnessStart : Nat) (state : EState)
    (actions : List Action) (target : Env)
    (witnessLocal : Spartan.piCcsPhaseOffset ≤ witnessStart)
    (stateAffine : Poseidon2.StateAffine state)
    (actionsAffine : ActionsInvocationInputsAffine actions)
    (index : Fin (invocationCount actions)) (lane : Fin 8) :
    let trace := compileActions phase rowStart witnessStart state actions
    let selectedInvocation := fun current : Fin (invocationCount actions) => trace.invocations.get
      (Fin.cast (compileActions_invocations_length phase rowStart witnessStart state actions).symm current)
    let outputs := fun current => List.ofFn fun coordinate : Fin 8 =>
      target ((selectedInvocation current).witnessStart + 584 + coordinate.val)
    let previous := PoseidonActionSemantics.previousState
      (List.ofFn (Layer.evalState (Spartan.pullback target) state)) outputs index
    (invocationInputCombination (selectedInvocation index) lane.val).toR1CS.eval target =
      match PoseidonActionSchedule.kindAt actions index with
      | .absorb block => previous.getD lane.val 0 +
          (block.getD lane.val (0 : Expr)).eval (Spartan.pullback target)
      | .squeezeFirst _ | .squeezeSecond => previous.getD lane.val 0 := by
  intro trace selectedInvocation outputs previous
  let previousExpr : EState := if index.val = 0 then state
    else permutationOutput (witnessStart + (index.val - 1) * 592)
  have previousAffine : Poseidon2.StateAffine previousExpr := by
    unfold previousExpr
    split
    · exact stateAffine
    · exact permutationOutput_affine _
  have previousValue : previous.getD lane.val 0 =
      (previousExpr lane).eval (Spartan.pullback target) := by
    by_cases first : index.val = 0
    · simp only [previous, PoseidonActionSemantics.previousState, dif_pos first,
        previousExpr, if_pos first]
      exact Lifecycle.PriorStateHash.ofFn_getD _ lane (0 : F)
    · let priorIndex : Fin (invocationCount actions) := ⟨index.val - 1, by
        have bounded := index.isLt
        omega⟩
      have position : (selectedInvocation priorIndex).witnessStart =
          Spartan.sourceToSpartan (witnessStart + (index.val - 1) * 592) :=
        congrArg PermutationInvocation.witnessStart
          (compiled_invocation phase rowStart witnessStart state actions priorIndex)
      have startLocal : Spartan.piCcsPhaseOffset ≤ witnessStart + (index.val - 1) * 592 := by omega
      calc
        previous.getD lane.val 0 = target ((selectedInvocation priorIndex).witnessStart + 584 + lane.val) := by
          simp only [previous, PoseidonActionSemantics.previousState, dif_neg first, outputs]
          exact Lifecycle.PriorStateHash.ofFn_getD _ lane (0 : F)
        _ = target (Spartan.sourceToSpartan (witnessStart + (index.val - 1) * 592) + 584 + lane.val) :=
          congrArg (fun start => target (start + 584 + lane.val)) position
        _ = target (Spartan.sourceToSpartan (witnessStart + (index.val - 1) * 592 + 584 + lane.val)) := by
          apply congrArg target
          have mapped := Spartan.sourceToSpartan_add_of_piCcsLocal
            (witnessStart + (index.val - 1) * 592) (584 + lane.val) startLocal
          simpa only [Nat.add_assoc] using mapped.symm
        _ = (previousExpr lane).eval (Spartan.pullback target) := by
          simp only [previousExpr, if_neg first, permutationOutput,
            Permutation.freshState, Expr.eval_var, Spartan.pullback]
  dsimp only [selectedInvocation, trace]
  rw [compiled_invocation phase rowStart witnessStart state actions index, invocation_input]
  cases found : PoseidonActionSchedule.kindAt actions index with
  | absorb block =>
      have blockAffine := selectedBlock_affine actions actionsAffine index block found
      have affine := Poseidon2.absorbE_affine previousExpr block previousAffine blockAffine
      rw [inputCombination_eval (affine lane)]
      change (previousExpr lane + block.getD lane.val (0 : Expr)).eval (Spartan.pullback target) = _
      rw [Expr.eval_hadd]
      exact congrArg (fun value : F => value +
        (block.getD lane.val (0 : Expr)).eval (Spartan.pullback target)) previousValue.symm
  | squeezeFirst expected =>
      rw [inputCombination_eval (previousAffine lane)]
      exact previousValue.symm
  | squeezeSecond =>
      rw [inputCombination_eval (previousAffine lane)]
      exact previousValue.symm

open NightstreamFPrime.Circuit.Quadratic

private theorem stored_output_lane
    (phase rowStart witnessStart : Nat) (state : EState)
    (actions : List Action) (target : Env)
    (witnessLocal : Spartan.piCcsPhaseOffset ≤ witnessStart)
    (index : Fin (invocationCount actions)) (lane : Fin 8) :
    let selected := (compileActions phase rowStart witnessStart state actions).invocations.get
      (Fin.cast (compileActions_invocations_length phase rowStart witnessStart state actions).symm index)
    (List.ofFn fun coordinate : Fin 8 =>
      target (selected.witnessStart + 584 + coordinate.val)).getD lane.val 0 =
      (permutationOutput (witnessStart + index.val * 592) lane).eval (Spartan.pullback target) := by
  intro selected
  have position : selected.witnessStart = Spartan.sourceToSpartan (witnessStart + index.val * 592) :=
    congrArg PermutationInvocation.witnessStart
      (compiled_invocation phase rowStart witnessStart state actions index)
  rw [Lifecycle.PriorStateHash.ofFn_getD, position]
  change target (Spartan.sourceToSpartan (witnessStart + index.val * 592) + 584 + lane.val) =
    target (Spartan.sourceToSpartan (witnessStart + index.val * 592 + 584 + lane.val))
  apply congrArg target
  have mapped := Spartan.sourceToSpartan_add_of_piCcsLocal
    (witnessStart + index.val * 592) (584 + lane.val) (by omega)
  simpa only [Nat.add_assoc] using mapped.symm

private theorem compiled_expected_at
    (witnessStart : Nat) (state : EState) (actions : List Action) (env : Env)
    (assertions : ConstraintsHold env (Formal.compile witnessStart state actions).assertions)
    (index : Nat) (expected : KExpr)
    (found : (PoseidonActionSchedule.kinds actions)[index]? = some (.squeezeFirst expected)) :
    expected.eval env =
      ⟨((if index = 0 then state
          else permutationOutput (witnessStart + (index - 1) * 592)) 0).eval env,
        (permutationOutput (witnessStart + index * 592) 0).eval env⟩ := by
  induction actions generalizing witnessStart state index expected with
  | nil => simp [PoseidonActionSchedule.kinds] at found
  | cons action actions inductionHypothesis =>
      cases action with
      | absorb input =>
          let blocks := Hash.inputChunks input
          change ((blocks.map PoseidonActionSchedule.Kind.absorb ++
            PoseidonActionSchedule.kinds actions)[index]?) = some (.squeezeFirst expected) at found
          by_cases inHead : index < blocks.length
          · rw [List.getElem?_append_left (by simpa only [List.length_map] using inHead),
              List.getElem?_map] at found
            cases selected : blocks[index]? <;> simp [selected] at found
          · rw [List.getElem?_append_right
              (by simpa only [List.length_map] using Nat.le_of_not_gt inHead), List.length_map] at found
            let absorbed := compileBlocks 0 0 witnessStart state blocks
            have tailAssertions : ConstraintsHold env
                (Formal.compile (witnessStart + blocks.length * 592) absorbed.state actions).assertions := by
              rw [show absorbed.state = (Hash.compileAbsorptions witnessStart state blocks).output from
                compileBlocks_state_eq 0 0 witnessStart state blocks]
              simpa only [Formal.compile, Hash.compileAbsorptions_recipes_length] using assertions
            have tail := inductionHypothesis (witnessStart + blocks.length * 592) absorbed.state
              tailAssertions (index - blocks.length) expected found
            have indexEq : blocks.length + (index - blocks.length) = index := by omega
            have previousEq := previous_shift state absorbed.state witnessStart blocks.length
              (index - blocks.length)
              (compileBlocks_last_or_initial 0 0 witnessStart state blocks)
            rw [indexEq] at previousEq
            have witnessEq : witnessStart + blocks.length * 592 + (index - blocks.length) * 592 =
                witnessStart + index * 592 := by omega
            simpa only [previousEq, witnessEq] using tail
      | squeezeK headExpected =>
          let squeezed := Squeeze.compile witnessStart state
          have parts : ConstraintsHold env (KExpr.equalities squeezed.sample headExpected) ∧
              ConstraintsHold env
                (Formal.compile (witnessStart + squeezed.recipes.length) squeezed.output actions).assertions := by
            exact (Permutation.constraintsHold_append env _ _).mp assertions
          cases index with
          | zero =>
              have same : headExpected = expected := by
                simpa [PoseidonActionSchedule.kinds, PoseidonActionSchedule.actionKinds] using found
              subst expected
              have sampleEq := (KExpr.equalities_hold_iff env squeezed.sample headExpected).mp parts.1
              calc
                headExpected.eval env = squeezed.sample.eval env := sampleEq.symm
                _ = ⟨(state 0).eval env, (permutationOutput witnessStart 0).eval env⟩ := by
                  rw [Squeeze.compile_sample_eq, ← permutationOutput_eq_compile witnessStart state]
                  rfl
                _ = _ := by simp
          | succ index =>
              cases index with
              | zero =>
                  simp [PoseidonActionSchedule.kinds, PoseidonActionSchedule.actionKinds] at found
              | succ index =>
                  have tailFound : (PoseidonActionSchedule.kinds actions)[index]? =
                      some (.squeezeFirst expected) := by
                    simpa [PoseidonActionSchedule.kinds, PoseidonActionSchedule.actionKinds] using found
                  have tailAssertions : ConstraintsHold env
                      (Formal.compile (witnessStart + 1184)
                        (permutationOutput (witnessStart + 592)) actions).assertions := by
                    simpa only [squeezed, Squeeze.compile_recipes_length,
                      ← squeezeOutput_eq_compile] using parts.2
                  have tail := inductionHypothesis (witnessStart + 1184)
                    (permutationOutput (witnessStart + 592)) tailAssertions index expected tailFound
                  have previousEq :
                      (if index = 0 then permutationOutput (witnessStart + 592)
                        else permutationOutput (witnessStart + 1184 + (index - 1) * 592)) =
                      permutationOutput (witnessStart + (index + 1) * 592) := by
                    by_cases first : index = 0
                    · subst index
                      simp
                    · rw [if_neg first]
                      apply congrArg permutationOutput
                      omega
                  have witnessEq : witnessStart + 1184 + index * 592 =
                      witnessStart + (index + 1 + 1) * 592 := by omega
                  simpa only [previousEq, witnessEq, Nat.succ_ne_zero, if_false,
                    Nat.add_sub_cancel] using tail

/-- The actual compiler assertions bind a selected squeeze expectation to
lane zero before its first permutation and lane zero after that permutation.
This is assertion custody only; permutation rows establish the separate
Poseidon2 relation. No affine-input or row-validity premise is needed here. -/
theorem compileActions_expected_eval
    (phase rowStart witnessStart : Nat) (state : EState)
    (actions : List Action) (target : Env)
    (witnessLocal : Spartan.piCcsPhaseOffset ≤ witnessStart)
    (assertions : ConstraintsHold (Spartan.pullback target)
      (Formal.compile witnessStart state actions).assertions)
    (index : Fin (invocationCount actions)) (expected : KExpr)
    (found : PoseidonActionSchedule.kindAt actions index = .squeezeFirst expected) :
    let trace := compileActions phase rowStart witnessStart state actions
    let selectedInvocation := fun current : Fin (invocationCount actions) => trace.invocations.get
      (Fin.cast (compileActions_invocations_length phase rowStart witnessStart state actions).symm current)
    let outputs := fun current => List.ofFn fun coordinate : Fin 8 =>
      target ((selectedInvocation current).witnessStart + 584 + coordinate.val)
    let previous := PoseidonActionSemantics.previousState
      (List.ofFn (Layer.evalState (Spartan.pullback target) state)) outputs index
    expected.eval (Spartan.pullback target) = ⟨previous.getD 0 0, (outputs index).getD 0 0⟩ := by
  intro trace selectedInvocation outputs previous
  have selectedKind : (PoseidonActionSchedule.kinds actions)[index.val]? =
      some (.squeezeFirst expected) := by
    rw [← PoseidonActionSchedule.kindAt_materializes, List.getElem?_ofFn, dif_pos index.isLt]
    exact congrArg some found
  have expectedEq := compiled_expected_at witnessStart state actions (Spartan.pullback target)
    assertions index.val expected selectedKind
  have currentValue : (outputs index).getD 0 0 =
      (permutationOutput (witnessStart + index.val * 592) 0).eval (Spartan.pullback target) :=
    stored_output_lane phase rowStart witnessStart state actions target witnessLocal index 0
  have previousValue : previous.getD 0 0 =
      ((if index.val = 0 then state
        else permutationOutput (witnessStart + (index.val - 1) * 592)) 0).eval
          (Spartan.pullback target) := by
    by_cases first : index.val = 0
    · simp only [previous, PoseidonActionSemantics.previousState, dif_pos first, if_pos first]
      exact Lifecycle.PriorStateHash.ofFn_getD _ (0 : Fin 8) (0 : F)
    · let priorIndex : Fin (invocationCount actions) := ⟨index.val - 1, by
        have bounded := index.isLt
        omega⟩
      simp only [previous, PoseidonActionSemantics.previousState, dif_neg first, if_neg first]
      exact stored_output_lane phase rowStart witnessStart state actions target witnessLocal priorIndex 0
  exact expectedEq.trans (congrArg₂ K.mk previousValue.symm currentValue.symm)

end NightstreamFPrime.Export.Stage1.InvocationInputLaw
