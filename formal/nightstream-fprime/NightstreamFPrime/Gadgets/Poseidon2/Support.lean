import NightstreamFPrime.Circuit.StraightLineSupport
import NightstreamFPrime.Gadgets.Poseidon2.Formal

/-!
Owns arbitrary variable-support propagation for the canonical Poseidon2 hash
compiler. It does not change the hash semantics, permutation schedule, or
physical layout.
-/

namespace NightstreamFPrime.Gadgets.Poseidon2.Support

open NightstreamFPrime.Circuit

def StateSupported (state : Layer.EState) (allowed : Nat → Prop) : Prop :=
  ∀ lane, (state lane).VarsSatisfy allowed

private theorem getE_supported (state : Layer.EState) (allowed : Nat → Prop)
    (stateSupported : StateSupported state allowed) (index : Nat) :
    (Layer.getE state index).VarsSatisfy allowed := by
  unfold Layer.getE
  split
  · exact stateSupported _
  · trivial

private theorem mat4E_supported (state : Layer.EState)
    (allowed : Nat → Prop) (stateSupported : StateSupported state allowed)
    (base lane : Nat) :
    (Layer.mat4E state base lane).VarsSatisfy allowed := by
  rcases lane with _ | _ | _ | lane <;>
    simp [Layer.mat4E, Expr.VarsSatisfy,
      getE_supported state allowed stateSupported]

private theorem externalE_supported (state : Layer.EState)
    (allowed : Nat → Prop) (stateSupported : StateSupported state allowed)
    (lane : Fin 8) :
    (Layer.externalE state lane).VarsSatisfy allowed := by
  simp [Layer.externalE, Layer.blockE, Expr.VarsSatisfy,
    mat4E_supported state allowed stateSupported]

private theorem sumE_supported (state : Layer.EState)
    (allowed : Nat → Prop) (stateSupported : StateSupported state allowed) :
    (Layer.sumE state).VarsSatisfy allowed := by
  simp [Layer.sumE, Expr.VarsSatisfy,
    getE_supported state allowed stateSupported]

private theorem internalE_supported (state : Layer.EState)
    (allowed : Nat → Prop) (stateSupported : StateSupported state allowed)
    (lane : Fin 8) :
    (Layer.internalE state lane).VarsSatisfy allowed := by
  exact ⟨⟨trivial, stateSupported lane⟩,
    sumE_supported state allowed stateSupported⟩

private theorem sboxRecipes_supported (start : Nat) (value : Expr)
    (allowed : Nat → Prop) (valueSupported : value.VarsSatisfy allowed)
    (targetsSupported : ∀ index, index < 4 → allowed (start + index)) :
    ∀ recipe ∈ Permutation.sboxRecipes start value,
      recipe.VarsSatisfy allowed := by
  intro recipe member
  simp only [Permutation.sboxRecipes, List.mem_cons,
    List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl
  · exact ⟨valueSupported, valueSupported⟩
  · exact ⟨targetsSupported 0 (by decide), targetsSupported 0 (by decide)⟩
  · exact ⟨targetsSupported 1 (by decide), targetsSupported 0 (by decide)⟩
  · exact ⟨targetsSupported 2 (by decide), valueSupported⟩

private theorem sboxOutput_supported (start : Nat) (allowed : Nat → Prop)
    (targetsSupported : ∀ index, index < 4 → allowed (start + index)) :
    (Permutation.sboxOutput start).VarsSatisfy allowed := by
  exact targetsSupported 3 (by decide)

private theorem compileSboxes_supported (start : Nat) (values : List Expr)
    (allowed : Nat → Prop)
    (valuesSupported : ∀ value ∈ values, value.VarsSatisfy allowed)
    (targetsSupported : ∀ index,
      index < (Permutation.compileSboxes start values).recipes.length →
      allowed (start + index)) :
    (∀ recipe ∈ (Permutation.compileSboxes start values).recipes,
        recipe.VarsSatisfy allowed) ∧
      ∀ output ∈ (Permutation.compileSboxes start values).outputs,
        output.VarsSatisfy allowed := by
  induction values generalizing start with
  | nil => simp [Permutation.compileSboxes]
  | cons value rest inductionHypothesis =>
      have headTargets : ∀ index, index < 4 →
          allowed (start + index) := by
        intro index indexBound
        apply targetsSupported index
        simp only [Permutation.compileSboxes, List.length_append,
          Permutation.sboxRecipes_length,
          Permutation.compileSboxes_recipes_length]
        omega
      have tailTargets : ∀ index,
          index < (Permutation.compileSboxes (start + 4) rest).recipes.length →
          allowed ((start + 4) + index) := by
        intro index indexBound
        have bound : 4 + index <
            (Permutation.compileSboxes start (value :: rest)).recipes.length := by
          rw [Permutation.compileSboxes_recipes_length] at indexBound ⊢
          simp only [List.length_cons]
          omega
        have target := targetsSupported (4 + index) bound
        simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using target
      have tail := inductionHypothesis (start + 4) (by
        intro current currentMember
        exact valuesSupported current (by simp [currentMember])) tailTargets
      constructor
      · intro recipe member
        simp only [Permutation.compileSboxes, List.mem_append] at member
        rcases member with member | member
        · exact sboxRecipes_supported start value allowed
            (valuesSupported value (by simp)) headTargets recipe member
        · exact tail.1 recipe member
      · intro output member
        simp only [Permutation.compileSboxes, List.mem_cons] at member
        rcases member with rfl | member
        · exact sboxOutput_supported start allowed headTargets
        · exact tail.2 output member

private theorem fullInputs_supported (rows : List (List Nat)) (round : Nat)
    (state : Layer.EState) (allowed : Nat → Prop)
    (stateSupported : StateSupported state allowed) :
    ∀ value ∈ Permutation.fullInputs rows round state,
      value.VarsSatisfy allowed := by
  unfold Permutation.fullInputs
  rw [List.forall_mem_ofFn_iff]
  intro lane
  exact ⟨stateSupported lane, trivial⟩

private theorem fullSboxState_supported (start : Nat)
    (rows : List (List Nat)) (round : Nat) (state : Layer.EState)
    (allowed : Nat → Prop)
    (outputsSupported : ∀ output ∈
      (Permutation.compileSboxes start
        (Permutation.fullInputs rows round state)).outputs,
      output.VarsSatisfy allowed) :
    StateSupported (Permutation.fullSboxState start rows round state) allowed := by
  intro lane
  unfold Permutation.fullSboxState
  rw [List.getD_eq_get (Permutation.compileSboxes start
    (Permutation.fullInputs rows round state)).outputs 0
    ⟨lane.val, by simp [Permutation.fullInputs]⟩]
  exact outputsSupported _ (List.get_mem _ _)

private theorem partialInput_supported (round : Nat) (state : Layer.EState)
    (allowed : Nat → Prop) (stateSupported : StateSupported state allowed) :
    (Permutation.partialInput round state).VarsSatisfy allowed := by
  exact ⟨stateSupported 0, trivial⟩

private theorem partialSboxState_supported (start round : Nat)
    (state : Layer.EState) (allowed : Nat → Prop)
    (stateSupported : StateSupported state allowed)
    (outputsSupported : ∀ output ∈
      (Permutation.compileSboxes start
        [Permutation.partialInput round state]).outputs,
      output.VarsSatisfy allowed) :
    StateSupported (Permutation.partialSboxState start round state) allowed := by
  intro lane
  by_cases hzero : lane.val = 0
  · simp only [Permutation.partialSboxState, hzero, if_true]
    rw [List.getD_eq_get (Permutation.compileSboxes start
      [Permutation.partialInput round state]).outputs 0 ⟨0, by simp⟩]
    exact outputsSupported _ (List.get_mem _ _)
  · simp only [Permutation.partialSboxState, hzero, if_false]
    exact stateSupported lane

private theorem freshState_supported (start size : Nat)
    (allowed : Nat → Prop)
    (targetsSupported : ∀ index, index < size → allowed (start + index))
    (sizeBound : 8 ≤ size) :
    StateSupported (Permutation.freshState start) allowed := by
  intro lane
  exact targetsSupported lane.val (Nat.lt_of_lt_of_le lane.isLt sizeBound)

private theorem step_supported (start : Nat) (step : Permutation.Step)
    (state : Layer.EState) (allowed : Nat → Prop)
    (stateSupported : StateSupported state allowed)
    (targetsSupported : ∀ index, index < Permutation.stepSize step →
      allowed (start + index)) :
    (∀ recipe ∈ Permutation.stepRecipes start step state,
        recipe.VarsSatisfy allowed) ∧
      StateSupported (Permutation.stepOutput start step) allowed := by
  cases step with
  | initialLayer =>
      constructor
      · simp only [Permutation.stepRecipes]
        rw [List.forall_mem_ofFn_iff]
        exact externalE_supported state allowed stateSupported
      · exact freshState_supported start 8 allowed (by
          intro index indexBound
          exact targetsSupported index (by
            simpa [Permutation.stepSize] using indexBound)) (by decide)
  | initialFullRound round =>
      have inputs := fullInputs_supported
        NightstreamFPrime.Spec.Poseidon2.initialConstants round state allowed
        stateSupported
      have sboxes := compileSboxes_supported start _ allowed inputs (by
        intro index indexBound
        have indexBound' : index < 32 := by
          rw [Permutation.compileSboxes_recipes_length] at indexBound
          simpa [Permutation.fullInputs] using indexBound
        exact targetsSupported index (by
          simp only [Permutation.stepSize]
          omega))
      constructor
      · intro recipe member
        simp only [Permutation.stepRecipes, List.mem_append] at member
        rcases member with member | member
        · exact sboxes.1 recipe member
        · rw [List.mem_ofFn'] at member
          rcases member with ⟨lane, rfl⟩
          exact externalE_supported _ allowed
            (fullSboxState_supported start _ round state allowed sboxes.2) lane
      · exact freshState_supported (start + 32) 8 allowed (by
          intro index indexBound
          have supported := targetsSupported (32 + index) (by
            simp [Permutation.stepSize]
            omega)
          simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using supported)
          (by decide)
  | partialRound round =>
      have input := partialInput_supported round state allowed stateSupported
      have sboxes := compileSboxes_supported start
        [Permutation.partialInput round state] allowed (by
          intro value member
          simp only [List.mem_singleton] at member
          subst value
          exact input) (by
            intro index indexBound
            exact targetsSupported index (by
              simp only [Permutation.compileSboxes_recipes_length,
                List.length_singleton] at indexBound
              simp [Permutation.stepSize]
              omega))
      constructor
      · intro recipe member
        simp only [Permutation.stepRecipes, List.mem_append] at member
        rcases member with member | member
        · exact sboxes.1 recipe member
        · rw [List.mem_ofFn'] at member
          rcases member with ⟨lane, rfl⟩
          exact internalE_supported _ allowed
            (partialSboxState_supported start round state allowed
              stateSupported sboxes.2) lane
      · exact freshState_supported (start + 4) 8 allowed (by
          intro index indexBound
          have supported := targetsSupported (4 + index) (by
            simp [Permutation.stepSize]
            omega)
          simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using supported)
          (by decide)
  | terminalFullRound round =>
      have inputs := fullInputs_supported
        NightstreamFPrime.Spec.Poseidon2.terminalConstants round state allowed
        stateSupported
      have sboxes := compileSboxes_supported start _ allowed inputs (by
        intro index indexBound
        have indexBound' : index < 32 := by
          rw [Permutation.compileSboxes_recipes_length] at indexBound
          simpa [Permutation.fullInputs] using indexBound
        exact targetsSupported index (by
          simp only [Permutation.stepSize]
          omega))
      constructor
      · intro recipe member
        simp only [Permutation.stepRecipes, List.mem_append] at member
        rcases member with member | member
        · exact sboxes.1 recipe member
        · rw [List.mem_ofFn'] at member
          rcases member with ⟨lane, rfl⟩
          exact externalE_supported _ allowed
            (fullSboxState_supported start _ round state allowed sboxes.2) lane
      · exact freshState_supported (start + 32) 8 allowed (by
          intro index indexBound
          have supported := targetsSupported (32 + index) (by
            simp [Permutation.stepSize]
            omega)
          simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using supported)
          (by decide)

/-- Every permutation recipe and output lane uses only supported inputs or
the exact contiguous target range allocated by the compiler. -/
theorem permutationCompile_supported (start : Nat) (state : Layer.EState)
    (steps : List Permutation.Step) (allowed : Nat → Prop)
    (stateSupported : StateSupported state allowed)
    (targetsSupported : ∀ index,
      index < (Permutation.compile start state steps).recipes.length →
      allowed (start + index)) :
    (∀ recipe ∈ (Permutation.compile start state steps).recipes,
        recipe.VarsSatisfy allowed) ∧
      StateSupported (Permutation.compile start state steps).output allowed := by
  induction steps generalizing start state with
  | nil => simpa [Permutation.compile] using stateSupported
  | cons step rest inductionHypothesis =>
      have headTargets : ∀ index, index < Permutation.stepSize step →
          allowed (start + index) := by
        intro index indexBound
        apply targetsSupported index
        simp only [Permutation.compile, List.length_append,
          Permutation.stepRecipes_length,
          Permutation.compile_recipes_length]
        omega
      have head := step_supported start step state allowed stateSupported
        headTargets
      have tailTargets : ∀ index,
          index < (Permutation.compile (start + Permutation.stepSize step)
            (Permutation.stepOutput start step) rest).recipes.length →
          allowed ((start + Permutation.stepSize step) + index) := by
        intro index indexBound
        have target := targetsSupported (Permutation.stepSize step + index) (by
          simp only [Permutation.compile, List.length_append,
            Permutation.stepRecipes_length]
          omega)
        simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using target
      have tail := inductionHypothesis
        (start + Permutation.stepSize step)
        (Permutation.stepOutput start step) head.2 tailTargets
      constructor
      · intro recipe member
        simp only [Permutation.compile, List.mem_append] at member
        rcases member with member | member
        · exact head.1 recipe member
        · exact tail.1 recipe member
      · exact tail.2

def BlocksSupported (blocks : List (List Expr))
    (allowed : Nat → Prop) : Prop :=
  ∀ block ∈ blocks, ∀ expression ∈ block,
    expression.VarsSatisfy allowed

private theorem inputChunks_supported (input : List Expr)
    (allowed : Nat → Prop)
    (inputSupported : ∀ expression ∈ input,
      expression.VarsSatisfy allowed) :
    BlocksSupported (Hash.inputChunks input) allowed := by
  intro block blockMember expression expressionMember
  simp only [Hash.inputChunks, List.mem_map] at blockMember
  rcases blockMember with ⟨chunk, _, rfl⟩
  apply inputSupported expression
  exact List.mem_of_mem_drop (List.mem_of_mem_take expressionMember)

private theorem absorbE_supported (state : Layer.EState) (block : List Expr)
    (allowed : Nat → Prop) (stateSupported : StateSupported state allowed)
    (blockSupported : ∀ expression ∈ block,
      expression.VarsSatisfy allowed) :
    StateSupported (Hash.absorbE state block) allowed := by
  intro lane
  apply Expr.VarsSatisfy.add
  · exact stateSupported lane
  · by_cases within : lane.val < block.length
    · rw [List.getD_eq_get block 0 ⟨lane.val, within⟩]
      exact blockSupported _ (List.get_mem _ _)
    · simp [List.getD, within, Expr.VarsSatisfy]

private theorem padE_supported (state : Layer.EState) (allowed : Nat → Prop)
    (stateSupported : StateSupported state allowed) :
    StateSupported (Hash.padE state) allowed := by
  intro lane
  by_cases hzero : lane.val = 0
  · simp [Hash.padE, hzero, Expr.VarsSatisfy, stateSupported lane]
  · simpa [Hash.padE, hzero] using stateSupported lane

private theorem compileAbsorptions_supported (start : Nat)
    (state : Layer.EState) (blocks : List (List Expr))
    (allowed : Nat → Prop) (stateSupported : StateSupported state allowed)
    (blocksSupported : BlocksSupported blocks allowed)
    (targetsSupported : ∀ index,
      index < (Hash.compileAbsorptions start state blocks).recipes.length →
      allowed (start + index)) :
    (∀ recipe ∈ (Hash.compileAbsorptions start state blocks).recipes,
        recipe.VarsSatisfy allowed) ∧
      StateSupported (Hash.compileAbsorptions start state blocks).output
        allowed := by
  induction blocks generalizing start state with
  | nil => simpa [Hash.compileAbsorptions] using stateSupported
  | cons block rest inductionHypothesis =>
      let permutation := Permutation.compile start (Hash.absorbE state block)
        Permutation.schedule
      have permutationLength : permutation.recipes.length = 592 := by
        simpa [permutation] using Permutation.compile_schedule_recipe_count
          start (Hash.absorbE state block)
      have blockSupported : ∀ expression ∈ block,
          expression.VarsSatisfy allowed :=
        blocksSupported block (by simp)
      have restSupported : BlocksSupported rest allowed := by
        intro current member
        exact blocksSupported current (by simp [member])
      have absorbedSupported := absorbE_supported state block allowed
        stateSupported blockSupported
      have headTargets : ∀ index, index < permutation.recipes.length →
          allowed (start + index) := by
        intro index indexBound
        apply targetsSupported index
        rw [Hash.compileAbsorptions_recipes_length]
        rw [permutationLength] at indexBound
        simp only [List.length_cons]
        omega
      have head := permutationCompile_supported start
        (Hash.absorbE state block) Permutation.schedule allowed
        absorbedSupported headTargets
      have tailTargets : ∀ index,
          index < (Hash.compileAbsorptions (start + 592)
            permutation.output rest).recipes.length →
          allowed ((start + 592) + index) := by
        intro index indexBound
        have target := targetsSupported (592 + index) (by
          rw [Hash.compileAbsorptions_recipes_length] at indexBound ⊢
          simp only [List.length_cons]
          omega)
        simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using target
      have tail := inductionHypothesis (start + 592) permutation.output
        head.2 restSupported tailTargets
      constructor
      · intro recipe member
        simp only [Hash.compileAbsorptions, List.mem_append] at member
        rcases member with member | member
        · exact head.1 recipe member
        · exact tail.1 recipe member
      · exact tail.2

private theorem hashProgram_output_eq {left right : Hash.Program}
    (equality : left = right) : left.output = right.output := by
  cases equality
  rfl

/-- Every complete sponge recipe and output lane uses only supported inputs or
the exact contiguous target range allocated by the hash compiler. -/
theorem hashCompile_supported (start : Nat) (input : List Expr)
    (allowed : Nat → Prop)
    (inputSupported : ∀ expression ∈ input,
      expression.VarsSatisfy allowed)
    (targetsSupported : ∀ index,
      index < (Hash.compile start input).recipes.length →
      allowed (start + index)) :
    (∀ recipe ∈ (Hash.compile start input).recipes,
        recipe.VarsSatisfy allowed) ∧
      StateSupported (Hash.compile start input).output allowed := by
  let blocks := Hash.inputChunks input
  let absorbed := Hash.compileAbsorptions start Hash.zeroE blocks
  let finalPermutation := Permutation.compile
    (start + absorbed.recipes.length) (Hash.padE absorbed.output)
    Permutation.schedule
  have compileEq : Hash.compile start input =
      ⟨absorbed.recipes ++ finalPermutation.recipes,
        finalPermutation.output⟩ := by
    rfl
  have absorbedLength : absorbed.recipes.length =
      (Hash.inputChunks input).length * 592 := by
    simpa [absorbed, blocks] using Hash.compileAbsorptions_recipes_length
      start Hash.zeroE (Hash.inputChunks input)
  have zeroSupported : StateSupported Hash.zeroE allowed := by
    intro lane
    trivial
  have blocksSupported : BlocksSupported blocks allowed :=
    inputChunks_supported input allowed inputSupported
  have absorbedTargets : ∀ index, index < absorbed.recipes.length →
      allowed (start + index) := by
    intro index indexBound
    apply targetsSupported index
    rw [Hash.compile_recipes_length]
    rw [absorbedLength] at indexBound
    exact Nat.lt_add_right 592 indexBound
  have absorbedProof := compileAbsorptions_supported start Hash.zeroE blocks
    allowed zeroSupported blocksSupported absorbedTargets
  have finalTargets : ∀ index,
      index < finalPermutation.recipes.length →
      allowed ((start + absorbed.recipes.length) + index) := by
    intro index indexBound
    have target := targetsSupported (absorbed.recipes.length + index) (by
      rw [Hash.compile_recipes_length, absorbedLength]
      have finalLength : finalPermutation.recipes.length = 592 := by
        simpa [finalPermutation] using
          Permutation.compile_schedule_recipe_count
            (start + absorbed.recipes.length) (Hash.padE absorbed.output)
      rw [finalLength] at indexBound
      exact Nat.add_lt_add_left indexBound _)
    simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using target
  have finalProof :
      (∀ recipe ∈ finalPermutation.recipes,
          recipe.VarsSatisfy allowed) ∧
        StateSupported finalPermutation.output allowed := by
    simpa only [finalPermutation] using permutationCompile_supported
      (start + absorbed.recipes.length) (Hash.padE absorbed.output)
      Permutation.schedule allowed (padE_supported absorbed.output allowed
        absorbedProof.2) finalTargets
  constructor
  · intro recipe member
    have recipesEq : (Hash.compile start input).recipes =
        absorbed.recipes ++ finalPermutation.recipes :=
      congrArg (fun program : Hash.Program => program.recipes) compileEq
    rw [recipesEq] at member
    rw [List.mem_append] at member
    rcases member with member | member
    · exact absorbedProof.1 recipe member
    · exact finalProof.1 recipe member
  · have outputEq := hashProgram_output_eq compileEq
    intro lane
    have laneEq := congrFun outputEq lane
    exact Eq.mpr
      (congrArg (fun expression => expression.VarsSatisfy allowed) laneEq)
      (finalProof.2 lane)

/-- The proof-carrying Poseidon2 hash circuit preserves one caller-selected
support set for every flattened recipe and digest-assertion row. -/
theorem formalFlatConstraints_supported
    (interface : Formal.Interface) (offset : Nat) (allowed : Nat → Prop)
    (inputSupported : ∀ expression ∈ interface.input offset,
      expression.VarsSatisfy allowed)
    (expectedSupported : ∀ lane,
      (interface.expected offset lane).VarsSatisfy allowed)
    (localSupported : ∀ index,
      offset ≤ index →
      index < offset + localLength (Circuit.ops (Formal.main interface) offset) →
      allowed index) :
    ∀ expression ∈
        flatConstraints (Circuit.ops (Formal.main interface) offset),
      expression.VarsSatisfy allowed := by
  let program := Hash.compile offset (interface.input offset)
  have targetSupported : ∀ index, index < program.recipes.length →
      allowed (offset + index) := by
    intro index indexBound
    apply localSupported (offset + index) (by omega)
    rw [Formal.main_ops, Formal.opsAt_localLength]
    exact Nat.add_lt_add_left indexBound offset
  have compiled := hashCompile_supported offset (interface.input offset)
    allowed inputSupported targetSupported
  intro expression member
  simp only [Formal.main_ops, flatConstraints, List.mem_flatMap] at member
  rcases member with ⟨operation, operationMember, constraintMember⟩
  simp only [Formal.opsAt, List.mem_cons] at operationMember
  rcases operationMember with rfl | operationMember
  · exact recipeConstraints_varsSatisfy offset program.recipes allowed
      compiled.1 targetSupported expression constraintMember
  · rw [Formal.assertions, List.mem_ofFn'] at operationMember
    rcases operationMember with ⟨lane, rfl⟩
    simp only [Op.flatConstraints, List.mem_singleton] at constraintMember
    subst expression
    apply Expr.VarsSatisfy.sub
    · exact compiled.2 ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩
    · exact expectedSupported lane

end NightstreamFPrime.Gadgets.Poseidon2.Support
