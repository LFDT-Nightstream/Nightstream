import NightstreamFPrime.Gadgets.Poseidon2.Formal
import NightstreamFPrime.Layout.R1CS

/-!
Owns the syntactic proof that the staged Poseidon2 witness program lowers to
one physical R1CS row per recipe. The proof follows compiler structure and
does not unfold an emitted hash schedule.
-/

namespace NightstreamFPrime.Layout.Poseidon2

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2

def StateAffine (state : Layer.EState) : Prop :=
  ∀ lane, R1CS.IsAffine (state lane)

theorem affine_sum4 {a b c d : Expr}
    (ha : R1CS.IsAffine a) (hb : R1CS.IsAffine b)
    (hc : R1CS.IsAffine c) (hd : R1CS.IsAffine d) :
    R1CS.IsAffine (a + b + c + d) :=
  R1CS.IsAffine.add (R1CS.IsAffine.add (R1CS.IsAffine.add ha hb) hc) hd

theorem getE_affine (state : Layer.EState) (stateAffine : StateAffine state)
    (index : Nat) : R1CS.IsAffine (Layer.getE state index) := by
  unfold Layer.getE
  split
  · exact stateAffine _
  · exact R1CS.isAffine_const _

theorem mat4E_affine (state : Layer.EState) (stateAffine : StateAffine state)
    (base lane : Nat) : R1CS.IsAffine (Layer.mat4E state base lane) := by
  rcases lane with _ | _ | _ | lane
  · exact affine_sum4
      (R1CS.IsAffine.const_mul 2 (getE_affine state stateAffine base))
      (R1CS.IsAffine.const_mul 3 (getE_affine state stateAffine (base + 1)))
      (getE_affine state stateAffine (base + 2))
      (getE_affine state stateAffine (base + 3))
  · exact affine_sum4
      (getE_affine state stateAffine base)
      (R1CS.IsAffine.const_mul 2 (getE_affine state stateAffine (base + 1)))
      (R1CS.IsAffine.const_mul 3 (getE_affine state stateAffine (base + 2)))
      (getE_affine state stateAffine (base + 3))
  · exact affine_sum4
      (getE_affine state stateAffine base)
      (getE_affine state stateAffine (base + 1))
      (R1CS.IsAffine.const_mul 2 (getE_affine state stateAffine (base + 2)))
      (R1CS.IsAffine.const_mul 3 (getE_affine state stateAffine (base + 3)))
  · exact affine_sum4
      (R1CS.IsAffine.const_mul 3 (getE_affine state stateAffine base))
      (getE_affine state stateAffine (base + 1))
      (getE_affine state stateAffine (base + 2))
      (R1CS.IsAffine.const_mul 2 (getE_affine state stateAffine (base + 3)))

theorem blockE_affine (state : Layer.EState) (stateAffine : StateAffine state)
    (index : Nat) : R1CS.IsAffine (Layer.blockE state index) :=
  mat4E_affine state stateAffine _ _

theorem externalE_affine (state : Layer.EState)
    (stateAffine : StateAffine state) : StateAffine (Layer.externalE state) := by
  intro lane
  exact R1CS.IsAffine.add
    (R1CS.IsAffine.add
      (blockE_affine state stateAffine lane.val)
      (blockE_affine state stateAffine (lane.val % 4)))
    (blockE_affine state stateAffine (lane.val % 4 + 4))

theorem sumE_affine (state : Layer.EState) (stateAffine : StateAffine state) :
    R1CS.IsAffine (Layer.sumE state) := by
  unfold Layer.sumE
  exact R1CS.IsAffine.add
    (R1CS.IsAffine.add
      (R1CS.IsAffine.add
        (R1CS.IsAffine.add
          (R1CS.IsAffine.add
            (R1CS.IsAffine.add
              (R1CS.IsAffine.add
                (getE_affine state stateAffine 0)
                (getE_affine state stateAffine 1))
              (getE_affine state stateAffine 2))
            (getE_affine state stateAffine 3))
          (getE_affine state stateAffine 4))
        (getE_affine state stateAffine 5))
      (getE_affine state stateAffine 6))
    (getE_affine state stateAffine 7)

theorem internalE_affine (state : Layer.EState)
    (stateAffine : StateAffine state) : StateAffine (Layer.internalE state) := by
  intro lane
  exact R1CS.IsAffine.add
    (R1CS.IsAffine.const_mul _ (stateAffine lane))
    (sumE_affine state stateAffine)

theorem freshState_affine (start : Nat) :
    StateAffine (Permutation.freshState start) := by
  intro lane
  exact R1CS.isAffine_var _

theorem recipesDirect_of_all_affine (output : Nat) (recipes : List Expr)
    (allAffine : ∀ recipe ∈ recipes, R1CS.IsAffine recipe) :
    R1CS.RecipesDirect output recipes := by
  induction recipes generalizing output with
  | nil => trivial
  | cons recipe rest ih =>
      exact ⟨
        R1CS.IsDirectRecipe.of_affine output (allAffine recipe (by simp)),
        ih (output + 1) (by
          intro current member
          exact allAffine current (by simp [member]))⟩

theorem sboxRecipes_direct (start : Nat) (value : Expr)
    (valueAffine : R1CS.IsAffine value) :
    R1CS.RecipesDirect start (Permutation.sboxRecipes start value) := by
  simp only [Permutation.sboxRecipes, R1CS.RecipesDirect]
  exact ⟨
    R1CS.IsDirectRecipe.mul start valueAffine valueAffine,
    R1CS.IsDirectRecipe.mul (start + 1)
      (R1CS.isAffine_var _) (R1CS.isAffine_var _),
    R1CS.IsDirectRecipe.mul (start + 1 + 1)
      (R1CS.isAffine_var _) (R1CS.isAffine_var _),
    R1CS.IsDirectRecipe.mul (start + 1 + 1 + 1)
      (R1CS.isAffine_var _) valueAffine,
    trivial⟩

theorem compileSboxes_direct (start : Nat) (values : List Expr)
    (valuesAffine : ∀ value ∈ values, R1CS.IsAffine value) :
    R1CS.RecipesDirect start
      (Permutation.compileSboxes start values).recipes := by
  induction values generalizing start with
  | nil => trivial
  | cons value rest ih =>
      unfold Permutation.compileSboxes
      apply R1CS.recipesDirect_append
      · exact sboxRecipes_direct start value (valuesAffine value (by simp))
      · simpa using ih (start + 4) (by
          intro current member
          exact valuesAffine current (by simp [member]))

theorem compileSboxes_outputs_affine (start : Nat) (values : List Expr)
    (output : Expr)
    (member : output ∈ (Permutation.compileSboxes start values).outputs) :
    R1CS.IsAffine output := by
  induction values generalizing start with
  | nil => simp [Permutation.compileSboxes] at member
  | cons value rest ih =>
      simp only [Permutation.compileSboxes, List.mem_cons] at member
      rcases member with rfl | member
      · exact R1CS.isAffine_var _
      · exact ih (start + 4) member

theorem fullInputs_affine (rows : List (List Nat)) (round : Nat)
    (state : Layer.EState) (stateAffine : StateAffine state) :
    ∀ value ∈ Permutation.fullInputs rows round state,
      R1CS.IsAffine value := by
  unfold Permutation.fullInputs
  rw [List.forall_mem_ofFn_iff]
  intro lane
  exact R1CS.IsAffine.add (stateAffine lane) (R1CS.isAffine_const _)

theorem fullSboxState_affine (start : Nat) (rows : List (List Nat))
    (round : Nat) (state : Layer.EState) :
    StateAffine (Permutation.fullSboxState start rows round state) := by
  intro lane
  unfold Permutation.fullSboxState
  rw [List.getD_eq_get
    (Permutation.compileSboxes start
      (Permutation.fullInputs rows round state)).outputs 0
    ⟨lane.val, by simp [Permutation.fullInputs]⟩]
  exact compileSboxes_outputs_affine start _ _ (List.get_mem _ _)

theorem partialInput_affine (round : Nat) (state : Layer.EState)
    (stateAffine : StateAffine state) :
    R1CS.IsAffine (Permutation.partialInput round state) :=
  R1CS.IsAffine.add (stateAffine 0) (R1CS.isAffine_const _)

theorem partialSboxState_affine (start round : Nat) (state : Layer.EState)
    (stateAffine : StateAffine state) :
    StateAffine (Permutation.partialSboxState start round state) := by
  intro lane
  by_cases zeroLane : lane.val = 0
  · simp only [Permutation.partialSboxState, zeroLane, if_true]
    rw [List.getD_eq_get
      (Permutation.compileSboxes start
        [Permutation.partialInput round state]).outputs 0
      ⟨0, by simp⟩]
    exact compileSboxes_outputs_affine start _ _ (List.get_mem _ _)
  · simp [Permutation.partialSboxState, zeroLane, stateAffine lane]

theorem listOfFn_direct (output : Nat) (state : Layer.EState)
    (stateAffine : StateAffine state) :
    R1CS.RecipesDirect output (List.ofFn state) := by
  apply recipesDirect_of_all_affine
  intro expression member
  rw [List.mem_ofFn'] at member
  rcases member with ⟨lane, rfl⟩
  exact stateAffine lane

theorem stepRecipes_direct (start : Nat) (step : Permutation.Step)
    (state : Layer.EState) (stateAffine : StateAffine state) :
    R1CS.RecipesDirect start (Permutation.stepRecipes start step state) := by
  cases step with
  | initialLayer =>
      exact listOfFn_direct start _ (externalE_affine state stateAffine)
  | initialFullRound round =>
      apply R1CS.recipesDirect_append
      · exact compileSboxes_direct start _
          (fullInputs_affine _ round state stateAffine)
      · exact listOfFn_direct _ _
          (externalE_affine _ (fullSboxState_affine start _ round state))
  | terminalFullRound round =>
      apply R1CS.recipesDirect_append
      · exact compileSboxes_direct start _
          (fullInputs_affine _ round state stateAffine)
      · exact listOfFn_direct _ _
          (externalE_affine _ (fullSboxState_affine start _ round state))
  | partialRound round =>
      apply R1CS.recipesDirect_append
      · exact compileSboxes_direct start _ (by
          intro value member
          simp only [List.mem_singleton] at member
          subst value
          exact partialInput_affine round state stateAffine)
      · exact listOfFn_direct _ _
          (internalE_affine _
            (partialSboxState_affine start round state stateAffine))

theorem stepOutput_affine (start : Nat) (step : Permutation.Step) :
    StateAffine (Permutation.stepOutput start step) := by
  cases step <;> exact freshState_affine _

theorem compile_direct (start : Nat) (state : Layer.EState)
    (steps : List Permutation.Step) (stateAffine : StateAffine state) :
    R1CS.RecipesDirect start
      (Permutation.compile start state steps).recipes := by
  induction steps generalizing start state with
  | nil => trivial
  | cons step rest ih =>
      apply R1CS.recipesDirect_append
      · exact stepRecipes_direct start step state stateAffine
      · simpa using ih (start + Permutation.stepSize step)
          (Permutation.stepOutput start step)
          (stepOutput_affine start step)

theorem compile_output_affine (start : Nat) (state : Layer.EState)
    (steps : List Permutation.Step) (stateAffine : StateAffine state) :
    StateAffine (Permutation.compile start state steps).output := by
  induction steps generalizing start state with
  | nil => simpa [Permutation.compile] using stateAffine
  | cons step rest ih =>
      exact ih (start + Permutation.stepSize step)
        (Permutation.stepOutput start step) (stepOutput_affine start step)

theorem compile_schedule_direct (start : Nat) (state : Layer.EState)
    (stateAffine : StateAffine state) :
    R1CS.RecipesDirect start
      (Permutation.compile start state Permutation.schedule).recipes :=
  compile_direct start state Permutation.schedule stateAffine

theorem compile_schedule_output_affine (start : Nat) (state : Layer.EState)
    (stateAffine : StateAffine state) :
    StateAffine
      (Permutation.compile start state Permutation.schedule).output :=
  compile_output_affine start state Permutation.schedule stateAffine

/-- The fixed schedule's final layer owns the eight variables beginning at
`start + 584`. This reduction is fixed-size and independent of hash length. -/
theorem compile_schedule_output_eq (start : Nat) (state : Layer.EState) :
    (Permutation.compile start state Permutation.schedule).output =
      Permutation.freshState (start + 584) := by
  rfl

def ListAffine (values : List Expr) : Prop :=
  ∀ expression ∈ values, R1CS.IsAffine expression

def BlocksAffine (blocks : List (List Expr)) : Prop :=
  ∀ block ∈ blocks, ListAffine block

theorem getD_affine (values : List Expr) (allAffine : ListAffine values)
    (index : Nat) (fallback : Expr) (fallbackAffine : R1CS.IsAffine fallback) :
    R1CS.IsAffine (values.getD index fallback) := by
  by_cases inBounds : index < values.length
  · rw [List.getD_eq_get values fallback ⟨index, inBounds⟩]
    exact allAffine _ (List.get_mem values ⟨index, inBounds⟩)
  · simp [List.getD, inBounds, fallbackAffine]

theorem zeroE_affine : StateAffine Hash.zeroE := by
  intro lane
  exact R1CS.isAffine_const _

theorem absorbE_affine (state : Layer.EState) (block : List Expr)
    (stateAffine : StateAffine state) (blockAffine : ListAffine block) :
    StateAffine (Hash.absorbE state block) := by
  intro lane
  exact R1CS.IsAffine.add (stateAffine lane)
    (getD_affine block blockAffine lane.val 0 (R1CS.isAffine_const _))

theorem padE_affine (state : Layer.EState) (stateAffine : StateAffine state) :
    StateAffine (Hash.padE state) := by
  intro lane
  unfold Hash.padE
  split
  · exact R1CS.IsAffine.add (stateAffine lane) (R1CS.isAffine_const _)
  · exact stateAffine lane

theorem inputChunks_affine (input : List Expr) (inputAffine : ListAffine input) :
    BlocksAffine (Hash.inputChunks input) := by
  intro block blockMember expression expressionMember
  simp only [Hash.inputChunks, List.mem_map] at blockMember
  rcases blockMember with ⟨chunk, _, rfl⟩
  apply inputAffine expression
  exact List.mem_of_mem_drop (List.mem_of_mem_take expressionMember)

theorem compileAbsorptions_direct (start : Nat) (state : Layer.EState)
    (blocks : List (List Expr)) (stateAffine : StateAffine state)
    (blocksAffine : BlocksAffine blocks) :
    R1CS.RecipesDirect start
      (Hash.compileAbsorptions start state blocks).recipes := by
  induction blocks generalizing start state with
  | nil => trivial
  | cons block rest ih =>
      have blockAffine : ListAffine block := blocksAffine block (by simp)
      have restAffine : BlocksAffine rest := by
        intro current member
        exact blocksAffine current (by simp [member])
      have absorbedAffine := absorbE_affine state block stateAffine blockAffine
      have headDirect := compile_schedule_direct start
        (Hash.absorbE state block) absorbedAffine
      have headOutputAffine := compile_schedule_output_affine start
        (Hash.absorbE state block) absorbedAffine
      have tailDirect := ih (start + 592)
        (Permutation.compile start (Hash.absorbE state block)
          Permutation.schedule).output headOutputAffine restAffine
      unfold Hash.compileAbsorptions
      apply R1CS.recipesDirect_append
      · exact headDirect
      · simpa using tailDirect

theorem compileAbsorptions_output_affine (start : Nat) (state : Layer.EState)
    (blocks : List (List Expr)) (stateAffine : StateAffine state)
    (blocksAffine : BlocksAffine blocks) :
    StateAffine (Hash.compileAbsorptions start state blocks).output := by
  induction blocks generalizing start state with
  | nil => simpa [Hash.compileAbsorptions] using stateAffine
  | cons block rest ih =>
      have blockAffine : ListAffine block := blocksAffine block (by simp)
      have restAffine : BlocksAffine rest := by
        intro current member
        exact blocksAffine current (by simp [member])
      have absorbedAffine := absorbE_affine state block stateAffine blockAffine
      have headOutputAffine := compile_schedule_output_affine start
        (Hash.absorbE state block) absorbedAffine
      have tailOutputAffine := ih (start + 592)
        (Permutation.compile start (Hash.absorbE state block)
          Permutation.schedule).output headOutputAffine restAffine
      simpa [Hash.compileAbsorptions] using tailOutputAffine

theorem hash_compile_direct (start : Nat) (input : List Expr)
    (inputAffine : ListAffine input) :
    R1CS.RecipesDirect start (Hash.compile start input).recipes := by
  let blocks := Hash.inputChunks input
  let absorbed := Hash.compileAbsorptions start Hash.zeroE blocks
  have blocksAffine : BlocksAffine blocks :=
    inputChunks_affine input inputAffine
  have absorbedDirect := compileAbsorptions_direct start Hash.zeroE blocks
    zeroE_affine blocksAffine
  have absorbedOutputAffine := compileAbsorptions_output_affine start
    Hash.zeroE blocks zeroE_affine blocksAffine
  have finalDirect := compile_schedule_direct
    (start + absorbed.recipes.length) (Hash.padE absorbed.output)
    (padE_affine absorbed.output absorbedOutputAffine)
  unfold Hash.compile
  apply R1CS.recipesDirect_append
  · exact absorbedDirect
  · exact finalDirect

theorem hash_compile_output_eq (start : Nat) (input : List Expr) :
    (Hash.compile start input).output =
      Permutation.freshState
        (start + (Hash.inputChunks input).length * 592 + 584) := by
  unfold Hash.compile
  dsimp only
  rw [compile_schedule_output_eq,
    Hash.compileAbsorptions_recipes_length]

theorem hash_recipeConstraints_freshCount (start : Nat) (input : List Expr)
    (inputAffine : ListAffine input) :
    R1CS.totalFreshCount
      (recipeConstraints start (Hash.compile start input).recipes) = 0 :=
  R1CS.recipeConstraints_totalFreshCount start _
    (hash_compile_direct start input inputAffine)

theorem hash_recipeConstraints_rowCount (start : Nat) (input : List Expr)
    (inputAffine : ListAffine input) :
    R1CS.totalRowCount
      (recipeConstraints start (Hash.compile start input).recipes) =
        (Hash.inputChunks input).length * 592 + 592 := by
  rw [R1CS.recipeConstraints_totalRowCount start _
    (hash_compile_direct start input inputAffine)]
  exact Hash.compile_recipes_length start input

def HashInterfaceAffine (interface : Formal.Interface) (offset : Nat) : Prop :=
  ListAffine (interface.input offset) ∧
    ∀ lane, R1CS.IsAffine (interface.expected offset lane)

def hashConstraints (interface : Formal.Interface) (offset : Nat) : List Expr :=
  flatConstraints (Circuit.ops (Formal.circuit interface).main offset)

theorem hashConstraints_noFresh (interface : Formal.Interface) (offset : Nat)
    (affine : HashInterfaceAffine interface offset) :
    ∀ expression ∈ hashConstraints interface offset,
      R1CS.constraintFreshCount expression = 0 := by
  have recipesDirect := hash_compile_direct offset (interface.input offset)
    affine.1
  intro expression member
  change expression ∈ flatConstraints (Formal.opsAt interface offset) at member
  simp only [flatConstraints, List.mem_flatMap] at member
  rcases member with ⟨operation, operationMember, constraintMember⟩
  simp only [Formal.opsAt, List.mem_cons] at operationMember
  rcases operationMember with rfl | operationMember
  · exact R1CS.recipeConstraints_noFresh offset _ recipesDirect expression
      constraintMember
  · rw [Formal.assertions, List.mem_ofFn'] at operationMember
    rcases operationMember with ⟨lane, rfl⟩
    simp only [Op.flatConstraints, List.mem_singleton] at constraintMember
    subst expression
    rw [hash_compile_output_eq]
    exact R1CS.constraintFreshCount_recipe_eq_zero _ _
      (R1CS.IsDirectRecipe.of_affine _ (affine.2 lane))

theorem hashConstraints_rowsOne (interface : Formal.Interface) (offset : Nat)
    (affine : HashInterfaceAffine interface offset) :
    ∀ expression ∈ hashConstraints interface offset,
      R1CS.constraintRowCount expression = 1 := by
  have recipesDirect := hash_compile_direct offset (interface.input offset)
    affine.1
  intro expression member
  change expression ∈ flatConstraints (Formal.opsAt interface offset) at member
  simp only [flatConstraints, List.mem_flatMap] at member
  rcases member with ⟨operation, operationMember, constraintMember⟩
  simp only [Formal.opsAt, List.mem_cons] at operationMember
  rcases operationMember with rfl | operationMember
  · exact R1CS.recipeConstraints_rowsOne offset _ recipesDirect expression
      constraintMember
  · rw [Formal.assertions, List.mem_ofFn'] at operationMember
    rcases operationMember with ⟨lane, rfl⟩
    simp only [Op.flatConstraints, List.mem_singleton] at constraintMember
    subst expression
    rw [hash_compile_output_eq]
    exact R1CS.constraintRowCount_recipe_eq_one _ _
      (R1CS.IsDirectRecipe.of_affine _ (affine.2 lane))

theorem hashConstraints_freshCount (interface : Formal.Interface)
    (offset : Nat) (affine : HashInterfaceAffine interface offset) :
    R1CS.totalFreshCount (hashConstraints interface offset) = 0 :=
  R1CS.totalFreshCount_eq_zero_of_noFresh _
    (hashConstraints_noFresh interface offset affine)

theorem hashConstraints_rowCount (interface : Formal.Interface)
    (offset : Nat) (affine : HashInterfaceAffine interface offset) :
    R1CS.totalRowCount (hashConstraints interface offset) =
      (Hash.inputChunks (interface.input offset)).length * 592 + 596 := by
  rw [R1CS.totalRowCount_eq_length_of_rowsOne _
    (hashConstraints_rowsOne interface offset affine)]
  unfold hashConstraints
  change (flatConstraints (Formal.opsAt interface offset)).length = _
  simp [Formal.opsAt, Formal.assertions, flatConstraints,
    Op.flatConstraints, Hash.compile_recipes_length]

theorem hashPhysical_complete (interface : Formal.Interface) (offset : Nat)
    (env : Env) (physicalStart : Nat)
    (affine : HashInterfaceAffine interface offset)
    (logical : ConstraintsHold env (hashConstraints interface offset)) :
    R1CS.RowsHold env
      (R1CS.lowerConstraints (hashConstraints interface offset)
        physicalStart).rows :=
  R1CS.lowerConstraints_complete_of_noFresh env _ physicalStart
    (hashConstraints_noFresh interface offset affine) logical

end NightstreamFPrime.Layout.Poseidon2
