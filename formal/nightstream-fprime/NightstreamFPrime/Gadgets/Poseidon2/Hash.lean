import NightstreamFPrime.Gadgets.Poseidon2.Permutation

/-!
Owns the logical Poseidon2 sponge compiler. It uses the proved permutation for
every absorb block and for the final `+1` padding permutation. The output is
the first four lanes. Physical rows, columns, and lifecycle serialization are
owned by later layers.
-/

namespace NightstreamFPrime.Gadgets.Poseidon2.Hash

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit

abbrev EState := Layer.EState
abbrev FState := Layer.FState

def evalList (env : Env) (values : List Expr) : List F :=
  values.map (Expr.eval env)

def inputChunks {Alpha : Type} (input : List Alpha) : List (List Alpha) :=
  (List.range ((input.length + Spec.Poseidon2.rate - 1) /
    Spec.Poseidon2.rate)).map fun chunk =>
      (input.drop (chunk * Spec.Poseidon2.rate)).take Spec.Poseidon2.rate

def zeroE : EState := fun _ => 0
def zeroF : FState := fun _ => 0

def absorbE (state : EState) (block : List Expr) : EState :=
  fun lane => state lane + block.getD lane.val 0

def absorbF (state : FState) (block : List F) : FState :=
  fun lane => state lane + block.getD lane.val 0

def padE (state : EState) : EState :=
  fun lane => if lane.val = 0 then state lane + 1 else state lane

def padF (state : FState) : FState :=
  fun lane => if lane.val = 0 then state lane + 1 else state lane

def digestE (state : EState) : Fin 4 → Expr :=
  fun lane => state ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩

def digestF (state : FState) : Fin 4 → F :=
  fun lane => state ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩

@[simp] theorem eval_zeroE (env : Env) :
    Layer.evalState env zeroE = zeroF := by
  funext lane
  rfl

@[simp] theorem eval_absorbE (env : Env) (state : EState)
    (block : List Expr) :
    Layer.evalState env (absorbE state block) =
      absorbF (Layer.evalState env state) (evalList env block) := by
  funext lane
  simp only [Layer.evalState, absorbE, absorbF, Expr.eval_hadd]
  congr 1
  change (block.getD lane.val (0 : Expr)).eval env =
    (block.map (Expr.eval env)).getD lane.val (0 : F)
  have evalZero : (0 : Expr).eval env = (0 : F) := by
    apply Fin.ext
    norm_num [Expr.eval, goldilocksModulus]
  rw [← evalZero]
  exact (List.getD_map (n := lane.val) block (0 : Expr) (Expr.eval env)).symm

@[simp] theorem eval_padE (env : Env) (state : EState) :
    Layer.evalState env (padE state) = padF (Layer.evalState env state) := by
  have evalOne : (1 : Expr).eval env = (1 : F) := by
    apply Fin.ext
    norm_num [Expr.eval, goldilocksModulus]
  funext lane
  by_cases hzero : lane.val = 0 <;>
    simp [Layer.evalState, padE, padF, hzero, evalOne]

@[simp] theorem eval_digestE (env : Env) (state : EState) :
    (fun lane => (digestE state lane).eval env) =
      digestF (Layer.evalState env state) := by
  rfl

def absorbManyF : FState → List (List F) → FState
  | state, [] => state
  | state, block :: rest =>
      absorbManyF
        (Permutation.runF Permutation.schedule (absorbF state block)) rest

def hashF (input : List F) : Fin 4 → F :=
  digestF (Permutation.runF Permutation.schedule
    (padF (absorbManyF zeroF (inputChunks input))))

/-- One straight-line program for a sequence of absorb permutations. -/
structure AbsorbProgram where
  recipes : List Expr
  output : EState

def compileAbsorptions (start : Nat) (state : EState) :
    List (List Expr) → AbsorbProgram
  | [] => ⟨[], state⟩
  | block :: rest =>
      let permutation := Permutation.compile start (absorbE state block)
        Permutation.schedule
      let tail := compileAbsorptions (start + 592) permutation.output rest
      ⟨permutation.recipes ++ tail.recipes, tail.output⟩

/-- Tail-recursive executable form of `compileAbsorptions`. The kernel keeps
the structural definition above; compiled emission uses this proved form. -/
@[inline] def compileAbsorptionsTR (start : Nat) (state : EState)
    (blocks : List (List Expr)) : AbsorbProgram :=
  go start state blocks []
where
  go : Nat → EState → List (List Expr) → List Expr → AbsorbProgram
    | _, current, [], recipesRev => ⟨recipesRev.reverse, current⟩
    | output, current, block :: rest, recipesRev =>
        let permutation := Permutation.compile output
          (absorbE current block) Permutation.schedule
        go (output + 592) permutation.output rest
          (permutation.recipes.reverse ++ recipesRev)

private theorem compileAbsorptionsTR_go_eq (start : Nat) (state : EState)
    (blocks : List (List Expr)) (recipesRev : List Expr) :
    compileAbsorptionsTR.go start state blocks recipesRev =
      let program := compileAbsorptions start state blocks
      ⟨recipesRev.reverse ++ program.recipes, program.output⟩ := by
  induction blocks generalizing start state recipesRev with
  | nil => simp [compileAbsorptionsTR.go, compileAbsorptions]
  | cons block rest inductionHypothesis =>
      simp only [compileAbsorptionsTR.go, compileAbsorptions]
      let permutation := Permutation.compile start
        (absorbE state block) Permutation.schedule
      rw [inductionHypothesis]
      apply congrArg₂ AbsorbProgram.mk
      · simp [List.reverse_append, List.append_assoc, permutation]
      · rfl

@[csimp] theorem compileAbsorptions_eq_compileAbsorptionsTR :
    @compileAbsorptions = @compileAbsorptionsTR := by
  funext start state blocks
  rw [compileAbsorptionsTR, compileAbsorptionsTR_go_eq]
  rfl

/-- The complete sponge program, including the final padding permutation. -/
structure Program where
  recipes : List Expr
  output : EState

def compile (start : Nat) (input : List Expr) : Program :=
  let absorbed := compileAbsorptions start zeroE (inputChunks input)
  let finalStart := start + absorbed.recipes.length
  let finalPermutation := Permutation.compile finalStart
    (padE absorbed.output) Permutation.schedule
  ⟨absorbed.recipes ++ finalPermutation.recipes, finalPermutation.output⟩

def BlocksBelow (bound : Nat) (blocks : List (List Expr)) : Prop :=
  ∀ block ∈ blocks, ∀ expression ∈ block, expression.VarsBelow bound

theorem blocksBelow_mono {lower upper : Nat} {blocks : List (List Expr)}
    (hblocks : BlocksBelow lower blocks) (hle : lower ≤ upper) :
    BlocksBelow upper blocks := by
  intro block hblock expression hexpression
  exact Expr.VarsBelow.mono expression
    (hblocks block hblock expression hexpression) hle

theorem inputChunks_below (input : List Expr) (bound : Nat)
    (hinput : ∀ expression ∈ input, expression.VarsBelow bound) :
    BlocksBelow bound (inputChunks input) := by
  intro block hblock expression hexpression
  simp only [inputChunks, List.mem_map] at hblock
  rcases hblock with ⟨chunk, _, rfl⟩
  apply hinput expression
  exact List.mem_of_mem_drop (List.mem_of_mem_take hexpression)

theorem absorbE_varsBelow (state : EState) (block : List Expr) {bound : Nat}
    (hstate : ∀ lane, (state lane).VarsBelow bound)
    (hblock : ∀ expression ∈ block, expression.VarsBelow bound)
    (lane : Fin 8) :
    (absorbE state block lane).VarsBelow bound := by
  simp only [absorbE, Expr.VarsBelow]
  constructor
  · exact hstate lane
  · by_cases hin : lane.val < block.length
    · rw [List.getD_eq_get block 0 ⟨lane.val, hin⟩]
      exact hblock (block.get ⟨lane.val, hin⟩) (List.get_mem _ _)
    · simp [List.getD, hin, Expr.VarsBelow]

theorem padE_varsBelow (state : EState) {bound : Nat}
    (hstate : ∀ lane, (state lane).VarsBelow bound) (lane : Fin 8) :
    (padE state lane).VarsBelow bound := by
  by_cases hzero : lane.val = 0
  · simp [padE, hzero, Expr.VarsBelow, hstate]
  · simp [padE, hzero, hstate]

@[simp] theorem compileAbsorptions_recipes_length (start : Nat)
    (state : EState) (blocks : List (List Expr)) :
    (compileAbsorptions start state blocks).recipes.length =
      blocks.length * 592 := by
  induction blocks generalizing start state with
  | nil => rfl
  | cons block rest ih =>
      simp only [compileAbsorptions, List.length_append,
        Permutation.compile_schedule_recipe_count, ih, List.length_cons]
      omega

@[simp] theorem compile_recipes_length (start : Nat) (input : List Expr) :
    (compile start input).recipes.length =
      (inputChunks input).length * 592 + 592 := by
  unfold compile
  dsimp only
  rw [List.length_append, compileAbsorptions_recipes_length,
    Permutation.compile_schedule_recipe_count]

theorem compileAbsorptions_causal (start : Nat) (state : EState)
    (blocks : List (List Expr))
    (hstate : ∀ lane, (state lane).VarsBelow start)
    (hblocks : BlocksBelow start blocks) :
    RecipesCausal start (compileAbsorptions start state blocks).recipes := by
  induction blocks generalizing start state with
  | nil => trivial
  | cons block rest ih =>
      have hblock : ∀ expression ∈ block, expression.VarsBelow start :=
        hblocks block (by simp)
      have hrest : BlocksBelow start rest := by
        intro current member
        exact hblocks current (by simp [member])
      have habsorb : ∀ lane, (absorbE state block lane).VarsBelow start :=
        absorbE_varsBelow state block hstate hblock
      have hpermutation := Permutation.compile_schedule_causal start
        (absorbE state block) habsorb
      have houtput : ∀ lane,
          ((Permutation.compile start (absorbE state block)
            Permutation.schedule).output lane).VarsBelow (start + 592) := by
        intro lane
        have outputBound := Permutation.compile_output_varsBelow start
          (absorbE state block) Permutation.schedule habsorb lane
        rw [Permutation.compile_schedule_recipe_count] at outputBound
        exact outputBound
      have htail := ih (start + 592)
        (Permutation.compile start (absorbE state block)
          Permutation.schedule).output houtput
        (blocksBelow_mono hrest (by omega))
      apply Permutation.recipesCausal_append_causal start _ _ hpermutation
      rw [Permutation.compile_schedule_recipe_count]
      exact htail

theorem compileAbsorptions_output_varsBelow (start : Nat) (state : EState)
    (blocks : List (List Expr))
    (hstate : ∀ lane, (state lane).VarsBelow start)
    (hblocks : BlocksBelow start blocks) (lane : Fin 8) :
    ((compileAbsorptions start state blocks).output lane).VarsBelow
      (start + (compileAbsorptions start state blocks).recipes.length) := by
  induction blocks generalizing start state with
  | nil => simpa [compileAbsorptions] using hstate lane
  | cons block rest ih =>
      have hblock : ∀ expression ∈ block, expression.VarsBelow start :=
        hblocks block (by simp)
      have hrest : BlocksBelow start rest := by
        intro current member
        exact hblocks current (by simp [member])
      have habsorb : ∀ lane, (absorbE state block lane).VarsBelow start :=
        absorbE_varsBelow state block hstate hblock
      have houtput : ∀ current,
          ((Permutation.compile start (absorbE state block)
            Permutation.schedule).output current).VarsBelow (start + 592) := by
        intro current
        have outputBound := Permutation.compile_output_varsBelow start
          (absorbE state block) Permutation.schedule habsorb current
        rw [Permutation.compile_schedule_recipe_count] at outputBound
        exact outputBound
      have tail := ih (start + 592)
        (Permutation.compile start (absorbE state block)
          Permutation.schedule).output houtput
        (blocksBelow_mono hrest (by omega))
      convert tail using 1 <;>
        simp only [compileAbsorptions, List.length_append,
          Permutation.compile_schedule_recipe_count,
          compileAbsorptions_recipes_length] <;> omega

theorem compileAbsorptions_sound (env : Env) (start : Nat) (state : EState)
    (blocks : List (List Expr))
    (hrows : ConstraintsHold env (recipeConstraints start
      (compileAbsorptions start state blocks).recipes)) :
    Layer.evalState env (compileAbsorptions start state blocks).output =
      absorbManyF (Layer.evalState env state) (blocks.map (evalList env)) := by
  induction blocks generalizing start state with
  | nil => rfl
  | cons block rest ih =>
      let permutation := Permutation.compile start (absorbE state block)
        Permutation.schedule
      have splitRows :
          ConstraintsHold env (recipeConstraints start permutation.recipes) ∧
          ConstraintsHold env (recipeConstraints (start + 592)
            (compileAbsorptions (start + 592) permutation.output rest).recipes) := by
        rw [compileAbsorptions, Permutation.recipeConstraints_append] at hrows
        have separated :=
          (Permutation.constraintsHold_append env _ _).mp hrows
        simpa [permutation] using separated
      have headSound := Permutation.compile_sound env start
        (absorbE state block) Permutation.schedule splitRows.1
      have tailSound := ih (start + 592) permutation.output splitRows.2
      simpa [compileAbsorptions, absorbManyF, permutation, headSound] using
        tailSound

theorem compile_causal (start : Nat) (input : List Expr)
    (hinput : ∀ expression ∈ input, expression.VarsBelow start) :
    RecipesCausal start (compile start input).recipes := by
  let blocks := inputChunks input
  let absorbed := compileAbsorptions start zeroE blocks
  have hzero : ∀ lane, (zeroE lane).VarsBelow start := by
    intro lane
    trivial
  have hblocks : BlocksBelow start blocks := inputChunks_below input start hinput
  have habsorbed := compileAbsorptions_causal start zeroE blocks hzero hblocks
  have houtput : ∀ lane, (absorbed.output lane).VarsBelow
      (start + absorbed.recipes.length) :=
    compileAbsorptions_output_varsBelow start zeroE blocks hzero hblocks
  have hfinal := Permutation.compile_schedule_causal
    (start + absorbed.recipes.length) (padE absorbed.output)
    (padE_varsBelow absorbed.output houtput)
  apply Permutation.recipesCausal_append_causal start _ _ habsorbed
  exact hfinal

/-- Every final sponge output expression reads only the input prefix and the
exact recipe interval allocated by `compile`. -/
theorem compile_output_varsBelow (start : Nat) (input : List Expr)
    (hinput : ∀ expression ∈ input, expression.VarsBelow start)
    (lane : Fin 8) :
    ((compile start input).output lane).VarsBelow
      (start + (compile start input).recipes.length) := by
  let blocks := inputChunks input
  let absorbed := compileAbsorptions start zeroE blocks
  have hzero : ∀ current, (zeroE current).VarsBelow start := by
    intro current
    trivial
  have hblocks : BlocksBelow start blocks :=
    inputChunks_below input start hinput
  have houtput : ∀ current, (absorbed.output current).VarsBelow
      (start + absorbed.recipes.length) :=
    compileAbsorptions_output_varsBelow start zeroE blocks hzero hblocks
  have finalOutput := Permutation.compile_output_varsBelow
    (start + absorbed.recipes.length) (padE absorbed.output)
    Permutation.schedule (padE_varsBelow absorbed.output houtput) lane
  change ((Permutation.compile (start + absorbed.recipes.length)
      (padE absorbed.output) Permutation.schedule).output lane).VarsBelow
    (start + (absorbed.recipes ++
      (Permutation.compile (start + absorbed.recipes.length)
        (padE absorbed.output) Permutation.schedule).recipes).length)
  convert finalOutput using 1 <;>
    simp only [List.length_append,
      Permutation.compile_schedule_recipe_count] <;> omega

private theorem ofFn_state {Alpha : Type} (state : Fin 8 → Alpha) :
    List.ofFn state =
      [state 0, state 1, state 2, state 3, state 4, state 5, state 6, state 7] := by
  simp [List.ofFn_succ]

theorem absorbF_input_eq_reference (state : FState) (block : List F) :
    List.ofFn (absorbF state block) =
      (List.range Spec.Poseidon2.width).map fun index =>
        (List.ofFn state).getD index 0 + block.getD index 0 := by
  rw [ofFn_state (absorbF state block), ofFn_state state]
  simp [absorbF, Spec.Poseidon2.width, List.range_succ]

theorem absorbStepF_eq_reference (state : FState) (block : List F) :
    List.ofFn (Permutation.runF Permutation.schedule (absorbF state block)) =
      Spec.Poseidon2.absorbBlock (List.ofFn state) block := by
  calc
    List.ofFn (Permutation.runF Permutation.schedule (absorbF state block)) =
        Permutation.runReference Permutation.schedule
          (List.ofFn (absorbF state block)) :=
      Permutation.runF_eq_reference _ _
    _ = Spec.Poseidon2.permute (List.ofFn (absorbF state block)) :=
      Permutation.runReference_schedule _
    _ = Spec.Poseidon2.absorbBlock (List.ofFn state) block := by
      rw [Spec.Poseidon2.absorbBlock, absorbF_input_eq_reference]

theorem absorbManyF_eq_reference (state : FState) (blocks : List (List F)) :
    List.ofFn (absorbManyF state blocks) =
      blocks.foldl Spec.Poseidon2.absorbBlock (List.ofFn state) := by
  induction blocks generalizing state with
  | nil => rfl
  | cons block rest ih =>
      calc
        List.ofFn (absorbManyF state (block :: rest)) =
            rest.foldl Spec.Poseidon2.absorbBlock
              (List.ofFn (Permutation.runF Permutation.schedule
                (absorbF state block))) :=
          ih _
        _ = rest.foldl Spec.Poseidon2.absorbBlock
              (Spec.Poseidon2.absorbBlock (List.ofFn state) block) := by
          rw [absorbStepF_eq_reference]
        _ = (block :: rest).foldl Spec.Poseidon2.absorbBlock
              (List.ofFn state) := by
          simp only [List.foldl_cons]

theorem padF_input_eq_reference (state : FState) :
    List.ofFn (padF state) =
      (List.range Spec.Poseidon2.width).map fun index =>
        if index = 0 then (List.ofFn state).getD 0 0 + 1
        else (List.ofFn state).getD index 0 := by
  rw [ofFn_state (padF state), ofFn_state state]
  simp [padF, Spec.Poseidon2.width, List.range_succ]

theorem digestF_eq_take (state : FState) :
    List.ofFn (digestF state) =
      (List.ofFn state).take Spec.Poseidon2.digestLen := by
  rw [ofFn_state state]
  simp [digestF, Spec.Poseidon2.digestLen, List.ofFn_succ]

theorem hashF_eq_reference (input : List F) :
    List.ofFn (hashF input) = Spec.Poseidon2.hash input := by
  let absorbed := absorbManyF zeroF (inputChunks input)
  calc
    List.ofFn (hashF input) =
        (List.ofFn (Permutation.runF Permutation.schedule
          (padF absorbed))).take Spec.Poseidon2.digestLen := by
      change List.ofFn (digestF (Permutation.runF Permutation.schedule
        (padF absorbed))) = _
      exact digestF_eq_take _
    _ = (Spec.Poseidon2.permute (List.ofFn (padF absorbed))).take
          Spec.Poseidon2.digestLen := by
      rw [Permutation.runF_eq_reference, Permutation.runReference_schedule]
    _ = (Spec.Poseidon2.permute
          ((List.range Spec.Poseidon2.width).map fun index =>
            if index = 0 then (List.ofFn absorbed).getD 0 0 + 1
            else (List.ofFn absorbed).getD index 0)).take
          Spec.Poseidon2.digestLen := by
      rw [padF_input_eq_reference]
    _ = Spec.Poseidon2.hash input := by
      rw [absorbManyF_eq_reference]
      rfl

theorem inputChunks_eval (env : Env) (input : List Expr) :
    (inputChunks input).map (evalList env) =
      inputChunks (evalList env input) := by
  simp [inputChunks, evalList, List.map_map, Function.comp_def]

/-- The complete generated sponge rows imply the exact executable Poseidon2
digest over the evaluated symbolic input. -/
theorem compile_sound (env : Env) (start : Nat) (input : List Expr)
    (hrows : ConstraintsHold env
      (recipeConstraints start (compile start input).recipes)) :
    List.ofFn (fun lane =>
      (digestE (compile start input).output lane).eval env) =
      Spec.Poseidon2.hash (evalList env input) := by
  let blocks := inputChunks input
  let absorbed := compileAbsorptions start zeroE blocks
  let finalStart := start + absorbed.recipes.length
  let finalPermutation := Permutation.compile finalStart
    (padE absorbed.output) Permutation.schedule
  have splitRows :
      ConstraintsHold env (recipeConstraints start absorbed.recipes) ∧
      ConstraintsHold env
        (recipeConstraints finalStart finalPermutation.recipes) := by
    rw [compile, Permutation.recipeConstraints_append] at hrows
    exact (Permutation.constraintsHold_append env _ _).mp hrows
  have absorbedSound := compileAbsorptions_sound env start zeroE blocks splitRows.1
  have finalSound := Permutation.compile_sound env finalStart
    (padE absorbed.output) Permutation.schedule splitRows.2
  have outputSound :
      Layer.evalState env finalPermutation.output =
        Permutation.runF Permutation.schedule
          (padF (absorbManyF zeroF (inputChunks (evalList env input)))) := by
    calc
      Layer.evalState env finalPermutation.output =
          Permutation.runF Permutation.schedule
            (Layer.evalState env (padE absorbed.output)) := finalSound
      _ = Permutation.runF Permutation.schedule
            (padF (Layer.evalState env absorbed.output)) := by
          rw [eval_padE]
      _ = Permutation.runF Permutation.schedule
            (padF (absorbManyF zeroF
              (blocks.map (evalList env)))) := by
          rw [absorbedSound, eval_zeroE]
      _ = Permutation.runF Permutation.schedule
            (padF (absorbManyF zeroF
              (inputChunks (evalList env input)))) := by
          rw [inputChunks_eval]
  calc
    List.ofFn (fun lane =>
        (digestE (compile start input).output lane).eval env) =
        List.ofFn (digestF (Layer.evalState env finalPermutation.output)) := by
      rfl
    _ = List.ofFn (hashF (evalList env input)) := by
      rw [outputSound]
      rfl
    _ = Spec.Poseidon2.hash (evalList env input) :=
      hashF_eq_reference _

end NightstreamFPrime.Gadgets.Poseidon2.Hash
