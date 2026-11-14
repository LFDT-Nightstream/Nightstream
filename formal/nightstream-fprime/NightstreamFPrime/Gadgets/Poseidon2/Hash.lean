import NightstreamFPrime.Gadgets.Poseidon2.Permutation
import NightstreamFPrime.Circuit.StraightLine

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

/-- `count` chunks of `rest`, one pass. -/
def chunksFrom {Alpha : Type} : Nat → List Alpha → List (List Alpha)
  | 0, _ => []
  | count + 1, rest =>
      rest.take Spec.Poseidon2.rate :: chunksFrom count (rest.drop Spec.Poseidon2.rate)

theorem chunksFrom_eq {Alpha : Type} :
    ∀ (count : Nat) (rest : List Alpha),
      chunksFrom count rest = (List.range count).map fun chunk =>
        (rest.drop (chunk * Spec.Poseidon2.rate)).take Spec.Poseidon2.rate
  | 0, _ => rfl
  | count + 1, rest => by
      rw [chunksFrom, chunksFrom_eq count, List.range_succ_eq_map]
      simp [List.drop_drop, Nat.succ_mul, Nat.add_comm]

/-- Compiled code chunks the input in one pass; `inputChunks` drops from the start for each
chunk. -/
def inputChunksFast {Alpha : Type} (input : List Alpha) : List (List Alpha) :=
  chunksFrom ((input.length + Spec.Poseidon2.rate - 1) / Spec.Poseidon2.rate) input

@[csimp] theorem inputChunks_eq_inputChunksFast : @inputChunks = @inputChunksFast := by
  funext Alpha input
  rw [inputChunksFast, chunksFrom_eq, inputChunks]

def zeroE : EState := fun _ => 0
def zeroF : FState := fun _ => 0

def absorbE (state : EState) (block : List Expr) : EState :=
  fun lane => state lane + block.getD lane.val 0

def absorbF (state : FState) (block : List F) : FState :=
  fun lane => state lane + block.getD lane.val 0

/-- The last block of `Spec.Poseidon2.pad`: the last words in their lanes, then the word 1 in the
next lane. -/
def padE (last : List Expr) (state : EState) : EState :=
  fun lane => if lane.val < last.length then state lane + last.getD lane.val 0
    else if lane.val = last.length then state lane + 1 else state lane

def padF (last : List F) (state : FState) : FState :=
  fun lane => if lane.val < last.length then state lane + last.getD lane.val 0
    else if lane.val = last.length then state lane + 1 else state lane

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
    norm_num [Expr.eval, fieldModulus]
  rw [← evalZero]
  exact (List.getD_map (n := lane.val) block (0 : Expr) (Expr.eval env)).symm

@[simp] theorem eval_padE (env : Env) (last : List Expr) (state : EState) :
    Layer.evalState env (padE last state) =
      padF (evalList env last) (Layer.evalState env state) := by
  have evalOne : (1 : Expr).eval env = (1 : F) := by
    apply Fin.ext
    norm_num [Expr.eval, fieldModulus]
  funext lane
  by_cases inside : lane.val < last.length
  · simp [Layer.evalState, padE, padF, inside, evalList]
  · by_cases next : lane.val = last.length <;>
      simp [Layer.evalState, padE, padF, inside, next, evalList, evalOne]

theorem eval_digestE (env : Env) (state : EState) :
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
    (padF (Spec.Poseidon2.lastWords input)
      (absorbManyF zeroF (Spec.Poseidon2.fullBlocks input))))

structure AbsorbProgram where
  recipes : List Expr
  output : EState

def compileAbsorptions (start : Nat) (state : EState) :
    List (List Expr) → AbsorbProgram
  | [] => ⟨[], state⟩
  | block :: rest =>
      let permutation := Permutation.compile start (absorbE state block)
        Permutation.schedule
      let tail := compileAbsorptions (start + 1174) permutation.output rest
      ⟨permutation.recipes ++ tail.recipes, tail.output⟩

@[inline] def compileAbsorptionsTR (start : Nat) (state : EState)
    (blocks : List (List Expr)) : AbsorbProgram :=
  go start state blocks []
where
  go : Nat → EState → List (List Expr) → List Expr → AbsorbProgram
    | _, current, [], recipesRev => ⟨recipesRev.reverse, current⟩
    | output, current, block :: rest, recipesRev =>
        let permutation := Permutation.compile output
          (absorbE current block) Permutation.schedule
        go (output + 1174) permutation.output rest
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
      · simp [List.reverse_append, List.append_assoc]
      · rfl

@[csimp] theorem compileAbsorptions_eq_compileAbsorptionsTR :
    @compileAbsorptions = @compileAbsorptionsTR := by
  funext start state blocks
  rw [compileAbsorptionsTR, compileAbsorptionsTR_go_eq]
  rfl

structure Program where
  recipes : List Expr
  output : EState

/-- `Spec.Poseidon2.hash_eq_last`: one permutation for each full block, then one for the last
words and the word 1. -/
def compile (start : Nat) (input : List Expr) : Program :=
  let absorbed := compileAbsorptions start zeroE (Spec.Poseidon2.fullBlocks input)
  let finalStart := start + absorbed.recipes.length
  let finalPermutation := Permutation.compile finalStart
    (padE (Spec.Poseidon2.lastWords input) absorbed.output) Permutation.schedule
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

theorem fullBlocks_below (input : List Expr) (bound : Nat)
    (hinput : ∀ expression ∈ input, expression.VarsBelow bound) :
    BlocksBelow bound (Spec.Poseidon2.fullBlocks input) := by
  intro block hblock expression hexpression
  simp only [Spec.Poseidon2.fullBlocks, List.mem_map] at hblock
  rcases hblock with ⟨chunk, _, rfl⟩
  apply hinput expression
  exact List.mem_of_mem_drop (List.mem_of_mem_take hexpression)

theorem absorbE_varsBelow (state : EState) (block : List Expr) {bound : Nat}
    (hstate : ∀ lane, (state lane).VarsBelow bound)
    (hblock : ∀ expression ∈ block, expression.VarsBelow bound)
    (lane : Fin 16) :
    (absorbE state block lane).VarsBelow bound := by
  simp only [absorbE, Expr.VarsBelow]
  constructor
  · exact hstate lane
  · by_cases hin : lane.val < block.length
    · rw [List.getD_eq_get block 0 ⟨lane.val, hin⟩]
      exact hblock (block.get ⟨lane.val, hin⟩) (List.get_mem _ _)
    · simp [List.getD, hin, Expr.VarsBelow]

theorem padE_varsBelow (input : List Expr) (state : EState) {bound : Nat}
    (hinput : ∀ expression ∈ input, expression.VarsBelow bound)
    (hstate : ∀ lane, (state lane).VarsBelow bound) (lane : Fin 16) :
    (padE (Spec.Poseidon2.lastWords input) state lane).VarsBelow bound := by
  unfold padE
  split
  · rename_i inside
    refine ⟨hstate lane, ?_⟩
    rw [List.getD_eq_getElem _ _ inside]
    exact hinput _ (List.mem_of_mem_drop (List.getElem_mem inside))
  · split
    · exact ⟨hstate lane, trivial⟩
    · exact hstate lane

@[simp] theorem compileAbsorptions_recipes_length (start : Nat)
    (state : EState) (blocks : List (List Expr)) :
    (compileAbsorptions start state blocks).recipes.length =
      blocks.length * 1174 := by
  induction blocks generalizing start state with
  | nil => rfl
  | cons block rest ih =>
      simp only [compileAbsorptions, List.length_append,
        Permutation.compile_schedule_recipe_count, ih, List.length_cons]
      omega

@[simp] theorem compile_recipes_length (start : Nat) (input : List Expr) :
    (compile start input).recipes.length =
      input.length / Spec.Poseidon2.rate * 1174 + 1174 := by
  unfold compile
  dsimp only
  rw [List.length_append, compileAbsorptions_recipes_length,
    Permutation.compile_schedule_recipe_count, Spec.Poseidon2.fullBlocks, List.length_map,
    List.length_range]

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
            Permutation.schedule).output lane).VarsBelow (start + 1174) := by
        intro lane
        have outputBound := Permutation.compile_output_varsBelow start
          (absorbE state block) Permutation.schedule habsorb lane
        rw [Permutation.compile_schedule_recipe_count] at outputBound
        exact outputBound
      have htail := ih (start + 1174)
        (Permutation.compile start (absorbE state block)
          Permutation.schedule).output houtput
        (blocksBelow_mono hrest (by omega))
      apply Permutation.recipesCausal_append_causal start _ _ hpermutation
      rw [Permutation.compile_schedule_recipe_count]
      exact htail

theorem compileAbsorptions_output_varsBelow (start : Nat) (state : EState)
    (blocks : List (List Expr))
    (hstate : ∀ lane, (state lane).VarsBelow start)
    (hblocks : BlocksBelow start blocks) (lane : Fin 16) :
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
            Permutation.schedule).output current).VarsBelow (start + 1174) := by
        intro current
        have outputBound := Permutation.compile_output_varsBelow start
          (absorbE state block) Permutation.schedule habsorb current
        rw [Permutation.compile_schedule_recipe_count] at outputBound
        exact outputBound
      have tail := ih (start + 1174)
        (Permutation.compile start (absorbE state block)
          Permutation.schedule).output houtput
        (blocksBelow_mono hrest (by omega))
      convert tail using 1 <;>
        simp only [compileAbsorptions, List.length_append,
          Permutation.compile_schedule_recipe_count,
          compileAbsorptions_recipes_length]; omega

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
          ConstraintsHold env (recipeConstraints (start + 1174)
            (compileAbsorptions (start + 1174) permutation.output rest).recipes) := by
        rw [compileAbsorptions, Permutation.recipeConstraints_append] at hrows
        have separated :=
          (Permutation.constraintsHold_append env _ _).mp hrows
        simpa [permutation] using! separated
      have headSound := Permutation.compile_sound env start
        (absorbE state block) Permutation.schedule splitRows.1
      have tailSound := ih (start + 1174) permutation.output splitRows.2
      simpa [compileAbsorptions, absorbManyF, permutation, headSound] using
        tailSound

theorem compile_causal (start : Nat) (input : List Expr)
    (hinput : ∀ expression ∈ input, expression.VarsBelow start) :
    RecipesCausal start (compile start input).recipes := by
  let blocks := Spec.Poseidon2.fullBlocks input
  let absorbed := compileAbsorptions start zeroE blocks
  have hzero : ∀ lane, (zeroE lane).VarsBelow start := by
    intro lane
    trivial
  have hblocks : BlocksBelow start blocks := fullBlocks_below input start hinput
  have habsorbed := compileAbsorptions_causal start zeroE blocks hzero hblocks
  have houtput : ∀ lane, (absorbed.output lane).VarsBelow
      (start + absorbed.recipes.length) :=
    compileAbsorptions_output_varsBelow start zeroE blocks hzero hblocks
  have hfinal := Permutation.compile_schedule_causal
    (start + absorbed.recipes.length)
    (padE (Spec.Poseidon2.lastWords input) absorbed.output)
    (padE_varsBelow input absorbed.output
      (fun expression member => Expr.VarsBelow.mono expression (hinput expression member)
        (Nat.le_add_right _ _)) houtput)
  apply Permutation.recipesCausal_append_causal start _ _ habsorbed
  exact hfinal

theorem compile_output_varsBelow (start : Nat) (input : List Expr)
    (hinput : ∀ expression ∈ input, expression.VarsBelow start)
    (lane : Fin 16) :
    ((compile start input).output lane).VarsBelow
      (start + (compile start input).recipes.length) := by
  let blocks := Spec.Poseidon2.fullBlocks input
  let absorbed := compileAbsorptions start zeroE blocks
  let last := Spec.Poseidon2.lastWords input
  have hzero : ∀ current, (zeroE current).VarsBelow start := by
    intro current
    trivial
  have hblocks : BlocksBelow start blocks :=
    fullBlocks_below input start hinput
  have houtput : ∀ current, (absorbed.output current).VarsBelow
      (start + absorbed.recipes.length) :=
    compileAbsorptions_output_varsBelow start zeroE blocks hzero hblocks
  have finalOutput := Permutation.compile_output_varsBelow
    (start + absorbed.recipes.length) (padE last absorbed.output)
    Permutation.schedule (padE_varsBelow input absorbed.output
      (fun expression member => Expr.VarsBelow.mono expression (hinput expression member)
        (Nat.le_add_right _ _)) houtput) lane
  change ((Permutation.compile (start + absorbed.recipes.length)
      (padE last absorbed.output) Permutation.schedule).output lane).VarsBelow
    (start + (absorbed.recipes ++
      (Permutation.compile (start + absorbed.recipes.length)
        (padE last absorbed.output) Permutation.schedule).recipes).length)
  convert finalOutput using 1;
    simp only [List.length_append,
      Permutation.compile_schedule_recipe_count]; omega

theorem absorbF_input_eq_reference (state : FState) (block : List F) :
    List.ofFn (absorbF state block) =
      (List.range Spec.Poseidon2.width).map fun index =>
        (List.ofFn state).getD index 0 + block.getD index 0 := by
  rw [Layer.ofFn_state (absorbF state block), Layer.ofFn_state state]
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

/-- The padding block adds the last words and the word 1; the lanes after them add zero. -/
theorem padF_eq_absorbF (last : List F) (state : FState) :
    padF last state = absorbF state (last ++ [1]) := by
  funext lane
  unfold padF absorbF
  split
  · rename_i inside
    rw [List.getD_eq_getElem _ _ inside, List.getD_eq_getElem _ _ (by simp; omega),
      List.getElem_append_left inside]
  · rename_i outside
    split
    · rename_i next
      rw [List.getD_eq_getElem _ _ (by simp; omega), List.getElem_append_right (by omega)]
      simp [next]
    · rename_i after
      rw [List.getD_eq_default _ _ (by simp; omega)]
      simp

theorem digestF_eq_take (state : FState) :
    List.ofFn (digestF state) =
      (List.ofFn state).take Spec.Poseidon2.digestLen := by
  rw [Layer.ofFn_state state]
  simp [digestF, Spec.Poseidon2.digestLen, List.ofFn_succ]

theorem hashF_eq_reference (input : List F) :
    List.ofFn (hashF input) = Spec.Poseidon2.hash input := by
  let absorbed := absorbManyF zeroF (Spec.Poseidon2.fullBlocks input)
  calc
    List.ofFn (hashF input) =
        (List.ofFn (Permutation.runF Permutation.schedule
          (absorbF absorbed (Spec.Poseidon2.lastWords input ++ [1])))).take
          Spec.Poseidon2.digestLen := by
      rw [← padF_eq_absorbF]
      exact digestF_eq_take _
    _ = (Spec.Poseidon2.absorbBlock (List.ofFn absorbed)
          (Spec.Poseidon2.lastWords input ++ [1])).take Spec.Poseidon2.digestLen := by
      rw [absorbStepF_eq_reference]
    _ = Spec.Poseidon2.hash input := by
      rw [absorbManyF_eq_reference, Spec.Poseidon2.hash_eq_last]
      rfl

theorem inputChunks_eval (env : Env) (input : List Expr) :
    (inputChunks input).map (evalList env) =
      inputChunks (evalList env input) := by
  simp [inputChunks, evalList, List.map_map, Function.comp_def]

theorem fullBlocks_eval (env : Env) (input : List Expr) :
    (Spec.Poseidon2.fullBlocks input).map (evalList env) =
      Spec.Poseidon2.fullBlocks (evalList env input) := by
  simp [Spec.Poseidon2.fullBlocks, evalList, List.map_map, Function.comp_def]

theorem lastWords_eval (env : Env) (input : List Expr) :
    evalList env (Spec.Poseidon2.lastWords input) =
      Spec.Poseidon2.lastWords (evalList env input) := by
  simp [Spec.Poseidon2.lastWords, evalList]

theorem compile_sound (env : Env) (start : Nat) (input : List Expr)
    (hrows : ConstraintsHold env
      (recipeConstraints start (compile start input).recipes)) :
    List.ofFn (fun lane =>
      (digestE (compile start input).output lane).eval env) =
      Spec.Poseidon2.hash (evalList env input) := by
  let blocks := Spec.Poseidon2.fullBlocks input
  let last := Spec.Poseidon2.lastWords input
  let absorbed := compileAbsorptions start zeroE blocks
  let finalStart := start + absorbed.recipes.length
  let finalPermutation := Permutation.compile finalStart
    (padE last absorbed.output) Permutation.schedule
  have splitRows :
      ConstraintsHold env (recipeConstraints start absorbed.recipes) ∧
      ConstraintsHold env
        (recipeConstraints finalStart finalPermutation.recipes) := by
    rw [compile, Permutation.recipeConstraints_append] at hrows
    exact (Permutation.constraintsHold_append env _ _).mp hrows
  have absorbedSound := compileAbsorptions_sound env start zeroE blocks splitRows.1
  have finalSound := Permutation.compile_sound env finalStart
    (padE last absorbed.output) Permutation.schedule splitRows.2
  have outputSound :
      Layer.evalState env finalPermutation.output =
        Permutation.runF Permutation.schedule
          (padF (Spec.Poseidon2.lastWords (evalList env input))
            (absorbManyF zeroF (Spec.Poseidon2.fullBlocks (evalList env input)))) := by
    calc
      Layer.evalState env finalPermutation.output =
          Permutation.runF Permutation.schedule
            (Layer.evalState env (padE last absorbed.output)) := finalSound
      _ = Permutation.runF Permutation.schedule
            (padF (evalList env last) (Layer.evalState env absorbed.output)) := by
          rw [eval_padE]
      _ = Permutation.runF Permutation.schedule
            (padF (evalList env last) (absorbManyF zeroF
              (blocks.map (evalList env)))) := by
          rw [absorbedSound, eval_zeroE]
      _ = Permutation.runF Permutation.schedule
            (padF (Spec.Poseidon2.lastWords (evalList env input))
              (absorbManyF zeroF (Spec.Poseidon2.fullBlocks (evalList env input)))) := by
          rw [fullBlocks_eval, lastWords_eval]
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
