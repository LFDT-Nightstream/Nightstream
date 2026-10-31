import NightstreamFPrime.Gadgets.Poseidon2.Support

/-! Owns one proof-carrying Poseidon2 sponge child: it absorbs an explicit list
of rate chunks into an initial state and exposes the final state to its parent.
Each chunk adds its words to the rate lanes and permutes
(`Spec.Poseidon2.absorbBlock`). Transcript framing is the parent's data. The
child has no assertion, so honest completion needs no premise. -/

namespace NightstreamFPrime.Lifecycle.Nebula.Sponge

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2

abbrev EState := Layer.EState

structure Interface where
  initial : Nat → EState
  chunks : Nat → List (List Expr)

def program (i : Interface) (offset : Nat) : Hash.AbsorbProgram :=
  Hash.compileAbsorptions offset (i.initial offset) (i.chunks offset)

/-- The final state, in closed form: the last permutation's fresh outputs, or
the initial state when there is no chunk (`output_eq_program`). Parents read it
without compiling the recipes. -/
def output (i : Interface) (offset : Nat) : EState :=
  match i.chunks offset with
  | [] => i.initial offset
  | chunks => Permutation.freshState (offset + (chunks.length - 1) * 1096 + 1080)

def opsAt (i : Interface) (offset : Nat) : List Op :=
  [Op.witness (WitnessBatch.arithmetic offset (program i offset).recipes)]

def main (i : Interface) : Circuit Unit := fun offset =>
  ((), offset + (i.chunks offset).length * 1096, opsAt i offset)

/-- The final state of a nonempty absorption is its last permutation's fresh
outputs. -/
theorem compileAbsorptions_output (chunks : List (List Expr)) (nonempty : chunks ≠ []) :
    ∀ (start : Nat) (state : EState), (Hash.compileAbsorptions start state chunks).output =
      Permutation.freshState (start + (chunks.length - 1) * 1096 + 1080) := by
  induction chunks with
  | nil => exact absurd rfl nonempty
  | cons b rest ih =>
    intro start state
    rw [Hash.compileAbsorptions]
    dsimp only
    cases rest with
    | nil =>
      rw [Hash.compileAbsorptions]
      dsimp only
      rw [← Permutation.scheduleOutput_eq_compile, Permutation.scheduleOutput]
      simp
    | cons c rest =>
      rw [ih (List.cons_ne_nil _ _)]
      congr 1
      simp only [List.length_cons]
      omega

theorem output_eq_program (i : Interface) (offset : Nat) : output i offset = (program i offset).output := by
  unfold output program
  cases h : i.chunks offset with
  | nil => rfl
  | cons b rest => exact (compileAbsorptions_output (b :: rest) (List.cons_ne_nil _ _) _ _).symm

def Assumptions (i : Interface) (offset : Nat) (_env : Env) : Prop :=
  (∀ lane, (i.initial offset lane).VarsBelow offset) ∧ Hash.BlocksBelow offset (i.chunks offset)

/-- A symbolic state as a Poseidon2 state. -/
def evalState (env : Env) (state : EState) : Spec.Poseidon2.State :=
  List.ofFn (Layer.evalState env state)

def SpecHolds (i : Interface) (offset : Nat) (env : Env) : Prop :=
  evalState env (output i offset) =
    ((i.chunks offset).map (Hash.evalList env)).foldl Spec.Poseidon2.absorbBlock
      (evalState env (i.initial offset))

@[simp] theorem main_ops (i : Interface) (offset : Nat) :
    Circuit.ops (main i) offset = opsAt i offset := rfl

theorem localLength_ops (i : Interface) (offset : Nat) :
    localLength (Circuit.ops (main i) offset) = (program i offset).recipes.length := by
  simp [opsAt, localLength, Op.localLength]

theorem localLength_eq (i : Interface) (offset : Nat) :
    localLength (Circuit.ops (main i) offset) = (i.chunks offset).length * 1096 := by
  rw [localLength_ops, program, Hash.compileAbsorptions_recipes_length]

theorem flatConstraints_eq (i : Interface) (offset : Nat) :
    flatConstraints (Circuit.ops (main i) offset) =
      recipeConstraints offset (program i offset).recipes := by
  simp [flatConstraints, opsAt, Op.flatConstraints]

theorem rowCount_eq (i : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (main i) offset)).length = (i.chunks offset).length * 1096 := by
  rw [flatConstraints_eq, recipeConstraints_length, program, Hash.compileAbsorptions_recipes_length]

theorem soundness (i : Interface) (env : Env) (offset : Nat)
    (_assumptions : Assumptions i offset env) (rows : holds env (Circuit.ops (main i) offset)) :
    SpecHolds i offset env := by
  have recipeRows : ConstraintsHold env (recipeConstraints offset (program i offset).recipes) :=
    rows (Op.witness (WitnessBatch.arithmetic offset (program i offset).recipes)) (by simp [opsAt])
  have computed := Hash.compileAbsorptions_sound env offset (i.initial offset) (i.chunks offset)
    recipeRows
  unfold SpecHolds evalState
  rw [output_eq_program, program, computed, Hash.absorbManyF_eq_reference]

theorem completeness (i : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions i offset env) :
    ∃ completed,
      AgreesOutside env completed offset (localLength (Circuit.ops (main i) offset)) ∧
      holdsFlat completed (Circuit.ops (main i) offset) := by
  have causal := Hash.compileAbsorptions_causal offset (i.initial offset) (i.chunks offset)
    assumptions.1 assumptions.2
  refine ⟨executeRecipes env offset (program i offset).recipes, ?_, ?_⟩
  · rw [localLength_ops]
    exact executeRecipes_agreesOutside env offset _
  · change ConstraintsHold _ (flatConstraints (Circuit.ops (main i) offset))
    rw [flatConstraints_eq]
    exact executeRecipes_holds_recipeConstraints env offset _ causal

theorem flatConstraints_varsBelow (i : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions i offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (main i) offset),
      expression.VarsBelow (offset + localLength (Circuit.ops (main i) offset)) := by
  have causal := Hash.compileAbsorptions_causal offset (i.initial offset) (i.chunks offset)
    assumptions.1 assumptions.2
  rw [flatConstraints_eq, localLength_ops]
  exact recipeConstraints_varsBelow_of_causal offset _ causal

theorem output_varsBelow (i : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions i offset env) (lane : Fin 16) :
    (output i offset lane).VarsBelow (offset + localLength (Circuit.ops (main i) offset)) := by
  rw [localLength_ops, output_eq_program]
  exact Hash.compileAbsorptions_output_varsBelow offset (i.initial offset) (i.chunks offset)
    assumptions.1 assumptions.2 lane

/-- Every row and every output lane reads only supported inputs or the child's
own recipe variables. -/
theorem supported (i : Interface) (offset : Nat) (allowed : Nat → Prop)
    (initialSupported : ∀ lane, (i.initial offset lane).VarsSatisfy allowed)
    (chunksSupported : ∀ chunk ∈ i.chunks offset, ∀ e ∈ chunk, e.VarsSatisfy allowed)
    (localSupported : ∀ index, offset ≤ index →
      index < offset + localLength (Circuit.ops (main i) offset) → allowed index) :
    (∀ expression ∈ flatConstraints (Circuit.ops (main i) offset),
      expression.VarsSatisfy allowed) ∧ ∀ lane, (output i offset lane).VarsSatisfy allowed := by
  have targets : ∀ index, index < (program i offset).recipes.length → allowed (offset + index) :=
    fun index bound => localSupported _ (by omega) (by rw [localLength_ops]; omega)
  have compiled := Support.compileAbsorptions_supported offset (i.initial offset) (i.chunks offset)
    allowed initialSupported chunksSupported targets
  refine ⟨?_, by rw [output_eq_program]; exact compiled.2⟩
  rw [flatConstraints_eq]
  exact recipeConstraints_varsSatisfy offset _ allowed compiled.1 targets

def circuit (i : Interface) : FormalCircuit where
  main := main i
  assumptions := Assumptions i
  spec := SpecHolds i
  soundness := soundness i
  completeness := fun env offset assumptions _ => completeness i env offset assumptions

end NightstreamFPrime.Lifecycle.Nebula.Sponge
