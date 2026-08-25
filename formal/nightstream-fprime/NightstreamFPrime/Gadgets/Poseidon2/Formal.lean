import NightstreamFPrime.Gadgets.Poseidon2.Hash

/-!
Owns the one proof-carrying Poseidon2 hash circuit. The interface supplies
external input and digest expressions for each call offset. The child exports
only its hash specification; its recipes and assertions remain opaque to
parents.
-/

namespace NightstreamFPrime.Gadgets.Poseidon2.Formal

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit

/-- External expressions selected by the parent at one call offset. -/
structure Interface where
  input : Nat → List Expr
  expected : Nat → Fin 4 → Expr

def Assumptions (interface : Interface) (offset : Nat) (_env : Env) : Prop :=
  (∀ expression ∈ interface.input offset, expression.VarsBelow offset) ∧
    ∀ lane, (interface.expected offset lane).VarsBelow offset

def SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  List.ofFn (fun lane => (interface.expected offset lane).eval env) =
    Spec.Poseidon2.hash (Hash.evalList env (interface.input offset))

def assertions (interface : Interface) (offset : Nat) : List Op :=
  let program := Hash.compile offset (interface.input offset)
  List.ofFn fun lane => Op.assertZero
    (Hash.digestE program.output lane - interface.expected offset lane)

def opsAt (interface : Interface) (offset : Nat) : List Op :=
  let program := Hash.compile offset (interface.input offset)
  Op.witness (WitnessBatch.arithmetic offset program.recipes) ::
    assertions interface offset

def main (interface : Interface) : Circuit Unit := fun offset =>
  let program := Hash.compile offset (interface.input offset)
  ((), offset + program.recipes.length, opsAt interface offset)

@[simp] theorem main_ops (interface : Interface) (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := by
  rfl

theorem assertions_localLength (interface : Interface) (offset : Nat) :
    localLength (assertions interface offset) = 0 := by
  change (List.ofFn (fun _ : Fin 4 => 0)).sum = 0
  simp

theorem opsAt_localLength (interface : Interface) (offset : Nat) :
    localLength (opsAt interface offset) =
      (Hash.compile offset (interface.input offset)).recipes.length := by
  unfold opsAt
  change (Hash.compile offset (interface.input offset)).recipes.length +
    localLength (assertions interface offset) = _
  rw [assertions_localLength]
  omega

theorem assertion_mem (interface : Interface) (offset : Nat) (lane : Fin 4) :
    Op.assertZero
      (Hash.digestE (Hash.compile offset (interface.input offset)).output lane -
        interface.expected offset lane) ∈ assertions interface offset := by
  rw [assertions, List.mem_ofFn']
  exact Set.mem_range_self lane

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (_assumptions : Assumptions interface offset env)
    (hholds : holds env (Circuit.ops (main interface) offset)) :
    SpecHolds interface offset env := by
  let program := Hash.compile offset (interface.input offset)
  have recipeRows : ConstraintsHold env
      (recipeConstraints offset program.recipes) := by
    have witnessHolds := hholds
      (Op.witness (WitnessBatch.arithmetic offset program.recipes)) (by
        simp [main_ops, opsAt, program])
    exact witnessHolds
  have computed := Hash.compile_sound env offset (interface.input offset)
    recipeRows
  have laneEquation (lane : Fin 4) :
      (interface.expected offset lane).eval env =
        (Hash.digestE program.output lane).eval env := by
    have assertionHolds := hholds
      (Op.assertZero (Hash.digestE program.output lane -
        interface.expected offset lane)) (by
          simp only [main_ops, opsAt, List.mem_cons]
          exact Or.inr (by simpa [program] using assertion_mem interface offset lane))
    change (Hash.digestE program.output lane -
      interface.expected offset lane).eval env = 0 at assertionHolds
    exact (sub_eq_zero.mp (by
      simpa only [Expr.eval_sub] using assertionHolds)).symm
  unfold SpecHolds
  calc
    List.ofFn (fun lane => (interface.expected offset lane).eval env) =
        List.ofFn (fun lane =>
          (Hash.digestE program.output lane).eval env) := by
      congr 1
      funext lane
      exact laneEquation lane
    _ = Spec.Poseidon2.hash (Hash.evalList env (interface.input offset)) :=
      computed

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  let program := Hash.compile offset (interface.input offset)
  let completed := executeRecipes env offset program.recipes
  have causal : RecipesCausal offset program.recipes :=
    Hash.compile_causal offset (interface.input offset) assumptions.1
  have recipeRows : ConstraintsHold completed
      (recipeConstraints offset program.recipes) :=
    executeRecipes_holds_recipeConstraints env offset program.recipes causal
  have agreesBelow : ∀ index, index < offset → completed index = env index :=
    executeRecipes_agrees_below env offset program.recipes
  have inputEval : Hash.evalList completed (interface.input offset) =
      Hash.evalList env (interface.input offset) := by
    unfold Hash.evalList
    apply List.map_congr_left
    intro expression member
    exact expression.eval_eq_of_agree_below offset completed env
      (assumptions.1 expression member) agreesBelow
  have expectedEval : (fun lane =>
      (interface.expected offset lane).eval completed) =
      fun lane => (interface.expected offset lane).eval env := by
    funext lane
    exact (interface.expected offset lane).eval_eq_of_agree_below offset
      completed env (assumptions.2 lane) agreesBelow
  have computed := Hash.compile_sound completed offset (interface.input offset)
    recipeRows
  have digestLists :
      List.ofFn (fun lane => (Hash.digestE program.output lane).eval completed) =
        List.ofFn (fun lane =>
          (interface.expected offset lane).eval completed) := by
    calc
      List.ofFn (fun lane => (Hash.digestE program.output lane).eval completed) =
          Spec.Poseidon2.hash
            (Hash.evalList completed (interface.input offset)) := computed
      _ = Spec.Poseidon2.hash
            (Hash.evalList env (interface.input offset)) := by rw [inputEval]
      _ = List.ofFn (fun lane =>
            (interface.expected offset lane).eval env) := specification.symm
      _ = List.ofFn (fun lane =>
            (interface.expected offset lane).eval completed) := by
          rw [expectedEval]
  have digestEquation (lane : Fin 4) :
      (Hash.digestE program.output lane).eval completed =
        (interface.expected offset lane).eval completed := by
    have selected := congrArg (fun values : List F => values.getD lane.val 0)
      digestLists
    fin_cases lane <;>
      simpa [Hash.digestE, Hash.digestF, Layer.evalState, List.ofFn_succ] using
        selected
  refine ⟨completed, ?_, ?_⟩
  · simpa [main_ops, opsAt_localLength, program, completed] using
      executeRecipes_agreesOutside env offset program.recipes
  · change ConstraintsHold completed
      (flatConstraints (opsAt interface offset))
    intro expression member
    simp only [flatConstraints, List.mem_flatMap] at member
    rcases member with ⟨operation, operationMember, constraintMember⟩
    simp only [opsAt, List.mem_cons] at operationMember
    rcases operationMember with rfl | operationMember
    · exact recipeRows expression constraintMember
    · rw [assertions, List.mem_ofFn'] at operationMember
      rcases operationMember with ⟨lane, rfl⟩
      simp only [Op.flatConstraints, List.mem_singleton] at constraintMember
      subst expression
      change (Hash.digestE program.output lane -
        interface.expected offset lane).eval completed = 0
      simp only [Expr.eval_sub]
      exact sub_eq_zero.mpr (digestEquation lane)

/-- The only proof-carrying Poseidon2 hash circuit exported to parents. -/
def circuit (interface : Interface) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  soundness := soundness interface
  completeness := completeness interface

end NightstreamFPrime.Gadgets.Poseidon2.Formal
