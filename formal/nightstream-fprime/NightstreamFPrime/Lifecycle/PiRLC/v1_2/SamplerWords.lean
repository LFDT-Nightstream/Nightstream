import NightstreamFPrime.Gadgets.Sampling.WideReduction.Program
import NightstreamFPrime.Circuit.StraightLineSupport

/-! Materialize each checked digit once for the existing R1CS ring recipes.
The word equality is checked before a coefficient enters the combination
recipes. Each scalar has exactly 54 output words. -/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_2.SamplerWords

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Sampling

def count : Nat := ringDegree

theorem count_eq : count = 54 := rfl

def recipe (rangeOffset : Nat) (index : Fin count) : Expr :=
  WideReduction.Program.outputWord rangeOffset index

def recipes (rangeOffset : Nat) : List Expr := List.ofFn (recipe rangeOffset)

def operations (rangeOffset offset : Nat) : List Op :=
  [.witness (WitnessBatch.arithmetic offset (recipes rangeOffset))]

def outputWord (offset : Nat) (position : Fin ringDegree) : Expr :=
  .var (offset + position.val)

def Assumptions (rangeOffset offset : Nat) : Prop := rangeOffset + WideReduction.Program.privateCount ≤ offset

def SpecHolds (rangeOffset offset : Nat) (env : Env) : Prop :=
  ∀ index : Fin count, env (offset + index.val) = (recipe rangeOffset index).eval env

theorem recipes_length (rangeOffset : Nat) : (recipes rangeOffset).length = count := List.length_ofFn

theorem recipe_below (rangeOffset : Nat) (index : Fin count) :
    (recipe rangeOffset index).VarsBelow (rangeOffset + WideReduction.Program.privateCount) := by
  exact (WideReduction.Program.outputChallenge_varsBelow rangeOffset index).1

private theorem causal_ofFn {n : Nat} (source : Fin n → Expr) (offset : Nat)
    (below : ∀ index, (source index).VarsBelow offset) : RecipesCausal offset (List.ofFn source) := by
  apply recipesCausal_of_all_below
  intro expression member
  obtain ⟨index, rfl⟩ := List.mem_ofFn.mp member
  exact below index

private theorem scope_ofFn {n : Nat} (source : Fin n → Expr) (offset : Nat)
    (below : ∀ index, (source index).VarsBelow offset) :
    ∀ expression ∈ flatConstraints [.witness (WitnessBatch.arithmetic offset (List.ofFn source))],
      expression.VarsBelow (offset + n) := by
  simpa only [flatConstraints, List.flatMap_cons, List.flatMap_nil, List.append_nil,
    Op.flatConstraints, WitnessBatch.arithmetic, List.length_ofFn] using
      recipeConstraints_varsBelow_of_causal offset (List.ofFn source) (causal_ofFn source offset below)

private theorem sound_ofFn {n : Nat} (source : Fin n → Expr) (offset : Nat) (env : Env)
    (rows : holds env [.witness (WitnessBatch.arithmetic offset (List.ofFn source))]) :
    ∀ index : Fin n, env (offset + index.val) = (source index).eval env := by
  have checked := rows (.witness (WitnessBatch.arithmetic offset (List.ofFn source))) (by simp)
  change ConstraintsHold env (recipeConstraints offset (List.ofFn source)) at checked
  intro index
  have value := recipeConstraints_value env offset (List.ofFn source) checked index.val
    (by rw [List.length_ofFn]; exact index.isLt)
  simpa only [List.get_ofFn] using! value

private theorem complete_ofFn {n : Nat} (source : Fin n → Expr) (offset : Nat) (env : Env)
    (below : ∀ index, (source index).VarsBelow offset) :
    ∃ completed, AgreesOutside env completed offset n ∧
      holdsFlat completed [.witness (WitnessBatch.arithmetic offset (List.ofFn source))] := by
  refine ⟨executeRecipes env offset (List.ofFn source), ?_, ?_⟩
  · simpa only [List.length_ofFn] using executeRecipes_agreesOutside env offset (List.ofFn source)
  · change ConstraintsHold _ (recipeConstraints offset (List.ofFn source) ++ [])
    rw [List.append_nil]
    exact executeRecipes_holds_recipeConstraints env offset _ (causal_ofFn source offset below)

theorem localLength_eq (rangeOffset offset : Nat) : localLength (operations rangeOffset offset) = count := by
  simp [operations, localLength, Op.localLength, WitnessBatch.arithmetic, WitnessBatch.outputLength, recipes_length]

theorem constraints_eq (rangeOffset offset : Nat) :
    flatConstraints (operations rangeOffset offset) = recipeConstraints offset (recipes rangeOffset) := by
  simp [operations, flatConstraints, Op.flatConstraints, WitnessBatch.arithmetic]

theorem flatConstraints_varsSatisfy (rangeOffset offset : Nat) (allowed : Nat → Prop)
    (locals : ∀ index, WideReduction.Program.coreOffset rangeOffset ≤ index →
      index < WideReduction.Program.coreOffset rangeOffset + WideReduction.privateCount → allowed index)
    (words : ∀ index, index < count → allowed (offset + index)) :
    ∀ expression ∈ flatConstraints (operations rangeOffset offset), expression.VarsSatisfy allowed := by
  rw [constraints_eq]
  apply recipeConstraints_varsSatisfy
  · intro expression member
    obtain ⟨position, rfl⟩ := List.mem_ofFn.mp member
    exact WideReduction.Program.outputWord_varsSatisfy rangeOffset position allowed locals
  · simpa only [recipes_length] using words

theorem rowCount_eq (rangeOffset offset : Nat) :
    (flatConstraints (operations rangeOffset offset)).length = count := by
  rw [constraints_eq, recipeConstraints_length, recipes_length]

theorem scope (rangeOffset offset : Nat) (inputs : Assumptions rangeOffset offset) :
    ∀ expression ∈ flatConstraints (operations rangeOffset offset), expression.VarsBelow (offset + count) :=
  scope_ofFn (recipe rangeOffset) offset
    (fun index => Expr.VarsBelow.mono _ (recipe_below rangeOffset index) inputs)

theorem soundness (rangeOffset offset : Nat) (env : Env) (rows : holds env (operations rangeOffset offset)) :
    SpecHolds rangeOffset offset env := sound_ofFn (recipe rangeOffset) offset env rows

theorem outputWord_eq (rangeOffset offset : Nat) (env : Env) (spec : SpecHolds rangeOffset offset env)
    (position : Fin ringDegree) :
    (outputWord offset position).eval env =
      (WideReduction.Program.outputWord rangeOffset position).eval env :=
  spec position

theorem complete (rangeOffset offset : Nat) (env : Env) (inputs : Assumptions rangeOffset offset) :
    ∃ completed, AgreesOutside env completed offset count ∧ holdsFlat completed (operations rangeOffset offset) :=
  complete_ofFn (recipe rangeOffset) offset env
    (fun index => Expr.VarsBelow.mono _ (recipe_below rangeOffset index) inputs)

def circuit (rangeOffset : Nat) : FormalCircuit where
  main := fun offset => ((), offset + count, operations rangeOffset offset)
  assumptions := fun offset _ => Assumptions rangeOffset offset
  spec := SpecHolds rangeOffset
  privateCount := fun _ => count
  rowCount := fun _ => count
  privateCount_eq := localLength_eq rangeOffset
  rowCount_eq := rowCount_eq rangeOffset
  soundness := fun env offset _ rows => soundness rangeOffset offset env rows
  completeness := fun env offset inputs _ => by
    rw [show localLength (Circuit.ops (fun start => ((), start + count, operations rangeOffset start)) offset) = count from
      localLength_eq rangeOffset offset]
    exact complete rangeOffset offset env inputs

theorem circuit_ops (rangeOffset offset : Nat) :
    Circuit.ops (circuit rangeOffset).main offset = operations rangeOffset offset := rfl

end NightstreamFPrime.Lifecycle.PiRLC.v1_2.SamplerWords
