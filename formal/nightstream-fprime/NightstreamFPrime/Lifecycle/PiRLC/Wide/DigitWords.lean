import Batteries.Data.Fin.Coding
import NightstreamFPrime.Lifecycle.PiRLC.Wide.Batch

/-! Materialize each checked digit once for the existing R1CS ring recipes.
These words are temporary: the CCS plan reconstructs them from three checked
bits. This avoids copying those three products into every convolution term. -/

namespace NightstreamFPrime.Lifecycle.PiRLC.Wide.DigitWords

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Sampling

def count : Nat := Batch.sourceCount * ringDegree

theorem count_eq : count = 918 := by rw [count, Batch.sourceCount_eq]; rfl

def recipe (samplerOffset : Nat) (index : Fin count) : Expr :=
  let decoded : Fin Batch.sourceCount × Fin ringDegree := Fin.decodeProd index
  WideReduction.Program.outputWord (Scalar.rangeOffset (Batch.sourceOffset samplerOffset decoded.1.val)) decoded.2

def recipes (samplerOffset : Nat) : List Expr := List.ofFn (recipe samplerOffset)

def operations (samplerOffset offset : Nat) : List Op :=
  [.witness (WitnessBatch.arithmetic offset (recipes samplerOffset))]

def outputWord (offset : Nat) (source : Fin Batch.sourceCount) (position : Fin ringDegree) : Expr :=
  .var (offset + (Fin.encodeProd (source, position)).val)

def Assumptions (samplerOffset offset : Nat) : Prop := samplerOffset + Batch.privateCount ≤ offset

def SpecHolds (samplerOffset offset : Nat) (env : Env) : Prop :=
  ∀ index : Fin count, env (offset + index.val) = (recipe samplerOffset index).eval env

theorem recipes_length (samplerOffset : Nat) : (recipes samplerOffset).length = count := List.length_ofFn

theorem recipe_below (samplerOffset : Nat) (index : Fin count) :
    (recipe samplerOffset index).VarsBelow (samplerOffset + Batch.privateCount) := by
  let decoded : Fin Batch.sourceCount × Fin ringDegree := Fin.decodeProd index
  have bound := Batch.outputChallenge_below samplerOffset decoded.1 decoded.2
  exact bound.1

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
  simpa only [List.get_ofFn] using value

private theorem complete_ofFn {n : Nat} (source : Fin n → Expr) (offset : Nat) (env : Env)
    (below : ∀ index, (source index).VarsBelow offset) :
    ∃ completed, AgreesOutside env completed offset n ∧
      holdsFlat completed [.witness (WitnessBatch.arithmetic offset (List.ofFn source))] := by
  refine ⟨executeRecipes env offset (List.ofFn source), ?_, ?_⟩
  · simpa only [List.length_ofFn] using executeRecipes_agreesOutside env offset (List.ofFn source)
  · change ConstraintsHold _ (recipeConstraints offset (List.ofFn source) ++ [])
    rw [List.append_nil]
    exact executeRecipes_holds_recipeConstraints env offset _ (causal_ofFn source offset below)

theorem localLength_eq (samplerOffset offset : Nat) : localLength (operations samplerOffset offset) = count := by
  simp [operations, localLength, Op.localLength, WitnessBatch.arithmetic, WitnessBatch.outputLength, recipes_length]

theorem constraints_eq (samplerOffset offset : Nat) :
    flatConstraints (operations samplerOffset offset) = recipeConstraints offset (recipes samplerOffset) := by
  simp [operations, flatConstraints, Op.flatConstraints, WitnessBatch.arithmetic]

theorem rowCount_eq (samplerOffset offset : Nat) :
    (flatConstraints (operations samplerOffset offset)).length = count := by
  rw [constraints_eq, recipeConstraints_length, recipes_length]

theorem scope (samplerOffset offset : Nat) (inputs : Assumptions samplerOffset offset) :
    ∀ expression ∈ flatConstraints (operations samplerOffset offset), expression.VarsBelow (offset + count) :=
  scope_ofFn (recipe samplerOffset) offset
    (fun index => Expr.VarsBelow.mono _ (recipe_below samplerOffset index) inputs)

theorem soundness (samplerOffset offset : Nat) (env : Env) (rows : holds env (operations samplerOffset offset)) :
    SpecHolds samplerOffset offset env := sound_ofFn (recipe samplerOffset) offset env rows

theorem outputWord_eq (samplerOffset offset : Nat) (env : Env) (spec : SpecHolds samplerOffset offset env)
    (source : Fin Batch.sourceCount) (position : Fin ringDegree) :
    (outputWord offset source position).eval env =
      (WideReduction.Program.outputWord (Scalar.rangeOffset (Batch.sourceOffset samplerOffset source.val)) position).eval env := by
  have value := spec (Fin.encodeProd (source, position))
  simpa only [outputWord, Expr.eval_var, recipe, Fin.decodeProd_encodeProd] using value

theorem complete (samplerOffset offset : Nat) (env : Env) (inputs : Assumptions samplerOffset offset) :
    ∃ completed, AgreesOutside env completed offset count ∧ holdsFlat completed (operations samplerOffset offset) :=
  complete_ofFn (recipe samplerOffset) offset env
    (fun index => Expr.VarsBelow.mono _ (recipe_below samplerOffset index) inputs)

def circuit (samplerOffset : Nat) : FormalCircuit where
  main := fun offset => ((), offset + count, operations samplerOffset offset)
  assumptions := fun offset _ => Assumptions samplerOffset offset
  spec := SpecHolds samplerOffset
  privateCount := fun _ => count
  rowCount := fun _ => count
  privateCount_eq := localLength_eq samplerOffset
  rowCount_eq := rowCount_eq samplerOffset
  soundness := fun env offset _ rows => soundness samplerOffset offset env rows
  completeness := fun env offset inputs _ => by
    rw [show localLength (Circuit.ops (fun start => ((), start + count, operations samplerOffset start)) offset) = count from
      localLength_eq samplerOffset offset]
    exact complete samplerOffset offset env inputs

theorem circuit_ops (samplerOffset offset : Nat) :
    Circuit.ops (circuit samplerOffset).main offset = operations samplerOffset offset := rfl

end NightstreamFPrime.Lifecycle.PiRLC.Wide.DigitWords
