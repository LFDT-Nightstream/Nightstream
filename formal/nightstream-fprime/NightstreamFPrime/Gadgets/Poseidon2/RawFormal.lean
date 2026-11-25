import NightstreamFPrime.Gadgets.Poseidon2.LiveRows
import NightstreamFPrime.Gadgets.Poseidon2.Hash
import NightstreamFPrime.Circuit.Execution

namespace NightstreamFPrime.Gadgets.Poseidon2.RawFormal

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit

structure Interface where
  input : Nat → List Expr

def program (interface : Interface) (offset : Nat) : Hash.Program :=
  Hash.compile offset (interface.input offset)

def digest (interface : Interface) (offset : Nat) (lane : Fin 4) : Expr :=
  Hash.digestE (program interface offset).output lane

def operations (interface : Interface) (offset : Nat) : List Op :=
  [.witness (WitnessBatch.arithmetic offset (program interface offset).recipes),
    .mark (Hash.marks offset (interface.input offset))]

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), offset + (program interface offset).recipes.length,
    operations interface offset)

@[simp] theorem main_ops (interface : Interface) (offset : Nat) :
    Circuit.ops (main interface) offset = operations interface offset := by
  rfl

def Assumptions (interface : Interface) (offset : Nat) (_env : Env) : Prop :=
  ∀ expression ∈ interface.input offset, expression.VarsBelow offset

def SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  List.ofFn (fun lane => (digest interface offset lane).eval env) =
    Spec.Poseidon2.hash (Hash.evalList env (interface.input offset))

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (main interface) offset) =
      (program interface offset).recipes.length := by
  rw [main_ops]
  change (WitnessBatch.arithmetic offset
    (program interface offset).recipes).outputLength = _
  exact WitnessBatch.arithmetic_outputLength _ _

theorem flatConstraints_eq (interface : Interface) (offset : Nat) :
    flatConstraints (Circuit.ops (main interface) offset) =
      recipeConstraints offset (program interface offset).recipes := by
  rw [main_ops]
  simp [operations, flatConstraints, Op.flatConstraints]

theorem flatConstraints_length_eq (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (main interface) offset)).length =
      (program interface offset).recipes.length := by
  rw [flatConstraints_eq, recipeConstraints_length]

theorem recipeConstraints_varsBelow
    (start : Nat) (recipes : List Expr)
    (causal : RecipesCausal start recipes) :
    ∀ expression ∈ recipeConstraints start recipes,
      expression.VarsBelow (start + recipes.length) := by
  induction recipes generalizing start with
  | nil =>
      intro expression member
      cases member
  | cons recipe rest inductionHypothesis =>
      intro expression member
      simp only [recipeConstraints, List.mem_cons] at member
      rcases member with rfl | member
      · apply Expr.VarsBelow.sub
        · simp [Expr.VarsBelow]
        · exact Expr.VarsBelow.mono recipe causal.1 (by
            simp only [List.length_cons]
            omega)
      · have below := inductionHypothesis (start := start + 1) causal.2
          expression member
        apply Expr.VarsBelow.mono expression below
        simp only [List.length_cons]
        omega

theorem flatConstraints_varsBelow
    (interface : Interface) (offset : Nat) {env : Env}
    (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (main interface) offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (main interface) offset)) := by
  rw [localLength_eq, flatConstraints_eq]
  exact recipeConstraints_varsBelow offset (program interface offset).recipes
    (Hash.compile_causal offset (interface.input offset) assumptions)

theorem digest_varsBelow (interface : Interface) (offset : Nat) {env : Env}
    (assumptions : Assumptions interface offset env) (lane : Fin 4) :
    (digest interface offset lane).VarsBelow
      (offset + (program interface offset).recipes.length) := by
  exact Hash.compile_output_varsBelow offset (interface.input offset)
    assumptions ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (_assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    SpecHolds interface offset env := by
  have recipeRows := rows
    (.witness (WitnessBatch.arithmetic offset
      (program interface offset).recipes)) (by
        rw [main_ops]
        simp [operations])
  exact Hash.compile_sound env offset (interface.input offset) recipeRows

theorem ordered (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    Witness.Ordered offset (witnesses (operations interface offset))
      (offset + localLength (operations interface offset)) :=
  Witness.ordered_batch rfl (localLength_eq interface offset)
    (Hash.compile_causal offset (interface.input offset) assumptions)

theorem computed_holdsFlat (interface : Interface) (env : Env) (offset : Nat)
    (computed : Witness.Computed env (witnesses (operations interface offset))) :
    holdsFlat env (operations interface offset) :=
  Witness.computed_holdsFlat (by simp [operations, Op.RecipesOnly]) computed

theorem complete (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) :=
  Witness.complete (ordered interface offset env assumptions) fun completed _ =>
    computed_holdsFlat interface completed offset

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (_specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) :=
  complete interface env offset assumptions

def circuit (interface : Interface) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  privateCount := fun offset => (program interface offset).recipes.length
  rowCount := fun offset => (program interface offset).recipes.length
  privateCount_eq := localLength_eq interface
  rowCount_eq := flatConstraints_length_eq interface
  soundness := soundness interface
  completeness := completeness interface

/-! ### The exporter's blocks (SPEC.md section 4.4) -/

theorem placed (interface : Interface) (offset : Nat) :
    Placed offset (localLength (operations interface offset)) (operations interface offset) := by
  rw [show localLength (operations interface offset) =
      (Hash.compile offset (interface.input offset)).recipes.length from
    localLength_batch _ _ _]
  exact hash_placed offset _

theorem liveRows (interface : Interface) (offset : Nat) {live : Nat → Prop}
    (inputAffine : ∀ word ∈ interface.input offset, word.Affine)
    (inputReads : ∀ word ∈ interface.input offset,
      word.VarsSatisfy fun column => column < offset ∧ live column)
    (own : OwnLive live offset (localLength (operations interface offset))
      (operations interface offset)) :
    LiveRows live (operations interface offset) := by
  rw [show localLength (operations interface offset) =
      (Hash.compile offset (interface.input offset)).recipes.length from
    localLength_batch _ _ _] at own
  exact hash_liveRows inputAffine inputReads own

theorem digest_live (interface : Interface) (offset : Nat) {live : Nat → Prop}
    (own : OwnLive live offset (localLength (operations interface offset))
      (operations interface offset)) (lane : Fin 4) :
    (digest interface offset lane).VarsSatisfy live := by
  rw [show localLength (operations interface offset) =
      (Hash.compile offset (interface.input offset)).recipes.length from
    localLength_batch _ _ _] at own
  exact hash_digest_live own lane

end NightstreamFPrime.Gadgets.Poseidon2.RawFormal
