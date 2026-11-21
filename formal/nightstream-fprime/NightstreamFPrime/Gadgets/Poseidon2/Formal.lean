import NightstreamFPrime.Gadgets.Poseidon2.LiveRows
import NightstreamFPrime.Gadgets.Poseidon2.RawFormal
import NightstreamFPrime.Circuit.Execution

namespace NightstreamFPrime.Gadgets.Poseidon2.Formal

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit

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
    Op.mark (Hash.marks offset (interface.input offset)) :: assertions interface offset

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
    (0 + localLength (assertions interface offset)) = _
  rw [assertions_localLength]
  omega

theorem assertions_rowCount (interface : Interface) (offset : Nat) :
    rowCount (assertions interface offset) = 4 := by
  change (List.ofFn (fun _ : Fin 4 => 1)).sum = 4
  decide

theorem flatConstraints_length_eq (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (main interface) offset)).length =
      (Hash.compile offset (interface.input offset)).recipes.length + 4 := by
  rw [flatConstraints_length_eq_rowCount, main_ops]
  change (Hash.compile offset (interface.input offset)).recipes.length +
      (0 + rowCount (assertions interface offset)) = _
  rw [assertions_rowCount]

theorem assertion_mem (interface : Interface) (offset : Nat) (lane : Fin 4) :
    Op.assertZero
      (Hash.digestE (Hash.compile offset (interface.input offset)).output lane -
        interface.expected offset lane) ∈ assertions interface offset := by
  rw [assertions, List.mem_ofFn']
  exact Set.mem_range_self lane

theorem flatConstraints_varsBelow
    (interface : Interface) (offset : Nat) {env : Env}
    (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (main interface) offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (main interface) offset)) := by
  rw [main_ops, opsAt_localLength]
  intro expression member
  simp only [flatConstraints, List.mem_flatMap] at member
  rcases member with ⟨operation, operationMember, constraintMember⟩
  simp only [opsAt, List.mem_cons] at operationMember
  rcases operationMember with rfl | rfl | operationMember
  · exact RawFormal.recipeConstraints_varsBelow offset
      (Hash.compile offset (interface.input offset)).recipes
      (Hash.compile_causal offset (interface.input offset) assumptions.1)
      expression constraintMember
  · simp [Op.flatConstraints] at constraintMember
  · rw [assertions, List.mem_ofFn'] at operationMember
    rcases operationMember with ⟨lane, rfl⟩
    simp only [Op.flatConstraints, List.mem_singleton] at constraintMember
    subst expression
    apply Expr.VarsBelow.sub
    · exact Hash.compile_output_varsBelow offset (interface.input offset)
        assumptions.1 ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩
    · exact Expr.VarsBelow.mono _ (assumptions.2 lane) (by omega)

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

theorem witnesses_opsAt (interface : Interface) (offset : Nat) :
    witnesses (opsAt interface offset) =
      [WitnessBatch.arithmetic offset (Hash.compile offset (interface.input offset)).recipes] :=
  rfl

theorem ordered (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    Witness.Ordered offset (witnesses (opsAt interface offset))
      (offset + localLength (opsAt interface offset)) :=
  Witness.ordered_batch (witnesses_opsAt interface offset) (opsAt_localLength interface offset)
    (Hash.compile_causal offset (interface.input offset) assumptions.1)

theorem computed_holdsFlat (interface : Interface) (env : Env) (offset : Nat)
    (specification : SpecHolds interface offset env)
    (computed : Witness.Computed env (witnesses (opsAt interface offset))) :
    holdsFlat env (opsAt interface offset) := by
  let program := Hash.compile offset (interface.input offset)
  have recipeRows : ConstraintsHold env (recipeConstraints offset program.recipes) :=
    Witness.computed_batch (witnesses_opsAt interface offset) computed
  have digestLists :
      List.ofFn (fun lane => (Hash.digestE program.output lane).eval env) =
        List.ofFn (fun lane => (interface.expected offset lane).eval env) :=
    (Hash.compile_sound env offset (interface.input offset) recipeRows).trans
      specification.symm
  have digestEquation (lane : Fin 4) :
      (Hash.digestE program.output lane).eval env =
        (interface.expected offset lane).eval env := by
    have selected := congrArg (fun values : List F => values.getD lane.val 0)
      digestLists
    fin_cases lane <;>
      simpa [Hash.digestE, Hash.digestF, Layer.evalState, List.ofFn_succ] using
        selected
  intro expression member
  simp only [flatConstraints, List.mem_flatMap] at member
  rcases member with ⟨operation, operationMember, constraintMember⟩
  simp only [opsAt, List.mem_cons] at operationMember
  rcases operationMember with rfl | rfl | operationMember
  · exact recipeRows expression constraintMember
  · simp [Op.flatConstraints] at constraintMember
  · rw [assertions, List.mem_ofFn'] at operationMember
    rcases operationMember with ⟨lane, rfl⟩
    simp only [Op.flatConstraints, List.mem_singleton] at constraintMember
    subst expression
    change (Hash.digestE program.output lane -
      interface.expected offset lane).eval env = 0
    simp only [Expr.eval_sub]
    exact sub_eq_zero.mpr (digestEquation lane)

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  refine Witness.complete (ordered interface offset env assumptions)
    fun completed below computed => computed_holdsFlat interface completed offset ?_ computed
  have inputEval : Hash.evalList completed (interface.input offset) =
      Hash.evalList env (interface.input offset) := by
    unfold Hash.evalList
    apply List.map_congr_left
    intro expression member
    exact expression.eval_eq_of_agree_below offset completed env
      (assumptions.1 expression member) below
  have expectedEval : (fun lane =>
      (interface.expected offset lane).eval completed) =
      fun lane => (interface.expected offset lane).eval env := by
    funext lane
    exact (interface.expected offset lane).eval_eq_of_agree_below offset
      completed env (assumptions.2 lane) below
  unfold SpecHolds
  rw [inputEval, expectedEval]
  exact specification

def circuit (interface : Interface) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  privateCount := fun offset =>
    (Hash.compile offset (interface.input offset)).recipes.length
  rowCount := fun offset =>
    (Hash.compile offset (interface.input offset)).recipes.length + 4
  privateCount_eq := by
    intro offset
    rw [main_ops, opsAt_localLength]
  rowCount_eq := flatConstraints_length_eq interface
  soundness := soundness interface
  completeness := completeness interface

/-! ### The exporter's blocks (SPEC.md section 4.4) -/

theorem assertions_rowsOnly (interface : Interface) (offset : Nat) :
    RowsOnly (assertions interface offset) :=
  RowsOnly.ofFn _

theorem opsAt_split (interface : Interface) (offset : Nat) :
    opsAt interface offset =
      [Op.witness (WitnessBatch.arithmetic offset (Hash.compile offset (interface.input offset)).recipes),
        Op.mark (Hash.marks offset (interface.input offset))] ++ assertions interface offset :=
  rfl

theorem placed (interface : Interface) (offset : Nat) :
    Placed offset (localLength (Circuit.ops (main interface) offset))
      (Circuit.ops (main interface) offset) := by
  rw [main_ops, opsAt_localLength, opsAt_split, ← Nat.add_zero (List.length _)]
  exact (hash_placed offset _).append
    (.of_marks_nil (assertions_rowsOnly interface offset).marks_nil)

/-- The hash's rows are its permutations' rows and its digest rows, which read its digest and its
expected words. -/
theorem liveRows (interface : Interface) (offset : Nat) {live : Nat → Prop}
    (inputAffine : ∀ word ∈ interface.input offset, word.Affine)
    (inputReads : ∀ word ∈ interface.input offset,
      word.VarsSatisfy fun column => column < offset ∧ live column)
    (expectedReads : ∀ lane, (interface.expected offset lane).VarsSatisfy live)
    (own : OwnLive live offset (localLength (Circuit.ops (main interface) offset))
      (Circuit.ops (main interface) offset)) :
    LiveRows live (Circuit.ops (main interface) offset) := by
  have hashOwn : HashOwnLive live offset (interface.input offset) := by
    rw [main_ops, opsAt_localLength, opsAt_split] at own
    refine own.mono ?_
    simp [(assertions_rowsOnly interface offset).marks_nil]
  rw [main_ops, opsAt_split]
  refine (hash_liveRows inputAffine inputReads hashOwn).append
    (.of_rows (assertions_rowsOnly interface offset) fun row member => ?_)
  rw [assertions, List.mem_ofFn'] at member
  obtain ⟨lane, equal⟩ := member
  cases equal
  exact Expr.VarsSatisfy.sub _ _ _ (hash_digest_live hashOwn lane) (expectedReads lane)

end NightstreamFPrime.Gadgets.Poseidon2.Formal
