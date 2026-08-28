import NightstreamFPrime.Circuit.Quadratic
import NightstreamFPrime.Lifecycle.XOut

/-!
Owns the canonical state-word checks used by the PiCCS statement boundary.

The pilot hashes fixed-width word arrays. These rows make each array a
canonical `serializePreimage` shape and bind the four verifier-context words
in both states to one verifier-owned public value. The state data and running
values stay in their existing zero-copy columns.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.StateBinding

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- One fixed word in the canonical state serialization. -/
structure FixedWord where
  index : Nat
  value : F
deriving DecidableEq

def tagWords : List FixedWord :=
  (List.finRange stateDomainTag.length).map fun index =>
    ⟨index.val, stateDomainTag.getD index.val 0⟩

def runningGroupStart (source : Nat) : Nat :=
  40 + cubeVariables * 2 + source * 2865

def runningPrefixWords : List FixedWord :=
  (List.finRange productionShape.runningCount).flatMap fun source =>
    [⟨runningGroupStart source.val, Poseidon2.ofNat 972⟩,
      ⟨runningGroupStart source.val + 973, Poseidon2.ofNat 270⟩,
      ⟨runningGroupStart source.val + 1244, Poseidon2.ofNat 1620⟩]

/-- All fixed tag, block-length, and program-counter words. -/
def fixedWords : List FixedWord :=
  tagWords ++
    [⟨23, Poseidon2.ofNat 4⟩,
      ⟨29, Poseidon2.ofNat 4⟩,
      ⟨34, Poseidon2.ofNat 4⟩,
      ⟨39, Poseidon2.ofNat (cubeVariables * 2)⟩] ++
    runningPrefixWords ++ [⟨45932, Poseidon2.ofNat 1⟩]

def contextWordStart : Nat := 24

theorem tagWords_length : tagWords.length = 23 := by
  simp [tagWords, stateDomainTag_length]

theorem runningPrefixWords_length : runningPrefixWords.length = 48 := by
  norm_num [runningPrefixWords, productionShape, productionProfile,
    Phi81MatrixSource.phi81Shape]

theorem fixedWords_length : fixedWords.length = 76 := by
  simp [fixedWords, tagWords_length, runningPrefixWords_length]

theorem fixedWord_index_lt (word : FixedWord) (member : word ∈ fixedWords) :
    word.index < 45933 := by
  simp only [fixedWords, List.mem_append] at member
  rcases member with ((tagMember | fixedMember) | runningMember) | pcMember
  · rw [tagWords, List.mem_map] at tagMember
    rcases tagMember with ⟨index, _indexMember, rfl⟩
    have bound := index.isLt
    have tagLength := stateDomainTag_length
    change index.val < 45933
    omega
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at fixedMember
    rcases fixedMember with rfl | rfl | rfl | rfl <;> norm_num
  · rw [runningPrefixWords, List.mem_flatMap] at runningMember
    rcases runningMember with ⟨source, _sourceMember, wordMember⟩
    simp only [List.mem_cons, List.not_mem_nil, or_false] at wordMember
    have sourceBound := source.isLt
    norm_num [productionShape, productionProfile,
      Phi81MatrixSource.phi81Shape] at sourceBound
    rcases wordMember with rfl | rfl | rfl <;>
      simp only [runningGroupStart, cubeVariables] <;> omega
  · simp only [List.mem_singleton] at pcMember
    subst word
    norm_num

structure Interface where
  priorState : Nat → Nat → Expr
  outputState : Nat → Nat → Expr
  expectedContext : Nat → Fin 4 → Expr

def stateAssertions (state : Nat → Expr) : List Expr :=
  fixedWords.map fun word => state word.index - Expr.const word.value

def contextAssertions (state : Nat → Expr)
    (expected : Fin 4 → Expr) : List Expr :=
  (List.finRange 4).map fun lane =>
    state (contextWordStart + lane.val) - expected lane

def assertions (interface : Interface) (offset : Nat) : List Expr :=
  stateAssertions (interface.priorState offset) ++ (
    stateAssertions (interface.outputState offset) ++
      contextAssertions (interface.priorState offset)
        (interface.expectedContext offset) ++
      contextAssertions (interface.outputState offset)
        (interface.expectedContext offset))

def StateCanonical (state : Nat → Expr) (env : Env) : Prop :=
  ∀ word ∈ fixedWords, (state word.index).eval env = word.value

def ContextPreserved (prior output : Nat → Expr) (env : Env) : Prop :=
  ∀ lane : Fin 4,
    (output (contextWordStart + lane.val)).eval env =
      (prior (contextWordStart + lane.val)).eval env

def ContextBound (state : Nat → Expr) (expected : Fin 4 → Expr)
    (env : Env) : Prop :=
  ∀ lane : Fin 4,
    (state (contextWordStart + lane.val)).eval env =
      (expected lane).eval env

structure SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop where
  priorCanonical : StateCanonical (interface.priorState offset) env
  outputCanonical : StateCanonical (interface.outputState offset) env
  priorContext : ContextBound (interface.priorState offset)
    (interface.expectedContext offset) env
  outputContext : ContextBound (interface.outputState offset)
    (interface.expectedContext offset) env

theorem SpecHolds.contextPreserved
    {interface : Interface} {offset : Nat} {env : Env}
    (specification : SpecHolds interface offset env) :
    ContextPreserved (interface.priorState offset)
      (interface.outputState offset) env := by
  intro lane
  rw [specification.outputContext lane, specification.priorContext lane]

def opsAt (interface : Interface) (offset : Nat) : List Op :=
  (assertions interface offset).map Op.assertZero

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), offset, opsAt interface offset)

@[simp] theorem main_ops (interface : Interface) (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := by
  rfl

private theorem assertion_holds_iff (left right : Expr) (env : Env) :
    (left - right).eval env = 0 ↔ left.eval env = right.eval env := by
  constructor
  · intro row
    exact sub_eq_zero.mp (by simpa using row)
  · intro equal
    simpa using sub_eq_zero.mpr equal

private theorem flatConstraints_assertions (expressions : List Expr) :
    flatConstraints (expressions.map Op.assertZero) = expressions := by
  induction expressions with
  | nil => rfl
  | cons expression rest inductionHypothesis =>
      change [expression] ++ flatConstraints (rest.map Op.assertZero) =
        expression :: rest
      rw [inductionHypothesis]
      rfl

@[simp] theorem flatConstraints_opsAt (interface : Interface) (offset : Nat) :
    flatConstraints (opsAt interface offset) = assertions interface offset := by
  exact flatConstraints_assertions _

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    SpecHolds interface offset env := by
  rw [main_ops] at rows
  have rowOfMember : ∀ expression ∈ assertions interface offset,
      expression.eval env = 0 := by
    intro expression member
    exact rows (Op.assertZero expression) (by
      rw [opsAt, List.mem_map]
      exact ⟨expression, member, rfl⟩)
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro word member
    have row := rowOfMember
      (interface.priorState offset word.index -
        Expr.const word.value) (by
          unfold assertions
          apply List.mem_append_left
          rw [stateAssertions, List.mem_map]
          exact ⟨word, member, rfl⟩)
    exact (assertion_holds_iff _ _ env).mp row
  · intro word member
    have row := rowOfMember
      (interface.outputState offset word.index -
        Expr.const word.value) (by
          unfold assertions
          apply List.mem_append_right
          apply List.mem_append_left
          apply List.mem_append_left
          rw [stateAssertions, List.mem_map]
          exact ⟨word, member, rfl⟩)
    exact (assertion_holds_iff _ _ env).mp row
  · intro lane
    have row := rowOfMember
        (interface.priorState offset
          (contextWordStart + lane.val) -
          interface.expectedContext offset lane) (by
          simp [assertions, contextAssertions])
    exact (assertion_holds_iff _ _ env).mp row
  · intro lane
    have row := rowOfMember
        (interface.outputState offset
          (contextWordStart + lane.val) -
          interface.expectedContext offset lane) (by
          simp [assertions, contextAssertions])
    exact (assertion_holds_iff _ _ env).mp row

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  refine ⟨env, ?_, ?_⟩
  · intro _ _
    rfl
  · rw [main_ops]
    change ConstraintsHold env (flatConstraints (opsAt interface offset))
    rw [flatConstraints_opsAt]
    intro expression sourceMember
    rw [assertions, List.mem_append] at sourceMember
    rcases sourceMember with priorMember | remainingMember
    · rw [stateAssertions, List.mem_map] at priorMember
      rcases priorMember with ⟨word, wordMember, rfl⟩
      apply (assertion_holds_iff _ _ env).mpr
      exact specification.priorCanonical word wordMember
    · rw [List.mem_append] at remainingMember
      rcases remainingMember with middleMember | outputContextMember
      · rw [List.mem_append] at middleMember
        rcases middleMember with outputMember | priorContextMember
        · rw [stateAssertions, List.mem_map] at outputMember
          rcases outputMember with ⟨word, wordMember, rfl⟩
          apply (assertion_holds_iff _ _ env).mpr
          exact specification.outputCanonical word wordMember
        · rw [contextAssertions, List.mem_map] at priorContextMember
          rcases priorContextMember with ⟨lane, _laneMember, rfl⟩
          apply (assertion_holds_iff _ _ env).mpr
          exact specification.priorContext lane
      · rw [contextAssertions, List.mem_map] at outputContextMember
        rcases outputContextMember with ⟨lane, _laneMember, rfl⟩
        apply (assertion_holds_iff _ _ env).mpr
        exact specification.outputContext lane

structure Assumptions (interface : Interface) (offset : Nat)
    (_env : Env) : Prop where
  priorFixed : ∀ word ∈ fixedWords,
    (interface.priorState offset word.index).VarsBelow offset
  outputFixed : ∀ word ∈ fixedWords,
    (interface.outputState offset word.index).VarsBelow offset
  priorContext : ∀ lane : Fin 4,
    (interface.priorState offset
      (contextWordStart + lane.val)).VarsBelow offset
  outputContext : ∀ lane : Fin 4,
    (interface.outputState offset
      (contextWordStart + lane.val)).VarsBelow offset
  expectedContext : ∀ lane : Fin 4,
    (interface.expectedContext offset lane).VarsBelow offset

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (main interface) offset),
      expression.VarsBelow offset := by
  intro expression member
  rw [main_ops, flatConstraints_opsAt] at member
  rw [assertions, List.mem_append] at member
  rcases member with priorMember | remainingMember
  · rw [stateAssertions, List.mem_map] at priorMember
    rcases priorMember with ⟨word, _wordMember, rfl⟩
    exact Expr.VarsBelow.sub _ _ _
      (assumptions.priorFixed word _wordMember) trivial
  · rw [List.mem_append] at remainingMember
    rcases remainingMember with middleMember | outputContextMember
    · rw [List.mem_append] at middleMember
      rcases middleMember with outputMember | priorContextMember
      · rw [stateAssertions, List.mem_map] at outputMember
        rcases outputMember with ⟨word, _wordMember, rfl⟩
        exact Expr.VarsBelow.sub _ _ _
          (assumptions.outputFixed word _wordMember) trivial
      · rw [contextAssertions, List.mem_map] at priorContextMember
        rcases priorContextMember with ⟨lane, _laneMember, rfl⟩
        exact Expr.VarsBelow.sub _ _ _
          (assumptions.priorContext lane) (assumptions.expectedContext lane)
    · rw [contextAssertions, List.mem_map] at outputContextMember
      rcases outputContextMember with ⟨lane, _laneMember, rfl⟩
      exact Expr.VarsBelow.sub _ _ _
        (assumptions.outputContext lane) (assumptions.expectedContext lane)

theorem specHolds_of_agree_below (interface : Interface) (offset : Nat)
    (before after : Env) (assumptions : Assumptions interface offset before)
    (agrees : ∀ index, index < offset → after index = before index)
    (specification : SpecHolds interface offset before) :
    SpecHolds interface offset after := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro word member
    calc
      (interface.priorState offset word.index).eval after =
          (interface.priorState offset word.index).eval before :=
        Expr.eval_eq_of_agree_below _ offset after before
          (assumptions.priorFixed word member) agrees
      _ = word.value := specification.priorCanonical word member
  · intro word member
    calc
      (interface.outputState offset word.index).eval after =
          (interface.outputState offset word.index).eval before :=
        Expr.eval_eq_of_agree_below _ offset after before
          (assumptions.outputFixed word member) agrees
      _ = word.value := specification.outputCanonical word member
  · intro lane
    calc
      (interface.priorState offset
          (contextWordStart + lane.val)).eval after =
          (interface.priorState offset
            (contextWordStart + lane.val)).eval before :=
        Expr.eval_eq_of_agree_below _ offset after before
          (assumptions.priorContext lane) agrees
      _ = (interface.expectedContext offset lane).eval before :=
        specification.priorContext lane
      _ = (interface.expectedContext offset lane).eval after := by
        symm
        exact Expr.eval_eq_of_agree_below _ offset after before
          (assumptions.expectedContext lane) agrees
  · intro lane
    calc
      (interface.outputState offset
          (contextWordStart + lane.val)).eval after =
          (interface.outputState offset
            (contextWordStart + lane.val)).eval before :=
        Expr.eval_eq_of_agree_below _ offset after before
          (assumptions.outputContext lane) agrees
      _ = (interface.expectedContext offset lane).eval before :=
        specification.outputContext lane
      _ = (interface.expectedContext offset lane).eval after := by
        symm
        exact Expr.eval_eq_of_agree_below _ offset after before
          (assumptions.expectedContext lane) agrees

private theorem localLength_assertions (expressions : List Expr) :
    localLength (expressions.map Op.assertZero) = 0 := by
  induction expressions with
  | nil => rfl
  | cons _ rest inductionHypothesis =>
      change 0 + localLength (rest.map Op.assertZero) = 0
      simpa using inductionHypothesis

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (main interface) offset) = 0 := by
  rw [main_ops, opsAt, localLength_assertions]

/-- The semantic state predicate satisfies the exact direct constraint list
without allocating or changing any value. -/
theorem constraintsHold_of_spec (interface : Interface) (env : Env)
    (offset : Nat) (specification : SpecHolds interface offset env) :
    ConstraintsHold env
      (flatConstraints (Circuit.ops (main interface) offset)) := by
  rcases completeness interface env offset specification with
    ⟨completed, agrees, rows⟩
  have completedEq : completed = env := by
    funext index
    apply agrees index
    rw [localLength_eq]
    omega
  simpa only [completedEq] using rows

theorem operations_length (interface : Interface) (offset : Nat) :
    (Circuit.ops (main interface) offset).length = 160 := by
  rw [main_ops]
  simp [opsAt, assertions, stateAssertions, contextAssertions,
    fixedWords_length]

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (main interface) offset)).length = 160 := by
  rw [main_ops, flatConstraints_opsAt]
  simp [assertions, stateAssertions, contextAssertions, fixedWords_length]

/-- The sole logical circuit for canonical state binding. -/
def circuit (interface : Interface) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  soundness := by
    intro env offset _assumptions rows
    exact soundness interface env offset rows
  completeness := by
    intro env offset _assumptions specification
    exact completeness interface env offset specification

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.StateBinding
