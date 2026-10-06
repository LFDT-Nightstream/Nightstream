import NightstreamFPrime.Circuit.Quadratic
import NightstreamFPrime.Lifecycle.XOut
import NightstreamFPrime.Spec.GoldilocksPrime
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits

/-!
Owns the canonical state-word checks used by the PiCCS statement boundary.

The pilot hashes fixed-width word arrays. These rows pin the constant domain
chunk of each array and bind the four verifier-context words in both states to
one verifier-owned public value. They also split every packed prior parent
word into the sixteen child digits that the PiCCS running statement reads:
each lane has one Boolean sign and sixteen digits that are zero or
`1 - 2 · sign`, and the three recomposed lanes pack into the hashed word. The
other running values stay in their existing zero-copy columns.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.StateBinding

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

/-- One fixed word in the canonical state serialization. -/
structure FixedWord where
  index : Nat
  value : F
deriving DecidableEq

/-- The fixed words: the constant domain chunk. -/
def fixedWords : List FixedWord :=
  (List.finRange stateDomainChunk.length).map fun index =>
    ⟨index.val, stateDomainChunk.getD index.val 0⟩

/-- The first packed parent word: after the domain chunk and the 27,704
running field words. -/
def packedWordStart : Nat := 27716

/-- The first verifier-context word: the tail follows the domain chunk and
the 27,794 running words. -/
def contextWordStart : Nat := 27806

theorem fixedWords_length : fixedWords.length = 12 := by
  simp [fixedWords, stateDomainChunk_length]

theorem fixedWord_index_lt (word : FixedWord) (member : word ∈ fixedWords) :
    word.index < 27819 := by
  rw [fixedWords, List.mem_map] at member
  rcases member with ⟨index, _indexMember, rfl⟩
  have bound := index.isLt
  have chunkLength := stateDomainChunk_length
  change index.val < 27819
  omega

/-- `Σ_j 2^j · digit_j` with the production radix weights. -/
def recomposeExpr (digits : Radix.ChildIndex → Expr) : Expr :=
  ((List.ofFn digits).zip
      (List.ofFn Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight)).foldr
    (fun pair suffix => Expr.const pair.2 * pair.1 + suffix) 0

def packWordExpr (low middle high : Expr) : Expr :=
  low + Expr.const packRadix * middle + Expr.const (packRadix * packRadix) * high

/-- A property closed under constants, sums and constant scaling holds on
every recomposition of digits that satisfy it. -/
theorem recomposeExpr_closed (Holds : Expr → Prop)
    (constant : ∀ value, Holds (Expr.const value))
    (add : ∀ left right, Holds left → Holds right → Holds (left + right))
    (scale : ∀ weight value, Holds value → Holds (Expr.const weight * value))
    (digits : Radix.ChildIndex → Expr) (digitHolds : ∀ child, Holds (digits child)) :
    Holds (recomposeExpr digits) := by
  unfold recomposeExpr
  generalize (List.ofFn Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight) = weights
  have values : ∀ value ∈ List.ofFn digits, Holds value := by
    intro value member
    rw [List.mem_ofFn'] at member
    rcases member with ⟨child, rfl⟩
    exact digitHolds child
  generalize List.ofFn digits = items at values
  induction items generalizing weights with
  | nil => exact constant _
  | cons item rest inductionHypothesis =>
      cases weights with
      | nil => exact constant _
      | cons weight weights =>
          exact add _ _ (scale weight item (values item (by simp)))
            (inductionHypothesis weights fun value member =>
              values value (by simp [member]))

private theorem weightedFold_eval (env : Env) :
    ∀ (values : List Expr) (weights : List F),
      ((values.zip weights).foldr
          (fun pair suffix => Expr.const pair.2 * pair.1 + suffix) 0).eval env =
        ((values.map fun value => value.eval env).zip weights).foldr
          (fun pair suffix => pair.2 * pair.1 + suffix) 0
  | [], _ => Fin.ext rfl
  | _ :: _, [] => Fin.ext rfl
  | value :: values, weight :: weights => by
      change weight * value.eval env +
          ((values.zip weights).foldr
            (fun pair suffix => Expr.const pair.2 * pair.1 + suffix) 0).eval env =
        weight * value.eval env +
          (((values.map fun item => item.eval env).zip weights).foldr
            (fun pair suffix => pair.2 * pair.1 + suffix) 0)
      rw [weightedFold_eval env values weights]

theorem recomposeExpr_eval (digits : Radix.ChildIndex → Expr) (env : Env) :
    (recomposeExpr digits).eval env =
      Radix.recomposeScalar fun child => (digits child).eval env := by
  unfold recomposeExpr
  rw [weightedFold_eval, List.map_ofFn,
    ← Radix.recomposeScalarList_eq]
  rfl

theorem packWordExpr_eval (low middle high : Expr) (env : Env) :
    (packWordExpr low middle high).eval env =
      packWord (low.eval env) (middle.eval env) (high.eval env) := by
  simp only [packWordExpr, Expr.eval_hadd, Expr.eval_hmul, Expr.eval_const, packWord]

theorem packWordExpr_closed (Holds : Expr → Prop)
    (add : ∀ left right, Holds left → Holds right → Holds (left + right))
    (scale : ∀ weight value, Holds value → Holds (Expr.const weight * value))
    {low middle high : Expr} (lowHolds : Holds low) (middleHolds : Holds middle)
    (highHolds : Holds high) :
    Holds (packWordExpr low middle high) :=
  add _ _ (add _ _ lowHolds (scale _ _ middleHolds)) (scale _ _ highHolds)

structure Interface where
  priorState : Nat → Nat → Expr
  outputState : Nat → Nat → Expr
  expectedContext : Nat → Fin 4 → Expr
  /-- Child digit of parent coordinate `3 · word + lane`. -/
  priorDigit : Nat → Fin packedParentWords → Fin 3 → Radix.ChildIndex → Expr
  /-- Boolean sign of that coordinate; the digit sign is `1 - 2 · bit`. -/
  priorSign : Nat → Fin packedParentWords → Fin 3 → Expr

def stateAssertions (state : Nat → Expr) : List Expr :=
  fixedWords.map fun word => state word.index - Expr.const word.value

def contextAssertions (state : Nat → Expr)
    (expected : Fin 4 → Expr) : List Expr :=
  (List.finRange 4).map fun lane =>
    state (contextWordStart + lane.val) - expected lane

def signedUnit (sign : Expr) : Expr :=
  Expr.const 1 - Expr.const 2 * sign

/-- The sign row, then one row per digit: zero or `1 - 2 · sign`. -/
def laneAssertions (sign : Expr) (digits : Radix.ChildIndex → Expr) : List Expr :=
  sign * (sign - Expr.const 1) ::
    List.ofFn fun child => digits child * (digits child - signedUnit sign)

def packedAssertion (interface : Interface) (offset : Nat)
    (word : Fin packedParentWords) : Expr :=
  interface.priorState offset (packedWordStart + word.val) -
    packWordExpr (recomposeExpr (interface.priorDigit offset word 0))
      (recomposeExpr (interface.priorDigit offset word 1))
      (recomposeExpr (interface.priorDigit offset word 2))

/-- Three lanes of sign and digit rows, then the packing row. -/
def wordAssertions (interface : Interface) (offset : Nat)
    (word : Fin packedParentWords) : List Expr :=
  (List.finRange 3).flatMap (fun lane =>
      laneAssertions (interface.priorSign offset word lane)
        (interface.priorDigit offset word lane)) ++
    [packedAssertion interface offset word]

def childAssertions (interface : Interface) (offset : Nat) : List Expr :=
  (List.finRange packedParentWords).flatMap (wordAssertions interface offset)

def stateWordAssertions (interface : Interface) (offset : Nat) : List Expr :=
  stateAssertions (interface.priorState offset) ++ (
    stateAssertions (interface.outputState offset) ++
      contextAssertions (interface.priorState offset)
        (interface.expectedContext offset) ++
      contextAssertions (interface.outputState offset)
        (interface.expectedContext offset))

def assertions (interface : Interface) (offset : Nat) : List Expr :=
  stateWordAssertions interface offset ++ childAssertions interface offset

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

def priorSignValue (interface : Interface) (offset : Nat) (env : Env)
    (word : Fin packedParentWords) (lane : Fin 3) : F :=
  (interface.priorSign offset word lane).eval env

def priorDigits (interface : Interface) (offset : Nat) (env : Env)
    (word : Fin packedParentWords) (lane : Fin 3) : Radix.ChildIndex → F :=
  fun child => (interface.priorDigit offset word lane child).eval env

/-- Every prior lane carries one Boolean sign and common-sign digits, and the
three recomposed lanes pack into the hashed prior word. -/
structure ChildrenSplit (interface : Interface) (offset : Nat) (env : Env) : Prop where
  sign : ∀ word lane, priorSignValue interface offset env word lane = 0 ∨
    priorSignValue interface offset env word lane = 1
  digit : ∀ word lane child,
    priorDigits interface offset env word lane child = 0 ∨
      priorDigits interface offset env word lane child =
        1 - 2 * priorSignValue interface offset env word lane
  packed : ∀ word,
    (interface.priorState offset (packedWordStart + word.val)).eval env =
      packWord (Radix.recomposeScalar (priorDigits interface offset env word 0))
        (Radix.recomposeScalar (priorDigits interface offset env word 1))
        (Radix.recomposeScalar (priorDigits interface offset env word 2))

/-- Every checked lane is in the Π_DEC accepted digit language. -/
theorem ChildrenSplit.constraint {interface : Interface} {offset : Nat} {env : Env}
    (split : ChildrenSplit interface offset env)
    (word : Fin packedParentWords) (lane : Fin 3) :
    Radix.UniformSignedDigits.ConstraintPredicate
      (1 - 2 * priorSignValue interface offset env word lane)
      (priorDigits interface offset env word lane) := by
  refine ⟨?_, split.digit word lane⟩
  rcases split.sign word lane with zero | one
  · right
    left
    rw [zero]
    decide
  · right
    right
    rw [one]
    decide

/-- The child-split predicate depends only on the values of the packed
words, digits, and signs that it reads. -/
theorem ChildrenSplit.congr {left right : Interface} {leftOffset rightOffset : Nat}
    {env : Env} (split : ChildrenSplit left leftOffset env)
    (packed : ∀ word : Fin packedParentWords,
      (left.priorState leftOffset (packedWordStart + word.val)).eval env =
        (right.priorState rightOffset (packedWordStart + word.val)).eval env)
    (digit : ∀ word lane child,
      (left.priorDigit leftOffset word lane child).eval env =
        (right.priorDigit rightOffset word lane child).eval env)
    (sign : ∀ word lane,
      (left.priorSign leftOffset word lane).eval env =
        (right.priorSign rightOffset word lane).eval env) :
    ChildrenSplit right rightOffset env := by
  have signSame (word : Fin packedParentWords) (lane : Fin 3) :
      priorSignValue right rightOffset env word lane =
        priorSignValue left leftOffset env word lane :=
    (sign word lane).symm
  have digitsSame (word : Fin packedParentWords) (lane : Fin 3) :
      priorDigits right rightOffset env word lane =
        priorDigits left leftOffset env word lane :=
    funext fun child => (digit word lane child).symm
  refine ⟨?_, ?_, ?_⟩
  · intro word lane
    rw [signSame]
    exact split.sign word lane
  · intro word lane child
    rw [digitsSame, signSame]
    exact split.digit word lane child
  · intro word
    rw [← packed word, digitsSame, digitsSame, digitsSame]
    exact split.packed word

structure SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop where
  priorCanonical : StateCanonical (interface.priorState offset) env
  outputCanonical : StateCanonical (interface.outputState offset) env
  priorContext : ContextBound (interface.priorState offset)
    (interface.expectedContext offset) env
  outputContext : ContextBound (interface.outputState offset)
    (interface.expectedContext offset) env
  priorChildren : ChildrenSplit interface offset env

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

/-! ## Child-row membership -/

private theorem signRow_mem (sign : Expr) (digits : Radix.ChildIndex → Expr) :
    sign * (sign - Expr.const 1) ∈ laneAssertions sign digits :=
  List.mem_cons.mpr (Or.inl rfl)

private theorem digitRow_mem (sign : Expr) (digits : Radix.ChildIndex → Expr)
    (child : Radix.ChildIndex) :
    digits child * (digits child - signedUnit sign) ∈ laneAssertions sign digits :=
  List.mem_cons_of_mem _ (List.mem_ofFn.mpr ⟨child, rfl⟩)

private theorem laneRow_mem {interface : Interface} {offset : Nat}
    {expression : Expr} (word : Fin packedParentWords) (lane : Fin 3)
    (member : expression ∈ laneAssertions (interface.priorSign offset word lane)
      (interface.priorDigit offset word lane)) :
    expression ∈ childAssertions interface offset :=
  List.mem_flatMap.mpr ⟨word, List.mem_finRange word,
    List.mem_append_left _ (List.mem_flatMap.mpr ⟨lane, List.mem_finRange lane, member⟩)⟩

private theorem packedRow_mem (interface : Interface) (offset : Nat)
    (word : Fin packedParentWords) :
    packedAssertion interface offset word ∈ childAssertions interface offset :=
  List.mem_flatMap.mpr ⟨word, List.mem_finRange word,
    List.mem_append_right _ (List.mem_singleton_self _)⟩

theorem childRow_cases {interface : Interface} {offset : Nat}
    {expression : Expr} (member : expression ∈ childAssertions interface offset) :
    ∃ word,
      (∃ lane, expression =
          interface.priorSign offset word lane *
            (interface.priorSign offset word lane - Expr.const 1) ∨
        ∃ child, expression =
          interface.priorDigit offset word lane child *
            (interface.priorDigit offset word lane child -
              signedUnit (interface.priorSign offset word lane))) ∨
        expression = packedAssertion interface offset word := by
  rcases List.mem_flatMap.mp member with ⟨word, _, wordMember⟩
  refine ⟨word, ?_⟩
  rcases List.mem_append.mp wordMember with laneMember | packedMember
  · rcases List.mem_flatMap.mp laneMember with ⟨lane, _, rowMember⟩
    left
    refine ⟨lane, ?_⟩
    rcases List.mem_cons.mp rowMember with signRow | digitMember
    · exact Or.inl signRow
    · rcases List.mem_ofFn.mp digitMember with ⟨child, rfl⟩
      exact Or.inr ⟨child, rfl⟩
  · exact Or.inr (List.mem_singleton.mp packedMember)

/-! ## Child-row values -/

private theorem signRow_eval (sign : Expr) (env : Env) :
    (sign * (sign - Expr.const 1)).eval env = sign.eval env * (sign.eval env - 1) := by
  simp only [Expr.eval_hmul, Expr.eval_sub, Expr.eval_const]

private theorem digitRow_eval (sign digit : Expr) (env : Env) :
    (digit * (digit - signedUnit sign)).eval env =
      digit.eval env * (digit.eval env - (1 - 2 * sign.eval env)) := by
  simp only [signedUnit, Expr.eval_hmul, Expr.eval_sub, Expr.eval_const]

private theorem packedRow_eval (interface : Interface) (offset : Nat) (env : Env)
    (word : Fin packedParentWords) :
    (packedAssertion interface offset word).eval env = 0 ↔
      (interface.priorState offset (packedWordStart + word.val)).eval env =
        packWord (Radix.recomposeScalar (priorDigits interface offset env word 0))
          (Radix.recomposeScalar (priorDigits interface offset env word 1))
          (Radix.recomposeScalar (priorDigits interface offset env word 2)) := by
  unfold packedAssertion
  rw [assertion_holds_iff, packWordExpr_eval, recomposeExpr_eval, recomposeExpr_eval,
    recomposeExpr_eval]
  rfl

private theorem childrenSplit_of_rows (interface : Interface) (offset : Nat) (env : Env)
    (rows : ∀ expression ∈ childAssertions interface offset, expression.eval env = 0) :
    ChildrenSplit interface offset env := by
  refine ⟨?_, ?_, ?_⟩
  · intro word lane
    have row := rows _ (laneRow_mem word lane (signRow_mem _ _))
    rw [signRow_eval] at row
    rcases GoldilocksPrime.baseFieldNoZeroDivisors _ _ row with zero | one
    · exact Or.inl zero
    · exact Or.inr (sub_eq_zero.mp one)
  · intro word lane child
    have row := rows _ (laneRow_mem word lane (digitRow_mem _ _ child))
    rw [digitRow_eval] at row
    rcases GoldilocksPrime.baseFieldNoZeroDivisors _ _ row with zero | signed
    · exact Or.inl zero
    · exact Or.inr (sub_eq_zero.mp signed)
  · intro word
    exact (packedRow_eval interface offset env word).mp
      (rows _ (packedRow_mem interface offset word))

private theorem childRows_of_split (interface : Interface) (offset : Nat) (env : Env)
    (split : ChildrenSplit interface offset env) :
    ∀ expression ∈ childAssertions interface offset, expression.eval env = 0 := by
  intro expression member
  rcases childRow_cases member with ⟨word, ⟨lane, signRow | ⟨child, digitRow⟩⟩ | packedRow⟩
  · subst expression
    rw [signRow_eval]
    rcases split.sign word lane with zero | one
    · rw [show (interface.priorSign offset word lane).eval env = 0 from zero]
      exact zero_mul _
    · rw [show (interface.priorSign offset word lane).eval env = 1 from one, sub_self]
      exact mul_zero _
  · subst expression
    rw [digitRow_eval]
    rcases split.digit word lane child with zero | signed
    · rw [show (interface.priorDigit offset word lane child).eval env = 0 from zero]
      exact zero_mul _
    · rw [show (interface.priorDigit offset word lane child).eval env =
          1 - 2 * (interface.priorSign offset word lane).eval env from signed, sub_self]
      exact mul_zero _
  · subst expression
    exact (packedRow_eval interface offset env word).mpr (split.packed word)

/-! ## Circuit contract -/

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
  have stateRow : ∀ expression ∈ stateWordAssertions interface offset,
      expression.eval env = 0 := fun expression member =>
    rowOfMember expression (List.mem_append_left _ member)
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro word member
    have row := stateRow
      (interface.priorState offset word.index -
        Expr.const word.value) (by
          unfold stateWordAssertions
          apply List.mem_append_left
          rw [stateAssertions, List.mem_map]
          exact ⟨word, member, rfl⟩)
    exact (assertion_holds_iff _ _ env).mp row
  · intro word member
    have row := stateRow
      (interface.outputState offset word.index -
        Expr.const word.value) (by
          unfold stateWordAssertions
          apply List.mem_append_right
          apply List.mem_append_left
          apply List.mem_append_left
          rw [stateAssertions, List.mem_map]
          exact ⟨word, member, rfl⟩)
    exact (assertion_holds_iff _ _ env).mp row
  · intro lane
    have row := stateRow
        (interface.priorState offset
          (contextWordStart + lane.val) -
          interface.expectedContext offset lane) (by
          simp [stateWordAssertions, contextAssertions])
    exact (assertion_holds_iff _ _ env).mp row
  · intro lane
    have row := stateRow
        (interface.outputState offset
          (contextWordStart + lane.val) -
          interface.expectedContext offset lane) (by
          simp [stateWordAssertions, contextAssertions])
    exact (assertion_holds_iff _ _ env).mp row
  · exact childrenSplit_of_rows interface offset env fun expression member =>
      rowOfMember expression (List.mem_append_right _ member)

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
    intro expression member
    rcases List.mem_append.mp member with stateMember | childMember
    · rw [stateWordAssertions, List.mem_append] at stateMember
      rcases stateMember with priorMember | remainingMember
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
    · exact childRows_of_split interface offset env specification.priorChildren
        expression childMember

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
  priorPacked : ∀ word : Fin packedParentWords,
    (interface.priorState offset (packedWordStart + word.val)).VarsBelow offset
  priorDigit : ∀ word lane child,
    (interface.priorDigit offset word lane child).VarsBelow offset
  priorSign : ∀ word lane, (interface.priorSign offset word lane).VarsBelow offset

/-- A property closed under constants, sums and products holds on every
child-split row whose packed word, digits and signs satisfy it. -/
theorem childAssertions_closed (Holds : Expr → Prop)
    (constant : ∀ value, Holds (Expr.const value))
    (add : ∀ left right, Holds left → Holds right → Holds (left + right))
    (mul : ∀ left right, Holds left → Holds right → Holds (left * right))
    (interface : Interface) (offset : Nat)
    (packedHolds : ∀ word : Fin packedParentWords,
      Holds (interface.priorState offset (packedWordStart + word.val)))
    (digitHolds : ∀ word lane child, Holds (interface.priorDigit offset word lane child))
    (signHolds : ∀ word lane, Holds (interface.priorSign offset word lane)) :
    ∀ expression ∈ childAssertions interface offset, Holds expression := by
  have sub : ∀ left right, Holds left → Holds right → Holds (left - right) :=
    fun left right leftHolds rightHolds =>
      add _ _ leftHolds (mul _ _ (constant _) rightHolds)
  have scale : ∀ weight value, Holds value → Holds (Expr.const weight * value) :=
    fun weight value valueHolds => mul _ _ (constant weight) valueHolds
  intro expression member
  rcases childRow_cases member with ⟨word, ⟨lane, signRow | ⟨child, digitRow⟩⟩ | packedRow⟩
  · subst expression
    exact mul _ _ (signHolds word lane) (sub _ _ (signHolds word lane) (constant _))
  · subst expression
    exact mul _ _ (digitHolds word lane child) (sub _ _ (digitHolds word lane child)
      (sub _ _ (constant _) (scale _ _ (signHolds word lane))))
  · subst expression
    have recomposed (lane : Fin 3) :
        Holds (recomposeExpr (interface.priorDigit offset word lane)) :=
      recomposeExpr_closed Holds constant add scale _ (digitHolds word lane)
    exact sub _ _ (packedHolds word)
      (packWordExpr_closed Holds add scale (recomposed 0) (recomposed 1) (recomposed 2))

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (main interface) offset),
      expression.VarsBelow offset := by
  intro expression member
  rw [main_ops, flatConstraints_opsAt] at member
  rcases List.mem_append.mp member with member | childMember
  · rw [stateWordAssertions, List.mem_append] at member
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
  · exact childAssertions_closed (fun expression => expression.VarsBelow offset)
      (fun _ => trivial) (fun left right => Expr.VarsBelow.add left right offset)
      (fun left right => Expr.VarsBelow.mul left right offset) interface offset
      assumptions.priorPacked assumptions.priorDigit assumptions.priorSign
      expression childMember

theorem specHolds_of_agree_below (interface : Interface) (offset : Nat)
    (before after : Env) (assumptions : Assumptions interface offset before)
    (agrees : ∀ index, index < offset → after index = before index)
    (specification : SpecHolds interface offset before) :
    SpecHolds interface offset after := by
  have same : ∀ expression : Expr, expression.VarsBelow offset →
      expression.eval after = expression.eval before := fun expression below =>
    Expr.eval_eq_of_agree_below expression offset after before below agrees
  have signSame (word : Fin packedParentWords) (lane : Fin 3) :
      priorSignValue interface offset after word lane =
        priorSignValue interface offset before word lane :=
    same _ (assumptions.priorSign word lane)
  have digitsSame (word : Fin packedParentWords) (lane : Fin 3) :
      priorDigits interface offset after word lane =
        priorDigits interface offset before word lane :=
    funext fun child => same _ (assumptions.priorDigit word lane child)
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro word member
    rw [same _ (assumptions.priorFixed word member)]
    exact specification.priorCanonical word member
  · intro word member
    rw [same _ (assumptions.outputFixed word member)]
    exact specification.outputCanonical word member
  · intro lane
    rw [same _ (assumptions.priorContext lane), same _ (assumptions.expectedContext lane)]
    exact specification.priorContext lane
  · intro lane
    rw [same _ (assumptions.outputContext lane), same _ (assumptions.expectedContext lane)]
    exact specification.outputContext lane
  · refine ⟨?_, ?_, ?_⟩
    · intro word lane
      rw [signSame]
      exact specification.priorChildren.sign word lane
    · intro word lane child
      rw [digitsSame, signSame]
      exact specification.priorChildren.digit word lane child
    · intro word
      rw [same _ (assumptions.priorPacked word), digitsSame, digitsSame, digitsSame]
      exact specification.priorChildren.packed word

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
  simpa only [completedEq] using! rows

private theorem wordAssertions_length (interface : Interface) (offset : Nat)
    (word : Fin packedParentWords) :
    (wordAssertions interface offset word).length = 52 := by
  simp [wordAssertions, laneAssertions, List.length_flatMap, productionGlobalParams]

theorem assertions_length (interface : Interface) (offset : Nat) :
    (assertions interface offset).length = 4712 := by
  have children : (childAssertions interface offset).length = 4680 := by
    rw [childAssertions, List.length_flatMap]
    simp [wordAssertions_length, packedParentWords]
  simp [assertions, stateWordAssertions, stateAssertions, contextAssertions,
    fixedWords_length, children]

theorem operations_length (interface : Interface) (offset : Nat) :
    (Circuit.ops (main interface) offset).length = 4712 := by
  rw [main_ops, opsAt, List.length_map, assertions_length]

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (main interface) offset)).length = 4712 := by
  rw [main_ops, flatConstraints_opsAt, assertions_length]

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
