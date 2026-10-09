import NightstreamFPrime.Circuit.Quadratic
import NightstreamFPrime.Lifecycle.PiDEC.v1_2.SignedSplitScalar
import NightstreamFPrime.Lifecycle.XOut
import NightstreamFPrime.Spec.GoldilocksPrime

/-!
Owns the canonical state-word checks used by the PiCCS statement boundary.

The pilot hashes fixed-width word arrays. These rows pin the constant domain
chunk of each array and bind the four verifier-context words in both states to
one verifier-owned public value. They also split every packed prior parent
word into the sixteen child digits that the PiCCS running statement reads.
Each lane owns one hinted sign column. The shared Π_DEC rows
(`PiDEC.v1_2.SignedSplitScalar`) check that the sign is Boolean and every
digit is zero or `1 - 2 · sign`, and the three recomposed lanes pack into the
hashed word. The hint is not authority; the rows bind it. The other running
values stay in their existing zero-copy columns.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_2.StateBinding

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

def packWordExpr (low middle high : Expr) : Expr :=
  low + Expr.const packRadix * middle + Expr.const (packRadix * packRadix) * high

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

/-! ## Hinted sign columns -/

/-- One hinted sign column per packed parent coordinate, word-major. -/
def signCount : Nat := 3 * packedParentWords

@[simp] theorem signCount_eq : signCount = 270 := by
  rfl

/-- Local column of the sign of parent coordinate `3 · word + lane`. -/
def signIndex (word : Fin packedParentWords) (lane : Fin 3) : Fin signCount :=
  ⟨3 * word.val + lane.val, by
    have wordBound := word.isLt
    have laneBound := lane.isLt
    unfold signCount
    omega⟩

def signWordOf (index : Fin signCount) : Fin packedParentWords :=
  ⟨index.val / 3, by
    have bound := index.isLt
    unfold signCount at bound
    omega⟩

def signLaneOf (index : Fin signCount) : Fin 3 :=
  ⟨index.val % 3, Nat.mod_lt _ (by decide)⟩

@[simp] theorem signWordOf_signIndex (word : Fin packedParentWords) (lane : Fin 3) :
    signWordOf (signIndex word lane) = word := by
  apply Fin.ext
  have laneBound := lane.isLt
  simp only [signWordOf, signIndex]
  omega

@[simp] theorem signLaneOf_signIndex (word : Fin packedParentWords) (lane : Fin 3) :
    signLaneOf (signIndex word lane) = lane := by
  apply Fin.ext
  have laneBound := lane.isLt
  simp only [signLaneOf, signIndex]
  omega

/-- The sign bit of parent coordinate `3 · word + lane`; the digit sign is
`1 - 2 · bit`. -/
def signBit (offset : Nat) (word : Fin packedParentWords) (lane : Fin 3) : Expr :=
  Expr.var (offset + (signIndex word lane).val)

structure Interface where
  priorState : Nat → Nat → Expr
  outputState : Nat → Nat → Expr
  expectedContext : Nat → Fin 4 → Expr
  /-- Child digit of parent coordinate `3 · word + lane`. -/
  priorDigit : Nat → Fin packedParentWords → Fin 3 → Radix.ChildIndex → Expr

/-- The non-authoritative sign of one lane: the centered sign of its own
recomposed digits. -/
def signHint (interface : Interface) (offset : Nat)
    (word : Fin packedParentWords) (lane : Fin 3) : Hint :=
  PiDEC.v1_2.SignedSplitScalar.bitHint (PiDEC.v1_2.SignedSplitScalar.recomposeDigits (interface.priorDigit offset word lane))

def signHints (interface : Interface) (offset : Nat) : List Hint :=
  List.ofFn fun index : Fin signCount =>
    signHint interface offset (signWordOf index) (signLaneOf index)

@[simp] theorem signHints_length (interface : Interface) (offset : Nat) :
    (signHints interface offset).length = signCount := by
  rw [signHints, List.length_ofFn]

theorem signHints_get (interface : Interface) (offset : Nat) (index : Fin signCount)
    (bound : index.val < (signHints interface offset).length) :
    (signHints interface offset).get ⟨index.val, bound⟩ =
      signHint interface offset (signWordOf index) (signLaneOf index) := by
  unfold signHints
  rw [List.get_ofFn]
  rfl

def stateAssertions (state : Nat → Expr) : List Expr :=
  fixedWords.map fun word => state word.index - Expr.const word.value

def contextAssertions (state : Nat → Expr)
    (expected : Fin 4 → Expr) : List Expr :=
  (List.finRange 4).map fun lane =>
    state (contextWordStart + lane.val) - expected lane

/-- The sign row, then one row per digit: zero or `1 - 2 · sign`. -/
def laneAssertions (sign : Expr) (digits : Radix.ChildIndex → Expr) : List Expr :=
  PiDEC.v1_2.SignedSplitScalar.signRow sign :: List.ofFn fun child => PiDEC.v1_2.SignedSplitScalar.digitRow sign (digits child)

def packedAssertion (interface : Interface) (offset : Nat)
    (word : Fin packedParentWords) : Expr :=
  interface.priorState offset (packedWordStart + word.val) -
    packWordExpr (PiDEC.v1_2.SignedSplitScalar.recomposeDigits (interface.priorDigit offset word 0))
      (PiDEC.v1_2.SignedSplitScalar.recomposeDigits (interface.priorDigit offset word 1))
      (PiDEC.v1_2.SignedSplitScalar.recomposeDigits (interface.priorDigit offset word 2))

/-- Three lanes of sign and digit rows, then the packing row. -/
def wordAssertions (interface : Interface) (offset : Nat)
    (word : Fin packedParentWords) : List Expr :=
  (List.finRange 3).flatMap (fun lane =>
      laneAssertions (signBit offset word lane)
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

def priorDigits (interface : Interface) (offset : Nat) (env : Env)
    (word : Fin packedParentWords) (lane : Fin 3) : Radix.ChildIndex → F :=
  fun child => (interface.priorDigit offset word lane child).eval env

/-- Every prior lane is a common-sign digit vector, and the three recomposed
lanes pack into the hashed prior word. The sign is a witness of the rows, not
part of the statement. -/
structure ChildrenSplit (interface : Interface) (offset : Nat) (env : Env) : Prop where
  digits : ∀ word lane, ∃ sign,
    Radix.UniformSignedDigits.ConstraintPredicate sign
      (priorDigits interface offset env word lane)
  packed : ∀ word,
    (interface.priorState offset (packedWordStart + word.val)).eval env =
      packWord (Radix.recomposeScalar (priorDigits interface offset env word 0))
        (Radix.recomposeScalar (priorDigits interface offset env word 1))
        (Radix.recomposeScalar (priorDigits interface offset env word 2))

structure SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop where
  priorCanonical : StateCanonical (interface.priorState offset) env
  outputCanonical : StateCanonical (interface.outputState offset) env
  priorContext : ContextBound (interface.priorState offset)
    (interface.expectedContext offset) env
  outputContext : ContextBound (interface.outputState offset)
    (interface.expectedContext offset) env
  priorChildren : ChildrenSplit interface offset env

/-- The split facts depend only on the evaluated packed words and digits. -/
theorem ChildrenSplit.congr {left right : Interface} {leftOffset rightOffset : Nat}
    {env : Env} (split : ChildrenSplit left leftOffset env)
    (packed : ∀ word : Fin packedParentWords,
      (left.priorState leftOffset (packedWordStart + word.val)).eval env =
        (right.priorState rightOffset (packedWordStart + word.val)).eval env)
    (digit : ∀ word lane child,
      (left.priorDigit leftOffset word lane child).eval env =
        (right.priorDigit rightOffset word lane child).eval env) :
    ChildrenSplit right rightOffset env := by
  have digitsSame (word : Fin packedParentWords) (lane : Fin 3) :
      priorDigits right rightOffset env word lane =
        priorDigits left leftOffset env word lane :=
    funext fun child => (digit word lane child).symm
  refine ⟨?_, ?_⟩
  · intro word lane
    rw [digitsSame]
    exact split.digits word lane
  · intro word
    rw [← packed word, digitsSame, digitsSame, digitsSame]
    exact split.packed word

theorem SpecHolds.contextPreserved
    {interface : Interface} {offset : Nat} {env : Env}
    (specification : SpecHolds interface offset env) :
    ContextPreserved (interface.priorState offset)
      (interface.outputState offset) env := by
  intro lane
  rw [specification.outputContext lane, specification.priorContext lane]

def opsAt (interface : Interface) (offset : Nat) : List Op :=
  .witness (WitnessBatch.hinted offset (signHints interface offset)) ::
    (assertions interface offset).map Op.assertZero

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), offset + signCount, opsAt interface offset)

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
  change recipeConstraints offset [] ++
      flatConstraints ((assertions interface offset).map Op.assertZero) = _
  rw [flatConstraints_assertions]
  rfl

/-! ## Child-row membership -/

private theorem signRow_mem (sign : Expr) (digits : Radix.ChildIndex → Expr) :
    PiDEC.v1_2.SignedSplitScalar.signRow sign ∈ laneAssertions sign digits :=
  List.mem_cons.mpr (Or.inl rfl)

private theorem digitRow_mem (sign : Expr) (digits : Radix.ChildIndex → Expr)
    (child : Radix.ChildIndex) :
    PiDEC.v1_2.SignedSplitScalar.digitRow sign (digits child) ∈ laneAssertions sign digits :=
  List.mem_cons_of_mem _ (List.mem_ofFn.mpr ⟨child, rfl⟩)

private theorem laneRow_mem {interface : Interface} {offset : Nat}
    {expression : Expr} (word : Fin packedParentWords) (lane : Fin 3)
    (member : expression ∈ laneAssertions (signBit offset word lane)
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
      (∃ lane, expression = PiDEC.v1_2.SignedSplitScalar.signRow (signBit offset word lane) ∨
        ∃ child, expression = PiDEC.v1_2.SignedSplitScalar.digitRow (signBit offset word lane)
          (interface.priorDigit offset word lane child)) ∨
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

private theorem packedRow_eval (interface : Interface) (offset : Nat) (env : Env)
    (word : Fin packedParentWords) :
    (packedAssertion interface offset word).eval env = 0 ↔
      (interface.priorState offset (packedWordStart + word.val)).eval env =
        packWord (Radix.recomposeScalar (priorDigits interface offset env word 0))
          (Radix.recomposeScalar (priorDigits interface offset env word 1))
          (Radix.recomposeScalar (priorDigits interface offset env word 2)) := by
  unfold packedAssertion
  rw [assertion_holds_iff, packWordExpr_eval, PiDEC.v1_2.SignedSplitScalar.recomposeDigits_eval,
    PiDEC.v1_2.SignedSplitScalar.recomposeDigits_eval, PiDEC.v1_2.SignedSplitScalar.recomposeDigits_eval]
  rfl

private theorem childrenSplit_of_rows (interface : Interface) (offset : Nat) (env : Env)
    (rows : ∀ expression ∈ childAssertions interface offset, expression.eval env = 0) :
    ChildrenSplit interface offset env := by
  refine ⟨?_, ?_⟩
  · intro word lane
    have signRow := rows _ (laneRow_mem word lane (signRow_mem _ _))
    rw [PiDEC.v1_2.SignedSplitScalar.signRow_eval] at signRow
    refine ⟨1 - 2 * (signBit offset word lane).eval env, ?_, ?_⟩
    · rcases GoldilocksPrime.baseFieldNoZeroDivisors _ _ signRow with zero | one
      · right
        left
        rw [zero]
        decide
      · right
        right
        rw [sub_eq_zero.mp one]
        decide
    · intro child
      have row := rows _ (laneRow_mem word lane (digitRow_mem _ _ child))
      rw [PiDEC.v1_2.SignedSplitScalar.digitRow_eval] at row
      rcases GoldilocksPrime.baseFieldNoZeroDivisors _ _ row with zero | signed
      · exact Or.inl zero
      · exact Or.inr (sub_eq_zero.mp signed)
  · intro word
    exact (packedRow_eval interface offset env word).mp
      (rows _ (packedRow_mem interface offset word))

/-- The child rows hold whenever the split holds and every sign column holds
the hinted sign of its own lane. -/
private theorem childRows_of_split (interface : Interface) (offset : Nat) (env : Env)
    (split : ChildrenSplit interface offset env)
    (signs : ∀ word lane, (signBit offset word lane).eval env =
      (signHint interface offset word lane).eval env) :
    ∀ expression ∈ childAssertions interface offset, expression.eval env = 0 := by
  intro expression member
  rcases childRow_cases member with ⟨word, ⟨lane, signRow | ⟨child, digitRow⟩⟩ | packedRow⟩
  · obtain ⟨_, constraint⟩ := split.digits word lane
    subst expression
    rw [PiDEC.v1_2.SignedSplitScalar.signRow_eval, signs word lane]
    exact (PiDEC.v1_2.SignedSplitScalar.signedDigitRows_of_constraint _ env constraint).1
  · obtain ⟨_, constraint⟩ := split.digits word lane
    subst expression
    rw [PiDEC.v1_2.SignedSplitScalar.digitRow_eval, signs word lane]
    exact (PiDEC.v1_2.SignedSplitScalar.signedDigitRows_of_constraint _ env constraint).2 child
  · subst expression
    exact (packedRow_eval interface offset env word).mpr (split.packed word)

/-- Every row holds whenever the semantic state predicate holds and every sign
column holds its hinted value. -/
theorem constraintsHold_of_signs (interface : Interface) (env : Env)
    (offset : Nat) (specification : SpecHolds interface offset env)
    (signs : ∀ word lane, (signBit offset word lane).eval env =
      (signHint interface offset word lane).eval env) :
    ConstraintsHold env (assertions interface offset) := by
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
  · exact childRows_of_split interface offset env specification.priorChildren signs
      expression childMember

/-! ## Circuit contract -/

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    SpecHolds interface offset env := by
  rw [main_ops] at rows
  have rowOfMember : ∀ expression ∈ assertions interface offset,
      expression.eval env = 0 := by
    intro expression member
    exact rows (Op.assertZero expression) (by
      rw [opsAt]
      exact List.mem_cons_of_mem _ (List.mem_map.mpr ⟨expression, member, rfl⟩))
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
    (signHolds : ∀ word lane, Holds (signBit offset word lane)) :
    ∀ expression ∈ childAssertions interface offset, Holds expression := by
  have sub : ∀ left right, Holds left → Holds right → Holds (left - right) :=
    fun left right leftHolds rightHolds =>
      add _ _ leftHolds (mul _ _ (constant _) rightHolds)
  have scale : ∀ weight value, Holds value → Holds (Expr.const weight * value) :=
    fun weight value valueHolds => mul _ _ (constant weight) valueHolds
  have one : Holds 1 := constant _
  have two : Holds 2 := constant _
  intro expression member
  rcases childRow_cases member with ⟨word, ⟨lane, signRow | ⟨child, digitRow⟩⟩ | packedRow⟩
  · subst expression
    exact mul _ _ (signHolds word lane) (sub _ _ (signHolds word lane) one)
  · subst expression
    exact mul _ _ (digitHolds word lane child) (sub _ _ (digitHolds word lane child)
      (sub _ _ one (mul _ _ two (signHolds word lane))))
  · subst expression
    have recomposed (lane : Fin 3) :
        Holds (PiDEC.v1_2.SignedSplitScalar.recomposeDigits (interface.priorDigit offset word lane)) :=
      PiDEC.v1_2.SignedSplitScalar.recomposeDigits_closed Holds constant add scale _ (digitHolds word lane)
    exact sub _ _ (packedHolds word)
      (packWordExpr_closed Holds add scale (recomposed 0) (recomposed 1) (recomposed 2))

private theorem signBit_varsBelow (offset : Nat) (word : Fin packedParentWords)
    (lane : Fin 3) : (signBit offset word lane).VarsBelow (offset + signCount) := by
  have bound := (signIndex word lane).isLt
  simp only [signBit, Expr.VarsBelow]
  omega

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (main interface) offset),
      expression.VarsBelow (offset + signCount) := by
  have mono : ∀ expression : Expr, expression.VarsBelow offset →
      expression.VarsBelow (offset + signCount) := fun expression below =>
    Expr.VarsBelow.mono expression below (by omega)
  intro expression member
  rw [main_ops, flatConstraints_opsAt] at member
  rcases List.mem_append.mp member with member | childMember
  · apply mono
    rw [stateWordAssertions, List.mem_append] at member
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
  · exact childAssertions_closed (fun expression => expression.VarsBelow (offset + signCount))
      (fun _ => trivial) (fun left right => Expr.VarsBelow.add left right _)
      (fun left right => Expr.VarsBelow.mul left right _) interface offset
      (fun word => mono _ (assumptions.priorPacked word))
      (fun word lane child => mono _ (assumptions.priorDigit word lane child))
      (signBit_varsBelow offset) expression childMember

theorem specHolds_of_agree_below (interface : Interface) (offset : Nat)
    (before after : Env) (assumptions : Assumptions interface offset before)
    (agrees : ∀ index, index < offset → after index = before index)
    (specification : SpecHolds interface offset before) :
    SpecHolds interface offset after := by
  have same : ∀ expression : Expr, expression.VarsBelow offset →
      expression.eval after = expression.eval before := fun expression below =>
    Expr.eval_eq_of_agree_below expression offset after before below agrees
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
  · refine ⟨?_, ?_⟩
    · intro word lane
      rw [digitsSame]
      exact specification.priorChildren.digits word lane
    · intro word
      rw [same _ (assumptions.priorPacked word), digitsSame, digitsSame, digitsSame]
      exact specification.priorChildren.packed word

/-! ## Hinted completion -/

/-- Every sign hint reads only prior digits, below the leaf offset. -/
theorem signHints_readBelow (interface : Interface) (offset : Nat) {env : Env}
    (assumptions : Assumptions interface offset env) :
    HintsReadBelow offset (signHints interface offset) := by
  intro hint member
  rw [signHints, List.mem_ofFn] at member
  rcases member with ⟨index, rfl⟩
  exact PiDEC.v1_2.SignedSplitScalar.recomposeDigits_closed (fun expression => expression.VarsBelow offset)
    (fun _ => trivial) (fun left right => Expr.VarsBelow.add left right offset)
    (fun weight value below => Expr.VarsBelow.mul _ value offset trivial below) _
    (assumptions.priorDigit _ _)

/-- Each lane's hint keeps its value when the environment changes only at or
above the leaf offset. -/
theorem signHint_eval_of_agree_below (interface : Interface) (offset : Nat)
    (before after : Env) (assumptions : Assumptions interface offset before)
    (agrees : ∀ index, index < offset → after index = before index)
    (word : Fin packedParentWords) (lane : Fin 3) :
    (signHint interface offset word lane).eval after =
      (signHint interface offset word lane).eval before := by
  apply Hint.eval_eq_of_agree_below _ offset after before _ agrees
  have member : signHint interface offset word lane ∈ signHints interface offset := by
    rw [signHints, List.mem_ofFn]
    exact ⟨signIndex word lane, by simp⟩
  exact signHints_readBelow interface offset assumptions _ member

def completeEnv (interface : Interface) (env : Env) (offset : Nat) : Env :=
  executeHints env offset (signHints interface offset)

theorem completeEnv_agrees_below (interface : Interface) (env : Env) (offset : Nat) :
    ∀ index, index < offset → completeEnv interface env offset index = env index :=
  executeHints_agrees_below env offset (signHints interface offset)

theorem completeEnv_sign (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (word : Fin packedParentWords) (lane : Fin 3) :
    (signBit offset word lane).eval (completeEnv interface env offset) =
      (signHint interface offset word lane).eval env := by
  have value := executeHints_value_of_readBelow env offset (signHints interface offset)
    (signHints_readBelow interface offset assumptions) (signIndex word lane).val
    (by rw [signHints_length]; exact (signIndex word lane).isLt)
  rw [signBit, Expr.eval_var, completeEnv, value, signHints_get, signWordOf_signIndex,
    signLaneOf_signIndex]

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (main interface) offset) = signCount := by
  rw [main_ops, opsAt]
  change (WitnessBatch.hinted offset (signHints interface offset)).outputLength +
      localLength ((assertions interface offset).map Op.assertZero) = signCount
  have assertions : localLength ((assertions interface offset).map Op.assertZero) = 0 := by
    generalize assertions interface offset = expressions
    induction expressions with
    | nil => rfl
    | cons _ rest inductionHypothesis =>
        change 0 + localLength (rest.map Op.assertZero) = 0
        simpa using inductionHypothesis
  rw [assertions, WitnessBatch.hinted_outputLength, signHints_length, Nat.add_zero]

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  let completed := completeEnv interface env offset
  have agrees := completeEnv_agrees_below interface env offset
  refine ⟨completed, ?_, ?_⟩
  · rw [localLength_eq]
    simpa [completed, completeEnv] using
      executeHints_agreesOutside env offset (signHints interface offset)
  · unfold holdsFlat
    rw [main_ops, flatConstraints_opsAt]
    apply constraintsHold_of_signs interface completed offset
      (specHolds_of_agree_below interface offset env completed assumptions agrees
        specification)
    intro word lane
    rw [completeEnv_sign interface env offset assumptions word lane,
      signHint_eval_of_agree_below interface offset env completed assumptions agrees]

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
    intro env offset assumptions specification
    exact completeness interface env offset assumptions specification

end NightstreamFPrime.Lifecycle.PiCCS.v1_2.StateBinding
