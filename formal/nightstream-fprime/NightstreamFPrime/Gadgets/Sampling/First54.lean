import NightstreamFPrime.Circuit.Sequence
import NightstreamFPrime.Gadgets.Sampling.First54Step
import NightstreamFPrime.Gadgets.Sampling.First54ValueStep
import NightstreamFPrime.Spec.Sampling.FirstAccepted

/-!
Owns the fixed 64-candidate, first-54 selector.

Each round calls one 55-slot position child and one 54-slot value child. The
final row requires the absorbing full slot to be one. The selector consumes
all 64 candidates and fails closed when fewer than 54 are accepted. Candidate
decoding and transcript evolution belong to the parent sampler.
-/

namespace NightstreamFPrime.Gadgets.Sampling.First54

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit

def candidateCount : Nat := 64
def outputCount : Nat := First54ValueStep.outputCount
def roundPrivateCount : Nat :=
  First54Step.slotCount + First54ValueStep.outputCount
def logicalPrivateCount : Nat := candidateCount * roundPrivateCount
def logicalRowCount : Nat := logicalPrivateCount + 1

structure Interface where
  accepted : Nat → Fin candidateCount → Expr
  symbol : Nat → Fin candidateCount → Expr

def candidateIndex (index : Nat) : Fin candidateCount :=
  ⟨index % candidateCount, Nat.mod_lt _ (by decide)⟩

def positionOffset (offset round : Nat) : Nat :=
  offset + round * roundPrivateCount

def valueOffset (offset round : Nat) : Nat :=
  positionOffset offset round + First54Step.slotCount

def finalOffset (offset : Nat) : Nat := offset + logicalPrivateCount

def initialPosition (slot : Fin First54Step.slotCount) : Expr :=
  if slot.val = 0 then 1 else 0

def priorPosition (offset round : Nat)
    (slot : Fin First54Step.slotCount) : Expr :=
  match round with
  | 0 => initialPosition slot
  | previous + 1 => First54Step.output (positionOffset offset previous) slot

def priorOutput (offset round : Nat)
    (slot : Fin First54ValueStep.outputCount) : Expr :=
  match round with
  | 0 => 0
  | previous + 1 => First54ValueStep.output (valueOffset offset previous) slot

def positionInterface (interface : Interface) (parentOffset round : Nat) :
    First54Step.Interface where
  accepted := fun _ => interface.accepted parentOffset (candidateIndex round)
  prior := fun _ => priorPosition parentOffset round

def valueInterface (interface : Interface) (parentOffset round : Nat) :
    First54ValueStep.Interface where
  accepted := fun _ => interface.accepted parentOffset (candidateIndex round)
  symbol := fun _ => interface.symbol parentOffset (candidateIndex round)
  priorPosition := fun _ => priorPosition parentOffset round
  priorOutput := fun _ => priorOutput parentOffset round

def positionCircuit (interface : Interface) (parentOffset round : Nat) :
    FormalCircuit :=
  First54Step.circuit (positionInterface interface parentOffset round)

def valueCircuit (interface : Interface) (parentOffset round : Nat) :
    FormalCircuit :=
  First54ValueStep.circuit (valueInterface interface parentOffset round)

def positionName (round : Nat) : String :=
  "first54.position_" ++ toString round

def valueName (round : Nat) : String :=
  "first54.value_" ++ toString round

def positionOp (interface : Interface) (offset round : Nat) : Op :=
  Sequence.childOp (positionName round)
    (positionCircuit interface offset round) (positionOffset offset round)

def valueOp (interface : Interface) (offset round : Nat) : Op :=
  Sequence.childOp (valueName round)
    (valueCircuit interface offset round) (valueOffset offset round)

def roundOps (interface : Interface) (offset round : Nat) : List Op :=
  [positionOp interface offset round, valueOp interface offset round]

def roundOpsPrefix (interface : Interface) (offset count : Nat) : List Op :=
  (List.range count).flatMap (roundOps interface offset)

def fullSlot : Fin First54Step.slotCount :=
  ⟨First54Step.fullSlot, by decide⟩

def finalFull (offset : Nat) : Expr :=
  First54Step.output (positionOffset offset (candidateCount - 1)) fullSlot

def finalAssertion (offset : Nat) : Op :=
  .assertZero (finalFull offset - 1)

def opsAt (interface : Interface) (offset : Nat) : List Op :=
  roundOpsPrefix interface offset candidateCount ++ [finalAssertion offset]

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), finalOffset offset, opsAt interface offset)

def output (offset : Nat) (slot : Fin outputCount) : Expr :=
  First54ValueStep.output
    (valueOffset offset (candidateCount - 1)) slot

structure Assumptions (interface : Interface) (offset : Nat)
    (env : Env) : Prop where
  acceptedBelow : ∀ candidate,
    (interface.accepted offset candidate).VarsBelow offset
  symbolBelow : ∀ candidate,
    (interface.symbol offset candidate).VarsBelow offset
  acceptedBoolean : ∀ candidate,
    (interface.accepted offset candidate).eval env = 0 ∨
      (interface.accepted offset candidate).eval env = 1

structure SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop where
  position : ∀ round : Fin candidateCount,
    First54Step.SpecHolds (positionInterface interface offset round.val)
      (positionOffset offset round.val) env
  value : ∀ round : Fin candidateCount,
    First54ValueStep.SpecHolds (valueInterface interface offset round.val)
      (valueOffset offset round.val) env
  full : (finalFull offset).eval env = 1

private theorem priorPositionBelow (offset round : Nat)
    (slot : Fin First54Step.slotCount) :
    (priorPosition offset round slot).VarsBelow (positionOffset offset round) := by
  cases round with
  | zero =>
      simp only [priorPosition]
      unfold initialPosition
      split <;> trivial
  | succ previous =>
      simp only [priorPosition, First54Step.output, Expr.VarsBelow]
      simp [positionOffset, roundPrivateCount, First54Step.slotCount,
        First54ValueStep.outputCount]
      omega

private theorem priorOutputBelow (offset round : Nat)
    (slot : Fin First54ValueStep.outputCount) :
    (priorOutput offset round slot).VarsBelow (positionOffset offset round) := by
  cases round with
  | zero =>
      simp [priorOutput, Expr.VarsBelow, positionOffset]
  | succ previous =>
      simp only [priorOutput, First54ValueStep.output, Expr.VarsBelow]
      simp [valueOffset, positionOffset, roundPrivateCount,
        First54Step.slotCount, First54ValueStep.outputCount]
      omega

theorem positionAssumptions (interface : Interface) (offset round : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    First54Step.Assumptions (positionInterface interface offset round)
      (positionOffset offset round) env := by
  constructor
  · apply Expr.VarsBelow.mono _
      (assumptions.acceptedBelow (candidateIndex round))
    simp [positionOffset]
  · exact priorPositionBelow offset round
  · exact assumptions.acceptedBoolean (candidateIndex round)

theorem valueAssumptions (interface : Interface) (offset round : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    First54ValueStep.Assumptions (valueInterface interface offset round)
      (valueOffset offset round) env := by
  constructor
  · apply Expr.VarsBelow.mono _
      (assumptions.acceptedBelow (candidateIndex round))
    simp [valueOffset, positionOffset]
    omega
  · apply Expr.VarsBelow.mono _
      (assumptions.symbolBelow (candidateIndex round))
    simp [valueOffset, positionOffset]
    omega
  · intro slot
    apply Expr.VarsBelow.mono _ (priorPositionBelow offset round slot)
    simp [valueOffset]
  · intro slot
    apply Expr.VarsBelow.mono _ (priorOutputBelow offset round slot)
    simp [valueOffset]

private theorem positionChildLength (interface : Interface)
    (offset round : Nat) :
    localLength (Circuit.ops (positionCircuit interface offset round).main
      (positionOffset offset round)) = First54Step.slotCount := by
  exact First54Step.localLength_eq _ _

private theorem valueChildLength (interface : Interface)
    (offset round : Nat) :
    localLength (Circuit.ops (valueCircuit interface offset round).main
      (valueOffset offset round)) = First54ValueStep.outputCount := by
  exact First54ValueStep.localLength_eq _ _

@[simp] private theorem positionOp_localLength (interface : Interface)
    (offset round : Nat) :
    (positionOp interface offset round).localLength =
      First54Step.slotCount := by
  rw [positionOp, Sequence.childOp_localLength]
  exact positionChildLength interface offset round

@[simp] private theorem valueOp_localLength (interface : Interface)
    (offset round : Nat) :
    (valueOp interface offset round).localLength =
      First54ValueStep.outputCount := by
  rw [valueOp, Sequence.childOp_localLength]
  exact valueChildLength interface offset round

@[simp] private theorem positionOp_rowCount (interface : Interface)
    (offset round : Nat) :
    (positionOp interface offset round).rowCount = First54Step.slotCount := by
  rfl

@[simp] private theorem valueOp_rowCount (interface : Interface)
    (offset round : Nat) :
    (valueOp interface offset round).rowCount =
      First54ValueStep.outputCount := by
  rfl

private theorem localLength_roundOpsPrefix (interface : Interface)
    (offset count : Nat) :
    localLength (roundOpsPrefix interface offset count) =
      count * roundPrivateCount := by
  induction count with
  | zero => simp [roundOpsPrefix, localLength]
  | succ count inductionHypothesis =>
      simp only [roundOpsPrefix] at inductionHypothesis ⊢
      rw [List.range_succ, List.flatMap_append,
        Sequence.localLength_append, inductionHypothesis]
      simp [roundOps, localLength, roundPrivateCount,
        First54Step.slotCount, First54ValueStep.outputCount]
      omega

private theorem rowCount_roundOpsPrefix (interface : Interface)
    (offset count : Nat) :
    rowCount (roundOpsPrefix interface offset count) =
      count * roundPrivateCount := by
  induction count with
  | zero => simp [roundOpsPrefix, rowCount]
  | succ count inductionHypothesis =>
      simp only [roundOpsPrefix] at inductionHypothesis ⊢
      rw [List.range_succ, List.flatMap_append, rowCount_append,
        inductionHypothesis]
      simp [roundOps, roundPrivateCount, rowCount,
        First54Step.slotCount, First54ValueStep.outputCount]
      omega

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (main interface) offset) = logicalPrivateCount := by
  change localLength (opsAt interface offset) = logicalPrivateCount
  rw [opsAt, Sequence.localLength_append,
    localLength_roundOpsPrefix]
  simp [finalAssertion, Op.localLength, logicalPrivateCount]

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (main interface) offset)).length =
      logicalRowCount := by
  rw [flatConstraints_length_eq_rowCount]
  change rowCount (opsAt interface offset) = logicalRowCount
  rw [opsAt, rowCount_append, rowCount_roundOpsPrefix]
  have finalRows : rowCount [finalAssertion offset] = 1 := by
    rfl
  rw [finalRows]
  rfl

private theorem positionScope (interface : Interface) (offset round : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (positionCircuit interface offset round).main
        (positionOffset offset round)),
      expression.VarsBelow
        (positionOffset offset round + First54Step.slotCount) := by
  have scope := First54Step.flatConstraints_varsBelow
    (positionInterface interface offset round) (positionOffset offset round)
      env (positionAssumptions interface offset round env assumptions)
  simpa [positionCircuit, First54Step.circuit] using scope

private theorem valueScope (interface : Interface) (offset round : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (valueCircuit interface offset round).main
        (valueOffset offset round)),
      expression.VarsBelow
        (valueOffset offset round + First54ValueStep.outputCount) := by
  have scope := First54ValueStep.flatConstraints_varsBelow
    (valueInterface interface offset round) (valueOffset offset round)
      env (valueAssumptions interface offset round env assumptions)
  simpa [valueCircuit, First54ValueStep.circuit] using scope

private theorem round_lt_of_member {round count : Nat}
    (member : round ∈ List.range count) : round < count :=
  List.mem_range.mp member

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    SpecHolds interface offset env := by
  change holds env (opsAt interface offset) at rows
  constructor
  · intro round
    have member : positionOp interface offset round.val ∈
        opsAt interface offset := by
      apply List.mem_append_left
      apply List.mem_flatMap.mpr
      exact ⟨round.val, List.mem_range.mpr round.isLt, by simp [roundOps]⟩
    have callHolds := rows _ member
    change First54Step.Assumptions
        (positionInterface interface offset round.val)
          (positionOffset offset round.val) env →
      First54Step.SpecHolds (positionInterface interface offset round.val)
        (positionOffset offset round.val) env at callHolds
    exact callHolds
      (positionAssumptions interface offset round.val env assumptions)
  · intro round
    have member : valueOp interface offset round.val ∈
        opsAt interface offset := by
      apply List.mem_append_left
      apply List.mem_flatMap.mpr
      exact ⟨round.val, List.mem_range.mpr round.isLt, by simp [roundOps]⟩
    have callHolds := rows _ member
    change First54ValueStep.Assumptions
        (valueInterface interface offset round.val)
          (valueOffset offset round.val) env →
      First54ValueStep.SpecHolds (valueInterface interface offset round.val)
        (valueOffset offset round.val) env at callHolds
    exact callHolds
      (valueAssumptions interface offset round.val env assumptions)
  · have assertionMember : finalAssertion offset ∈ opsAt interface offset :=
      List.mem_append_right _ (by simp)
    have assertionHolds := rows (finalAssertion offset) assertionMember
    change (finalFull offset - 1).eval env = 0 at assertionHolds
    apply sub_eq_zero.mp
    simpa only [Expr.eval_sub] using assertionHolds

set_option maxRecDepth 100000 in -- fixed-size: one 64-candidate selector, not artifact data
private theorem holdsFlat_of_spec (interface : Interface) (env : Env)
    (offset : Nat) (specification : SpecHolds interface offset env) :
    holdsFlat env (opsAt interface offset) := by
  unfold holdsFlat
  intro expression member
  rcases List.mem_flatMap.mp member with ⟨operation, operationMember,
    expressionMember⟩
  rcases List.mem_append.mp operationMember with roundMember | finalMember
  · rcases List.mem_flatMap.mp roundMember with
      ⟨round, roundInRange, operationInRound⟩
    have roundLt := round_lt_of_member roundInRange
    let roundFin : Fin candidateCount := ⟨round, by
      simpa [candidateCount] using roundLt⟩
    simp [roundOps] at operationInRound
    rcases operationInRound with rfl | rfl
    · have childRows := First54Step.holdsFlat_of_spec
        (positionInterface interface offset round) env
          (positionOffset offset round) (specification.position roundFin)
      apply childRows expression
      simpa [positionOp, Sequence.childOp] using expressionMember
    · have childRows := First54ValueStep.holdsFlat_of_spec
        (valueInterface interface offset round) env
          (valueOffset offset round) (specification.value roundFin)
      apply childRows expression
      simpa [valueOp, Sequence.childOp] using expressionMember
  · simp only [List.mem_singleton] at finalMember
    subst operation
    simp only [finalAssertion, Op.flatConstraints, List.mem_singleton] at expressionMember
    rw [expressionMember]
    calc
      (finalFull offset - 1).eval env =
          (finalFull offset).eval env - (1 : F) := Expr.eval_sub _ _ _
      _ = 0 := sub_eq_zero.mpr specification.full

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (_assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  refine ⟨env, ?_, ?_⟩
  · intro _ _
    rfl
  · exact holdsFlat_of_spec interface env offset specification

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (main interface) offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (main interface) offset)) := by
  change ∀ expression ∈ flatConstraints (opsAt interface offset),
    expression.VarsBelow (offset + localLength (opsAt interface offset))
  have lengthEq : localLength (opsAt interface offset) =
      logicalPrivateCount := by
    simpa using localLength_eq interface offset
  rw [lengthEq]
  intro expression member
  rcases List.mem_flatMap.mp member with ⟨operation, operationMember,
    expressionMember⟩
  rcases List.mem_append.mp operationMember with roundMember | finalMember
  · rcases List.mem_flatMap.mp roundMember with
      ⟨round, roundInRange, operationInRound⟩
    have roundLt := round_lt_of_member roundInRange
    have roundBound : round < 64 := by
      simpa [candidateCount] using roundLt
    simp [roundOps] at operationInRound
    rcases operationInRound with rfl | rfl
    · apply Expr.VarsBelow.mono expression
        (positionScope interface offset round env assumptions expression (by
          simpa [positionOp, Sequence.childOp] using expressionMember))
      simp [positionOffset, logicalPrivateCount, candidateCount,
        roundPrivateCount, First54Step.slotCount,
        First54ValueStep.outputCount]
      omega
    · apply Expr.VarsBelow.mono expression
        (valueScope interface offset round env assumptions expression (by
          simpa [valueOp, Sequence.childOp] using expressionMember))
      simp [valueOffset, positionOffset, logicalPrivateCount, candidateCount,
        roundPrivateCount, First54Step.slotCount,
        First54ValueStep.outputCount]
      omega
  · simp only [List.mem_singleton] at finalMember
    subst operation
    simp only [finalAssertion, Op.flatConstraints, List.mem_singleton] at expressionMember
    subst expression
    unfold finalFull First54Step.output
    exact Expr.VarsBelow.sub _ _ _ (by
      simp [fullSlot, Expr.VarsBelow, positionOffset, logicalPrivateCount,
        candidateCount, roundPrivateCount, First54Step.fullSlot,
        First54Step.slotCount, First54ValueStep.outputCount]) trivial

theorem assumptionsAtPrefix (interface : Interface) (env : Env)
    (offset : Nat) (assumptions : Assumptions interface offset env)
    (completedPrefix : Sequence.Prefix env offset) :
    Assumptions interface offset completedPrefix.current := by
  have agreeBelow : ∀ index, index < offset →
      completedPrefix.current index = env index := by
    intro index below
    exact completedPrefix.agrees index (Or.inl below)
  constructor
  · exact assumptions.acceptedBelow
  · exact assumptions.symbolBelow
  · intro candidate
    have evaluation := Expr.eval_eq_of_agree_below
      (interface.accepted offset candidate) offset completedPrefix.current env
        (assumptions.acceptedBelow candidate) agreeBelow
    rw [evaluation]
    exact assumptions.acceptedBoolean candidate

/-- Deterministically complete the first `count` position/value child pairs.
The prefix contains no final shortfall assertion. -/
theorem completeRounds (interface : Interface) (env : Env) (offset count : Nat)
    (assumptions : Assumptions interface offset env)
    (bounded : count ≤ candidateCount) :
    ∃ completed : Sequence.Prefix env offset,
      completed.operations = roundOpsPrefix interface offset count := by
  induction count with
  | zero =>
      exact ⟨Sequence.empty env offset, rfl⟩
  | succ count inductionHypothesis =>
      have countLt : count < candidateCount := by omega
      rcases inductionHypothesis (by omega) with ⟨before, beforeOps⟩
      have beforeAssumptions := assumptionsAtPrefix interface env offset
        assumptions before
      have positionStart : offset + localLength before.operations =
          positionOffset offset count := by
        rw [beforeOps, localLength_roundOpsPrefix]
        simp [positionOffset]
      have positionChildAssumptions := positionAssumptions interface offset count
        before.current beforeAssumptions
      rcases First54Step.complete (positionInterface interface offset count)
          before.current (positionOffset offset count)
          positionChildAssumptions with
        ⟨positionEnv, positionAgrees, positionRows⟩
      rcases Sequence.appendBuiltAt before (positionName count)
          (positionCircuit interface offset count) (positionOffset offset count)
          positionStart (positionScope interface offset count before.current
            beforeAssumptions) positionEnv positionAgrees positionRows with
        ⟨afterPosition, positionOps, positionEnd, _, _⟩
      have afterPositionAssumptions := assumptionsAtPrefix interface env offset
        assumptions afterPosition
      have valueStart : offset + localLength afterPosition.operations =
          valueOffset offset count := by
        rw [positionEnd, positionChildLength]
        simp [valueOffset]
      have valueChildAssumptions := valueAssumptions interface offset count
        afterPosition.current afterPositionAssumptions
      rcases First54ValueStep.complete (valueInterface interface offset count)
          afterPosition.current (valueOffset offset count)
          valueChildAssumptions with
        ⟨valueEnv, valueAgrees, valueRows⟩
      rcases Sequence.appendBuiltAt afterPosition (valueName count)
          (valueCircuit interface offset count) (valueOffset offset count)
          valueStart (valueScope interface offset count afterPosition.current
            afterPositionAssumptions) valueEnv valueAgrees valueRows with
        ⟨completed, valueOps, _, _, _⟩
      refine ⟨completed, ?_⟩
      rw [valueOps, positionOps, beforeOps]
      simp [roundOpsPrefix, List.range_succ, roundOps, positionOp, valueOp]

def circuit (interface : Interface) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  privateCount := fun _ => logicalPrivateCount
  rowCount := fun _ => logicalRowCount
  privateCount_eq := localLength_eq interface
  rowCount_eq := flatConstraints_length interface
  soundness := soundness interface
  completeness := completeness interface

end NightstreamFPrime.Gadgets.Sampling.First54
