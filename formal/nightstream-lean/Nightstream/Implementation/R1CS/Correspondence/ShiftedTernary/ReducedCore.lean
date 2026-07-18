import Nightstream.Implementation.R1CS.Correspondence.ShiftedTernary.ShiftedTernaryComplete

/-!
Contract: model-level 123-gate core for one balanced Goldilocks opening.

Owns: the common centered-unit gate, the retained negative-indicator
definition, the retained borrow transition, and proofs that the omitted
opening obligations follow.

Does not own: production row selection, encoded row indices, witness
materialization, SIS commitment binding, or authorization to delete rows.

Emits constraints: no. `gates` is a model schedule, not a production artifact.

Authority boundary: reconstruction may be omitted only when
`SharedFieldDigitAlias` is supplied by a separately checked lowering. A digest
or a self-consistent witness is not such an alias.

| Core branch | Count | Mathematical obligation | Omitted obligations proved here |
|---|---:|---|---|
| `centeredUnits` | 41 | `d^3 - d = 0` in Goldilocks | exact centered alphabet |
| `negativeDefinitions` | 41 | `d(d-1) = 2n` | `n` bitness and `n(d+1)=0` |
| `borrowTransitions` | 41 | radix-three comparison step | 40 internal borrow bitness rows |
| shared alias | 0 | field slot decodes from the same 41 digits | reconstruction row |
-/

namespace Nightstream.Implementation.R1CS.ShiftedTernaryReducedCore

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernarySound
open Nightstream.Implementation.R1CS.ShiftedTernaryComplete

set_option maxRecDepth 262144

/-- Exact polynomial used by the common gadget-native centered-unit gate:
`d^3 - d = 0`, with `-1` represented by `p - 1`. -/
def CenteredUnitGateHolds (value : Nat) : Prop :=
  (value ^ 3 + (goldilocksP - 1) * value) % goldilocksP = 0

instance (value : Nat) : Decidable (CenteredUnitGateHolds value) := by
  unfold CenteredUnitGateHolds
  infer_instance

/-- One heterogeneous gate in the model schedule. -/
inductive Gate where
  | centeredUnit (column : Nat)
  | r1cs (row : Row)
deriving DecidableEq, Repr

def Gate.Holds (assignment : Nat → Nat) : Gate → Prop
  | .centeredUnit column => CenteredUnitGateHolds (assignment column)
  | .r1cs row => RowHolds assignment row

instance (assignment : Nat → Nat) (gate : Gate) :
    Decidable (gate.Holds assignment) := by
  cases gate <;> simp only [Gate.Holds] <;> infer_instance

def centeredUnitGates : List Gate :=
  (List.range digitCount).map fun index =>
    .centeredUnit (ShiftedTernary.digitCols.getD index 0)

def negativeDefinitionGates : List Gate :=
  (List.range digitCount).map fun index =>
    .r1cs (negativeDefinitionRow
      (ShiftedTernary.digitCols.getD index 0)
      (ShiftedTernary.negativeCols.getD index 0))

def borrowTransitionGates : List Gate :=
  (List.range digitCount).map fun index => .r1cs (borrowRow index)

/-- The proposed model core: 41 + 41 + 41 gates. -/
def gates : List Gate :=
  centeredUnitGates ++ negativeDefinitionGates ++ borrowTransitionGates

theorem centeredUnitGates_length : centeredUnitGates.length = 41 := by
  decide

theorem negativeDefinitionGates_length :
    negativeDefinitionGates.length = 41 := by
  decide

theorem borrowTransitionGates_length :
    borrowTransitionGates.length = 41 := by
  decide

theorem gates_length : gates.length = 123 := by
  decide

/-- Compact phase predicate corresponding exactly to `gates`. -/
structure Accepts (assignment : Nat → Nat) : Prop where
  centeredUnit : ∀ index, index < digitCount →
    CenteredUnitGateHolds
      (assignment (ShiftedTernary.digitCols.getD index 0))
  negativeDefinition : ∀ index, index < digitCount →
    RowHolds assignment (negativeDefinitionRow
      (ShiftedTernary.digitCols.getD index 0)
      (ShiftedTernary.negativeCols.getD index 0))
  borrowTransition : ∀ index, index < digitCount →
    RowHolds assignment (borrowRow index)

/-- The lowering-owned alias needed to make the old reconstruction row
redundant. It says that the field slot and the 41 digit slots decode to the
same Goldilocks residue. -/
def SharedFieldDigitAlias (assignment : Nat → Nat) : Prop :=
  assignment ShiftedTernary.fieldCol % goldilocksP =
    lowValue (centeredDigit assignment) digitCount % goldilocksP

instance (assignment : Nat → Nat) :
    Decidable (SharedFieldDigitAlias assignment) := by
  unfold SharedFieldDigitAlias
  infer_instance

theorem accepts_iff_gate_schedule (assignment : Nat → Nat) :
    Accepts assignment ↔
      ∀ gate ∈ gates, gate.Holds assignment := by
  constructor
  · intro accepts gate member
    simp only [gates, List.mem_append] at member
    rcases member with headMember | transition
    · rcases headMember with centered | definition
      · rcases List.mem_map.mp centered with ⟨index, indexMember, rfl⟩
        exact accepts.centeredUnit index (List.mem_range.mp indexMember)
      · rcases List.mem_map.mp definition with ⟨index, indexMember, rfl⟩
        exact accepts.negativeDefinition index
          (List.mem_range.mp indexMember)
    · rcases List.mem_map.mp transition with ⟨index, indexMember, rfl⟩
      exact accepts.borrowTransition index
        (List.mem_range.mp indexMember)
  · intro schedule
    constructor
    · intro index indexLt
      exact schedule (.centeredUnit
        (ShiftedTernary.digitCols.getD index 0)) (by
          unfold gates
          apply List.mem_append_left
          apply List.mem_append_left
          unfold centeredUnitGates
          apply List.mem_map.mpr
          exact ⟨index, List.mem_range.mpr indexLt, rfl⟩)
    · intro index indexLt
      exact schedule (.r1cs (negativeDefinitionRow
        (ShiftedTernary.digitCols.getD index 0)
        (ShiftedTernary.negativeCols.getD index 0))) (by
          unfold gates
          apply List.mem_append_left
          apply List.mem_append_right
          unfold negativeDefinitionGates
          apply List.mem_map.mpr
          exact ⟨index, List.mem_range.mpr indexLt, rfl⟩)
    · intro index indexLt
      exact schedule (.r1cs (borrowRow index)) (by
        unfold gates
        apply List.mem_append_right
        unfold borrowTransitionGates
        apply List.mem_map.mpr
        exact ⟨index, List.mem_range.mpr indexLt, rfl⟩)

private theorem add_pred_mod_zero {value : Nat}
    (valueLt : value < goldilocksP)
    (zero : (value + (goldilocksP - 1)) % goldilocksP = 0) :
    value = 1 := by
  simp only [goldilocksP] at valueLt zero ⊢
  cases value with
  | zero =>
      simp only [Nat.zero_add] at zero
      rw [Nat.mod_eq_of_lt (by decide)] at zero
      omega
  | succ predecessor =>
      have predecessorLt : predecessor < 18446744069414584321 := by
        omega
      have sumEq :
          Nat.succ predecessor + (18446744069414584321 - 1) =
            18446744069414584321 + predecessor := by
        omega
      rw [sumEq, Nat.add_mod] at zero
      simp only [Nat.mod_self, Nat.zero_add,
        Nat.mod_eq_of_lt predecessorLt] at zero
      omega

private theorem succ_mod_zero {value : Nat}
    (valueLt : value < goldilocksP)
    (zero : (value + 1) % goldilocksP = 0) :
    value = goldilocksP - 1 := by
  simp only [goldilocksP] at valueLt zero ⊢
  have valueSuccLe : value + 1 ≤ 18446744069414584321 := by
    omega
  rcases Nat.lt_or_eq_of_le valueSuccLe with strict | equal
  · rw [Nat.mod_eq_of_lt strict] at zero
    omega
  · omega

/-- The actual common cubic gate has exactly the centered-unit roots among
canonical Goldilocks residues. -/
theorem centeredUnitGate_sound
    (prime : EuclidPrime goldilocksP)
    {value : Nat} (canonical : value < goldilocksP)
    (holds : CenteredUnitGateHolds value) :
    value = 0 ∨ value = 1 ∨ value = goldilocksP - 1 := by
  have factored :
      value * (value ^ 2 + (goldilocksP - 1)) % goldilocksP = 0 := by
    have identity :
        value * (value ^ 2 + (goldilocksP - 1)) =
          value ^ 3 + (goldilocksP - 1) * value := by
      simp [Nat.mul_add, Nat.pow_succ, Nat.mul_comm]
    rw [identity]
    simpa [CenteredUnitGateHolds] using holds
  rcases prime value (value ^ 2 + (goldilocksP - 1)) factored with
    valueZero | quadraticZero
  · left
    simpa [Nat.mod_eq_of_lt canonical] using valueZero
  · right
    have productZero :
        (value + (goldilocksP - 1)) * (value + 1) %
            goldilocksP = 0 := by
      have identity :
          (value + (goldilocksP - 1)) * (value + 1) =
            (value ^ 2 + (goldilocksP - 1)) +
              goldilocksP * value := by
        have combine :
            value + (goldilocksP - 1) * value =
              goldilocksP * value := by
          calc
            value + (goldilocksP - 1) * value =
                1 * value + (goldilocksP - 1) * value := by simp
            _ = (1 + (goldilocksP - 1)) * value := by
              rw [Nat.add_mul]
            _ = goldilocksP * value := by
              rw [show 1 + (goldilocksP - 1) = goldilocksP by
                native_decide]
        calc
          (value + (goldilocksP - 1)) * (value + 1) =
              value ^ 2 +
                (value + (goldilocksP - 1) * value) +
                  (goldilocksP - 1) := by
                    simp [Nat.add_mul, Nat.mul_add, Nat.pow_two]
                    omega
          _ = value ^ 2 + goldilocksP * value +
                (goldilocksP - 1) := by rw [combine]
          _ = (value ^ 2 + (goldilocksP - 1)) +
                goldilocksP * value := by omega
      rw [identity, Nat.add_mod, quadraticZero]
      simp
    rcases prime (value + (goldilocksP - 1)) (value + 1)
        productZero with minusOne | plusOne
    · exact Or.inl (add_pred_mod_zero canonical minusOne)
    · exact Or.inr (succ_mod_zero canonical plusOne)

theorem centeredUnitGate_complete
    {value negative : Nat} (digit : Digit value negative) :
    CenteredUnitGateHolds value := by
  cases digit with
  | neg valueEq _ => subst value; native_decide
  | zero valueEq _ => subst value; native_decide
  | pos valueEq _ => subst value; native_decide

private theorem double_mod_two {value : Nat}
    (valueLt : value < goldilocksP)
    (two : 2 * value % goldilocksP = 2) :
    value = 1 := by
  simp only [goldilocksP] at valueLt two ⊢
  by_cases below : 2 * value < 18446744069414584321
  · rw [Nat.mod_eq_of_lt below] at two
    omega
  · have modulusLe : 18446744069414584321 ≤ 2 * value :=
      Nat.le_of_not_gt below
    rw [Nat.mod_eq_sub_mod modulusLe] at two
    have reducedLt :
        2 * value - 18446744069414584321 <
          18446744069414584321 := by
      apply (Nat.sub_lt_iff_lt_add' modulusLe).mpr
      simpa [Nat.two_mul] using
        (Nat.mul_lt_mul_left (a := 2) (b := value)
          (c := 18446744069414584321) (by decide)).mpr valueLt
    rw [Nat.mod_eq_of_lt reducedLt] at two
    have impossible :
        2 * value = 18446744069414584321 + 2 :=
      (Nat.sub_eq_iff_eq_add' modulusLe).mp two
    have parity := congrArg (fun number => number % 2) impossible
    simp at parity

/-- The common cubic gate plus the retained definition row determines the
exact existing `Digit` semantics; no negative bitness or support row is used. -/
theorem digit_of_centeredUnit_and_definition
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    {digitCol negativeCol : Nat}
    (centered : CenteredUnitGateHolds (assignment digitCol))
    (definition : RowHolds assignment
      (negativeDefinitionRow digitCol negativeCol)) :
    Digit (assignment digitCol) (assignment negativeCol) := by
  have digitCases := centeredUnitGate_sound prime (canonical digitCol) centered
  have negativeLt := canonical negativeCol
  rcases digitCases with digitZero | digitOne | digitMinusOne
  · apply Digit.zero digitZero
    have equation := definition
    simp [RowHolds, negativeDefinitionRow, lcEval, one, digitZero,
      goldilocksP] at equation
    have productZero : 2 * assignment negativeCol % goldilocksP = 0 := by
      simpa [goldilocksP] using equation.symm
    rcases prime 2 (assignment negativeCol) productZero with
      impossible | negativeZero
    · exfalso
      have twoNonzero : 2 % goldilocksP ≠ 0 := by native_decide
      exact twoNonzero impossible
    · simpa [Nat.mod_eq_of_lt negativeLt] using negativeZero
  · apply Digit.pos digitOne
    have equation := definition
    simp [RowHolds, negativeDefinitionRow, lcEval, one, digitOne,
      goldilocksP] at equation
    have productZero : 2 * assignment negativeCol % goldilocksP = 0 := by
      simpa [goldilocksP] using equation.symm
    rcases prime 2 (assignment negativeCol) productZero with
      impossible | negativeZero
    · exfalso
      have twoNonzero : 2 % goldilocksP ≠ 0 := by native_decide
      exact twoNonzero impossible
    · simpa [Nat.mod_eq_of_lt negativeLt] using negativeZero
  · apply Digit.neg digitMinusOne
    have equation := definition
    have doubled : 2 * assignment negativeCol % goldilocksP = 2 := by
      simpa [RowHolds, negativeDefinitionRow, lcEval, one,
        digitMinusOne, goldilocksP] using equation.symm
    exact double_mod_two negativeLt doubled

/-- Indexed exact digit semantics supplied by the first two core branches. -/
theorem Accepts.digit
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} (accepted : Accepts assignment)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    {index : Nat} (indexLt : index < digitCount) :
    Digit
      (assignment (ShiftedTernary.digitCols.getD index 0))
      (assignment (ShiftedTernary.negativeCols.getD index 0)) :=
  digit_of_centeredUnit_and_definition prime canonical one
    (accepted.centeredUnit index indexLt)
    (accepted.negativeDefinition index indexLt)

theorem Accepts.digits
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} (accepted : Accepts assignment)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) :
    DigitsHold assignment := by
  intro pair member
  have lengths : ShiftedTernary.digitCols.length = digitCount ∧
      ShiftedTernary.negativeCols.length = digitCount := by
    decide
  rcases List.mem_iff_getElem.mp member with ⟨index, indexLt, pairEq⟩
  have digitLt : index < digitCount := by
    simpa [List.length_zip, lengths.1, lengths.2] using indexLt
  have columns :
      (ShiftedTernary.digitCols.zip ShiftedTernary.negativeCols)[index] =
        (ShiftedTernary.digitCols.getD index 0,
          ShiftedTernary.negativeCols.getD index 0) := by
    rw [List.getElem_zip]
    simp only [List.getD_eq_getElem?_getD]
    have digitColumnLt : index < ShiftedTernary.digitCols.length := by
      simpa [lengths.1] using digitLt
    have negativeColumnLt :
        index < ShiftedTernary.negativeCols.length := by
      simpa [lengths.2] using digitLt
    rw [List.getElem?_eq_getElem digitColumnLt,
      List.getElem?_eq_getElem negativeColumnLt]
    simp
  rw [← pairEq]
  exact columns.symm ▸ accepted.digit prime canonical one digitLt

/-- Negative indicators are binary consequences, not core assumptions. -/
theorem Accepts.negative_le_one
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} (accepted : Accepts assignment)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    {index : Nat} (indexLt : index < digitCount) :
    assignment (ShiftedTernary.negativeCols.getD index 0) ≤ 1 := by
  have digit := accepted.digit prime canonical one indexLt
  cases digit <;> omega

private theorem bitRow_holds_of_le_one
    {assignment : Nat → Nat} (one : assignment 0 = 1)
    {column : Nat} (bounded : assignment column ≤ 1) :
    RowHolds assignment (bitRow column) := by
  have cases : assignment column = 0 ∨ assignment column = 1 := by
    omega
  rcases cases with zero | oneValue
  · simp [RowHolds, bitRow, lcEval, one, zero]
  · simp [RowHolds, bitRow, lcEval, one, oneValue, goldilocksP]

theorem Accepts.negative_bitness_follows
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} (accepted : Accepts assignment)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    {index : Nat} (indexLt : index < digitCount) :
    RowHolds assignment
      (bitRow (ShiftedTernary.negativeCols.getD index 0)) :=
  bitRow_holds_of_le_one one
    (accepted.negative_le_one prime canonical one indexLt)

theorem Accepts.negative_support_follows
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} (accepted : Accepts assignment)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    {index : Nat} (indexLt : index < digitCount) :
    RowHolds assignment (negativeSupportRow
      (ShiftedTernary.digitCols.getD index 0)
      (ShiftedTernary.negativeCols.getD index 0)) :=
  digitSupport_complete one
    (accepted.digit prime canonical one indexLt)

/-- Borrow recursion from the verifier-fixed zero input sentinel. No internal
borrow bitness hypothesis appears in the induction. -/
theorem Accepts.borrowTrace_from_zero_sentinel
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} (accepted : Accepts assignment)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) :
    ∀ count, count ≤ digitCount →
      borrowAt assignment count =
        expectedBorrow (assignmentTrit assignment) boundDigit 0 count := by
  intro count countLe
  induction count with
  | zero => simp [borrowAt, expectedBorrow]
  | succ count inductionHypothesis =>
      have countLt : count < digitCount := by omega
      have prefixEq := inductionHypothesis (by omega)
      have currentLe : borrowAt assignment count ≤ 1 := by
        rw [prefixEq]
        exact expectedBorrow_le_one _ _ 0 (by omega) count
      have step := borrowRow_forces_step canonical one countLt
        (accepted.digit prime canonical one countLt) currentLe
        (accepted.borrowTransition count countLt)
      simpa [expectedBorrow, assignmentTrit, boundDigit, prefixEq] using step

theorem Accepts.borrowAt_le_one
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} (accepted : Accepts assignment)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    {count : Nat} (countLe : count ≤ digitCount) :
    borrowAt assignment count ≤ 1 := by
  rw [accepted.borrowTrace_from_zero_sentinel prime canonical one count countLe]
  exact expectedBorrow_le_one _ _ 0 (by omega) count

/-- Every materialized internal borrow is binary by the zero-sentinel
induction, so its separate common bitness gate is semantically redundant. -/
theorem Accepts.borrow_le_one
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} (accepted : Accepts assignment)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    {index : Nat} (indexLt : index < digitCount - 1) :
    assignment (ShiftedTernary.borrowCols.getD index 0) ≤ 1 := by
  have bounded := accepted.borrowAt_le_one prime canonical one
    (count := index + 1) (by omega)
  have nonzero : index + 1 ≠ 0 := by omega
  have nonterminal : index + 1 ≠ digitCount := by omega
  simpa [borrowAt, nonzero, nonterminal] using bounded

theorem Accepts.borrow_bitness_follows
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} (accepted : Accepts assignment)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    {index : Nat} (indexLt : index < digitCount - 1) :
    RowHolds assignment
      (bitRow (ShiftedTernary.borrowCols.getD index 0)) :=
  bitRow_holds_of_le_one one
    (accepted.borrow_le_one prime canonical one indexLt)

/-- The shared field/digit alias implies the exact old reconstruction row. -/
theorem reconstructionRow_holds_of_shared_alias
    {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (alias : SharedFieldDigitAlias assignment) :
    RowHolds assignment reconstructionRow := by
  have equation :
      (assignment ShiftedTernary.fieldCol +
        negativeWeightedValue assignment digitCount) % goldilocksP = 0 := by
    calc
      (assignment ShiftedTernary.fieldCol +
          negativeWeightedValue assignment digitCount) % goldilocksP =
          (assignment ShiftedTernary.fieldCol % goldilocksP +
            negativeWeightedValue assignment digitCount % goldilocksP) %
              goldilocksP := Nat.add_mod ..
      _ = (lowValue (centeredDigit assignment) digitCount % goldilocksP +
            negativeWeightedValue assignment digitCount % goldilocksP) %
              goldilocksP := by rw [alias]
      _ = (lowValue (centeredDigit assignment) digitCount +
            negativeWeightedValue assignment digitCount) % goldilocksP :=
              (Nat.add_mod ..).symm
      _ = (negativeWeightedValue assignment digitCount +
            lowValue (centeredDigit assignment) digitCount) % goldilocksP := by
              rw [Nat.add_comm]
      _ = 0 := negative_add_centered_mod_zero assignment digitCount Nat.le.refl
  simp only [RowHolds, reconstructionRow, lcEval, List.foldl, one,
    Nat.one_mul, Nat.mul_one, Nat.zero_add, Nat.zero_mod]
  simp only [List.foldl_map]
  have folded := foldl_range_eq_negativeWeightedValue assignment
    (assignment ShiftedTernary.fieldCol) digitCount
  unfold centeredDigit at folded
  rw [folded]
  simpa [goldilocksP] using equation

theorem Accepts.digitRows
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} (accepted : Accepts assignment)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) :
    Satisfies digitRows assignment := by
  intro row member
  unfold ShiftedTernaryCompiler.digitRows at member
  rw [List.mem_flatMap] at member
  rcases member with ⟨pair, pairMember, rowMember⟩
  have digit := accepted.digits prime canonical one pair pairMember
  simp only [List.mem_cons, List.not_mem_nil, or_false] at rowMember
  rcases rowMember with rfl | rfl
  · exact digitDefinition_complete one digit
  · exact digitSupport_complete one digit

theorem Accepts.borrowRows
    {assignment : Nat → Nat} (accepted : Accepts assignment) :
    Satisfies borrowRows assignment := by
  intro row member
  unfold ShiftedTernaryCompiler.borrowRows at member
  rw [List.mem_map] at member
  rcases member with ⟨index, indexMember, rfl⟩
  exact accepted.borrowTransition index (List.mem_range.mp indexMember)

/-- Reduced core plus the separately checked alias reconstructs every one of
the old 124 canonical-opening rows. -/
theorem canonicalRows_of_reduced
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : Accepts assignment)
    (alias : SharedFieldDigitAlias assignment) :
    Satisfies canonicalRows assignment := by
  intro row member
  unfold canonicalRows at member
  rw [List.mem_append] at member
  rcases member with prefixMember | borrowMember
  · rw [List.mem_append] at prefixMember
    rcases prefixMember with digitMember | reconstructionMember
    · exact accepted.digitRows prime canonical one row digitMember
    · simp only [List.mem_singleton] at reconstructionMember
      subst row
      exact reconstructionRow_holds_of_shared_alias one alias
  · exact accepted.borrowRows row borrowMember

/-- The old canonical rows imply the reduced core. This direction uses the
old digit proof only to recover the common cubic gate. -/
theorem reduced_of_canonicalRows
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies canonicalRows assignment) :
    Accepts assignment := by
  have digits := allDigits_sound_of_canonicalRows prime canonical one satisfies
  constructor
  · intro index indexLt
    exact centeredUnitGate_complete (digits.atIndex indexLt)
  · intro index indexLt
    exact digitDefinition_complete one (digits.atIndex indexLt)
  · intro index indexLt
    exact borrowRow_holds_of_satisfies satisfies indexLt

/-- Exact model-level equivalence to the existing full 124-row opening
semantics. The alias is explicit because it is not derivable from the 123 core
without assuming how the shared field slot is decoded. -/
theorem reduced_iff_canonicalRows
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) :
    (Accepts assignment ∧ SharedFieldDigitAlias assignment) ↔
      Satisfies canonicalRows assignment := by
  constructor
  · rintro ⟨accepted, alias⟩
    exact canonicalRows_of_reduced prime canonical one accepted alias
  · intro satisfies
    exact ⟨reduced_of_canonicalRows prime canonical one satisfies,
      field_eq_centered_mod one satisfies⟩

theorem canonicalOpening_of_reduced
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : Accepts assignment)
    (alias : SharedFieldDigitAlias assignment) :
    CanonicalOpening assignment :=
  canonicalOpening_of_canonicalRows prime canonical one
    (canonicalRows_of_reduced prime canonical one accepted alias)

/-- Honest native witness completeness for the reduced model. No production
row correspondence or row deletion is claimed. -/
theorem CanonicalWitness.reducedCore_complete
    {assignment : Nat → Nat} (witness : CanonicalWitness assignment) :
    Accepts assignment ∧ SharedFieldDigitAlias assignment := by
  constructor
  · constructor
    · intro index indexLt
      exact centeredUnitGate_complete (witness.digits.atIndex indexLt)
    · intro index indexLt
      exact digitDefinition_complete witness.one
        (witness.digits.atIndex indexLt)
    · intro index indexLt
      exact witness.borrowRow_complete indexLt
  · simpa [SharedFieldDigitAlias] using witness.field_eq_centered_mod

end Nightstream.Implementation.R1CS.ShiftedTernaryReducedCore
