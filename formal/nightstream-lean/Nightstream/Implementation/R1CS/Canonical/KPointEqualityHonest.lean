import Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport
import Nightstream.Implementation.R1CS.Canonical.KMulChainHonest

/-!
Contract: honest completeness for the canonical point-equality row program.

Owns: one assignment that first realizes every independent affine factor and
then realizes their product chain, plus preservation of every source column
below the allocation base.

Does not own: semantic selection of the two points.  The caller supplies only
syntactic source-placement facts; the row program computes the equality
polynomial from those sources.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KPointEqualityHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHornerHonest
open Nightstream.Implementation.R1CS.Canonical.KHornerSupport
open Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport

/-! ## Independent factor witnesses -/

def factorRowsFor {variables : Nat}
    (input : KPointEquality.Input variables)
    (entries : List (Fin variables)) : List Row :=
  entries.flatMap fun index =>
    KMul.rows (input.left index) (KPointEquality.slope input index)
      (KPointEquality.factorFrame input index)

/-- Apply the independent factor writes from right to left.  The order is
irrelevant to their values because all operands precede `frameBase`; using a
right fold makes the induction follow the emitted `flatMap` directly. -/
def factorWitness {variables : Nat}
    (assignment : Nat → Nat) (input : KPointEquality.Input variables) :
    List (Fin variables) → (Nat → Nat)
  | [] => assignment
  | index :: rest =>
      KMulHonest.witness (factorWitness assignment input rest)
        (input.left index) (KPointEquality.slope input index)
        (KPointEquality.factorFrame input index)

theorem factorWitness_off_below
    {variables : Nat}
    (assignment : Nat → Nat) (input : KPointEquality.Input variables) :
    ∀ (entries : List (Fin variables)) (column : Nat),
      column < input.frameBase →
      factorWitness assignment input entries column = assignment column
  | [], _, _ => rfl
  | index :: rest, column, below => by
      rw [factorWitness,
        KMulHonest.witness_off_frame
          (factorWitness assignment input rest)
          (input.left index) (KPointEquality.slope input index)
          (KPointEquality.factorFrame input index) column]
      · exact factorWitness_off_below assignment input rest column below
      all_goals
        simp only [KPointEquality.factorFrame, KFrames.frameAt,
          KFrames.frameColumn, KFrames.columnsPerFrame]
        omega

private theorem intercept_below
    {variables : Nat}
    (input : KPointEquality.Input variables)
    (positive : 0 < input.frameBase)
    (rightBelow :
      ∀ index, CarriedBelow (input.right index) input.frameBase)
    (index : Fin variables) :
    CarriedBelow (KPointEquality.intercept input index) input.frameBase := by
  unfold KPointEquality.intercept KLinear.oneMinus
  exact sub_below (one_below input.frameBase positive) (rightBelow index)

private theorem slope_below
    {variables : Nat}
    (input : KPointEquality.Input variables)
    (positive : 0 < input.frameBase)
    (rightBelow :
      ∀ index, CarriedBelow (input.right index) input.frameBase)
    (index : Fin variables) :
    CarriedBelow (KPointEquality.slope input index) input.frameBase := by
  unfold KPointEquality.slope
  exact sub_below (rightBelow index)
    (intercept_below input positive rightBelow index)

private theorem tail_row_misses_head_frame
    {variables : Nat}
    (input : KPointEquality.Input variables)
    (positive : 0 < input.frameBase)
    (leftBelow :
      ∀ index, CarriedBelow (input.left index) input.frameBase)
    (rightBelow :
      ∀ index, CarriedBelow (input.right index) input.frameBase)
    (head : Fin variables) (rest : List (Fin variables))
    (headNotMem : head ∉ rest)
    (row : Row) (member : row ∈ factorRowsFor input rest)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨ Mentions row.c column) :
    column ≠ (KPointEquality.factorFrame input head).lowLow ∧
      column ≠ (KPointEquality.factorFrame input head).highHigh ∧
      column ≠ (KPointEquality.factorFrame input head).cross := by
  rcases List.mem_flatMap.1 member with
    ⟨other, otherMember, rowMember⟩
  have different : head.val ≠ other.val := by
    intro equal
    have same : head = other := Fin.ext equal
    subst other
    exact headNotMem otherMember
  rcases KMulOwnership.rows_conservation
      (input.left other) (KPointEquality.slope input other)
      (KPointEquality.factorFrame input other)
      row rowMember column mentioned with inOperand | inFrame
  · have below : column < input.frameBase := by
      rcases inOperand with leftLow | leftHigh | slopeLow | slopeHigh
      · exact (leftBelow other).1 column leftLow
      · exact (leftBelow other).2 column leftHigh
      · exact (slope_below input positive rightBelow other).1 column slopeLow
      · exact (slope_below input positive rightBelow other).2 column slopeHigh
    constructor
    · simp only [KPointEquality.factorFrame, KFrames.frameAt,
        KFrames.frameColumn, KFrames.columnsPerFrame]
      omega
    constructor <;>
      simp only [KPointEquality.factorFrame, KFrames.frameAt,
        KFrames.frameColumn, KFrames.columnsPerFrame] <;> omega
  · rcases inFrame with rfl | rfl | rfl
    all_goals
      constructor
      · simp only [KPointEquality.factorFrame, KFrames.frameAt,
          KFrames.frameColumn, KFrames.columnsPerFrame]
        omega
      constructor <;>
        simp only [KPointEquality.factorFrame, KFrames.frameAt,
          KFrames.frameColumn, KFrames.columnsPerFrame] <;> omega

theorem factorRowsFor_honest
    {variables : Nat}
    (assignment : Nat → Nat) (input : KPointEquality.Input variables)
    (positive : 0 < input.frameBase)
    (leftBelow :
      ∀ index, CarriedBelow (input.left index) input.frameBase)
    (rightBelow :
      ∀ index, CarriedBelow (input.right index) input.frameBase) :
    ∀ (entries : List (Fin variables)), entries.Nodup →
      Satisfies (factorRowsFor input entries)
        (factorWitness assignment input entries)
  | [], _ => by
      intro row member
      simp [factorRowsFor] at member
  | head :: rest, nodup => by
      have restNodup : rest.Nodup := (List.nodup_cons.1 nodup).2
      have headNotMem : head ∉ rest := (List.nodup_cons.1 nodup).1
      let tailWitness := factorWitness assignment input rest
      let finalWitness :=
        KMulHonest.witness tailWitness
          (input.left head) (KPointEquality.slope input head)
          (KPointEquality.factorFrame input head)
      have headSatisfied :
          Satisfies
            (KMul.rows (input.left head) (KPointEquality.slope input head)
              (KPointEquality.factorFrame input head))
            finalWitness := by
        exact KMulHonest.witness_satisfies tailWitness
          (input.left head) (KPointEquality.slope input head)
          (KPointEquality.factorFrame input head)
          (KMulHonest.canonical_distinct input.frameBase head.val)
          (KMulChainHonest.fresh_of_belowStep _
            input.frameBase head.val
            (KMulChainHonest.belowBase_to_belowStep _
              input.frameBase head.val (leftBelow head).1))
          (KMulChainHonest.fresh_of_belowStep _
            input.frameBase head.val
            (KMulChainHonest.belowBase_to_belowStep _
              input.frameBase head.val (leftBelow head).2))
          (KMulChainHonest.fresh_of_belowStep _
            input.frameBase head.val
            (KMulChainHonest.belowBase_to_belowStep _
              input.frameBase head.val
              (slope_below input positive rightBelow head).1))
          (KMulChainHonest.fresh_of_belowStep _
            input.frameBase head.val
            (KMulChainHonest.belowBase_to_belowStep _
              input.frameBase head.val
              (slope_below input positive rightBelow head).2))
      have tailSatisfied :
          Satisfies (factorRowsFor input rest) tailWitness :=
        factorRowsFor_honest assignment input positive leftBelow rightBelow
          rest restNodup
      have tailPreserved :
          Satisfies (factorRowsFor input rest) finalWitness := by
        apply satisfies_extend _ tailWitness finalWitness
        · intro row member column mentioned
          symm
          apply KMulHonest.witness_off_frame
          · exact
              (tail_row_misses_head_frame input positive leftBelow rightBelow
                head rest headNotMem row member column mentioned).1
          · exact
              (tail_row_misses_head_frame input positive leftBelow rightBelow
                head rest headNotMem row member column mentioned).2.1
          · exact
              (tail_row_misses_head_frame input positive leftBelow rightBelow
                head rest headNotMem row member column mentioned).2.2
        · exact tailSatisfied
      intro row member
      change RowHolds finalWitness row
      simp only [factorRowsFor, List.flatMap_cons, List.mem_append] at member
      exact member.elim (headSatisfied row) (tailPreserved row)

private theorem nodup_ofFn_of_injective
    {α : Type} :
    ∀ {n : Nat} (function : Fin n → α),
      Function.Injective function → (List.ofFn function).Nodup
  | 0, _, _ => by simp
  | _ + 1, function, injective => by
      rw [List.ofFn_succ, List.nodup_cons]
      constructor
      · intro member
        rcases List.mem_ofFn.mp member with ⟨index, equal⟩
        exact Fin.succ_ne_zero index (injective equal)
      · exact nodup_ofFn_of_injective
          (fun index => function index.succ)
          (fun first second equal => Fin.succ_inj.mp (injective equal))

private theorem nodup_ofFn_id (variables : Nat) :
    (List.ofFn (fun index : Fin variables => index)).Nodup :=
  nodup_ofFn_of_injective _ (fun _ _ equal => equal)

/-! ## Product-chain assembly -/

private theorem factor_below_productBase
    {variables : Nat}
    (input : KPointEquality.Input variables)
    (positive : 0 < input.frameBase)
    (rightBelow :
      ∀ index, CarriedBelow (input.right index) input.frameBase)
    (index : Fin variables) :
    CarriedBelow (KPointEquality.factor input index)
      (KPointEquality.productBase input) := by
  unfold KPointEquality.factor
  apply add_below
  · exact carried_mono (intercept_below input positive rightBelow index)
      (by unfold KPointEquality.productBase; omega)
  · unfold KPointEquality.factorProduct
    apply frame_output_below
    unfold KPointEquality.productBase
    omega

private theorem all_factors_below_productBase
    {variables : Nat}
    (input : KPointEquality.Input variables)
    (positive : 0 < input.frameBase)
    (rightBelow :
      ∀ index, CarriedBelow (input.right index) input.frameBase) :
    ∀ factor ∈ KPointEquality.factors input,
      CarriedBelow factor (KPointEquality.productBase input) := by
  intro factor member
  rcases List.mem_map.1 member with ⟨index, _, rfl⟩
  exact factor_below_productBase input positive rightBelow index

private theorem factorRows_below_productBase
    {variables : Nat}
    (input : KPointEquality.Input variables)
    (positive : 0 < input.frameBase)
    (leftBelow :
      ∀ index, CarriedBelow (input.left index) input.frameBase)
    (rightBelow :
      ∀ index, CarriedBelow (input.right index) input.frameBase) :
    RowsBelow (KPointEquality.factorRows input)
      (KPointEquality.productBase input) := by
  intro row member column mentioned
  rcases List.mem_flatMap.1 member with ⟨index, _, rowMember⟩
  apply mul_rows_below
    (input.left index) (KPointEquality.slope input index)
    input.frameBase index.val (KPointEquality.productBase input)
  · exact carried_mono (leftBelow index)
      (by unfold KPointEquality.productBase; omega)
  · exact carried_mono (slope_below input positive rightBelow index)
      (by unfold KPointEquality.productBase; omega)
  · unfold KPointEquality.productBase
    omega
  · exact rowMember
  · exact mentioned

def witness {variables : Nat}
    (assignment : Nat → Nat) (input : KPointEquality.Input variables) :
    Nat → Nat :=
  let factorsAssignment :=
    factorWitness assignment input (KPointEquality.indices variables)
  match KPointEquality.factors input with
  | [] => factorsAssignment
  | first :: rest =>
      KMulChainHonest.witness factorsAssignment first rest
        (KPointEquality.productBase input) 0

theorem witness_off_block
    {variables : Nat}
    (assignment : Nat → Nat) (input : KPointEquality.Input variables)
    (column : Nat) (below : column < input.frameBase) :
    witness assignment input column = assignment column := by
  unfold witness
  generalize equal : KPointEquality.factors input = factors
  cases factors with
  | nil =>
    exact factorWitness_off_below assignment input
      (KPointEquality.indices variables) column below
  | cons first rest =>
    change
      KMulChainHonest.witness
          (factorWitness assignment input (KPointEquality.indices variables))
          first rest (KPointEquality.productBase input) 0 column =
        assignment column
    rw [KMulChainHonest.witness_off_before
      (factorWitness assignment input (KPointEquality.indices variables))
      first (KPointEquality.productBase input) rest 0 column
      (by unfold KPointEquality.productBase; omega)]
    exact factorWitness_off_below assignment input
      (KPointEquality.indices variables) column below

private theorem satisfies_append
    {left right : List Row} {assignment : Nat → Nat}
    (leftSatisfied : Satisfies left assignment)
    (rightSatisfied : Satisfies right assignment) :
    Satisfies (left ++ right) assignment := by
  intro row member
  exact (List.mem_append.1 member).elim
    (leftSatisfied row) (rightSatisfied row)

/-- Every authoritative pair of source points has one satisfying execution.
No equality value or paper conclusion is supplied as a premise. -/
theorem rows_honest
    {variables : Nat}
    (assignment : Nat → Nat) (input : KPointEquality.Input variables)
    (positive : 0 < input.frameBase)
    (leftBelow :
      ∀ index, CarriedBelow (input.left index) input.frameBase)
    (rightBelow :
      ∀ index, CarriedBelow (input.right index) input.frameBase) :
    Satisfies (KPointEquality.rows input) (witness assignment input) := by
  let factorsAssignment :=
    factorWitness assignment input (KPointEquality.indices variables)
  have factorSatisfied :
      Satisfies (KPointEquality.factorRows input) factorsAssignment := by
    change Satisfies
      (factorRowsFor input (KPointEquality.indices variables))
      factorsAssignment
    exact factorRowsFor_honest assignment input positive leftBelow rightBelow
      (KPointEquality.indices variables) (nodup_ofFn_id variables)
  unfold witness
  split
  next empty =>
    have noProduct :
        Satisfies (KPointEquality.productRows input) factorsAssignment := by
      intro row member
      simp [KPointEquality.productRows, empty] at member
    exact satisfies_append factorSatisfied noProduct
  next first rest equal =>
    let finalAssignment :=
      KMulChainHonest.witness factorsAssignment first rest
        (KPointEquality.productBase input) 0
    have productSatisfied :
        Satisfies (KPointEquality.productRows input) finalAssignment := by
      have firstBelow :=
        all_factors_below_productBase input positive rightBelow first (by
          rw [equal]
          exact List.mem_cons_self)
      have restBelow :
          ∀ factor ∈ rest,
            CarriedBelow factor (KPointEquality.productBase input) := by
        intro factor member
        exact all_factors_below_productBase input positive rightBelow
          factor (by
            rw [equal]
            exact List.mem_cons_of_mem first member)
      simpa [KPointEquality.productRows, equal, finalAssignment] using
        KMulChainHonest.witness_satisfies_from_base
          factorsAssignment first rest (KPointEquality.productBase input)
          firstBelow.1 firstBelow.2 restBelow
    have factorPreserved :
        Satisfies (KPointEquality.factorRows input) finalAssignment := by
      apply satisfies_extend _ factorsAssignment finalAssignment
      · intro row member column mentioned
        symm
        apply KMulChainHonest.witness_off_before
        exact factorRows_below_productBase input positive leftBelow rightBelow
          row member column mentioned
      · exact factorSatisfied
    exact satisfies_append factorPreserved productSatisfied

end Nightstream.Implementation.R1CS.Canonical.KPointEqualityHonest
