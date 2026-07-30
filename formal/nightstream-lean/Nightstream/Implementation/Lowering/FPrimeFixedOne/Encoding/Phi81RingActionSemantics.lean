import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Common

/-!
Contract: exact semantic decoding of the Lean-owned Phi81 ring-action rows.

Owns:
- extraction of every raw equation from owned-row satisfaction;
- exact decoding of every schoolbook product cell;
- exact evaluation of the unreduced and reduced combinations;
- soundness of the complete row program against `Phi81RingAction.combine`.

Does not own: honest completion, freshness, placement, activation, codecs,
selected-NIFS assembly, Rust, or generated artifacts.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm

private theorem rawSatisfies_member
    {source : List Row}
    {assignment : ColumnId → F}
    (satisfied : RawSatisfies source assignment)
    {row : Row}
    (member : row ∈ source) :
    row.Holds assignment := by
  induction source with
  | nil =>
      simp at member
  | cons head tail inductionHypothesis =>
      rcases List.mem_cons.mp member with rfl | tailMember
      · exact satisfied.1
      · exact inductionHypothesis satisfied.2 tailMember

private theorem satisfies_ownRows_iff
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (source : List Row)
    (assignment : ColumnId → F) :
    Satisfies (ownRows owner firstOrdinal source) assignment ↔
      RawSatisfies source assignment := by
  induction source generalizing firstOrdinal with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      simp only [ownRows, satisfies_cons, rawSatisfies_cons,
        inductionHypothesis (firstOrdinal + 1)]

theorem satisfies_rows_iff
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F) :
    Satisfies (rows frame) assignment ↔
      RawSatisfies (rawRows frame) assignment := by
  exact satisfies_ownRows_iff frame.owner frame.firstOrdinal
    (rawRows frame) assignment

private theorem linearCombination_eval_append
    (left right : LinearCombination)
    (assignment : ColumnId → F) :
    (left ++ right).eval assignment =
      left.eval assignment + right.eval assignment := by
  induction left with
  | nil =>
      simp
  | cons term tail inductionHypothesis =>
      simp only [List.cons_append, LinearCombination.eval,
        inductionHypothesis, Lean.Grind.Fin.add_assoc]

private theorem linearCombination_eval_negate
    (combination : LinearCombination)
    (assignment : ColumnId → F) :
    (negate combination).eval assignment =
      -(combination.eval assignment) := by
  induction combination with
  | nil =>
      simp [negate, LinearCombination.eval,
        Lean.Grind.AddCommGroup.neg_zero]
  | cons term tail inductionHypothesis =>
      change
        (-term.coefficient) * assignment term.column +
            LinearCombination.eval assignment (negate tail) =
          -(term.coefficient * assignment term.column +
            LinearCombination.eval assignment tail)
      rw [inductionHypothesis, Lean.Grind.Fin.neg_mul,
        Lean.Grind.AddCommGroup.neg_add]

/-- Every product row fixes its allocated cell to the corresponding decoded
challenge/value coefficient product. -/
theorem product_exact_of_satisfied
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F)
    (satisfied : Satisfies (rows frame) assignment)
    (source : Fin count)
    (left right : Fin ringDegree) :
    assignment
        (frame.productColumn source.val left.val right.val) =
      decoded assignment (frame.challenges source) left *
        decoded assignment (frame.values source) right := by
  have rawSatisfied :=
    (satisfies_rows_iff frame assignment).mp satisfied
  have member :
      productRow frame source.val left.val right.val ∈ rawRows frame := by
    apply List.mem_append_left
    unfold productRows
    apply List.mem_flatMap.2
    refine ⟨source.val, List.mem_range.mpr source.isLt, ?_⟩
    apply List.mem_flatMap.2
    refine ⟨left.val, List.mem_range.mpr left.isLt, ?_⟩
    apply List.mem_map.2
    exact ⟨right.val, List.mem_range.mpr right.isLt, rfl⟩
  have holds := rawSatisfies_member rawSatisfied member
  simp only [productRow, dif_pos source.isLt, dif_pos left.isLt,
    dif_pos right.isLt, Row.Holds, Goldilocks.singleton,
    LinearCombination.eval, Fin.one_mul, Fin.add_zero] at holds
  simpa using holds.symm

private theorem sourceTerms_eval
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F)
    (source : Fin count)
    (degree : Nat)
    (indices : List Nat)
    (indicesBound :
      ∀ index ∈ indices, index < ringDegree)
    (products :
      ∀ left right : Fin ringDegree,
        assignment
            (frame.productColumn source.val left.val right.val) =
          decoded assignment (frame.challenges source) left *
            decoded assignment (frame.values source) right) :
    LinearCombination.eval assignment
        (sourceTerms frame source.val degree indices) =
      Product.fieldListSum indices
        (Product.rawTerm
          (decoded assignment (frame.challenges source))
          (decoded assignment (frame.values source))
          degree) := by
  induction indices with
  | nil =>
      rfl
  | cons index rest inductionHypothesis =>
      have indexLt : index < ringDegree :=
        indicesBound index (by simp)
      have restBound :
          ∀ item ∈ rest, item < ringDegree := by
        intro item member
        exact indicesBound item (by simp [member])
      change
        (if Product.supportActive degree index then
              ({ column :=
                    frame.productColumn source.val index (degree - index)
                 coefficient := 1 } : Term)
            else
              ({ column := frame.one, coefficient := 0 } : Term)).coefficient *
            assignment
              (if Product.supportActive degree index then
                  ({ column :=
                        frame.productColumn source.val index (degree - index)
                     coefficient := 1 } : Term)
                else
                  ({ column := frame.one, coefficient := 0 } : Term)).column +
          LinearCombination.eval assignment
            (sourceTerms frame source.val degree rest) =
        Product.fieldListSum (index :: rest)
          (Product.rawTerm
            (decoded assignment (frame.challenges source))
            (decoded assignment (frame.values source))
            degree)
      rw [inductionHypothesis restBound]
      by_cases active : Product.supportActive degree index
      · have rightLt : degree - index < ringDegree :=
          active.2
        rw [if_pos active,
          products ⟨index, indexLt⟩ ⟨degree - index, rightLt⟩]
        simp [Product.fieldListSum, Product.rawTerm, active,
          ringFCoeff, decoded, indexLt, rightLt, Fin.one_mul]
      · rw [if_neg active]
        simp [Product.fieldListSum, Product.rawTerm, active,
          Fin.zero_mul, Fin.zero_add]

/-- One source combination evaluates to the exact executable raw convolution
for that source. -/
theorem sourceRawCombination_eval
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F)
    (source : Fin count)
    (degree : Nat)
    (products :
      ∀ left right : Fin ringDegree,
        assignment
            (frame.productColumn source.val left.val right.val) =
          decoded assignment (frame.challenges source) left *
            decoded assignment (frame.values source) right) :
    (sourceRawCombination frame source.val degree).eval assignment =
      rawMulCoeffF
        (decoded assignment (frame.challenges source))
        (decoded assignment (frame.values source))
        degree := by
  rw [Product.rawMulCoeffF_eq_fieldListSum]
  unfold sourceRawCombination
  apply sourceTerms_eval frame assignment source degree
  · intro index member
    exact List.mem_range.mp member
  · exact products

/-- Source-wise raw convolution sum, in the same head-first recursion as
`combine`. -/
def rawProductSum :
    {count : Nat} →
      (Fin count → RingF) →
      (Fin count → RingF) →
      Nat → F
  | 0, _, _, _ => 0
  | _ + 1, challenges, values, degree =>
      rawMulCoeffF (challenges 0) (values 0) degree +
        rawProductSum
          (fun source => challenges source.succ)
          (fun source => values source.succ)
          degree

/-- The emitted raw combination denotes the exact source-wise raw convolution
sum. -/
theorem rawCombination_eval
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F)
    (products :
      ∀ source : Fin count,
        ∀ left right : Fin ringDegree,
          assignment
              (frame.productColumn source.val left.val right.val) =
            decoded assignment (frame.challenges source) left *
              decoded assignment (frame.values source) right)
    (degree : Nat) :
    (rawCombination frame degree).eval assignment =
      rawProductSum
        (fun source => decoded assignment (frame.challenges source))
        (fun source => decoded assignment (frame.values source))
        degree := by
  induction count with
  | zero =>
      rfl
  | succ count inductionHypothesis =>
      rw [rawCombination, linearCombination_eval_append]
      have headExact :=
        sourceRawCombination_eval frame assignment
          (0 : Fin (count + 1)) degree (products 0)
      have headExactNat :
          (sourceRawCombination frame 0 degree).eval assignment =
            rawMulCoeffF
              (decoded assignment (frame.challenges 0))
              (decoded assignment (frame.values 0)) degree := by
        simpa using headExact
      rw [headExactNat]
      rw [inductionHypothesis (frame := tailFrame frame)
        (products := fun source left right =>
          products source.succ left right)]
      rfl

private theorem add_sub_pair
    (a₁ a₂ b₁ b₂ : F) :
    (a₁ + a₂) - (b₁ + b₂) =
      (a₁ - b₁) + (a₂ - b₂) := by
  simp only [Fin.sub_eq_add_neg, Lean.Grind.AddCommGroup.neg_add]
  letI : Std.Associative (fun (a b : F) => a + b) :=
    ⟨Lean.Grind.Fin.add_assoc⟩
  letI : Std.Commutative (fun (a b : F) => a + b) :=
    ⟨Lean.Grind.Fin.add_comm⟩
  ac_rfl

private theorem add_sub_add_triple
    (a₁ a₂ b₁ b₂ c₁ c₂ : F) :
    (a₁ + a₂) - (b₁ + b₂) + (c₁ + c₂) =
      (a₁ - b₁ + c₁) + (a₂ - b₂ + c₂) := by
  rw [add_sub_pair]
  simp only [Fin.sub_eq_add_neg]
  letI : Std.Associative (fun (a b : F) => a + b) :=
    ⟨Lean.Grind.Fin.add_assoc⟩
  letI : Std.Commutative (fun (a b : F) => a + b) :=
    ⟨Lean.Grind.Fin.add_comm⟩
  ac_rfl

private theorem add_sub_pair_zero
    (a₁ a₂ b₁ b₂ : F) :
    (a₁ + a₂) - (b₁ + b₂) + 0 =
      (a₁ - b₁ + 0) + (a₂ - b₂ + 0) := by
  rw [Fin.add_zero, Fin.add_zero, Fin.add_zero]
  exact add_sub_pair _ _ _ _

/-- Reducing the source-wise raw sum is exactly the recursive semantic
combination. -/
theorem rawProductSum_reduces_to_combine
    {count : Nat}
    (challenges values : Fin count → RingF)
    (output : Fin ringDegree) :
    let folded :=
      if output.val < ringMiddleDegree then
        output.val + ringDegree
      else
        output.val + ringMiddleDegree
    let twice :=
      if output.val + 81 ≤ 106 then
        rawProductSum challenges values (output.val + 81)
      else
        0
    rawProductSum challenges values output.val -
        rawProductSum challenges values folded + twice =
      combine challenges values output := by
  induction count with
  | zero =>
      simp [rawProductSum, combine, ringFZero]
  | succ count inductionHypothesis =>
      simp only [rawProductSum, combine, ringFAdd]
      by_cases low : output.val < ringMiddleDegree
      · by_cases twice : output.val + 81 ≤ 106
        · simp only [if_pos low, if_pos twice]
          rw [add_sub_add_triple]
          have tailExact := inductionHypothesis
              (fun source => challenges source.succ)
              (fun source => values source.succ)
          simp only [if_pos low, if_pos twice] at tailExact
          rw [tailExact]
          simp [ringFMul, low, twice]
        · simp only [if_pos low, if_neg twice]
          rw [add_sub_pair_zero]
          have tailExact := inductionHypothesis
              (fun source => challenges source.succ)
              (fun source => values source.succ)
          simp only [if_pos low, if_neg twice] at tailExact
          rw [tailExact]
          simp [ringFMul, low, twice]
      · by_cases twice : output.val + 81 ≤ 106
        · simp only [if_neg low, if_pos twice]
          rw [add_sub_add_triple]
          have tailExact := inductionHypothesis
              (fun source => challenges source.succ)
              (fun source => values source.succ)
          simp only [if_neg low, if_pos twice] at tailExact
          rw [tailExact]
          simp [ringFMul, low, twice]
        · simp only [if_neg low, if_neg twice]
          rw [add_sub_pair_zero]
          have tailExact := inductionHypothesis
              (fun source => challenges source.succ)
              (fun source => values source.succ)
          simp only [if_neg low, if_neg twice] at tailExact
          rw [tailExact]
          simp [ringFMul, low, twice]

/-- The emitted reduced combination denotes the exact semantic source fold at
one output lane. -/
theorem reducedCombination_eval
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F)
    (products :
      ∀ source : Fin count,
        ∀ left right : Fin ringDegree,
          assignment
              (frame.productColumn source.val left.val right.val) =
            decoded assignment (frame.challenges source) left *
              decoded assignment (frame.values source) right)
    (output : Fin ringDegree) :
    (reducedCombination frame output.val).eval assignment =
      combine
        (fun source => decoded assignment (frame.challenges source))
        (fun source => decoded assignment (frame.values source))
        output := by
  unfold reducedCombination
  simp only [linearCombination_eval_append,
    linearCombination_eval_negate]
  by_cases low : output.val < ringMiddleDegree
  · by_cases twice : output.val + 81 ≤ 106
    · simp only [if_pos low, if_pos twice]
      rw [rawCombination_eval frame assignment products,
        rawCombination_eval frame assignment products,
        rawCombination_eval frame assignment products]
      simpa only [if_pos low, if_pos twice, Fin.sub_eq_add_neg,
        Lean.Grind.Fin.add_assoc] using
        rawProductSum_reduces_to_combine
          (fun source => decoded assignment (frame.challenges source))
          (fun source => decoded assignment (frame.values source))
          output
    · simp only [if_pos low, if_neg twice,
        LinearCombination.eval]
      rw [rawCombination_eval frame assignment products,
        rawCombination_eval frame assignment products]
      simpa only [if_pos low, if_neg twice, Fin.sub_eq_add_neg,
        Lean.Grind.Fin.add_assoc, Fin.add_zero] using
        rawProductSum_reduces_to_combine
          (fun source => decoded assignment (frame.challenges source))
          (fun source => decoded assignment (frame.values source))
          output
  · by_cases twice : output.val + 81 ≤ 106
    · simp only [if_neg low, if_pos twice]
      rw [rawCombination_eval frame assignment products,
        rawCombination_eval frame assignment products,
        rawCombination_eval frame assignment products]
      simpa only [if_neg low, if_pos twice, Fin.sub_eq_add_neg,
        Lean.Grind.Fin.add_assoc] using
        rawProductSum_reduces_to_combine
          (fun source => decoded assignment (frame.challenges source))
          (fun source => decoded assignment (frame.values source))
          output
    · simp only [if_neg low, if_neg twice,
        LinearCombination.eval]
      rw [rawCombination_eval frame assignment products,
        rawCombination_eval frame assignment products]
      simpa only [if_neg low, if_neg twice, Fin.sub_eq_add_neg,
        Lean.Grind.Fin.add_assoc, Fin.add_zero] using
        rawProductSum_reduces_to_combine
          (fun source => decoded assignment (frame.challenges source))
          (fun source => decoded assignment (frame.values source))
          output

/-- Satisfaction fixes one visible output lane to the exact semantic ring
action. -/
theorem output_exact_of_satisfied
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F)
    (constantOne : assignment frame.one = 1)
    (satisfied : Satisfies (rows frame) assignment)
    (output : Fin ringDegree) :
    decoded assignment frame.output output =
      combine
        (fun source => decoded assignment (frame.challenges source))
        (fun source => decoded assignment (frame.values source))
        output := by
  have rawSatisfied :=
    (satisfies_rows_iff frame assignment).mp satisfied
  have member :
      outputRow frame output.val ∈ rawRows frame := by
    apply List.mem_append_right
    unfold outputRows
    exact List.mem_map.2
      ⟨output.val, List.mem_range.mpr output.isLt, rfl⟩
  have holds := rawSatisfies_member rawSatisfied member
  have products :
      ∀ source : Fin count,
        ∀ left right : Fin ringDegree,
          assignment
              (frame.productColumn source.val left.val right.val) =
            decoded assignment (frame.challenges source) left *
              decoded assignment (frame.values source) right := by
    intro source left right
    exact product_exact_of_satisfied frame assignment satisfied
      source left right
  have reducedExact :=
    reducedCombination_eval frame assignment products output
  simp only [outputRow, dif_pos output.isLt, Row.Holds,
    Goldilocks.singleton, LinearCombination.eval, constantOne,
    Fin.mul_one, Fin.add_zero] at holds
  rw [reducedExact] at holds
  exact holds.symm

/-- **Headline soundness.** Every satisfying assignment carries exactly the
frozen semantic sum of Phi81 challenge actions on the visible output ring. -/
theorem rows_sound
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F)
    (constantOne : assignment frame.one = 1)
    (satisfied : Satisfies (rows frame) assignment) :
    decoded assignment frame.output =
      combine
        (fun source => decoded assignment (frame.challenges source))
        (fun source => decoded assignment (frame.values source)) := by
  funext output
  exact output_exact_of_satisfied frame assignment constantOne
    satisfied output

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction
