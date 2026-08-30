import NightstreamFPrime.Layout.ProductionRelation.ProductSumPlan

/-!
Owns the fixed direct product schedule for one Phi81 multiplication lane.
Each of the three raw convolutions has 54 ordered positions. Positions outside
the selected degree support are zero terms. The resulting 162 terms form 33
five-product groups and 34 rows including the final affine pin.

This module does not select concrete Stage 1 source or assignment columns.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.Phi81ProductPlan

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Layout.ProductionRelation.ProductSumPlan

abbrev State (logicalWidth : Nat) := Fin ringDegree → SparseForm logicalWidth

def evalState {logicalWidth : Nat} (assignment : Assignment F logicalWidth)
    (state : State logicalWidth) : RingF :=
  fun lane => (state lane).eval assignment

def rawTerm {logicalWidth : Nat} (coefficient : F)
    (left right : State logicalWidth) (degree source : Nat) :
    Term logicalWidth :=
  if sourceBound : source < ringDegree then
    if support : source ≤ degree ∧ degree - source < ringDegree then
      { left := SparseForm.scale coefficient (left ⟨source, sourceBound⟩)
        right := right ⟨degree - source, support.2⟩ }
    else
      .zero
  else
    .zero

/-- One fixed 54-position raw convolution. -/
def rawTerms {logicalWidth : Nat} (coefficient : F)
    (left right : State logicalWidth) (degree : Nat) :
    List (Term logicalWidth) :=
  (List.range ringDegree).map (rawTerm coefficient left right degree)

@[simp] theorem rawTerms_length {logicalWidth : Nat} (coefficient : F)
    (left right : State logicalWidth) (degree : Nat) :
    (rawTerms coefficient left right degree).length = 54 := by
  simp [rawTerms, ringDegree]

def rawProduct (left right : RingF) (degree source : Nat) : F :=
  if source ≤ degree ∧ degree - source < ringDegree then
    ringFCoeff left source * ringFCoeff right (degree - source)
  else
    0

private theorem foldl_if_add_eq_sum (term : Nat → F)
    (selected : Nat → Prop) [DecidablePred selected] :
    ∀ (indices : List Nat) (initial : F),
      indices.foldl (fun accumulated index =>
          if selected index then accumulated + term index else accumulated)
          initial =
        initial +
          (indices.map fun index =>
            if selected index then term index else 0).sum
  | [], initial => by simp
  | index :: rest, initial => by
      simp only [List.foldl_cons, List.map_cons, List.sum_cons]
      by_cases active : selected index
      · rw [if_pos active, if_pos active,
          foldl_if_add_eq_sum term selected rest (initial + term index)]
        exact baseLaws.add_assoc _ _ _
      · rw [if_neg active, if_neg active,
          foldl_if_add_eq_sum term selected rest initial]
        simp

theorem rawMulCoeffF_eq_sum (left right : RingF) (degree : Nat) :
    rawMulCoeffF left right degree =
      ((List.range ringDegree).map (rawProduct left right degree)).sum := by
  unfold rawMulCoeffF rawProduct
  have folded := foldl_if_add_eq_sum
    (fun source => ringFCoeff left source *
      ringFCoeff right (degree - source))
    (fun source => source ≤ degree ∧ degree - source < ringDegree)
    (List.range ringDegree) 0
  simpa only [zero_add] using folded

theorem rawTerm_eval {logicalWidth : Nat} (coefficient : F)
    (left right : State logicalWidth) (degree source : Nat)
    (sourceBound : source < ringDegree)
    (assignment : Assignment F logicalWidth) :
    (rawTerm coefficient left right degree source).eval assignment =
      coefficient * rawProduct (evalState assignment left)
        (evalState assignment right) degree source := by
  unfold rawTerm rawProduct
  rw [dif_pos sourceBound]
  by_cases support : source ≤ degree ∧ degree - source < ringDegree
  · rw [dif_pos support, if_pos support]
    simp [Term.eval, evalState, ringFCoeff, sourceBound, support, mul_assoc]
  · rw [dif_neg support, if_neg support]
    simp

private theorem sum_scaled (coefficient : F) :
    ∀ values : List F,
      (values.map fun value => coefficient * value).sum =
        coefficient * values.sum
  | [] => by simp
  | value :: rest => by
      simp [sum_scaled coefficient rest, mul_add]

/-- Exact evaluation of one signed fixed raw-convolution term block. -/
theorem rawTerms_total {logicalWidth : Nat} (coefficient : F)
    (left right : State logicalWidth) (degree : Nat)
    (assignment : Assignment F logicalWidth) :
    total assignment (rawTerms coefficient left right degree) =
      coefficient * rawMulCoeffF (evalState assignment left)
        (evalState assignment right) degree := by
  unfold total rawTerms
  rw [List.map_map]
  have evaluated :
      (List.range ringDegree).map
          (Term.eval assignment ∘ rawTerm coefficient left right degree) =
        (List.range ringDegree).map fun source =>
          coefficient * rawProduct (evalState assignment left)
            (evalState assignment right) degree source := by
    apply List.map_congr_left
    intro source member
    exact rawTerm_eval coefficient left right degree source
      (List.mem_range.mp member) assignment
  rw [evaluated]
  calc
    ((List.range ringDegree).map fun source =>
        coefficient * rawProduct (evalState assignment left)
          (evalState assignment right) degree source).sum =
        (((List.range ringDegree).map fun source =>
          rawProduct (evalState assignment left)
            (evalState assignment right) degree source).map fun value =>
              coefficient * value).sum := by
          rw [List.map_map]
          apply congrArg List.sum
          apply List.map_congr_left
          intro source _member
          rfl
    _ = coefficient *
          ((List.range ringDegree).map fun source =>
            rawProduct (evalState assignment left)
              (evalState assignment right) degree source).sum :=
        sum_scaled coefficient _
    _ = coefficient * rawMulCoeffF (evalState assignment left)
          (evalState assignment right) degree := by
        rw [rawMulCoeffF_eq_sum]

def foldedDegree (lane : Fin ringDegree) : Nat :=
  if lane.val < ringMiddleDegree then lane.val + ringDegree
  else lane.val + ringMiddleDegree

def twiceCoefficient (lane : Fin ringDegree) : F :=
  if lane.val + 81 ≤ 106 then 1 else 0

/-- Fixed 162-term schedule in the exact Φ81 reduction order. -/
def terms {logicalWidth : Nat} (left right : State logicalWidth)
    (lane : Fin ringDegree) : List (Term logicalWidth) :=
  rawTerms 1 left right lane.val ++
    rawTerms (-1) left right (foldedDegree lane) ++
      rawTerms (twiceCoefficient lane) left right (lane.val + 81)

@[simp] theorem terms_length {logicalWidth : Nat}
    (left right : State logicalWidth) (lane : Fin ringDegree) :
    (terms left right lane).length = 162 := by
  simp [terms]

@[simp] theorem groups_length {logicalWidth : Nat}
    (left right : State logicalWidth) (lane : Fin ringDegree) :
    (groups (terms left right lane)).length = 33 := by
  rfl

/-- The fixed term schedule is exactly the SuperNeo Φ81 product lane. -/
theorem terms_total {logicalWidth : Nat}
    (left right : State logicalWidth) (lane : Fin ringDegree)
    (assignment : Assignment F logicalWidth) :
    total assignment (terms left right lane) =
      ringFMul (evalState assignment left) (evalState assignment right) lane := by
  unfold terms
  rw [total_append, total_append]
  rw [rawTerms_total, rawTerms_total, rawTerms_total]
  by_cases low : lane.val < ringMiddleDegree
  · by_cases twice : lane.val + 81 ≤ 106
    · simp [ringFMul, foldedDegree, twiceCoefficient, low, twice,
        sub_eq_add_neg]
    · simp [ringFMul, foldedDegree, twiceCoefficient, low, twice,
        sub_eq_add_neg]
  · by_cases twice : lane.val + 81 ≤ 106
    · simp [ringFMul, foldedDegree, twiceCoefficient, low, twice,
        sub_eq_add_neg]
    · simp [ringFMul, foldedDegree, twiceCoefficient, low, twice,
        sub_eq_add_neg]

/-- Every lane uses 33 five-product rows and one final pin. -/
theorem rows_length {logicalWidth : Nat}
    (oneColumn : Fin logicalWidth) (left right : State logicalWidth)
    (lane : Fin ringDegree)
    (groupOutput : Fin (groups (terms left right lane)).length →
      SparseForm logicalWidth)
    (prior output : SparseForm logicalWidth) :
    let interface : ProductSumPlan.Interface logicalWidth :=
      { oneColumn := oneColumn
        terms := terms left right lane
        groupOutput := groupOutput
        prior := prior
        output := output }
    (ProductSumPlan.rows interface).length = 34 := by
  simp

end NightstreamFPrime.Layout.ProductionRelation.Phi81ProductPlan
