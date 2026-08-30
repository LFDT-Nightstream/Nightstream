import Mathlib.Data.List.GetD
import NightstreamFPrime.Layout.ProductionRelation.PinRow
import NightstreamFPrime.Layout.ProductionRelation.ProductSumRow

/-!
Owns the proof-oriented grouping core for direct five-product selective rows.
Products remain in source order and are split into groups of at most five.

This module does not select a concrete Stage 1 product schedule.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.ProductSumPlan

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- One product of two sparse forms. -/
structure Term (logicalWidth : Nat) where
  left : SparseForm logicalWidth
  right : SparseForm logicalWidth

namespace Term

def zero {logicalWidth : Nat} : Term logicalWidth :=
  { left := .empty, right := .empty }

def eval {logicalWidth : Nat} (assignment : Assignment F logicalWidth)
    (term : Term logicalWidth) : F :=
  term.left.eval assignment * term.right.eval assignment

@[simp] theorem eval_zero {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth) :
    (zero : Term logicalWidth).eval assignment = 0 := by
  simp [zero, eval]

end Term

/-- Fixed-width grouping in exact source order. -/
def groups {Alpha : Type} : List Alpha → List (List Alpha)
  | [] => []
  | [a] => [[a]]
  | [a, b] => [[a, b]]
  | [a, b, c] => [[a, b, c]]
  | [a, b, c, d] => [[a, b, c, d]]
  | a :: b :: c :: d :: e :: rest =>
      [a, b, c, d, e] :: groups rest

theorem groups_join {Alpha : Type} :
    ∀ values : List Alpha, (groups values).flatten = values
  | [] => rfl
  | [a] => rfl
  | [a, b] => rfl
  | [a, b, c] => rfl
  | [a, b, c, d] => rfl
  | a :: b :: c :: d :: e :: rest => by
      simp [groups, groups_join rest]

theorem group_length_le {Alpha : Type} :
    ∀ (values : List Alpha) (group : List Alpha),
      group ∈ groups values → group.length ≤ 5
  | [], group, member => by simp [groups] at member
  | [a], group, member => by
      simp [groups] at member
      subst group
      norm_num
  | [a, b], group, member => by
      simp [groups] at member
      subst group
      norm_num
  | [a, b, c], group, member => by
      simp [groups] at member
      subst group
      norm_num
  | [a, b, c, d], group, member => by
      simp [groups] at member
      subst group
      norm_num
  | a :: b :: c :: d :: e :: rest, group, member => by
      simp only [groups, List.mem_cons] at member
      rcases member with rfl | member
      · norm_num
      · exact group_length_le rest group member

def termAt {logicalWidth : Nat} (group : List (Term logicalWidth))
    (lane : Fin 5) : Term logicalWidth :=
  group.getD lane.val .zero

def groupTotal {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth)
    (group : List (Term logicalWidth)) : F :=
  Spec.ProductionRelation.RowSemantics.productTotal
    (fun lane => (termAt group lane).left.eval assignment)
    (fun lane => (termAt group lane).right.eval assignment)

/-- A padded five-product row evaluates to the exact unpadded group sum. -/
theorem groupTotal_eq_sum {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth)
    (group : List (Term logicalWidth)) (bound : group.length ≤ 5) :
    groupTotal assignment group = (group.map (Term.eval assignment)).sum := by
  rcases group with _ | ⟨a, rest⟩
  · simp [groupTotal, termAt,
      Spec.ProductionRelation.RowSemantics.productTotal, Term.eval, Term.zero]
  rcases rest with _ | ⟨b, rest⟩
  · simp [groupTotal, termAt,
      Spec.ProductionRelation.RowSemantics.productTotal, Term.eval, Term.zero]
  rcases rest with _ | ⟨c, rest⟩
  · simp [groupTotal, termAt,
      Spec.ProductionRelation.RowSemantics.productTotal, Term.eval, Term.zero]
    <;> try abel
  rcases rest with _ | ⟨d, rest⟩
  · simp [groupTotal, termAt,
      Spec.ProductionRelation.RowSemantics.productTotal, Term.eval, Term.zero]
    <;> try abel
  rcases rest with _ | ⟨e, rest⟩
  · simp [groupTotal, termAt,
      Spec.ProductionRelation.RowSemantics.productTotal, Term.eval, Term.zero]
    <;> try abel
  rcases rest with _ | ⟨f, rest⟩
  · simp [groupTotal, termAt,
      Spec.ProductionRelation.RowSemantics.productTotal, Term.eval, Term.zero]
    <;> try abel
  · simp only [List.length_cons] at bound
    omega

def total {logicalWidth : Nat} (assignment : Assignment F logicalWidth)
    (terms : List (Term logicalWidth)) : F :=
  (terms.map (Term.eval assignment)).sum

@[simp] theorem total_append {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth)
    (left right : List (Term logicalWidth)) :
    total assignment (left ++ right) =
      total assignment left + total assignment right := by
  simp [total]

def groupTotals {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth)
    (terms : List (Term logicalWidth)) : List F :=
  List.ofFn fun group : Fin (groups terms).length =>
    groupTotal assignment ((groups terms).get group)

private theorem sum_map_join {Alpha : Type} [AddCommMonoid Alpha]
    (families : List (List Alpha)) :
    families.flatten.sum = (families.map List.sum).sum := by
  induction families with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [List.sum_append, inductionHypothesis]

/-- Grouping changes neither product order nor the exact field total. -/
theorem groupTotals_sum {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth)
    (terms : List (Term logicalWidth)) :
    (groupTotals assignment terms).sum = total assignment terms := by
  unfold groupTotals total
  have enumerate :
      List.ofFn (fun group : Fin (groups terms).length =>
          groupTotal assignment ((groups terms).get group)) =
        (groups terms).map (groupTotal assignment) := by
    simpa only [List.get_eq_getElem] using
      List.ofFn_getElem_eq_map (groups terms) (groupTotal assignment)
  rw [enumerate]
  have perGroup :
      (groups terms).map (groupTotal assignment) =
        (groups terms).map fun group =>
          (group.map (Term.eval assignment)).sum := by
    apply List.map_congr_left
    intro group member
    exact groupTotal_eq_sum assignment group
      (group_length_le terms group member)
  rw [perGroup]
  calc
    (List.map (fun group =>
        (List.map (Term.eval assignment) group).sum) (groups terms)).sum =
        (List.map (List.map (Term.eval assignment))
          (groups terms)).flatten.sum := by
            rw [sum_map_join]
            induction groups terms with
            | nil => rfl
            | cons group rest inductionHypothesis =>
                simp only [List.map_cons, List.sum_cons]
                rw [inductionHypothesis]
    _ = (List.map (Term.eval assignment)
          (groups terms).flatten).sum := by
            rw [List.map_flatten]
    _ = (List.map (Term.eval assignment) terms).sum := by
          rw [groups_join]

def sumForms {logicalWidth : Nat} :
    List (SparseForm logicalWidth) → SparseForm logicalWidth
  | [] => .empty
  | form :: rest => SparseForm.add form (sumForms rest)

@[simp] theorem sumForms_eval {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth) :
    ∀ forms : List (SparseForm logicalWidth),
      (sumForms forms).eval assignment =
        (forms.map fun form => form.eval assignment).sum
  | [] => by simp [sumForms]
  | form :: rest => by
      simp [sumForms, sumForms_eval assignment rest]

/-- Sparse forms supplied by one direct product-sum computation. -/
structure Interface (logicalWidth : Nat) where
  oneColumn : Fin logicalWidth
  terms : List (Term logicalWidth)
  groupOutput : Fin (groups terms).length → SparseForm logicalWidth
  prior : SparseForm logicalWidth
  output : SparseForm logicalWidth

def selector {logicalWidth : Nat} (interface : Interface logicalWidth) :
    SparseForm logicalWidth :=
  SparseForm.singleton interface.oneColumn 1

def groupAt {logicalWidth : Nat} (interface : Interface logicalWidth)
    (group : Fin (groups interface.terms).length) :
    List (Term logicalWidth) :=
  (groups interface.terms).get group

def productRow {logicalWidth : Nat} (interface : Interface logicalWidth)
    (group : Fin (groups interface.terms).length) :
    ProductSumRow.Forms logicalWidth :=
  { selector := selector interface
    left := fun lane => (termAt (groupAt interface group) lane).left
    right := fun lane => (termAt (groupAt interface group) lane).right
    output := interface.groupOutput group }

def productRows {logicalWidth : Nat} (interface : Interface logicalWidth) :
    List (ProductSumRow.Forms logicalWidth) :=
  List.ofFn (productRow interface)

def groupOutputForms {logicalWidth : Nat}
    (interface : Interface logicalWidth) : List (SparseForm logicalWidth) :=
  List.ofFn interface.groupOutput

def finalDifference {logicalWidth : Nat}
    (interface : Interface logicalWidth) : SparseForm logicalWidth :=
  SparseForm.add
    (SparseForm.add interface.output (SparseForm.scale (-1) interface.prior))
    (SparseForm.scale (-1) (sumForms (groupOutputForms interface)))

def finalRow {logicalWidth : Nat} (interface : Interface logicalWidth) :
    PinRow.Forms logicalWidth :=
  { selector := selector interface
    value := finalDifference interface }

def ProductRowsZero {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth) : Prop :=
  ∀ row ∈ productRows interface, row.residual assignment = 0

def FinalRowZero {logicalWidth : Nat} (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth) : Prop :=
  (finalRow interface).residual assignment = 0

/-- Exact equations represented by the compact product rows and final pin. -/
structure Equations {logicalWidth : Nat} (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth) : Prop where
  groups : ∀ group,
    (interface.groupOutput group).eval assignment =
      groupTotal assignment (groupAt interface group)
  final : interface.output.eval assignment =
    interface.prior.eval assignment + total assignment interface.terms

private theorem productRow_preserves {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1)
    (group : Fin (groups interface.terms).length) :
    (productRow interface group).Preserves assignment
      (fun lane => (termAt (groupAt interface group) lane).left.eval assignment)
      (fun lane => (termAt (groupAt interface group) lane).right.eval assignment)
      ((interface.groupOutput group).eval assignment) := by
  refine ⟨?_, fun _ => rfl, fun _ => rfl, rfl⟩
  simp [productRow, selector, one]

theorem productRow_zero_iff {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1)
    (group : Fin (groups interface.terms).length) :
    (productRow interface group).residual assignment = 0 ↔
      (interface.groupOutput group).eval assignment =
        groupTotal assignment (groupAt interface group) := by
  have equivalence := ProductSumRow.Forms.residual_zero_iff
    (productRow interface group) assignment
    (fun lane => (termAt (groupAt interface group) lane).left.eval assignment)
    (fun lane => (termAt (groupAt interface group) lane).right.eval assignment)
    ((interface.groupOutput group).eval assignment)
    (productRow_preserves interface assignment one group)
  simpa [groupTotal, eq_comm] using equivalence

theorem productRowsZero_iff {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1) :
    ProductRowsZero interface assignment ↔
      ∀ group,
        (interface.groupOutput group).eval assignment =
          groupTotal assignment (groupAt interface group) := by
  constructor
  · intro rowsZero group
    apply (productRow_zero_iff interface assignment one group).mp
    exact rowsZero (productRow interface group)
      (List.mem_ofFn.mpr ⟨group, rfl⟩)
  · intro equations row member
    rcases List.mem_ofFn.mp member with ⟨group, rfl⟩
    exact (productRow_zero_iff interface assignment one group).mpr
      (equations group)

theorem groupOutputValues_eq_groupTotals {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (equations : ∀ group,
      (interface.groupOutput group).eval assignment =
        groupTotal assignment (groupAt interface group)) :
    (groupOutputForms interface).map
        (fun form => form.eval assignment) =
      groupTotals assignment interface.terms := by
  unfold groupOutputForms groupTotals
  rw [List.map_ofFn]
  apply congrArg List.ofFn
  funext group
  exact equations group

theorem finalDifference_eval {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth) :
    (finalDifference interface).eval assignment =
      interface.output.eval assignment - interface.prior.eval assignment -
        ((groupOutputForms interface).map
          (fun form => form.eval assignment)).sum := by
  simp [finalDifference, sumForms_eval, sub_eq_add_neg]

theorem finalRow_zero_iff {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1) :
    FinalRowZero interface assignment ↔
      interface.output.eval assignment =
        interface.prior.eval assignment +
          ((groupOutputForms interface).map
            (fun form => form.eval assignment)).sum := by
  have preserves : (finalRow interface).Preserves assignment
      ((finalDifference interface).eval assignment) := by
    refine ⟨?_, rfl⟩
    simp [finalRow, selector, one]
  unfold FinalRowZero
  rw [PinRow.Forms.residual_zero_iff
    (finalRow interface) assignment _ preserves]
  rw [finalDifference_eval]
  constructor
  · intro hypothesis
    have reduced :=
      Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp hypothesis
    have added := congrArg
      (fun value : F => value + interface.prior.eval assignment) reduced
    have reordered :
        interface.output.eval assignment =
          ((groupOutputForms interface).map
            (fun form => form.eval assignment)).sum +
            interface.prior.eval assignment := by
      simpa using added
    simpa [add_comm] using reordered
  · intro hypothesis
    rw [hypothesis]
    abel

theorem equations_imply_total {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (groupEquations : ∀ group,
      (interface.groupOutput group).eval assignment =
        groupTotal assignment (groupAt interface group))
    (finalEquation : interface.output.eval assignment =
      interface.prior.eval assignment +
        ((groupOutputForms interface).map
          (fun form => form.eval assignment)).sum) :
    interface.output.eval assignment =
      interface.prior.eval assignment + total assignment interface.terms := by
  rw [groupOutputValues_eq_groupTotals interface assignment groupEquations]
    at finalEquation
  rw [groupTotals_sum] at finalEquation
  exact finalEquation

inductive Row (logicalWidth : Nat) where
  | product : ProductSumRow.Forms logicalWidth → Row logicalWidth
  | pin : PinRow.Forms logicalWidth → Row logicalWidth

namespace Row

def meaningfulForm {logicalWidth : Nat} (row : Row logicalWidth)
    (port : Fin Spec.ProductionRelation.meaningfulPortCount) :
    SparseForm logicalWidth :=
  match row with
  | .product forms => forms.meaningfulForm port
  | .pin forms => forms.meaningfulForm port

def residual {logicalWidth : Nat} (row : Row logicalWidth)
    (assignment : Assignment F logicalWidth) : F :=
  match row with
  | .product forms => forms.residual assignment
  | .pin forms => forms.residual assignment

def portForm {logicalWidth : Nat} (row : Row logicalWidth)
    (port : Fin Spec.ProductionRelation.matrixCount) : SparseForm logicalWidth :=
  match ProductionRelation.meaningfulPort? port with
  | some meaningful => row.meaningfulForm meaningful
  | none => .empty

def portImages {logicalWidth : Nat} (row : Row logicalWidth)
    (assignment : Assignment F logicalWidth) :
    Fin Spec.ProductionRelation.matrixCount → F :=
  fun port => (row.portForm port).eval assignment

theorem polynomial_eq_residual {logicalWidth : Nat} (row : Row logicalWidth)
    (assignment : Assignment F logicalWidth) :
    evaluatePolynomial
        Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps
        Spec.ProductionRelation.polynomial (row.portImages assignment) =
      row.residual assignment := by
  cases row <;> rfl

end Row

def rows {logicalWidth : Nat} (interface : Interface logicalWidth) :
    List (Row logicalWidth) :=
  (productRows interface).map Row.product ++ [Row.pin (finalRow interface)]

@[simp] theorem productRows_length {logicalWidth : Nat}
    (interface : Interface logicalWidth) :
    (productRows interface).length = (groups interface.terms).length := by
  simp [productRows]

@[simp] theorem rows_length {logicalWidth : Nat}
    (interface : Interface logicalWidth) :
    (rows interface).length = (groups interface.terms).length + 1 := by
  simp [rows]

def RowsZero {logicalWidth : Nat} (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth) : Prop :=
  ∀ row ∈ rows interface, row.residual assignment = 0

theorem rowsZero_implies_productRowsZero {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (rowsZero : RowsZero interface assignment) :
    ProductRowsZero interface assignment := by
  intro forms member
  exact rowsZero (Row.product forms) (by
    unfold rows
    exact List.mem_append_left _ (List.mem_map_of_mem member))

theorem rowsZero_implies_finalRowZero {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (rowsZero : RowsZero interface assignment) :
    FinalRowZero interface assignment := by
  exact rowsZero (Row.pin (finalRow interface)) (by simp [rows])

/-- The direct row list is sound and complete for its exact grouped-product
equations. -/
theorem rowsZero_iff_equations {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1) :
    RowsZero interface assignment ↔ Equations interface assignment := by
  constructor
  · intro rowsZero
    have groupEquations :=
      (productRowsZero_iff interface assignment one).mp
        (rowsZero_implies_productRowsZero interface assignment rowsZero)
    have finalEquation :=
      (finalRow_zero_iff interface assignment one).mp
        (rowsZero_implies_finalRowZero interface assignment rowsZero)
    exact ⟨groupEquations,
      equations_imply_total interface assignment groupEquations finalEquation⟩
  · intro equations row member
    simp only [rows, List.mem_append, List.mem_map, List.mem_singleton] at member
    rcases member with ⟨forms, formsMember, rfl⟩ | rfl
    · exact (productRowsZero_iff interface assignment one).mpr
        equations.groups forms formsMember
    · apply (finalRow_zero_iff interface assignment one).mpr
      have valuesEqual := groupOutputValues_eq_groupTotals
        interface assignment equations.groups
      have sumsEqual := congrArg List.sum valuesEqual
      rw [groupTotals_sum] at sumsEqual
      calc
        interface.output.eval assignment =
            interface.prior.eval assignment +
              total assignment interface.terms := equations.final
        _ = interface.prior.eval assignment +
              ((groupOutputForms interface).map
                (fun form => form.eval assignment)).sum := by
            rw [sumsEqual]

/-- Actual 14-matrix plan for one grouped-product computation. -/
def plan {logicalWidth : Nat} (interface : Interface logicalWidth)
    (rowCount_le : (rows interface).length ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables) :
    ProductionRelation.Plan logicalWidth where
  rowCount := (rows interface).length
  rowCount_le := rowCount_le
  forms := fun row port => (rows interface).get row |>.meaningfulForm port

theorem plan_rowImage_at {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (rowCount_le : (rows interface).length ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth)
    (row : Fin (rows interface).length) :
    (plan interface rowCount_le).rowImage assignment
        ((plan interface rowCount_le).rowLayout.toVertex row) =
      ((rows interface).get row).portImages assignment := by
  funext port
  unfold ProductionRelation.Plan.rowImage
  rw [(plan interface rowCount_le).rowLayout.toColumn_toVertex]
  rfl

theorem plan_residual_at {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (rowCount_le : (rows interface).length ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth)
    (row : Fin (rows interface).length) :
    evaluatePolynomial
        Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps
        Spec.ProductionRelation.polynomial
        ((plan interface rowCount_le).rowImage assignment
          ((plan interface rowCount_le).rowLayout.toVertex row)) =
      ((rows interface).get row).residual assignment := by
  rw [plan_rowImage_at]
  exact Row.polynomial_eq_residual _ _

def PlanRowsZero {logicalWidth : Nat} (interface : Interface logicalWidth)
    (rowCount_le : (rows interface).length ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth) : Prop :=
  ∀ row : Fin (rows interface).length,
    evaluatePolynomial
        Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps
        Spec.ProductionRelation.polynomial
        ((plan interface rowCount_le).rowImage assignment
          ((plan interface rowCount_le).rowLayout.toVertex row)) = 0

theorem planRowsZero_iff_equations {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (rowCount_le : (rows interface).length ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1) :
    PlanRowsZero interface rowCount_le assignment ↔
      Equations interface assignment := by
  rw [← rowsZero_iff_equations interface assignment one]
  constructor
  · intro planRowsZero row member
    rcases List.mem_iff_get.mp member with ⟨index, rfl⟩
    rw [← plan_residual_at interface rowCount_le assignment index]
    exact planRowsZero index
  · intro rowsZero index
    rw [plan_residual_at interface rowCount_le assignment index]
    exact rowsZero ((rows interface).get index) (List.get_mem _ _)

theorem planRowsZero_implies_total {logicalWidth : Nat}
    (interface : Interface logicalWidth)
    (rowCount_le : (rows interface).length ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1)
    (rowsZero : PlanRowsZero interface rowCount_le assignment) :
    interface.output.eval assignment =
      interface.prior.eval assignment + total assignment interface.terms :=
  ((planRowsZero_iff_equations interface rowCount_le assignment one).mp
    rowsZero).final

end NightstreamFPrime.Layout.ProductionRelation.ProductSumPlan
