import Mathlib.Data.List.OfFn
import NightstreamFPrime.Export.Stage1.PiDECInputCheck
import NightstreamFPrime.Export.Stage1.PiDECOrdinaryDirectSource
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork

/-!
Counted indexed generation of the selected PiDEC commitment packet. One
coordinate builds sixteen child terms in A, constant one in B, and its parent in C.
The existing affine recipe lowering fixes that exact row, including order.
No complete packet, package row list, or function-valued source is evaluated.
Counts are named operations; they are not machine instructions or wall time.
Dispatch, predecessor extraction, helper calls, value/index projections,
scalar literals, arithmetic, and data constructors are counted separately.
Calls include passing existing arguments; callee work is added separately.
Clock fields and clock arithmetic are instrumentation and are excluded.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECOrdinarySourceWork

open _root_.NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open _root_.NightstreamFPrime.Circuit
open _root_.NightstreamFPrime.Layout
open _root_.NightstreamFPrime.Layout.Stage1
open _root_.NightstreamFPrime.Lifecycle
open _root_.NightstreamFPrime.Lifecycle.PiDEC.v1_1
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

attribute [local irreducible] R1CS.lowerConstraints PiDECOrdinaryDirectSource.commitmentProgramRow

private def weightedExpr : List (Nat × F) → Expr
  | [] => 0
  | term :: rest => Expr.const term.2 * Expr.var term.1 + weightedExpr rest

private def affineResult (terms : List (Nat × F)) : R1CS.AffineResult (weightedExpr terms) where
  combination := ⟨0, terms⟩
  sound := by
    intro env
    induction terms with
    | nil => rfl
    | cons term rest ih =>
        simpa [R1CS.LinearCombination.eval, weightedExpr, Expr.eval] using
          congrArg (fun tail : F => term.2 * env term.1 + tail) ih

private theorem lowerAffine_weighted (terms : List (Nat × F)) :
    R1CS.lowerAffine (weightedExpr terms) = some (affineResult terms) := by
  induction terms with
  | nil => rfl
  | cons term rest ih =>
      simp [weightedExpr, R1CS.lowerAffine, ih, affineResult,
        R1CS.LinearCombination.ofVar, R1CS.LinearCombination.add, R1CS.LinearCombination.scale]

private def rawRow (parent : Nat) (terms : List (Nat × F)) : R1CS.Row :=
  ⟨⟨0, terms⟩, R1CS.LinearCombination.one, R1CS.LinearCombination.ofVar parent⟩

private theorem lowerConstraint_weighted (parent start : Nat) (terms : List (Nat × F)) :
    R1CS.lowerConstraint (Expr.var parent - weightedExpr terms) start =
      ⟨start, [rawRow parent terms]⟩ := by
  change R1CS.lowerConstraint
    (.add (.var parent) (.mul (.const (-1)) (weightedExpr terms))) start = _
  simp only [R1CS.lowerConstraint, R1CS.directConstraint,
    R1CS.directRecipeRow, lowerAffine_weighted, R1CS.affineRecipeRow, affineResult, rawRow]
  simp only [dite_true]

private theorem lowerRows_ofFn : ∀ {count : Nat}
    (parents : Fin count → Nat) (terms : Fin count → List (Nat × F)) (start : Nat),
    (R1CS.lowerConstraints
      (List.ofFn fun index => Expr.var (parents index) - weightedExpr (terms index)) start).rows =
      List.ofFn (fun index => rawRow (parents index) (terms index))
  | 0, _, _, _ => by simp only [List.ofFn_zero, R1CS.lowerConstraints]
  | count + 1, parents, terms, start => by
      rw [List.ofFn_succ, R1CS.lowerConstraints, lowerConstraint_weighted, List.ofFn_succ]
      change rawRow (parents 0) (terms 0) ::
        (R1CS.lowerConstraints
          (List.ofFn fun index : Fin count => Expr.var (parents index.succ) - weightedExpr (terms index.succ))
          start).rows = _
      rw [lowerRows_ofFn]

private theorem weightedExpr_ofFn : ∀ {count : Nat}
    (columns : Fin count → Nat) (weights : Fin count → F),
    ((List.ofFn fun index => Expr.var (columns index)).zip (List.ofFn weights)).foldr
      (fun pair suffix => Expr.const pair.2 * pair.1 + suffix) 0 =
        weightedExpr (List.ofFn fun index => (columns index, weights index))
  | 0, _, _ => rfl
  | _ + 1, columns, weights => by
      simp only [List.ofFn_succ, List.zip_cons_cons, List.foldr_cons, weightedExpr]
      rw [weightedExpr_ofFn]

private def sourceTerms (coordinate : Fin 1188) : List (Nat × F) :=
  List.ofFn fun child : Fin 16 =>
    (PiDECInputs.childCommitmentStart child + coordinate.val,
      Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight child)

private theorem coordinate_flat (coordinate : Fin 1188) :
    (CommitmentRecomposition.coordinates coordinate).1.val * 54 +
      (CommitmentRecomposition.coordinates coordinate).2.val = coordinate.val := by
  change coordinate.val / 54 * 54 + coordinate.val % 54 = coordinate.val
  exact Nat.div_add_mod' coordinate.val 54

private abbrev scalarInterface (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  CommitmentRecomposition.scalarInterface
    (Formal.commitmentInterface (Formal.atOffset
      (PiDECArithmetic.phaseInterface logicalWidth publicFits) PiDECInputs.phaseOffset))

private theorem parent_expr {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (coordinate : Fin 1188) :
    (scalarInterface logicalWidth publicFits).parent
      (Formal.commitmentOffset PiDECInputs.phaseOffset) coordinate =
        Expr.var (PiDECSourceSupport.parentCommitmentStart + coordinate.val) := by
  change (PiRLC.v1_1.CommitmentCombination.output
    (PiRLC.v1_1.Formal.commitmentInterface (PiDECInputs.piRlcSharedInterface logicalWidth publicFits))
    PiRLCStarts.commitmentLogicalStart (CommitmentRecomposition.coordinates coordinate).1
      (CommitmentRecomposition.coordinates coordinate).2) = _
  simp only [PiRLC.v1_1.CommitmentCombination.output, PiRLC.v1_1.CombinationFamily.output,
    PiRLC.v1_1.CombinationStep.output, PiRLC.v1_1.CombinationStep.indexOf,
    PiRLC.v1_1.CommitmentCombination.cell, finProdFinEquiv]
  congr 1
  change PiDECSourceSupport.parentCommitmentStart +
      (0 + 1 * (CommitmentRecomposition.coordinates coordinate).2.val +
        (ringDegree * 1) * (CommitmentRecomposition.coordinates coordinate).1.val) = _
  have flattened := coordinate_flat coordinate
  simp only [Nat.zero_add, Nat.one_mul, Nat.mul_one]
  change PiDECSourceSupport.parentCommitmentStart +
      ((CommitmentRecomposition.coordinates coordinate).2.val +
        54 * (CommitmentRecomposition.coordinates coordinate).1.val) = _
  omega

private theorem child_expr {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (coordinate : Fin 1188) (child : Fin 16) :
    (scalarInterface logicalWidth publicFits).child
      (Formal.commitmentOffset PiDECInputs.phaseOffset) child coordinate =
        Expr.var (PiDECInputs.childCommitmentStart child + coordinate.val) := by
  change Expr.var (PiDECInputs.childCommitmentStart child +
    (CommitmentRecomposition.coordinates coordinate).1.val * 54 +
      (CommitmentRecomposition.coordinates coordinate).2.val) = _
  congr 1
  have flattened := coordinate_flat coordinate
  omega

private theorem constraints_eq {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth} :
    PiDECOrdinaryDirectSource.commitmentConstraints logicalWidth publicFits =
      List.ofFn (fun coordinate : Fin 1188 =>
        Expr.var (PiDECSourceSupport.parentCommitmentStart + coordinate.val) - weightedExpr (sourceTerms coordinate)) := by
  change flatConstraints (Circuit.ops
    (RadixRecomposition.circuit (scalarInterface logicalWidth publicFits)).main
    (Formal.commitmentOffset PiDECInputs.phaseOffset)) = _
  rw [RadixRecomposition.circuit_ops, RadixRecomposition.flatConstraints_operations]
  unfold RadixRecomposition.constraints
  apply congrArg List.ofFn
  funext coordinate
  rw [RadixRecomposition.constraint,
    parent_expr (logicalWidth := logicalWidth) (publicFits := publicFits)]
  apply congrArg (fun expression : Expr =>
    Expr.var (PiDECSourceSupport.parentCommitmentStart + coordinate.val) - expression)
  simp only [RadixRecomposition.recomposeExpr,
    child_expr (logicalWidth := logicalWidth) (publicFits := publicFits)]
  exact weightedExpr_ofFn (count := 16)
    (fun child => PiDECInputs.childCommitmentStart child + coordinate.val)
    Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight

private theorem source_row {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (coordinate : Fin 1188) :
    PiDECOrdinaryDirectSource.commitmentProgramRow relation coordinate =
      Spartan.remapRow (rawRow (PiDECSourceSupport.parentCommitmentStart + coordinate.val) (sourceTerms coordinate)) := by
  have rows : PiDECOrdinaryDirectSource.commitmentRows logicalWidth publicFits =
      List.ofFn (fun coordinate : Fin 1188 =>
        Spartan.remapRow (rawRow (PiDECSourceSupport.parentCommitmentStart + coordinate.val) (sourceTerms coordinate))) := by
    rw [PiDECOrdinaryDirectSource.commitmentRows, constraints_eq, lowerRows_ofFn]
    simp only [Spartan.remapRows, List.map_ofFn, Function.comp_def]
  exact congrFun (List.ofFn_injective
    ((PiDECOrdinaryDirectSource.commitmentProgramRows_eq relation).trans rows)) coordinate

private theorem late_column (start index : Nat) (late : Spartan.piCcsPhaseOffset ≤ start) :
    Spartan.sourceToSpartan (start + index) =
      Spartan.piCcsLocalStart + (start + index - Spartan.piCcsPhaseOffset) := by
  have first : ¬ start + index < Spartan.pilotSourceColumnCount := by
    change 14751804 ≤ start at late
    change ¬ start + index < 14722512
    omega
  have second : ¬ start + index < Spartan.proofInputSourceStart := by
    change 14751804 ≤ start at late
    change ¬ start + index < 14722516
    omega
  have third : ¬ start + index < Spartan.piCcsPhaseOffset := by omega
  simp only [Spartan.sourceToSpartan, if_neg first, if_neg second, if_neg third]

/-- The constants are the proved selected starts after Spartan's column
permutation. One constant/index read, add, and return execute here. -/
private def parentColumn (coordinate : Fin 1188) : Result Nat :=
  ⟨20347121 + coordinate.val, 4⟩

private theorem parentColumn_value (coordinate : Fin 1188) :
    (parentColumn coordinate).value =
      Spartan.sourceToSpartan (PiDECSourceSupport.parentCommitmentStart + coordinate.val) := by
  rw [late_column _ _ (by rw [PiDECSourceSupport.parentCommitmentStart_eq]; decide)]
  rw [PiDECSourceSupport.parentCommitmentStart_eq]
  change 20347121 + coordinate.val = 14751526 + (20347399 + coordinate.val - 14751804)
  omega

/-- Two literals, the Fin value read, multiplication, two additions, and
the Result constructor. The child is already a natural argument. -/
private def childColumn (coordinate : Fin 1188) (child : Nat) : Result Nat :=
  ⟨28972970 + child * 1188 + coordinate.val, 7⟩

private theorem childColumn_value (coordinate : Fin 1188) (child : Fin 16) :
    (childColumn coordinate child.val).value =
      Spartan.sourceToSpartan (PiDECInputs.childCommitmentStart child + coordinate.val) := by
  rw [late_column _ _ (by
    change 14751804 ≤ 28973248 + child.val * 1188
    omega)]
  change 28972970 + child.val * 1188 + coordinate.val =
    14751526 + (28973248 + child.val * 1188 + coordinate.val - 14751804)
  omega

/-- The base charges dispatch, its literal, and Result. A step charges
dispatch, predecessor, recursive call, value read, literal, multiply, and Result. -/
private def powerTwo : Nat → Result Nat
  | 0 => ⟨1, 3⟩
  | count + 1 =>
      let previous := powerTwo count
      ⟨previous.value * 2, previous.work + 7⟩

private theorem powerTwo_value (count : Nat) : (powerTwo count).value = 2 ^ count := by
  induction count with
  | zero => rfl
  | succ count ih => simp only [powerTwo, ih, Nat.pow_succ]

private theorem powerTwo_work (count : Nat) : (powerTwo count).work = count * 7 + 3 := by
  induction count with
  | zero => rfl
  | succ count ih => simp only [powerTwo, ih, Nat.add_mul, Nat.one_mul]

/-- Six operations: power call, value read, modulus literal, reduction,
field constructor, and Result. Selected child exponents are less than 16. -/
private def weight (child : Nat) : Result F :=
  let power := powerTwo child
  let value : F := ⟨power.value % goldilocksModulus, Nat.mod_lt _ (by decide)⟩
  ⟨value, power.work + 6⟩

private theorem weight_value (child : Fin 16) :
    (weight child.val).value = Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight child := by
  apply Fin.ext
  change (powerTwo child.val).value % goldilocksModulus = 2 ^ child.val % goldilocksModulus
  rw [powerTwo_value]

/-- Two calls, two value reads, the pair constructor, and Result. -/
private def childTerm (coordinate : Fin 1188) (child : Nat) : Result (Nat × F) :=
  let column := childColumn coordinate child
  let coefficient := weight child
  ⟨(column.value, coefficient.value), column.work + coefficient.work + 6⟩

private theorem childTerm_work_le (coordinate : Fin 1188) (child : Nat) (bound : child < 16) :
    (childTerm coordinate child).work ≤ 127 := by
  simp only [childTerm, childColumn, weight, powerTwo_work]
  omega

/-- Each step charges dispatch/predecessor (2), two calls, the index literal
and addition (2), two value reads, cons, and Result (10). The empty case
charges dispatch, nil, and Result. Indices advance without accessor closures. -/
private def childTerms (coordinate : Fin 1188) : Nat → Nat → Result (List (Nat × F))
  | 0, _ => ⟨[], 3⟩
  | remaining + 1, next =>
      let term := childTerm coordinate next
      let suffix := childTerms coordinate remaining (next + 1)
      ⟨term.value :: suffix.value, term.work + suffix.work + 10⟩

private theorem childTerms_value (coordinate : Fin 1188) : ∀ (remaining next : Nat),
    (childTerms coordinate remaining next).value =
      List.ofFn (fun index : Fin remaining => (childTerm coordinate (next + index.val)).value)
  | 0, _ => rfl
  | remaining + 1, next => by
      rw [childTerms, List.ofFn_succ, childTerms_value]
      simp only [Fin.val_succ]
      apply congrArg (List.cons (childTerm coordinate next).value)
      apply congrArg List.ofFn
      funext index
      have same : next + 1 + index.val = next + (index.val + 1) := by omega
      rw [same]

private theorem childTerms_work_le (coordinate : Fin 1188) : ∀ (remaining next : Nat),
    next + remaining ≤ 16 → (childTerms coordinate remaining next).work ≤ remaining * 137 + 3
  | 0, _, _ => by simp [childTerms]
  | remaining + 1, next, bound => by
      have head := childTerm_work_le coordinate next (by omega)
      have tail := childTerms_work_le coordinate remaining (next + 1) (by omega)
      simp only [childTerms, Nat.add_mul, Nat.one_mul]
      omega

private theorem allChildTerms_value (coordinate : Fin 1188) :
    (childTerms coordinate 16 0).value =
      (sourceTerms coordinate).map (fun term => (Spartan.sourceToSpartan term.1, term.2)) := by
  rw [childTerms_value]
  simp only [Nat.zero_add, sourceTerms, List.map_ofFn]
  apply congrArg List.ofFn
  funext child
  change ((childColumn coordinate child.val).value, (weight child.val).value) = _
  exact congrArg₂ Prod.mk (childColumn_value coordinate child) (weight_value child)

attribute [irreducible] childTerms

/-- Final overhead: two Nat literals, two calls, two value reads, four F
literals, two nils, pair/cons, three affine records, Row, and Result (19).
The affine records are constructed here, with no uncharged helper call. -/
def commitmentRow (coordinate : Fin 1188) : Result R1CS.Row :=
  let children := childTerms coordinate 16 0
  let parent := parentColumn coordinate
  ⟨⟨⟨0, children.value⟩, ⟨1, []⟩, ⟨0, [(parent.value, 1)]⟩⟩,
    children.work + parent.work + 19⟩

private theorem commitmentRow_mapped (coordinate : Fin 1188) :
    (commitmentRow coordinate).value =
      Spartan.remapRow (rawRow (PiDECSourceSupport.parentCommitmentStart + coordinate.val) (sourceTerms coordinate)) := by
  simp only [commitmentRow, allChildTerms_value, parentColumn_value, Spartan.remapRow,
    Spartan.remapCombination, rawRow, R1CS.LinearCombination.one, R1CS.LinearCombination.ofVar,
    List.map_nil, List.map_cons]

theorem commitmentRow_value (coordinate : Fin 1188) :
    (commitmentRow coordinate).value =
      PiDECOrdinaryDirectSource.commitmentProgramRow PiDECInputCheck.relation coordinate :=
  (commitmentRow_mapped coordinate).trans
    (source_row (logicalWidth := PiDECInputCheck.logicalWidth) (publicFits := PiDECInputCheck.publicFits)
      PiDECInputCheck.relation coordinate).symm

theorem commitmentRow_lengths (coordinate : Fin 1188) :
    (commitmentRow coordinate).value.a.terms.length = 16 ∧
      (commitmentRow coordinate).value.b.terms.length = 0 ∧
      (commitmentRow coordinate).value.c.terms.length = 1 := by
  simp only [commitmentRow, childTerms_value, List.length_ofFn,
    List.length_nil, List.length_cons, and_self]

/-- Sixteen calls bounded by their actual power, column, and constructor
loops; list initialization, the parent column, and final construction. -/
def commitmentRowWork : Nat := 16 * 137 + 3 + 4 + 19

theorem commitmentRow_work_le (coordinate : Fin 1188) :
    (commitmentRow coordinate).work ≤ commitmentRowWork := by
  have children := childTerms_work_le coordinate 16 0 (by decide)
  change (childTerms coordinate 16 0).work + 4 + 19 ≤ commitmentRowWork
  unfold commitmentRowWork
  omega

end NightstreamFPrime.Export.Stage1.PiDECOrdinarySourceWork
