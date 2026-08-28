import NightstreamFPrime.Layout.ProductionRelation.OrdinaryRow
import NightstreamFPrime.Layout.ProductionRelation.RetainedSlot

/-!
Owns the source-field substitution used by the production selective compiler.
A substitution maps each bounded source field to one sparse form over the
final low-norm assignment. Linear combinations and ordinary R1CS rows compile
through that map with exact evaluation preservation.

This module does not choose the production retained set or affine rewrite
schedule.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.SourceCompiler

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- Sparse reconstruction form for every source field in a fixed domain. -/
structure SourceMap (sourceWidth logicalWidth : Nat) where
  form : Fin sourceWidth → SparseForm logicalWidth

namespace SourceMap

/-- The sparse forms reconstruct the exact source assignment. -/
def Preserves {sourceWidth logicalWidth : Nat}
    (sourceMap : SourceMap sourceWidth logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Circuit.Env) : Prop :=
  ∀ column, (sourceMap.form column).eval assignment = source column.val

end SourceMap

/-- Every source index used by one sparse R1CS linear combination is inside
the declared source domain. -/
def CombinationBounded (sourceWidth : Nat)
    (combination : R1CS.LinearCombination) : Prop :=
  ∀ term ∈ combination.terms, term.1 < sourceWidth

/-- All three source forms of one R1CS row are bounded. -/
def RowBounded (sourceWidth : Nat) (row : R1CS.Row) : Prop :=
  CombinationBounded sourceWidth row.a ∧
    CombinationBounded sourceWidth row.b ∧
    CombinationBounded sourceWidth row.c

/-- Proof-carrying result of bounded affine source recognition. -/
structure AffineSource (sourceWidth : Nat) where
  expression : Circuit.Expr
  combination : R1CS.LinearCombination
  bounded : CombinationBounded sourceWidth combination
  sound : ∀ env, combination.eval env = expression.eval env

private def termsBoundedDecidable (sourceWidth : Nat) :
    (terms : List (Nat × F)) →
      Decidable (∀ term ∈ terms, term.1 < sourceWidth)
  | [] => isTrue (by simp)
  | head :: tail =>
      match inferInstanceAs (Decidable (head.1 < sourceWidth)),
          termsBoundedDecidable sourceWidth tail with
      | isTrue headBounded, isTrue tailBounded =>
          isTrue (by
            intro term member
            rcases List.mem_cons.mp member with rfl | member
            · exact headBounded
            · exact tailBounded term member)
      | isFalse headUnbounded, _ =>
          isFalse (fun all => headUnbounded (all head (by simp)))
      | _, isFalse tailUnbounded =>
          isFalse (fun all => tailUnbounded fun term member =>
            all term (by simp [member]))

def combinationBoundedDecidable (sourceWidth : Nat)
    (combination : R1CS.LinearCombination) :
    Decidable (CombinationBounded sourceWidth combination) :=
  termsBoundedDecidable sourceWidth combination.terms

/-- Fail-closed affine recognition in one declared source domain. -/
def lowerAffine? (sourceWidth : Nat) (expression : Circuit.Expr) :
    Option (AffineSource sourceWidth) :=
  match R1CS.lowerAffine expression with
  | none => none
  | some lowered =>
      match combinationBoundedDecidable sourceWidth lowered.combination with
      | isTrue bounded =>
          some
            { expression := expression
              combination := lowered.combination
              bounded := bounded
              sound := lowered.sound }
      | isFalse _ => none

/-- A successful bounded affine recognition evaluates exactly as its source
expression. -/
theorem lowerAffine?_sound {sourceWidth : Nat} (expression : Circuit.Expr)
    (result : AffineSource sourceWidth)
    (found : lowerAffine? sourceWidth expression = some result)
    (env : Circuit.Env) :
    result.combination.eval env = expression.eval env := by
  unfold lowerAffine? at found
  cases loweredEq : R1CS.lowerAffine expression with
  | none => simp [loweredEq] at found
  | some lowered =>
      cases boundedEq : combinationBoundedDecidable sourceWidth
        lowered.combination with
      | isTrue bounded =>
          simp [loweredEq, boundedEq] at found
          subst result
          exact lowered.sound env
      | isFalse unbounded =>
          simp [loweredEq, boundedEq] at found

/-- Compile only the variable terms of one source linear combination. -/
def compileTerms {sourceWidth logicalWidth : Nat}
    (sourceMap : SourceMap sourceWidth logicalWidth) :
    (terms : List (Nat × F)) →
    (∀ term ∈ terms, term.1 < sourceWidth) → SparseForm logicalWidth
  | [], _ => .empty
  | term :: rest, bounded =>
      SparseForm.add
        (SparseForm.scale term.2
          (sourceMap.form ⟨term.1, bounded term (by simp)⟩))
        (compileTerms sourceMap rest fun candidate member =>
          bounded candidate (by simp [member]))

/-- Compiled source terms evaluate to the exact source term sum. -/
theorem compileTerms_eval {sourceWidth logicalWidth : Nat}
    (sourceMap : SourceMap sourceWidth logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Circuit.Env)
    (preserves : sourceMap.Preserves assignment source)
    (terms : List (Nat × F))
    (bounded : ∀ term ∈ terms, term.1 < sourceWidth) :
    (compileTerms sourceMap terms bounded).eval assignment =
      (terms.map fun term => term.2 * source term.1).sum := by
  induction terms with
  | nil => simp [compileTerms]
  | cons term rest inductionHypothesis =>
      have headBound : term.1 < sourceWidth := bounded term (by simp)
      have restBound : ∀ candidate ∈ rest, candidate.1 < sourceWidth :=
        fun candidate member => bounded candidate (by simp [member])
      simp only [compileTerms, SparseForm.add_eval, SparseForm.scale_eval,
        List.map_cons, List.sum_cons]
      rw [preserves ⟨term.1, headBound⟩]
      exact congrArg (fun value => term.2 * source term.1 + value)
        (inductionHypothesis restBound)

/-- Compile one complete affine source form. The constant is carried by a
verifier-owned coordinate whose assignment value is proved to be one. -/
def compileCombination {sourceWidth logicalWidth : Nat}
    (sourceMap : SourceMap sourceWidth logicalWidth)
    (oneColumn : Fin logicalWidth) (combination : R1CS.LinearCombination)
    (bounded : CombinationBounded sourceWidth combination) :
    SparseForm logicalWidth :=
  SparseForm.add (SparseForm.singleton oneColumn combination.constant)
    (compileTerms sourceMap combination.terms bounded)

/-- Exact source semantics of one compiled affine form. -/
theorem compileCombination_eval {sourceWidth logicalWidth : Nat}
    (sourceMap : SourceMap sourceWidth logicalWidth)
    (oneColumn : Fin logicalWidth) (combination : R1CS.LinearCombination)
    (bounded : CombinationBounded sourceWidth combination)
    (assignment : Assignment F logicalWidth) (source : Circuit.Env)
    (one : assignment oneColumn = 1)
    (preserves : sourceMap.Preserves assignment source) :
    (compileCombination sourceMap oneColumn combination bounded).eval assignment =
      combination.eval source := by
  rw [compileCombination, SparseForm.add_eval, SparseForm.singleton_eval,
    one, mul_one, compileTerms_eval sourceMap assignment source preserves]
  rfl

/-- Compile one bounded ordinary R1CS row into the four live selective forms. -/
def compileRow {sourceWidth logicalWidth : Nat}
    (sourceMap : SourceMap sourceWidth logicalWidth)
    (oneColumn : Fin logicalWidth) (row : R1CS.Row)
    (bounded : RowBounded sourceWidth row) : OrdinaryRow.Forms logicalWidth :=
  { selector := SparseForm.singleton oneColumn 1
    a := compileCombination sourceMap oneColumn row.a bounded.1
    b := compileCombination sourceMap oneColumn row.b bounded.2.1
    c := compileCombination sourceMap oneColumn row.c bounded.2.2 }

/-- A compiled bounded row satisfies the exact preservation premise of the
ordinary-row selective semantics. -/
theorem compileRow_preserves {sourceWidth logicalWidth : Nat}
    (sourceMap : SourceMap sourceWidth logicalWidth)
    (oneColumn : Fin logicalWidth) (row : R1CS.Row)
    (bounded : RowBounded sourceWidth row)
    (assignment : Assignment F logicalWidth) (source : Circuit.Env)
    (one : assignment oneColumn = 1)
    (preserves : sourceMap.Preserves assignment source) :
    (compileRow sourceMap oneColumn row bounded).Preserves
      assignment source row := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [compileRow, one]
  · exact compileCombination_eval sourceMap oneColumn row.a bounded.1
      assignment source one preserves
  · exact compileCombination_eval sourceMap oneColumn row.b bounded.2.1
      assignment source one preserves
  · exact compileCombination_eval sourceMap oneColumn row.c bounded.2.2
      assignment source one preserves

/-- Total R1CS environment induced by one bounded source assignment. Indices
outside the declared source domain are canonical zero. -/
def sourceEnv {sourceWidth : Nat} (source : Fin sourceWidth → F) : Circuit.Env :=
  fun index => if bounded : index < sourceWidth then source ⟨index, bounded⟩ else 0

@[simp] theorem sourceEnv_at {sourceWidth : Nat}
    (source : Fin sourceWidth → F) (column : Fin sourceWidth) :
    sourceEnv source column.val = source column := by
  unfold sourceEnv
  rw [dif_pos column.isLt]

/-- A source map that retains every source coordinate at one owned slot. -/
structure RetainedMap {sourceWidth : Nat}
    (slots : List (LowNormAssignment.Slot sourceWidth)) where
  slotFor : Fin sourceWidth → Fin slots.length
  owns : ∀ column, (slots.get (slotFor column)).source = column

namespace RetainedMap

def sourceMap {sourceWidth : Nat}
    {slots : List (LowNormAssignment.Slot sourceWidth)}
    (retained : RetainedMap slots) :
    SourceMap sourceWidth (ProductionAssignment.logicalWidth slots) where
  form := fun column => RetainedSlot.form slots (retained.slotFor column)

/-- Retained slot ownership is sufficient for exact source reconstruction. -/
theorem preserves {sourceWidth : Nat}
    (publicInput : Fin ProductionAssignment.publicWidth → F)
    (slots : List (LowNormAssignment.Slot sourceWidth))
    (source : Fin sourceWidth → F) (retained : RetainedMap slots) :
    retained.sourceMap.Preserves
      (ProductionAssignment.logicalAssignment publicInput slots source)
      (sourceEnv source) := by
  intro column
  rw [sourceMap, RetainedSlot.form_eval, retained.owns column, sourceEnv_at]

end RetainedMap

/-- The verifier-owned constant-one column in the recursive public prefix. -/
def markerColumn {sourceWidth : Nat}
    {slots : List (LowNormAssignment.Slot sourceWidth)} :
    Fin (ProductionAssignment.logicalWidth slots) :=
  ProductionAssignment.publicColumn Lifecycle.encHashMarkerIndex

/-- The canonical recursive public input fixes the marker column to one. -/
theorem markerColumn_eq_one {sourceWidth : Nat} (digest : Lifecycle.Digest)
    (slots : List (LowNormAssignment.Slot sourceWidth))
    (source : Fin sourceWidth → F) :
    ProductionAssignment.logicalAssignment
        (Lifecycle.encHash (publicFits := ProductionAssignment.publicFits slots) digest)
        slots source markerColumn = 1 := by
  rw [markerColumn, ProductionAssignment.logicalAssignment_publicColumn]
  exact Lifecycle.encHash_marker digest

end NightstreamFPrime.Layout.ProductionRelation.SourceCompiler
