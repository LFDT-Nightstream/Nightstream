import NightstreamFPrime.Export.Stage1.ConstraintRenaming
import NightstreamFPrime.Layout.PiDEC.v1_1.PublicInputSplit
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.RingKRecomposition

/-! Column transport for the two PiDEC leaves. Each identity is structural in
one leaf and exposes an interface that its parent can compose. -/

namespace NightstreamFPrime.Export.Stage1.PiDECLeafRenaming

open NightstreamFPrime.Circuit NightstreamFPrime.Layout NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle.PiDEC.v1_1
open CompactRows

private theorem weightedFold_rename (column : Nat → Nat) :
    ∀ (values : List Expr) (weights : List F),
      renameExpr column ((values.zip weights).foldr
        (fun pair suffix => Expr.const pair.2 * pair.1 + suffix) 0) =
      (((values.map (renameExpr column)).zip weights).foldr
        (fun pair suffix => Expr.const pair.2 * pair.1 + suffix) 0)
  | [], _ => rfl
  | _ :: _, [] => rfl
  | value :: values, weight :: weights => by
    exact congrArg (Expr.add (Expr.mul (Expr.const weight) (renameExpr column value)))
      (weightedFold_rename column values weights)

theorem signedSplit (column : Nat → Nat)
    (before after : SignedSplitScalar.Interface) (start finish : Nat)
    (sign : column start = finish)
    (parent : renameExpr column (before.parent start) = after.parent finish)
    (digit : ∀ index, renameExpr column (before.digit start index) = after.digit finish index) :
    (SignedSplitScalar.constraints before start).map (renameExpr column) =
      SignedSplitScalar.constraints after finish := by
  have recompose : renameExpr column (SignedSplitScalar.recomposeExpr before start) =
      SignedSplitScalar.recomposeExpr after finish := by
    unfold SignedSplitScalar.recomposeExpr
    rw [weightedFold_rename, List.map_ofFn]
    simp only [Function.comp_def]
    rw [funext digit]
  have signExpr : renameExpr column (SignedSplitScalar.signExpr start) =
      SignedSplitScalar.signExpr finish := by
    simp only [SignedSplitScalar.signExpr, SignedSplitScalar.signBitExpr, renameExpr_sub]
    change 1 - 2 * Expr.var (column start) = 1 - 2 * Expr.var finish
    rw [sign]
  simp only [SignedSplitScalar.constraints, List.map_cons, List.map_append, List.map_nil]
  apply congrArg₂ List.cons
  · change Expr.mul (Expr.var (column start)) (Expr.var (column start) - 1) =
      Expr.mul (Expr.var finish) (Expr.var finish - 1)
    rw [sign]
  · apply congrArg₂ List.append
    · simp only [SignedSplitScalar.digitConstraints, List.map_ofFn]
      apply congrArg List.ofFn
      funext index
      change Expr.mul (renameExpr column (before.digit start index))
          (renameExpr column (before.digit start index - SignedSplitScalar.signExpr start)) = _
      rw [renameExpr_sub, digit, signExpr]
      rfl
    · apply congrArg (fun value => [value])
      change renameExpr column (SignedSplitScalar.recomposeExpr before start - before.parent start) = _
      rw [renameExpr_sub, recompose, parent]
      rfl

theorem signedSplit_rows (column : Nat → Nat)
    (before after : SignedSplitScalar.Interface) (start finish : Nat)
    (sign : column start = finish)
    (parent : renameExpr column (before.parent start) = after.parent finish)
    (digit : ∀ index, renameExpr column (before.digit start index) = after.digit finish index) :
    (flatConstraints (Circuit.ops (SignedSplitScalar.circuit before).main start)).map (renameExpr column) =
      flatConstraints (Circuit.ops (SignedSplitScalar.circuit after).main finish) := by
  change (flatConstraints (SignedSplitScalar.operations before start)).map _ =
    flatConstraints (SignedSplitScalar.operations after finish)
  rw [SignedSplitScalar.flatConstraints_operations, SignedSplitScalar.flatConstraints_operations]
  exact signedSplit column before after start finish sign parent digit

theorem radix {count : Nat} (column : Nat → Nat)
    (before after : RadixRecomposition.Interface count) (start finish : Nat)
    (parent : ∀ coordinate, renameExpr column (before.parent start coordinate) = after.parent finish coordinate)
    (child : ∀ index coordinate, renameExpr column (before.child start index coordinate) =
      after.child finish index coordinate) :
    (RadixRecomposition.constraints before start).map (renameExpr column) =
      RadixRecomposition.constraints after finish := by
  simp only [RadixRecomposition.constraints, List.map_ofFn]
  apply congrArg List.ofFn
  funext coordinate
  simp only [Function.comp_def, RadixRecomposition.constraint]
  rw [renameExpr_sub, parent]
  apply congrArg (fun value => after.parent finish coordinate - value)
  unfold RadixRecomposition.recomposeExpr
  rw [weightedFold_rename, List.map_ofFn]
  simp only [Function.comp_def]
  rw [funext (fun index => child index coordinate)]

theorem radix_rows {count : Nat} (column : Nat → Nat)
    (before after : RadixRecomposition.Interface count) (start finish : Nat)
    (parent : ∀ coordinate, renameExpr column (before.parent start coordinate) = after.parent finish coordinate)
    (child : ∀ index coordinate, renameExpr column (before.child start index coordinate) =
      after.child finish index coordinate) :
    (flatConstraints (Circuit.ops (RadixRecomposition.circuit before).main start)).map (renameExpr column) =
      flatConstraints (Circuit.ops (RadixRecomposition.circuit after).main finish) := by
  change (flatConstraints (RadixRecomposition.operations before start)).map _ =
    flatConstraints (RadixRecomposition.operations after finish)
  rw [RadixRecomposition.flatConstraints_operations, RadixRecomposition.flatConstraints_operations]
  exact radix column before after start finish parent child

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint NightstreamFPrime.Lifecycle.PaperAlgebra

/-- The public-input parent composes the scalar certificate in source order. -/
theorem publicInput_rows {width : Nat}
    {fits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}
    (column : Nat → Nat) (before after : PublicInputSplit.Interface width fits)
    (start finish : Nat)
    (sign : ∀ source, source < PublicInputSplit.coordinateCount width fits →
      column (PublicInputSplit.sourceOffset start source) = PublicInputSplit.sourceOffset finish source)
    (parent : ∀ coordinate, renameExpr column (before.parent start coordinate) = after.parent finish coordinate)
    (digit : ∀ child coordinate, renameExpr column (before.digit start child coordinate) =
      after.digit finish child coordinate) :
    (PiDEC.v1_1.PublicInputSplit.logicalConstraints before start).map (renameExpr column) =
      PiDEC.v1_1.PublicInputSplit.logicalConstraints after finish := by
  rw [PiDEC.v1_1.PublicInputSplit.logicalConstraints_eq_ordered,
    PiDEC.v1_1.PublicInputSplit.logicalConstraints_eq_ordered]
  unfold PiDEC.v1_1.PublicInputSplit.orderedConstraints PiDEC.v1_1.PublicInputSplit.childConstraintLists
  rw [List.map_flatten, List.map_map]
  apply congrArg List.flatten
  apply List.map_congr_left
  intro source member
  have bounded := List.mem_range.mp member
  dsimp only [Function.comp_def]
  unfold PiDEC.v1_1.PublicInputSplit.childConstraints
  simp only [PiDEC.v1_1.PublicInputSplit.Logical.childOp, PublicInputSplit.childOp, dif_pos bounded]
  change (flatConstraints (Circuit.ops (SignedSplitScalar.circuit _).main _)).map _ =
    flatConstraints (Circuit.ops (SignedSplitScalar.circuit _).main _)
  apply signedSplit_rows
  · exact sign source bounded
  · exact parent ⟨source, bounded⟩
  · intro child
    exact digit child ⟨source, bounded⟩

/-- Both extension cells keep their order under the scalar row map. -/
theorem ringK_rows {blocks : Nat} (column : Nat → Nat)
    (before after : RingKRecomposition.Interface blocks) (start finish : Nat)
    (parent : ∀ block lane,
      renameExpr column (before.parent start block lane).c0 = (after.parent finish block lane).c0 ∧
      renameExpr column (before.parent start block lane).c1 = (after.parent finish block lane).c1)
    (child : ∀ index block lane,
      renameExpr column (before.child start index block lane).c0 = (after.child finish index block lane).c0 ∧
      renameExpr column (before.child start index block lane).c1 = (after.child finish index block lane).c1) :
    (flatConstraints (Circuit.ops (RingKRecomposition.circuit before).main start)).map (renameExpr column) =
      flatConstraints (Circuit.ops (RingKRecomposition.circuit after).main finish) := by
  apply radix_rows
  · intro coordinate
    dsimp only [RingKRecomposition.scalarInterface]
    unfold RingKRecomposition.expressionCell
    split
    · exact (parent _ _).1
    · exact (parent _ _).2
  · intro index coordinate
    dsimp only [RingKRecomposition.scalarInterface]
    unfold RingKRecomposition.expressionCell
    split
    · exact (child _ _ _).1
    · exact (child _ _ _).2

end NightstreamFPrime.Export.Stage1.PiDECLeafRenaming
