import NightstreamFPrime.Circuit.VariableSupport
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.RadixRecomposition
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.RingKRecomposition

/-!
Owns generic variable-support preservation for the two PiDEC circuit shapes:
the signed public-input split and fixed-radix recomposition.

This module selects no Stage 1 column and changes no circuit definition.
-/

namespace NightstreamFPrime.Lifecycle.PiDEC.v1_1

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

namespace Support

private theorem weightedFold_varsSatisfy (allowed : Nat → Prop) :
    ∀ (values : List Expr) (weights : List F),
      (∀ value ∈ values, value.VarsSatisfy allowed) →
      ((values.zip weights).foldr
        (fun pair suffix => Expr.const pair.2 * pair.1 + suffix) 0
      ).VarsSatisfy allowed
  | [], _, _ => trivial
  | _ :: _, [], _ => trivial
  | value :: values, _ :: weights, supported =>
      ⟨⟨trivial, supported value (by simp)⟩,
        weightedFold_varsSatisfy allowed values weights
          (fun item member => supported item (by simp [member]))⟩

end Support

namespace SignedSplitScalar

theorem recomposeExpr_varsSatisfy
    (allowed : Nat → Prop) (interface : Interface) (offset : Nat)
    (digits : ∀ index, (interface.digit offset index).VarsSatisfy allowed) :
    (recomposeExpr interface offset).VarsSatisfy allowed := by
  unfold recomposeExpr
  apply Support.weightedFold_varsSatisfy
  intro expression member
  rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
  exact digits index

theorem flatConstraints_varsSatisfy
    (allowed : Nat → Prop) (interface : Interface) (offset : Nat)
    (parent : (interface.parent offset).VarsSatisfy allowed)
    (digits : ∀ index, (interface.digit offset index).VarsSatisfy allowed)
    (localCell : allowed offset) :
    ∀ expression ∈ flatConstraints (operations interface offset),
      expression.VarsSatisfy allowed := by
  rw [flatConstraints_operations]
  intro expression member
  rw [constraints] at member
  rcases List.mem_cons.mp member with signMember | tailMember
  · subst expression
    simp only [signConstraint, signBitExpr, Expr.sub, Expr.neg,
      Expr.VarsSatisfy]
    exact ⟨localCell, ⟨localCell, ⟨trivial, trivial⟩⟩⟩
  · rcases List.mem_append.mp tailMember with digitMember | recompositionMember
    · rcases List.mem_ofFn.mp digitMember with ⟨index, rfl⟩
      have signSupported : (signExpr offset).VarsSatisfy allowed := by
        simp only [signExpr, signBitExpr, Expr.sub, Expr.neg,
          Expr.VarsSatisfy]
        exact ⟨trivial, ⟨trivial, ⟨trivial, localCell⟩⟩⟩
      simp only [digitConstraint, Expr.sub, Expr.neg, Expr.VarsSatisfy]
      exact ⟨digits index, ⟨digits index, ⟨trivial, signSupported⟩⟩⟩
    · rw [List.mem_singleton] at recompositionMember
      subst expression
      simp only [recompositionConstraint, Expr.sub, Expr.neg,
        Expr.VarsSatisfy]
      exact ⟨recomposeExpr_varsSatisfy allowed interface offset digits,
        trivial, parent⟩

end SignedSplitScalar

namespace RadixRecomposition

theorem recomposeExpr_varsSatisfy {coordinateCount : Nat}
    (allowed : Nat → Prop) (interface : Interface coordinateCount)
    (offset : Nat) (coordinate : Fin coordinateCount)
    (children : ∀ child,
      (interface.child offset child coordinate).VarsSatisfy allowed) :
    (recomposeExpr interface offset coordinate).VarsSatisfy allowed := by
  unfold recomposeExpr
  apply Support.weightedFold_varsSatisfy
  intro expression member
  rcases List.mem_ofFn.mp member with ⟨child, rfl⟩
  exact children child

theorem flatConstraints_varsSatisfy {coordinateCount : Nat}
    (allowed : Nat → Prop) (interface : Interface coordinateCount)
    (offset : Nat)
    (parent : ∀ coordinate,
      (interface.parent offset coordinate).VarsSatisfy allowed)
    (children : ∀ child coordinate,
      (interface.child offset child coordinate).VarsSatisfy allowed) :
    ∀ expression ∈ flatConstraints (operations interface offset),
      expression.VarsSatisfy allowed := by
  rw [flatConstraints_operations]
  intro expression member
  rcases List.mem_ofFn.mp member with ⟨coordinate, rfl⟩
  simp only [constraint, Expr.sub, Expr.neg, Expr.VarsSatisfy]
  exact ⟨parent coordinate, trivial,
    recomposeExpr_varsSatisfy allowed interface offset coordinate
      (fun child => children child coordinate)⟩

end RadixRecomposition

namespace RingKRecomposition

def KSupported (allowed : Nat → Prop) (value : KExpr) : Prop :=
  value.c0.VarsSatisfy allowed ∧ value.c1.VarsSatisfy allowed

theorem expressionCell_varsSatisfy
    (allowed : Nat → Prop) (cell : Fin cellCount) (value : KExpr)
    (supported : KSupported allowed value) :
    (expressionCell cell value).VarsSatisfy allowed := by
  fin_cases cell
  · simpa [expressionCell, cellCount] using supported.1
  · simpa [expressionCell, cellCount] using supported.2

theorem flatConstraints_varsSatisfy {blockCount : Nat}
    (allowed : Nat → Prop) (interface : Interface blockCount) (offset : Nat)
    (parent : ∀ block lane,
      KSupported allowed (interface.parent offset block lane))
    (children : ∀ child block lane,
      KSupported allowed (interface.child offset child block lane)) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (circuit interface).main offset),
      expression.VarsSatisfy allowed := by
  rw [RadixRecomposition.circuit_ops]
  apply RadixRecomposition.flatConstraints_varsSatisfy allowed
  · intro coordinate
    apply expressionCell_varsSatisfy
    exact parent (coordinates coordinate).1 (coordinates coordinate).2.1
  · intro child coordinate
    apply expressionCell_varsSatisfy
    exact children child (coordinates coordinate).1
      (coordinates coordinate).2.1

end RingKRecomposition

namespace PublicInputSplit

theorem flatConstraints_varsSatisfy
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (allowed : Nat → Prop) (interface : Interface logicalWidth publicFits)
    (offset : Nat)
    (parent : ∀ coordinate,
      (interface.parent offset coordinate).VarsSatisfy allowed)
    (digits : ∀ child coordinate,
      (interface.digit offset child coordinate).VarsSatisfy allowed)
    (locals : ∀ index, offset ≤ index →
      index < offset + logicalPrivateCount logicalWidth publicFits →
      allowed index) :
    ∀ expression ∈ flatConstraints (Circuit.ops (main interface) offset),
      expression.VarsSatisfy allowed := by
  change ∀ expression ∈ flatConstraints (opsAt interface offset), _
  intro expression member
  rcases List.mem_flatMap.mp member with
    ⟨operation, operationMember, expressionMember⟩
  rcases List.mem_map.mp operationMember with ⟨source, sourceMember, rfl⟩
  have sourceLt := List.mem_range.mp sourceMember
  have childMember : expression ∈ flatConstraints
      (SignedSplitScalar.operations
        (childInterface interface offset source sourceLt)
        (sourceOffset offset source)) := by
    simpa [childOp, dif_pos sourceLt, Sequence.childOp] using expressionMember
  apply SignedSplitScalar.flatConstraints_varsSatisfy allowed
    (childInterface interface offset source sourceLt)
    (sourceOffset offset source)
  · exact parent ⟨source, sourceLt⟩
  · intro child
    exact digits child ⟨source, sourceLt⟩
  · apply locals (sourceOffset offset source)
    · simp [sourceOffset]
    · have sourceLt270 : source < 270 := by
        simpa only [coordinateCount_eq] using sourceLt
      rw [logicalPrivateCount_eq]
      simp [sourceOffset, SignedSplitScalar.exactPrivateCount]
      omega
  · exact childMember

end PublicInputSplit

end NightstreamFPrime.Lifecycle.PiDEC.v1_1
