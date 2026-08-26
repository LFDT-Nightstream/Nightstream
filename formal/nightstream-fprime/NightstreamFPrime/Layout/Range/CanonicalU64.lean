import NightstreamFPrime.Gadgets.Range.CanonicalU64
import NightstreamFPrime.Layout.R1CS.Completeness

/-!
Owns physical R1CS lowering for one canonical Goldilocks field lane.

The logical child allocates 66 values. The current optimized lowering adds
197 multiplication values and emits 264 rows. This owner changes no logical
predicate and adds no boundary-copy row.
-/

namespace NightstreamFPrime.Layout.Range.CanonicalU64

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Range
open NightstreamFPrime.Layout

namespace Logical

abbrev Interface :=
  NightstreamFPrime.Gadgets.Range.CanonicalU64.Interface
abbrev auxiliaryCount :=
  NightstreamFPrime.Gadgets.Range.CanonicalU64.auxiliaryCount
abbrev exactRowCount :=
  NightstreamFPrime.Gadgets.Range.CanonicalU64.exactRowCount
abbrev flagRecipe :=
  NightstreamFPrime.Gadgets.Range.CanonicalU64.flagRecipe
abbrev booleanConstraint :=
  NightstreamFPrime.Gadgets.Range.CanonicalU64.booleanConstraint
abbrev booleanConstraints :=
  NightstreamFPrime.Gadgets.Range.CanonicalU64.booleanConstraints
abbrev recompositionConstraint :=
  NightstreamFPrime.Gadgets.Range.CanonicalU64.recompositionConstraint
abbrev canonicalityConstraint :=
  NightstreamFPrime.Gadgets.Range.CanonicalU64.canonicalityConstraint
abbrev operations :=
  NightstreamFPrime.Gadgets.Range.CanonicalU64.operations
abbrev flatConstraints_operations :=
  NightstreamFPrime.Gadgets.Range.CanonicalU64.flatConstraints_operations
abbrev circuit :=
  NightstreamFPrime.Gadgets.Range.CanonicalU64.circuit
abbrev Assumptions :=
  NightstreamFPrime.Gadgets.Range.CanonicalU64.Assumptions
abbrev SpecHolds :=
  NightstreamFPrime.Gadgets.Range.CanonicalU64.SpecHolds
abbrev soundness :=
  NightstreamFPrime.Gadgets.Range.CanonicalU64.soundness
abbrev complete :=
  NightstreamFPrime.Gadgets.Range.CanonicalU64.complete
abbrev completeness :=
  NightstreamFPrime.Gadgets.Range.CanonicalU64.completeness
abbrev localLength_eq :=
  NightstreamFPrime.Gadgets.Range.CanonicalU64.localLength_eq
abbrev flatConstraints_varsBelow :=
  NightstreamFPrime.Gadgets.Range.CanonicalU64.flatConstraints_varsBelow

end Logical

structure InputsAffine (interface : Logical.Interface) (offset : Nat) : Prop where
  source : R1CS.IsAffine (interface.source offset)

def logicalConstraints (interface : Logical.Interface) (offset : Nat) :
    List Expr :=
  flatConstraints (Logical.operations interface offset)

private theorem sub_affine {left right : Expr}
    (leftAffine : R1CS.IsAffine left) (rightAffine : R1CS.IsAffine right) :
    R1CS.IsAffine (left - right) :=
  R1CS.IsAffine.add leftAffine
    (R1CS.IsAffine.const_mul (-1) rightAffine)

private theorem foldl_weighted_affine (indices : List Nat)
    (coefficient : Nat → NightstreamFPrime.Spec.F) (bit : Nat → Expr)
    (bitsAffine : ∀ index, R1CS.IsAffine (bit index))
    (initial : Expr) (initialAffine : R1CS.IsAffine initial) :
    R1CS.IsAffine (indices.foldl (fun value index =>
      value + Expr.const (coefficient index) * bit index) initial) := by
  induction indices generalizing initial with
  | nil => exact initialAffine
  | cons index rest inductionHypothesis =>
      apply inductionHypothesis
      exact R1CS.IsAffine.add initialAffine
        (R1CS.IsAffine.const_mul _ (bitsAffine index))

private theorem weighted_affine (offset start count : Nat) :
    R1CS.IsAffine (CanonicalU64.weightedExpr offset start count) := by
  unfold CanonicalU64.weightedExpr
  exact foldl_weighted_affine _ _ _
    (fun _ => R1CS.isAffine_var _) 0 (R1CS.isAffine_const _)

private theorem word_affine (offset : Nat) :
    R1CS.IsAffine (CanonicalU64.wordExpr offset) := by
  unfold CanonicalU64.wordExpr CanonicalU64.lowExpr CanonicalU64.highExpr
  exact R1CS.IsAffine.add (weighted_affine _ _ _)
    (R1CS.IsAffine.const_mul _ (weighted_affine _ _ _))

private theorem recomposition_affine (interface : Logical.Interface)
    (offset : Nat) (inputs : InputsAffine interface offset) :
    R1CS.IsAffine (Logical.recompositionConstraint interface offset) := by
  unfold Logical.recompositionConstraint
  exact sub_affine inputs.source (word_affine offset)

private theorem flagFreshCount_eq (offset : Nat) :
    R1CS.totalFreshCount (recipeConstraints
      (offset + CanonicalU64.bitCount + 1) [Logical.flagRecipe offset]) = 36 := by
  rfl

private theorem flagRowCount_eq (offset : Nat) :
    R1CS.totalRowCount (recipeConstraints
      (offset + CanonicalU64.bitCount + 1) [Logical.flagRecipe offset]) = 37 := by
  rfl

private theorem booleanFreshCount_eq (offset index : Nat) :
    R1CS.constraintFreshCount (Logical.booleanConstraint offset index) = 2 := by
  rfl

private theorem booleanRowCount_eq (offset index : Nat) :
    R1CS.constraintRowCount (Logical.booleanConstraint offset index) = 3 := by
  rfl

private theorem booleanFreshTotal_eq (offset : Nat) :
    R1CS.totalFreshCount (Logical.booleanConstraints offset) = 128 := by
  unfold R1CS.totalFreshCount
  change (List.map R1CS.constraintFreshCount
    ((List.range CanonicalU64.bitCount).map
      (Logical.booleanConstraint offset))).sum = 128
  rw [List.map_map]
  simp [booleanFreshCount_eq, CanonicalU64.bitCount, Function.comp_def]

private theorem booleanRowTotal_eq (offset : Nat) :
    R1CS.totalRowCount (Logical.booleanConstraints offset) = 192 := by
  unfold R1CS.totalRowCount
  change (List.map R1CS.constraintRowCount
    ((List.range CanonicalU64.bitCount).map
      (Logical.booleanConstraint offset))).sum = 192
  rw [List.map_map]
  simp [booleanRowCount_eq, CanonicalU64.bitCount, Function.comp_def]

private theorem canonicalityFreshCount_eq (offset : Nat) :
    R1CS.constraintFreshCount (Logical.canonicalityConstraint offset) = 33 := by
  rfl

private theorem canonicalityRowCount_eq (offset : Nat) :
    R1CS.constraintRowCount (Logical.canonicalityConstraint offset) = 34 := by
  rfl

theorem totalFreshCount_eq (interface : Logical.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    R1CS.totalFreshCount (logicalConstraints interface offset) = 197 := by
  unfold logicalConstraints
  rw [Logical.flatConstraints_operations,
    R1CS.totalFreshCount_append, R1CS.totalFreshCount_append,
    flagFreshCount_eq, booleanFreshTotal_eq]
  simp only [R1CS.totalFreshCount, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, Nat.add_zero]
  rw [R1CS.constraintFreshCount_eq_zero_of_affine _
      (recomposition_affine interface offset inputs),
    canonicalityFreshCount_eq]

theorem totalRowCount_eq (interface : Logical.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    R1CS.totalRowCount (logicalConstraints interface offset) = 264 := by
  unfold logicalConstraints
  rw [Logical.flatConstraints_operations,
    R1CS.totalRowCount_append, R1CS.totalRowCount_append,
    flagRowCount_eq, booleanRowTotal_eq]
  simp only [R1CS.totalRowCount, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, Nat.add_zero]
  rw [R1CS.constraintRowCount_eq_one_of_affine _
      (recomposition_affine interface offset inputs),
    canonicalityRowCount_eq]

def footprint (interface : Logical.Interface)
    (inputs : ∀ offset, InputsAffine interface offset) :
    R1CS.CircuitFootprint (Logical.circuit interface) where
  freshColumnCount := fun _ => 197
  physicalRowCount := fun _ => 264
  freshColumnCount_eq := fun offset =>
    totalFreshCount_eq interface offset (inputs offset)
  physicalRowCount_eq := fun offset =>
    totalRowCount_eq interface offset (inputs offset)

theorem physicalPrivateColumnCount_eq (interface : Logical.Interface)
    (offset : Nat) (inputs : InputsAffine interface offset) :
    localLength (Circuit.ops (Logical.circuit interface).main offset) +
      R1CS.totalFreshCount (logicalConstraints interface offset) = 263 := by
  change localLength (Logical.operations interface offset) +
    R1CS.totalFreshCount (logicalConstraints interface offset) = 263
  rw [Logical.localLength_eq, totalFreshCount_eq interface offset inputs]
  rfl

def plan (interface : Logical.Interface) (offset : Nat) : R1CS.LoweringPlan where
  constraints := logicalConstraints interface offset
  firstFresh := offset + Logical.auxiliaryCount

def PhysicalHolds (interface : Logical.Interface) (offset : Nat)
    (env : Env) : Prop :=
  R1CS.RowsHold env (plan interface offset).rows

theorem physical_implies_spec (interface : Logical.Interface) (offset : Nat)
    (env : Env) (assumptions : Logical.Assumptions interface offset env)
    (physical : PhysicalHolds interface offset env) :
    Logical.SpecHolds interface offset env := by
  apply Logical.soundness interface env offset assumptions
  apply holdsFlat_implies_holds
  change ConstraintsHold env (logicalConstraints interface offset)
  exact R1CS.LoweringPlan.sound (plan interface offset) env physical

theorem physical_complete (interface : Logical.Interface) (offset : Nat)
    (env : Env) (inputs : InputsAffine interface offset)
    (assumptions : Logical.Assumptions interface offset env)
    (specification : Logical.SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset 263 ∧
      PhysicalHolds interface offset completed := by
  rcases Logical.completeness interface env offset assumptions specification with
    ⟨logicalEnv, logicalAgrees, logicalRows⟩
  have logicalAgreesFixed :
      AgreesOutside env logicalEnv offset Logical.auxiliaryCount := by
    rw [Logical.localLength_eq] at logicalAgrees
    exact logicalAgrees
  have scope : ∀ expression ∈ logicalConstraints interface offset,
      expression.VarsBelow (offset + Logical.auxiliaryCount) :=
    Logical.flatConstraints_varsBelow interface offset assumptions
  rcases R1CS.lowerConstraints_complete logicalEnv
      (logicalConstraints interface offset) (offset + Logical.auxiliaryCount)
      scope logicalRows with
    ⟨completed, physicalAgrees, rows⟩
  refine ⟨completed, ?_, rows⟩
  have combined := logicalAgreesFixed.append physicalAgrees
  rw [totalFreshCount_eq interface offset inputs] at combined
  simpa [Logical.auxiliaryCount] using combined

end NightstreamFPrime.Layout.Range.CanonicalU64
