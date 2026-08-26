import NightstreamFPrime.Gadgets.Sampling.Candidate16Five
import NightstreamFPrime.Layout.R1CS.Completeness

/-!
Owns physical R1CS lowering for one 16-bit PiRLC candidate decoded from a
canonical-u64 bit vector.

The logical decoder is generic over caller expressions. This owner fixes its
only production shape: 16 adjacent bits selected from one canonical 64-bit
word. Lowering starts after the decoder's 17 logical variables. Exact physical
counts include every multiplication intermediate introduced by lowering.
-/

namespace NightstreamFPrime.Layout.Sampling.Candidate16Five

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Range
open NightstreamFPrime.Layout

namespace Logical

abbrev Interface :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.Interface
abbrev candidateBitCount :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.candidateBitCount
abbrev auxiliaryCount :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.auxiliaryCount
abbrev weightedExpr :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.weightedExpr
abbrev canonicalWindowInterface :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.canonicalWindowInterface
abbrev operations :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.operations
abbrev flatConstraints_operations :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.flatConstraints_operations
abbrev rejectRecipe :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.rejectRecipe
abbrev productExpr :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.productExpr
abbrev quotientBooleanConstraint :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.quotientBooleanConstraint
abbrev quotientBitExpr :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.quotientBitExpr
abbrev quotientExpr :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.quotientExpr
abbrev remainderExpr :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.remainderExpr
abbrev quotientRecompositionConstraint :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.quotientRecompositionConstraint
abbrev quotientWordExpr :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.quotientWordExpr
abbrev divisionConstraint :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.divisionConstraint
abbrev remainderConstraint :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.remainderConstraint
abbrev quotientBitCount :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.quotientBitCount
abbrev Assumptions :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.Assumptions
abbrev SpecHolds :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.SpecHolds
abbrev circuit :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.circuit
abbrev soundness :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.soundness
abbrev completeness :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.completeness
abbrev localLength_eq :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.localLength_eq
abbrev flatConstraints_varsBelow :=
  NightstreamFPrime.Gadgets.Sampling.Candidate16Five.flatConstraints_varsBelow

end Logical

/-- Exact low or high 16-bit window of one canonical-u64 decomposition. -/
def interface (wordOffset : Nat) (part : Fin 2) : Logical.Interface where
  candidate := (Logical.canonicalWindowInterface wordOffset part).candidate
  candidateBit :=
    (Logical.canonicalWindowInterface wordOffset part).candidateBit

def logicalConstraints (wordOffset : Nat) (part : Fin 2)
    (offset : Nat) : List Expr :=
  flatConstraints (Logical.operations (interface wordOffset part) offset)

def plan (wordOffset : Nat) (part : Fin 2) (offset : Nat) :
    R1CS.LoweringPlan where
  constraints := logicalConstraints wordOffset part offset
  firstFresh := offset + Logical.auxiliaryCount

private theorem sub_affine {left right : Expr}
    (leftAffine : R1CS.IsAffine left) (rightAffine : R1CS.IsAffine right) :
    R1CS.IsAffine (left - right) :=
  R1CS.IsAffine.add leftAffine (R1CS.IsAffine.const_mul (-1) rightAffine)

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

private theorem logicalWeighted_affine (bit : Nat → Expr) (count : Nat)
    (bitsAffine : ∀ index, R1CS.IsAffine (bit index)) :
    R1CS.IsAffine (Logical.weightedExpr bit count) := by
  unfold Logical.weightedExpr
  exact foldl_weighted_affine _ _ bit bitsAffine 0 (R1CS.isAffine_const _)

private theorem canonicalWeighted_affine (wordOffset start count : Nat) :
    R1CS.IsAffine (CanonicalU64.weightedExpr wordOffset start count) := by
  unfold CanonicalU64.weightedExpr
  exact foldl_weighted_affine _ _ _
    (fun index => R1CS.isAffine_var _) 0 (R1CS.isAffine_const _)

private theorem quotientRecomposition_affine (offset : Nat) :
    R1CS.IsAffine
      (NightstreamFPrime.Gadgets.Sampling.Candidate16Five.quotientRecompositionConstraint
        offset) := by
  apply sub_affine (R1CS.isAffine_var _)
  exact logicalWeighted_affine _ _ (fun index => R1CS.isAffine_var _)

private theorem division_affine (wordOffset : Nat) (part : Fin 2)
    (offset : Nat) :
    R1CS.IsAffine
      (NightstreamFPrime.Gadgets.Sampling.Candidate16Five.divisionConstraint
        (interface wordOffset part) offset) := by
  apply sub_affine
  · exact canonicalWeighted_affine _ _ _
  · exact R1CS.IsAffine.add
      (R1CS.IsAffine.const_mul _ (R1CS.isAffine_var _))
      (R1CS.isAffine_var _)

private theorem rejectFreshCount_eq (wordOffset : Nat) (part : Fin 2)
    (offset : Nat) :
    R1CS.totalFreshCount (recipeConstraints (offset + 16)
      [Logical.rejectRecipe (interface wordOffset part) offset]) = 17 := by
  rfl

private theorem rejectRowCount_eq (wordOffset : Nat) (part : Fin 2)
    (offset : Nat) :
    R1CS.totalRowCount (recipeConstraints (offset + 16)
      [Logical.rejectRecipe (interface wordOffset part) offset]) = 18 := by
  rfl

private theorem booleanFreshCount_eq (offset index : Nat) :
    R1CS.constraintFreshCount
      (NightstreamFPrime.Gadgets.Sampling.Candidate16Five.quotientBooleanConstraint
        offset index) = 2 := by
  rfl

private theorem booleanRowCount_eq (offset index : Nat) :
    R1CS.constraintRowCount
      (NightstreamFPrime.Gadgets.Sampling.Candidate16Five.quotientBooleanConstraint
        offset index) = 3 := by
  rfl

private theorem remainderFreshCount_eq (offset : Nat) :
    R1CS.constraintFreshCount
      (NightstreamFPrime.Gadgets.Sampling.Candidate16Five.remainderConstraint
        offset) = 8 := by
  rfl

private theorem remainderRowCount_eq (offset : Nat) :
    R1CS.constraintRowCount
      (NightstreamFPrime.Gadgets.Sampling.Candidate16Five.remainderConstraint
        offset) = 9 := by
  rfl

private theorem booleanFreshTotal_eq (offset : Nat) :
    (List.map
      (R1CS.constraintFreshCount ∘
        NightstreamFPrime.Gadgets.Sampling.Candidate16Five.quotientBooleanConstraint
          offset)
      (List.range
        NightstreamFPrime.Gadgets.Sampling.Candidate16Five.quotientBitCount)
      ).sum = 28 := by
  simp [booleanFreshCount_eq,
    NightstreamFPrime.Gadgets.Sampling.Candidate16Five.quotientBitCount,
    Function.comp_def]

private theorem booleanRowTotal_eq (offset : Nat) :
    (List.map
      (R1CS.constraintRowCount ∘
        NightstreamFPrime.Gadgets.Sampling.Candidate16Five.quotientBooleanConstraint
          offset)
      (List.range
        NightstreamFPrime.Gadgets.Sampling.Candidate16Five.quotientBitCount)
      ).sum = 42 := by
  simp [booleanRowCount_eq,
    NightstreamFPrime.Gadgets.Sampling.Candidate16Five.quotientBitCount,
    Function.comp_def]

theorem totalFreshCount_eq (wordOffset : Nat) (part : Fin 2)
    (offset : Nat) :
    R1CS.totalFreshCount (logicalConstraints wordOffset part offset) = 53 := by
  unfold logicalConstraints
  rw [Logical.flatConstraints_operations,
    R1CS.totalFreshCount_append, R1CS.totalFreshCount_append,
    rejectFreshCount_eq]
  simp only [R1CS.totalFreshCount, List.map_map]
  rw [booleanFreshTotal_eq]
  simp only [List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, Nat.add_zero]
  rw [R1CS.constraintFreshCount_eq_zero_of_affine _
      (quotientRecomposition_affine offset),
    R1CS.constraintFreshCount_eq_zero_of_affine _
      (division_affine wordOffset part offset),
    remainderFreshCount_eq]

theorem totalRowCount_eq (wordOffset : Nat) (part : Fin 2)
    (offset : Nat) :
    R1CS.totalRowCount (logicalConstraints wordOffset part offset) = 71 := by
  unfold logicalConstraints
  rw [Logical.flatConstraints_operations,
    R1CS.totalRowCount_append, R1CS.totalRowCount_append,
    rejectRowCount_eq]
  simp only [R1CS.totalRowCount, List.map_map]
  rw [booleanRowTotal_eq]
  simp only [List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, Nat.add_zero]
  rw [R1CS.constraintRowCount_eq_one_of_affine _
      (quotientRecomposition_affine offset),
    R1CS.constraintRowCount_eq_one_of_affine _
      (division_affine wordOffset part offset),
    remainderRowCount_eq]

def footprint (wordOffset : Nat) (part : Fin 2) :
    R1CS.CircuitFootprint (Logical.circuit (interface wordOffset part)) where
  freshColumnCount := fun _ => 53
  physicalRowCount := fun _ => 71
  freshColumnCount_eq := by
    intro offset
    change R1CS.totalFreshCount
      (logicalConstraints wordOffset part offset) = 53
    exact totalFreshCount_eq wordOffset part offset
  physicalRowCount_eq := by
    intro offset
    change R1CS.totalRowCount
      (logicalConstraints wordOffset part offset) = 71
    exact totalRowCount_eq wordOffset part offset

theorem physicalPrivateColumnCount_eq (wordOffset : Nat) (part : Fin 2)
    (offset : Nat) :
    localLength (Circuit.ops
        (Logical.circuit (interface wordOffset part)).main offset) +
      R1CS.totalFreshCount (logicalConstraints wordOffset part offset) = 70 := by
  change localLength (Logical.operations (interface wordOffset part) offset) +
    R1CS.totalFreshCount (logicalConstraints wordOffset part offset) = 70
  rw [Logical.localLength_eq, totalFreshCount_eq]
  rfl

def physicalRows (wordOffset : Nat) (part : Fin 2)
    (offset : Nat) : List R1CS.Row :=
  (plan wordOffset part offset).rows

def PhysicalHolds (wordOffset : Nat) (part : Fin 2)
    (offset : Nat) (env : Env) : Prop :=
  R1CS.RowsHold env (physicalRows wordOffset part offset)

theorem physical_implies_spec (wordOffset : Nat) (part : Fin 2)
    (offset : Nat) (env : Env)
    (assumptions : Logical.Assumptions (interface wordOffset part) offset env)
    (physical : PhysicalHolds wordOffset part offset env) :
    Logical.SpecHolds (interface wordOffset part) offset env := by
  apply Logical.soundness (interface wordOffset part) env offset assumptions
  apply holdsFlat_implies_holds
  change ConstraintsHold env (logicalConstraints wordOffset part offset)
  exact R1CS.LoweringPlan.sound (plan wordOffset part offset) env physical

theorem physical_complete (wordOffset : Nat) (part : Fin 2)
    (offset : Nat) (env : Env)
    (assumptions : Logical.Assumptions (interface wordOffset part) offset env)
    (specification : Logical.SpecHolds (interface wordOffset part) offset env) :
    ∃ completed,
      AgreesOutside env completed offset 70 ∧
      PhysicalHolds wordOffset part offset completed := by
  rcases Logical.completeness (interface wordOffset part) env offset assumptions
      specification with ⟨logicalEnv, logicalAgrees, logicalRows⟩
  have logicalAgreesFixed :
      AgreesOutside env logicalEnv offset Logical.auxiliaryCount := by
    rw [Logical.localLength_eq] at logicalAgrees
    exact logicalAgrees
  have scope : ∀ expression ∈ logicalConstraints wordOffset part offset,
      expression.VarsBelow (offset + Logical.auxiliaryCount) := by
    exact Logical.flatConstraints_varsBelow (interface wordOffset part) offset
      assumptions.1 assumptions.2.1
  have logicalHolds :
      ConstraintsHold logicalEnv (logicalConstraints wordOffset part offset) := by
    exact logicalRows
  rcases R1CS.lowerConstraints_complete logicalEnv
      (logicalConstraints wordOffset part offset)
      (offset + Logical.auxiliaryCount) scope logicalHolds with
    ⟨completed, physicalAgrees, rows⟩
  refine ⟨completed, ?_, rows⟩
  have combined := logicalAgreesFixed.append physicalAgrees
  rw [totalFreshCount_eq wordOffset part offset] at combined
  simpa [Logical.auxiliaryCount] using combined

end NightstreamFPrime.Layout.Sampling.Candidate16Five
