import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafModel

/-!
Contract: structural operand-permutation invariance for the reusable PiRLC
Poseidon2 leaf.

Assurance tier: model-level.

Owns: invariance of source linear-combination actions, final port actions,
matrix points, and selective-CCS residuals under exact `List.Perm` witnesses.

Does not own: generated permutation witnesses, Rust row identity, call
placement, selector activation, lifecycle authority, or cryptographic
security.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallPermutation

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel.Wire

theorem sum_eq_of_perm {left right : List F}
    (permutation : left.Perm right) : sum left = sum right := by
  induction permutation with
  | nil => rfl
  | cons value permutation inductionHypothesis =>
      simp only [sum]
      rw [inductionHypothesis]
  | swap left right tail =>
      simp only [sum]
      ac_rfl
  | trans first second firstHypothesis secondHypothesis =>
      exact firstHypothesis.trans secondHypothesis

theorem sum_map_eq_of_perm {α : Type} {left right : List α}
    (permutation : left.Perm right) (value : α → F) :
    sum (left.map value) = sum (right.map value) :=
  sum_eq_of_perm (permutation.map value)

structure SourceLinearCombinationPermutes
    (left right : SourceLinearCombination) : Prop where
  constant : left.constant = right.constant
  terms : left.terms.Perm right.terms

theorem sourceAction_eq_of_perm
    {left right : SourceLinearCombination}
    (permutation : SourceLinearCombinationPermutes left right)
    (assignment : SourceAssignment) :
    sourceAction left assignment = sourceAction right assignment := by
  unfold sourceAction
  rw [permutation.constant]
  rw [sum_map_eq_of_perm permutation.terms]

structure PortPermutes (left right : Port) : Prop where
  explicit : left.explicit.Perm right.explicit
  geometric : left.geometric.Perm right.geometric

theorem portAction_eq_of_perm {left right : Port}
    (permutation : PortPermutes left right)
    (assignment : FinalAssignment) :
    portAction left assignment = portAction right assignment := by
  unfold portAction
  rw [sum_map_eq_of_perm permutation.explicit]
  rw [sum_map_eq_of_perm permutation.geometric]

def RowPermutes (left right : Wire.Row) : Prop :=
  ∀ index, PortPermutes (left.port index) (right.port index)

theorem point_eq_of_perm {left right : Wire.Row}
    (permutation : RowPermutes left right)
    (assignment : FinalAssignment) :
    point left assignment = point right assignment := by
  funext index
  exact portAction_eq_of_perm (permutation index) assignment

theorem residual_eq_of_perm {left right : Wire.Row}
    (permutation : RowPermutes left right)
    (assignment : FinalAssignment) :
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel.residual
        left assignment =
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel.residual
        right assignment := by
  unfold Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel.residual
  rw [point_eq_of_perm permutation]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallPermutation
