import NightstreamFPrime.Layout.BalancedTernary
import tests.AxiomAudit

/-!
Review artifact: the existing low-norm encoding is not the only digit vector
that reconstructs a given field value. This does not assert that the complete
FPrime matrices accept a bad transition or this alternative digit vector.
-/

namespace FPrimeProve2EncodingReview

open NightstreamFPrime.Spec
open NightstreamFPrime.Layout
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix
open NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm

def value : F := fieldOfNat 9223372034707292160

def alternative : List F :=
  (BalancedTernary.digitsNat BalancedTernary.width 9223372034707292161).map
    (fun digit => -digit)

theorem alternative_length : alternative.length = BalancedTernary.width := by
  simp [alternative]

theorem alternative_norm : ∀ digit ∈ alternative, centeredMagnitude digit < 2 := by
  have unsigned : ∀ count value digit, digit ∈ BalancedTernary.digitsNat count value →
      centeredMagnitude digit < 2 := by
    intro count
    induction count with
    | zero => simp [BalancedTernary.digitsNat]
    | succ count inductionHypothesis =>
      intro value digit member
      simp only [BalancedTernary.digitsNat, List.mem_cons] at member
      rcases member with rfl | member
      · exact BalancedTernary.digit_norm value
      · exact inductionHypothesis _ _ member
  intro digit member
  rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
  simpa only [Centered.centeredMagnitude_neg] using unsigned _ _ _ sourceMember

theorem alternative_recomposes : BalancedTernary.recompose alternative = value := by
  -- The local structural argument matches the private map-negation lemma in
  -- Layout/BalancedTernary.lean at reviewed commit 4fc02857.
  have mapNeg (values : List F) :
      BalancedTernary.recompose (values.map fun digit => -digit) =
        -BalancedTernary.recompose values := by
    induction values with
    | nil => exact Lean.Grind.AddCommGroup.neg_zero.symm
    | cons digit values inductionHypothesis =>
      simp only [List.map_cons, BalancedTernary.recompose, inductionHypothesis]
      have mulNeg : fieldOfNat 3 * -BalancedTernary.recompose values =
          -(fieldOfNat 3 * BalancedTernary.recompose values) := by
        calc
          fieldOfNat 3 * -BalancedTernary.recompose values =
              -BalancedTernary.recompose values * fieldOfNat 3 := Fin.mul_comm _ _
          _ = -(BalancedTernary.recompose values * fieldOfNat 3) :=
            Lean.Grind.Fin.neg_mul _ _
          _ = -(fieldOfNat 3 * BalancedTernary.recompose values) := by
            rw [Fin.mul_comm (BalancedTernary.recompose values) (fieldOfNat 3)]
      rw [mulNeg]
      exact (Lean.Grind.AddCommGroup.neg_add _ _).symm
  unfold alternative
  rw [mapNeg, BalancedTernary.recompose_digitsNat _ _ (by decide)]
  decide

theorem alternative_not_canonical : alternative ≠ BalancedTernary.digits value := by
  decide

#audit_axioms alternative_length
#audit_axioms alternative_norm
#audit_axioms alternative_recomposes
#audit_axioms alternative_not_canonical

end FPrimeProve2EncodingReview
