import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.CenteredTernary
import Nightstream.SuperNeo.Concrete.Algebra

/-!
Intrinsic Goldilocks encoding and Phi81 packing arithmetic. Security-selected
widths belong to `Ajtai.EstimatorModel`.
-/

namespace Ajtai.Parameters

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.CenteredTernaryField
open Nightstream.SuperNeo.Concrete

/-- Number of distinct balanced-ternary words of a fixed width. -/
def balancedTernaryWordCount (width : Nat) : Nat :=
  3 ^ width

/-- A width has enough balanced-ternary words to represent every Goldilocks
residue. -/
def CoversGoldilocks (width : Nat) : Prop :=
  goldilocksP ≤ balancedTernaryWordCount width

/-- The R1CS and SuperNeo models use the same Goldilocks modulus. -/
theorem goldilocksModulus_agrees :
    goldilocksP = goldilocksModulus := by
  rfl

/-- Kernel-reduced cardinality window for the Goldilocks modulus. -/
theorem goldilocks_balancedTernary_window :
    3 ^ 40 < goldilocksP ∧ goldilocksP < 3 ^ 41 := by
  decide

/-- Every canonical Goldilocks residue has the concrete 41-trit opening. -/
theorem everyCanonicalGoldilocks_has_41_trit_opening
    {source : Nat} (canonical : source < goldilocksP) :
    Represents source (encodeDigit source) :=
  encodeDigit_represents canonical

/-- Forty balanced trits do not provide enough distinct words. -/
theorem forty_trits_insufficient :
    ¬ CoversGoldilocks 40 := by
  unfold CoversGoldilocks balancedTernaryWordCount
  exact Nat.not_le_of_gt goldilocks_balancedTernary_window.1

/-- Forty-one balanced trits provide enough distinct words. -/
theorem forty_one_trits_sufficient :
    CoversGoldilocks digitCount := by
  unfold CoversGoldilocks balancedTernaryWordCount
  simpa only [digitCount] using
    Nat.le_of_lt goldilocks_balancedTernary_window.2

/-- Every balanced-ternary width below 41 has insufficient cardinality. -/
theorem fewer_than_41_trits_insufficient
    {width : Nat} (widthLt : width < digitCount) :
    ¬ CoversGoldilocks width := by
  intro covers
  have widthLe : width ≤ 40 := by
    simp only [digitCount] at widthLt
    omega
  have powerLe : 3 ^ width ≤ 3 ^ 40 :=
    Nat.pow_le_pow_right (by decide) widthLe
  have modulusLe : goldilocksP ≤ 3 ^ 40 := by
    exact Nat.le_trans covers powerLe
  exact Nat.not_le_of_lt goldilocks_balancedTernary_window.1 modulusLe

/-- `41` is the least balanced-ternary width that covers Goldilocks. -/
theorem digitCount_is_least_sufficient :
    CoversGoldilocks digitCount ∧
      ∀ width, CoversGoldilocks width → digitCount ≤ width := by
  constructor
  · exact forty_one_trits_sufficient
  · intro width covers
    apply Nat.le_of_not_gt
    intro widthLt
    exact fewer_than_41_trits_insufficient widthLt covers

/-- Rank selected for the long protocol-binding Ajtai map. -/
def protocolBindingRank : Nat := 2

/-- Generic field capacity of a packed ring message. -/
def maxPackedFields
    (digitsPerField coefficientsPerRingColumn ringColumns : Nat) : Nat :=
  ringColumns * coefficientsPerRingColumn / digitsPerField

/-- Ring columns needed to pack the balanced-ternary openings of `sourceFields`
Goldilocks values into the configured Phi81 ring. -/
def requiredRingColumns (sourceFields : Nat) : Nat :=
  (sourceFields * digitCount + ringDegree - 1) / ringDegree

theorem phi81_ringDegree_eq :
    ringDegree = 54 := by
  rfl

end Ajtai.Parameters
