import NightstreamFPrime.Lifecycle.RandomOracleUniqueness
import NightstreamFPrime.Lifecycle.Nifs.VerifierErrorBudget

/-! Random-oracle model, SECURITY_MODEL.md Lemmas 5 and 6. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.RandomOracleKnowledge

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.TranscriptCoverage
open NightstreamFPrime.Lifecycle.RandomOracleTest
open NightstreamFPrime.Lifecycle.RandomOracleExtraction
open NightstreamFPrime.Lifecycle.RandomOracleUniqueness
open StrongReduction ConcreteCarrier

attribute [local instance low] Classical.propDecidable

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  {Output : Type}
  (adversary : OracleComp (Point logicalWidth publicFits (ProductionKey.degreeBound relation))
    Answer Output)
  (claim : Output → Claim relation)

local notation "Oracle" => Point logicalWidth publicFits (ProductionKey.degreeBound relation) → Answer
local notation "Retries" => Fin Nifs.PaperProfile.arity.total → Answer

def Extracts (oracle : Oracle) (retries : Retries) : Prop :=
  ∃ valid : Valid relation ajtai adversary claim oracle retries,
    SourceHolds extensionOps K.embed (PaperAlgebra.openingMaps ajtai) productionGlobalParams
      ((ProductionKey.key relation ajtai).statement (claimed relation adversary claim oracle).running
        (claimed relation adversary claim oracle).fresh)
      (extractedWitness relation ajtai adversary claim oracle retries valid.1 valid.2)

noncomputable def statisticalError (queries : Nat) : ℝ :=
  (Nifs.PaperProfile.arity).total *
      (((queries + (Nifs.PaperProfile.arity).total : Nat) : ℝ) * sampleError) +
    ((queries + challenges.length : Nat) : ℝ) * IndependentExecution.testError productionShape 8

theorem statisticalError_le (queries : Nat) :
    statisticalError queries ≤ ((queries : ℝ) + 74) * (50 / (87 * 2 ^ 115)) := by
  have sample : sampleError < 1 / 5 ^ 54 + 1 / 2 ^ 132 := by
    have distance := Spec.Folding.Nifs.NonInteractive.PiRlcSampler.distance_lt
    have count : (Spec.Folding.Nifs.NonInteractive.PiRlcSampler.scalarCount : ℝ) = 5 ^ 54 := by
      norm_num [Spec.Folding.Nifs.NonInteractive.PiRlcSampler.scalarCount,
        Phi81StrongSet.alphabetSize, Phi81StrongSet.coefficientCount]
    unfold sampleError
    rw [count]
    linarith
  have test := Nifs.VerifierErrorBudget.test_error_eq
  have perQuery : 17 * sampleError + IndependentExecution.testError productionShape 8 <
      50 / (87 * 2 ^ 115) := by
    have numbers : (17 : ℝ) * (1 / 5 ^ 54 + 1 / 2 ^ 132) +
        4589 / (goldilocksModulus : ℝ) ^ 2 < 50 / (87 * 2 ^ 115) := by
      norm_num [goldilocksModulus]
    rw [test]
    linarith
  have sampleNonnegative := sampleError_nonnegative
  have testNonnegative : 0 ≤ IndependentExecution.testError productionShape 8 := by
    rw [test]
    positivity
  have queriesNonnegative : (0 : ℝ) ≤ queries := Nat.cast_nonneg queries
  unfold statisticalError
  rw [Nifs.PaperProfile.arity_total, challenges_length]
  push_cast
  calc
    _ ≤ ((queries : ℝ) + 74) * (17 * sampleError + IndependentExecution.testError productionShape 8) := by
      nlinarith
    _ ≤ _ := mul_le_mul_of_nonneg_left perQuery.le (by positivity)

theorem statisticalError_two_pow_64 : statisticalError (2 ^ 64) < 1 / 2 ^ 51 := by
  refine (statisticalError_le (2 ^ 64)).trans_lt ?_
  norm_num

noncomputable def knowledgeError (queries : Nat) : ℝ :=
  statisticalError queries +
  (𝔼 oracle, (if Succeeds relation ajtai adversary claim oracle then
      ∑ index, mismatchChance relation ajtai adversary claim index oracle else 0) +
    collisionChance relation ajtai adversary claim + runningChance relation ajtai adversary claim)

theorem knowledge_error_le {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, (if Succeeds relation ajtai adversary claim oracle then (1 : ℝ) else 0) ≤
      𝔼 oracle, ∑ retries, weight relation ajtai adversary claim oracle retries *
          (if Extracts relation ajtai adversary claim oracle retries then 1 else 0) +
        knowledgeError relation ajtai adversary claim queries := by
  have pointwise (oracle : Oracle) :
      (if Succeeds relation ajtai adversary claim oracle then (1 : ℝ) else 0) ≤
        ∑ retries, weight relation ajtai adversary claim oracle retries *
            (if Extracts relation ajtai adversary claim oracle retries then 1 else 0) +
          (if Succeeds relation ajtai adversary claim oracle then
            ∑ retries, retryWeight (retrySets relation ajtai adversary claim oracle) retries *
              (if ∀ index, ForkValid relation ajtai adversary claim oracle retries index then 0 else 1)
          else 0) +
          ∑ retries, weight relation ajtai adversary claim oracle retries *
            (if SourceFails relation ajtai adversary claim oracle retries then 1 else 0) := by
    have extracts : 0 ≤ ∑ retries, weight relation ajtai adversary claim oracle retries *
        (if Extracts relation ajtai adversary claim oracle retries then 1 else 0) :=
      Finset.sum_nonneg fun retries _ => mul_nonneg (weight_nonnegative relation ajtai adversary claim
        oracle retries) (by split <;> norm_num)
    have fails : 0 ≤ ∑ retries, weight relation ajtai adversary claim oracle retries *
        (if SourceFails relation ajtai adversary claim oracle retries then 1 else 0) :=
      Finset.sum_nonneg fun retries _ => mul_nonneg (weight_nonnegative relation ajtai adversary claim
        oracle retries) (by split <;> norm_num)
    by_cases succeeds : Succeeds relation ajtai adversary claim oracle
    · rw [ite_eq_left succeeds, ite_eq_left succeeds, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
      have total : ∑ retries, weight relation ajtai adversary claim oracle retries = 1 :=
        retryWeight_sum_eq_one _ fun index =>
          mass_ne_zero (base_mem_retrySets relation ajtai adversary claim succeeds index)
      refine total.symm.le.trans (Finset.sum_le_sum fun retries _ => ?_)
      have nonnegative := weight_nonnegative relation ajtai adversary claim oracle retries
      change weight relation ajtai adversary claim oracle retries ≤
        weight relation ajtai adversary claim oracle retries *
            (if Extracts relation ajtai adversary claim oracle retries then 1 else 0) +
          weight relation ajtai adversary claim oracle retries *
            (if ∀ index, ForkValid relation ajtai adversary claim oracle retries index then 0 else 1) +
          weight relation ajtai adversary claim oracle retries *
            (if SourceFails relation ajtai adversary claim oracle retries then 1 else 0)
      rw [← mul_add, ← mul_add]
      refine le_mul_of_one_le_right nonnegative ?_
      have first : (0 : ℝ) ≤ if Extracts relation ajtai adversary claim oracle retries then 1 else 0 := by
        split <;> norm_num
      have third : (0 : ℝ) ≤ if SourceFails relation ajtai adversary claim oracle retries then 1 else 0 := by
        split <;> norm_num
      by_cases forks : ∀ index, ForkValid relation ajtai adversary claim oracle retries index
      · rw [ite_eq_left forks]
        by_cases holds : SourceHolds extensionOps K.embed (PaperAlgebra.openingMaps ajtai)
            productionGlobalParams
            ((ProductionKey.key relation ajtai).statement (claimed relation adversary claim oracle).running
              (claimed relation adversary claim oracle).fresh)
            (extractedWitness relation ajtai adversary claim oracle retries succeeds forks)
        · have extracted : Extracts relation ajtai adversary claim oracle retries :=
            ⟨⟨succeeds, forks⟩, holds⟩
          rw [ite_eq_left extracted]
          linarith
        · have failed : SourceFails relation ajtai adversary claim oracle retries :=
            ⟨⟨succeeds, forks⟩, holds⟩
          rw [ite_eq_left failed]
          linarith
      · rw [ite_eq_right forks]
        linarith
    · rw [ite_eq_right succeeds, ite_eq_right succeeds]
      linarith
  have lemma5 := fork_failure_le relation ajtai adversary claim bounded
  have lemma6 := source_error_le relation ajtai adversary claim bounded
  have summed := Finset.expect_le_expect (s := Finset.univ) fun oracle (_ : oracle ∈ Finset.univ) =>
    pointwise oracle
  rw [Finset.expect_add_distrib, Finset.expect_add_distrib] at summed
  unfold knowledgeError statisticalError
  linarith

noncomputable def extract (oracle : Oracle) (retries : Retries) :
    Option (OutputWitness productionShape (Phi81CarrierLayout.carrierWidth logicalWidth)) :=
  if Extracts relation ajtai adversary claim oracle retries then
    witnessOf relation ajtai adversary claim oracle retries
  else none

theorem extract_eq_none_iff (oracle : Oracle) (retries : Retries) :
    extract relation ajtai adversary claim oracle retries = none ↔
      ¬ Extracts relation ajtai adversary claim oracle retries := by
  unfold extract witnessOf
  by_cases extracts : Extracts relation ajtai adversary claim oracle retries
  · simp [extracts, extracts.fst]
  · simp [extracts]

noncomputable def runWeight (run : Oracle × Retries) : ℝ :=
  (if Succeeds relation ajtai adversary claim run.1 then
      weight relation ajtai adversary claim run.1 run.2
    else 1 / Fintype.card Retries) / Fintype.card Oracle

theorem weight_sum_eq_one {oracle : Oracle} (succeeds : Succeeds relation ajtai adversary claim oracle) :
    ∑ retries, weight relation ajtai adversary claim oracle retries = 1 :=
  retryWeight_sum_eq_one _ fun index =>
    mass_ne_zero (base_mem_retrySets relation ajtai adversary claim succeeds index)

theorem runWeight_nonnegative (run : Oracle × Retries) :
    0 ≤ runWeight relation ajtai adversary claim run := by
  unfold runWeight
  have : 0 ≤ if Succeeds relation ajtai adversary claim run.1 then
      weight relation ajtai adversary claim run.1 run.2 else 1 / (Fintype.card Retries : ℝ) := by
    split
    · exact weight_nonnegative relation ajtai adversary claim _ _
    · positivity
  positivity

theorem runWeight_sum : ∑ run, runWeight relation ajtai adversary claim run = 1 := by
  have oracles : (0 : ℝ) < Fintype.card Oracle := by exact_mod_cast Fintype.card_pos
  have each (oracle : Oracle) :
      ∑ retries, runWeight relation ajtai adversary claim (oracle, retries) =
        1 / Fintype.card Oracle := by
    unfold runWeight
    rw [← Finset.sum_div]
    refine congrArg (· / (Fintype.card Oracle : ℝ)) ?_
    by_cases succeeds : Succeeds relation ajtai adversary claim oracle
    · simp only [ite_eq_left succeeds]
      exact weight_sum_eq_one relation ajtai adversary claim succeeds
    · have retries : (0 : ℝ) < Fintype.card Retries := by exact_mod_cast Fintype.card_pos
      simp only [ite_eq_right succeeds, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
      field_simp
  rw [Fintype.sum_prod_type]
  simp only [each, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  field_simp

theorem extract_holds {oracle : Oracle} {retries : Retries} {witness}
    (returned : extract relation ajtai adversary claim oracle retries = some witness) :
    SourceHolds extensionOps K.embed (PaperAlgebra.openingMaps ajtai) productionGlobalParams
      ((ProductionKey.key relation ajtai).statement (claimed relation adversary claim oracle).running
        (claimed relation adversary claim oracle).fresh) witness := by
  unfold extract at returned
  by_cases extracts : Extracts relation ajtai adversary claim oracle retries
  · rw [ite_eq_left extracts] at returned
    obtain ⟨valid, holds⟩ := extracts
    unfold witnessOf at returned
    rw [dite_eq_left valid] at returned
    cases returned
    exact holds
  · rw [ite_eq_right extracts] at returned
    cases returned

theorem failure_eq :
    ∑ run, runWeight relation ajtai adversary claim run *
        (if Succeeds relation ajtai adversary claim run.1 ∧
            extract relation ajtai adversary claim run.1 run.2 = none then 1 else 0) =
      𝔼 oracle, (if Succeeds relation ajtai adversary claim oracle then (1 : ℝ) else 0) -
        𝔼 oracle, ∑ retries, weight relation ajtai adversary claim oracle retries *
          (if Extracts relation ajtai adversary claim oracle retries then 1 else 0) := by
  calc
    _ = ∑ oracle : Oracle, ∑ retries : Retries,
          (if Succeeds relation ajtai adversary claim oracle then
            weight relation ajtai adversary claim oracle retries -
              weight relation ajtai adversary claim oracle retries *
                (if Extracts relation ajtai adversary claim oracle retries then 1 else 0)
          else 0) / Fintype.card Oracle := by
      rw [Fintype.sum_prod_type]
      refine Finset.sum_congr rfl fun oracle _ => Finset.sum_congr rfl fun retries _ => ?_
      by_cases succeeds : Succeeds relation ajtai adversary claim oracle
      · by_cases extracts : Extracts relation ajtai adversary claim oracle retries
        · have returned : extract relation ajtai adversary claim oracle retries ≠ none :=
            fun none => (extract_eq_none_iff relation ajtai adversary claim oracle retries).mp none
              extracts
          simp [runWeight, succeeds, extracts, returned]
        · have none : extract relation ajtai adversary claim oracle retries = none :=
            (extract_eq_none_iff relation ajtai adversary claim oracle retries).mpr extracts
          simp [runWeight, succeeds, extracts, none]
      · simp only [succeeds, false_and, ite_false, mul_zero, zero_div]
    _ = ((∑ oracle : Oracle, if Succeeds relation ajtai adversary claim oracle then (1 : ℝ) else 0) -
          ∑ oracle : Oracle, ∑ retries, weight relation ajtai adversary claim oracle retries *
            (if Extracts relation ajtai adversary claim oracle retries then 1 else 0)) /
        Fintype.card Oracle := by
      rw [← Finset.sum_sub_distrib, Finset.sum_div]
      refine Finset.sum_congr rfl fun oracle _ => ?_
      rw [← Finset.sum_div]
      refine congrArg (· / (Fintype.card Oracle : ℝ)) ?_
      by_cases succeeds : Succeeds relation ajtai adversary claim oracle
      · simp only [ite_eq_left succeeds]
        rw [Finset.sum_sub_distrib, weight_sum_eq_one relation ajtai adversary claim succeeds]
      · have never (retries : Retries) : ¬ Extracts relation ajtai adversary claim oracle retries :=
          fun extracts => succeeds extracts.fst.1
        simp [succeeds]
        exact Finset.sum_eq_zero fun retries _ => ite_eq_right (never retries)
    _ = _ := by
      rw [Finset.expect_eq_sum_div_card, Finset.expect_eq_sum_div_card, Finset.card_univ, sub_div]

end NightstreamFPrime.Lifecycle.RandomOracleKnowledge
