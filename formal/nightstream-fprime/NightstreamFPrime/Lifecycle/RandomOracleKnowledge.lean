import NightstreamFPrime.Lifecycle.RandomOracleUniqueness

/-!
Owns the knowledge error of the production NIFS key in the random-oracle
model: Lemmas 5 and 6 of ROM_KNOWLEDGE_SOUNDNESS.md together.

Inputs: an adaptive oracle adversary with at most `Q` queries that outputs a
claim (running and fresh statements, a NIFS proof, and child witnesses).

Outputs:
- `Extracts`: the extractor's retries form a complete `Π_RLC` fork, and the
  fork's witness satisfies the source relation of the claimed statement;
- `knowledge_error_le`: the extractor succeeds with probability at least the
  acceptance probability minus `17 (Q + 17) sampleError + (Q + 74) testError`,
  the running-mismatch chance of Lemma 5, and the two binding-reduction
  chances of Lemma 6.

The extractor's retry law has total mass one after every acceptance
(`RandomOracle.retryWeight_sum_eq_one`), so a missing retry counts as a
failure. It runs `17 (Q + 17)` expected retries
(`RandomOracleExtraction.expected_retries_le`).

Does not own: the hardness of MSIS or of the state hash, which bound the
binding-reduction and running-mismatch chances, or the fit of the oracle to
Poseidon2.
-/

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

/-- The extractor succeeds: the retries form a complete `Π_RLC` fork, and the
fork's witness satisfies the source relation of the claimed statement. -/
def Extracts (oracle : Oracle) (retries : Retries) : Prop :=
  ∃ valid : Valid relation ajtai adversary claim oracle retries,
    SourceHolds extensionOps K.embed (PaperAlgebra.openingMaps ajtai) productionGlobalParams
      ((ProductionKey.key relation ajtai).statement (claimed relation adversary claim oracle).running
        (claimed relation adversary claim oracle).fresh)
      (extractedWitness relation ajtai adversary claim oracle retries valid.1 valid.2)

/-- **ROM knowledge soundness of the production NIFS.** The extractor
succeeds with probability at least the acceptance probability minus the
statistical error `17 (Q + 17) sampleError + (Q + 74) testError` and the
chances of a running mismatch (Lemma 5) and of the binding reduction
(Lemma 6). -/
theorem knowledge_error_le {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, (if Succeeds relation ajtai adversary claim oracle then (1 : ℝ) else 0) ≤
      𝔼 oracle, ∑ retries, weight relation ajtai adversary claim oracle retries *
          (if Extracts relation ajtai adversary claim oracle retries then 1 else 0) +
        ((Nifs.PaperProfile.arity).total *
            (((queries + (Nifs.PaperProfile.arity).total : Nat) : ℝ) * sampleError) +
          ((queries + challenges.length : Nat) : ℝ) * IndependentExecution.testError productionShape 9) +
        (𝔼 oracle, (if Succeeds relation ajtai adversary claim oracle then
            ∑ index, mismatchChance relation ajtai adversary claim index oracle else 0) +
          collisionChance relation ajtai adversary claim + runningChance relation ajtai adversary claim) := by
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
    · rw [if_pos succeeds, if_pos succeeds, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
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
      · rw [if_pos forks]
        by_cases holds : SourceHolds extensionOps K.embed (PaperAlgebra.openingMaps ajtai)
            productionGlobalParams
            ((ProductionKey.key relation ajtai).statement (claimed relation adversary claim oracle).running
              (claimed relation adversary claim oracle).fresh)
            (extractedWitness relation ajtai adversary claim oracle retries succeeds forks)
        · have extracted : Extracts relation ajtai adversary claim oracle retries :=
            ⟨⟨succeeds, forks⟩, holds⟩
          rw [if_pos extracted]
          linarith
        · have failed : SourceFails relation ajtai adversary claim oracle retries :=
            ⟨⟨succeeds, forks⟩, holds⟩
          rw [if_pos failed]
          linarith
      · rw [if_neg forks]
        linarith
    · rw [if_neg succeeds, if_neg succeeds]
      linarith
  have lemma5 := fork_failure_le relation ajtai adversary claim bounded
  have lemma6 := source_error_le relation ajtai adversary claim bounded
  have summed := Finset.expect_le_expect (s := Finset.univ) fun oracle (_ : oracle ∈ Finset.univ) =>
    pointwise oracle
  rw [Finset.expect_add_distrib, Finset.expect_add_distrib] at summed
  linarith

end NightstreamFPrime.Lifecycle.RandomOracleKnowledge
