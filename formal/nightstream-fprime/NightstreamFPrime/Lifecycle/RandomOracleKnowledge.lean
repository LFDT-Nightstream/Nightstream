import NightstreamFPrime.Lifecycle.RandomOracleUniqueness
import NightstreamFPrime.Spec.KnowledgeContract

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
  chances of Lemma 6;
- `contract`: the same result as a `Spec.KnowledgeContract`, which answers
  the six auditor questions.

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

/-- The knowledge error: the statistical error
`17 (Q + 17) sampleError + (Q + 74) testError`, the chance that a retry
changes the running statement (Lemma 5), and the two binding-reduction
chances (Lemma 6). -/
noncomputable def knowledgeError (queries : Nat) : ℝ :=
  ((Nifs.PaperProfile.arity).total *
      (((queries + (Nifs.PaperProfile.arity).total : Nat) : ℝ) * sampleError) +
    ((queries + challenges.length : Nat) : ℝ) * IndependentExecution.testError productionShape 9) +
  (𝔼 oracle, (if Succeeds relation ajtai adversary claim oracle then
      ∑ index, mismatchChance relation ajtai adversary claim index oracle else 0) +
    collisionChance relation ajtai adversary claim + runningChance relation ajtai adversary claim)

/-- **ROM knowledge soundness of the production NIFS.** The extractor
succeeds with probability at least the acceptance probability minus the
knowledge error. -/
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
  unfold knowledgeError
  linarith

/-! ## The knowledge contract -/

/-- The extractor's output: the fork's witness when it satisfies the source
relation of the claimed statement, else nothing. The last step is the source
relation's own check. -/
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

/-- The law of a run: a uniform oracle, then the extractor's retries after an
acceptance. Retries after a rejection are uniform; no event reads them. -/
noncomputable def runWeight (run : Oracle × Retries) : ℝ :=
  (if Succeeds relation ajtai adversary claim run.1 then
      weight relation ajtai adversary claim run.1 run.2
    else 1 / Fintype.card Retries) / Fintype.card Oracle

/-- The retries of an accepted run have total weight one. -/
theorem weight_sum_eq_one {oracle : Oracle} (succeeds : Succeeds relation ajtai adversary claim oracle) :
    ∑ retries, weight relation ajtai adversary claim oracle retries = 1 :=
  retryWeight_sum_eq_one _ fun index =>
    mass_ne_zero (base_mem_retrySets relation ajtai adversary claim succeeds index)

/-- The knowledge contract of the production NIFS in the random-oracle model,
for an adversary with at most `queries` oracle queries. -/
noncomputable def contract {queries : Nat} (bounded : adversary.QueryBound queries) :
    KnowledgeContract where
  Run := Oracle × Retries
  weight := runWeight relation ajtai adversary claim
  weight_nonnegative run := by
    unfold runWeight
    have : 0 ≤ if Succeeds relation ajtai adversary claim run.1 then
        weight relation ajtai adversary claim run.1 run.2 else 1 / (Fintype.card Retries : ℝ) := by
      split
      · exact weight_nonnegative relation ajtai adversary claim _ _
      · positivity
    positivity
  weight_sum := by
    have oracles : (0 : ℝ) < Fintype.card Oracle := by exact_mod_cast Fintype.card_pos
    have each (oracle : Oracle) :
        ∑ retries, runWeight relation ajtai adversary claim (oracle, retries) =
          1 / Fintype.card Oracle := by
      unfold runWeight
      rw [← Finset.sum_div]
      refine congrArg (· / (Fintype.card Oracle : ℝ)) ?_
      by_cases succeeds : Succeeds relation ajtai adversary claim oracle
      · simp only [if_pos succeeds]
        exact weight_sum_eq_one relation ajtai adversary claim succeeds
      · have retries : (0 : ℝ) < Fintype.card Retries := by exact_mod_cast Fintype.card_pos
        simp only [if_neg succeeds, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
        field_simp
    rw [Fintype.sum_prod_type]
    simp only [each, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    field_simp
  Statement := Running (logicalWidth := logicalWidth) (publicFits := publicFits) ×
    Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)
  statement run := ((claimed relation adversary claim run.1).running,
    (claimed relation adversary claim run.1).fresh)
  accepts run := Succeeds relation ajtai adversary claim run.1
  Witness := OutputWitness productionShape (Phi81CarrierLayout.carrierWidth logicalWidth)
  extract run := extract relation ajtai adversary claim run.1 run.2
  Holds statement witness := SourceHolds extensionOps K.embed (PaperAlgebra.openingMaps ajtai)
    productionGlobalParams ((ProductionKey.key relation ajtai).statement statement.1 statement.2) witness
  witness_statement := by
    rintro ⟨oracle, retries⟩ witness returned
    change extract relation ajtai adversary claim oracle retries = some witness at returned
    unfold extract at returned
    by_cases extracts : Extracts relation ajtai adversary claim oracle retries
    · rw [if_pos extracts] at returned
      obtain ⟨valid, holds⟩ := extracts
      unfold witnessOf at returned
      rw [dif_pos valid] at returned
      cases returned
      exact holds
    · rw [if_neg extracts] at returned
      cases returned
  error := knowledgeError relation ajtai adversary claim queries
  knowledge_sound := by
    have main := knowledge_error_le relation ajtai adversary claim bounded
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
        · simp only [succeeds, false_and, if_false, mul_zero, zero_div]
      _ = ((∑ oracle : Oracle, if Succeeds relation ajtai adversary claim oracle then (1 : ℝ) else 0) -
            ∑ oracle : Oracle, ∑ retries, weight relation ajtai adversary claim oracle retries *
              (if Extracts relation ajtai adversary claim oracle retries then 1 else 0)) /
          Fintype.card Oracle := by
        rw [← Finset.sum_sub_distrib, Finset.sum_div]
        refine Finset.sum_congr rfl fun oracle _ => ?_
        rw [← Finset.sum_div]
        refine congrArg (· / (Fintype.card Oracle : ℝ)) ?_
        by_cases succeeds : Succeeds relation ajtai adversary claim oracle
        · simp only [if_pos succeeds]
          rw [Finset.sum_sub_distrib, weight_sum_eq_one relation ajtai adversary claim succeeds]
        · have never (retries : Retries) : ¬ Extracts relation ajtai adversary claim oracle retries :=
            fun extracts => succeeds extracts.fst.1
          simp [succeeds, never]
      _ ≤ knowledgeError relation ajtai adversary claim queries := by
        rw [Finset.expect_eq_sum_div_card, Finset.expect_eq_sum_div_card, Finset.card_univ] at main
        rw [sub_div]
        linarith

end NightstreamFPrime.Lifecycle.RandomOracleKnowledge
