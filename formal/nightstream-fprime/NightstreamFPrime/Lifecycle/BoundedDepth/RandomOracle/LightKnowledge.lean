import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.LightUniqueness

/-! The light fold's knowledge bound in the random-oracle model: `Uniqueness.knowledge_error_le`
on the light game, the shape of `RandomOracleLink.knowledge_error_le_linked`.

A valid pair extracts (`Extracts`) when its answer gives the walk's `Extracted` on the base
attempt: the input claim at the base bound plus `T · (b − 1)` and the fresh instance. A valid
pair extracts or fails (`splits`, `LightUniqueness.extracted_of_truth`). `knowledge_error_le`: a
successful run extracts, weighted by the retry law, except with `knowledgeError`, the sum of
the statistical error (`statisticalError`: `(Q + 1) · sampleError` for the fork step and
`(Q + 58) · 574/q²` for the test step), the prior-state collision chances of the fork step
(`LightFork.collisionChance`) and the uniqueness step (`stateChance`), and the kernel chance
(`LightUniqueness.kernelChance`). `statisticalError_le`: the statistical error is at most
`(Q + 58) · 575/2^128`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.LightKnowledge

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthFolds (freshToLight)
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.RandomOracle.Test (Answer Point)
open NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.ForkLink (Attempt)
open NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.LightUniqueness (Retries)

attribute [local instance low] Classical.propDecidable

/-- The statistical error of the light fold: `(Q + 1) · sampleError` for the fork step and
`(Q + 58) · 574/q²` for the test step. -/
noncomputable def statisticalError (queries : Nat) : ℝ :=
  ((queries + 1 : Nat) : ℝ) * Fork.sampleError +
    ((queries + 58 : Nat) : ℝ) * ((574 : ℝ) / (fieldModulus : ℝ) ^ 2)

theorem statisticalError_eq {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}
    (queries : Nat) :
    Uniqueness.statisticalError (LightSchedule.schedule logicalWidth publicFits) 8 queries =
      statisticalError queries := by
  have test : IndependentExecution.testError
      (LightSchedule.schedule logicalWidth publicFits).shape 8 =
        (574 : ℝ) / (fieldModulus : ℝ) ^ 2 :=
    LightTest.testError_eq
  unfold Uniqueness.statisticalError statisticalError
  rw [test, LightTest.challenges_length]
  show ((1 : Nat) : ℝ) * (((queries + 1 : Nat) : ℝ) * Fork.sampleError) + _ = _
  rw [Nat.cast_one, one_mul]

/-- The statistical error is at most `(Q + 58) · 575/2^128`. -/
theorem statisticalError_le (queries : Nat) :
    statisticalError queries ≤ ((queries : ℝ) + 58) * (575 / 2 ^ 128) := by
  have sample : Fork.sampleError < 1 / 5 ^ 64 + 1 / 2 ^ 173 := by
    have distance := Spec.Folding.Nifs.NonInteractive.PiRlcSampler.distance_lt
    have count : (Spec.Folding.Nifs.NonInteractive.PiRlcSampler.scalarCount : ℝ) = 5 ^ 64 := by
      norm_num [Spec.Folding.Nifs.NonInteractive.PiRlcSampler.scalarCount,
        RingStrongSet.alphabetSize, RingStrongSet.coefficientCount, ringDegree]
    unfold Fork.sampleError
    rw [count]
    linarith
  have perQuery : Fork.sampleError + (574 : ℝ) / (fieldModulus : ℝ) ^ 2 <
      575 / 2 ^ 128 := by
    have numbers : (1 / 5 ^ 64 + 1 / 2 ^ 173 : ℝ) + 574 / (fieldModulus : ℝ) ^ 2 <
        575 / 2 ^ 128 := by
      norm_num [fieldModulus]
    linarith
  have sampleNonnegative := Fork.sampleError_nonnegative
  have testNonnegative : (0 : ℝ) ≤ 574 / (fieldModulus : ℝ) ^ 2 := by positivity
  have queriesNonnegative : (0 : ℝ) ≤ queries := Nat.cast_nonneg queries
  unfold statisticalError
  push_cast
  calc
    _ ≤ ((queries : ℝ) + 58) *
        (Fork.sampleError + (574 : ℝ) / (fieldModulus : ℝ) ^ 2) := by
      nlinarith
    _ ≤ _ := mul_le_mul_of_nonneg_left perQuery.le (by positivity)

theorem statisticalError_two_pow_64 : statisticalError (2 ^ 64) < 1 / 2 ^ 54 := by
  refine (statisticalError_le (2 ^ 64)).trans_lt ?_
  norm_num

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (key : CircuitKey logicalWidth publicFits)
  (adversary : OracleComp (Point (LightSchedule.schedule logicalWidth publicFits) 8) Answer
    (Attempt logicalWidth publicFits))

/-- A valid pair whose answer gives the walk's `Extracted` on the base attempt. -/
def Extracts (oracle : LightFork.Oracle logicalWidth publicFits)
    (retries : Retries logicalWidth publicFits) : Prop :=
  ∃ valid : Uniqueness.Valid (LightFork.game ajtai key) adversary oracle retries,
    UniquenessLink.Extracted ajtai (adversary.run oracle)
      (LightUniqueness.answerOf (LightFork.extractedWitness ajtai valid.1 valid.2))

/-- A valid pair extracts or fails. -/
theorem splits {oracle : LightFork.Oracle logicalWidth publicFits}
    {retries : Retries logicalWidth publicFits}
    (valid : Uniqueness.Valid (LightFork.game ajtai key) adversary oracle retries) :
    Extracts ajtai key adversary oracle retries ∨
      LightUniqueness.Fails ajtai key adversary oracle retries := by
  by_cases truth : LightUniqueness.Truth ajtai key.relation
      (LightFork.runningOf (adversary.run oracle).link.start.1)
      (freshToLight (adversary.run oracle).link.fold.fresh)
      (LightFork.extractedWitness ajtai valid.1 valid.2)
  · exact Or.inl ⟨valid, LightUniqueness.extracted_of_truth ajtai valid.1 valid.2 truth⟩
  · exact Or.inr ⟨valid, truth⟩

/-- The chance that a resampled valid run gives a collision of the base and resampled prior
states (`UniquenessLink.stateChance`). -/
noncomputable def stateChance : ℝ :=
  UniquenessLink.stateChance ajtai key (LightFork.folds ajtai)
    (fun attempt => freshToLight attempt.link.fold.fresh)
    (fun attempt => LightSchedule.messages (LightFork.proofOf attempt.link.fold.proof)) adversary

/-- The light knowledge error: the statistical error, the prior-state collision chances of the
fork and uniqueness steps, and the kernel chance. -/
noncomputable def knowledgeError (queries : Nat) : ℝ :=
  statisticalError queries +
    (𝔼 oracle, (if Fork.Succeeds (LightFork.game ajtai key) adversary oracle then
        ∑ index, LightFork.collisionChance ajtai key adversary index oracle else 0) +
      LightUniqueness.kernelChance ajtai key adversary + stateChance ajtai key adversary)

/-- The light fold's knowledge bound in the random-oracle model. -/
theorem knowledge_error_le {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, (if Fork.Succeeds (LightFork.game ajtai key) adversary oracle then (1 : ℝ)
        else 0) ≤
      𝔼 oracle, ∑ retries,
          Uniqueness.weight (LightFork.game ajtai key) adversary oracle retries *
            (if Extracts ajtai key adversary oracle retries then 1 else 0) +
        knowledgeError ajtai key adversary queries := by
  have bound := Uniqueness.knowledge_error_le (LightFork.game ajtai key) adversary
    (LightUniqueness.witnessOf ajtai key adversary) (LightUniqueness.source ajtai key adversary)
    (Extracts ajtai key adversary) (LightUniqueness.Fails ajtai key adversary)
    (splits ajtai key adversary) (fun fails => fails.1) (LightUniqueness.agree ajtai)
    (LightUniqueness.source_degreeCovers ajtai key adversary)
    (LightFork.collisionChance ajtai key adversary)
    (fun _ succeeds index => ForkLink.mismatchChance_le ajtai key (LightFork.folds ajtai)
      (statement := fun attempt => freshToLight attempt.link.fold.fresh)
      (fun attempt => LightSchedule.messages (LightFork.proofOf attempt.link.fold.proof))
      adversary (fun same => LightFork.freshToLight_injective same) succeeds index)
    bounded
  have state : Uniqueness.startChance (LightFork.game ajtai key) adversary ≤
      stateChance ajtai key adversary :=
    UniquenessLink.startChance_le_stateChance ajtai key (LightFork.folds ajtai)
      (statement := fun attempt => freshToLight attempt.link.fold.fresh)
      (fun attempt => LightSchedule.messages (LightFork.proofOf attempt.link.fold.proof))
      adversary (fun same => LightFork.freshToLight_injective same)
  have kernel := LightUniqueness.collisionChance_le_kernelChance ajtai key adversary
  rw [statisticalError_eq] at bound
  unfold knowledgeError
  linarith

end NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.LightKnowledge
