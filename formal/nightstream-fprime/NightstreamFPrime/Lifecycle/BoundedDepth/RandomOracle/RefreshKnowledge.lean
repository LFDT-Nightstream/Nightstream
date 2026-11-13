import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RefreshUniqueness

/-! The refresh fold's knowledge bound in the random-oracle model: `Uniqueness.knowledge_error_le`
on the refresh game, the shape of `RandomOracleLink.knowledge_error_le_linked`.

A valid pair extracts (`Extracts`) when its answer gives the walk's `Extracted` on the base
attempt: the input claim at the refresh bound `B` and the fresh instance. A valid pair extracts
or fails (`splits`, `RefreshUniqueness.extracted_of_truth`). `knowledge_error_le`: a successful
run extracts, weighted by the retry law, except with `knowledgeError`, the sum of the
statistical error (`statisticalError`: `17 · (Q + 17) · sampleError` for the fork step and
`(Q + 74) · 5389/q²` for the test step), the prior-state collision chances of the fork step
(`RefreshFork.collisionChance`) and the uniqueness step (`stateChance`), and the kernel chance
(`RefreshUniqueness.kernelChance`). `statisticalError_le`: the statistical error is at most
`(Q + 74) · 5390/2^128`, the bound of the one-circuit fold. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RefreshKnowledge

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.RandomOracle.Test (Answer Point)
open NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.ForkLink (Attempt)
open NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RefreshUniqueness (Retries)

attribute [local instance low] Classical.propDecidable

/-- The statistical error of the refresh fold: `17 · (Q + 17) · sampleError` for the fork step
and `(Q + 74) · 5389/q²` for the test step. -/
noncomputable def statisticalError (queries : Nat) : ℝ :=
  17 * (((queries + 17 : Nat) : ℝ) * Fork.sampleError) +
    ((queries + 74 : Nat) : ℝ) * ((5389 : ℝ) / (fieldModulus : ℝ) ^ 2)

theorem statisticalError_eq {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}
    (queries : Nat) :
    Uniqueness.statisticalError (RefreshSchedule.schedule logicalWidth publicFits) 8 queries =
      statisticalError queries := by
  have test : IndependentExecution.testError
      (RefreshSchedule.schedule logicalWidth publicFits).shape 8 =
        (5389 : ℝ) / (fieldModulus : ℝ) ^ 2 :=
    Nifs.VerifierErrorBudget.test_error_eq
  unfold Uniqueness.statisticalError statisticalError
  rw [test, RefreshTest.challenges_length]
  show ((17 : Nat) : ℝ) * (((queries + 17 : Nat) : ℝ) * Fork.sampleError) + _ = _
  norm_num

/-- The statistical error is at most `(Q + 74) · 5390/2^128`. -/
theorem statisticalError_le (queries : Nat) :
    statisticalError queries ≤ ((queries : ℝ) + 74) * (5390 / 2 ^ 128) := by
  have sample : Fork.sampleError < 1 / 5 ^ 64 + 1 / 2 ^ 173 := by
    have distance := Spec.Folding.Nifs.NonInteractive.PiRlcSampler.distance_lt
    have count : (Spec.Folding.Nifs.NonInteractive.PiRlcSampler.scalarCount : ℝ) = 5 ^ 64 := by
      norm_num [Spec.Folding.Nifs.NonInteractive.PiRlcSampler.scalarCount,
        RingStrongSet.alphabetSize, RingStrongSet.coefficientCount]
    unfold Fork.sampleError
    rw [count]
    linarith
  have perQuery : 17 * Fork.sampleError + (5389 : ℝ) / (fieldModulus : ℝ) ^ 2 <
      5390 / 2 ^ 128 := by
    have numbers : (17 : ℝ) * (1 / 5 ^ 64 + 1 / 2 ^ 173) +
        5389 / (fieldModulus : ℝ) ^ 2 < 5390 / 2 ^ 128 := by
      norm_num [fieldModulus]
    linarith
  have sampleNonnegative := Fork.sampleError_nonnegative
  have testNonnegative : (0 : ℝ) ≤ 5389 / (fieldModulus : ℝ) ^ 2 := by positivity
  have queriesNonnegative : (0 : ℝ) ≤ queries := Nat.cast_nonneg queries
  unfold statisticalError
  push_cast
  calc
    _ ≤ ((queries : ℝ) + 74) *
        (17 * Fork.sampleError + (5389 : ℝ) / (fieldModulus : ℝ) ^ 2) := by
      nlinarith
    _ ≤ _ := mul_le_mul_of_nonneg_left perQuery.le (by positivity)

theorem statisticalError_two_pow_64 : statisticalError (2 ^ 64) < 1 / 2 ^ 51 := by
  refine (statisticalError_le (2 ^ 64)).trans_lt ?_
  norm_num

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (key : CircuitKey logicalWidth publicFits)
  (adversary : OracleComp (Point (RefreshSchedule.schedule logicalWidth publicFits) 8) Answer
    (Attempt logicalWidth publicFits))

/-- A valid pair whose answer gives the walk's `Extracted` on the base attempt. -/
def Extracts (oracle : RefreshFork.Oracle logicalWidth publicFits)
    (retries : Retries logicalWidth publicFits) : Prop :=
  ∃ valid : Uniqueness.Valid (RefreshFork.game ajtai key) adversary oracle retries,
    UniquenessLink.Extracted ajtai (adversary.run oracle)
      (RefreshUniqueness.answerOf ajtai key.relation
        (RefreshFork.extractedWitness ajtai valid.1 valid.2))

/-- A valid pair extracts or fails. -/
theorem splits {oracle : RefreshFork.Oracle logicalWidth publicFits}
    {retries : Retries logicalWidth publicFits}
    (valid : Uniqueness.Valid (RefreshFork.game ajtai key) adversary oracle retries) :
    Extracts ajtai key adversary oracle retries ∨
      RefreshUniqueness.Fails ajtai key adversary oracle retries := by
  by_cases truth : RefreshUniqueness.Truth ajtai key.relation
      (toClaim (key.system ajtai) (adversary.run oracle).link.start.1)
      (adversary.run oracle).link.fold.fresh
      (RefreshFork.proofOf (adversary.run oracle).link.fold.proof).split
      (RefreshFork.extractedWitness ajtai valid.1 valid.2)
  · exact Or.inl ⟨valid, RefreshUniqueness.extracted_of_truth ajtai valid.1 valid.2 truth⟩
  · exact Or.inr ⟨valid, truth⟩

/-- The chance that a resampled valid run gives a collision of the base and resampled prior
states (`UniquenessLink.stateChance`). -/
noncomputable def stateChance : ℝ :=
  UniquenessLink.stateChance ajtai key (RefreshFork.folds ajtai)
    (fun attempt => (attempt.link.fold.fresh, (RefreshFork.proofOf attempt.link.fold.proof).split))
    (fun attempt => RefreshSchedule.messages (RefreshFork.proofOf attempt.link.fold.proof).nifs)
    adversary

/-- The refresh knowledge error: the statistical error, the prior-state collision chances of
the fork and uniqueness steps, and the kernel chance. -/
noncomputable def knowledgeError (queries : Nat) : ℝ :=
  statisticalError queries +
    (𝔼 oracle, (if Fork.Succeeds (RefreshFork.game ajtai key) adversary oracle then
        ∑ index, RefreshFork.collisionChance ajtai key adversary index oracle else 0) +
      RefreshUniqueness.kernelChance ajtai key adversary + stateChance ajtai key adversary)

/-- The refresh fold's knowledge bound in the random-oracle model. -/
theorem knowledge_error_le {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, (if Fork.Succeeds (RefreshFork.game ajtai key) adversary oracle then (1 : ℝ)
        else 0) ≤
      𝔼 oracle, ∑ retries,
          Uniqueness.weight (RefreshFork.game ajtai key) adversary oracle retries *
            (if Extracts ajtai key adversary oracle retries then 1 else 0) +
        knowledgeError ajtai key adversary queries := by
  have bound := Uniqueness.knowledge_error_le (RefreshFork.game ajtai key) adversary
    (RefreshUniqueness.witnessOf ajtai key adversary)
    (RefreshUniqueness.source ajtai key adversary)
    (Extracts ajtai key adversary) (RefreshUniqueness.Fails ajtai key adversary)
    (splits ajtai key adversary) (fun fails => fails.1) (RefreshUniqueness.agree ajtai)
    (RefreshUniqueness.source_degreeCovers ajtai key adversary)
    (RefreshFork.collisionChance ajtai key adversary)
    (fun _ succeeds index => ForkLink.mismatchChance_le ajtai key (RefreshFork.folds ajtai)
      (statement := fun attempt =>
        (attempt.link.fold.fresh, (RefreshFork.proofOf attempt.link.fold.proof).split))
      (fun attempt => RefreshSchedule.messages (RefreshFork.proofOf attempt.link.fold.proof).nifs)
      adversary (fun same => congrArg Prod.fst same) succeeds index)
    bounded
  have state : Uniqueness.startChance (RefreshFork.game ajtai key) adversary ≤
      stateChance ajtai key adversary :=
    UniquenessLink.startChance_le_stateChance ajtai key (RefreshFork.folds ajtai)
      (statement := fun attempt =>
        (attempt.link.fold.fresh, (RefreshFork.proofOf attempt.link.fold.proof).split))
      (fun attempt => RefreshSchedule.messages (RefreshFork.proofOf attempt.link.fold.proof).nifs)
      adversary (fun same => congrArg Prod.fst same)
  have kernel := RefreshUniqueness.collisionChance_le_kernelChance ajtai key adversary
  rw [statisticalError_eq] at bound
  unfold knowledgeError
  linarith

end NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RefreshKnowledge
