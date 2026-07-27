import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveCarrier
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveRootCounting
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.DelayedCombinedNc.Acceptance

/-!
Finite mixing-root bounds for production block/lane NC.

Assurance tier: model-level registered-deviation refinement.

Owns: explicit lane and block Boolean-table controllers, the exact
constant-first source gamma polynomial, the delayed degree-one residual
identity, and finite-root bounds for the actual named NC events.

Does not own: FE mixing, SumCheck round collisions, Fiat--Shamir, Poseidon2,
closed concrete field certificates, Rust/R1CS, artifacts, costs, or rows.

Emits constraints: no.

| Boundary | Owned equation |
| --- | --- |
| Selectors | Lane and block controllers evaluate to the exact named root events |
| Gamma/residual | Coefficient evaluation equals the source gamma or delayed residual identity |
| Union | NC failure probability retains lane + (block + (gamma + residual)) ordering |
-/

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveNcMixing

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.SumCheck.Finite
open IdealInteractiveCarrier
open IdealInteractiveRootCounting

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws

private noncomputable def propositionEvent (proposition : Prop) : Bool :=
  @ite Bool proposition (Classical.propDecidable proposition) true false

/-- Exact production NC coins from the selected finite seed coordinates. -/
def coins
    (betaA : BetaAWord)
    (betaBlock : BetaBlockWord)
    (gamma : K) :
    Mixing.Coins PiCcsDomains.production.nc where
  betaBlock := cubePoint betaBlock
  betaA := cubePoint betaA
  gamma := gamma

/-! ## Lane selector -/

/-- One source/block lane table before the sampled `betaA` specialization. -/
def laneTable
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (block : BooleanVertex PiCcsDomains.production.nc.blockVariables) :
    BooleanTable K PiCcsDomains.production.nc.laneVariables :=
  BooleanTable.tabulate fun lane =>
    SourceProjection.rangeValueAt covers data source {
      block := block.toCubePoint ops
      lane := lane.toCubePoint ops
    }

/-- The lane table's MLE at `betaA` is exactly the repository's named lane
specialization. -/
theorem laneTable_evaluate
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (block : BooleanVertex PiCcsDomains.production.nc.blockVariables)
    (betaA : BetaAWord)
    (betaBlock : BetaBlockWord)
    (gamma : K) :
    (laneTable covers data source block).evaluate ops (cubePoint betaA) =
      InitialSum.laneResidualAtBeta covers data
        (coins betaA betaBlock gamma) source block := by
  exact
    (BooleanReproduction.equalityWeighted_tabulate_eq_evaluate
      ops laws (cubePoint betaA) (fun lane =>
        SourceProjection.rangeValueAt covers data source {
          block := block.toCubePoint ops
          lane := lane.toCubePoint ops
        })).symm

/-- A nonzero complete NC relation supplies one fixed nonzero lane table.
The controller is derived from authoritative source data and is not prover
input. -/
theorem laneController_of_relationNonzero
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (relationNonzero :
      ¬ SourceProjection.BooleanResidualsZero covers data) :
    ∃ source block,
      ¬ (laneTable covers data source block).AllEntriesZero ops := by
  classical
  obtain ⟨source, sourceFailure⟩ :=
    Classical.not_forall.mp relationNonzero
  obtain ⟨block, blockFailure⟩ :=
    Classical.not_forall.mp sourceFailure
  obtain ⟨lane, leafNonzero⟩ :=
    Classical.not_forall.mp blockFailure
  refine ⟨source, BlockNcDomain.blockVertex block, ?_⟩
  intro allZero
  have leafZero :=
    (BooleanTable.tabulate_allEntriesZero_iff ops _).mp allZero
      (BlockNcDomain.laneVertex lane)
  apply leafNonzero
  simpa [
    laneTable,
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.booleanPoint
  ] using leafZero

/-- Exact named lane-selector event for one sampled lane word. -/
noncomputable def laneSelectorRootEvent
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (betaBlock : BetaBlockWord)
    (gamma : K)
    (betaA : BetaAWord) : Bool :=
  propositionEvent
    (MixingSoundness.LaneSelectorRoot covers data
      (coins betaA betaBlock gamma))

/-- Finite multilinear bound for the actual `LaneSelectorRoot`. -/
theorem laneSelectorRoot_probability_le
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (betaBlock : BetaBlockWord)
    (gamma : K)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (alphabet : Support K) :
    ((Support.challengeVectors alphabet
      PiCcsDomains.production.nc.laneVariables).uniform).probabilityBool
        (laneSelectorRootEvent covers data betaBlock gamma) <=
      ratio PiCcsDomains.production.nc.laneVariables
        alphabet.cardinality := by
  classical
  by_cases relationZero :
      SourceProjection.BooleanResidualsZero covers data
  · have eventFalse :
        laneSelectorRootEvent covers data betaBlock gamma =
          fun _betaA => false := by
      funext betaA
      apply Bool.eq_false_iff.mpr
      intro eventTrue
      have root :
          MixingSoundness.LaneSelectorRoot covers data
            (coins betaA betaBlock gamma) := by
        simpa [laneSelectorRootEvent, propositionEvent] using eventTrue
      exact root.relationNonzero relationZero
    rw [eventFalse]
    have probabilityZero :
        ((Support.challengeVectors alphabet
          PiCcsDomains.production.nc.laneVariables).uniform
            ).probabilityBool (fun _betaA => false) = 0 := by
      unfold Experiment.probabilityBool Experiment.countBool
      simp [Rat.div_def]
    calc
      ((Support.challengeVectors alphabet
          PiCcsDomains.production.nc.laneVariables).uniform
            ).probabilityBool (fun _betaA => false) =
          0 := probabilityZero
      _ <= ratio PiCcsDomains.production.nc.laneVariables
          alphabet.cardinality :=
        ratio_nonneg _ alphabet.cardinality_pos
  · obtain ⟨source, block, controllerNonzero⟩ :=
      laneController_of_relationNonzero covers data relationZero
    let zeroEvent : BetaAWord -> Bool := fun betaA =>
      decide
        ((laneTable covers data source block).evaluateCoordinates ops
          (List.ofFn betaA) = ops.zero)
    have eventToZero :
        ∀ betaA,
          laneSelectorRootEvent covers data betaBlock gamma betaA = true ->
            zeroEvent betaA = true := by
      intro betaA eventTrue
      have root :
          MixingSoundness.LaneSelectorRoot covers data
            (coins betaA betaBlock gamma) := by
        simpa [laneSelectorRootEvent, propositionEvent] using eventTrue
      have selectedZero :=
        root.everyLaneSpecializationZero source block
      have evaluationZero :
          (laneTable covers data source block).evaluate
              ops (cubePoint betaA) = ops.zero := by
        rw [laneTable_evaluate covers data source block betaA betaBlock gamma]
        exact selectedZero
      simpa [zeroEvent, BooleanTable.evaluate, cubePoint] using evaluationZero
    calc
      ((Support.challengeVectors alphabet
          PiCcsDomains.production.nc.laneVariables).uniform).probabilityBool
            (laneSelectorRootEvent covers data betaBlock gamma) <=
          ((Support.challengeVectors alphabet
            PiCcsDomains.production.nc.laneVariables).uniform).probabilityBool
              zeroEvent :=
        Experiment.probabilityBool_mono _ eventToZero
      _ <= ratio PiCcsDomains.production.nc.laneVariables
            alphabet.cardinality := by
        exact multilinearZero_probability_le ops laws noZeroDivisors
          (laneTable covers data source block) alphabet controllerNonzero

/-! ## Block selector -/

/-- One source block table after the sampled lane word and before the sampled
`betaBlock`. The table is fixed by authoritative source data and the prior
lane challenge. -/
def blockTable
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (betaA : BetaAWord)
    (source : Fin shape.sourceCount) :
    BooleanTable K PiCcsDomains.production.nc.blockVariables :=
  BooleanTable.tabulate fun block =>
    InitialSum.laneResidualAtBeta covers data
      (coins betaA (fun _ => K.zero) K.zero) source block

/-- The block table's MLE at `betaBlock` is exactly the repository's named
source specialization. -/
theorem blockTable_evaluate
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (betaA : BetaAWord)
    (source : Fin shape.sourceCount)
    (betaBlock : BetaBlockWord)
    (gamma : K) :
    (blockTable covers data betaA source).evaluate ops
        (cubePoint betaBlock) =
      InitialSum.sourceResidualAtBeta covers data
        (coins betaA betaBlock gamma) source := by
  calc
    (blockTable covers data betaA source).evaluate ops
        (cubePoint betaBlock) =
      BooleanReproduction.equalityWeighted ops (cubePoint betaBlock)
        (fun block =>
          InitialSum.laneResidualAtBeta covers data
            (coins betaA (fun _ => K.zero) K.zero) source block) :=
        (BooleanReproduction.equalityWeighted_tabulate_eq_evaluate
          ops laws (cubePoint betaBlock) (fun block =>
            InitialSum.laneResidualAtBeta covers data
              (coins betaA (fun _ => K.zero) K.zero) source block)).symm
    _ = BooleanReproduction.equalityWeighted ops (cubePoint betaBlock)
        (fun block =>
          InitialSum.laneResidualAtBeta covers data
            (coins betaA betaBlock gamma) source block) := by
          rfl
    _ = InitialSum.sourceResidualAtBeta covers data
        (coins betaA betaBlock gamma) source := rfl

/-- Exact named block-selector event for one sampled block word. -/
noncomputable def blockSelectorRootEvent
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (betaA : BetaAWord)
    (gamma : K)
    (betaBlock : BetaBlockWord) : Bool :=
  propositionEvent
    (MixingSoundness.BlockSelectorRoot covers data
      (coins betaA betaBlock gamma))

/-- Finite multilinear bound for the actual `BlockSelectorRoot`, conditioned
on the complete prior lane word. -/
theorem blockSelectorRoot_probability_le
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (betaA : BetaAWord)
    (gamma : K)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (alphabet : Support K) :
    ((Support.challengeVectors alphabet
      PiCcsDomains.production.nc.blockVariables).uniform).probabilityBool
        (blockSelectorRootEvent covers data betaA gamma) <=
      ratio PiCcsDomains.production.nc.blockVariables
        alphabet.cardinality := by
  classical
  by_cases survives : ∃ source,
      ¬ (blockTable covers data betaA source).AllEntriesZero ops
  · obtain ⟨source, controllerNonzero⟩ := survives
    let zeroEvent : BetaBlockWord -> Bool := fun betaBlock =>
      decide
        ((blockTable covers data betaA source).evaluateCoordinates ops
          (List.ofFn betaBlock) = ops.zero)
    have eventToZero :
        ∀ betaBlock,
          blockSelectorRootEvent covers data betaA gamma betaBlock = true ->
            zeroEvent betaBlock = true := by
      intro betaBlock eventTrue
      have root :
          MixingSoundness.BlockSelectorRoot covers data
            (coins betaA betaBlock gamma) := by
        simpa [blockSelectorRootEvent, propositionEvent] using eventTrue
      have selectedZero := root.everySourceSpecializationZero source
      have evaluationZero :
          (blockTable covers data betaA source).evaluate
              ops (cubePoint betaBlock) = ops.zero := by
        rw [blockTable_evaluate covers data betaA source betaBlock gamma]
        exact selectedZero
      simpa [zeroEvent, BooleanTable.evaluate, cubePoint] using evaluationZero
    calc
      ((Support.challengeVectors alphabet
          PiCcsDomains.production.nc.blockVariables).uniform).probabilityBool
            (blockSelectorRootEvent covers data betaA gamma) <=
          ((Support.challengeVectors alphabet
            PiCcsDomains.production.nc.blockVariables).uniform
              ).probabilityBool zeroEvent :=
        Experiment.probabilityBool_mono _ eventToZero
      _ <= ratio PiCcsDomains.production.nc.blockVariables
            alphabet.cardinality := by
        exact multilinearZero_probability_le ops laws noZeroDivisors
          (blockTable covers data betaA source) alphabet controllerNonzero
  · have eventFalse :
        blockSelectorRootEvent covers data betaA gamma =
          fun _betaBlock => false := by
      funext betaBlock
      apply Bool.eq_false_iff.mpr
      intro eventTrue
      have root :
          MixingSoundness.BlockSelectorRoot covers data
            (coins betaA betaBlock gamma) := by
        simpa [blockSelectorRootEvent, propositionEvent] using eventTrue
      obtain ⟨source, block, laneNonzero⟩ :=
        root.someLaneSpecializationNonzero
      apply survives
      refine ⟨source, ?_⟩
      intro allZero
      have blockZero :=
        (BooleanTable.tabulate_allEntriesZero_iff ops _).mp allZero block
      apply laneNonzero
      simpa [blockTable, coins] using blockZero
    rw [eventFalse]
    have probabilityZero :
        ((Support.challengeVectors alphabet
          PiCcsDomains.production.nc.blockVariables).uniform
            ).probabilityBool (fun _betaBlock => false) = 0 := by
      unfold Experiment.probabilityBool Experiment.countBool
      simp [Rat.div_def]
    calc
      ((Support.challengeVectors alphabet
          PiCcsDomains.production.nc.blockVariables).uniform
            ).probabilityBool (fun _betaBlock => false) =
          0 := probabilityZero
      _ <= ratio PiCcsDomains.production.nc.blockVariables
          alphabet.cardinality :=
        ratio_nonneg _ alphabet.cardinality_pos

/-! ## Shared gamma polynomial -/

/-- Constant-first NC source coefficients after both selector words. The
coefficients are fixed before the shared `gamma` is sampled/evaluated. -/
def gammaCoefficients
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (betaA : BetaAWord)
    (betaBlock : BetaBlockWord) : List K :=
  (MixingSoundness.gammaPolynomial covers data
    (coins betaA betaBlock K.zero)).coefficients

theorem gammaCoefficients_length
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (betaA : BetaAWord)
    (betaBlock : BetaBlockWord) :
    (gammaCoefficients covers data betaA betaBlock).length =
      shape.sourceCount := by
  simp [gammaCoefficients, MixingSoundness.gammaPolynomial,
    canonicalFinIndices_length]

/-- Exact named gamma-polynomial event for the single gamma shared by FE and
NC. No second NC gamma is introduced. -/
noncomputable def gammaPolynomialRootEvent
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (betaA : BetaAWord)
    (betaBlock : BetaBlockWord)
    (gamma : K) : Bool :=
  propositionEvent
    (MixingSoundness.GammaPolynomialRoot covers data
      (coins betaA betaBlock gamma))

/-- Finite univariate bound for the actual NC gamma polynomial, conditioned
on the prior lane and block words. -/
theorem gammaPolynomialRoot_probability_le
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (betaA : BetaAWord)
    (betaBlock : BetaBlockWord)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (alphabet : Support K) :
    alphabet.uniform.probabilityBool
        (gammaPolynomialRootEvent covers data betaA betaBlock) <=
      ratio (shape.sourceCount - 1) alphabet.cardinality := by
  classical
  by_cases survives : ∃ source,
      InitialSum.sourceResidualAtBeta covers data
        (coins betaA betaBlock K.zero) source ≠ K.zero
  · obtain ⟨source, coefficientNonzero⟩ := survives
    have sourceCountPositive : 0 < shape.sourceCount := by
      exact Nat.zero_lt_of_lt source.isLt
    have coefficientCount :
        (gammaCoefficients covers data betaA betaBlock).length =
          (shape.sourceCount - 1) + 1 := by
      rw [gammaCoefficients_length]
      omega
    have coefficientsNonzero :
        ¬ CoefficientRootCounting.AllZero ops
          (gammaCoefficients covers data betaA betaBlock) := by
      intro allZero
      apply coefficientNonzero
      apply allZero
      simp [gammaCoefficients, MixingSoundness.gammaPolynomial,
        canonicalFinIndices]
    let zeroEvent : K -> Bool := fun gamma =>
      decide
        (Message.evaluateCoefficients ops.toOps gamma
          (gammaCoefficients covers data betaA betaBlock) = ops.zero)
    have eventToZero :
        ∀ gamma,
          gammaPolynomialRootEvent covers data betaA betaBlock gamma = true ->
            zeroEvent gamma = true := by
      intro gamma eventTrue
      have root :
          MixingSoundness.GammaPolynomialRoot covers data
            (coins betaA betaBlock gamma) := by
        simpa [gammaPolynomialRootEvent, propositionEvent] using eventTrue
      simpa [zeroEvent, gammaCoefficients,
        MixingSoundness.gammaPolynomial, Message.evaluate] using
          root.polynomialRoot
    calc
      alphabet.uniform.probabilityBool
          (gammaPolynomialRootEvent covers data betaA betaBlock) <=
        alphabet.uniform.probabilityBool zeroEvent :=
          Experiment.probabilityBool_mono _ eventToZero
      _ <= ratio (shape.sourceCount - 1) alphabet.cardinality := by
        exact coefficientZero_probability_le ops laws noZeroDivisors
          (shape.sourceCount - 1)
          (gammaCoefficients covers data betaA betaBlock)
          coefficientCount alphabet coefficientsNonzero
  · have eventFalse :
        gammaPolynomialRootEvent covers data betaA betaBlock =
          fun _gamma => false := by
      funext gamma
      apply Bool.eq_false_iff.mpr
      intro eventTrue
      have root :
          MixingSoundness.GammaPolynomialRoot covers data
            (coins betaA betaBlock gamma) := by
        simpa [gammaPolynomialRootEvent, propositionEvent] using eventTrue
      obtain ⟨source, coefficientNonzero⟩ :=
        root.someCoefficientNonzero
      apply survives
      refine ⟨source, ?_⟩
      exact coefficientNonzero
    rw [eventFalse]
    have probabilityZero :
        alphabet.uniform.probabilityBool (fun _gamma => false) = 0 := by
      unfold Experiment.probabilityBool Experiment.countBool
      simp [Rat.div_def]
    calc
      alphabet.uniform.probabilityBool (fun _gamma => false) =
          0 := probabilityZero
      _ <= ratio (shape.sourceCount - 1) alphabet.cardinality :=
        ratio_nonneg _ alphabet.cardinality_pos

/-! ## Delayed residual weight -/

private theorem neg_eq_zero_implies_eq_zero
    {value : K}
    (negZero : ops.neg value = ops.zero) :
    value = ops.zero := by
  calc
    value = ops.add value ops.zero := (laws.add_zero value).symm
    _ = ops.add value (ops.neg value) := by rw [negZero]
    _ = ops.zero := laws.add_neg value

private theorem mul_neg_right
    (left right : K) :
    ops.mul left (ops.neg right) = ops.neg (ops.mul left right) := by
  calc
    ops.mul left (ops.neg right) =
        ops.mul (ops.neg right) left := laws.mul_comm _ _
    _ = ops.neg (ops.mul right left) := laws.neg_mul _ _
    _ = ops.neg (ops.mul left right) := by rw [laws.mul_comm right left]

/-- The constant-first difference polynomial for the exact delayed identity:
`[0,parent] - [ordinary,raw]`. -/
def residualWeightDifference
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (betaA : BetaAWord)
    (betaBlock : BetaBlockWord)
    (gamma : K)
    (weights : RunningWeights shape)
    (producerBeta parentProjection : K)
    (oldBlock :
      CubePoint K PiCcsDomains.production.nc.blockVariables) : List K :=
  [
    ops.neg
      (InitialSum.mixedResidualAtBeta covers data
        (coins betaA betaBlock gamma)),
    ops.sub parentProjection
      (authoritativeRunningProjection covers data weights producerBeta
        oldBlock)
  ]

theorem residualWeightDifference_length
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (betaA : BetaAWord)
    (betaBlock : BetaBlockWord)
    (gamma : K)
    (weights : RunningWeights shape)
    (producerBeta parentProjection : K)
    (oldBlock :
      CubePoint K PiCcsDomains.production.nc.blockVariables) :
    (residualWeightDifference covers data betaA betaBlock gamma weights
      producerBeta parentProjection oldBlock).length = 1 + 1 := by
  rfl

/-- Exact repository residual-weight event as a Boolean predicate of the last
delayed scalar. -/
noncomputable def residualWeightRootEvent
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (betaA : BetaAWord)
    (betaBlock : BetaBlockWord)
    (gamma : K)
    (weights : RunningWeights shape)
    (producerBeta parentProjection : K)
    (oldBlock :
      CubePoint K PiCcsDomains.production.nc.blockVariables)
    (batchWeight : K) : Bool :=
  propositionEvent
    (DelayedCombinedNc.Acceptance.ResidualWeightRoot covers data
      (coins betaA betaBlock gamma) weights producerBeta batchWeight
      parentProjection oldBlock)

private theorem residualWeightDifference_nonzero_of_notExact
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (betaA : BetaAWord)
    (betaBlock : BetaBlockWord)
    (gamma : K)
    (weights : RunningWeights shape)
    (producerBeta parentProjection batchWeight : K)
    (oldBlock :
      CubePoint K PiCcsDomains.production.nc.blockVariables)
    (notExact :
      ¬ (DelayedCombinedNc.Acceptance.residualWeightIdentity covers data
        (coins betaA betaBlock gamma) weights producerBeta batchWeight
        parentProjection oldBlock).Exact) :
    ¬ CoefficientRootCounting.AllZero ops
      (residualWeightDifference covers data betaA betaBlock gamma weights
        producerBeta parentProjection oldBlock) := by
  intro allZero
  have constantZero :
      ops.neg
        (InitialSum.mixedResidualAtBeta covers data
          (coins betaA betaBlock gamma)) = ops.zero :=
    allZero _ (by
      simp [residualWeightDifference])
  have linearZero :
      ops.sub parentProjection
        (authoritativeRunningProjection covers data weights producerBeta
          oldBlock) = ops.zero :=
    allZero _ (by
      simp [residualWeightDifference])
  apply notExact
  apply
    (DelayedCombinedNc.Acceptance.residualWeightIdentity_exact_iff
      covers data (coins betaA betaBlock gamma) weights producerBeta
      batchWeight parentProjection oldBlock).2
  exact ⟨neg_eq_zero_implies_eq_zero constantZero,
    (FiniteSumAlgebra.sub_eq_zero_iff ops laws _ _).mp linearZero⟩

private theorem residualWeightRoot_implies_difference_zero
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (betaA : BetaAWord)
    (betaBlock : BetaBlockWord)
    (gamma : K)
    (weights : RunningWeights shape)
    (producerBeta parentProjection batchWeight : K)
    (oldBlock :
      CubePoint K PiCcsDomains.production.nc.blockVariables)
    (root :
      DelayedCombinedNc.Acceptance.ResidualWeightRoot covers data
        (coins betaA betaBlock gamma) weights producerBeta batchWeight
        parentProjection oldBlock) :
    Message.evaluateCoefficients ops.toOps batchWeight
        (residualWeightDifference covers data betaA betaBlock gamma weights
          producerBeta parentProjection oldBlock) = ops.zero := by
  let ordinary :=
    InitialSum.mixedResidualAtBeta covers data
      (coins betaA betaBlock gamma)
  let raw :=
    authoritativeRunningProjection covers data weights producerBeta oldBlock
  have collision :
      ops.mul batchWeight parentProjection =
        ops.add ordinary (ops.mul batchWeight raw) := by
    simpa [ordinary, raw] using
      DelayedCombinedNc.Acceptance.residualWeightRoot_equation
        covers data (coins betaA betaBlock gamma) weights producerBeta
        batchWeight parentProjection oldBlock root
  letI : Std.Associative ops.add := ⟨laws.add_assoc⟩
  letI : Std.Commutative ops.add := ⟨laws.add_comm⟩
  simp only [residualWeightDifference, Message.evaluateCoefficients,
    laws.mul_zero, laws.add_zero]
  change
    ops.add (ops.neg ordinary)
      (ops.mul batchWeight (ops.sub parentProjection raw)) = ops.zero
  rw [InterpolationOps.sub, laws.left_distrib, mul_neg_right]
  rw [collision]
  calc
    ops.add (ops.neg ordinary)
        (ops.add (ops.add ordinary (ops.mul batchWeight raw))
          (ops.neg (ops.mul batchWeight raw))) =
      ops.add
        (ops.add (ops.neg ordinary) ordinary)
        (ops.add (ops.mul batchWeight raw)
          (ops.neg (ops.mul batchWeight raw))) := by ac_rfl
    _ = ops.add ops.zero ops.zero := by
      rw [laws.add_comm (ops.neg ordinary) ordinary, laws.add_neg,
        laws.add_neg]
    _ = ops.zero := laws.zero_add ops.zero

/-- Exact degree-one root bound for the delayed residual identity. The
identity is fixed before `batchWeight`; no cancellation premise is used. -/
theorem residualWeightRoot_probability_le
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (betaA : BetaAWord)
    (betaBlock : BetaBlockWord)
    (gamma : K)
    (weights : RunningWeights shape)
    (producerBeta parentProjection : K)
    (oldBlock :
      CubePoint K PiCcsDomains.production.nc.blockVariables)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (alphabet : Support K) :
    alphabet.uniform.probabilityBool
        (residualWeightRootEvent covers data betaA betaBlock gamma weights
          producerBeta parentProjection oldBlock) <=
      ratio 1 alphabet.cardinality := by
  classical
  by_cases exact :
      (DelayedCombinedNc.Acceptance.residualWeightIdentity covers data
        (coins betaA betaBlock gamma) weights producerBeta K.zero
        parentProjection oldBlock).Exact
  · have eventFalse :
        residualWeightRootEvent covers data betaA betaBlock gamma weights
            producerBeta parentProjection oldBlock =
          fun _batchWeight => false := by
      funext batchWeight
      apply Bool.eq_false_iff.mpr
      intro eventTrue
      have root :
          DelayedCombinedNc.Acceptance.ResidualWeightRoot covers data
            (coins betaA betaBlock gamma) weights producerBeta batchWeight
            parentProjection oldBlock := by
        simpa [residualWeightRootEvent, propositionEvent] using eventTrue
      apply root.notExact
      exact exact
    rw [eventFalse]
    have probabilityZero :
        alphabet.uniform.probabilityBool
            (fun _batchWeight => false) = 0 := by
      unfold Experiment.probabilityBool Experiment.countBool
      simp [Rat.div_def]
    calc
      alphabet.uniform.probabilityBool (fun _batchWeight => false) =
          0 := probabilityZero
      _ <= ratio 1 alphabet.cardinality :=
        ratio_nonneg _ alphabet.cardinality_pos
  · have coefficientsNonzero :
        ¬ CoefficientRootCounting.AllZero ops
          (residualWeightDifference covers data betaA betaBlock gamma weights
            producerBeta parentProjection oldBlock) :=
      residualWeightDifference_nonzero_of_notExact covers data betaA
        betaBlock gamma weights producerBeta parentProjection K.zero oldBlock
        exact
    let zeroEvent : K -> Bool := fun batchWeight =>
      decide
        (Message.evaluateCoefficients ops.toOps batchWeight
          (residualWeightDifference covers data betaA betaBlock gamma weights
            producerBeta parentProjection oldBlock) = ops.zero)
    have eventToZero :
        ∀ batchWeight,
          residualWeightRootEvent covers data betaA betaBlock gamma weights
              producerBeta parentProjection oldBlock batchWeight = true ->
            zeroEvent batchWeight = true := by
      intro batchWeight eventTrue
      have root :
          DelayedCombinedNc.Acceptance.ResidualWeightRoot covers data
            (coins betaA betaBlock gamma) weights producerBeta batchWeight
            parentProjection oldBlock := by
        simpa [residualWeightRootEvent, propositionEvent] using eventTrue
      simpa [zeroEvent] using
        residualWeightRoot_implies_difference_zero covers data betaA betaBlock
          gamma weights producerBeta parentProjection batchWeight oldBlock root
    calc
      alphabet.uniform.probabilityBool
          (residualWeightRootEvent covers data betaA betaBlock gamma weights
            producerBeta parentProjection oldBlock) <=
        alphabet.uniform.probabilityBool zeroEvent :=
          Experiment.probabilityBool_mono _ eventToZero
      _ <= ratio 1 alphabet.cardinality := by
        exact coefficientZero_probability_le ops laws noZeroDivisors 1
          (residualWeightDifference covers data betaA betaBlock gamma weights
            producerBeta parentProjection oldBlock)
          (residualWeightDifference_length covers data betaA betaBlock gamma
            weights producerBeta parentProjection oldBlock)
          alphabet coefficientsNonzero

/-! ## Exact pre-SumCheck event and ordered support bounds -/

/-- Lane-root probability over the complete pre-SumCheck carrier. Product
transpositions below are finite Fubini equalities only; the protocol schedule
continues to sample `betaA` before shared `gamma` and `betaBlock`. -/
theorem laneSelectorRoot_pre_probability_le
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (alphabet : Support K) :
    ((preSupport (shape := shape) alphabet).uniform).probabilityBool
        (fun seed =>
          laneSelectorRootEvent covers data seed.betaBlock seed.gamma
            seed.betaA) <=
      ratio PiCcsDomains.production.nc.laneVariables
        alphabet.cardinality := by
  calc
    ((preSupport (shape := shape) alphabet).uniform).probabilityBool
        (fun seed =>
          laneSelectorRootEvent covers data seed.betaBlock seed.gamma
            seed.betaA) =
      (((delayedSupport alphabet).product
        (engineSupport (shape := shape) alphabet)).uniform).probabilityBool
        (fun seed =>
          laneSelectorRootEvent covers data seed.1.1.1 seed.2.2
            seed.2.1.1.2) := by
              unfold preSupport PreSeed.betaBlock PreSeed.gamma PreSeed.betaA
              exact product_swap_probabilityBool
                (engineSupport (shape := shape) alphabet)
                (delayedSupport alphabet)
                (fun engine delayed =>
                  laneSelectorRootEvent covers data delayed.1.1 engine.2
                    engine.1.1.2)
    _ <= ratio PiCcsDomains.production.nc.laneVariables
          alphabet.cardinality := by
      refine product_probabilityBool_le_of_components
        (delayedSupport alphabet)
        (engineSupport (shape := shape) alphabet)
        (fun delayed engine =>
          laneSelectorRootEvent covers data delayed.1.1 engine.2
            engine.1.1.2)
        (ratio PiCcsDomains.production.nc.laneVariables
          alphabet.cardinality) ?_
      intro delayed _delayedMember
      calc
        (engineSupport (shape := shape) alphabet).uniform.probabilityBool
            (fun engine =>
              laneSelectorRootEvent covers data delayed.1.1 engine.2
                engine.1.1.2) =
          ((alphabet.product
            (enginePrefixSupport (shape := shape) alphabet)).uniform
              ).probabilityBool
            (fun seed =>
              laneSelectorRootEvent covers data delayed.1.1 seed.1
                seed.2.1.2) := by
                  unfold engineSupport
                  exact product_swap_probabilityBool
                    (enginePrefixSupport (shape := shape) alphabet) alphabet
                    (fun engineHead gamma =>
                      laneSelectorRootEvent covers data delayed.1.1 gamma
                        engineHead.1.2)
        _ <= ratio PiCcsDomains.production.nc.laneVariables
              alphabet.cardinality := by
          refine product_probabilityBool_le_of_components alphabet
            (enginePrefixSupport (shape := shape) alphabet)
            (fun gamma engineHead =>
              laneSelectorRootEvent covers data delayed.1.1 gamma
                engineHead.1.2)
            (ratio PiCcsDomains.production.nc.laneVariables
              alphabet.cardinality) ?_
          intro gamma _gammaMember
          unfold enginePrefixSupport
          let laneSupport :=
            Support.challengeVectors alphabet
              PiCcsDomains.production.nc.laneVariables
          let rowSupport :=
            Support.challengeVectors alphabet shape.rowVariables
          calc
            (((laneSupport.product laneSupport).product rowSupport).uniform
                ).probabilityBool
                (fun engineHead =>
                  laneSelectorRootEvent covers data delayed.1.1 gamma
                    engineHead.1.2) =
              (laneSupport.product laneSupport).uniform.probabilityBool
                (fun pair =>
                  laneSelectorRootEvent covers data delayed.1.1 gamma
                    pair.2) :=
              Support.product_uniform_probabilityBool_first
                (laneSupport.product laneSupport) rowSupport
                (fun pair =>
                  laneSelectorRootEvent covers data delayed.1.1 gamma pair.2)
            _ = laneSupport.uniform.probabilityBool
                (laneSelectorRootEvent covers data delayed.1.1 gamma) :=
              Support.product_uniform_probabilityBool_second
                laneSupport laneSupport
                (laneSelectorRootEvent covers data delayed.1.1 gamma)
            _ <= ratio PiCcsDomains.production.nc.laneVariables
                  alphabet.cardinality :=
              laneSelectorRoot_probability_le covers data delayed.1.1 gamma
                noZeroDivisors alphabet

/-- Block-selector probability over the complete ordered pre-SumCheck
carrier. `betaBlock` is averaged only after fixing the engine batch. -/
theorem blockSelectorRoot_pre_probability_le
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (alphabet : Support K) :
    ((preSupport (shape := shape) alphabet).uniform).probabilityBool
        (fun seed =>
          blockSelectorRootEvent covers data seed.betaA seed.gamma
            seed.betaBlock) <=
      ratio PiCcsDomains.production.nc.blockVariables
        alphabet.cardinality := by
  unfold preSupport PreSeed.betaA PreSeed.gamma PreSeed.betaBlock
  refine product_probabilityBool_le_of_components
    (engineSupport (shape := shape) alphabet)
    (delayedSupport alphabet)
    (fun engine delayed =>
      blockSelectorRootEvent covers data engine.1.1.2 engine.2 delayed.1.1)
    (ratio PiCcsDomains.production.nc.blockVariables
      alphabet.cardinality) ?_
  intro engine _engineMember
  unfold delayedSupport
  calc
    ((delayedPrefixSupport alphabet).product alphabet).uniform.probabilityBool
        (fun delayed =>
          blockSelectorRootEvent covers data engine.1.1.2 engine.2
            delayed.1.1) =
      (delayedPrefixSupport alphabet).uniform.probabilityBool
        (fun delayedHead =>
          blockSelectorRootEvent covers data engine.1.1.2 engine.2
            delayedHead.1) :=
      Support.product_uniform_probabilityBool_first
        (delayedPrefixSupport alphabet) alphabet
        (fun delayedHead =>
          blockSelectorRootEvent covers data engine.1.1.2 engine.2
            delayedHead.1)
    _ = ((Support.challengeVectors alphabet
          PiCcsDomains.production.nc.blockVariables).uniform
            ).probabilityBool
          (blockSelectorRootEvent covers data engine.1.1.2 engine.2) := by
      unfold delayedPrefixSupport
      exact Support.product_uniform_probabilityBool_first
        (Support.challengeVectors alphabet
          PiCcsDomains.production.nc.blockVariables)
        alphabet
        (blockSelectorRootEvent covers data engine.1.1.2 engine.2)
    _ <= ratio PiCcsDomains.production.nc.blockVariables
          alphabet.cardinality :=
      blockSelectorRoot_probability_le covers data engine.1.1.2 engine.2
        noZeroDivisors alphabet

/-- Shared-gamma NC root probability over the complete pre-SumCheck carrier.
The finite product is transposed only to condition on the later
`betaBlock`; the protocol still owns one earlier gamma used by both FE and
NC. -/
theorem gammaPolynomialRoot_pre_probability_le
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (alphabet : Support K) :
    ((preSupport (shape := shape) alphabet).uniform).probabilityBool
        (fun seed =>
          gammaPolynomialRootEvent covers data seed.betaA seed.betaBlock
            seed.gamma) <=
      ratio (shape.sourceCount - 1) alphabet.cardinality := by
  calc
    ((preSupport (shape := shape) alphabet).uniform).probabilityBool
        (fun seed =>
          gammaPolynomialRootEvent covers data seed.betaA seed.betaBlock
            seed.gamma) =
      (((delayedSupport alphabet).product
        (engineSupport (shape := shape) alphabet)).uniform).probabilityBool
        (fun seed =>
          gammaPolynomialRootEvent covers data seed.2.1.1.2 seed.1.1.1
            seed.2.2) := by
              unfold preSupport PreSeed.betaA PreSeed.betaBlock PreSeed.gamma
              exact product_swap_probabilityBool
                (engineSupport (shape := shape) alphabet)
                (delayedSupport alphabet)
                (fun engine delayed =>
                  gammaPolynomialRootEvent covers data engine.1.1.2
                    delayed.1.1 engine.2)
    _ <= ratio (shape.sourceCount - 1) alphabet.cardinality := by
      refine product_probabilityBool_le_of_components
        (delayedSupport alphabet)
        (engineSupport (shape := shape) alphabet)
        (fun delayed engine =>
          gammaPolynomialRootEvent covers data engine.1.1.2 delayed.1.1
            engine.2)
        (ratio (shape.sourceCount - 1) alphabet.cardinality) ?_
      intro delayed _delayedMember
      unfold engineSupport
      refine product_probabilityBool_le_of_components
        (enginePrefixSupport (shape := shape) alphabet) alphabet
        (fun engineHead gamma =>
          gammaPolynomialRootEvent covers data engineHead.1.2 delayed.1.1
            gamma)
        (ratio (shape.sourceCount - 1) alphabet.cardinality) ?_
      intro engineHead _engineHeadMember
      exact gammaPolynomialRoot_probability_le covers data engineHead.1.2
        delayed.1.1 noZeroDivisors alphabet

/-- Delayed residual-weight probability over the complete pre-SumCheck
carrier. `batchWeight` remains the final delayed coordinate, after
`producerBeta`. -/
theorem residualWeightRoot_pre_probability_le
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (pending : ProductionDelayedBlockLane)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (alphabet : Support K) :
    ((preSupport (shape := shape) alphabet).uniform).probabilityBool
        (fun seed =>
          residualWeightRootEvent covers data seed.betaA seed.betaBlock
            seed.gamma weights seed.producerBeta
            (DelayedPackedProjection.projectedValue pending.parentYZcol
              seed.producerBeta)
            pending.oldBlock seed.batchWeight) <=
      ratio 1 alphabet.cardinality := by
  unfold preSupport PreSeed.betaA PreSeed.betaBlock PreSeed.gamma
    PreSeed.producerBeta PreSeed.batchWeight
  refine product_probabilityBool_le_of_components
    (engineSupport (shape := shape) alphabet)
    (delayedSupport alphabet)
    (fun engine delayed =>
      residualWeightRootEvent covers data engine.1.1.2 delayed.1.1 engine.2
        weights delayed.1.2
        (DelayedPackedProjection.projectedValue pending.parentYZcol
          delayed.1.2)
        pending.oldBlock delayed.2)
    (ratio 1 alphabet.cardinality) ?_
  intro engine _engineMember
  unfold delayedSupport
  refine product_probabilityBool_le_of_components
    (delayedPrefixSupport alphabet) alphabet
    (fun delayedHead batchWeight =>
      residualWeightRootEvent covers data engine.1.1.2 delayedHead.1 engine.2
        weights delayedHead.2
        (DelayedPackedProjection.projectedValue pending.parentYZcol
          delayedHead.2)
        pending.oldBlock batchWeight)
    (ratio 1 alphabet.cardinality) ?_
  intro delayedHead _delayedHeadMember
  exact residualWeightRoot_probability_le covers data engine.1.1.2
    delayedHead.1 engine.2 weights delayedHead.2
    (DelayedPackedProjection.projectedValue pending.parentYZcol
      delayedHead.2)
    pending.oldBlock noZeroDivisors alphabet

private theorem probabilityBool_or_le_of_bounds
    {Outcome : Type}
    (experiment : Experiment Outcome)
    (left right : Outcome -> Bool)
    (leftBudget rightBudget : Rat)
    (leftBound : experiment.probabilityBool left <= leftBudget)
    (rightBound : experiment.probabilityBool right <= rightBudget) :
    experiment.probabilityBool
        (fun outcome => left outcome || right outcome) <=
      leftBudget + rightBudget := by
  exact Rat.le_trans
    (experiment.probabilityBool_or_le left right)
    (Rat.le_trans
      ((Rat.add_le_add_right
        (c := experiment.probabilityBool right)).mpr leftBound)
      ((Rat.add_le_add_left (c := leftBudget)).mpr rightBound))

/-- The exact optional delayed event. Base steps carry no pending value and
therefore have no residual-weight root branch. -/
noncomputable def optionalResidualWeightRootEvent
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (pending : Option ProductionDelayedBlockLane)
    (seed : PreSeed shape) : Bool :=
  match pending with
  | none => false
  | some delayed =>
      residualWeightRootEvent covers data seed.betaA seed.betaBlock
        seed.gamma weights seed.producerBeta
        (DelayedPackedProjection.projectedValue delayed.parentYZcol
          seed.producerBeta)
        delayed.oldBlock seed.batchWeight

@[simp] theorem optionalResidualWeightRootEvent_eq_true_iff
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (pending : Option ProductionDelayedBlockLane)
    (seed : PreSeed shape) :
    optionalResidualWeightRootEvent covers data weights pending seed = true ↔
      ∃ delayed,
        pending = some delayed ∧
          DelayedCombinedNc.Acceptance.ResidualWeightRoot covers data
            (coins seed.betaA seed.betaBlock seed.gamma)
            weights seed.producerBeta seed.batchWeight
            (DelayedPackedProjection.projectedValue delayed.parentYZcol
              seed.producerBeta)
            delayed.oldBlock := by
  cases pending with
  | none =>
      simp [optionalResidualWeightRootEvent]
  | some delayed =>
      simp [optionalResidualWeightRootEvent, residualWeightRootEvent,
        propositionEvent]

theorem optionalResidualWeightRoot_pre_probability_le
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (pending : Option ProductionDelayedBlockLane)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (alphabet : Support K) :
    ((preSupport (shape := shape) alphabet).uniform).probabilityBool
        (optionalResidualWeightRootEvent covers data weights pending) <=
      ratio 1 alphabet.cardinality := by
  cases pending with
  | none =>
      have probabilityZero :
          ((preSupport (shape := shape) alphabet).uniform).probabilityBool
              (fun _seed => false) = 0 := by
        unfold Experiment.probabilityBool Experiment.countBool
        simp [Rat.div_def]
      calc
        ((preSupport (shape := shape) alphabet).uniform).probabilityBool
            (optionalResidualWeightRootEvent covers data weights none) =
          0 := by
            simpa [optionalResidualWeightRootEvent] using probabilityZero
        _ <= ratio 1 alphabet.cardinality :=
          ratio_nonneg _ alphabet.cardinality_pos
  | some delayed =>
      simpa [optionalResidualWeightRootEvent] using
        residualWeightRoot_pre_probability_le covers data weights delayed
          noZeroDivisors alphabet

/-- Exact right-nested production NC mixing event. Its order matches
`ProductionRefinement.NcFailure`: lane, block, shared-gamma polynomial, then
delayed residual weight. -/
noncomputable def ncMixingRootEvent
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (pending : Option ProductionDelayedBlockLane)
    (seed : PreSeed shape) : Bool :=
  laneSelectorRootEvent covers data seed.betaBlock seed.gamma seed.betaA ||
    (blockSelectorRootEvent covers data seed.betaA seed.gamma seed.betaBlock ||
      (gammaPolynomialRootEvent covers data seed.betaA seed.betaBlock
          seed.gamma ||
        optionalResidualWeightRootEvent covers data weights pending seed))

@[simp] theorem ncMixingRootEvent_eq_true_iff
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (pending : Option ProductionDelayedBlockLane)
    (seed : PreSeed shape) :
    ncMixingRootEvent covers data weights pending seed = true ↔
      MixingSoundness.LaneSelectorRoot covers data
          (coins seed.betaA seed.betaBlock seed.gamma) ∨
        (MixingSoundness.BlockSelectorRoot covers data
            (coins seed.betaA seed.betaBlock seed.gamma) ∨
          (MixingSoundness.GammaPolynomialRoot covers data
              (coins seed.betaA seed.betaBlock seed.gamma) ∨
            ∃ delayed,
              pending = some delayed ∧
                DelayedCombinedNc.Acceptance.ResidualWeightRoot covers data
                  (coins seed.betaA seed.betaBlock seed.gamma)
                  weights seed.producerBeta seed.batchWeight
                  (DelayedPackedProjection.projectedValue
                    delayed.parentYZcol seed.producerBeta)
                  delayed.oldBlock)) := by
  simp [ncMixingRootEvent, laneSelectorRootEvent, blockSelectorRootEvent,
    gammaPolynomialRootEvent, propositionEvent, Bool.or_eq_true]

/-- Explicit finite union bound for every actual NC mixing-root constructor.
The loss expression is intentionally right-associated in the frozen event
order. -/
theorem ncMixingRoot_probability_le
    {shape : SemanticShape}
    (covers : PiCcsDomains.production.nc.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (pending : Option ProductionDelayedBlockLane)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (alphabet : Support K) :
    ((preSupport (shape := shape) alphabet).uniform).probabilityBool
        (ncMixingRootEvent covers data weights pending) <=
      ratio PiCcsDomains.production.nc.laneVariables alphabet.cardinality +
        (ratio PiCcsDomains.production.nc.blockVariables
            alphabet.cardinality +
          (ratio (shape.sourceCount - 1) alphabet.cardinality +
            ratio 1 alphabet.cardinality)) := by
  let experiment := (preSupport (shape := shape) alphabet).uniform
  let lane : PreSeed shape -> Bool := fun seed =>
    laneSelectorRootEvent covers data seed.betaBlock seed.gamma seed.betaA
  let block : PreSeed shape -> Bool := fun seed =>
    blockSelectorRootEvent covers data seed.betaA seed.gamma seed.betaBlock
  let gammaRoot : PreSeed shape -> Bool := fun seed =>
    gammaPolynomialRootEvent covers data seed.betaA seed.betaBlock seed.gamma
  let residual : PreSeed shape -> Bool :=
    optionalResidualWeightRootEvent covers data weights pending
  have laneBound :
      experiment.probabilityBool lane <=
        ratio PiCcsDomains.production.nc.laneVariables
          alphabet.cardinality := by
    simpa [experiment, lane] using
      laneSelectorRoot_pre_probability_le covers data noZeroDivisors alphabet
  have blockBound :
      experiment.probabilityBool block <=
        ratio PiCcsDomains.production.nc.blockVariables
          alphabet.cardinality := by
    simpa [experiment, block] using
      blockSelectorRoot_pre_probability_le covers data noZeroDivisors alphabet
  have gammaBound :
      experiment.probabilityBool gammaRoot <=
        ratio (shape.sourceCount - 1) alphabet.cardinality := by
    simpa [experiment, gammaRoot] using
      gammaPolynomialRoot_pre_probability_le covers data noZeroDivisors
        alphabet
  have residualBound :
      experiment.probabilityBool residual <=
        ratio 1 alphabet.cardinality := by
    simpa [experiment, residual] using
      optionalResidualWeightRoot_pre_probability_le covers data weights
        pending noZeroDivisors alphabet
  have gammaResidual :
      experiment.probabilityBool
          (fun seed => gammaRoot seed || residual seed) <=
        ratio (shape.sourceCount - 1) alphabet.cardinality +
          ratio 1 alphabet.cardinality :=
    probabilityBool_or_le_of_bounds experiment gammaRoot residual
      (ratio (shape.sourceCount - 1) alphabet.cardinality)
      (ratio 1 alphabet.cardinality) gammaBound residualBound
  have blockTail :
      experiment.probabilityBool
          (fun seed =>
            block seed || (gammaRoot seed || residual seed)) <=
        ratio PiCcsDomains.production.nc.blockVariables
            alphabet.cardinality +
          (ratio (shape.sourceCount - 1) alphabet.cardinality +
            ratio 1 alphabet.cardinality) :=
    probabilityBool_or_le_of_bounds experiment block
      (fun seed => gammaRoot seed || residual seed)
      (ratio PiCcsDomains.production.nc.blockVariables alphabet.cardinality)
      (ratio (shape.sourceCount - 1) alphabet.cardinality +
        ratio 1 alphabet.cardinality)
      blockBound gammaResidual
  have allRoots :
      experiment.probabilityBool
          (fun seed =>
            lane seed ||
              (block seed || (gammaRoot seed || residual seed))) <=
        ratio PiCcsDomains.production.nc.laneVariables alphabet.cardinality +
          (ratio PiCcsDomains.production.nc.blockVariables
              alphabet.cardinality +
            (ratio (shape.sourceCount - 1) alphabet.cardinality +
              ratio 1 alphabet.cardinality)) :=
    probabilityBool_or_le_of_bounds experiment lane
      (fun seed => block seed || (gammaRoot seed || residual seed))
      (ratio PiCcsDomains.production.nc.laneVariables alphabet.cardinality)
      (ratio PiCcsDomains.production.nc.blockVariables alphabet.cardinality +
        (ratio (shape.sourceCount - 1) alphabet.cardinality +
          ratio 1 alphabet.cardinality))
      laneBound blockTail
  simpa [experiment, lane, block, gammaRoot, residual,
    ncMixingRootEvent] using allRoots

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveNcMixing
