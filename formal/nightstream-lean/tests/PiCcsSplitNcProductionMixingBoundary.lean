import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialDegree.Support

/-!
Focused regressions for the production Split-NC mixing-carrier obstruction and
its concrete algebra boundary.
-/

set_option autoImplicit false

namespace tests.PiCcsSplitNcProductionMixingBoundary

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc
open Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary

private abbrev ops := ConcreteCarrier.extensionOps

private def polynomialLaws : FixedPolynomial.Laws ops.toOps :=
  ProtocolPolynomialDegree.Support.polynomialLaws
    ConcreteCarrier.extensionLaws

/-! Exact production event inventory and unchanged collision theorem. -/

#check ProductionRefinement.FeFailure.sumcheck
#check SumCheck.Fe.BadEvent.mixingRoot
#check SumCheck.Fe.BadEvent.roundCollision
#check ProductionRefinement.NcFailure.laneSelectorRoot
#check ProductionRefinement.NcFailure.blockSelectorRoot
#check ProductionRefinement.NcFailure.gammaPolynomialRoot
#check ProductionRefinement.NcFailure.residualWeightRoot
#check ProductionRefinement.NcFailure.roundCollision
#check feFailure_exact_cases
#check ncFailure_exact_cases
#check CausalSoundness.splitCollision_probability_le

/-! Concrete field-law transport. -/

#check goldilocksBaseNoZeroDivisors
#check productionExtensionNoZeroDivisors

/-! Support well-formedness and malformed-list rejection. -/

def twoPointSupport : Support K where
  values := [K.zero, K.one]
  nodup := by decide
  nonempty := by decide

example : twoPointSupport.cardinality = 2 := by
  rfl

example : twoPointSupport.values.Nodup :=
  twoPointSupport.nodup

example : twoPointSupport.values ≠ [] :=
  twoPointSupport.nonempty

example : ¬ ∃ support : Support K, support.values = [] := by
  rintro ⟨support, empty⟩
  exact support.nonempty empty

example (value : K) :
    ¬ ∃ support : Support K, support.values = [value, value] := by
  rintro ⟨support, duplicate⟩
  have nodup := support.nodup
  rw [duplicate] at nodup
  have duplicateNotNodup : ¬ [value, value].Nodup := by
    simp
  exact duplicateNotNodup nodup

/-! Zero, degree-zero, and degree-at-least-cardinality edge cases. -/

example (degree : Nat) :
    (fun point : K =>
      (FixedPolynomial.zero ops.toOps degree).evaluate ops.toOps point) =
      fun _ => ops.zero := by
  funext point
  exact FixedPolynomial.evaluate_zero ops.toOps polynomialLaws degree point

example (degree : Nat) :
    ¬ ((fun point : K =>
        (FixedPolynomial.zero ops.toOps degree).evaluate ops.toOps point) ≠
      fun _ => ops.zero) := by
  intro nonzero
  apply nonzero
  funext point
  exact FixedPolynomial.evaluate_zero ops.toOps polynomialLaws degree point

def degreeZeroOne : FixedPolynomial K 0 :=
  FixedPolynomial.constant ops.one

theorem degreeZeroOne_nonzero :
    (fun point : K => degreeZeroOne.evaluate ops.toOps point) ≠
      fun _ => ops.zero := by
  intro equal
  have atZero := congrFun equal K.zero
  have one_ne_zero : ops.one ≠ ops.zero := by decide
  apply one_ne_zero
  simpa [degreeZeroOne, polynomialLaws] using atZero

example
    (euclid : NormRange.GoldilocksModulusEuclid)
    (sevenNonresidue : ConcreteCarrier.SevenProjectiveNonresidue) :
    twoPointSupport.values.countP (fun value =>
      decide (degreeZeroOne.evaluate ops.toOps value = ops.zero)) <= 0 := by
  exact FiniteRootCounting.roots_count_le_degree
    ops ConcreteCarrier.extensionLaws
    (productionExtensionNoZeroDivisors euclid sevenNonresidue)
    0 degreeZeroOne twoPointSupport.values twoPointSupport.nodup
    degreeZeroOne_nonzero

def widenedOne (degree : Nat) : FixedPolynomial K degree :=
  FixedPolynomial.widen ops.toOps (Nat.zero_le degree) degreeZeroOne

theorem widenedOne_nonzero (degree : Nat) :
    (fun point : K => (widenedOne degree).evaluate ops.toOps point) ≠
      fun _ => ops.zero := by
  intro equal
  have atZero := congrFun equal K.zero
  have one_ne_zero : ops.one ≠ ops.zero := by decide
  apply one_ne_zero
  simpa [widenedOne, degreeZeroOne, polynomialLaws] using atZero

example : twoPointSupport.cardinality <= 3 := by
  decide

example
    (euclid : NormRange.GoldilocksModulusEuclid)
    (sevenNonresidue : ConcreteCarrier.SevenProjectiveNonresidue) :
    twoPointSupport.values.countP (fun value =>
      decide ((widenedOne 3).evaluate ops.toOps value = ops.zero)) <= 3 := by
  exact FiniteRootCounting.roots_count_le_degree
    ops ConcreteCarrier.extensionLaws
    (productionExtensionNoZeroDivisors euclid sevenNonresidue)
    3 (widenedOne 3) twoPointSupport.values twoPointSupport.nodup
    (widenedOne_nonzero 3)

/-! Exact denominator/support obstruction on the production context carrier. -/

universe uState

section ProductionContext

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
  {arity : BatchArity productionGlobalParams}

example
    (input : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (alphabet : Support K) :
    ChallengeSupportAligned input.full alphabet ↔
      input.full.challengeSetSize = alphabet.cardinality := by
  rfl

example
    (input : Context shape State publicRingColumns publicFits verifierRows arity) :
    ¬ ∃ alphabet : Support K,
      ChallengeSupportAligned (withChallengeSetSize input 0) alphabet :=
  zeroChallengeSetSize_has_no_aligned_support input

#check withChallengeSetSize_statement
#check withChallengeSetSize_priorState
#check withChallengeSetSize_schedule
#check withChallengeSetSize_feCoins
#check withChallengeSetSize_ncCoins

end ProductionContext

/-! Shared-gamma correlation and exact delayed ordering. -/

universe uVerifierKey uInput

section CoreSchedule

variable
  {shape : SemanticShape}
  {State : Type uState}
  {VerifierKey : Type uVerifierKey}
  {Input : Type uInput}
  {domains : Domains}

example
    (schedule : Schedule VerifierKey Input shape domains State)
    (value : K)
    (priorState : State)
    (statement : Statement VerifierKey Input) :
    let challenges :=
      (derivePreSumcheck
        (replaceCoreChallenges schedule
          (fun _ => constantCoreChallenges value))
        priorState statement).challenges
    challenges.feCoins.gamma = value ∧
      challenges.ncCoins.gamma = value ∧
      challenges.ncCoins.gamma = challenges.feCoins.gamma ∧
      challenges.feCoins.betaA.coordinates =
        List.replicate domains.laneVariables value ∧
      challenges.ncCoins.betaA.coordinates =
        List.replicate domains.laneVariables value :=
  derivePreSumcheck_constantCore_shared_gamma
    schedule value priorState statement

#check derivePreSumcheck_replaceCore_producerBeta
#check derivePreSumcheck_replaceCore_batchWeight
#check derivePreSumcheck_replaceCore_state
#check replaceCoreChallenges_same_state
#check replaceCoreChallenges_different_challenges

end CoreSchedule

/-! FE first, then adaptive but prefix-causal NC. -/

section AdaptiveNc

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness

variable
  {shape : SemanticShape}
  {input : PublicInput shape}
  {feDomain : FlatNcDomain}
  {ncRounds : Nat}

example
    (strategy : SplitStrategy input feDomain ncRounds)
    (feWord : Fin (feRoundCount shape feDomain) -> K) :
    NcStrategy ncRounds :=
  strategy.nc feWord

-- The callback receives only a typed prior prefix, never the current challenge.
#check NcStrategy.message

end AdaptiveNc

end tests.PiCcsSplitNcProductionMixingBoundary
