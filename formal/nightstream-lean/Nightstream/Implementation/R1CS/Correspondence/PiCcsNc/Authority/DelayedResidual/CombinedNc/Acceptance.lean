import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.MixingSoundness
import Nightstream.SuperNeo.ProjectionCheck
import Nightstream.SuperNeo.SumCheck.FixedPhase

/-!
Contract: derive the deterministic semantic outcome of the production-native
combined NC SumCheck whose terminal is evaluated directly from the
authoritative raw assignment polynomial.

Assurance tier: model-level.

Owns: exact block-before-lane Boolean-cube serialization, fixed-degree
representability of every combined round, the degree-one residual-weight
identity, and the truth/parent-projection-or-named-root decomposition of
semantic combined-NC acceptance.

Does not own: transcript sampling or domain separation, a concrete terminal
gadget, one-fold state continuity, commitment binding, Rust, generated rows,
costs, or row-removal permission.

Emits constraints: no.

Authority boundary: the accepted terminal is
`CombinedNc.sumcheckPolynomial ... finalPoint.coordinates`, hence is computed
from `Sources.Data.runningAssignments`. No output message, `CeClaim.y_zcol`,
digest, caller-provided terminal equality, or `ProjectionCheck.Accepted`
premise enters this leaf. The physical verifier must later refine its terminal
gadget to this raw-witness evaluator.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.nc.delayed_running.acceptance.cube` | recursive SumCheck truth enumerates the exact block-then-lane combined cube | derived | `semanticInitial_eq_ordinary_add_weightedProjection` |
| `pi_ccs.nc.delayed_running.acceptance.degree` | all exact-arity expected rounds fit the existing five-slot quartic carrier | derived | `expectedRoundsRepresentable` |
| `pi_ccs.nc.delayed_running.acceptance.residual_weight` | `[0,parent] = [ordinary,raw]` at `batchWeight`, including the zero-weight case | security boundary | `ResidualWeightRoot` |
| `pi_ccs.nc.delayed_running.acceptance.soundness` | acceptance yields NC truth and the delayed parent identity, or one named selector/gamma/residual/SumCheck event | derived | `accepted_implies_truth_and_parentProjection_or_badEvent` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.Acceptance

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.MixingSoundness
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree
open Nightstream.SuperNeo.SumCheck.Finite

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws

private def projectionOps : Nightstream.SuperNeo.ProjectionCheck.Ops K where
  zero := K.zero
  add := K.add
  mul := K.mul

private theorem projectionEval_pair
    (constant linear point : K) :
    Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
        [constant, linear] point =
      K.add constant (K.mul point linear) := by
  unfold Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
  simp only [List.foldr]
  change ops.add constant
      (ops.mul point (ops.add linear (ops.mul point ops.zero))) =
    ops.add constant (ops.mul point linear)
  rw [laws.mul_zero, laws.add_zero]

/-! ## Exact combined cube seen by generic SumCheck -/

/-- Exact-arity serialization evaluates the same typed combined polynomial.
This is a raw-witness terminal bridge, not a message-binding premise. -/
theorem sumcheckPolynomial_coordinates_eq_combinedAtPoint
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (point : Point domain) :
    sumcheckPolynomial covers data coins weights producerBeta
        batchWeight oldBlock point.coordinates =
      combinedAtPoint covers data coins weights producerBeta
        batchWeight oldBlock point := by
  unfold sumcheckPolynomial
  rw [dif_pos point.coordinates_length]
  rw [Point.ofCoordinates_coordinates]

/-- Recursive Boolean completions enumerate the exact product cube in
production block-before-lane order. -/
private theorem sumcheckHypercubeSum_eq_combinedHypercubeSum
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables) :
    HypercubeTruth.sumCompletions ops.toOps
        (sumcheckPolynomial covers data coins weights producerBeta
          batchWeight oldBlock) []
        (domain.blockVariables + domain.laneVariables) =
      combinedHypercubeSum covers data coins weights producerBeta
        batchWeight oldBlock := by
  rw [HypercubeTruth.sumCompletions_add]
  rw [SumCheckTruthPath.sumCompletions_eq_vertexSum ops laws]
  unfold combinedHypercubeSum FiniteSumAlgebra.sumMap
  simp only [List.nil_append]
  congr 1
  apply List.map_congr_left
  intro block _
  rw [SumCheckTruthPath.sumCompletions_eq_vertexSum ops laws]
  congr 1
  apply List.map_congr_left
  intro lane _
  exact sumcheckPolynomial_coordinates_eq_combinedAtPoint covers data coins
    weights producerBeta batchWeight oldBlock {
      block := block.toCubePoint ops
      lane := lane.toCubePoint ops }

/-- The semantic initial value of the combined polynomial is the ordinary
canonical NC source mixture plus the weighted projection of the authoritative
running assignments. -/
theorem semanticInitial_eq_ordinary_add_weightedProjection
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (point : Point domain) :
    FixedPhase.semanticInitial ops.toOps
        (sumcheckPolynomial covers data coins weights producerBeta
          batchWeight oldBlock) point.coordinates.length =
      K.add (InitialSum.mixedResidualAtBeta covers data coins)
        (K.mul batchWeight
          (authoritativeRunningProjection covers data weights
            producerBeta oldBlock)) := by
  unfold FixedPhase.semanticInitial
  rw [point.coordinates_length]
  rw [sumcheckHypercubeSum_eq_combinedHypercubeSum]
  rw [combinedHypercubeSum_eq_ordinary_add_weightedProjection]
  rw [InitialSum.hypercubeSum_eq_mixedResidualAtBeta]

/-! ## Degree-one residual-weight event -/

/-- The verifier's carried-parent claim and the independently normalized cube
are compared as two fixed-width degree-one coefficient vectors:
`[0, parent]` and `[ordinary NC residual, authoritative raw projection]`.
Its evaluation point is exactly `batchWeight`. -/
def residualWeightIdentity
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight parentProjection : K)
    (oldBlock : CubePoint K domain.blockVariables) :
    Nightstream.SuperNeo.ProjectionCheck.Identity K where
  lhs := [K.zero, parentProjection]
  rhs := [
    InitialSum.mixedResidualAtBeta covers data coins,
    authoritativeRunningProjection covers data weights producerBeta
      oldBlock]
  beta := batchWeight
  maxDegree := 1

/-- A non-exact degree-one residual identity that nevertheless agrees at the
sampled batch weight. In particular, `batchWeight = 0` does not authorize
cancellation: a parent-projection mismatch with zero ordinary residual lands
in this event. -/
def ResidualWeightRoot
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight parentProjection : K)
    (oldBlock : CubePoint K domain.blockVariables) : Prop :=
  Nightstream.SuperNeo.ProjectionCheck.BadRoot projectionOps
    (residualWeightIdentity covers data coins weights producerBeta
      batchWeight parentProjection oldBlock)

theorem residualWeightIdentity_exact_iff
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight parentProjection : K)
    (oldBlock : CubePoint K domain.blockVariables) :
    (residualWeightIdentity covers data coins weights producerBeta
        batchWeight parentProjection oldBlock).Exact ↔
      InitialSum.mixedResidualAtBeta covers data coins = K.zero ∧
      parentProjection =
        authoritativeRunningProjection covers data weights
          producerBeta oldBlock := by
  simp [Nightstream.SuperNeo.ProjectionCheck.Identity.Exact,
    residualWeightIdentity, eq_comm]

private theorem residualWeightIdentity_accepted_iff
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight parentProjection : K)
    (oldBlock : CubePoint K domain.blockVariables) :
    Nightstream.SuperNeo.ProjectionCheck.Accepted projectionOps
        (residualWeightIdentity covers data coins weights producerBeta
          batchWeight parentProjection oldBlock) ↔
      K.mul batchWeight parentProjection =
        K.add (InitialSum.mixedResidualAtBeta covers data coins)
          (K.mul batchWeight
            (authoritativeRunningProjection covers data weights
              producerBeta oldBlock)) := by
  constructor
  · intro accepted
    have collision := accepted.2
    change Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
        [K.zero, parentProjection] batchWeight =
      Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
        [InitialSum.mixedResidualAtBeta covers data coins,
          authoritativeRunningProjection covers data weights producerBeta
            oldBlock] batchWeight at collision
    rw [projectionEval_pair, projectionEval_pair] at collision
    change ops.add ops.zero (ops.mul batchWeight parentProjection) =
      ops.add (InitialSum.mixedResidualAtBeta covers data coins)
        (ops.mul batchWeight
          (authoritativeRunningProjection covers data weights producerBeta
            oldBlock)) at collision
    rw [laws.zero_add] at collision
    exact collision
  · intro weightedEquation
    constructor
    · simp [Nightstream.SuperNeo.ProjectionCheck.Identity.WellFormed,
        residualWeightIdentity]
    · change Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
          [K.zero, parentProjection] batchWeight =
        Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
          [InitialSum.mixedResidualAtBeta covers data coins,
            authoritativeRunningProjection covers data weights producerBeta
              oldBlock] batchWeight
      rw [projectionEval_pair, projectionEval_pair]
      change ops.add ops.zero (ops.mul batchWeight parentProjection) =
        ops.add (InitialSum.mixedResidualAtBeta covers data coins)
          (ops.mul batchWeight
            (authoritativeRunningProjection covers data weights producerBeta
              oldBlock))
      rw [laws.zero_add]
      exact weightedEquation

/-! ## Expected-round representability -/

private theorem expectedPolynomialsFrom_representable
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (fixed challenges : List K)
    (arity : fixed.length + challenges.length =
      domain.blockVariables + domain.laneVariables) :
    ∀ expected ∈ HypercubeTruth.expectedPolynomialsFrom ops.toOps
        (sumcheckPolynomial covers data coins weights producerBeta
          batchWeight oldBlock) fixed challenges,
      ∃ polynomial : FixedPolynomial K ncSumcheckDegreeBound,
        FixedPhase.Represents ops.toOps polynomial expected := by
  induction challenges generalizing fixed with
  | nil =>
      simp [HypercubeTruth.expectedPolynomialsFrom]
  | cons challenge challenges inductionHypothesis =>
      intro expected expectedIn
      simp only [HypercubeTruth.expectedPolynomialsFrom,
        List.mem_cons] at expectedIn
      rcases expectedIn with rfl | expectedIn
      · rcases expectedRound_quartic covers data coins weights
          producerBeta batchWeight oldBlock fixed challenges.length (by
            simp only [List.length_cons] at arity
            omega) with ⟨polynomial, represents⟩
        exact ⟨polynomial, represents⟩
      · exact inductionHypothesis (fixed := fixed ++ [challenge]) (by
          simp only [List.length_cons] at arity
          simp only [List.length_append, List.length_singleton]
          omega) expected expectedIn

/-- The combined polynomial's proved quartic slices discharge generic
fixed-phase representability; the caller supplies no degree assumption. -/
theorem expectedRoundsRepresentable
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (point : Point domain) :
    FixedPhase.ExpectedRoundsRepresentable ops.toOps
      (sumcheckPolynomial covers data coins weights producerBeta
        batchWeight oldBlock)
      ncSumcheckDegreeBound point.coordinates := by
  intro expected expectedIn
  exact expectedPolynomialsFrom_representable covers data coins weights
    producerBeta batchWeight oldBlock [] point.coordinates
    (by simpa using point.coordinates_length) expected (by
      simpa [FixedPhase.expectedRounds, HypercubeTruth.expectedPolynomials]
        using expectedIn)

/-! ## Deterministic acceptance decomposition -/

/-- Production-native combined-NC acceptance yields both current NC truth and
the exact delayed parent scalar, unless one precise existing selector/gamma
event, the degree-one residual-weight root, or a fixed-degree SumCheck round
collision occurs.

The premise's terminal is definitionally the authoritative combined
polynomial at `point.coordinates`; no output-message binding is assumed. -/
theorem accepted_implies_truth_and_parentProjection_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight parentProjection : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (point : Point domain)
    (certificate : FixedPhase.Certificate K
      ncSumcheckDegreeBound)
    (challengeSetSize : Nat)
    (accepted : FixedPhase.Accepted ops.toOps
      (sumcheckPolynomial covers data coins weights producerBeta
        batchWeight oldBlock)
      (K.mul batchWeight parentProjection) point.coordinates certificate) :
    (Semantics.Nc.Truth data ∧
        parentProjection =
          authoritativeRunningProjection covers data weights
            producerBeta oldBlock) ∨
      LaneSelectorRoot covers data coins ∨
      BlockSelectorRoot covers data coins ∨
      GammaPolynomialRoot covers data coins ∨
      ResidualWeightRoot covers data coins weights producerBeta batchWeight
        parentProjection oldBlock ∨
      ∃ round,
        FixedPhase.BadChallenge ops.toOps
          (sumcheckPolynomial covers data coins weights producerBeta
            batchWeight oldBlock)
          ncSumcheckDegreeBound challengeSetSize
          (K.mul batchWeight parentProjection) point.coordinates certificate
          round := by
  by_cases claimTrue :
      K.mul batchWeight parentProjection =
        FixedPhase.semanticInitial ops.toOps
          (sumcheckPolynomial covers data coins weights producerBeta
            batchWeight oldBlock) point.coordinates.length
  · have weightedAccepted :
        Nightstream.SuperNeo.ProjectionCheck.Accepted projectionOps
          (residualWeightIdentity covers data coins weights producerBeta
            batchWeight parentProjection oldBlock) :=
      (residualWeightIdentity_accepted_iff covers data coins weights
        producerBeta batchWeight parentProjection oldBlock).2 <| by
          rw [claimTrue]
          exact semanticInitial_eq_ordinary_add_weightedProjection covers data
            coins weights producerBeta batchWeight oldBlock point
    rcases Nightstream.SuperNeo.ProjectionCheck.accepted_implies_exact_or_badRoot
        projectionOps
        (residualWeightIdentity covers data coins weights producerBeta
          batchWeight parentProjection oldBlock)
        weightedAccepted with exact | residualRoot
    · have exactParts :=
        (residualWeightIdentity_exact_iff covers data coins weights
          producerBeta batchWeight parentProjection oldBlock).1 exact
      rcases
        (mixedResidualAtBeta_eq_zero_iff_truth_or_laneSelectorRoot_or_blockSelectorRoot_or_gammaPolynomialRoot
          noZeroDivisors covers data coins).1 exactParts.1 with
        truth | laneRoot | blockRoot | gammaRoot
      · exact Or.inl ⟨truth, exactParts.2⟩
      · exact Or.inr (Or.inl laneRoot)
      · exact Or.inr (Or.inr (Or.inl blockRoot))
      · exact Or.inr (Or.inr (Or.inr (Or.inl gammaRoot)))
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl residualRoot))))
  · apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inr
    exact FixedPhase.false_acceptance_implies_bad_challenge ops.toOps
      (sumcheckPolynomial covers data coins weights producerBeta
        batchWeight oldBlock)
      challengeSetSize (K.mul batchWeight parentProjection)
      point.coordinates certificate
      (expectedRoundsRepresentable covers data coins weights producerBeta
        batchWeight oldBlock point)
      accepted claimTrue

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.Acceptance
