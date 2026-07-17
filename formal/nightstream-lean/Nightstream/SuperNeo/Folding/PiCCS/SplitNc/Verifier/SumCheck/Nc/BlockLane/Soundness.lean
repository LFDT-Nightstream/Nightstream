import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.BlockLane.Completeness
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.MixingSoundness

/-!
Deterministic soundness decomposition for canonical block×lane NC SumCheck.

Assurance tier: model-level.

Owns: reduction of a false accepted initial claim to a fixed-degree round
collision and the complete truth-or-lane-selector-or-block-selector-or-gamma-
polynomial-or-round-collision decomposition.

Does not own: Fiat–Shamir sampling, probability bounds, packed-output
terminal authority, Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: acceptance in the final theorem uses the independent
semantic terminal `q(point.coordinates)`. A later output-authority theorem
must prove that a concrete message-derived terminal equals that value. This
module does not assume that a digest or scalar terminal is authoritative.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.block_lane.sumcheck.soundness.round` | false accepted initial claim exposes a fixed-degree collision | security boundary | `false_acceptance_implies_bad_challenge` |
| `nifs.pi_ccs.nc.block_lane.sumcheck.soundness.events` | classify lane-selector, block-selector, gamma-polynomial, and round roots | security boundary | `BadEvent` |
| `nifs.pi_ccs.nc.block_lane.sumcheck.soundness.decompose` | semantic acceptance implies truth or one named event | derived | `accepted_implies_truth_or_badEvent` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.BlockLane

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.SumCheck.Finite

private abbrev ops := ConcreteCarrier.extensionOps

/-- Under exact terminal binding and the independently proved round-degree
bound, a false accepted initial claim exposes the generic root-count-ready
SumCheck collision. No sampling or probability bound is asserted. -/
theorem false_acceptance_implies_bad_challenge
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (challengeSetSize : Nat)
    (initial terminal : K)
    (point : Point domain)
    (certificate : Certificate)
    (terminalBinding : terminal =
      InitialSum.sumcheckPolynomial covers data coins point.coordinates)
    (accepted : Accepted initial point.coordinates terminal certificate)
    (falseClaim : initial ≠
      FixedPhase.semanticInitial ops.toOps
        (InitialSum.sumcheckPolynomial covers data coins)
        point.coordinates.length) :
    ∃ round,
      FixedPhase.BadChallenge ops.toOps
        (InitialSum.sumcheckPolynomial covers data coins)
        Polynomial.Nc.Degree.ncSumcheckDegreeBound challengeSetSize
        initial point.coordinates certificate round := by
  exact FixedPhase.false_acceptance_implies_bad_challenge ops.toOps
    (InitialSum.sumcheckPolynomial covers data coins)
    challengeSetSize initial point.coordinates certificate
    (expectedRoundsRepresentable covers data coins point)
    (semanticAccepted_of_terminal_binding covers data coins
      initial terminal point certificate terminalBinding accepted)
    falseClaim

/-- Exhaustive deterministic reasons why semantic NC SumCheck may accept
without the independent full-carrier norm relation. -/
inductive BadEvent
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (point : Point domain)
    (certificate : Certificate)
    (challengeSetSize : Nat) : Prop where
  | laneSelectorRoot
      (root : MixingSoundness.LaneSelectorRoot covers data coins) :
      BadEvent covers data coins point certificate challengeSetSize
  | blockSelectorRoot
      (root : MixingSoundness.BlockSelectorRoot covers data coins) :
      BadEvent covers data coins point certificate challengeSetSize
  | gammaPolynomialRoot
      (root : MixingSoundness.GammaPolynomialRoot covers data coins) :
      BadEvent covers data coins point certificate challengeSetSize
  | roundCollision
      (round : Nightstream.SuperNeo.SumCheck.Round K K)
      (collision :
        FixedPhase.BadChallenge ops.toOps
          (InitialSum.sumcheckPolynomial covers data coins)
          Polynomial.Nc.Degree.ncSumcheckDegreeBound challengeSetSize
          InitialSum.claimedInitial point.coordinates certificate round) :
      BadEvent covers data coins point certificate challengeSetSize

/-- Semantic acceptance is sound up to the named lane-selector,
block-selector, gamma-polynomial, and fixed-degree round-collision events.
The theorem is deterministic and expects the independent semantic terminal;
it is not yet an output-message or Fiat–Shamir soundness theorem. -/
theorem accepted_implies_truth_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (point : Point domain)
    (certificate : Certificate)
    (challengeSetSize : Nat)
    (accepted :
      Accepted InitialSum.claimedInitial point.coordinates
        (InitialSum.sumcheckPolynomial covers data coins point.coordinates)
        certificate) :
    Semantics.Nc.Truth data ∨
      BadEvent covers data coins point certificate challengeSetSize := by
  by_cases mixtureZero :
      InitialSum.mixedResidualAtBeta covers data coins = K.zero
  · rcases
      (MixingSoundness.mixedResidualAtBeta_eq_zero_iff_truth_or_laneSelectorRoot_or_blockSelectorRoot_or_gammaPolynomialRoot
        noZeroDivisors covers data coins).mp mixtureZero with
      truth | laneSelectorRoot | blockSelectorRoot | gammaPolynomialRoot
    · exact Or.inl truth
    · exact Or.inr (.laneSelectorRoot laneSelectorRoot)
    · exact Or.inr (.blockSelectorRoot blockSelectorRoot)
    · exact Or.inr (.gammaPolynomialRoot gammaPolynomialRoot)
  · apply Or.inr
    have semanticInitial_eq_mixture :
        FixedPhase.semanticInitial ops.toOps
            (InitialSum.sumcheckPolynomial covers data coins)
            point.coordinates.length =
          InitialSum.mixedResidualAtBeta covers data coins := by
      unfold FixedPhase.semanticInitial
      rw [point.coordinates_length]
      change InitialSum.sumcheckHypercubeSum covers data coins = _
      rw [InitialSum.sumcheckHypercubeSum_eq_hypercubeSum,
        InitialSum.hypercubeSum_eq_mixedResidualAtBeta]
    have falseClaim :
        InitialSum.claimedInitial ≠
          FixedPhase.semanticInitial ops.toOps
            (InitialSum.sumcheckPolynomial covers data coins)
            point.coordinates.length := by
      intro claimedEqualsSemantic
      apply mixtureZero
      calc
        InitialSum.mixedResidualAtBeta covers data coins =
            FixedPhase.semanticInitial ops.toOps
              (InitialSum.sumcheckPolynomial covers data coins)
              point.coordinates.length := semanticInitial_eq_mixture.symm
        _ = InitialSum.claimedInitial := claimedEqualsSemantic.symm
        _ = K.zero := rfl
    rcases false_acceptance_implies_bad_challenge covers data coins
        challengeSetSize InitialSum.claimedInitial
        (InitialSum.sumcheckPolynomial covers data coins point.coordinates)
        point certificate rfl accepted falseClaim with
      ⟨round, collision⟩
    exact .roundCollision round collision

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.BlockLane
