import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.BlockLane

/-!
Focused regressions for canonical block×lane NC SumCheck semantics.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.block_lane.sumcheck.round_count` | accepted rounds equal typed block-plus-lane arity | accepting malformed certificate length |
| `nifs.pi_ccs.nc.block_lane.sumcheck.complete` | independent NC truth constructs a five-slot certificate | certificate-shaped semantic assumptions |
| `nifs.pi_ccs.nc.block_lane.sumcheck.soundness.decompose` | accepted zero claim yields truth or a named root family | unclassified soundness gap |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.BlockLane.Tests

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.SumCheck.Finite

/-- Typed points make accepted certificate arity structural. -/
example
    {domain : BlockNcDomain}
    (initial terminal : K)
    (point : Point domain)
    (certificate : Certificate)
    (accepted : Accepted initial point.coordinates terminal certificate) :
    certificate.rounds.length =
      domain.blockVariables + domain.laneVariables :=
  accepted_rounds_length initial terminal point certificate accepted

/-- Completeness begins from independent NC truth. -/
example
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (truth : Semantics.Nc.Truth data)
    (point : Point domain) :
    ∃ certificate : Certificate,
      Accepted InitialSum.claimedInitial point.coordinates
        (InitialSum.sumcheckPolynomial covers data coins point.coordinates)
        certificate :=
  complete_of_truth covers data coins truth point

/-- Deterministic soundness names every selector, gamma, or SumCheck root;
it makes no sampling-probability claim. -/
example
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
      BadEvent covers data coins point certificate challengeSetSize :=
  accepted_implies_truth_or_badEvent noZeroDivisors covers data coins
    point certificate challengeSetSize accepted

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.BlockLane.Tests
