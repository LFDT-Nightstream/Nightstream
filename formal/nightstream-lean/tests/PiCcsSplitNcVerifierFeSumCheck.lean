import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe

/-!
Focused regressions for mixed-width Split-NC FE SumCheck replay.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.fe.sumcheck.serialization.order` | physical messages split exactly into row prefix and lane suffix | phase reorder or boundary drift |
| `nifs.pi_ccs.fe.sumcheck.serialization.lane` | every physical lane message has exactly three slots | accidental serialization of the widened proof view |
| `nifs.pi_ccs.fe.sumcheck.proof_view.evaluation` | semantic zero extension preserves physical lane evaluation | invalid generic-chain reuse |
| `nifs.pi_ccs.fe.sumcheck.chain` | one chain consumes row then lane challenges without reset | artificial phase-boundary claim |
| `nifs.pi_ccs.fe.sumcheck.honesty` | physical phase honesty implies generic semantic honesty | hidden widening premise |
| `nifs.pi_ccs.fe.sumcheck.degree` | independently derived rounds fit the verifier-owned row ceiling | caller-supplied degree premise |
| `nifs.pi_ccs.fe.sumcheck.completeness` | FE truth and honest physical rounds imply acceptance | semantic/claimed-chain drift |
| `nifs.pi_ccs.fe.sumcheck.soundness` | acceptance reduces to FE truth or named bad events | unclassified false acceptance |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Tests

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe
open Nightstream.SuperNeo.SumCheck.Finite

private abbrev ops := ConcreteCarrier.extensionOps

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (certificate : Certificate input domain) :
    certificate.rawRounds.take shape.rowVariables =
        certificate.rowRawRounds ∧
      certificate.rawRounds.drop shape.rowVariables =
        certificate.laneRawRounds := by
  simp

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (certificate : Certificate input domain) :
    ∀ message ∈ certificate.rawRounds.drop shape.rowVariables,
      message.coefficients.length = 3 := by
  intro message member
  rw [Certificate.rawRounds_drop_rowVariables] at member
  exact Certificate.laneRawRounds_width certificate message member

/-- The physical lane message remains three slots while only the semantic
proof view has `Drow + 1` slots. The statement is polymorphic in the input,
including inputs whose row ceiling is strictly larger than two. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (certificate : Certificate input domain)
    (lane : Fin domain.laneVariables) :
    (certificate.laneRounds lane).toMessage.coefficients.length = 3 ∧
      (laneToUniform input (certificate.laneRounds lane)).toMessage.coefficients.length =
        Drow input + 1 := by
  constructor
  · simpa [laneSumcheckDegreeBound] using
      (certificate.laneRounds lane).toMessage_coefficients_length
  · exact
      (laneToUniform input (certificate.laneRounds lane)).toMessage_coefficients_length

example
    {shape : SemanticShape}
    {input : PublicInput shape}
    (message : LaneMessage)
    (point : K) :
    (laneToUniform input message).evaluate ops.toOps point =
      message.evaluate ops.toOps point :=
  lane_evaluate_uniform input message point

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (initial terminal : K)
    (point : Point shape domain)
    (certificate : Certificate input domain) :
    Accepted initial terminal point certificate ↔
      FixedPhase.Chain ops.toOps initial certificate.uniformRounds
        (point.row.coordinates ++ point.lane.coordinates) terminal := by
  rfl

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (initial terminal : K)
    (point : Point shape domain)
    (certificate : Certificate input domain) :
    check initial terminal point certificate = true ↔
      Accepted initial terminal point certificate :=
  check_eq_true_iff_accepted initial terminal point certificate

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (q : List K -> K)
    (point : Point shape domain)
    (certificate : Certificate input domain)
    (honest : HonestAt q point certificate) :
    FixedPhase.Honest ops.toOps q point.coordinates
      { rounds := certificate.uniformRounds } :=
  honestAt_implies_fixedPhaseHonest q point certificate honest

/-- Honest physical messages exist at every fixed verifier point without
materializing a uniform-width wire certificate. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (point : Point shape domain) :
    exists certificate : Certificate (PublicInput.ofSources data) domain,
      HonestAt (InitialSum.sumcheckPolynomial profile data coins)
        point certificate :=
  exists_honestAt profile data coins point

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (point : Point shape domain) :
    FixedPhase.ExpectedRoundsRepresentable ops.toOps
      (InitialSum.sumcheckPolynomial profile data coins)
      (Drow (PublicInput.ofSources data)) point.coordinates :=
  expectedRoundsRepresentable profile data coins point

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (truth : Semantics.Fe.Truth data)
    (point : Point shape domain)
    (certificate : Certificate (PublicInput.ofSources data) domain)
    (honest : HonestAt
      (InitialSum.sumcheckPolynomial profile data coins) point certificate) :
    Accepted
      (initial profile (PublicInput.ofSources data) coins)
      (InitialSum.sumcheckPolynomial profile data coins point.coordinates)
      point certificate :=
  complete_of_truth_and_honestAt profile data coins truth point certificate honest

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (point : Point shape domain)
    (certificate : Certificate (PublicInput.ofSources data) domain)
    (challengeSetSize : Nat)
    (accepted :
      Accepted
        (initial profile (PublicInput.ofSources data) coins)
        (InitialSum.sumcheckPolynomial profile data coins point.coordinates)
        point certificate) :
    Semantics.Fe.Truth data ∨
      BadEvent profile data coins point certificate challengeSetSize :=
  accepted_implies_truth_or_badEvent profile data coins point certificate
    challengeSetSize accepted

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Tests
