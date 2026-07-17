import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.Interface

/-!
Semantic bridge for fixed-width canonical block×lane NC SumCheck.

Assurance tier: model-level.

Owns: binding a verifier-visible terminal to the independent total
polynomial and proving that every exact-arity expected round is representable
by the shared five-slot carrier.

Does not own: certificate round count, transcript replay, completeness,
soundness events, packed-output authority, Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: source data enters only through an explicit terminal
equality and the independently defined polynomial. The physical checker
receives no source assignment or expected polynomial. Exact arity and
block-before-lane order come from the typed point; the raw five-slot decoder
alone does not enforce either property.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.block_lane.sumcheck.semantic_terminal` | forwarded terminal equals independent `q(challenges)` | explicit proof boundary | `semanticAccepted_of_terminal_binding` |
| `nifs.pi_ccs.nc.block_lane.sumcheck.degree` | every exact-arity semantic round fits five slots | derived | `expectedRoundsRepresentable` |
| `nifs.pi_ccs.nc.block_lane.sumcheck.round_count` | accepted certificate length equals typed point arity | derived | `accepted_rounds_length` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.BlockLane

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.SumCheck.Finite

private abbrev ops := ConcreteCarrier.extensionOps

/-- Explicit terminal equality upgrades verifier-visible chain acceptance to
the independent semantic fixed-phase relation. -/
theorem semanticAccepted_of_terminal_binding
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (initial terminal : K)
    (point : Point domain)
    (certificate : Certificate)
    (terminalBinding : terminal =
      InitialSum.sumcheckPolynomial covers data coins point.coordinates)
    (accepted : Accepted initial point.coordinates terminal certificate) :
    FixedPhase.Accepted ops.toOps
      (InitialSum.sumcheckPolynomial covers data coins)
      initial point.coordinates certificate := by
  unfold FixedPhase.Accepted
  rw [← terminalBinding]
  exact accepted

private theorem expectedPolynomialsFrom_representable
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (fixed challenges : List K)
    (arity : fixed.length + challenges.length =
      domain.blockVariables + domain.laneVariables) :
    ∀ expected ∈ HypercubeTruth.expectedPolynomialsFrom ops.toOps
        (InitialSum.sumcheckPolynomial covers data coins)
        fixed challenges,
      ∃ polynomial : RoundMessage,
        FixedPhase.Represents ops.toOps polynomial expected := by
  induction challenges generalizing fixed with
  | nil =>
      simp [HypercubeTruth.expectedPolynomialsFrom]
  | cons challenge challenges inductionHypothesis =>
      intro expected expectedIn
      simp only [HypercubeTruth.expectedPolynomialsFrom,
        List.mem_cons] at expectedIn
      rcases expectedIn with rfl | expectedIn
      · rcases Degree.SumCheck.expectedRound_quartic covers data coins
          fixed challenges.length (by
            simp only [List.length_cons] at arity
            omega) with ⟨polynomial, represents⟩
        exact ⟨polynomial, represents⟩
      · exact inductionHypothesis (fixed := fixed ++ [challenge]) (by
          simp only [List.length_cons] at arity
          simp only [List.length_append, List.length_singleton]
          omega) expected expectedIn

/-- The independent round-degree proof discharges the generic fixed-phase
representability premise for every exact block×lane challenge vector. -/
theorem expectedRoundsRepresentable
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (point : Point domain) :
    FixedPhase.ExpectedRoundsRepresentable ops.toOps
      (InitialSum.sumcheckPolynomial covers data coins)
      Polynomial.Nc.Degree.ncSumcheckDegreeBound point.coordinates := by
  intro expected expectedIn
  exact expectedPolynomialsFrom_representable covers data coins
    [] point.coordinates (by simpa using point.coordinates_length) expected (by
      simpa [FixedPhase.expectedRounds, HypercubeTruth.expectedPolynomials]
        using expectedIn)

/-- Acceptance at a typed block×lane point forces exactly one certificate
round per coordinate. For the fixed profile, a later domain theorem reduces
this symbolic count to nine. -/
theorem accepted_rounds_length
    {domain : BlockNcDomain}
    (initial terminal : K)
    (point : Point domain)
    (certificate : Certificate)
    (accepted : Accepted initial point.coordinates terminal certificate) :
    certificate.rounds.length =
      domain.blockVariables + domain.laneVariables := by
  rw [← point.coordinates_length]
  exact FixedPhase.Chain.rounds_length_eq_challenges_length ops.toOps
    initial terminal certificate.rounds point.coordinates accepted

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.BlockLane
