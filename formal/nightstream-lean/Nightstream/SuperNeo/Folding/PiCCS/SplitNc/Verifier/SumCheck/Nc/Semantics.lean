import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.MixingSoundness
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.Interface

/-!
Semantic completeness and deterministic soundness decomposition for exact-
width Split-NC NC SumCheck.

Owns: terminal binding to the independent polynomial, degree representability,
fixed-challenge completeness, and the named selector/gamma/round-collision
bad-event decomposition.

Does not own: the physical certificate/checker interface, transcript sampling,
`yZcol` authority, probability bounds, Rust, R1CS, rows, removals, or costs.

Emits constraints: no.

Authority boundary: source data enters only through explicit semantic terminal
binding and the independent NC polynomial. The executable interface receives
no source assignment, expected polynomial, or semantic witness.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.sumcheck.semantic_terminal` | bind the forwarded terminal to independent `q(challenges)` | explicit proof boundary | `semanticAccepted_of_terminal_binding` |
| `nifs.pi_ccs.nc.sumcheck.degree` | every semantic round fits five slots | derived | `expectedRoundsRepresentable` |
| `nifs.pi_ccs.nc.sumcheck.completeness` | honest NC truth has an accepted exact-width certificate | model-level | `complete_of_truth` |
| `nifs.pi_ccs.nc.sumcheck.soundness.round` | a false accepted initial claim yields a fixed-degree collision | security boundary | `false_acceptance_implies_bad_challenge` |
| `nifs.pi_ccs.nc.sumcheck.soundness` | false acceptance implies selector, gamma, or round collision | security boundary | `accepted_implies_truth_or_badEvent` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing
open Nightstream.SuperNeo.SumCheck.Finite

private abbrev ops := ConcreteCarrier.extensionOps

/-- An explicit terminal equality upgrades verifier-visible acceptance to the
semantic fixed-phase relation. This is the only direction in which hidden
source data enters the soundness reduction. -/
theorem semanticAccepted_of_terminal_binding
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (initial terminal : K)
    (challenges : List K)
    (certificate : Certificate)
    (terminalBinding : terminal =
      InitialSum.sumcheckPolynomial convention covers data coins challenges)
    (accepted : Accepted initial challenges terminal certificate) :
    FixedPhase.Accepted ops.toOps
      (InitialSum.sumcheckPolynomial convention covers data coins)
      initial challenges certificate := by
  unfold FixedPhase.Accepted
  rw [← terminalBinding]
  exact accepted

private theorem expectedPolynomialsFrom_representable
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (fixed challenges : List K)
    (arity : fixed.length + challenges.length =
      domain.columnVariables + domain.laneVariables) :
    ∀ expected ∈ HypercubeTruth.expectedPolynomialsFrom ops.toOps
        (InitialSum.sumcheckPolynomial convention covers data coins)
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
      · rcases Degree.expectedRound_quartic convention covers data coins
          fixed challenges.length (by
            simp only [List.length_cons] at arity
            omega) with ⟨polynomial, represents⟩
        exact ⟨polynomial, represents⟩
      · exact inductionHypothesis (fixed := fixed ++ [challenge]) (by
          simp only [List.length_cons] at arity
          simp only [List.length_append, List.length_singleton]
          omega) expected expectedIn

/-- The independent quartic theorem discharges the generic fixed-phase degree
premise for every exact-arity NC challenge vector. This is the only bridge
from protocol-specific degree reasoning into generic SumCheck replay. -/
theorem expectedRoundsRepresentable
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (challenges : List K)
    (arity : challenges.length =
      domain.columnVariables + domain.laneVariables) :
    FixedPhase.ExpectedRoundsRepresentable ops.toOps
      (InitialSum.sumcheckPolynomial convention covers data coins)
      Degree.ncSumcheckDegreeBound challenges := by
  intro expected expectedIn
  exact expectedPolynomialsFrom_representable convention covers data coins
    [] challenges (by simpa using arity) expected (by
      simpa [FixedPhase.expectedRounds, HypercubeTruth.expectedPolynomials]
        using expectedIn)

/-- Under exact terminal binding and the independently proved quartic bound,
a false accepted initial claim exposes the generic root-count-ready SumCheck
collision. This theorem does not sample challenges or bound the event. -/
theorem false_acceptance_implies_bad_challenge
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (challengeSetSize : Nat)
    (initial terminal : K)
    (challenges : List K)
    (certificate : Certificate)
    (arity : challenges.length =
      domain.columnVariables + domain.laneVariables)
    (terminalBinding : terminal =
      InitialSum.sumcheckPolynomial convention covers data coins challenges)
    (accepted : Accepted initial challenges terminal certificate)
    (falseClaim : initial ≠
      FixedPhase.semanticInitial ops.toOps
        (InitialSum.sumcheckPolynomial convention covers data coins)
        challenges.length) :
    ∃ round,
      FixedPhase.BadChallenge ops.toOps
        (InitialSum.sumcheckPolynomial convention covers data coins)
        Degree.ncSumcheckDegreeBound challengeSetSize initial challenges
        certificate round := by
  exact FixedPhase.false_acceptance_implies_bad_challenge ops.toOps
    (InitialSum.sumcheckPolynomial convention covers data coins)
    challengeSetSize initial challenges certificate
    (expectedRoundsRepresentable convention covers data coins challenges arity)
    (semanticAccepted_of_terminal_binding convention covers data coins
      initial terminal challenges certificate terminalBinding accepted)
    falseClaim

/-- Exhaustive deterministic reasons why paper-relative NC SumCheck may accept
without the independent full-carrier norm relation. The first two events are
owned by semantic compression; the third is owned by SumCheck challenge
sampling. No probability bound is asserted here. -/
inductive BadEvent
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (challenges : List K)
    (certificate : Certificate)
    (challengeSetSize : Nat) : Prop where
  | selectorRoot
      (root :
        MixingSoundness.SelectorRoot covers data coins) :
      BadEvent covers data coins challenges certificate challengeSetSize
  | gammaRoot
      (root :
        MixingSoundness.GammaRoot covers data coins) :
      BadEvent covers data coins challenges certificate challengeSetSize
  | roundCollision
      (round : Nightstream.SuperNeo.SumCheck.Round K K)
      (collision :
        FixedPhase.BadChallenge ops.toOps
          (InitialSum.sumcheckPolynomial .paperNc covers data coins)
          Degree.ncSumcheckDegreeBound challengeSetSize
          InitialSum.claimedInitial challenges certificate round) :
      BadEvent covers data coins challenges certificate challengeSetSize

/-- Honest full-carrier norm truth has a complete typed five-slot NC
certificate for every exact-arity verifier challenge vector.

This is algebraic completeness after a challenge vector is fixed. It is not
Fiat--Shamir prover construction: that later theorem must derive each next
challenge only after absorbing the preceding message. -/
theorem complete_of_truth
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (truth : Semantics.Nc.Truth data)
    (challenges : List K)
    (arity : challenges.length =
      domain.columnVariables + domain.laneVariables) :
    ∃ certificate : Certificate,
      Accepted InitialSum.claimedInitial challenges
        (InitialSum.sumcheckPolynomial convention covers data coins challenges)
        certificate := by
  let polynomial := InitialSum.sumcheckPolynomial convention covers data coins
  have representable :
      FixedPhase.ExpectedRoundsRepresentable ops.toOps polynomial
        Degree.ncSumcheckDegreeBound challenges := by
    simpa [polynomial] using expectedRoundsRepresentable
      convention covers data coins challenges arity
  rcases FixedPhase.exists_honest_certificate ops.toOps polynomial
      Degree.ncSumcheckDegreeBound challenges representable with
    ⟨certificate, honest⟩
  have initialIsTrue :
      InitialSum.claimedInitial =
        FixedPhase.semanticInitial ops.toOps polynomial challenges.length := by
    rw [InitialSum.claimedInitial_eq_sumcheckHypercubeSum_of_truth
      convention covers data coins truth]
    unfold InitialSum.sumcheckHypercubeSum FixedPhase.semanticInitial
    rw [arity]
  refine ⟨certificate, ?_⟩
  unfold Accepted
  exact FixedPhase.complete ops.toOps polynomial InitialSum.claimedInitial
    challenges certificate initialIsTrue honest

/-- The executable checker is perfectly complete for the same independent
post-challenge algebraic path. It does not close transcript completeness. -/
theorem check_complete_of_truth
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (truth : Semantics.Nc.Truth data)
    (challenges : List K)
    (arity : challenges.length =
      domain.columnVariables + domain.laneVariables) :
    ∃ certificate : Certificate,
      check InitialSum.claimedInitial challenges
        (InitialSum.sumcheckPolynomial convention covers data coins challenges)
        certificate = true := by
  rcases complete_of_truth convention covers data coins truth challenges arity with
    ⟨certificate, accepted⟩
  exact ⟨certificate,
    (check_eq_true_iff_accepted InitialSum.claimedInitial challenges
      (InitialSum.sumcheckPolynomial convention covers data coins challenges)
      certificate).2 accepted⟩

/-- Paper-relative NC acceptance is sound up to the named selector, gamma, and
fixed-degree round-collision events. The theorem is deterministic: transcript
replay and probability bounds must later justify that these events are rare.

The proof does not assume the initial sum is zero. It first asks whether the
independently derived source mixture is zero. A zero mixture decomposes into
truth or a semantic compression root; a nonzero mixture makes the verifier's
literal zero claim false, so generic fixed-phase soundness exposes a round
collision. -/
theorem accepted_implies_truth_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (challenges : List K)
    (certificate : Certificate)
    (challengeSetSize : Nat)
    (arity : challenges.length =
      domain.columnVariables + domain.laneVariables)
    (accepted :
      Accepted InitialSum.claimedInitial challenges
        (InitialSum.sumcheckPolynomial .paperNc covers data coins challenges)
        certificate) :
    Semantics.Nc.Truth data ∨
      BadEvent covers data coins challenges certificate challengeSetSize := by
  by_cases mixtureZero :
      InitialSum.mixedResidualAtBeta .paperNc covers data coins = K.zero
  · rcases
      (MixingSoundness.paperNc_mixedResidualAtBeta_eq_zero_iff_truth_or_selectorRoot_or_gammaRoot
        noZeroDivisors covers data coins).mp mixtureZero with
      truth | selectorRoot | gammaRoot
    · exact Or.inl truth
    · exact Or.inr (.selectorRoot selectorRoot)
    · exact Or.inr (.gammaRoot gammaRoot)
  · apply Or.inr
    have semanticInitial_eq_mixture :
        FixedPhase.semanticInitial ops.toOps
            (InitialSum.sumcheckPolynomial .paperNc covers data coins)
            challenges.length =
          InitialSum.mixedResidualAtBeta .paperNc covers data coins := by
      unfold FixedPhase.semanticInitial
      rw [arity]
      change InitialSum.sumcheckHypercubeSum .paperNc covers data coins = _
      rw [InitialSum.sumcheckHypercubeSum_eq_hypercubeSum,
        InitialSum.hypercubeSum_eq_mixedResidualAtBeta]
    have falseClaim :
        InitialSum.claimedInitial ≠
          FixedPhase.semanticInitial ops.toOps
            (InitialSum.sumcheckPolynomial .paperNc covers data coins)
            challenges.length := by
      intro claimedEqualsSemantic
      apply mixtureZero
      calc
        InitialSum.mixedResidualAtBeta .paperNc covers data coins =
            FixedPhase.semanticInitial ops.toOps
              (InitialSum.sumcheckPolynomial .paperNc covers data coins)
              challenges.length := semanticInitial_eq_mixture.symm
        _ = InitialSum.claimedInitial := claimedEqualsSemantic.symm
        _ = K.zero := rfl
    rcases false_acceptance_implies_bad_challenge .paperNc covers data coins
        challengeSetSize InitialSum.claimedInitial
        (InitialSum.sumcheckPolynomial .paperNc covers data coins challenges)
        challenges certificate arity rfl accepted falseClaim with
      ⟨round, collision⟩
    exact .roundCollision round collision

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc
