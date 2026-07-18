import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc

/-!
Focused regressions for exact-width Split-NC SumCheck replay.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.sumcheck.message.decode` | only width five parses | empty, short, or overlong acceptance |
| `nifs.pi_ccs.nc.sumcheck.chain` | NC adapter and generic fixed-phase replay coincide | duplicate checker/spec drift |
| `nifs.pi_ccs.nc.sumcheck.degree` | quartic NC rounds discharge generic representability | degree/phase disconnect |
| `nifs.pi_ccs.nc.sumcheck.completeness` | honest NC truth has a five-slot certificate | semantic/computational drift |
| `nifs.pi_ccs.nc.sumcheck.soundness` | accepted paper-NC is truth or one named bad event | omitted semantic failure mode |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.Tests

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing

private def rawWithWidth (width : Nat) :
    Nightstream.SuperNeo.SumCheck.Finite.Message K where
  coefficients := List.replicate width K.zero

private abbrev ops := ConcreteCarrier.extensionOps

private def zeroRound : RoundMessage :=
  Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial.zero
    ops.toOps Degree.ncSumcheckDegreeBound

example : (RoundMessage.decode (rawWithWidth 0)).isSome = false := by
  simp [RoundMessage.decode, rawWithWidth, Degree.ncMessageWidth,
    Degree.ncSumcheckDegreeBound]

example : (RoundMessage.decode (rawWithWidth 1)).isSome = false := by
  simp [RoundMessage.decode, rawWithWidth, Degree.ncMessageWidth,
    Degree.ncSumcheckDegreeBound]

example : (RoundMessage.decode (rawWithWidth 4)).isSome = false := by
  simp [RoundMessage.decode, rawWithWidth, Degree.ncMessageWidth,
    Degree.ncSumcheckDegreeBound]

example : (RoundMessage.decode (rawWithWidth 5)).isSome = true := by
  simp [RoundMessage.decode, rawWithWidth, Degree.ncMessageWidth,
    Degree.ncSumcheckDegreeBound]

example : (RoundMessage.decode (rawWithWidth 6)).isSome = false := by
  simp [RoundMessage.decode, rawWithWidth, Degree.ncMessageWidth,
    Degree.ncSumcheckDegreeBound]

example : zeroRound.toRaw.coefficients.length = 5 ∧
    zeroRound.toRaw.degreeUpperBound = 4 := by
  simp

/-- Fixed-width padding is intentional: the all-zero five-slot polynomial is
valid for NC even though the older variable-width canonical language rejects
its redundant trailing zeros. -/
example :
    ¬ zeroRound.toRaw.Canonical ops.toOps := by
  simp [zeroRound, RoundMessage.toRaw,
    Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial.zero,
    Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial.toMessage,
    Nightstream.SuperNeo.SumCheck.Finite.Message.Canonical,
    Degree.ncSumcheckDegreeBound]

example
    (initial terminal : K)
    (challenges : List K)
    (certificate : Certificate) :
    check initial challenges terminal certificate = true ↔
      Accepted initial challenges terminal certificate :=
  check_eq_true_iff_accepted initial challenges terminal certificate

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (challenges : List K)
    (arity : challenges.length =
      domain.columnVariables + domain.laneVariables) :
    Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.ExpectedRoundsRepresentable
      ops.toOps
      (InitialSum.sumcheckPolynomial convention covers data coins)
      Degree.ncSumcheckDegreeBound challenges :=
  expectedRoundsRepresentable convention covers data coins challenges arity

example
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
        certificate = true :=
  check_complete_of_truth convention covers data coins truth challenges arity

example
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
      BadEvent covers data coins challenges certificate challengeSetSize :=
  accepted_implies_truth_or_badEvent noZeroDivisors covers data coins
    challenges certificate challengeSetSize arity accepted

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.Tests
