import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe

/-!
Focused regressions for the canonical verifier-visible Split-NC FE phase.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.fe.verify.chain` | executable and logical phase acceptance coincide | checker/specification drift |
| `nifs.pi_ccs.fe.verify.completeness` | truth plus source-bound honest messages is accepted | hidden terminal mismatch |
| `nifs.pi_ccs.fe.verify.soundness` | acceptance reduces to truth, output mismatch, or named bad event | unclassified false acceptance |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe.Tests

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type}
    (machine : Transcript.Fe.Machine State)
    (initialState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (input : PublicInput shape)
    (coins : Polynomial.Fe.Coins shape domain)
    (message : OutputMessage shape)
    (certificate : SumCheck.Fe.Certificate input domain) :
    check machine initialState profile input coins message certificate = true ↔
      Accepted machine initialState profile input coins message certificate :=
  check_eq_true_iff_accepted machine initialState profile input coins message
    certificate

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type}
    (machine : Transcript.Fe.Machine State)
    (initialState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape)
    (coins : Polynomial.Fe.Coins shape domain)
    (message : OutputMessage shape)
    (certificate :
      SumCheck.Fe.Certificate (PublicInput.ofSources data) domain)
    (truth : Semantics.Fe.Truth data)
    (messageBound :
      message.yRing =
        Polynomial.Fe.sourceYRingAt data
          (Transcript.Fe.derive machine initialState certificate).challengePoint.row)
    (honest :
      SumCheck.Fe.HonestAt
        (Polynomial.Fe.InitialSum.sumcheckPolynomial profile data coins)
        (Transcript.Fe.derive machine initialState certificate).challengePoint
        certificate) :
    Accepted machine initialState profile (PublicInput.ofSources data) coins
      message certificate :=
  accepted_of_truth_and_honestAt machine initialState profile data coins
    message certificate truth messageBound honest

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type}
    (machine : Transcript.Fe.Machine State)
    (initialState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape)
    (coins : Polynomial.Fe.Coins shape domain)
    (message : OutputMessage shape)
    (certificate :
      SumCheck.Fe.Certificate (PublicInput.ofSources data) domain)
    (challengeSetSize : Nat)
    (accepted :
      Accepted machine initialState profile (PublicInput.ofSources data) coins
        message certificate) :
    Semantics.Fe.Truth data ∨
      Polynomial.Fe.OutputMismatch data
        (Transcript.Fe.derive machine initialState certificate).challengePoint
        message ∨
      SumCheck.Fe.BadEvent profile data coins
        (Transcript.Fe.derive machine initialState certificate).challengePoint
        certificate challengeSetSize :=
  accepted_implies_truth_or_mismatch_or_badEvent machine initialState profile
    data coins message certificate challengeSetSize accepted

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe.Tests
