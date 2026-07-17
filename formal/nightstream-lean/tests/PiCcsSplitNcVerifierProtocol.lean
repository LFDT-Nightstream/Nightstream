import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol

/-!
Focused regressions for protocol-level Split-NC `Pi_CCS` composition.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.semantic.split_exact` | split truth is exactly fresh CCS, all-source norm, and carried evaluation truth | semantic reorder or omission |
| `nifs.pi_ccs.verify.chain` | executable and logical FE/NC acceptance coincide | phase-composition drift |
| `nifs.pi_ccs.verify.output_authority` | row binding is exact whole-function equality at the FE point | self-consistent raw output promoted to authority |
| `nifs.pi_ccs.verify.soundness` | acceptance reduces to paper obligations, failed output binding, or named phase events | unclassified production false acceptance |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.Tests

open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

example
    {shape : SemanticShape}
    (data : Data shape) :
    Semantics.Truth data ↔ Semantics.Paper.Holds data :=
  Semantics.truth_iff_paperHolds data

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type}
    (feMachine : Transcript.Fe.Machine State)
    (ncMachine : Transcript.Nc.Machine State)
    (initialState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (input : PublicInput shape)
    (feCoins : Polynomial.Fe.Coins shape domain)
    (ncCoins : Polynomial.Nc.Mixing.Coins domain)
    (certificate : Certificate input domain) :
    check feMachine ncMachine initialState profile input feCoins ncCoins
        certificate = true ↔
      Accepted feMachine ncMachine initialState profile input feCoins ncCoins
        certificate :=
  check_eq_true_iff_accepted feMachine ncMachine initialState profile input
    feCoins ncCoins certificate

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type}
    (data : Data shape)
    (execution : Execution shape domain State)
    (message : OutputMessage shape) :
    YRingBoundToSources data execution.outputPoints message ↔
      message.yRing =
        Polynomial.Fe.sourceYRingAt data execution.fePoint.row :=
  yRingBoundToSources_iff_yRing_eq data execution message

example
    (noZeroDivisors :
      PaperJoint.NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type}
    (covers : domain.Covers shape)
    (feMachine : Transcript.Fe.Machine State)
    (ncMachine : Transcript.Nc.Machine State)
    (initialState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape)
    (feCoins : Polynomial.Fe.Coins shape domain)
    (ncCoins : Polynomial.Nc.Mixing.Coins domain)
    (certificate :
      Certificate (PublicInput.ofSources data) domain)
    (challengeSetSize : Nat)
    (accepted :
      Accepted feMachine ncMachine initialState profile
        (PublicInput.ofSources data) feCoins ncCoins certificate) :
    let execution :=
      derive feMachine ncMachine initialState certificate
    Semantics.Paper.Holds data ∨
      ¬ BoundToSources covers data execution.outputPoints certificate.output ∨
      BadEvent profile covers data feCoins ncCoins execution certificate
        challengeSetSize :=
  accepted_implies_paperObligations_or_unbound_or_badEvent
    noZeroDivisors covers feMachine ncMachine initialState profile data feCoins
    ncCoins certificate challengeSetSize accepted

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.Tests
