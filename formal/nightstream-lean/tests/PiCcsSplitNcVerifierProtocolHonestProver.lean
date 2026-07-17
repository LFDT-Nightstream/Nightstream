import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.HonestProver

/-!
Focused theorem regression for honest sequential protocol composition.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.prover.handoff` | NC starts from the constructed FE final state | disconnected phase transcripts |
| `nifs.pi_ccs.prover.output` | output claims use the same derived FE/NC points | caller-selected output points |
| `nifs.pi_ccs.prover.completeness` | paper obligations produce accepted, source-bound payload | soundness-only verifier model |
-/

namespace NightstreamTests.PiCcsSplitNcVerifierProtocolHonestProver

open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.HonestProver

example
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
    (obligations : Semantics.Paper.Holds data) :
    ∃ certificate :
        Protocol.Certificate (PublicInput.ofSources data) domain,
      Protocol.Accepted feMachine ncMachine initialState profile
          (PublicInput.ofSources data) feCoins ncCoins certificate ∧
        BoundToSources covers data
          (Protocol.derive feMachine ncMachine initialState certificate).outputPoints
          certificate.output :=
  complete_of_paperObligations covers feMachine ncMachine initialState
    profile data feCoins ncCoins obligations

end NightstreamTests.PiCcsSplitNcVerifierProtocolHonestProver
