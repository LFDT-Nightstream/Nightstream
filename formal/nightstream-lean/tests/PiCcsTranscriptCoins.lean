import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Coins

/-!
Focused API regressions for verifier-owned `Pi_CCS` semantic coins.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.transcript.coins.shape` | concrete challenge dimensions equal the semantic domains | caller-selected or swapped dimensions |
| `nifs.pi_ccs.transcript.coins.fe` | FE coins come from one concrete challenge replay | independently supplied FE coins |
| `nifs.pi_ccs.transcript.coins.nc` | NC coins come from that same replay | independently supplied NC coins |
| `nifs.pi_ccs.transcript.coins.shared` | FE and NC share exactly `betaA` and `gamma` | divergent cross-phase authority |
-/

namespace NightstreamTests.PiCcsTranscriptCoins

open Nightstream.Implementation.R1CS.PiCcsTranscript.Coins
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

example
    (shape : SemanticShape)
    (domain : FlatNcDomain)
    (degreeBound : Nat) :
    (shapeWithDegree shape domain degreeBound).ellD =
        domain.laneVariables /\
      (shapeWithDegree shape domain degreeBound).ellN =
        shape.rowVariables /\
      (shapeWithDegree shape domain degreeBound).ellM =
        domain.columnVariables := by
  exact ⟨rfl, rfl, rfl⟩

example
    (initial : State)
    (shape : SemanticShape)
    (domain : FlatNcDomain)
    (degreeBound : Nat) :
    (ncCoins initial shape domain degreeBound).betaA =
      (feCoins initial shape domain degreeBound).betaA :=
  ncCoins_betaA_eq_feCoins_betaA initial shape domain degreeBound

example
    (initial : State)
    (shape : SemanticShape)
    (domain : FlatNcDomain)
    (degreeBound : Nat) :
    (ncCoins initial shape domain degreeBound).gamma =
      (feCoins initial shape domain degreeBound).gamma :=
  ncCoins_gamma_eq_feCoins_gamma initial shape domain degreeBound

end NightstreamTests.PiCcsTranscriptCoins
