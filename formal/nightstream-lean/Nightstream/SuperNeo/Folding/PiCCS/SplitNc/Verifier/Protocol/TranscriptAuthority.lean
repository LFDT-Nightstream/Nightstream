import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Types

/-!
Transcript authority for the legacy flat-column Split-NC `Pi_CCS` path.

Assurance tier: model-level diagnostic path.

Protocol: SuperNeo `Pi_CCS`.
Phase: statement binding, pre-SumCheck sampling, FE/NC replay, and output
handoff.
Constraint family: typed verifier dataflow only; this file emits no rows.

Owns: one typed public statement; one shared pre-SumCheck challenge record;
deterministic statement binding and sampling; construction of the FE and NC
transcript machines from one schedule; and the post-`Pi_CCS` output handoff.

Does not own: the canonical block×lane NC path, a concrete encoding,
Poseidon2, collision or random-oracle security, polynomial truth, SumCheck
acceptance, honest prover construction, `Pi_RLC`, Rust, R1CS, rows, costs,
necessity, or row removal.

Emits constraints: no.

Authority boundary: FE and NC do not receive independent coin records.
`betaA` and `gamma` occur once in `Challenges` and both phase projections read
that same value. The state entering FE is derived by binding the complete typed
statement and then executing the configured pre-SumCheck sampler. Abstract
schedule functions may still collide or ignore inputs; concrete transcript
refinement and security must name and bound those events. `betaM` and this
module's flat NC machine are retained only for the legacy 15-round diagnostic
path; they are not authority for canonical block×lane NC.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.transcript.statement` | verifier key and exact public input product form one typed statement | verifier-owned input | `Statement` |
| `nifs.pi_ccs.transcript.bind` | prior state and complete statement determine the pre-challenge state | computed | `Schedule.bindStatement`, `derivePreSumcheck` |
| `nifs.pi_ccs.transcript.coins.shared` | `alpha`, `betaA`, `betaR`, `gamma`, and `betaM` occur once | computed | `Challenges` |
| `nifs.pi_ccs.transcript.coins.fe` | FE coins are a projection of the shared record | direct dataflow | `Challenges.feCoins` |
| `nifs.pi_ccs.transcript.coins.nc` | NC coins are a projection of the same shared record | direct dataflow | `Challenges.ncCoins` |
| `nifs.pi_ccs.transcript.fe` | the verifier-owned initial claim parameterizes the sole FE entry | computed | `feMachine` |
| `nifs.pi_ccs.transcript.nc` | NC entry and rounds use the same chained schedule | computed | `ncMachine` |
| `nifs.pi_ccs.transcript.output` | final FE/NC state and complete output message determine the handoff | computed | `Schedule.absorbOutput` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uVerifierKey uInput uState

/-- Complete public statement bound before any `Pi_CCS` challenge is sampled.
The concrete encoding of either field remains a later refinement. -/
structure Statement
    (VerifierKey : Type uVerifierKey)
    (Input : Type uInput) where
  verifierKey : VerifierKey
  input : Input

/-- The unique pre-SumCheck verifier challenge record.

The shared `betaA` and `gamma` fields are intentionally not duplicated into
separate FE and NC records. -/
structure Challenges
    (shape : SemanticShape)
    (domain : FlatNcDomain) where
  alpha : CubePoint K domain.laneVariables
  betaA : CubePoint K domain.laneVariables
  betaR : CubePoint K shape.rowVariables
  gamma : K
  betaM : CubePoint K domain.columnVariables

namespace Challenges

/-- Exact FE view of the one shared pre-SumCheck record. -/
def feCoins
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (challenges : Challenges shape domain) :
    Polynomial.Fe.Coins shape domain where
  alpha := challenges.alpha
  betaA := challenges.betaA
  betaR := challenges.betaR
  gamma := challenges.gamma

/-- Exact NC view of the same shared pre-SumCheck record. -/
def ncCoins
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (challenges : Challenges shape domain) :
    Polynomial.Nc.Mixing.Coins domain where
  betaM := challenges.betaM
  betaA := challenges.betaA
  gamma := challenges.gamma

/-- FE and NC cannot acquire different lane/Ajtai challenges. -/
@[simp] theorem ncCoins_betaA_eq_feCoins_betaA
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (challenges : Challenges shape domain) :
    challenges.ncCoins.betaA = challenges.feCoins.betaA := by
  rfl

/-- FE and NC cannot acquire different source-mixing challenges. -/
@[simp] theorem ncCoins_gamma_eq_feCoins_gamma
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (challenges : Challenges shape domain) :
    challenges.ncCoins.gamma = challenges.feCoins.gamma := by
  rfl

end Challenges

/-- Pre-SumCheck challenge output and the sole state entering FE. -/
structure PreSumcheck
    (shape : SemanticShape)
    (domain : FlatNcDomain)
    (State : Type uState) where
  challenges : Challenges shape domain
  state : State

/-- Abstract deterministic transcript schedule.

Every operation is verifier-owned and receives all values it is responsible
for binding. A later concrete refinement must instantiate these operations
with the exact Poseidon2 encoding and prove the corresponding security
properties. -/
structure Schedule
    (VerifierKey : Type uVerifierKey)
    (Input : Type uInput)
    (shape : SemanticShape)
    (domain : FlatNcDomain)
    (State : Type uState) where
  bindStatement : State -> Statement VerifierKey Input -> State
  derivePreSumcheck : State -> PreSumcheck shape domain State
  enterFe : State -> K -> State
  absorbFeRound :
    State -> Nightstream.SuperNeo.SumCheck.Finite.Message K -> State
  squeezeFeChallenge : State -> K × State
  enterNc : State -> State
  absorbNcRound : State -> Transcript.Nc.RoundMessage -> State
  squeezeNcChallenge : State -> K × State
  absorbOutput : State -> OutputMessage shape -> State

/-- Bind the complete typed statement, then derive the one shared challenge
record and FE-entry state. -/
def derivePreSumcheck
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (schedule : Schedule VerifierKey Input shape domain State)
    (priorState : State)
    (statement : Statement VerifierKey Input) :
    PreSumcheck shape domain State :=
  schedule.derivePreSumcheck (schedule.bindStatement priorState statement)

/-- FE machine whose only phase-entry parameter is the verifier-computed
initial claim. -/
def feMachine
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (schedule : Schedule VerifierKey Input shape domain State)
    (initialClaim : K) :
    Transcript.Fe.Machine State where
  enterFe state := schedule.enterFe state initialClaim
  absorbRound := schedule.absorbFeRound
  squeezeChallenge := schedule.squeezeFeChallenge

/-- NC machine projected from the same chained schedule. -/
def ncMachine
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (schedule : Schedule VerifierKey Input shape domain State) :
    Transcript.Nc.Machine State where
  enterNc := schedule.enterNc
  absorbRound := schedule.absorbNcRound
  squeezeChallenge := schedule.squeezeNcChallenge

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority
