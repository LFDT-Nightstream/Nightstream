import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Mixing
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.BlockLane
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Types

/-!
Transcript authority for the canonical Split-NC FE plus block×lane NC path.

Assurance tier: model-level.

Owns: one typed public statement; one domain record whose FE and NC views
share a single lane width; one core challenge record; structurally ordered
producer-projection and residual-weight sampling; deterministic statement
binding; FE and NC machine projections; and the raw output-message handoff.

Does not own: the legacy flat-column NC verifier, a concrete encoding,
Poseidon2, collision or random-oracle security, polynomial truth, SumCheck
acceptance, output binding, Rust, R1CS, costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: `Domains` makes divergent FE/NC lane dimensions
unrepresentable. `betaA` and `gamma` occur once in `Challenges`; both phase
projections read those same fields. The schedule binds the complete statement,
derives the core coins, enters and samples the typed `.producerBeta` domain,
then enters and samples the distinct `.batchWeight` domain, before FE starts.
A later Poseidon2 refinement must still prove the exact encoding of these
kernel-distinct tags.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.transcript.block_lane.domain` | FE and NC derive from one lane width | computed by construction | `Domains`, `Domains.fe`, `Domains.nc` |
| `nifs.pi_ccs.transcript.block_lane.statement` | key and complete public input form one typed statement | verifier-owned input | `Statement` |
| `nifs.pi_ccs.transcript.block_lane.bind` | statement binding precedes all challenge sampling | computed | `Schedule.bindStatement`, `derivePreSumcheck` |
| `nifs.pi_ccs.transcript.block_lane.coins.core` | `alpha`, `betaA`, `betaR`, `gamma`, and `betaBlock` occur once | computed | `CoreChallenges` |
| `nifs.pi_ccs.transcript.block_lane.coins.domain` | producer and residual roles are distinct typed transcript domains | typed | `DelayedChallengeDomain` |
| `nifs.pi_ccs.transcript.block_lane.coins.producer` | sample `producerBeta` only after statement binding and core derivation | computed/security boundary | `Schedule.enterDelayedDomain`, `Schedule.squeezeDelayedChallenge`, `derivePreSumcheck` |
| `nifs.pi_ccs.transcript.block_lane.coins.residual` | sample `batchWeight` in the typed residual domain after `producerBeta` | computed/security boundary | `Schedule.enterDelayedDomain`, `Schedule.squeezeDelayedChallenge`, `derivePreSumcheck` |
| `nifs.pi_ccs.transcript.block_lane.coins.fe` | FE coins are a projection of the shared record | direct dataflow | `Challenges.feCoins` |
| `nifs.pi_ccs.transcript.block_lane.coins.nc` | block×lane NC coins are a projection of the same record | direct dataflow | `Challenges.ncCoins` |
| `nifs.pi_ccs.transcript.block_lane.fe` | verifier initial claim parameterizes the sole FE entry | computed | `feMachine` |
| `nifs.pi_ccs.transcript.block_lane.nc` | NC entry and rounds use the same chained schedule | computed | `ncMachine` |
| `nifs.pi_ccs.transcript.block_lane.output` | final state and complete raw output determine the handoff | computed | `Schedule.absorbOutput` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uVerifierKey uInput uState

/-- The two arithmetization views used by canonical Split-NC.

The flat column width remains only because the current FE type is
parameterized by `FlatNcDomain`; canonical NC consumes `blockVariables`
instead. Both views are constructed from the sole `laneVariables` field. -/
structure Domains where
  columnVariables : Nat
  blockVariables : Nat
  laneVariables : Nat
deriving Repr, DecidableEq

namespace Domains

/-- FE view of the shared dimensions. Its column axis is not consumed by FE. -/
def fe (domains : Domains) : FlatNcDomain where
  columnVariables := domains.columnVariables
  laneVariables := domains.laneVariables

/-- Canonical block×lane NC view of the shared dimensions. -/
def nc (domains : Domains) : BlockNcDomain where
  blockVariables := domains.blockVariables
  laneVariables := domains.laneVariables

@[simp] theorem fe_laneVariables (domains : Domains) :
    domains.fe.laneVariables = domains.laneVariables := by
  rfl

@[simp] theorem nc_laneVariables (domains : Domains) :
    domains.nc.laneVariables = domains.laneVariables := by
  rfl

end Domains

/-- Complete public statement bound before any `Pi_CCS` challenge is sampled.
The concrete encoding of either field remains a later refinement. -/
structure Statement
    (VerifierKey : Type uVerifierKey)
    (Input : Type uInput) where
  verifierKey : VerifierKey
  input : Input

/-- Typed domains for the two delayed-projection challenges. These roles are
kernel-distinct before a concrete transcript chooses their Poseidon2 tags. -/
inductive DelayedChallengeDomain where
  | producerBeta
  | batchWeight
deriving Repr, DecidableEq

@[simp] theorem DelayedChallengeDomain.producerBeta_ne_batchWeight :
    DelayedChallengeDomain.producerBeta ≠
      DelayedChallengeDomain.batchWeight := by
  decide

/-- Core FE plus block×lane NC challenges derived after statement binding and
before the two delayed-projection challenges. -/
structure CoreChallenges
    (shape : SemanticShape)
    (domains : Domains) where
  alpha : CubePoint K domains.laneVariables
  betaA : CubePoint K domains.laneVariables
  betaR : CubePoint K shape.rowVariables
  gamma : K
  betaBlock : CubePoint K domains.blockVariables

/-- The unique pre-SumCheck challenge record. The two scalar fields are not
part of the opaque core derivation: `derivePreSumcheck` samples them in the
typed order `producerBeta`, then `batchWeight`, using distinct domain-entry
operations. -/
structure Challenges
    (shape : SemanticShape)
    (domains : Domains)
    extends CoreChallenges shape domains where
  producerBeta : K
  batchWeight : K

namespace Challenges

/-- Exact FE view of the shared pre-SumCheck challenge record. -/
def feCoins
    {shape : SemanticShape}
    {domains : Domains}
    (challenges : Challenges shape domains) :
    Polynomial.Fe.Coins shape domains.fe where
  alpha := challenges.alpha
  betaA := challenges.betaA
  betaR := challenges.betaR
  gamma := challenges.gamma

/-- Exact canonical NC view of the same challenge record. -/
def ncCoins
    {shape : SemanticShape}
    {domains : Domains}
    (challenges : Challenges shape domains) :
    Polynomial.Nc.BlockLane.Mixing.Coins domains.nc where
  betaBlock := challenges.betaBlock
  betaA := challenges.betaA
  gamma := challenges.gamma

/-- FE and canonical NC cannot acquire different lane challenges. -/
@[simp] theorem ncCoins_betaA_eq_feCoins_betaA
    {shape : SemanticShape}
    {domains : Domains}
    (challenges : Challenges shape domains) :
    challenges.ncCoins.betaA = challenges.feCoins.betaA := by
  rfl

/-- FE and canonical NC cannot acquire different source-mixing challenges. -/
@[simp] theorem ncCoins_gamma_eq_feCoins_gamma
    {shape : SemanticShape}
    {domains : Domains}
    (challenges : Challenges shape domains) :
    challenges.ncCoins.gamma = challenges.feCoins.gamma := by
  rfl

end Challenges

/-- Core challenge output and the state immediately preceding the two
delayed-projection sampling domains. -/
structure CorePreSumcheck
    (shape : SemanticShape)
    (domains : Domains)
    (State : Type uState) where
  challenges : CoreChallenges shape domains
  state : State

/-- Complete pre-SumCheck challenge output and the sole state entering FE. -/
structure PreSumcheck
    (shape : SemanticShape)
    (domains : Domains)
    (State : Type uState) where
  challenges : Challenges shape domains
  state : State

/-- Abstract deterministic transcript schedule for canonical FE plus
block×lane NC.

A concrete refinement must instantiate every operation with the exact
Poseidon2 encoding and prove its transcript/security obligations. -/
structure Schedule
    (VerifierKey : Type uVerifierKey)
    (Input : Type uInput)
    (shape : SemanticShape)
  (domains : Domains)
  (State : Type uState) where
  bindStatement : State -> Statement VerifierKey Input -> State
  deriveCore : State -> CorePreSumcheck shape domains State
  enterDelayedDomain : DelayedChallengeDomain -> State -> State
  squeezeDelayedChallenge : State -> K × State
  enterFe : State -> K -> State
  absorbFeRound :
    State -> Nightstream.SuperNeo.SumCheck.Finite.Message K -> State
  squeezeFeChallenge : State -> K × State
  enterNc : State -> State
  absorbNcRound : State -> Transcript.Nc.RoundMessage -> State
  squeezeNcChallenge : State -> K × State
  absorbOutput : State -> OutputMessage shape -> State

/-- Bind the complete typed statement before deriving the shared challenge
record and FE-entry state. -/
def derivePreSumcheck
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    (schedule : Schedule VerifierKey Input shape domains State)
    (priorState : State)
    (statement : Statement VerifierKey Input) :
    PreSumcheck shape domains State :=
  let boundState := schedule.bindStatement priorState statement
  let core := schedule.deriveCore boundState
  let producerStep :=
    schedule.squeezeDelayedChallenge
      (schedule.enterDelayedDomain .producerBeta core.state)
  let batchStep :=
    schedule.squeezeDelayedChallenge
      (schedule.enterDelayedDomain .batchWeight producerStep.2)
  {
    challenges := {
      alpha := core.challenges.alpha
      betaA := core.challenges.betaA
      betaR := core.challenges.betaR
      gamma := core.challenges.gamma
      betaBlock := core.challenges.betaBlock
      producerBeta := producerStep.1
      batchWeight := batchStep.1
    }
    state := batchStep.2
  }

/-- The producer challenge is exactly the first delayed scalar sampled after
the bound statement's core challenge derivation. -/
@[simp] theorem derivePreSumcheck_producerBeta
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    (schedule : Schedule VerifierKey Input shape domains State)
    (priorState : State)
    (statement : Statement VerifierKey Input) :
    (derivePreSumcheck schedule priorState statement).challenges.producerBeta =
      (schedule.squeezeDelayedChallenge
        (schedule.enterDelayedDomain .producerBeta
          (schedule.deriveCore
            (schedule.bindStatement priorState statement)).state)).1 := by
  rfl

/-- The residual weight is sampled in its own entry domain from the state
returned by the producer challenge squeeze. -/
@[simp] theorem derivePreSumcheck_batchWeight
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    (schedule : Schedule VerifierKey Input shape domains State)
    (priorState : State)
    (statement : Statement VerifierKey Input) :
    (derivePreSumcheck schedule priorState statement).challenges.batchWeight =
      (schedule.squeezeDelayedChallenge
        (schedule.enterDelayedDomain .batchWeight
          (schedule.squeezeDelayedChallenge
            (schedule.enterDelayedDomain .producerBeta
              (schedule.deriveCore
                (schedule.bindStatement priorState statement)).state)).2)).1 := by
  rfl

/-- FE machine whose sole phase-entry parameter is the verifier-computed
initial claim. -/
def feMachine
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    (schedule : Schedule VerifierKey Input shape domains State)
    (initialClaim : K) :
    Transcript.Fe.Machine State where
  enterFe state := schedule.enterFe state initialClaim
  absorbRound := schedule.absorbFeRound
  squeezeChallenge := schedule.squeezeFeChallenge

/-- Canonical NC machine projected from the same chained schedule. -/
def ncMachine
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    (schedule : Schedule VerifierKey Input shape domains State) :
    Transcript.Nc.Machine State where
  enterNc := schedule.enterNc
  absorbRound := schedule.absorbNcRound
  squeezeChallenge := schedule.squeezeNcChallenge

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane
