import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

/-!
Opaque-core schedule obstruction for production Split-NC mixing.

Assurance tier: model-level obstruction.

Owns: an exact record update showing that the five core challenge values may
be replaced arbitrarily while preserving the state returned by `deriveCore`,
both explicitly ordered delayed squeezes, the state entering FE, and the
shared `betaA`/`gamma` projections.

Does not own: an internal order, finite support, distribution, or causal
visibility boundary for the five values returned atomically by `deriveCore`;
a claim about a future refined schedule; Fiat--Shamir; Rust/R1CS; or rows.

Emits constraints: no.

| Boundary | Owned equation | Excluded boundary |
|---|---|---|
| opaque core carrier | arbitrary values preserve the returned state | internal core sampling order |
| delayed schedule | producer beta then batch weight remain unchanged | Fiat--Shamir realization |
| FE/NC projections | betaA and gamma are shared | independence premise |
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

universe uState uVerifierKey uInput

variable
  {shape : SemanticShape}
  {State : Type uState}

/-- A point whose complete typed coordinate list is one repeated value. -/
def constantCubePoint
    (variables : Nat)
    (value : K) :
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CubePoint K variables where
  coordinates := List.replicate variables value
  dimension := by simp

/-- A maximally correlated core record: every scalar coordinate in every
core challenge is the same value. The current schedule carrier states neither
an independence law nor an internal sampling order for `deriveCore`. -/
def constantCoreChallenges
    {domains : Domains}
    (value : K) : CoreChallenges shape domains where
  alpha := constantCubePoint domains.laneVariables value
  betaA := constantCubePoint domains.laneVariables value
  betaR := constantCubePoint shape.rowVariables value
  gamma := value
  betaBlock := constantCubePoint domains.blockVariables value

/-- Replace the five opaque core challenge values while preserving the exact
state returned by `deriveCore` and every other schedule operation. -/
def replaceCoreChallenges
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {domains : Domains}
    (schedule : Schedule VerifierKey Input shape domains State)
    (replacement : State -> CoreChallenges shape domains) :
    Schedule VerifierKey Input shape domains State := {
  schedule with
  deriveCore := fun state =>
    let original := schedule.deriveCore state
    {
      challenges := replacement state
      state := original.state
    }
}

@[simp] theorem replaceCoreChallenges_deriveCore_challenges
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {domains : Domains}
    (schedule : Schedule VerifierKey Input shape domains State)
    (replacement : State -> CoreChallenges shape domains)
    (state : State) :
    ((replaceCoreChallenges schedule replacement).deriveCore state).challenges =
      replacement state := by
  rfl

@[simp] theorem replaceCoreChallenges_deriveCore_state
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {domains : Domains}
    (schedule : Schedule VerifierKey Input shape domains State)
    (replacement : State -> CoreChallenges shape domains)
    (state : State) :
    ((replaceCoreChallenges schedule replacement).deriveCore state).state =
      (schedule.deriveCore state).state := by
  rfl

/-- Two arbitrary replacement records can induce the same externally visible
`deriveCore` state. This obstructs deriving an internal challenge order or
sampling law from the current `Schedule` carrier alone. -/
theorem replaceCoreChallenges_same_state
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {domains : Domains}
    (schedule : Schedule VerifierKey Input shape domains State)
    (left right : CoreChallenges shape domains)
    (state : State) :
    ((replaceCoreChallenges schedule (fun _ => left)).deriveCore state).state =
      ((replaceCoreChallenges schedule (fun _ => right)).deriveCore state).state := by
  rfl

/-- The same-state replacement retains the supplied distinction between
challenge records; state equality does not pin their values. -/
theorem replaceCoreChallenges_different_challenges
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {domains : Domains}
    (schedule : Schedule VerifierKey Input shape domains State)
    (left right : CoreChallenges shape domains)
    (different : left ≠ right)
    (state : State) :
    ((replaceCoreChallenges schedule (fun _ => left)).deriveCore state).challenges ≠
      ((replaceCoreChallenges schedule (fun _ => right)).deriveCore state).challenges := by
  simpa only [replaceCoreChallenges_deriveCore_challenges] using different

/-- Replacing all five core values does not alter the first delayed challenge,
whose domain entry and squeeze remain after the opaque core state. -/
@[simp] theorem derivePreSumcheck_replaceCore_producerBeta
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {domains : Domains}
    (schedule : Schedule VerifierKey Input shape domains State)
    (replacement : State -> CoreChallenges shape domains)
    (priorState : State)
    (statement : Statement VerifierKey Input) :
    (derivePreSumcheck
      (replaceCoreChallenges schedule replacement) priorState statement).challenges.producerBeta =
      (derivePreSumcheck schedule priorState statement).challenges.producerBeta := by
  rfl

/-- The second delayed challenge remains after the first and is unchanged by
replacement of the opaque core values. -/
@[simp] theorem derivePreSumcheck_replaceCore_batchWeight
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {domains : Domains}
    (schedule : Schedule VerifierKey Input shape domains State)
    (replacement : State -> CoreChallenges shape domains)
    (priorState : State)
    (statement : Statement VerifierKey Input) :
    (derivePreSumcheck
      (replaceCoreChallenges schedule replacement) priorState statement).challenges.batchWeight =
      (derivePreSumcheck schedule priorState statement).challenges.batchWeight := by
  rfl

/-- The state entering FE is unchanged when only the opaque core challenge
values are replaced. -/
@[simp] theorem derivePreSumcheck_replaceCore_state
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {domains : Domains}
    (schedule : Schedule VerifierKey Input shape domains State)
    (replacement : State -> CoreChallenges shape domains)
    (priorState : State)
    (statement : Statement VerifierKey Input) :
    (derivePreSumcheck
      (replaceCoreChallenges schedule replacement) priorState statement).state =
      (derivePreSumcheck schedule priorState statement).state := by
  rfl

/-- Shared `betaA` remains a definitional projection in the replacement
countermodel; the obstruction is missing sampling/order evidence, not an
FE/NC fork. -/
@[simp] theorem replaceCore_shared_betaA
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {domains : Domains}
    (schedule : Schedule VerifierKey Input shape domains State)
    (replacement : State -> CoreChallenges shape domains)
    (priorState : State)
    (statement : Statement VerifierKey Input) :
    let challenges :=
      (derivePreSumcheck
        (replaceCoreChallenges schedule replacement) priorState statement).challenges
    challenges.ncCoins.betaA = challenges.feCoins.betaA := by
  rfl

/-- Shared `gamma` likewise remains one field of the common challenge record;
no independence premise is introduced. -/
@[simp] theorem replaceCore_shared_gamma
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {domains : Domains}
    (schedule : Schedule VerifierKey Input shape domains State)
    (replacement : State -> CoreChallenges shape domains)
    (priorState : State)
    (statement : Statement VerifierKey Input) :
    let challenges :=
      (derivePreSumcheck
        (replaceCoreChallenges schedule replacement) priorState statement).challenges
    challenges.ncCoins.gamma = challenges.feCoins.gamma := by
  rfl

/-- The opaque core interface admits complete FE/NC correlation while still
using one shared `gamma`. This rules out silently deriving independence from
the current carrier; it does not obstruct a refined ordered sampler. -/
theorem derivePreSumcheck_constantCore_shared_gamma
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {domains : Domains}
    (schedule : Schedule VerifierKey Input shape domains State)
    (value : K)
    (priorState : State)
    (statement : Statement VerifierKey Input) :
    let challenges :=
      (derivePreSumcheck
        (replaceCoreChallenges schedule
          (fun _ => constantCoreChallenges value))
        priorState statement).challenges
    challenges.feCoins.gamma = value ∧
      challenges.ncCoins.gamma = value ∧
      challenges.ncCoins.gamma = challenges.feCoins.gamma ∧
      challenges.feCoins.betaA.coordinates =
        List.replicate domains.laneVariables value ∧
      challenges.ncCoins.betaA.coordinates =
        List.replicate domains.laneVariables value := by
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary
