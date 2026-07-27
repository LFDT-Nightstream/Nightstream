import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.CausalSoundness
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.ChallengeSupport
import Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Product

/-!
Selected ideal-interactive challenge carrier for production Split-NC.

Assurance tier: model-level protocol refinement.

Owns: one explicit nonempty finite support; the production challenge order
`(alpha, betaA, betaR, gamma)`, then `betaBlock`, `producerBeta`,
`batchWeight`, then FE and NC round words; a schedule interpreter whose
messages are absorbed before the corresponding round challenge is revealed;
and a constructor into the existing fixed-active production context with
`challengeSetSize` derived from the sampled alphabet.

Does not own: a claim about arbitrary opaque `Schedule.deriveCore` values,
Fiat--Shamir, Poseidon2, a bounded concrete sampler, closed Goldilocks
arithmetic certificates, Rust/R1CS, artifacts, costs, or rows.

Emits constraints: no.

The obstruction in `CoreSchedule` remains valid.  Positive results in this
module apply only to contexts constructed by `input`; they do not infer a
sampling law for an arbitrary existing schedule.

| Stage | Owned data | Visibility |
|---|---|---|
| engine batch | `alpha`, `betaA`, `betaR`, shared `gamma` | before FE |
| block suffix | `betaBlock` | before FE and NC |
| delayed suffix | `producerBeta`, then `batchWeight` | before FE and NC |
| FE | one challenge after each absorbed FE message | prior FE prefix only |
| NC | one challenge after each absorbed NC message | complete FE word plus prior NC prefix |
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveCarrier

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

universe uState

/-! ## Ordered finite seed -/

/-- One exact engine-batch alpha word. -/
abbrev AlphaWord :=
  Fin PiCcsDomains.production.laneVariables -> K

/-- One exact engine-batch lane-selector word. -/
abbrev BetaAWord :=
  Fin PiCcsDomains.production.laneVariables -> K

/-- One exact engine-batch row-selector word. -/
abbrev BetaRWord (shape : SemanticShape) :=
  Fin shape.rowVariables -> K

/-- The production engine batch in serialized order:
`((alpha, betaA), betaR), gamma`. -/
abbrev EngineSeed (shape : SemanticShape) :=
  ((AlphaWord × BetaAWord) × BetaRWord shape) × K

/-- One exact block-selector word sampled after the engine batch. -/
abbrev BetaBlockWord :=
  Fin PiCcsDomains.production.blockVariables -> K

/-- The post-engine suffix in serialized order:
`(betaBlock, producerBeta), batchWeight`. -/
abbrev DelayedSeed :=
  (BetaBlockWord × K) × K

/-- All pre-SumCheck verifier challenges in exact transcript order. -/
abbrev PreSeed (shape : SemanticShape) :=
  EngineSeed shape × DelayedSeed

/-- Exact physical FE challenge word. -/
abbrev FeWord (shape : SemanticShape) :=
  Fin
      (Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness.feRoundCount
        shape PiCcsDomains.production.fe) ->
    K

/-- Exact physical block-plus-lane NC challenge word. -/
abbrev NcWord :=
  Fin CausalSoundness.ncRoundCount -> K

/-- FE completes before the fresh NC word is consumed. -/
abbrev SumCheckSeed (shape : SemanticShape) :=
  FeWord shape × NcWord

/-- Complete ideal-interactive production seed. -/
abbrev Seed (shape : SemanticShape) :=
  PreSeed shape × SumCheckSeed shape

namespace PreSeed

def alpha {shape : SemanticShape} (seed : PreSeed shape) : AlphaWord :=
  seed.1.1.1.1

def betaA {shape : SemanticShape} (seed : PreSeed shape) : BetaAWord :=
  seed.1.1.1.2

def betaR {shape : SemanticShape} (seed : PreSeed shape) : BetaRWord shape :=
  seed.1.1.2

def gamma {shape : SemanticShape} (seed : PreSeed shape) : K :=
  seed.1.2

def betaBlock {shape : SemanticShape} (seed : PreSeed shape) : BetaBlockWord :=
  seed.2.1.1

def producerBeta {shape : SemanticShape} (seed : PreSeed shape) : K :=
  seed.2.1.2

def batchWeight {shape : SemanticShape} (seed : PreSeed shape) : K :=
  seed.2.2

end PreSeed

namespace Seed

def alpha {shape : SemanticShape} (seed : Seed shape) : AlphaWord :=
  seed.1.alpha

def betaA {shape : SemanticShape} (seed : Seed shape) : BetaAWord :=
  seed.1.betaA

def betaR {shape : SemanticShape} (seed : Seed shape) : BetaRWord shape :=
  seed.1.betaR

def gamma {shape : SemanticShape} (seed : Seed shape) : K :=
  seed.1.gamma

def betaBlock {shape : SemanticShape} (seed : Seed shape) : BetaBlockWord :=
  seed.1.betaBlock

def producerBeta {shape : SemanticShape} (seed : Seed shape) : K :=
  seed.1.producerBeta

def batchWeight {shape : SemanticShape} (seed : Seed shape) : K :=
  seed.1.batchWeight

def feWord {shape : SemanticShape} (seed : Seed shape) : FeWord shape :=
  seed.2.1

def ncWord {shape : SemanticShape} (seed : Seed shape) : NcWord :=
  seed.2.2

end Seed

/-- Convert a finite challenge word to the paper-owned typed cube point
without changing coordinate order. -/
def cubePoint
    {variables : Nat}
    (word : Fin variables -> K) :
    CubePoint K variables where
  coordinates := List.ofFn word
  dimension := by simp

/-- Erase the ordered pre-SumCheck seed into the existing single shared core
record.  `gamma` occurs once; FE and NC later project that same field. -/
def coreChallenges
    {shape : SemanticShape}
    (seed : Seed shape) :
    CoreChallenges shape PiCcsDomains.production where
  alpha := cubePoint seed.alpha
  betaA := cubePoint seed.betaA
  betaR := cubePoint seed.betaR
  gamma := seed.gamma
  betaBlock := cubePoint seed.betaBlock

/-- Complete pre-SumCheck record obtained after the two delayed scalar
domains. -/
def challenges
    {shape : SemanticShape}
    (seed : Seed shape) :
    Challenges shape PiCcsDomains.production where
  alpha := cubePoint seed.alpha
  betaA := cubePoint seed.betaA
  betaR := cubePoint seed.betaR
  gamma := seed.gamma
  betaBlock := cubePoint seed.betaBlock
  producerBeta := seed.producerBeta
  batchWeight := seed.batchWeight

@[simp] theorem coreChallenges_shared_gamma
    {shape : SemanticShape}
    (seed : Seed shape) :
    (coreChallenges seed).gamma = seed.gamma := by
  rfl

@[simp] theorem coreChallenges_shared_betaA
    {shape : SemanticShape}
    (seed : Seed shape) :
    (coreChallenges seed).betaA.coordinates = List.ofFn seed.betaA := by
  rfl

/-- Engine coordinates sampled before the shared gamma. -/
def enginePrefixSupport
    {shape : SemanticShape}
    (alphabet : Support K) :
    Support ((AlphaWord × BetaAWord) × BetaRWord shape) :=
  ((Support.challengeVectors alphabet
      PiCcsDomains.production.laneVariables).product
    (Support.challengeVectors alphabet
      PiCcsDomains.production.laneVariables)).product
    (Support.challengeVectors alphabet shape.rowVariables)

/-- Exact support for the batched engine challenges. -/
def engineSupport
    {shape : SemanticShape}
    (alphabet : Support K) :
    Support (EngineSeed shape) :=
  (enginePrefixSupport (shape := shape) alphabet).product alphabet

/-- Delayed coordinates sampled before the final batch weight. -/
def delayedPrefixSupport
    (alphabet : Support K) :
    Support (BetaBlockWord × K) :=
  (Support.challengeVectors alphabet
    PiCcsDomains.production.blockVariables).product alphabet

/-- Exact support for `betaBlock`, then the two ordered delayed scalars. -/
def delayedSupport
    (alphabet : Support K) :
    Support DelayedSeed :=
  (delayedPrefixSupport alphabet).product alphabet

/-- Exact support for every challenge sampled before FE starts. -/
def preSupport
    {shape : SemanticShape}
    (alphabet : Support K) :
    Support (PreSeed shape) :=
  (engineSupport (shape := shape) alphabet).product
    (delayedSupport alphabet)

/-- Exact causal FE-then-NC round-word support. -/
def sumCheckSupport
    {shape : SemanticShape}
    (alphabet : Support K) :
    Support (SumCheckSeed shape) :=
  (Support.challengeVectors alphabet
      (Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness.feRoundCount
        shape PiCcsDomains.production.fe)).product
    (Support.challengeVectors alphabet CausalSoundness.ncRoundCount)

/-- One nonempty duplicate-free support for the complete ideal execution. -/
def support
    {shape : SemanticShape}
    (alphabet : Support K) :
    Support (Seed shape) :=
  (preSupport (shape := shape) alphabet).product
    (sumCheckSupport (shape := shape) alphabet)

@[simp] theorem support_cardinality
    {shape : SemanticShape}
    (alphabet : Support K) :
    (support (shape := shape) alphabet).cardinality =
      (((alphabet.cardinality ^
          PiCcsDomains.production.laneVariables *
        alphabet.cardinality ^
          PiCcsDomains.production.laneVariables) *
        alphabet.cardinality ^ shape.rowVariables) *
        alphabet.cardinality) *
      (((alphabet.cardinality ^
          PiCcsDomains.production.blockVariables *
        alphabet.cardinality) *
        alphabet.cardinality) *
      ((alphabet.cardinality ^
          (Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness.feRoundCount
            shape PiCcsDomains.production.fe)) *
        alphabet.cardinality ^ CausalSoundness.ncRoundCount)) := by
  simp [support, preSupport, sumCheckSupport, engineSupport,
    enginePrefixSupport, delayedSupport, delayedPrefixSupport]
  ac_rfl

/-! ## Schedule interpreter -/

/-- Proof-owned transcript state.  The base state continues to follow the
existing schedule; the ideal seed and two round cursors are immutable
sampling authority. -/
structure ReplayState
    (BaseState : Type uState)
    (shape : SemanticShape) where
  base : BaseState
  seed : Seed shape
  delayedDomain : Option DelayedChallengeDomain
  feIndex : Nat
  ncIndex : Nat

namespace ReplayState

def initial
    {BaseState : Type uState}
    {shape : SemanticShape}
    (base : BaseState)
    (seed : Seed shape) :
    ReplayState BaseState shape where
  base := base
  seed := seed
  delayedDomain := none
  feIndex := 0
  ncIndex := 0

end ReplayState

/-- Total lookup used by the schedule carrier.  Exact replay theorems below
show that every protocol call is in range; the default is not protocol
authority. -/
def wordAt
    {count : Nat}
    (word : Fin count -> K)
    (index : Nat) : K :=
  if inRange : index < count then
    word ⟨index, inRange⟩
  else
    K.zero

@[simp] theorem wordAt_fin
    {count : Nat}
    (word : Fin count -> K)
    (index : Fin count) :
    wordAt word index.val = word index := by
  simp [wordAt, index.isLt]

/-- Lift an existing state-transition schedule while replacing only its
challenge source with the explicit ordered ideal seed.  Base-state absorption
and successor-state threading are preserved operation by operation. -/
def schedule
    {BaseState : Type uState}
    {shape : SemanticShape}
    {VerifierKey Input : Type}
    (baseSchedule :
      Schedule VerifierKey Input shape PiCcsDomains.production BaseState) :
    Schedule VerifierKey Input shape PiCcsDomains.production
      (ReplayState BaseState shape) where
  bindStatement := fun state statement =>
    { state with
      base := baseSchedule.bindStatement state.base statement
      delayedDomain := none
      feIndex := 0
      ncIndex := 0 }
  deriveCore := fun state =>
    let baseCore := baseSchedule.deriveCore state.base
    {
      challenges := coreChallenges state.seed
      state := { state with base := baseCore.state }
    }
  enterDelayedDomain := fun domain state =>
    { state with
      base := baseSchedule.enterDelayedDomain domain state.base
      delayedDomain := some domain }
  squeezeDelayedChallenge := fun state =>
    let baseSample := baseSchedule.squeezeDelayedChallenge state.base
    let challenge :=
      match state.delayedDomain with
      | some .producerBeta => state.seed.producerBeta
      | some .batchWeight => state.seed.batchWeight
      | none => K.zero
    (challenge,
      { state with
        base := baseSample.2
        delayedDomain := none })
  enterFe := fun state initial =>
    { state with
      base := baseSchedule.enterFe state.base initial
      feIndex := 0 }
  absorbFeRound := fun state message =>
    { state with
      base := baseSchedule.absorbFeRound state.base message }
  squeezeFeChallenge := fun state =>
    let baseSample := baseSchedule.squeezeFeChallenge state.base
    (wordAt state.seed.feWord state.feIndex,
      { state with
        base := baseSample.2
        feIndex := state.feIndex + 1 })
  enterNc := fun state =>
    { state with
      base := baseSchedule.enterNc state.base
      ncIndex := 0 }
  absorbNcRound := fun state message =>
    { state with
      base := baseSchedule.absorbNcRound state.base message }
  squeezeNcChallenge := fun state =>
    let baseSample := baseSchedule.squeezeNcChallenge state.base
    (wordAt state.seed.ncWord state.ncIndex,
      { state with
        base := baseSample.2
        ncIndex := state.ncIndex + 1 })
  absorbOutput := fun state output =>
    { state with
      base := baseSchedule.absorbOutput state.base output }

/-- `deriveCore` erases exactly to the existing core challenge carrier. -/
@[simp] theorem schedule_deriveCore_challenges
    {BaseState : Type uState}
    {shape : SemanticShape}
    {VerifierKey Input : Type}
    (baseSchedule :
      Schedule VerifierKey Input shape PiCcsDomains.production BaseState)
    (state : ReplayState BaseState shape) :
    ((schedule baseSchedule).deriveCore state).challenges =
      coreChallenges state.seed := by
  rfl

/-- The selected carrier keeps one shared gamma across FE and NC. -/
@[simp] theorem derivePreSumcheck_shared_gamma
    {BaseState : Type uState}
    {shape : SemanticShape}
    {VerifierKey Input : Type}
    (baseSchedule :
      Schedule VerifierKey Input shape PiCcsDomains.production BaseState)
    (prior : ReplayState BaseState shape)
    (statement : Statement VerifierKey Input) :
    let challenges :=
      (derivePreSumcheck (schedule baseSchedule) prior statement).challenges
    challenges.feCoins.gamma = prior.seed.gamma ∧
      challenges.ncCoins.gamma = prior.seed.gamma ∧
      challenges.ncCoins.gamma = challenges.feCoins.gamma := by
  exact ⟨rfl, rfl, rfl⟩

/-- The two delayed challenges are consumed in their typed order. -/
@[simp] theorem derivePreSumcheck_delayed
    {BaseState : Type uState}
    {shape : SemanticShape}
    {VerifierKey Input : Type}
    (baseSchedule :
      Schedule VerifierKey Input shape PiCcsDomains.production BaseState)
    (prior : ReplayState BaseState shape)
    (statement : Statement VerifierKey Input) :
    let challenges :=
      (derivePreSumcheck (schedule baseSchedule) prior statement).challenges
    challenges.producerBeta = prior.seed.producerBeta ∧
      challenges.batchWeight = prior.seed.batchWeight := by
  exact ⟨rfl, rfl⟩

/-- Lift the unrelated PiRLC state machine without changing its candidate
stream.  This is needed only because the fixed-active context shares one
state type across protocol phases. -/
def piRlcMachine
    {BaseState : Type uState}
    {shape : SemanticShape}
    (baseMachine :
      Nifs.NonInteractive.PiRlcSampler.ProductionSchedule.Machine BaseState) :
    Nifs.NonInteractive.PiRlcSampler.ProductionSchedule.Machine
      (ReplayState BaseState shape) where
  enterScalar := fun state coordinate =>
    { state with
      base := baseMachine.enterScalar state.base coordinate }
  digestBlock := fun state counter =>
    let result := baseMachine.digestBlock state.base counter
    ({ state with base := result.1 }, result.2)

/-! ## Existing production-context constructor -/

variable
  {shape : SemanticShape}
  {BaseState : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Install the selected ideal seed into the existing typed production input.
All statement/source fields are inherited from the authoritative base input;
only transcript state/schedule and the denominator are refined. -/
def input
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := BaseState)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (seed : Seed shape) :
    ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := ReplayState BaseState shape)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits) where
  carrier := baseInput.carrier
  context := {
    covers := baseInput.context.covers
    key := baseInput.context.key
    alignment := baseInput.context.alignment
    input := baseInput.context.input
    pending := baseInput.context.pending
    piCcsInput := baseInput.context.piCcsInput
    priorState := ReplayState.initial baseInput.context.priorState seed
    piCcsSchedule := schedule baseInput.context.piCcsSchedule
    piRlcMachine := piRlcMachine baseInput.context.piRlcMachine
    profile := baseInput.context.profile
    challengeSetSize := alphabet.cardinality
  }

@[simp] theorem input_data
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := BaseState)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (seed : Seed shape) :
    (input alphabet baseInput seed).data = baseInput.data := by
  rfl

@[simp] theorem input_challengeSetSize
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := BaseState)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (seed : Seed shape) :
    (input alphabet baseInput seed).full.challengeSetSize =
      alphabet.cardinality := by
  rfl

@[simp] theorem input_feCoins
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := BaseState)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (seed : Seed shape) :
    (input alphabet baseInput seed).full.feCoins =
      (challenges seed).feCoins := by
  rfl

@[simp] theorem input_ncCoins
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := BaseState)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (seed : Seed shape) :
    (input alphabet baseInput seed).full.ncCoins =
      (challenges seed).ncCoins := by
  rfl

@[simp] theorem input_producerBeta
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := BaseState)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (seed : Seed shape) :
    (input alphabet baseInput seed).full.producerBeta =
      seed.producerBeta := by
  rfl

@[simp] theorem input_batchWeight
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := BaseState)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (seed : Seed shape) :
    (input alphabet baseInput seed).full.batchWeight =
      seed.batchWeight := by
  rfl

@[simp] theorem input_profile
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := BaseState)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (seed : Seed shape) :
    (input alphabet baseInput seed).full.profile =
      baseInput.full.profile := by
  rfl

@[simp] theorem input_covers
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := BaseState)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (seed : Seed shape) :
    (input alphabet baseInput seed).full.covers =
      baseInput.full.covers := by
  rfl

@[simp] theorem input_pending
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := BaseState)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (seed : Seed shape) :
    (input alphabet baseInput seed).full.pending =
      baseInput.full.pending := by
  rfl

@[simp] theorem input_productionWeights
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := BaseState)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (seed : Seed shape) :
    ProductionProjection.productionWeights
        (input alphabet baseInput seed).full =
      ProductionProjection.productionWeights baseInput.full := by
  rfl

/-- The new constructor, unlike the unrestricted old carrier, supplies the
exact denominator/support alignment required by root counting. -/
theorem input_supportAligned
    (alphabet : Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := BaseState)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (seed : Seed shape) :
    ChallengeSupportAligned (input alphabet baseInput seed).full alphabet := by
  rfl

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveCarrier
