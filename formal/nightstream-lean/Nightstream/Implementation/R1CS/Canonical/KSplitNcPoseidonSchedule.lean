import Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

/-!
Contract: the Lean-selected value-level Poseidon2 schedule for operational
Split-NC `Pi_CCS`.

This module instantiates every operation of
`TranscriptAuthority.BlockLane.Schedule` with the width-8 overwrite duplex.
The tags and the serialization framing are Lean-owned encoding choices:

* every typed domain has a distinct tag;
* every variable-width payload is prefixed by its field length;
* quadratic-extension values are serialized low limb then high limb;
* FE and NC messages are absorbed before their challenge squeeze;
* the state returned by FE is the state entering NC;
* the raw output is absorbed only after NC.

`Serialization` is deliberately the remaining boundary in this module.  A
selected call must construct it from the authoritative call-frame codecs.
It is data-only and cannot carry a challenge, acceptance result, transcript
state, or semantic conclusion.

Does not own: physical rows, call-frame decoding, source authority, the
selected serialization inhabitant, Fiat--Shamir security, Rust, or artifacts.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcPoseidonSchedule

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule

universe uVerifierKey uInput

abbrev State := Poseidon2Duplex.State

/-- Kernel-distinct domains used by the selected operational replay. -/
inductive Tag where
  | statement
  | alpha
  | betaA
  | betaR
  | gamma
  | betaBlock
  | producerBeta
  | batchWeight
  | feEntry
  | feRound
  | ncEntry
  | ncRound
  | output
deriving Repr, DecidableEq

/-- Concrete field words for the typed domains.  These are canonical Lean
encoding constants, not values imported from a production emitter. -/
def Tag.code : Tag → Nat
  | .statement => 64
  | .alpha => 65
  | .betaA => 66
  | .betaR => 67
  | .gamma => 68
  | .betaBlock => 69
  | .producerBeta => 70
  | .batchWeight => 71
  | .feEntry => 72
  | .feRound => 73
  | .ncEntry => 74
  | .ncRound => 75
  | .output => 76

theorem Tag.code_lt_modulus (tag : Tag) :
    tag.code < goldilocksP := by
  cases tag <;> decide

theorem Tag.code_injective : Function.Injective Tag.code := by
  intro left right equal
  cases left <;> cases right <;> simp_all [Tag.code]

@[simp] theorem Tag.code_eq_iff (left right : Tag) :
    left.code = right.code ↔ left = right :=
  Tag.code_injective.eq_iff

/-- The only two data serializations not fixed by the transcript carrier
itself.  A selected profile must encode the complete typed statement and the
complete raw output message. -/
structure Serialization
    (VerifierKey : Type uVerifierKey)
    (Input : Type uInput)
    (shape : SemanticShape) where
  statementFields : Statement VerifierKey Input → List Nat
  outputFields : OutputMessage shape → List Nat

/-- Canonical low/high serialization of one quadratic-extension element. -/
def kFields (value : K) : List Nat :=
  [value.c0.val, value.c1.val]

@[simp] theorem kFields_length (value : K) :
    (kFields value).length = 2 := rfl

@[simp] theorem flatMap_kFields_length (values : List K) :
    (values.flatMap kFields).length = values.length * 2 := by
  induction values with
  | nil => rfl
  | cons value values inductionHypothesis =>
      simp [kFields, inductionHypothesis, Nat.succ_mul]

/-- Constant-first coefficient serialization of one FE message. -/
def feMessageFields
    (message : Nightstream.SuperNeo.SumCheck.Finite.Message K) : List Nat :=
  message.coefficients.flatMap kFields

/-- Constant-first five-coefficient serialization of one NC message. -/
def ncMessageFields
    (message :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.RoundMessage) :
    List Nat :=
  message.coefficients.flatMap kFields

@[simp] theorem ncMessageFields_length
    (message :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.RoundMessage) :
    (ncMessageFields message).length = 10 := by
  rw [ncMessageFields, flatMap_kFields_length,
    message.coefficients_length]
  decide

/-- Length-delimited typed absorption.  Including the payload length prevents
the trailing-zero ambiguity of the unframed binding sponge. -/
def absorbTagged
    (constants : Constants) (tag : Tag) (payload : List Nat)
    (state : State) : State :=
  Poseidon2Duplex.absorbList constants
    (tag.code :: payload.length % goldilocksP ::
      payload.map (fun value => value % goldilocksP)) state

/-- Convert the row-layer duplex challenge to the semantic concrete carrier
without changing either Goldilocks coordinate. -/
def squeezeK (constants : Constants) (state : State) : K × State :=
  let sampled := SymbolicDuplexSemantics.squeezeKValue constants state
  (KConcreteFixedPhaseBridge.ofProjection sampled.1, sampled.2)

/-- Squeeze exactly `count` extension values, threading the unique state. -/
def squeezeManyK (constants : Constants) :
    Nat → State → List K × State
  | 0, state => ([], state)
  | count + 1, state =>
      let sampled := squeezeK constants state
      let rest := squeezeManyK constants count sampled.2
      (sampled.1 :: rest.1, rest.2)

@[simp] theorem squeezeManyK_length
    (constants : Constants) (count : Nat) (state : State) :
    (squeezeManyK constants count state).1.length = count := by
  induction count generalizing state with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [squeezeManyK, List.length_cons]
      rw [inductionHypothesis]

/-- Enter one typed challenge domain, then squeeze its exact vector. -/
def sampleVector
    (constants : Constants) (tag : Tag) (count : Nat) (state : State) :
    List K × State :=
  squeezeManyK constants count (absorbTagged constants tag [] state)

@[simp] theorem sampleVector_length
    (constants : Constants) (tag : Tag) (count : Nat) (state : State) :
    (sampleVector constants tag count state).1.length = count :=
  squeezeManyK_length constants count _

/-- Package an exact sampled list as the verifier's typed cube point. -/
def sampledPoint
    (constants : Constants) (tag : Tag) (count : Nat) (state : State) :
    CubePoint K count × State :=
  let sampled := sampleVector constants tag count state
  ({
    coordinates := sampled.1
    dimension := sampleVector_length constants tag count state
  }, sampled.2)

/-- One typed scalar sample. -/
def sampleScalar
    (constants : Constants) (tag : Tag) (state : State) : K × State :=
  squeezeK constants
    (absorbTagged constants tag [] state)

/-- Derive all core challenges in their unique typed order. -/
def deriveCore
    {shape : SemanticShape}
    {domains : Domains}
    (constants : Constants) (state : State) :
    CorePreSumcheck shape domains State :=
  let alpha := sampledPoint constants .alpha domains.laneVariables state
  let betaA := sampledPoint constants .betaA domains.laneVariables alpha.2
  let betaR := sampledPoint constants .betaR shape.rowVariables betaA.2
  let gamma := sampleScalar constants .gamma betaR.2
  let betaBlock :=
    sampledPoint constants .betaBlock domains.blockVariables gamma.2
  {
    challenges := {
      alpha := alpha.1
      betaA := betaA.1
      betaR := betaR.1
      gamma := gamma.1
      betaBlock := betaBlock.1
    }
    state := betaBlock.2
  }

/-- Map the kernel-distinct delayed roles to distinct concrete tags. -/
def delayedTag : DelayedChallengeDomain → Tag
  | .producerBeta => .producerBeta
  | .batchWeight => .batchWeight

theorem delayedTag_injective : Function.Injective delayedTag := by
  intro left right equal
  cases left <;> cases right <;> simp_all [delayedTag]

/-- Selected operational schedule.  Every function is an explicit duplex
operation; no challenge or successor state is supplied by a certificate. -/
def schedule
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    (constants : Constants)
    (serialization : Serialization VerifierKey Input shape) :
    Schedule VerifierKey Input shape domains State where
  bindStatement state statement :=
    absorbTagged constants .statement
      (serialization.statementFields statement) state
  deriveCore state := deriveCore constants state
  enterDelayedDomain domain state :=
    absorbTagged constants (delayedTag domain) [] state
  squeezeDelayedChallenge state :=
    squeezeK constants state
  enterFe state initialClaim :=
    absorbTagged constants .feEntry (kFields initialClaim) state
  absorbFeRound state message :=
    absorbTagged constants .feRound (feMessageFields message) state
  squeezeFeChallenge state :=
    squeezeK constants state
  enterNc state :=
    absorbTagged constants .ncEntry [] state
  absorbNcRound state message :=
    absorbTagged constants .ncRound (ncMessageFields message) state
  squeezeNcChallenge state :=
    squeezeK constants state
  absorbOutput state output :=
    absorbTagged constants .output (serialization.outputFields output) state

/-! ## Exact operation equations

These lemmas make later symbolic refinement depend on stable named
operations rather than unfolding the schedule record ad hoc. -/

@[simp] theorem schedule_bindStatement
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    (constants : Constants)
    (serialization : Serialization VerifierKey Input shape)
    (state : State) (statement : Statement VerifierKey Input) :
    (schedule (domains := domains) constants serialization).bindStatement
        state statement =
      absorbTagged constants .statement
        (serialization.statementFields statement) state := rfl

@[simp] theorem schedule_deriveCore
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    (constants : Constants)
    (serialization : Serialization VerifierKey Input shape)
    (state : State) :
    (schedule (domains := domains) constants serialization).deriveCore state =
      deriveCore constants state := rfl

@[simp] theorem schedule_enterDelayedDomain
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    (constants : Constants)
    (serialization : Serialization VerifierKey Input shape)
    (domain : DelayedChallengeDomain) (state : State) :
    (schedule (domains := domains) constants serialization).enterDelayedDomain
        domain state =
      absorbTagged constants (delayedTag domain) [] state := rfl

@[simp] theorem schedule_enterFe
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    (constants : Constants)
    (serialization : Serialization VerifierKey Input shape)
    (state : State) (initialClaim : K) :
    (schedule (domains := domains) constants serialization).enterFe
        state initialClaim =
      absorbTagged constants .feEntry (kFields initialClaim) state := rfl

@[simp] theorem schedule_absorbFeRound
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    (constants : Constants)
    (serialization : Serialization VerifierKey Input shape)
    (state : State)
    (message : Nightstream.SuperNeo.SumCheck.Finite.Message K) :
    (schedule (domains := domains) constants serialization).absorbFeRound
        state message =
      absorbTagged constants .feRound (feMessageFields message) state := rfl

@[simp] theorem schedule_enterNc
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    (constants : Constants)
    (serialization : Serialization VerifierKey Input shape)
    (state : State) :
    (schedule (domains := domains) constants serialization).enterNc state =
      absorbTagged constants .ncEntry [] state := rfl

@[simp] theorem schedule_absorbNcRound
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    (constants : Constants)
    (serialization : Serialization VerifierKey Input shape)
    (state : State)
    (message :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.RoundMessage) :
    (schedule (domains := domains) constants serialization).absorbNcRound
        state message =
      absorbTagged constants .ncRound (ncMessageFields message) state := rfl

@[simp] theorem schedule_absorbOutput
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    (constants : Constants)
    (serialization : Serialization VerifierKey Input shape)
    (state : State) (output : OutputMessage shape) :
    (schedule (domains := domains) constants serialization).absorbOutput
        state output =
      absorbTagged constants .output
        (serialization.outputFields output) state := rfl

end Nightstream.Implementation.R1CS.Canonical.KSplitNcPoseidonSchedule
