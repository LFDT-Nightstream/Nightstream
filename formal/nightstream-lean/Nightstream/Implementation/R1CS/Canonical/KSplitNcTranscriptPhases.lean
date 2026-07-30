import Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptSemantics

/-!
Contract: phase-level refinement of the operational Split-NC transcript.

This module constructs the exact FE and block×lane NC certificates from the
same typed round columns consumed by the numeric row program.  It then proves
that the symbolic Poseidon2 replay and the selected value-level verifier replay
absorb the same messages, derive the same challenges, and thread the identical
FE-to-NC transcript state.

No challenge, phase-boundary state, certificate message, or acceptance result
is supplied independently of the physical columns.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptPhases

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
open Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptSemantics
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

abbrev ValueState := Poseidon2Duplex.State

private theorem cubePoint_eq_of_coordinates_eq
    {variables : Nat}
    (left right : CubePoint K variables)
    (coordinates : left.coordinates = right.coordinates) :
    left = right := by
  cases left
  cases right
  simp_all

/-- The exact mixed-width FE certificate decoded from the physical round
columns. -/
def feCertificate
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    SumCheck.Fe.Certificate polynomialInput domains.fe where
  rowRounds index :=
    (input.fe.rowRounds index).paperPolynomial assignment
  laneRounds index :=
    (input.fe.laneRounds index).paperPolynomial assignment

/-- The exact block-prefix then lane-suffix NC certificate decoded from the
physical round columns. -/
def ncCertificate
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Transcript.Nc.BlockLane.Certificate domains.nc where
  rounds :=
    Fin.addCases
      (fun index =>
        (input.nc.blockRounds index).paperPolynomial assignment)
      (fun index =>
        (input.nc.laneRounds index).paperPolynomial assignment)

/-- FE point read from the transcript's own row/lane squeeze outputs. -/
def decodedFePoint
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Polynomial.Fe.Point shape domains.fe where
  row := {
    coordinates :=
      decodedColumnList assignment
        (KSplitNcTranscript.feRowReplay input).challenges
    dimension := by
      rw [decodedColumnList_length]
      unfold KSplitNcTranscript.feRowReplay
      rw [
        KSplitNcTranscript.replayRounds_challenges_length]
      simp
  }
  lane := {
    coordinates :=
      decodedColumnList assignment
        (KSplitNcTranscript.feLaneReplay input).challenges
    dimension := by
      rw [decodedColumnList_length]
      unfold KSplitNcTranscript.feLaneReplay
      rw [
        KSplitNcTranscript.replayRounds_challenges_length]
      simp [Domains.fe]
  }

/-- Canonical NC point read from the transcript's own block/lane squeeze
outputs. -/
def decodedNcPoint
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Polynomial.Nc.BlockLane.Point domains.nc where
  block := {
    coordinates :=
      decodedColumnList assignment
        (KSplitNcTranscript.ncBlockReplay input).challenges
    dimension := by
      rw [decodedColumnList_length]
      unfold KSplitNcTranscript.ncBlockReplay
      rw [
        KSplitNcTranscript.replayRounds_challenges_length]
      simp [Domains.nc]
  }
  lane := {
    coordinates :=
      decodedColumnList assignment
        (KSplitNcTranscript.ncLaneReplay input).challenges
    dimension := by
      rw [decodedColumnList_length]
      unfold KSplitNcTranscript.ncLaneReplay
      rw [
        KSplitNcTranscript.replayRounds_challenges_length]
      simp [Domains.nc]
  }

@[simp] theorem feCertificate_rowRounds
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    List.ofFn (feCertificate assignment input).rowRounds =
      (List.ofFn input.fe.rowRounds).map
        (fun round => round.paperPolynomial assignment) := by
  rw [List.map_ofFn]
  rfl

@[simp] theorem feCertificate_laneRounds
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    List.ofFn (feCertificate assignment input).laneRounds =
      (List.ofFn input.fe.laneRounds).map
        (fun round => round.paperPolynomial assignment) := by
  rw [List.map_ofFn]
  rfl

@[simp] theorem ncCertificate_rounds
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    List.ofFn (ncCertificate assignment input).rounds =
      (List.ofFn input.nc.blockRounds).map
          (fun round => round.paperPolynomial assignment) ++
        (List.ofFn input.nc.laneRounds).map
          (fun round => round.paperPolynomial assignment) := by
  rw [List.ofFn_add]
  rw [List.map_ofFn, List.map_ofFn]
  congr
  · funext index
    simpa [ncCertificate, Fin.castAdd] using
      (Fin.addCases_left
        (motive := fun _ =>
          SumCheck.Nc.RoundMessage)
        index)
  · funext index
    simpa [ncCertificate] using
      (Fin.addCases_right
        (motive := fun _ =>
          SumCheck.Nc.RoundMessage)
        index)

@[simp] theorem ncCertificate_blockRounds
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    (ncCertificate assignment input).blockRounds =
      (List.ofFn input.nc.blockRounds).map
        (fun round => round.paperPolynomial assignment) := by
  unfold Transcript.Nc.BlockLane.Certificate.blockRounds
  unfold Transcript.Nc.BlockLane.Certificate.rawRounds
  rw [ncCertificate_rounds]
  calc
    (((List.ofFn input.nc.blockRounds).map
          (fun round => round.paperPolynomial assignment)) ++
        ((List.ofFn input.nc.laneRounds).map
          (fun round => round.paperPolynomial assignment))).take
        domains.nc.blockVariables =
      (((List.ofFn input.nc.blockRounds).map
          (fun round => round.paperPolynomial assignment)) ++
        ((List.ofFn input.nc.laneRounds).map
          (fun round => round.paperPolynomial assignment))).take
        ((List.ofFn input.nc.blockRounds).map
          (fun round => round.paperPolynomial assignment)).length := by
      congr 2
      simp [Domains.nc]
    _ = _ := List.take_append_length

@[simp] theorem ncCertificate_laneRounds
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    (ncCertificate assignment input).laneRounds =
      (List.ofFn input.nc.laneRounds).map
        (fun round => round.paperPolynomial assignment) := by
  unfold Transcript.Nc.BlockLane.Certificate.laneRounds
  unfold Transcript.Nc.BlockLane.Certificate.rawRounds
  rw [ncCertificate_rounds]
  calc
    (((List.ofFn input.nc.blockRounds).map
          (fun round => round.paperPolynomial assignment)) ++
        ((List.ofFn input.nc.laneRounds).map
          (fun round => round.paperPolynomial assignment))).drop
        domains.nc.blockVariables =
      (((List.ofFn input.nc.blockRounds).map
          (fun round => round.paperPolynomial assignment)) ++
        ((List.ofFn input.nc.laneRounds).map
          (fun round => round.paperPolynomial assignment))).drop
        ((List.ofFn input.nc.blockRounds).map
          (fun round => round.paperPolynomial assignment)).length := by
      congr 2
      simp [Domains.nc]
    _ = _ := List.drop_append_length

private theorem fePayloads_eq
    {degree : Nat}
    (assignment : Nat → Nat) :
    ∀ rounds : List (RoundColumns degree),
      rounds.map
          (fun round =>
            fieldValues assignment (KSplitNcTranscript.roundFields round)) =
        (rounds.map
          (fun round => round.paperPolynomial assignment)).map
            (fun polynomial =>
              KSplitNcPoseidonSchedule.feMessageFields
                polynomial.toMessage)
  | [] => rfl
  | round :: rounds => by
      simp only [List.map_cons]
      rw [fieldValues_roundFields assignment round,
        fePayloads_eq assignment rounds]
      rfl

private theorem ncPayloads_eq
    (assignment : Nat → Nat) :
    ∀ rounds : List (RoundColumns 4),
      rounds.map
          (fun round =>
            fieldValues assignment (KSplitNcTranscript.roundFields round)) =
        (rounds.map
          (fun round => round.paperPolynomial assignment)).map
            KSplitNcPoseidonSchedule.ncMessageFields
  | [] => rfl
  | round :: rounds => by
      simp only [List.map_cons]
      rw [fieldValues_roundFields assignment round,
        ncPayloads_eq assignment rounds]
      rfl

theorem feRow_payloads_eq
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    (List.ofFn input.fe.rowRounds).map
        (fun round =>
          fieldValues assignment (KSplitNcTranscript.roundFields round)) =
      (feCertificate assignment input).rowRawRounds.map
        KSplitNcPoseidonSchedule.feMessageFields := by
  simp only [SumCheck.Fe.Certificate.rowRawRounds,
    feCertificate_rowRounds, List.map_map]
  simpa only [List.map_map, Function.comp_apply] using
    (fePayloads_eq assignment (List.ofFn input.fe.rowRounds))

theorem feLane_payloads_eq
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    (List.ofFn input.fe.laneRounds).map
        (fun round =>
          fieldValues assignment (KSplitNcTranscript.roundFields round)) =
      (feCertificate assignment input).laneRawRounds.map
        KSplitNcPoseidonSchedule.feMessageFields := by
  rw [SumCheck.Fe.Certificate.laneRawRounds]
  rw [feCertificate_laneRounds]
  calc
    _ = ((List.ofFn input.fe.laneRounds).map
          (fun round => round.paperPolynomial assignment)).map
            (fun polynomial =>
              KSplitNcPoseidonSchedule.feMessageFields
                polynomial.toMessage) :=
      fePayloads_eq assignment (List.ofFn input.fe.laneRounds)
    _ = _ := by
      exact (List.map_map
        (g := KSplitNcPoseidonSchedule.feMessageFields)
        (f := Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial.toMessage)
        (l := (List.ofFn input.fe.laneRounds).map
          (fun round => round.paperPolynomial assignment))).symm

theorem ncBlockLane_payloads_eq
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    ((List.ofFn input.nc.blockRounds).map
          (fun round =>
            fieldValues assignment (KSplitNcTranscript.roundFields round)) ++
        (List.ofFn input.nc.laneRounds).map
          (fun round =>
            fieldValues assignment (KSplitNcTranscript.roundFields round))) =
      (ncCertificate assignment input).rawRounds.map
        KSplitNcPoseidonSchedule.ncMessageFields := by
  rw [Transcript.Nc.BlockLane.Certificate.rawRounds]
  rw [ncCertificate_rounds]
  calc
    _ = ((List.ofFn input.nc.blockRounds).map
            (fun round => round.paperPolynomial assignment)).map
              KSplitNcPoseidonSchedule.ncMessageFields ++
          ((List.ofFn input.nc.laneRounds).map
            (fun round => round.paperPolynomial assignment)).map
              KSplitNcPoseidonSchedule.ncMessageFields := by
        rw [ncPayloads_eq assignment, ncPayloads_eq assignment]
    _ = _ := by
      exact (List.map_append
        (f := KSplitNcPoseidonSchedule.ncMessageFields)
        (l₁ := (List.ofFn input.nc.blockRounds).map
          (fun round => round.paperPolynomial assignment))
        (l₂ := (List.ofFn input.nc.laneRounds).map
          (fun round => round.paperPolynomial assignment))).symm

theorem ncBlock_payloads_eq
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    (List.ofFn input.nc.blockRounds).map
        (fun round =>
          fieldValues assignment (KSplitNcTranscript.roundFields round)) =
      (ncCertificate assignment input).blockRounds.map
        KSplitNcPoseidonSchedule.ncMessageFields := by
  rw [ncCertificate_blockRounds]
  exact ncPayloads_eq assignment (List.ofFn input.nc.blockRounds)

theorem ncLane_payloads_eq
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    (List.ofFn input.nc.laneRounds).map
        (fun round =>
          fieldValues assignment (KSplitNcTranscript.roundFields round)) =
      (ncCertificate assignment input).laneRounds.map
        KSplitNcPoseidonSchedule.ncMessageFields := by
  rw [ncCertificate_laneRounds]
  exact ncPayloads_eq assignment (List.ofFn input.nc.laneRounds)

/-- The value replay helper is definitionally the selected FE verifier replay
once both are fed the same physical message serialization. -/
theorem valueReplayPayloads_eq_fe
    {shape : SemanticShape}
    {domains : Domains}
    (constants : Poseidon2Schedule.Constants)
    (serialization : KSplitNcPoseidonSchedule.Serialization Unit Unit shape)
    (initialClaim : K) :
    ∀ (messages : List (Nightstream.SuperNeo.SumCheck.Finite.Message K))
      (state : ValueState),
      valueReplayPayloads constants .feRound
          (messages.map KSplitNcPoseidonSchedule.feMessageFields) state =
        Transcript.Fe.runRoundsFrom
          (feMachine
            (KSplitNcPoseidonSchedule.schedule
              (domains := domains) constants serialization)
            initialClaim)
          state messages
  | [], _ => rfl
  | message :: messages, state => by
      simp only [valueReplayPayloads, List.map_cons,
        Transcript.Fe.runRoundsFrom, Transcript.Fe.runRound,
        feMachine, KSplitNcPoseidonSchedule.schedule]
      rw [valueReplayPayloads_eq_fe
        (domains := domains) constants serialization initialClaim messages]
      rfl

/-- The value replay helper is definitionally the selected NC verifier replay
once both are fed the same physical message serialization. -/
theorem valueReplayPayloads_eq_nc
    {shape : SemanticShape}
    {domains : Domains}
    (constants : Poseidon2Schedule.Constants)
    (serialization : KSplitNcPoseidonSchedule.Serialization Unit Unit shape) :
    ∀ (messages : List Transcript.Nc.RoundMessage)
      (state : ValueState),
      valueReplayPayloads constants .ncRound
          (messages.map KSplitNcPoseidonSchedule.ncMessageFields) state =
        Transcript.Nc.runRoundsFrom
          (ncMachine
            (KSplitNcPoseidonSchedule.schedule
              (domains := domains) constants serialization))
          state messages
  | [], _ => rfl
  | message :: messages, state => by
      simp only [valueReplayPayloads, List.map_cons,
        Transcript.Nc.runRoundsFrom, Transcript.Nc.runRound,
        ncMachine, KSplitNcPoseidonSchedule.schedule]
      rw [valueReplayPayloads_eq_nc
        (domains := domains) constants serialization messages]
      rfl

/-! ## Selected semantic execution -/

def semanticPre
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    PreSumcheck shape domains ValueState :=
  derivePreSumcheck
    (valueSchedule constants assignment input)
    (priorState assignment input) unitStatement

def semanticFeInitial
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) : K :=
  Polynomial.Fe.initial profile polynomialInput
    (semanticPre constants assignment input).challenges.feCoins

def semanticFeMachine
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Transcript.Fe.Machine ValueState :=
  feMachine
    (valueSchedule constants assignment input)
    (semanticFeInitial profile constants assignment input)

def semanticFeExecution
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Transcript.Fe.Derived shape domains.fe ValueState :=
  Transcript.Fe.derive
    (semanticFeMachine profile constants assignment input)
    (semanticPre constants assignment input).state
    (feCertificate assignment input)

def semanticNcExecution
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Transcript.Nc.BlockLane.Derived domains.nc ValueState :=
  Transcript.Nc.BlockLane.derive
    (ncMachine (valueSchedule constants assignment input))
    (semanticFeExecution profile constants assignment input).finalState
    (ncCertificate assignment input)

private theorem feEntry_to_output_extends
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Extends (KSplitNcTranscript.feEntryBuilder input)
      (KSplitNcTranscript.outputBuilder input) :=
  (feRow_extends input).trans
    ((feLane_extends input).trans
      ((ncEntry_extends input).trans
        ((ncBlock_extends input).trans
          ((ncLane_extends input).trans (output_extends input)))))

private theorem feRow_to_output_extends
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Extends (KSplitNcTranscript.feRowReplay input).builder
      (KSplitNcTranscript.outputBuilder input) :=
  (feLane_extends input).trans
    ((ncEntry_extends input).trans
      ((ncBlock_extends input).trans
        ((ncLane_extends input).trans (output_extends input))))

private theorem feLane_to_output_extends
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Extends (KSplitNcTranscript.feLaneReplay input).builder
      (KSplitNcTranscript.outputBuilder input) :=
  (ncEntry_extends input).trans
    ((ncBlock_extends input).trans
      ((ncLane_extends input).trans (output_extends input)))

private theorem ncEntry_to_output_extends
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Extends (KSplitNcTranscript.ncEntryBuilder input)
      (KSplitNcTranscript.outputBuilder input) :=
  (ncBlock_extends input).trans
    ((ncLane_extends input).trans (output_extends input))

private theorem ncBlock_to_output_extends
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Extends (KSplitNcTranscript.ncBlockReplay input).builder
      (KSplitNcTranscript.outputBuilder input) :=
  (ncLane_extends input).trans (output_extends input)

theorem decoded_feEntry
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (input : KSplitNcTranscript.Input polynomialInput domains)
    (valid :
      Valid input.transcriptBase constants assignment
        (KSplitNcTranscript.outputBuilder input))
    (initial :
      decodedColumns assignment input.fe.initial =
        semanticFeInitial profile constants assignment input) :
    decodedBuilder assignment (KSplitNcTranscript.feEntryBuilder input) =
      (semanticFeMachine profile constants assignment input).enterFe
        (semanticPre constants assignment input).state := by
  have feEntryValid :
      Valid input.transcriptBase constants assignment
        (KSplitNcTranscript.feEntryBuilder input) :=
    valid.of_extends (feEntry_to_output_extends input)
  have entered :=
    decoded_absorbTagged input.transcriptBase constants assignment
      constantWire .feEntry
      (KSplitNcTranscript.carriedFields (carried input.fe.initial))
      (KSplitNcTranscript.batchSample input).2 feEntryValid
  have payload :=
    fieldValues_carriedFields assignment (carried input.fe.initial)
  rw [decodeCarried_carried] at payload
  have pre :=
    decoded_preSumcheck constants assignment constantWire input valid
  have preState :
      decodedBuilder assignment
          (KSplitNcTranscript.batchSample input).2 =
        (semanticPre constants assignment input).state := by
    simpa only [KSplitNcTranscript.replay, semanticPre] using pre.state
  change
    ofProjection (input.fe.initial.value assignment) =
      semanticFeInitial profile constants assignment input at initial
  rw [payload, initial, preState] at entered
  simpa only [KSplitNcTranscript.feEntryBuilder,
    semanticFeMachine, semanticFeInitial, semanticPre,
    valueSchedule, KSplitNcPoseidonSchedule.schedule,
    feMachine] using entered

theorem decoded_feRow
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (input : KSplitNcTranscript.Input polynomialInput domains)
    (valid :
      Valid input.transcriptBase constants assignment
        (KSplitNcTranscript.outputBuilder input))
    (initial :
      decodedColumns assignment input.fe.initial =
        semanticFeInitial profile constants assignment input) :
    (decodedColumnList assignment
        (KSplitNcTranscript.feRowReplay input).challenges,
      decodedBuilder assignment
        (KSplitNcTranscript.feRowReplay input).builder) =
      Transcript.Fe.runRoundsFrom
        (semanticFeMachine profile constants assignment input)
        ((semanticFeMachine profile constants assignment input).enterFe
          (semanticPre constants assignment input).state)
        (feCertificate assignment input).rowRawRounds := by
  have rowValid :
      Valid input.transcriptBase constants assignment
        (KSplitNcTranscript.feRowReplay input).builder :=
    valid.of_extends (feRow_to_output_extends input)
  have decoded :=
    decoded_replayRounds input.transcriptBase constants assignment
      constantWire .feRound (List.ofFn input.fe.rowRounds)
      (KSplitNcTranscript.feEntryBuilder input) rowValid
  rw [feRow_payloads_eq assignment input,
    decoded_feEntry profile constants assignment constantWire input
      valid initial] at decoded
  rw [valueReplayPayloads_eq_fe
    (domains := domains) constants
    (valueSerialization assignment input)
    (semanticFeInitial profile constants assignment input)
    (feCertificate assignment input).rowRawRounds] at decoded
  simpa only [KSplitNcTranscript.feRowReplay,
    semanticFeMachine, valueSchedule] using decoded

theorem decoded_feLane
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (input : KSplitNcTranscript.Input polynomialInput domains)
    (valid :
      Valid input.transcriptBase constants assignment
        (KSplitNcTranscript.outputBuilder input))
    (initial :
      decodedColumns assignment input.fe.initial =
        semanticFeInitial profile constants assignment input) :
    (decodedColumnList assignment
        (KSplitNcTranscript.feLaneReplay input).challenges,
      decodedBuilder assignment
        (KSplitNcTranscript.feLaneReplay input).builder) =
      Transcript.Fe.runRoundsFrom
        (semanticFeMachine profile constants assignment input)
        (Transcript.Fe.runRoundsFrom
          (semanticFeMachine profile constants assignment input)
          ((semanticFeMachine profile constants assignment input).enterFe
            (semanticPre constants assignment input).state)
          (feCertificate assignment input).rowRawRounds).2
        (feCertificate assignment input).laneRawRounds := by
  have laneValid :
      Valid input.transcriptBase constants assignment
        (KSplitNcTranscript.feLaneReplay input).builder :=
    valid.of_extends (feLane_to_output_extends input)
  have row :=
    decoded_feRow profile constants assignment constantWire input
      valid initial
  have rowState := congrArg Prod.snd row
  simp only at rowState
  have decoded :=
    decoded_replayRounds input.transcriptBase constants assignment
      constantWire .feRound (List.ofFn input.fe.laneRounds)
      (KSplitNcTranscript.feRowReplay input).builder laneValid
  rw [feLane_payloads_eq assignment input, rowState] at decoded
  rw [valueReplayPayloads_eq_fe
    (domains := domains) constants
    (valueSerialization assignment input)
    (semanticFeInitial profile constants assignment input)
    (feCertificate assignment input).laneRawRounds] at decoded
  simpa only [KSplitNcTranscript.feLaneReplay,
    semanticFeMachine, valueSchedule] using decoded

structure FeReplayAgrees
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) : Prop where
  point :
    decodedFePoint assignment input =
      (semanticFeExecution profile constants assignment input).challengePoint
  state :
    decodedBuilder assignment
        (KSplitNcTranscript.feLaneReplay input).builder =
      (semanticFeExecution profile constants assignment input).finalState

theorem decoded_fe
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (input : KSplitNcTranscript.Input polynomialInput domains)
    (valid :
      Valid input.transcriptBase constants assignment
        (KSplitNcTranscript.outputBuilder input))
    (initial :
      decodedColumns assignment input.fe.initial =
        semanticFeInitial profile constants assignment input) :
    FeReplayAgrees profile constants assignment input := by
  let machine := semanticFeMachine profile constants assignment input
  let pre := semanticPre constants assignment input
  let certificate := feCertificate assignment input
  let rowResult :=
    Transcript.Fe.runRoundsFrom machine (machine.enterFe pre.state)
      certificate.rowRawRounds
  let laneResult :=
    Transcript.Fe.runRoundsFrom machine rowResult.2
      certificate.laneRawRounds
  have row :=
    decoded_feRow profile constants assignment constantWire input
      valid initial
  have lane :=
    decoded_feLane profile constants assignment constantWire input
      valid initial
  have rowValues := congrArg Prod.fst row
  have laneValues := congrArg Prod.fst lane
  have laneState := congrArg Prod.snd lane
  simp only at rowValues laneValues laneState
  have split :=
    Transcript.Fe.replay_eq_row_then_lane
      machine pre.state certificate
  have derived :=
    Transcript.Fe.derive_coordinates_finalState
      machine pre.state certificate
  change
    Transcript.Fe.runRoundsFrom machine (machine.enterFe pre.state)
        certificate.rawRounds =
      (rowResult.1 ++ laneResult.1, laneResult.2) at split
  change
    ((semanticFeExecution profile constants assignment input).challengePoint.coordinates,
      (semanticFeExecution profile constants assignment input).finalState) =
      Transcript.Fe.runRoundsFrom machine (machine.enterFe pre.state)
        certificate.rawRounds at derived
  rw [split] at derived
  change
    decodedColumnList assignment
        (KSplitNcTranscript.feRowReplay input).challenges =
      rowResult.1 at rowValues
  change
    decodedColumnList assignment
        (KSplitNcTranscript.feLaneReplay input).challenges =
      laneResult.1 at laneValues
  change
    decodedBuilder assignment
        (KSplitNcTranscript.feLaneReplay input).builder =
      laneResult.2 at laneState
  rw [← rowValues, ← laneValues, ← laneState] at derived
  have coordinateEq := congrArg Prod.fst derived
  have stateEq := congrArg Prod.snd derived
  simp only at coordinateEq stateEq
  have appendEq :
      (semanticFeExecution profile constants assignment input).challengePoint.row.coordinates ++
          (semanticFeExecution profile constants assignment input).challengePoint.lane.coordinates =
        (decodedFePoint assignment input).row.coordinates ++
          (decodedFePoint assignment input).lane.coordinates := by
    simpa only [Polynomial.Fe.Point.coordinates,
      decodedFePoint] using coordinateEq
  have componentEq :=
    List.append_inj appendEq (by
      rw [
        (semanticFeExecution profile constants assignment input).challengePoint.row.dimension,
        (decodedFePoint assignment input).row.dimension])
  refine {
    point := ?_
    state := stateEq.symm
  }
  apply Polynomial.Fe.Point.ext
  · exact cubePoint_eq_of_coordinates_eq _ _ componentEq.1.symm
  · exact cubePoint_eq_of_coordinates_eq _ _ componentEq.2.symm

theorem decoded_ncEntry
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (input : KSplitNcTranscript.Input polynomialInput domains)
    (valid :
      Valid input.transcriptBase constants assignment
        (KSplitNcTranscript.outputBuilder input))
    (initial :
      decodedColumns assignment input.fe.initial =
        semanticFeInitial profile constants assignment input) :
    decodedBuilder assignment (KSplitNcTranscript.ncEntryBuilder input) =
      (ncMachine (valueSchedule constants assignment input)).enterNc
        (semanticFeExecution profile constants assignment input).finalState := by
  have entryValid :
      Valid input.transcriptBase constants assignment
        (KSplitNcTranscript.ncEntryBuilder input) :=
    valid.of_extends (ncEntry_to_output_extends input)
  have entered :=
    decoded_absorbTagged input.transcriptBase constants assignment
      constantWire .ncEntry [] (KSplitNcTranscript.feLaneReplay input).builder
      entryValid
  have fe :=
    decoded_fe profile constants assignment constantWire input valid initial
  rw [fe.state] at entered
  simpa only [KSplitNcTranscript.ncEntryBuilder, fieldValues,
    valueSchedule, ncMachine, KSplitNcPoseidonSchedule.schedule] using entered

theorem decoded_ncBlock
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (input : KSplitNcTranscript.Input polynomialInput domains)
    (valid :
      Valid input.transcriptBase constants assignment
        (KSplitNcTranscript.outputBuilder input))
    (initial :
      decodedColumns assignment input.fe.initial =
        semanticFeInitial profile constants assignment input) :
    (decodedColumnList assignment
        (KSplitNcTranscript.ncBlockReplay input).challenges,
      decodedBuilder assignment
        (KSplitNcTranscript.ncBlockReplay input).builder) =
      Transcript.Nc.runRoundsFrom
        (ncMachine (valueSchedule constants assignment input))
        ((ncMachine (valueSchedule constants assignment input)).enterNc
          (semanticFeExecution profile constants assignment input).finalState)
        (ncCertificate assignment input).blockRounds := by
  have blockValid :
      Valid input.transcriptBase constants assignment
        (KSplitNcTranscript.ncBlockReplay input).builder :=
    valid.of_extends (ncBlock_to_output_extends input)
  have decoded :=
    decoded_replayRounds input.transcriptBase constants assignment
      constantWire .ncRound (List.ofFn input.nc.blockRounds)
      (KSplitNcTranscript.ncEntryBuilder input) blockValid
  rw [ncBlock_payloads_eq assignment input,
    decoded_ncEntry profile constants assignment constantWire input
      valid initial] at decoded
  rw [valueReplayPayloads_eq_nc
    (domains := domains) constants
    (valueSerialization assignment input)
    (ncCertificate assignment input).blockRounds] at decoded
  simpa only [KSplitNcTranscript.ncBlockReplay,
    valueSchedule] using decoded

theorem decoded_ncLane
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (input : KSplitNcTranscript.Input polynomialInput domains)
    (valid :
      Valid input.transcriptBase constants assignment
        (KSplitNcTranscript.outputBuilder input))
    (initial :
      decodedColumns assignment input.fe.initial =
        semanticFeInitial profile constants assignment input) :
    (decodedColumnList assignment
        (KSplitNcTranscript.ncLaneReplay input).challenges,
      decodedBuilder assignment
        (KSplitNcTranscript.ncLaneReplay input).builder) =
      Transcript.Nc.runRoundsFrom
        (ncMachine (valueSchedule constants assignment input))
        (Transcript.Nc.runRoundsFrom
          (ncMachine (valueSchedule constants assignment input))
          ((ncMachine (valueSchedule constants assignment input)).enterNc
            (semanticFeExecution profile constants assignment input).finalState)
          (ncCertificate assignment input).blockRounds).2
        (ncCertificate assignment input).laneRounds := by
  have laneValid :
      Valid input.transcriptBase constants assignment
        (KSplitNcTranscript.ncLaneReplay input).builder :=
    valid.of_extends (output_extends input)
  have block :=
    decoded_ncBlock profile constants assignment constantWire input
      valid initial
  have blockState := congrArg Prod.snd block
  simp only at blockState
  have decoded :=
    decoded_replayRounds input.transcriptBase constants assignment
      constantWire .ncRound (List.ofFn input.nc.laneRounds)
      (KSplitNcTranscript.ncBlockReplay input).builder laneValid
  rw [ncLane_payloads_eq assignment input, blockState] at decoded
  rw [valueReplayPayloads_eq_nc
    (domains := domains) constants
    (valueSerialization assignment input)
    (ncCertificate assignment input).laneRounds] at decoded
  simpa only [KSplitNcTranscript.ncLaneReplay,
    valueSchedule] using decoded

structure NcReplayAgrees
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) : Prop where
  point :
    decodedNcPoint assignment input =
      (semanticNcExecution profile constants assignment input).challengePoint
  state :
    decodedBuilder assignment
        (KSplitNcTranscript.ncLaneReplay input).builder =
      (semanticNcExecution profile constants assignment input).finalState

theorem decoded_nc
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (input : KSplitNcTranscript.Input polynomialInput domains)
    (valid :
      Valid input.transcriptBase constants assignment
        (KSplitNcTranscript.outputBuilder input))
    (initial :
      decodedColumns assignment input.fe.initial =
        semanticFeInitial profile constants assignment input) :
    NcReplayAgrees profile constants assignment input := by
  let machine := ncMachine (valueSchedule constants assignment input)
  let certificate := ncCertificate assignment input
  let initialState :=
    (semanticFeExecution profile constants assignment input).finalState
  let blockResult :=
    Transcript.Nc.runRoundsFrom machine (machine.enterNc initialState)
      certificate.blockRounds
  let laneResult :=
    Transcript.Nc.runRoundsFrom machine blockResult.2
      certificate.laneRounds
  have block :=
    decoded_ncBlock profile constants assignment constantWire input
      valid initial
  have lane :=
    decoded_ncLane profile constants assignment constantWire input
      valid initial
  have blockValues := congrArg Prod.fst block
  have laneValues := congrArg Prod.fst lane
  have laneState := congrArg Prod.snd lane
  simp only at blockValues laneValues laneState
  have split :=
    Transcript.Nc.BlockLane.replay_eq_block_then_lane
      machine initialState certificate
  have derived :=
    Transcript.Nc.BlockLane.derive_coordinates_finalState
      machine initialState certificate
  change
    Transcript.Nc.runRoundsFrom machine (machine.enterNc initialState)
        certificate.rawRounds =
      (blockResult.1 ++ laneResult.1, laneResult.2) at split
  change
    ((semanticNcExecution profile constants assignment input).challengePoint.coordinates,
      (semanticNcExecution profile constants assignment input).finalState) =
      Transcript.Nc.runRoundsFrom machine (machine.enterNc initialState)
        certificate.rawRounds at derived
  rw [split] at derived
  change
    decodedColumnList assignment
        (KSplitNcTranscript.ncBlockReplay input).challenges =
      blockResult.1 at blockValues
  change
    decodedColumnList assignment
        (KSplitNcTranscript.ncLaneReplay input).challenges =
      laneResult.1 at laneValues
  change
    decodedBuilder assignment
        (KSplitNcTranscript.ncLaneReplay input).builder =
      laneResult.2 at laneState
  rw [← blockValues, ← laneValues, ← laneState] at derived
  have coordinateEq := congrArg Prod.fst derived
  have stateEq := congrArg Prod.snd derived
  simp only at coordinateEq stateEq
  have appendEq :
      (semanticNcExecution profile constants assignment input).challengePoint.block.coordinates ++
          (semanticNcExecution profile constants assignment input).challengePoint.lane.coordinates =
        (decodedNcPoint assignment input).block.coordinates ++
          (decodedNcPoint assignment input).lane.coordinates := by
    simpa only [Polynomial.Nc.BlockLane.Point.coordinates,
      decodedNcPoint] using coordinateEq
  have componentEq :=
    List.append_inj appendEq (by
      rw [
        (semanticNcExecution profile constants assignment input).challengePoint.block.dimension,
        (decodedNcPoint assignment input).block.dimension])
  refine {
    point := ?_
    state := stateEq.symm
  }
  apply Polynomial.Nc.BlockLane.Point.ext
  · exact cubePoint_eq_of_coordinates_eq _ _ componentEq.1.symm
  · exact cubePoint_eq_of_coordinates_eq _ _ componentEq.2.symm

theorem decoded_output
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (input : KSplitNcTranscript.Input polynomialInput domains)
    (valid :
      Valid input.transcriptBase constants assignment
        (KSplitNcTranscript.outputBuilder input))
    (initial :
      decodedColumns assignment input.fe.initial =
        semanticFeInitial profile constants assignment input) :
    decodedBuilder assignment (KSplitNcTranscript.outputBuilder input) =
      (valueSchedule constants assignment input).absorbOutput
        (semanticNcExecution profile constants assignment input).finalState
        (zeroOutput shape) := by
  have decoded :=
    decoded_absorbTagged input.transcriptBase constants assignment
      constantWire .output input.outputFields
      (KSplitNcTranscript.ncLaneReplay input).builder valid
  have nc :=
    decoded_nc profile constants assignment constantWire input valid initial
  rw [nc.state] at decoded
  simpa only [KSplitNcTranscript.outputBuilder, valueSchedule,
    KSplitNcPoseidonSchedule.schedule, valueSerialization,
    zeroOutput] using decoded

/-! ## Static projections into the numeric claimed-chain rows -/

theorem feAgrees
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    KSplitNcFeRows.Agrees
      (KSplitNcTranscript.numericColumns input).fe assignment
      (decodedColumns assignment input.fe.initial)
      (decodedColumns assignment input.fe.terminal)
      (decodedFePoint assignment input)
      (feCertificate assignment input) := by
  refine {
    initial := rfl
    rowRounds := ?_
    rowChallenges := rfl
    boundary := rfl
    laneRounds := ?_
    laneChallenges := rfl
    terminal := rfl
  }
  · simpa only [KSplitNcTranscript.numericColumns,
      KSplitNcFeRows.Columns.rowSource,
      SourceColumns.paperRounds] using
        (feCertificate_rowRounds assignment input).symm
  · simpa only [KSplitNcTranscript.numericColumns,
      KSplitNcFeRows.Columns.laneSource,
      SourceColumns.paperRounds] using
        (feCertificate_laneRounds assignment input).symm

theorem ncAgrees
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    KSplitNcNcRows.Agrees
      (KSplitNcTranscript.numericColumns input).nc assignment
      (decodedColumns assignment input.nc.initial)
      (decodedColumns assignment input.nc.terminal)
      (decodedNcPoint assignment input)
      (ncCertificate assignment input) := by
  refine {
    initial := rfl
    rounds := ?_
    challenges := ?_
    terminal := rfl
  }
  · simpa only [KSplitNcTranscript.numericColumns,
      SourceColumns.paperRounds,
      Transcript.Nc.BlockLane.Certificate.toSumCheck,
      Transcript.Nc.BlockLane.Certificate.rawRounds,
      List.map_append] using
        (ncCertificate_rounds assignment input).symm
  · simp only [KSplitNcTranscript.numericColumns,
      KSplitNcTranscript.replay,
      SourceColumns.paperChallenges,
      Polynomial.Nc.BlockLane.Point.coordinates,
      decodedNcPoint, decodedColumnList,
      List.map_append]
    rfl

end Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptPhases
