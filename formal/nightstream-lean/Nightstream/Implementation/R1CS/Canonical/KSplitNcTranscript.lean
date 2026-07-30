import Nightstream.Implementation.R1CS.Canonical.KSplitNcBlockLaneRows
import Nightstream.Implementation.R1CS.Canonical.KSplitNcPoseidonSchedule

/-!
Contract: the Lean-owned symbolic Poseidon2 replay for operational Split-NC.

The builder mirrors `KSplitNcPoseidonSchedule.schedule` exactly.  It owns the
physical challenge columns: core challenges, delayed challenges, every FE
round challenge, and every NC round challenge are output ports of a forced
Poseidon2 permutation.  They are not caller-selected columns.

The FE and NC message carriers are structurally exact:

* one syntax-width FE message per row coordinate;
* one three-slot FE message per lane coordinate;
* one five-slot NC message per block coordinate and per lane coordinate.

`numericColumns` constructs the claimed-chain program from those generated
challenge columns, so a later composition cannot accidentally connect the
SumCheck rows to a different challenge vector.

Does not own: evaluation semantics, the authoritative statement/output
codec, honest witness values, global call placement, or the enclosing NIFS
recipe.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscript

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

private abbrev KColumns :=
  Nightstream.Implementation.R1CS.ProjectionProgram.KColumns

/-- A verifier-known field word on the shared constant wire. -/
def word (value : Nat) : LinComb :=
  [(0, value % goldilocksP)]

/-- Canonical low/high expression order for one extension value. -/
def carriedFields (value : Carried) : List LinComb :=
  [value.low, value.high]

/-- The exact symbolic framing used by the selected value schedule. -/
def taggedFields
    (tag : KSplitNcPoseidonSchedule.Tag)
    (payload : List LinComb) : List LinComb :=
  word tag.code :: word payload.length :: payload

def absorbTagged
    (base : Nat) (tag : KSplitNcPoseidonSchedule.Tag)
    (payload : List LinComb) (builder : SymbolicDuplex.Builder) :
    SymbolicDuplex.Builder :=
  SymbolicDuplex.absorbMany base (taggedFields tag payload) builder

/-- Output columns of the forced permutation used by one squeeze. -/
def squeezeColumns (base : Nat) (builder : SymbolicDuplex.Builder) :
    KColumns :=
  let before := SymbolicDuplex.absorb base SymbolicDuplex.one builder
  let call := before.entries.length
  {
    c0 := (SymbolicDuplex.layoutAt base call).outputPort ⟨0, by decide⟩
    c1 := (SymbolicDuplex.layoutAt base call).outputPort ⟨1, by decide⟩
  }

/-- One symbolic extension squeeze, retaining the exact physical columns. -/
def squeeze (base : Nat) (builder : SymbolicDuplex.Builder) :
    KColumns × SymbolicDuplex.Builder :=
  (squeezeColumns base builder, SymbolicDuplex.gate base builder)

@[simp] theorem squeeze_builder
    (base : Nat) (builder : SymbolicDuplex.Builder) :
    (squeeze base builder).2 = (SymbolicDuplex.squeezeK base builder).2 := rfl

@[simp] theorem squeeze_carried
    (base : Nat) (builder : SymbolicDuplex.Builder) :
    carried (squeeze base builder).1 =
      (SymbolicDuplex.squeezeK base builder).1 := by
  rfl

/-- Squeeze an exact vector of extension challenges. -/
def squeezeMany (base : Nat) :
    Nat → SymbolicDuplex.Builder → List KColumns × SymbolicDuplex.Builder
  | 0, builder => ([], builder)
  | count + 1, builder =>
      let sampled := squeeze base builder
      let rest := squeezeMany base count sampled.2
      (sampled.1 :: rest.1, rest.2)

@[simp] theorem squeezeMany_length
    (base count : Nat) (builder : SymbolicDuplex.Builder) :
    (squeezeMany base count builder).1.length = count := by
  induction count generalizing builder with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [squeezeMany, List.length_cons]
      rw [inductionHypothesis]

def sampleVector
    (base : Nat) (tag : KSplitNcPoseidonSchedule.Tag)
    (count : Nat) (builder : SymbolicDuplex.Builder) :
    List KColumns × SymbolicDuplex.Builder :=
  squeezeMany base count (absorbTagged base tag [] builder)

@[simp] theorem sampleVector_length
    (base : Nat) (tag : KSplitNcPoseidonSchedule.Tag)
    (count : Nat) (builder : SymbolicDuplex.Builder) :
    (sampleVector base tag count builder).1.length = count :=
  squeezeMany_length base count _

/-- All five core challenge families, in the selected schedule order. -/
structure Core
    (shape : SemanticShape) (domains : Domains) where
  alpha : List KColumns
  alpha_length : alpha.length = domains.laneVariables
  betaA : List KColumns
  betaA_length : betaA.length = domains.laneVariables
  betaR : List KColumns
  betaR_length : betaR.length = shape.rowVariables
  gamma : KColumns
  betaBlock : List KColumns
  betaBlock_length : betaBlock.length = domains.blockVariables
  builder : SymbolicDuplex.Builder

def deriveCore
    (base : Nat) (shape : SemanticShape) (domains : Domains)
    (builder : SymbolicDuplex.Builder) : Core shape domains :=
  let alpha := sampleVector base .alpha domains.laneVariables builder
  let betaA := sampleVector base .betaA domains.laneVariables alpha.2
  let betaR := sampleVector base .betaR shape.rowVariables betaA.2
  let gamma := squeeze base (absorbTagged base .gamma [] betaR.2)
  let betaBlock :=
    sampleVector base .betaBlock domains.blockVariables gamma.2
  {
    alpha := alpha.1
    alpha_length := sampleVector_length _ _ _ _
    betaA := betaA.1
    betaA_length := sampleVector_length _ _ _ _
    betaR := betaR.1
    betaR_length := sampleVector_length _ _ _ _
    gamma := gamma.1
    betaBlock := betaBlock.1
    betaBlock_length := sampleVector_length _ _ _ _
    builder := betaBlock.2
  }

/-- Convert one exact row-layer message to low/high field expressions. -/
def roundFields
    {degree : Nat}
    (round :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence.RoundColumns
        degree) : List LinComb :=
  round.coefficients.flatMap fun columns => carriedFields (carried columns)

structure RoundReplay where
  challenges : List KColumns
  builder : SymbolicDuplex.Builder

/-- Absorb each message and squeeze its challenge before proceeding. -/
def replayRounds
    {degree : Nat}
    (base : Nat) (tag : KSplitNcPoseidonSchedule.Tag) :
    List
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence.RoundColumns
          degree) →
      SymbolicDuplex.Builder → RoundReplay
  | [], builder => ⟨[], builder⟩
  | round :: rounds, builder =>
      let absorbed := absorbTagged base tag (roundFields round) builder
      let sampled := squeeze base absorbed
      let rest := replayRounds base tag rounds sampled.2
      ⟨sampled.1 :: rest.challenges, rest.builder⟩

@[simp] theorem replayRounds_challenges_length
    {degree : Nat}
    (base : Nat) (tag : KSplitNcPoseidonSchedule.Tag)
    (rounds :
      List
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence.RoundColumns
          degree))
    (builder : SymbolicDuplex.Builder) :
    (replayRounds base tag rounds builder).challenges.length =
      rounds.length := by
  induction rounds generalizing builder with
  | nil => rfl
  | cons round rounds inductionHypothesis =>
      simp only [replayRounds, List.length_cons]
      rw [inductionHypothesis]

/-- Exact non-challenge FE columns.  Round counts are carried by finite
domains rather than list-length premises. -/
structure FeColumns
    {shape : SemanticShape}
    (input : PublicInput shape) (domains : Domains) where
  initial : KColumns
  rowRounds :
    Fin shape.rowVariables →
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence.RoundColumns
        (SumCheck.Fe.Drow input)
  boundary : KColumns
  laneRounds :
    Fin domains.laneVariables →
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence.RoundColumns
        2
  terminal : KColumns

/-- Exact non-challenge NC columns, split structurally at the block/lane cut. -/
structure NcColumns (domains : Domains) where
  initial : KColumns
  blockRounds :
    Fin domains.blockVariables →
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence.RoundColumns
        4
  laneRounds :
    Fin domains.laneVariables →
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence.RoundColumns
        4
  terminal : KColumns

/-- Complete symbolic transcript input.  Statement and output fields are
still expressions over the authoritative caller-owned prefix; no digest or
accepted proposition appears here. -/
structure Input
    {shape : SemanticShape}
    (polynomialInput : PublicInput shape) (domains : Domains) where
  transcriptBase : Nat
  priorLanes : Poseidon2Core.State
  priorAbsorbed : Nat
  statementFields : List LinComb
  outputFields : List LinComb
  fe : FeColumns polynomialInput domains
  nc : NcColumns domains

/-- Full replay record.  The four round challenge lists are retained
separately so the FE row/lane and NC block/lane cuts cannot drift. -/
structure Replay
    (shape : SemanticShape) (domains : Domains) where
  core : Core shape domains
  producerBeta : KColumns
  batchWeight : KColumns
  afterPreSumcheck : SymbolicDuplex.Builder
  feRow : RoundReplay
  feLane : RoundReplay
  ncBlock : RoundReplay
  ncLane : RoundReplay
  beforeOutput : SymbolicDuplex.Builder
  afterOutput : SymbolicDuplex.Builder

/-- Caller-owned prior transcript state as a symbolic builder. -/
def initialBuilder
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) : SymbolicDuplex.Builder :=
  SymbolicDuplex.start input.priorLanes input.priorAbsorbed

def statementBuilder
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) : SymbolicDuplex.Builder :=
  absorbTagged input.transcriptBase .statement input.statementFields
    (initialBuilder input)

def coreReplay
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) : Core shape domains :=
  deriveCore input.transcriptBase shape domains (statementBuilder input)

def producerSample
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) :
    KColumns × SymbolicDuplex.Builder :=
  squeeze input.transcriptBase
    (absorbTagged input.transcriptBase .producerBeta []
      (coreReplay input).builder)

def batchSample
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) :
    KColumns × SymbolicDuplex.Builder :=
  squeeze input.transcriptBase
    (absorbTagged input.transcriptBase .batchWeight []
      (producerSample input).2)

def feEntryBuilder
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) : SymbolicDuplex.Builder :=
  absorbTagged input.transcriptBase .feEntry
    (carriedFields (carried input.fe.initial)) (batchSample input).2

def feRowReplay
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) : RoundReplay :=
  replayRounds input.transcriptBase .feRound
    (List.ofFn input.fe.rowRounds) (feEntryBuilder input)

def feLaneReplay
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) : RoundReplay :=
  replayRounds input.transcriptBase .feRound
    (List.ofFn input.fe.laneRounds) (feRowReplay input).builder

def ncEntryBuilder
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) : SymbolicDuplex.Builder :=
  absorbTagged input.transcriptBase .ncEntry [] (feLaneReplay input).builder

def ncBlockReplay
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) : RoundReplay :=
  replayRounds input.transcriptBase .ncRound
    (List.ofFn input.nc.blockRounds) (ncEntryBuilder input)

def ncLaneReplay
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) : RoundReplay :=
  replayRounds input.transcriptBase .ncRound
    (List.ofFn input.nc.laneRounds) (ncBlockReplay input).builder

def outputBuilder
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) : SymbolicDuplex.Builder :=
  absorbTagged input.transcriptBase .output input.outputFields
    (ncLaneReplay input).builder

/-- Execute the exact selected symbolic schedule. -/
def replay
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) : Replay shape domains :=
  {
    core := coreReplay input
    producerBeta := (producerSample input).1
    batchWeight := (batchSample input).1
    afterPreSumcheck := (batchSample input).2
    feRow := feRowReplay input
    feLane := feLaneReplay input
    ncBlock := ncBlockReplay input
    ncLane := ncLaneReplay input
    beforeOutput := (ncLaneReplay input).builder
    afterOutput := outputBuilder input
  }

/-- Claimed-chain columns whose challenge coordinates are definitionally the
Poseidon2 squeeze outputs of this replay. -/
def numericColumns
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) :
    KSplitNcBlockLaneRows.Columns polynomialInput domains :=
  let execution := replay input
  {
    fe := {
      initial := input.fe.initial
      rowRounds := List.ofFn input.fe.rowRounds
      rowChallenges := execution.feRow.challenges
      rowSameLength := by
        dsimp only [execution, replay]
        unfold feRowReplay
        rw [replayRounds_challenges_length]
      boundary := input.fe.boundary
      laneRounds := List.ofFn input.fe.laneRounds
      laneChallenges := execution.feLane.challenges
      laneSameLength := by
        dsimp only [execution, replay]
        unfold feLaneReplay
        rw [replayRounds_challenges_length]
      terminal := input.fe.terminal
    }
    nc := {
      current := input.nc.initial
      rounds :=
        List.ofFn input.nc.blockRounds ++ List.ofFn input.nc.laneRounds
      challenges :=
        execution.ncBlock.challenges ++ execution.ncLane.challenges
      terminal := input.nc.terminal
      sameLength := by
        dsimp only [execution, replay]
        unfold ncBlockReplay ncLaneReplay
        simp only [List.length_append]
        rw [replayRounds_challenges_length,
          replayRounds_challenges_length]
    }
  }

/-- All Poseidon2 rows emitted by the selected replay. -/
def rows
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (constants : Poseidon2Schedule.Constants)
    (input : Input polynomialInput domains) : List Row :=
  SymbolicDuplex.rows input.transcriptBase constants (replay input).afterOutput

/-- Exact transcript cost derived from its actual permutation-entry list. -/
def cost
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : Input polynomialInput domains) : Cost :=
  SymbolicDuplex.cost (replay input).afterOutput

theorem rows_cost
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (constants : Poseidon2Schedule.Constants)
    (input : Input polynomialInput domains) :
    (rows constants input).length = (cost input).recurringRows :=
  SymbolicDuplex.cost_rows input.transcriptBase constants
    (replay input).afterOutput

end Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscript
