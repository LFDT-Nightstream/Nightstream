import Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscript
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexPhysical

/-!
Contract: call-order closure for the Lean-owned operational Split-NC
transcript.

This module proves only that every symbolic Poseidon2 entry receives its list
position as call ID.  It deliberately makes no claim about the authority or
placement of the expressions absorbed by those calls.  That weaker invariant
is exactly what allocation coverage needs.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptCallOrder

open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexPhysical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

theorem absorbTagged
    (base : Nat) (tag : KSplitNcPoseidonSchedule.Tag)
    (payload : List LinCombNormal.LinComb)
    (builder : SymbolicDuplex.Builder)
    (ordered : CallOrdered builder) :
    CallOrdered
      (KSplitNcTranscript.absorbTagged base tag payload builder) := by
  unfold KSplitNcTranscript.absorbTagged
  exact callOrdered_absorbMany base _ builder ordered

theorem squeeze
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (ordered : CallOrdered builder) :
    CallOrdered (KSplitNcTranscript.squeeze base builder).2 := by
  unfold KSplitNcTranscript.squeeze
  exact callOrdered_gate base builder ordered

theorem squeezeMany
    (base : Nat) :
    ∀ (count : Nat) (builder : SymbolicDuplex.Builder),
      CallOrdered builder →
        CallOrdered (KSplitNcTranscript.squeezeMany base count builder).2
  | 0, _, ordered => ordered
  | count + 1, builder, ordered => by
      unfold KSplitNcTranscript.squeezeMany
      exact squeezeMany base count
        (KSplitNcTranscript.squeeze base builder).2
        (squeeze base builder ordered)

theorem sampleVector
    (base : Nat) (tag : KSplitNcPoseidonSchedule.Tag)
    (count : Nat) (builder : SymbolicDuplex.Builder)
    (ordered : CallOrdered builder) :
    CallOrdered
      (KSplitNcTranscript.sampleVector base tag count builder).2 := by
  unfold KSplitNcTranscript.sampleVector
  exact squeezeMany base count _ (absorbTagged base tag [] builder ordered)

theorem deriveCore
    (base : Nat) (shape : SemanticShape) (domains : Domains)
    (builder : SymbolicDuplex.Builder)
    (ordered : CallOrdered builder) :
    CallOrdered
      (KSplitNcTranscript.deriveCore base shape domains builder).builder := by
  let alpha :=
    KSplitNcTranscript.sampleVector base .alpha
      domains.laneVariables builder
  have alphaOrdered : CallOrdered alpha.2 :=
    sampleVector base .alpha domains.laneVariables builder ordered
  let betaA :=
    KSplitNcTranscript.sampleVector base .betaA
      domains.laneVariables alpha.2
  have betaAOrdered : CallOrdered betaA.2 :=
    sampleVector base .betaA domains.laneVariables alpha.2 alphaOrdered
  let betaR :=
    KSplitNcTranscript.sampleVector base .betaR
      shape.rowVariables betaA.2
  have betaROrdered : CallOrdered betaR.2 :=
    sampleVector base .betaR shape.rowVariables betaA.2 betaAOrdered
  let gamma :=
    KSplitNcTranscript.squeeze base
      (KSplitNcTranscript.absorbTagged base .gamma [] betaR.2)
  have gammaOrdered : CallOrdered gamma.2 :=
    squeeze base _ (absorbTagged base .gamma [] betaR.2 betaROrdered)
  let betaBlock :=
    KSplitNcTranscript.sampleVector base .betaBlock
      domains.blockVariables gamma.2
  have betaBlockOrdered : CallOrdered betaBlock.2 :=
    sampleVector base .betaBlock domains.blockVariables gamma.2 gammaOrdered
  simpa only [KSplitNcTranscript.deriveCore, alpha, betaA, betaR, gamma,
    betaBlock] using betaBlockOrdered

theorem replayRounds
    {degree : Nat}
    (base : Nat) (tag : KSplitNcPoseidonSchedule.Tag) :
    ∀ (rounds :
        List
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence.RoundColumns
            degree))
      (builder : SymbolicDuplex.Builder),
      CallOrdered builder →
        CallOrdered
          (KSplitNcTranscript.replayRounds base tag rounds builder).builder
  | [], _, ordered => ordered
  | round :: rounds, builder, ordered => by
      unfold KSplitNcTranscript.replayRounds
      let absorbed :=
        KSplitNcTranscript.absorbTagged base tag
          (KSplitNcTranscript.roundFields round) builder
      have absorbedOrdered : CallOrdered absorbed :=
        absorbTagged base tag _ builder ordered
      let sampled := KSplitNcTranscript.squeeze base absorbed
      have sampledOrdered : CallOrdered sampled.2 :=
        squeeze base absorbed absorbedOrdered
      exact replayRounds base tag rounds sampled.2 sampledOrdered

theorem initialBuilder
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    CallOrdered (KSplitNcTranscript.initialBuilder input) :=
  callOrdered_start input.priorLanes input.priorAbsorbed

theorem statementBuilder
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    CallOrdered (KSplitNcTranscript.statementBuilder input) :=
  absorbTagged input.transcriptBase .statement input.statementFields
    _ (initialBuilder input)

theorem coreReplay
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    CallOrdered (KSplitNcTranscript.coreReplay input).builder :=
  deriveCore input.transcriptBase shape domains _
    (statementBuilder input)

theorem producerSample
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    CallOrdered (KSplitNcTranscript.producerSample input).2 := by
  unfold KSplitNcTranscript.producerSample
  exact squeeze input.transcriptBase _
    (absorbTagged input.transcriptBase .producerBeta []
      (KSplitNcTranscript.coreReplay input).builder (coreReplay input))

theorem batchSample
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    CallOrdered (KSplitNcTranscript.batchSample input).2 := by
  unfold KSplitNcTranscript.batchSample
  exact squeeze input.transcriptBase _
    (absorbTagged input.transcriptBase .batchWeight []
      (KSplitNcTranscript.producerSample input).2
      (producerSample input))

theorem feEntryBuilder
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    CallOrdered (KSplitNcTranscript.feEntryBuilder input) :=
  absorbTagged input.transcriptBase .feEntry _ _
    (batchSample input)

theorem feRowReplay
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    CallOrdered (KSplitNcTranscript.feRowReplay input).builder :=
  replayRounds input.transcriptBase .feRound
    (List.ofFn input.fe.rowRounds) _ (feEntryBuilder input)

theorem feLaneReplay
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    CallOrdered (KSplitNcTranscript.feLaneReplay input).builder :=
  replayRounds input.transcriptBase .feRound
    (List.ofFn input.fe.laneRounds) _ (feRowReplay input)

theorem ncEntryBuilder
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    CallOrdered (KSplitNcTranscript.ncEntryBuilder input) :=
  absorbTagged input.transcriptBase .ncEntry [] _ (feLaneReplay input)

theorem ncBlockReplay
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    CallOrdered (KSplitNcTranscript.ncBlockReplay input).builder :=
  replayRounds input.transcriptBase .ncRound
    (List.ofFn input.nc.blockRounds) _ (ncEntryBuilder input)

theorem ncLaneReplay
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    CallOrdered (KSplitNcTranscript.ncLaneReplay input).builder :=
  replayRounds input.transcriptBase .ncRound
    (List.ofFn input.nc.laneRounds) _ (ncBlockReplay input)

theorem outputBuilder
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    CallOrdered (KSplitNcTranscript.outputBuilder input) :=
  absorbTagged input.transcriptBase .output input.outputFields _
    (ncLaneReplay input)

end Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptCallOrder
