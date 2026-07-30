import Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptControl
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexCount

/-!
Contract: count the complete Split-NC transcript replay without constructing
its serialized field lists or symbolic permutation entries.

Assurance tier: model-level canonical encoding.

Owns: a compact control replay and its exact refinement to the physical
`KSplitNcTranscript` builder.

Does not own: statement or output serialization authority, field values,
physical columns, verifier acceptance, or security.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptCount

open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

private abbrev Control := SymbolicDuplexCount.Control

/-- Count one tagged payload. The tag and payload length add two fields. -/
def absorbTagged (payloadLength : Nat) (control : Control) : Control :=
  SymbolicDuplexCount.absorbManyFast (payloadLength + 2) control

/-- Count one extension-field squeeze. -/
def squeeze (control : Control) : Control :=
  SymbolicDuplexCount.gate control

/-- Count an exact vector of extension-field squeezes. -/
def squeezeMany : Nat → Control → Control
  | 0, control => control
  | count + 1, control => squeezeMany count (squeeze control)

/-- Count one empty tagged challenge family and its squeezes. -/
def sampleVector (count : Nat) (control : Control) : Control :=
  squeezeMany count (absorbTagged 0 control)

/-- Count the five core challenge families. -/
def deriveCore
    (rowVariables laneVariables blockVariables : Nat)
    (control : Control) : Control :=
  let alpha := sampleVector laneVariables control
  let betaA := sampleVector laneVariables alpha
  let betaR := sampleVector rowVariables betaA
  let gamma := squeeze (absorbTagged 0 betaR)
  sampleVector blockVariables gamma

/-- Count one fixed-degree round message and its challenge. -/
def replayRound (degree : Nat) (control : Control) : Control :=
  squeeze (absorbTagged ((degree + 1) * 2) control)

/-- Count an exact number of equal-width round messages. -/
def replayRounds (degree : Nat) : Nat → Control → Control
  | 0, control => control
  | count + 1, control =>
      replayRounds degree count (replayRound degree control)

/-- Compact control replay for the exact selected Split-NC schedule. -/
def afterOutput
    (rowVariables rowDegree laneVariables blockVariables : Nat)
    (priorAbsorbed statementLength outputLength : Nat) : Control :=
  let initial : Control := ⟨0, priorAbsorbed⟩
  let statement := absorbTagged statementLength initial
  let core :=
    deriveCore rowVariables laneVariables blockVariables statement
  let producer := squeeze (absorbTagged 0 core)
  let batch := squeeze (absorbTagged 0 producer)
  let feEntry := absorbTagged 2 batch
  let feRows := replayRounds rowDegree rowVariables feEntry
  let feLanes := replayRounds 2 laneVariables feRows
  let ncEntry := absorbTagged 0 feLanes
  let ncBlocks := replayRounds 4 blockVariables ncEntry
  let ncLanes := replayRounds 4 laneVariables ncBlocks
  absorbTagged outputLength ncLanes

theorem ofBuilder_absorbTagged
    (base : Nat) (tag : KSplitNcPoseidonSchedule.Tag)
    (payload : List LinCombNormal.LinComb)
    (builder : SymbolicDuplex.Builder) :
    SymbolicDuplexCount.ofBuilder
        (KSplitNcTranscript.absorbTagged base tag payload builder) =
      absorbTagged payload.length
        (SymbolicDuplexCount.ofBuilder builder) := by
  unfold KSplitNcTranscript.absorbTagged absorbTagged
  rw [SymbolicDuplexCount.ofBuilder_absorbMany]
  rw [SymbolicDuplexCount.absorbMany_eq_fast]
  congr 1

@[simp] theorem ofBuilder_squeeze
    (base : Nat) (builder : SymbolicDuplex.Builder) :
    SymbolicDuplexCount.ofBuilder
        (KSplitNcTranscript.squeeze base builder).2 =
      squeeze (SymbolicDuplexCount.ofBuilder builder) := by
  unfold KSplitNcTranscript.squeeze squeeze
  exact SymbolicDuplexCount.ofBuilder_gate base builder

theorem ofBuilder_squeezeMany
    (base count : Nat) (builder : SymbolicDuplex.Builder) :
    SymbolicDuplexCount.ofBuilder
        (KSplitNcTranscript.squeezeMany base count builder).2 =
      squeezeMany count (SymbolicDuplexCount.ofBuilder builder) := by
  induction count generalizing builder with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [KSplitNcTranscript.squeezeMany, squeezeMany]
      rw [inductionHypothesis, ofBuilder_squeeze]

theorem ofBuilder_sampleVector
    (base : Nat) (tag : KSplitNcPoseidonSchedule.Tag)
    (count : Nat) (builder : SymbolicDuplex.Builder) :
    SymbolicDuplexCount.ofBuilder
        (KSplitNcTranscript.sampleVector base tag count builder).2 =
      sampleVector count (SymbolicDuplexCount.ofBuilder builder) := by
  unfold KSplitNcTranscript.sampleVector sampleVector
  rw [ofBuilder_squeezeMany, ofBuilder_absorbTagged]
  rfl

theorem ofBuilder_deriveCore
    (base : Nat) (shape : SemanticShape) (domains : Domains)
    (builder : SymbolicDuplex.Builder) :
    SymbolicDuplexCount.ofBuilder
        (KSplitNcTranscript.deriveCore base shape domains builder).builder =
      deriveCore shape.rowVariables domains.laneVariables
        domains.blockVariables (SymbolicDuplexCount.ofBuilder builder) := by
  let alpha :=
    KSplitNcTranscript.sampleVector base .alpha
      domains.laneVariables builder
  let betaA :=
    KSplitNcTranscript.sampleVector base .betaA
      domains.laneVariables alpha.2
  let betaR :=
    KSplitNcTranscript.sampleVector base .betaR
      shape.rowVariables betaA.2
  let gamma :=
    KSplitNcTranscript.squeeze base
      (KSplitNcTranscript.absorbTagged base .gamma [] betaR.2)
  have alphaCount :
      SymbolicDuplexCount.ofBuilder alpha.2 =
        sampleVector domains.laneVariables
          (SymbolicDuplexCount.ofBuilder builder) :=
    ofBuilder_sampleVector base .alpha domains.laneVariables builder
  have betaACount :
      SymbolicDuplexCount.ofBuilder betaA.2 =
        sampleVector domains.laneVariables
          (sampleVector domains.laneVariables
            (SymbolicDuplexCount.ofBuilder builder)) := by
    calc
      SymbolicDuplexCount.ofBuilder betaA.2 =
          sampleVector domains.laneVariables
            (SymbolicDuplexCount.ofBuilder alpha.2) :=
        ofBuilder_sampleVector base .betaA domains.laneVariables alpha.2
      _ = _ := congrArg (sampleVector domains.laneVariables) alphaCount
  have betaRCount :
      SymbolicDuplexCount.ofBuilder betaR.2 =
        sampleVector shape.rowVariables
          (sampleVector domains.laneVariables
            (sampleVector domains.laneVariables
              (SymbolicDuplexCount.ofBuilder builder))) := by
    calc
      SymbolicDuplexCount.ofBuilder betaR.2 =
          sampleVector shape.rowVariables
            (SymbolicDuplexCount.ofBuilder betaA.2) :=
        ofBuilder_sampleVector base .betaR shape.rowVariables betaA.2
      _ = _ := congrArg (sampleVector shape.rowVariables) betaACount
  have gammaCount :
      SymbolicDuplexCount.ofBuilder gamma.2 =
        squeeze
          (absorbTagged 0
            (sampleVector shape.rowVariables
              (sampleVector domains.laneVariables
                (sampleVector domains.laneVariables
                  (SymbolicDuplexCount.ofBuilder builder))))) := by
    calc
      SymbolicDuplexCount.ofBuilder gamma.2 =
          squeeze
            (SymbolicDuplexCount.ofBuilder
              (KSplitNcTranscript.absorbTagged base .gamma [] betaR.2)) :=
        ofBuilder_squeeze base _
      _ = squeeze
          (absorbTagged 0
            (SymbolicDuplexCount.ofBuilder betaR.2)) := by
        rw [ofBuilder_absorbTagged]
        rfl
      _ = _ := congrArg (fun control => squeeze (absorbTagged 0 control))
        betaRCount
  change
    SymbolicDuplexCount.ofBuilder
        (KSplitNcTranscript.sampleVector base .betaBlock
          domains.blockVariables gamma.2).2 =
      sampleVector domains.blockVariables
        (squeeze
          (absorbTagged 0
            (sampleVector shape.rowVariables
              (sampleVector domains.laneVariables
                (sampleVector domains.laneVariables
                  (SymbolicDuplexCount.ofBuilder builder))))))
  calc
    SymbolicDuplexCount.ofBuilder
        (KSplitNcTranscript.sampleVector base .betaBlock
          domains.blockVariables gamma.2).2 =
      sampleVector domains.blockVariables
        (SymbolicDuplexCount.ofBuilder gamma.2) :=
      ofBuilder_sampleVector base .betaBlock domains.blockVariables gamma.2
    _ = _ := congrArg (sampleVector domains.blockVariables) gammaCount

theorem ofBuilder_replayRounds
    {degree : Nat}
    (base : Nat) (tag : KSplitNcPoseidonSchedule.Tag) :
    ∀ (rounds :
        List (KFixedPhaseSemanticOccurrence.RoundColumns degree))
      (builder : SymbolicDuplex.Builder),
      SymbolicDuplexCount.ofBuilder
          (KSplitNcTranscript.replayRounds base tag rounds builder).builder =
        replayRounds degree rounds.length
          (SymbolicDuplexCount.ofBuilder builder)
  | [], _ => rfl
  | round :: rounds, builder => by
      simp only [KSplitNcTranscript.replayRounds, List.length_cons,
        replayRounds]
      rw [ofBuilder_replayRounds base tag rounds]
      apply congrArg (replayRounds degree rounds.length)
      unfold replayRound
      rw [ofBuilder_squeeze, ofBuilder_absorbTagged]
      congr 2
      exact KSplitNcTranscriptControl.roundFields_length round

/-- The compact replay has exactly the control state of the physical
transcript builder. -/
theorem replay_refines
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    SymbolicDuplexCount.ofBuilder
        (KSplitNcTranscript.replay input).afterOutput =
      afterOutput shape.rowVariables
        (SumCheck.Fe.Drow polynomialInput)
        domains.laneVariables domains.blockVariables
        input.priorAbsorbed input.statementFields.length
        input.outputFields.length := by
  have statementCount :
      SymbolicDuplexCount.ofBuilder
          (KSplitNcTranscript.statementBuilder input) =
        absorbTagged input.statementFields.length
          ⟨0, input.priorAbsorbed⟩ := by
    rw [KSplitNcTranscript.statementBuilder, ofBuilder_absorbTagged]
    rfl
  have coreCount :
      SymbolicDuplexCount.ofBuilder
          (KSplitNcTranscript.coreReplay input).builder =
        deriveCore shape.rowVariables domains.laneVariables
          domains.blockVariables
          (absorbTagged input.statementFields.length
            ⟨0, input.priorAbsorbed⟩) := by
    rw [KSplitNcTranscript.coreReplay, ofBuilder_deriveCore, statementCount]
  have producerCount :
      SymbolicDuplexCount.ofBuilder
          (KSplitNcTranscript.producerSample input).2 =
        squeeze
          (absorbTagged 0
            (deriveCore shape.rowVariables domains.laneVariables
              domains.blockVariables
              (absorbTagged input.statementFields.length
                ⟨0, input.priorAbsorbed⟩))) := by
    rw [KSplitNcTranscript.producerSample, ofBuilder_squeeze,
      ofBuilder_absorbTagged, coreCount]
    rfl
  have batchCount :
      SymbolicDuplexCount.ofBuilder
          (KSplitNcTranscript.batchSample input).2 =
        squeeze
          (absorbTagged 0
            (squeeze
              (absorbTagged 0
                (deriveCore shape.rowVariables domains.laneVariables
                  domains.blockVariables
                  (absorbTagged input.statementFields.length
                    ⟨0, input.priorAbsorbed⟩))))) := by
    rw [KSplitNcTranscript.batchSample, ofBuilder_squeeze,
      ofBuilder_absorbTagged, producerCount]
    rfl
  have feEntryCount :
      SymbolicDuplexCount.ofBuilder
          (KSplitNcTranscript.feEntryBuilder input) =
        absorbTagged 2
          (squeeze
            (absorbTagged 0
              (squeeze
                (absorbTagged 0
                      (deriveCore shape.rowVariables domains.laneVariables
                    domains.blockVariables
                    (absorbTagged input.statementFields.length
                      ⟨0, input.priorAbsorbed⟩)))))) := by
    rw [KSplitNcTranscript.feEntryBuilder, ofBuilder_absorbTagged]
    change
      absorbTagged 2
          (SymbolicDuplexCount.ofBuilder
            (KSplitNcTranscript.batchSample input).2) =
        _
    rw [batchCount]
  have feRowCount :
      SymbolicDuplexCount.ofBuilder
          (KSplitNcTranscript.feRowReplay input).builder =
        replayRounds (SumCheck.Fe.Drow polynomialInput)
          shape.rowVariables
          (absorbTagged 2
            (squeeze
              (absorbTagged 0
                (squeeze
                  (absorbTagged 0
                    (deriveCore shape.rowVariables domains.laneVariables
                      domains.blockVariables
                      (absorbTagged input.statementFields.length
                        ⟨0, input.priorAbsorbed⟩))))))) := by
    rw [KSplitNcTranscript.feRowReplay, ofBuilder_replayRounds]
    simp only [List.length_ofFn]
    rw [feEntryCount]
  have feLaneCount :
      SymbolicDuplexCount.ofBuilder
          (KSplitNcTranscript.feLaneReplay input).builder =
        replayRounds 2 domains.laneVariables
          (replayRounds (SumCheck.Fe.Drow polynomialInput)
            shape.rowVariables
            (absorbTagged 2
              (squeeze
                (absorbTagged 0
                  (squeeze
                    (absorbTagged 0
                      (deriveCore shape.rowVariables domains.laneVariables
                        domains.blockVariables
                        (absorbTagged input.statementFields.length
                          ⟨0, input.priorAbsorbed⟩)))))))) := by
    rw [KSplitNcTranscript.feLaneReplay, ofBuilder_replayRounds]
    simp only [List.length_ofFn]
    rw [feRowCount]
  have ncEntryCount :
      SymbolicDuplexCount.ofBuilder
          (KSplitNcTranscript.ncEntryBuilder input) =
        absorbTagged 0
          (replayRounds 2 domains.laneVariables
            (replayRounds (SumCheck.Fe.Drow polynomialInput)
              shape.rowVariables
              (absorbTagged 2
                (squeeze
                  (absorbTagged 0
                    (squeeze
                      (absorbTagged 0
                        (deriveCore shape.rowVariables
                          domains.laneVariables domains.blockVariables
                          (absorbTagged input.statementFields.length
                            ⟨0, input.priorAbsorbed⟩))))))))) := by
    rw [KSplitNcTranscript.ncEntryBuilder, ofBuilder_absorbTagged,
      feLaneCount]
    rfl
  have ncBlockCount :
      SymbolicDuplexCount.ofBuilder
          (KSplitNcTranscript.ncBlockReplay input).builder =
        replayRounds 4 domains.blockVariables
          (absorbTagged 0
            (replayRounds 2 domains.laneVariables
              (replayRounds (SumCheck.Fe.Drow polynomialInput)
                shape.rowVariables
                (absorbTagged 2
                  (squeeze
                    (absorbTagged 0
                      (squeeze
                        (absorbTagged 0
                          (deriveCore shape.rowVariables
                            domains.laneVariables domains.blockVariables
                            (absorbTagged input.statementFields.length
                              ⟨0, input.priorAbsorbed⟩)))))))))) := by
    rw [KSplitNcTranscript.ncBlockReplay, ofBuilder_replayRounds]
    simp only [List.length_ofFn]
    rw [ncEntryCount]
  have ncLaneCount :
      SymbolicDuplexCount.ofBuilder
          (KSplitNcTranscript.ncLaneReplay input).builder =
        replayRounds 4 domains.laneVariables
          (replayRounds 4 domains.blockVariables
            (absorbTagged 0
              (replayRounds 2 domains.laneVariables
                (replayRounds (SumCheck.Fe.Drow polynomialInput)
                  shape.rowVariables
                  (absorbTagged 2
                    (squeeze
                      (absorbTagged 0
                        (squeeze
                          (absorbTagged 0
                            (deriveCore shape.rowVariables
                              domains.laneVariables domains.blockVariables
                              (absorbTagged input.statementFields.length
                                ⟨0, input.priorAbsorbed⟩))))))))))) := by
    rw [KSplitNcTranscript.ncLaneReplay, ofBuilder_replayRounds]
    simp only [List.length_ofFn]
    rw [ncBlockCount]
  change
    SymbolicDuplexCount.ofBuilder
        (KSplitNcTranscript.outputBuilder input) =
      afterOutput shape.rowVariables (SumCheck.Fe.Drow polynomialInput)
        domains.laneVariables domains.blockVariables input.priorAbsorbed
        input.statementFields.length input.outputFields.length
  unfold KSplitNcTranscript.outputBuilder afterOutput
  rw [ofBuilder_absorbTagged, ncLaneCount]

/-- Exact compact transcript cost. -/
def cost
    (rowVariables rowDegree laneVariables blockVariables : Nat)
    (priorAbsorbed statementLength outputLength : Nat) :
    Nightstream.Implementation.Lowering.Typed.Cost :=
  let entries :=
    (afterOutput rowVariables rowDegree laneVariables blockVariables
      priorAbsorbed statementLength outputLength).entries
  ⟨entries * 352, 0, 0, entries * 352⟩

/-- Compact counting preserves the exact physical transcript cost. -/
theorem cost_eq
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    KSplitNcTranscript.cost input =
      cost shape.rowVariables (SumCheck.Fe.Drow polynomialInput)
        domains.laneVariables domains.blockVariables input.priorAbsorbed
        input.statementFields.length input.outputFields.length := by
  unfold KSplitNcTranscript.cost SymbolicDuplex.cost cost
  have refined := replay_refines input
  have entriesEqual :
      (KSplitNcTranscript.replay input).afterOutput.entries.length =
        (afterOutput shape.rowVariables
          (SumCheck.Fe.Drow polynomialInput)
          domains.laneVariables domains.blockVariables input.priorAbsorbed
          input.statementFields.length input.outputFields.length).entries :=
    congrArg SymbolicDuplexCount.Control.entries refined
  rw [entriesEqual]

end Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptCount
