import Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscript
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexControl

/-!
Contract: prove that the Split-NC transcript's physical permutation count
depends only on protocol shape and serialized field counts.

Assurance tier: model-level canonical encoding.

Owns: control-state equivalence for the full transcript replay.

Does not own: serialization authority, field values, physical columns,
verifier acceptance, or security.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptControl

open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

private abbrev Equivalent := SymbolicDuplexControl.Equivalent

theorem taggedFields_length
    (tag : KSplitNcPoseidonSchedule.Tag)
    (payload : List LinCombNormal.LinComb) :
    (KSplitNcTranscript.taggedFields tag payload).length =
      payload.length + 2 := by
  simp [KSplitNcTranscript.taggedFields]

theorem absorbTagged
    (leftBase rightBase : Nat)
    (tag : KSplitNcPoseidonSchedule.Tag)
    (leftPayload rightPayload : List LinCombNormal.LinComb)
    {left right : SymbolicDuplex.Builder}
    (payloadLength : leftPayload.length = rightPayload.length)
    (equivalent : Equivalent left right) :
    Equivalent
      (KSplitNcTranscript.absorbTagged leftBase tag leftPayload left)
      (KSplitNcTranscript.absorbTagged rightBase tag rightPayload right) := by
  unfold KSplitNcTranscript.absorbTagged
  apply SymbolicDuplexControl.absorbMany
  · rw [taggedFields_length, taggedFields_length, payloadLength]
  · exact equivalent

theorem squeeze
    (leftBase rightBase : Nat)
    {left right : SymbolicDuplex.Builder}
    (equivalent : Equivalent left right) :
    Equivalent
      (KSplitNcTranscript.squeeze leftBase left).2
      (KSplitNcTranscript.squeeze rightBase right).2 := by
  rw [KSplitNcTranscript.squeeze_builder,
    KSplitNcTranscript.squeeze_builder]
  exact SymbolicDuplexControl.gate leftBase rightBase equivalent

theorem squeezeMany
    (leftBase rightBase count : Nat)
    {left right : SymbolicDuplex.Builder}
    (equivalent : Equivalent left right) :
    Equivalent
      (KSplitNcTranscript.squeezeMany leftBase count left).2
      (KSplitNcTranscript.squeezeMany rightBase count right).2 := by
  induction count generalizing left right with
  | zero =>
      exact equivalent
  | succ count inductionHypothesis =>
      simp only [KSplitNcTranscript.squeezeMany]
      apply inductionHypothesis
      exact squeeze leftBase rightBase equivalent

theorem sampleVector
    (leftBase rightBase : Nat)
    (tag : KSplitNcPoseidonSchedule.Tag)
    (count : Nat)
    {left right : SymbolicDuplex.Builder}
    (equivalent : Equivalent left right) :
    Equivalent
      (KSplitNcTranscript.sampleVector leftBase tag count left).2
      (KSplitNcTranscript.sampleVector rightBase tag count right).2 := by
  unfold KSplitNcTranscript.sampleVector
  apply squeezeMany
  exact absorbTagged leftBase rightBase tag [] [] rfl equivalent

theorem deriveCore
    (leftBase rightBase : Nat)
    (shape : SemanticShape) (domains : Domains)
    {left right : SymbolicDuplex.Builder}
    (equivalent : Equivalent left right) :
    Equivalent
      (KSplitNcTranscript.deriveCore leftBase shape domains left).builder
      (KSplitNcTranscript.deriveCore rightBase shape domains right).builder := by
  let leftAlpha :=
    KSplitNcTranscript.sampleVector leftBase .alpha
      domains.laneVariables left
  let rightAlpha :=
    KSplitNcTranscript.sampleVector rightBase .alpha
      domains.laneVariables right
  have alphaEquivalent : Equivalent leftAlpha.2 rightAlpha.2 :=
    sampleVector leftBase rightBase .alpha domains.laneVariables equivalent
  let leftBetaA :=
    KSplitNcTranscript.sampleVector leftBase .betaA
      domains.laneVariables leftAlpha.2
  let rightBetaA :=
    KSplitNcTranscript.sampleVector rightBase .betaA
      domains.laneVariables rightAlpha.2
  have betaAEquivalent : Equivalent leftBetaA.2 rightBetaA.2 :=
    sampleVector leftBase rightBase .betaA domains.laneVariables
      alphaEquivalent
  let leftBetaR :=
    KSplitNcTranscript.sampleVector leftBase .betaR
      shape.rowVariables leftBetaA.2
  let rightBetaR :=
    KSplitNcTranscript.sampleVector rightBase .betaR
      shape.rowVariables rightBetaA.2
  have betaREquivalent : Equivalent leftBetaR.2 rightBetaR.2 :=
    sampleVector leftBase rightBase .betaR shape.rowVariables betaAEquivalent
  let leftGamma :=
    KSplitNcTranscript.squeeze leftBase
      (KSplitNcTranscript.absorbTagged leftBase .gamma [] leftBetaR.2)
  let rightGamma :=
    KSplitNcTranscript.squeeze rightBase
      (KSplitNcTranscript.absorbTagged rightBase .gamma [] rightBetaR.2)
  have gammaEquivalent : Equivalent leftGamma.2 rightGamma.2 := by
    apply squeeze
    exact absorbTagged leftBase rightBase .gamma [] [] rfl betaREquivalent
  let leftBetaBlock :=
    KSplitNcTranscript.sampleVector leftBase .betaBlock
      domains.blockVariables leftGamma.2
  let rightBetaBlock :=
    KSplitNcTranscript.sampleVector rightBase .betaBlock
      domains.blockVariables rightGamma.2
  have betaBlockEquivalent : Equivalent leftBetaBlock.2 rightBetaBlock.2 :=
    sampleVector leftBase rightBase .betaBlock domains.blockVariables
      gammaEquivalent
  simpa [KSplitNcTranscript.deriveCore, leftAlpha, rightAlpha,
    leftBetaA, rightBetaA, leftBetaR, rightBetaR, leftGamma, rightGamma,
    leftBetaBlock, rightBetaBlock] using betaBlockEquivalent

theorem roundFields_length
    {degree : Nat}
    (round : KFixedPhaseSemanticOccurrence.RoundColumns degree) :
    (KSplitNcTranscript.roundFields round).length =
      (degree + 1) * 2 := by
  unfold KSplitNcTranscript.roundFields
  rw [List.length_flatMap]
  simp only [KSplitNcTranscript.carriedFields, List.length_cons,
    List.length_nil, Nat.reduceAdd]
  have sumMapConst :
      ∀ values : List ProjectionProgram.KColumns,
        (values.map fun _ => 2).sum = values.length * 2 := by
    intro values
    induction values with
    | nil => rfl
    | cons _ rest inductionHypothesis =>
        simp only [List.map_cons, List.sum_cons, List.length_cons,
          Nat.succ_mul, inductionHypothesis]
        omega
  rw [sumMapConst, round.coefficients_length]

theorem replayRounds
    {degree : Nat}
    (leftBase rightBase : Nat)
    (tag : KSplitNcPoseidonSchedule.Tag) :
    ∀ (leftRounds rightRounds :
        List (KFixedPhaseSemanticOccurrence.RoundColumns degree))
      (left right : SymbolicDuplex.Builder),
      leftRounds.length = rightRounds.length →
      Equivalent left right →
      Equivalent
        (KSplitNcTranscript.replayRounds leftBase tag leftRounds left).builder
        (KSplitNcTranscript.replayRounds rightBase tag rightRounds right).builder
  | [], [], _, _, _, equivalent => equivalent
  | [], _ :: _, _, _, lengths, _ => by cases lengths
  | _ :: _, [], _, _, lengths, _ => by cases lengths
  | leftRound :: leftRest, rightRound :: rightRest, left, right,
      lengths, equivalent => by
      simp only [List.length_cons, Nat.succ.injEq] at lengths
      simp only [KSplitNcTranscript.replayRounds]
      apply replayRounds leftBase rightBase tag leftRest rightRest
      · exact lengths
      · apply squeeze
        apply absorbTagged
        · rw [roundFields_length, roundFields_length]
        · exact equivalent

theorem replay_afterOutput
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (left right : KSplitNcTranscript.Input polynomialInput domains)
    (priorAbsorbed : left.priorAbsorbed = right.priorAbsorbed)
    (statementLength :
      left.statementFields.length = right.statementFields.length)
    (outputLength :
      left.outputFields.length = right.outputFields.length) :
    Equivalent
      (KSplitNcTranscript.replay left).afterOutput
      (KSplitNcTranscript.replay right).afterOutput := by
  have initialEquivalent :
      Equivalent
        (KSplitNcTranscript.initialBuilder left)
        (KSplitNcTranscript.initialBuilder right) := by
    exact ⟨rfl, priorAbsorbed⟩
  have statementEquivalent :
      Equivalent
        (KSplitNcTranscript.statementBuilder left)
        (KSplitNcTranscript.statementBuilder right) := by
    exact absorbTagged left.transcriptBase right.transcriptBase .statement
      left.statementFields right.statementFields statementLength
      initialEquivalent
  have coreEquivalent :
      Equivalent
        (KSplitNcTranscript.coreReplay left).builder
        (KSplitNcTranscript.coreReplay right).builder := by
    exact deriveCore left.transcriptBase right.transcriptBase shape domains
      statementEquivalent
  have producerEquivalent :
      Equivalent
        (KSplitNcTranscript.producerSample left).2
        (KSplitNcTranscript.producerSample right).2 := by
    apply squeeze
    exact absorbTagged left.transcriptBase right.transcriptBase
      .producerBeta [] [] rfl coreEquivalent
  have batchEquivalent :
      Equivalent
        (KSplitNcTranscript.batchSample left).2
        (KSplitNcTranscript.batchSample right).2 := by
    apply squeeze
    exact absorbTagged left.transcriptBase right.transcriptBase
      .batchWeight [] [] rfl producerEquivalent
  have feEntryEquivalent :
      Equivalent
        (KSplitNcTranscript.feEntryBuilder left)
        (KSplitNcTranscript.feEntryBuilder right) := by
    apply absorbTagged
    · simp [KSplitNcTranscript.carriedFields]
    · exact batchEquivalent
  have feRowEquivalent :
      Equivalent
        (KSplitNcTranscript.feRowReplay left).builder
        (KSplitNcTranscript.feRowReplay right).builder := by
    apply replayRounds
    · simp
    · exact feEntryEquivalent
  have feLaneEquivalent :
      Equivalent
        (KSplitNcTranscript.feLaneReplay left).builder
        (KSplitNcTranscript.feLaneReplay right).builder := by
    apply replayRounds
    · simp
    · exact feRowEquivalent
  have ncEntryEquivalent :
      Equivalent
        (KSplitNcTranscript.ncEntryBuilder left)
        (KSplitNcTranscript.ncEntryBuilder right) := by
    exact absorbTagged left.transcriptBase right.transcriptBase
      .ncEntry [] [] rfl feLaneEquivalent
  have ncBlockEquivalent :
      Equivalent
        (KSplitNcTranscript.ncBlockReplay left).builder
        (KSplitNcTranscript.ncBlockReplay right).builder := by
    apply replayRounds
    · simp
    · exact ncEntryEquivalent
  have ncLaneEquivalent :
      Equivalent
        (KSplitNcTranscript.ncLaneReplay left).builder
        (KSplitNcTranscript.ncLaneReplay right).builder := by
    apply replayRounds
    · simp
    · exact ncBlockEquivalent
  change
    Equivalent
      (KSplitNcTranscript.outputBuilder left)
      (KSplitNcTranscript.outputBuilder right)
  exact absorbTagged left.transcriptBase right.transcriptBase .output
    left.outputFields right.outputFields outputLength ncLaneEquivalent

/-- Equal transcript control state gives equal exact row and allocation
cost. Field values and physical bases do not enter the result. -/
theorem cost_eq
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (left right : KSplitNcTranscript.Input polynomialInput domains)
    (priorAbsorbed : left.priorAbsorbed = right.priorAbsorbed)
    (statementLength :
      left.statementFields.length = right.statementFields.length)
    (outputLength :
      left.outputFields.length = right.outputFields.length) :
    KSplitNcTranscript.cost left = KSplitNcTranscript.cost right := by
  have equivalent :=
    replay_afterOutput left right priorAbsorbed statementLength outputLength
  simp [KSplitNcTranscript.cost, SymbolicDuplex.cost,
    equivalent.entries]

/-- The numeric claimed-chain cost depends only on the indexed FE/NC round
domains and the selected polynomial degree. It does not depend on message or
challenge columns. -/
theorem numericCost_eq
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (left right : KSplitNcTranscript.Input polynomialInput domains) :
    KSplitNcBlockLaneRows.cost
        (KSplitNcTranscript.numericColumns left) =
      KSplitNcBlockLaneRows.cost
        (KSplitNcTranscript.numericColumns right) := by
  simp [KSplitNcBlockLaneRows.cost, KSplitNcFeRows.cost,
    KSplitNcNcRows.cost, KFixedPhaseSumCheck.chainCost,
    KSplitNcTranscript.numericColumns]

end Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptControl
