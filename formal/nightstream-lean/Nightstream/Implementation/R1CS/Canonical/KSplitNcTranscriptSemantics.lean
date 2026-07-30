import Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscript

/-!
Contract: row-satisfaction semantics for the operational Split-NC Poseidon2
transcript.

This module proves that the symbolic replay denotes the selected value-level
`KSplitNcPoseidonSchedule`.  In particular, every challenge column consumed
by the FE/NC claimed-chain rows is decoded from the output lanes of the exact
preceding Poseidon2 squeeze.

No transcript value, challenge vector, SumCheck acceptance result, or
semantic conclusion is a premise.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptSemantics

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck
open Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

private abbrev KColumns :=
  Nightstream.Implementation.R1CS.ProjectionProgram.KColumns

abbrev ValueState := Poseidon2Duplex.State

/-- Decode one exact pair of physical challenge columns. -/
def decodedColumns (assignment : Nat → Nat) (columns : KColumns) : K :=
  ofProjection (columns.value assignment)

def decodedColumnList
    (assignment : Nat → Nat) (columns : List KColumns) : List K :=
  columns.map (decodedColumns assignment)

@[simp] theorem decodedColumnList_length
    (assignment : Nat → Nat) (columns : List KColumns) :
    (decodedColumnList assignment columns).length = columns.length := by
  simp [decodedColumnList]

/-- Evaluate a symbolic serialized field list. -/
def fieldValues
    (assignment : Nat → Nat) (fields : List LinComb) : List Nat :=
  fields.map (lcEval assignment)

@[simp] theorem fieldValues_length
    (assignment : Nat → Nat) (fields : List LinComb) :
    (fieldValues assignment fields).length = fields.length := by
  simp [fieldValues]

theorem word_eval
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (value : Nat) :
    lcEval assignment (KSplitNcTranscript.word value) =
      value % goldilocksP := by
  unfold KSplitNcTranscript.word
  rw [lcEval_eq_rawSum, rawSum_cons, constantWire]
  simp only [rawSum, List.foldl_nil, Nat.add_zero, Nat.mul_one,
    Nat.mod_mod]

theorem fieldValues_mod
    (assignment : Nat → Nat) (fields : List LinComb) :
    (fieldValues assignment fields).map
        (fun value => value % goldilocksP) =
      fieldValues assignment fields := by
  unfold fieldValues
  rw [List.map_map]
  apply List.map_congr_left
  intro field _
  simp only [Function.comp_apply]
  exact Nat.mod_eq_of_lt (lcEval_lt assignment field)

theorem fieldValues_tagged
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (tag : KSplitNcPoseidonSchedule.Tag) (payload : List LinComb) :
    fieldValues assignment (KSplitNcTranscript.taggedFields tag payload) =
      tag.code :: payload.length % goldilocksP ::
        fieldValues assignment payload := by
  unfold fieldValues KSplitNcTranscript.taggedFields
  simp only [List.map_cons]
  rw [word_eval assignment constantWire,
    word_eval assignment constantWire,
    Nat.mod_eq_of_lt (KSplitNcPoseidonSchedule.Tag.code_lt_modulus tag)]

/-! ## Physical extension -/

theorem absorbTagged_extends
    (base : Nat) (tag : KSplitNcPoseidonSchedule.Tag)
    (payload : List LinComb) (builder : SymbolicDuplex.Builder) :
    Extends builder (KSplitNcTranscript.absorbTagged base tag payload builder) :=
  absorbMany_extends base (KSplitNcTranscript.taggedFields tag payload) builder

theorem squeeze_extends
    (base : Nat) (builder : SymbolicDuplex.Builder) :
    Extends builder (KSplitNcTranscript.squeeze base builder).2 := by
  simpa only [KSplitNcTranscript.squeeze_builder] using
    squeezeK_extends base builder

theorem squeezeMany_extends (base : Nat) :
    ∀ (count : Nat) (builder : SymbolicDuplex.Builder),
      Extends builder (KSplitNcTranscript.squeezeMany base count builder).2
  | 0, builder => Extends.refl builder
  | count + 1, builder =>
      (squeeze_extends base builder).trans
        (squeezeMany_extends base count
          (KSplitNcTranscript.squeeze base builder).2)

theorem sampleVector_extends
    (base : Nat) (tag : KSplitNcPoseidonSchedule.Tag)
    (count : Nat) (builder : SymbolicDuplex.Builder) :
    Extends builder
      (KSplitNcTranscript.sampleVector base tag count builder).2 :=
  (absorbTagged_extends base tag [] builder).trans
    (squeezeMany_extends base count
      (KSplitNcTranscript.absorbTagged base tag [] builder))

/-! ## Operation semantics -/

theorem decoded_absorbTagged
    (base : Nat) (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (tag : KSplitNcPoseidonSchedule.Tag) (payload : List LinComb)
    (builder : SymbolicDuplex.Builder)
    (valid :
      Valid base constants assignment
        (KSplitNcTranscript.absorbTagged base tag payload builder)) :
    decodedBuilder assignment
        (KSplitNcTranscript.absorbTagged base tag payload builder) =
      KSplitNcPoseidonSchedule.absorbTagged constants tag
        (fieldValues assignment payload)
        (decodedBuilder assignment builder) := by
  have absorbed :=
    decodedBuilder_absorbMany base constants assignment
      (KSplitNcTranscript.taggedFields tag payload) builder valid
  have taggedValues :=
    fieldValues_tagged assignment constantWire tag payload
  change
    (KSplitNcTranscript.taggedFields tag payload).map
        (lcEval assignment) =
      tag.code :: payload.length % goldilocksP ::
        fieldValues assignment payload at taggedValues
  rw [taggedValues] at absorbed
  unfold KSplitNcPoseidonSchedule.absorbTagged
  rw [fieldValues_mod]
  simpa only [KSplitNcTranscript.absorbTagged,
    fieldValues_length] using absorbed

theorem decoded_squeeze
    (base : Nat) (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (builder : SymbolicDuplex.Builder)
    (valid :
      Valid base constants assignment
        (KSplitNcTranscript.squeeze base builder).2) :
    (decodedColumns assignment
        (KSplitNcTranscript.squeeze base builder).1,
      decodedBuilder assignment
        (KSplitNcTranscript.squeeze base builder).2) =
      KSplitNcPoseidonSchedule.squeezeK constants
        (decodedBuilder assignment builder) := by
  have raw :=
    decoded_squeezeK base constants assignment builder constantWire
      (by
        simpa only [KSplitNcTranscript.squeeze_builder] using valid)
  have mapped := congrArg
    (fun result :
        Nightstream.Implementation.R1CS.ProjectionProgram.K × ValueState =>
      (ofProjection result.1, result.2))
    raw
  have firstEq :
      decodedColumns assignment
          (KSplitNcTranscript.squeeze base builder).1 =
        ofProjection
          (decodeCarried assignment
            (SymbolicDuplex.squeezeK base builder).1) := by
    unfold decodedColumns
    rw [← decodeCarried_carried assignment
      (KSplitNcTranscript.squeeze base builder).1]
    rw [KSplitNcTranscript.squeeze_carried]
  rw [firstEq]
  simpa only [KSplitNcPoseidonSchedule.squeezeK,
    KSplitNcTranscript.squeeze_builder] using mapped

theorem decoded_squeezeMany
    (base : Nat) (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1) :
    ∀ (count : Nat) (builder : SymbolicDuplex.Builder),
      Valid base constants assignment
          (KSplitNcTranscript.squeezeMany base count builder).2 →
      (decodedColumnList assignment
          (KSplitNcTranscript.squeezeMany base count builder).1,
        decodedBuilder assignment
          (KSplitNcTranscript.squeezeMany base count builder).2) =
        KSplitNcPoseidonSchedule.squeezeManyK constants count
          (decodedBuilder assignment builder)
  | 0, _, _ => rfl
  | count + 1, builder, valid => by
      let sampled := KSplitNcTranscript.squeeze base builder
      have headValid :
          Valid base constants assignment sampled.2 :=
        valid.of_extends
          (squeezeMany_extends base count sampled.2)
      have head :=
        decoded_squeeze base constants assignment constantWire builder
          headValid
      have tail :=
        decoded_squeezeMany base constants assignment constantWire
          count sampled.2 valid
      have headValue := congrArg Prod.fst head
      have headState := congrArg Prod.snd head
      simp only at headValue headState
      rw [headState] at tail
      have tailValues := congrArg Prod.fst tail
      have tailState := congrArg Prod.snd tail
      simp only at tailValues tailState
      have tailValues' :
          List.map (decodedColumns assignment)
              (KSplitNcTranscript.squeezeMany base count
                (KSplitNcTranscript.squeeze base builder).2).1 =
            (KSplitNcPoseidonSchedule.squeezeManyK constants count
              (KSplitNcPoseidonSchedule.squeezeK constants
                (decodedBuilder assignment builder)).2).1 := by
        simpa only [decodedColumnList, sampled] using tailValues
      simp only [KSplitNcTranscript.squeezeMany,
        KSplitNcPoseidonSchedule.squeezeManyK, decodedColumnList,
        List.map_cons]
      apply Prod.ext
      · simp only
        rw [headValue, tailValues']
      · exact tailState

theorem decoded_sampleVector
    (base : Nat) (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (tag : KSplitNcPoseidonSchedule.Tag) (count : Nat)
    (builder : SymbolicDuplex.Builder)
    (valid :
      Valid base constants assignment
        (KSplitNcTranscript.sampleVector base tag count builder).2) :
    (decodedColumnList assignment
        (KSplitNcTranscript.sampleVector base tag count builder).1,
      decodedBuilder assignment
        (KSplitNcTranscript.sampleVector base tag count builder).2) =
      KSplitNcPoseidonSchedule.sampleVector constants tag count
        (decodedBuilder assignment builder) := by
  let entered :=
    KSplitNcTranscript.absorbTagged base tag [] builder
  have enteredValid :
      Valid base constants assignment entered :=
    valid.of_extends (squeezeMany_extends base count entered)
  have enteredEq :=
    decoded_absorbTagged base constants assignment constantWire tag []
      builder enteredValid
  have sampledEq :=
    decoded_squeezeMany base constants assignment constantWire count
      entered valid
  unfold KSplitNcTranscript.sampleVector
  unfold KSplitNcPoseidonSchedule.sampleVector
  rw [sampledEq, enteredEq]
  rfl

theorem sampleScalar_extends
    (base : Nat) (tag : KSplitNcPoseidonSchedule.Tag)
    (builder : SymbolicDuplex.Builder) :
    Extends builder
      (KSplitNcTranscript.squeeze base
        (KSplitNcTranscript.absorbTagged base tag [] builder)).2 :=
  (absorbTagged_extends base tag [] builder).trans
    (squeeze_extends base
      (KSplitNcTranscript.absorbTagged base tag [] builder))

theorem decoded_sampleScalar
    (base : Nat) (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (tag : KSplitNcPoseidonSchedule.Tag)
    (builder : SymbolicDuplex.Builder)
    (valid :
      Valid base constants assignment
        (KSplitNcTranscript.squeeze base
          (KSplitNcTranscript.absorbTagged base tag [] builder)).2) :
    (decodedColumns assignment
        (KSplitNcTranscript.squeeze base
          (KSplitNcTranscript.absorbTagged base tag [] builder)).1,
      decodedBuilder assignment
        (KSplitNcTranscript.squeeze base
          (KSplitNcTranscript.absorbTagged base tag [] builder)).2) =
      KSplitNcPoseidonSchedule.sampleScalar constants tag
        (decodedBuilder assignment builder) := by
  let entered := KSplitNcTranscript.absorbTagged base tag [] builder
  have enteredValid :
      Valid base constants assignment entered :=
    valid.of_extends (squeeze_extends base entered)
  have enteredEq :=
    decoded_absorbTagged base constants assignment constantWire tag []
      builder enteredValid
  have sampledEq :=
    decoded_squeeze base constants assignment constantWire entered valid
  unfold KSplitNcPoseidonSchedule.sampleScalar
  rw [sampledEq, enteredEq]
  rfl

/-! ## Round-message serialization and replay -/

theorem fieldValues_carriedFields
    (assignment : Nat → Nat) (value : Carried) :
    fieldValues assignment (KSplitNcTranscript.carriedFields value) =
      KSplitNcPoseidonSchedule.kFields
        (ofProjection (decodeCarried assignment value)) := by
  rfl

theorem fieldValues_flatMap_columns
    (assignment : Nat → Nat) :
    ∀ columns : List KColumns,
      fieldValues assignment
          (columns.flatMap fun value =>
            KSplitNcTranscript.carriedFields (carried value)) =
        (columns.map fun value => decodedColumns assignment value).flatMap
          KSplitNcPoseidonSchedule.kFields
  | [] => rfl
  | column :: columns => by
      simp only [List.flatMap_cons, List.map_cons]
      change
        fieldValues assignment
            (KSplitNcTranscript.carriedFields (carried column)) ++
              fieldValues assignment
                (columns.flatMap fun value =>
                  KSplitNcTranscript.carriedFields (carried value)) =
          KSplitNcPoseidonSchedule.kFields
              (decodedColumns assignment column) ++
            (columns.map fun value =>
              decodedColumns assignment value).flatMap
                KSplitNcPoseidonSchedule.kFields
      rw [fieldValues_carriedFields,
        decodeCarried_carried,
        fieldValues_flatMap_columns assignment columns]
      rfl

theorem fieldValues_roundFields
    {degree : Nat}
    (assignment : Nat → Nat)
    (round :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence.RoundColumns
        degree) :
    fieldValues assignment (KSplitNcTranscript.roundFields round) =
      (round.paperPolynomial assignment).coefficients.flatMap
        KSplitNcPoseidonSchedule.kFields := by
  unfold KSplitNcTranscript.roundFields
  rw [fieldValues_flatMap_columns]
  rfl

/-- Value replay for a list of already-decoded message payloads. -/
def valueReplayPayloads
    (constants : Poseidon2Schedule.Constants)
    (tag : KSplitNcPoseidonSchedule.Tag) :
    List (List Nat) → ValueState → List K × ValueState
  | [], state => ([], state)
  | payload :: payloads, state =>
      let entered :=
        KSplitNcPoseidonSchedule.absorbTagged constants tag payload state
      let sampled := KSplitNcPoseidonSchedule.squeezeK constants entered
      let rest := valueReplayPayloads constants tag payloads sampled.2
      (sampled.1 :: rest.1, rest.2)

theorem replayRounds_extends
    {degree : Nat}
    (base : Nat) (tag : KSplitNcPoseidonSchedule.Tag) :
    ∀ (rounds :
        List
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence.RoundColumns
            degree))
      (builder : SymbolicDuplex.Builder),
      Extends builder
        (KSplitNcTranscript.replayRounds base tag rounds builder).builder
  | [], builder => Extends.refl builder
  | round :: rounds, builder =>
      let entered :=
        KSplitNcTranscript.absorbTagged base tag
          (KSplitNcTranscript.roundFields round) builder
      let sampled := KSplitNcTranscript.squeeze base entered
      (absorbTagged_extends base tag
          (KSplitNcTranscript.roundFields round) builder).trans
        ((squeeze_extends base entered).trans
          (replayRounds_extends base tag rounds sampled.2))

theorem decoded_replayRounds
    {degree : Nat}
    (base : Nat) (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (tag : KSplitNcPoseidonSchedule.Tag) :
    ∀ (rounds :
        List
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence.RoundColumns
            degree))
      (builder : SymbolicDuplex.Builder),
      Valid base constants assignment
          (KSplitNcTranscript.replayRounds base tag rounds builder).builder →
      (decodedColumnList assignment
          (KSplitNcTranscript.replayRounds base tag rounds builder).challenges,
        decodedBuilder assignment
          (KSplitNcTranscript.replayRounds base tag rounds builder).builder) =
        valueReplayPayloads constants tag
          (rounds.map fun round =>
            fieldValues assignment (KSplitNcTranscript.roundFields round))
          (decodedBuilder assignment builder)
  | [], _, _ => rfl
  | round :: rounds, builder, valid => by
      let entered :=
        KSplitNcTranscript.absorbTagged base tag
          (KSplitNcTranscript.roundFields round) builder
      let sampled := KSplitNcTranscript.squeeze base entered
      have sampledValid :
          Valid base constants assignment sampled.2 :=
        valid.of_extends
          (replayRounds_extends base tag rounds sampled.2)
      have enteredValid :
          Valid base constants assignment entered :=
        sampledValid.of_extends (squeeze_extends base entered)
      have enteredEq :=
        decoded_absorbTagged base constants assignment constantWire tag
          (KSplitNcTranscript.roundFields round) builder enteredValid
      have sampledEq :=
        decoded_squeeze base constants assignment constantWire entered
          sampledValid
      have restEq :=
        decoded_replayRounds base constants assignment constantWire tag
          rounds sampled.2 valid
      have sampledValue := congrArg Prod.fst sampledEq
      have sampledState := congrArg Prod.snd sampledEq
      simp only at sampledValue sampledState
      rw [enteredEq] at sampledValue sampledState
      rw [sampledState] at restEq
      have restValues := congrArg Prod.fst restEq
      have restState := congrArg Prod.snd restEq
      simp only at restValues restState
      simp only [KSplitNcTranscript.replayRounds,
        valueReplayPayloads, decodedColumnList, List.map_cons]
      apply Prod.ext
      · simp only
        rw [sampledValue]
        congr 1
      · simpa only [sampled] using restState

/-! ## Core challenge derivation -/

structure CoreAgrees
    (assignment : Nat → Nat)
    {shape : SemanticShape} {domains : Domains}
    (symbolic : KSplitNcTranscript.Core shape domains)
    (semantic : CorePreSumcheck shape domains ValueState) : Prop where
  alpha :
    decodedColumnList assignment symbolic.alpha =
      semantic.challenges.alpha.coordinates
  betaA :
    decodedColumnList assignment symbolic.betaA =
      semantic.challenges.betaA.coordinates
  betaR :
    decodedColumnList assignment symbolic.betaR =
      semantic.challenges.betaR.coordinates
  gamma :
    decodedColumns assignment symbolic.gamma =
      semantic.challenges.gamma
  betaBlock :
    decodedColumnList assignment symbolic.betaBlock =
      semantic.challenges.betaBlock.coordinates
  state :
    decodedBuilder assignment symbolic.builder = semantic.state

theorem deriveCore_extends
    (base : Nat) (shape : SemanticShape) (domains : Domains)
    (builder : SymbolicDuplex.Builder) :
    Extends builder
      (KSplitNcTranscript.deriveCore base shape domains builder).builder := by
  let alpha :=
    KSplitNcTranscript.sampleVector base .alpha
      domains.laneVariables builder
  let betaA :=
    KSplitNcTranscript.sampleVector base .betaA
      domains.laneVariables alpha.2
  let betaR :=
    KSplitNcTranscript.sampleVector base .betaR
      shape.rowVariables betaA.2
  let gammaEntered :=
    KSplitNcTranscript.absorbTagged base .gamma [] betaR.2
  let gamma := KSplitNcTranscript.squeeze base gammaEntered
  let betaBlock :=
    KSplitNcTranscript.sampleVector base .betaBlock
      domains.blockVariables gamma.2
  have alphaExt :=
    sampleVector_extends base .alpha domains.laneVariables builder
  have betaAExt :=
    sampleVector_extends base .betaA domains.laneVariables alpha.2
  have betaRExt :=
    sampleVector_extends base .betaR shape.rowVariables betaA.2
  have gammaExt := sampleScalar_extends base .gamma betaR.2
  have betaBlockExt :=
    sampleVector_extends base .betaBlock domains.blockVariables gamma.2
  simpa only [KSplitNcTranscript.deriveCore, alpha, betaA, betaR,
    gammaEntered, gamma, betaBlock] using
      alphaExt.trans
        (betaAExt.trans
          (betaRExt.trans (gammaExt.trans betaBlockExt)))

theorem decoded_deriveCore
    (base : Nat) (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (shape : SemanticShape) (domains : Domains)
    (builder : SymbolicDuplex.Builder)
    (valid :
      Valid base constants assignment
        (KSplitNcTranscript.deriveCore base shape domains builder).builder) :
    CoreAgrees assignment
      (KSplitNcTranscript.deriveCore base shape domains builder)
      (KSplitNcPoseidonSchedule.deriveCore constants
        (decodedBuilder assignment builder)) := by
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
  let betaBlock :=
    KSplitNcTranscript.sampleVector base .betaBlock
      domains.blockVariables gamma.2
  have gammaValid :
      Valid base constants assignment gamma.2 :=
    valid.of_extends
      (sampleVector_extends base .betaBlock domains.blockVariables gamma.2)
  have betaRValid :
      Valid base constants assignment betaR.2 :=
    gammaValid.of_extends (sampleScalar_extends base .gamma betaR.2)
  have betaAValid :
      Valid base constants assignment betaA.2 :=
    betaRValid.of_extends
      (sampleVector_extends base .betaR shape.rowVariables betaA.2)
  have alphaValid :
      Valid base constants assignment alpha.2 :=
    betaAValid.of_extends
      (sampleVector_extends base .betaA domains.laneVariables alpha.2)
  have alphaEq :=
    decoded_sampleVector base constants assignment constantWire .alpha
      domains.laneVariables builder alphaValid
  have alphaValues := congrArg Prod.fst alphaEq
  have alphaState := congrArg Prod.snd alphaEq
  simp only at alphaValues alphaState
  have betaAEq :=
    decoded_sampleVector base constants assignment constantWire .betaA
      domains.laneVariables alpha.2 betaAValid
  rw [alphaState] at betaAEq
  have betaAValues := congrArg Prod.fst betaAEq
  have betaAState := congrArg Prod.snd betaAEq
  simp only at betaAValues betaAState
  have betaREq :=
    decoded_sampleVector base constants assignment constantWire .betaR
      shape.rowVariables betaA.2 betaRValid
  rw [betaAState] at betaREq
  have betaRValues := congrArg Prod.fst betaREq
  have betaRState := congrArg Prod.snd betaREq
  simp only at betaRValues betaRState
  have gammaEq :=
    decoded_sampleScalar base constants assignment constantWire .gamma
      betaR.2 gammaValid
  rw [betaRState] at gammaEq
  have gammaValue := congrArg Prod.fst gammaEq
  have gammaState := congrArg Prod.snd gammaEq
  simp only at gammaValue gammaState
  have betaBlockEq :=
    decoded_sampleVector base constants assignment constantWire .betaBlock
      domains.blockVariables gamma.2 valid
  rw [gammaState] at betaBlockEq
  have betaBlockValues := congrArg Prod.fst betaBlockEq
  have betaBlockState := congrArg Prod.snd betaBlockEq
  simp only at betaBlockValues betaBlockState
  refine {
    alpha := ?_
    betaA := ?_
    betaR := ?_
    gamma := ?_
    betaBlock := ?_
    state := ?_
  }
  · simpa only [KSplitNcTranscript.deriveCore,
      KSplitNcPoseidonSchedule.deriveCore,
      KSplitNcPoseidonSchedule.sampledPoint,
      alpha, betaA, betaR, gamma, betaBlock] using alphaValues
  · simpa only [KSplitNcTranscript.deriveCore,
      KSplitNcPoseidonSchedule.deriveCore,
      KSplitNcPoseidonSchedule.sampledPoint,
      alpha, betaA, betaR, gamma, betaBlock] using betaAValues
  · simpa only [KSplitNcTranscript.deriveCore,
      KSplitNcPoseidonSchedule.deriveCore,
      KSplitNcPoseidonSchedule.sampledPoint,
      alpha, betaA, betaR, gamma, betaBlock] using betaRValues
  · simpa only [KSplitNcTranscript.deriveCore,
      KSplitNcPoseidonSchedule.deriveCore,
      KSplitNcPoseidonSchedule.sampledPoint,
      alpha, betaA, betaR, gamma, betaBlock] using gammaValue
  · simpa only [KSplitNcTranscript.deriveCore,
      KSplitNcPoseidonSchedule.deriveCore,
      KSplitNcPoseidonSchedule.sampledPoint,
      alpha, betaA, betaR, gamma, betaBlock] using betaBlockValues
  · simpa only [KSplitNcTranscript.deriveCore,
      KSplitNcPoseidonSchedule.deriveCore,
      KSplitNcPoseidonSchedule.sampledPoint,
      alpha, betaA, betaR, gamma, betaBlock] using betaBlockState

/-! ## Full symbolic stage extension -/

theorem statement_extends
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Extends (KSplitNcTranscript.initialBuilder input)
      (KSplitNcTranscript.statementBuilder input) :=
  absorbTagged_extends input.transcriptBase .statement
    input.statementFields (KSplitNcTranscript.initialBuilder input)

theorem core_extends
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Extends (KSplitNcTranscript.statementBuilder input)
      (KSplitNcTranscript.coreReplay input).builder :=
  deriveCore_extends input.transcriptBase shape domains
    (KSplitNcTranscript.statementBuilder input)

theorem producer_extends
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Extends (KSplitNcTranscript.coreReplay input).builder
      (KSplitNcTranscript.producerSample input).2 :=
  sampleScalar_extends input.transcriptBase .producerBeta
    (KSplitNcTranscript.coreReplay input).builder

theorem batch_extends
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Extends (KSplitNcTranscript.producerSample input).2
      (KSplitNcTranscript.batchSample input).2 :=
  sampleScalar_extends input.transcriptBase .batchWeight
    (KSplitNcTranscript.producerSample input).2

theorem feEntry_extends
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Extends (KSplitNcTranscript.batchSample input).2
      (KSplitNcTranscript.feEntryBuilder input) :=
  absorbTagged_extends input.transcriptBase .feEntry
    (KSplitNcTranscript.carriedFields (carried input.fe.initial))
    (KSplitNcTranscript.batchSample input).2

theorem feRow_extends
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Extends (KSplitNcTranscript.feEntryBuilder input)
      (KSplitNcTranscript.feRowReplay input).builder :=
  replayRounds_extends input.transcriptBase .feRound
    (List.ofFn input.fe.rowRounds)
    (KSplitNcTranscript.feEntryBuilder input)

theorem feLane_extends
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Extends (KSplitNcTranscript.feRowReplay input).builder
      (KSplitNcTranscript.feLaneReplay input).builder :=
  replayRounds_extends input.transcriptBase .feRound
    (List.ofFn input.fe.laneRounds)
    (KSplitNcTranscript.feRowReplay input).builder

theorem ncEntry_extends
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Extends (KSplitNcTranscript.feLaneReplay input).builder
      (KSplitNcTranscript.ncEntryBuilder input) :=
  absorbTagged_extends input.transcriptBase .ncEntry []
    (KSplitNcTranscript.feLaneReplay input).builder

theorem ncBlock_extends
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Extends (KSplitNcTranscript.ncEntryBuilder input)
      (KSplitNcTranscript.ncBlockReplay input).builder :=
  replayRounds_extends input.transcriptBase .ncRound
    (List.ofFn input.nc.blockRounds)
    (KSplitNcTranscript.ncEntryBuilder input)

theorem ncLane_extends
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Extends (KSplitNcTranscript.ncBlockReplay input).builder
      (KSplitNcTranscript.ncLaneReplay input).builder :=
  replayRounds_extends input.transcriptBase .ncRound
    (List.ofFn input.nc.laneRounds)
    (KSplitNcTranscript.ncBlockReplay input).builder

theorem output_extends
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Extends (KSplitNcTranscript.ncLaneReplay input).builder
      (KSplitNcTranscript.outputBuilder input) :=
  absorbTagged_extends input.transcriptBase .output input.outputFields
    (KSplitNcTranscript.ncLaneReplay input).builder

theorem batch_to_output_extends
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Extends (KSplitNcTranscript.batchSample input).2
      (KSplitNcTranscript.outputBuilder input) :=
  (feEntry_extends input).trans
    ((feRow_extends input).trans
      ((feLane_extends input).trans
        ((ncEntry_extends input).trans
          ((ncBlock_extends input).trans
            ((ncLane_extends input).trans (output_extends input))))))

theorem producer_to_output_extends
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Extends (KSplitNcTranscript.producerSample input).2
      (KSplitNcTranscript.outputBuilder input) :=
  (batch_extends input).trans (batch_to_output_extends input)

theorem core_to_output_extends
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Extends (KSplitNcTranscript.coreReplay input).builder
      (KSplitNcTranscript.outputBuilder input) :=
  (producer_extends input).trans (producer_to_output_extends input)

theorem statement_to_output_extends
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Extends (KSplitNcTranscript.statementBuilder input)
      (KSplitNcTranscript.outputBuilder input) :=
  (core_extends input).trans (core_to_output_extends input)

/-! ## Value-level selected instance -/

def unitStatement : Statement Unit Unit where
  verifierKey := ()
  input := ()

def zeroOutput (shape : SemanticShape) : OutputMessage shape where
  yRing := fun _ _ _ => K.zero
  yZcol := fun _ _ => K.zero

/-- The serialization induced by one authoritative symbolic input under an
assignment.  Later call-frame decoding proves that these evaluated fields
are exactly the selected structured statement and output encodings. -/
def valueSerialization
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    KSplitNcPoseidonSchedule.Serialization Unit Unit shape where
  statementFields _ := fieldValues assignment input.statementFields
  outputFields _ := fieldValues assignment input.outputFields

def valueSchedule
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    Schedule Unit Unit shape domains ValueState :=
  KSplitNcPoseidonSchedule.schedule constants
    (valueSerialization assignment input)

def priorState
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains) : ValueState :=
  decodedBuilder assignment (KSplitNcTranscript.initialBuilder input)

structure PreAgrees
    (assignment : Nat → Nat)
    {shape : SemanticShape} {domains : Domains}
    (execution : KSplitNcTranscript.Replay shape domains)
    (semantic : PreSumcheck shape domains ValueState) : Prop where
  alpha :
    decodedColumnList assignment execution.core.alpha =
      semantic.challenges.alpha.coordinates
  betaA :
    decodedColumnList assignment execution.core.betaA =
      semantic.challenges.betaA.coordinates
  betaR :
    decodedColumnList assignment execution.core.betaR =
      semantic.challenges.betaR.coordinates
  gamma :
    decodedColumns assignment execution.core.gamma =
      semantic.challenges.gamma
  betaBlock :
    decodedColumnList assignment execution.core.betaBlock =
      semantic.challenges.betaBlock.coordinates
  producerBeta :
    decodedColumns assignment execution.producerBeta =
      semantic.challenges.producerBeta
  batchWeight :
    decodedColumns assignment execution.batchWeight =
      semantic.challenges.batchWeight
  state :
    decodedBuilder assignment execution.afterPreSumcheck =
      semantic.state

theorem decoded_preSumcheck
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (input : KSplitNcTranscript.Input polynomialInput domains)
    (valid :
      Valid input.transcriptBase constants assignment
        (KSplitNcTranscript.outputBuilder input)) :
    PreAgrees assignment (KSplitNcTranscript.replay input)
      (derivePreSumcheck
        (valueSchedule constants assignment input)
        (priorState assignment input) unitStatement) := by
  have statementValid :
      Valid input.transcriptBase constants assignment
        (KSplitNcTranscript.statementBuilder input) :=
    valid.of_extends (statement_to_output_extends input)
  have coreValid :
      Valid input.transcriptBase constants assignment
        (KSplitNcTranscript.coreReplay input).builder :=
    valid.of_extends (core_to_output_extends input)
  have producerValid :
      Valid input.transcriptBase constants assignment
        (KSplitNcTranscript.producerSample input).2 :=
    valid.of_extends (producer_to_output_extends input)
  have batchValid :
      Valid input.transcriptBase constants assignment
        (KSplitNcTranscript.batchSample input).2 :=
    valid.of_extends (batch_to_output_extends input)
  have statementEq :=
    decoded_absorbTagged input.transcriptBase constants assignment
      constantWire .statement input.statementFields
      (KSplitNcTranscript.initialBuilder input) statementValid
  have statementEq' :
      decodedBuilder assignment
          (KSplitNcTranscript.statementBuilder input) =
        KSplitNcPoseidonSchedule.absorbTagged constants .statement
          (fieldValues assignment input.statementFields)
          (decodedBuilder assignment
            (KSplitNcTranscript.initialBuilder input)) := by
    simpa only [KSplitNcTranscript.statementBuilder] using statementEq
  have coreEq :=
    decoded_deriveCore input.transcriptBase constants assignment constantWire
      shape domains (KSplitNcTranscript.statementBuilder input) coreValid
  rw [statementEq'] at coreEq
  have coreState :
      decodedBuilder assignment
          (KSplitNcTranscript.coreReplay input).builder =
        (KSplitNcPoseidonSchedule.deriveCore
          (shape := shape) (domains := domains) constants
          (KSplitNcPoseidonSchedule.absorbTagged constants .statement
            (fieldValues assignment input.statementFields)
            (decodedBuilder assignment
              (KSplitNcTranscript.initialBuilder input)))).state := by
    simpa only [KSplitNcTranscript.coreReplay] using coreEq.state
  have producerEq :=
    decoded_sampleScalar input.transcriptBase constants assignment
      constantWire .producerBeta
      (KSplitNcTranscript.coreReplay input).builder producerValid
  rw [coreState] at producerEq
  have producerValue := congrArg Prod.fst producerEq
  have producerState := congrArg Prod.snd producerEq
  simp only at producerValue producerState
  have producerState' :
      decodedBuilder assignment
          (KSplitNcTranscript.producerSample input).2 =
        (KSplitNcPoseidonSchedule.sampleScalar constants .producerBeta
          (KSplitNcPoseidonSchedule.deriveCore
            (shape := shape) (domains := domains) constants
            (KSplitNcPoseidonSchedule.absorbTagged constants .statement
              (fieldValues assignment input.statementFields)
              (decodedBuilder assignment
                (KSplitNcTranscript.initialBuilder input)))).state).2 := by
    simpa only [KSplitNcTranscript.producerSample] using producerState
  have batchEq :=
    decoded_sampleScalar input.transcriptBase constants assignment
      constantWire .batchWeight
      (KSplitNcTranscript.producerSample input).2 batchValid
  rw [producerState'] at batchEq
  have batchValue := congrArg Prod.fst batchEq
  have batchState := congrArg Prod.snd batchEq
  simp only at batchValue batchState
  refine {
    alpha := ?_
    betaA := ?_
    betaR := ?_
    gamma := ?_
    betaBlock := ?_
    producerBeta := ?_
    batchWeight := ?_
    state := ?_
  }
  · simpa only [KSplitNcTranscript.replay, valueSchedule,
      KSplitNcPoseidonSchedule.schedule, derivePreSumcheck,
      KSplitNcTranscript.coreReplay, valueSerialization,
      unitStatement] using coreEq.alpha
  · simpa only [KSplitNcTranscript.replay, valueSchedule,
      KSplitNcPoseidonSchedule.schedule, derivePreSumcheck,
      KSplitNcTranscript.coreReplay, valueSerialization,
      unitStatement] using coreEq.betaA
  · simpa only [KSplitNcTranscript.replay, valueSchedule,
      KSplitNcPoseidonSchedule.schedule, derivePreSumcheck,
      KSplitNcTranscript.coreReplay, valueSerialization,
      unitStatement] using coreEq.betaR
  · simpa only [KSplitNcTranscript.replay, valueSchedule,
      KSplitNcPoseidonSchedule.schedule, derivePreSumcheck,
      KSplitNcTranscript.coreReplay, valueSerialization,
      unitStatement] using coreEq.gamma
  · simpa only [KSplitNcTranscript.replay, valueSchedule,
      KSplitNcPoseidonSchedule.schedule, derivePreSumcheck,
      KSplitNcTranscript.coreReplay, valueSerialization,
      unitStatement] using coreEq.betaBlock
  · simpa only [KSplitNcTranscript.replay,
      KSplitNcTranscript.producerSample,
      valueSchedule, KSplitNcPoseidonSchedule.schedule,
      KSplitNcPoseidonSchedule.delayedTag,
      derivePreSumcheck, valueSerialization, unitStatement] using
        producerValue
  · simpa only [KSplitNcTranscript.replay,
      KSplitNcTranscript.batchSample,
      valueSchedule, KSplitNcPoseidonSchedule.schedule,
      KSplitNcPoseidonSchedule.delayedTag,
      derivePreSumcheck, valueSerialization, unitStatement] using
        batchValue
  · simpa only [KSplitNcTranscript.replay,
      valueSchedule, KSplitNcPoseidonSchedule.schedule,
      KSplitNcPoseidonSchedule.delayedTag,
      derivePreSumcheck, valueSerialization, unitStatement] using
        batchState

end Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptSemantics
