import Nightstream.Implementation.R1CS.Canonical.KPiCcsTranscriptSemantics
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiatShamir

/-!
Contract: refinement of the Lean-owned PiCCS Poseidon2 replay to the unchanged
paper-joint Fiat--Shamir schedule.

`KPiCcsTranscriptSemantics` proves that row satisfaction determines one exact
value-level replay.  This module packages that same replay as the abstract
`PaperJoint.FiatShamir.Oracle` selected by the paper verifier.  The paper
certificate is decoded from the fixed-width row messages; it carries no
challenge or transcript-state field.

This is an arithmetic/schedule refinement only.  Collision resistance and
challenge-distribution bounds remain separate named security obligations.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.Canonical.KPiCcsPaperFiatShamir

open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.KPiCcsTranscriptSemantics
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.SumCheck.Finite

abbrev ValueState := KPiCcsTranscriptSemantics.ValueState
abbrev ValueK := KPiCcsTranscriptSemantics.ValueK

/-- Value-level encoding of one row-layer numeric word. -/
def valueWord (value : Nat) : Nat :=
  value % goldilocksP

/-- Canonical low/high serialization of one quadratic-extension value. -/
def valueKFields (value : ValueK) : List Nat :=
  [value.c0.val, value.c1.val]

/-- Value-level serialization of one paper SumCheck message. -/
def valueRoundFields
    (roundIndex : Nat) (message : Message ValueK) : List Nat :=
  [valueWord 45, valueWord roundIndex,
    valueWord message.coefficients.length] ++
    message.coefficients.flatMap valueKFields

/-- One indexed extension-field squeeze. -/
def valueSqueezeAt
    (constants : Constants) (label index : Nat) (state : ValueState) :
    ValueK × ValueState :=
  let absorbed := Poseidon2Duplex.absorbList constants
    [valueWord label, valueWord index] state
  SymbolicDuplexSemantics.squeezeKValue constants absorbed

/-- The row-backed paper oracle.  Its context is the complete typed PiCCS
input, so statement binding happens inside `initialState`. -/
def oracle
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat) :
    FiatShamir.Oracle
      (KPiCcsTranscript.Input shape degree) ValueK ValueState shape where
  initialState input :=
    Poseidon2Duplex.absorbList constants
      (fieldValues assignment (KPiCcsTranscript.statementFields input))
      (SymbolicDuplexSemantics.decodedBuilder assignment
        (SymbolicDuplex.start input.priorLanes 0))
  absorbRound state round message :=
    Poseidon2Duplex.absorbList constants
      (valueRoundFields round.val message) state
  squeeze state label :=
    match label with
    | .alpha coordinate =>
        valueSqueezeAt constants 42 coordinate.val state
    | .gamma =>
        valueSqueezeLabel constants assignment 43 state
    | .sumcheck round =>
        valueSqueezeAt constants 46 round.val state

/-- The fixed-width row certificate projected to the paper's finite message
format in canonical round order. -/
def roundAt
    {shape : Shape} {degree : Nat}
    (input : KPiCcsTranscript.Input shape degree)
    (round : Fin shape.cubeVariables) :
    KFixedPhaseSumCheck.Round degree :=
  input.rounds.get
    ⟨round.val, by rw [input.rounds_length]; exact round.isLt⟩

def certificate
    {shape : Shape} {degree : Nat}
    (assignment : Nat → Nat)
    (input : KPiCcsTranscript.Input shape degree) :
    FiatShamir.Certificate ValueK shape where
  rounds := fun round =>
    (KFixedPhaseSumCheck.Round.polynomial
      (roundAt input round) assignment).toMessage

theorem valueWord_eq_word_eval
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (value : Nat) :
    valueWord value =
      lcEval assignment (KPiCcsTranscript.word value) := by
  unfold valueWord KPiCcsTranscript.word
  rw [lcEval_eq_rawSum, rawSum_cons, constantWire]
  simp only [rawSum, List.foldl_nil, Nat.add_zero, Nat.mul_one, Nat.mod_mod]

theorem fieldValues_carriedFields
    (assignment : Nat → Nat) (value : KMul.Carried) :
    fieldValues assignment (KPiCcsTranscript.carriedFields value) =
      valueKFields (KFixedPhaseSumCheck.decodeCarried assignment value) := by
  rfl

theorem map_flatMap_carriedFields
    (assignment : Nat → Nat) :
    ∀ values : List KMul.Carried,
      (values.flatMap KPiCcsTranscript.carriedFields).map
          (lcEval assignment) =
        (values.map (KFixedPhaseSumCheck.decodeCarried assignment)).flatMap
          valueKFields
  | [] => rfl
  | value :: rest => by
      simp only [List.flatMap_cons, List.map_append, List.map_cons]
      change
        fieldValues assignment (KPiCcsTranscript.carriedFields value) ++
            (rest.flatMap KPiCcsTranscript.carriedFields).map
              (lcEval assignment) =
          valueKFields (KFixedPhaseSumCheck.decodeCarried assignment value) ++
            (rest.map
              (KFixedPhaseSumCheck.decodeCarried assignment)).flatMap
                valueKFields
      rw [fieldValues_carriedFields,
        map_flatMap_carriedFields assignment rest]

theorem fieldValues_roundFields
    {degree : Nat}
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (roundIndex : Nat) (round : KFixedPhaseSumCheck.Round degree) :
    fieldValues assignment
        (KPiCcsTranscript.roundFields roundIndex round) =
      valueRoundFields roundIndex
        (KFixedPhaseSumCheck.Round.polynomial round assignment).toMessage := by
  unfold KPiCcsTranscript.roundFields valueRoundFields
  simp only [fieldValues, List.map_append, List.map_cons, List.map_nil,
    FixedPolynomial.toMessage, KFixedPhaseSumCheck.Round.polynomial]
  rw [← valueWord_eq_word_eval assignment constantWire 45,
    ← valueWord_eq_word_eval assignment constantWire roundIndex,
    ← valueWord_eq_word_eval assignment constantWire
      round.coefficients.length]
  simp only [List.length_map]
  rw [map_flatMap_carriedFields]

theorem fieldValues_tag_index
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (label index : Nat) :
    fieldValues assignment
        [KPiCcsTranscript.word label, KPiCcsTranscript.word index] =
      [valueWord label, valueWord index] := by
  simp only [fieldValues, List.map_cons, List.map_nil]
  rw [← valueWord_eq_word_eval assignment constantWire label,
    ← valueWord_eq_word_eval assignment constantWire index]

theorem valueSqueezeIndexedGo_one
    (constants : Constants) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (label index : Nat) (state : ValueState) :
    valueSqueezeIndexedGo constants assignment label index 1 state =
      ([ (valueSqueezeAt constants label index state).1 ],
        (valueSqueezeAt constants label index state).2) := by
  unfold valueSqueezeIndexedGo valueSqueezeAt
  rw [fieldValues_tag_index assignment constantWire]
  rfl

/-- Replay indexed labels from an explicit natural-number list. -/
def valueRunIndices
    (constants : Constants) (label : Nat) :
    List Nat → ValueState → List ValueK × ValueState
  | [], state => ([], state)
  | index :: rest, state =>
      let sampled := valueSqueezeAt constants label index state
      let tail := valueRunIndices constants label rest sampled.2
      (sampled.1 :: tail.1, tail.2)

theorem valueRunIndices_range'
    (constants : Constants) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (label : Nat) :
    ∀ index count state,
      valueRunIndices constants label (List.range' index count) state =
        valueSqueezeIndexedGo constants assignment
          label index count state
  | _, 0, _ => rfl
  | index, count + 1, state => by
      simp only [List.range'_succ, valueRunIndices,
        valueSqueezeIndexedGo]
      rw [fieldValues_tag_index assignment constantWire]
      have tailEq :=
        valueRunIndices_range' constants assignment constantWire
        label (index + 1) count
          (valueSqueezeAt constants label index state).2
      exact congrArg
        (fun tail : List ValueK × ValueState =>
          ((valueSqueezeAt constants label index state).1 :: tail.1,
            tail.2))
        tailEq

theorem canonicalFinIndices_values (count : Nat) :
    (canonicalFinIndices count).map (fun index => index.val) =
      List.range count := by
  apply List.ext_getElem
  · simp [canonicalFinIndices]
  · intro index leftBound rightBound
    simp [canonicalFinIndices]

theorem squeezeMany_alpha_agrees
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (indices : List (Fin shape.cubeVariables)) (state : ValueState) :
    FiatShamir.squeezeMany
        (oracle (shape := shape) (degree := degree) constants assignment)
        state (indices.map FiatShamir.ChallengeLabel.alpha) =
      valueRunIndices constants 42 (indices.map fun index => index.val) state := by
  induction indices generalizing state with
  | nil => rfl
  | cons index rest inductionHypothesis =>
      let sampled := valueSqueezeAt constants 42 index.val state
      have tailEq :
          FiatShamir.squeezeMany
              (oracle (shape := shape) (degree := degree)
                constants assignment)
              sampled.2
              (rest.map FiatShamir.ChallengeLabel.alpha) =
            valueRunIndices constants 42
              (rest.map fun coordinate => coordinate.val) sampled.2 := by
        exact inductionHypothesis sampled.2
      simp only [List.map_cons, FiatShamir.squeezeMany, oracle,
        valueRunIndices, sampled]
      exact congrArg
        (fun tail : List ValueK × ValueState =>
          (sampled.1 :: tail.1, tail.2))
        tailEq

theorem alphaSchedule_agrees
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1) (state : ValueState) :
    FiatShamir.squeezeMany
        (oracle (shape := shape) (degree := degree) constants assignment)
        state (FiatShamir.alphaLabels shape) =
      valueSqueezeIndexed constants assignment
        42 shape.cubeVariables state := by
  rw [FiatShamir.alphaLabels,
    squeezeMany_alpha_agrees constants assignment]
  rw [canonicalFinIndices_values, List.range_eq_range']
  exact valueRunIndices_range' constants assignment constantWire
    42 0 shape.cubeVariables state

theorem derivePreSumcheck_agrees
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (input : KPiCcsTranscript.Input shape degree) :
    let paper :=
      FiatShamir.derivePreSumcheck (oracle constants assignment) input
    let concrete := valueDerivePreSumcheck constants assignment input
    paper.alpha.coordinates = concrete.alpha ∧
      paper.gamma = concrete.gamma ∧
      paper.state = concrete.state := by
  let initial :=
    SymbolicDuplexSemantics.decodedBuilder assignment
      (SymbolicDuplex.start input.priorLanes 0)
  let statement :=
    Poseidon2Duplex.absorbList constants
      (fieldValues assignment (KPiCcsTranscript.statementFields input))
      initial
  let paperAlpha :=
    FiatShamir.squeezeMany
      (oracle (shape := shape) (degree := degree) constants assignment)
      statement (FiatShamir.alphaLabels shape)
  let concreteAlpha :=
    valueSqueezeIndexed constants assignment
      42 shape.cubeVariables statement
  have alphaPair :
      paperAlpha = concreteAlpha := by
    exact alphaSchedule_agrees constants assignment constantWire statement
  have alphaValues := congrArg Prod.fst alphaPair
  have alphaState := congrArg Prod.snd alphaPair
  unfold FiatShamir.derivePreSumcheck valueDerivePreSumcheck
  change paperAlpha.1 = concreteAlpha.1 ∧
    (valueSqueezeLabel constants assignment 43 paperAlpha.2).1 =
      (valueSqueezeLabel constants assignment 43 concreteAlpha.2).1 ∧
    (valueSqueezeLabel constants assignment 43 paperAlpha.2).2 =
      (valueSqueezeLabel constants assignment 43 concreteAlpha.2).2
  refine ⟨alphaValues, ?_⟩
  rw [alphaState]
  exact ⟨rfl, rfl⟩

theorem certificate_round
    {shape : Shape} {degree : Nat}
    (assignment : Nat → Nat)
    (input : KPiCcsTranscript.Input shape degree)
    (round : Fin shape.cubeVariables) :
    (certificate assignment input).rounds round =
      (KFixedPhaseSumCheck.Round.polynomial
        (roundAt input round) assignment).toMessage := rfl

theorem deriveRoundsFrom_agrees
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (input : KPiCcsTranscript.Input shape degree) :
    ∀ (indices : List (Fin shape.cubeVariables))
      (rounds : List (KFixedPhaseSumCheck.Round degree))
      (index : Nat) (state : ValueState),
      indices.map (roundAt input) = rounds →
      indices.map (fun coordinate => coordinate.val) =
        List.range' index rounds.length →
      FiatShamir.deriveRoundsFrom
          (oracle (shape := shape) (degree := degree) constants assignment)
          (certificate assignment input).rounds
          state indices =
        let replay :=
          valueReplayRounds constants assignment rounds index state
        (replay.challenges, replay.state)
  | [], rounds, _, _, roundsEq, _ => by
      have empty : rounds = [] := by
        simpa only [List.map_nil] using roundsEq.symm
      subst rounds
      rfl
  | coordinate :: rest, [], _, _, roundsEq, _ => by
      simp only [List.map_cons, List.cons_ne_nil] at roundsEq
  | coordinate :: rest, round :: rounds, index, state,
      roundsEq, indicesEq => by
      simp only [List.map_cons, List.cons.injEq] at roundsEq
      simp only [List.map_cons, List.length_cons, List.range'_succ,
        List.cons.injEq] at indicesEq
      have roundEq := roundsEq.1
      have restRoundsEq := roundsEq.2
      have coordinateEq := indicesEq.1
      have restIndicesEq := indicesEq.2
      subst round
      subst index
      let absorbed :=
        Poseidon2Duplex.absorbList constants
          (fieldValues assignment
            (KPiCcsTranscript.roundFields coordinate.val
              (roundAt input coordinate))) state
      let sampled :=
        valueSqueezeIndexedGo constants assignment
          46 coordinate.val 1 absorbed
      have serialized :
          Poseidon2Duplex.absorbList constants
              (valueRoundFields coordinate.val
                ((certificate assignment input).rounds coordinate)) state =
            absorbed := by
        rw [certificate_round, ← fieldValues_roundFields
          assignment constantWire coordinate.val (roundAt input coordinate)]
      have sampledEq :
          ([(valueSqueezeAt constants 46 coordinate.val absorbed).1],
              (valueSqueezeAt constants 46 coordinate.val absorbed).2) =
            sampled := by
        exact (valueSqueezeIndexedGo_one constants assignment constantWire
          46 coordinate.val absorbed).symm
      have paperStep :
          (oracle (shape := shape) (degree := degree)
              constants assignment).squeeze
              ((oracle (shape := shape) (degree := degree)
                constants assignment).absorbRound state coordinate
                ((certificate assignment input).rounds coordinate))
              (.sumcheck coordinate) =
            valueSqueezeAt constants 46 coordinate.val absorbed := by
        change valueSqueezeAt constants 46 coordinate.val
            (Poseidon2Duplex.absorbList constants
              (valueRoundFields coordinate.val
                ((certificate assignment input).rounds coordinate)) state) =
          valueSqueezeAt constants 46 coordinate.val absorbed
        rw [serialized]
      have tail :=
        deriveRoundsFrom_agrees constants assignment constantWire input
          rest rounds (coordinate.val + 1)
          (valueSqueezeAt constants 46 coordinate.val absorbed).2
          restRoundsEq restIndicesEq
      simp only [FiatShamir.deriveRoundsFrom]
      rw [paperStep, tail]
      simp only [valueReplayRounds]
      have sampledState :=
        congrArg Prod.snd sampledEq
      have sampledValues :=
        congrArg Prod.fst sampledEq
      simp only at sampledState sampledValues
      rw [← sampledValues, ← sampledState]
      rfl

theorem canonicalRoundList
    {shape : Shape} {degree : Nat}
    (input : KPiCcsTranscript.Input shape degree) :
    (canonicalFinIndices shape.cubeVariables).map (roundAt input) =
      input.rounds := by
  apply List.ext_getElem
  · simp only [List.length_map, canonicalFinIndices_length,
      input.rounds_length]
  · intro index leftBound rightBound
    simp only [List.getElem_map]
    have sourceBound :
        index < (canonicalFinIndices shape.cubeVariables).length := by
      simpa only [List.length_map] using leftBound
    change roundAt input
        ((canonicalFinIndices shape.cubeVariables)[index]'sourceBound) =
      input.rounds[index]
    simp only [canonicalFinIndices, List.getElem_ofFn, roundAt]
    congr 1

theorem deriveRounds_agrees
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (input : KPiCcsTranscript.Input shape degree)
    (state : ValueState) :
    FiatShamir.deriveRoundsFrom
        (oracle (shape := shape) (degree := degree) constants assignment)
        (certificate assignment input).rounds
        state (canonicalFinIndices shape.cubeVariables) =
      let replay :=
        valueReplayRounds constants assignment input.rounds 0 state
      (replay.challenges, replay.state) := by
  apply deriveRoundsFrom_agrees constants assignment constantWire input
  · exact canonicalRoundList input
  · rw [canonicalFinIndices_values, input.rounds_length,
      List.range_eq_range']

theorem derive_alpha_agrees
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (input : KPiCcsTranscript.Input shape degree) :
    (FiatShamir.derive
        (oracle (shape := shape) (degree := degree) constants assignment)
        input (certificate assignment input)).alpha.coordinates =
      (valueReplay constants assignment input).alpha := by
  change
    (FiatShamir.derivePreSumcheck
      (oracle (shape := shape) (degree := degree) constants assignment)
      input).alpha.coordinates =
      (valueDerivePreSumcheck constants assignment input).alpha
  exact (derivePreSumcheck_agrees constants assignment constantWire input).1

theorem derive_gamma_agrees
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (input : KPiCcsTranscript.Input shape degree) :
    (FiatShamir.derive
        (oracle (shape := shape) (degree := degree) constants assignment)
        input (certificate assignment input)).gamma =
      (valueReplay constants assignment input).gamma := by
  change
    (FiatShamir.derivePreSumcheck
      (oracle (shape := shape) (degree := degree) constants assignment)
      input).gamma =
      (valueDerivePreSumcheck constants assignment input).gamma
  exact (derivePreSumcheck_agrees constants assignment constantWire input).2.1

theorem derive_roundPoint_agrees
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (input : KPiCcsTranscript.Input shape degree) :
    (FiatShamir.derive
        (oracle (shape := shape) (degree := degree) constants assignment)
        input (certificate assignment input)).roundPoint.coordinates =
      (valueReplay constants assignment input).point := by
  let paperPre :=
    FiatShamir.derivePreSumcheck
      (oracle (shape := shape) (degree := degree) constants assignment)
      input
  let concretePre := valueDerivePreSumcheck constants assignment input
  change
    (FiatShamir.deriveRoundsFrom
      (oracle (shape := shape) (degree := degree) constants assignment)
      (certificate assignment input).rounds paperPre.state
      (canonicalFinIndices shape.cubeVariables)).1 =
      (valueReplayRounds constants assignment input.rounds
        0 concretePre.state).challenges
  have preState :
      paperPre.state = concretePre.state :=
    (derivePreSumcheck_agrees constants assignment constantWire input).2.2
  have roundsEq :=
    deriveRounds_agrees constants assignment constantWire input
      paperPre.state
  rw [preState] at roundsEq
  rw [preState]
  rw [roundsEq]

theorem derive_finalState_agrees
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (input : KPiCcsTranscript.Input shape degree) :
    (FiatShamir.derive
        (oracle (shape := shape) (degree := degree) constants assignment)
        input (certificate assignment input)).finalState =
      (valueReplay constants assignment input).beforeOutput := by
  let paperPre :=
    FiatShamir.derivePreSumcheck
      (oracle (shape := shape) (degree := degree) constants assignment)
      input
  let concretePre := valueDerivePreSumcheck constants assignment input
  change
    (FiatShamir.deriveRoundsFrom
      (oracle (shape := shape) (degree := degree) constants assignment)
      (certificate assignment input).rounds paperPre.state
      (canonicalFinIndices shape.cubeVariables)).2 =
      (valueReplayRounds constants assignment input.rounds
        0 concretePre.state).state
  have preState :
      paperPre.state = concretePre.state :=
    (derivePreSumcheck_agrees constants assignment constantWire input).2.2
  have roundsEq :=
    deriveRounds_agrees constants assignment constantWire input
      paperPre.state
  rw [preState] at roundsEq
  rw [preState]
  rw [roundsEq]

theorem derive_agrees
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (input : KPiCcsTranscript.Input shape degree) :
    let paper :=
      FiatShamir.derive
        (oracle (shape := shape) (degree := degree) constants assignment)
        input (certificate assignment input)
    let concrete := valueReplay constants assignment input
    paper.alpha.coordinates = concrete.alpha ∧
      paper.gamma = concrete.gamma ∧
      paper.roundPoint.coordinates = concrete.point ∧
      paper.finalState = concrete.beforeOutput := by
  exact ⟨derive_alpha_agrees constants assignment constantWire input,
    derive_gamma_agrees constants assignment constantWire input,
    derive_roundPoint_agrees constants assignment constantWire input,
    derive_finalState_agrees constants assignment constantWire input⟩

/-- Headline paper-schedule refinement: satisfying the emitted transcript rows
forces exactly the alpha, gamma, and SumCheck point derived by the unchanged
paper Fiat--Shamir machine.  The outgoing state remains explicit because the
paper machine stops immediately before the output message is absorbed. -/
theorem rows_derive_paper_schedule
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (input : KPiCcsTranscript.Input shape degree)
    (residues : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Nightstream.Implementation.R1CS.Satisfies
        (KPiCcsTranscript.rows constants input) assignment) :
    let paper :=
      FiatShamir.derive
        (oracle (shape := shape) (degree := degree) constants assignment)
        input (certificate assignment input)
    (KPiCcsTranscript.replay input).alpha.map (decoded assignment) =
        paper.alpha.coordinates ∧
      decoded assignment (KPiCcsTranscript.replay input).gamma =
        paper.gamma ∧
      (KPiCcsTranscript.replay input).point.map (decoded assignment) =
        paper.roundPoint.coordinates ∧
      SymbolicDuplexSemantics.decodedBuilder assignment
          (KPiCcsTranscript.replay input).beforeOutput =
        paper.finalState ∧
      SymbolicDuplexSemantics.decodedBuilder assignment
          (KPiCcsTranscript.replay input).afterOutput =
        (valueReplay constants assignment input).afterOutput := by
  have physical :=
    rows_replay_semantics constants assignment input residues
      constantWire satisfied
  have paper :=
    derive_agrees constants assignment constantWire input
  exact
    ⟨physical.1.trans paper.1.symm,
      physical.2.1.trans paper.2.1.symm,
      physical.2.2.1.trans paper.2.2.1.symm,
      physical.2.2.2.1.trans paper.2.2.2.symm,
      physical.2.2.2.2⟩

end Nightstream.Implementation.R1CS.Canonical.KPiCcsPaperFiatShamir
