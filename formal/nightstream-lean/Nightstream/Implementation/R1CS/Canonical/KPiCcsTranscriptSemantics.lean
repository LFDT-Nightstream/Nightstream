import Nightstream.Implementation.R1CS.Canonical.KPiCcsTranscript
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiatShamir

/-!
Contract: row-satisfaction refinement of the concrete paper-joint PiCCS
transcript.

`KPiCcsTranscript` fixes the serialization and constructs the symbolic duplex
program.  This module gives that program its value-level meaning.  In
particular, alpha, gamma, and every SumCheck point coordinate are decoded from
the Poseidon2 states forced by the emitted rows; none is a free certificate
field or a caller-supplied transcript fact.

The first layer below mirrors the selected serialization exactly.  The final
layer packages that replay as the unchanged paper `FiatShamir.Oracle`
schedule.  Keeping the two layers separate makes serialization mistakes
observable rather than hiding them in an oracle definition.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.Canonical.KPiCcsTranscriptSemantics

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev ValueState := Poseidon2Duplex.State
abbrev ValueK := K

/-- Decode one carried transcript expression to the concrete row-layer
quadratic extension. -/
def decoded (assignment : Nat → Nat) (value : KMul.Carried) : ValueK :=
  KFixedPhaseSumCheck.decodeCarried assignment value

/-- Evaluate one serialized base-field list under the authoritative
assignment. -/
def fieldValues (assignment : Nat → Nat) (fields : List LinCombNormal.LinComb) :
    List Nat :=
  fields.map (lcEval assignment)

theorem fieldValues_cons
    (assignment : Nat → Nat) (head : LinCombNormal.LinComb)
    (tail : List LinCombNormal.LinComb) :
    fieldValues assignment (head :: tail) =
      lcEval assignment head :: fieldValues assignment tail := rfl

/-! ## Value-level replay of the selected serialization -/

def valueSqueezeLabel (constants : Constants) (assignment : Nat → Nat)
    (label : Nat) (state : ValueState) : ValueK × ValueState :=
  SymbolicDuplexSemantics.squeezeKValue constants
    (Poseidon2Duplex.absorbElem constants
      (lcEval assignment (KPiCcsTranscript.word label)) state)

def valueSqueezeIndexedGo
    (constants : Constants) (assignment : Nat → Nat)
    (label index : Nat) :
    Nat → ValueState → List ValueK × ValueState
  | 0, state => ([], state)
  | remaining + 1, state =>
      let absorbed := Poseidon2Duplex.absorbList constants
        (fieldValues assignment
          [KPiCcsTranscript.word label, KPiCcsTranscript.word index])
        state
      let sampled := SymbolicDuplexSemantics.squeezeKValue constants absorbed
      let tail := valueSqueezeIndexedGo constants assignment label
        (index + 1) remaining sampled.2
      (sampled.1 :: tail.1, tail.2)

def valueSqueezeIndexed
    (constants : Constants) (assignment : Nat → Nat)
    (label count : Nat) (state : ValueState) :
    List ValueK × ValueState :=
  valueSqueezeIndexedGo constants assignment label 0 count state

theorem valueSqueezeIndexedGo_length
    (constants : Constants) (assignment : Nat → Nat)
    (label index count : Nat) (state : ValueState) :
    (valueSqueezeIndexedGo constants assignment label index count state).1.length =
      count := by
  induction count generalizing index state with
  | zero => rfl
  | succ remaining inductionHypothesis =>
      simp only [valueSqueezeIndexedGo, List.length_cons]
      rw [inductionHypothesis]

theorem valueSqueezeIndexed_length
    (constants : Constants) (assignment : Nat → Nat)
    (label count : Nat) (state : ValueState) :
    (valueSqueezeIndexed constants assignment label count state).1.length =
      count :=
  valueSqueezeIndexedGo_length constants assignment label 0 count state

structure ValuePreSumcheck (shape : Shape) where
  alpha : List ValueK
  alpha_length : alpha.length = shape.cubeVariables
  gamma : ValueK
  state : ValueState

def valueDerivePreSumcheck
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (input : KPiCcsTranscript.Input shape degree) :
    ValuePreSumcheck shape :=
  let initial :=
    decodedBuilder assignment
      (SymbolicDuplex.start input.priorLanes 0)
  let statement :=
    Poseidon2Duplex.absorbList constants
      (fieldValues assignment (KPiCcsTranscript.statementFields input))
      initial
  let alpha :=
    valueSqueezeIndexed constants assignment 42 shape.cubeVariables statement
  let gamma :=
    valueSqueezeLabel constants assignment 43 alpha.2
  { alpha := alpha.1
    alpha_length := valueSqueezeIndexed_length _ _ _ _ _
    gamma := gamma.1
    state := gamma.2 }

structure ValueRoundReplay where
  challenges : List ValueK
  state : ValueState

def valueReplayRounds
    {degree : Nat} (constants : Constants) (assignment : Nat → Nat) :
    List (KFixedPhaseSumCheck.Round degree) → Nat → ValueState →
      ValueRoundReplay
  | [], _, state => { challenges := [], state }
  | round :: rest, index, state =>
      let absorbed := Poseidon2Duplex.absorbList constants
        (fieldValues assignment
          (KPiCcsTranscript.roundFields index round)) state
      let sampled :=
        valueSqueezeIndexedGo constants assignment 46 index 1 absorbed
      let tail := valueReplayRounds constants assignment rest
        (index + 1) sampled.2
      { challenges := sampled.1 ++ tail.challenges
        state := tail.state }

theorem valueReplayRounds_challenges_length
    {degree : Nat} (constants : Constants) (assignment : Nat → Nat) :
    ∀ (rounds : List (KFixedPhaseSumCheck.Round degree)) index state,
      (valueReplayRounds constants assignment rounds index state).challenges.length =
        rounds.length
  | [], _, _ => rfl
  | round :: rest, index, state => by
      simp only [valueReplayRounds, List.length_append,
        valueSqueezeIndexed_length,
        valueReplayRounds_challenges_length constants assignment rest
          (index + 1)]
      exact Nat.add_comm 1 rest.length

structure ValueReplay (shape : Shape) where
  alpha : List ValueK
  alpha_length : alpha.length = shape.cubeVariables
  gamma : ValueK
  point : List ValueK
  point_length : point.length = shape.cubeVariables
  beforeOutput : ValueState
  afterOutput : ValueState

def valueReplay
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (input : KPiCcsTranscript.Input shape degree) :
    ValueReplay shape :=
  let pre := valueDerivePreSumcheck constants assignment input
  let rounds := valueReplayRounds constants assignment input.rounds 0 pre.state
  let outgoing := Poseidon2Duplex.absorbList constants
    (fieldValues assignment (KPiCcsTranscript.outputFields input))
    rounds.state
  { alpha := pre.alpha
    alpha_length := pre.alpha_length
    gamma := pre.gamma
    point := rounds.challenges
    point_length := by
      rw [valueReplayRounds_challenges_length, input.rounds_length]
    beforeOutput := rounds.state
    afterOutput := outgoing }

/-! ## Physical extension facts -/

theorem squeezeIndexedGo_extends (base label index : Nat) :
    ∀ (count : Nat) (builder : SymbolicDuplex.Builder),
      Extends builder
        (KPiCcsTranscript.squeezeIndexedGo base label index count builder).2
  | 0, builder => Extends.refl builder
  | remaining + 1, builder =>
      let absorbed := SymbolicDuplex.absorbMany base
        [KPiCcsTranscript.word label, KPiCcsTranscript.word index] builder
      let sampled := SymbolicDuplex.squeezeK base absorbed
      (SymbolicDuplexSemantics.absorbMany_extends base _ builder).trans
        ((SymbolicDuplexSemantics.squeezeK_extends base absorbed).trans
          (squeezeIndexedGo_extends base label (index + 1)
            remaining sampled.2))

theorem squeezeIndexed_extends (base label count : Nat)
    (builder : SymbolicDuplex.Builder) :
    Extends builder
      (KPiCcsTranscript.squeezeIndexed base label count builder).2 :=
  squeezeIndexedGo_extends base label 0 count builder

theorem squeezeLabel_extends (base label : Nat)
    (builder : SymbolicDuplex.Builder) :
    Extends builder
      (KPiCcsTranscript.squeezeLabel base label builder).2 :=
  (SymbolicDuplexSemantics.absorb_extends base
      (KPiCcsTranscript.word label) builder).trans
    (SymbolicDuplexSemantics.squeezeK_extends base
      (SymbolicDuplex.absorb base
        (KPiCcsTranscript.word label) builder))

theorem replayRounds_extends
    {degree : Nat} (base : Nat) :
    ∀ (rounds : List (KFixedPhaseSumCheck.Round degree)) index builder,
      Extends builder
        (KPiCcsTranscript.replayRounds base rounds index builder).builder
  | [], _, builder => Extends.refl builder
  | round :: rest, index, builder =>
      let absorbed :=
        SymbolicDuplex.absorbMany base
          (KPiCcsTranscript.roundFields index round) builder
      let sampled :=
        KPiCcsTranscript.squeezeIndexedGo base 46 index 1 absorbed
      (SymbolicDuplexSemantics.absorbMany_extends base _ builder).trans
        ((squeezeIndexedGo_extends base 46 index 1 absorbed).trans
          (replayRounds_extends base rest (index + 1) sampled.2))

/-! ## Row-satisfaction refinement -/

theorem squeezeLabel_semantics
    (base label : Nat) (constants : Constants)
    (assignment : Nat → Nat) (builder : SymbolicDuplex.Builder)
    (constantWire : assignment 0 = 1)
    (valid : Valid base constants assignment
      (KPiCcsTranscript.squeezeLabel base label builder).2) :
    (decoded assignment
        (KPiCcsTranscript.squeezeLabel base label builder).1,
      decodedBuilder assignment
        (KPiCcsTranscript.squeezeLabel base label builder).2) =
      valueSqueezeLabel constants assignment label
        (decodedBuilder assignment builder) := by
  let absorbed :=
    SymbolicDuplex.absorb base (KPiCcsTranscript.word label) builder
  have absorbedValid :
      Valid base constants assignment absorbed := by
    exact valid.of_extends
      (SymbolicDuplexSemantics.squeezeK_extends base absorbed)
  have absorbedEq :=
    SymbolicDuplexSemantics.decodedBuilder_absorb base constants assignment
      (KPiCcsTranscript.word label) builder absorbedValid
  have squeezeEq :=
    SymbolicDuplexSemantics.decoded_squeezeK base constants assignment
      absorbed constantWire valid
  unfold KPiCcsTranscript.squeezeLabel valueSqueezeLabel
  change
    (decoded assignment (SymbolicDuplex.squeezeK base absorbed).1,
      decodedBuilder assignment
        (SymbolicDuplex.squeezeK base absorbed).2) = _
  calc
    _ = SymbolicDuplexSemantics.squeezeKValue constants
          (decodedBuilder assignment absorbed) := squeezeEq
    _ = SymbolicDuplexSemantics.squeezeKValue constants
          (Poseidon2Duplex.absorbElem constants
            (lcEval assignment (KPiCcsTranscript.word label))
            (decodedBuilder assignment builder)) := by
          rw [absorbedEq]

theorem squeezeIndexedGo_semantics
    (base label : Nat) (constants : Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1) :
    ∀ (index count : Nat) (builder : SymbolicDuplex.Builder),
      Valid base constants assignment
          (KPiCcsTranscript.squeezeIndexedGo
            base label index count builder).2 →
      ((KPiCcsTranscript.squeezeIndexedGo
          base label index count builder).1.map (decoded assignment),
        decodedBuilder assignment
          (KPiCcsTranscript.squeezeIndexedGo
            base label index count builder).2) =
        valueSqueezeIndexedGo constants assignment
          label index count (decodedBuilder assignment builder)
  | _, 0, _, _ => rfl
  | index, remaining + 1, builder, valid => by
      let absorbed :=
        SymbolicDuplex.absorbMany base
          [KPiCcsTranscript.word label, KPiCcsTranscript.word index] builder
      let sampled := SymbolicDuplex.squeezeK base absorbed
      have sampledValid :
          Valid base constants assignment sampled.2 := by
        exact valid.of_extends
          (squeezeIndexedGo_extends base label (index + 1)
            remaining sampled.2)
      have absorbedValid :
          Valid base constants assignment absorbed := by
        exact sampledValid.of_extends
          (SymbolicDuplexSemantics.squeezeK_extends base absorbed)
      have absorbedEq :=
        SymbolicDuplexSemantics.decodedBuilder_absorbMany
          base constants assignment
          [KPiCcsTranscript.word label, KPiCcsTranscript.word index]
          builder absorbedValid
      have sampledEq :=
        SymbolicDuplexSemantics.decoded_squeezeK
          base constants assignment absorbed constantWire sampledValid
      have tailEq :=
        squeezeIndexedGo_semantics base label constants assignment
          constantWire (index + 1) remaining sampled.2 valid
      simp only [KPiCcsTranscript.squeezeIndexedGo,
        valueSqueezeIndexedGo, List.map_cons]
      change
        (decoded assignment sampled.1 ::
            (KPiCcsTranscript.squeezeIndexedGo base label
              (index + 1) remaining sampled.2).1.map (decoded assignment),
          decodedBuilder assignment
            (KPiCcsTranscript.squeezeIndexedGo base label
              (index + 1) remaining sampled.2).2) = _
      have sampledValueEq := congrArg Prod.fst sampledEq
      have sampledStateEq := congrArg Prod.snd sampledEq
      simp only at sampledValueEq sampledStateEq
      rw [absorbedEq] at sampledValueEq sampledStateEq
      have tailValuesEq := congrArg Prod.fst tailEq
      have tailStateEq := congrArg Prod.snd tailEq
      simp only at tailValuesEq tailStateEq
      rw [sampledStateEq] at tailValuesEq tailStateEq
      simp only [decoded, sampled, absorbed, fieldValues] at sampledValueEq sampledStateEq
      simp only [decoded, sampled, absorbed, fieldValues] at tailValuesEq tailStateEq
      simp only [decoded, sampled, absorbed, fieldValues]
      apply Prod.ext
      · calc
          _ = (SymbolicDuplexSemantics.squeezeKValue constants
                  (Poseidon2Duplex.absorbList constants
                    (List.map (lcEval assignment)
                      [KPiCcsTranscript.word label,
                        KPiCcsTranscript.word index])
                    (decodedBuilder assignment builder))).1 ::
                (KPiCcsTranscript.squeezeIndexedGo base label
                  (index + 1) remaining
                  (SymbolicDuplex.squeezeK base
                    (SymbolicDuplex.absorbMany base
                      [KPiCcsTranscript.word label,
                        KPiCcsTranscript.word index] builder)).2).1.map
                    (decoded assignment) := by
              exact congrArg
                (fun head => head ::
                  (KPiCcsTranscript.squeezeIndexedGo base label
                    (index + 1) remaining
                    (SymbolicDuplex.squeezeK base
                      (SymbolicDuplex.absorbMany base
                        [KPiCcsTranscript.word label,
                          KPiCcsTranscript.word index] builder)).2).1.map
                      (decoded assignment))
                sampledValueEq
          _ = _ := by
              exact congrArg
                (List.cons
                  (SymbolicDuplexSemantics.squeezeKValue constants
                    (Poseidon2Duplex.absorbList constants
                      (List.map (lcEval assignment)
                        [KPiCcsTranscript.word label,
                          KPiCcsTranscript.word index])
                      (decodedBuilder assignment builder))).1)
                tailValuesEq
      · exact tailStateEq

theorem squeezeIndexed_semantics
    (base label count : Nat) (constants : Constants)
    (assignment : Nat → Nat) (builder : SymbolicDuplex.Builder)
    (constantWire : assignment 0 = 1)
    (valid : Valid base constants assignment
      (KPiCcsTranscript.squeezeIndexed base label count builder).2) :
    ((KPiCcsTranscript.squeezeIndexed base label count builder).1.map
        (decoded assignment),
      decodedBuilder assignment
        (KPiCcsTranscript.squeezeIndexed base label count builder).2) =
      valueSqueezeIndexed constants assignment label count
        (decodedBuilder assignment builder) :=
  squeezeIndexedGo_semantics base label constants assignment constantWire
    0 count builder valid

theorem derivePreSumcheck_semantics
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (input : KPiCcsTranscript.Input shape degree)
    (constantWire : assignment 0 = 1)
    (valid : Valid input.transcriptBase constants assignment
      (KPiCcsTranscript.derivePreSumcheck input).builder) :
    ((KPiCcsTranscript.derivePreSumcheck input).alpha.map
          (decoded assignment) =
        (valueDerivePreSumcheck constants assignment input).alpha) ∧
      decoded assignment
          (KPiCcsTranscript.derivePreSumcheck input).gamma =
        (valueDerivePreSumcheck constants assignment input).gamma ∧
      decodedBuilder assignment
          (KPiCcsTranscript.derivePreSumcheck input).builder =
        (valueDerivePreSumcheck constants assignment input).state := by
  let initial := SymbolicDuplex.start input.priorLanes 0
  let statement :=
    SymbolicDuplex.absorbMany input.transcriptBase
      (KPiCcsTranscript.statementFields input) initial
  let alpha :=
    KPiCcsTranscript.squeezeIndexed input.transcriptBase
      42 shape.cubeVariables statement
  let gamma :=
    KPiCcsTranscript.squeezeLabel input.transcriptBase 43 alpha.2
  have alphaValid :
      Valid input.transcriptBase constants assignment alpha.2 := by
    exact valid.of_extends
      (squeezeLabel_extends input.transcriptBase 43 alpha.2)
  have statementValid :
      Valid input.transcriptBase constants assignment statement := by
    exact alphaValid.of_extends
      (squeezeIndexed_extends input.transcriptBase 42
        shape.cubeVariables statement)
  have statementEq :=
    SymbolicDuplexSemantics.decodedBuilder_absorbMany
      input.transcriptBase constants assignment
      (KPiCcsTranscript.statementFields input) initial statementValid
  have alphaPair :=
    squeezeIndexed_semantics input.transcriptBase 42 shape.cubeVariables
      constants assignment statement constantWire alphaValid
  have alphaValuesEq := congrArg Prod.fst alphaPair
  have alphaStateEq := congrArg Prod.snd alphaPair
  simp only at alphaValuesEq alphaStateEq
  rw [statementEq] at alphaValuesEq alphaStateEq
  have gammaPair :=
    squeezeLabel_semantics input.transcriptBase 43 constants assignment
      alpha.2 constantWire valid
  have gammaValueEq := congrArg Prod.fst gammaPair
  have gammaStateEq := congrArg Prod.snd gammaPair
  simp only at gammaValueEq gammaStateEq
  rw [alphaStateEq] at gammaValueEq gammaStateEq
  simpa only [KPiCcsTranscript.derivePreSumcheck, valueDerivePreSumcheck,
    initial, statement, alpha, gamma] using
      And.intro alphaValuesEq (And.intro gammaValueEq gammaStateEq)

theorem replayRounds_semantics
    {degree : Nat} (base : Nat) (constants : Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1) :
    ∀ (rounds : List (KFixedPhaseSumCheck.Round degree))
      (index : Nat) (builder : SymbolicDuplex.Builder),
      Valid base constants assignment
          (KPiCcsTranscript.replayRounds base rounds index builder).builder →
      ((KPiCcsTranscript.replayRounds base rounds index builder).challenges.map
          (decoded assignment) =
        (valueReplayRounds constants assignment rounds index
          (decodedBuilder assignment builder)).challenges) ∧
      decodedBuilder assignment
          (KPiCcsTranscript.replayRounds base rounds index builder).builder =
        (valueReplayRounds constants assignment rounds index
          (decodedBuilder assignment builder)).state
  | [], _, _, _ => ⟨rfl, rfl⟩
  | round :: rest, index, builder, valid => by
      let absorbed :=
        SymbolicDuplex.absorbMany base
          (KPiCcsTranscript.roundFields index round) builder
      let sampled :=
        KPiCcsTranscript.squeezeIndexedGo base 46 index 1 absorbed
      have sampledValid :
          Valid base constants assignment sampled.2 := by
        exact valid.of_extends
          (replayRounds_extends base rest (index + 1) sampled.2)
      have absorbedValid :
          Valid base constants assignment absorbed := by
        exact sampledValid.of_extends
          (squeezeIndexedGo_extends base 46 index 1 absorbed)
      have absorbedEq :=
        SymbolicDuplexSemantics.decodedBuilder_absorbMany
          base constants assignment
          (KPiCcsTranscript.roundFields index round) builder absorbedValid
      have sampledPair :=
        squeezeIndexedGo_semantics base 46 constants assignment
          constantWire index 1 absorbed sampledValid
      have sampledValuesEq := congrArg Prod.fst sampledPair
      have sampledStateEq := congrArg Prod.snd sampledPair
      simp only at sampledValuesEq sampledStateEq
      rw [absorbedEq] at sampledValuesEq sampledStateEq
      have tail :=
        replayRounds_semantics base constants assignment constantWire rest
          (index + 1) sampled.2 valid
      have tailValuesEq := tail.1
      have tailStateEq := tail.2
      rw [sampledStateEq] at tailValuesEq tailStateEq
      simp only [KPiCcsTranscript.replayRounds, valueReplayRounds,
        List.map_append, sampled, absorbed, fieldValues] at sampledValuesEq sampledStateEq
      simp only [KPiCcsTranscript.replayRounds, valueReplayRounds,
        List.map_append, sampled, absorbed, fieldValues] at tailValuesEq tailStateEq
      refine ⟨?_, tailStateEq⟩
      simp only [KPiCcsTranscript.replayRounds, valueReplayRounds,
        List.map_append, fieldValues]
      rw [sampledValuesEq, tailValuesEq]

theorem replay_semantics
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (input : KPiCcsTranscript.Input shape degree)
    (constantWire : assignment 0 = 1)
    (valid : Valid input.transcriptBase constants assignment
      (KPiCcsTranscript.replay input).afterOutput) :
    ((KPiCcsTranscript.replay input).alpha.map (decoded assignment) =
        (valueReplay constants assignment input).alpha) ∧
      decoded assignment (KPiCcsTranscript.replay input).gamma =
        (valueReplay constants assignment input).gamma ∧
      (KPiCcsTranscript.replay input).point.map (decoded assignment) =
        (valueReplay constants assignment input).point ∧
      decodedBuilder assignment
          (KPiCcsTranscript.replay input).beforeOutput =
        (valueReplay constants assignment input).beforeOutput ∧
      decodedBuilder assignment
          (KPiCcsTranscript.replay input).afterOutput =
        (valueReplay constants assignment input).afterOutput := by
  let pre := KPiCcsTranscript.derivePreSumcheck input
  let rounds :=
    KPiCcsTranscript.replayRounds input.transcriptBase input.rounds 0 pre.builder
  let outgoing :=
    SymbolicDuplex.absorbMany input.transcriptBase
      (KPiCcsTranscript.outputFields input) rounds.builder
  have roundsValid :
      Valid input.transcriptBase constants assignment rounds.builder := by
    exact valid.of_extends
      (SymbolicDuplexSemantics.absorbMany_extends input.transcriptBase
        (KPiCcsTranscript.outputFields input) rounds.builder)
  have preValid :
      Valid input.transcriptBase constants assignment pre.builder := by
    exact roundsValid.of_extends
      (replayRounds_extends input.transcriptBase input.rounds 0 pre.builder)
  have preSem :=
    derivePreSumcheck_semantics constants assignment input constantWire preValid
  have roundsSem :=
    replayRounds_semantics input.transcriptBase constants assignment
      constantWire input.rounds 0 pre.builder roundsValid
  have roundsValuesEq := roundsSem.1
  have roundsStateEq := roundsSem.2
  rw [preSem.2.2] at roundsValuesEq roundsStateEq
  have outgoingEq :=
    SymbolicDuplexSemantics.decodedBuilder_absorbMany
      input.transcriptBase constants assignment
      (KPiCcsTranscript.outputFields input) rounds.builder valid
  rw [roundsStateEq] at outgoingEq
  simpa only [KPiCcsTranscript.replay, valueReplay, pre, rounds, outgoing] using
    And.intro preSem.1
      (And.intro preSem.2.1
        (And.intro roundsValuesEq
          (And.intro roundsStateEq outgoingEq)))

theorem transcriptRows_satisfied
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (input : KPiCcsTranscript.Input shape degree)
    (satisfied :
      Satisfies (KPiCcsTranscript.rows constants input) assignment) :
    Satisfies
      (SymbolicDuplex.rows input.transcriptBase constants
        (KPiCcsTranscript.replay input).afterOutput)
      assignment := by
  intro row member
  exact satisfied row (List.mem_append_left _ member)

/-- Headline challenge-binding theorem: combined row satisfaction determines
the exact value-level alpha, gamma, round point, and outgoing transcript state.
No transcript-equality premise is accepted. -/
theorem rows_replay_semantics
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (input : KPiCcsTranscript.Input shape degree)
    (residues : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies (KPiCcsTranscript.rows constants input) assignment) :
    ((KPiCcsTranscript.replay input).alpha.map (decoded assignment) =
        (valueReplay constants assignment input).alpha) ∧
      decoded assignment (KPiCcsTranscript.replay input).gamma =
        (valueReplay constants assignment input).gamma ∧
      (KPiCcsTranscript.replay input).point.map (decoded assignment) =
        (valueReplay constants assignment input).point ∧
      decodedBuilder assignment
          (KPiCcsTranscript.replay input).beforeOutput =
        (valueReplay constants assignment input).beforeOutput ∧
      decodedBuilder assignment
          (KPiCcsTranscript.replay input).afterOutput =
        (valueReplay constants assignment input).afterOutput := by
  apply replay_semantics constants assignment input constantWire
  apply SymbolicDuplexSemantics.valid_of_satisfied
    input.transcriptBase constants
    (KPiCcsTranscript.replay input).afterOutput assignment
    residues constantWire
  exact transcriptRows_satisfied constants assignment input satisfied

end Nightstream.Implementation.R1CS.Canonical.KPiCcsTranscriptSemantics
