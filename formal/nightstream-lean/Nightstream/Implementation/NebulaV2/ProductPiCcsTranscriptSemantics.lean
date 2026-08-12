import Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptRows
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics

/-!
Contract: row-satisfaction semantics for the exact V2 PiCCS transcript.

Owns the proof that satisfying the generated Poseidon2 rows makes every
symbolic alpha, gamma, SumCheck challenge, pre-output state, and full-output
state equal the value-level replay of the same serialized fields.

Does not own the typed placement of those fields into a paper NIFS input,
PiRLC, PiDEC, cryptographic transcript security, Rust, or the deployed
verifier.

No transcript value or verifier acceptance result is an assumption.
-/

set_option autoImplicit false
set_option maxRecDepth 30000

namespace Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptSemantics

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptRows
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics

abbrev ValueState := Poseidon2Duplex.State
abbrev ValueK := Nightstream.Implementation.R1CS.ProjectionProgram.K

def decoded (assignment : Nat -> Nat) (value : Carried) : ValueK :=
  KFixedPhaseSumCheck.decodeCarried assignment value

def fieldValues (assignment : Nat -> Nat) (fields : List LinComb) :
    List Nat :=
  fields.map (lcEval assignment)

theorem lcEval_word (assignment : Nat -> Nat) (one : assignment 0 = 1)
    (value : Nat) :
    lcEval assignment (word value) = value % goldilocksP := by
  simp [word, lcEval, rawSum, one, goldilocksP]

theorem fieldValues_words (assignment : Nat -> Nat)
    (one : assignment 0 = 1) (values : List Nat) :
    fieldValues assignment (values.map word) =
      values.map fun value => value % goldilocksP := by
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [fieldValues, lcEval_word assignment one, inductionHypothesis]

def StateCanonical (state : ValueState) : Prop :=
  forall lane, state.lanes lane < goldilocksP

theorem empty_canonical : StateCanonical Poseidon2Duplex.empty := by
  intro lane
  simp [StateCanonical, Poseidon2Duplex.empty, goldilocksP]

theorem permute_canonical (state : ValueState) :
    StateCanonical (Poseidon2Duplex.permute ProductPoseidon2.constants state) := by
  intro lane
  exact Poseidon2Honest.refTerminal_lt _ _ _ _

theorem guarded_canonical (state : ValueState)
    (canonical : StateCanonical state) :
    StateCanonical (Poseidon2Duplex.guarded ProductPoseidon2.constants state) := by
  unfold Poseidon2Duplex.guarded
  split
  · exact permute_canonical state
  · exact canonical

theorem absorbElem_canonical (value : Nat) (state : ValueState)
    (canonical : StateCanonical state) :
    StateCanonical
      (Poseidon2Duplex.absorbElem ProductPoseidon2.constants value state) := by
  intro lane
  unfold Poseidon2Duplex.absorbElem
  let target := Poseidon2Duplex.guarded ProductPoseidon2.constants state
  have targetCanonical : StateCanonical target := guarded_canonical state canonical
  change (if lane.val = target.absorbed then value % goldilocksP
    else target.lanes lane) < goldilocksP
  split
  · exact Nat.mod_lt _ (by decide)
  · exact targetCanonical lane

theorem absorbList_canonical (values : List Nat) (state : ValueState)
    (canonical : StateCanonical state) :
    StateCanonical
      (Poseidon2Duplex.absorbList ProductPoseidon2.constants values state) := by
  induction values generalizing state with
  | nil => exact canonical
  | cons head tail inductionHypothesis =>
      exact inductionHypothesis _ (absorbElem_canonical head state canonical)

theorem initialStateForStatement_canonical
    (statementId : ProductPoseidon2.StatementId) :
    StateCanonical (ProductPoseidon2.initialStateForStatement statementId) := by
  unfold ProductPoseidon2.initialStateForStatement
  apply absorbList_canonical
  exact empty_canonical

theorem decoded_initialBuilder (assignment : Nat -> Nat)
    (one : assignment 0 = 1) (input : Input) :
    decodedBuilder assignment (initialBuilder input) =
      ProductPoseidon2.initialStateForStatement input.statementId := by
  rw [Poseidon2Duplex.State.mk.injEq]
  refine ⟨?_, rfl⟩
  funext lane
  change lcEval assignment
      (word ((ProductPoseidon2.initialStateForStatement
        input.statementId).lanes lane)) =
    (ProductPoseidon2.initialStateForStatement input.statementId).lanes lane
  rw [lcEval_word assignment one]
  exact Nat.mod_eq_of_lt
    (initialStateForStatement_canonical input.statementId lane)

def valueAbsorbPublic (assignment : Nat -> Nat) (input : Input) :
    ValueState :=
  Poseidon2Duplex.absorbList ProductPoseidon2.constants
    (fieldValues assignment input.publicNifsFields)
    (ProductPoseidon2.initialStateForStatement input.statementId)

/-- Public absorption depends only on the statement identifier and the
decoded public fields. This avoids unfolding unrelated verifier data. -/
theorem valueAbsorbPublic_eq_of_fields
    (assignment : Nat -> Nat) (input : Input) (values : List Nat)
    (fieldsEq : fieldValues assignment input.publicNifsFields = values) :
    valueAbsorbPublic assignment input =
      Poseidon2Duplex.absorbList ProductPoseidon2.constants values
        (ProductPoseidon2.initialStateForStatement input.statementId) := by
  unfold valueAbsorbPublic
  rw [fieldsEq]

def valueAbsorbStatement (assignment : Nat -> Nat) (input : Input) :
    ValueState :=
  Poseidon2Duplex.absorbList ProductPoseidon2.constants
    (fieldValues assignment (statementFields input))
    (valueAbsorbPublic assignment input)

def valueSqueezeVerifierChallenge
    (assignment : Nat -> Nat)
    (eventIndex challengeIndex challengeType : Nat)
    (coordinates : List Nat) (state : ValueState) : ValueK × ValueState :=
  squeezeKValue ProductPoseidon2.constants
    (Poseidon2Duplex.absorbList ProductPoseidon2.constants
      (fieldValues assignment
        (verifierChallengeFields eventIndex challengeIndex challengeType
          coordinates)) state)

def valueDeriveAlphaGo (assignment : Nat -> Nat) (input : Input) :
    Nat -> Nat -> ValueState -> List ValueK × ValueState
  | _, 0, state => ([], state)
  | index, remaining + 1, state =>
      let sampled := valueSqueezeVerifierChallenge assignment 1 1 42
        [index] state
      let tail := valueDeriveAlphaGo assignment input (index + 1)
        remaining sampled.2
      (sampled.1 :: tail.1, tail.2)

def valueDeriveAlpha (assignment : Nat -> Nat) (input : Input) :
    List ValueK × ValueState :=
  valueDeriveAlphaGo assignment input 0 selectedShape.cubeVariables
    (valueAbsorbStatement assignment input)

def valueDeriveGamma (assignment : Nat -> Nat) (input : Input) :
    ValueK × ValueState :=
  valueSqueezeVerifierChallenge assignment 2 2 43 []
    (valueDeriveAlpha assignment input).2

structure ValueRoundReplay where
  challenges : List ValueK
  state : ValueState

def valueReplayRoundsGo (assignment : Nat -> Nat) (input : Input) :
    List Round -> Nat -> ValueState -> ValueRoundReplay
  | [], _, state => { challenges := [], state }
  | round :: rest, index, state =>
      let absorbed := Poseidon2Duplex.absorbList ProductPoseidon2.constants
        (fieldValues assignment (roundFields index round)) state
      let sampled := valueSqueezeVerifierChallenge assignment
        (4 + 2 * index) (3 + index) 46 [] absorbed
      let tail := valueReplayRoundsGo assignment input rest (index + 1)
        sampled.2
      { challenges := sampled.1 :: tail.challenges
        state := tail.state }

def valueReplayRounds (assignment : Nat -> Nat) (input : Input) :
    ValueRoundReplay :=
  valueReplayRoundsGo assignment input (List.ofFn input.rounds) 0
    (valueDeriveGamma assignment input).2

def valueAfterFullOutput (assignment : Nat -> Nat) (input : Input) :
    ValueState :=
  Poseidon2Duplex.absorbList ProductPoseidon2.constants
    (fieldValues assignment (fullOutputFields input))
    (valueReplayRounds assignment input).state

/-! ## Builder extension -/

theorem absorbPublic_extends (input : Input) :
    Extends (initialBuilder input) (absorbPublicInput input) :=
  absorbMany_extends input.transcriptBase input.publicNifsFields _

theorem absorbStatement_extends (input : Input) :
    Extends (absorbPublicInput input) (absorbStatement input) :=
  absorbMany_extends input.transcriptBase (statementFields input) _

theorem squeezeVerifierChallenge_extends
    (eventIndex challengeIndex challengeType : Nat)
    (coordinates : List Nat) (input : Input)
    (builder : SymbolicDuplex.Builder) :
    Extends builder
      (squeezeVerifierChallenge eventIndex challengeIndex challengeType
        coordinates input builder).2 :=
  (absorbMany_extends input.transcriptBase
      (verifierChallengeFields eventIndex challengeIndex challengeType
        coordinates) builder).trans
    (squeezeK_extends input.transcriptBase _)

theorem deriveAlphaGo_extends (input : Input) :
    forall index count builder,
      Extends builder (deriveAlphaGo input index count builder).2
  | _, 0, builder => Extends.refl builder
  | index, remaining + 1, builder =>
      let sampled := squeezeVerifierChallenge 1 1 42 [index] input builder
      (squeezeVerifierChallenge_extends 1 1 42 [index] input builder).trans
        (deriveAlphaGo_extends input (index + 1) remaining sampled.2)

theorem deriveAlpha_extends (input : Input) :
    Extends (absorbStatement input) (deriveAlpha input).2 :=
  deriveAlphaGo_extends input 0 selectedShape.cubeVariables _

theorem deriveGamma_extends (input : Input) :
    Extends (deriveAlpha input).2 (deriveGamma input).2 :=
  squeezeVerifierChallenge_extends 2 2 43 [] input _

theorem replayRoundsGo_extends (input : Input) :
    forall rounds index builder,
      Extends builder (replayRoundsGo input rounds index builder).builder
  | [], _, builder => Extends.refl builder
  | round :: rest, index, builder =>
      let absorbed := SymbolicDuplex.absorbMany input.transcriptBase
        (roundFields index round) builder
      let sampled := squeezeVerifierChallenge
        (4 + 2 * index) (3 + index) 46 [] input absorbed
      (absorbMany_extends input.transcriptBase
          (roundFields index round) builder).trans
        ((squeezeVerifierChallenge_extends
            (4 + 2 * index) (3 + index) 46 [] input absorbed).trans
          (replayRoundsGo_extends input rest (index + 1) sampled.2))

theorem replayRounds_extends (input : Input) :
    Extends (deriveGamma input).2 (replayRounds input).builder :=
  replayRoundsGo_extends input (List.ofFn input.rounds) 0 _

theorem afterFullOutput_extends (input : Input) :
    Extends (replayRounds input).builder (afterFullOutput input) :=
  absorbMany_extends input.transcriptBase (fullOutputFields input) _

/-! ## Operation semantics -/

theorem squeezeVerifierChallenge_semantics
    (assignment : Nat -> Nat) (one : assignment 0 = 1)
    (eventIndex challengeIndex challengeType : Nat)
    (coordinates : List Nat) (input : Input)
    (builder : SymbolicDuplex.Builder)
    (valid : Valid input.transcriptBase ProductPoseidon2.constants assignment
      (squeezeVerifierChallenge eventIndex challengeIndex challengeType
        coordinates input builder).2) :
    (decoded assignment
        (squeezeVerifierChallenge eventIndex challengeIndex challengeType
          coordinates input builder).1,
      decodedBuilder assignment
        (squeezeVerifierChallenge eventIndex challengeIndex challengeType
          coordinates input builder).2) =
      valueSqueezeVerifierChallenge assignment eventIndex challengeIndex
        challengeType coordinates (decodedBuilder assignment builder) := by
  let absorbed := SymbolicDuplex.absorbMany input.transcriptBase
    (verifierChallengeFields eventIndex challengeIndex challengeType
      coordinates) builder
  have absorbedValid :
      Valid input.transcriptBase ProductPoseidon2.constants assignment
        absorbed := by
    exact valid.of_extends (squeezeK_extends input.transcriptBase absorbed)
  have absorbedEq := decodedBuilder_absorbMany input.transcriptBase
    ProductPoseidon2.constants assignment
    (verifierChallengeFields eventIndex challengeIndex challengeType
      coordinates) builder absorbedValid
  change decodedBuilder assignment absorbed =
    Poseidon2Duplex.absorbList ProductPoseidon2.constants
      (fieldValues assignment
        (verifierChallengeFields eventIndex challengeIndex challengeType
          coordinates)) (decodedBuilder assignment builder) at absorbedEq
  have sampledEq := decoded_squeezeK input.transcriptBase
    ProductPoseidon2.constants assignment absorbed one valid
  unfold squeezeVerifierChallenge valueSqueezeVerifierChallenge
  change
    (decoded assignment (SymbolicDuplex.squeezeK input.transcriptBase absorbed).1,
      decodedBuilder assignment
        (SymbolicDuplex.squeezeK input.transcriptBase absorbed).2) = _
  rw [← absorbedEq]
  simpa [decoded] using sampledEq

theorem deriveAlphaGo_semantics
    (assignment : Nat -> Nat) (one : assignment 0 = 1) (input : Input) :
    forall index count builder,
      Valid input.transcriptBase ProductPoseidon2.constants assignment
          (deriveAlphaGo input index count builder).2 ->
      ((deriveAlphaGo input index count builder).1.map (decoded assignment),
        decodedBuilder assignment
          (deriveAlphaGo input index count builder).2) =
        valueDeriveAlphaGo assignment input index count
          (decodedBuilder assignment builder)
  | _, 0, _, _ => rfl
  | index, remaining + 1, builder, valid => by
      let sampled := squeezeVerifierChallenge 1 1 42 [index] input builder
      have sampledValid :
          Valid input.transcriptBase ProductPoseidon2.constants assignment
            sampled.2 := by
        exact valid.of_extends
          (deriveAlphaGo_extends input (index + 1) remaining sampled.2)
      have sampledEq := squeezeVerifierChallenge_semantics assignment one
        1 1 42 [index] input builder sampledValid
      have tailEq := deriveAlphaGo_semantics assignment one input
        (index + 1) remaining sampled.2 valid
      have sampledValueEq := congrArg Prod.fst sampledEq
      have sampledStateEq := congrArg Prod.snd sampledEq
      have tailValueEq := congrArg Prod.fst tailEq
      have tailStateEq := congrArg Prod.snd tailEq
      simp only at sampledValueEq sampledStateEq tailValueEq tailStateEq
      rw [sampledStateEq] at tailValueEq tailStateEq
      simp only [deriveAlphaGo, valueDeriveAlphaGo, List.map_cons]
      exact Prod.ext
        (congrArg₂ List.cons sampledValueEq tailValueEq)
        tailStateEq

theorem replayRoundsGo_semantics
    (assignment : Nat -> Nat) (one : assignment 0 = 1) (input : Input) :
    forall rounds index builder,
      Valid input.transcriptBase ProductPoseidon2.constants assignment
          (replayRoundsGo input rounds index builder).builder ->
      ((replayRoundsGo input rounds index builder).challenges.map
          (decoded assignment),
        decodedBuilder assignment
          (replayRoundsGo input rounds index builder).builder) =
        ((valueReplayRoundsGo assignment input rounds index
          (decodedBuilder assignment builder)).challenges,
        (valueReplayRoundsGo assignment input rounds index
          (decodedBuilder assignment builder)).state)
  | [], _, _, _ => rfl
  | round :: rest, index, builder, valid => by
      let absorbed := SymbolicDuplex.absorbMany input.transcriptBase
        (roundFields index round) builder
      let sampled := squeezeVerifierChallenge
        (4 + 2 * index) (3 + index) 46 [] input absorbed
      have sampledValid :
          Valid input.transcriptBase ProductPoseidon2.constants assignment
            sampled.2 := by
        exact valid.of_extends
          (replayRoundsGo_extends input rest (index + 1) sampled.2)
      have absorbedValid :
          Valid input.transcriptBase ProductPoseidon2.constants assignment
            absorbed := by
        exact sampledValid.of_extends
          (squeezeVerifierChallenge_extends
            (4 + 2 * index) (3 + index) 46 [] input absorbed)
      have absorbedEq := decodedBuilder_absorbMany input.transcriptBase
        ProductPoseidon2.constants assignment (roundFields index round)
        builder absorbedValid
      have sampledEq := squeezeVerifierChallenge_semantics assignment one
        (4 + 2 * index) (3 + index) 46 [] input absorbed sampledValid
      have tailEq := replayRoundsGo_semantics assignment one input rest
        (index + 1) sampled.2 valid
      have sampledValueEq := congrArg Prod.fst sampledEq
      have sampledStateEq := congrArg Prod.snd sampledEq
      rw [absorbedEq] at sampledValueEq sampledStateEq
      have tailValueEq := congrArg Prod.fst tailEq
      have tailStateEq := congrArg Prod.snd tailEq
      simp only at sampledValueEq sampledStateEq tailValueEq tailStateEq
      rw [sampledStateEq] at tailValueEq tailStateEq
      simp only [replayRoundsGo, valueReplayRoundsGo, List.map_cons]
      exact Prod.ext
        (congrArg₂ List.cons sampledValueEq tailValueEq)
        tailStateEq

/-! ## Complete replay theorem -/

/-- Satisfying the complete PiCCS transcript rows fixes the decoded state
immediately after the public NIFS input to the value-level Poseidon2 replay.

This theorem exposes the exact prefix needed by production-profile bridges.
It does not assume a challenge, a verifier result, or a transcript state. -/
theorem absorbPublicInput_rows_semantics
    (assignment : Nat -> Nat) (input : Input)
    (residues : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    decodedBuilder assignment (absorbPublicInput input) =
      valueAbsorbPublic assignment input := by
  have valid : Valid input.transcriptBase ProductPoseidon2.constants assignment
      (afterFullOutput input) := by
    apply valid_of_satisfied input.transcriptBase ProductPoseidon2.constants
      (afterFullOutput input) assignment residues one
    exact transcriptRows_satisfied input assignment satisfied
  have roundsValid : Valid input.transcriptBase ProductPoseidon2.constants
      assignment (replayRounds input).builder :=
    valid.of_extends (afterFullOutput_extends input)
  have gammaValid : Valid input.transcriptBase ProductPoseidon2.constants
      assignment (deriveGamma input).2 :=
    roundsValid.of_extends (replayRounds_extends input)
  have alphaValid : Valid input.transcriptBase ProductPoseidon2.constants
      assignment (deriveAlpha input).2 :=
    gammaValid.of_extends (deriveGamma_extends input)
  have statementValid : Valid input.transcriptBase ProductPoseidon2.constants
      assignment (absorbStatement input) :=
    alphaValid.of_extends (deriveAlpha_extends input)
  have publicValid : Valid input.transcriptBase ProductPoseidon2.constants
      assignment (absorbPublicInput input) :=
    statementValid.of_extends (absorbStatement_extends input)
  have publicEq := decodedBuilder_absorbMany input.transcriptBase
    ProductPoseidon2.constants assignment input.publicNifsFields
    (initialBuilder input) publicValid
  rw [decoded_initialBuilder assignment one input] at publicEq
  exact publicEq

theorem rows_replay_semantics (assignment : Nat -> Nat) (input : Input)
    (residues : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    ((deriveAlpha input).1.map (decoded assignment) =
        (valueDeriveAlpha assignment input).1) /\
      decoded assignment (deriveGamma input).1 =
        (valueDeriveGamma assignment input).1 /\
      (replayRounds input).challenges.map (decoded assignment) =
        (valueReplayRounds assignment input).challenges /\
      decodedBuilder assignment (replayRounds input).builder =
        (valueReplayRounds assignment input).state /\
      decodedBuilder assignment (afterFullOutput input) =
        valueAfterFullOutput assignment input := by
  have valid : Valid input.transcriptBase ProductPoseidon2.constants assignment
      (afterFullOutput input) := by
    apply valid_of_satisfied input.transcriptBase ProductPoseidon2.constants
      (afterFullOutput input) assignment residues one
    exact transcriptRows_satisfied input assignment satisfied
  have roundsValid : Valid input.transcriptBase ProductPoseidon2.constants
      assignment (replayRounds input).builder :=
    valid.of_extends (afterFullOutput_extends input)
  have gammaValid : Valid input.transcriptBase ProductPoseidon2.constants
      assignment (deriveGamma input).2 :=
    roundsValid.of_extends (replayRounds_extends input)
  have alphaValid : Valid input.transcriptBase ProductPoseidon2.constants
      assignment (deriveAlpha input).2 :=
    gammaValid.of_extends (deriveGamma_extends input)
  have statementValid : Valid input.transcriptBase ProductPoseidon2.constants
      assignment (absorbStatement input) :=
    alphaValid.of_extends (deriveAlpha_extends input)
  have publicValid : Valid input.transcriptBase ProductPoseidon2.constants
      assignment (absorbPublicInput input) :=
    statementValid.of_extends (absorbStatement_extends input)
  have publicEq := decodedBuilder_absorbMany input.transcriptBase
    ProductPoseidon2.constants assignment input.publicNifsFields
    (initialBuilder input) publicValid
  rw [decoded_initialBuilder assignment one input] at publicEq
  change decodedBuilder assignment (absorbPublicInput input) =
    valueAbsorbPublic assignment input at publicEq
  have statementEq := decodedBuilder_absorbMany input.transcriptBase
    ProductPoseidon2.constants assignment (statementFields input)
    (absorbPublicInput input) statementValid
  change decodedBuilder assignment (absorbStatement input) =
    Poseidon2Duplex.absorbList ProductPoseidon2.constants
      (fieldValues assignment (statementFields input))
      (decodedBuilder assignment (absorbPublicInput input)) at statementEq
  rw [publicEq] at statementEq
  change decodedBuilder assignment (absorbStatement input) =
    valueAbsorbStatement assignment input at statementEq
  have alphaEq := deriveAlphaGo_semantics assignment one input 0
    selectedShape.cubeVariables (absorbStatement input) alphaValid
  rw [statementEq] at alphaEq
  change
    ((deriveAlpha input).1.map (decoded assignment),
      decodedBuilder assignment (deriveAlpha input).2) =
      valueDeriveAlpha assignment input at alphaEq
  have alphaValuesEq := congrArg Prod.fst alphaEq
  have alphaStateEq := congrArg Prod.snd alphaEq
  simp only at alphaValuesEq alphaStateEq
  have gammaEq := squeezeVerifierChallenge_semantics assignment one
    2 2 43 [] input (deriveAlpha input).2 gammaValid
  rw [alphaStateEq] at gammaEq
  change
    (decoded assignment (deriveGamma input).1,
      decodedBuilder assignment (deriveGamma input).2) =
      valueDeriveGamma assignment input at gammaEq
  have gammaValueEq := congrArg Prod.fst gammaEq
  have gammaStateEq := congrArg Prod.snd gammaEq
  simp only at gammaValueEq gammaStateEq
  have roundsEq := replayRoundsGo_semantics assignment one input
    (List.ofFn input.rounds) 0 (deriveGamma input).2 roundsValid
  rw [gammaStateEq] at roundsEq
  change
    ((replayRounds input).challenges.map (decoded assignment),
      decodedBuilder assignment (replayRounds input).builder) =
      ((valueReplayRounds assignment input).challenges,
        (valueReplayRounds assignment input).state) at roundsEq
  have roundValuesEq := congrArg Prod.fst roundsEq
  have roundStateEq := congrArg Prod.snd roundsEq
  simp only at roundValuesEq roundStateEq
  have fullEq := decodedBuilder_absorbMany input.transcriptBase
    ProductPoseidon2.constants assignment (fullOutputFields input)
    (replayRounds input).builder valid
  rw [roundStateEq] at fullEq
  exact
    ⟨alphaValuesEq, gammaValueEq, roundValuesEq, roundStateEq,
      by simpa [afterFullOutput, valueAfterFullOutput] using fullEq⟩

end Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptSemantics
