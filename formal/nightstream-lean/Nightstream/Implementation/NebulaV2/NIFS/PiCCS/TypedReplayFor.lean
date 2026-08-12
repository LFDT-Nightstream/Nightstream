import Nightstream.Implementation.NebulaV2.NIFS.PiCCS.TranscriptSemanticsFor
import Nightstream.Implementation.NebulaV2.NIFS.PiCCS.TypedBridge

/-!
Contract: key-independent typed replay of the complete product PiCCS
transcript at an arbitrary generated-relation exponent.

The theorem connects the row-level public, statement, alpha, gamma, and
SumCheck round replay to `ProductPoseidon2.transcriptFor rowVariables`.
Placement contains only serialization equalities. It contains no challenge,
derived state, SumCheck result, PiCCS Boolean, or verifier result.

Does not own arithmetic acceptance, a concrete key, physical placement,
PiRLC, PiDEC, cryptographic security, or implementation refinement.

Assurance tier: exponent-indexed typed transcript refinement.
-/

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductPiCcsTypedReplayFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptRowsFor
open Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptSemanticsFor
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev PaperStatement (rowVariables : Nat) :=
  ProtocolVerifier.Statement K ProductPoseidon2.State
    (ProductNifsCodec.shapeFor rowVariables)

abbrev PaperCertificate (rowVariables : Nat) :=
  FiatShamir.Certificate K (ProductNifsCodec.shapeFor rowVariables)

structure Placement {rowVariables : Nat}
    (input : Input rowVariables)
    (statement : PaperStatement rowVariables)
    (certificate : PaperCertificate rowVariables)
    (assignment : Nat -> Nat) : Prop where
  publicState :
    valueAbsorbPublic assignment input = statement.priorState
  statementSerialization :
    fieldValues assignment (statementFields input) =
      ProductPoseidon2.statementFieldsFor rowVariables statement
  roundSerialization : forall round,
    fieldValues assignment (roundFields round.val (input.rounds round)) =
      ProductPoseidon2.roundFieldsFor round (certificate.rounds round)

theorem valueSqueeze_eq_concrete
    (assignment : Nat -> Nat) (one : assignment 0 = 1)
    (eventIndex challengeIndex challengeType : Nat)
    (coordinates : List Nat) (state : ProductPoseidon2.State) :
    (ofProjection
        (valueSqueezeVerifierChallenge assignment eventIndex challengeIndex
          challengeType coordinates state).1,
      (valueSqueezeVerifierChallenge assignment eventIndex challengeIndex
        challengeType coordinates state).2) =
      ProductPoseidon2.squeezeVerifierChallenge eventIndex challengeIndex
        challengeType coordinates state := by
  simpa [ProductPiCcsTranscriptSemanticsFor.valueSqueezeVerifierChallenge,
    ProductPiCcsTranscriptSemantics.valueSqueezeVerifierChallenge,
    ProductPiCcsTranscriptSemanticsFor.fieldValues,
    ProductPiCcsTranscriptSemantics.fieldValues,
    ProductPiCcsTranscriptRowsFor.verifierChallengeFields] using
      ProductPiCcsTypedBridge.valueSqueeze_eq_concrete assignment one
        eventIndex challengeIndex challengeType coordinates state

def concreteAlphaIndices :
    List Nat -> ProductPoseidon2.State -> List K × ProductPoseidon2.State
  | [], state => ([], state)
  | index :: rest, state =>
      let sampled := ProductPoseidon2.squeezeVerifierChallenge
        1 1 42 [index] state
      let tail := concreteAlphaIndices rest sampled.2
      (sampled.1 :: tail.1, tail.2)

theorem valueAlphaGo_eq_concrete {rowVariables : Nat}
    (assignment : Nat -> Nat) (one : assignment 0 = 1)
    (input : Input rowVariables) :
    forall index count state,
      ((valueDeriveAlphaGo assignment input index count state).1.map
          ofProjection,
        (valueDeriveAlphaGo assignment input index count state).2) =
      concreteAlphaIndices (List.range' index count) state
  | _, 0, _ => rfl
  | index, count + 1, state => by
      let valueSample := valueSqueezeVerifierChallenge assignment 1 1 42
        [index] state
      let concreteSample := ProductPoseidon2.squeezeVerifierChallenge
        1 1 42 [index] state
      have sampleEq := valueSqueeze_eq_concrete assignment one 1 1 42
        [index] state
      have sampleValueEq := congrArg Prod.fst sampleEq
      have sampleStateEq := congrArg Prod.snd sampleEq
      simp only at sampleValueEq sampleStateEq
      simp only [valueDeriveAlphaGo, List.range'_succ,
        concreteAlphaIndices, List.map_cons]
      rw [sampleValueEq, sampleStateEq]
      exact congrArg
        (fun tail : List K × ProductPoseidon2.State =>
          (concreteSample.1 :: tail.1, tail.2))
        (valueAlphaGo_eq_concrete assignment one input
          (index + 1) count concreteSample.2)

theorem squeezeMany_alpha_eq_indices {rowVariables : Nat}
    (indices : List
      (Fin (ProductNifsCodec.shapeFor rowVariables).cubeVariables))
    (state : ProductPoseidon2.State) :
    FiatShamir.squeezeMany (ProductPoseidon2.transcriptFor rowVariables) state
        (indices.map FiatShamir.ChallengeLabel.alpha) =
      concreteAlphaIndices (indices.map fun index => index.val) state := by
  induction indices generalizing state with
  | nil => rfl
  | cons index rest inductionHypothesis =>
      let sampled := ProductPoseidon2.squeezeVerifierChallenge
        1 1 42 [index.val] state
      have tailEq := inductionHypothesis sampled.2
      simpa [FiatShamir.squeezeMany, ProductPoseidon2.transcriptFor,
        concreteAlphaIndices, sampled] using
          congrArg
            (fun tail : List K × ProductPoseidon2.State =>
              (sampled.1 :: tail.1, tail.2)) tailEq

theorem valueAlpha_eq_paper {rowVariables : Nat}
    (assignment : Nat -> Nat) (one : assignment 0 = 1)
    (input : Input rowVariables) (initialState : ProductPoseidon2.State) :
    ((valueDeriveAlphaGo assignment input 0
        (Shape rowVariables).cubeVariables initialState).1.map ofProjection,
      (valueDeriveAlphaGo assignment input 0
        (Shape rowVariables).cubeVariables initialState).2) =
      FiatShamir.squeezeMany (ProductPoseidon2.transcriptFor rowVariables)
        initialState
        (FiatShamir.alphaLabels (ProductNifsCodec.shapeFor rowVariables)) := by
  have valueEq := valueAlphaGo_eq_concrete assignment one input 0
    (Shape rowVariables).cubeVariables initialState
  have paperEq := squeezeMany_alpha_eq_indices
    (canonicalFinIndices (Shape rowVariables).cubeVariables) initialState
  unfold FiatShamir.alphaLabels at paperEq
  rw [ProductPiCcsTypedBridge.canonicalFinIndices_values,
    List.range_eq_range'] at paperEq
  exact valueEq.trans paperEq.symm

theorem valueAbsorbStatement_eq_paperInitial {rowVariables : Nat}
    (input : Input rowVariables) (statement : PaperStatement rowVariables)
    (certificate : PaperCertificate rowVariables)
    (assignment : Nat -> Nat)
    (placement : Placement input statement certificate assignment) :
    valueAbsorbStatement assignment input =
      (ProductPoseidon2.transcriptFor rowVariables).initialState statement := by
  unfold valueAbsorbStatement
  rw [placement.publicState, placement.statementSerialization]
  rfl

theorem valuePreSumcheck_eq_paper {rowVariables : Nat}
    (input : Input rowVariables) (statement : PaperStatement rowVariables)
    (certificate : PaperCertificate rowVariables)
    (assignment : Nat -> Nat) (one : assignment 0 = 1)
    (placement : Placement input statement certificate assignment) :
    let paper := FiatShamir.derivePreSumcheck
      (ProductPoseidon2.transcriptFor rowVariables) statement
    (valueDeriveAlpha assignment input).1.map ofProjection =
        paper.alpha.coordinates /\
      ofProjection (valueDeriveGamma assignment input).1 = paper.gamma /\
      (valueDeriveGamma assignment input).2 = paper.state := by
  dsimp only
  let paper := FiatShamir.derivePreSumcheck
    (ProductPoseidon2.transcriptFor rowVariables) statement
  have initialEq := valueAbsorbStatement_eq_paperInitial input statement
    certificate assignment placement
  have alphaEq := valueAlpha_eq_paper assignment one input
    (valueAbsorbStatement assignment input)
  have gammaEq := valueSqueeze_eq_concrete assignment one 2 2 43 []
    (valueDeriveAlpha assignment input).2
  change
    ((valueDeriveAlpha assignment input).1.map ofProjection,
      (valueDeriveAlpha assignment input).2) =
      FiatShamir.squeezeMany (ProductPoseidon2.transcriptFor rowVariables)
        (valueAbsorbStatement assignment input)
        (FiatShamir.alphaLabels
          (ProductNifsCodec.shapeFor rowVariables)) at alphaEq
  rw [initialEq] at alphaEq
  have alphaValuesEq := congrArg Prod.fst alphaEq
  have alphaStateEq := congrArg Prod.snd alphaEq
  simp only at alphaValuesEq alphaStateEq
  have gammaValueEq := congrArg Prod.fst gammaEq
  have gammaStateEq := congrArg Prod.snd gammaEq
  simp only at gammaValueEq gammaStateEq
  unfold FiatShamir.derivePreSumcheck
  change
    (valueDeriveAlpha assignment input).1.map ofProjection =
        (FiatShamir.squeezeMany (ProductPoseidon2.transcriptFor rowVariables)
          ((ProductPoseidon2.transcriptFor rowVariables).initialState statement)
          (FiatShamir.alphaLabels
            (ProductNifsCodec.shapeFor rowVariables))).1 /\
      ofProjection (valueDeriveGamma assignment input).1 =
        ((ProductPoseidon2.transcriptFor rowVariables).squeeze
          (FiatShamir.squeezeMany
            (ProductPoseidon2.transcriptFor rowVariables)
            ((ProductPoseidon2.transcriptFor rowVariables).initialState statement)
            (FiatShamir.alphaLabels
              (ProductNifsCodec.shapeFor rowVariables))).2 .gamma).1 /\
      (valueDeriveGamma assignment input).2 =
        ((ProductPoseidon2.transcriptFor rowVariables).squeeze
          (FiatShamir.squeezeMany
            (ProductPoseidon2.transcriptFor rowVariables)
            ((ProductPoseidon2.transcriptFor rowVariables).initialState statement)
            (FiatShamir.alphaLabels
              (ProductNifsCodec.shapeFor rowVariables))).2 .gamma).2
  refine ⟨alphaValuesEq, ?_, ?_⟩
  · rw [← alphaStateEq]
    simpa [valueDeriveGamma, ProductPoseidon2.transcriptFor] using gammaValueEq
  · rw [← alphaStateEq]
    simpa [valueDeriveGamma, ProductPoseidon2.transcriptFor] using gammaStateEq

theorem valueReplayRoundsGo_eq_paper {rowVariables : Nat}
    (input : Input rowVariables) (statement : PaperStatement rowVariables)
    (certificate : PaperCertificate rowVariables)
    (assignment : Nat -> Nat) (one : assignment 0 = 1)
    (placement : Placement input statement certificate assignment) :
    forall
      (indices : List (Fin (Shape rowVariables).cubeVariables))
      (rounds : List Round) (index : Nat)
      (state : ProductPoseidon2.State),
      indices.map input.rounds = rounds ->
      indices.map (fun coordinate => coordinate.val) =
        List.range' index rounds.length ->
      ((valueReplayRoundsGo assignment input rounds index state).challenges.map
          ofProjection,
        (valueReplayRoundsGo assignment input rounds index state).state) =
        FiatShamir.deriveRoundsFrom
          (ProductPoseidon2.transcriptFor rowVariables)
          certificate.rounds state indices
  | [], rounds, _, _, roundsEq, _ => by
      have empty : rounds = [] := by
        simpa only [List.map_nil] using roundsEq.symm
      subst rounds
      rfl
  | _ :: _, [], _, _, roundsEq, _ => by
      change _ :: _ = [] at roundsEq
      cases roundsEq
  | coordinate :: rest, round :: rounds, index, state,
      roundsEq, indicesEq => by
      change input.rounds coordinate :: rest.map input.rounds =
        round :: rounds at roundsEq
      have roundEq := (List.cons.inj roundsEq).1
      have restRoundsEq := (List.cons.inj roundsEq).2
      change coordinate.val :: rest.map (fun item => item.val) =
        index :: List.range' (index + 1) rounds.length at indicesEq
      have coordinateEq := (List.cons.inj indicesEq).1
      have restIndicesEq := (List.cons.inj indicesEq).2
      subst round
      subst index
      let valueAbsorbed :=
        Poseidon2Duplex.absorbList ProductPoseidon2.constants
          (fieldValues assignment
            (roundFields coordinate.val (input.rounds coordinate))) state
      let paperAbsorbed :=
        Poseidon2Duplex.absorbList ProductPoseidon2.constants
          (ProductPoseidon2.roundFieldsFor coordinate
            (certificate.rounds coordinate)) state
      have absorbedEq : valueAbsorbed = paperAbsorbed := by
        unfold valueAbsorbed paperAbsorbed
        rw [placement.roundSerialization coordinate]
      let valueSample := valueSqueezeVerifierChallenge assignment
        (4 + 2 * coordinate.val) (3 + coordinate.val) 46 [] valueAbsorbed
      let paperSample := ProductPoseidon2.squeezeVerifierChallenge
        (4 + 2 * coordinate.val) (3 + coordinate.val) 46 [] paperAbsorbed
      have sampleEq := valueSqueeze_eq_concrete assignment one
        (4 + 2 * coordinate.val) (3 + coordinate.val) 46 [] valueAbsorbed
      change (ofProjection valueSample.1, valueSample.2) = _ at sampleEq
      rw [absorbedEq] at sampleEq
      change (ofProjection valueSample.1, valueSample.2) = paperSample at sampleEq
      have sampleValueEq := congrArg Prod.fst sampleEq
      have sampleStateEq := congrArg Prod.snd sampleEq
      simp only at sampleValueEq sampleStateEq
      simp only [valueReplayRoundsGo, FiatShamir.deriveRoundsFrom,
        ProductPoseidon2.transcriptFor, List.map_cons]
      change
        (ofProjection valueSample.1 ::
            (valueReplayRoundsGo assignment input rounds
              (coordinate.val + 1) valueSample.2).challenges.map ofProjection,
          (valueReplayRoundsGo assignment input rounds
            (coordinate.val + 1) valueSample.2).state) =
          (paperSample.1 ::
            (FiatShamir.deriveRoundsFrom
              (ProductPoseidon2.transcriptFor rowVariables)
              certificate.rounds paperSample.2 rest).1,
            (FiatShamir.deriveRoundsFrom
              (ProductPoseidon2.transcriptFor rowVariables)
              certificate.rounds paperSample.2 rest).2)
      rw [sampleValueEq, sampleStateEq]
      exact congrArg
        (fun tail : List K × ProductPoseidon2.State =>
          (paperSample.1 :: tail.1, tail.2))
        (valueReplayRoundsGo_eq_paper input statement certificate assignment
          one placement rest rounds (coordinate.val + 1) paperSample.2
          restRoundsEq restIndicesEq)

theorem canonicalRoundWires {rowVariables : Nat}
    (input : Input rowVariables) :
    (canonicalFinIndices (Shape rowVariables).cubeVariables).map input.rounds =
      List.ofFn input.rounds := by
  apply List.ext_getElem
  · simp only [List.length_map, canonicalFinIndices_length,
      List.length_ofFn]
  · intro index leftBound rightBound
    simp only [List.getElem_map]
    have sourceBound :
        index < (canonicalFinIndices
          (Shape rowVariables).cubeVariables).length := by
      simpa only [List.length_map] using leftBound
    change input.rounds
        ((canonicalFinIndices (Shape rowVariables).cubeVariables
          )[index]'sourceBound) = (List.ofFn input.rounds)[index]
    simp only [canonicalFinIndices, List.getElem_ofFn]
    congr 1

theorem decodedAlpha_coordinates_eq {rowVariables : Nat}
    (input : Input rowVariables) (assignment : Nat -> Nat) :
    (KPiCcsOccurrence.decodedAlpha (occurrenceInput input) assignment
      ).coordinates =
      (deriveAlpha input).1.map fun value =>
        ofProjection (decoded assignment value) := by
  apply List.ext_getElem
  · simp only [KPiCcsOccurrence.decodedAlpha,
      KPiCcsOccurrence.terminalInput, KPiCcsTerminal.decodedAlpha,
      KPiCcsTerminal.alphaEqualityInput, KPointEquality.decodedRight,
      KPointEquality.indices, List.length_map, List.length_ofFn,
      deriveAlpha_length]
  · intro index leftBound rightBound
    simp only [KPiCcsOccurrence.decodedAlpha,
      KPiCcsOccurrence.terminalInput, KPiCcsTerminal.decodedAlpha,
      KPiCcsTerminal.alphaEqualityInput, KPointEquality.decodedRight,
      KPointEquality.indices, List.getElem_map, List.getElem_ofFn,
      occurrenceInput, alphaAt, KPointEquality.decoded, decoded]
    congr 3

theorem decodedPoint_coordinates_eq {rowVariables : Nat}
    (input : Input rowVariables) (assignment : Nat -> Nat) :
    (KPiCcsOccurrence.decodedPoint (occurrenceInput input) assignment
      ).coordinates =
      (replayRounds input).challenges.map fun value =>
        ofProjection (decoded assignment value) := by
  apply List.ext_getElem
  · simp only [KPiCcsOccurrence.decodedPoint,
      KPiCcsOccurrence.terminalInput, KPiCcsTerminal.decodedPoint,
      KPiCcsTerminal.alphaEqualityInput, KPointEquality.decodedLeft,
      KPointEquality.indices, List.length_map, List.length_ofFn,
      replayRounds_length]
  · intro index leftBound rightBound
    simp only [KPiCcsOccurrence.decodedPoint,
      KPiCcsOccurrence.terminalInput, KPiCcsTerminal.decodedPoint,
      KPiCcsTerminal.alphaEqualityInput, KPointEquality.decodedLeft,
      KPointEquality.indices, List.getElem_map, List.getElem_ofFn,
      occurrenceInput, pointAt, KPointEquality.decoded, decoded]
    congr 3

theorem valueReplay_eq_derived {rowVariables : Nat}
    (input : Input rowVariables) (statement : PaperStatement rowVariables)
    (certificate : PaperCertificate rowVariables)
    (assignment : Nat -> Nat) (one : assignment 0 = 1)
    (placement : Placement input statement certificate assignment) :
    let derived := FiatShamir.derive
      (ProductPoseidon2.transcriptFor rowVariables) statement certificate
    (valueDeriveAlpha assignment input).1.map ofProjection =
        derived.alpha.coordinates /\
      ofProjection (valueDeriveGamma assignment input).1 = derived.gamma /\
      (valueReplayRounds assignment input).challenges.map ofProjection =
        derived.roundPoint.coordinates /\
      (valueReplayRounds assignment input).state = derived.finalState := by
  dsimp only
  let pre := FiatShamir.derivePreSumcheck
    (ProductPoseidon2.transcriptFor rowVariables) statement
  have preEq := valuePreSumcheck_eq_paper input statement certificate
    assignment one placement
  change
    (valueDeriveAlpha assignment input).1.map ofProjection =
        pre.alpha.coordinates /\
      ofProjection (valueDeriveGamma assignment input).1 = pre.gamma /\
      (valueDeriveGamma assignment input).2 = pre.state at preEq
  have roundsEq := valueReplayRoundsGo_eq_paper input statement certificate
    assignment one placement
    (canonicalFinIndices (Shape rowVariables).cubeVariables)
    (List.ofFn input.rounds) 0 (valueDeriveGamma assignment input).2
    (canonicalRoundWires input)
    (by
      simp [ProductPiCcsTypedBridge.canonicalFinIndices_values,
        List.range_eq_range'])
  change
    ((valueReplayRounds assignment input).challenges.map ofProjection,
      (valueReplayRounds assignment input).state) =
      FiatShamir.deriveRoundsFrom
        (ProductPoseidon2.transcriptFor rowVariables) certificate.rounds
        (valueDeriveGamma assignment input).2
        (canonicalFinIndices (Shape rowVariables).cubeVariables) at roundsEq
  rw [preEq.2.2] at roundsEq
  have roundValuesEq := congrArg Prod.fst roundsEq
  have roundStateEq := congrArg Prod.snd roundsEq
  simp only at roundValuesEq roundStateEq
  exact ⟨preEq.1, preEq.2.1, roundValuesEq, roundStateEq⟩

theorem valueReplay_eq_derived_of_components {rowVariables : Nat}
    (input : Input rowVariables) (priorState : ProductPoseidon2.State)
    (verifierInput : ProtocolPolynomial.VerifierInput K
      (ProductNifsCodec.shapeFor rowVariables))
    (certificateRounds : Fin (Shape rowVariables).cubeVariables ->
      SumCheck.Finite.Message K)
    (assignment : Nat -> Nat) (one : assignment 0 = 1)
    (publicState : valueAbsorbPublic assignment input = priorState)
    (statementSerialization :
      fieldValues assignment (statementFields input) =
        ProductPoseidon2.statementFieldsFor rowVariables
          ({ priorState := priorState, input := verifierInput } :
            PaperStatement rowVariables))
    (roundSerialization : forall round,
      fieldValues assignment (roundFields round.val (input.rounds round)) =
        ProductPoseidon2.roundFieldsFor round (certificateRounds round)) :
    let statement : PaperStatement rowVariables :=
      { priorState := priorState, input := verifierInput }
    let certificate : PaperCertificate rowVariables :=
      { rounds := certificateRounds }
    let derived := FiatShamir.derive
      (ProductPoseidon2.transcriptFor rowVariables) statement certificate
    (valueDeriveAlpha assignment input).1.map ofProjection =
        derived.alpha.coordinates /\
      ofProjection (valueDeriveGamma assignment input).1 = derived.gamma /\
      (valueReplayRounds assignment input).challenges.map ofProjection =
        derived.roundPoint.coordinates /\
      (valueReplayRounds assignment input).state = derived.finalState := by
  exact valueReplay_eq_derived input
    { priorState := priorState, input := verifierInput }
    { rounds := certificateRounds } assignment one
    { publicState := publicState
      statementSerialization := statementSerialization
      roundSerialization := roundSerialization }

end Nightstream.Implementation.NebulaV2.ProductPiCcsTypedReplayFor
