import Nightstream.Implementation.NebulaV2.NIFS.PiCCS.TypedBridge

/-!
Contract: key-independent typed replay of the exact product PiCCS transcript.

Owns the proof that the public, statement, alpha, gamma, and fixed SumCheck
round fields in one `ProductPiCcsTranscriptRows.Input` compute the exact
Poseidon2 Fiat-Shamir coins for an independently defined paper statement and
certificate.

The placement contains only value serialization equalities. It contains no
challenge, derived state, SumCheck result, PiCCS Boolean, or verifier result.

Does not own arithmetic PiCCS acceptance, a concrete key, generated column
placement, PiRLC, PiDEC, cryptographic security, or implementation refinement.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductPiCcsTypedReplay

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptRows
open Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptSemantics
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev PaperStatement :=
  ProtocolVerifier.Statement K ProductPoseidon2.State ProductNifsCodec.shape

abbrev PaperCertificate :=
  FiatShamir.Certificate K ProductNifsCodec.shape

/-- Serialization facts for one transcript replay. Each equality has a
physical row expression on the left and an independent paper value on the
right. -/
structure Placement
    (input : Input)
    (statement : PaperStatement)
    (certificate : PaperCertificate)
    (assignment : Nat -> Nat) : Prop where
  publicState :
    valueAbsorbPublic assignment input = statement.priorState
  statementSerialization :
    fieldValues assignment (statementFields input) =
      ProductPoseidon2.statementFields statement
  roundSerialization : forall round,
    fieldValues assignment (roundFields round.val (input.rounds round)) =
      ProductPoseidon2.roundFields round (certificate.rounds round)

/-- Public and statement absorption reaches the exact paper initial state. -/
theorem valueAbsorbStatement_eq_paperInitial
    (input : Input) (statement : PaperStatement)
    (certificate : PaperCertificate) (assignment : Nat -> Nat)
    (placement : Placement input statement certificate assignment) :
    valueAbsorbStatement assignment input =
      ProductPoseidon2.transcript.initialState statement := by
  unfold valueAbsorbStatement
  rw [placement.publicState, placement.statementSerialization]
  rfl

/-- The row pre-SumCheck replay gives the paper alpha, gamma, and state. -/
theorem valuePreSumcheck_eq_paper
    (input : Input) (statement : PaperStatement)
    (certificate : PaperCertificate) (assignment : Nat -> Nat)
    (one : assignment 0 = 1)
    (placement : Placement input statement certificate assignment) :
    let paper :=
      FiatShamir.derivePreSumcheck ProductPoseidon2.transcript statement
    (valueDeriveAlpha assignment input).1.map
          KConcreteFixedPhaseBridge.ofProjection = paper.alpha.coordinates /\
      KConcreteFixedPhaseBridge.ofProjection
          (valueDeriveGamma assignment input).1 = paper.gamma /\
      (valueDeriveGamma assignment input).2 = paper.state := by
  dsimp only
  let paper := FiatShamir.derivePreSumcheck ProductPoseidon2.transcript statement
  have initialEq := valueAbsorbStatement_eq_paperInitial input statement
    certificate assignment placement
  have alphaEq := ProductPiCcsTypedBridge.valueAlpha_eq_paper assignment one
    input (valueAbsorbStatement assignment input)
  have gammaEq := ProductPiCcsTypedBridge.valueSqueeze_eq_concrete assignment
    one 2 2 43 [] (valueDeriveAlpha assignment input).2
  change
    ((valueDeriveAlpha assignment input).1.map
        KConcreteFixedPhaseBridge.ofProjection,
      (valueDeriveAlpha assignment input).2) =
      FiatShamir.squeezeMany ProductPoseidon2.transcript
        (valueAbsorbStatement assignment input)
        (FiatShamir.alphaLabels ProductNifsCodec.shape) at alphaEq
  rw [initialEq] at alphaEq
  have alphaValuesEq := congrArg Prod.fst alphaEq
  have alphaStateEq := congrArg Prod.snd alphaEq
  simp only at alphaValuesEq alphaStateEq
  have gammaValueEq := congrArg Prod.fst gammaEq
  have gammaStateEq := congrArg Prod.snd gammaEq
  simp only at gammaValueEq gammaStateEq
  unfold FiatShamir.derivePreSumcheck
  change
    (valueDeriveAlpha assignment input).1.map
        KConcreteFixedPhaseBridge.ofProjection =
        (FiatShamir.squeezeMany ProductPoseidon2.transcript
          (ProductPoseidon2.transcript.initialState statement)
          (FiatShamir.alphaLabels ProductNifsCodec.shape)).1 /\
      KConcreteFixedPhaseBridge.ofProjection
          (valueDeriveGamma assignment input).1 =
        (ProductPoseidon2.transcript.squeeze
          (FiatShamir.squeezeMany ProductPoseidon2.transcript
            (ProductPoseidon2.transcript.initialState statement)
            (FiatShamir.alphaLabels ProductNifsCodec.shape)).2
          .gamma).1 /\
      (valueDeriveGamma assignment input).2 =
        (ProductPoseidon2.transcript.squeeze
          (FiatShamir.squeezeMany ProductPoseidon2.transcript
            (ProductPoseidon2.transcript.initialState statement)
            (FiatShamir.alphaLabels ProductNifsCodec.shape)).2
          .gamma).2
  refine ⟨alphaValuesEq, ?_, ?_⟩
  · rw [← alphaStateEq]
    simpa [valueDeriveGamma, ProductPoseidon2.transcript] using gammaValueEq
  · rw [← alphaStateEq]
    simpa [valueDeriveGamma, ProductPoseidon2.transcript] using gammaStateEq

/-- Replay any canonical suffix of fixed SumCheck rounds. -/
theorem valueReplayRoundsGo_eq_paper
    (input : Input) (statement : PaperStatement)
    (certificate : PaperCertificate) (assignment : Nat -> Nat)
    (one : assignment 0 = 1)
    (placement : Placement input statement certificate assignment) :
    forall
      (indices : List (Fin ProductNifsCodec.shape.cubeVariables))
      (rounds : List Round) (index : Nat)
      (state : ProductPoseidon2.State),
      indices.map input.rounds = rounds ->
      indices.map (fun coordinate => coordinate.val) =
        List.range' index rounds.length ->
      ((valueReplayRoundsGo assignment input rounds index state).challenges.map
          KConcreteFixedPhaseBridge.ofProjection,
        (valueReplayRoundsGo assignment input rounds index state).state) =
        FiatShamir.deriveRoundsFrom ProductPoseidon2.transcript
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
          (ProductPoseidon2.roundFields coordinate
            (certificate.rounds coordinate)) state
      have absorbedEq : valueAbsorbed = paperAbsorbed := by
        unfold valueAbsorbed paperAbsorbed
        rw [placement.roundSerialization coordinate]
      let valueSample := valueSqueezeVerifierChallenge assignment
        (4 + 2 * coordinate.val) (3 + coordinate.val) 46 [] valueAbsorbed
      let paperSample := ProductPoseidon2.squeezeVerifierChallenge
        (4 + 2 * coordinate.val) (3 + coordinate.val) 46 [] paperAbsorbed
      have sampleEq := ProductPiCcsTypedBridge.valueSqueeze_eq_concrete
        assignment one (4 + 2 * coordinate.val) (3 + coordinate.val) 46 []
        valueAbsorbed
      change
        (KConcreteFixedPhaseBridge.ofProjection valueSample.1,
          valueSample.2) = _ at sampleEq
      rw [absorbedEq] at sampleEq
      change
        (KConcreteFixedPhaseBridge.ofProjection valueSample.1,
          valueSample.2) = paperSample at sampleEq
      have sampleValueEq := congrArg Prod.fst sampleEq
      have sampleStateEq := congrArg Prod.snd sampleEq
      simp only at sampleValueEq sampleStateEq
      simp only [valueReplayRoundsGo, FiatShamir.deriveRoundsFrom,
        ProductPoseidon2.transcript, List.map_cons]
      change
        (KConcreteFixedPhaseBridge.ofProjection valueSample.1 ::
            (valueReplayRoundsGo assignment input rounds
              (coordinate.val + 1) valueSample.2).challenges.map
                KConcreteFixedPhaseBridge.ofProjection,
          (valueReplayRoundsGo assignment input rounds
            (coordinate.val + 1) valueSample.2).state) =
          (paperSample.1 ::
            (FiatShamir.deriveRoundsFrom ProductPoseidon2.transcript
              certificate.rounds paperSample.2 rest).1,
            (FiatShamir.deriveRoundsFrom ProductPoseidon2.transcript
              certificate.rounds paperSample.2 rest).2)
      rw [sampleValueEq, sampleStateEq]
      exact congrArg
        (fun tail : List K × ProductPoseidon2.State =>
          (paperSample.1 :: tail.1, tail.2))
        (valueReplayRoundsGo_eq_paper input statement certificate assignment
          one placement rest rounds (coordinate.val + 1) paperSample.2
          restRoundsEq restIndicesEq)

/-- Canonical finite indices select the physical round function in order. -/
theorem canonicalRoundWires (input : Input) :
    (canonicalFinIndices ProductNifsCodec.shape.cubeVariables).map
        input.rounds =
      List.ofFn input.rounds := by
  apply List.ext_getElem
  · simp only [List.length_map, canonicalFinIndices_length,
      List.length_ofFn, selectedShape]
  · intro index leftBound rightBound
    simp only [List.getElem_map]
    have sourceBound :
        index <
          (canonicalFinIndices ProductNifsCodec.shape.cubeVariables).length := by
      simpa only [List.length_map] using leftBound
    change input.rounds
        ((canonicalFinIndices ProductNifsCodec.shape.cubeVariables
          )[index]'sourceBound) =
      (List.ofFn input.rounds)[index]
    simp only [canonicalFinIndices, List.getElem_ofFn]
    congr 1

/-- The arithmetic occurrence's alpha point is the canonical decoded list
from the same transcript replay. This statement is independent of any key. -/
theorem decodedAlpha_coordinates_eq
    (input : Input) (assignment : Nat -> Nat) :
    (KPiCcsOccurrence.decodedAlpha
      (ProductPiCcsTranscriptRows.occurrenceInput input) assignment
      ).coordinates =
      (ProductPiCcsTranscriptRows.deriveAlpha input).1.map fun value =>
        KConcreteFixedPhaseBridge.ofProjection
          (ProductPiCcsTranscriptSemantics.decoded assignment value) := by
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
      ProductPiCcsTranscriptRows.occurrenceInput,
      ProductPiCcsTranscriptRows.alphaAt, KPointEquality.decoded,
      ProductPiCcsTranscriptSemantics.decoded]
    congr 3

/-- The arithmetic occurrence's SumCheck point is the canonical decoded list
from the same transcript replay. This statement is independent of any key. -/
theorem decodedPoint_coordinates_eq
    (input : Input) (assignment : Nat -> Nat) :
    (KPiCcsOccurrence.decodedPoint
      (ProductPiCcsTranscriptRows.occurrenceInput input) assignment
      ).coordinates =
      (ProductPiCcsTranscriptRows.replayRounds input).challenges.map fun value =>
        KConcreteFixedPhaseBridge.ofProjection
          (ProductPiCcsTranscriptSemantics.decoded assignment value) := by
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
      ProductPiCcsTranscriptRows.occurrenceInput,
      ProductPiCcsTranscriptRows.pointAt, KPointEquality.decoded,
      ProductPiCcsTranscriptSemantics.decoded]
    congr 3

/-- The full fixed-width replay gives the paper round point and final state. -/
theorem valueRounds_eq_paper
    (input : Input) (statement : PaperStatement)
    (certificate : PaperCertificate) (assignment : Nat -> Nat)
    (one : assignment 0 = 1)
    (placement : Placement input statement certificate assignment) :
    let pre := FiatShamir.derivePreSumcheck ProductPoseidon2.transcript statement
    ((valueReplayRounds assignment input).challenges.map
        KConcreteFixedPhaseBridge.ofProjection,
      (valueReplayRounds assignment input).state) =
      FiatShamir.deriveRoundsFrom ProductPoseidon2.transcript
        certificate.rounds pre.state
        (canonicalFinIndices ProductNifsCodec.shape.cubeVariables) := by
  dsimp only
  let pre := FiatShamir.derivePreSumcheck ProductPoseidon2.transcript statement
  have preEq := valuePreSumcheck_eq_paper input statement certificate assignment
    one placement
  change
    (valueDeriveAlpha assignment input).1.map
        KConcreteFixedPhaseBridge.ofProjection = pre.alpha.coordinates /\
      KConcreteFixedPhaseBridge.ofProjection
          (valueDeriveGamma assignment input).1 = pre.gamma /\
      (valueDeriveGamma assignment input).2 = pre.state at preEq
  have replay := valueReplayRoundsGo_eq_paper input statement certificate
    assignment one placement
    (canonicalFinIndices ProductNifsCodec.shape.cubeVariables)
    (List.ofFn input.rounds) 0 (valueDeriveGamma assignment input).2
    (canonicalRoundWires input)
    (by
      simp [selectedShape, ProductPiCcsTypedBridge.canonicalFinIndices_values,
        List.range_eq_range'])
  change
    ((valueReplayRounds assignment input).challenges.map
        KConcreteFixedPhaseBridge.ofProjection,
      (valueReplayRounds assignment input).state) =
      FiatShamir.deriveRoundsFrom ProductPoseidon2.transcript
        certificate.rounds (valueDeriveGamma assignment input).2
        (canonicalFinIndices ProductNifsCodec.shape.cubeVariables) at replay
  exact replay.trans (congrArg
    (fun state =>
      FiatShamir.deriveRoundsFrom ProductPoseidon2.transcript
        certificate.rounds state
        (canonicalFinIndices ProductNifsCodec.shape.cubeVariables))
    preEq.2.2)

private theorem derive_components
    {Context Field State : Type}
    {shape : Shape}
    (oracle : FiatShamir.Oracle Context Field State shape)
    (context : Context) (certificate : FiatShamir.Certificate Field shape) :
    let pre := FiatShamir.derivePreSumcheck oracle context
    let rounds := FiatShamir.deriveRoundsFrom oracle certificate.rounds
      pre.state (canonicalFinIndices shape.cubeVariables)
    let derived := FiatShamir.derive oracle context certificate
    derived.alpha = pre.alpha /\
      derived.gamma = pre.gamma /\
      derived.roundPoint.coordinates = rounds.1 /\
      derived.finalState = rounds.2 := by
  exact ⟨rfl, rfl, rfl, rfl⟩

/-- The complete row replay is the exact Fiat-Shamir coin record. -/
theorem valueReplay_eq_derived
    (input : Input) (statement : PaperStatement)
    (certificate : PaperCertificate) (assignment : Nat -> Nat)
    (one : assignment 0 = 1)
    (placement : Placement input statement certificate assignment) :
    let derived := FiatShamir.derive ProductPoseidon2.transcript statement
      certificate
    (valueDeriveAlpha assignment input).1.map
          KConcreteFixedPhaseBridge.ofProjection = derived.alpha.coordinates /\
      KConcreteFixedPhaseBridge.ofProjection
          (valueDeriveGamma assignment input).1 = derived.gamma /\
      (valueReplayRounds assignment input).challenges.map
          KConcreteFixedPhaseBridge.ofProjection =
        derived.roundPoint.coordinates /\
      (valueReplayRounds assignment input).state = derived.finalState := by
  dsimp only
  let pre := FiatShamir.derivePreSumcheck ProductPoseidon2.transcript statement
  let roundResult := FiatShamir.deriveRoundsFrom ProductPoseidon2.transcript
    certificate.rounds pre.state
    (canonicalFinIndices ProductNifsCodec.shape.cubeVariables)
  let derived := FiatShamir.derive ProductPoseidon2.transcript statement
    certificate
  have preEq := valuePreSumcheck_eq_paper input statement certificate assignment
    one placement
  change
    (valueDeriveAlpha assignment input).1.map
        KConcreteFixedPhaseBridge.ofProjection = pre.alpha.coordinates /\
      KConcreteFixedPhaseBridge.ofProjection
          (valueDeriveGamma assignment input).1 = pre.gamma /\
      (valueDeriveGamma assignment input).2 = pre.state at preEq
  have roundsEq := valueRounds_eq_paper input statement certificate assignment
    one placement
  change
    ((valueReplayRounds assignment input).challenges.map
        KConcreteFixedPhaseBridge.ofProjection,
      (valueReplayRounds assignment input).state) = roundResult at roundsEq
  have roundValuesEq := congrArg Prod.fst roundsEq
  have roundStateEq := congrArg Prod.snd roundsEq
  simp only at roundValuesEq roundStateEq
  have components := derive_components ProductPoseidon2.transcript statement
    certificate
  change derived.alpha = pre.alpha /\
      derived.gamma = pre.gamma /\
      derived.roundPoint.coordinates = roundResult.1 /\
      derived.finalState = roundResult.2 at components
  exact
    ⟨preEq.1.trans (congrArg CubePoint.coordinates components.1.symm),
      preEq.2.1.trans components.2.1.symm,
      roundValuesEq.trans components.2.2.1.symm,
      roundStateEq.trans components.2.2.2.symm⟩

/-- Direct form of `valueReplay_eq_derived` for callers that already have
the three physical serialization facts. Keeping the dependent record
construction in this key-independent module avoids unfolding a concrete key
at the call site. -/
theorem valueReplay_eq_derived_of_serializations
    (input : Input) (statement : PaperStatement)
    (certificate : PaperCertificate) (assignment : Nat -> Nat)
    (one : assignment 0 = 1)
    (publicState : valueAbsorbPublic assignment input = statement.priorState)
    (statementSerialization :
      fieldValues assignment (statementFields input) =
        ProductPoseidon2.statementFields statement)
    (roundSerialization : forall round,
      fieldValues assignment (roundFields round.val (input.rounds round)) =
        ProductPoseidon2.roundFields round (certificate.rounds round)) :
    let derived := FiatShamir.derive ProductPoseidon2.transcript statement
      certificate
    (valueDeriveAlpha assignment input).1.map
          KConcreteFixedPhaseBridge.ofProjection = derived.alpha.coordinates /\
      KConcreteFixedPhaseBridge.ofProjection
          (valueDeriveGamma assignment input).1 = derived.gamma /\
      (valueReplayRounds assignment input).challenges.map
          KConcreteFixedPhaseBridge.ofProjection =
        derived.roundPoint.coordinates /\
      (valueReplayRounds assignment input).state = derived.finalState := by
  exact valueReplay_eq_derived input statement certificate assignment one
    { publicState := publicState
      statementSerialization := statementSerialization
      roundSerialization := roundSerialization }

/-- Component form of the typed replay. The concrete caller supplies only
the verifier-owned prior state and verifier input, plus the exact prover round
messages. No concrete dependent record must be reduced at the call site. -/
theorem valueReplay_eq_derived_of_components
    (input : Input) (priorState : ProductPoseidon2.State)
    (verifierInput : ProtocolPolynomial.VerifierInput K ProductNifsCodec.shape)
    (certificateRounds : Fin ProductNifsCodec.shape.cubeVariables ->
      SumCheck.Finite.Message K)
    (assignment : Nat -> Nat) (one : assignment 0 = 1)
    (publicState : valueAbsorbPublic assignment input = priorState)
    (statementSerialization :
      fieldValues assignment (statementFields input) =
        ProductPoseidon2.statementFields
          ({ priorState := priorState
             input := verifierInput } : PaperStatement))
    (roundSerialization : forall round,
      fieldValues assignment (roundFields round.val (input.rounds round)) =
        ProductPoseidon2.roundFields round (certificateRounds round)) :
    let statement : PaperStatement :=
      { priorState := priorState, input := verifierInput }
    let certificate : PaperCertificate := { rounds := certificateRounds }
    let derived := FiatShamir.derive ProductPoseidon2.transcript statement
      certificate
    (valueDeriveAlpha assignment input).1.map
          KConcreteFixedPhaseBridge.ofProjection = derived.alpha.coordinates /\
      KConcreteFixedPhaseBridge.ofProjection
          (valueDeriveGamma assignment input).1 = derived.gamma /\
      (valueReplayRounds assignment input).challenges.map
          KConcreteFixedPhaseBridge.ofProjection =
        derived.roundPoint.coordinates /\
      (valueReplayRounds assignment input).state = derived.finalState := by
  exact valueReplay_eq_derived_of_serializations input
    { priorState := priorState, input := verifierInput }
    { rounds := certificateRounds } assignment one publicState
    statementSerialization roundSerialization

end Nightstream.Implementation.NebulaV2.ProductPiCcsTypedReplay
