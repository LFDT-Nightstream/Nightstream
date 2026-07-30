import Nightstream.Implementation.R1CS.Canonical.KPiCcsPaperFiatShamir

/-!
Contract: bind the deterministic PiCCS row reduction and its named events to
the exact Fiat--Shamir coins derived by the unchanged paper schedule.

`KPiCcsOccurrence` already derives table truth or the paper's named
`MixingRoot`, `SumCheckCollision`, and `OutputMismatch` events from arithmetic
row satisfaction.  `KPiCcsPaperFiatShamir` independently proves that the
challenge expressions in those rows equal the verifier-derived Poseidon2
schedule.  This module composes the two facts after translating the row
carrier to the paper's concrete quadratic-extension carrier.

No probability bound, source-binding premise discharge, or cryptographic
assumption is introduced here.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.Canonical.KPiCcsEventBinding

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
open Nightstream.Implementation.R1CS.Canonical.KPiCcsTranscriptSemantics

abbrev ConcreteK := Nightstream.SuperNeo.Concrete.K
abbrev ValueK := KPiCcsTranscriptSemantics.ValueK
abbrev Constants := Poseidon2Schedule.Constants

/-- Coordinate-preserving translation from the row transcript carrier to the
paper semantic carrier. -/
def concretePoint {variables : Nat}
    (point : CubePoint ValueK variables) :
    CubePoint ConcreteK variables where
  coordinates := point.coordinates.map ofProjection
  dimension := by
    rw [List.length_map, point.dimension]

private theorem cubePoint_eq_of_coordinates_eq
    {Field : Type} {variables : Nat}
    {left right : CubePoint Field variables}
    (equal : left.coordinates = right.coordinates) :
    left = right := by
  cases left with
  | mk leftCoordinates leftDimension =>
      cases right with
      | mk rightCoordinates rightDimension =>
          simp only at equal
          subst rightCoordinates
          rfl

theorem alphaAt_enumerates_replay
    {shape : Shape} (execution : KPiCcsTranscript.Replay shape) :
    (KPointEquality.indices shape.cubeVariables).map
        (KPiCcsTranscript.alphaAt execution) =
      execution.alpha := by
  apply List.ext_getElem
  · rw [List.length_map, KPointEquality.indices_length,
      execution.alpha_length]
  · intro index leftBound rightBound
    simp [KPointEquality.indices, KPiCcsTranscript.alphaAt]

theorem pointAt_enumerates_replay
    {shape : Shape} (execution : KPiCcsTranscript.Replay shape) :
    (KPointEquality.indices shape.cubeVariables).map
        (KPiCcsTranscript.pointAt execution) =
      execution.point := by
  apply List.ext_getElem
  · rw [List.length_map, KPointEquality.indices_length,
      execution.point_length]
  · intro index leftBound rightBound
    simp [KPointEquality.indices, KPiCcsTranscript.pointAt]

theorem decodedAlpha_coordinates
    {shape : Shape} {degree : Nat}
    (input : KPiCcsTranscript.Input shape degree)
    (assignment : Nat → Nat) :
    (KPiCcsOccurrence.decodedAlpha
        (KPiCcsTranscript.occurrenceInput input) assignment).coordinates =
      (KPiCcsTranscript.replay input).alpha.map
        (fun value =>
          ofProjection
            (KPiCcsTranscriptSemantics.decoded assignment value)) := by
  unfold KPiCcsOccurrence.decodedAlpha
    KPiCcsTerminal.decodedAlpha
    KPiCcsTerminal.alphaEqualityInput
    KPiCcsOccurrence.terminalInput
    KPiCcsTranscript.occurrenceInput
    KPointEquality.decodedRight
    KPointEquality.decoded
  change
    (KPointEquality.indices shape.cubeVariables).map
        (fun index =>
          ofProjection
            (KPiCcsTranscriptSemantics.decoded assignment
              (KPiCcsTranscript.alphaAt
                (KPiCcsTranscript.replay input) index))) =
      _
  have enumerated :=
    alphaAt_enumerates_replay (KPiCcsTranscript.replay input)
  simpa only [List.map_map, Function.comp_apply] using
    congrArg
      (List.map fun value =>
        ofProjection (KPiCcsTranscriptSemantics.decoded assignment value))
      enumerated

theorem decodedPoint_coordinates
    {shape : Shape} {degree : Nat}
    (input : KPiCcsTranscript.Input shape degree)
    (assignment : Nat → Nat) :
    (KPiCcsOccurrence.decodedPoint
        (KPiCcsTranscript.occurrenceInput input) assignment).coordinates =
      (KPiCcsTranscript.replay input).point.map
        (fun value =>
          ofProjection
            (KPiCcsTranscriptSemantics.decoded assignment value)) := by
  unfold KPiCcsOccurrence.decodedPoint
    KPiCcsTerminal.decodedPoint
    KPiCcsTerminal.alphaEqualityInput
    KPiCcsOccurrence.terminalInput
    KPiCcsTranscript.occurrenceInput
    KPointEquality.decodedLeft
    KPointEquality.decoded
  change
    (KPointEquality.indices shape.cubeVariables).map
        (fun index =>
          ofProjection
            (KPiCcsTranscriptSemantics.decoded assignment
              (KPiCcsTranscript.pointAt
                (KPiCcsTranscript.replay input) index))) =
      _
  have enumerated :=
    pointAt_enumerates_replay (KPiCcsTranscript.replay input)
  simpa only [List.map_map, Function.comp_apply] using
    congrArg
      (List.map fun value =>
        ofProjection (KPiCcsTranscriptSemantics.decoded assignment value))
      enumerated

/-- The unchanged paper verifier derivation for this exact row occurrence. -/
def paperDerivation
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (input : KPiCcsTranscript.Input shape degree) :
    FiatShamir.DerivedCoins ValueK
      KPiCcsTranscriptSemantics.ValueState shape :=
  FiatShamir.derive
    (KPiCcsPaperFiatShamir.oracle constants assignment)
    input (KPiCcsPaperFiatShamir.certificate assignment input)

/-- The verifier-derived alpha point, represented in the paper semantic
carrier used by PiCCS. -/
def paperAlpha
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (input : KPiCcsTranscript.Input shape degree) :
    CubePoint ConcreteK shape.cubeVariables :=
  concretePoint (paperDerivation constants assignment input).alpha

/-- The verifier-derived mixing challenge in the paper semantic carrier. -/
def paperGamma
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (input : KPiCcsTranscript.Input shape degree) : ConcreteK :=
  ofProjection (paperDerivation constants assignment input).gamma

/-- The verifier-derived fixed-phase SumCheck point in the paper semantic
carrier. -/
def paperPoint
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (input : KPiCcsTranscript.Input shape degree) :
    CubePoint ConcreteK shape.cubeVariables :=
  concretePoint (paperDerivation constants assignment input).roundPoint

theorem decodedAlpha_eq_paper
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (input : KPiCcsTranscript.Input shape degree)
    (residues : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies (KPiCcsTranscript.rows constants input) assignment) :
    KPiCcsOccurrence.decodedAlpha
        (KPiCcsTranscript.occurrenceInput input) assignment =
      paperAlpha constants assignment input := by
  apply cubePoint_eq_of_coordinates_eq
  rw [decodedAlpha_coordinates]
  have schedule :=
    KPiCcsPaperFiatShamir.rows_derive_paper_schedule
      constants assignment input residues constantWire satisfied
  have mapped := congrArg (List.map ofProjection) schedule.1
  simpa only [paperAlpha, concretePoint, paperDerivation, List.map_map,
    Function.comp_apply] using mapped

theorem decodedGamma_eq_paper
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (input : KPiCcsTranscript.Input shape degree)
    (residues : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies (KPiCcsTranscript.rows constants input) assignment) :
    KPiCcsOccurrence.decodedGamma
        (KPiCcsTranscript.occurrenceInput input) assignment =
      paperGamma constants assignment input := by
  have schedule :=
    KPiCcsPaperFiatShamir.rows_derive_paper_schedule
      constants assignment input residues constantWire satisfied
  have mapped := congrArg ofProjection schedule.2.1
  simpa [KPiCcsOccurrence.decodedGamma,
    KPiCcsTerminal.decoded, KPointEquality.decoded,
    KPiCcsOccurrence.terminalInput, KPiCcsTranscript.occurrenceInput,
    paperGamma, paperDerivation] using mapped

theorem decodedPoint_eq_paper
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (input : KPiCcsTranscript.Input shape degree)
    (residues : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies (KPiCcsTranscript.rows constants input) assignment) :
    KPiCcsOccurrence.decodedPoint
        (KPiCcsTranscript.occurrenceInput input) assignment =
      paperPoint constants assignment input := by
  apply cubePoint_eq_of_coordinates_eq
  rw [decodedPoint_coordinates]
  have schedule :=
    KPiCcsPaperFiatShamir.rows_derive_paper_schedule
      constants assignment input residues constantWire satisfied
  have mapped := congrArg (List.map ofProjection) schedule.2.2.1
  simpa only [paperPoint, concretePoint, paperDerivation, List.map_map,
    Function.comp_apply] using mapped

/-- **Transcript-bound deterministic PiCCS reduction.**

Satisfying the complete Poseidon2-transcript and arithmetic row program yields
the unchanged paper table relation or one of its exact named algebraic events,
all instantiated at the verifier-derived Fiat--Shamir coins.  No decoded or
caller-provided challenge remains in the conclusion. -/
theorem rows_imply_tableTruth_or_paperBadEvent
    {shape : Shape} {degree : Nat}
    (constants : Constants) (assignment : Nat → Nat)
    (input : KPiCcsTranscript.Input shape degree)
    (residues : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies (KPiCcsTranscript.rows constants input) assignment)
    (data : ProtocolPolynomial.Data ConcreteK shape)
    (sourceBinding :
      data.toVerifierInput =
        KPiCcsOccurrence.decodedVerifierInput
          (KPiCcsTranscript.occurrenceInput input) assignment)
    (degreeCovers :
      data.toVerifierInput.sumcheckDegreeBound ≤ degree)
    (challengeSetSize : Nat) :
    (TableResidualData.toTableObligations ConcreteCarrier.extensionOps
        (SignedCoefficientObject.toTableResidualData
          ConcreteCarrier.extensionOps
          (data.toJointData ConcreteCarrier.extensionOps))).AllHold ∨
      SignedCoefficientObject.MixingRoot ConcreteCarrier.extensionOps
        (data.toJointData ConcreteCarrier.extensionOps)
        (paperAlpha constants assignment input)
        (paperGamma constants assignment input) ∨
      ProtocolPolynomial.FixedWidth.SumCheckCollision
        ConcreteCarrier.extensionOps data
        (paperAlpha constants assignment input)
        (paperGamma constants assignment input)
        degree challengeSetSize
        (paperPoint constants assignment input)
        (KPiCcsOccurrence.decodedCertificate
          (KPiCcsTranscript.occurrenceInput input) assignment) ∨
      ProtocolPolynomial.OutputMismatch ConcreteCarrier.extensionOps data
        (paperAlpha constants assignment input)
        (paperGamma constants assignment input)
        (paperPoint constants assignment input)
        (KPiCcsOccurrence.decodedMessage
          (KPiCcsTranscript.occurrenceInput input) assignment) := by
  have arithmetic :=
    KPiCcsOccurrence.rows_imply_tableTruth_or_badEvent
      (KPiCcsTranscript.occurrenceInput input) assignment constantWire
      (KPiCcsTranscript.occurrence_satisfied
        constants input assignment satisfied)
      data sourceBinding degreeCovers challengeSetSize
  have alphaEq :=
    decodedAlpha_eq_paper constants assignment input residues
      constantWire satisfied
  have gammaEq :=
    decodedGamma_eq_paper constants assignment input residues
      constantWire satisfied
  have pointEq :=
    decodedPoint_eq_paper constants assignment input residues
      constantWire satisfied
  rw [alphaEq, gammaEq, pointEq] at arithmetic
  exact arithmetic

end Nightstream.Implementation.R1CS.Canonical.KPiCcsEventBinding
