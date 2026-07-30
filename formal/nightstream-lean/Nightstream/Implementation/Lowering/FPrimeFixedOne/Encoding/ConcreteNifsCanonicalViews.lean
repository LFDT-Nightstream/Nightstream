import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofCodec
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile

/-!
Contract: exact codec-coordinate views for the Lean-owned selected NIFS
running, fresh, and proof codecs.

Owns: the index arithmetic induced by the canonical codec composition.  Every
view is derived from the codec's left-to-right product and increasing `Fin`
orders.  No index is copied from Rust or from an artifact.

Does not own: verifier acceptance, transcript replay, relation matrices,
application data, physical rows, Rust, or artifacts.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalViews

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCodecCore
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRunningCodec
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofCodec
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCarrierViews
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

private abbrev Domains :=
  Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production

noncomputable def fieldView :
    FView fieldCodec id where
  index := ⟨0, by decide⟩
  encodeValue := by
    intro value
    rfl

noncomputable def kView :
    KView kCodec id := by
  let c0 :
      FView kCodec (fun value : Concrete.K => value.c0) :=
    FView.throughPullback kPair kPair_injective
      (FView.productLeft (right := fieldCodec) fieldView)
  let c1 :
      FView kCodec (fun value : Concrete.K => value.c1) :=
    FView.throughPullback kPair kPair_injective
      (FView.productRight (left := fieldCodec) fieldView)
  exact {
    c0Index := c0.index
    c1Index := c1.index
    encodeC0 := c0.encodeValue
    encodeC1 := c1.encodeValue
  }

noncomputable def ringFView
    (lane : Fin ringDegree) :
    FView ringFCodec (fun value => value lane) :=
  FView.finElement ringDegree lane fieldView

noncomputable def ringKView
    (lane : Fin ringDegree) :
    KView ringKCodec (fun value => value lane) :=
  KView.finElement ringDegree lane kView

noncomputable def commitmentView
    (verifierRows : Nat)
    (row : Fin verifierRows)
    (lane : Fin ringDegree) :
    FView (commitmentCodec verifierRows)
      (fun commitment => commitment row lane) :=
  FView.finElement verifierRows row (ringFView lane)

noncomputable def publicInputView
    (publicWidth : Nat)
    (column : Fin publicWidth) :
    FView (publicInputCodec publicWidth)
      (fun input => input column) :=
  FView.finElement publicWidth column fieldView

noncomputable def pointView
    (variables : Nat)
    (coordinate : Fin variables) :
    KView (pointCodec variables)
      (ConcreteNifsCarrierViews.pointCoordinate coordinate) := by
  let listView :
      KView (Codec.fixedList variables K.zero kCodec)
        (fun coordinates => coordinates.getD coordinate.val K.zero) :=
    KView.fixedListElement variables K.zero coordinate kView
  let pulled :
      KView (pointCodec variables)
        (fun point => point.coordinates.getD coordinate.val K.zero) :=
    KView.throughPullback pointData pointData_injective listView
  exact pulled.congrValue (by
    intro point
    unfold ConcreteNifsCarrierViews.pointCoordinate
    rw [List.getD_eq_getElem?_getD,
      List.getElem?_eq_getElem (by
        rw [point.dimension]
        exact coordinate.isLt),
      Option.getD_some]
    simp only [List.get_eq_getElem])

noncomputable def evaluationView
    (matrixCount : Nat)
    (matrix : Fin matrixCount)
    (lane : Fin ringDegree) :
    KView (evaluationsCodec matrixCount)
      (fun evaluations => evaluations.getD matrix.val ringKZero lane) :=
  KView.fixedArrayElement matrixCount ringKZero matrix (ringKView lane)

/-! ## Proof-coordinate product -/

private abbrev ClaimedYRingData (shape : SemanticShape) :=
  Fin shape.runningCount → Fin shape.matrixCount → RingK

private abbrev FeRowData
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount) :=
  Fin shape.rowVariables →
    Fin (rowSlotCount constraintPolynomial) → Concrete.K

private abbrev FeLaneData :=
  Fin Domains.fe.laneVariables → Fin 3 → Concrete.K

private abbrev NcData :=
  Fin (Domains.nc.blockVariables + Domains.nc.laneVariables) →
    Fin 5 → Concrete.K

private abbrev OutputYRingData (shape : SemanticShape) :=
  Fin shape.sourceCount → Fin shape.matrixCount → RingK

private abbrev OutputYZcolData (shape : SemanticShape) :=
  Fin shape.sourceCount → RingK

private abbrev PiRlcChallengeData :=
  Fin FixedActive.arity.total → RingF

private abbrev PiDecPayloadData
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :=
  Fin productionGlobalParams.k →
    PiDecChildPayload
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows)

private noncomputable def proofTailI
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    Codec
      (PiRlcChallengeData ×
        PiDecPayloadData
          shape publicRingColumns verifierRows publicFits) :=
  Codec.product
    (Codec.finFunction FixedActive.arity.total ringFCodec)
    (Codec.finFunction productionGlobalParams.k
      (payloadCodec shape publicRingColumns verifierRows publicFits))

private noncomputable def proofTailH
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    Codec
      (OutputYZcolData shape ×
        (PiRlcChallengeData ×
          PiDecPayloadData
            shape publicRingColumns verifierRows publicFits)) :=
  Codec.product
    (Codec.finFunction shape.sourceCount ringKCodec)
    (proofTailI shape publicRingColumns verifierRows publicFits)

private noncomputable def proofTailG
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    Codec
      (OutputYRingData shape ×
        (OutputYZcolData shape ×
          (PiRlcChallengeData ×
            PiDecPayloadData
              shape publicRingColumns verifierRows publicFits))) :=
  Codec.product
    (Codec.finFunction shape.sourceCount
      (Codec.finFunction shape.matrixCount ringKCodec))
    (proofTailH shape publicRingColumns verifierRows publicFits)

private noncomputable def proofTailF
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    Codec
      (NcData ×
        (OutputYRingData shape ×
          (OutputYZcolData shape ×
            (PiRlcChallengeData ×
              PiDecPayloadData
                shape publicRingColumns verifierRows publicFits)))) :=
  Codec.product
    (Codec.finFunction
      (Domains.nc.blockVariables + Domains.nc.laneVariables)
      (Codec.finFunction 5 kCodec))
    (proofTailG shape publicRingColumns verifierRows publicFits)

private noncomputable def proofTailE
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    Codec
      (FeLaneData ×
        (NcData ×
          (OutputYRingData shape ×
            (OutputYZcolData shape ×
              (PiRlcChallengeData ×
                PiDecPayloadData
                  shape publicRingColumns verifierRows publicFits))))) :=
  Codec.product
    (Codec.finFunction Domains.fe.laneVariables
      (Codec.finFunction 3 kCodec))
    (proofTailF shape publicRingColumns verifierRows publicFits)

private noncomputable def proofTailD
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    Codec
      (FeRowData shape constraintPolynomial ×
        (FeLaneData ×
          (NcData ×
            (OutputYRingData shape ×
              (OutputYZcolData shape ×
                (PiRlcChallengeData ×
                  PiDecPayloadData
                    shape publicRingColumns verifierRows publicFits)))))) :=
  Codec.product
    (Codec.finFunction shape.rowVariables
      (Codec.finFunction
        (rowSlotCount constraintPolynomial) kCodec))
    (proofTailE shape publicRingColumns verifierRows publicFits)

private noncomputable def proofTailC
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    Codec
      ((Fin 8 → Nat) ×
        (FeRowData shape constraintPolynomial ×
          (FeLaneData ×
            (NcData ×
              (OutputYRingData shape ×
                (OutputYZcolData shape ×
                  (PiRlcChallengeData ×
                    PiDecPayloadData
                      shape publicRingColumns verifierRows publicFits))))))) :=
  Codec.product
    (Codec.finFunction 8 priorLaneCodec)
    (proofTailD shape constraintPolynomial
      publicRingColumns verifierRows publicFits)

private noncomputable def proofTailB
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    Codec
      (ClaimedYRingData shape ×
        ((Fin 8 → Nat) ×
          (FeRowData shape constraintPolynomial ×
            (FeLaneData ×
              (NcData ×
                (OutputYRingData shape ×
                  (OutputYZcolData shape ×
                    (PiRlcChallengeData ×
                      PiDecPayloadData
                        shape publicRingColumns verifierRows
                          publicFits)))))))) :=
  Codec.product
    (Codec.finFunction shape.runningCount
      (Codec.finFunction shape.matrixCount ringKCodec))
    (proofTailC shape constraintPolynomial
      publicRingColumns verifierRows publicFits)

private noncomputable def proofRawCodec
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :=
  Codec.product
    (pointCodec shape.rowVariables)
    (proofTailB shape constraintPolynomial
      publicRingColumns verifierRows publicFits)

private noncomputable def priorLaneBaseView :
    FView priorLaneCodec NumericRowBridge.residue :=
  FView.throughPullbackOn NumericRowBridge.residue
    (fun _ _ => True.intro)
    (fun leftLt rightLt equal =>
      NumericRowBridge.residue_injective_of_lt leftLt rightLt equal)
    fieldView

private noncomputable def liftProofF
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    {value :
      Coordinates shape constraintPolynomial
        publicRingColumns verifierRows publicFits → Field}
    (view :
      FView
        (coordinatesCodec shape constraintPolynomial
          publicRingColumns verifierRows publicFits)
        value) :
    FView
      (proofCodec shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits)
      (fun proof =>
        value (proofCoordinates constraintPolynomial proof)) :=
  FView.throughPullbackOn
    (proofCoordinates constraintPolynomial)
    (fun _ admissible => proofCoordinates_admissible admissible)
    (fun leftAdmissible rightAdmissible equal =>
      proofCoordinates_injective leftAdmissible rightAdmissible equal)
    view

private noncomputable def liftProofK
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    {value :
      Coordinates shape constraintPolynomial
        publicRingColumns verifierRows publicFits → Concrete.K}
    (view :
      KView
        (coordinatesCodec shape constraintPolynomial
          publicRingColumns verifierRows publicFits)
        value) :
    KView
      (proofCodec shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits)
      (fun proof =>
        value (proofCoordinates constraintPolynomial proof)) :=
  KView.throughPullbackOn
    (proofCoordinates constraintPolynomial)
    (fun _ admissible => proofCoordinates_admissible admissible)
    (fun leftAdmissible rightAdmissible equal =>
      proofCoordinates_injective leftAdmissible rightAdmissible equal)
    view

private noncomputable def coordinatesAView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    {value : CubePoint Concrete.K shape.rowVariables → Concrete.K}
    (view : KView (pointCodec shape.rowVariables) value) :
    KView
      (coordinatesCodec shape constraintPolynomial
        publicRingColumns verifierRows publicFits)
      (fun coordinates => value coordinates.priorPoint) :=
  KView.throughPullback coordinateData coordinateData_injective
    (KView.productLeft
      (right := proofTailB shape constraintPolynomial
        publicRingColumns verifierRows publicFits)
      view)

private noncomputable def coordinatesBView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    {value : ClaimedYRingData shape → Concrete.K}
    (view :
      KView
        (Codec.finFunction shape.runningCount
          (Codec.finFunction shape.matrixCount ringKCodec))
        value) :
    KView
      (coordinatesCodec shape constraintPolynomial
        publicRingColumns verifierRows publicFits)
      (fun coordinates => value coordinates.claimedYRing) :=
  KView.throughPullback coordinateData coordinateData_injective
    (KView.productRight (left := pointCodec shape.rowVariables)
      (KView.productLeft
        (right := proofTailC shape constraintPolynomial
          publicRingColumns verifierRows publicFits)
        view))

private noncomputable def coordinatesCView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    {value : (Fin 8 → Nat) → Field}
    (view : FView (Codec.finFunction 8 priorLaneCodec) value) :
    FView
      (coordinatesCodec shape constraintPolynomial
        publicRingColumns verifierRows publicFits)
      (fun coordinates => value coordinates.priorLanes) :=
  FView.throughPullback coordinateData coordinateData_injective
    (FView.productRight (left := pointCodec shape.rowVariables)
      (FView.productRight
        (left :=
          Codec.finFunction shape.runningCount
            (Codec.finFunction shape.matrixCount ringKCodec))
        (FView.productLeft
          (right := proofTailD shape constraintPolynomial
            publicRingColumns verifierRows publicFits)
          view)))

private noncomputable def coordinatesDView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    {value : FeRowData shape constraintPolynomial → Concrete.K}
    (view :
      KView
        (Codec.finFunction shape.rowVariables
          (Codec.finFunction
            (rowSlotCount constraintPolynomial) kCodec))
        value) :
    KView
      (coordinatesCodec shape constraintPolynomial
        publicRingColumns verifierRows publicFits)
      (fun coordinates => value coordinates.feRow) :=
  KView.throughPullback coordinateData coordinateData_injective
    (KView.productRight (left := pointCodec shape.rowVariables)
      (KView.productRight
        (left :=
          Codec.finFunction shape.runningCount
            (Codec.finFunction shape.matrixCount ringKCodec))
        (KView.productRight
          (left := Codec.finFunction 8 priorLaneCodec)
          (KView.productLeft
            (right := proofTailE shape
              publicRingColumns verifierRows publicFits)
            view))))

private noncomputable def coordinatesEView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    {value : FeLaneData → Concrete.K}
    (view :
      KView
        (Codec.finFunction Domains.fe.laneVariables
          (Codec.finFunction 3 kCodec))
        value) :
    KView
      (coordinatesCodec shape constraintPolynomial
        publicRingColumns verifierRows publicFits)
      (fun coordinates => value coordinates.feLane) :=
  KView.throughPullback coordinateData coordinateData_injective
    (KView.productRight (left := pointCodec shape.rowVariables)
      (KView.productRight
        (left :=
          Codec.finFunction shape.runningCount
            (Codec.finFunction shape.matrixCount ringKCodec))
        (KView.productRight
          (left := Codec.finFunction 8 priorLaneCodec)
          (KView.productRight
            (left :=
              Codec.finFunction shape.rowVariables
                (Codec.finFunction
                  (rowSlotCount constraintPolynomial) kCodec))
            (KView.productLeft
              (right := proofTailF shape
                publicRingColumns verifierRows publicFits)
              view)))))

private noncomputable def coordinatesFView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    {value : NcData → Concrete.K}
    (view :
      KView
        (Codec.finFunction
          (Domains.nc.blockVariables + Domains.nc.laneVariables)
          (Codec.finFunction 5 kCodec))
        value) :
    KView
      (coordinatesCodec shape constraintPolynomial
        publicRingColumns verifierRows publicFits)
      (fun coordinates => value coordinates.nc) :=
  KView.throughPullback coordinateData coordinateData_injective
    (KView.productRight (left := pointCodec shape.rowVariables)
      (KView.productRight
        (left :=
          Codec.finFunction shape.runningCount
            (Codec.finFunction shape.matrixCount ringKCodec))
        (KView.productRight
          (left := Codec.finFunction 8 priorLaneCodec)
          (KView.productRight
            (left :=
              Codec.finFunction shape.rowVariables
                (Codec.finFunction
                  (rowSlotCount constraintPolynomial) kCodec))
            (KView.productRight
              (left :=
                Codec.finFunction Domains.fe.laneVariables
                  (Codec.finFunction 3 kCodec))
              (KView.productLeft
                (right := proofTailG shape
                  publicRingColumns verifierRows publicFits)
                view))))))

private noncomputable def coordinatesGView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    {value : OutputYRingData shape → Concrete.K}
    (view :
      KView
        (Codec.finFunction shape.sourceCount
          (Codec.finFunction shape.matrixCount ringKCodec))
        value) :
    KView
      (coordinatesCodec shape constraintPolynomial
        publicRingColumns verifierRows publicFits)
      (fun coordinates => value coordinates.outputYRing) :=
  KView.throughPullback coordinateData coordinateData_injective
    (KView.productRight (left := pointCodec shape.rowVariables)
      (KView.productRight
        (left :=
          Codec.finFunction shape.runningCount
            (Codec.finFunction shape.matrixCount ringKCodec))
        (KView.productRight
          (left := Codec.finFunction 8 priorLaneCodec)
          (KView.productRight
            (left :=
              Codec.finFunction shape.rowVariables
                (Codec.finFunction
                  (rowSlotCount constraintPolynomial) kCodec))
            (KView.productRight
              (left :=
                Codec.finFunction Domains.fe.laneVariables
                  (Codec.finFunction 3 kCodec))
              (KView.productRight
                (left :=
                  Codec.finFunction
                    (Domains.nc.blockVariables + Domains.nc.laneVariables)
                    (Codec.finFunction 5 kCodec))
                (KView.productLeft
                  (right := proofTailH shape
                    publicRingColumns verifierRows publicFits)
                  view)))))))

private noncomputable def coordinatesHView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    {value : OutputYZcolData shape → Concrete.K}
    (view :
      KView (Codec.finFunction shape.sourceCount ringKCodec) value) :
    KView
      (coordinatesCodec shape constraintPolynomial
        publicRingColumns verifierRows publicFits)
      (fun coordinates => value coordinates.outputYZcol) :=
  KView.throughPullback coordinateData coordinateData_injective
    (KView.productRight (left := pointCodec shape.rowVariables)
      (KView.productRight
        (left :=
          Codec.finFunction shape.runningCount
            (Codec.finFunction shape.matrixCount ringKCodec))
        (KView.productRight
          (left := Codec.finFunction 8 priorLaneCodec)
          (KView.productRight
            (left :=
              Codec.finFunction shape.rowVariables
                (Codec.finFunction
                  (rowSlotCount constraintPolynomial) kCodec))
            (KView.productRight
              (left :=
                Codec.finFunction Domains.fe.laneVariables
                  (Codec.finFunction 3 kCodec))
              (KView.productRight
                (left :=
                  Codec.finFunction
                    (Domains.nc.blockVariables + Domains.nc.laneVariables)
                    (Codec.finFunction 5 kCodec))
                (KView.productRight
                  (left :=
                    Codec.finFunction shape.sourceCount
                      (Codec.finFunction shape.matrixCount ringKCodec))
                  (KView.productLeft
                    (right := proofTailI shape
                      publicRingColumns verifierRows publicFits)
                    view))))))))

private noncomputable def coordinatesIView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    {value : PiRlcChallengeData → Field}
    (view :
      FView
        (Codec.finFunction FixedActive.arity.total ringFCodec)
        value) :
    FView
      (coordinatesCodec shape constraintPolynomial
        publicRingColumns verifierRows publicFits)
      (fun coordinates => value coordinates.piRlcChallenges) :=
  FView.throughPullback coordinateData coordinateData_injective
    (FView.productRight (left := pointCodec shape.rowVariables)
      (FView.productRight
        (left :=
          Codec.finFunction shape.runningCount
            (Codec.finFunction shape.matrixCount ringKCodec))
        (FView.productRight
          (left := Codec.finFunction 8 priorLaneCodec)
          (FView.productRight
            (left :=
              Codec.finFunction shape.rowVariables
                (Codec.finFunction
                  (rowSlotCount constraintPolynomial) kCodec))
            (FView.productRight
              (left :=
                Codec.finFunction Domains.fe.laneVariables
                  (Codec.finFunction 3 kCodec))
              (FView.productRight
                (left :=
                  Codec.finFunction
                    (Domains.nc.blockVariables + Domains.nc.laneVariables)
                    (Codec.finFunction 5 kCodec))
                (FView.productRight
                  (left :=
                    Codec.finFunction shape.sourceCount
                      (Codec.finFunction shape.matrixCount ringKCodec))
                  (FView.productRight
                    (left :=
                      Codec.finFunction shape.sourceCount ringKCodec)
                    (FView.productLeft
                      (right :=
                        Codec.finFunction productionGlobalParams.k
                          (payloadCodec shape publicRingColumns verifierRows
                            publicFits))
                      view)))))))))

private noncomputable def coordinatesJFView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    {value :
      PiDecPayloadData
        shape publicRingColumns verifierRows publicFits → Field}
    (view :
      FView
        (Codec.finFunction productionGlobalParams.k
          (payloadCodec shape publicRingColumns verifierRows publicFits))
        value) :
    FView
      (coordinatesCodec shape constraintPolynomial
        publicRingColumns verifierRows publicFits)
      (fun coordinates => value coordinates.piDecPayloads) :=
  FView.throughPullback coordinateData coordinateData_injective
    (FView.productRight (left := pointCodec shape.rowVariables)
      (FView.productRight
        (left :=
          Codec.finFunction shape.runningCount
            (Codec.finFunction shape.matrixCount ringKCodec))
        (FView.productRight
          (left := Codec.finFunction 8 priorLaneCodec)
          (FView.productRight
            (left :=
              Codec.finFunction shape.rowVariables
                (Codec.finFunction
                  (rowSlotCount constraintPolynomial) kCodec))
            (FView.productRight
              (left :=
                Codec.finFunction Domains.fe.laneVariables
                  (Codec.finFunction 3 kCodec))
              (FView.productRight
                (left :=
                  Codec.finFunction
                    (Domains.nc.blockVariables + Domains.nc.laneVariables)
                    (Codec.finFunction 5 kCodec))
                (FView.productRight
                  (left :=
                    Codec.finFunction shape.sourceCount
                      (Codec.finFunction shape.matrixCount ringKCodec))
                  (FView.productRight
                    (left :=
                      Codec.finFunction shape.sourceCount ringKCodec)
                    (FView.productRight
                      (left :=
                        Codec.finFunction FixedActive.arity.total ringFCodec)
                      view)))))))))

private noncomputable def coordinatesJKView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    {value :
      PiDecPayloadData
        shape publicRingColumns verifierRows publicFits → Concrete.K}
    (view :
      KView
        (Codec.finFunction productionGlobalParams.k
          (payloadCodec shape publicRingColumns verifierRows publicFits))
        value) :
    KView
      (coordinatesCodec shape constraintPolynomial
        publicRingColumns verifierRows publicFits)
      (fun coordinates => value coordinates.piDecPayloads) :=
  KView.throughPullback coordinateData coordinateData_injective
    (KView.productRight (left := pointCodec shape.rowVariables)
      (KView.productRight
        (left :=
          Codec.finFunction shape.runningCount
            (Codec.finFunction shape.matrixCount ringKCodec))
        (KView.productRight
          (left := Codec.finFunction 8 priorLaneCodec)
          (KView.productRight
            (left :=
              Codec.finFunction shape.rowVariables
                (Codec.finFunction
                  (rowSlotCount constraintPolynomial) kCodec))
            (KView.productRight
              (left :=
                Codec.finFunction Domains.fe.laneVariables
                  (Codec.finFunction 3 kCodec))
              (KView.productRight
                (left :=
                  Codec.finFunction
                    (Domains.nc.blockVariables + Domains.nc.laneVariables)
                    (Codec.finFunction 5 kCodec))
                (KView.productRight
                  (left :=
                    Codec.finFunction shape.sourceCount
                      (Codec.finFunction shape.matrixCount ringKCodec))
                  (KView.productRight
                    (left :=
                      Codec.finFunction shape.sourceCount ringKCodec)
                    (KView.productRight
                      (left :=
                        Codec.finFunction FixedActive.arity.total ringFCodec)
                      view)))))))))

/-! ## Selected-proof views -/

noncomputable def coordinatesPriorLaneView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (lane : Fin 8) :
    FView
      (coordinatesCodec shape constraintPolynomial
        publicRingColumns verifierRows publicFits)
      (fun coordinates =>
        NumericRowBridge.residue (coordinates.priorLanes lane)) :=
  coordinatesCView shape constraintPolynomial
    publicRingColumns verifierRows publicFits
    (FView.finElement 8 lane priorLaneBaseView)

noncomputable def proofPriorLaneView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (lane : Fin 8) :
    FView
      (proofCodec shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits)
      (fun proof => NumericRowBridge.residue (proof.priorState.lanes lane)) :=
  liftProofF shape constraintPolynomial priorAbsorbed
    publicRingColumns verifierRows publicFits
    (coordinatesPriorLaneView shape constraintPolynomial
      publicRingColumns verifierRows publicFits lane)

noncomputable def proofPriorPointView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (coordinate : Fin shape.rowVariables) :
    KView
      (proofCodec shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits)
      (priorPointCoordinate coordinate) :=
  liftProofK shape constraintPolynomial priorAbsorbed
    publicRingColumns verifierRows publicFits
    (coordinatesAView shape constraintPolynomial
      publicRingColumns verifierRows publicFits
      (pointView shape.rowVariables coordinate))

noncomputable def proofClaimedYRingView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (running : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    KView
      (proofCodec shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits)
      (claimedYRingCoordinate running matrix lane) :=
  liftProofK shape constraintPolynomial priorAbsorbed
    publicRingColumns verifierRows publicFits
    (coordinatesBView shape constraintPolynomial
      publicRingColumns verifierRows publicFits
      (KView.finElement shape.runningCount running
        (KView.finElement shape.matrixCount matrix (ringKView lane))))

noncomputable def proofFeRowView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (round : Fin shape.rowVariables)
    (slot : Fin (rowSlotCount constraintPolynomial)) :
    KView
      (proofCodec shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits)
      (fun proof =>
        (proof.certificate.piCcs.fe.rowRounds round).coefficients.getD
          slot.val Concrete.K.zero) :=
  liftProofK shape constraintPolynomial priorAbsorbed
    publicRingColumns verifierRows publicFits
    (coordinatesDView shape constraintPolynomial
      publicRingColumns verifierRows publicFits
      (KView.finElement shape.rowVariables round
        (KView.finElement
          (rowSlotCount constraintPolynomial) slot kView)))

noncomputable def proofFeLaneView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (round : Fin Domains.fe.laneVariables)
    (slot : Fin 3) :
    KView
      (proofCodec shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits)
      (fun proof =>
        (proof.certificate.piCcs.fe.laneRounds round).coefficients.getD
          slot.val Concrete.K.zero) :=
  liftProofK shape constraintPolynomial priorAbsorbed
    publicRingColumns verifierRows publicFits
    (coordinatesEView shape constraintPolynomial
      publicRingColumns verifierRows publicFits
      (KView.finElement Domains.fe.laneVariables round
        (KView.finElement 3 slot kView)))

noncomputable def proofNcView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (round :
      Fin (Domains.nc.blockVariables + Domains.nc.laneVariables))
    (slot : Fin 5) :
    KView
      (proofCodec shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits)
      (fun proof =>
        (proof.certificate.piCcs.nc.rounds round).coefficients.getD
          slot.val Concrete.K.zero) :=
  liftProofK shape constraintPolynomial priorAbsorbed
    publicRingColumns verifierRows publicFits
    (coordinatesFView shape constraintPolynomial
      publicRingColumns verifierRows publicFits
      (KView.finElement
        (Domains.nc.blockVariables + Domains.nc.laneVariables) round
        (KView.finElement 5 slot kView)))

noncomputable def proofOutputYRingView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (source : Fin shape.sourceCount)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    KView
      (proofCodec shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits)
      (outputYRingCoordinate source matrix lane) :=
  liftProofK shape constraintPolynomial priorAbsorbed
    publicRingColumns verifierRows publicFits
    (coordinatesGView shape constraintPolynomial
      publicRingColumns verifierRows publicFits
      (KView.finElement shape.sourceCount source
        (KView.finElement shape.matrixCount matrix (ringKView lane))))

noncomputable def proofOutputYZcolView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (source : Fin shape.sourceCount)
    (lane : Fin ringDegree) :
    KView
      (proofCodec shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits)
      (outputYZcolCoordinate source lane) :=
  liftProofK shape constraintPolynomial priorAbsorbed
    publicRingColumns verifierRows publicFits
    (coordinatesHView shape constraintPolynomial
      publicRingColumns verifierRows publicFits
      (KView.finElement shape.sourceCount source (ringKView lane)))

noncomputable def proofChallengeView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (coordinate : Fin FixedActive.arity.total)
    (lane : Fin ringDegree) :
    FView
      (proofCodec shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits)
      (fun proof =>
        proof.certificate.piRlcChallenges coordinate lane) :=
  liftProofF shape constraintPolynomial priorAbsorbed
    publicRingColumns verifierRows publicFits
    (coordinatesIView shape constraintPolynomial
      publicRingColumns verifierRows publicFits
      (FView.finElement FixedActive.arity.total coordinate
        (ringFView lane)))

private noncomputable def payloadCommitmentView
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (row : Fin verifierRows)
    (lane : Fin ringDegree) :
  FView
      (payloadCodec shape publicRingColumns verifierRows publicFits)
      (fun payload => payload.commitment row lane) :=
  FView.throughPullback payloadData payloadData_injective
    (FView.productLeft
      (right :=
        Codec.product
          (publicInputCodec (ringDegree * publicRingColumns))
          (evaluationsCodec shape.matrixCount))
      (commitmentView verifierRows row lane))

private noncomputable def payloadPublicView
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (column : Fin (ringDegree * publicRingColumns)) :
  FView
      (payloadCodec shape publicRingColumns verifierRows publicFits)
      (fun payload => payload.publicInput column) :=
  FView.throughPullback payloadData payloadData_injective
    (FView.productRight (left := commitmentCodec verifierRows)
      (FView.productLeft
        (right := evaluationsCodec shape.matrixCount)
        (publicInputView
          (ringDegree * publicRingColumns) column)))

private noncomputable def payloadEvaluationView
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
  KView
      (payloadCodec shape publicRingColumns verifierRows publicFits)
      (fun payload =>
        payload.evaluations.getD matrix.val ringKZero lane) :=
  KView.throughPullback payloadData payloadData_injective
    (KView.productRight (left := commitmentCodec verifierRows)
      (KView.productRight
        (left := publicInputCodec (ringDegree * publicRingColumns))
        (evaluationView shape.matrixCount matrix lane)))

noncomputable def proofPayloadCommitmentView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (child : Fin productionGlobalParams.k)
    (row : Fin verifierRows)
    (lane : Fin ringDegree) :
    FView
      (proofCodec shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits)
      (payloadCommitmentCoordinate child row lane) :=
  liftProofF shape constraintPolynomial priorAbsorbed
    publicRingColumns verifierRows publicFits
    (coordinatesJFView shape constraintPolynomial
      publicRingColumns verifierRows publicFits
      (FView.finElement productionGlobalParams.k child
        (payloadCommitmentView
          shape publicRingColumns verifierRows publicFits row lane)))

noncomputable def proofPayloadPublicView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (child : Fin productionGlobalParams.k)
    (column : Fin (ringDegree * publicRingColumns)) :
    FView
      (proofCodec shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits)
      (payloadPublicCoordinate child column) :=
  liftProofF shape constraintPolynomial priorAbsorbed
    publicRingColumns verifierRows publicFits
    (coordinatesJFView shape constraintPolynomial
      publicRingColumns verifierRows publicFits
      (FView.finElement productionGlobalParams.k child
        (payloadPublicView
          shape publicRingColumns verifierRows publicFits column)))

noncomputable def proofPayloadEvaluationView
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (child : Fin productionGlobalParams.k)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    KView
      (proofCodec shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits)
      (payloadEvaluationCoordinate child matrix lane) :=
  liftProofK shape constraintPolynomial priorAbsorbed
    publicRingColumns verifierRows publicFits
    (coordinatesJKView shape constraintPolynomial
      publicRingColumns verifierRows publicFits
      (KView.finElement productionGlobalParams.k child
        (payloadEvaluationView
          shape publicRingColumns verifierRows publicFits matrix lane)))

/-! ## Complete payload views -/

noncomputable def completeCommitmentView
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (row : Fin verifierRows)
    (lane : Fin ringDegree) :
    FView
      (completePayloadCodec
        shape publicRingColumns verifierRows publicFits)
      (fun payload => payload.1 row lane) :=
  FView.productLeft
    (right :=
      Codec.product
        (publicInputCodec (ringDegree * publicRingColumns))
        (Codec.product
          (pointCodec shape.rowVariables)
          (evaluationsCodec shape.matrixCount)))
    (commitmentView verifierRows row lane)

noncomputable def completePublicView
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (column : Fin (ringDegree * publicRingColumns)) :
    FView
      (completePayloadCodec
        shape publicRingColumns verifierRows publicFits)
      (fun payload => payload.2.1 column) :=
  FView.productRight (left := commitmentCodec verifierRows)
    (FView.productLeft
      (right :=
        Codec.product
          (pointCodec shape.rowVariables)
          (evaluationsCodec shape.matrixCount))
      (publicInputView (ringDegree * publicRingColumns) column))

noncomputable def completePointView
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (coordinate : Fin shape.rowVariables) :
    KView
      (completePayloadCodec
        shape publicRingColumns verifierRows publicFits)
      (fun payload =>
        ConcreteNifsCarrierViews.pointCoordinate coordinate payload.2.2.1) :=
  KView.productRight (left := commitmentCodec verifierRows)
    (KView.productRight
      (left := publicInputCodec (ringDegree * publicRingColumns))
      (KView.productLeft
        (right := evaluationsCodec shape.matrixCount)
        (pointView shape.rowVariables coordinate)))

noncomputable def completeEvaluationView
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    KView
      (completePayloadCodec
        shape publicRingColumns verifierRows publicFits)
      (fun payload => payload.2.2.2.getD matrix.val ringKZero lane) :=
  KView.productRight (left := commitmentCodec verifierRows)
    (KView.productRight
      (left := publicInputCodec (ringDegree * publicRingColumns))
      (KView.productRight
        (left := pointCodec shape.rowVariables)
        (evaluationView shape.matrixCount matrix lane)))

/-! ## Running and fresh views -/

noncomputable def parentCommitmentView
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (row : Fin verifierRows)
    (lane : Fin ringDegree) :
    FView
      (parentPayloadCodec
        shape publicRingColumns verifierRows publicFits)
      (fun payload => payload.commitment row lane) :=
  FView.throughPullback parentPayloadData parentPayloadData_injective
    (completeCommitmentView
      shape publicRingColumns verifierRows publicFits row lane)

noncomputable def parentPublicView
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (column : Fin (ringDegree * publicRingColumns)) :
    FView
      (parentPayloadCodec
        shape publicRingColumns verifierRows publicFits)
      (fun payload => payload.publicInput column) :=
  FView.throughPullback parentPayloadData parentPayloadData_injective
    (completePublicView
      shape publicRingColumns verifierRows publicFits column)

noncomputable def parentPointView
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (coordinate : Fin shape.rowVariables) :
    KView
      (parentPayloadCodec
        shape publicRingColumns verifierRows publicFits)
      (fun payload =>
        ConcreteNifsCarrierViews.pointCoordinate coordinate payload.point) :=
  KView.throughPullback parentPayloadData parentPayloadData_injective
    (completePointView
      shape publicRingColumns verifierRows publicFits coordinate)

noncomputable def parentEvaluationView
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    KView
      (parentPayloadCodec
        shape publicRingColumns verifierRows publicFits)
      (fun payload => payload.evaluations.getD matrix.val ringKZero lane) :=
  KView.throughPullback parentPayloadData parentPayloadData_injective
    (completeEvaluationView
      shape publicRingColumns verifierRows publicFits matrix lane)

noncomputable def childCommitmentPayloadView
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (row : Fin verifierRows)
    (lane : Fin ringDegree) :
    FView
      (runningPayloadCodec
        shape publicRingColumns verifierRows publicFits)
      (fun payload => payload.commitment row lane) :=
  FView.throughPullback runningPayloadData runningPayloadData_injective
    (completeCommitmentView
      shape publicRingColumns verifierRows publicFits row lane)

noncomputable def childPublicPayloadView
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (column : Fin (ringDegree * publicRingColumns)) :
    FView
      (runningPayloadCodec
        shape publicRingColumns verifierRows publicFits)
      (fun payload => payload.publicInput column) :=
  FView.throughPullback runningPayloadData runningPayloadData_injective
    (completePublicView
      shape publicRingColumns verifierRows publicFits column)

noncomputable def childPointPayloadView
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (coordinate : Fin shape.rowVariables) :
    KView
      (runningPayloadCodec
        shape publicRingColumns verifierRows publicFits)
      (fun payload =>
        ConcreteNifsCarrierViews.pointCoordinate coordinate payload.point) :=
  KView.throughPullback runningPayloadData runningPayloadData_injective
    (completePointView
      shape publicRingColumns verifierRows publicFits coordinate)

noncomputable def childEvaluationPayloadView
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    KView
      (runningPayloadCodec
        shape publicRingColumns verifierRows publicFits)
      (fun payload => payload.evaluations.getD matrix.val ringKZero lane) :=
  KView.throughPullback runningPayloadData runningPayloadData_injective
    (completeEvaluationView
      shape publicRingColumns verifierRows publicFits matrix lane)

noncomputable def runningViews
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    RunningViews
      (runningCodec shape publicRingColumns verifierRows publicFits) where
  parentCommitment row lane :=
    FView.throughPullback runningData runningData_injective
      (FView.productLeft
        (right :=
          Codec.finFunction productionGlobalParams.k
            (runningPayloadCodec
              shape publicRingColumns verifierRows publicFits))
        (parentCommitmentView
          shape publicRingColumns verifierRows publicFits row lane))
  childCommitment child row lane :=
    FView.throughPullback runningData runningData_injective
      (FView.productRight
        (left :=
          parentPayloadCodec
            shape publicRingColumns verifierRows publicFits)
        (FView.finElement productionGlobalParams.k child
          (childCommitmentPayloadView
            shape publicRingColumns verifierRows publicFits row lane)))
  parentPublic column :=
    FView.throughPullback runningData runningData_injective
      (FView.productLeft
        (right :=
          Codec.finFunction productionGlobalParams.k
            (runningPayloadCodec
              shape publicRingColumns verifierRows publicFits))
        (parentPublicView
          shape publicRingColumns verifierRows publicFits column))
  childPublic child column :=
    FView.throughPullback runningData runningData_injective
      (FView.productRight
        (left :=
          parentPayloadCodec
            shape publicRingColumns verifierRows publicFits)
        (FView.finElement productionGlobalParams.k child
          (childPublicPayloadView
            shape publicRingColumns verifierRows publicFits column)))
  parentPoint coordinate :=
    KView.throughPullback runningData runningData_injective
      (KView.productLeft
        (right :=
          Codec.finFunction productionGlobalParams.k
            (runningPayloadCodec
              shape publicRingColumns verifierRows publicFits))
        (parentPointView
          shape publicRingColumns verifierRows publicFits coordinate))
  childPoint child coordinate :=
    KView.throughPullback runningData runningData_injective
      (KView.productRight
        (left :=
          parentPayloadCodec
            shape publicRingColumns verifierRows publicFits)
        (KView.finElement productionGlobalParams.k child
          (childPointPayloadView
            shape publicRingColumns verifierRows publicFits coordinate)))
  parentEvaluation matrix lane :=
    KView.throughPullback runningData runningData_injective
      (KView.productLeft
        (right :=
          Codec.finFunction productionGlobalParams.k
            (runningPayloadCodec
              shape publicRingColumns verifierRows publicFits))
        (parentEvaluationView
          shape publicRingColumns verifierRows publicFits matrix lane))
  childEvaluation child matrix lane :=
    KView.throughPullback runningData runningData_injective
      (KView.productRight
        (left :=
          parentPayloadCodec
            shape publicRingColumns verifierRows publicFits)
        (KView.finElement productionGlobalParams.k child
          (childEvaluationPayloadView
            shape publicRingColumns verifierRows publicFits matrix lane)))
  parentEvaluationsSize running admissible :=
    (runningCodec_admissible_iff running).mp admissible |>.1
  childEvaluationsSize running admissible :=
    (runningCodec_admissible_iff running).mp admissible |>.2

noncomputable def freshViews
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    FreshViews
      (freshCodec shape publicRingColumns verifierRows publicFits) where
  commitment row lane :=
    FView.throughPullback freshPayloadData freshPayloadData_injective
      (FView.productLeft
        (right := publicInputCodec (ringDegree * publicRingColumns))
        (commitmentView verifierRows row lane))
  publicInput column :=
    FView.throughPullback freshPayloadData freshPayloadData_injective
      (FView.productRight
        (left := commitmentCodec verifierRows)
        (publicInputView (ringDegree * publicRingColumns) column))

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalViews
