import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofCodec
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalViews
import Nightstream.Implementation.Lowering.Goldilocks.CodecRecovery

/-!
Contract: reconstruct one admissible selected NIFS proof from every complete
canonical dynamic-coordinate value.

Owns: the inverse semantic construction for `proofCoordinates` when the
selected prior duplex is empty and its absorbed cursor is zero.

Does not own: physical column decoding, verifier acceptance, transcript
soundness, application inputs, Rust, or generated artifacts.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofRecovery

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofCodec
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalViews
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

private abbrev TranscriptState := Poseidon2Duplex.State
private abbrev Domains :=
  Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production

private def kOfPair (pair : Field × Field) : K where
  c0 := pair.1
  c1 := pair.2

private theorem kPair_kOfPair (pair : Field × Field) :
    ConcreteNifsCanonicalCodecCore.kPair (kOfPair pair) = pair := by
  cases pair
  rfl

private def pointOfList
    (variables : Nat)
    (values : List K) :
    CubePoint K variables :=
  if lengthExact : values.length = variables then
    ⟨values, lengthExact⟩
  else
    ⟨List.replicate variables K.zero, by simp⟩

private theorem pointData_pointOfList
    (variables : Nat)
    (values : List K)
    (admissible :
      (Codec.fixedList variables K.zero
        ConcreteNifsCanonicalCodecCore.kCodec).Admissible values) :
    ConcreteNifsCanonicalCodecCore.pointData
        (pointOfList variables values) =
      values := by
  simp [pointOfList, admissible.1,
    ConcreteNifsCanonicalCodecCore.pointData]

theorem priorLaneCodec_exactWidthRecoverable :
    priorLaneCodec.ExactWidthRecoverable := by
  apply Codec.pullbackOn_exactWidthRecoverable
    fieldCodec
    (fun value =>
      value < Nightstream.Implementation.R1CS.goldilocksP)
    NumericRowBridge.residue
    (fun _ _ => True.intro)
    (fun leftLt rightLt equal =>
      NumericRowBridge.residue_injective_of_lt leftLt rightLt equal)
    (fun value => value.val)
  · intro value targetAdmissible
    exact value.isLt
  · intro value targetAdmissible
    exact NumericRowBridge.residue_field_val value
  · exact Codec.fieldCodec_exactWidthRecoverable

theorem kCodec_exactWidthRecoverable :
    ConcreteNifsCanonicalCodecCore.kCodec.ExactWidthRecoverable := by
  apply Codec.pullback_exactWidthRecoverable
    (Codec.product fieldCodec fieldCodec)
    ConcreteNifsCanonicalCodecCore.kPair
    ConcreteNifsCanonicalCodecCore.kPair_injective
    kOfPair
    (fun pair _ => kPair_kOfPair pair)
  exact Codec.product_exactWidthRecoverable
    fieldCodec fieldCodec
    Codec.fieldCodec_exactWidthRecoverable
    Codec.fieldCodec_exactWidthRecoverable

theorem ringFCodec_exactWidthRecoverable :
    ConcreteNifsCanonicalCodecCore.ringFCodec.ExactWidthRecoverable := by
  exact Codec.finFunction_exactWidthRecoverable
    fieldCodec Codec.fieldCodec_exactWidthRecoverable ringDegree

theorem ringKCodec_exactWidthRecoverable :
    ConcreteNifsCanonicalCodecCore.ringKCodec.ExactWidthRecoverable := by
  exact Codec.finFunction_exactWidthRecoverable
    ConcreteNifsCanonicalCodecCore.kCodec
    kCodec_exactWidthRecoverable ringDegree

theorem pointCodec_exactWidthRecoverable
    (variables : Nat) :
    (ConcreteNifsCanonicalCodecCore.pointCodec variables
      ).ExactWidthRecoverable := by
  apply Codec.pullback_exactWidthRecoverable
    (Codec.fixedList variables K.zero
      ConcreteNifsCanonicalCodecCore.kCodec)
    ConcreteNifsCanonicalCodecCore.pointData
    ConcreteNifsCanonicalCodecCore.pointData_injective
    (pointOfList variables)
    (fun values admissible =>
      pointData_pointOfList variables values admissible)
  exact Codec.fixedList_exactWidthRecoverable
    variables K.zero ConcreteNifsCanonicalCodecCore.kCodec
    kCodec_exactWidthRecoverable

theorem evaluationsCodec_exactWidthRecoverable
    (matrixCount : Nat) :
    (ConcreteNifsCanonicalCodecCore.evaluationsCodec matrixCount
      ).ExactWidthRecoverable := by
  exact Codec.fixedArray_exactWidthRecoverable
    matrixCount ringKZero ConcreteNifsCanonicalCodecCore.ringKCodec
    ringKCodec_exactWidthRecoverable

theorem commitmentCodec_exactWidthRecoverable
    (verifierRows : Nat) :
    (ConcreteNifsCanonicalCodecCore.commitmentCodec verifierRows
      ).ExactWidthRecoverable := by
  exact Codec.finFunction_exactWidthRecoverable
    ConcreteNifsCanonicalCodecCore.ringFCodec
    ringFCodec_exactWidthRecoverable verifierRows

theorem publicInputCodec_exactWidthRecoverable
    (publicWidth : Nat) :
    (ConcreteNifsCanonicalCodecCore.publicInputCodec publicWidth
      ).ExactWidthRecoverable := by
  exact Codec.finFunction_exactWidthRecoverable
    fieldCodec Codec.fieldCodec_exactWidthRecoverable publicWidth

theorem payloadCodec_exactWidthRecoverable
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    (payloadCodec shape publicRingColumns verifierRows publicFits
      ).ExactWidthRecoverable := by
  unfold payloadCodec
  apply Codec.pullback_exactWidthRecoverable
    (Codec.product
      (ConcreteNifsCanonicalCodecCore.commitmentCodec verifierRows)
      (Codec.product
        (ConcreteNifsCanonicalCodecCore.publicInputCodec
          (ringDegree * publicRingColumns))
        (ConcreteNifsCanonicalCodecCore.evaluationsCodec
          shape.matrixCount)))
    ConcreteNifsCanonicalProofCodec.payloadData
    ConcreteNifsCanonicalProofCodec.payloadData_injective
    (fun data => {
      commitment := data.1
      publicInput := data.2.1
      evaluations := data.2.2
    })
    (by
      intro data admissible
      cases data
      rfl)
  exact Codec.product_exactWidthRecoverable
    (ConcreteNifsCanonicalCodecCore.commitmentCodec verifierRows)
    (Codec.product
      (ConcreteNifsCanonicalCodecCore.publicInputCodec
        (ringDegree * publicRingColumns))
      (ConcreteNifsCanonicalCodecCore.evaluationsCodec shape.matrixCount))
    (commitmentCodec_exactWidthRecoverable verifierRows)
    (Codec.product_exactWidthRecoverable
      (ConcreteNifsCanonicalCodecCore.publicInputCodec
        (ringDegree * publicRingColumns))
      (ConcreteNifsCanonicalCodecCore.evaluationsCodec shape.matrixCount)
      (publicInputCodec_exactWidthRecoverable
        (ringDegree * publicRingColumns))
      (evaluationsCodec_exactWidthRecoverable shape.matrixCount))

theorem coordinatesCodec_exactWidthRecoverable
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    (coordinatesCodec shape constraintPolynomial
      publicRingColumns verifierRows publicFits).ExactWidthRecoverable := by
  let claimedRecoverable :=
    Codec.finFunction_exactWidthRecoverable
      (Codec.finFunction shape.matrixCount
        ConcreteNifsCanonicalCodecCore.ringKCodec)
      (Codec.finFunction_exactWidthRecoverable
        ConcreteNifsCanonicalCodecCore.ringKCodec
        ringKCodec_exactWidthRecoverable shape.matrixCount)
      shape.runningCount
  let priorRecoverable :=
    Codec.finFunction_exactWidthRecoverable
      priorLaneCodec priorLaneCodec_exactWidthRecoverable 8
  let feRowRecoverable :=
    Codec.finFunction_exactWidthRecoverable
      (Codec.finFunction (rowSlotCount constraintPolynomial)
        ConcreteNifsCanonicalCodecCore.kCodec)
      (Codec.finFunction_exactWidthRecoverable
        ConcreteNifsCanonicalCodecCore.kCodec
        kCodec_exactWidthRecoverable
        (rowSlotCount constraintPolynomial))
      shape.rowVariables
  let feLaneRecoverable :=
    Codec.finFunction_exactWidthRecoverable
      (Codec.finFunction 3 ConcreteNifsCanonicalCodecCore.kCodec)
      (Codec.finFunction_exactWidthRecoverable
        ConcreteNifsCanonicalCodecCore.kCodec
        kCodec_exactWidthRecoverable 3)
      Domains.fe.laneVariables
  let ncRecoverable :=
    Codec.finFunction_exactWidthRecoverable
      (Codec.finFunction 5 ConcreteNifsCanonicalCodecCore.kCodec)
      (Codec.finFunction_exactWidthRecoverable
        ConcreteNifsCanonicalCodecCore.kCodec
        kCodec_exactWidthRecoverable 5)
      (Domains.nc.blockVariables + Domains.nc.laneVariables)
  let outputYRingRecoverable :=
    Codec.finFunction_exactWidthRecoverable
      (Codec.finFunction shape.matrixCount
        ConcreteNifsCanonicalCodecCore.ringKCodec)
      (Codec.finFunction_exactWidthRecoverable
        ConcreteNifsCanonicalCodecCore.ringKCodec
        ringKCodec_exactWidthRecoverable shape.matrixCount)
      shape.sourceCount
  let outputYZcolRecoverable :=
    Codec.finFunction_exactWidthRecoverable
      ConcreteNifsCanonicalCodecCore.ringKCodec
      ringKCodec_exactWidthRecoverable shape.sourceCount
  let piRlcRecoverable :=
    Codec.finFunction_exactWidthRecoverable
      ConcreteNifsCanonicalCodecCore.ringFCodec
      ringFCodec_exactWidthRecoverable FixedActive.arity.total
  let piDecRecoverable :=
    Codec.finFunction_exactWidthRecoverable
      (payloadCodec shape publicRingColumns verifierRows publicFits)
      (payloadCodec_exactWidthRecoverable shape
        publicRingColumns verifierRows publicFits)
      productionGlobalParams.k
  unfold coordinatesCodec
  apply Codec.pullback_exactWidthRecoverable
    _
    coordinateData
    coordinateData_injective
    (fun data => {
      priorPoint := data.1
      claimedYRing := data.2.1
      priorLanes := data.2.2.1
      feRow := data.2.2.2.1
      feLane := data.2.2.2.2.1
      nc := data.2.2.2.2.2.1
      outputYRing := data.2.2.2.2.2.2.1
      outputYZcol := data.2.2.2.2.2.2.2.1
      piRlcChallenges := data.2.2.2.2.2.2.2.2.1
      piDecPayloads := data.2.2.2.2.2.2.2.2.2
    })
    (by
      intro data admissible
      rcases data with
        ⟨priorPoint, claimedYRing, priorLanes, feRow, feLane, nc,
          outputYRing, outputYZcol, piRlcChallenges, piDecPayloads⟩
      rfl)
  exact Codec.product_exactWidthRecoverable
    (ConcreteNifsCanonicalCodecCore.pointCodec shape.rowVariables)
    _
    (pointCodec_exactWidthRecoverable shape.rowVariables)
    (Codec.product_exactWidthRecoverable
      (Codec.finFunction shape.runningCount
        (Codec.finFunction shape.matrixCount
          ConcreteNifsCanonicalCodecCore.ringKCodec))
      _
      claimedRecoverable
      (Codec.product_exactWidthRecoverable
        (Codec.finFunction 8 priorLaneCodec)
        _
        priorRecoverable
        (Codec.product_exactWidthRecoverable
          (Codec.finFunction shape.rowVariables
            (Codec.finFunction (rowSlotCount constraintPolynomial)
              ConcreteNifsCanonicalCodecCore.kCodec))
          _
          feRowRecoverable
          (Codec.product_exactWidthRecoverable
            (Codec.finFunction Domains.fe.laneVariables
              (Codec.finFunction 3
                ConcreteNifsCanonicalCodecCore.kCodec))
            _
            feLaneRecoverable
            (Codec.product_exactWidthRecoverable
              (Codec.finFunction
                (Domains.nc.blockVariables + Domains.nc.laneVariables)
                (Codec.finFunction 5
                  ConcreteNifsCanonicalCodecCore.kCodec))
              _
              ncRecoverable
              (Codec.product_exactWidthRecoverable
                (Codec.finFunction shape.sourceCount
                  (Codec.finFunction shape.matrixCount
                    ConcreteNifsCanonicalCodecCore.ringKCodec))
                _
                outputYRingRecoverable
                (Codec.product_exactWidthRecoverable
                  (Codec.finFunction shape.sourceCount
                    ConcreteNifsCanonicalCodecCore.ringKCodec)
                  _
                  outputYZcolRecoverable
                  (Codec.product_exactWidthRecoverable
                    (Codec.finFunction FixedActive.arity.total
                      ConcreteNifsCanonicalCodecCore.ringFCodec)
                    (Codec.finFunction productionGlobalParams.k
                      (payloadCodec shape publicRingColumns
                        verifierRows publicFits))
                piRlcRecoverable
                piDecRecoverable))))))))

/-- Rebuild the complete selected proof from the dynamic coordinate carrier
and the two setup-owned values omitted by the codec. -/
def proofOfCoordinates
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (coordinates :
      Coordinates shape constraintPolynomial
        publicRingColumns verifierRows publicFits) :
    SelectedProof shape TranscriptState publicRingColumns publicFits
      verifierRows where
  piCcsInput := {
    constraintPolynomial := constraintPolynomial
    priorPoint := coordinates.priorPoint
    claimedYRing := coordinates.claimedYRing
  }
  priorState := Poseidon2Duplex.empty
  certificate := {
    piCcs := {
      fe := {
        rowRounds := fun round => {
          coefficients := List.ofFn (coordinates.feRow round)
          coefficients_length := by
            change
              (List.ofFn (coordinates.feRow round)).length =
                rowSlotCount constraintPolynomial
            simp
        }
        laneRounds := fun round => {
          coefficients := List.ofFn (coordinates.feLane round)
          coefficients_length := by
            simp [Polynomial.Fe.laneSumcheckDegreeBound]
        }
      }
      nc := {
        rounds := fun round => {
          coefficients := List.ofFn (coordinates.nc round)
          coefficients_length := by
            simp [Polynomial.Nc.Degree.ncSumcheckDegreeBound]
        }
      }
      output := {
        yRing := coordinates.outputYRing
        yZcol := coordinates.outputYZcol
      }
    }
    piRlcChallenges := coordinates.piRlcChallenges
    piDecPayloads := coordinates.piDecPayloads
  }

/-- The selected proof carrier is inhabited independently of any physical
proof coordinates. This witness is used only on branches where the NIFS proof
is semantically inactive. -/
theorem selectedProof_nonempty
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    Nonempty
      (SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) := by
  let codec :=
    coordinatesCodec shape constraintPolynomial
      publicRingColumns verifierRows publicFits
  let fields : List Field := List.replicate codec.width 0
  have lengthExact : fields.length = codec.width := by
    simp [fields]
  rcases
      (coordinatesCodec_exactWidthRecoverable shape constraintPolynomial
        publicRingColumns verifierRows publicFits) fields lengthExact with
    ⟨coordinates, _, _⟩
  exact
    ⟨proofOfCoordinates shape constraintPolynomial
      publicRingColumns verifierRows publicFits coordinates⟩

/-- The reconstructed proof is inside the exact setup-owned source domain of
the canonical proof codec. -/
theorem proofOfCoordinates_admissible
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (coordinates :
      Coordinates shape constraintPolynomial
        publicRingColumns verifierRows publicFits)
    (coordinatesAdmissible :
      (coordinatesCodec shape constraintPolynomial
        publicRingColumns verifierRows publicFits).Admissible coordinates) :
    ProofAdmissible constraintPolynomial 0
      (proofOfCoordinates shape constraintPolynomial
        publicRingColumns verifierRows publicFits coordinates) := by
  constructor
  · rfl
  · rfl
  · rfl
  · intro child
    have payloadAdmissible :
        ∀ child,
          (payloadCodec shape publicRingColumns verifierRows publicFits
            ).Admissible (coordinates.piDecPayloads child) :=
      coordinatesAdmissible.2.2.2.2.2.2.2.2.2
    exact (payloadAdmissible child).2.2.1

/-- Reconstructing a proof preserves every dynamic coordinate exactly. -/
theorem proofCoordinates_proofOfCoordinates
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (coordinates :
      Coordinates shape constraintPolynomial
        publicRingColumns verifierRows publicFits)
    (priorLanes :
      Poseidon2Duplex.empty.lanes = coordinates.priorLanes) :
    proofCoordinates constraintPolynomial
        (proofOfCoordinates shape constraintPolynomial
          publicRingColumns verifierRows publicFits coordinates) =
      coordinates := by
  cases coordinates
  simp [proofCoordinates, proofOfCoordinates, priorLanes]
  constructor
  · funext round slot
    refine Fin.cases rfl (fun slot => ?_) slot
    refine Fin.cases rfl (fun slot => ?_) slot
    exact Fin.cases rfl (fun impossible => Fin.elim0 impossible) slot
  · funext round slot
    refine Fin.cases rfl (fun slot => ?_) slot
    refine Fin.cases rfl (fun slot => ?_) slot
    refine Fin.cases rfl (fun slot => ?_) slot
    refine Fin.cases rfl (fun slot => ?_) slot
    exact Fin.cases rfl (fun impossible => Fin.elim0 impossible) slot

/-- Exact-width coordinates decode to one selected proof once the physical
program has established the omitted empty-duplex lane condition. -/
theorem proofCodec_decode_exists_of_priorLanes
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (fieldCoordinates : List Field)
    (lengthExact :
      fieldCoordinates.length =
        (proofCodec shape constraintPolynomial 0
          publicRingColumns verifierRows publicFits).width)
    (priorLanesZero :
      ∀ coordinates,
        (coordinatesCodec shape constraintPolynomial
          publicRingColumns verifierRows publicFits).Admissible
            coordinates →
        (coordinatesCodec shape constraintPolynomial
          publicRingColumns verifierRows publicFits).encode coordinates =
            fieldCoordinates →
        Poseidon2Duplex.empty.lanes = coordinates.priorLanes) :
    ∃ proof,
      (proofCodec shape constraintPolynomial 0
        publicRingColumns verifierRows publicFits).decode
          fieldCoordinates =
        some proof := by
  have coordinateLength :
      fieldCoordinates.length =
        (coordinatesCodec shape constraintPolynomial
          publicRingColumns verifierRows publicFits).width := by
    simpa [proofCodec, Codec.pullbackOn, Codec.ofInjectiveEncoding] using
      lengthExact
  rcases
      (coordinatesCodec_exactWidthRecoverable shape constraintPolynomial
        publicRingColumns verifierRows publicFits)
        fieldCoordinates coordinateLength with
    ⟨coordinates, coordinatesAdmissible, coordinatesEncoded⟩
  let proof :=
    proofOfCoordinates shape constraintPolynomial
      publicRingColumns verifierRows publicFits coordinates
  have proofAdmissible :
      ProofAdmissible constraintPolynomial 0 proof :=
    proofOfCoordinates_admissible shape constraintPolynomial
      publicRingColumns verifierRows publicFits coordinates
      coordinatesAdmissible
  have coordinatesExact :
      proofCoordinates constraintPolynomial proof = coordinates :=
    proofCoordinates_proofOfCoordinates shape constraintPolynomial
      publicRingColumns verifierRows publicFits coordinates
      (priorLanesZero coordinates coordinatesAdmissible coordinatesEncoded)
  refine ⟨proof, ?_⟩
  have decoded :=
    (proofCodec shape constraintPolynomial 0
      publicRingColumns verifierRows publicFits).decode_encode
        proof proofAdmissible
  have encoded :
      (proofCodec shape constraintPolynomial 0
        publicRingColumns verifierRows publicFits).encode proof =
          fieldCoordinates := by
    change
      (coordinatesCodec shape constraintPolynomial
        publicRingColumns verifierRows publicFits).encode
          (proofCoordinates constraintPolynomial proof) =
        fieldCoordinates
    rw [coordinatesExact, coordinatesEncoded]
  rwa [encoded] at decoded

/-- Exact-width physical proof coordinates decode once the eight explicit
prior-duplex coordinates are all zero.  This is the form consumed by the
physical canonicality rows. -/
theorem proofCodec_decode_exists_of_priorLaneCoordinatesZero
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (fieldCoordinates : List Field)
    (lengthExact :
      fieldCoordinates.length =
        (proofCodec shape constraintPolynomial 0
          publicRingColumns verifierRows publicFits).width)
    (priorLaneCoordinatesZero :
      ∀ lane : Fin 8,
        fieldCoordinates.getD
            (coordinatesPriorLaneView shape constraintPolynomial
              publicRingColumns verifierRows publicFits lane).index.val
            0 =
          0) :
    ∃ proof,
      (proofCodec shape constraintPolynomial 0
        publicRingColumns verifierRows publicFits).decode
          fieldCoordinates =
        some proof := by
  apply proofCodec_decode_exists_of_priorLanes
    shape constraintPolynomial publicRingColumns verifierRows publicFits
    fieldCoordinates lengthExact
  intro coordinates coordinatesAdmissible coordinatesEncoded
  funext lane
  have selected :=
    congrArg
      (fun values =>
        values.getD
          (coordinatesPriorLaneView shape constraintPolynomial
            publicRingColumns verifierRows publicFits lane).index.val
          0)
      coordinatesEncoded
  change
    ((coordinatesCodec shape constraintPolynomial
      publicRingColumns verifierRows publicFits).encode coordinates).getD
        (coordinatesPriorLaneView shape constraintPolynomial
          publicRingColumns verifierRows publicFits lane).index.val
        0 =
      fieldCoordinates.getD
        (coordinatesPriorLaneView shape constraintPolynomial
          publicRingColumns verifierRows publicFits lane).index.val
        0 at selected
  rw [
    (coordinatesPriorLaneView shape constraintPolynomial
      publicRingColumns verifierRows publicFits lane).encodeValue
        coordinates,
    priorLaneCoordinatesZero lane
  ] at selected
  have reduced :
      coordinates.priorLanes lane %
          Nightstream.Implementation.R1CS.goldilocksP =
        0 := by
    simpa only [NumericRowBridge.residue] using congrArg Fin.val selected
  have laneBound :
      coordinates.priorLanes lane <
        Nightstream.Implementation.R1CS.goldilocksP :=
    coordinatesAdmissible.2.2.1 lane
  rw [Nat.mod_eq_of_lt laneBound] at reduced
  simpa [Poseidon2Duplex.empty] using reduced.symm

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofRecovery
