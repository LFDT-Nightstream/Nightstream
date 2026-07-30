import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRunningCodec
import Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
import Nightstream.Implementation.R1CS.Canonical.KSplitNcStaticInput

/-!
Contract: one canonical field-coordinate codec for the raw selected
ConcretePhi81 NIFS proof.

Owns: the exact coordinate order of every dynamic public-input claim, prior
duplex lane, FE and NC message coefficient, raw output claim, PiRLC
challenge, and PiDEC child payload.

The selected constraint polynomial and prior absorb cursor are setup fields.
They are fixed by the admissible domain and therefore do not consume prover
coordinates.  Every omitted field is proved equal to its selected setup value.

Does not own: verifier acceptance, transcript challenges, relation matrices,
application state, physical columns, Rust, or artifacts.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofCodec

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCodecCore
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRunningCodec
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

private abbrev TranscriptState := Poseidon2Duplex.State
private abbrev Domains :=
  Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production

/-- Number of physical coefficients in one FE row-phase message for the
selected setup polynomial. -/
def rowSlotCount
    {shape : SemanticShape}
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount) : Nat :=
  SumCheck.Fe.Drow
      (KSplitNcStaticInput.layoutInput constraintPolynomial) + 1

private abbrev ClaimedYRing (shape : SemanticShape) :=
  Fin shape.runningCount → Fin shape.matrixCount → RingK

private abbrev FeRowCoefficients
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount) :=
  Fin shape.rowVariables → Fin (rowSlotCount constraintPolynomial) → K

private abbrev FeLaneCoefficients :=
  Fin Domains.fe.laneVariables → Fin 3 → K

private abbrev NcCoefficients :=
  Fin (Domains.nc.blockVariables + Domains.nc.laneVariables) → Fin 5 → K

private abbrev OutputYRing (shape : SemanticShape) :=
  Fin shape.sourceCount → Fin shape.matrixCount → RingK

private abbrev OutputYZcol (shape : SemanticShape) :=
  Fin shape.sourceCount → RingK

private abbrev PiRlcChallenges :=
  Fin FixedActive.arity.total → RingF

/-- One canonical prior-duplex lane.

The encoding is the field residue for every natural number.  The admissible
domain restricts the lane to its unique canonical representative, so the
coordinate law remains total while decoding remains injective. -/
noncomputable def priorLaneCodec : Codec Nat :=
  Codec.pullbackOn fieldCodec
    (fun value =>
      value < Nightstream.Implementation.R1CS.goldilocksP)
    NumericRowBridge.residue
    (fun _ _ => True.intro)
    (fun leftLt rightLt equal =>
      NumericRowBridge.residue_injective_of_lt leftLt rightLt equal)

private abbrev PayloadData
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :=
  CommitmentValue verifierRows ×
    (Phi81Relation.PublicInput
        (RelationShape shape publicRingColumns publicFits) ×
      Array RingK)

def payloadData
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (payload :
      PiDecChildPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) :
    PayloadData shape publicRingColumns verifierRows publicFits :=
  (payload.commitment, (payload.publicInput, payload.evaluations))

theorem payloadData_injective
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth} :
    Function.Injective
      (payloadData
        (shape := shape) (publicRingColumns := publicRingColumns)
        (verifierRows := verifierRows) (publicFits := publicFits)) := by
  intro left right equal
  cases left
  cases right
  cases equal
  rfl

noncomputable def payloadCodec
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    Codec
      (PiDecChildPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) :=
  Codec.pullback
    (Codec.product
      (commitmentCodec verifierRows)
      (Codec.product
        (publicInputCodec (ringDegree * publicRingColumns))
        (evaluationsCodec shape.matrixCount)))
    payloadData payloadData_injective

theorem payloadCodec_admissible
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (payload :
      PiDecChildPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows))
    (evaluationsSize :
      payload.evaluations.size = shape.matrixCount) :
    (payloadCodec shape publicRingColumns verifierRows publicFits).Admissible
      payload := by
  exact
    ⟨commitmentCodec_admissible payload.commitment,
      publicInputCodec_admissible payload.publicInput,
      evaluationsCodec_admissible payload.evaluations evaluationsSize⟩

/-- Fixed-coordinate view of all dynamic proof data. -/
structure Coordinates
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) where
  priorPoint : CubePoint K shape.rowVariables
  claimedYRing : ClaimedYRing shape
  priorLanes : Fin 8 → Nat
  feRow : FeRowCoefficients shape constraintPolynomial
  feLane : FeLaneCoefficients
  nc : NcCoefficients
  outputYRing : OutputYRing shape
  outputYZcol : OutputYZcol shape
  piRlcChallenges : PiRlcChallenges
  piDecPayloads : Fin productionGlobalParams.k →
    PiDecChildPayload
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows)

private abbrev CoordinateData
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :=
  CubePoint K shape.rowVariables ×
    ClaimedYRing shape ×
      (Fin 8 → Nat) ×
        FeRowCoefficients shape constraintPolynomial ×
          FeLaneCoefficients ×
            NcCoefficients ×
              OutputYRing shape ×
                OutputYZcol shape ×
                  PiRlcChallenges ×
                    (Fin productionGlobalParams.k →
                      PiDecChildPayload
                        (RelationShape shape publicRingColumns publicFits)
                        (CommitmentValue verifierRows))

def coordinateData
    {shape : SemanticShape}
    {constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (coordinates :
      Coordinates shape constraintPolynomial
        publicRingColumns verifierRows publicFits) :
    CoordinateData shape constraintPolynomial
      publicRingColumns verifierRows publicFits :=
  (coordinates.priorPoint,
    (coordinates.claimedYRing,
      (coordinates.priorLanes,
        (coordinates.feRow,
          (coordinates.feLane,
            (coordinates.nc,
              (coordinates.outputYRing,
                (coordinates.outputYZcol,
                  (coordinates.piRlcChallenges,
                    coordinates.piDecPayloads)))))))))

theorem coordinateData_injective
    {shape : SemanticShape}
    {constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth} :
    Function.Injective
      (coordinateData
        (shape := shape)
        (constraintPolynomial := constraintPolynomial)
        (publicRingColumns := publicRingColumns)
        (verifierRows := verifierRows) (publicFits := publicFits)) := by
  intro left right equal
  cases left
  cases right
  cases equal
  rfl

noncomputable def coordinatesCodec
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    Codec
      (Coordinates shape constraintPolynomial
        publicRingColumns verifierRows publicFits) :=
  Codec.pullback
    (Codec.product
      (pointCodec shape.rowVariables)
      (Codec.product
        (Codec.finFunction shape.runningCount
          (Codec.finFunction shape.matrixCount ringKCodec))
        (Codec.product
          (Codec.finFunction 8 priorLaneCodec)
          (Codec.product
            (Codec.finFunction shape.rowVariables
              (Codec.finFunction
                (rowSlotCount constraintPolynomial) kCodec))
            (Codec.product
              (Codec.finFunction Domains.fe.laneVariables
                (Codec.finFunction 3 kCodec))
              (Codec.product
                (Codec.finFunction
                  (Domains.nc.blockVariables + Domains.nc.laneVariables)
                  (Codec.finFunction 5 kCodec))
                (Codec.product
                  (Codec.finFunction shape.sourceCount
                    (Codec.finFunction shape.matrixCount ringKCodec))
                  (Codec.product
                    (Codec.finFunction shape.sourceCount ringKCodec)
                    (Codec.product
                      (Codec.finFunction FixedActive.arity.total ringFCodec)
                      (Codec.finFunction productionGlobalParams.k
                        (payloadCodec shape publicRingColumns verifierRows
                          publicFits)))))))))))
    coordinateData coordinateData_injective

/-- The exact source domain omitted from the physical proof coordinates.

The constraint polynomial and initial transcript state are verifier-owned
setup data.  Each PiDEC evaluation array must have the selected matrix count.
No acceptance equation or transcript challenge is an admissibility field.
-/
structure ProofAdmissible
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) : Prop where
  constraintPolynomial_eq :
    proof.piCcsInput.constraintPolynomial = constraintPolynomial
  priorState_eq :
    proof.priorState = Poseidon2Duplex.empty
  priorAbsorbed_eq : proof.priorState.absorbed = priorAbsorbed
  piDecEvaluations_size :
    ∀ child,
      (proof.certificate.piDecPayloads child).evaluations.size =
        shape.matrixCount

namespace ProofAdmissible

/-- Verifier ownership of the initial duplex state also supplies the
canonical-residue bound required by the physical field codec. -/
theorem priorLane_lt
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount}
    {priorAbsorbed : Nat}
    {proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows}
    (admissible :
      ProofAdmissible constraintPolynomial priorAbsorbed proof)
    (lane : Fin 8) :
    proof.priorState.lanes lane <
      Nightstream.Implementation.R1CS.goldilocksP := by
  rw [admissible.priorState_eq]
  simp [Poseidon2Duplex.empty]
  decide

end ProofAdmissible

/-- Remove only the two selected setup fields and retain every dynamic proof
coordinate in its canonical order. -/
def proofCoordinates
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) :
    Coordinates shape constraintPolynomial
      publicRingColumns verifierRows publicFits where
  priorPoint := proof.piCcsInput.priorPoint
  claimedYRing := proof.piCcsInput.claimedYRing
  priorLanes := proof.priorState.lanes
  feRow := fun round slot =>
    (proof.certificate.piCcs.fe.rowRounds round).coefficients.getD
      slot.val K.zero
  feLane := fun round slot =>
    (proof.certificate.piCcs.fe.laneRounds round).coefficients.getD
      slot.val K.zero
  nc := fun round slot =>
    (proof.certificate.piCcs.nc.rounds round).coefficients.getD
      slot.val K.zero
  outputYRing := proof.certificate.piCcs.output.yRing
  outputYZcol := proof.certificate.piCcs.output.yZcol
  piRlcChallenges := proof.certificate.piRlcChallenges
  piDecPayloads := proof.certificate.piDecPayloads

theorem proofCoordinates_admissible
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount}
    {priorAbsorbed : Nat}
    {proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows}
    (admissible :
      ProofAdmissible constraintPolynomial priorAbsorbed proof) :
    (coordinatesCodec shape constraintPolynomial
      publicRingColumns verifierRows publicFits).Admissible
        (proofCoordinates constraintPolynomial proof) := by
  exact
    ⟨pointCodec_admissible proof.piCcsInput.priorPoint,
      (fun running =>
        fun matrix =>
          ringKCodec_admissible
            (proof.piCcsInput.claimedYRing running matrix)),
      (fun lane => admissible.priorLane_lt lane),
      (fun round =>
        fun slot =>
          kCodec_admissible
            ((proof.certificate.piCcs.fe.rowRounds round).coefficients.getD
              slot.val K.zero)),
      (fun round =>
        fun slot =>
          kCodec_admissible
            ((proof.certificate.piCcs.fe.laneRounds round).coefficients.getD
              slot.val K.zero)),
      (fun round =>
        fun slot =>
          kCodec_admissible
            ((proof.certificate.piCcs.nc.rounds round).coefficients.getD
              slot.val K.zero)),
      (fun source =>
        fun matrix =>
          ringKCodec_admissible
            (proof.certificate.piCcs.output.yRing source matrix)),
      (fun source =>
        ringKCodec_admissible
          (proof.certificate.piCcs.output.yZcol source)),
      (fun coordinate =>
        ringFCodec_admissible
          (proof.certificate.piRlcChallenges coordinate)),
      (fun child =>
        payloadCodec_admissible
          (proof.certificate.piDecPayloads child)
          (admissible.piDecEvaluations_size child))⟩

private theorem list_eq_of_getD_eq
    {α : Type}
    (default : α)
    {count : Nat}
    (left right : List α)
    (leftLength : left.length = count)
    (rightLength : right.length = count)
    (valuesEqual : ∀ index : Fin count,
      left.getD index.val default = right.getD index.val default) :
    left = right := by
  apply List.ext_get
  · exact leftLength.trans rightLength.symm
  · intro index leftLt rightLt
    let typed : Fin count :=
      ⟨index, by
        rw [← leftLength]
        exact leftLt⟩
    have selected := valuesEqual typed
    rw [List.getD_eq_getElem?_getD,
      List.getElem?_eq_getElem leftLt] at selected
    rw [List.getD_eq_getElem?_getD,
      List.getElem?_eq_getElem rightLt] at selected
    exact selected

private theorem fixedPolynomial_eq_of_coefficients_eq
    {α : Type}
    {degree : Nat}
    (left right :
      Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial α degree)
    (coefficients : left.coefficients = right.coefficients) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem fixedPolynomial_heq_of_degree_eq
    {α : Type}
    {leftDegree rightDegree : Nat}
    (left :
      Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial α leftDegree)
    (right :
      Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial α rightDegree)
    (degrees : leftDegree = rightDegree)
    (coefficients : left.coefficients = right.coefficients) :
    HEq left right := by
  subst rightDegree
  exact heq_of_eq
    (fixedPolynomial_eq_of_coefficients_eq left right coefficients)

private theorem feCertificate_heq_of_input_eq
    {shape : SemanticShape}
    {leftInput rightInput : PublicInput shape}
    {domain : FlatNcDomain}
    (left : SumCheck.Fe.Certificate leftInput domain)
    (right : SumCheck.Fe.Certificate rightInput domain)
    (inputs : leftInput = rightInput)
    (rowRounds :
      ∀ round, HEq (left.rowRounds round) (right.rowRounds round))
    (laneRounds :
      ∀ round, left.laneRounds round = right.laneRounds round) :
    HEq left right := by
  subst rightInput
  apply heq_of_eq
  cases left
  cases right
  simp only at rowRounds laneRounds
  congr
  · funext round
    exact eq_of_heq (rowRounds round)
  · funext round
    exact laneRounds round

private theorem ncCertificate_eq_of_rounds
    {domain : BlockNcDomain}
    (left right : Transcript.Nc.BlockLane.Certificate domain)
    (rounds : left.rounds = right.rounds) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem piCcsCertificate_heq_of_input_eq
    {shape : SemanticShape}
    {leftInput rightInput : PublicInput shape}
    (left :
      Protocol.BlockLane.Certificate leftInput Domains)
    (right :
      Protocol.BlockLane.Certificate rightInput Domains)
    (inputs : leftInput = rightInput)
    (fe : HEq left.fe right.fe)
    (nc : left.nc = right.nc)
    (output : left.output = right.output) :
    HEq left right := by
  subst rightInput
  apply heq_of_eq
  cases left
  cases right
  simp only at fe nc output
  cases eq_of_heq fe
  cases nc
  cases output
  rfl

private theorem outerCertificate_heq_of_input_eq
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {leftInput rightInput : PublicInput shape}
    (left :
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Certificate
        (arity := FixedActive.arity)
        publicRingColumns publicFits verifierRows leftInput)
    (right :
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Certificate
        (arity := FixedActive.arity)
        publicRingColumns publicFits verifierRows rightInput)
    (inputs : leftInput = rightInput)
    (piCcs : HEq left.piCcs right.piCcs)
    (piRlcChallenges :
      left.piRlcChallenges = right.piRlcChallenges)
    (piDecPayloads :
      left.piDecPayloads = right.piDecPayloads) :
    HEq left right := by
  subst rightInput
  apply heq_of_eq
  cases left
  cases right
  simp only at piCcs piRlcChallenges piDecPayloads
  cases eq_of_heq piCcs
  cases piRlcChallenges
  cases piDecPayloads
  rfl

private theorem selectedProof_eq_of_parts
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (left right :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (piCcsInput : left.piCcsInput = right.piCcsInput)
    (priorState : left.priorState = right.priorState)
    (certificate : HEq left.certificate right.certificate) :
    left = right := by
  cases left
  cases right
  simp only at piCcsInput priorState certificate
  cases piCcsInput
  cases priorState
  cases eq_of_heq certificate
  rfl

private theorem feRow_coefficients_length
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount}
    {priorAbsorbed : Nat}
    {proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows}
    (admissible :
      ProofAdmissible constraintPolynomial priorAbsorbed proof)
    (round : Fin shape.rowVariables) :
    (proof.certificate.piCcs.fe.rowRounds round).coefficients.length =
      rowSlotCount constraintPolynomial := by
  rw [(proof.certificate.piCcs.fe.rowRounds round).coefficients_length]
  unfold rowSlotCount SumCheck.Fe.Drow
  unfold Polynomial.Fe.rowSumcheckDegreeBound
  rw [admissible.constraintPolynomial_eq]
  rfl

private theorem feLane_coefficients_length
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (round : Fin Domains.fe.laneVariables) :
    (proof.certificate.piCcs.fe.laneRounds round).coefficients.length = 3 := by
  simpa [Polynomial.Fe.laneSumcheckDegreeBound] using
    (proof.certificate.piCcs.fe.laneRounds round).coefficients_length

private theorem nc_coefficients_length
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (round :
      Fin (Domains.nc.blockVariables + Domains.nc.laneVariables)) :
    (proof.certificate.piCcs.nc.rounds round).coefficients.length = 5 := by
  simpa [Polynomial.Nc.Degree.ncSumcheckDegreeBound] using
    (proof.certificate.piCcs.nc.rounds round).coefficients_length

/-- The dynamic coordinate projection loses no admissible proof data. -/
theorem proofCoordinates_injective
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount}
    {priorAbsorbed : Nat}
    {left right :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows}
    (leftAdmissible :
      ProofAdmissible constraintPolynomial priorAbsorbed left)
    (rightAdmissible :
      ProofAdmissible constraintPolynomial priorAbsorbed right)
    (coordinates :
      proofCoordinates constraintPolynomial left =
        proofCoordinates constraintPolynomial right) :
    left = right := by
  have priorPoint :
      left.piCcsInput.priorPoint = right.piCcsInput.priorPoint :=
    congrArg Coordinates.priorPoint coordinates
  have claimedYRing :
      left.piCcsInput.claimedYRing = right.piCcsInput.claimedYRing :=
    congrArg Coordinates.claimedYRing coordinates
  have piCcsInput : left.piCcsInput = right.piCcsInput := by
    apply PublicInput.ext
    · exact leftAdmissible.constraintPolynomial_eq.trans
        rightAdmissible.constraintPolynomial_eq.symm
    · exact priorPoint
    · exact claimedYRing
  have priorState : left.priorState = right.priorState :=
    leftAdmissible.priorState_eq.trans
      rightAdmissible.priorState_eq.symm
  have feRowCoordinates :
      (proofCoordinates constraintPolynomial left).feRow =
        (proofCoordinates constraintPolynomial right).feRow :=
    congrArg Coordinates.feRow coordinates
  have feLaneCoordinates :
      (proofCoordinates constraintPolynomial left).feLane =
        (proofCoordinates constraintPolynomial right).feLane :=
    congrArg Coordinates.feLane coordinates
  have ncCoordinates :
      (proofCoordinates constraintPolynomial left).nc =
        (proofCoordinates constraintPolynomial right).nc :=
    congrArg Coordinates.nc coordinates
  have fe :
      HEq left.certificate.piCcs.fe right.certificate.piCcs.fe := by
    apply feCertificate_heq_of_input_eq _ _ piCcsInput
    · intro round
      have coefficients :
          (left.certificate.piCcs.fe.rowRounds round).coefficients =
            (right.certificate.piCcs.fe.rowRounds round).coefficients := by
        apply list_eq_of_getD_eq K.zero _ _
          (feRow_coefficients_length leftAdmissible round)
          (feRow_coefficients_length rightAdmissible round)
        intro slot
        exact congrFun (congrFun feRowCoordinates round) slot
      exact fixedPolynomial_heq_of_degree_eq _ _
        (congrArg SumCheck.Fe.Drow piCcsInput) coefficients
    · intro round
      apply fixedPolynomial_eq_of_coefficients_eq
      apply list_eq_of_getD_eq K.zero _ _
        (feLane_coefficients_length left round)
        (feLane_coefficients_length right round)
      intro slot
      exact congrFun (congrFun feLaneCoordinates round) slot
  have nc : left.certificate.piCcs.nc = right.certificate.piCcs.nc := by
    apply ncCertificate_eq_of_rounds
    funext round
    apply fixedPolynomial_eq_of_coefficients_eq
    apply list_eq_of_getD_eq K.zero _ _
      (nc_coefficients_length left round)
      (nc_coefficients_length right round)
    intro slot
    exact congrFun (congrFun ncCoordinates round) slot
  have output :
      left.certificate.piCcs.output = right.certificate.piCcs.output := by
    apply Claims.ext
    · intro source matrix lane
      exact congrFun
        (congrFun
          (congrFun
            (congrArg Coordinates.outputYRing coordinates) source)
          matrix)
        lane
    · intro source lane
      exact congrFun
        (congrFun
          (congrArg Coordinates.outputYZcol coordinates) source)
        lane
  have piCcs :
      HEq left.certificate.piCcs right.certificate.piCcs :=
    piCcsCertificate_heq_of_input_eq _ _ piCcsInput fe nc output
  have piRlcChallenges :
      left.certificate.piRlcChallenges =
        right.certificate.piRlcChallenges :=
    congrArg Coordinates.piRlcChallenges coordinates
  have piDecPayloads :
      left.certificate.piDecPayloads =
        right.certificate.piDecPayloads :=
    congrArg Coordinates.piDecPayloads coordinates
  have certificate :
      HEq left.certificate right.certificate :=
    outerCertificate_heq_of_input_eq _ _ piCcsInput piCcs
      piRlcChallenges piDecPayloads
  exact selectedProof_eq_of_parts left right piCcsInput priorState certificate

/-- Canonical codec for the complete selected proof.  The encoding contains
every dynamic prover coordinate and omits only setup-fixed fields proved by
`ProofAdmissible`. -/
noncomputable def proofCodec
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    Codec
      (SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) :=
  Codec.pullbackOn
    (coordinatesCodec shape constraintPolynomial
      publicRingColumns verifierRows publicFits)
    (ProofAdmissible constraintPolynomial priorAbsorbed)
    (proofCoordinates constraintPolynomial)
    (fun _ admissible => proofCoordinates_admissible admissible)
    (fun leftAdmissible rightAdmissible equal =>
      proofCoordinates_injective
        leftAdmissible rightAdmissible equal)

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofCodec
