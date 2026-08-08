import Nightstream.Implementation.Rust.NifsProductionGolden.FixedRelation
import Nightstream.Implementation.Rust.NifsProductionGolden.PiCcsReplay
import Nightstream.Implementation.Rust.NifsProductionGolden.Receipt
import Nightstream.Implementation.Rust.PiCcsExecution.Checker
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConstraintPolynomialLift
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialFixedWidth

/-! Independent replay of the exact six-round production `Pi_CCS` verifier. -/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Rust.NifsProductionGolden.PiCcsChecker

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial
open Nightstream.Implementation.Rust.NifsProductionGolden
open Nightstream.Implementation.Rust.PiCcsExecution

def shape : Shape where
  cubeVariables := 6
  freshCount := 1
  runningCount := 0
  matrixCount := 4
  coefficientCount := 54

def polynomial : ConstraintPolynomial K shape.matrixCount :=
  ConstraintPolynomialLift.liftConstraintPolynomial K.embed
    FixedRelation.polynomial

def expectedRelationMatrices : List Nat :=
  ([0, 1] ++ List.replicate 52 0) ++
    ([1] ++ List.replicate 53 0) ++
    ([0, 1] ++ List.replicate 52 0)

def expectedFixtureAssignment : List Nat :=
  [1, 1] ++ List.replicate 52 0

def publicPrefix : List Nat :=
  [40, 2, 6, 1, 0, 4, 54, 54, 64, 4, 1, 54]

def statementFields : List Nat :=
  [41, 6, 1, 0, 4, 54, 2, 2,
    1, 0, 0, 1, 1, 0,
    goldilocksModulus - 1, 0, 0, 0, 0, 1,
    47]

def publicProfileCheck (statement : PiCcsCanonicalStatement) : Bool :=
  decide (statement.publicFields.length = 27) &&
    decide (statement.publicFields.take 12 = publicPrefix) &&
    decide ((statement.publicFields.drop 12).take 4 = statement.relationId) &&
    decide (statement.publicFields.getD 16 0 = 47) &&
    decide (statement.publicFields.getD 21 0 = 0) &&
    decide (statement.publicFields.getD 22 0 = 1) &&
    canonicalFields statement.publicFields

def roundCount : Nat := 6

def roundCoefficientCount : Nat := 5

def proofByteCount : Nat :=
  32 + roundCount * roundCoefficientCount * 16

def coefficientOffset (round coefficient : Nat) : Nat :=
  32 + (round * roundCoefficientCount + coefficient) * 16

def proofCoefficient (bytes : Array Nat) (round coefficient : Nat) : RawK :=
  let offset := coefficientOffset round coefficient
  { low := readU64LE bytes offset
    high := readU64LE bytes (offset + 8) }

def proofCoefficientsWellFormed (bytes : Array Nat) : Bool :=
  (List.range roundCount).all fun round =>
    (List.range roundCoefficientCount).all fun coefficient =>
      (proofCoefficient bytes round coefficient).wellFormed

def proofBytesWellFormed (bytes : List Nat) : Bool :=
  let bytesArray := bytes.toArray
  decide (bytes.length = proofByteCount) &&
    bytes.all (fun byte => decide (byte < 256)) &&
    decide (readU64LE bytesArray 0 = proofTag) &&
    decide (readU64LE bytesArray 8 = proofVersion) &&
    decide (readU64LE bytesArray 16 = roundCount) &&
    decide (readU64LE bytesArray 24 = roundCoefficientCount) &&
    proofCoefficientsWellFormed bytesArray

def proofCertificate (bytes : List Nat) : SumCheck.Finite.Certificate K :=
  let bytesArray := bytes.toArray
  { rounds := (List.range roundCount).map fun round =>
      { coefficients := (List.range roundCoefficientCount).map fun coefficient =>
          (proofCoefficient bytesArray round coefficient).decode } }

def receiptShapeCheck (receipt : ProductionReceipt) : Bool :=
  NifsProductionGolden.relationShapeCheck receipt &&
    NifsProductionGolden.poseidonTraceShapeCheck receipt &&
    decide (receipt.relationMatrices = expectedRelationMatrices) &&
    decide (receipt.fixtureAssignment = expectedFixtureAssignment) &&
    decide (receipt.piCcsStatement.transcriptState.length = 8) &&
    canonicalFields receipt.piCcsStatement.transcriptState &&
    decide (receipt.piCcsStatement.transcriptAbsorbed <= 4) &&
    publicProfileCheck receipt.piCcsStatement &&
    decide (receipt.piCcsStatement.piCcsStatementFields = statementFields) &&
    canonicalFields receipt.piCcsStatement.piCcsStatementFields &&
    decide (receipt.piCcsStatement.priorPoint = []) &&
    decide (receipt.piCcsStatement.claimedCoefficients = []) &&
    proofBytesWellFormed receipt.piCcsProof.proofBytes &&
    decide (receipt.piCcsProof.fullOutput.length = 4 * 54) &&
    canonicalKValues receipt.piCcsProof.fullOutput

def zeroPoint : CubePoint K shape.cubeVariables where
  coordinates := List.replicate 6 K.zero
  dimension := by simp [shape]

def verifierInput : VerifierInput K shape where
  constraintPolynomial := polynomial
  priorPoint := zeroPoint
  claimedCoefficient := fun coordinate => Fin.elim0 coordinate.running

def fullOutputIndex (matrix coefficient : Nat) : Nat :=
  matrix * 54 + coefficient

def outputMessage (fullOutput : Array RawK) : OutputMessage K shape where
  freshMatrixImage := fun _ matrix =>
    rawValueAt fullOutput (fullOutputIndex matrix.val 0)
  sourceAssignment := fun _ =>
    rawValueAt fullOutput (fullOutputIndex 0 0)
  carriedImage := fun coordinate => Fin.elim0 coordinate.running

structure DecodedReceipt where
  alpha : CubePoint K shape.cubeVariables
  gamma : K
  roundPoint : CubePoint K shape.cubeVariables
  message : OutputMessage K shape
  certificate : SumCheck.Finite.Certificate K
  finalTranscript :
    Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.State

def finishDecode (receipt : ProductionReceipt)
    (certificate : SumCheck.Finite.Certificate K)
    (replayed : PiCcsReplay.Result) : Option DecodedReceipt :=
  match @cubePoint? shape.cubeVariables replayed.alphaValues,
      @cubePoint? shape.cubeVariables replayed.roundValues with
  | some alpha, some roundPoint =>
      some {
        alpha := alpha
        gamma := replayed.gamma
        roundPoint := roundPoint
        message := outputMessage receipt.piCcsProof.fullOutput.toArray
        certificate := certificate
        finalTranscript := replayed.finalTranscript }
  | _, _ => none

def decodeReceipt (receipt : ProductionReceipt) : Option DecodedReceipt :=
  if receiptShapeCheck receipt then
    let certificate := proofCertificate receipt.piCcsProof.proofBytes
    finishDecode receipt certificate (PiCcsReplay.reference receipt certificate)
  else
    none

def decodeReceiptCertified (receipt : ProductionReceipt) : Option DecodedReceipt :=
  if receiptShapeCheck receipt then
    let certificate := proofCertificate receipt.piCcsProof.proofBytes
    match PiCcsReplay.replay? receipt certificate with
    | none => none
    | some replayed => finishDecode receipt certificate replayed
  else
    none

theorem decodeReceiptCertified_sound (receipt : ProductionReceipt)
    (decoded : DecodedReceipt)
    (accepted : decodeReceiptCertified receipt = some decoded) :
    decodeReceipt receipt = some decoded := by
  unfold decodeReceiptCertified at accepted
  split at accepted
  · rename_i shapeChecked
    let certificate := proofCertificate receipt.piCcsProof.proofBytes
    cases replayEq : PiCcsReplay.replay? receipt certificate with
    | none => simp [certificate, replayEq] at accepted
    | some replayed =>
      have replayedEq := PiCcsReplay.replay?_sound receipt certificate
        replayed replayEq
      unfold decodeReceipt
      rw [if_pos shapeChecked]
      dsimp only
      rw [<- replayedEq]
      simpa [certificate, replayEq] using accepted
  · contradiction

def checkReceipt (receipt : ProductionReceipt) : Bool :=
  match decodeReceiptCertified receipt with
  | none => false
  | some decoded =>
      ProtocolPolynomial.FixedWidth.check ConcreteCarrier.extensionOps 4
        verifierInput decoded.alpha decoded.gamma decoded.roundPoint
        decoded.message decoded.certificate

namespace PaperPiCCS

def Accepts (receipt : ProductionReceipt) : Prop :=
  exists decoded : DecodedReceipt,
    decodeReceipt receipt = some decoded /\
      exists fixed : SumCheck.Finite.FixedPhase.Certificate K 4,
        SumCheck.Finite.FixedPhase.RawCertificate.decode 4
            decoded.certificate = some fixed /\
          SumCheck.Finite.FixedPhase.Chain
            ConcreteCarrier.extensionOps.toOps
            (verifierInput.initial ConcreteCarrier.extensionOps decoded.gamma)
            fixed.rounds decoded.roundPoint.coordinates
            (ProtocolPolynomial.terminalFromMessage
              ConcreteCarrier.extensionOps verifierInput decoded.alpha
              decoded.gamma decoded.roundPoint decoded.message)

end PaperPiCCS

theorem checkReceipt_sound (receipt : ProductionReceipt) :
    checkReceipt receipt = true -> PaperPiCCS.Accepts receipt := by
  intro checked
  unfold checkReceipt at checked
  cases decodedCertifiedEq : decodeReceiptCertified receipt with
  | none => simp [decodedCertifiedEq] at checked
  | some decoded =>
      have decodedEq := decodeReceiptCertified_sound receipt decoded
        decodedCertifiedEq
      have accepted :=
        (ProtocolPolynomial.FixedWidth.check_eq_true_iff
          ConcreteCarrier.extensionOps 4 verifierInput decoded.alpha
          decoded.gamma decoded.roundPoint decoded.message
          decoded.certificate).1
            (by simpa [decodedCertifiedEq] using checked)
      exact ⟨decoded, decodedEq, accepted⟩

end Nightstream.Implementation.Rust.NifsProductionGolden.PiCcsChecker
