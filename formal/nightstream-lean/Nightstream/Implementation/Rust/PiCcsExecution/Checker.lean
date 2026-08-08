import Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentity
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial.Semantics
import Nightstream.Implementation.Rust.PiCcsExecution.CachedDuplex
import Nightstream.Implementation.Rust.PiCcsExecution.Receipt
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConstraintPolynomialLift
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialFixedWidth

/-!
Independent Lean replay of one selected Rust `Pi_CCS` execution.

Owns: binding to an externally selected relation identifier, exact rectangular
profile and sparse-polynomial checks, transcript replay, output projection, and
the reduction from executable receipt acceptance to the paper fixed-width
SumCheck relation.

Does not own: authority for the expected relation identifier, proof that a
production matrix artifact has that identifier, universal refinement of Rust,
or Fiat--Shamir security.

Emits constraints: no.

Assurance tier: model-level for `checkReceipt_sound`. A concrete generated
receipt becomes artifact-checked only when a Rust drift test pins its bytes.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Rust.PiCcsExecution

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core

namespace SelectedPolynomial

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial

/-- Exact paper shape selected for the production rectangular relation. -/
def shape : Shape :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity.shape

/-- The exact 66-term polynomial with the ignored identity-matrix input,
lifted coefficient-by-coefficient into the quadratic extension. -/
def polynomial : ConstraintPolynomial K shape.matrixCount :=
  ConstraintPolynomialLift.liftConstraintPolynomial K.embed
    (ConstraintPolynomialPrepend.prependIgnoredVariable Semantics.polynomial)

/-- Rust's field encoding for one original 13-variable sparse term. The first
zero is the imaginary coefficient limb; the second is the prepended identity
matrix exponent. -/
def encodeTerm (term : Monomial F 13) : List Nat :=
  [term.coefficient.val, 0, 0] ++
    (canonicalFinIndices 13).map term.exponents

/-- Exact compact statement absorption emitted by Rust for this polynomial. -/
def statementFields : List Nat :=
  [41, 24, 1, 14, 14, 54, 8, 66] ++
    Semantics.terms.flatMap encodeTerm ++ [47]

end SelectedPolynomial

/-- Every value in a base-field list has a canonical representative. -/
def canonicalFields (values : List Nat) : Bool :=
  values.all fun value => decide (value < goldilocksModulus)

/-- Every quadratic-extension value has two canonical representatives. -/
def canonicalKValues (values : List RawK) : Bool :=
  values.all RawK.wellFormed

/-- Exact fixed public-profile prefix before the four-field relation ID. -/
def selectedPublicPrefix : List Nat :=
  [40, 2, 24, 1, 14, 14, 54, 11437038, 16777216, 9,
    14944219, 11437038]

/-- Check the compact public absorption, including the copy of the relation
identifier, the compact-binding tag, and the mandatory running handle. The
four public-digest and four handle fields remain statement data. -/
def publicProfileCheck (statement : PiCcsCanonicalStatement) : Bool :=
  decide (statement.publicFields.length = 27) &&
    decide (statement.publicFields.take 12 = selectedPublicPrefix) &&
    decide ((statement.publicFields.drop 12).take 4 = statement.relationId) &&
    decide (statement.publicFields.getD 16 0 = 47) &&
    decide (statement.publicFields.getD 21 0 = 14) &&
    decide (statement.publicFields.getD 22 0 = 1) &&
    canonicalFields statement.publicFields

/-- All fail-closed structural checks before dependent paper values are built. -/
def receiptShapeCheck
    (expectedRelationId : List Nat)
    (statement : PiCcsCanonicalStatement)
    (rustProof : PiCcsExecutionProof) : Bool :=
  decide (expectedRelationId.length = 4) &&
    canonicalFields expectedRelationId &&
    decide (statement.relationId = expectedRelationId) &&
    decide (statement.relationId.length = 4) &&
    canonicalFields statement.relationId &&
    decide (statement.transcriptState.length = 8) &&
    canonicalFields statement.transcriptState &&
    decide (statement.transcriptAbsorbed <=
      Poseidon2Sponge.rate) &&
    publicProfileCheck statement &&
    decide (statement.piCcsStatementFields =
      SelectedPolynomial.statementFields) &&
    canonicalFields statement.piCcsStatementFields &&
    decide (statement.priorPoint.length = 24) &&
    canonicalKValues statement.priorPoint &&
    decide (statement.claimedCoefficients.length = 10584) &&
    canonicalKValues statement.claimedCoefficients &&
    proofBytesWellFormed rustProof.proofBytes &&
    decide (rustProof.fullOutput.length = 11340) &&
    canonicalKValues rustProof.fullOutput

/-- Rebuild the exact captured value-level transcript state. -/
def initialTranscript (statement : PiCcsCanonicalStatement) :
    Poseidon2Duplex.State where
  lanes := fun lane => statement.transcriptState.getD lane.val 0
  absorbed := statement.transcriptAbsorbed

/-- Absorb raw canonical fields with the selected overwrite duplex. -/
def absorbFields (values : List Nat) (state : Poseidon2Duplex.State) :
    Poseidon2Duplex.State :=
  CachedDuplex.absorbList Poseidon2CanonicalConstants.selected values state

/-- Cached receipt absorption is the canonical overwrite duplex. -/
theorem absorbFields_eq_reference
    (values : List Nat) (state : Poseidon2Duplex.State) :
    absorbFields values state =
      Poseidon2Duplex.absorbList Poseidon2CanonicalConstants.selected
        values state :=
  CachedDuplex.absorbList_eq_reference
    Poseidon2CanonicalConstants.selected values state

/-- Read the first two freshly permuted lanes as the concrete paper extension
carrier. -/
def squeezeK (state : Poseidon2Duplex.State) : K × Poseidon2Duplex.State :=
  let next := CachedDuplex.gate Poseidon2CanonicalConstants.selected state
  ({ c0 := ⟨next.lanes ⟨0, by decide⟩ % goldilocksModulus,
        Nat.mod_lt _ (by decide)⟩
     c1 := ⟨next.lanes ⟨1, by decide⟩ % goldilocksModulus,
        Nat.mod_lt _ (by decide)⟩ }, next)

/-- Cached challenge squeezing is the canonical pre-squeeze duplex gate. -/
theorem squeezeK_eq_reference (state : Poseidon2Duplex.State) :
    squeezeK state =
      let next :=
        Poseidon2Duplex.gate Poseidon2CanonicalConstants.selected state
      ({ c0 := ⟨next.lanes ⟨0, by decide⟩ % goldilocksModulus,
            Nat.mod_lt _ (by decide)⟩
         c1 := ⟨next.lanes ⟨1, by decide⟩ % goldilocksModulus,
            Nat.mod_lt _ (by decide)⟩ }, next) := by
  let project : Poseidon2Duplex.State -> K × Poseidon2Duplex.State :=
    fun next =>
      ({ c0 := ⟨next.lanes ⟨0, by decide⟩ % goldilocksModulus,
            Nat.mod_lt _ (by decide)⟩
         c1 := ⟨next.lanes ⟨1, by decide⟩ % goldilocksModulus,
            Nat.mod_lt _ (by decide)⟩ }, next)
  change project (CachedDuplex.gate Poseidon2CanonicalConstants.selected state) =
    project (Poseidon2Duplex.gate Poseidon2CanonicalConstants.selected state)
  exact congrArg project
    (CachedDuplex.gate_eq_reference
      Poseidon2CanonicalConstants.selected state)

/-- Absorb an indexed domain tag and squeeze one extension value. -/
def squeezeIndexed (tag index : Nat) (state : Poseidon2Duplex.State) :
    K × Poseidon2Duplex.State :=
  squeezeK (absorbFields [tag, index] state)

/-- Absorb a non-indexed domain tag and squeeze one extension value. -/
def squeezeSingle (tag : Nat) (state : Poseidon2Duplex.State) :
    K × Poseidon2Duplex.State :=
  squeezeK (absorbFields [tag] state)

/-- Derive a consecutive indexed challenge family, preserving state order. -/
def deriveIndexed (tag start : Nat) : Nat -> Poseidon2Duplex.State ->
    List K × Poseidon2Duplex.State
  | 0, state => ([], state)
  | count + 1, state =>
      let sampled := squeezeIndexed tag start state
      let rest := deriveIndexed tag (start + 1) count sampled.2
      (sampled.1 :: rest.1, rest.2)

/-- Exact low/high field encoding of one decoded extension value. -/
def encodeK (value : K) : List Nat := [value.c0.val, value.c1.val]

/-- Exact Rust round-message absorption. -/
def roundFields (index : Nat) (message : SumCheck.Finite.Message K) :
    List Nat :=
  [45, index, message.coefficients.length] ++
    message.coefficients.flatMap encodeK

/-- Replay all round challenges in message order. -/
def deriveRoundChallenges : Nat -> List (SumCheck.Finite.Message K) ->
    Poseidon2Duplex.State -> List K × Poseidon2Duplex.State
  | _, [], state => ([], state)
  | index, message :: messages, state =>
      let afterMessage := absorbFields (roundFields index message) state
      let sampled := squeezeIndexed 46 index afterMessage
      let rest := deriveRoundChallenges (index + 1) messages sampled.2
      (sampled.1 :: rest.1, rest.2)

/-- Fail-closed dependent point construction. -/
def cubePoint? {variables : Nat} (coordinates : List K) :
    Option (CubePoint K variables) :=
  if dimension : coordinates.length = variables then
    some { coordinates := coordinates, dimension := dimension }
  else
    none

/-- Read one canonical raw extension value by flat index. Arrays keep the
10,584-coordinate initial and terminal folds linear rather than quadratic. -/
def rawValueAt (values : Array RawK) (index : Nat) : K :=
  (values.getD index default).decode

/-- The verifier input uses only the selected polynomial and the two public
claim families carried in the canonical statement. -/
def verifierInput
    (claimedCoefficients : Array RawK)
    (priorPoint : CubePoint K SelectedPolynomial.shape.cubeVariables) :
    VerifierInput K SelectedPolynomial.shape where
  constraintPolynomial := SelectedPolynomial.polynomial
  priorPoint := priorPoint
  claimedCoefficient := fun coordinate =>
    rawValueAt claimedCoefficients
      coordinate.localGammaExponent

/-- One target term with a zero-coefficient fast path. The paper target is
unchanged: multiplication by zero proves that the skipped power has no effect. -/
def zeroAwareTargetTerm
    (input : VerifierInput K SelectedPolynomial.shape)
    (gamma : K)
    (coordinate : CarriedCoordinate SelectedPolynomial.shape) : K :=
  let coefficient := input.claimedCoefficient coordinate
  if coefficient = ConcreteCarrier.extensionOps.zero then
    ConcreteCarrier.extensionOps.zero
  else
    TargetPolynomial.term ConcreteCarrier.extensionOps.toOps
      input.targetCoefficients .coherentAbsolute gamma coordinate

/-- Linear executable form of the paper target for sparse receipts. It avoids
computing a power for a coefficient that is exactly zero. -/
def zeroAwareInitial
    (input : VerifierInput K SelectedPolynomial.shape)
    (gamma : K) : K :=
  ((canonicalCarriedCoordinates SelectedPolynomial.shape).map
      (zeroAwareTargetTerm input gamma)).foldr
    ConcreteCarrier.extensionOps.add ConcreteCarrier.extensionOps.zero

private theorem zeroAwareTerms_eq_paperTerms
    (input : VerifierInput K SelectedPolynomial.shape)
    (gamma : K) :
    forall coordinates : List (CarriedCoordinate SelectedPolynomial.shape),
      (coordinates.map (zeroAwareTargetTerm input gamma)).foldr
          ConcreteCarrier.extensionOps.add
          ConcreteCarrier.extensionOps.zero =
        (coordinates.map fun coordinate =>
          TargetPolynomial.term ConcreteCarrier.extensionOps.toOps
            input.targetCoefficients .coherentAbsolute gamma coordinate).foldr
          ConcreteCarrier.extensionOps.add
          ConcreteCarrier.extensionOps.zero
  | [] => rfl
  | coordinate :: coordinates => by
      simp only [List.map_cons, List.foldr_cons]
      rw [zeroAwareTerms_eq_paperTerms input gamma coordinates]
      by_cases coefficientZero :
          input.claimedCoefficient coordinate =
            ConcreteCarrier.extensionOps.zero
      · simp only [zeroAwareTargetTerm, coefficientZero, ↓reduceIte,
          TargetPolynomial.term, VerifierInput.targetCoefficients]
        rw [ConcreteCarrier.extensionLaws.mul_zero]
      · simp only [zeroAwareTargetTerm, coefficientZero, ↓reduceIte]

/-- The zero-aware executable target is exactly the selected paper target. -/
theorem zeroAwareInitial_eq_paperInitial
    (input : VerifierInput K SelectedPolynomial.shape)
    (gamma : K) :
    zeroAwareInitial input gamma =
      input.initial ConcreteCarrier.extensionOps gamma := by
  change
    ((canonicalCarriedCoordinates SelectedPolynomial.shape).map
      (zeroAwareTargetTerm input gamma)).foldr
        ConcreteCarrier.extensionOps.add
        ConcreteCarrier.extensionOps.zero =
      ((canonicalCarriedCoordinates SelectedPolynomial.shape).map fun coordinate =>
        TargetPolynomial.term ConcreteCarrier.extensionOps.toOps
          input.targetCoefficients .coherentAbsolute gamma coordinate).foldr
        ConcreteCarrier.extensionOps.add
        ConcreteCarrier.extensionOps.zero
  exact zeroAwareTerms_eq_paperTerms input gamma _

/-- Flat source/matrix/coefficient index in Rust's complete `y'` payload. -/
def fullOutputIndex (source matrix coefficient : Nat) : Nat :=
  source * 14 * 54 + matrix * 54 + coefficient

/-- Project the complete Rust output family to the three values read by the
paper verifier. No output field outside these projections becomes authority. -/
def outputMessage (fullOutput : Array RawK) :
    OutputMessage K SelectedPolynomial.shape where
  freshMatrixImage := fun source matrix =>
    rawValueAt fullOutput
      (fullOutputIndex source.val matrix.val 0)
  sourceAssignment := fun source =>
    rawValueAt fullOutput
      (fullOutputIndex source.val 0 0)
  carriedImage := fun coordinate =>
    rawValueAt fullOutput
      (fullOutputIndex (1 + coordinate.running.val)
        coordinate.matrix.val coordinate.coefficient.val)

/-- One gamma-weighted output with a zero-value fast path. -/
def zeroAwareGammaTerm (gamma : K) (exponent : Nat) (value : K) : K :=
  if value = ConcreteCarrier.extensionOps.zero then
    ConcreteCarrier.extensionOps.zero
  else
    SignedJointIdentity.gammaTerm ConcreteCarrier.extensionOps
      gamma exponent value

/-- Skipping the power for a zero value preserves the paper gamma term. -/
theorem zeroAwareGammaTerm_eq_paperTerm
    (gamma : K) (exponent : Nat) (value : K) :
    zeroAwareGammaTerm gamma exponent value =
      SignedJointIdentity.gammaTerm ConcreteCarrier.extensionOps
        gamma exponent value := by
  by_cases valueZero : value = ConcreteCarrier.extensionOps.zero
  · simp only [zeroAwareGammaTerm, valueZero, ↓reduceIte,
      SignedJointIdentity.gammaTerm]
    rw [ConcreteCarrier.extensionLaws.mul_zero]
  · simp only [zeroAwareGammaTerm, valueZero, ↓reduceIte]

/-- Sparse executable form of the paper carried-output term. -/
def zeroAwareCarriedAtMessage
    (input : VerifierInput K SelectedPolynomial.shape)
    (gamma : K)
    (point : CubePoint K SelectedPolynomial.shape.cubeVariables)
    (message : OutputMessage K SelectedPolynomial.shape) : K :=
  ConcreteCarrier.extensionOps.mul
    (SumCheckTruthPath.pointEquality ConcreteCarrier.extensionOps
      point input.priorPoint) <|
    SignedJointIdentity.sumMap ConcreteCarrier.extensionOps
      (canonicalCarriedCoordinates SelectedPolynomial.shape) fun coordinate =>
        zeroAwareGammaTerm gamma coordinate.localGammaExponent
          (message.carriedImage coordinate)

/-- The sparse carried-output evaluator is exactly the paper evaluator. -/
theorem zeroAwareCarriedAtMessage_eq_paper
    (input : VerifierInput K SelectedPolynomial.shape)
    (gamma : K)
    (point : CubePoint K SelectedPolynomial.shape.cubeVariables)
    (message : OutputMessage K SelectedPolynomial.shape) :
    zeroAwareCarriedAtMessage input gamma point message =
      ProtocolPolynomial.carriedAtMessage ConcreteCarrier.extensionOps
        input gamma point message := by
  unfold zeroAwareCarriedAtMessage ProtocolPolynomial.carriedAtMessage
  congr 2
  funext coordinate
  exact zeroAwareGammaTerm_eq_paperTerm gamma
    coordinate.localGammaExponent (message.carriedImage coordinate)

/-- Complete selected paper terminal with the sparse carried-output evaluator. -/
def zeroAwareTerminalFromMessage
    (input : VerifierInput K SelectedPolynomial.shape)
    (alpha : CubePoint K SelectedPolynomial.shape.cubeVariables)
    (gamma : K)
    (point : CubePoint K SelectedPolynomial.shape.cubeVariables)
    (message : OutputMessage K SelectedPolynomial.shape) : K :=
  ConcreteCarrier.extensionOps.add
    (ConcreteCarrier.extensionOps.mul
      (SumCheckTruthPath.pointEquality ConcreteCarrier.extensionOps point alpha)
      (ConcreteCarrier.extensionOps.add
        (ProtocolPolynomial.ccsAtMessage ConcreteCarrier.extensionOps
          input gamma message)
        (SignedJointIdentity.gammaTerm ConcreteCarrier.extensionOps gamma
          SelectedPolynomial.shape.freshCount
          (ProtocolPolynomial.normAtMessage ConcreteCarrier.extensionOps
            gamma message))))
    (SignedJointIdentity.gammaTerm ConcreteCarrier.extensionOps gamma
      SelectedPolynomial.shape.carriedEvaluationOffset
      (zeroAwareCarriedAtMessage input gamma point message))

/-- The sparse executable terminal is exactly the selected paper terminal. -/
theorem zeroAwareTerminalFromMessage_eq_paper
    (input : VerifierInput K SelectedPolynomial.shape)
    (alpha : CubePoint K SelectedPolynomial.shape.cubeVariables)
    (gamma : K)
    (point : CubePoint K SelectedPolynomial.shape.cubeVariables)
    (message : OutputMessage K SelectedPolynomial.shape) :
    zeroAwareTerminalFromMessage input alpha gamma point message =
      ProtocolPolynomial.terminalFromMessage ConcreteCarrier.extensionOps
        input alpha gamma point message := by
  unfold zeroAwareTerminalFromMessage ProtocolPolynomial.terminalFromMessage
  rw [zeroAwareCarriedAtMessage_eq_paper]

/-- Fully decoded values consumed by the existing paper verifier. -/
structure DecodedReceipt where
  input : VerifierInput K SelectedPolynomial.shape
  alpha : CubePoint K SelectedPolynomial.shape.cubeVariables
  gamma : K
  roundPoint : CubePoint K SelectedPolynomial.shape.cubeVariables
  message : OutputMessage K SelectedPolynomial.shape
  certificate : SumCheck.Finite.Certificate K

/-- Parse and replay one receipt. Every challenge is recomputed from the
captured state and exact canonical fields; no challenge is receipt data. -/
def decodeReceipt
    (expectedRelationId : List Nat)
    (statement : PiCcsCanonicalStatement)
    (rustProof : PiCcsExecutionProof) : Option DecodedReceipt :=
  if receiptShapeCheck expectedRelationId statement rustProof then
    let certificate := proofCertificate rustProof.proofBytes
    let afterPublic := absorbFields statement.publicFields
      (initialTranscript statement)
    let afterStatement := absorbFields statement.piCcsStatementFields
      afterPublic
    let alphaResult := deriveIndexed 42 0 24 afterStatement
    let gammaResult := squeezeSingle 43 alphaResult.2
    let roundResult := deriveRoundChallenges 0 certificate.rounds
      gammaResult.2
    let claimedCoefficients := statement.claimedCoefficients.toArray
    let fullOutput := rustProof.fullOutput.toArray
    match cubePoint? (variables := SelectedPolynomial.shape.cubeVariables)
          (statement.priorPoint.map RawK.decode),
        cubePoint? (variables := SelectedPolynomial.shape.cubeVariables)
          alphaResult.1,
        cubePoint? (variables := SelectedPolynomial.shape.cubeVariables)
          roundResult.1 with
    | some priorPoint, some alpha, some roundPoint =>
        some {
          input := verifierInput claimedCoefficients priorPoint
          alpha := alpha
          gamma := gammaResult.1
          roundPoint := roundPoint
          message := outputMessage fullOutput
          certificate := certificate }
    | _, _, _ => none
  else
    none

/-- Independent executable receipt checker for the selected one-SumCheck
paper relation. -/
def checkReceipt
    (expectedRelationId : List Nat)
    (statement : PiCcsCanonicalStatement)
    (rustProof : PiCcsExecutionProof) : Bool :=
  match decodeReceipt expectedRelationId statement rustProof with
  | none => false
  | some decoded =>
      SumCheck.Finite.FixedPhase.RawCertificate.check
        ConcreteCarrier.extensionOps.toOps 9
        (zeroAwareInitial decoded.input decoded.gamma)
        decoded.roundPoint.coordinates
        (zeroAwareTerminalFromMessage decoded.input decoded.alpha
          decoded.gamma decoded.roundPoint decoded.message)
        decoded.certificate

/-- The executable receipt checker calls the exact paper checker after a
proved sparse-target optimization. -/
theorem checkReceipt_eq_paperCheck
    (decoded : DecodedReceipt) :
    SumCheck.Finite.FixedPhase.RawCertificate.check
        ConcreteCarrier.extensionOps.toOps 9
        (zeroAwareInitial decoded.input decoded.gamma)
        decoded.roundPoint.coordinates
        (zeroAwareTerminalFromMessage decoded.input decoded.alpha
          decoded.gamma decoded.roundPoint decoded.message)
        decoded.certificate =
      ProtocolPolynomial.FixedWidth.check ConcreteCarrier.extensionOps 9
        decoded.input decoded.alpha decoded.gamma decoded.roundPoint
        decoded.message decoded.certificate := by
  rw [zeroAwareInitial_eq_paperInitial]
  rw [zeroAwareTerminalFromMessage_eq_paper]
  rfl

namespace PaperPiCCS

/-- Logical paper acceptance exposed by an accepted decoded receipt. It names
the exact decoded fixed-width certificate and complete claimed chain. -/
def Accepts
    (expectedRelationId : List Nat)
    (statement : PiCcsCanonicalStatement)
    (rustProof : PiCcsExecutionProof) : Prop :=
  exists decoded : DecodedReceipt,
    decodeReceipt expectedRelationId statement rustProof = some decoded /\
      exists fixed : SumCheck.Finite.FixedPhase.Certificate K 9,
        SumCheck.Finite.FixedPhase.RawCertificate.decode 9
            decoded.certificate = some fixed /\
          SumCheck.Finite.FixedPhase.Chain
            ConcreteCarrier.extensionOps.toOps
            (decoded.input.initial ConcreteCarrier.extensionOps decoded.gamma)
            fixed.rounds decoded.roundPoint.coordinates
            (ProtocolPolynomial.terminalFromMessage
              ConcreteCarrier.extensionOps decoded.input decoded.alpha
              decoded.gamma decoded.roundPoint decoded.message)

end PaperPiCCS

/-- An accepted receipt supplies the exact logical paper `Pi_CCS` chain. -/
theorem checkReceipt_sound
    (expectedRelationId : List Nat)
    (statement : PiCcsCanonicalStatement)
    (rustProof : PiCcsExecutionProof) :
    checkReceipt expectedRelationId statement rustProof = true ->
      PaperPiCCS.Accepts expectedRelationId statement rustProof := by
  intro checked
  unfold checkReceipt at checked
  cases decodedEq : decodeReceipt expectedRelationId statement rustProof with
  | none => simp [decodedEq] at checked
  | some decoded =>
      have optimizedChecked :
          SumCheck.Finite.FixedPhase.RawCertificate.check
              ConcreteCarrier.extensionOps.toOps 9
              (zeroAwareInitial decoded.input decoded.gamma)
              decoded.roundPoint.coordinates
              (zeroAwareTerminalFromMessage decoded.input decoded.alpha
                decoded.gamma decoded.roundPoint decoded.message)
              decoded.certificate = true := by
        simpa [decodedEq] using checked
      rw [checkReceipt_eq_paperCheck] at optimizedChecked
      have accepted :=
        (ProtocolPolynomial.FixedWidth.check_eq_true_iff
          ConcreteCarrier.extensionOps 9 decoded.input decoded.alpha
          decoded.gamma decoded.roundPoint decoded.message
          decoded.certificate).1 optimizedChecked
      exact ⟨decoded, decodedEq, accepted⟩

end Nightstream.Implementation.Rust.PiCcsExecution
