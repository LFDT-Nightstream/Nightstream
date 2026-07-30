import Nightstream.Implementation.R1CS.Canonical.KConcreteHorner
import Nightstream.Implementation.R1CS.Canonical.KSparsePolynomial
import Nightstream.Implementation.R1CS.Canonical.KStrictNorm
import Nightstream.Implementation.R1CS.Canonical.KEquality

/-!
Contract: the exact message-derived terminal of paper-joint PiCCS.

Owns the canonical row construction for:

* every fresh-source sparse CCS polynomial evaluation;
* every source strict-`b = 2` norm residual;
* the equality selectors at `alpha` and the public prior point;
* the exact gamma layout of CCS, norm, and carried-message values;
* the two final extension products; and
* binding the result to the fixed-phase SumCheck terminal.

All row and column counts are derived from the emitted programs.  The sparse
polynomial allocation depends on explicit monomial exponents, never on
declared degree metadata.  No Rust row, measured dimension, terminal fact, or
caller-supplied semantic conclusion enters the construction.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KPiCcsTerminal

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner

abbrev ConcreteK := Nightstream.SuperNeo.Concrete.K

structure Input (shape : Shape) where
  constraintPolynomial :
    CCSResidualTable.ConstraintPolynomial ConcreteK shape.matrixCount
  gamma : Carried
  alpha : Fin shape.cubeVariables → Carried
  point : Fin shape.cubeVariables → Carried
  priorPoint : Fin shape.cubeVariables → Carried
  claimedCoefficient : CarriedCoordinate shape → Carried
  freshMatrixImage :
    Fin shape.freshCount → Fin shape.matrixCount → Carried
  sourceAssignment : Fin shape.sourceCount → Carried
  carriedImage : CarriedCoordinate shape → Carried
  terminal : Carried
  frameBase : Nat

def polynomialDegreeSum {shape : Shape} (input : Input shape) : Nat :=
  KSparsePolynomial.totalDegreeSum input.constraintPolynomial.terms

def sparseRowsPerSource {shape : Shape} (input : Input shape) : Nat :=
  3 * polynomialDegreeSum input

def pointEqualityRows (variables : Nat) : Nat :=
  3 * variables + 3 * (variables - 1)

def ccsInput {shape : Shape} (input : Input shape)
    (source : Fin shape.freshCount) :
    KSparsePolynomial.Input shape.matrixCount where
  polynomial := input.constraintPolynomial
  point := input.freshMatrixImage source
  frameBase := input.frameBase + source.val * sparseRowsPerSource input

def ccsRows {shape : Shape} (input : Input shape) : List Row :=
  (canonicalFinIndices shape.freshCount).flatMap fun source =>
    KSparsePolynomial.rows (ccsInput input source)

def ccsOutputs {shape : Shape} (input : Input shape) : List Carried :=
  (canonicalFinIndices shape.freshCount).map fun source =>
    KSparsePolynomial.output (ccsInput input source)

def normBase {shape : Shape} (input : Input shape) : Nat :=
  input.frameBase + shape.freshCount * sparseRowsPerSource input

def normInput {shape : Shape} (input : Input shape)
    (source : Fin shape.sourceCount) : KStrictNorm.Input where
  value := input.sourceAssignment source
  frameBase := normBase input + 6 * source.val

def normRows {shape : Shape} (input : Input shape) : List Row :=
  (canonicalFinIndices shape.sourceCount).flatMap fun source =>
    KStrictNorm.rows (normInput input source)

def normOutputs {shape : Shape} (input : Input shape) : List Carried :=
  (canonicalFinIndices shape.sourceCount).map fun source =>
    KStrictNorm.output (normInput input source)

def alphaEqualityBase {shape : Shape} (input : Input shape) : Nat :=
  normBase input + 6 * shape.sourceCount

def alphaEqualityInput {shape : Shape} (input : Input shape) :
    KPointEquality.Input shape.cubeVariables where
  left := input.point
  right := input.alpha
  frameBase := alphaEqualityBase input

def priorEqualityBase {shape : Shape} (input : Input shape) : Nat :=
  alphaEqualityBase input + pointEqualityRows shape.cubeVariables

def priorEqualityInput {shape : Shape} (input : Input shape) :
    KPointEquality.Input shape.cubeVariables where
  left := input.point
  right := input.priorPoint
  frameBase := priorEqualityBase input

def combinedCoefficients {shape : Shape} (input : Input shape) : List Carried :=
  ccsOutputs input ++ normOutputs input

theorem combinedCoefficients_length {shape : Shape} (input : Input shape) :
    (combinedCoefficients input).length =
      shape.freshCount + shape.sourceCount := by
  unfold combinedCoefficients ccsOutputs normOutputs
  rw [List.length_append, List.length_map, List.length_map,
    canonicalFinIndices_length, canonicalFinIndices_length]

def combinedBase {shape : Shape} (input : Input shape) : Nat :=
  priorEqualityBase input + pointEqualityRows shape.cubeVariables

def combinedOutput {shape : Shape} (input : Input shape) : Carried :=
  hornerCarried input.gamma (KFrames.frameAt (combinedBase input))
    (combinedCoefficients input) 0

def combinedRows {shape : Shape} (input : Input shape) : List Row :=
  hornerRows input.gamma (KFrames.frameAt (combinedBase input))
    (combinedCoefficients input) 0

def carriedCoefficients {shape : Shape} (input : Input shape) : List Carried :=
  List.replicate shape.carriedEvaluationOffset KLinear.zeroCarried ++
    (canonicalCarriedCoordinates shape).map input.carriedImage

theorem carriedCoefficients_length {shape : Shape} (input : Input shape) :
    (carriedCoefficients input).length = shape.jointCoefficientCount := by
  unfold carriedCoefficients Shape.jointCoefficientCount
    Shape.carriedEvaluationOffset
  rw [List.length_append, List.length_replicate, List.length_map,
    canonicalCarriedCoordinates_length]

def carriedBase {shape : Shape} (input : Input shape) : Nat :=
  combinedBase input +
    3 * ((combinedCoefficients input).length - 1)

def carriedOutput {shape : Shape} (input : Input shape) : Carried :=
  hornerCarried input.gamma (KFrames.frameAt (carriedBase input))
    (carriedCoefficients input) 0

def carriedRows {shape : Shape} (input : Input shape) : List Row :=
  hornerRows input.gamma (KFrames.frameAt (carriedBase input))
    (carriedCoefficients input) 0

def finalBase {shape : Shape} (input : Input shape) : Nat :=
  carriedBase input + 3 * ((carriedCoefficients input).length - 1)

def alphaProductFrame {shape : Shape} (input : Input shape) : Frame :=
  KFrames.frameAt (finalBase input) 0

def priorProductFrame {shape : Shape} (input : Input shape) : Frame :=
  KFrames.frameAt (finalBase input) 1

def alphaProduct {shape : Shape} (input : Input shape) : Carried :=
  KMulChain.frameOutput (alphaProductFrame input)

def priorProduct {shape : Shape} (input : Input shape) : Carried :=
  KMulChain.frameOutput (priorProductFrame input)

def terminalExpression {shape : Shape} (input : Input shape) : Carried :=
  KLinear.addCarried (alphaProduct input) (priorProduct input)

def rowGroups {shape : Shape} (input : Input shape) : List (List Row) :=
  [ ccsRows input,
    normRows input,
    KPointEquality.rows (alphaEqualityInput input),
    KPointEquality.rows (priorEqualityInput input),
    combinedRows input,
    carriedRows input,
    KMul.rows
      (KPointEquality.equalityCarried (alphaEqualityInput input))
      (combinedOutput input) (alphaProductFrame input),
    KMul.rows
      (KPointEquality.equalityCarried (priorEqualityInput input))
      (carriedOutput input) (priorProductFrame input),
    KEquality.rows (terminalExpression input) input.terminal ]

def rows {shape : Shape} (input : Input shape) : List Row :=
  (rowGroups input).flatten

private theorem constantFlatMapLength
    {α β : Type} (items : List α) (program : α → List β) (count : Nat)
    (each : ∀ item ∈ items, (program item).length = count) :
    (items.flatMap program).length = count * items.length := by
  induction items with
  | nil => rfl
  | cons item rest inductionHypothesis =>
      rw [List.flatMap_cons, List.length_append,
        each item (by simp)]
      have tailEach : ∀ value ∈ rest, (program value).length = count := by
        intro value member
        exact each value (by simp [member])
      rw [inductionHypothesis tailEach]
      simp only [List.length_cons, Nat.mul_succ]
      omega

theorem ccsRows_length {shape : Shape} (input : Input shape) :
    (ccsRows input).length =
      sparseRowsPerSource input * shape.freshCount := by
  unfold ccsRows
  calc
    ((canonicalFinIndices shape.freshCount).flatMap fun source =>
        KSparsePolynomial.rows (ccsInput input source)).length =
        sparseRowsPerSource input *
          (canonicalFinIndices shape.freshCount).length := by
      apply constantFlatMapLength _ _ (sparseRowsPerSource input)
      intro source _
      rw [KSparsePolynomial.rows_length]
      rfl
    _ = sparseRowsPerSource input * shape.freshCount := by
      rw [canonicalFinIndices_length]

theorem normRows_length {shape : Shape} (input : Input shape) :
    (normRows input).length = 6 * shape.sourceCount := by
  unfold normRows
  rw [constantFlatMapLength _ _ _ (fun source _ =>
    KStrictNorm.rows_length (normInput input source)),
    canonicalFinIndices_length]

theorem rows_length {shape : Shape} (input : Input shape) :
    (rows input).length =
      sparseRowsPerSource input * shape.freshCount
        + 6 * shape.sourceCount
        + 2 * pointEqualityRows shape.cubeVariables
        + 3 * (shape.freshCount + shape.sourceCount - 1)
        + 3 * (shape.jointCoefficientCount - 1)
        + 8 := by
  unfold rows rowGroups
  simp only [List.flatten_cons, List.flatten_nil, List.nil_append,
    List.length_append, List.length_nil, ccsRows_length, normRows_length,
    KPointEquality.rows_length, combinedRows, carriedRows,
    hornerRows_length, KMul.rows_length, KEquality.rows_length,
    combinedCoefficients_length, carriedCoefficients_length,
    pointEqualityRows]
  omega

/-! ## Extraction from the assembled satisfaction predicate -/

theorem satisfies_group
    {shape : Shape} (input : Input shape) (assignment : Nat → Nat)
    (satisfied : Satisfies (rows input) assignment)
    (group : List Row) (member : group ∈ rowGroups input) :
    Satisfies group assignment := by
  intro row rowMember
  exact satisfied row (List.mem_flatten.2 ⟨group, member, rowMember⟩)

theorem ccs_source_satisfied
    {shape : Shape} (input : Input shape) (assignment : Nat → Nat)
    (satisfied : Satisfies (rows input) assignment)
    (source : Fin shape.freshCount) :
    Satisfies (KSparsePolynomial.rows (ccsInput input source)) assignment := by
  have group : Satisfies (ccsRows input) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  intro row member
  exact group row (List.mem_flatMap.2
    ⟨source, List.mem_ofFn.mpr ⟨source, rfl⟩, member⟩)

theorem norm_source_satisfied
    {shape : Shape} (input : Input shape) (assignment : Nat → Nat)
    (satisfied : Satisfies (rows input) assignment)
    (source : Fin shape.sourceCount) :
    Satisfies (KStrictNorm.rows (normInput input source)) assignment := by
  have group : Satisfies (normRows input) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  intro row member
  exact group row (List.mem_flatMap.2
    ⟨source, List.mem_ofFn.mpr ⟨source, rfl⟩, member⟩)

/-! ## Authoritative decoding -/

def decoded (assignment : Nat → Nat) (value : Carried) : ConcreteK :=
  KPointEquality.decoded assignment value

def decodedInput {shape : Shape} (input : Input shape)
    (assignment : Nat → Nat) :
    ProtocolPolynomial.VerifierInput ConcreteK shape where
  constraintPolynomial := input.constraintPolynomial
  priorPoint := KPointEquality.decodedRight
    (priorEqualityInput input) assignment
  claimedCoefficient := fun coordinate =>
    decoded assignment (input.claimedCoefficient coordinate)

def decodedMessage {shape : Shape} (input : Input shape)
    (assignment : Nat → Nat) :
    ProtocolPolynomial.OutputMessage ConcreteK shape where
  freshMatrixImage := fun source matrix =>
    decoded assignment (input.freshMatrixImage source matrix)
  sourceAssignment := fun source =>
    decoded assignment (input.sourceAssignment source)
  carriedImage := fun coordinate =>
    decoded assignment (input.carriedImage coordinate)

def decodedAlpha {shape : Shape} (input : Input shape)
    (assignment : Nat → Nat) : CubePoint ConcreteK shape.cubeVariables :=
  KPointEquality.decodedRight (alphaEqualityInput input) assignment

def decodedPoint {shape : Shape} (input : Input shape)
    (assignment : Nat → Nat) : CubePoint ConcreteK shape.cubeVariables :=
  KPointEquality.decodedLeft (alphaEqualityInput input) assignment

theorem decodedPoint_prior_left
    {shape : Shape} (input : Input shape) (assignment : Nat → Nat) :
    KPointEquality.decodedLeft (priorEqualityInput input) assignment =
      decodedPoint input assignment := rfl

/-! ## Soundness of the two gamma blocks -/

theorem ccsOutputs_decoded
    {shape : Shape} (input : Input shape) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    (ccsOutputs input).map (decoded assignment) =
      (canonicalFinIndices shape.freshCount).map fun source =>
        CCSResidualTable.evaluatePolynomial ConcreteCarrier.extensionOps
          input.constraintPolynomial
          (fun matrix =>
            decoded assignment (input.freshMatrixImage source matrix)) := by
  unfold ccsOutputs
  rw [List.map_map]
  apply List.map_congr_left
  intro source _
  exact KSparsePolynomial.rows_sound (ccsInput input source) assignment
    constantWire (ccs_source_satisfied input assignment satisfied source)

theorem normOutputs_decoded
    {shape : Shape} (input : Input shape) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    (normOutputs input).map (decoded assignment) =
      (canonicalFinIndices shape.sourceCount).map fun source =>
        ProtocolPolynomial.strictNormResidual ConcreteCarrier.extensionOps
          (decoded assignment (input.sourceAssignment source)) := by
  unfold normOutputs
  rw [List.map_map]
  apply List.map_congr_left
  intro source _
  exact KStrictNorm.rows_sound (normInput input source) assignment
    constantWire (norm_source_satisfied input assignment satisfied source)

theorem combined_sound
    {shape : Shape} (input : Input shape) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    decoded assignment (combinedOutput input) =
      ConcreteCarrier.extensionOps.add
        (ProtocolPolynomial.ccsAtMessage ConcreteCarrier.extensionOps
          (decodedInput input assignment) (decoded assignment input.gamma)
          (decodedMessage input assignment))
        (SignedJointIdentity.gammaTerm ConcreteCarrier.extensionOps
          (decoded assignment input.gamma) shape.freshCount
          (ProtocolPolynomial.normAtMessage ConcreteCarrier.extensionOps
            (decoded assignment input.gamma)
            (decodedMessage input assignment))) := by
  have group : Satisfies (combinedRows input) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have computed :=
    KConcreteHorner.rows_sound assignment input.gamma
      (KFrames.frameAt (combinedBase input)) (combinedCoefficients input) 0
      group
  have computed' :
      decoded assignment (combinedOutput input) =
        Message.evaluateCoefficients ConcreteCarrier.extensionOps.toOps
          (decoded assignment input.gamma)
          ((combinedCoefficients input).map (decoded assignment)) := by
    simpa [decoded, KConcreteHorner.decoded, combinedOutput] using computed
  change decoded assignment (combinedOutput input) = _
  rw [computed']
  unfold combinedCoefficients
  rw [List.map_append, ccsOutputs_decoded input assignment constantWire
      satisfied,
    normOutputs_decoded input assignment constantWire satisfied,
    SignedCoefficientPolynomial.evaluate_append ConcreteCarrier.extensionOps
      ConcreteCarrier.extensionLaws,
    SignedCoefficientPolynomial.evaluate_canonicalFinMap_eq_gammaSum
      ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws,
    SignedCoefficientPolynomial.evaluate_canonicalFinMap_eq_gammaSum
      ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws,
    List.length_map, canonicalFinIndices_length]
  unfold ProtocolPolynomial.ccsAtMessage ProtocolPolynomial.normAtMessage
    decodedInput decodedMessage SignedJointIdentity.gammaTerm
  rfl

private theorem evaluate_zeros (gamma : ConcreteK) :
    ∀ count,
      Message.evaluateCoefficients ConcreteCarrier.extensionOps.toOps gamma
          (List.replicate count ConcreteCarrier.extensionOps.zero) =
        ConcreteCarrier.extensionOps.zero
  | 0 => rfl
  | count + 1 => by
      simp only [List.replicate_succ, Message.evaluateCoefficients]
      rw [evaluate_zeros gamma count,
        ConcreteCarrier.extensionLaws.mul_zero,
        ConcreteCarrier.extensionLaws.zero_add]

private theorem finiteSum_eq_foldr :
    ∀ values : List ConcreteK,
      BooleanTable.finiteSum ConcreteCarrier.extensionOps values =
        values.foldr ConcreteCarrier.extensionOps.add
          ConcreteCarrier.extensionOps.zero
  | [] => rfl
  | _ :: values => by
      simp only [BooleanTable.finiteSum, List.foldr]
      rw [finiteSum_eq_foldr values]

def decodedCarriedCoefficients
    {shape : Shape} (input : Input shape) (assignment : Nat → Nat) :
    TargetPolynomial.CarriedTargetCoefficients ConcreteK shape where
  coefficient := fun coordinate =>
    decoded assignment (input.carriedImage coordinate)

theorem carried_sound
    {shape : Shape} (input : Input shape) (assignment : Nat → Nat)
    (satisfied : Satisfies (rows input) assignment) :
    decoded assignment (carriedOutput input) =
      SignedJointIdentity.gammaTerm ConcreteCarrier.extensionOps
        (decoded assignment input.gamma) shape.carriedEvaluationOffset
        (SignedJointIdentity.sumMap ConcreteCarrier.extensionOps
          (canonicalCarriedCoordinates shape) fun coordinate =>
            SignedJointIdentity.gammaTerm ConcreteCarrier.extensionOps
              (decoded assignment input.gamma)
              coordinate.localGammaExponent
              (decoded assignment (input.carriedImage coordinate))) := by
  have group : Satisfies (carriedRows input) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have computed :=
    KConcreteHorner.rows_sound assignment input.gamma
      (KFrames.frameAt (carriedBase input)) (carriedCoefficients input) 0 group
  have computed' :
      decoded assignment (carriedOutput input) =
        Message.evaluateCoefficients ConcreteCarrier.extensionOps.toOps
          (decoded assignment input.gamma)
          ((carriedCoefficients input).map (decoded assignment)) := by
    simpa [decoded, KConcreteHorner.decoded, carriedOutput] using computed
  change decoded assignment (carriedOutput input) = _
  rw [computed']
  unfold carriedCoefficients
  rw [List.map_append, List.map_replicate]
  have zeroDecoded :
      decoded assignment KLinear.zeroCarried =
        ConcreteCarrier.extensionOps.zero :=
    KSparsePolynomial.decoded_zero assignment
  rw [zeroDecoded,
    SignedCoefficientPolynomial.evaluate_append ConcreteCarrier.extensionOps
      ConcreteCarrier.extensionLaws,
    evaluate_zeros,
    ConcreteCarrier.extensionLaws.zero_add, List.length_replicate]
  rw [List.map_map]
  have localEvaluation :=
    SignedCoefficientPolynomial.evaluate_canonicalCarriedMap_eq_targetLocal
      ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws
      (decoded assignment input.gamma)
      (decodedCarriedCoefficients input assignment)
  change
    Message.evaluateCoefficients ConcreteCarrier.extensionOps.toOps
        (decoded assignment input.gamma)
        ((canonicalCarriedCoordinates shape).map fun coordinate =>
          decoded assignment (input.carriedImage coordinate)) =
      TargetPolynomial.evaluateLocal ConcreteCarrier.extensionOps.toOps
        (decodedCarriedCoefficients input assignment)
        (decoded assignment input.gamma) at localEvaluation
  change
    ConcreteCarrier.extensionOps.mul
        (TargetPolynomial.power ConcreteCarrier.extensionOps.toOps
          (decoded assignment input.gamma) shape.carriedEvaluationOffset)
        (Message.evaluateCoefficients ConcreteCarrier.extensionOps.toOps
          (decoded assignment input.gamma)
          ((canonicalCarriedCoordinates shape).map fun coordinate =>
            decoded assignment (input.carriedImage coordinate))) =
      _
  rw [localEvaluation]
  rw [TargetPolynomial.evaluateLocal_eq_foldr]
  unfold SignedJointIdentity.sumMap
  rw [finiteSum_eq_foldr]
  unfold SignedJointIdentity.gammaTerm TargetPolynomial.term
  rfl

/-! ## Final semantic binding -/

theorem rows_sound
    {shape : Shape} (input : Input shape) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    decoded assignment input.terminal =
      ProtocolPolynomial.terminalFromMessage ConcreteCarrier.extensionOps
        (decodedInput input assignment) (decodedAlpha input assignment)
        (decoded assignment input.gamma) (decodedPoint input assignment)
      (decodedMessage input assignment) := by
  have alphaEqualitySatisfied :
      Satisfies (KPointEquality.rows (alphaEqualityInput input)) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have priorEqualitySatisfied :
      Satisfies (KPointEquality.rows (priorEqualityInput input)) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have alphaProductSatisfied :
      Satisfies
        (KMul.rows
          (KPointEquality.equalityCarried (alphaEqualityInput input))
          (combinedOutput input) (alphaProductFrame input)) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have priorProductSatisfied :
      Satisfies
        (KMul.rows
          (KPointEquality.equalityCarried (priorEqualityInput input))
          (carriedOutput input) (priorProductFrame input)) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have bindingSatisfied :
      Satisfies (KEquality.rows (terminalExpression input) input.terminal)
        assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have alphaEquality :=
    KPointEquality.rows_sound (alphaEqualityInput input) assignment
      constantWire alphaEqualitySatisfied
  have priorEquality :=
    KPointEquality.rows_sound (priorEqualityInput input) assignment
      constantWire priorEqualitySatisfied
  have alphaProductPair :=
    KMulChain.frameOutput_sound assignment
      (KPointEquality.equalityCarried (alphaEqualityInput input))
      (combinedOutput input) (alphaProductFrame input) alphaProductSatisfied
  have priorProductPair :=
    KMulChain.frameOutput_sound assignment
      (KPointEquality.equalityCarried (priorEqualityInput input))
      (carriedOutput input) (priorProductFrame input) priorProductSatisfied
  have binding :=
    KEquality.rows_sound assignment (terminalExpression input) input.terminal
      constantWire bindingSatisfied
  have bindingPair :
      carriedValue assignment (terminalExpression input) =
        carriedValue assignment input.terminal := by
    unfold carriedValue
    simp only [Pair.mk.injEq]
    exact binding
  have combinedSemantic :=
    combined_sound input assignment constantWire satisfied
  have carriedSemantic :=
    carried_sound input assignment satisfied
  unfold decoded
  change
    KPointEquality.decoded assignment (combinedOutput input) =
      _ at combinedSemantic
  change
    KPointEquality.decoded assignment (carriedOutput input) =
      _ at carriedSemantic
  apply KConcreteBridge.ofConcrete_injective
  rw [KPointEquality.ofConcrete_decoded]
  rw [← bindingPair]
  unfold terminalExpression
  rw [KLinear.carriedValue_add, alphaProduct, alphaProductPair,
    priorProduct, priorProductPair]
  rw [← KPointEquality.ofConcrete_decoded assignment
    (KPointEquality.equalityCarried (alphaEqualityInput input)),
    alphaEquality,
    ← KPointEquality.ofConcrete_decoded assignment (combinedOutput input),
    combinedSemantic,
    ← KPointEquality.ofConcrete_decoded assignment
      (KPointEquality.equalityCarried (priorEqualityInput input)),
    priorEquality,
    ← KPointEquality.ofConcrete_decoded assignment (carriedOutput input),
    carriedSemantic]
  rw [← KConcreteBridge.ofConcrete_mul, ← KConcreteBridge.ofConcrete_mul,
    ← KConcreteBridge.ofConcrete_add]
  apply congrArg KConcreteBridge.ofConcrete
  unfold ProtocolPolynomial.terminalFromMessage
  unfold ProtocolPolynomial.carriedAtMessage
  simp only [decoded, decodedPoint, decodedAlpha, decodedInput, decodedMessage,
    alphaEqualityInput, priorEqualityInput]
  unfold SignedJointIdentity.gammaTerm
  have samePoint :
      KPointEquality.decodedLeft
          { left := input.point, right := input.priorPoint,
            frameBase := priorEqualityBase input } assignment =
        KPointEquality.decodedLeft
          { left := input.point, right := input.alpha,
            frameBase := alphaEqualityBase input } assignment := rfl
  rw [samePoint]
  congr 1
  exact
    (ConcreteCarrier.extensionLaws.mul_assoc _ _ _).symm.trans <|
      (congrArg
        (fun value => ConcreteCarrier.extensionOps.mul value _)
        (ConcreteCarrier.extensionLaws.mul_comm _ _)).trans <|
          ConcreteCarrier.extensionLaws.mul_assoc _ _ _

end Nightstream.Implementation.R1CS.Canonical.KPiCcsTerminal
