import Nightstream.Implementation.Nebula.NIFS.PiCCS.TranscriptCursorFor
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexCount

/-!
Contract: exact product-PiCCS row geometry at one augmented-relation exponent.

The transcript count is derived from the emitted field lists and the exact
duplex control. The arithmetic count is derived from the sparse polynomial
term count and degree sum. No row count is supplied by a caller.

This module does not select an exponent or claim that a complete generated
F-prime relation fits a row cube.

Assurance tier: exponent-indexed row implementation.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Nebula.ProductPiCcsTranscriptGeometryFor

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductPiCcsTranscriptRowsFor
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

private theorem length_flatMap_uniform
    {Alpha Beta : Type} (items : List Alpha) (values : Alpha -> List Beta)
    (count : Nat) (uniform : forall item, (values item).length = count) :
    (items.flatMap values).length = items.length * count := by
  induction items with
  | nil => simp
  | cons head tail inductionHypothesis =>
      simp [uniform, inductionHypothesis, Nat.add_mul, Nat.add_comm]

private theorem builder_entries_length_of_control
    {builder : SymbolicDuplex.Builder}
    {control : SymbolicDuplexCount.Control}
    (equal : SymbolicDuplexCount.ofBuilder builder = control) :
    builder.entries.length = control.entries := by
  change (SymbolicDuplexCount.ofBuilder builder).entries = control.entries
  exact congrArg SymbolicDuplexCount.Control.entries equal

private theorem value_absorbList_absorbed
    (constants : Poseidon2Schedule.Constants)
    (values : List Nat) (state : Poseidon2Duplex.State) :
    (Poseidon2Duplex.absorbList constants values state).absorbed =
      SymbolicDuplexCursor.after state.absorbed values.length := by
  induction values generalizing state with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      rw [Poseidon2Duplex.absorbList, inductionHypothesis]
      simp only [SymbolicDuplexCursor.after]
      unfold Poseidon2Duplex.absorbElem Poseidon2Duplex.guarded
        SymbolicDuplexCursor.step
      by_cases full : Poseidon2Sponge.rate <= state.absorbed
      · simp [full, Poseidon2Duplex.permute]
      · simp [full]

private theorem after_two_four_mul (blocks : Nat) :
    SymbolicDuplexCursor.after 2 (4 * blocks) = 2 := by
  induction blocks with
  | zero => rfl
  | succ blocks inductionHypothesis =>
      rw [Nat.mul_succ, SymbolicDuplexCursor.after_add,
        inductionHypothesis]
      decide

private theorem initialStateForStatement_absorbed
    (statementId : ProductPoseidon2.StatementId) :
    (ProductPoseidon2.initialStateForStatement statementId).absorbed = 2 := by
  rw [ProductPoseidon2.initialStateForStatement,
    value_absorbList_absorbed,
    ProductPoseidon2.statementIdentifierFields,
    ProductPoseidon2.proofPrefixFields_length]
  change SymbolicDuplexCursor.after 0 366 = 2
  rw [show 366 = 2 + 4 * 91 by decide,
    SymbolicDuplexCursor.after_add]
  change SymbolicDuplexCursor.after 2 (4 * 91) = 2
  exact after_two_four_mul 91

theorem monomialFields_length
    {rowVariables : Nat}
    (monomial : CCSResidualTable.Monomial
      Nightstream.SuperNeo.Concrete.K (Shape rowVariables).matrixCount) :
    (monomialFields monomial).length = 16 := by
  simp [monomialFields, constantKFields, canonicalFinIndices_length,
    ProductPiCcsTranscriptRows.monomialFields,
    ProductPiCcsTranscriptRows.constantKFields,
    ProductPiCcsTranscriptRowsFor.Shape, ProductNifsCodec.shapeFor]

theorem polynomialFields_length
    {rowVariables : Nat}
    (polynomial : CCSResidualTable.ConstraintPolynomial
      Nightstream.SuperNeo.Concrete.K (Shape rowVariables).matrixCount) :
    (polynomialFields polynomial).length =
      2 + polynomial.terms.length * 16 := by
  simp only [polynomialFields,
    ProductPiCcsTranscriptRows.polynomialFields, List.length_cons,
    List.length_flatMap]
  have mapped :
      polynomial.terms.map (fun term =>
        (ProductPiCcsTranscriptRows.monomialFields term).length) =
        polynomial.terms.map (fun _ => 16) := by
    apply List.map_congr_left
    intro term _
    simpa [monomialFields] using monomialFields_length term
  rw [mapped]
  induction polynomial.terms with
  | nil => simp
  | cons head tail inductionHypothesis =>
      simp at inductionHypothesis ⊢
      omega

theorem pointFields_length
    {count : Nat} (point : Fin count -> Carried) :
    (pointFields point).length = count * 2 := by
  simp only [pointFields, ProductPiCcsTranscriptRows.pointFields]
  rw [length_flatMap_uniform _ _ 2 (fun _ => rfl),
    canonicalFinIndices_length]

theorem statementFields_length
    {rowVariables : Nat} (input : Input rowVariables)
    (termCount : input.constraintPolynomial.terms.length = 74) :
    (statementFields input).length = 25386 + 2 * rowVariables := by
  have polynomialLength := polynomialFields_length input.constraintPolynomial
  have carriedLength :
      ((canonicalCarriedCoordinates (Shape rowVariables)).flatMap
        fun coordinate => carriedFields (input.claimedCoefficient coordinate)
      ).length = 24192 := by
    rw [length_flatMap_uniform _ _ 2 (fun _ => rfl),
      canonicalCarriedCoordinates_length]
    rfl
  unfold statementFields shapeFields
  simp only [List.length_append, List.length_cons, List.length_nil,
    polynomialLength, pointFields_length, carriedLength, termCount]
  simp [ProductPiCcsTranscriptRows.shapeFields,
    ProductPiCcsTranscriptRowsFor.Shape, ProductNifsCodec.shapeFor]
  omega

theorem verifierChallengeFields_length
    (eventIndex challengeIndex challengeType : Nat)
    (coordinates : List Nat) :
    (verifierChallengeFields eventIndex challengeIndex challengeType
      coordinates).length = 60 + coordinates.length := by
  have domainLength : ProductPoseidon2.construction3DomainFields.length = 36 := by
    decide
  have labelLength : ProductPoseidon2.verifierChallengeLabelFields.length = 20 := by
    decide
  simp [verifierChallengeFields,
    ProductPiCcsTranscriptRows.verifierChallengeFields,
    domainLength, labelLength]
  omega

theorem roundFields_length (index : Nat) (round : Round) :
    (roundFields index round).length = 40 := by
  have labelLength : ProductPoseidon2.proverMessageLabelFields.length = 16 := by
    decide
  have payloadLength :
      (round.coefficients.flatMap carriedFields).length = 20 := by
    rw [length_flatMap_uniform _ _ 2 (fun _ => rfl),
      round.coefficients_length]
  simp [roundFields, ProductPiCcsTranscriptRows.roundFields,
    proverMessageFields, ProductPiCcsTranscriptRows.proverMessageFields,
    payloadLength, labelLength]

def initialControl : SymbolicDuplexCount.Control :=
  { entries := 0, absorbed := 2 }

def publicControl (rowVariables : Nat) : SymbolicDuplexCount.Control :=
  SymbolicDuplexCount.absorbManyFast (publicFieldCount rowVariables)
    initialControl

def statementControl (rowVariables : Nat) : SymbolicDuplexCount.Control :=
  SymbolicDuplexCount.absorbManyFast (25386 + 2 * rowVariables)
    (publicControl rowVariables)

private def alphaControlGo : Nat -> SymbolicDuplexCount.Control ->
    SymbolicDuplexCount.Control
  | 0, control => control
  | count + 1, control =>
      alphaControlGo count
        (SymbolicDuplexCount.gate
          (SymbolicDuplexCount.absorbManyFast 61 control))

def alphaControl (rowVariables : Nat) : SymbolicDuplexCount.Control :=
  alphaControlGo rowVariables (statementControl rowVariables)

def gammaControl (rowVariables : Nat) : SymbolicDuplexCount.Control :=
  SymbolicDuplexCount.gate
    (SymbolicDuplexCount.absorbManyFast 60 (alphaControl rowVariables))

private def roundControlGo : Nat -> SymbolicDuplexCount.Control ->
    SymbolicDuplexCount.Control
  | 0, control => control
  | count + 1, control =>
      roundControlGo count
        (SymbolicDuplexCount.gate
          (SymbolicDuplexCount.absorbManyFast 60
            (SymbolicDuplexCount.absorbManyFast 40 control)))

def roundsControl (rowVariables : Nat) : SymbolicDuplexCount.Control :=
  roundControlGo rowVariables (gammaControl rowVariables)

def finalControl (rowVariables : Nat) : SymbolicDuplexCount.Control :=
  SymbolicDuplexCount.absorbManyFast 25724 (roundsControl rowVariables)

private theorem deriveAlphaGo_control
    {rowVariables : Nat} (input : Input rowVariables)
    (index count : Nat) (builder : SymbolicDuplex.Builder) :
    SymbolicDuplexCount.ofBuilder
        (deriveAlphaGo input index count builder).2 =
      alphaControlGo count (SymbolicDuplexCount.ofBuilder builder) := by
  induction count generalizing index builder with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [deriveAlphaGo]
      rw [inductionHypothesis]
      unfold squeezeVerifierChallenge SymbolicDuplex.squeezeK
      rw [SymbolicDuplexCount.ofBuilder_gate,
        SymbolicDuplexCount.ofBuilder_absorbMany,
        verifierChallengeFields_length,
        SymbolicDuplexCount.absorbMany_eq_fast]
      simp only [List.length_singleton, Nat.add_comm 60 1]
      rfl

private theorem replayRoundsGo_control
    {rowVariables : Nat} (input : Input rowVariables)
    (rounds : List Round) (index : Nat)
    (builder : SymbolicDuplex.Builder) :
    SymbolicDuplexCount.ofBuilder
        (replayRoundsGo input rounds index builder).builder =
      roundControlGo rounds.length
        (SymbolicDuplexCount.ofBuilder builder) := by
  induction rounds generalizing index builder with
  | nil => rfl
  | cons round rest inductionHypothesis =>
      simp only [replayRoundsGo, List.length_cons]
      rw [inductionHypothesis]
      unfold squeezeVerifierChallenge SymbolicDuplex.squeezeK
      rw [SymbolicDuplexCount.ofBuilder_gate,
        SymbolicDuplexCount.ofBuilder_absorbMany,
        verifierChallengeFields_length,
        SymbolicDuplexCount.absorbMany_eq_fast,
        SymbolicDuplexCount.ofBuilder_absorbMany,
        roundFields_length,
        SymbolicDuplexCount.absorbMany_eq_fast]
      simp only [List.length_nil, Nat.add_zero]
      rfl

private theorem initialBuilder_control
    {rowVariables : Nat} (input : Input rowVariables) :
    SymbolicDuplexCount.ofBuilder (initialBuilder input) = initialControl := by
  simp only [initialBuilder, SymbolicDuplexCount.ofBuilder,
    SymbolicDuplex.start, List.length_nil, initialControl]
  rw [initialStateForStatement_absorbed]

theorem absorbPublicInput_control
    {rowVariables : Nat} (input : Input rowVariables) :
    SymbolicDuplexCount.ofBuilder (absorbPublicInput input) =
      publicControl rowVariables := by
  unfold absorbPublicInput publicControl
  rw [SymbolicDuplexCount.ofBuilder_absorbMany,
    input.publicNifsFields_length, initialBuilder_control,
    SymbolicDuplexCount.absorbMany_eq_fast]

theorem absorbStatement_control
    {rowVariables : Nat} (input : Input rowVariables)
    (termCount : input.constraintPolynomial.terms.length = 74) :
    SymbolicDuplexCount.ofBuilder (absorbStatement input) =
      statementControl rowVariables := by
  unfold absorbStatement statementControl
  rw [SymbolicDuplexCount.ofBuilder_absorbMany,
    statementFields_length input termCount, absorbPublicInput_control,
    SymbolicDuplexCount.absorbMany_eq_fast]

theorem deriveAlpha_control
    {rowVariables : Nat} (input : Input rowVariables)
    (termCount : input.constraintPolynomial.terms.length = 74) :
    SymbolicDuplexCount.ofBuilder (deriveAlpha input).2 =
      alphaControl rowVariables := by
  unfold deriveAlpha alphaControl
  rw [deriveAlphaGo_control, absorbStatement_control input termCount]
  simp [ProductPiCcsTranscriptRowsFor.Shape, ProductNifsCodec.shapeFor]

theorem deriveGamma_control
    {rowVariables : Nat} (input : Input rowVariables)
    (termCount : input.constraintPolynomial.terms.length = 74) :
    SymbolicDuplexCount.ofBuilder (deriveGamma input).2 =
      gammaControl rowVariables := by
  unfold deriveGamma gammaControl squeezeVerifierChallenge
    SymbolicDuplex.squeezeK
  rw [SymbolicDuplexCount.ofBuilder_gate,
    SymbolicDuplexCount.ofBuilder_absorbMany,
    verifierChallengeFields_length,
    SymbolicDuplexCount.absorbMany_eq_fast,
    deriveAlpha_control input termCount]
  simp only [List.length_nil, Nat.add_zero]

theorem replayRounds_control
    {rowVariables : Nat} (input : Input rowVariables)
    (termCount : input.constraintPolynomial.terms.length = 74) :
    SymbolicDuplexCount.ofBuilder (replayRounds input).builder =
      roundsControl rowVariables := by
  unfold replayRounds roundsControl
  rw [replayRoundsGo_control, List.length_ofFn,
    deriveGamma_control input termCount]
  simp [ProductPiCcsTranscriptRowsFor.Shape, ProductNifsCodec.shapeFor]

theorem afterFullOutput_control
    {rowVariables : Nat} (input : Input rowVariables)
    (termCount : input.constraintPolynomial.terms.length = 74) :
    SymbolicDuplexCount.ofBuilder (afterFullOutput input) =
      finalControl rowVariables := by
  unfold afterFullOutput finalControl
  rw [SymbolicDuplexCount.ofBuilder_absorbMany,
    ProductPiCcsTranscriptCursorFor.fullOutputFields_length,
    replayRounds_control input termCount,
    SymbolicDuplexCount.absorbMany_eq_fast]

def transcriptRowCount (rowVariables : Nat) : Nat :=
  (finalControl rowVariables).entries * SymbolicDuplex.stride

theorem transcript_rows_length
    {rowVariables : Nat} (input : Input rowVariables)
    (termCount : input.constraintPolynomial.terms.length = 74) :
    (SymbolicDuplex.rows input.transcriptBase ProductPoseidon2.constants
      (afterFullOutput input)).length = transcriptRowCount rowVariables := by
  rw [SymbolicDuplex.rows_length]
  have count := builder_entries_length_of_control
    (afterFullOutput_control input termCount)
  rw [count]
  rfl

def occurrenceRowCount (rowVariables : Nat) : Nat :=
  let shape := ProductPiCcsTranscriptRowsFor.Shape rowVariables
  (3 * (shape.jointCoefficientCount - 1) + 2) +
    (shape.cubeVariables * (3 * 9 + 2) + 2) +
    (3 * 324 * shape.freshCount
      + 6 * shape.sourceCount
      + 2 * KPiCcsTerminal.pointEqualityRows shape.cubeVariables
      + 3 * (shape.freshCount + shape.sourceCount - 1)
      + 3 * (shape.jointCoefficientCount - 1)
      + 8)

theorem occurrence_rows_length
    {rowVariables : Nat} (input : Input rowVariables)
    (degreeSum : KSparsePolynomial.totalDegreeSum
      input.constraintPolynomial.terms = 324) :
    (KPiCcsOccurrence.rows (occurrenceInput input)).length =
      occurrenceRowCount rowVariables := by
  have terminalDegree :
      KSparsePolynomial.totalDegreeSum
        (KPiCcsOccurrence.terminalInput
          (occurrenceInput input)).constraintPolynomial.terms = 324 := by
    simpa [KPiCcsOccurrence.terminalInput, occurrenceInput] using degreeSum
  rw [KPiCcsOccurrence.rows_length]
  simp only [KPiCcsTerminal.sparseRowsPerSource,
    KPiCcsTerminal.polynomialDegreeSum]
  rw [terminalDegree]
  rfl

def rowCount (rowVariables : Nat) : Nat :=
  transcriptRowCount rowVariables + occurrenceRowCount rowVariables

theorem rows_length_exact
    {rowVariables : Nat} (input : Input rowVariables)
    (termCount : input.constraintPolynomial.terms.length = 74)
    (degreeSum : KSparsePolynomial.totalDegreeSum
      input.constraintPolynomial.terms = 324) :
    (rows input).length = rowCount rowVariables := by
  unfold rows rowCount
  rw [List.length_append, transcript_rows_length input termCount,
    occurrence_rows_length input degreeSum]

theorem rowCount_25 : rowCount 25 = 12124196 := by decide

theorem rowCount_26 : rowCount 26 = 12139373 := by decide

theorem exponent_26_adds_15177_rows :
    rowCount 26 = rowCount 25 + 15177 := by decide

end Nightstream.Implementation.Nebula.ProductPiCcsTranscriptGeometryFor
