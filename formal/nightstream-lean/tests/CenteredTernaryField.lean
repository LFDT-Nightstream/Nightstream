import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.DerivedNegative
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.LinearCompiler
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.Refinement.DerivedBorrow
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.LayoutManifest
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.LayoutWidthFloor

/-! Narrow compile-time checks for ordinary private centered-field encoding. -/

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernaryReducedCore
open Nightstream.Implementation.R1CS.CenteredTernaryField
open Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged
open Nightstream.Implementation.R1CS.CenteredTernaryDerivedNegative
open Nightstream.Implementation.R1CS.CenteredTernaryLinearCompiler
open Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow
open Nightstream.Implementation.R1CS.FPrimeFieldLayout

example (prime : EuclidPrime goldilocksP)
    {value : Nat} (canonical : value < goldilocksP) :
    CenteredUnitGateHolds value ↔ CenteredResidue value :=
  centeredUnitGate_iff prime canonical

example {source : Nat} (canonical : source < goldilocksP) :
    decodeWord (encodeDigit source) = source :=
  decode_encodeDigit canonical

example {digits : Nat → Nat} (represented : Represents 0 digits) :
    ∀ index, index < digitCount → digits index = 0 :=
  represents_zero_unique represented

example : 3 ^ 40 < goldilocksP ∧ goldilocksP < 3 ^ digitCount :=
  width_floor

example (encoded : AcceptedEncoding) :
    encodeChosen (decodeChosen encoded) = encoded :=
  encodeChosen_decodeChosen encoded

example (witness : ChosenWitness) :
    decodeChosen (encodeChosen witness) = witness :=
  decodeChosen_encodeChosen witness

example :
    rawTargetWord 0 ≠ rawTargetWord goldilocksP ∧
      decodeFiniteWord (rawTargetWord 0) =
        decodeFiniteWord (rawTargetWord goldilocksP) :=
  ⟨duplicate_words_differ, duplicate_words_decode_same⟩

example (sourcePredicate : Nat → Prop) :
    (∃ witness, AugmentedRelation sourcePredicate witness) ↔
      ∃ source, source < goldilocksP ∧ sourcePredicate source :=
  augmented_exists_iff_semantic_exists sourcePredicate

example {fieldCount : Nat}
    (encoded : AcceptedPrivateEncoding fieldCount) :
    encodeChosenPrivate (decodeChosenPrivate encoded) = encoded :=
  encodeChosenPrivate_decodeChosenPrivate encoded

example {fieldCount : Nat}
    (sourcePredicate : (Fin fieldCount → Nat) → Prop) :
    (∃ witness, AugmentedPrivateRelation sourcePredicate witness) ↔
      ∃ sources, (∀ field, sources field < goldilocksP) ∧
        sourcePredicate sources :=
  augmented_private_exists_iff_semantic_exists sourcePredicate

example (value : Nightstream.SuperNeo.Concrete.F) :
    Nightstream.SuperNeo.Concrete.centeredMagnitude value < 2 ↔
      CenteredResidue value.val :=
  concrete_norm_two_iff_centeredResidue value

example : normDischargedGates.length = 82 :=
  normDischargedGates_length

example {value : Nat} (centered : CenteredResidue value) :
    derivedNegative value = negativeIndicator value :=
  derivedNegative_eq_indicator centered

example : derivedBorrowSchedule.length = 41 :=
  derivedBorrowSchedule_length

example {index : Nat} (indexLt : index < digitCount) :
    (derivedBorrowEquation index).degree ≤ 3 :=
  derivedBorrowEquation_degree_le_three indexLt

example : maximumDerivedBorrowDegree = 3 :=
  maximumDerivedBorrowDegree_eq_three

example (assignment : Nat → Nat) :
    DerivedAccepts assignment ↔
      ∀ equation ∈ derivedBorrowEquations, equation.Holds assignment :=
  derivedAccepts_iff_polynomial_schedule assignment

example (role : SlotRole) :
    role.Eligible ∨ role.ExplicitlyExcluded :=
  SlotRole.eligible_or_explicitlyExcluded role

example {Owner : Type} {runOf : Owner → CoordinateRun}
    {count : Nat} {first second : Owner} {tail : List Owner}
    (partition : ExactPartition runOf count (first :: second :: tail)) :
    (runOf first).start = 0 ∧
      0 < (runOf first).length ∧
      (runOf first).endExclusive = (runOf second).start :=
  ⟨ExactPartition.firstStartsAtZero partition,
    ExactPartition.ownerLengthPositive partition (by simp),
    ExactPartition.firstAbutsSecond partition⟩

example {Owner : Type} {runOf : Owner → CoordinateRun}
    {count : Nat} {owners : List Owner}
    (partition : ExactPartition runOf count owners)
    (nonempty : owners ≠ []) :
    ∃ owner : Owner,
      owner ∈ owners ∧ (runOf owner).endExclusive = count :=
  ExactPartition.finalEndsAtCount partition nonempty

example (manifest : Manifest) (valid : manifest.Valid)
    {column : Nat} (columnLt : column < manifest.sourceColumnCount) :
    ∃ segment : SourceSegment,
      segment ∈ manifest.sourceSegments ∧
      segment.source.Contains column ∧
      ∀ other : SourceSegment,
        other ∈ manifest.sourceSegments →
        other.source.Contains column →
        other = segment :=
  valid.existsUniqueSlotForSource columnLt

example (manifest : Manifest) (valid : manifest.Valid)
    {coordinate : Nat} (coordinateLt : coordinate < manifest.encodedColumnCount) :
    ∃ owner : CoordinateOwnerRun,
      owner ∈ manifest.coordinateOwners ∧
      owner.encoded.Contains coordinate ∧
      ∀ other : CoordinateOwnerRun,
        other ∈ manifest.coordinateOwners →
        other.encoded.Contains coordinate →
        other = owner :=
  valid.encodedCoordinateHasUniqueOwner coordinateLt

example (manifest : Manifest) (valid : manifest.Valid)
    {coordinate : Nat} (coordinateLt : coordinate < manifest.ceAssignmentLength) :
    ∃ owner : CoordinateOwnerRun,
      owner ∈ manifest.coordinateOwners ∧
      owner.ce.Contains coordinate ∧
      ∀ other : CoordinateOwnerRun,
        other ∈ manifest.coordinateOwners →
        other.ce.Contains coordinate →
        other = owner :=
  valid.ceCoordinateHasUniqueOwner coordinateLt

example (manifest : Manifest) (valid : manifest.Valid) :
    (∃ segment : SourceSegment,
      segment ∈ manifest.sourceSegments ∧
      segment.source.Contains 0 ∧
      segment.role = .constantOne) ∧
    (∃ owner : CoordinateOwnerRun,
      owner ∈ manifest.coordinateOwners ∧
      owner.encoded.Contains 0 ∧
      owner.role.ExplicitlyExcluded) ∧
    (∃ owner : CoordinateOwnerRun,
      owner ∈ manifest.coordinateOwners ∧
      owner.ce.Contains 0 ∧
      owner.role.ExplicitlyExcluded) :=
  ⟨valid.sourceZeroHasConstantOneOwner,
    valid.encodedZeroHasExcludedOwner,
    valid.ceZeroHasExcludedOwner⟩

example (manifest : Manifest) (valid : manifest.Valid)
    {segment : SourceSegment}
    (segmentMember : segment ∈ manifest.sourceSegments)
    (eligible : segment.role.Eligible) :
    ∃ owner : CoordinateOwnerRun,
      owner ∈ manifest.coordinateOwners ∧
      owner.owner = .source segment ∧
      owner.encoded.length = segment.source.length * digitCount ∧
      owner.ce.length = segment.source.length * digitCount ∧
      ∀ other : CoordinateOwnerRun,
        other ∈ manifest.coordinateOwners →
        other.owner = .source segment →
        other = owner :=
  valid.ordinaryOwnerFor segmentMember eligible

example (manifest : Manifest) (valid : manifest.Valid)
    {owner : CoordinateOwnerRun}
    (ownerMember : owner ∈ manifest.coordinateOwners)
    {ownerPath : String} {role : SlotRole}
    (coordinateOnly : owner.owner = .coordinateOnly ownerPath role) :
    role.ExplicitlyExcluded :=
  valid.coordinateOnlyOwnerIsExcluded ownerMember coordinateOnly

example {manifest : Manifest} (valid : manifest.Valid) :
    eligibleEncodedLength manifest.coordinateOwners ≤
      manifest.encodedColumnCount :=
  encodedEligibleLength_le_total valid

example {manifest : Manifest} (valid : manifest.Valid) :
    eligibleCeLength manifest.coordinateOwners ≤
      manifest.ceAssignmentLength :=
  ceEligibleLength_le_total valid

example :
    (∀ digit : Fin digitCount, NormBoundTwo (finiteEncode 2 digit)) ∧
      decodeFiniteWord (finiteEncode 2) = 2 ∧
      ¬ CenteredResidue 2 :=
  normBounded_word_can_decode_nonCentered_source

example (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (norm : DigitNormBoundTwo assignment) :
    CenteredTernaryNormDischarged.Accepts assignment ↔
      DerivedAccepts assignment ∧ NegativesMaterialized assignment :=
  conservative_iff_derived_and_materialized prime canonical one norm

example {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (norm : DigitNormBoundTwo assignment) :
    CenteredTernaryNormDischarged.Accepts
        (materializeNegatives assignment) ↔
      DerivedAccepts assignment :=
  materialized_accepts_iff_derived one norm

example {fieldCount : Nat} (layout : Layout fieldCount)
    (encoded : Nat → Nat)
    (norm : PrivateCoordinatesNormBoundTwo layout encoded) :
    encodeChosenPrivate (parse layout encoded norm) =
      acceptedProjection layout encoded norm :=
  reemit_parsed_projection layout encoded norm

example {fieldCount : Nat} (layout : Layout fieldCount)
    (sourceRows : List Row) (encoded : Nat → Nat) :
    Satisfies (loweredRows layout sourceRows) encoded ↔
      Satisfies sourceRows (decodedAssignment layout encoded) :=
  loweredRows_iff_sourceRows layout sourceRows encoded

example {fieldCount : Nat} {layout : Layout fieldCount}
    (materializer : HonestMaterializer layout)
    (sourceRows : List Row) {source : Nat → Nat}
    (canonical : ∀ column, source column < goldilocksP)
    (accepted : Satisfies sourceRows source) :
    ∃ encoded,
      PrivateCoordinatesNormBoundTwo layout encoded ∧
      Satisfies (loweredRows layout sourceRows) encoded ∧
      decodedAssignment layout encoded = source :=
  honest_complete materializer sourceRows canonical accepted
