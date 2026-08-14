import Nightstream.Implementation.R1CS.Core.LinearSubstitution
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.CenteredSeptenaryFreshCcsAuthority

/-!
Contract: exact linear substitution for a finite tuple of 23-coordinate
centered-septenary private fields.

Assurance tier: model-level for property
`FPRIME-R4-LINEAR-SUBSTITUTION-SAME-ASSIGNMENT`.

Owns: proof-carrying source and encoded placement, radix-seven decoding of
each source field, exact transport of arbitrary R1CS rows, honest
materialization, and transfer of a fresh radix-four CCS norm to the exact
typed carrier used by the transformed rows.

Does not own: generated Rust placements, selective row indices, affine
temporary elimination, selector composition, complete source-row coverage,
or a production row or column deletion claim.

Emits constraints: no. It proves equivalence for rows emitted by a separate
compiler and uses the verifier-owned outer norm as domain authority.
-/

set_option autoImplicit false
set_option maxRecDepth 262144

namespace Nightstream.Implementation.R1CS.CenteredSeptenaryLinearCompiler

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CenteredSeptenaryField
open Nightstream.Implementation.R1CS.CenteredSeptenaryNormDischarged
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

universe uCommitment

/-- Exact source-to-word layout. Every designated source field expands to one
complete contiguous radix-seven word. -/
structure Layout (fieldCount : Nat) where
  words : CenteredSeptenaryLayout.Layout fieldCount
  sourceColumn : Fin fieldCount → Nat
  expansion : LinearSubstitution.ColumnExpansion
  privateExpansion : ∀ field,
    expansion (sourceColumn field) =
      (List.range digitCount).map fun index =>
        (words.wordStart field + index, radix ^ index % goldilocksP)

def encodedColumn {fieldCount : Nat}
    (layout : Layout fieldCount) (field : Fin fieldCount)
    (digit : Nat) : Nat :=
  layout.words.wordStart field + digit

def encodedWord {fieldCount : Nat}
    (layout : Layout fieldCount) (encoded : Nat → Nat)
    (field : Fin fieldCount) : FiniteWord :=
  fun digit => encoded (encodedColumn layout field digit.val)

/-- Exact coordinate-wise projection of the outer strict `b = 4` norm. -/
def PrivateCoordinatesNormBoundFour {fieldCount : Nat}
    (layout : Layout fieldCount) (encoded : Nat → Nat) : Prop :=
  ∀ (field : Fin fieldCount) (digit : Fin digitCount),
    NormBoundFour (encoded (encodedColumn layout field digit.val))

theorem privateWordsAccepted_of_norm
    {fieldCount : Nat} {layout : Layout fieldCount}
    {encoded : Nat → Nat}
    (norm : PrivateCoordinatesNormBoundFour layout encoded) :
    ∀ field, FiniteAlphabetWord (encodedWord layout encoded field) := by
  intro field digit
  exact normBoundFour_iff_centeredResidue.mp (norm field digit)

/-- Source assignment decoded by the complete sparse substitution map. -/
def decodedAssignment {fieldCount : Nat}
    (layout : Layout fieldCount) (encoded : Nat → Nat) : Nat → Nat :=
  LinearSubstitution.assignment layout.expansion encoded

def loweredRows {fieldCount : Nat}
    (layout : Layout fieldCount) (sourceRows : List Row) : List Row :=
  sourceRows.map (LinearSubstitution.row layout.expansion)

private theorem foldl_range_eq_lowValue
    (digits : Nat → Nat) (start count : Nat) :
    (List.range count).foldl
        (fun total index => total + radix ^ index * digits index) start =
      start + lowValue digits count := by
  induction count with
  | zero => simp [lowValue]
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.foldl_append]
      simp only [List.foldl]
      rw [inductionHypothesis]
      rw [Nat.mul_comm (radix ^ count) (digits count)]
      change start + lowValue digits count + digits count * radix ^ count =
        start + (lowValue digits count + digits count * radix ^ count)
      omega

/-- Every designated source column is exactly the radix-seven decode of the
same 23 committed coordinates used by the transformed rows. -/
theorem decodedPrivateColumn
    {fieldCount : Nat} (layout : Layout fieldCount)
    (encoded : Nat → Nat) (field : Fin fieldCount) :
    decodedAssignment layout encoded (layout.sourceColumn field) =
      decodeFiniteWord (encodedWord layout encoded field) := by
  unfold decodedAssignment LinearSubstitution.assignment
  rw [layout.privateExpansion field]
  unfold decodeFiniteWord decodeWord
  have folded := foldl_range_eq_lowValue
    (fun index => encoded (encodedColumn layout field index)) 0 digitCount
  simpa [R1CS.lcEval, List.foldl_map,
    CenteredSeptenaryField.wordAt, encodedColumn] using
      congrArg (fun value => value % goldilocksP) folded

theorem loweredRows_iff_sourceRows
    {fieldCount : Nat} (layout : Layout fieldCount)
    (sourceRows : List Row) (encoded : Nat → Nat) :
    Satisfies (loweredRows layout sourceRows) encoded ↔
      Satisfies sourceRows (decodedAssignment layout encoded) := by
  exact LinearSubstitution.satisfies_mapped_iff sourceRows
    layout.expansion encoded

/-- Soundness for arbitrary norm-accepted words. No honest-encoder image is
assumed. -/
theorem loweredRows_sound
    {fieldCount : Nat} (layout : Layout fieldCount)
    (sourceRows : List Row) {encoded : Nat → Nat}
    (norm : PrivateCoordinatesNormBoundFour layout encoded)
    (accepted : Satisfies (loweredRows layout sourceRows) encoded) :
    Satisfies sourceRows (decodedAssignment layout encoded) ∧
      ∀ field,
        decodedAssignment layout encoded (layout.sourceColumn field) =
            decodeFiniteWord (encodedWord layout encoded field) ∧
          FiniteAlphabetWord (encodedWord layout encoded field) := by
  constructor
  · exact (loweredRows_iff_sourceRows layout sourceRows encoded).mp accepted
  · intro field
    exact ⟨decodedPrivateColumn layout encoded field,
      privateWordsAccepted_of_norm norm field⟩

/-- A fixed generated layout must provide a right inverse for all source
columns, not only the designated private tuple. -/
structure HonestMaterializer
    {fieldCount : Nat} (layout : Layout fieldCount) where
  encodeAssignment : (Nat → Nat) → Nat → Nat
  decode_encode : ∀ source,
    (∀ column, source column < goldilocksP) →
      decodedAssignment layout (encodeAssignment source) = source
  privateCoordinates : ∀ source field digit,
    encodeAssignment source (encodedColumn layout field digit.val) =
      finiteEncode (source (layout.sourceColumn field)) digit

theorem HonestMaterializer.private_norm
    {fieldCount : Nat} {layout : Layout fieldCount}
    (materializer : HonestMaterializer layout)
    (source : Nat → Nat) :
    PrivateCoordinatesNormBoundFour layout
      (materializer.encodeAssignment source) := by
  intro field digit
  rw [materializer.privateCoordinates source field digit]
  exact normBoundFour_iff_centeredResidue.mpr
    (finiteEncode_alphabet (source (layout.sourceColumn field)) digit)

theorem honest_complete
    {fieldCount : Nat} {layout : Layout fieldCount}
    (materializer : HonestMaterializer layout)
    (sourceRows : List Row) {source : Nat → Nat}
    (canonical : ∀ column, source column < goldilocksP)
    (accepted : Satisfies sourceRows source) :
    ∃ encoded,
      PrivateCoordinatesNormBoundFour layout encoded ∧
        Satisfies (loweredRows layout sourceRows) encoded ∧
        decodedAssignment layout encoded = source := by
  let encoded := materializer.encodeAssignment source
  have decoded : decodedAssignment layout encoded = source :=
    materializer.decode_encode source canonical
  refine ⟨encoded, materializer.private_norm source, ?_, decoded⟩
  apply (loweredRows_iff_sourceRows layout sourceRows encoded).mpr
  rw [decoded]
  exact accepted

/-! ## Exact typed fresh-CCS authority -/

/-- Numeric view of the exact typed assignment. Values outside the carrier
are zero, but every layout-owned word is proved to stay inside the carrier. -/
def typedEncodedNat {shape : Shape}
    (assignment : Assignment shape) : Nat → Nat :=
  fun column =>
    if within : column < shape.carrierWidth then
      (assignment ⟨column, within⟩).val
    else 0

theorem typedEncodedNat_of_lt
    {shape : Shape} (assignment : Assignment shape)
    {column : Nat} (within : column < shape.carrierWidth) :
    typedEncodedNat assignment column =
      (assignment ⟨column, within⟩).val := by
  simp [typedEncodedNat, within]

/-- One fresh radix-four CCS opening supplies the private-word norm for the
exact same assignment consumed by the transformed rows. -/
theorem privateNorm_of_fresh_ccsHolds
    {fieldCount : Nat} (layout : Layout fieldCount)
    {shape : Shape}
    (widthExact : layout.words.encodedColumnCount = shape.carrierWidth)
    {Commitment : Type uCommitment}
    (commit : Assignment shape → Commitment)
    (params : GlobalParams)
    (baseFour : params.b = 4)
    (statement : CCSStatement shape Commitment)
    (fresh : statement.stage = .fresh)
    (assignment : Assignment shape)
    (holds : CCS.Holds (relationSemantics commit) params statement assignment) :
    PrivateCoordinatesNormBoundFour layout (typedEncodedNat assignment) := by
  have outerNorm :=
    CenteredSeptenaryFreshCcsAuthority.norm_four_of_fresh_ccsHolds
      commit params baseFour statement fresh assignment holds
  intro field digit
  have within : encodedColumn layout field digit.val < shape.carrierWidth := by
    rw [← widthExact]
    have fits := layout.words.wordFits field
    unfold encodedColumn
    omega
  rw [typedEncodedNat_of_lt assignment within]
  apply normBoundFour_iff_centeredResidue.mpr
  exact (concrete_norm_four_iff_centeredResidue
    (assignment ⟨encodedColumn layout field digit.val, within⟩)).mp
      (outerNorm ⟨encodedColumn layout field digit.val, within⟩)

/-- End-to-end model boundary for the row family: transformed rows plus one
fresh radix-four CCS opening recover the source rows and every exact private
word from one assignment. -/
theorem freshCcs_loweredRows_sound
    {fieldCount : Nat} (layout : Layout fieldCount)
    (sourceRows : List Row)
    {shape : Shape}
    (widthExact : layout.words.encodedColumnCount = shape.carrierWidth)
    {Commitment : Type uCommitment}
    (commit : Assignment shape → Commitment)
    (params : GlobalParams)
    (baseFour : params.b = 4)
    (statement : CCSStatement shape Commitment)
    (fresh : statement.stage = .fresh)
    (assignment : Assignment shape)
    (holds : CCS.Holds (relationSemantics commit) params statement assignment)
    (accepted : Satisfies (loweredRows layout sourceRows)
      (typedEncodedNat assignment)) :
    Satisfies sourceRows
        (decodedAssignment layout (typedEncodedNat assignment)) ∧
      ∀ field,
        decodedAssignment layout (typedEncodedNat assignment)
              (layout.sourceColumn field) =
            decodeFiniteWord
              (encodedWord layout (typedEncodedNat assignment) field) ∧
          FiniteAlphabetWord
            (encodedWord layout (typedEncodedNat assignment) field) := by
  exact loweredRows_sound layout sourceRows
    (privateNorm_of_fresh_ccsHolds layout widthExact commit params baseFour
      statement fresh assignment holds) accepted

end Nightstream.Implementation.R1CS.CenteredSeptenaryLinearCompiler
