import Nightstream.Implementation.R1CS.Core.LinearSubstitution
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.NormDischarged

/-!
Contract: generic compiler theorem for a finite tuple of exact 41-word
ordinary private fields.

Owns: a proof-carrying source/encoded slot layout, parsing of the exact words
under the external `b = 2` coordinate norm, exact linear substitution into
arbitrary source R1CS rows, and soundness/completeness relative to an honest
assignment materializer.

Does not own: the fixed F-prime slot census, production column numbers, Rust
emission, an outer-CE norm bridge, public bits, canonical-u64 interfaces,
SIS-authoritative openings, or a claim that the current 222,762 profiler count
is the eligible tuple length.

Emits constraints: no. Source rows are transformed by the already-proved
`LinearSubstitution.row`; the external norm discharges word acceptance.

Authority boundary: exact encoded words are retained by the parser and remain
commitment authority. Decoded source residues are only the values consumed by
the source rows.

| Branch | Mathematical obligation | Main result | Tier |
|---|---|---|---|
| layout | each private source expands to 41 weighted coordinates | `Layout.privateExpansion` | refinement interface |
| parser | norm-authorized words parse/re-emit exactly | `reemit_parsed_projection` | kernel model |
| source decode | substituted source column equals word decode | `decodedPrivateColumn` | kernel model |
| arbitrary rows | mapped rows iff source rows on decoded assignment | `loweredRows_iff_sourceRows` | kernel model |
| honest completeness | a right-inverse materializer lifts any source witness | `honest_complete` | conditional generic theorem |
-/

namespace Nightstream.Implementation.R1CS.CenteredTernaryLinearCompiler

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CenteredTernaryField
open Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernarySound

set_option maxRecDepth 262144

/-- Proof-carrying placement of a finite tuple of private source fields into
an otherwise arbitrary linear-substitution layout. `encodedColumn` is queried
only below `digitCount`; keeping it Nat-indexed makes the emitted sparse list
definition direct. -/
structure Layout (fieldCount : Nat) where
  sourceColumn : Fin fieldCount → Nat
  encodedColumn : Fin fieldCount → Nat → Nat
  expansion : LinearSubstitution.ColumnExpansion
  privateExpansion : ∀ field,
    expansion (sourceColumn field) =
      (List.range digitCount).map fun index =>
        (encodedColumn field index, 3 ^ index % goldilocksP)

/-- Exact private word projected from an encoded assignment. -/
def encodedWord {fieldCount : Nat} (layout : Layout fieldCount)
    (encoded : Nat → Nat) (field : Fin fieldCount) : FiniteWord :=
  fun digit => encoded (layout.encodedColumn field digit.val)

/-- The coordinate-wise projection of the authoritative outer `b = 2` norm
onto exactly the ordinary private fields owned by this layout. -/
def PrivateCoordinatesNormBoundTwo {fieldCount : Nat}
    (layout : Layout fieldCount) (encoded : Nat → Nat) : Prop :=
  ∀ (field : Fin fieldCount) (digit : Fin digitCount),
    NormBoundTwo (encoded (layout.encodedColumn field digit.val))

theorem privateWordsAccepted_of_norm
    {fieldCount : Nat} {layout : Layout fieldCount}
    {encoded : Nat → Nat}
    (norm : PrivateCoordinatesNormBoundTwo layout encoded) :
    PrivateWordsAccepted (fun field => encodedWord layout encoded field) := by
  intro field digit
  exact normBoundTwo_iff_centeredResidue.mp (norm field digit)

/-- Exact accepted projection used as the input to the two-sided chosen-word
parser. -/
def acceptedProjection
    {fieldCount : Nat} (layout : Layout fieldCount)
    (encoded : Nat → Nat)
    (norm : PrivateCoordinatesNormBoundTwo layout encoded) :
    AcceptedPrivateEncoding fieldCount :=
  ⟨fun field => encodedWord layout encoded field,
    privateWordsAccepted_of_norm norm⟩

/-- Parse while retaining every exact coordinate choice. -/
def parse
    {fieldCount : Nat} (layout : Layout fieldCount)
    (encoded : Nat → Nat)
    (norm : PrivateCoordinatesNormBoundTwo layout encoded) :
    ChosenPrivateWitness fieldCount :=
  decodeChosenPrivate (acceptedProjection layout encoded norm)

/-- HyperNova H.2-shaped direction for the generic finite layout: parsing an
arbitrary norm-accepted projection and re-emitting it returns the same exact
word tuple. -/
theorem reemit_parsed_projection
    {fieldCount : Nat} (layout : Layout fieldCount)
    (encoded : Nat → Nat)
    (norm : PrivateCoordinatesNormBoundTwo layout encoded) :
    encodeChosenPrivate (parse layout encoded norm) =
      acceptedProjection layout encoded norm := by
  exact encodeChosenPrivate_decodeChosenPrivate _

theorem parsed_coordinate_exact
    {fieldCount : Nat} (layout : Layout fieldCount)
    (encoded : Nat → Nat)
    (norm : PrivateCoordinatesNormBoundTwo layout encoded)
    (field : Fin fieldCount) (digit : Fin digitCount) :
    (parse layout encoded norm).words field digit =
      encoded (layout.encodedColumn field digit.val) := by
  rfl

/-- Source assignment decoded by the complete sparse substitution map. -/
def decodedAssignment {fieldCount : Nat} (layout : Layout fieldCount)
    (encoded : Nat → Nat) : Nat → Nat :=
  LinearSubstitution.assignment layout.expansion encoded

/-- Exact lowering of arbitrary source rows. -/
def loweredRows {fieldCount : Nat} (layout : Layout fieldCount)
    (sourceRows : List Row) : List Row :=
  sourceRows.map (LinearSubstitution.row layout.expansion)

/-- Each designated private source column is exactly the radix-three decode
of the retained word at that slot. -/
theorem decodedPrivateColumn
    {fieldCount : Nat} (layout : Layout fieldCount)
    (encoded : Nat → Nat) (field : Fin fieldCount) :
    decodedAssignment layout encoded (layout.sourceColumn field) =
      decodeFiniteWord (encodedWord layout encoded field) := by
  unfold decodedAssignment LinearSubstitution.assignment
  rw [layout.privateExpansion field]
  unfold decodeFiniteWord decodeWord
  have folded := foldl_range_eq_lowValue
    (fun index => encoded (layout.encodedColumn field index)) 0 digitCount
  simpa [lcEval, List.foldl_map, CenteredTernaryField.wordAt] using
    congrArg (fun value => value % goldilocksP) folded

theorem decodedPrivateColumn_eq_parsedSource
    {fieldCount : Nat} (layout : Layout fieldCount)
    (encoded : Nat → Nat)
    (norm : PrivateCoordinatesNormBoundTwo layout encoded)
    (field : Fin fieldCount) :
    decodedAssignment layout encoded (layout.sourceColumn field) =
      (parse layout encoded norm).sources field := by
  rw [decodedPrivateColumn]
  rfl

/-- Exact soundness/completeness for any finite list of source R1CS rows. No
row-shape assumption is needed beyond ordinary R1CS semantics. -/
theorem loweredRows_iff_sourceRows
    {fieldCount : Nat} (layout : Layout fieldCount)
    (sourceRows : List Row) (encoded : Nat → Nat) :
    Satisfies (loweredRows layout sourceRows) encoded ↔
      Satisfies sourceRows (decodedAssignment layout encoded) := by
  exact LinearSubstitution.satisfies_mapped_iff
    sourceRows layout.expansion encoded

/-- Soundness with the exact parser boundary made explicit. -/
theorem loweredRows_sound
    {fieldCount : Nat} (layout : Layout fieldCount)
    (sourceRows : List Row) {encoded : Nat → Nat}
    (norm : PrivateCoordinatesNormBoundTwo layout encoded)
    (accepted : Satisfies (loweredRows layout sourceRows) encoded) :
    Satisfies sourceRows (decodedAssignment layout encoded) ∧
      ∀ field,
        decodedAssignment layout encoded (layout.sourceColumn field) =
          (parse layout encoded norm).sources field := by
  constructor
  · exact (loweredRows_iff_sourceRows layout sourceRows encoded).mp accepted
  · exact decodedPrivateColumn_eq_parsedSource layout encoded norm

/-- A fixed layout must separately provide this right inverse. It includes
all non-private columns, so generic field encoding cannot manufacture it from
the private tuple alone. -/
structure HonestMaterializer
    {fieldCount : Nat} (layout : Layout fieldCount) where
  encodeAssignment : (Nat → Nat) → Nat → Nat
  decode_encode : ∀ source,
    (∀ column, source column < goldilocksP) →
      decodedAssignment layout (encodeAssignment source) = source
  privateCoordinates : ∀ source field digit,
    encodeAssignment source (layout.encodedColumn field digit.val) =
      finiteEncode (source (layout.sourceColumn field)) digit

theorem HonestMaterializer.private_norm
    {fieldCount : Nat} {layout : Layout fieldCount}
    (materializer : HonestMaterializer layout)
    (source : Nat → Nat) :
    PrivateCoordinatesNormBoundTwo layout
      (materializer.encodeAssignment source) := by
  unfold PrivateCoordinatesNormBoundTwo
  intro field digit
  rw [materializer.privateCoordinates source field digit]
  exact normBoundTwo_iff_centeredResidue.mpr
    (finiteEncode_alphabet (source (layout.sourceColumn field)) digit)

/-- Generic honest completeness for arbitrary source rows. Fixed F-prime must
instantiate `Layout` and `HonestMaterializer` from generated evidence; this
theorem contains no hard-coded field count. -/
theorem honest_complete
    {fieldCount : Nat} {layout : Layout fieldCount}
    (materializer : HonestMaterializer layout)
    (sourceRows : List Row) {source : Nat → Nat}
    (canonical : ∀ column, source column < goldilocksP)
    (accepted : Satisfies sourceRows source) :
    ∃ encoded,
      PrivateCoordinatesNormBoundTwo layout encoded ∧
      Satisfies (loweredRows layout sourceRows) encoded ∧
      decodedAssignment layout encoded = source := by
  let encoded := materializer.encodeAssignment source
  have decoded : decodedAssignment layout encoded = source :=
    materializer.decode_encode source canonical
  refine ⟨encoded, materializer.private_norm source, ?_, decoded⟩
  apply (loweredRows_iff_sourceRows layout sourceRows encoded).mpr
  rw [decoded]
  exact accepted

end Nightstream.Implementation.R1CS.CenteredTernaryLinearCompiler
