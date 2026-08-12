import Nightstream.Implementation.NebulaV2.Memory.Carry.Parser
import Nightstream.Implementation.NebulaV2.Core.PublicBitBlock

/-!
Contract: complete public-bit row bridge for one exact V2 recursive memory
carry.

Assurance tier: implementation model.

Owns the 3,433 consecutive authority-bearing public bits, all counter and
closed-state rows, all 52 canonical Goldilocks limb blocks, and derivation of
the exact typed carry in every circuit value column from the fail-closed
parser result.

Does not own the state-hash permutation, enclosing recursive state, absolute
generated columns, or Rust/container refinement. Verifier-owned chain-header
column placement remains an explicit trust-boundary input.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryCarryPublicRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2.MemoryCarryCodec
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime

abbrev ParserCounter := MemoryCarryParser.Counter
abbrev RegularCounter := MemoryCarryRows.RegularCounter

def RegularCounter.parser : RegularCounter → ParserCounter
  | .phase => .phase
  | .segmentIndex => .segmentIndex
  | .globalTimestamp => .globalTimestamp
  | .segmentStartTimestamp => .segmentStartTimestamp
  | .segmentActiveAccessCount => .segmentActiveAccessCount
  | .segmentEndTimestamp => .segmentEndTimestamp

theorem RegularCounter.parser_width (counter : RegularCounter) :
    counter.parser.width = counter.width := by
  cases counter <;> rfl

theorem RegularCounter.parser_offset (counter : RegularCounter) :
    counter.parser.bitOffset = counter.bitOffset := by
  cases counter <;> rfl

theorem RegularCounter.parser_tag (counter : RegularCounter) :
    counter.parser.tag = counter.tag := by
  cases counter <;> rfl

structure Layout where
  carry : MemoryCarryRows.Layout
  fieldColumnMap : MemoryCarryFieldRows.Slot → List Nat
  fieldMapsConstantOne : ∀ slot,
    Relabel.column (fieldColumnMap slot) 0 = 0
  fieldValueColumn : ∀ slot,
    Relabel.column (fieldColumnMap slot) CanonicalU64.varCol =
      carry.fieldColumn slot.tag

def Layout.publicBits (layout : Layout) : PublicBitBlock.Layout :=
  { publicBitStart := layout.carry.publicBitStart }

def Layout.fields (layout : Layout) : MemoryCarryFieldRows.Layout :=
  { publicBitStart := layout.carry.publicBitStart
    columnMap := layout.fieldColumnMap
    mapsConstantOne := layout.fieldMapsConstantOne }

def Layout.parserWord (layout : Layout) (counter : ParserCounter) :
    BoundedWordRows.Layout :=
  { width := counter.width
    valueColumn := layout.carry.fieldColumn counter.tag
    bitStart := layout.carry.publicBitStart + counter.bitOffset }

def rows (layout : Layout) : List Row :=
  MemoryCarryRows.rows layout.carry ++
    MemoryCarryFieldRows.rows layout.fields

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 7094 := by
  rw [rows, List.length_append, MemoryCarryRows.rows_length_exact,
    MemoryCarryFieldRows.rows_length_exact]

private theorem carry_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (MemoryCarryRows.rows layout.carry) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem field_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (MemoryCarryFieldRows.rows layout.fields) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

theorem counter_digits_eq_block
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (counter : ParserCounter) :
    (layout.parserWord counter).digits assignment =
      (MemoryCarryParser.counterWord block counter).val := by
  symm
  change
    (FixedBits.slice block counter.bitOffset counter.width _).val =
      (List.range counter.width).map fun index =>
        assignment
          (layout.carry.publicBitStart + counter.bitOffset + index)
  exact PublicBitBlock.slice_eq_columns placed counter.bitOffset counter.width _

private theorem regular_column_eq_parser_word
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment)
    (counter : RegularCounter) :
    assignment (layout.carry.fieldColumn counter.tag) =
      MemoryCarryParser.counterValue block counter.parser := by
  have recomposed := BoundedWordRows.recomposition_sound
    counter.fitsGoldilocks canonical one
    (MemoryCarryRows.regular_rows_hold (carry_rows_hold holds) counter)
  have adjusted :
      assignment (layout.carry.fieldColumn counter.tag) =
        BoundedWordRows.decoded (layout.parserWord counter.parser)
          assignment := by
    simpa [Layout.parserWord, MemoryCarryRows.Layout.regularWord,
      RegularCounter.parser_width, RegularCounter.parser_offset,
      RegularCounter.parser_tag] using recomposed
  calc
    assignment (layout.carry.fieldColumn counter.tag) =
        BoundedWordRows.decoded (layout.parserWord counter.parser)
          assignment := adjusted
    _ = Nat.ofDigits 2
        ((layout.parserWord counter.parser).digits assignment) := rfl
    _ = Nat.ofDigits 2
        (MemoryCarryParser.counterWord block counter.parser).val :=
      congrArg (Nat.ofDigits 2)
        (counter_digits_eq_block placed counter.parser)
    _ = MemoryCarryParser.counterValue block counter.parser := rfl

private theorem step_column_eq_parser_word
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment) :
    assignment (layout.carry.fieldColumn .stepIndex) =
      MemoryCarryParser.counterValue block .stepIndex := by
  have stepRows := MemoryCarryRows.step_rows_hold (carry_rows_hold holds)
  have valueRows := LessThanConstantRows.value_rows_hold stepRows
  have fits : 2 ^ MemoryCarryParser.Counter.width (.stepIndex) ≤
      goldilocksP := by decide
  have recomposed := BoundedWordRows.recomposition_sound fits canonical one
    valueRows
  have adjusted :
      assignment (layout.carry.fieldColumn .stepIndex) =
        BoundedWordRows.decoded (layout.parserWord .stepIndex) assignment := by
    simpa [Layout.parserWord, MemoryCarryRows.Layout.stepWord,
      LessThanConstantRows.Layout.valueWord] using recomposed
  calc
    assignment (layout.carry.fieldColumn .stepIndex) =
        BoundedWordRows.decoded (layout.parserWord .stepIndex) assignment :=
      adjusted
    _ = Nat.ofDigits 2
        ((layout.parserWord .stepIndex).digits assignment) := rfl
    _ = Nat.ofDigits 2
        (MemoryCarryParser.counterWord block .stepIndex).val :=
      congrArg (Nat.ofDigits 2)
        (counter_digits_eq_block placed .stepIndex)
    _ = MemoryCarryParser.counterValue block .stepIndex := rfl

/-- Every counter value column is fixed by the exact parser word. -/
theorem counter_column_eq_parser_word
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment)
    (counter : ParserCounter) :
    assignment (layout.carry.fieldColumn counter.tag) =
      MemoryCarryParser.counterValue block counter := by
  cases counter with
  | phase => exact regular_column_eq_parser_word canonical one placed holds .phase
  | segmentIndex =>
      exact regular_column_eq_parser_word canonical one placed holds .segmentIndex
  | stepIndex => exact step_column_eq_parser_word canonical one placed holds
  | globalTimestamp =>
      exact regular_column_eq_parser_word canonical one placed holds
        .globalTimestamp
  | segmentStartTimestamp =>
      exact regular_column_eq_parser_word canonical one placed holds
        .segmentStartTimestamp
  | segmentActiveAccessCount =>
      exact regular_column_eq_parser_word canonical one placed holds
        .segmentActiveAccessCount
  | segmentEndTimestamp =>
      exact regular_column_eq_parser_word canonical one placed holds
        .segmentEndTimestamp

theorem fields_place_parser_words
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    (placed : PublicBitBlock.Placed layout.publicBits assignment block) :
    MemoryCarryFieldRows.Places layout.fields assignment
      (MemoryCarryParser.rawWords block) := by
  intro slot
  change
    (MemoryCarryParser.fieldWord block slot).val =
      CanonicalFieldSchemaRows.rawDigits layout.fields.schema assignment slot
  change
    (FixedBits.slice block slot.bitOffset CanonicalFieldBits.bitCount
      (MemoryCarryParser.field_slice_fits slot)).val =
      CanonicalFieldSchemaRows.rawDigits layout.fields.schema assignment slot
  rw [PublicBitBlock.slice_eq_columns placed slot.bitOffset
    CanonicalFieldBits.bitCount (MemoryCarryParser.field_slice_fits slot)]
  simp [PublicBitBlock.sliceColumns,
    CanonicalFieldSchemaRows.rawDigits,
    MemoryCarryFieldRows.Layout.schema,
    MemoryCarryFieldRows.Layout.rawColumns, Layout.publicBits, Layout.fields,
    List.getD]
  intro index bound
  have rangeBound : index < (List.range CanonicalFieldBits.bitCount).length := by
    simpa using bound
  rw [List.getElem?_eq_getElem rangeBound]
  simp

/-- Satisfying canonical-field rows force all 52 parser words below the
Goldilocks modulus. This is a row conclusion, not a native-parser premise. -/
private theorem fieldsCanonical_of_rows
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment) :
    MemoryCarryParser.fieldsCanonical block = true := by
  rw [MemoryCarryParser.fieldsCanonical, List.all_eq_true]
  intro slot _member
  rcases MemoryCarryFieldRows.rows_force_native_acceptance canonical one
      (field_rows_hold holds) (fields_place_parser_words placed) slot with
    ⟨value, accepted, _⟩
  exact decide_eq_true
    ((FieldCodec.nativeDecode_some_iff _ value).mp accepted).1

/-- The deterministic typed value decoded from the public bits occupies all
carry value columns. No caller supplies a parsed value. -/
private theorem decodedValue_placed
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment) :
    MemoryCarryRows.Placed layout.carry assignment
      (MemoryCarryParser.decodedValue block) := by
  have fieldsCanonical := fieldsCanonical_of_rows canonical one placed holds
  have nativeParses : MemoryCarryFieldRows.NativeParses
      (MemoryCarryParser.rawWords block)
      (MemoryCarryParser.decodedValue block) := by
    intro slot
    rw [MemoryCarryParser.rawWords,
      MemoryCarryParser.decodedValue_canonicalValue]
    exact MemoryCarryParser.nativeDecode_field fieldsCanonical slot
  have fieldColumns := MemoryCarryFieldRows.typed_columns_of_rows canonical one
    (field_rows_hold holds) (fields_place_parser_words placed) nativeParses
  intro tag
  cases tag with
  | phase =>
      exact (counter_column_eq_parser_word canonical one placed holds
        .phase).trans
          (MemoryCarryParser.decodedValue_counterValue block .phase).symm
  | segmentIndex =>
      exact (counter_column_eq_parser_word canonical one placed holds
        .segmentIndex).trans
          (MemoryCarryParser.decodedValue_counterValue block
            .segmentIndex).symm
  | stepIndex =>
      exact (counter_column_eq_parser_word canonical one placed holds
        .stepIndex).trans
          (MemoryCarryParser.decodedValue_counterValue block .stepIndex).symm
  | globalTimestamp =>
      exact (counter_column_eq_parser_word canonical one placed holds
        .globalTimestamp).trans
          (MemoryCarryParser.decodedValue_counterValue block
            .globalTimestamp).symm
  | segmentStartTimestamp =>
      exact (counter_column_eq_parser_word canonical one placed holds
        .segmentStartTimestamp).trans
          (MemoryCarryParser.decodedValue_counterValue block
            .segmentStartTimestamp).symm
  | segmentActiveAccessCount =>
      exact (counter_column_eq_parser_word canonical one placed holds
        .segmentActiveAccessCount).trans
          (MemoryCarryParser.decodedValue_counterValue block
            .segmentActiveAccessCount).symm
  | segmentEndTimestamp =>
      exact (counter_column_eq_parser_word canonical one placed holds
        .segmentEndTimestamp).trans
          (MemoryCarryParser.decodedValue_counterValue block
            .segmentEndTimestamp).symm
  | challenge repetition coordinate limb =>
      let slot : MemoryCarryFieldRows.Slot :=
        .challenge repetition coordinate limb
      calc
        assignment (layout.carry.fieldColumn slot.tag) =
            assignment
              (Relabel.column (layout.fieldColumnMap slot)
                CanonicalU64.varCol) :=
          congrArg assignment (layout.fieldValueColumn slot).symm
        _ = (MemoryCarryParser.decodedValue block).fieldValue slot.tag :=
          fieldColumns slot
  | product repetition role limb =>
      let slot : MemoryCarryFieldRows.Slot := .product repetition role limb
      calc
        assignment (layout.carry.fieldColumn slot.tag) =
            assignment
              (Relabel.column (layout.fieldColumnMap slot)
                CanonicalU64.varCol) :=
          congrArg assignment (layout.fieldValueColumn slot).symm
        _ = (MemoryCarryParser.decodedValue block).fieldValue slot.tag :=
          fieldColumns slot
  | root source lane =>
      let slot : MemoryCarryFieldRows.Slot := .root source lane
      calc
        assignment (layout.carry.fieldColumn slot.tag) =
            assignment
              (Relabel.column (layout.fieldColumnMap slot)
                CanonicalU64.varCol) :=
          congrArg assignment (layout.fieldValueColumn slot).symm
        _ = (MemoryCarryParser.decodedValue block).fieldValue slot.tag :=
          fieldColumns slot

/-- **The 7,094 carry-parser rows force native parser acceptance.**

The conclusion fixes the returned value to the deterministic decode of the
same 3,433 public bits. There is no parser-success or canonical-carry premise. -/
theorem rows_force_parse
    {layout : Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (headersPlaced : MemoryCarryRows.HeadersPlaced layout.carry assignment
      headers)
    (holds : Satisfies (rows layout) assignment) :
    MemoryCarryParser.parse headers block =
      some (MemoryCarryParser.decodedValue block) := by
  have valuePlaced := decodedValue_placed canonical one placed holds
  have valueCanonical := MemoryCarryRows.value_canonical_of_rows canonical one
    valuePlaced headersPlaced (carry_rows_hold holds)
  have stepBound : MemoryCarryParser.counterValue block .stepIndex <
      Lifecycle.claimsPerSegment := by
    rw [← MemoryCarryParser.decodedValue_counterValue block .stepIndex]
    exact valueCanonical.stepIndex
  have fieldsCanonical := fieldsCanonical_of_rows canonical one placed holds
  have closed : MemoryCarryParser.closedCheck headers
      (MemoryCarryParser.decodedValue block) :=
    valueCanonical.closedFields
  unfold MemoryCarryParser.parse
  rw [dif_pos stepBound, dif_pos fieldsCanonical, dif_pos closed]

theorem field_columns_eq_parsed_value
    {layout : Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {block : MemoryCarryParser.Block} {value : Value}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment)
    (accepted : MemoryCarryParser.parse headers block = some value) :
    ∀ slot,
      assignment
          (Relabel.column (layout.fieldColumnMap slot) CanonicalU64.varCol) =
        value.fieldValue slot.tag := by
  exact MemoryCarryFieldRows.typed_columns_of_rows canonical one
    (field_rows_hold holds) (fields_place_parser_words placed)
    (MemoryCarryParser.parse_native_parses accepted)

/-- The typed carry placement required by the counter/closed-state block is
derived from parser bits and rows. -/
theorem carry_places_parsed_value
    {layout : Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {block : MemoryCarryParser.Block} {value : Value}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment)
    (accepted : MemoryCarryParser.parse headers block = some value) :
    MemoryCarryRows.Placed layout.carry assignment value := by
  intro tag
  cases tag with
  | phase =>
      exact (counter_column_eq_parser_word canonical one placed holds .phase).trans
        (MemoryCarryParser.parse_counterValue accepted .phase).symm
  | segmentIndex =>
      exact (counter_column_eq_parser_word canonical one placed holds
        .segmentIndex).trans
          (MemoryCarryParser.parse_counterValue accepted .segmentIndex).symm
  | stepIndex =>
      exact (counter_column_eq_parser_word canonical one placed holds
        .stepIndex).trans
          (MemoryCarryParser.parse_counterValue accepted .stepIndex).symm
  | globalTimestamp =>
      exact (counter_column_eq_parser_word canonical one placed holds
        .globalTimestamp).trans
          (MemoryCarryParser.parse_counterValue accepted .globalTimestamp).symm
  | segmentStartTimestamp =>
      exact (counter_column_eq_parser_word canonical one placed holds
        .segmentStartTimestamp).trans
          (MemoryCarryParser.parse_counterValue accepted
            .segmentStartTimestamp).symm
  | segmentActiveAccessCount =>
      exact (counter_column_eq_parser_word canonical one placed holds
        .segmentActiveAccessCount).trans
          (MemoryCarryParser.parse_counterValue accepted
            .segmentActiveAccessCount).symm
  | segmentEndTimestamp =>
      exact (counter_column_eq_parser_word canonical one placed holds
        .segmentEndTimestamp).trans
          (MemoryCarryParser.parse_counterValue accepted
            .segmentEndTimestamp).symm
  | challenge repetition coordinate limb =>
      let slot : MemoryCarryFieldRows.Slot :=
        .challenge repetition coordinate limb
      calc
        assignment (layout.carry.fieldColumn slot.tag) =
            assignment
              (Relabel.column (layout.fieldColumnMap slot)
                CanonicalU64.varCol) :=
          congrArg assignment (layout.fieldValueColumn slot).symm
        _ = value.fieldValue slot.tag :=
          field_columns_eq_parsed_value canonical one placed holds accepted slot
  | product repetition role limb =>
      let slot : MemoryCarryFieldRows.Slot := .product repetition role limb
      calc
        assignment (layout.carry.fieldColumn slot.tag) =
            assignment
              (Relabel.column (layout.fieldColumnMap slot)
                CanonicalU64.varCol) :=
          congrArg assignment (layout.fieldValueColumn slot).symm
        _ = value.fieldValue slot.tag :=
          field_columns_eq_parsed_value canonical one placed holds accepted slot
  | root source lane =>
      let slot : MemoryCarryFieldRows.Slot := .root source lane
      calc
        assignment (layout.carry.fieldColumn slot.tag) =
            assignment
              (Relabel.column (layout.fieldColumnMap slot)
                CanonicalU64.varCol) :=
          congrArg assignment (layout.fieldValueColumn slot).symm
        _ = value.fieldValue slot.tag :=
          field_columns_eq_parsed_value canonical one placed holds accepted slot

structure ParsedColumnsMatch
    (layout : Layout) (assignment : Nat → Nat)
    (headers : ChainHeaders Digest.Value) (value : Value) : Prop where
  placed : MemoryCarryRows.Placed layout.carry assignment value
  headersPlaced : MemoryCarryRows.HeadersPlaced layout.carry assignment
    headers
  rowCanonical : value.Canonical headers
  parserCanonical : value.Canonical headers

/-- Complete local carry soundness. Closed-state canonicality is derived
both by the parser and by the concrete conditional rows. -/
theorem parsed_columns_match
    {layout : Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {block : MemoryCarryParser.Block} {value : Value}
    (canonicalAssignment : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (headersPlaced : MemoryCarryRows.HeadersPlaced layout.carry assignment
      headers)
    (holds : Satisfies (rows layout) assignment)
    (accepted : MemoryCarryParser.parse headers block = some value) :
    ParsedColumnsMatch layout assignment headers value := by
  have valuePlaced := carry_places_parsed_value canonicalAssignment one placed
    holds accepted
  exact
    { placed := valuePlaced
      headersPlaced := headersPlaced
      rowCanonical := MemoryCarryRows.value_canonical_of_rows
        canonicalAssignment one valuePlaced headersPlaced (carry_rows_hold holds)
      parserCanonical := MemoryCarryParser.parse_value_canonical accepted }

/-- Complete local carry parsing is a conclusion of public-bit placement,
header placement, and the exact 7,094 rows. -/
theorem rows_force_parsed_columns_match
    {layout : Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (headersPlaced : MemoryCarryRows.HeadersPlaced layout.carry assignment
      headers)
    (holds : Satisfies (rows layout) assignment) :
    ParsedColumnsMatch layout assignment headers
      (MemoryCarryParser.decodedValue block) := by
  exact parsed_columns_match canonical one placed headersPlaced holds
    (rows_force_parse canonical one placed headersPlaced holds)

end Nightstream.Implementation.NebulaV2.MemoryCarryPublicRows
