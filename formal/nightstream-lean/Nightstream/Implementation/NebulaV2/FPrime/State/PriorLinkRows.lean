import Nightstream.Implementation.NebulaV2.FPrime.State.AuthoritativeOutputRows
import Nightstream.Implementation.NebulaV2.Core.CanonicalFieldSchemaRows
import Nightstream.Implementation.NebulaV2.FPrime.Claim.EnvelopeRows
import Nightstream.Implementation.NebulaV2.Memory.Claim.BoundCcsPublic
import Nightstream.Implementation.NebulaV2.Memory.Claim.PoseidonRows
import Nightstream.Implementation.R1CS.Core.ConstantPins

/-!
Contract: exact recursive-link rows from one parsed incoming V2 memory carry
and memory suffix, plus one typed prior non-memory state, to the selected
540-coordinate fresh CCS public image.

Assurance tier: implementation model and cryptographic primitive semantics.

Owns the complete prior state-output Poseidon2 computation, canonical
decomposition of all four state-output lanes, the complete memory-suffix
Poseidon2 computation, canonical decomposition of its four output lanes, the
affine-one coordinate, all 27 zero-padding coordinates, and the exact source
columns inside the complete fresh-claim envelope.

Does not own the incoming carry or memory-claim parser rows, the transition
that produced the prior non-memory payload, NIFS verification, Poseidon2
collision resistance, absolute generated columns, or Rust conformance.

Emits constraints: yes.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.NebulaV2.PriorStateLinkRows

open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CanonicalU64
open Nightstream.Protocol.NebulaV2

def ccsPublicBitCount : Nat := MemoryBoundCcsPublic.coordinateCount
def digestBitCount : Nat := MemoryBoundCcsPublic.digestBitCount
def memoryDigestBitCount : Nat := MemoryBoundCcsPublic.digestBitCount
def paddingBitCount : Nat := MemoryBoundCcsPublic.paddingBitCount

theorem exactCarrierGeometry :
    ccsPublicBitCount =
        1 + digestBitCount + memoryDigestBitCount + paddingBitCount ∧
      digestBitCount = 4 * CanonicalFieldBits.bitCount ∧
      memoryDigestBitCount = 4 * CanonicalFieldBits.bitCount := by
  decide

/-- Exact generated-column layout. The canonical-u64 maps allocate their
local helper columns, but their value columns are the four Poseidon2 outputs. -/
structure Layout (widths : CompilerWidths) where
  fullClaim : FullClaimEnvelopeRows.Layout widths
  stateOutput : AuthoritativeStateOutputRows.Layout
  memoryDigest : MemoryClaimPoseidonRows.Layout
  digestColumnMap : Fin 4 → List Nat
  digestMapsConstantOne : ∀ lane,
    Relabel.column (digestColumnMap lane) 0 = 0
  digestValueColumn : ∀ lane,
    Relabel.column (digestColumnMap lane) varCol =
      stateOutput.hash.stateOutput.trace.outputColumns.getD lane.val 0
  memoryDigestColumnMap : Fin 4 → List Nat
  memoryDigestMapsConstantOne : ∀ lane,
    Relabel.column (memoryDigestColumnMap lane) 0 = 0
  memoryDigestValueColumn : ∀ lane,
    Relabel.column (memoryDigestColumnMap lane) varCol =
      memoryDigest.trace.outputColumns.getD lane.val 0

def Layout.ccsBitColumn {widths : CompilerWidths}
    (layout : Layout widths) (offset : Nat) : Nat :=
  layout.fullClaim.claimBitStart + Section.ccsPublic.bitOffset widths + offset

def Layout.digestRawColumns {widths : CompilerWidths}
    (layout : Layout widths) (lane : Fin 4) : List Nat :=
  (List.range CanonicalFieldBits.bitCount).map fun bit =>
    layout.ccsBitColumn (1 + CanonicalFieldBits.bitCount * lane.val + bit)

theorem Layout.digestRawColumns_length {widths : CompilerWidths}
    (layout : Layout widths) (lane : Fin 4) :
    (layout.digestRawColumns lane).length = CanonicalFieldBits.bitCount := by
  simp [Layout.digestRawColumns]

def Layout.digestSchema {widths : CompilerWidths}
    (layout : Layout widths) : CanonicalFieldSchemaRows.Layout (Fin 4) where
  columnMap := layout.digestColumnMap
  rawColumns := layout.digestRawColumns
  rawColumnsLength := layout.digestRawColumns_length
  mapsConstantOne := layout.digestMapsConstantOne

def Layout.memoryDigestRawColumns {widths : CompilerWidths}
    (layout : Layout widths) (lane : Fin 4) : List Nat :=
  (List.range CanonicalFieldBits.bitCount).map fun bit =>
    layout.ccsBitColumn
      (1 + digestBitCount + CanonicalFieldBits.bitCount * lane.val + bit)

theorem Layout.memoryDigestRawColumns_length {widths : CompilerWidths}
    (layout : Layout widths) (lane : Fin 4) :
    (layout.memoryDigestRawColumns lane).length =
      CanonicalFieldBits.bitCount := by
  simp [Layout.memoryDigestRawColumns]

def Layout.memoryDigestSchema {widths : CompilerWidths}
    (layout : Layout widths) : CanonicalFieldSchemaRows.Layout (Fin 4) where
  columnMap := layout.memoryDigestColumnMap
  rawColumns := layout.memoryDigestRawColumns
  rawColumnsLength := layout.memoryDigestRawColumns_length
  mapsConstantOne := layout.memoryDigestMapsConstantOne

def digestSlots : List (Fin 4) := List.ofFn id

theorem digestSlots_length : digestSlots.length = 4 := by
  simp [digestSlots]

theorem digestSlot_mem (lane : Fin 4) : lane ∈ digestSlots := by
  fin_cases lane <;> simp [digestSlots]

def publicPins {widths : CompilerWidths}
    (layout : Layout widths) : List (Nat × Nat) :=
  [(layout.ccsBitColumn 0, 1)] ++
    (List.range paddingBitCount).map fun padding =>
      (layout.ccsBitColumn
        (1 + digestBitCount + memoryDigestBitCount + padding), 0)

def rows {widths : CompilerWidths} (layout : Layout widths) : List Row :=
  AuthoritativeStateOutputRows.rows layout.stateOutput ++
    MemoryClaimPoseidonRows.rows layout.memoryDigest ++
    CanonicalFieldSchemaRows.schemaRows digestSlots layout.digestSchema ++
    CanonicalFieldSchemaRows.schemaRows digestSlots layout.memoryDigestSchema ++
    ConstantPins.rows (publicPins layout)

/-- Pure structural validity. It contains no assignment, digest, or row
satisfaction conclusion. -/
structure Layout.Valid {widths : CompilerWidths}
    (layout : Layout widths) : Prop where
  stateOutputValid : layout.stateOutput.Valid
  memoryDigestValid : layout.memoryDigest.Valid
  exactCcsPublicWidth : widths.ccsPublicBits = ccsPublicBitCount

theorem publicPins_length {widths : CompilerWidths}
    (layout : Layout widths) : (publicPins layout).length = 28 := by
  simp [publicPins, paddingBitCount,
    MemoryBoundCcsPublic.paddingBitCount]

theorem publicPins_valuesCanonical {widths : CompilerWidths}
    (layout : Layout widths) :
    ConstantPins.ValuesCanonical (publicPins layout) := by
  intro pin member
  simp only [publicPins, List.mem_append, List.mem_singleton,
    List.mem_map] at member
  rcases member with rfl | ⟨padding, _paddingMember, rfl⟩
  · norm_num [goldilocksP]
  · norm_num [goldilocksP]

theorem rows_length_exact {widths : CompilerWidths}
    {layout : Layout widths} (valid : layout.Valid) :
    (rows layout).length = 40090 := by
  simp [rows,
    AuthoritativeStateOutputRows.rows_length_exact valid.stateOutputValid,
    MemoryClaimPoseidonRows.rows_length_exact valid.memoryDigestValid,
    CanonicalFieldSchemaRows.schemaRows_length, digestSlots_length,
    ConstantPins.rows, publicPins_length]

private theorem state_rows_hold
    {widths : CompilerWidths} {layout : Layout widths}
    {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (AuthoritativeStateOutputRows.rows layout.stateOutput)
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem memory_digest_rows_hold
    {widths : CompilerWidths} {layout : Layout widths}
    {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (MemoryClaimPoseidonRows.rows layout.memoryDigest)
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

/-- Public projection of the complete prior-state link to its independently
typed state-output subrelation. The lifetime owner uses this result across
different generated-column layouts. -/
theorem stateOutput_rows_hold
    {widths : CompilerWidths} {layout : Layout widths}
    {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (AuthoritativeStateOutputRows.rows layout.stateOutput)
      assignment :=
  state_rows_hold holds

private theorem digest_rows_hold
    {widths : CompilerWidths} {layout : Layout widths}
    {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies
      (CanonicalFieldSchemaRows.schemaRows digestSlots layout.digestSchema)
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem memory_digest_schema_rows_hold
    {widths : CompilerWidths} {layout : Layout widths}
    {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies
      (CanonicalFieldSchemaRows.schemaRows digestSlots
        layout.memoryDigestSchema) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem pin_rows_hold
    {widths : CompilerWidths} {layout : Layout widths}
    {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (ConstantPins.rows (publicPins layout)) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem selfIncluded (program : List Row) :
    rowsIncluded program program = true := by
  rw [rowsIncluded, List.all_eq_true]
  intro row member
  exact decide_eq_true member

theorem pinFacts
    {widths : CompilerWidths} {layout : Layout widths}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    ∀ pin ∈ publicPins layout, assignment pin.1 = pin.2 := by
  exact ConstantPins.sound (publicPins_valuesCanonical layout)
    (selfIncluded (ConstantPins.rows (publicPins layout))) canonical one
    (pin_rows_hold holds)

def outputColumn {widths : CompilerWidths}
    (layout : Layout widths) (lane : Fin 4) : Nat :=
  layout.stateOutput.hash.stateOutput.trace.outputColumns.getD lane.val 0

def outputDigest {widths : CompilerWidths}
    (layout : Layout widths) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    Fin 4 → ShiftedTernary41V1.CanonicalGoldilocks :=
  fun lane => ⟨assignment (outputColumn layout lane), canonical _⟩

def memoryOutputColumn {widths : CompilerWidths}
    (layout : Layout widths) (lane : Fin 4) : Nat :=
  layout.memoryDigest.trace.outputColumns.getD lane.val 0

def outputMemoryDigest {widths : CompilerWidths}
    (layout : Layout widths) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    Fin 4 → ShiftedTernary41V1.CanonicalGoldilocks :=
  fun lane => ⟨assignment (memoryOutputColumn layout lane), canonical _⟩

/-- The exact 540-coordinate carrier binds both the prior state and the
complete canonical memory suffix. -/
def ccsEncoding
    (digest : MemoryBoundCcsPublic.CanonicalDigest)
    (memory : MemoryClaimCodec.Claim) : List Nat :=
  MemoryBoundCcsPublic.encode digest memory

theorem ccsEncoding_length
    (digest : MemoryBoundCcsPublic.CanonicalDigest)
    (memory : MemoryClaimCodec.Claim) :
    (ccsEncoding digest memory).length = ccsPublicBitCount := by
  exact MemoryBoundCcsPublic.encode_length digest memory

theorem ccsEncoding_binary
    (digest : MemoryBoundCcsPublic.CanonicalDigest)
    (memory : MemoryClaimCodec.Claim)
    (digit : Nat) (member : digit ∈ ccsEncoding digest memory) : digit < 2 :=
  MemoryBoundCcsPublic.encode_binary digest memory digit member

/-- Exact typed 540-coordinate public carrier. -/
def ccsPublicWord
    (digest : MemoryBoundCcsPublic.CanonicalDigest)
    (memory : MemoryClaimCodec.Claim) :
    FixedBits.Word ccsPublicBitCount :=
  MemoryBoundCcsPublic.word digest memory

theorem ccsEncoding_get_digest
    (digest : MemoryBoundCcsPublic.CanonicalDigest)
    (memory : MemoryClaimCodec.Claim)
    (lane : Fin 4) (bit : Fin CanonicalFieldBits.bitCount) :
    (ccsEncoding digest memory).getD
        (1 + CanonicalFieldBits.bitCount * lane.val + bit.val) 0 =
      (CanonicalFieldBits.encode (digest lane)).val.getD bit.val 0 :=
  MemoryBoundCcsPublic.encode_get_stateDigest digest memory lane bit

theorem ccsEncoding_get_memory_digest
    (digest : MemoryBoundCcsPublic.CanonicalDigest)
    (memory : MemoryClaimCodec.Claim)
    (lane : Fin 4) (bit : Fin CanonicalFieldBits.bitCount) :
    (ccsEncoding digest memory).getD
        (1 + digestBitCount +
          CanonicalFieldBits.bitCount * lane.val + bit.val) 0 =
      (CanonicalFieldBits.encode
        (MemoryBoundCcsPublic.memoryDigest memory lane)).val.getD bit.val 0 :=
  MemoryBoundCcsPublic.encode_get_memoryDigest digest memory lane bit

theorem ccsEncoding_get_padding
    (digest : MemoryBoundCcsPublic.CanonicalDigest)
    (memory : MemoryClaimCodec.Claim)
    (padding : Nat) (paddingBound : padding < paddingBitCount) :
    (ccsEncoding digest memory).getD
      (1 + digestBitCount + memoryDigestBitCount + padding) 0 = 0 :=
  MemoryBoundCcsPublic.encode_get_padding digest memory padding paddingBound

def claimDigestWord {widths : CompilerWidths}
    {layout : Layout widths} (valid : layout.Valid)
    (claim : Value widths) (lane : Fin 4) : CanonicalFieldBits.Word :=
  FixedBits.slice claim.ccsPublic
    (1 + CanonicalFieldBits.bitCount * lane.val)
    CanonicalFieldBits.bitCount (by
      rw [valid.exactCcsPublicWidth]
      have laneBound := lane.isLt
      norm_num [ccsPublicBitCount, CanonicalFieldBits.bitCount,
        MemoryBoundCcsPublic.coordinateCount] at *
      omega)

def claimDigestWords {widths : CompilerWidths}
    {layout : Layout widths} (valid : layout.Valid)
    (claim : Value widths) :
    CanonicalFieldSchemaRows.RawWords (Fin 4) :=
  fun lane => claimDigestWord valid claim lane

def claimMemoryDigestWord {widths : CompilerWidths}
    {layout : Layout widths} (valid : layout.Valid)
    (claim : Value widths) (lane : Fin 4) : CanonicalFieldBits.Word :=
  FixedBits.slice claim.ccsPublic
    (1 + digestBitCount + CanonicalFieldBits.bitCount * lane.val)
    CanonicalFieldBits.bitCount (by
      rw [valid.exactCcsPublicWidth]
      have laneBound := lane.isLt
      norm_num [ccsPublicBitCount, digestBitCount,
        CanonicalFieldBits.bitCount, MemoryBoundCcsPublic.coordinateCount,
        MemoryBoundCcsPublic.digestBitCount] at *
      omega)

def claimMemoryDigestWords {widths : CompilerWidths}
    {layout : Layout widths} (valid : layout.Valid)
    (claim : Value widths) :
    CanonicalFieldSchemaRows.RawWords (Fin 4) :=
  fun lane => claimMemoryDigestWord valid claim lane

private theorem claimCcsBitPlaced
    {widths : CompilerWidths} {layout : Layout widths}
    {assignment : Nat → Nat} {claim : Value widths}
    {input : FixedBits.Word widths.totalBits}
    (placed : FullClaimEnvelopeRows.Placed layout.fullClaim assignment claim input)
    (offset : Nat) (bound : offset < widths.ccsPublicBits) :
    assignment (layout.ccsBitColumn offset) =
      claim.ccsPublic.val.getD offset 0 := by
  let global : Fin widths.totalBits :=
    ⟨Section.ccsPublic.bitOffset widths + offset, by
      have fits := Section.slice_fits widths .ccsPublic
      simp only [Section.width] at fits
      omega⟩
  have source := (placed global).1
  have sectionValue := claim.encode_get_section .ccsPublic ⟨offset, bound⟩
  have getDBound : offset < claim.ccsPublic.val.length := by
    simpa [claim.ccsPublic.property.1] using bound
  calc
    assignment (layout.ccsBitColumn offset) =
        FullClaimEnvelopeRows.envelopeBit claim global := by
      simpa [global, Layout.ccsBitColumn, Nat.add_assoc] using source
    _ = claim.ccsPublic.val[offset]'getDBound := by
      simpa [FullClaimEnvelopeRows.envelopeBit, global] using sectionValue
    _ = claim.ccsPublic.val.getD offset 0 := by
      simp [List.getD_eq_getElem?_getD, getDBound]

def Layout.ccsPublicBits {widths : CompilerWidths}
    (layout : Layout widths) : PublicBitBlock.Layout :=
  { publicBitStart := layout.ccsBitColumn 0 }

private theorem claimCcsPublicPlaced
    {widths : CompilerWidths} {layout : Layout widths}
    {assignment : Nat → Nat} {claim : Value widths}
    {input : FixedBits.Word widths.totalBits}
    (placed : FullClaimEnvelopeRows.Placed layout.fullClaim assignment claim input) :
    PublicBitBlock.Placed layout.ccsPublicBits assignment claim.ccsPublic := by
  intro offset bound
  have exact := claimCcsBitPlaced placed offset bound
  have getDBound : offset < claim.ccsPublic.val.length := by
    simpa [claim.ccsPublic.property.1] using bound
  simpa [Layout.ccsPublicBits, Layout.ccsBitColumn, Nat.add_assoc,
    List.getD_eq_getElem?_getD, getDBound] using exact

/-- Exact claim-section placement supplies the raw canonical-u64 words used by
the digest-link rows. No independent digest-word witness is accepted. -/
theorem claimDigestWordsPlaced
    {widths : CompilerWidths} {layout : Layout widths}
    (valid : layout.Valid) {assignment : Nat → Nat}
    {claim : Value widths} {input : FixedBits.Word widths.totalBits}
    (placed : FullClaimEnvelopeRows.Placed layout.fullClaim assignment claim input) :
    CanonicalFieldSchemaRows.Places layout.digestSchema assignment
      (claimDigestWords valid claim) := by
  intro lane
  have fits :
      1 + CanonicalFieldBits.bitCount * lane.val +
          CanonicalFieldBits.bitCount ≤ widths.ccsPublicBits := by
    rw [valid.exactCcsPublicWidth]
    have laneBound := lane.isLt
    norm_num [ccsPublicBitCount, CanonicalFieldBits.bitCount,
      MemoryBoundCcsPublic.coordinateCount] at *
    omega
  have sliced := PublicBitBlock.slice_eq_columns
    (claimCcsPublicPlaced placed)
    (1 + CanonicalFieldBits.bitCount * lane.val)
    CanonicalFieldBits.bitCount fits
  simpa [claimDigestWords, claimDigestWord,
    CanonicalFieldSchemaRows.rawDigits, Layout.digestSchema,
    Layout.digestRawColumns, PublicBitBlock.sliceColumns,
    Layout.ccsPublicBits, Layout.ccsBitColumn, Nat.add_assoc] using sliced

theorem claimMemoryDigestWordsPlaced
    {widths : CompilerWidths} {layout : Layout widths}
    (valid : layout.Valid) {assignment : Nat → Nat}
    {claim : Value widths} {input : FixedBits.Word widths.totalBits}
    (placed : FullClaimEnvelopeRows.Placed layout.fullClaim assignment claim input) :
    CanonicalFieldSchemaRows.Places layout.memoryDigestSchema assignment
      (claimMemoryDigestWords valid claim) := by
  intro lane
  have fits :
      1 + digestBitCount + CanonicalFieldBits.bitCount * lane.val +
          CanonicalFieldBits.bitCount ≤ widths.ccsPublicBits := by
    rw [valid.exactCcsPublicWidth]
    have laneBound := lane.isLt
    norm_num [ccsPublicBitCount, digestBitCount,
      CanonicalFieldBits.bitCount, MemoryBoundCcsPublic.coordinateCount,
      MemoryBoundCcsPublic.digestBitCount] at *
    omega
  have sliced := PublicBitBlock.slice_eq_columns
    (claimCcsPublicPlaced placed)
    (1 + digestBitCount + CanonicalFieldBits.bitCount * lane.val)
    CanonicalFieldBits.bitCount fits
  simpa [claimMemoryDigestWords, claimMemoryDigestWord,
    CanonicalFieldSchemaRows.rawDigits, Layout.memoryDigestSchema,
    Layout.memoryDigestRawColumns, PublicBitBlock.sliceColumns,
    Layout.ccsPublicBits, Layout.ccsBitColumn, Nat.add_assoc] using sliced

/-- Every digest source word in a satisfying relation is the unique canonical
encoding of the corresponding Poseidon2 output lane. -/
theorem claimDigestWord_eq_outputEncoding
    {widths : CompilerWidths} {layout : Layout widths}
    (valid : layout.Valid) {assignment : Nat → Nat}
    {claim : Value widths} {input : FixedBits.Word widths.totalBits}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : FullClaimEnvelopeRows.Placed layout.fullClaim assignment claim input)
    (holds : Satisfies (rows layout) assignment)
    (lane : Fin 4) :
    claimDigestWord valid claim lane =
      CanonicalFieldBits.encode (outputDigest layout assignment canonical lane) := by
  have rawPlaced := claimDigestWordsPlaced valid placed
  have decoded := CanonicalFieldSchemaRows.slot_sound canonical one
    (digest_rows_hold holds) rawPlaced (digestSlot_mem lane)
  rcases decoded with ⟨value, accepted, valueEqual⟩
  have outputEqual : value = outputDigest layout assignment canonical lane := by
    apply Subtype.ext
    change value.val = assignment (outputColumn layout lane)
    exact valueEqual.trans
      (congrArg assignment (layout.digestValueColumn lane))
  subst value
  apply CanonicalFieldBits.decode_injective
  have decodedAccepted := (FieldCodec.nativeDecode_some_iff _ _).mp accepted
  rw [CanonicalFieldBits.decode_encode]
  exact decodedAccepted.2.symm

theorem claimMemoryDigestWord_eq_outputEncoding
    {widths : CompilerWidths} {layout : Layout widths}
    (valid : layout.Valid) {assignment : Nat → Nat}
    {claim : Value widths} {input : FixedBits.Word widths.totalBits}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : FullClaimEnvelopeRows.Placed layout.fullClaim assignment claim input)
    (holds : Satisfies (rows layout) assignment)
    (lane : Fin 4) :
    claimMemoryDigestWord valid claim lane =
      CanonicalFieldBits.encode
        (outputMemoryDigest layout assignment canonical lane) := by
  have rawPlaced := claimMemoryDigestWordsPlaced valid placed
  have decoded := CanonicalFieldSchemaRows.slot_sound canonical one
    (memory_digest_schema_rows_hold holds) rawPlaced (digestSlot_mem lane)
  rcases decoded with ⟨value, accepted, valueEqual⟩
  have outputEqual :
      value = outputMemoryDigest layout assignment canonical lane := by
    apply Subtype.ext
    change value.val = assignment (memoryOutputColumn layout lane)
    exact valueEqual.trans
      (congrArg assignment (layout.memoryDigestValueColumn lane))
  subst value
  apply CanonicalFieldBits.decode_injective
  have decodedAccepted := (FieldCodec.nativeDecode_some_iff _ _).mp accepted
  rw [CanonicalFieldBits.decode_encode]
  exact decodedAccepted.2.symm

theorem claimMemoryDigestWord_eq_claimEncoding
    {widths : CompilerWidths} {layout : Layout widths}
    (valid : layout.Valid) {assignment : Nat → Nat}
    {claim : Value widths} {input : FixedBits.Word widths.totalBits}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : FullClaimEnvelopeRows.Placed layout.fullClaim assignment claim input)
    (memoryParsed : MemoryClaimRows.ParsedColumnsMatch
      layout.memoryDigest.frame.claim assignment claim.memory)
    (holds : Satisfies (rows layout) assignment)
    (lane : Fin 4) :
    claimMemoryDigestWord valid claim lane =
      CanonicalFieldBits.encode
        (MemoryBoundCcsPublic.memoryDigest claim.memory lane) := by
  rw [claimMemoryDigestWord_eq_outputEncoding valid canonical one placed
    holds lane]
  congr 1
  apply Subtype.ext
  have output := MemoryClaimPoseidonRows.output_columns_eq_digest
    valid.memoryDigestValid canonical one memoryParsed
      (memory_digest_rows_hold holds) lane
  simpa [outputMemoryDigest, memoryOutputColumn,
    MemoryBoundCcsPublic.memoryDigest,
    MemoryClaimPoseidonBinding.canonicalDigest] using output

/-- Exact, non-overlapping description of all 540 selected CCS public
coordinates. -/
structure CcsPublicExact
    {widths : CompilerWidths} {layout : Layout widths}
    (valid : layout.Valid) (assignment : Nat → Nat) (claim : Value widths)
    (canonical : ∀ column, assignment column < goldilocksP) : Prop where
  width : claim.ccsPublic.val.length = ccsPublicBitCount
  affineOne : claim.ccsPublic.val.getD 0 0 = 1
  digestWords : ∀ lane,
    claimDigestWord valid claim lane =
      CanonicalFieldBits.encode (outputDigest layout assignment canonical lane)
  memoryDigestWords : ∀ lane,
    claimMemoryDigestWord valid claim lane =
      CanonicalFieldBits.encode
        (MemoryBoundCcsPublic.memoryDigest claim.memory lane)
  paddingZero : ∀ padding, padding < paddingBitCount →
    claim.ccsPublic.val.getD
      (1 + digestBitCount + memoryDigestBitCount + padding) 0 = 0

namespace CcsPublicExact

def typedWord
    {widths : CompilerWidths} {layout : Layout widths}
    (valid : layout.Valid)
    (digest : MemoryBoundCcsPublic.CanonicalDigest)
    (memory : MemoryClaimCodec.Claim) :
    FixedBits.Word widths.ccsPublicBits :=
  ⟨ccsEncoding digest memory, by
      rw [valid.exactCcsPublicWidth]
      exact ccsEncoding_length digest memory,
    ccsEncoding_binary digest memory⟩

/-- One fieldwise digest equality, expressed at the exact coordinate in the
complete CCS public word. This result removes the slice object from the
lifetime proof. -/
theorem getD_digest
    {widths : CompilerWidths} {layout : Layout widths}
    {assignment : Nat → Nat} {claim : Value widths}
    {canonical : ∀ column, assignment column < goldilocksP}
    {valid : layout.Valid}
    (exact : CcsPublicExact valid assignment claim canonical)
    (lane : Fin 4) (bit : Fin CanonicalFieldBits.bitCount) :
    claim.ccsPublic.val.getD
        (1 + CanonicalFieldBits.bitCount * lane.val + bit.val) 0 =
      (CanonicalFieldBits.encode
        (outputDigest layout assignment canonical lane)).val.getD bit.val 0 := by
  have fits :
      1 + CanonicalFieldBits.bitCount * lane.val +
          CanonicalFieldBits.bitCount ≤ widths.ccsPublicBits := by
    rw [valid.exactCcsPublicWidth]
    have laneBound := lane.isLt
    norm_num [ccsPublicBitCount, CanonicalFieldBits.bitCount,
      MemoryBoundCcsPublic.coordinateCount] at *
    omega
  have source := FixedBits.slice_getD claim.ccsPublic
    (1 + CanonicalFieldBits.bitCount * lane.val)
    CanonicalFieldBits.bitCount fits bit.val bit.isLt
  have wordEqual := congrArg
    (fun word : CanonicalFieldBits.Word => word.val.getD bit.val 0)
    (exact.digestWords lane)
  calc
    claim.ccsPublic.val.getD
          (1 + CanonicalFieldBits.bitCount * lane.val + bit.val) 0 =
        (claimDigestWord valid claim lane).val.getD bit.val 0 := by
      simpa [claimDigestWord] using source.symm
    _ = (CanonicalFieldBits.encode
          (outputDigest layout assignment canonical lane)).val.getD bit.val 0 :=
      wordEqual

theorem getD_memoryDigest
    {widths : CompilerWidths} {layout : Layout widths}
    {assignment : Nat → Nat} {claim : Value widths}
    {canonical : ∀ column, assignment column < goldilocksP}
    {valid : layout.Valid}
    (exact : CcsPublicExact valid assignment claim canonical)
    (lane : Fin 4) (bit : Fin CanonicalFieldBits.bitCount) :
    claim.ccsPublic.val.getD
        (1 + digestBitCount +
          CanonicalFieldBits.bitCount * lane.val + bit.val) 0 =
      (CanonicalFieldBits.encode
        (MemoryBoundCcsPublic.memoryDigest claim.memory lane)).val.getD
          bit.val 0 := by
  have fits :
      1 + digestBitCount + CanonicalFieldBits.bitCount * lane.val +
          CanonicalFieldBits.bitCount ≤ widths.ccsPublicBits := by
    rw [valid.exactCcsPublicWidth]
    have laneBound := lane.isLt
    norm_num [ccsPublicBitCount, digestBitCount,
      CanonicalFieldBits.bitCount, MemoryBoundCcsPublic.coordinateCount,
      MemoryBoundCcsPublic.digestBitCount] at *
    omega
  have source := FixedBits.slice_getD claim.ccsPublic
    (1 + digestBitCount + CanonicalFieldBits.bitCount * lane.val)
    CanonicalFieldBits.bitCount fits bit.val bit.isLt
  have wordEqual := congrArg
    (fun word : CanonicalFieldBits.Word => word.val.getD bit.val 0)
    (exact.memoryDigestWords lane)
  calc
    claim.ccsPublic.val.getD
          (1 + digestBitCount +
            CanonicalFieldBits.bitCount * lane.val + bit.val) 0 =
        (claimMemoryDigestWord valid claim lane).val.getD bit.val 0 := by
      simpa [claimMemoryDigestWord, Nat.add_assoc] using source.symm
    _ = (CanonicalFieldBits.encode
          (MemoryBoundCcsPublic.memoryDigest claim.memory lane)).val.getD
            bit.val 0 := wordEqual

/-- The fieldwise row result assembles to one exact carrier equality. This
prevents a lifetime proof from replacing the selected claim by another value
that only has the same memory suffix. -/
theorem ccsPublic_eq_ccsPublicWord
    {widths : CompilerWidths} {layout : Layout widths}
    {assignment : Nat → Nat} {claim : Value widths}
    {canonical : ∀ column, assignment column < goldilocksP}
    {valid : layout.Valid}
    (exact : CcsPublicExact valid assignment claim canonical) :
    claim.ccsPublic =
      typedWord valid (outputDigest layout assignment canonical)
        claim.memory := by
  apply Subtype.ext
  apply List.ext_get
  · rw [exact.width]
    exact (ccsEncoding_length (outputDigest layout assignment canonical)
      claim.memory).symm
  · intro index leftBound rightBound
    change claim.ccsPublic.val[index] =
      (ccsEncoding (outputDigest layout assignment canonical)
        claim.memory)[index]
    have indexLimit : index < ccsPublicBitCount := by
      rw [← exact.width]
      exact leftBound
    have leftGetD : claim.ccsPublic.val.getD index 0 =
        claim.ccsPublic.val[index] := by
      simp [List.getD_eq_getElem?_getD, leftBound]
    have encodingBound :
        index < (ccsEncoding
          (outputDigest layout assignment canonical) claim.memory).length := by
      rw [ccsEncoding_length]
      exact indexLimit
    have rightGetD :
        (ccsEncoding (outputDigest layout assignment canonical)
          claim.memory).getD index 0 =
          (ccsEncoding
            (outputDigest layout assignment canonical) claim.memory)[index] := by
      simp [List.getD_eq_getElem?_getD, encodingBound]
    have getDEqual :
        claim.ccsPublic.val.getD index 0 =
          (ccsEncoding
            (outputDigest layout assignment canonical) claim.memory).getD
              index 0 := by
      by_cases affine : index = 0
      · subst index
        simpa [ccsEncoding, MemoryBoundCcsPublic.encode,
          List.getD_eq_getElem?_getD] using exact.affineOne
      by_cases digestRegion : index < 1 + digestBitCount
      · let flat := index - 1
        have flatBound : flat < digestBitCount := by
          simp only [flat]
          omega
        let lane : Fin 4 := ⟨flat / CanonicalFieldBits.bitCount, by
          norm_num [digestBitCount, CanonicalFieldBits.bitCount,
            MemoryBoundCcsPublic.digestBitCount] at flatBound ⊢
          omega⟩
        let bit : Fin CanonicalFieldBits.bitCount :=
          ⟨flat % CanonicalFieldBits.bitCount,
            Nat.mod_lt _ (by norm_num [CanonicalFieldBits.bitCount])⟩
        have indexEq :
            1 + CanonicalFieldBits.bitCount * lane.val + bit.val = index := by
          change 1 + CanonicalFieldBits.bitCount *
              (flat / CanonicalFieldBits.bitCount) +
              flat % CanonicalFieldBits.bitCount = index
          have division := Nat.mod_add_div flat CanonicalFieldBits.bitCount
          have flatEq : 1 + flat = index := by
            simp only [flat]
            omega
          omega
        rw [← indexEq]
        exact (exact.getD_digest lane bit).trans
          (ccsEncoding_get_digest
            (outputDigest layout assignment canonical) claim.memory
            lane bit).symm
      · by_cases memoryRegion :
            index < 1 + digestBitCount + memoryDigestBitCount
        · let flat := index - (1 + digestBitCount)
          have flatBound : flat < memoryDigestBitCount := by
            simp only [flat]
            omega
          let lane : Fin 4 :=
            ⟨flat / CanonicalFieldBits.bitCount, by
              norm_num [memoryDigestBitCount,
                CanonicalFieldBits.bitCount,
                MemoryBoundCcsPublic.digestBitCount] at flatBound ⊢
              omega⟩
          let bit : Fin CanonicalFieldBits.bitCount :=
            ⟨flat % CanonicalFieldBits.bitCount,
              Nat.mod_lt _ (by norm_num [CanonicalFieldBits.bitCount])⟩
          have indexEq :
              1 + digestBitCount +
                  CanonicalFieldBits.bitCount * lane.val + bit.val = index := by
            change 1 + digestBitCount + CanonicalFieldBits.bitCount *
                (flat / CanonicalFieldBits.bitCount) +
                flat % CanonicalFieldBits.bitCount = index
            have division := Nat.mod_add_div flat CanonicalFieldBits.bitCount
            have flatEq : 1 + digestBitCount + flat = index := by
              simp only [flat]
              omega
            omega
          rw [← indexEq]
          exact (exact.getD_memoryDigest lane bit).trans
            (ccsEncoding_get_memory_digest
              (outputDigest layout assignment canonical) claim.memory
              lane bit).symm
        · let padding :=
            index - (1 + digestBitCount + memoryDigestBitCount)
          have paddingBound : padding < paddingBitCount := by
            simp only [padding]
            norm_num [ccsPublicBitCount, digestBitCount,
              memoryDigestBitCount, paddingBitCount,
              MemoryBoundCcsPublic.coordinateCount,
              MemoryBoundCcsPublic.digestBitCount,
              MemoryBoundCcsPublic.paddingBitCount] at *
            omega
          have leftZero := exact.paddingZero padding paddingBound
          have rightZero := ccsEncoding_get_padding
            (outputDigest layout assignment canonical) claim.memory
            padding paddingBound
          have indexEq :
              1 + digestBitCount + memoryDigestBitCount + padding = index := by
            simp only [padding]
            omega
          rw [← indexEq]
          exact leftZero.trans rightZero.symm
    exact leftGetD.symm.trans (getDEqual.trans rightGetD)

end CcsPublicExact

/-- Satisfying rows derive the complete selected carrier. No field in this
predicate is supplied as a witness or assumption. -/
theorem claimCcsPublicExact
    {widths : CompilerWidths} {layout : Layout widths}
    (valid : layout.Valid) {assignment : Nat → Nat}
    {claim : Value widths} {input : FixedBits.Word widths.totalBits}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : FullClaimEnvelopeRows.Placed layout.fullClaim assignment claim input)
    (memoryParsed : MemoryClaimRows.ParsedColumnsMatch
      layout.memoryDigest.frame.claim assignment claim.memory)
    (holds : Satisfies (rows layout) assignment) :
    CcsPublicExact valid assignment claim canonical := by
  refine {
    width := ?_
    affineOne := ?_
    digestWords := ?_
    memoryDigestWords := ?_
    paddingZero := ?_
  }
  · rw [claim.ccsPublic.property.1, valid.exactCcsPublicWidth]
  · have source := claimCcsBitPlaced placed 0 (by
      rw [valid.exactCcsPublicWidth]
      decide)
    have pin := pinFacts canonical one holds
      (layout.ccsBitColumn 0, 1) (by simp [publicPins])
    exact source.symm.trans pin
  · exact claimDigestWord_eq_outputEncoding valid canonical one placed holds
  · exact claimMemoryDigestWord_eq_claimEncoding valid canonical one placed
      memoryParsed holds
  · intro padding paddingBound
    have offsetBound :
        1 + digestBitCount + memoryDigestBitCount + padding <
          widths.ccsPublicBits := by
      rw [valid.exactCcsPublicWidth]
      norm_num [ccsPublicBitCount, digestBitCount, memoryDigestBitCount,
        paddingBitCount, MemoryBoundCcsPublic.coordinateCount,
        MemoryBoundCcsPublic.digestBitCount,
        MemoryBoundCcsPublic.paddingBitCount] at *
      omega
    have source := claimCcsBitPlaced placed
      (1 + digestBitCount + memoryDigestBitCount + padding) offsetBound
    have pin := pinFacts canonical one holds
      (layout.ccsBitColumn
        (1 + digestBitCount + memoryDigestBitCount + padding), 0)
      (by
        simp only [publicPins, List.mem_append, List.mem_singleton,
          List.mem_map]
        exact Or.inr ⟨padding, List.mem_range.mpr paddingBound, rfl⟩)
    exact source.symm.trans pin

/-- The four output lanes used above are not free witnesses. Satisfying the
same row block computes them from the exact incoming carry block and typed
prior non-memory payload through both fixed Poseidon2 sponges. -/
theorem outputDigest_eq_typedPriorState
    {widths : CompilerWidths} {layout : Layout widths}
    (valid : layout.Valid) {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (carryPlaced : PublicBitBlock.Placed
      layout.stateOutput.hash.carry.frame.packing.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment) :
    ∀ lane : Fin 4,
      (outputDigest layout assignment canonical lane).val =
        StateOutputPoseidonRows.pureDigest
          (StateOutputAuthorityRows.fullFrame
            (StateOutputAuthorityRows.payload layout.stateOutput.authority
              assignment)
            (MemoryCarryPoseidonRows.carryDigest block)) lane.val := by
  have output :=
    AuthoritativeStateOutputRows.output_columns_eq_typed_stateDigest
      valid.stateOutputValid canonical one carryPlaced (state_rows_hold holds)
  intro lane
  simpa [outputDigest, outputColumn] using output lane

end Nightstream.Implementation.NebulaV2.PriorStateLinkRows
