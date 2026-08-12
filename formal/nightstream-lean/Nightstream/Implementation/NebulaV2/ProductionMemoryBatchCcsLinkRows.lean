import Nightstream.Implementation.NebulaV2.CanonicalFieldSchemaRows
import Nightstream.Implementation.NebulaV2.ProductionFieldNativeFullClaim
import Nightstream.Implementation.NebulaV2.ProductionMemoryBatchPoseidonRows
import Nightstream.Implementation.R1CS.Core.ConstantPins

/-!
Contract: exponent-independent generated-row link from one checked production
memory batch and one authoritative state digest to all 540 fresh CCS public
coordinates.

The relation recomputes the complete candidate-specific memory-batch digest,
canonically decomposes all four state and four memory digest lanes, fixes the
affine coordinate to one, and fixes all 27 padding coordinates to zero.
Satisfying rows derive `ProductionMemoryBoundCcsPublic.FullMatches`; this is
not a claim-canonicality premise.

Does not own the computation of the four state-digest source columns, NIFS
verification, absolute generated-column allocation, Poseidon2 collision
security, Rust refinement, candidate selection, or a verifier key.

Emits constraints: yes.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductionMemoryBatchCcsLinkRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CanonicalU64
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete

abbrev CanonicalDigest := ProductionMemoryBoundCcsPublic.CanonicalDigest

def digestSlots : List (Fin 4) := List.ofFn id

theorem digestSlots_length : digestSlots.length = 4 := by
  simp [digestSlots]

theorem digestSlot_mem (lane : Fin 4) : lane ∈ digestSlots := by
  fin_cases lane <;> simp [digestSlots]

/-- Offset of the 540-coordinate CCS image in each field-native claim
carrier. The exponent-indexed adapter proves that its carrier uses this same
offset. -/
def ccsPublicOffset : Nat := 0

structure Layout (candidate : Id) where
  carrierStart : Nat
  batch : ProductionMemoryBatchPoseidonRows.Layout candidate
  stateDigestColumn : Fin 4 -> Nat
  stateDigestColumnMap : Fin 4 -> List Nat
  stateDigestMapsConstantOne : forall lane,
    Relabel.column (stateDigestColumnMap lane) 0 = 0
  stateDigestValueColumn : forall lane,
    Relabel.column (stateDigestColumnMap lane) varCol =
      stateDigestColumn lane
  memoryDigestColumnMap : Fin 4 -> List Nat
  memoryDigestMapsConstantOne : forall lane,
    Relabel.column (memoryDigestColumnMap lane) 0 = 0
  memoryDigestValueColumn : forall lane,
    Relabel.column (memoryDigestColumnMap lane) varCol =
      batch.trace.outputColumns.getD lane.val 0

def Layout.ccsBitColumn {candidate : Id}
    (layout : Layout candidate) (offset : Nat) : Nat :=
  layout.carrierStart + ccsPublicOffset + offset

def Layout.ccsBits {candidate : Id}
    (layout : Layout candidate) : PublicBitBlock.Layout where
  publicBitStart := layout.ccsBitColumn 0

def Layout.stateDigestRawColumns {candidate : Id}
    (layout : Layout candidate) (lane : Fin 4) : List Nat :=
  (List.range CanonicalFieldBits.bitCount).map fun bit =>
    layout.ccsBitColumn
      (1 + CanonicalFieldBits.bitCount * lane.val + bit)

theorem Layout.stateDigestRawColumns_length {candidate : Id}
    (layout : Layout candidate) (lane : Fin 4) :
    (layout.stateDigestRawColumns lane).length =
      CanonicalFieldBits.bitCount := by
  simp [Layout.stateDigestRawColumns]

def Layout.stateDigestSchema {candidate : Id}
    (layout : Layout candidate) :
    CanonicalFieldSchemaRows.Layout (Fin 4) where
  columnMap := layout.stateDigestColumnMap
  rawColumns := layout.stateDigestRawColumns
  rawColumnsLength := layout.stateDigestRawColumns_length
  mapsConstantOne := layout.stateDigestMapsConstantOne

def Layout.memoryDigestRawColumns {candidate : Id}
    (layout : Layout candidate) (lane : Fin 4) : List Nat :=
  (List.range CanonicalFieldBits.bitCount).map fun bit =>
    layout.ccsBitColumn
      (1 + MemoryBoundCcsPublic.digestBitCount +
        CanonicalFieldBits.bitCount * lane.val + bit)

theorem Layout.memoryDigestRawColumns_length {candidate : Id}
    (layout : Layout candidate) (lane : Fin 4) :
    (layout.memoryDigestRawColumns lane).length =
      CanonicalFieldBits.bitCount := by
  simp [Layout.memoryDigestRawColumns]

def Layout.memoryDigestSchema {candidate : Id}
    (layout : Layout candidate) :
    CanonicalFieldSchemaRows.Layout (Fin 4) where
  columnMap := layout.memoryDigestColumnMap
  rawColumns := layout.memoryDigestRawColumns
  rawColumnsLength := layout.memoryDigestRawColumns_length
  mapsConstantOne := layout.memoryDigestMapsConstantOne

def publicPins {candidate : Id}
    (layout : Layout candidate) : List (Nat × Nat) :=
  [(layout.ccsBitColumn 0, 1)] ++
    (List.range MemoryBoundCcsPublic.paddingBitCount).map
      fun padding =>
        (layout.ccsBitColumn
          (1 + MemoryBoundCcsPublic.digestBitCount +
            MemoryBoundCcsPublic.digestBitCount + padding), 0)

def rows {candidate : Id} (layout : Layout candidate) : List Row :=
  ProductionMemoryBatchPoseidonRows.rows layout.batch ++
    CanonicalFieldSchemaRows.schemaRows digestSlots
      layout.stateDigestSchema ++
    CanonicalFieldSchemaRows.schemaRows digestSlots
      layout.memoryDigestSchema ++
    ConstantPins.rows (publicPins layout)

/-- Structural certificate only. It contains no assignment or authority
result. -/
structure Layout.Valid {candidate : Id} (layout : Layout candidate) : Prop where
  batchValid : layout.batch.Valid
  memoryValid : layout.batch.frame.memory.Valid

def StateDigestPlaced {candidate : Id}
    (layout : Layout candidate) (assignment : Nat -> Nat)
    (digest : CanonicalDigest) : Prop :=
  forall lane, assignment (layout.stateDigestColumn lane) = (digest lane).val

theorem publicPins_length {candidate : Id} (layout : Layout candidate) :
    (publicPins layout).length = 28 := by
  simp [publicPins, MemoryBoundCcsPublic.paddingBitCount]

theorem publicPins_valuesCanonical {candidate : Id}
    (layout : Layout candidate) :
    ConstantPins.ValuesCanonical (publicPins layout) := by
  intro pin member
  simp only [publicPins, List.mem_append, List.mem_singleton,
    List.mem_map] at member
  rcases member with rfl | ⟨padding, _paddingMember, rfl⟩
  · norm_num [goldilocksP]
  · norm_num [goldilocksP]

def rowCount (candidate : Id) : Nat :=
  ProductionMemoryBatchPoseidonRows.rowCount candidate + 1092

theorem rows_length_exact {candidate : Id} {layout : Layout candidate}
    (valid : layout.Valid) : (rows layout).length = rowCount candidate := by
  simp [rows, rowCount,
    ProductionMemoryBatchPoseidonRows.rows_length_exact valid.batchValid,
    CanonicalFieldSchemaRows.schemaRows_length, digestSlots_length,
    ConstantPins.rows, publicPins_length]

theorem candidate_row_count_table :
    rowCount .e1 = 15593 /\
      rowCount .e4 = 53042 /\
      rowCount .e8 = 103174 /\
      rowCount .e16 = 203438 := by
  decide

private theorem batch_rows_hold
    {candidate : Id} {layout : Layout candidate}
    {assignment : Nat -> Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (ProductionMemoryBatchPoseidonRows.rows layout.batch)
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem state_schema_rows_hold
    {candidate : Id} {layout : Layout candidate}
    {assignment : Nat -> Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies
      (CanonicalFieldSchemaRows.schemaRows digestSlots
        layout.stateDigestSchema) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem memory_schema_rows_hold
    {candidate : Id} {layout : Layout candidate}
    {assignment : Nat -> Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies
      (CanonicalFieldSchemaRows.schemaRows digestSlots
        layout.memoryDigestSchema) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem pin_rows_hold
    {candidate : Id} {layout : Layout candidate}
    {assignment : Nat -> Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (ConstantPins.rows (publicPins layout)) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem selfIncluded (program : List Row) :
    rowsIncluded program program = true := by
  rw [rowsIncluded, List.all_eq_true]
  intro row member
  exact decide_eq_true member

private theorem pinFacts
    {candidate : Id} {layout : Layout candidate}
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    forall pin, pin ∈ publicPins layout -> assignment pin.1 = pin.2 := by
  exact ConstantPins.sound (publicPins_valuesCanonical layout)
    (selfIncluded _) canonical one (pin_rows_hold holds)

def CcsPublicPlaced
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (layout : Layout candidate) (assignment : Nat -> Nat)
    (value : ProductionFieldNativeFullClaim.Value candidate fullShape) : Prop :=
  forall index : Fin 540,
    assignment (layout.ccsBitColumn index.val) =
      value.ccsPublic.val.get
        ⟨index.val, by
          rw [value.ccsPublic.property.1]
          exact index.isLt⟩

private theorem publicBitBlockPlaced_of_ccsPublicPlaced
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    {layout : Layout candidate} {assignment : Nat -> Nat}
    {value : ProductionFieldNativeFullClaim.Value candidate fullShape}
    (placed : CcsPublicPlaced layout assignment value) :
    PublicBitBlock.Placed layout.ccsBits assignment value.ccsPublic := by
  intro index bound
  have source := placed
    ⟨index, by simpa [value.ccsPublic.property.1] using bound⟩
  simpa [Layout.ccsBits] using source

def claimStateDigestWord {candidate : Id} {fullShape : Phi81Relation.Shape}
    (value : ProductionFieldNativeFullClaim.Value candidate fullShape)
    (lane : Fin 4) : CanonicalFieldBits.Word :=
  FixedBits.slice value.ccsPublic
    (1 + CanonicalFieldBits.bitCount * lane.val)
    CanonicalFieldBits.bitCount (by
      have laneBound := lane.isLt
      norm_num [MemoryBoundCcsPublic.coordinateCount,
        CanonicalFieldBits.bitCount] at *
      omega)

def claimMemoryDigestWord {candidate : Id} {fullShape : Phi81Relation.Shape}
    (value : ProductionFieldNativeFullClaim.Value candidate fullShape)
    (lane : Fin 4) : CanonicalFieldBits.Word :=
  FixedBits.slice value.ccsPublic
    (1 + MemoryBoundCcsPublic.digestBitCount +
      CanonicalFieldBits.bitCount * lane.val)
    CanonicalFieldBits.bitCount (by
      have laneBound := lane.isLt
      norm_num [MemoryBoundCcsPublic.coordinateCount,
        MemoryBoundCcsPublic.digestBitCount,
        CanonicalFieldBits.bitCount] at *
      omega)

def claimStateDigestWords {candidate : Id} {fullShape : Phi81Relation.Shape}
    (value : ProductionFieldNativeFullClaim.Value candidate fullShape) :
    CanonicalFieldSchemaRows.RawWords (Fin 4) :=
  fun lane => claimStateDigestWord value lane

def claimMemoryDigestWords {candidate : Id} {fullShape : Phi81Relation.Shape}
    (value : ProductionFieldNativeFullClaim.Value candidate fullShape) :
    CanonicalFieldSchemaRows.RawWords (Fin 4) :=
  fun lane => claimMemoryDigestWord value lane

private theorem claimStateDigestWordsPlaced
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    {layout : Layout candidate} {assignment : Nat -> Nat}
    {value : ProductionFieldNativeFullClaim.Value candidate fullShape}
    (placed : CcsPublicPlaced layout assignment value) :
    CanonicalFieldSchemaRows.Places layout.stateDigestSchema assignment
      (claimStateDigestWords value) := by
  intro lane
  have sliced := PublicBitBlock.slice_eq_columns
    (publicBitBlockPlaced_of_ccsPublicPlaced placed)
    (1 + CanonicalFieldBits.bitCount * lane.val)
    CanonicalFieldBits.bitCount (by
      have laneBound := lane.isLt
      norm_num [MemoryBoundCcsPublic.coordinateCount,
        CanonicalFieldBits.bitCount] at *
      omega)
  simpa [claimStateDigestWords, claimStateDigestWord,
    CanonicalFieldSchemaRows.rawDigits, Layout.stateDigestSchema,
    Layout.stateDigestRawColumns, PublicBitBlock.sliceColumns,
    Layout.ccsBits, Nat.add_assoc] using sliced

private theorem claimMemoryDigestWordsPlaced
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    {layout : Layout candidate} {assignment : Nat -> Nat}
    {value : ProductionFieldNativeFullClaim.Value candidate fullShape}
    (placed : CcsPublicPlaced layout assignment value) :
    CanonicalFieldSchemaRows.Places layout.memoryDigestSchema assignment
      (claimMemoryDigestWords value) := by
  intro lane
  have sliced := PublicBitBlock.slice_eq_columns
    (publicBitBlockPlaced_of_ccsPublicPlaced placed)
    (1 + MemoryBoundCcsPublic.digestBitCount +
      CanonicalFieldBits.bitCount * lane.val)
    CanonicalFieldBits.bitCount (by
      have laneBound := lane.isLt
      norm_num [MemoryBoundCcsPublic.coordinateCount,
        MemoryBoundCcsPublic.digestBitCount,
        CanonicalFieldBits.bitCount] at *
      omega)
  simpa [claimMemoryDigestWords, claimMemoryDigestWord,
    CanonicalFieldSchemaRows.rawDigits, Layout.memoryDigestSchema,
    Layout.memoryDigestRawColumns, PublicBitBlock.sliceColumns,
    Layout.ccsBits, Nat.add_assoc] using sliced

theorem claimStateDigestWord_eq_encoding_of_ccsPublicPlaced
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    {layout : Layout candidate} {assignment : Nat -> Nat}
    {value : ProductionFieldNativeFullClaim.Value candidate fullShape}
    {stateDigest : CanonicalDigest}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : CcsPublicPlaced layout assignment value)
    (statePlaced : StateDigestPlaced layout assignment stateDigest)
    (holds : Satisfies (rows layout) assignment)
    (lane : Fin 4) :
    claimStateDigestWord value lane =
      CanonicalFieldBits.encode (stateDigest lane) := by
  have rawPlaced := claimStateDigestWordsPlaced placed
  rcases CanonicalFieldSchemaRows.slot_sound canonical one
      (state_schema_rows_hold holds) rawPlaced (digestSlot_mem lane) with
    ⟨decoded, accepted, valueEqual⟩
  have decodedEqual : decoded = stateDigest lane := by
    apply Subtype.ext
    calc
      decoded.val = assignment
          (Relabel.column (layout.stateDigestColumnMap lane) varCol) :=
        valueEqual
      _ = assignment (layout.stateDigestColumn lane) := by
        rw [layout.stateDigestValueColumn lane]
      _ = (stateDigest lane).val := statePlaced lane
  subst decoded
  apply CanonicalFieldBits.decode_injective
  have acceptedExact := (FieldCodec.nativeDecode_some_iff _ _).mp accepted
  rw [CanonicalFieldBits.decode_encode]
  exact acceptedExact.2.symm

theorem claimMemoryDigestWord_eq_encoding_of_ccsPublicPlaced
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    {layout : Layout candidate} {assignment : Nat -> Nat}
    {headers : FPrime.ChainHeaders Digest.Value}
    {value : ProductionFieldNativeFullClaim.Value candidate fullShape}
    (valid : layout.Valid)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : CcsPublicPlaced layout assignment value)
    (memory : ProductionMemoryCheckedBatchRows.Result
      layout.batch.frame.memory assignment headers)
    (holds : Satisfies (rows layout) assignment)
    (lane : Fin 4) :
    claimMemoryDigestWord value lane =
      CanonicalFieldBits.encode
        (ProductionMemoryBoundCcsPublic.memoryDigest
          memory.suffixBatch lane) := by
  have rawPlaced := claimMemoryDigestWordsPlaced placed
  rcases CanonicalFieldSchemaRows.slot_sound canonical one
      (memory_schema_rows_hold holds) rawPlaced (digestSlot_mem lane) with
    ⟨decoded, accepted, valueEqual⟩
  have outputEqual :=
    ProductionMemoryBatchPoseidonRows.output_columns_eq_digest
      valid.batchValid canonical one memory (batch_rows_hold holds) lane
  have decodedEqual : decoded =
      ProductionMemoryBoundCcsPublic.memoryDigest memory.suffixBatch lane := by
    apply Subtype.ext
    calc
      decoded.val = assignment
          (Relabel.column (layout.memoryDigestColumnMap lane) varCol) :=
        valueEqual
      _ = assignment (layout.batch.trace.outputColumns.getD lane.val 0) := by
        rw [layout.memoryDigestValueColumn lane]
      _ = ProductionMemoryBatchPoseidonBinding.digest
          memory.suffixBatch lane := outputEqual
      _ = (ProductionMemoryBoundCcsPublic.memoryDigest
          memory.suffixBatch lane).val := rfl
  subst decoded
  apply CanonicalFieldBits.decode_injective
  have acceptedExact := (FieldCodec.nativeDecode_some_iff _ _).mp accepted
  rw [CanonicalFieldBits.decode_encode]
  exact acceptedExact.2.symm

structure CcsPublicExact
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (value : ProductionFieldNativeFullClaim.Value candidate fullShape)
    (stateDigest : CanonicalDigest)
    (batch : ProductionMemoryBatchPoseidonBinding.Batch candidate) : Prop where
  affineOne : value.ccsPublic.val.getD 0 0 = 1
  stateDigestWords : forall lane,
    claimStateDigestWord value lane =
      CanonicalFieldBits.encode (stateDigest lane)
  memoryDigestWords : forall lane,
    claimMemoryDigestWord value lane =
      CanonicalFieldBits.encode
        (ProductionMemoryBoundCcsPublic.memoryDigest batch lane)
  paddingZero : forall padding,
    padding < MemoryBoundCcsPublic.paddingBitCount ->
      value.ccsPublic.val.getD
        (1 + MemoryBoundCcsPublic.digestBitCount +
          MemoryBoundCcsPublic.digestBitCount + padding) 0 = 0

theorem claimCcsPublicExact_of_ccsPublicPlaced
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    {layout : Layout candidate} {assignment : Nat -> Nat}
    {headers : FPrime.ChainHeaders Digest.Value}
    {value : ProductionFieldNativeFullClaim.Value candidate fullShape}
    {stateDigest : CanonicalDigest}
    (valid : layout.Valid)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : CcsPublicPlaced layout assignment value)
    (statePlaced : StateDigestPlaced layout assignment stateDigest)
    (memory : ProductionMemoryCheckedBatchRows.Result
      layout.batch.frame.memory assignment headers)
    (holds : Satisfies (rows layout) assignment) :
    CcsPublicExact value stateDigest memory.suffixBatch := by
  refine {
    affineOne := ?_
    stateDigestWords := ?_
    memoryDigestWords := ?_
    paddingZero := ?_
  }
  · have source := placed (0 : Fin 540)
    have pin := pinFacts canonical one holds
      (layout.ccsBitColumn 0, 1) (by simp [publicPins])
    have zeroBound : 0 < value.ccsPublic.val.length := by
      rw [value.ccsPublic.property.1]
      decide
    simpa [List.getD_eq_getElem?_getD, zeroBound] using source.symm.trans pin
  · exact claimStateDigestWord_eq_encoding_of_ccsPublicPlaced canonical one placed
      statePlaced holds
  · exact claimMemoryDigestWord_eq_encoding_of_ccsPublicPlaced valid canonical one placed
      memory holds
  · intro padding paddingBound
    have offsetBound :
        1 + MemoryBoundCcsPublic.digestBitCount +
          MemoryBoundCcsPublic.digestBitCount + padding < 540 := by
      norm_num [MemoryBoundCcsPublic.digestBitCount,
        MemoryBoundCcsPublic.paddingBitCount] at *
      omega
    let offset := 1 + MemoryBoundCcsPublic.digestBitCount +
      MemoryBoundCcsPublic.digestBitCount + padding
    have source := placed ⟨offset, offsetBound⟩
    have pin := pinFacts canonical one holds
      (layout.ccsBitColumn offset, 0) (by
        simp only [publicPins, List.mem_append, List.mem_singleton,
          List.mem_map]
        exact Or.inr ⟨padding, List.mem_range.mpr paddingBound, rfl⟩)
    have sourceBound : offset < value.ccsPublic.val.length := by
      rw [value.ccsPublic.property.1]
      exact offsetBound
    simpa [offset, List.getD_eq_getElem?_getD, sourceBound] using
      source.symm.trans pin

namespace CcsPublicExact

theorem getD_stateDigest
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    {value : ProductionFieldNativeFullClaim.Value candidate fullShape}
    {stateDigest : CanonicalDigest}
    {batch : ProductionMemoryBatchPoseidonBinding.Batch candidate}
    (exact : CcsPublicExact value stateDigest batch)
    (lane : Fin 4) (bit : Fin CanonicalFieldBits.bitCount) :
    value.ccsPublic.val.getD
        (1 + CanonicalFieldBits.bitCount * lane.val + bit.val) 0 =
      (CanonicalFieldBits.encode (stateDigest lane)).val.getD bit.val 0 := by
  have selected := FixedBits.slice_getD value.ccsPublic
    (1 + CanonicalFieldBits.bitCount * lane.val)
    CanonicalFieldBits.bitCount (by
      have laneBound := lane.isLt
      norm_num [MemoryBoundCcsPublic.coordinateCount,
        CanonicalFieldBits.bitCount] at *
      omega) bit.val bit.isLt
  have wordEqual := congrArg
    (fun word : CanonicalFieldBits.Word => word.val.getD bit.val 0)
    (exact.stateDigestWords lane)
  calc
    value.ccsPublic.val.getD
        (1 + CanonicalFieldBits.bitCount * lane.val + bit.val) 0 =
      (claimStateDigestWord value lane).val.getD bit.val 0 := selected.symm
    _ = (CanonicalFieldBits.encode (stateDigest lane)).val.getD bit.val 0 :=
      wordEqual

theorem getD_memoryDigest
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    {value : ProductionFieldNativeFullClaim.Value candidate fullShape}
    {stateDigest : CanonicalDigest}
    {batch : ProductionMemoryBatchPoseidonBinding.Batch candidate}
    (exact : CcsPublicExact value stateDigest batch)
    (lane : Fin 4) (bit : Fin CanonicalFieldBits.bitCount) :
    value.ccsPublic.val.getD
        (1 + MemoryBoundCcsPublic.digestBitCount +
          CanonicalFieldBits.bitCount * lane.val + bit.val) 0 =
      (CanonicalFieldBits.encode
        (ProductionMemoryBoundCcsPublic.memoryDigest batch lane)).val.getD
          bit.val 0 := by
  have selected := FixedBits.slice_getD value.ccsPublic
    (1 + MemoryBoundCcsPublic.digestBitCount +
      CanonicalFieldBits.bitCount * lane.val)
    CanonicalFieldBits.bitCount (by
      have laneBound := lane.isLt
      norm_num [MemoryBoundCcsPublic.coordinateCount,
        MemoryBoundCcsPublic.digestBitCount,
        CanonicalFieldBits.bitCount] at *
      omega) bit.val bit.isLt
  have wordEqual := congrArg
    (fun word : CanonicalFieldBits.Word => word.val.getD bit.val 0)
    (exact.memoryDigestWords lane)
  calc
    value.ccsPublic.val.getD
        (1 + MemoryBoundCcsPublic.digestBitCount +
          CanonicalFieldBits.bitCount * lane.val + bit.val) 0 =
      (claimMemoryDigestWord value lane).val.getD bit.val 0 := selected.symm
    _ = (CanonicalFieldBits.encode
        (ProductionMemoryBoundCcsPublic.memoryDigest batch lane)).val.getD
          bit.val 0 := wordEqual

/-- All fieldwise row results assemble to the exact 540-coordinate authority
relation. -/
theorem fullMatches
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    {value : ProductionFieldNativeFullClaim.Value candidate fullShape}
    {stateDigest : CanonicalDigest}
    {batch : ProductionMemoryBatchPoseidonBinding.Batch candidate}
    (exact : CcsPublicExact value stateDigest batch) :
    ProductionMemoryBoundCcsPublic.FullMatches
      value.ccsPublic stateDigest batch := by
  unfold ProductionMemoryBoundCcsPublic.FullMatches
  apply List.ext_get
  · exact value.ccsPublic.property.1.trans
      (ProductionMemoryBoundCcsPublic.encode_length stateDigest batch).symm
  · intro index leftBound rightBound
    have indexLimit : index < MemoryBoundCcsPublic.coordinateCount := by
      simpa [value.ccsPublic.property.1] using leftBound
    have leftGetD : value.ccsPublic.val.getD index 0 =
        value.ccsPublic.val[index] := by
      simp [List.getD_eq_getElem?_getD, leftBound]
    have encodingBound : index <
        (ProductionMemoryBoundCcsPublic.encode stateDigest batch).length := by
      rw [ProductionMemoryBoundCcsPublic.encode_length]
      exact indexLimit
    have rightGetD :
        (ProductionMemoryBoundCcsPublic.encode stateDigest batch).getD
            index 0 =
          (ProductionMemoryBoundCcsPublic.encode stateDigest batch)[index] := by
      simp [List.getD_eq_getElem?_getD, encodingBound]
    have getDEqual : value.ccsPublic.val.getD index 0 =
        (ProductionMemoryBoundCcsPublic.encode stateDigest batch).getD
          index 0 := by
      by_cases affine : index = 0
      · subst index
        simpa [ProductionMemoryBoundCcsPublic.encode,
          List.getD_eq_getElem?_getD] using exact.affineOne
      by_cases stateRegion :
          index < 1 + MemoryBoundCcsPublic.digestBitCount
      · let flat := index - 1
        have flatBound : flat <
            MemoryBoundCcsPublic.digestBitCount := by
          simp only [flat]
          omega
        let lane : Fin 4 :=
          ⟨flat / CanonicalFieldBits.bitCount, by
            norm_num [MemoryBoundCcsPublic.digestBitCount,
              CanonicalFieldBits.bitCount] at flatBound ⊢
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
        exact (exact.getD_stateDigest lane bit).trans
          (ProductionMemoryBoundCcsPublic.encode_get_stateDigest
            stateDigest batch lane bit).symm
      · by_cases memoryRegion :
          index < 1 + MemoryBoundCcsPublic.digestBitCount +
            MemoryBoundCcsPublic.digestBitCount
        · let flat := index -
            (1 + MemoryBoundCcsPublic.digestBitCount)
          have flatBound : flat <
              MemoryBoundCcsPublic.digestBitCount := by
            simp only [flat]
            omega
          let lane : Fin 4 :=
            ⟨flat / CanonicalFieldBits.bitCount, by
              norm_num [MemoryBoundCcsPublic.digestBitCount,
                CanonicalFieldBits.bitCount] at flatBound ⊢
              omega⟩
          let bit : Fin CanonicalFieldBits.bitCount :=
            ⟨flat % CanonicalFieldBits.bitCount,
              Nat.mod_lt _ (by norm_num [CanonicalFieldBits.bitCount])⟩
          have indexEq :
              1 + MemoryBoundCcsPublic.digestBitCount +
                CanonicalFieldBits.bitCount * lane.val + bit.val = index := by
            change 1 + MemoryBoundCcsPublic.digestBitCount +
                CanonicalFieldBits.bitCount *
                  (flat / CanonicalFieldBits.bitCount) +
                flat % CanonicalFieldBits.bitCount = index
            have division := Nat.mod_add_div flat CanonicalFieldBits.bitCount
            have flatEq :
                1 + MemoryBoundCcsPublic.digestBitCount + flat =
                  index := by
              simp only [flat]
              omega
            omega
          rw [← indexEq]
          exact (exact.getD_memoryDigest lane bit).trans
            (ProductionMemoryBoundCcsPublic.encode_get_memoryDigest
              stateDigest batch lane bit).symm
        · let padding := index -
            (1 + MemoryBoundCcsPublic.digestBitCount +
              MemoryBoundCcsPublic.digestBitCount)
          have paddingBound :
              padding < MemoryBoundCcsPublic.paddingBitCount := by
            simp only [padding]
            norm_num [MemoryBoundCcsPublic.coordinateCount,
              MemoryBoundCcsPublic.digestBitCount,
              MemoryBoundCcsPublic.paddingBitCount] at *
            omega
          have indexEq :
              1 + MemoryBoundCcsPublic.digestBitCount +
                MemoryBoundCcsPublic.digestBitCount + padding =
                  index := by
            simp only [padding]
            omega
          rw [← indexEq]
          exact (exact.paddingZero padding paddingBound).trans
            (ProductionMemoryBoundCcsPublic.encode_get_padding
              stateDigest batch padding paddingBound).symm
    exact leftGetD.symm.trans (getDEqual.trans rightGetD)

end CcsPublicExact

/-- Main exponent-independent result. Only the 540 physically placed CCS
coordinates are needed; the running-state width is irrelevant to this
section. -/
theorem rows_imply_fullMatches_of_ccsPublicPlaced
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    {layout : Layout candidate} {assignment : Nat -> Nat}
    {headers : FPrime.ChainHeaders Digest.Value}
    {value : ProductionFieldNativeFullClaim.Value candidate fullShape}
    {stateDigest : CanonicalDigest}
    (valid : layout.Valid)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : CcsPublicPlaced layout assignment value)
    (statePlaced : StateDigestPlaced layout assignment stateDigest)
    (memory : ProductionMemoryCheckedBatchRows.Result
      layout.batch.frame.memory assignment headers)
    (holds : Satisfies (rows layout) assignment) :
    ProductionMemoryBoundCcsPublic.FullMatches
      value.ccsPublic stateDigest memory.suffixBatch :=
  (claimCcsPublicExact_of_ccsPublicPlaced valid canonical one placed
    statePlaced memory holds).fullMatches

end Nightstream.Implementation.NebulaV2.ProductionMemoryBatchCcsLinkRows
