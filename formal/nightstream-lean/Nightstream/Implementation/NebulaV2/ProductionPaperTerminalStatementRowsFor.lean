import Nightstream.Implementation.NebulaV2.ProductionPaperTerminalPublicRowsFor
import Nightstream.Implementation.NebulaV2.ProductionWasmStatePublicRows

/-!
Contract: exact V2 public-statement bit recomposition for the terminal relation.

The source is the complete 7,868-bit `WasmPublicStatementCodec` image. The
rows derive every recombined value used by terminal public checks: both full
application states, segment count, final timestamp, real application-row
count, and all four final-memory-root lanes.

`BitsPlaced` is the only parser boundary. It states exact placement of the
verifier-owned public bits. It contains none of the recombined equalities in
`ProductionPaperTerminalPublicRowsFor.StatementPlaced`.

The four digest lanes use 64-bit recomposition with a proof that the exact
decoded word is below Goldilocks. This prevents the deterministic `0` versus
`q` alias.

Assurance tier: exponent-indexed implementation-to-codec bridge.

Does not own byte parsing, generated public-column allocation, Rust
refinement, or terminal proof verification.

Emits constraints: 177 rows.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.NebulaV2.ProductionPaperTerminalStatementRowsFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.BoundedWordRows
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.WasmPublicStatementEncoding
open Nightstream.Protocol.NebulaV2.WasmStateEncoding
open Nightstream.Protocol.NebulaV2.WasmStatement

def statementBitCount : Nat := 7868
def initialStateOffset : Nat := 2880
def segmentCountOffset : Nat := 5173
def finalTimestampOffset : Nat := 5180
def resultOffset : Nat := 5203
def realRowsOffset : Nat := resultOffset
def finalStateOffset : Nat := resultOffset + 18
def finalMemoryRootOffset : Nat := resultOffset + 2409

structure Layout where
  publicBitStart : Nat
  statement : ProductionPaperTerminalPublicRowsFor.StatementLayout

def Layout.initialState (layout : Layout) :
    ProductionWasmStatePublicRows.Layout where
  publicBitStart := layout.publicBitStart + initialStateOffset
  fieldColumn := layout.statement.initialApplicationColumn

def Layout.finalState (layout : Layout) :
    ProductionWasmStatePublicRows.Layout where
  publicBitStart := layout.publicBitStart + finalStateOffset
  fieldColumn := layout.statement.finalApplicationColumn

def Layout.segmentCountWord (layout : Layout) : BoundedWordRows.Layout where
  width := segmentCountBitWidth
  valueColumn := layout.statement.segmentCountColumn
  bitStart := layout.publicBitStart + segmentCountOffset

def Layout.finalTimestampWord (layout : Layout) : BoundedWordRows.Layout where
  width := finalTimestampBitWidth
  valueColumn := layout.statement.finalTimestampColumn
  bitStart := layout.publicBitStart + finalTimestampOffset

def Layout.realRowsWord (layout : Layout) : BoundedWordRows.Layout where
  width := 18
  valueColumn := layout.statement.realApplicationRowsColumn
  bitStart := layout.publicBitStart + realRowsOffset

def Layout.finalMemoryRootWord (layout : Layout) (lane : Fin 4) :
    BoundedWordRows.Layout where
  width := 64
  valueColumn := layout.statement.finalMemoryRootColumn lane
  bitStart := layout.publicBitStart + finalMemoryRootOffset + lane.val * 64

def scalarRows (layout : Layout) : List Row :=
  [ layout.segmentCountWord.recompositionRow
  , layout.finalTimestampWord.recompositionRow
  , layout.realRowsWord.recompositionRow
  ]

def rootRows (layout : Layout) : List Row :=
  List.ofFn fun lane : Fin 4 =>
    (layout.finalMemoryRootWord lane).recompositionRow

def rows (layout : Layout) : List Row :=
  ProductionWasmStatePublicRows.rows layout.initialState ++
    (scalarRows layout ++
      (ProductionWasmStatePublicRows.rows layout.finalState ++
        rootRows layout))

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 177 := by
  simp [rows, scalarRows, rootRows,
    ProductionWasmStatePublicRows.rows_length_exact]

/-- Exact placement of all verifier-owned public bits. -/
def BitsPlaced (layout : Layout) (assignment : Nat -> Nat)
    (image : PublicImage) : Prop :=
  forall offset (bound : offset < statementBitCount),
    assignment (layout.publicBitStart + offset) =
      (WasmPublicStatementCodec.encode image).get
        ⟨offset, by
          simpa [statementBitCount,
            WasmPublicStatementCodec.encode_length] using bound⟩

private def leafBlocks (image : PublicImage) : List (List Nat) :=
  [ WasmPublicStatementCodec.encodeIdentity image.identity
  , WasmStateCodec.encode image.initialApplicationState
  , WasmStateCodec.encodeWord segmentCountBitWidth image.segmentCount
  , WasmStateCodec.encodeWord finalTimestampBitWidth
      image.finalGlobalTimestamp
  , WasmStateCodec.encodeWord 18 image.result.realApplicationRowCount
  , WasmStateCodec.encode image.result.finalApplicationState
  , WasmStateCodec.encodeWord 1
      (WasmResultCodec.modeValue image.result.terminationMode)
  , WasmStateCodec.encodeWord 32 image.result.exitCode
  , WasmStateCodec.encodeWord 1 image.result.outputPresent.toNat
  , WasmStateCodec.encodeWord 32 image.result.outputValueLow
  , WasmStateCodec.encodeWord 32 image.result.outputValueHigh
  , WasmStateCodec.encodeWord 64
      (image.result.finalMemoryRoot.lanes ⟨0, by
        norm_num [Nightstream.Protocol.NebulaV2.Digest.laneCount]⟩).val
  , WasmStateCodec.encodeWord 64
      (image.result.finalMemoryRoot.lanes ⟨1, by
        norm_num [Nightstream.Protocol.NebulaV2.Digest.laneCount]⟩).val
  , WasmStateCodec.encodeWord 64
      (image.result.finalMemoryRoot.lanes ⟨2, by
        norm_num [Nightstream.Protocol.NebulaV2.Digest.laneCount]⟩).val
  , WasmStateCodec.encodeWord 64
      (image.result.finalMemoryRoot.lanes ⟨3, by
        norm_num [Nightstream.Protocol.NebulaV2.Digest.laneCount]⟩).val
  ]

private theorem encode_eq_leafBlocks (image : PublicImage) :
    WasmPublicStatementCodec.encode image = (leafBlocks image).flatten := by
  simp [leafBlocks, WasmPublicStatementCodec.encode,
    WasmPublicStatementCodec.blocks, WasmResultCodec.encode,
    WasmResultCodec.blocks, WasmResultCodec.encodeDigest,
    WasmResultCodec.encodeDigestAux,
    Nightstream.Protocol.NebulaV2.Digest.laneCount,
    Nightstream.Protocol.NebulaV2.Digest.laneBitWidth]

private theorem leafSlice (image : PublicImage) (index : Fin 15) :
    ((WasmPublicStatementCodec.encode image).drop
        (TaggedBitSlices.offsetAt List.length (leafBlocks image) index.val)).take
          ((leafBlocks image).get index).length =
      (leafBlocks image).get index := by
  rw [encode_eq_leafBlocks]
  simpa [TaggedBitSlices.flatten] using
    (TaggedBitSlices.slice_flatten_at
      (fun block : List Nat => block) List.length (fun _ => rfl)
      (leafBlocks image) index.val (by simpa [leafBlocks] using index.isLt))

private theorem initialStateSlice (image : PublicImage) :
    ((WasmPublicStatementCodec.encode image).drop initialStateOffset).take
        2293 = WasmStateCodec.encode image.initialApplicationState := by
  simpa [leafBlocks, TaggedBitSlices.offsetAt, initialStateOffset,
    WasmPublicStatementCodec.encodeIdentity_length,
    WasmPublicStatementEncoding.identitySerializedBitCount_eq,
    WasmStateCodec.encode_exact_length] using
    (leafSlice image ⟨1, by decide⟩)

private theorem segmentCountSlice (image : PublicImage) :
    ((WasmPublicStatementCodec.encode image).drop segmentCountOffset).take
        segmentCountBitWidth =
      WasmStateCodec.encodeWord segmentCountBitWidth image.segmentCount := by
  simpa [leafBlocks, TaggedBitSlices.offsetAt, segmentCountOffset,
    WasmPublicStatementCodec.encodeIdentity_length,
    WasmPublicStatementEncoding.identitySerializedBitCount_eq,
    WasmStateCodec.encode_exact_length, WasmStateCodec.encodeWord_length,
    segmentCountBitWidth] using (leafSlice image ⟨2, by decide⟩)

private theorem finalTimestampSlice (image : PublicImage) :
    ((WasmPublicStatementCodec.encode image).drop finalTimestampOffset).take
        finalTimestampBitWidth =
      WasmStateCodec.encodeWord finalTimestampBitWidth
        image.finalGlobalTimestamp := by
  simpa [leafBlocks, TaggedBitSlices.offsetAt, finalTimestampOffset,
    WasmPublicStatementCodec.encodeIdentity_length,
    WasmPublicStatementEncoding.identitySerializedBitCount_eq,
    WasmStateCodec.encode_exact_length, WasmStateCodec.encodeWord_length,
    segmentCountBitWidth, finalTimestampBitWidth] using
    (leafSlice image ⟨3, by decide⟩)

private theorem realRowsSlice (image : PublicImage) :
    ((WasmPublicStatementCodec.encode image).drop realRowsOffset).take 18 =
      WasmStateCodec.encodeWord 18 image.result.realApplicationRowCount := by
  simpa [leafBlocks, TaggedBitSlices.offsetAt, realRowsOffset, resultOffset,
    WasmPublicStatementCodec.encodeIdentity_length,
    WasmPublicStatementEncoding.identitySerializedBitCount_eq,
    WasmStateCodec.encode_exact_length, WasmStateCodec.encodeWord_length,
    segmentCountBitWidth, finalTimestampBitWidth] using
    (leafSlice image ⟨4, by decide⟩)

private theorem finalStateSlice (image : PublicImage) :
    ((WasmPublicStatementCodec.encode image).drop finalStateOffset).take 2293 =
      WasmStateCodec.encode image.result.finalApplicationState := by
  simpa [leafBlocks, TaggedBitSlices.offsetAt, finalStateOffset, resultOffset,
    WasmPublicStatementCodec.encodeIdentity_length,
    WasmPublicStatementEncoding.identitySerializedBitCount_eq,
    WasmStateCodec.encode_exact_length, WasmStateCodec.encodeWord_length,
    segmentCountBitWidth, finalTimestampBitWidth] using
    (leafSlice image ⟨5, by decide⟩)

private theorem finalMemoryRootSlice (image : PublicImage) (lane : Fin 4) :
    ((WasmPublicStatementCodec.encode image).drop
        (finalMemoryRootOffset + lane.val * 64)).take 64 =
      WasmStateCodec.encodeWord 64
        (image.result.finalMemoryRoot.lanes lane).val := by
  fin_cases lane
  · simpa [leafBlocks, TaggedBitSlices.offsetAt, finalMemoryRootOffset,
      resultOffset, WasmPublicStatementCodec.encodeIdentity_length,
      WasmPublicStatementEncoding.identitySerializedBitCount_eq,
      WasmStateCodec.encode_exact_length, WasmStateCodec.encodeWord_length,
      segmentCountBitWidth, finalTimestampBitWidth] using
      (leafSlice image ⟨11, by decide⟩)
  · simpa [leafBlocks, TaggedBitSlices.offsetAt, finalMemoryRootOffset,
      resultOffset, WasmPublicStatementCodec.encodeIdentity_length,
      WasmPublicStatementEncoding.identitySerializedBitCount_eq,
      WasmStateCodec.encode_exact_length, WasmStateCodec.encodeWord_length,
      segmentCountBitWidth, finalTimestampBitWidth] using
      (leafSlice image ⟨12, by decide⟩)
  · simpa [leafBlocks, TaggedBitSlices.offsetAt, finalMemoryRootOffset,
      resultOffset, WasmPublicStatementCodec.encodeIdentity_length,
      WasmPublicStatementEncoding.identitySerializedBitCount_eq,
      WasmStateCodec.encode_exact_length, WasmStateCodec.encodeWord_length,
      segmentCountBitWidth, finalTimestampBitWidth] using
      (leafSlice image ⟨13, by decide⟩)
  · simpa [leafBlocks, TaggedBitSlices.offsetAt, finalMemoryRootOffset,
      resultOffset, WasmPublicStatementCodec.encodeIdentity_length,
      WasmPublicStatementEncoding.identitySerializedBitCount_eq,
      WasmStateCodec.encode_exact_length, WasmStateCodec.encodeWord_length,
      segmentCountBitWidth, finalTimestampBitWidth] using
      (leafSlice image ⟨14, by decide⟩)

private theorem sectionWordDigits
    {layout : Layout} {assignment : Nat -> Nat} {image : PublicImage}
    (placed : BitsPlaced layout assignment image)
    (sourceOffset width valueColumn value : Nat)
    (inside : sourceOffset + width ≤ statementBitCount)
    (sliceExact :
      ((WasmPublicStatementCodec.encode image).drop sourceOffset).take width =
        WasmStateCodec.encodeWord width value) :
    (BoundedWordRows.Layout.digits
      { width := width
        valueColumn := valueColumn
        bitStart := layout.publicBitStart + sourceOffset }
      assignment) = WasmStateCodec.encodeWord width value := by
  rw [← sliceExact]
  apply List.ext_get
  · rw [BoundedWordRows.Layout.digits_length, List.length_take,
      List.length_drop, WasmPublicStatementCodec.encode_length]
    have leSub : width ≤ statementBitCount - sourceOffset := by omega
    simpa [statementBitCount] using (min_eq_left leSub).symm
  · intro offset leftBound rightBound
    have offsetBound : offset < width := by
      simpa [BoundedWordRows.Layout.digits] using leftBound
    have globalBound : sourceOffset + offset < statementBitCount := by omega
    have source := placed (sourceOffset + offset) globalBound
    simpa [BoundedWordRows.Layout.digits,
      BoundedWordRows.Layout.bitColumn, List.getElem_drop,
      Nat.add_assoc] using source

private theorem getD_eq_getElem_of_lt
    {Alpha : Type} (values : List Alpha) (index : Nat)
    (default : Alpha) (bounded : index < values.length) :
    values.getD index default = values[index] := by
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem bounded]
  rfl

private theorem initialBitsPlaced
    {layout : Layout} {assignment : Nat -> Nat} {image : PublicImage}
    (placed : BitsPlaced layout assignment image) :
    ProductionWasmStatePublicRows.BitsPlaced layout.initialState assignment
      image.initialApplicationState := by
  intro offset bound
  have source := placed (initialStateOffset + offset) (by
    simp [statementBitCount, initialStateOffset]
    omega)
  have selected := congrArg
    (fun bits => bits.getD offset 0)
    (initialStateSlice image)
  dsimp only at selected
  have stateBound :
      offset < (WasmStateCodec.encode image.initialApplicationState).length := by
    simpa [WasmStateCodec.encode_exact_length] using bound
  have sliceBound :
      offset < (((WasmPublicStatementCodec.encode image).drop
        initialStateOffset).take 2293).length := by
    rw [initialStateSlice image]
    exact stateBound
  rw [getD_eq_getElem_of_lt _ _ _ sliceBound,
    getD_eq_getElem_of_lt _ _ _ stateBound] at selected
  have selectedGlobal :
      (WasmPublicStatementCodec.encode image).get
          ⟨initialStateOffset + offset, by
            simpa [WasmPublicStatementCodec.encode_length,
              initialStateOffset] using
              (show initialStateOffset + offset < statementBitCount by
                simp [statementBitCount, initialStateOffset]
                omega)⟩ =
        (WasmStateCodec.encode image.initialApplicationState).get
          ⟨offset, stateBound⟩ := by
    simpa [List.getElem_take, List.getElem_drop] using selected
  simpa [ProductionWasmStatePublicRows.BitsPlaced, Layout.initialState,
    initialStateOffset, Nat.add_assoc] using source.trans selectedGlobal

private theorem finalBitsPlaced
    {layout : Layout} {assignment : Nat -> Nat} {image : PublicImage}
    (placed : BitsPlaced layout assignment image) :
    ProductionWasmStatePublicRows.BitsPlaced layout.finalState assignment
      image.result.finalApplicationState := by
  intro offset bound
  have source := placed (finalStateOffset + offset) (by
    simp [statementBitCount, finalStateOffset, resultOffset]
    omega)
  have selected := congrArg
    (fun bits => bits.getD offset 0)
    (finalStateSlice image)
  dsimp only at selected
  have stateBound :
      offset < (WasmStateCodec.encode
        image.result.finalApplicationState).length := by
    simpa [WasmStateCodec.encode_exact_length] using bound
  have sliceBound :
      offset < (((WasmPublicStatementCodec.encode image).drop
        finalStateOffset).take 2293).length := by
    rw [finalStateSlice image]
    exact stateBound
  rw [getD_eq_getElem_of_lt _ _ _ sliceBound,
    getD_eq_getElem_of_lt _ _ _ stateBound] at selected
  have selectedGlobal :
      (WasmPublicStatementCodec.encode image).get
          ⟨finalStateOffset + offset, by
            simpa [WasmPublicStatementCodec.encode_length,
              finalStateOffset, resultOffset] using
              (show finalStateOffset + offset < statementBitCount by
                simp [statementBitCount, finalStateOffset, resultOffset]
                omega)⟩ =
        (WasmStateCodec.encode image.result.finalApplicationState).get
          ⟨offset, stateBound⟩ := by
    simpa [List.getElem_take, List.getElem_drop] using selected
  simpa [ProductionWasmStatePublicRows.BitsPlaced, Layout.finalState,
    finalStateOffset, resultOffset, Nat.add_assoc] using
      source.trans selectedGlobal

private theorem initial_rows_hold
    {layout : Layout} {assignment : Nat -> Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (ProductionWasmStatePublicRows.rows layout.initialState)
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem final_rows_hold
    {layout : Layout} {assignment : Nat -> Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (ProductionWasmStatePublicRows.rows layout.finalState)
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem scalar_row_mem
    (layout : Layout) {row : Row} (member : row ∈ scalarRows layout) :
    row ∈ rows layout := by
  simp [rows, member]

private theorem root_row_mem (layout : Layout) (lane : Fin 4) :
    (layout.finalMemoryRootWord lane).recompositionRow ∈ rows layout := by
  apply List.mem_append_right
  apply List.mem_append_right
  apply List.mem_append_right
  exact List.mem_ofFn.mpr ⟨lane, rfl⟩

private theorem narrowWordValue
    {layout : Layout} {assignment : Nat -> Nat} {image : PublicImage}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : BitsPlaced layout assignment image)
    (sourceOffset width valueColumn value : Nat)
    (inside : sourceOffset + width ≤ statementBitCount)
    (rangeFits : 2 ^ width ≤ goldilocksP)
    (valueBound : value < 2 ^ width)
    (sliceExact :
      ((WasmPublicStatementCodec.encode image).drop sourceOffset).take width =
        WasmStateCodec.encodeWord width value)
    (rowHolds : RowHolds assignment
      ({ width := width
         valueColumn := valueColumn
         bitStart := layout.publicBitStart + sourceOffset } :
        BoundedWordRows.Layout).recompositionRow) :
    assignment valueColumn = value := by
  let word : BoundedWordRows.Layout :=
    { width := width
      valueColumn := valueColumn
      bitStart := layout.publicBitStart + sourceOffset }
  have digits := sectionWordDigits placed sourceOffset width valueColumn value
    inside sliceExact
  have decodedBound : BoundedWordRows.decoded word assignment < 2 ^ width := by
    unfold BoundedWordRows.decoded word
    rw [digits, WasmStateCodec.ofDigits_encodeWord_of_bound valueBound]
    exact valueBound
  have exact := BoundedWordRows.recompositionRow_sound_of_decoded_bound
    rangeFits canonical one decodedBound rowHolds
  change assignment word.valueColumn = value
  rw [exact]
  unfold BoundedWordRows.decoded
  rw [digits, WasmStateCodec.ofDigits_encodeWord_of_bound valueBound]

private theorem rootWordValue
    {layout : Layout} {assignment : Nat -> Nat} {image : PublicImage}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : BitsPlaced layout assignment image)
    (lane : Fin 4)
    (rowHolds : RowHolds assignment
      (layout.finalMemoryRootWord lane).recompositionRow) :
    assignment (layout.statement.finalMemoryRootColumn lane) =
      (image.result.finalMemoryRoot.lanes lane).val := by
  let value := (image.result.finalMemoryRoot.lanes lane).val
  let word := layout.finalMemoryRootWord lane
  have inside : finalMemoryRootOffset + lane.val * 64 + 64 ≤
      statementBitCount := by
    fin_cases lane <;>
      norm_num [finalMemoryRootOffset, resultOffset, statementBitCount]
  have digits := sectionWordDigits placed
    (finalMemoryRootOffset + lane.val * 64) 64
    (layout.statement.finalMemoryRootColumn lane) value inside
    (finalMemoryRootSlice image lane)
  have digitsExact : word.digits assignment =
      WasmStateCodec.encodeWord 64 value := by
    simpa [word, Layout.finalMemoryRootWord, Nat.add_assoc] using digits
  have decodedBelowField : BoundedWordRows.decoded word assignment <
      goldilocksP := by
    unfold BoundedWordRows.decoded word
    rw [digitsExact, WasmStateCodec.ofDigits_encodeWord_of_bound]
    · exact (image.result.finalMemoryRoot.lanes lane).property
    · exact Nat.lt_trans (image.result.finalMemoryRoot.lanes lane).property
        CanonicalFieldBits.modulus_lt_capacity
  have canonicalTerms : CanonicalTerms word.terms := by
    apply word.terms_canonical_of_weight_bound
    intro offset offsetBound
    change offset < 64 at offsetBound
    have power : 2 ^ offset ≤ 2 ^ 63 :=
      Nat.pow_le_pow_right (by decide) (by omega)
    exact power.trans_lt (by norm_num [goldilocksP])
  have exact := BoundedWordRows.recompositionRow_sound_of_field_bound
    canonicalTerms canonical one decodedBelowField rowHolds
  change assignment word.valueColumn = value
  rw [exact]
  unfold BoundedWordRows.decoded
  rw [digitsExact, WasmStateCodec.ofDigits_encodeWord_of_bound]
  exact Nat.lt_trans (image.result.finalMemoryRoot.lanes lane).property
    CanonicalFieldBits.modulus_lt_capacity

/-- The exact 177 recomposition rows derive the former
`StatementPlaced` boundary from raw public bits. -/
theorem rows_imply_statementPlaced
    {Program : Type} {expectedProfile : Profile.Identity}
    {layout : Layout} {assignment : Nat -> Nat} {image : PublicImage}
    {statement : ProductionStatement Program}
    (decoded : image.DecodesFor expectedProfile statement)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : BitsPlaced layout assignment image)
    (holds : Satisfies (rows layout) assignment) :
    ProductionPaperTerminalPublicRowsFor.StatementPlaced layout.statement
      assignment statement := by
  have initialFields := ProductionWasmStatePublicRows.fields_exact
    decoded.initial_state_canonical canonical one (initialBitsPlaced placed)
    (initial_rows_hold holds)
  have finalFields := ProductionWasmStatePublicRows.fields_exact
    decoded.image_result_decodes.final_state_canonical canonical one
    (finalBitsPlaced placed) (final_rows_hold holds)
  have segment := narrowWordValue canonical one placed segmentCountOffset
    segmentCountBitWidth layout.statement.segmentCountColumn image.segmentCount
    (by norm_num [segmentCountOffset, segmentCountBitWidth,
      statementBitCount])
    (by norm_num [segmentCountBitWidth, goldilocksP])
    decoded.image_segment_count_bound (segmentCountSlice image)
    (holds _ (scalar_row_mem layout (by simp [scalarRows,
      Layout.segmentCountWord])))
  have timestamp := narrowWordValue canonical one placed finalTimestampOffset
    finalTimestampBitWidth layout.statement.finalTimestampColumn
    image.finalGlobalTimestamp
    (by norm_num [finalTimestampOffset, finalTimestampBitWidth,
      statementBitCount])
    (by norm_num [finalTimestampBitWidth, goldilocksP])
    decoded.image_final_timestamp_bound (finalTimestampSlice image)
    (holds _ (scalar_row_mem layout (by simp [scalarRows,
      Layout.finalTimestampWord])))
  have realRows := narrowWordValue canonical one placed realRowsOffset 18
    layout.statement.realApplicationRowsColumn
    image.result.realApplicationRowCount
    (by norm_num [realRowsOffset, resultOffset, statementBitCount])
    (by norm_num [goldilocksP])
    (by
      have resultDecoded := decoded.image_result_decodes
      rw [resultDecoded.exactImage]
      simpa [ResultImage.ofResult, Completion.realApplicationRowLimit] using
        resultDecoded.realRowCountBound)
    (realRowsSlice image)
    (holds _ (scalar_row_mem layout (by simp [scalarRows,
      Layout.realRowsWord])))
  have imageExact := decoded.exactImage
  refine
    { segmentCount := ?_
      finalTimestamp := ?_
      initialApplication := ?_
      realApplicationRows := ?_
      finalApplication := ?_
      finalMemoryRoot := ?_ }
  · rw [segment, imageExact]
    rfl
  · rw [timestamp, imageExact]
    rfl
  · intro index
    have coordinate := congrArg
      (fun values : List Nat => values.getD index.val 0) initialFields
    dsimp only at coordinate
    have leftBound : index.val <
        (List.ofFn fun coordinate : Fin 85 =>
          assignment (layout.initialState.fieldColumn coordinate)).length := by
      simp
    have rightBound : index.val <
        (ProductionWasmStateFields.encode image.initialApplicationState).length := by
      simpa [ProductionWasmStateFields.encode_length] using index.isLt
    rw [getD_eq_getElem_of_lt _ _ _ leftBound,
      getD_eq_getElem_of_lt _ _ _ rightBound] at coordinate
    simp only [List.getElem_ofFn] at coordinate
    simpa [Layout.initialState, imageExact, PublicImage.ofStatement] using
      coordinate
  · rw [realRows, imageExact]
    rfl
  · intro index
    have coordinate := congrArg
      (fun values : List Nat => values.getD index.val 0) finalFields
    dsimp only at coordinate
    have leftBound : index.val <
        (List.ofFn fun coordinate : Fin 85 =>
          assignment (layout.finalState.fieldColumn coordinate)).length := by
      simp
    have rightBound : index.val <
        (ProductionWasmStateFields.encode
          image.result.finalApplicationState).length := by
      simpa [ProductionWasmStateFields.encode_length] using index.isLt
    rw [getD_eq_getElem_of_lt _ _ _ leftBound,
      getD_eq_getElem_of_lt _ _ _ rightBound] at coordinate
    simp only [List.getElem_ofFn] at coordinate
    simpa [Layout.finalState, imageExact, PublicImage.ofStatement] using
      coordinate
  · intro lane
    have root := rootWordValue canonical one placed lane
      (holds _ (root_row_mem layout lane))
    simpa [imageExact, PublicImage.ofStatement] using root

end Nightstream.Implementation.NebulaV2.ProductionPaperTerminalStatementRowsFor
