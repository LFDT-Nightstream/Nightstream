import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270.PublicWriteTrace
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.ExecutionAudit

/-!
Active post-PiDEC public-write artifact refinement for Carrier270.

Assurance tier: artifact-checked and fixed-profile Rust-conformant for the
active recursive public-write schedule; model-proved for its typed Carrier270
consequence.

Owns: fail-closed decoding of the two generated 135-record runtime shards;
their exact, disjoint coverage of public columns `0..269`; construction of the
previously pending one-arm exporter certificate; and specialization of the
generic public-write refinement without a caller-supplied trace certificate.

Does not own: the caller's conventional constant-one source fact, private
assignment decoding, final sparse A/B/C equality, CCS/CE membership,
commitment-key alignment, protocol acceptance, or row removal.

Emits constraints: none.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.fixed_point.assignment.public_write_trace.execution.profile` | generated header is the active 270-coordinate recursive profile | checked artifact |
| `f_prime.fixed_point.assignment.public_write_trace.execution.chunk.{0,1}` | each 135-record runtime shard decodes to its exact canonical writes | checked artifact |
| `f_prime.fixed_point.assignment.public_write_trace.execution.coverage` | the two shards cover `0..269` exactly once | derived |
| `f_prime.fixed_point.assignment.public_write_trace.execution.refinement` | executing the artifact trace refines the typed Carrier270 public input | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.ProductionPublicWriteTrace

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicAssignment
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PhysicalPublicAssignment
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicWriteTrace

abbrev RuntimeRawPublicWrite :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.RawPublicWrite

private def chunk0 : List RuntimeRawPublicWrite :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.PublicWrites.Chunk0.values

private def chunk1 : List RuntimeRawPublicWrite :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.PublicWrites.Chunk1.values

private def header :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.Header.value

def productionArm : Nat := 2
def shardWidth : Nat := 135
def directBuilderOffset : Nat := 7802

/-- The generated execution header pins the active recursive profile used by
the two public-write shards.  The proof reduces only the named scalar fields;
it does not evaluate the header's unrelated transcript payload. -/
theorem productionProfile_exact :
    header.schemaVersion = 1 /\
      header.branch = productionArm /\
      header.logicalColumns = 11437038 /\
      header.packedRows = 54 /\
      header.packedColumns = 211797 /\
      header.finalColumns = 11437038 /\
      header.publicWriteCount = PublicDecoder.alignedPublicWidth :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

private def decodedKind : Nat -> RawPublicWriteKind
  | 0 => .constantOne
  | 1 => .directSource
  | _ => .initializedZero

private def sourceShapeMatches (column : Fin PublicDecoder.alignedPublicWidth)
    (raw : RuntimeRawPublicWrite) : Bool :=
  if column.val = 0 then
    raw.sourceKind == 0 &&
      raw.builderColumn == some 0 &&
      raw.normalizedSourceColumn == none &&
      raw.width == 1 &&
      !raw.centered &&
      raw.aliasSource == none &&
      raw.value == 1
  else if column.val < legacyPublicWidth then
    raw.sourceKind == 1 &&
      raw.builderColumn == some (directBuilderOffset + column.val) &&
      raw.normalizedSourceColumn == some column.val &&
      raw.width == 1 &&
      !raw.centered &&
      raw.aliasSource == none
  else
    raw.sourceKind == 2 &&
      raw.builderColumn == none &&
      raw.normalizedSourceColumn == none &&
      raw.width == 0 &&
      !raw.centered &&
      raw.aliasSource == none &&
      raw.value == 0

/-- Fail-closed translation from one runtime artifact record to the generic
one-arm write language.  Builder resolution, packed address, normalized
source and target, width, centeredness, aliasing, and fixed values are all
checked before the record is admitted. -/
def decodeWrite (column : Fin PublicDecoder.alignedPublicWidth)
    (raw : RuntimeRawPublicWrite) : Option RawPublicWrite :=
  if raw.schemaVersion == 1 &&
      raw.logicalColumn == column.val &&
      raw.packedRow == column.val % 54 &&
      raw.packedColumn == column.val / 54 &&
      raw.normalizedColumn == column.val &&
      sourceShapeMatches column raw then
    some
      { schemaVersion := raw.schemaVersion
        arm := productionArm
        logicalColumn := raw.logicalColumn
        normalizedSourceColumn := raw.normalizedSourceColumn
        finalColumn := raw.normalizedColumn
        kind := decodedKind raw.sourceKind
        width := raw.width
        centered := raw.centered
        aliasSource := raw.aliasSource }
  else
    none

private def invalidWrite : RawPublicWrite :=
  { schemaVersion := 0
    arm := productionArm
    logicalColumn := 0
    normalizedSourceColumn := none
    finalColumn := 0
    kind := .initializedZero
    width := 0
    centered := false
    aliasSource := none }

private def firstColumn (index : Fin shardWidth) :
    Fin PublicDecoder.alignedPublicWidth :=
  ⟨index.val, by
    have := index.isLt
    simp only [shardWidth, PublicDecoder.alignedPublicWidth] at this ⊢
    omega⟩

private def secondColumn (index : Fin shardWidth) :
    Fin PublicDecoder.alignedPublicWidth :=
  ⟨shardWidth + index.val, by
    have := index.isLt
    simp only [shardWidth, PublicDecoder.alignedPublicWidth] at this ⊢
    omega⟩

def RuntimeValueShape (column : Fin PublicDecoder.alignedPublicWidth)
    (raw : RuntimeRawPublicWrite) : Prop :=
  (column.val = 0 -> raw.value = 1) /\
    (legacyPublicWidth <= column.val -> raw.value = 0)

instance (column : Fin PublicDecoder.alignedPublicWidth)
    (raw : RuntimeRawPublicWrite) :
    Decidable (RuntimeValueShape column raw) := by
  unfold RuntimeValueShape
  infer_instance

/-- Executable certificate over exactly 135 proof-free runtime records. -/
theorem generated_chunk0_runtime_exact :
    forall index : Fin shardWidth,
      decodeWrite (firstColumn index) (chunk0.getD index.val default) =
          some (expectedWrite productionArm (firstColumn index)) /\
        RuntimeValueShape (firstColumn index)
          (chunk0.getD index.val default) := by
  native_decide

/-- Executable certificate over exactly 135 proof-free runtime records. -/
theorem generated_chunk1_runtime_exact :
    forall index : Fin shardWidth,
      decodeWrite (secondColumn index) (chunk1.getD index.val default) =
          some (expectedWrite productionArm (secondColumn index)) /\
        RuntimeValueShape (secondColumn index)
          (chunk1.getD index.val default) := by
  native_decide

/-- Exact raw runtime record at each active public column.  The split is total,
disjoint, and has no remainder because `270 = 135 + 135`. -/
def productionRawWrite
    (column : Fin PublicDecoder.alignedPublicWidth) : RuntimeRawPublicWrite :=
  if first : column.val < shardWidth then
    chunk0.getD column.val default
  else
    chunk1.getD (column.val - shardWidth) default

private theorem productionRawWrite_firstColumn (index : Fin shardWidth) :
    productionRawWrite (firstColumn index) =
      chunk0.getD index.val default := by
  rw [productionRawWrite, dif_pos]
  · rfl
  · have indexBound := index.isLt
    simpa [firstColumn] using indexBound

private theorem productionRawWrite_secondColumn (index : Fin shardWidth) :
    productionRawWrite (secondColumn index) =
      chunk1.getD index.val default := by
  rw [productionRawWrite, dif_neg]
  · have offsetExact : (secondColumn index).val - shardWidth = index.val := by
      simp only [secondColumn]
      omega
    rw [offsetExact]
  · simp [secondColumn, shardWidth]

theorem productionRawWrite_decodes
    (column : Fin PublicDecoder.alignedPublicWidth) :
    decodeWrite column (productionRawWrite column) =
      some (expectedWrite productionArm column) := by
  by_cases first : column.val < shardWidth
  · have columnExact : firstColumn ⟨column.val, first⟩ = column := by
      apply Fin.ext
      rfl
    rw [← columnExact, productionRawWrite_firstColumn]
    exact (generated_chunk0_runtime_exact ⟨column.val, first⟩).1
  · have offsetBound : column.val - shardWidth < shardWidth := by
      have columnBound := column.isLt
      simp only [shardWidth, PublicDecoder.alignedPublicWidth] at first columnBound ⊢
      omega
    let index : Fin shardWidth := ⟨column.val - shardWidth, offsetBound⟩
    have columnExact : secondColumn index = column := by
      apply Fin.ext
      simp only [secondColumn, index]
      have columnBound := column.isLt
      simp only [shardWidth, PublicDecoder.alignedPublicWidth] at first columnBound ⊢
      omega
    rw [← columnExact, productionRawWrite_secondColumn]
    exact (generated_chunk1_runtime_exact index).1

theorem productionRawWrite_valueShape
    (column : Fin PublicDecoder.alignedPublicWidth) :
    RuntimeValueShape column (productionRawWrite column) := by
  by_cases first : column.val < shardWidth
  · have columnExact : firstColumn ⟨column.val, first⟩ = column := by
      apply Fin.ext
      rfl
    rw [← columnExact, productionRawWrite_firstColumn]
    exact (generated_chunk0_runtime_exact ⟨column.val, first⟩).2
  · have offsetBound : column.val - shardWidth < shardWidth := by
      have columnBound := column.isLt
      simp only [shardWidth, PublicDecoder.alignedPublicWidth] at first columnBound ⊢
      omega
    let index : Fin shardWidth := ⟨column.val - shardWidth, offsetBound⟩
    have columnExact : secondColumn index = column := by
      apply Fin.ext
      simp only [secondColumn, index]
      have columnBound := column.isLt
      simp only [shardWidth, PublicDecoder.alignedPublicWidth] at first columnBound ⊢
      omega
    rw [← columnExact, productionRawWrite_secondColumn]
    exact (generated_chunk1_runtime_exact index).2

/-- Exact two-shard trace. Decoder failure returns an invalid record, so no
malformed generated record can become canonical through the fallback. -/
def productionTrace : OneArmTrace :=
  fun column =>
    (decodeWrite column (productionRawWrite column)).getD invalidWrite

theorem generated_chunk0_exact :
    forall index : Fin shardWidth,
      (decodeWrite (firstColumn index)
        (chunk0.getD index.val default)).getD invalidWrite =
          expectedWrite productionArm (firstColumn index) := by
  intro index
  rw [(generated_chunk0_runtime_exact index).1]
  rfl

theorem generated_chunk1_exact :
    forall index : Fin shardWidth,
      (decodeWrite (secondColumn index)
        (chunk1.getD index.val default)).getD invalidWrite =
          expectedWrite productionArm (secondColumn index) := by
  intro index
  rw [(generated_chunk1_runtime_exact index).1]
  rfl

/-- The two 135-record shards cover every public coordinate exactly once and
construct the formerly pending exporter certificate. -/
theorem productionTrace_certificate :
    PendingProductionExporterCertificate productionArm productionTrace := by
  intro column
  rw [productionTrace, productionRawWrite_decodes]
  rfl

/-- Actual generated active-arm trace refinement.  Unlike the generic parent
theorem, this statement has no exporter-certificate or public-dataflow
premise.  The remaining constant-one premise belongs to its physical source
row and is deliberately not fabricated by this schedule artifact. -/
theorem production_projectPhysical270_execute_eq_projectPublicInput
    (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (suffix : Fin PublicPaddingRefinement.Artifact.relationColumns -> F)
    (constantOne : SourceConstantOne dimensions legacy) :
    projectPhysical270 dimensions
        (executePhysical productionArm productionTrace
          (sourcePublicPrefix dimensions legacy) suffix) =
      projectPublicInput (assignment dimensions legacy) := by
  exact projectPhysical270_execute_eq_projectPublicInput dimensions legacy
    productionArm productionTrace suffix productionTrace_certificate
    constantOne

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.ProductionPublicWriteTrace
