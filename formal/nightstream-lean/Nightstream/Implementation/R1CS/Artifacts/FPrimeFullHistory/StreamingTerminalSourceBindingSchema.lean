import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import Nightstream.Implementation.R1CS.Core.Program

/-!
Contract: compact exact-row schema for terminal source-field decoding.

Rust owns the emitted rows. Decoder blocks coalesce equal retained-slot
encodings. Composite decoders keep each affine segment explicit. This schema
does not make a digest authoritative.

Emits constraints: no. It describes Rust-emitted constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalSourceBinding.Artifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

structure Range where
  start : Nat
  stop : Nat
deriving DecidableEq, Repr

abbrev Range.length (range : Range) : Nat := range.stop - range.start

abbrev Range.ValidWithin (range : Range) (bound : Nat) : Prop :=
  range.start ≤ range.stop ∧ range.stop ≤ bound

structure DecoderSegment where
  finalColumns : Range
  radix : Nat
  scale : Nat
deriving DecidableEq, Repr

def decoderTerms (start width radix scale : Nat) : List (Nat × Nat) :=
  (List.range width).map fun offset =>
    (start + offset, (scale * radix ^ offset) % goldilocksP)

def DecoderSegment.terms (segment : DecoderSegment) : List (Nat × Nat) :=
  decoderTerms segment.finalColumns.start segment.finalColumns.length
    segment.radix segment.scale

structure DecoderBlock where
  owner : String
  sourceFields : Range
  decodedColumns : Range
  finalColumns : Range
  width : Nat
  radix : Nat
  scale : Nat
deriving DecidableEq, Repr

def DecoderBlock.count (block : DecoderBlock) : Nat :=
  block.decodedColumns.length

def DecoderBlock.termsAt (block : DecoderBlock) (index : Nat) : List (Nat × Nat) :=
  decoderTerms (block.finalColumns.start + index * block.width)
    block.width block.radix block.scale

structure CompositeDecoder where
  owner : String
  sourceField : Nat
  decodedColumn : Nat
  segments : List DecoderSegment
deriving DecidableEq, Repr

def CompositeDecoder.terms (decoder : CompositeDecoder) : List (Nat × Nat) :=
  decoder.segments.flatMap DecoderSegment.terms

inductive DecoderGroup where
  | block : DecoderBlock → DecoderGroup
  | composite : CompositeDecoder → DecoderGroup
deriving DecidableEq, Repr

def DecoderGroup.count : DecoderGroup → Nat
  | .block item => item.count
  | .composite _ => 1

/-- Canonical CSC order has decoder terms before the newly allocated output.
Rust builder order differs only by this permutation. -/
def decoderRow (output : Nat) (terms : List (Nat × Nat)) : Row :=
  ⟨negateTerms terms ++ [(output, 1)], [(0, 1)], []⟩

def DecoderBlock.rows (block : DecoderBlock) : List Row :=
  (List.range block.count).map fun index =>
    decoderRow (block.decodedColumns.start + index) (block.termsAt index)

def DecoderGroup.rows : DecoderGroup → List Row
  | .block item => item.rows
  | .composite decoder => [decoderRow decoder.decodedColumn decoder.terms]

def DecoderBlock.Canonical (block : DecoderBlock) : Prop :=
  CanonicalTerms (decoderTerms 0 block.width block.radix block.scale)

def DecoderSegment.Canonical (segment : DecoderSegment) : Prop :=
  CanonicalTerms segment.terms

def CompositeDecoder.Canonical (decoder : CompositeDecoder) : Prop :=
  ∀ segment ∈ decoder.segments, segment.Canonical

instance (block : DecoderBlock) : Decidable block.Canonical := by
  unfold DecoderBlock.Canonical
  infer_instance

instance (segment : DecoderSegment) : Decidable segment.Canonical := by
  unfold DecoderSegment.Canonical
  infer_instance

instance (decoder : CompositeDecoder) : Decidable decoder.Canonical := by
  unfold CompositeDecoder.Canonical
  infer_instance

def DecoderGroup.Canonical : DecoderGroup → Prop
  | .block item => item.Canonical
  | .composite decoder => decoder.Canonical

instance (group : DecoderGroup) : Decidable group.Canonical := by
  cases group <;> simp only [DecoderGroup.Canonical] <;> infer_instance

def ownerValid (owner : String) : Prop :=
  owner = "x_out" ∨ owner = "nebula_lane" ∨
    owner = "local_state" ∨ owner = "delayed_payload"

instance (owner : String) : Decidable (ownerValid owner) := by
  unfold ownerValid
  infer_instance

def DecoderSegment.Valid
    (segment : DecoderSegment) (columns : Nat) (finalAssignment : Range) : Prop :=
  segment.finalColumns.ValidWithin columns ∧
    finalAssignment.start ≤ segment.finalColumns.start ∧
    segment.finalColumns.stop ≤ finalAssignment.stop ∧
    0 < segment.finalColumns.length ∧
    (segment.radix = 2 ∨ segment.radix = 3 ∨ segment.radix = 7) ∧
    0 < segment.scale ∧ segment.scale < goldilocksP

instance (segment : DecoderSegment) (columns : Nat) (finalAssignment : Range) :
    Decidable (segment.Valid columns finalAssignment) := by
  unfold DecoderSegment.Valid
  infer_instance

def DecoderGroup.Valid
    (group : DecoderGroup) (columns : Nat)
    (finalAssignment decoded : Range) : Prop :=
  match group with
  | .block item =>
      ownerValid item.owner ∧
      item.sourceFields.length = item.decodedColumns.length ∧
      0 < item.count ∧ 0 < item.width ∧
      item.finalColumns.length = item.count * item.width ∧
      item.decodedColumns.ValidWithin columns ∧
      decoded.start ≤ item.decodedColumns.start ∧
      item.decodedColumns.stop ≤ decoded.stop ∧
      item.finalColumns.ValidWithin columns ∧
      finalAssignment.start ≤ item.finalColumns.start ∧
      item.finalColumns.stop ≤ finalAssignment.stop ∧
      (item.radix = 2 ∨ item.radix = 3 ∨ item.radix = 7) ∧
      0 < item.scale ∧ item.scale < goldilocksP
  | .composite decoder =>
      ownerValid decoder.owner ∧
      decoded.start ≤ decoder.decodedColumn ∧ decoder.decodedColumn < decoded.stop ∧
      decoder.decodedColumn < columns ∧ decoder.segments ≠ [] ∧
      ∀ segment ∈ decoder.segments, segment.Valid columns finalAssignment

instance (group : DecoderGroup) (columns : Nat)
    (finalAssignment decoded : Range) :
    Decidable (group.Valid columns finalAssignment decoded) := by
  cases group <;> simp only [DecoderGroup.Valid] <;> infer_instance

def DecoderBlock.Holds (block : DecoderBlock) (assignment : Nat → Nat) : Prop :=
  ∀ index, index < block.count →
    assignment (block.decodedColumns.start + index) =
      lcEval assignment (block.termsAt index)

def CompositeDecoder.Holds
    (decoder : CompositeDecoder) (assignment : Nat → Nat) : Prop :=
  assignment decoder.decodedColumn = lcEval assignment decoder.terms

def DecoderGroup.Holds (group : DecoderGroup) (assignment : Nat → Nat) : Prop :=
  match group with
  | .block item => item.Holds assignment
  | .composite decoder => decoder.Holds assignment

structure RawArtifact where
  schemaVersion : Nat
  profileId : String
  sourceArtifactIdentity : String
  finalArtifactIdentity : String
  lifecycleScope : String
  rowFamily : String
  rowStart : Nat
  rowStop : Nat
  columnCount : Nat
  finalAssignmentColumns : Range
  decodedColumns : Range
  decoderGroups : List DecoderGroup
deriving DecidableEq, Repr

def RawArtifact.rows (artifact : RawArtifact) : List Row :=
  artifact.decoderGroups.flatMap DecoderGroup.rows

def RawArtifact.expectedRows (artifact : RawArtifact) : Nat :=
  (artifact.decoderGroups.map DecoderGroup.count).sum

def RawArtifact.Satisfied
    (artifact : RawArtifact) (assignment : Nat → Nat) : Prop :=
  Satisfies artifact.rows assignment

structure RawArtifact.Valid (artifact : RawArtifact) : Prop where
  schemaVersion : artifact.schemaVersion = 1
  profileId : artifact.profileId =
    "nightstream/goldilocks/streaming-terminal-slice/v1"
  sourceArtifactIdentity : artifact.sourceArtifactIdentity =
    "rust:nightstream/streaming-terminal-lifecycle/source-rows/v1"
  finalArtifactIdentity : artifact.finalArtifactIdentity =
    "rust:nightstream/streaming-selective-ccs/final-rows/v1"
  lifecycleScope : artifact.lifecycleScope = "recursive-terminal-arm-435"
  rowFamily : artifact.rowFamily = "terminal.streaming.source_binding"
  rowCount : artifact.rowStop - artifact.rowStart = artifact.expectedRows
  decodedCount : artifact.decodedColumns.length = artifact.expectedRows
  finalAssignmentWithin : artifact.finalAssignmentColumns.ValidWithin artifact.columnCount
  decodedWithin : artifact.decodedColumns.ValidWithin artifact.columnCount
  groupsValid : ∀ group ∈ artifact.decoderGroups,
    group.Valid artifact.columnCount artifact.finalAssignmentColumns
      artifact.decodedColumns
  groupsCanonical : ∀ group ∈ artifact.decoderGroups, group.Canonical

theorem DecoderBlock.rows_length (block : DecoderBlock) :
    block.rows.length = block.count := by
  simp [DecoderBlock.rows]

theorem DecoderGroup.rows_length (group : DecoderGroup) :
    group.rows.length = group.count := by
  cases group with
  | block item => simpa [DecoderGroup.rows, DecoderGroup.count] using item.rows_length
  | composite decoder => simp [DecoderGroup.rows, DecoderGroup.count]

theorem RawArtifact.rows_length (artifact : RawArtifact) :
    artifact.rows.length = artifact.expectedRows := by
  unfold RawArtifact.rows RawArtifact.expectedRows
  induction artifact.decoderGroups with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.flatMap_cons, List.length_append, DecoderGroup.rows_length,
        List.map_cons, List.sum_cons]
      omega

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalSourceBinding.Artifact
