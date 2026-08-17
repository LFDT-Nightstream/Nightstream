import Mathlib.Data.List.Basic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder

/-!
Contract: structural validation of the compact production PiRLC family-body
source decoder.

Assurance tier: artifact-checked for property
`FPRIME-PIRLC-FAMILY-BODY-DECODER-COVER` under the supported Goldilocks
`b = 2`, `k_rho = 16` profile.

Owns exact source bounds, final-slot bounds, affine template and residual
geometry, complete source-column cover, and one reference to every compact
source owner for both parity arms.

Does not own source-row semantics, matrix soundness, assignment values,
linear-definition reconstruction, trace-elimination soundness, selector
authority, or lifecycle soundness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyDecoder

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoderSchema
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder

structure OwnerGeometry where
  sourceStart : Nat
  count : Nat
  stride : Nat
  width : Nat
deriving DecidableEq, Repr

private def resolutionValidAt
    (arm : RawArm)
    (sourceColumn offset directBase referenceBase referenceFinalBase : Nat) :
    RawResolutionRun → Bool
  | .direct start startStride width _ =>
      let finalStart := directBase + start + startStride * offset
      decide (0 < width ∧ finalStart + width ≤ arm.finalColumns)
  | .decompositionAlias source sourceStride _digit _digitStride start
      startStride _ =>
      let aliasSource := referenceBase + source + sourceStride * offset
      let aliasStart := referenceFinalBase + start + startStride * offset
      decide (aliasSource < sourceColumn ∧ aliasSource < arm.sourceEnd ∧
        aliasStart < arm.finalColumns)
  | .equalityAlias source sourceStride start startStride width _ =>
      let aliasSource := referenceBase + source + sourceStride * offset
      let aliasStart := referenceFinalBase + start + startStride * offset
      decide (0 < width ∧ aliasSource < sourceColumn ∧
        aliasSource < arm.sourceEnd ∧ aliasStart + width ≤ arm.finalColumns)
  | .linearDefinition => true
  | .traceEliminated => true

private def relativeRunsValidFrom (sourceWidth cursor : Nat) :
    List RawRun → Bool
  | [] => decide (cursor = sourceWidth)
  | run :: runs =>
      decide (run.sourceStart = cursor) && decide (0 < run.length) &&
        decide (run.sourceStart + run.length ≤ sourceWidth) &&
        relativeRunsValidFrom sourceWidth (run.sourceStart + run.length) runs

private def relativeRunsValid (template : RawTemplate) : Bool :=
  decide (0 < template.sourceWidth) && decide (!template.relativeRuns.isEmpty) &&
    relativeRunsValidFrom template.sourceWidth 0 template.relativeRuns

private def templateResolutionValidAt
    (arm : RawArm) (batch : RawTemplateInstances) (run : RawRun)
    (instanceIndex offset : Nat) : Bool :=
  let sourceBase := batch.sourceStart + batch.sourceStride * instanceIndex
  let directBase := batch.finalStart + batch.finalStride * instanceIndex
  let referenceBase :=
    batch.referenceStart + batch.referenceStride * instanceIndex
  let referenceFinalBase :=
    batch.referenceFinalStart + batch.referenceFinalStride * instanceIndex
  resolutionValidAt arm (sourceBase + run.sourceStart + offset) offset
    directBase referenceBase referenceFinalBase run.resolution

/-- Every decoder coordinate is affine in the instance and run offsets.
These four corners therefore own the extrema used by the bound checks. -/
private def templateRunCornersValid
    (arm : RawArm) (batch : RawTemplateInstances) (run : RawRun) : Bool :=
  let lastInstance := batch.count - 1
  let lastOffset := run.length - 1
  templateResolutionValidAt arm batch run 0 0 &&
    templateResolutionValidAt arm batch run 0 lastOffset &&
    templateResolutionValidAt arm batch run lastInstance 0 &&
    templateResolutionValidAt arm batch run lastInstance lastOffset

private def templateBatchValid
    (arm : RawArm) (template : RawTemplate)
    (batch : RawTemplateInstances) : Bool :=
  decide (0 < batch.count) &&
    decide (batch.count = 1 ∧ batch.sourceStride = 0 ∨
      1 < batch.count ∧ template.sourceWidth ≤ batch.sourceStride) &&
    template.relativeRuns.all (templateRunCornersValid arm batch)

private def templateValid (arm : RawArm) (template : RawTemplate) : Bool :=
  relativeRunsValid template && decide (!template.instances.isEmpty) &&
    template.instances.all (templateBatchValid arm template)

private def templatesValid (arm : RawArm) : Bool :=
  arm.templates.all (templateValid arm)

private def residualResolutionValidAt
    (arm : RawArm) (batch : RawResidualBatch)
    (instanceIndex offset : Nat) : Bool :=
  let sourceColumn :=
    batch.sourceStart + batch.instanceStride * instanceIndex + offset
  resolutionValidAt arm sourceColumn offset 0 0 0 batch.resolution

private def residualCornersValid
    (arm : RawArm) (batch : RawResidualBatch) : Bool :=
  let lastInstance := batch.instanceCount - 1
  let lastOffset := batch.width - 1
  residualResolutionValidAt arm batch 0 0 &&
    residualResolutionValidAt arm batch 0 lastOffset &&
    residualResolutionValidAt arm batch lastInstance 0 &&
    residualResolutionValidAt arm batch lastInstance lastOffset

private def residualBatchValid
    (arm : RawArm) (batch : RawResidualBatch) : Bool :=
  decide (0 < batch.instanceCount) && decide (0 < batch.width) &&
    decide (batch.instanceCount = 1 ∧ batch.instanceStride = 0 ∨
      1 < batch.instanceCount ∧ batch.width ≤ batch.instanceStride) &&
    residualCornersValid arm batch

private def residualBatchesValid (arm : RawArm) : Bool :=
  arm.residualBatches.all (residualBatchValid arm)

private def ownerGeometry (arm : RawArm) :
    RawOwnerRef → Option OwnerGeometry
  | .template templateIndex batchIndex =>
      match arm.templates[templateIndex]? with
      | none => none
      | some template =>
          match template.instances[batchIndex]? with
          | none => none
          | some batch => some {
              sourceStart := batch.sourceStart
              count := batch.count
              stride := batch.sourceStride
              width := template.sourceWidth
            }
  | .residual batchIndex =>
      match arm.residualBatches[batchIndex]? with
      | none => none
      | some batch => some {
          sourceStart := batch.sourceStart
          count := batch.instanceCount
          stride := batch.instanceStride
          width := batch.width
        }

private def ownerRepetitionValid
    (groupCount groupStride : Nat) (geometry : OwnerGeometry) : Bool :=
  if groupCount = 1 then
    decide (geometry.count = 1 ∧ geometry.stride = 0)
  else
    decide (1 < groupCount ∧ geometry.count = groupCount ∧
      geometry.stride = groupStride)

private def ownersTileFrom
    (arm : RawArm) (groupCount groupStride terminal : Nat) :
    Nat → List RawOwnerRef → Bool
  | cursor, [] => decide (cursor = terminal)
  | cursor, owner :: owners =>
      match ownerGeometry arm owner with
      | none => false
      | some geometry =>
          decide (geometry.sourceStart = cursor) && decide (0 < geometry.width) &&
            ownerRepetitionValid groupCount groupStride geometry &&
            ownersTileFrom arm groupCount groupStride terminal
              (cursor + geometry.width) owners

private def coverGroupValid (arm : RawArm) (group : RawCoverGroup) : Bool :=
  decide (0 < group.count) && decide (0 < group.stride) &&
    decide (!group.owners.isEmpty) &&
    ownersTileFrom arm group.count group.stride
      (group.sourceStart + group.stride) group.sourceStart group.owners

/-- A structural source cover. It advances by one affine group envelope and
never allocates a value for an expanded source column. -/
def coverGroupsFrom (arm : RawArm) : Nat → List RawCoverGroup → Bool
  | cursor, [] => decide (cursor = arm.sourceEnd)
  | cursor, group :: groups =>
      decide (group.sourceStart = cursor) && coverGroupValid arm group &&
        coverGroupsFrom arm (group.sourceStart + group.count * group.stride)
          groups

private def templateOwnerRefsFrom :
    Nat → List RawTemplate → List RawOwnerRef
  | _, [] => []
  | templateIndex, template :: templates =>
      (List.range template.instances.length |>.map fun batchIndex =>
        .template templateIndex batchIndex) ++
      templateOwnerRefsFrom (templateIndex + 1) templates

private def allOwnerRefs (arm : RawArm) : List RawOwnerRef :=
  templateOwnerRefsFrom 0 arm.templates ++
    (List.range arm.residualBatches.length |>.map RawOwnerRef.residual)

private def coverOwnerRefs (arm : RawArm) : List RawOwnerRef :=
  arm.coverGroups.flatMap (fun group => group.owners)

private def ownerRefsExact (arm : RawArm) : Bool :=
  let expected := allOwnerRefs arm
  let actual := coverOwnerRefs arm
  decide expected.Nodup && decide actual.Nodup &&
    decide (actual.length = expected.length) &&
    actual.all fun owner => decide (owner ∈ expected)

def templateColumnCount (arm : RawArm) : Nat :=
  (arm.templates.map fun template =>
    template.sourceWidth *
      (template.instances.map (fun batch => batch.count)).sum).sum

def residualColumnCount (arm : RawArm) : Nat :=
  (arm.residualBatches.map fun batch =>
    batch.instanceCount * batch.width).sum

private def exactShape
    (expectedArm expectedSourceEnd : Nat) (arm : RawArm) : Bool :=
  decide (arm.schemaVersion = supportedSchemaVersion ∧
    arm.arm = expectedArm ∧ arm.sourceStart = 1 ∧
    arm.sourceEnd = expectedSourceEnd ∧ arm.finalColumns = 8858862 ∧
    arm.templates.length = 3 ∧ arm.residualBatches.length = 67 ∧
    arm.coverGroups.length = 98)

private def columnCensusExact (arm : RawArm) : Bool :=
  decide (templateColumnCount arm + residualColumnCount arm =
    arm.sourceEnd - arm.sourceStart)

/-- Compact proof obligations for one parity decoder. No field contains an
expanded source-column set, final-column set, or witness assignment. -/
structure ArmValidFor
    (expectedArm expectedSourceEnd : Nat) (arm : RawArm) : Prop where
  shape : exactShape expectedArm expectedSourceEnd arm = true
  templates : templatesValid arm = true
  residualBatches : residualBatchesValid arm = true
  ownerRefs : ownerRefsExact arm = true
  sourceCover : coverGroupsFrom arm arm.sourceStart arm.coverGroups = true
  columnCensus : columnCensusExact arm = true

def EvenValid : Prop := ArmValidFor 0 1301126 evenArm

def OddValid : Prop := ArmValidFor 1 1302326 oddArm

/-- Exact source and final bounds, projected from the Rust-generated data. -/
theorem dimensions_exact :
    evenArm.sourceStart = 1 ∧ evenArm.sourceEnd = 1301126 ∧
      oddArm.sourceStart = 1 ∧ oddArm.sourceEnd = 1302326 ∧
      evenArm.finalColumns = 8858862 ∧ oddArm.finalColumns = 8858862 := by
  decide

/-- Exact compact certificate sizes. The proof checks 390 shared relative
runs per arm, 66 template batches, 134 residual batches, and 196 groups. -/
theorem certificate_input_lengths_exact :
    (evenArm.templates.map (fun template => template.relativeRuns.length)).sum = 390 ∧
      (oddArm.templates.map (fun template => template.relativeRuns.length)).sum = 390 ∧
      (evenArm.templates.map (fun template => template.instances.length)).sum = 33 ∧
      (oddArm.templates.map (fun template => template.instances.length)).sum = 33 ∧
      evenArm.residualBatches.length = 67 ∧ oddArm.residualBatches.length = 67 ∧
      evenArm.coverGroups.length = 98 ∧ oddArm.coverGroups.length = 98 := by
  decide

private theorem even_shape_exact : exactShape 0 1301126 evenArm = true := by
  decide

private theorem odd_shape_exact : exactShape 1 1302326 oddArm = true := by
  decide

private theorem even_templates_valid : templatesValid evenArm = true := by
  decide

private theorem odd_templates_valid : templatesValid oddArm = true := by
  decide

private theorem even_residual_batches_valid :
    residualBatchesValid evenArm = true := by
  decide

private theorem odd_residual_batches_valid :
    residualBatchesValid oddArm = true := by
  decide

private theorem even_owner_refs_exact : ownerRefsExact evenArm = true := by
  decide

private theorem odd_owner_refs_exact : ownerRefsExact oddArm = true := by
  decide

private theorem even_source_cover_exact :
    coverGroupsFrom evenArm evenArm.sourceStart evenArm.coverGroups = true := by
  decide

private theorem odd_source_cover_exact :
    coverGroupsFrom oddArm oddArm.sourceStart oddArm.coverGroups = true := by
  decide

/-- The compact template and residual counts equal the complete even source
range. -/
theorem even_column_census_exact :
    templateColumnCount evenArm = 1239996 ∧
      residualColumnCount evenArm = 61129 ∧
      templateColumnCount evenArm + residualColumnCount evenArm = 1301125 := by
  decide

/-- The compact template and residual counts equal the complete odd source
range. -/
theorem odd_column_census_exact :
    templateColumnCount oddArm = 1241196 ∧
      residualColumnCount oddArm = 61129 ∧
      templateColumnCount oddArm + residualColumnCount oddArm = 1302325 := by
  decide

/-- The generated even decoder satisfies every compact structural leaf. The
proof does not expand its 1,301,125 source columns. -/
theorem even_valid : EvenValid := by
  exact {
    shape := even_shape_exact
    templates := even_templates_valid
    residualBatches := even_residual_batches_valid
    ownerRefs := even_owner_refs_exact
    sourceCover := even_source_cover_exact
    columnCensus := by decide
  }

/-- The generated odd decoder satisfies every compact structural leaf. The
proof does not expand its 1,302,325 source columns. -/
theorem odd_valid : OddValid := by
  exact {
    shape := odd_shape_exact
    templates := odd_templates_valid
    residualBatches := odd_residual_batches_valid
    ownerRefs := odd_owner_refs_exact
    sourceCover := odd_source_cover_exact
    columnCensus := by decide
  }

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyDecoder
