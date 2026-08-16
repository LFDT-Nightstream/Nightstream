import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder

/-!
Contract: independent structural validation of the compact production PiRLC
family-body source decoder.

Assurance tier: artifact-checked for property
`FPRIME-PIRLC-FAMILY-BODY-DECODER-COVER`.

Owns exact source bounds, final-slot bounds, template expansion, complete
source-column cover, and absence of duplicate ownership for both parity arms.

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

structure Ownership where
  slots : Array Bool
  marked : Nat

private def resolutionValidAt
    (arm : RawArm)
    (sourceColumn offset directBase referenceBase referenceFinalBase : Nat) :
    RawResolutionRun -> Bool
  | .direct start startStride width _ =>
      let finalStart := directBase + start + startStride * offset
      decide (0 < width /\ finalStart + width <= arm.finalColumns)
  | .decompositionAlias source sourceStride _digit _digitStride start startStride _ =>
      let aliasSource := referenceBase + source + sourceStride * offset
      let aliasStart := referenceFinalBase + start + startStride * offset
      decide (aliasSource < sourceColumn /\ aliasSource < arm.sourceEnd /\
        aliasStart < arm.finalColumns)
  | .equalityAlias source sourceStride start startStride width _ =>
      let aliasSource := referenceBase + source + sourceStride * offset
      let aliasStart := referenceFinalBase + start + startStride * offset
      decide (0 < width /\ aliasSource < sourceColumn /\
        aliasSource < arm.sourceEnd /\ aliasStart + width <= arm.finalColumns)
  | .linearDefinition => true
  | .traceEliminated => true

private def markColumn
    (arm : RawArm)
    (sourceColumn : Nat)
    (state : Ownership) : Option Ownership :=
  if sourceColumn < arm.sourceStart || arm.sourceEnd <= sourceColumn then
    none
  else
    let index := sourceColumn - arm.sourceStart
    match state.slots[index]? with
    | some false =>
        some { slots := state.slots.set! index true, marked := state.marked + 1 }
    | _ => none

private def relativeRunsValidFrom (sourceWidth cursor : Nat) : List RawRun -> Bool
  | [] => decide (cursor = sourceWidth)
  | run :: runs =>
      if run.sourceStart = cursor && 0 < run.length &&
          run.sourceStart + run.length <= sourceWidth then
        relativeRunsValidFrom sourceWidth (run.sourceStart + run.length) runs
      else
        false

private def relativeRunsValid (template : RawTemplate) : Bool :=
  0 < template.sourceWidth &&
    !template.relativeRuns.isEmpty &&
    relativeRunsValidFrom template.sourceWidth 0 template.relativeRuns

private def checkTemplateRunColumns
    (arm : RawArm)
    (sourceBase directBase referenceBase referenceFinalBase : Nat)
    (run : RawRun) : Nat -> Nat -> Ownership -> Option Ownership
  | _, 0, state => some state
  | offset, remaining + 1, state => do
      let sourceColumn := sourceBase + run.sourceStart + offset
      if !resolutionValidAt arm sourceColumn offset directBase referenceBase
          referenceFinalBase run.resolution then
        none
      else
        let state <- markColumn arm sourceColumn state
        checkTemplateRunColumns arm sourceBase directBase referenceBase
          referenceFinalBase run (offset + 1) remaining state

private def checkTemplateRuns
    (arm : RawArm)
    (sourceBase directBase referenceBase referenceFinalBase : Nat) :
    List RawRun -> Ownership -> Option Ownership
  | [], state => some state
  | run :: runs, state => do
      let state <- checkTemplateRunColumns arm sourceBase directBase
        referenceBase referenceFinalBase run 0 run.length state
      checkTemplateRuns arm sourceBase directBase referenceBase
        referenceFinalBase runs state

private def checkTemplateBatchInstances
    (arm : RawArm)
    (template : RawTemplate)
    (batch : RawTemplateInstances) : Nat -> Nat -> Ownership -> Option Ownership
  | _, 0, state => some state
  | index, remaining + 1, state => do
      let sourceBase := batch.sourceStart + batch.sourceStride * index
      let directBase := batch.finalStart + batch.finalStride * index
      let referenceBase := batch.referenceStart + batch.referenceStride * index
      let referenceFinalBase :=
        batch.referenceFinalStart + batch.referenceFinalStride * index
      let state <- checkTemplateRuns arm sourceBase directBase referenceBase
        referenceFinalBase template.relativeRuns state
      checkTemplateBatchInstances arm template batch (index + 1) remaining state

private def checkTemplateBatches
    (arm : RawArm)
    (template : RawTemplate) :
    List RawTemplateInstances -> Ownership -> Option Ownership
  | [], state => some state
  | batch :: batches, state =>
      if batch.count = 0 || batch.sourceStride = 0 then
        none
      else do
        let state <- checkTemplateBatchInstances arm template batch 0 batch.count state
        checkTemplateBatches arm template batches state

private def checkTemplates
    (arm : RawArm) : List RawTemplate -> Ownership -> Option Ownership
  | [], state => some state
  | template :: templates, state =>
      if !relativeRunsValid template || template.instances.isEmpty then
        none
      else do
        let state <- checkTemplateBatches arm template template.instances state
        checkTemplates arm templates state

private def checkResidualRunColumns
    (arm : RawArm)
    (run : RawStridedRun) : Nat -> Nat -> Ownership -> Option Ownership
  | _, 0, state => some state
  | offset, remaining + 1, state => do
      let sourceColumn := run.sourceStart + run.sourceStride * offset
      if !resolutionValidAt arm sourceColumn offset 0 0 0 run.resolution then
        none
      else
        let state <- markColumn arm sourceColumn state
        checkResidualRunColumns arm run (offset + 1) remaining state

private def checkResidualRuns
    (arm : RawArm) : List RawStridedRun -> Ownership -> Option Ownership
  | [], state => some state
  | run :: runs, state =>
      if run.count = 0 || run.sourceStride = 0 then
        none
      else do
        let state <- checkResidualRunColumns arm run 0 run.count state
        checkResidualRuns arm runs state

def templateColumnCount (arm : RawArm) : Nat :=
  (arm.templates.map fun template =>
    template.sourceWidth * (template.instances.map (fun batch => batch.count)).sum).sum

def residualColumnCount (arm : RawArm) : Nat :=
  (arm.residualRuns.map fun run => run.count).sum

private def maximumList (values : List Nat) : Nat :=
  values.foldl Nat.max 0

def maximumCheckRun (arm : RawArm) : Nat :=
  Nat.max
    (maximumList (arm.templates.flatMap fun template =>
      template.relativeRuns.map fun run => run.length))
    (maximumList (arm.residualRuns.map fun run => run.count))

def exactShape
    (expectedArm expectedSourceEnd : Nat)
    (arm : RawArm) : Bool :=
  decide (arm.schemaVersion = supportedSchemaVersion /\
    arm.arm = expectedArm /\
    arm.sourceStart = 1 /\
    arm.sourceEnd = expectedSourceEnd /\
    arm.finalColumns = 2484972 /\
    arm.templates.length = 3 /\
    arm.residualRuns.length = 16)

def validateArm
    (expectedArm expectedSourceEnd : Nat)
    (arm : RawArm) : Bool :=
  if !exactShape expectedArm expectedSourceEnd arm then
    false
  else
    let sourceLength := arm.sourceEnd - arm.sourceStart
    let initial : Ownership :=
      { slots := Array.replicate sourceLength false, marked := 0 }
    match checkTemplates arm arm.templates initial with
    | none => false
    | some afterTemplates =>
        match checkResidualRuns arm arm.residualRuns afterTemplates with
        | none => false
        | some complete => decide (complete.marked = sourceLength)

def EvenValid : Prop := validateArm 0 559136 evenArm = true

def OddValid : Prop := validateArm 1 560336 oddArm = true

/-- The even check owns exactly 559,135 requested source columns. -/
theorem even_source_length_exact :
    evenArm.sourceEnd - evenArm.sourceStart = 559135 := by
  decide

/-- The odd check owns exactly 560,335 requested source columns. -/
theorem odd_source_length_exact :
    oddArm.sourceEnd - oddArm.sourceStart = 560335 := by
  decide

/-- The largest tail-recursive column run in either check has 43,794 items. -/
theorem maximum_check_run_exact :
    maximumCheckRun evenArm = 43794 /\ maximumCheckRun oddArm = 43794 := by
  native_decide

/-- The compact template and residual counts equal the complete even source range. -/
theorem even_column_census_exact :
    templateColumnCount evenArm = 511020 /\
      residualColumnCount evenArm = 48115 /\
      templateColumnCount evenArm + residualColumnCount evenArm = 559135 := by
  native_decide

/-- The compact template and residual counts equal the complete odd source range. -/
theorem odd_column_census_exact :
    templateColumnCount oddArm = 512220 /\
      residualColumnCount oddArm = 48115 /\
      templateColumnCount oddArm + residualColumnCount oddArm = 560335 := by
  native_decide

/-- The generated even decoder has exact cover, no duplicate source owner,
and only in-bounds source references and final slot spans. -/
theorem even_valid : EvenValid := by
  unfold EvenValid
  native_decide

/-- The generated odd decoder has exact cover, no duplicate source owner,
and only in-bounds source references and final slot spans. -/
theorem odd_valid : OddValid := by
  unfold OddValid
  native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyDecoder
