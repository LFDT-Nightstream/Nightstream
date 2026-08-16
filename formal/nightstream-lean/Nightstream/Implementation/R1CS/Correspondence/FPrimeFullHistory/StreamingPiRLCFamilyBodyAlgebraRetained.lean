import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.PiRlcFamilySelectiveCcs
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyAlgebraRetained
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.SourceImage

/-!
Contract: independent validation of the compact production PiRLC algebra
retained-row scan receipt.

Assurance tier: Rust-conformant for property
`FPRIME-PIRLC-FAMILY-BODY-ALGEBRA-RETAINED-PORT-IMAGE`.

Owns cross-artifact agreement with the 43,794-row PiRLC source recipe, the
first retained interval in both parity arms, and the source decoder's exact
radix-three and radix-seven slot map. It independently recomputes every
source and final nonzero census from the algebra dimensions.

Does not own matrix authority in Lean, assignment values, row satisfaction,
selector authority, the remaining normalized rows, recursive orchestration,
or lifecycle soundness. The Rust drift test exhaustively compares the actual
source and final matrices with the recipe represented by this receipt.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyAlgebraRetainedSchema
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoderSchema
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedgerSchema
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage

abbrev audit :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyAlgebraRetained.audit

abbrev sourceRecipe :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.PiRlcFamilySelectiveCcs.rawArtifact

abbrev rowLedger :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger.ledger

abbrev evenDecoder :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder.evenArm

abbrev oddDecoder :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder.oddArm

def reductionWidth (degree : Nat) : Nat :=
  if degree < 54 then 1 else if degree < 81 then 2 else 1

def reducedProductNnz : Nat :=
  ((List.range 54).map fun left =>
    ((List.range 54).map fun right => reductionWidth (left + right)).sum).sum

def sourceNnzExpected : List Nat :=
  [sourceRecipe.productRows * 2 + sourceRecipe.laneCount,
    sourceRecipe.productRows + sourceRecipe.sourceCount * reducedProductNnz,
    sourceRecipe.productRows + sourceRecipe.laneCount]

def finalPortNnzExpected : List Nat :=
  let generalWidth := 23
  let inputWidth := 41
  let arms := 2
  [0,
    arms * sourceRecipe.rows,
    arms * (sourceRecipe.productRows * (1 + generalWidth) +
      sourceRecipe.laneCount),
    arms * (sourceRecipe.productRows * inputWidth +
      sourceRecipe.sourceCount * reducedProductNnz * generalWidth),
    arms * ((sourceRecipe.productRows + sourceRecipe.laneCount) *
      generalWidth),
    0, 0, 0, 0, 0, 0, 0, 0]

def decoderPrefixExpected : List RawStridedRun :=
  [ { sourceStart := 1, count := 640, sourceStride := 1,
      resolution := .direct 1 1 1 false }
  , { sourceStart := 641, count := 810, sourceStride := 1,
      resolution := .direct 702 23 23 false }
  , { sourceStart := 1451, count := 810, sourceStride := 1,
      resolution := .direct 19332 41 41 false }
  , { sourceStart := 2261, count := 43794, sourceStride := 1,
      resolution := .direct 52542 23 23 false }
  ]

def retainedIntervalsExpected : List RawRetainedRun :=
  [ { arm := 0, sourceStart := 0, length := 43794,
      emittedStart := 34168 }
  , { arm := 1, sourceStart := 0, length := 43794,
      emittedStart := 156526 }
  ]

def algebraRetainedIntervals :=
  rowLedger.retainedRuns.filter fun run =>
    run.sourceStart == 0 && run.length == audit.sourceRows

def exactShape : Prop :=
  audit.schemaVersion =
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyAlgebraRetainedSchema.supportedSchemaVersion /\
    audit.sourceRows = 43794 /\
    audit.localColumns = 45415 /\
    audit.sourceColumnShift = 640 /\
    audit.finalRows = 279089 /\
    audit.finalColumns = 2484972 /\
    audit.selectorColumns = [648, 649] /\
    audit.emittedStarts = [34168, 156526] /\
    audit.sourceStarts = [641, 1451, 2261, 2315] /\
    audit.finalStarts = [702, 19332, 52542, 53784] /\
    audit.widths = [23, 41, 23, 23] /\
    audit.radices = audit.widths.map (fun width => (slotRadix width).val) /\
    audit.sourceNnz = sourceNnzExpected /\
    audit.finalPortNnz = finalPortNnzExpected

def sourceRecipeCoherent : Prop :=
  sourceRecipe.rows = audit.sourceRows /\
    sourceRecipe.columns = audit.localColumns /\
    sourceRecipe.challengeStart + audit.sourceColumnShift = 641 /\
    sourceRecipe.inputStart + audit.sourceColumnShift = 1451 /\
    sourceRecipe.outputStart + audit.sourceColumnShift = 2261 /\
    sourceRecipe.productStart + audit.sourceColumnShift = 2315 /\
    sourceRecipe.productStart + sourceRecipe.productRows =
      audit.localColumns

def decoderCoherent : Prop :=
  evenDecoder.residualRuns.take 4 = decoderPrefixExpected /\
    oddDecoder.residualRuns.take 4 = decoderPrefixExpected /\
    52542 + sourceRecipe.laneCount * 23 = 53784 /\
    53784 + sourceRecipe.productRows * 23 <= audit.finalColumns

def rowLedgerCoherent : Prop :=
  algebraRetainedIntervals = retainedIntervalsExpected /\
    audit.emittedStarts.map (fun start => start + audit.sourceRows) =
      [77962, 200320] /\
    (audit.emittedStarts.all fun start =>
      decide (start + audit.sourceRows <= audit.finalRows)) = true

def AuditValid : Prop :=
  exactShape /\ sourceRecipeCoherent /\ decoderCoherent /\ rowLedgerCoherent

/-- The Phi81 reduction recipe has 3,996 product coefficients per source. -/
theorem reduced_product_nnz_exact : reducedProductNnz = 3996 := by
  native_decide

/-- The receipt's source and final nonzero counts are recomputed from the
15-source, 54-lane algebra recipe and the 23/41-coordinate slot widths. -/
theorem nonzero_census_exact :
    audit.sourceNnz = sourceNnzExpected /\
      audit.finalPortNnz = finalPortNnzExpected := by
  native_decide

/-- Both parity decoders use the same radix-seven challenge/output/product
image and the same radix-three input image for the complete algebra block. -/
theorem decoder_prefix_exact : decoderCoherent := by
  unfold decoderCoherent
  native_decide

/-- The row ledger maps source rows 0 through 43,793 to the two exact emitted
intervals scanned by Rust. -/
theorem retained_intervals_exact : rowLedgerCoherent := by
  unfold rowLedgerCoherent
  native_decide

/-- The generated receipt agrees with the independent source recipe, decoder,
row ledger, and recomputed nonzero census. -/
theorem audit_valid : AuditValid := by
  unfold AuditValid exactShape sourceRecipeCoherent decoderCoherent
    rowLedgerCoherent
  native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained
