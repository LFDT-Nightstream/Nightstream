import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.PiRlcFamilySelectiveCcs
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyAlgebraRetained
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.SourceImage

/-!
Contract: structural validation of the compact production PiRLC algebra
retained-row scan receipt.

Assurance tier: Rust-conformant for property
`FPRIME-PIRLC-FAMILY-BODY-ALGEBRA-RETAINED-PORT-IMAGE` under the supported
Goldilocks `b = 2`, `k_rho = 16` profile.

Owns cross-artifact agreement with the 49,626-row PiRLC source recipe, the
first retained interval in both parity arms, and the source decoder's exact
radix-three slot map. It recomputes the source and final nonzero censuses from
the algebra dimensions.

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
  let width := 41
  let arms := 2
  [0,
    arms * sourceRecipe.rows,
    arms * (sourceRecipe.productRows * (1 + width) + sourceRecipe.laneCount),
    arms * (sourceRecipe.productRows * width +
      sourceRecipe.sourceCount * reducedProductNnz * width),
    arms * ((sourceRecipe.productRows + sourceRecipe.laneCount) * width),
    0, 0, 0, 0, 0, 0, 0, 0]

def decoderPrefixExpected : List RawResidualBatch :=
  [ { sourceStart := 1, instanceCount := 1, instanceStride := 0, width := 640,
      resolution := .direct 1 1 1 false }
  , { sourceStart := 641, instanceCount := 1, instanceStride := 0,
      width := 51462, resolution := .direct 702 41 41 false }
  ]

def retainedIntervalsExpected : List RawRetainedRun :=
  [ { arm := 0, sourceStart := 0, length := 49626,
      emittedStart := 19830 }
  , { arm := 1, sourceStart := 0, length := 49626,
      emittedStart := 255341 }
  ]

def algebraRetainedIntervals :=
  rowLedger.retainedRuns.filter fun run =>
    run.sourceStart == 0 && run.length == audit.sourceRows

private def exactShape : Prop :=
  audit.schemaVersion =
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyAlgebraRetainedSchema.supportedSchemaVersion ∧
    audit.sourceRows = 49626 ∧ audit.localColumns = 51463 ∧
    audit.sourceColumnShift = 640 ∧ audit.finalRows = 491046 ∧
    audit.finalColumns = 8858862 ∧ audit.selectorColumns = [648, 649] ∧
    audit.emittedStarts = [19830, 255341] ∧
    audit.sourceStarts = [641, 1559, 2477, 2531] ∧
    audit.finalStarts = [702, 38340, 75978, 78192] ∧
    audit.widths = [41, 41, 41, 41] ∧
    audit.radices = audit.widths.map (fun width => (slotRadix width).val) ∧
    audit.sourceNnz = sourceNnzExpected ∧
    audit.finalPortNnz = finalPortNnzExpected

private def sourceRecipeCoherent : Prop :=
  sourceRecipe.sourceCount = 17 ∧ sourceRecipe.laneCount = 54 ∧
    sourceRecipe.rows = audit.sourceRows ∧
    sourceRecipe.columns = audit.localColumns ∧
    sourceRecipe.challengeStart + audit.sourceColumnShift = 641 ∧
    sourceRecipe.inputStart + audit.sourceColumnShift = 1559 ∧
    sourceRecipe.outputStart + audit.sourceColumnShift = 2477 ∧
    sourceRecipe.productStart + audit.sourceColumnShift = 2531 ∧
    sourceRecipe.productStart + sourceRecipe.productRows =
      audit.localColumns

private def decoderCoherent : Prop :=
  evenDecoder.residualBatches.take 2 = decoderPrefixExpected ∧
    oddDecoder.residualBatches.take 2 = decoderPrefixExpected ∧
    audit.finalStarts = audit.sourceStarts.map (fun start =>
      702 + (start - 641) * 41) ∧
    702 + (audit.localColumns - audit.sourceColumnShift - 1) * 41 + 41 ≤
      audit.finalColumns

private def rowLedgerCoherent : Prop :=
  algebraRetainedIntervals = retainedIntervalsExpected ∧
    audit.emittedStarts.map (fun start => start + audit.sourceRows) =
      [69456, 304967] ∧
    (audit.emittedStarts.all fun start =>
      decide (start + audit.sourceRows ≤ audit.finalRows)) = true

/-- The independent leaf obligations for the algebra scan receipt. -/
structure AuditValid : Prop where
  shape : exactShape
  sourceRecipe : sourceRecipeCoherent
  decoder : decoderCoherent
  rowLedger : rowLedgerCoherent

/-- The Phi81 reduction recipe has 3,996 product coefficients per source. -/
theorem reduced_product_nnz_exact : reducedProductNnz = 3996 := by
  decide

/-- The receipt's source and final nonzero counts are recomputed from the
17-source, 54-lane algebra recipe and the 41-coordinate slot width. -/
theorem nonzero_census_exact :
    audit.sourceNnz = sourceNnzExpected ∧
      audit.finalPortNnz = finalPortNnzExpected := by
  unfold sourceNnzExpected finalPortNnzExpected
  rw [reduced_product_nnz_exact]
  decide

private theorem shape_exact : exactShape := by
  unfold exactShape
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl,
    by decide, nonzero_census_exact.1, nonzero_census_exact.2⟩

private theorem source_recipe_exact : sourceRecipeCoherent := by
  unfold sourceRecipeCoherent
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, by decide⟩

/-- Both parity decoders use the same direct radix-three image for the
complete algebra block. -/
theorem decoder_prefix_exact : decoderCoherent := by
  unfold decoderCoherent decoderPrefixExpected
  exact ⟨rfl, rfl, rfl, by decide⟩

/-- The row ledger maps source rows 0 through 49,625 to the two exact emitted
intervals scanned by Rust. -/
theorem retained_intervals_exact : rowLedgerCoherent := by
  unfold rowLedgerCoherent algebraRetainedIntervals retainedIntervalsExpected
  exact ⟨rfl, rfl, rfl⟩

/-- The generated receipt agrees with the independent source recipe, decoder,
row ledger, and recomputed nonzero census. -/
theorem audit_valid : AuditValid := {
  shape := shape_exact
  sourceRecipe := source_recipe_exact
  decoder := decoder_prefix_exact
  rowLedger := retained_intervals_exact
}

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained
