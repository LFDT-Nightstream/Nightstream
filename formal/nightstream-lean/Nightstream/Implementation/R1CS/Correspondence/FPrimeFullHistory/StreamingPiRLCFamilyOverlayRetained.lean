import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputBindingProductionSetup
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyOverlayRetained
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.SourceImage

/-!
Contract: independent validation of the compact production PiRLC
family-overlay retained-row receipt.

Assurance tier: Rust-conformant for property
`FPRIME-PIRLC-FAMILY-OVERLAY-RETAINED-PORT-IMAGE`.

Owns the affine selector and retained-row geometry for all 110 family arms,
the direct centered-digit and radix-seven output slots, the exact six seed
chunks, and independent compact-block and explicit nonzero censuses.

Does not own matrix authority in Lean, assignment values, body-to-overlay
links, selector authority, recursive orchestration, lifecycle soundness, or
Module-SIS hardness. The Rust drift test checks every source block, every
final block, and every retained explicit final row.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingProductionSetup
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyOverlayRetainedSchema
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage

abbrev audit :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyOverlayRetained.audit

def expectedSchedule : SeededPhi81.SeedSchedule :=
  SeededAjtai.schedule productionSetup.setup.seed.bytes verifierRows
    messageColumnCount productionSetup.setup.rejectionFuel

def sourceExplicitNnzExpected : List Nat :=
  let rows := 110 * 108
  [0, rows, rows]

def finalBlockCountsExpected : List Nat :=
  [0, 0, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def finalExplicitPortNnzExpected : List Nat :=
  let rows := 110 * 108
  [0, rows, 0, rows, rows * 23, 0, 0, 0, 0, 0, 0, 0, 0]

def exactShape : Prop :=
  audit.schemaVersion =
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyOverlayRetainedSchema.supportedSchemaVersion /\
    audit.familyCount = 110 /\
    audit.sourceRows = 108 /\
    audit.sourceColumns = 33360 /\
    audit.finalRows = 28627 /\
    audit.finalColumns = 35856 /\
    audit.selectorStart = 1 /\
    audit.selectorCount = audit.familyCount /\
    audit.retainedStart = 16737 /\
    audit.retainedStride = audit.sourceRows /\
    audit.sourceStarts = [1, 42, 33252] /\
    audit.finalStarts = [111, 152, 33362] /\
    audit.widths = [1, 23] /\
    audit.radices = [2, 7]

def affineGeometryCoherent : Prop :=
  audit.selectorStart + audit.selectorCount = 111 /\
    audit.retainedStart + audit.familyCount * audit.retainedStride = 28617 /\
    audit.retainedStart + audit.familyCount * audit.retainedStride <=
      audit.finalRows /\
    audit.finalStarts.getD 0 0 =
      audit.sourceStarts.getD 0 0 + audit.selectorCount /\
    audit.finalStarts.getD 1 0 =
      audit.sourceStarts.getD 1 0 + audit.selectorCount /\
    audit.finalStarts.getD 2 0 + 108 * 23 = 35846 /\
    audit.finalStarts.getD 2 0 + 108 * 23 <= audit.finalColumns

def seedScheduleCoherent : Prop :=
  audit.chunkSize = expectedSchedule.chunkSize /\
    audit.chunkSeedsByRow = expectedSchedule.seedsByOutput /\
    expectedSchedule.rejectionFuel = 16 /\
    audit.chunkSeedsByRow.length = 2 /\
    (audit.chunkSeedsByRow.all fun row => decide (row.length = 3)) = true /\
    (audit.chunkSeedsByRow.all fun row =>
      row.all fun seed =>
        decide (seed.length = 32) &&
          seed.all fun byte => decide (byte < 256)) = true

def censusCoherent : Prop :=
  audit.sourceExplicitNnz = sourceExplicitNnzExpected /\
    audit.finalBlockCounts = finalBlockCountsExpected /\
    audit.finalExplicitPortNnz = finalExplicitPortNnzExpected

def AuditValid : Prop :=
  exactShape /\ affineGeometryCoherent /\ seedScheduleCoherent /\
    censusCoherent

/-- The generated chunks are exactly the verifier-owned production schedule,
not a digest or a prover-supplied commitment identity. -/
theorem seed_schedule_exact : seedScheduleCoherent := by
  unfold seedScheduleCoherent expectedSchedule
  native_decide

/-- The 110 selectors and 110 retained 108-row blocks occupy the exact
affine intervals scanned by Rust. -/
theorem affine_geometry_exact : affineGeometryCoherent := by
  unfold affineGeometryCoherent
  native_decide

/-- The explicit and compact nonzero counts follow from one selector, one
constant-one term, one 23-coordinate output, and one compact seeded block per
retained row family. -/
theorem nonzero_census_exact : censusCoherent := by
  unfold censusCoherent sourceExplicitNnzExpected finalBlockCountsExpected
    finalExplicitPortNnzExpected
  native_decide

/-- The generated receipt agrees with the independent production seed
schedule, exact affine geometry, and recomputed compact/explicit censuses. -/
theorem audit_valid : AuditValid := by
  unfold AuditValid exactShape affineGeometryCoherent seedScheduleCoherent
    censusCoherent expectedSchedule sourceExplicitNnzExpected
    finalBlockCountsExpected finalExplicitPortNnzExpected
  native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained
