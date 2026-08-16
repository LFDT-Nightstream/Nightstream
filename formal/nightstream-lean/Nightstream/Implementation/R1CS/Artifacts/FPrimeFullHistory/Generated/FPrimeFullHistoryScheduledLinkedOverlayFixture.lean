import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.ScheduledLinkedOverlayFixtureSchema

/-! Generated file: exact schedule-linked private-overlay fixture.

Owns the Rust-emitted row ranges, selector columns, schedule maps, linked
field digit ranges, and selective port indices used by the exhaustive matrix
test.

Does not own component semantics, production dimensions, or the complete
recursive and terminal F-prime relations. Lean recomputes every link row.

Emits constraints: overlay selector equality, activation, exact decoded-field
equality, and ring-padding rows for this fixture.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryScheduledLinkedOverlayFixture

open Nightstream.Implementation.R1CS.FPrimeFullHistoryScheduledLinkedOverlayFixture.Artifact

def rawArtifact : RawArtifact where
  schemaVersion := 1
  rows := 384
  columns := 540
  publicColumns := 54
  scheduledRowEnd := 348
  overlayRowEnd := 376
  overlayKindEqualityRowEnd := 378
  overlayActivationRowEnd := 381
  fieldLinkRowEnd := 383
  ringPaddingRowEnd := 384
  ringPaddingColumnStart := 539
  portCount := 13
  generalSelectorPort := 1
  aPort := 2
  bPort := 3
  cPort := 4
  scheduleSelectorColumns := [378, 379, 380]
  overlaySelectorColumns := [432, 433]
  lifecycleGroups := [0, 1, 1]
  phaseKinds := [0, 1, 0]
  overlayKinds := [0, 1, 0]
  phaseFieldStarts := [270, 270]
  overlayFieldStarts := [434, 434]
  fieldWidths := [41, 41]
  fieldRadices := [3, 3]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryScheduledLinkedOverlayFixture
