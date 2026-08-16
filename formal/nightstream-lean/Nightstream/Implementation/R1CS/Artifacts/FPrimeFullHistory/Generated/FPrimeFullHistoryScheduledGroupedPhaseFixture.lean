import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.ScheduledGroupedPhaseFixtureSchema

/-! Generated file: exact schedule-over-grouped-phase composition fixture.

Owns the Rust-emitted row ranges, selector columns, schedule maps, cursor-bit
ranges, and selective port indices used by the exhaustive matrix test.

Does not own component semantics, the production 400-arm schedule, or the
complete recursive and terminal F-prime relations. Lean recomputes each row.

Emits constraints: this fixture's schedule total, group equality, activation,
and exact cursor rows.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryScheduledGroupedPhaseFixture

open Nightstream.Implementation.R1CS.FPrimeFullHistoryScheduledGroupedPhaseFixture.Artifact

def rawArtifact : RawArtifact where
  schemaVersion := 1
  rows := 406
  columns := 324
  publicColumns := 54
  commonRowEnd := 169
  phaseRowEnd := 338
  scheduleTotalRowEnd := 339
  lifecycleEqualityRowEnd := 341
  phaseKindEqualityRowEnd := 343
  lifecycleActivationRowEnd := 346
  phaseKindActivationRowEnd := 349
  cursorBindingRowEnd := 355
  portCount := 13
  generalSelectorPort := 1
  aPort := 2
  bPort := 3
  cPort := 4
  commonSelectorColumns := [54, 55]
  phaseKindSelectorColumns := [162, 163]
  scheduleSelectorColumns := [270, 271, 272]
  lifecycleGroups := [0, 1, 1]
  phaseKinds := [0, 1, 0]
  beforeCursorStart := 1
  beforeCursorEnd := 3
  afterCursorStart := 3
  afterCursorEnd := 5

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryScheduledGroupedPhaseFixture
