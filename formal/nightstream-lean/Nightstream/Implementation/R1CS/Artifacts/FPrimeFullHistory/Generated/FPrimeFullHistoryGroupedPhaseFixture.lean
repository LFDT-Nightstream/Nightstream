import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.GroupedPhaseFixtureSchema

/-! Generated file: exact grouped-phase composition fixture.

Owns: Rust-emitted row ranges, shared width, selector columns, phase groups,
and selective port indices for the exhaustive grouped-phase matrix test.

Does not own: source-component rows, production phase counts, or Nebula F-prime
semantics. Lean recomputes every group-equality and activation row.

Emits constraints: no. This file contains checked recipe data.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryGroupedPhaseFixture

open Nightstream.Implementation.R1CS.FPrimeFullHistoryGroupedPhaseFixture.Artifact

def rawArtifact : RawArtifact where
  schemaVersion := 1
  rows := 340
  columns := 270
  publicColumns := 54
  commonRowEnd := 166
  phaseRowEnd := 335
  groupEqualityRowEnd := 337
  phaseActivationRowEnd := 340
  portCount := 13
  generalSelectorPort := 1
  aPort := 2
  bPort := 3
  cPort := 4
  commonSelectorColumns := [54, 55]
  phaseSelectorColumns := [162, 163, 164]
  phaseGroups := [0, 1, 1]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryGroupedPhaseFixture
