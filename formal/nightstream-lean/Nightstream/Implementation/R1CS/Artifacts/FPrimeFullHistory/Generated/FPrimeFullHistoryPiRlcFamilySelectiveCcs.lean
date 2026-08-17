import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.PiRlcFamilySelectiveCcsSchema

/-! Generated file: compact exact row recipe for one production PiRLC family.

Owns: the Rust-declared dimensions, canonical column starts, selective port
indices, and Goldilocks coefficients used by the exhaustive row audit.

Does not own: semantic truth, PiCCS input authority, Poseidon2 binding, or the
complete recursive and terminal F-prime relations. Lean recomputes every row.

Emits constraints: no. This file contains recipe data, not a trusted digest.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiRlcFamilySelectiveCcs

open Nightstream.Implementation.R1CS.FPrimeFullHistoryPiRlcFamilySelectiveCcs.Artifact

def rawArtifact : RawArtifact where
  schemaVersion := 1
  sourceCount := 17
  laneCount := 54
  challengeStart := 1
  inputStart := 919
  outputStart := 1837
  productStart := 1891
  productRows := 49572
  rows := 49626
  columns := 51463
  rowVariables := 16
  portCount := 13
  generalSelectorPort := 1
  aPort := 2
  bPort := 3
  cPort := 4
  minusOne := 18446744069414584320
  minusTwo := 18446744069414584319

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiRlcFamilySelectiveCcs
