import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.PiCcsRoundSelectiveCcsSchema

/-! Generated file: exact compact recipe for one production PiCCS round.

Owns the Rust-declared degree-nine dimensions, canonical column starts,
selective port indices, and Goldilocks coefficients checked against the real
compact phase emitter.

Does not own semantic truth, Poseidon2 replay, recursive orchestration, or the
complete recursive and terminal F-prime relations. Lean recomputes every row.

Emits constraints: no. This file contains recipe data, not a trusted digest.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiCcsRoundSelectiveCcs

open Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsRoundSelectiveCcs.Artifact

def rawArtifact : RawArtifact where
  schemaVersion := 1
  degree := 9
  coefficientCount := 10
  currentStart := 1
  coefficientStart := 3
  challengeStart := 23
  nextStart := 25
  auxiliaryStart := 27
  rows := 31
  columns := 54
  rowVariables := 5
  portCount := 13
  generalSelectorPort := 1
  aPort := 2
  bPort := 3
  cPort := 4
  nonresidue := 7
  minusOne := 18446744069414584320

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiCcsRoundSelectiveCcs
