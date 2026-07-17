import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.DiagnosticProfile

/-!
External checks for the conditional fixed-profile Pi_RLC paper bridge.

Assurance tier: model-level theorem surface with fail-closed dependency guards.

| Check | Mathematical boundary |
|---|---|
| `public_role_count` | a matrix-count-indexed tree has exactly `23 + 2t` leaves |
| diagnostic specialization | the legacy three-row fixture has exactly 29 leaves |
| fixed arities | recursive and terminal attempts use the expected input counts |
| `equations_of_refinement` | exactly three equation artifacts imply generic public equations without sampler membership |
| `output_eq_piDecParent_of_artifacts` | codec/output-shape/parent artifacts identify the strict-PiDEC parent separately |
| trusted-assumption audit | neither exported theorem adds trusted computation or project postulates |
-/

namespace tests.FPrimeFullHistoryPiRlcPaper

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc

example (matrixCount : Nat) :
    (publicOrder matrixCount).length = 23 + 2 * matrixCount :=
  public_role_count matrixCount

example :
    (publicOrder DiagnosticProfile.matrixCount).length =
      DiagnosticProfile.publicLeafCount := by
  simpa [DiagnosticProfile.publicLeafCount] using
    public_role_count DiagnosticProfile.matrixCount

example : DiagnosticProfile.publicLeafCount = 29 :=
  DiagnosticProfile.publicLeafCount_eq_29

example : recursiveArity.total = 1 := by rfl

example : terminalArity.total = 15 := by rfl

#check equations_of_refinement
#check output_eq_piDecParent_of_artifacts

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.equations_of_refinement' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms equations_of_refinement

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.output_eq_piDecParent_of_artifacts' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms output_eq_piDecParent_of_artifacts

end tests.FPrimeFullHistoryPiRlcPaper
