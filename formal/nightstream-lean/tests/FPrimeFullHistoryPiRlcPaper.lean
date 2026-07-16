import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc

/-!
External checks for the conditional fixed-profile Pi_RLC paper bridge.

| Check | Mathematical boundary |
|---|---|
| `public_role_count` | the paper-public projection tree has exactly 29 leaves |
| fixed arities | recursive and terminal attempts use the expected input counts |
| `accepted_of_refinement` | the four explicit implementation artifacts imply generic paper acceptance |
| `output_eq_piDecParent_of_refinement` | the combined output is the strict-PiDEC parent carrier |
| trusted-assumption audit | neither exported theorem adds trusted computation or project postulates |
-/

namespace tests.FPrimeFullHistoryPiRlcPaper

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc

example : publicOrder.length = 29 := public_role_count

example : recursiveArity.total = 1 := by rfl

example : terminalArity.total = 15 := by rfl

#check accepted_of_refinement
#check output_eq_piDecParent_of_refinement

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.accepted_of_refinement' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms accepted_of_refinement

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.output_eq_piDecParent_of_refinement' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms output_eq_piDecParent_of_refinement

end tests.FPrimeFullHistoryPiRlcPaper
