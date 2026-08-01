import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationCertification
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.RunningCheckRecipe
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.FreshCheckRecipe
import tests.Axioms.Support

namespace NightstreamTests.Axioms.Poseidon23ApplicationCertification

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23HashPriorRecipe.recipe' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon23HashPriorRecipe.recipe

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23HashNextRecipe.recipe' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon23HashNextRecipe.recipe

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.RunningCheckRecipe.recipe' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RunningCheckRecipe.recipe

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.FreshCheckRecipe.recipe' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms FreshCheckRecipe.recipe

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationCertification.poseidon23' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ApplicationCertification.poseidon23

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationCertification.hash_outputs_distinct' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ApplicationCertification.hash_outputs_distinct

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationCertification.terminal_calls_distinct' does not depend on any axioms -/
#guard_msgs in
#audit_axioms ApplicationCertification.terminal_calls_distinct

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationCertification.call_multiplicities' does not depend on any axioms -/
#guard_msgs in
#audit_axioms ApplicationCertification.call_multiplicities

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationCertification.hashPriorCost_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ApplicationCertification.hashPriorCost_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationCertification.hashNextCost_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ApplicationCertification.hashNextCost_exact

end NightstreamTests.Axioms.Poseidon23ApplicationCertification
