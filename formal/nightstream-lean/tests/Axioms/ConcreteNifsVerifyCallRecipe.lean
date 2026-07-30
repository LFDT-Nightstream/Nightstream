import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsVerifyCallRecipe
import tests.Axioms.Support

namespace NightstreamTests.Axioms.ConcreteNifsVerifyCallRecipe

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsVerifyCallRecipe.active_soundness' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsVerifyCallRecipe.active_soundness

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsVerifyCallRecipe.active_honest_completeness' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsVerifyCallRecipe.active_honest_completeness

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsVerifyCallRecipe.inactive_satisfiable' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsVerifyCallRecipe.inactive_satisfiable

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsVerifyCallRecipe.receipt_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsVerifyCallRecipe.receipt_exact

end NightstreamTests.Axioms.ConcreteNifsVerifyCallRecipe
