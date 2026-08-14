import Nightstream.Implementation.Nebula.NIFS.Core.PaperAlgebra
import tests.Axioms.Support

/-! Fail-closed dependency guard for the exact V2 product paper algebra. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductPaperAlgebra.canonicalStructure_matrixSource' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPaperAlgebra.canonicalStructure_matrixSource

/-- info: 'Nightstream.Implementation.Nebula.ProductPaperAlgebra.ambientAgreement' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPaperAlgebra.ambientAgreement

/-- info: 'Nightstream.Implementation.Nebula.ProductPaperAlgebra.openingAgreement' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPaperAlgebra.openingAgreement

/-- info: 'Nightstream.Implementation.Nebula.ProductPaperAlgebra.evaluations_combine' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPaperAlgebra.evaluations_combine

/-- info: 'Nightstream.Implementation.Nebula.ProductPaperAlgebra.evaluations_recompose' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPaperAlgebra.evaluations_recompose
