import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive
import Nightstream.SuperNeo.Folding.PiDEC.Necessity.ProductionChildSubstitution
import tests.Axioms.Support

/-! Fail-closed dependency gate for the fixed-active paper profile. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.PaperVerifier.childPublicInputs_eq_of_parentPublicInput_eq' depends on axioms: [Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.PaperVerifier.childPublicInputs_eq_of_parentPublicInput_eq

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.PaperVerifier.OutputAccepted.publicInputs_eq_of_parentPublicInput_eq' depends on axioms: [Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.PaperVerifier.OutputAccepted.publicInputs_eq_of_parentPublicInput_eq

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.PaperVerifier.OutputAccepted.childEvaluations_size' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.PaperVerifier.OutputAccepted.childEvaluations_size

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.PaperVerifier.OutputAccepted.parentEvaluations_size' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.PaperVerifier.OutputAccepted.parentEvaluations_size

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.PaperVerifier.output_complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.PaperVerifier.output_complete

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.PaperVerifier.reduce_knowledge' depends on axioms: [Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.PaperVerifier.reduce_knowledge

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PublicInput.splitPublicInput_project' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PublicInput.splitPublicInput_project

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PublicInput.splitPublicInput_recompose' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PublicInput.splitPublicInput_recompose

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.Necessity.ProductionChildSubstitution.Fixture.rightNotPaperAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.Necessity.ProductionChildSubstitution.Fixture.rightNotPaperAccepted

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.Necessity.ProductionChildSubstitution.Fixture.trailingEvaluationChildren_notPaperAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.Necessity.ProductionChildSubstitution.Fixture.trailingEvaluationChildren_notPaperAccepted

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperProfile.arity_total' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.PaperProfile.arity_total

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperProfile.complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.PaperProfile.complete

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperProfile.Realization.parentOpening_eq_recompose_or_bindingCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.PaperProfile.Realization.parentOpening_eq_recompose_or_bindingCollision

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.Profile.sourceCount_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.Profile.sourceCount_eq

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.Realization.toGeneric' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.Realization.toGeneric

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.complete

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.Realization.parentOpening_eq_recompose_or_bindingCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.Realization.parentOpening_eq_recompose_or_bindingCollision

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.resultTransition_iff_exists_paperDecomposition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.resultTransition_iff_exists_paperDecomposition

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.ResultTransition.toPaperProfile' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.ResultTransition.toPaperProfile
