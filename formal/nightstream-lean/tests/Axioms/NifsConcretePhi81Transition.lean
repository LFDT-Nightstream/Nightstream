import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
import tests.Axioms.Support

/-! Fail-closed dependency gate for the exact concrete Phi81 NIFS composition. -/

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Context.piCcsStatement_sources' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Context.piCcsStatement_sources

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Context.piCcsStatement_polynomial' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Context.piCcsStatement_polynomial

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Context.piCcsStatement_runningParent' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Context.piCcsStatement_runningParent

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Context.system_eq_firstFresh' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Context.system_eq_firstFresh

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Context.ncCoins_betaA_eq_feCoins_betaA' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Context.ncCoins_betaA_eq_feCoins_betaA

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Context.ncCoins_gamma_eq_feCoins_gamma' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Context.ncCoins_gamma_eq_feCoins_gamma

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive_piRlcInitialState' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive_piRlcInitialState

/-! Canonical outgoing PiRLC retained/eliminated obligation ledger. -/

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiRlc.equations_iff_sourceStructures' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiRlc.equations_iff_sourceStructures

/-! Executable bounded-sampler checker. -/

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.certificateCheck_eq_true_iff_accepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.certificateCheck_eq_true_iff_accepted

/-! Canonical outgoing PiDEC retained/eliminated obligation ledger. -/

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiDec.accepted_iff_recomposition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiDec.accepted_iff_recomposition

/-! Executable outgoing PiDEC recomposition checker. -/

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiDec.Checker.check_eq_true_iff_recomposition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiDec.Checker.check_eq_true_iff_recomposition

/-! Canonical fixed-active structure/stage carrier. -/

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical.Context.sourceStructures' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical.Context.sourceStructures

/-! Minimal executable incoming checked-parent authority. -/

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.RunningAuthority.Accepted.parentAbsent_of_bootstrap' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.RunningAuthority.Accepted.parentAbsent_of_bootstrap

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.RunningAuthority.Accepted.iff_parentAbsent_of_bootstrap' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.RunningAuthority.Accepted.iff_parentAbsent_of_bootstrap

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.RunningAuthority.Accepted.iff_nonemptyBound_of_active' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.RunningAuthority.Accepted.iff_nonemptyBound_of_active

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.RunningAuthority.Accepted.rejects_parent_in_bootstrap' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.RunningAuthority.Accepted.rejects_parent_in_bootstrap

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical.RunningAuthority.check_eq_true_iff_accepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical.RunningAuthority.check_eq_true_iff_accepted

/-! Complete executable canonical NIFS checker. -/

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical.Checker.check_eq_true_iff_accepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical.Checker.check_eq_true_iff_accepted

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.RunningAuthority.Accepted.children_sharePoint' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.RunningAuthority.Accepted.children_sharePoint

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.TailAccepted.piRlcAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.TailAccepted.piRlcAccepted

/-! Certificate-independent fold refinement. -/

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.Realization.canonicalChildren' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.Realization.canonicalChildren

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.Holds.canonicalChildren' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.Holds.canonicalChildren

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Result.ResultTransition.canonicalChildren' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Result.ResultTransition.canonicalChildren

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.ObligationPlan.exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.ObligationPlan.exact

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.ObligationPlan.Necessity.Realization.parentNecessary' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.ObligationPlan.Necessity.Realization.parentNecessary

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.ObligationPlan.Necessity.Realization.challengeNecessary' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.ObligationPlan.Necessity.Realization.challengeNecessary

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.ObligationPlan.Necessity.Realization.childrenNecessary' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.ObligationPlan.Necessity.Realization.childrenNecessary

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Result.resultTransition_iff_exists_obligationPlan' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Result.resultTransition_iff_exists_obligationPlan

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CertificateRefinement.toSemanticFold' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CertificateRefinement.toSemanticFold

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Result.resultOf_refines' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Result.resultOf_refines

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Result.ResultTransition.inputRunningOpenings' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Result.ResultTransition.inputRunningOpenings

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Result.ResultTransition.runningAuthority' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Result.ResultTransition.runningAuthority

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.accepted_implies_paper_or_outputUnbound_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.accepted_implies_paper_or_outputUnbound_or_badEvent

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.accepted_implies_refinement_or_outputUnbound_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.accepted_implies_refinement_or_outputUnbound_or_badEvent

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.accepted_implies_transition_or_outputUnbound_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.accepted_implies_transition_or_outputUnbound_or_badEvent

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.complete_of_paperObligations' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.complete_of_paperObligations

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.complete_or_samplerShortfall' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.complete_or_samplerShortfall

/-! Fixed-active result and evaluator boundary. -/

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.ResultTransition.canonicalChildren' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.ResultTransition.canonicalChildren

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.ResultTransition.children_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.ResultTransition.children_transition

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.ResultTransition.inputRunningOpenings' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.ResultTransition.inputRunningOpenings

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.ResultTransition.inputRunningPiDec' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.ResultTransition.inputRunningPiDec

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.ResultTransition.runningStructure_eq_fresh' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.ResultTransition.runningStructure_eq_fresh

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.ResultTransition.childStructure_eq_fresh' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.ResultTransition.childStructure_eq_fresh

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.ResultTransition.parent_eq_of_children_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.ResultTransition.parent_eq_of_children_eq

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.transition_iff_exists_resultTransition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.transition_iff_exists_resultTransition

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Evaluator.run_eq_some_iff_accepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Evaluator.run_eq_some_iff_accepted

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Evaluator.run_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Evaluator.run_sound

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Evaluator.run_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Evaluator.run_complete

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Evaluator.run_complete_or_samplerShortfall' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Evaluator.run_complete_or_samplerShortfall
