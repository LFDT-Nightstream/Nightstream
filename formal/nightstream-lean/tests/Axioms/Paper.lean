import Nightstream.SuperNeo
import tests.Axioms.Support

/-!
Fail-closed paper-model axioms gate. Every expectation is checked when this
module is built; the aggregate entrypoint imports all ownership groups.
-/

/-- info: 'Nightstream.SuperNeo.Concrete.ccsMembership_iff' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.ccsMembership_iff

/-- info: 'Nightstream.SuperNeo.Concrete.ceMembership_iff' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.ceMembership_iff

/-- info: 'Nightstream.SuperNeo.Concrete.canonicalCCS_holds' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.canonicalCCS_holds

/-- info: 'Nightstream.SuperNeo.Concrete.canonicalCE_holds' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.canonicalCE_holds

/-- info: 'Nightstream.SuperNeo.GlobalParams.rlc_bound_for' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.GlobalParams.rlc_bound_for

/-- info: 'Nightstream.SuperNeo.SumCheck.false_acceptance_implies_bad_challenge' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.SumCheck.false_acceptance_implies_bad_challenge

/-- info: 'Nightstream.SuperNeo.SumCheck.check_eq_true_iff_accepted' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.SumCheck.check_eq_true_iff_accepted

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.strong_extract_or_bad_challenge' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.strong_extract_or_bad_challenge

/-- info: 'Nightstream.SuperNeo.Folding.BatchArity.total_le' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.BatchArity.total_le

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.product_complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.product_complete

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.complete

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.combinedOutput_holds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.combinedOutput_holds

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.complete

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.same_phi_extractions_unique_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.same_phi_extractions_unique_or_collision

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.reduce_knowledge' depends on axioms: [Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.reduce_knowledge

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.complete' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.complete

/-- info: 'Nightstream.SuperNeo.Folding.Composition.fold_knowledge_or_bad_event' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Composition.fold_knowledge_or_bad_event

/-- info: 'Nightstream.SuperNeo.ProjectionCheck.batchAccepted_implies_exact_or_badRoot' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.ProjectionCheck.batchAccepted_implies_exact_or_badRoot
