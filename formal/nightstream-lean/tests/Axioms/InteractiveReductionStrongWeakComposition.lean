import Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition
import tests.Axioms.Support

/-! Fail-closed dependency guard for the paper strong--weak composition theorem. -/

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.reductionOfKnowledge' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.reductionOfKnowledge
