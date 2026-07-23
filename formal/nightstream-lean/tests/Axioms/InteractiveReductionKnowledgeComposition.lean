import Nightstream.SuperNeo.InteractiveReduction.KnowledgeComposition
import tests.Axioms.Support

/-! Fail-closed dependency guard for sequential knowledge composition. -/

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.KnowledgeComposition.reductionOfKnowledge' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.KnowledgeComposition.reductionOfKnowledge
