import Nightstream.SuperNeo.SumCheck.FixedPhase.SemanticView
import tests.Axioms.Support

open Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.SemanticView

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.SemanticView.accepted_implies_symbolicAccepted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms accepted_implies_symbolicAccepted

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.SemanticView.accepted_implies_truthPath' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms accepted_implies_truthPath
