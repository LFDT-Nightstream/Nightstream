import Nightstream.SuperNeo.Folding.PiRLC.PaperForkAlgebra
import tests.Axioms.Support

/-!
Fail-closed trusted-dependency gate for the artifact-independent `Pi_RLC`
fork algebra.
-/

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.PaperForkAlgebra.coordinateIsolation' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.PaperForkAlgebra.coordinateIsolation

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.PaperForkAlgebra.inverseActionCancellation' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.PaperForkAlgebra.inverseActionCancellation
