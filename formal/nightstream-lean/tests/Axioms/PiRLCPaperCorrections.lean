import Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections
import tests.Axioms.Support

/-! Fail-closed trusted-dependency gate for the corrected ambient bound. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.midpointResidue_not_literalAmbientBounded' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.midpointResidue_not_literalAmbientBounded

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.all_centeredMagnitude_lt_correctedAmbientBound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.all_centeredMagnitude_lt_correctedAmbientBound

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.production_correctedAmbientBoundFor_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.production_correctedAmbientBoundFor_eq
