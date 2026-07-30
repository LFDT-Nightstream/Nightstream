import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsParameters
import tests.Axioms.Support

/-!
Fail-closed axiom guards for the selected paper-NIFS lowering parameters.

The expected reports are measured from raw `#print axioms` output. Neither
headline theorem may acquire compiler trust or a protocol assumption.
-/

namespace NightstreamTests.Axioms.CanonicalPaperNifsParameters

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsParameters.selected_setup_nifs' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PaperNifsParameters.selected_setup_nifs

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsParameters.callEval_nifsVerify' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PaperNifsParameters.callEval_nifsVerify

end NightstreamTests.Axioms.CanonicalPaperNifsParameters
