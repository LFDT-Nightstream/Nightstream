import Nightstream.Implementation.R1CS.Correspondence.Gadgets.TerminalCeProgramSound

/-! Focused compile gate for whole-program direct terminal-CE soundness. -/

set_option autoImplicit false

namespace tests.TerminalCeProgramSound

open Nightstream.Implementation.R1CS.TerminalCeProgramSound

#check Structural
#check decodedEvaluations_eq_expected_of_fields
#check decodedNc_eq_expected_of_fields
#check rows_sound

end tests.TerminalCeProgramSound
