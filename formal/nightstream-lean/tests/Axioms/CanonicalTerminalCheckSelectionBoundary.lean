import Nightstream.Implementation.R1CS.Canonical.TerminalCheckSelectionBoundary
import tests.Axioms.Support
namespace NightstreamTests.Axioms.CanonicalTerminalCheckSelectionBoundary
open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TerminalCheckSelectionBoundary.runningCheck_is_a_real_choice' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalCheckSelectionBoundary.runningCheck_is_a_real_choice

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TerminalCheckSelectionBoundary.freshCheck_is_a_real_choice' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalCheckSelectionBoundary.freshCheck_is_a_real_choice

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TerminalCheckSelectionBoundary.running_valid_fresh_invalid' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalCheckSelectionBoundary.running_valid_fresh_invalid

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TerminalCheckSelectionBoundary.fresh_valid_running_invalid' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalCheckSelectionBoundary.fresh_valid_running_invalid

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TerminalCheckSelectionBoundary.nifs_accepts_while_fresh_terminal_is_false' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms TerminalCheckSelectionBoundary.nifs_accepts_while_fresh_terminal_is_false

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TerminalCheckSelectionBoundary.lawful_checkers_still_disagree' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalCheckSelectionBoundary.lawful_checkers_still_disagree

end NightstreamTests.Axioms.CanonicalTerminalCheckSelectionBoundary
