import Nightstream.Implementation.Nebula.FPrime.Terminal.ProductRelationBridge
import tests.Axioms.Support

open Nightstream.Implementation.Nebula

/-- info: 'Nightstream.Implementation.Nebula.ProductTerminalRelation.holds_of_common_openings' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductTerminalRelation.holds_of_common_openings

/-- info: 'Nightstream.Implementation.Nebula.ProductTerminalRelation.core_of_holds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms ProductTerminalRelation.core_of_holds

/-- info: 'Nightstream.Implementation.Nebula.ProductTerminalRelation.canonical_children_hold' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductTerminalRelation.canonical_children_hold

/-- info: 'Nightstream.Implementation.Nebula.ProductTerminalRelation.commitment_of_holds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms ProductTerminalRelation.commitment_of_holds

/-- info: 'Nightstream.Implementation.Nebula.ProductTerminalRelation.combined_children_can_satisfy_ce_but_not_terminal' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductTerminalRelation.combined_children_can_satisfy_ce_but_not_terminal

/-- info: 'Nightstream.Implementation.Nebula.TerminalProductRelationBridge.terminal_children_hold_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalProductRelationBridge.terminal_children_hold_of_rows

/-- info: 'Nightstream.Implementation.Nebula.TerminalProductRelationBridge.terminal_core_check_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalProductRelationBridge.terminal_core_check_complete
