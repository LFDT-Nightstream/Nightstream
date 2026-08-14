import Nightstream.Implementation.Nebula.NIFS.Terminal.Acceptance
import tests.Axioms.Support

open Nightstream.Implementation.Nebula

/-- info: 'Nightstream.Implementation.Nebula.ProductTerminalAcceptance.acceptedOfRows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductTerminalAcceptance.acceptedOfRows

/-- info: 'Nightstream.Implementation.Nebula.ProductTerminalAcceptance.consumes_exact_selected_trailing_claim' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductTerminalAcceptance.consumes_exact_selected_trailing_claim

/-- info: 'Nightstream.Implementation.Nebula.ProductTerminalAcceptance.common_product_witnesses' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductTerminalAcceptance.common_product_witnesses
