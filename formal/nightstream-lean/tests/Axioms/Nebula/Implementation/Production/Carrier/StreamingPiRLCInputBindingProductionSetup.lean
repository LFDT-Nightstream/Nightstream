import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputBindingProductionSetup
import tests.Axioms.Support

/-! Dependency audit for the fixed production PiRLC input setup. -/

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingProductionSetup

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingProductionSetup.exact_identity' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms exact_identity
