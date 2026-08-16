import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilyCarryRows
import tests.Axioms.Support

/-! Dependency audit for the production PiRLC challenge and cursor rows. -/

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.rows_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_length

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.rows_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_sound
