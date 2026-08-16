import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputFamilyRows
import tests.Axioms.Support

/-! Dependency audit for the complete production PiRLC input-family rows. -/

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.rows_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_length

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.rows_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_sound
