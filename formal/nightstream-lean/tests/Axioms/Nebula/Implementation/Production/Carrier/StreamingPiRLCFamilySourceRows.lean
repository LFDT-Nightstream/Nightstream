import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilySourceRows
import tests.Axioms.Support

/-! Dependency audit for the complete production PiRLC family source rows. -/

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows.rows_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_length

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows.rows_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_sound
