import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputPhaseRows
import tests.Axioms.Support

/-! Dependency audit for the exact production PiRLC family commitment rows. -/

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.exact_chunk_geometry' does not depend on any axioms -/
#guard_msgs in
#audit_axioms exact_chunk_geometry

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.sourceRows_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms sourceRows_length

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.coordinateBlock_inputValue_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms coordinateBlock_inputValue_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.compact_output_exact_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms compact_output_exact_of_rows

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.rows_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_length

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.Exact.output_at' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Exact.output_at

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.rows_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_sound
