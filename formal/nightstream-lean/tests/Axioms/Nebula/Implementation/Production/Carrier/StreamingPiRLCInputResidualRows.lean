import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputResidualRows
import tests.Axioms.Support

/-! Dependency audit for the production PiRLC residual-link rows. -/

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows.rows_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms rows_length

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows.residualField_exact_of_row' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms residualField_exact_of_row

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows.rows_imply_addResidualFields' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_imply_addResidualFields

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows.rows_imply_concreteResidualTransition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_imply_concreteResidualTransition

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows.rows_complete' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms rows_complete
