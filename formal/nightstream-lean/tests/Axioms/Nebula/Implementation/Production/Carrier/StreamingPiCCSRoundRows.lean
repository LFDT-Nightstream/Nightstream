import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCCSRoundRows
import tests.Axioms.Support

/-! Dependency audit for the exact production PiCCS round rows. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsRoundRows.rows_imply_concrete_round' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiCcsRoundRows.rows_imply_concrete_round

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsRoundRows.rows_imply_roundPhaseRelation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiCcsRoundRows.rows_imply_roundPhaseRelation
