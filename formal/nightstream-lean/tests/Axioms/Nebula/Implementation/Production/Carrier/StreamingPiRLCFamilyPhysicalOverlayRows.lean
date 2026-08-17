import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilyPhysicalOverlayRows
import tests.Axioms.Support

/-! Dependency audit for the physical production PiRLC family overlay. -/

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows.fieldLinkCount_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms fieldLinkCount_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows.link_run_geometry_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms link_run_geometry_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows.physicalSourceColumnsExact_of_links' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms physicalSourceColumnsExact_of_links

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows.rows_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_sound

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows.AcceptedRows.sound' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AcceptedRows.sound
