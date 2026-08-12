import Nightstream.Implementation.NebulaV2.ProductPiRlcRingCombinationSound
import tests.Axioms.Support

/-! Dependency audit for one exact V2 PiRLC ring-combination occurrence. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiRlcRingCombinationRows.rows_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiRlcRingCombinationRows.rows_sound

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiRlcRingCombinationSound.sourceOutputTerms_field' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiRlcRingCombinationSound.sourceOutputTerms_field

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiRlcRingCombinationSound.rows_imply_ring_combination' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiRlcRingCombinationSound.rows_imply_ring_combination
