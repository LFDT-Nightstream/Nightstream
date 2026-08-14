import Nightstream.Implementation.Nebula.NIFS.PiRLC.RingCombinationSound
import tests.Axioms.Support

/-! Dependency audit for one exact V2 PiRLC ring-combination occurrence. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationRows.rows_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationRows.rows_sound

/-- info: 'Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationSound.sourceOutputTerms_field' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationSound.sourceOutputTerms_field

/-- info: 'Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationSound.rows_imply_ring_combination' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationSound.rows_imply_ring_combination
