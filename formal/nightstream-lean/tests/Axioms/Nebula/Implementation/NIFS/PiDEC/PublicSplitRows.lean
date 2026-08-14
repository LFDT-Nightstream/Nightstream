import Nightstream.Implementation.Nebula.NIFS.PiDEC.PublicSplitRows
import tests.Axioms.Support

/-! Dependency audit for the exact V2 PiDEC public-input split rows. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductPiDecPublicSplitRows.rows_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiDecPublicSplitRows.rows_sound
