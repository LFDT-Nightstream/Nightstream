import Nightstream.Implementation.NebulaV2.Production.NIFS.Core.NifsOutputRowsFor
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductionProductNifsOutputRowsFor

/-! Dependency gate for the complete paper-NIFS output carrier. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionProductNifsOutputRowsFor.rows_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionProductNifsOutputRowsFor.rows_sound

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionProductNifsOutputRowsFor.sources_of_nifs_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionProductNifsOutputRowsFor.sources_of_nifs_rows

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionProductNifsOutputRowsFor.section_rows_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionProductNifsOutputRowsFor.section_rows_sound

end tests.Axioms.NebulaV2ProductionProductNifsOutputRowsFor
