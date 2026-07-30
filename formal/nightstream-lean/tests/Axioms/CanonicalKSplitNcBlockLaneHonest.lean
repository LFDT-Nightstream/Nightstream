import Nightstream.Implementation.R1CS.Canonical.KSplitNcBlockLaneHonest
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.CanonicalKSplitNcBlockLaneHonest

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence.SourceColumns.numericRows_honest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  KFixedPhaseSemanticOccurrence.SourceColumns.numericRows_honest

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence.SourceColumns.numericRows_columns_below_end' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  KFixedPhaseSemanticOccurrence.SourceColumns.numericRows_columns_below_end

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcBlockLaneHonest.witness_off_block' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcBlockLaneHonest.witness_off_block

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcBlockLaneHonest.rows_honest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcBlockLaneHonest.rows_honest

end NightstreamTests.Axioms.CanonicalKSplitNcBlockLaneHonest
