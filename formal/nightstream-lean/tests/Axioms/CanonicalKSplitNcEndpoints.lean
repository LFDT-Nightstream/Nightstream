import Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointsSemanticHonest
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKSplitNcEndpoints

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpoints.endpointAgrees_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcEndpoints.endpointAgrees_of_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointsHonest.witness_off_source' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcEndpointsHonest.witness_off_source

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointsHonest.rows_honest_of_bindings' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcEndpointsHonest.rows_honest_of_bindings

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointsSemanticHonest.bindings_of_endpointAgrees' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcEndpointsSemanticHonest.bindings_of_endpointAgrees

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointsSemanticHonest.rows_honest_of_endpointAgrees' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcEndpointsSemanticHonest.rows_honest_of_endpointAgrees

end NightstreamTests.Axioms.CanonicalKSplitNcEndpoints
