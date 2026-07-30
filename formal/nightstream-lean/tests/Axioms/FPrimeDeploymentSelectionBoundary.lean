import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DeploymentSelectionBoundary
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.FPrimeDeploymentSelectionBoundary

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DeploymentSelectionBoundary.fixed_call_footprints_equal' does not depend on any axioms -/
#guard_msgs in
#audit_axioms DeploymentSelectionBoundary.fixed_call_footprints_equal

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DeploymentSelectionBoundary.footprint_fields_do_not_determine_step_or_nifs_rows' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms DeploymentSelectionBoundary.footprint_fields_do_not_determine_step_or_nifs_rows

end NightstreamTests.Axioms.FPrimeDeploymentSelectionBoundary
