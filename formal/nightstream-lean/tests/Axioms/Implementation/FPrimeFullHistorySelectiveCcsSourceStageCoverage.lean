import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.RadixFourSourceStageCoverageArtifact
import tests.Axioms.Support

/-! Fail-closed axiom guard for the radix-four source-stage census. -/

namespace NightstreamTests.Axioms.Implementation.FPrimeFullHistorySelectiveCcsSourceStageCoverage

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RadixFourSourceStageCoverageArtifact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RadixFourSourceStageCoverageArtifact.candidate_coverage_valid' does not depend on any axioms -/
#guard_msgs in
#audit_axioms candidate_coverage_valid

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RadixFourSourceStageCoverageArtifact.candidate_source_fields_partition' does not depend on any axioms -/
#guard_msgs in
#audit_axioms candidate_source_fields_partition

end NightstreamTests.Axioms.Implementation.FPrimeFullHistorySelectiveCcsSourceStageCoverage
