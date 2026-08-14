import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.RadixFourFirstAcceptedSelectionArtifact
import tests.Axioms.Support

/-! Fail-closed axiom guard for the radix-four selection schedule. -/

namespace NightstreamTests.Axioms.Implementation.FPrimeFullHistoryRadixFourFirstAcceptedSelection

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RadixFourFirstAcceptedSelectionArtifact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RadixFourFirstAcceptedSelectionArtifact.candidate_occurrence_count_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms candidate_occurrence_count_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RadixFourFirstAcceptedSelectionArtifact.candidate_coverage_valid' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms candidate_coverage_valid

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RadixFourFirstAcceptedSelectionArtifact.generated_currentAt_iff_aggregateAt' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms generated_currentAt_iff_aggregateAt

end NightstreamTests.Axioms.Implementation.FPrimeFullHistoryRadixFourFirstAcceptedSelection
