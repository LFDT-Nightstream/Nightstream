import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.CanonicalOpeningArtifactRows
import tests.Axioms.Support

/-! Fail-closed dependency gate for the generated canonical-opening row bridge. -/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningArtifactRows.finiteRowTransition' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningArtifactRows.finiteRowTransition

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningArtifactRows.artifactRows_imply_chunkScheduleHolds' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningArtifactRows.artifactRows_imply_chunkScheduleHolds
