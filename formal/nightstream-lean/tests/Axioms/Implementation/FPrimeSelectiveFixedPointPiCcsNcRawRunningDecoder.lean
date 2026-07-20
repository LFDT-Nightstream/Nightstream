import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc
import tests.Axioms.Support

/-! Kernel dependency report for the bounded running-`X` prefix decoder. -/

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

/-- info: 'Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Exact.sourceColumnMap_injective' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Exact.sourceColumnMap_injective

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.ArtifactRefinement.sourceColumnRowsBind_decodedData' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ArtifactRefinement.sourceColumnRowsBind_decodedData

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.ArtifactRefinement.decodedVirtual_live_eq_rawRunningAssignment' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ArtifactRefinement.decodedVirtual_live_eq_rawRunningAssignment

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.Production.acceptedPair_implies_previousSemanticFold_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.Production.acceptedPair_implies_previousSemanticFold_or_badEvent

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionChecker.check_eq_true_iff_accepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionChecker.check_eq_true_iff_accepted

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.Production.check_eq_true_iff_accepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.Production.check_eq_true_iff_accepted

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.Production.checkedPair_implies_previousSemanticFold_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.Production.checkedPair_implies_previousSemanticFold_or_badEvent

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionChecker.stateBindingCheck_eq_true_iff' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionChecker.stateBindingCheck_eq_true_iff

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.Production.checkedPair_of_stateChecks_implies_previousSemanticFold_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.Production.checkedPair_of_stateChecks_implies_previousSemanticFold_or_badEvent

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.Production.checkedTerminal_implies_semanticFold_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.Production.checkedTerminal_implies_semanticFold_or_badEvent

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.Production.checkedPair_implies_previousSemanticFold_or_namedFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.Production.checkedPair_implies_previousSemanticFold_or_namedFailure
