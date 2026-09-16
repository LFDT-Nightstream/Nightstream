import tests.FreshWitnessEvidence
import tests.EvidenceTargets

/-! Checker-owned invocations of the approved target and metadata checks.
Candidate declarations cannot omit these invocations from an acceptance run.
-/

#evidence_closed LeanGraph.Targets.PilotAssignment by LeanGraph.Targets.pilotAssignment
#evidence_closed LeanGraph.Targets.PiCCSAssignment by LeanGraph.Targets.piCCSAssignment

#evidence_closed LeanGraph.Targets.PiCCSPublicAssignment by LeanGraph.Targets.piCCSPublicAssignment
#evidence_closed LeanGraph.Targets.Stage1Assignment by LeanGraph.Targets.stage1Assignment
#evidence_closed LeanGraph.Targets.Stage1TerminalAssignment by LeanGraph.Targets.stage1TerminalAssignment
#evidence_closed LeanGraph.Targets.Stage1TerminalParent by LeanGraph.Targets.stage1TerminalParent
#evidence_closed LeanGraph.Targets.HyperNovaLinearSecurity by LeanGraph.Targets.hyperNovaLinearSecurity

#evidence_closed LeanGraph.Targets.HyperNovaTerminalFalseAcceptance by LeanGraph.Targets.hyperNovaTerminalFalseAcceptance
#evidence_closed LeanGraph.Targets.PiRLCWitnessReplay by LeanGraph.Targets.piRLCWitnessReplay
#evidence_closed LeanGraph.Targets.PiDECWitnessReplay by LeanGraph.Targets.piDECWitnessReplay
#evidence_closed LeanGraph.Targets.PiDECCommitmentReplay by LeanGraph.Targets.piDECCommitmentReplay
#evidence_closed LeanGraph.Targets.PiDECChildEvaluationReplay by LeanGraph.Targets.piDECChildEvaluationReplay

#evidence_closed LeanGraph.Targets.PiCCSFirstRoundKernel by LeanGraph.Targets.piCCSFirstRoundKernel

#evidence_closed LeanGraph.Targets.PiCCSFirstRoundReplayKernel by LeanGraph.Targets.piCCSFirstRoundReplayKernel

#evidence_closed LeanGraph.Targets.PiCCSNormPrefixKernel by LeanGraph.Targets.piCCSNormPrefixKernel

#evidence_closed LeanGraph.Targets.PiCCSPrefixNormAccumulation by LeanGraph.Targets.piCCSPrefixNormAccumulation

#evidence_closed LeanGraph.Targets.PiCCSRetainedPrefixKernel by LeanGraph.Targets.piCCSRetainedPrefixKernel

#evidence_closed LeanGraph.Targets.PiCCSOriginalEvaluationKernel by LeanGraph.Targets.piCCSOriginalEvaluationKernel

#evidence_closed LeanGraph.Targets.FreshWitnessKernels by LeanGraph.Targets.freshWitnessKernels
