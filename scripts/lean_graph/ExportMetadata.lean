import tests.EvidenceTargets

/-! Standalone inspection driver. Each witness graph includes its exact target.
Run through validate.sh after building the library and tests.EvidenceTargets.
-/

#evidence_export LeanGraph.Targets.pilotAssignment
#evidence_export LeanGraph.Targets.piCCSAssignment
#evidence_export LeanGraph.Targets.piCCSPublicAssignment
#evidence_export LeanGraph.Targets.stage1Assignment
#evidence_export LeanGraph.Targets.stage1TerminalAssignment
#evidence_export LeanGraph.Targets.stage1TerminalParent

-- The final history target.
#evidence_export LeanGraph.Targets.hyperNovaLinearSecurity
#evidence_export LeanGraph.Targets.hyperNovaTerminalFalseAcceptance
#evidence_export LeanGraph.Targets.piRLCWitnessReplay
#evidence_export NightstreamFPrime.Export.Stage1.PiRLCWitnessHonestResponse.preparedWitnessBlockPartials_honestResponse
