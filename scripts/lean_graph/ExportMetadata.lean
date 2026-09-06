import tests.EvidenceTargets

/-! Standalone inspection driver. Each witness graph includes its exact target.
Run through validate.sh after building tests.EvidenceTargets.
-/

#evidence_export LeanGraph.Targets.pilotAssignment
#evidence_export LeanGraph.Targets.piCCSAssignment
#evidence_export LeanGraph.Targets.piCCSPublicAssignment
