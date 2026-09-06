import tests.EvidenceTargets

/-! Checker-owned invocations of the approved target and metadata checks.
Candidate declarations cannot omit these invocations from an acceptance run.
-/

#evidence_closed LeanGraph.Targets.PilotAssignment by LeanGraph.Targets.pilotAssignment
#evidence_closed LeanGraph.Targets.PiCCSAssignment by LeanGraph.Targets.piCCSAssignment

#evidence_closed LeanGraph.Targets.PiCCSPublicAssignment by LeanGraph.Targets.piCCSPublicAssignment
