import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.SemanticHandoff
import tests.Axioms.Support

/-!
Fail-closed kernel dependency expectations for the lossless Split-NC output
projection and conditional exact transcript handoff.
-/

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Projection.SplitNc.projectOutputs_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Projection.SplitNc.projectOutputs_injective

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.SemanticHandoff.accepted_refines_run' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.SemanticHandoff.accepted_refines_run
