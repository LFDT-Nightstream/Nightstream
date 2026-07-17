import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.SemanticHandoff
import tests.Axioms.Support

/-! Fail-closed dependency gate for the typed terminal sampler handoff. -/

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.SemanticHandoff.accepted_refines_semanticHandoffBound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.SemanticHandoff.accepted_refines_semanticHandoffBound
