import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.Certificate
import tests.Axioms.Support

/-! Fail-closed dependency gate for the terminal concrete sampler certificate. -/

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Certificate.accepted_refines_certificateAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Certificate.accepted_refines_certificateAccepted
