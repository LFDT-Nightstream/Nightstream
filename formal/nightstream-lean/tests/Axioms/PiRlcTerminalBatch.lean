import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.Batch
import tests.Axioms.Support

/-! Fail-closed dependency gate for the terminal Π_RLC batch refinement. -/

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.FirstAccepted.enoughAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.FirstAccepted.enoughAccepted

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.MachineOutput.enoughAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.MachineOutput.enoughAccepted

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Batch.stateAt_refines' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Batch.stateAt_refines

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Batch.execution_exists' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Batch.execution_exists

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Batch.accepted_refines_batch' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Batch.accepted_refines_batch

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Batch.RefinedBatch.challenge_eq_machineChallenge' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Batch.RefinedBatch.challenge_eq_machineChallenge

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Batch.accepted_refines_initialStateBound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Batch.accepted_refines_initialStateBound
