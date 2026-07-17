import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.BlockLane.NcRefinement
import tests.Axioms.Support

/-! Fail-closed dependency gate for canonical Block×Lane NC Poseidon2 replay refinement. -/

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.BlockLane.NcRefinement.concreteRounds_eq_block_then_lane' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.BlockLane.NcRefinement.concreteRounds_eq_block_then_lane

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.BlockLane.NcRefinement.derive_refines_runRounds' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.BlockLane.NcRefinement.derive_refines_runRounds
