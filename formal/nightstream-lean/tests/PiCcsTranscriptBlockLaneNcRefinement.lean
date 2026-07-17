import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.BlockLane.NcRefinement
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain

/-!
Focused regressions for canonical Block×Lane NC Poseidon2 replay refinement.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.block_lane.poseidon.count` | fixed carrier projects to exactly nine concrete rounds | accidental legacy 15-round reuse |
| `nifs.pi_ccs.nc.block_lane.poseidon.phase_cut` | block and lane messages remain contiguous | reset, marker, or reorder at phase cut |
| `nifs.pi_ccs.nc.block_lane.poseidon.replay` | typed point/state equals concrete Poseidon2 replay | semantic/concrete transcript drift |
-/

namespace NightstreamTests.PiCcsTranscriptBlockLaneNcRefinement

open Nightstream.Implementation.R1CS.PiCcsTranscript.BlockLane.NcRefinement
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.BlockLane

example (certificate : Certificate PiCcsDomain.blockDomain) :
    (concreteRounds certificate).length = 9 := by
  simpa [PiCcsDomain.blockDomain_blockVariables,
    PiCcsDomain.blockDomain_laneVariables] using
    concreteRounds_length certificate

#check concreteRounds_eq_block_then_lane
#check derive_refines_runRounds

end NightstreamTests.PiCcsTranscriptBlockLaneNcRefinement
