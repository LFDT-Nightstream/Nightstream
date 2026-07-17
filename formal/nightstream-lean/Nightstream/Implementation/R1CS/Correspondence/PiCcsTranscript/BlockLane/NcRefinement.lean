import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.NcRefinement
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.BlockLane

/-!
Poseidon2 replay refinement for canonical block×lane NC certificates.

Assurance tier: model-level executable Poseidon2 refinement.

Owns: lossless projection of the exact block-then-lane certificate into the
existing five-extension/ten-field concrete round carrier; preservation of
the product-domain round count and phase cut; and equality between the typed
Block×Lane derivation and concrete Poseidon2 round replay.

Does not own: pre-SumCheck challenge derivation, a complete transcript
schedule, a protocol tag for `betaBlock`, output serialization, generated
rows, artifact hashes, Rust conformance, R1CS soundness, costs, or row removal.

Emits constraints: no.

Authority boundary: this adapter reuses only the domain-independent NC
prologue, exact round serialization, absorb/squeeze machine, and transport
proofs. The legacy flat 15-round schedule and generated terminal ranges are
not imported as canonical authority.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.block_lane.poseidon.message` | five `K` coefficients serialize as the existing ten fields | direct dataflow | `concreteRounds` |
| `nifs.pi_ccs.nc.block_lane.poseidon.count` | concrete replay has exactly `blockVariables + laneVariables` rounds | derived | `concreteRounds_length` |
| `nifs.pi_ccs.nc.block_lane.poseidon.phase_cut` | concrete block prefix is followed directly by the lane suffix | direct dataflow | `concreteRounds_eq_block_then_lane` |
| `nifs.pi_ccs.nc.block_lane.poseidon.replay` | typed point and successor state equal one concrete Poseidon2 replay | derived | `derive_refines_runRounds` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.BlockLane.NcRefinement

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiCcsTranscript.Transport
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc

private abbrev toConcreteRound :=
  Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.toConcreteRound

/-- Canonical concrete message list in physical block-then-lane order. -/
def concreteRounds
    {domain : BlockNcDomain}
    (certificate : BlockLane.Certificate domain) :
    List SumCheck.RoundMessage :=
  certificate.rawRounds.map toConcreteRound

/-- Concrete projection preserves the exact product-domain arity. -/
@[simp] theorem concreteRounds_length
    {domain : BlockNcDomain}
    (certificate : BlockLane.Certificate domain) :
    (concreteRounds certificate).length =
      domain.blockVariables + domain.laneVariables := by
  simp [concreteRounds, BlockLane.Certificate.rawRounds_length,
    BlockLane.roundCount]

/-- The concrete list preserves the semantic block/lane cut without adding
a prologue, reset, marker, or reordering at the boundary. -/
theorem concreteRounds_eq_block_then_lane
    {domain : BlockNcDomain}
    (certificate : BlockLane.Certificate domain) :
    concreteRounds certificate =
      certificate.blockRounds.map toConcreteRound ++
        certificate.laneRounds.map toConcreteRound := by
  rw [← List.map_append,
    BlockLane.Certificate.blockRounds_append_laneRounds]
  rfl

private theorem concreteRounds_eq_raw
    {domain : BlockNcDomain}
    (certificate : BlockLane.Certificate domain) :
    concreteRounds certificate =
      certificate.rawRounds.map
        Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.toConcreteRound :=
  rfl

/-- Full typed Block×Lane derivation equals concrete NC Poseidon2 replay,
jointly for the challenge vector and successor state. -/
theorem derive_refines_runRounds
    {domain : BlockNcDomain}
    (initial : State)
    (certificate : BlockLane.Certificate domain) :
    ((BlockLane.derive
        Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.machine
        initial certificate).challengePoint.coordinates,
      (BlockLane.derive
        Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.machine
        initial certificate).finalState) =
      let concrete := SumCheck.runRounds
        (Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.machine.enterNc
          initial)
        (concreteRounds certificate)
      (concrete.2.map toK, concrete.1) := by
  rw [BlockLane.derive_coordinates_finalState]
  rw [concreteRounds_eq_raw]
  exact
    Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.runRoundsFrom_refines
      (Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.machine.enterNc
        initial)
      certificate.rawRounds

end Nightstream.Implementation.R1CS.PiCcsTranscript.BlockLane.NcRefinement
