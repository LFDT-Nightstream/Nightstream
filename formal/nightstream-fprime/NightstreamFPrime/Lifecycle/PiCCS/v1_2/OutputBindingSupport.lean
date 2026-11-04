import NightstreamFPrime.Gadgets.Poseidon2.Duplex.WiringSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.OutputBinding

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_2.OutputBinding

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex

private theorem outputChunks_nonempty (interface : Interface) (offset : Nat) :
    Hash.inputChunks (outputWords interface offset) ≠ [] := by
  apply List.ne_nil_of_length_pos
  unfold Hash.inputChunks
  simp only [List.length_map, List.length_range, outputWords_length]
  norm_num [Spec.Poseidon2.rate]

theorem finalState_supported_from_offset (interface : Interface)
    (offset : Nat) :
    Formal.StateSupported (finalState interface offset)
      (fun index => offset <= index) := by
  let input := outputWords interface offset
  have chunksNonempty : Hash.inputChunks input ≠ [] :=
    outputChunks_nonempty interface offset
  rw [finalState_eq_compile]
  rw [← (Formal.compileWiring_matches offset (interface.initialState offset)
    (actions interface offset)).2]
  change Formal.StateSupported
    (Formal.compileWiring offset (interface.initialState offset)
      [.absorb input]).output (fun index => offset <= index)
  simp only [Formal.compileWiring]
  cases chunks : Hash.inputChunks input with
  | nil => exact False.elim (chunksNonempty chunks)
  | cons block rest =>
      simpa [chunks] using
        NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.compileAbsorbWiring_output_supported_from_start
          offset (interface.initialState offset) block rest

end NightstreamFPrime.Lifecycle.PiCCS.v1_2.OutputBinding
