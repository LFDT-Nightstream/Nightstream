import NightstreamFPrime.Gadgets.Poseidon2.Duplex.WiringSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding

/-!
Owns the variable-support fact for the PiCCS output-binding transcript
endpoint. The proof uses the recipe-free Duplex wiring projection; the full
compiler remains the semantic and row authority.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex

private theorem framedChunks_nonempty (words : List Expr) :
    Hash.inputChunks (StatementAbsorption.blockExpr words) ≠ [] := by
  apply List.ne_nil_of_length_pos
  unfold Hash.inputChunks
  simp only [List.length_map, List.length_range]
  apply Nat.div_pos
  · simp [StatementAbsorption.blockExpr, Spec.Poseidon2.rate]
  · norm_num [Spec.Poseidon2.rate]

/-- Every exposed lane of the output-binding endpoint is allocated at or
after the PiCCS output-binding offset. The result does not depend on the
27,540 output word contents. -/
theorem finalState_supported_from_offset (interface : Interface)
    (offset : Nat) :
    Formal.StateSupported (finalState interface offset)
      (fun index => offset <= index) := by
  let input := StatementAbsorption.blockExpr (outputWords interface offset)
  have chunksNonempty : Hash.inputChunks input ≠ [] := by
    simpa [input] using framedChunks_nonempty (outputWords interface offset)
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

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding
