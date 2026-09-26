import NightstreamFPrime.Lifecycle.Stage1.Application
import NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1Circuit

/-! The hash-chain application uses the shared application compiler. -/

namespace NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec

/-- The closed verifier-owned application program. -/
def program : Application.Program where
  witnessWordCount := messageWordCount
  step := step
  circuit := circuit
  spec_iff := spec_iff
  assumptions_of_inputsBelow := assumptions_of_inputsBelow
  constraintsSupported := constraintsSupported

@[simp] theorem program_witnessWordCount : program.witnessWordCount = 4 := by
  rfl

@[simp] theorem program_step : program.step = step := by
  rfl

theorem program_localLength
    (interface : Application.Interface program.witnessWordCount)
    (offset : Nat) :
    localLength (Circuit.ops (program.circuit interface).main offset) =
      7696 := by
  exact circuit_localLength interface offset

theorem program_rowCount
    (interface : Application.Interface program.witnessWordCount)
    (offset : Nat) :
    (flatConstraints (Circuit.ops (program.circuit interface).main offset)).length =
      7700 := by
  exact circuit_rowCount interface offset

end NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1
