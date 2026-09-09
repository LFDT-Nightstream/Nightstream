import NightstreamFPrime.Spec.AjtaiSetupV1.IndexEncoding
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup

/-! Instantiates the indexed setup's encoding bounds at the selected production
key dimensions. No key coefficients or circuit rows are materialized. -/

namespace NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.AjtaiSetupV1

theorem setup_index_bounds (row : Fin verifierRows) (block : Fin messageColumns)
    (lane : Fin ringDegree) :
    row.val < ChaCha20.wordModulus ∧
      block.val < ChaCha20.wordModulus ^ 2 ∧
      lane.val < ChaCha20.wordModulus := by
  have rowLimit : verifierRows < ChaCha20.wordModulus := by
    rw [verifierRows_eq]
    decide
  have blockLimit : messageColumns < ChaCha20.wordModulus ^ 2 := by
    rw [messageColumns_eq]
    decide
  have laneLimit : ringDegree < ChaCha20.wordModulus := by decide
  exact ⟨Nat.lt_trans row.isLt rowLimit, Nat.lt_trans block.isLt blockLimit,
    Nat.lt_trans lane.isLt laneLimit⟩

/-- No two distinct production key coordinates use the same initial state. -/
theorem production_index_injective
    (leftRow rightRow : Fin verifierRows)
    (leftBlock rightBlock : Fin messageColumns)
    (leftLane rightLane : Fin ringDegree)
    (same : ChaCha20.initialState productionSeedBytes
        leftRow.val leftBlock.val leftLane.val =
      ChaCha20.initialState productionSeedBytes
        rightRow.val rightBlock.val rightLane.val) :
    leftRow = rightRow ∧ leftBlock = rightBlock ∧ leftLane = rightLane := by
  have leftBounds := setup_index_bounds leftRow leftBlock leftLane
  have rightBounds := setup_index_bounds rightRow rightBlock rightLane
  have exactIndices := ChaCha20.initialState_index_injective
    productionSeedBytes productionSeedBytes
    leftRow.val leftBlock.val leftLane.val rightRow.val rightBlock.val rightLane.val
    leftBounds.1 rightBounds.1 leftBounds.2.1 rightBounds.2.1
    leftBounds.2.2 rightBounds.2.2 same
  exact ⟨Fin.ext exactIndices.1, Fin.ext exactIndices.2.1, Fin.ext exactIndices.2.2⟩

end NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
