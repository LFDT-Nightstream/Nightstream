import NightstreamFPrime.Spec.AjtaiSetupV1.IndexEncoding
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup

/-! Instantiates the indexed setup's encoding bounds at the selected production
key dimensions. No key coefficients or circuit rows are materialized. -/

namespace NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.AjtaiSetupV1

theorem setup_index_bounds (row : Fin verifierRows) (block : Fin messageColumns) :
    row.val < 2 ^ 32 ∧ block.val < 2 ^ 64 := by
  have rowLimit : verifierRows < 2 ^ 32 := by
    rw [verifierRows_eq]
    decide
  have blockLimit : messageColumns < 2 ^ 64 := by
    rw [messageColumns_eq]
    decide
  exact ⟨Nat.lt_trans row.isLt rowLimit, Nat.lt_trans block.isLt blockLimit⟩

/-- No two distinct production key elements use the same SHAKE128 input. -/
theorem production_index_injective
    (leftRow rightRow : Fin verifierRows)
    (leftBlock rightBlock : Fin messageColumns)
    (same : elementInput productionSeedBytes leftRow.val leftBlock.val =
      elementInput productionSeedBytes rightRow.val rightBlock.val) :
    leftRow = rightRow ∧ leftBlock = rightBlock := by
  have leftBounds := setup_index_bounds leftRow leftBlock
  have rightBounds := setup_index_bounds rightRow rightBlock
  have exactIndices := elementInput_injective productionSeed productionSeed
    leftRow.val leftBlock.val rightRow.val rightBlock.val
    leftBounds.1 rightBounds.1 leftBounds.2 rightBounds.2 same
  exact ⟨Fin.ext exactIndices.2.1, Fin.ext exactIndices.2.2⟩

end NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
