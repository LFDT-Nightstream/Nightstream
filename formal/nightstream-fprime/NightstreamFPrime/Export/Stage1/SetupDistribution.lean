import NightstreamFPrime.Spec.AjtaiSetupV1.ReductionBias
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup

/-! The ideal modular-reduction error budget at the exact selected key size.
The product-distribution interpretation requires independent uniform 256-bit
inputs. This file makes no pseudorandomness claim for the public setup seed. -/

namespace NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup

open NightstreamFPrime.Spec

def setupCoefficientCount : Nat := verifierRows * messageColumns * ringDegree

theorem setupCoefficientCount_eq : setupCoefficientCount = 5566248072 := by
  rw [setupCoefficientCount, verifierRows_eq, messageColumns_eq]
  rfl

/-- Sum of the single-coordinate error bounds across this exact key. -/
def idealReductionErrorBudget : ℚ :=
  setupCoefficientCount * ((2 ^ 256 % goldilocksModulus : Nat) : ℚ) / 2 ^ 256

theorem idealReductionErrorBudget_eq :
    idealReductionErrorBudget = (5566248072 : ℚ) * 4294967295 / 2 ^ 256 := by
  rw [idealReductionErrorBudget, setupCoefficientCount_eq,
    AjtaiSetupV1.ReductionBias.wide_remainder_eq]
  rfl

/-- The exponent 191 is derived from this exact coefficient count and the
checked remainder; it is not a target for cryptographic security. -/
theorem idealReductionErrorBudget_lt : idealReductionErrorBudget < 1 / 2 ^ 191 := by
  rw [idealReductionErrorBudget_eq]
  norm_num

end NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
