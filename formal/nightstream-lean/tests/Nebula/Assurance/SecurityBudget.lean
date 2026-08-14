import Nightstream.Assurance.Nebula.SecurityBudget

set_option autoImplicit false

namespace tests.NebulaSecurityBudget

open Nightstream.Assurance.Nebula.SecurityBudget
open Nightstream.Assurance.Nebula.SeededSetupSecurity

def idealAdditional : AdditionalBounds where
  poseidonAndTranscript := 0
  fiatShamirProgramming := 0
  piRlcSamplerDistribution := 0
  nifsCommonWitness := 0
  compactTerminal := 0
  poseidonNonnegative := by norm_num
  fiatShamirNonnegative := by norm_num
  samplerDistributionNonnegative := by norm_num
  commonWitnessNonnegative := by norm_num
  terminalNonnegative := by norm_num
  poseidonBound := by norm_num [requirement128, dyadic]
  fiatShamirBound := by norm_num [requirement128, dyadic]
  samplerDistributionBound := by norm_num [requirement128, dyadic]
  commonWitnessBound := by norm_num [requirement128, dyadic]
  terminalBound := by norm_num [requirement128, dyadic]

theorem exact_terms_fit_when_named_assumptions_hold :
    total 0 idealAdditional < target96 :=
  (total_lt_target96 (by norm_num)
    (by norm_num [setupRequirement127, dyadic] :
      (0 : ℚ) < setupRequirement127)
    idealAdditional).2

/-- A deterministic refinement defect is intentionally not a constructor of
`AdditionalBounds`; it cannot be hidden under a small probability. -/
theorem five_named_additional_terms :
    idealAdditional.total = 0 := by
  norm_num [AdditionalBounds.total, idealAdditional]

/-- Honest sampler abort is a completeness loss. This equality fails if that
loss is inserted into the false-acceptance total. -/
theorem honest_sampler_abort_is_not_in_soundness_total :
    total 0 idealAdditional =
      fieldBound + coordinateForkBound + fingerprintBound := by
  simp [total, AdditionalBounds.total, idealAdditional]

end tests.NebulaSecurityBudget
