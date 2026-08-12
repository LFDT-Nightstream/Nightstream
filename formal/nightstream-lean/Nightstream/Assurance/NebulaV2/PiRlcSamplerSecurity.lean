import Nightstream.Assurance.NebulaV2.SeededSetupSecurity
import Nightstream.Implementation.NebulaV2.ProductPiRlcFirstAcceptedBatchRows
import Nightstream.Protocol.NebulaV2.Lifecycle

/-!
Contract: exact ideal public-coin abort term for the V2 full-field PiRLC
sampler.

The generated V2 relation selects 810 coefficients per fold. Each coefficient
uses three independently framed Goldilocks candidates and rejects only the
canonical value `q - 1`. Under independent uniform candidates, one coordinate
aborts with probability `1 / q^3`. This module proves the exact lifetime union
expression and its 166-bit floor.

This is arithmetic for the ideal public-coin experiment. It does not prove
Poseidon2 or Fiat--Shamir uniformity, independence, or adaptive programming.
Those remain separate transcript-game obligations.

Assurance tier: quantitative public-coin model.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Assurance.NebulaV2.PiRlcSamplerSecurity

open Nightstream.Assurance.NebulaV2.SeededSetupSecurity
open Nightstream.Protocol.NebulaV2.Fingerprint

/-- Maximum number of NIFS folds in the selected factor-one V2 lifetime. -/
def lifetimeFolds : Nat :=
  Nightstream.Protocol.NebulaV2.Lifecycle.totalClaims
    Nightstream.Protocol.NebulaV2.Lifecycle.maximumSegments

/-- Number of independently selected PiRLC coefficient positions per fold. -/
def coefficientsPerFold : Nat :=
  Nightstream.Implementation.NebulaV2.ProductPiRlcFirstAcceptedBatchRows.coordinateCount

/-- Number of full-field attempts per coefficient position. -/
def attemptsPerCoefficient : Nat :=
  Nightstream.Implementation.NebulaV2.ProductPiRlcTranscriptRows.attemptCount

theorem exact_schedule :
    lifetimeFolds = 69632 ∧
      coefficientsPerFold = 810 ∧ attemptsPerCoefficient = 3 := by
  exact
    ⟨Nightstream.Protocol.NebulaV2.Lifecycle.maximum_claim_count,
      Nightstream.Implementation.NebulaV2.ProductPiRlcFirstAcceptedBatchRows.coordinateCount_eq,
      Nightstream.Implementation.NebulaV2.ProductPiRlcTranscriptRows.attemptCount_eq⟩

/-- Conservative lifetime union bound for exhaustion of all three attempts at
one coefficient position. -/
def lifetimeAbortBound : ℚ :=
  (lifetimeFolds * coefficientsPerFold : ℚ) /
    (goldilocksModulus : ℚ) ^ attemptsPerCoefficient

theorem lifetimeAbortBound_nonnegative : 0 ≤ lifetimeAbortBound := by
  unfold lifetimeAbortBound
  rw [exact_schedule.1, exact_schedule.2.1, exact_schedule.2.2]
  norm_num [goldilocksModulus]

/-- The exact V2 full-field sampler loss has at least 166 bits of security. -/
theorem lifetimeAbortBound_le_166 :
    lifetimeAbortBound ≤ dyadic 166 := by
  unfold lifetimeAbortBound
  rw [exact_schedule.1, exact_schedule.2.1, exact_schedule.2.2]
  norm_num [goldilocksModulus, dyadic]

/-- The same expression is larger than `2^-167`; its integer security floor
is exactly 166 bits. -/
theorem dyadic_167_lt_lifetimeAbortBound :
    dyadic 167 < lifetimeAbortBound := by
  unfold lifetimeAbortBound
  rw [exact_schedule.1, exact_schedule.2.1, exact_schedule.2.2]
  norm_num [goldilocksModulus, dyadic]

end Nightstream.Assurance.NebulaV2.PiRlcSamplerSecurity
