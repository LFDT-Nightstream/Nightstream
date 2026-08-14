import Nightstream.Assurance.Nebula.FingerprintSecurity
import Nightstream.Assurance.Nebula.PiRlcSamplerSecurity
import Nightstream.Assurance.Nebula.SeededSetupSecurity

/-!
Contract: complete conditional soundness budget for the fixed V2 lifetime.

Assurance tier: quantitative cryptographic-assumption boundary.

Owns exact rational bounds for the SuperNeo field term, coordinate fork,
two-repetition memory fingerprint, seven-role seeded setup, and every other
soundness event that the final release reduction must name. It also records the
full-field PiRLC honest-abort bound, but does not add that completeness loss to
the soundness total. It proves that the stated soundness requirements sum to
less than `2^-96`.

Does not prove Poseidon2 security, Fiat--Shamir programming, Module-SIS,
ChaCha8 pseudorandomness, common-witness extraction, or compact-backend
security. These are explicit inputs with required bounds. The sampler-abort
term is the proved ideal public-coin completeness expression. Sampler
distribution and native/circuit agreement remain a separate soundness input.
Deterministic parser or refinement defects are not probability terms and
cannot be supplied to this budget.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Assurance.Nebula.SecurityBudget

open Nightstream.Assurance.Nebula.FingerprintSecurity
open Nightstream.Assurance.Nebula.PiRlcSamplerSecurity
open Nightstream.Assurance.Nebula.SeededSetupSecurity
open Nightstream.Protocol.Nebula.Fingerprint

def folds : Nat := 69632
def superNeoFieldNumerator : Nat := 10824
def coordinateForkNumerator : Nat := 16
def coordinateAlphabetSize : Nat := 5
def coordinateCount : Nat := 54

def fieldBound : ℚ :=
  (folds * superNeoFieldNumerator : ℚ) /
    (goldilocksModulus : ℚ) ^ 2

def coordinateForkBound : ℚ :=
  (folds * coordinateForkNumerator : ℚ) /
    (coordinateAlphabetSize : ℚ) ^ coordinateCount

def fingerprintBound : ℚ := planningFingerprintBound
def samplerAbortBound : ℚ := lifetimeAbortBound

def target96 : ℚ := dyadic 96
def requirement128 : ℚ := dyadic 128
def setupRequirement127 : ℚ := dyadic 127

theorem fieldBound_le_98 : fieldBound ≤ dyadic 98 := by
  norm_num [fieldBound, folds, superNeoFieldNumerator, dyadic,
    goldilocksModulus]

theorem coordinateForkBound_le_105 :
    coordinateForkBound ≤ dyadic 105 := by
  norm_num [coordinateForkBound, folds, coordinateForkNumerator,
    coordinateAlphabetSize, coordinateCount, dyadic]

theorem fingerprintBound_le_186 :
    fingerprintBound ≤ dyadic 186 := by
  norm_num [fingerprintBound, planningFingerprintBound, planningLoss,
    maxSegmentFactors, Nightstream.Protocol.Nebula.scannedCells,
    Nightstream.Protocol.Nebula.romCells,
    Nightstream.Protocol.Nebula.ramCells,
    goldilocksModulus, dyadic]

theorem samplerAbortBound_le_166 :
    samplerAbortBound ≤ dyadic 166 :=
  lifetimeAbortBound_le_166

/-- Remaining computational events after deterministic refinement. Each is a
separate game or backend assumption. No field may contain a final execution
or verifier-soundness conclusion. -/
structure AdditionalBounds where
  poseidonAndTranscript : ℚ
  fiatShamirProgramming : ℚ
  piRlcSamplerDistribution : ℚ
  nifsCommonWitness : ℚ
  compactTerminal : ℚ
  poseidonNonnegative : 0 ≤ poseidonAndTranscript
  fiatShamirNonnegative : 0 ≤ fiatShamirProgramming
  samplerDistributionNonnegative : 0 ≤ piRlcSamplerDistribution
  commonWitnessNonnegative : 0 ≤ nifsCommonWitness
  terminalNonnegative : 0 ≤ compactTerminal
  poseidonBound : poseidonAndTranscript ≤ requirement128
  fiatShamirBound : fiatShamirProgramming ≤ requirement128
  samplerDistributionBound : piRlcSamplerDistribution ≤ requirement128
  commonWitnessBound : nifsCommonWitness ≤ requirement128
  terminalBound : compactTerminal ≤ requirement128

def AdditionalBounds.total (bounds : AdditionalBounds) : ℚ :=
  bounds.poseidonAndTranscript + bounds.fiatShamirProgramming +
    bounds.piRlcSamplerDistribution + bounds.nifsCommonWitness +
      bounds.compactTerminal

theorem AdditionalBounds.total_le_five_128 (bounds : AdditionalBounds) :
    bounds.total ≤ 5 * requirement128 := by
  unfold AdditionalBounds.total
  linarith [bounds.poseidonBound, bounds.fiatShamirBound,
    bounds.samplerDistributionBound, bounds.commonWitnessBound,
    bounds.terminalBound]

/-- Complete soundness advantage after all deterministic bridges have been
proved. `setupAdvantage` is the seven-role ChaCha8/Module-SIS hybrid sum. -/
def total
    (setupAdvantage : ℚ) (additional : AdditionalBounds) : ℚ :=
  fieldBound + coordinateForkBound + fingerprintBound + setupAdvantage +
    additional.total

/-- Exact arithmetic release gate. The small 98-bit field margin is enough
only if the five assumed computational terms each meet 128 bits and the
seven-role setup sum meets 127 bits. Honest sampler abort is not a false-accept
event and is not part of this theorem. -/
theorem total_lt_target96
    {setupAdvantage : ℚ}
    (setupNonnegative : 0 ≤ setupAdvantage)
    (setupBound : setupAdvantage < setupRequirement127)
    (additional : AdditionalBounds) :
    0 ≤ total setupAdvantage additional ∧
      total setupAdvantage additional < target96 := by
  constructor
  · unfold total
    have fieldNonnegative : 0 ≤ fieldBound := by
      unfold fieldBound
      positivity
    have forkNonnegative : 0 ≤ coordinateForkBound := by
      unfold coordinateForkBound
      positivity
    have fingerprintNonnegative : 0 ≤ fingerprintBound := by
      unfold fingerprintBound planningFingerprintBound
      positivity
    have additionalNonnegative : 0 ≤ additional.total := by
      unfold AdditionalBounds.total
      linarith [additional.poseidonNonnegative,
        additional.fiatShamirNonnegative,
        additional.samplerDistributionNonnegative,
        additional.commonWitnessNonnegative,
        additional.terminalNonnegative]
    linarith
  · have additionalBound : additional.total ≤ 5 * dyadic 128 := by
      simpa [requirement128] using additional.total_le_five_128
    have setupBound' : setupAdvantage < dyadic 127 := by
      simpa [setupRequirement127] using setupBound
    have componentBound :
        total setupAdvantage additional ≤
          dyadic 98 + dyadic 105 + dyadic 186 + setupAdvantage +
            additional.total := by
      unfold total
      linarith [fieldBound_le_98, coordinateForkBound_le_105,
        fingerprintBound_le_186]
    calc
      total setupAdvantage additional ≤
          dyadic 98 + dyadic 105 + dyadic 186 + setupAdvantage +
            additional.total := componentBound
      _ <
          dyadic 98 + dyadic 105 + dyadic 186 + dyadic 127 +
            5 * dyadic 128 := by
        linarith
      _ < target96 := by
        norm_num [dyadic, target96]

end Nightstream.Assurance.Nebula.SecurityBudget
