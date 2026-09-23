import Mathlib.Algebra.BigOperators.Fin
import NightstreamFPrime.Spec.Algebra
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet

/-!
Owns the whole-vector PiRLC scalar sampler map. Four transcript field values
are read as one base-`p` integer `X < p^4`. The scalar is the 54-digit
base-five expansion of `X mod 5^54`; each coefficient is its digit minus two.
The map is total: it has no rejection, retry or failure event.

This module fixes the deterministic map only. `Law` owns its comparison with
uniform scalars. No law is assigned to Poseidon2.
-/

namespace NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler
open ProductionAlphabet ProductionStrongSet

/-- Transcript field values consumed by one scalar. -/
def drawWidth : Nat := 4

/-- The ordered transcript field values for one scalar. -/
abbrev Draw := Fin drawWidth → F

/-- Number of draws, `p^4`. -/
def drawCount : Nat := goldilocksModulus ^ drawWidth

/-- Number of scalars, `5^54`. -/
def scalarCount : Nat := alphabetSize ^ coefficientCount

/-- Full residue blocks of `5^54` below `p^4`. -/
def quotient : Nat := drawCount / scalarCount

/-- Size of the incomplete final residue block. -/
def remainder : Nat := drawCount % scalarCount

theorem scalarCount_pos : 0 < scalarCount := by decide

theorem drawCount_pos : 0 < drawCount := by decide

theorem drawCount_eq : drawCount = quotient * scalarCount + remainder := by decide

theorem remainder_lt : remainder < scalarCount := by decide

/-- Base-`p` reading, `h₀ + p h₁ + p² h₂ + p³ h₃`. -/
def drawIndex : Draw ≃ Fin drawCount := finFunctionFinEquiv

/-- Base-five reading of a scalar, coefficient zero least significant. -/
def scalarIndex : Scalar ≃ Fin scalarCount := finFunctionFinEquiv

/-- Reduction of the draw integer modulo `5^54`. -/
def reduce (value : Fin drawCount) : Fin scalarCount :=
  ⟨value.val % scalarCount, Nat.mod_lt _ scalarCount_pos⟩

/-- The sampled scalar. -/
def sample (draw : Draw) : Scalar :=
  scalarIndex.symm (reduce (drawIndex draw))

theorem drawIndex_val (draw : Draw) :
    (drawIndex draw).val = ∑ index, (draw index).val * goldilocksModulus ^ index.val :=
  finFunctionFinEquiv_apply draw

/-- Coefficient `j` is digit `j` of `X mod 5^54`. -/
theorem sample_val (draw : Draw) (position : Fin coefficientCount) :
    (sample draw position).val =
      (drawIndex draw).val % scalarCount / alphabetSize ^ position.val % alphabetSize :=
  rfl

theorem sample_eq_iff (draw : Draw) (scalar : Scalar) :
    sample draw = scalar ↔ (drawIndex draw).val % scalarCount = (scalarIndex scalar).val := by
  rw [sample, Equiv.symm_apply_eq, Fin.ext_iff]
  rfl

end NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler
