import Mathlib.Algebra.BigOperators.Fin
import NightstreamFPrime.Spec.Algebra
import NightstreamFPrime.Spec.Phi81StrongSet

/-!
Owns the whole-vector PiRLC scalar sampler map. Four transcript field values
are read as one base-`p` integer `X < p^4`. The scalar is the 54-digit
base-five expansion of `X mod 5^54`; each coefficient is its digit minus two.
The map is total: it has no rejection, retry or failure event.

This module fixes the deterministic map only. `Law` owns its comparison with
uniform scalars. No law is assigned to Poseidon2.
-/

namespace NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler
open NightstreamFPrime.Spec.Phi81StrongSet

def drawWidth : Nat := 4

abbrev Draw := Fin drawWidth → F

def drawCount : Nat := goldilocksModulus ^ drawWidth

def scalarCount : Nat := alphabetSize ^ coefficientCount

def quotient : Nat := drawCount / scalarCount

def remainder : Nat := drawCount % scalarCount

theorem scalarCount_pos : 0 < scalarCount := by decide

theorem drawCount_pos : 0 < drawCount := by decide

theorem drawCount_eq : drawCount = quotient * scalarCount + remainder := by decide

theorem remainder_lt : remainder < scalarCount := by decide

def drawIndex : Draw ≃ Fin drawCount := finFunctionFinEquiv

def scalarIndex : Scalar ≃ Fin scalarCount := finFunctionFinEquiv

def reduce (value : Fin drawCount) : Fin scalarCount :=
  ⟨value.val % scalarCount, Nat.mod_lt _ scalarCount_pos⟩

def sample (draw : Draw) : Scalar :=
  scalarIndex.symm (reduce (drawIndex draw))

theorem drawIndex_val (draw : Draw) :
    (drawIndex draw).val = ∑ index, (draw index).val * goldilocksModulus ^ index.val :=
  finFunctionFinEquiv_apply draw

theorem sample_val (draw : Draw) (position : Fin coefficientCount) :
    (sample draw position).val =
      (drawIndex draw).val % scalarCount / alphabetSize ^ position.val % alphabetSize :=
  rfl

theorem sample_eq_iff (draw : Draw) (scalar : Scalar) :
    sample draw = scalar ↔ (drawIndex draw).val % scalarCount = (scalarIndex scalar).val := by
  rw [sample, Equiv.symm_apply_eq, Fin.ext_iff]
  rfl

end NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler
