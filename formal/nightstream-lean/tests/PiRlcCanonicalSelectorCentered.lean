import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelector

/-!
Regression for the affine centering performed by the canonical `Pi_RLC`
selector output row.

The raw sampler alphabet is `0 .. 4`, while the quotient relation consumes
the centered Goldilocks images `-2 .. 2`.  These mutations isolate the two
most easily confused values: raw zero must become `-2`, and raw two must
become zero.
-/

set_option autoImplicit false

namespace NightstreamTests.PiRlcCanonicalSelectorCentered

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelector
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

def rawZero : ProductionAlphabet.Coefficient := ⟨0, by decide⟩
def rawTwo : ProductionAlphabet.Coefficient := ⟨2, by decide⟩

theorem rawZero_embeds_as_minusTwo :
    (Phi81StrongSet.embedCoefficient rawZero).val = goldilocksP - 2 := by
  simpa [rawZero, goldilocksP] using embedCoefficient_val_eq_shift rawZero

theorem rawTwo_embeds_as_zero :
    (Phi81StrongSet.embedCoefficient rawTwo).val = 0 := by
  simpa [rawTwo, goldilocksP] using embedCoefficient_val_eq_shift rawTwo

theorem omitting_centering_changes_rawZero :
    rawZero.val ≠ (Phi81StrongSet.embedCoefficient rawZero).val := by
  rw [rawZero_embeds_as_minusTwo]
  decide

theorem omitting_centering_changes_rawTwo :
    rawTwo.val ≠ (Phi81StrongSet.embedCoefficient rawTwo).val := by
  rw [rawTwo_embeds_as_zero]
  decide

end NightstreamTests.PiRlcCanonicalSelectorCentered
