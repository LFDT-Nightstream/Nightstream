import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.RootCountingSecurity

/-!
Focused theorem-surface and boundary checks for finite paper-joint mixing
soundness.

The positive theorem uses the verifier-owned alpha/gamma product support.
Production Fiat--Shamir sampling remains separate.
-/

set_option autoImplicit false

namespace tests.PiCcsPaperJointMixingSoundness

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.MixingSoundness
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.RootCountingSecurity
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

#check MultilinearRootCounting.zeros_count_le
#check CoefficientRootCounting.roots_count_le_degree
#check alphaGammaZero_probability_le
#check verifierAlphaGamma_marginal
#check mixingRoot_probability_le
#check mixingRootProbabilityContract_of_rootCounting
#check fixedFirstBadBound_of_rootCounting
#check extraction_after_first_success_of_rootCounting
#check extraction_after_success_gate_of_rootCounting

/-- The strict coefficient-level nonzero premise rejects the zero polynomial;
root counting cannot silently instantiate it. -/
example
    {Field : Type}
    (ops : InterpolationOps Field)
    (coefficients : List Field)
    (allZero : CoefficientRootCounting.AllZero ops coefficients) :
    ¬ Not (CoefficientRootCounting.AllZero ops coefficients) := by
  exact fun nonzero => nonzero allZero

/-- An explicit support cannot be empty. -/
example {Challenge : Type} (support : Support Challenge) :
    support.values ≠ [] :=
  support.nonempty

/-- An explicit support cannot contain a duplicate list. -/
example {Challenge : Type} (value : Challenge) :
    ¬ ∃ support : Support Challenge, support.values = [value, value] := by
  rintro ⟨support, duplicate⟩
  have nodup := support.nodup
  rw [duplicate] at nodup
  simpa using nodup

/- The verifier marginal theorem is deliberately joint in alpha and gamma:
it uses only the paper's joint alpha/gamma strategy. -/
#check VerifierCoins.gamma
#check VerifierCoins.support

end tests.PiCcsPaperJointMixingSoundness
