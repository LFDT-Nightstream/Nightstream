import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.NoZeroDivisors
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteRootCounting

/-!
Concrete algebra boundary for production Split-NC finite-root arguments.

Assurance tier: model-level.

Owns: exact transport from the Goldilocks modulus Euclid divisor property and
the production `u² = 7` irreducibility premise to the no-zero-divisor law used
by the repository's finite-root theorem.

Does not own: arithmetic certificates for either visible number-theory
premise, challenge sampling, probability, Fiat--Shamir, Rust/R1CS, or rows.

Emits constraints: no.

| Boundary | Owned equation | Excluded boundary |
|---|---|---|
| Goldilocks base | Euclid premise implies base no-zero-divisors | closed arithmetic certificate |
| production extension | base law plus seven-nonresidue implies extension no-zero-divisors | sampling and probability |
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- The exact Goldilocks Euclid boundary implies the concrete base-field
no-zero-divisor law already used by production polynomial semantics. -/
theorem goldilocksBaseNoZeroDivisors
    (euclid : NormRange.GoldilocksModulusEuclid) :
    NormRange.BaseFieldNoZeroDivisors :=
  NormRange.baseFieldNoZeroDivisors_of_modulusEuclid euclid

/-- The exact active Goldilocks and `u² = 7` boundaries imply precisely the
production extension no-zero-divisor law consumed by finite root counting.
No second field carrier is introduced. -/
theorem productionExtensionNoZeroDivisors
    (euclid : NormRange.GoldilocksModulusEuclid)
    (sevenNonresidue : ConcreteCarrier.SevenProjectiveNonresidue) :
    FiniteRootCounting.NoZeroDivisors ConcreteCarrier.extensionOps :=
  ConcreteCarrier.extensionNoZeroDivisors_of_base_and_seven
    (goldilocksBaseNoZeroDivisors euclid) sevenNonresidue

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary
