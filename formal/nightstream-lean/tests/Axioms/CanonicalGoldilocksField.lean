import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalGoldilocksField

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-- info: 'Nightstream.Implementation.R1CS.Canonical.GoldilocksField.powMod_cast' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms powMod_cast

/-- info: 'Nightstream.Implementation.R1CS.Canonical.GoldilocksField.fermat_zmod' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms fermat_zmod

/-- info: 'Nightstream.Implementation.R1CS.Canonical.GoldilocksField.prime_divisor_order' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms prime_divisor_order

/-- info: 'Nightstream.Implementation.R1CS.Canonical.GoldilocksField.goldilocks_natPrime' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms goldilocks_natPrime

/-- info: 'Nightstream.Implementation.R1CS.Canonical.GoldilocksField.goldilocks_euclidPrime' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms goldilocks_euclidPrime

/-- info: 'Nightstream.Implementation.R1CS.Canonical.GoldilocksField.goldilocksInverseValue_correct' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms goldilocksInverseValue_correct

/-- info: 'Nightstream.Implementation.R1CS.Canonical.GoldilocksField.goldilocksFieldInverse' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms goldilocksFieldInverse

end NightstreamTests.Axioms.CanonicalGoldilocksField
