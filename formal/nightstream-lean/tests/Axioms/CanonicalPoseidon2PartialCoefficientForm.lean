import Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientForm
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalPoseidon2PartialCoefficientForm

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientForm.lcEval_coefficientForm' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2PartialCoefficientForm.lcEval_coefficientForm

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientForm.lcEval_addConstant_coefficientForm' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2PartialCoefficientForm.lcEval_addConstant_coefficientForm

end NightstreamTests.Axioms.CanonicalPoseidon2PartialCoefficientForm
