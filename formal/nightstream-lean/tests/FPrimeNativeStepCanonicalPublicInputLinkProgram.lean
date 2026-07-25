import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLinkProgramRefinement

/-!
Focused elaboration boundary for the Rust-emitted native public-link program.
-/

namespace NightstreamTests.FPrimeNativeStepCanonicalPublicInputLinkProgram

open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep
open CanonicalPlainCarrierLink
open CanonicalPublicInputLinkProgram
open CanonicalPublicInputLinkProgramRefinement

#check generated_plain_eq_canonical
#check generated_plain_cost
#check generated_run_eq_sourceCheck
#check generated_run_reduces_to_logicalPaperLink

example :
    cost generatedPlain = 273 :=
  generated_plain_cost

end NightstreamTests.FPrimeNativeStepCanonicalPublicInputLinkProgram
