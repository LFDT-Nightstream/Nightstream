import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement

/-!
Focused elaboration boundary for the Rust-emitted XOut preimage programs.
-/

namespace NightstreamTests.FPrimeNativeStepStateXOutProgram

open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep
open StateXOutProgram
open StateXOutProgramRefinement

#check generated_eq_canonical
#check generated_starts_with_exact_domain
#check generated_execute_eq_encodeStateXOutPreimage
#check generated_publicLink_accepts_computedXOut

example :
    cost (GeneratedProgram.select false false) = 23 :=
  generated_statelessPlain_cost

example :
    cost (GeneratedProgram.select true true) = 32 :=
  generated_statefulNebula_cost

end NightstreamTests.FPrimeNativeStepStateXOutProgram
