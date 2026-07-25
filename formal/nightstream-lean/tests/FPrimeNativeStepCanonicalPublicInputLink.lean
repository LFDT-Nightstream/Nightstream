import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLink

/-!
Focused regression for the typed canonical fresh-public link.
-/

namespace Nightstream.Tests.FPrimeNativeStepCanonicalPublicInputLink

open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLink

#check check_eq_true_iff
#check equalityFactorization

example (digest : Nightstream.Implementation.Encoding.FPrime.Digest) :
    check digest
      (Nightstream.Implementation.Encoding.FPrime.encodePublicInput digest) =
        true :=
  (check_eq_true_iff digest _).2 rfl

end Nightstream.Tests.FPrimeNativeStepCanonicalPublicInputLink
