import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierLink

/-!
Focused regression for the typed 270-coordinate plain fresh-public carrier.
-/

namespace Nightstream.Tests.FPrimeNativeStepCanonicalPlainCarrierLink

open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierLink

#check paddingWidth_eq_thirteen
#check check_eq_true_iff
#check equalityFactorization
#check rawCheck_eq_true_iff
#check rawCheck_reduces_to_typedCarrier
#check check_reduces_to_logicalPaperLink

example (digest : Nightstream.Implementation.Encoding.FPrime.Digest) :
    check digest (encodeClaim digest) = true :=
  (check_eq_true_iff digest _).2 rfl

example (digest : Nightstream.Implementation.Encoding.FPrime.Digest) :
    rawCheck digest (encodeRawClaim digest) = true :=
  (rawCheck_eq_true_iff digest _).2 rfl

end Nightstream.Tests.FPrimeNativeStepCanonicalPlainCarrierLink
