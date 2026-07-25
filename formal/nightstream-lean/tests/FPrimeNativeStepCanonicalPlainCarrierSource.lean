import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSource

/-!
Focused regression for the source-shaped plain public-link refinement.
-/

namespace Nightstream.Tests.FPrimeNativeStepCanonicalPlainCarrierSource

open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierLink
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSource

#check sourceCheck_eq_true_iff
#check sourceCheck_eq_rawCheck
#check sourceCheck_reduces_to_logicalPaperLink
#check sourceBatchCheck_reduces_to_logicalPaperLink

example (digest : Nightstream.Implementation.Encoding.FPrime.Digest) :
    sourceCheck digest (encodeRawClaim digest) = true :=
  (sourceCheck_eq_true_iff digest _).2 rfl

end Nightstream.Tests.FPrimeNativeStepCanonicalPlainCarrierSource
