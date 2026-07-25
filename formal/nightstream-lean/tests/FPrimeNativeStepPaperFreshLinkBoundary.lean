import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.PaperFreshLinkBoundary

/-!
Focused regression for the exact fresh-link factorization missing from the
current native-step-to-paper interface.
-/

namespace Nightstream.Tests.FPrimeNativeStepPaperFreshLinkBoundary

open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.PaperFreshLinkBoundary

#check overlappingLink_not_equalityFactorized
#check currentInterface_admits_nonFactorizingFreshLink

example :
    Not (EqualityFactorization counterSemantics.freshLink id id) :=
  currentInterface_admits_nonFactorizingFreshLink id id

end Nightstream.Tests.FPrimeNativeStepPaperFreshLinkBoundary
