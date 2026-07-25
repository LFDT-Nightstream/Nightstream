import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter

/-!
Focused elaboration boundary for the fixed-one native-to-lowering semantic
adapter.
-/

namespace NightstreamTests.FPrimeNativeStepFixedOneLoweringAdapter

open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter

#check parameters
#check CallAlignment.step
#check CallAlignment.hashPrior
#check CallAlignment.hashNext
#check CallAlignment.nifsVerify
#check CallAlignment.runningCheck
#check CallAlignment.freshCheck
#check stepAccepts_iff_directHolds
#check terminalAccepts_iff_transition

end NightstreamTests.FPrimeNativeStepFixedOneLoweringAdapter
