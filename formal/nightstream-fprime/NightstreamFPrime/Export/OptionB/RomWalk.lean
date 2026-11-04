import NightstreamFPrime.Export.OptionB.IvcWalk
import NightstreamFPrime.Lifecycle.OptionB.RandomOracle.Fidelity

/-! The random-oracle events of `Lifecycle.OptionB.RandomOracle.RomModel` in the walk's names.
On a deployed oracle (`Fidelity.Deployed`), the random-oracle success event of an attempt is the
walk's `IvcWalk.WalkSucceeds` (`succeeds_iff_walkSucceeds`, from `Fidelity.succeeds_iff`). The
random-oracle extraction event is the walk's `IvcWalk.WalkExtracted` on every oracle
(`extracted_iff_walkExtracted`). So `Assumption1.romAssumption1` bounds the walk's Assumption 1
events in the random-oracle model, with the oracle drawn uniformly; at one deployed oracle the
events match, but a single oracle has no probability. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.OptionB.RomWalk

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.OptionB.RandomOracle
open NightstreamFPrime.Export.OptionB.IvcWalk (WalkAttempt WalkSucceeds WalkExtracted)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- On a deployed oracle, the random-oracle success event is the walk's success event. -/
theorem succeeds_iff_walkSucceeds {oracle : RomModel.Oracle logicalWidth publicFits}
    (deployed : Fidelity.Deployed oracle) (attempt : WalkAttempt logicalWidth publicFits) :
    RomModel.Succeeds ajtai (oracle, attempt) ↔ WalkSucceeds ajtai attempt :=
  Fidelity.succeeds_iff ajtai deployed attempt

/-- The random-oracle extraction event is the walk's extraction event. -/
theorem extracted_iff_walkExtracted (oracle : RomModel.Oracle logicalWidth publicFits)
    (attempt : WalkAttempt logicalWidth publicFits)
    (answer : IvcWalk.Answer logicalWidth publicFits) :
    RomModel.Extracted ajtai (oracle, attempt) answer ↔ WalkExtracted ajtai attempt answer :=
  Iff.rfl

end NightstreamFPrime.Export.OptionB.RomWalk
