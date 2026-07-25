import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalEncodingRealization
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.HonestAssignment
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepSoundness
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepPhysicalCompleteness
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalTerminalSoundness
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalTerminalPhysicalCompleteness

/-!
Focused obligation-10 surface for the paper-authoritative fixed-one verifier.

Assurance tier: model-proved.  The checked realization constructs exact Step
and Terminal certificates from the selected typed recipes.  It establishes
source/receipt conservation, unique physical ownership, receipt-derived
four-way cost, exact selected local cost, and minimum inside the stated finite
rewrite class.  It does not claim that production Rust or a generated R1CS
artifact realizes the selected program; that is obligation 11.
-/

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

#check CanonicalEncodingRealization.step
#check CanonicalEncodingRealization.terminal
#check CanonicalEncodingRealization.stepObligation10
#check CanonicalEncodingRealization.terminalObligation10
#check CanonicalEncoding.Step.obligation10_of_certificate
#check CanonicalEncoding.Terminal.obligation10_of_certificate
#check CanonicalEncoding.Step.exactCost
#check CanonicalEncoding.Terminal.exactCost
#check CanonicalEncoding.Step.minimum
#check CanonicalEncoding.Terminal.minimum
#check SourceOwners.stepProgramOwnersExact
#check SourceOwners.terminalProgramOwnersExact
#check PrimitivePlan.activeSound
#check PrimitivePlan.activeComplete
#check PrimitivePlan.inactiveComplete
#check HonestAssignment.exists_encodes
#check CanonicalStepSoundness.physicalSound
#check CanonicalStepSoundness.physicalSoundAligned
#check CanonicalStepCompleteness.physicalComplete
#check CanonicalTerminalSoundness.physicalSound
#check CanonicalTerminalCompleteness.physicalComplete
