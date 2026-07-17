import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Interface

/-!
Fail-closed import regression for the physical FE SumCheck interface.

The interface may expose raw verifier carriers and physical degree parameters,
but importing it alone must not expose independent source data or NC semantic
truth.
-/

namespace tests.PiCcsSplitNcVerifierFeInterfaceFirewall

#check Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate

/-- error: Unknown identifier `Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources.Data` -/
#guard_msgs in
#check Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources.Data

/-- error: Unknown identifier `Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.diagonal` -/
#guard_msgs in
#check Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.diagonal

end tests.PiCcsSplitNcVerifierFeInterfaceFirewall
