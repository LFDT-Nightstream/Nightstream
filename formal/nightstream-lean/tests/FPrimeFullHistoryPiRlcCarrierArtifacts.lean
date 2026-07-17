import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.RecursiveCarrierArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.TerminalCarrierArtifact

/-!
Regression checks for the exact recursive and terminal PiRLC carrier split.

Assurance tier: artifact-checked regression surface.

Owns: compile-time checks for the generated public/delayed trace boundary.

Does not own: carrier semantics, generated data, row emission, or soundness.

Authority boundary: theorem names imported here must derive the public prefix
and delayed suffix from the generated 31-trace profile; this test supplies no
replacement witness or digest.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_rlc.recursive.carrier.census` | public tree equals generated `take 29`; delayed suffix has length two | reflexive or shifted public census |
| `nifs.pi_rlc.recursive.carrier.wiring` | equation inputs, outputs, and point are the direct batch-carrier fields | accidental dependence on codec, widths, or census |
| `nifs.pi_rlc.terminal.carrier.census` | public tree equals generated `take 29`; delayed suffix has length two | reflexive or shifted public census |
| `nifs.pi_rlc.terminal.carrier.wiring` | equation inputs, outputs, and point are the direct batch-carrier fields | accidental dependence on codec, widths, or census |

Emits constraints: no.
-/

namespace tests.FPrimeFullHistoryPiRlcCarrierArtifacts

namespace Recursive

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveCarrierArtifact

#check parentClaim
#check columns
#check publicTrace_census
#check publicTrace_positions
#check publicTrace_count
#check delayedTrace_count
#check trace_partition
#check inputWidth
#check outputWidth
#check carrierArtifact
#check parentArtifact
#check challengeWiringArtifact
#check equationWiringArtifact

end Recursive

namespace Terminal

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalCarrierArtifact

#check parentClaim
#check columns
#check publicTrace_census
#check publicTrace_positions
#check publicTrace_count
#check delayedTrace_count
#check trace_partition
#check inputWidth
#check outputWidth
#check carrierArtifact
#check parentArtifact
#check challengeWiringArtifact
#check equationWiringArtifact

end Terminal

end tests.FPrimeFullHistoryPiRlcCarrierArtifacts
