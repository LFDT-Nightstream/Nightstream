import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.FinalRound.Artifact

/-!
Focused regressions for the exact final terminal-NC artifact owners.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.round.14.message.length` | embedded length row pins ten fields | lost mixed-piece ownership |
| `nifs.pi_ccs.nc_sumcheck.round.14.message.permute` | two exact message calls are independently accepted | wrong call order or columns |
| `nifs.pi_ccs.nc_sumcheck.round.14.challenge.marker` | squeeze marker is an accepted verifier constant | prover-controlled marker |
| `nifs.pi_ccs.nc_sumcheck.round.14.challenge.permute` | exact final squeeze call is independently accepted | wrong terminal call |
-/

open Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FinalRound

#check Artifact.firstMessagePiece_eq
#check Artifact.secondMessagePiece_eq
#check Artifact.finalSqueezePiece_eq
#check Artifact.finalSqueezeOutputBase_eq
#check Artifact.messageLengthPins_included
#check Artifact.squeezeMarkerPins_included
#check Artifact.firstMessageCallAccepted
#check Artifact.secondMessageCallAccepted
#check Artifact.finalSqueezeCallAccepted
#check Artifact.Facts
#check Artifact.facts
