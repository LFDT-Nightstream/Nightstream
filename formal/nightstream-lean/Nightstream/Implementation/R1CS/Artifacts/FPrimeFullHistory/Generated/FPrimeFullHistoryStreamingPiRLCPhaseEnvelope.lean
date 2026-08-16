import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCPhaseEnvelopeSchema

/-! Generated file: compact exact geometry for both Rust-emitted PiRLC
carry-phase semantic envelopes. The Rust generator exhaustively checks
every represented row before it emits this data.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact

def artifactSha256 : String := "cb194404b349350f72333220a4ac4a4252e4512d642cb4a87a879e5d6ef40a80"

def phaseConstantValues : List Nat := [57, 30521782141150574, 31069335676202596, 27422324158721583, 30796712690673199, 27414614995316581, 29396737889036653, 30792317818729313, 33266151269363297, 49, 2169]

def evenArm : RawArm :=
  { sourceIdentity := "rust:pi-rlc-family-even/body-v3", sourceRowsSha256 := "2ef4f3217310c361be90d53c37e852f9ea362786aeb4e7cd212cf56ea8e4cfce",
    bodyRows := 1232857, bodyColumns := 1233086,
    phaseRowStart := 558380, phaseRowEnd := 1221351, phaseColumnStart := 558608, phaseColumnEnd := 1221579,
    beforeLocalSourceColumns := [558598, 558599, 558600, 558601], afterLocalSourceColumns := [416983, 416984, 416985, 416986],
    beforeLocalAliasColumns := [558608, 558609, 558610, 558611], afterLocalAliasColumns := [560781, 560782, 560783, 560784],
    payloadStartColumn := 558612, beforeHashConstantStartColumn := 560785,
    afterHashConstantStartColumn := 891182,
    beforeSemanticDigestColumns := [891174, 891175, 891176, 891177], afterSemanticDigestColumns := [1221571, 1221572, 1221573, 1221574],
    beforeXOutSemanticColumns := [891174, 891175, 891176, 891177], afterXOutSemanticColumns := [1221571, 1221572, 1221573, 1221574] }

def oddArm : RawArm :=
  { sourceIdentity := "rust:pi-rlc-family-odd/body-v3", sourceRowsSha256 := "45612a50dd5521e239f48594315aa86ed28ff53df81a3b73e1bd825d5b3c1f50",
    bodyRows := 1234057, bodyColumns := 1234286,
    phaseRowStart := 559580, phaseRowEnd := 1222551, phaseColumnStart := 559808, phaseColumnEnd := 1222779,
    beforeLocalSourceColumns := [559798, 559799, 559800, 559801], afterLocalSourceColumns := [418183, 418184, 418185, 418186],
    beforeLocalAliasColumns := [559808, 559809, 559810, 559811], afterLocalAliasColumns := [561981, 561982, 561983, 561984],
    payloadStartColumn := 559812, beforeHashConstantStartColumn := 561985,
    afterHashConstantStartColumn := 892382,
    beforeSemanticDigestColumns := [892374, 892375, 892376, 892377], afterSemanticDigestColumns := [1222771, 1222772, 1222773, 1222774],
    beforeXOutSemanticColumns := [892374, 892375, 892376, 892377], afterXOutSemanticColumns := [1222771, 1222772, 1222773, 1222774] }

def rawArtifact : RawArtifact :=
  { schemaVersion := 1, profileId := "nebula-f-prime-streaming-pi-rlc-phase-envelope-v1",
    constantValues := phaseConstantValues, even := evenArm, odd := oddArm }

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope
