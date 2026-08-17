import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCPhaseEnvelopeSchema

/-! Generated file: compact exact geometry for both Rust-emitted PiRLC
carry-phase semantic envelopes. The Rust generator exhaustively checks
every represented row before it emits this data.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact

def artifactSha256 : String := "7a67dec986d7cb629c2499582788f44a264485b8837c181b5a65b3d4410b3876"

def phaseConstantValues : List Nat := [57, 30521782141150574, 31069335676202596, 27422324158721583, 30796712690673199, 27414614995316581, 29396737889036653, 30792317818729313, 33266151269363297, 49, 2169]

def evenArm : RawArm :=
  { sourceIdentity := "rust:pi-rlc-family-even/body-v3", sourceRowsSha256 := "8d2b2b82c6cc9499da67d1c940e59c06af7ac1d13b8ca9530b3e98ad113e0a62",
    bodyRows := 1300897, bodyColumns := 1301126,
    phaseRowStart := 626420, phaseRowEnd := 1289391, phaseColumnStart := 626648, phaseColumnEnd := 1289619,
    beforeLocalSourceColumns := [626638, 626639, 626640, 626641], afterLocalSourceColumns := [468823, 468824, 468825, 468826],
    beforeLocalAliasColumns := [626648, 626649, 626650, 626651], afterLocalAliasColumns := [628821, 628822, 628823, 628824],
    payloadStartColumn := 626652, beforeHashConstantStartColumn := 628825,
    afterHashConstantStartColumn := 959222,
    beforeSemanticDigestColumns := [959214, 959215, 959216, 959217], afterSemanticDigestColumns := [1289611, 1289612, 1289613, 1289614],
    beforeXOutSemanticColumns := [959214, 959215, 959216, 959217], afterXOutSemanticColumns := [1289611, 1289612, 1289613, 1289614] }

def oddArm : RawArm :=
  { sourceIdentity := "rust:pi-rlc-family-odd/body-v3", sourceRowsSha256 := "d83076a6fd25819e173b5f5f9aa2f0ceadc355bb2dbb7195624889df99b7aa01",
    bodyRows := 1302097, bodyColumns := 1302326,
    phaseRowStart := 627620, phaseRowEnd := 1290591, phaseColumnStart := 627848, phaseColumnEnd := 1290819,
    beforeLocalSourceColumns := [627838, 627839, 627840, 627841], afterLocalSourceColumns := [470023, 470024, 470025, 470026],
    beforeLocalAliasColumns := [627848, 627849, 627850, 627851], afterLocalAliasColumns := [630021, 630022, 630023, 630024],
    payloadStartColumn := 627852, beforeHashConstantStartColumn := 630025,
    afterHashConstantStartColumn := 960422,
    beforeSemanticDigestColumns := [960414, 960415, 960416, 960417], afterSemanticDigestColumns := [1290811, 1290812, 1290813, 1290814],
    beforeXOutSemanticColumns := [960414, 960415, 960416, 960417], afterXOutSemanticColumns := [1290811, 1290812, 1290813, 1290814] }

def rawArtifact : RawArtifact :=
  { schemaVersion := 1, profileId := "nebula-f-prime-streaming-pi-rlc-phase-envelope-v1",
    constantValues := phaseConstantValues, even := evenArm, odd := oddArm }

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope
