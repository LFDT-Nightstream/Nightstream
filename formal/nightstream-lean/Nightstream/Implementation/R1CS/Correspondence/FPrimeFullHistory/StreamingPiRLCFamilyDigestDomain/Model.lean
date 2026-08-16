import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayDigestDomain.Checkpoint3

/-!
Contract: byte framing and bounded checkpoint model for the streaming PiRLC
family-state transcript domain.

Owns the exact application-domain bytes, the changed fourth four-word block,
the two-word remainder, and the expected fourth checkpoint values. The first
three blocks and checkpoints are definitionally shared with the certified
claim-state domain.

Does not own generated family-state columns, public digest words, or lifecycle
integration.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyDigestDomain

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain

/-- Exact bytes of the PiRLC family-state digest application label. -/
def stateDigestDomain : List Nat :=
  [110, 101, 111, 46, 102, 111, 108, 100, 46, 99, 108, 101, 97, 110,
   47, 110, 101, 98, 117, 108, 97, 47, 102, 45, 112, 114, 105, 109,
   101, 47, 115, 116, 114, 101, 97, 109, 105, 110, 103, 45, 112, 105,
   114, 108, 99, 45, 115, 116, 97, 116, 101, 47, 118, 49]

def domainBlock4 : List Nat :=
  [27422324158721583, 30796712690673199,
   27414614995316581, 29678212865747309]

def domainRemainder : List Nat :=
  [27431110772485234, 212436215156]

def domainCertificateIndices : List Nat :=
  List.range 4 ++
    (List.range 4).map (4 + ·) ++
    (List.range 4).map (8 + ·) ++
    (List.range 4).map (12 + ·) ++
    (List.range 2).map (16 + ·)

/-- The PiRLC domain certificate covers exactly 18 framed words. -/
theorem domain_certificate_partition_exact :
    domainBlock1.length = 4 ∧
      domainBlock2.length = 4 ∧
      domainBlock3.length = 4 ∧
      domainBlock4.length = 4 ∧
      domainRemainder.length = 2 ∧
      domainCertificateIndices = List.range 18 ∧
      domainCertificateIndices.Nodup := by
  native_decide

/-- The byte-level PiRLC framing is exactly the bounded checkpoint
partition. -/
theorem domain_framing_words_exact :
    [1] ++ packedBytesWithLen transcriptApplicationDomain ++
        packedBytesWithLen stateDigestDomain =
      domainBlock1 ++ domainBlock2 ++ domainBlock3 ++ domainBlock4 ++
        domainRemainder := by
  native_decide

def checkpoint4Values : List Nat :=
  [1384628267568061624, 9165867117000812810,
   17097887264920317608, 5045320036345004475,
   2778744841876081974, 15963932331085499115,
   9417555079406418274, 11480438335165740302]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyDigestDomain
