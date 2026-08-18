import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayDigestDomain.Model
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.ExtractedReference

/-!
Contract: byte framing and fixed Poseidon2 checkpoints for the Prelude
replay-state transcript domain.

Owns the exact application-domain bytes, five four-word framing blocks, four
checkpoint outputs, and the cursor-four state produced by transcript setup.
It does not own generated columns, transcript call rows, or lifecycle use.

Assurance tier: model-level until the checkpoint leaves are composed.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeDigestDomain

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain

/-- Exact bytes of the Prelude replay-state digest application label. -/
def stateDigestDomain : List Nat :=
  [110, 101, 111, 46, 102, 111, 108, 100, 46, 99, 108, 101, 97, 110,
   47, 110, 101, 98, 117, 108, 97, 47, 102, 45, 112, 114, 105, 109,
   101, 47, 115, 116, 114, 101, 97, 109, 105, 110, 103, 45, 112, 114,
   105, 111, 114, 45, 115, 116, 97, 116, 101, 45, 114, 101, 112, 108,
   97, 121, 45, 115, 116, 97, 116, 101, 47, 118, 49]

def domainBlock1 : List Nat :=
  [1, 44, 27428916078536046, 32774695491433326]

def domainBlock2 : List Nat :=
  [32492151232362031, 12721823848622437,
   31362922327076711, 12728458466782051]

def domainBlock3 : List Nat :=
  [13426, 67, 30521782141150574, 31069335676202596]

def domainBlock4 : List Nat :=
  [27422324158721583, 30796712690673199,
   27414614995316581, 32211487656143213]

def domainBlock5 : List Nat :=
  [27431110773469033, 30522878494336372,
   32758250074896737, 829828965]

def checkpoint1Values : List Nat :=
  [15335109097073140235, 17619563798142362813,
   11645649210628966215, 4436364367798357556,
   14285079556494616407, 10815961559651698320,
   18260705026218357875, 8424582804285107233]

def checkpoint2Values : List Nat :=
  [12824850162859434436, 16884249588232806831,
   1238414030862021266, 16194180760988878864,
   2250958222046521952, 15440103732085447511,
   6547395164335706312, 10272340994444577429]

def checkpoint3Values : List Nat :=
  [8217990886473854477, 15285686427137095851,
   1718373255794792806, 11998897845846645487,
   4304991313695166870, 4447228957773253884,
   17155955104328901778, 2704665992612756743]

def checkpoint4Values : List Nat :=
  [13454411644540670803, 3076890293650919384,
   16306137016845071370, 1053363896557079446,
   1988541141149579427, 4859373221894732330,
   9937262314844071878, 8401668388730343368]

def stateFromValues
    (values : List Nat) (absorbed : Fin (rate + 1)) : State where
  lanes := fun lane => fieldValue (values.getD lane.val 0)
  absorbed

def checkpointState (values : List Nat) : State :=
  stateFromValues values ⟨0, by decide⟩

def checkpoint1InputValues : List Nat :=
  domainBlock1 ++ [0, 0, 0, 0]

def checkpoint1InputState : State :=
  stateFromValues checkpoint1InputValues ⟨4, by decide⟩

theorem domain_framing_words_exact :
    [1] ++ packedBytesWithLen transcriptApplicationDomain ++
        packedBytesWithLen stateDigestDomain =
      domainBlock1 ++ domainBlock2 ++ domainBlock3 ++ domainBlock4 ++
        domainBlock5 := by
  native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeDigestDomain
