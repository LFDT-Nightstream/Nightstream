import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayDigestDomain.Checkpoint2
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptModel

/-!
Contract: independent application-domain model for the terminal Nebula gamma
transcript.

Owns the exact application-label bytes, the 15 setup words, the third
permutation input, and the expected eight-lane state at cursor three. It does
not own generated columns, finalizer rows, gamma challenges, or lifecycle
closure.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaDomainModel

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptModel

def gammaDomainBytes : List Nat :=
  asciiBytes "neo.fold.clean/nebula/gamma/v3"

def domainInitialState : State :=
  appendMessage emptyState transcriptApplicationDomain gammaDomainBytes

def gammaDomainBlock3 : List Nat :=
  [13426, 30, 30521782141150574, 31069335676202596]

def gammaDomainRemainder : List Nat :=
  [27422324158721583, 13336446519830319, 13174]

def gammaInitialStateValues : List Nat :=
  [27422324158721583, 13336446519830319, 13174,
    17411973590883579087, 6939038333896971149,
    3171679524884682263, 2890321166649729893,
    13044081322747540714]

def expectedDomainState : State :=
  stateFromValues gammaInitialStateValues ⟨3, by decide⟩

def checkpoint3InputValues : List Nat :=
  gammaDomainBlock3 ++ checkpoint2Values.drop 4

def checkpoint3InputState : State :=
  stateFromValues checkpoint3InputValues ⟨4, by decide⟩

theorem gamma_domain_framing_exact :
    [1] ++ packedBytesWithLen transcriptApplicationDomain ++
        packedBytesWithLen gammaDomainBytes =
      domainBlock1 ++ domainBlock2 ++ gammaDomainBlock3 ++
        gammaDomainRemainder := by
  rfl

theorem domain_block1_full :
    (absorbWords emptyState domainBlock1).absorbed.val = rate := by
  rfl

theorem domain_block2_full :
    (absorbWords (checkpointState checkpoint1Values)
      domainBlock2).absorbed.val = rate := by
  rfl

theorem gamma_domain_block3_full :
    (absorbWords (checkpointState checkpoint2Values)
      gammaDomainBlock3).absorbed.val = rate := by
  rfl

theorem checkpoint3_input_exact :
    absorbWords (checkpointState checkpoint2Values) gammaDomainBlock3 =
      checkpoint3InputState := by
  apply stateView_injective
  rfl

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaDomainModel
