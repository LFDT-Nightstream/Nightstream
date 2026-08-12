import Nightstream.Assurance.NebulaV2.CompactSequenceSecurity

set_option autoImplicit false

namespace Nightstream.Tests.NebulaV2CompactSequenceSecurity

open Nightstream.Assurance.NebulaV2.CompactSequenceSecurity
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.CompactChain
open Nightstream.Protocol.NebulaV2.CompactCommit
open Nightstream.Protocol.NebulaV2.IdealAcceptance

example
    {ChallengeField Plan Seed Digest : Type}
    [Field ChallengeField]
    {config :
      Config ChallengeField Profile.Identity Plan CommitmentEncoding Digest}
    (hash : HashInput Plan Digest → Digest)
    (key : Key Plan Seed)
    (chainRootExact : config.chainRoot = chainRoot hash key)
    (failure : Failure config) :
    ReleaseFailure config hash key :=
  classify_failure hash key chainRootExact failure

end Nightstream.Tests.NebulaV2CompactSequenceSecurity
