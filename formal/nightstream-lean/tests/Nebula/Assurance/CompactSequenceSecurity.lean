import Nightstream.Assurance.Nebula.CompactSequenceSecurity

set_option autoImplicit false

namespace Nightstream.Tests.NebulaCompactSequenceSecurity

open Nightstream.Assurance.Nebula.CompactSequenceSecurity
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.CompactChain
open Nightstream.Protocol.Nebula.CompactCommit
open Nightstream.Protocol.Nebula.IdealAcceptance

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

end Nightstream.Tests.NebulaCompactSequenceSecurity
