import Nightstream.Implementation.Nebula.FPrime.Manifest.BaseSchema

/-! Compile gate for the Nebula V2 base manifest schema. -/

namespace tests.NebulaBaseManifest

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.R1CS

example {widths : FullClaimEnvelope.CompilerWidths}
    (artifact : BaseManifestSchema.Artifact widths) :
    55084 ≤ artifact.programRows.length :=
  artifact.knownRows_lower_bound

example {widths : FullClaimEnvelope.CompilerWidths}
    {artifact : BaseManifestSchema.Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies artifact.other.challengeAuthority assignment :=
  artifact.challengeAuthority_satisfied satisfies

example {widths : FullClaimEnvelope.CompilerWidths}
    (artifact : BaseManifestSchema.Artifact widths) :
    artifact.partRows .challengeAuthority ≠ [] :=
  artifact.other.challengeAuthorityNonempty

end tests.NebulaBaseManifest
