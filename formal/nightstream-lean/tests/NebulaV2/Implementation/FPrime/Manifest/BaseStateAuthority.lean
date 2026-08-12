import Nightstream.Implementation.NebulaV2.FPrime.Manifest.BaseStateAuthority

/-! Compile gate for row-derived Nebula V2 base state authority. -/

namespace tests.NebulaV2BaseManifestStateAuthority

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS

#check BaseManifestStateAuthority.Call.initialAccepted
#check BaseManifestStateAuthority.Call.outgoingAccepted

example {widths : FullClaimEnvelope.CompilerWidths}
    {artifact : BaseManifestSchema.Artifact widths}
    {assignment : Nat → Nat}
    (call : BaseManifestStateAuthority.Call artifact assignment)
    (satisfies : Satisfies artifact.programRows assignment) :
    InitialMemoryCarryRows.Exact call.initialValue call.initialMemoryRoot :=
  call.initialExact satisfies

end tests.NebulaV2BaseManifestStateAuthority
