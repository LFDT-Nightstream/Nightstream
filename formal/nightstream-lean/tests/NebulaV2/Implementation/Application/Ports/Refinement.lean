import Nightstream.Implementation.NebulaV2.Application.Ports.Refinement
import Nightstream.Implementation.NebulaV2.Memory.Segment.CheckedRows

/-! Focused gates for exact 3-by-21 application-port refinement. -/

set_option autoImplicit false

namespace tests.NebulaV2ApplicationPortRefinement

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ApplicationPortRefinement
open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt
open Nightstream.Implementation.NebulaV2.RecursiveManifestSchema
open Nightstream.Implementation.NebulaV2.SegmentCheckedRows
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.Ports

variable {widths : CompilerWidths}
variable {artifact : Artifact widths} {selected : SelectedVerifier widths}

theorem invocation_rows_preserve_physical_access_order
    (invocation : Invocation artifact selected)
    (kinds : ApplicationRowIndex → NormalizedRowKind) :
    (invocation.applicationRows kinds).flatMap NormalizedRow.accesses =
      invocation.applicationAccesses :=
  invocation.applicationRows_flatMap_accesses kinds

theorem invocation_active_count_is_row_derived
    (invocation : Invocation artifact selected) :
    invocation.applicationAccesses.length =
      invocation.call.claim.memory.activeAccessCount :=
  invocation.applicationAccesses_length

theorem invocation_accesses_are_strictly_ordered
    (invocation : Invocation artifact selected) :
    Ordered invocation.call.claim.memory.timestampIn
      invocation.applicationAccesses
      invocation.call.claim.memory.timestampOut :=
  invocation.applicationAccessesOrdered

theorem invocation_reads_are_exact_application_reads
    (invocation : Invocation artifact selected) :
    invocation.chunk.reads =
      (Memory.readTuples invocation.applicationAccesses : Multiset MemTuple) :=
  invocation.chunk_reads_eq

theorem invocation_writes_are_exact_application_writes
    (invocation : Invocation artifact selected) :
    invocation.chunk.writes =
      (Memory.writeTuples invocation.applicationAccesses : Multiset MemTuple) :=
  invocation.chunk_writes_eq

end tests.NebulaV2ApplicationPortRefinement
