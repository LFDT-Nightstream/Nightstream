import Nightstream.Implementation.NebulaV2.Memory.Segment.Soundness

/-! Focused gates for non-circular row-derived segment memory soundness. -/

set_option autoImplicit false

namespace tests.NebulaV2SegmentMemorySoundness

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt
open Nightstream.Implementation.NebulaV2.RecursiveManifestSchema
open Nightstream.Implementation.NebulaV2.SegmentCheckedRows
open Nightstream.Implementation.NebulaV2.SegmentMemoryCoverage
open Nightstream.Implementation.NebulaV2.SegmentMemorySoundness
open Nightstream.Implementation.NebulaV2.SegmentSnapshotCoverage
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.IdealFingerprint
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.SuperNeo.Concrete

variable {widths : CompilerWidths}
variable {artifact : Artifact widths} {selected : SelectedVerifier widths}

/-- This gate states the exact central result. Its only non-row premises are
the canonical opening conditions. -/
theorem checked_segment_executes_or_has_concrete_fingerprint_failure
    {active : ActiveCarry Digest.Value (Challenges K) (State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations (.closed closed))
    (startsAtZero : active.stepIndex.val = 0)
    (openingProducts :
      active.products = MemoryCarryCodec.oneProductsK) :
    Memory.Executes
        (CheckedRun.snapshot run startsAtZero .initialSnapshot).tuples
        active.globalTimestamp
        (accesses invocations)
        (CheckedRun.snapshot run startsAtZero .finalSnapshot).tuples
        closed.globalTimestamp ∨
      EvaluationFailure (fingerprintCheck run startsAtZero) :=
  executesOrEvaluationFailure run startsAtZero openingProducts

-- The production-facing theorem obtains both opening conditions from one
-- canonical global segment.
#check globallyOpenedExecutesOrEvaluationFailure

end tests.NebulaV2SegmentMemorySoundness
