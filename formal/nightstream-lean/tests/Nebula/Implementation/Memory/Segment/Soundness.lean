import Nightstream.Implementation.Nebula.Memory.Segment.Soundness

/-! Focused gates for non-circular row-derived segment memory soundness. -/

set_option autoImplicit false

namespace tests.NebulaSegmentMemorySoundness

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.FullClaimEnvelope
open Nightstream.Implementation.Nebula.FullClaimNifsReceipt
open Nightstream.Implementation.Nebula.RecursiveManifestSchema
open Nightstream.Implementation.Nebula.SegmentCheckedRows
open Nightstream.Implementation.Nebula.SegmentMemoryCoverage
open Nightstream.Implementation.Nebula.SegmentMemorySoundness
open Nightstream.Implementation.Nebula.SegmentSnapshotCoverage
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.IdealFingerprint
open Nightstream.Protocol.Nebula.ProductState
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

end tests.NebulaSegmentMemorySoundness
