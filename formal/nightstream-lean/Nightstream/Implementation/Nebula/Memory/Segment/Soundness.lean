import Mathlib.Algebra.Field.TransferInstance
import Nightstream.Implementation.Nebula.Memory.Segment.Coverage
import Nightstream.Protocol.Nebula.GlobalFPrime
import Nightstream.Protocol.Nebula.IdealFingerprint

/-!
Contract: row-derived memory soundness for one complete V2 segment.

Assurance tier: implementation-to-protocol bridge.

Owns the reduction from one exact active-to-closed checked-row run to either
the independent sequential memory execution or the concrete event that a
nonzero two-variable fingerprint polynomial evaluates to zero at both V2
challenge pairs.

The premises are row satisfaction, exact carry chaining, canonical step zero,
and the canonical all-one opening product state. No premise states multiset
balance, fingerprint acceptance, record coverage, or memory execution.

Does not own challenge unpredictability, Fiat--Shamir probability, NIFS row
soundness, root-chain binding, application transition rows, or terminal
verification.

Emits constraints: no. It gives aggregate meaning to existing checked rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.SegmentMemorySoundness

open Nightstream.Implementation.Nebula.ConcreteField
open Nightstream.Implementation.Nebula.FullClaimEnvelope
open Nightstream.Implementation.Nebula.FullClaimNifsReceipt
open Nightstream.Implementation.Nebula.MemoryClaimProductUpdate
open Nightstream.Implementation.Nebula.RecursiveManifestSchema
open Nightstream.Implementation.Nebula.SegmentCheckedRows
open Nightstream.Implementation.Nebula.SegmentMemoryCoverage
open Nightstream.Implementation.Nebula.SegmentSnapshotCoverage
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.IdealFingerprint
open Nightstream.Protocol.Nebula.ProductState
open Nightstream.SuperNeo.Concrete

variable {widths : CompilerWidths}
variable {artifact : Artifact widths} {selected : SelectedVerifier widths}

/-- The exact ideal fingerprint check represented by a complete checked-row
segment. All record bounds are derived from snapshot rows, operation rows, and
the active carry's row-derived well-formedness. -/
def fingerprintCheck
    {active : ActiveCarry Digest.Value (Challenges K) (State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations (.closed closed))
    (startsAtZero : active.stepIndex.val = 0) :
    IdealFingerprint.Check encode
      (CheckedRun.snapshot run startsAtZero .initialSnapshot)
      (SegmentMemoryCoverage.accesses invocations)
      (CheckedRun.snapshot run startsAtZero .finalSnapshot) where
  bounds := by
    have wellFormed := run.activeWellFormed
    have startInRange : active.segmentStartTimestamp < timestampLimit :=
      (wellFormed.2.2.2.2.1.trans wellFormed.2.2.2.2.2).trans_lt
        wellFormed.2.2.2.1
    exact RecordBounds.ofValidAt
      (CheckedRun.snapshotValidAt run startsAtZero .initialSnapshot)
      startInRange
      (CheckedRun.snapshotValidAt run startsAtZero .finalSnapshot)
      wellFormed.2.2.2.1
      (SegmentMemoryCoverage.orderedActiveToClosed run)
  challenges := mapChallenges active.challenge

/-- Closing product rows and exact record coverage force both concrete
fingerprint repetitions to accept. This theorem does not assume fingerprint
acceptance. -/
theorem fingerprintAccepted
    {active : ActiveCarry Digest.Value (Challenges K) (State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations (.closed closed))
    (startsAtZero : active.stepIndex.val = 0)
    (openingProducts :
      active.products = MemoryCarryCodec.oneProductsK) :
    (fingerprintCheck run startsAtZero).Accepts := by
  have accumulated := run.accumulatedFromOneBalanced openingProducts
  have coverage := SegmentMemoryCoverage.covers run startsAtZero
  have expectedBalanced :
      ProductState.Balanced
        (ProductState.expected (fingerprintCheck run startsAtZero)) := by
    rw [← ProductState.accumulate_one_eq_expected
      (fingerprintCheck run startsAtZero) coverage]
    simpa [fingerprintCheck] using accumulated
  exact (ProductState.accepts_iff_expected_balanced
    (fingerprintCheck run startsAtZero)).mpr expectedBalanced

/-- A complete checked-row segment has exact multiset balance unless the
concrete nonzero fingerprint polynomial passes both evaluations. -/
theorem balanceOrEvaluationFailure
    {active : ActiveCarry Digest.Value (Challenges K) (State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations (.closed closed))
    (startsAtZero : active.stepIndex.val = 0)
    (openingProducts :
      active.products = MemoryCarryCodec.oneProductsK) :
    Memory.Balanced
        (CheckedRun.snapshot run startsAtZero .initialSnapshot).tuples
        (SegmentMemoryCoverage.accesses invocations)
        (CheckedRun.snapshot run startsAtZero .finalSnapshot).tuples ∨
      IdealFingerprint.EvaluationFailure
        (fingerprintCheck run startsAtZero) := by
  exact IdealFingerprint.balance_or_evaluationFailure
    ConcreteField.encode_injective_below_goldilocks
    (fingerprintCheck run startsAtZero)
    (fingerprintAccepted run startsAtZero openingProducts)

/-- Central non-circular segment theorem. Satisfying rows reconstruct the
exact application-ordered memory execution, except for the named concrete
fingerprint evaluation event. -/
theorem executesOrEvaluationFailure
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
        (SegmentMemoryCoverage.accesses invocations)
        (CheckedRun.snapshot run startsAtZero .finalSnapshot).tuples
        closed.globalTimestamp ∨
      IdealFingerprint.EvaluationFailure
        (fingerprintCheck run startsAtZero) := by
  rcases balanceOrEvaluationFailure run startsAtZero openingProducts with
    balance | failure
  · exact Or.inl (Memory.balanced_implies_executes
      (SegmentMemoryCoverage.orderedActiveToClosed run) balance)
  · exact Or.inr failure

noncomputable section

/-- The field structure transported through the proved coefficient
equivalence uses the same concrete multiplicative identity as the row layer. -/
local instance concreteKField : Field K :=
  ConcreteField.superNeoEquiv.field

private theorem fieldOneProducts_eq_concreteOne :
    (ProductState.one : State K) = MemoryCarryCodec.oneProductsK := by
  funext repetition
  apply ProductState.Four.ext <;>
    apply ConcreteField.superNeoEquiv.injective <;>
    simp only [ProductState.one, MemoryCarryCodec.oneProductsK,
      ConcreteField.superNeoEquiv_one]
  all_goals
    change ConcreteField.superNeoEquiv
      (ConcreteField.superNeoEquiv.symm (1 : ChallengeField)) = 1
    exact ConcreteField.superNeoEquiv.apply_symm_apply 1

/-- The abstract global F-prime segment supplies both canonical opening facts
needed by the row theorem. Callers cannot select step zero or all-one products
independently. -/
theorem globallyOpenedExecutesOrEvaluationFailure
    {before : ClosedCarry Digest.Value}
    (global : GlobalFPrime.SegmentRun
      (protocolSchema widths (PackedProof selected)) Digest.Value
      (VerifyClaim selected) before)
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active global.active) invocations
      (.closed global.after)) :
    Memory.Executes
        (CheckedRun.snapshot run global.startsAtStepZero
          .initialSnapshot).tuples
        global.active.globalTimestamp
        (SegmentMemoryCoverage.accesses invocations)
        (CheckedRun.snapshot run global.startsAtStepZero
          .finalSnapshot).tuples
        global.after.globalTimestamp ∨
      IdealFingerprint.EvaluationFailure
        (fingerprintCheck run global.startsAtStepZero) := by
  have openingProducts :
      global.active.products = MemoryCarryCodec.oneProductsK := by
    calc
      global.active.products = ProductState.one :=
        global.startsFromExactClosedCarry.2.2.2.1
      _ = MemoryCarryCodec.oneProductsK :=
        fieldOneProducts_eq_concreteOne
  exact executesOrEvaluationFailure run global.startsAtStepZero openingProducts

end

end Nightstream.Implementation.Nebula.SegmentMemorySoundness
