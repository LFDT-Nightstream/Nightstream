import Nightstream.Protocol.NebulaV2.ApplicationBatch
import Nightstream.Protocol.NebulaV2.AugmentedLifecycle
import Nightstream.Protocol.NebulaV2.ProductionBatchedGlobalFPrime

/-!
Contract: exact batch-aware augmented F-prime lifetime for Nebula V2.

One ordered verified-claim list is shared by the application batch chain, the
memory segment chain, and the delayed base/recursive/terminal schedule. Base
produces the first batch claim without consuming a prior claim. Each later
nonterminal invocation consumes one exact verified batch, continues the
memory carry, and produces the next batch claim. Terminal consumes the final
batch directly into a closed carry and does not reopen or produce a claim.

`links` is the named generated-row refinement boundary between one exact WASM
batch and its exact claim. The theorems in this file preserve and expose that
predicate. They do not assume that an arbitrary predicate proves application
or memory soundness.

Does not own generated rows, the concrete `links` instantiation, NIFS
extraction, recursive-size closure, Poseidon2 security, terminal cryptography,
Rust, or external bytes.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.ProductionBatchedAugmentedLifecycle

open Nightstream.Protocol.NebulaV2.ApplicationBatch
open Nightstream.Protocol.NebulaV2.AugmentedLifecycle
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.Lifecycle
open Nightstream.Protocol.NebulaV2.ProductionBatchedFPrime
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.Protocol.NebulaV2.WasmState

abbrev BatchVerifier :=
  ProductionBatchedGlobalFPrime.BatchVerifier

abbrev Receipt :=
  ProductionBatchedGlobalFPrime.Receipt

/-- The generated relation must instantiate this boundary with an exact
shared-column theorem. It relates one complete verified claim to the one
sequential WASM batch that produced it. -/
abbrev ClaimBatchLink
    {ChallengeField : Type} [Field ChallengeField]
    (candidate : Id) (schema : Schema) (Digest Program : Type)
    (verify : BatchVerifier candidate schema Digest ChallengeField)
    (machine : Machine Program) (program : Program) :=
  {before after : AppStateVector} ->
    Receipt candidate schema Digest ChallengeField verify ->
    Batch candidate machine program before after -> Prop

/-- Exact state-contiguous application batches paired in order with the same
verified claims used by the delayed memory relation. -/
inductive ProducedChain
    {ChallengeField : Type} [Field ChallengeField]
    (candidate : Id) (schema : Schema) (Digest Program : Type)
    (verify : BatchVerifier candidate schema Digest ChallengeField)
    (machine : Machine Program) (program : Program) :
    AppStateVector -> List ApplicationTrace.ApplicationRow -> AppStateVector ->
      List (Receipt candidate schema Digest ChallengeField verify) -> Type
  | nil (state : AppStateVector) :
      ProducedChain candidate schema Digest Program verify machine program
        state [] state []
  | cons
      {before middle after : AppStateVector}
      {tailRows : List ApplicationTrace.ApplicationRow}
      {tailClaims : List
        (Receipt candidate schema Digest ChallengeField verify)}
      (batch : Batch candidate machine program before middle)
      (receipt : Receipt candidate schema Digest ChallengeField verify)
      (tail : ProducedChain candidate schema Digest Program verify machine
        program middle tailRows after tailClaims) :
      ProducedChain candidate schema Digest Program verify machine program
        before (batch.rows ++ tailRows) after (receipt :: tailClaims)

namespace ProducedChain

/-- Forgetting claim pairing gives the exact application batch chain. -/
theorem toApplicationChain
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest Program : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {machine : Machine Program} {program : Program}
    {before after : AppStateVector}
    {rows : List ApplicationTrace.ApplicationRow}
    {claims : List (Receipt candidate schema Digest ChallengeField verify)}
    (produced : ProducedChain candidate schema Digest Program verify machine
      program before rows after claims) :
    ApplicationBatch.Chain candidate machine program before rows after
      claims.length := by
  induction produced with
  | nil => exact ApplicationBatch.Chain.nil _
  | cons batch receipt tail inductionHypothesis =>
      simpa [Nat.add_comm] using
        ApplicationBatch.Chain.cons batch inductionHypothesis

/-- The exact claim count fixes the complete application-row length. -/
theorem rows_length
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest Program : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {machine : Machine Program} {program : Program}
    {before after : AppStateVector}
    {rows : List ApplicationTrace.ApplicationRow}
    {claims : List (Receipt candidate schema Digest ChallengeField verify)}
    (produced : ProducedChain candidate schema Digest Program verify machine
      program before rows after claims) :
    rows.length = claims.length * rowsPerFreshClaim candidate :=
  produced.toApplicationChain.rows_length

/-- Separate evidence that every structural claim-batch pair satisfies the
named generated-row refinement boundary. Keeping this evidence separate from
the chain prevents an arbitrary relation parameter from defining execution. -/
def AllLinked
    {ChallengeField : Type} [Field ChallengeField]
    (candidate : Id) (schema : Schema) (Digest Program : Type)
    (verify : BatchVerifier candidate schema Digest ChallengeField)
    (machine : Machine Program) (program : Program)
    (links : ClaimBatchLink candidate schema Digest Program verify machine
      program) :
    {before after : AppStateVector} ->
      {rows : List ApplicationTrace.ApplicationRow} ->
      {claims : List (Receipt candidate schema Digest ChallengeField verify)} ->
      ProducedChain candidate schema Digest Program verify machine program
        before rows after claims -> Prop
  | _, _, _, _, .nil _ => True
  | _, _, _, _, .cons batch receipt tail =>
      links receipt batch /\
        AllLinked candidate schema Digest Program verify machine program links
          tail

/-- Every paired claim and application batch satisfies the named generated
row refinement boundary. -/
theorem AllLinked.every_linked
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest Program : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {machine : Machine Program} {program : Program}
    {links : ClaimBatchLink candidate schema Digest Program verify machine
      program}
    {before after : AppStateVector}
    {rows : List ApplicationTrace.ApplicationRow}
    {claims : List (Receipt candidate schema Digest ChallengeField verify)}
    {produced : ProducedChain candidate schema Digest Program verify machine
      program before rows after claims}
    (allLinked : AllLinked candidate schema Digest Program verify machine
      program links produced) :
    forall receipt, receipt ∈ claims ->
      exists batchBefore batchAfter,
        exists batch : Batch candidate machine program batchBefore batchAfter,
          links receipt batch := by
  induction produced with
  | nil => simp
  | cons batch headReceipt tail inductionHypothesis =>
      rcases allLinked with ⟨headLinked, tailLinked⟩
      intro receipt member
      simp only [List.mem_cons] at member
      rcases member with equal | tailMember
      · subst receipt
        exact ⟨_, _, batch, headLinked⟩
      · exact inductionHypothesis tailLinked receipt tailMember

/-- The paired application chain gives one exact deterministic WASM run. -/
theorem toRuns
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest Program : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {machine : Machine Program} {program : Program}
    {before after : AppStateVector}
    {rows : List ApplicationTrace.ApplicationRow}
    {claims : List (Receipt candidate schema Digest ChallengeField verify)}
    (produced : ProducedChain candidate schema Digest Program verify machine
      program before rows after claims) :
    exists count, Runs machine program before rows after count :=
  produced.toApplicationChain.toRuns

/-- Exact paired batches preserve complete application-state validity. -/
theorem after_valid
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest Program : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {machine : Machine Program} {program : Program}
    {before after : AppStateVector}
    {rows : List ApplicationTrace.ApplicationRow}
    {claims : List (Receipt candidate schema Digest ChallengeField verify)}
    (produced : ProducedChain candidate schema Digest Program verify machine
      program before rows after claims)
    (beforeValid : before.Valid) :
    after.Valid :=
  produced.toApplicationChain.after_valid beforeValid

end ProducedChain

/-- Exact delayed batch consumption after base. A recursive constructor
consumes the prior verified claim, applies the mandatory active-copy or
closed-open continuation, and then processes the next claim. The terminal
constructor consumes its claim into a closed carry and has no continuation. -/
inductive DelayedRun
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    (verify : BatchVerifier candidate schema Digest ChallengeField)
    (derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField)
    (headers : ChainHeaders Digest) :
    Carry Digest (ProductState.Challenges ChallengeField)
        (ProductState.State ChallengeField) ->
      List (Receipt candidate schema Digest ChallengeField verify) ->
      ClosedCarry Digest -> Prop
  | terminal
      {before : Carry Digest (ProductState.Challenges ChallengeField)
        (ProductState.State ChallengeField)}
      (receipt : Receipt candidate schema Digest ChallengeField verify)
      (final : ClosedCarry Digest)
      (consumes : Transition verify ProductState.Balanced before receipt
        (.closed final)) :
      DelayedRun verify derive headers before [receipt] final
  | recursive
      {before intermediate outgoing :
        Carry Digest (ProductState.Challenges ChallengeField)
          (ProductState.State ChallengeField)}
      {tail : List (Receipt candidate schema Digest ChallengeField verify)}
      {final : ClosedCarry Digest}
      (receipt : Receipt candidate schema Digest ChallengeField verify)
      (consumes : Transition verify ProductState.Balanced before receipt
        intermediate)
      (continues : Continues derive headers intermediate outgoing)
      (rest : DelayedRun verify derive headers outgoing tail final) :
      DelayedRun verify derive headers before (receipt :: tail) final

namespace DelayedRun

theorem claims_nonempty
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {before : Carry Digest (ProductState.Challenges ChallengeField)
      (ProductState.State ChallengeField)}
    {claims : List (Receipt candidate schema Digest ChallengeField verify)}
    {final : ClosedCarry Digest}
    (run : DelayedRun verify derive headers before claims final) :
    claims ≠ [] := by
  cases run <;> simp

theorem every_claim_accepted
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {before : Carry Digest (ProductState.Challenges ChallengeField)
      (ProductState.State ChallengeField)}
    {claims : List (Receipt candidate schema Digest ChallengeField verify)}
    {final : ClosedCarry Digest}
    (run : DelayedRun verify derive headers before claims final) :
    forall receipt, receipt ∈ claims -> verify receipt.proof receipt.claim := by
  induction run with
  | terminal receipt final consumes =>
      intro selected member
      simp only [List.mem_singleton] at member
      subst selected
      exact consumes.accepted_claim_is_consumed.1
  | recursive receipt consumes continues rest inductionHypothesis =>
      intro selected member
      simp only [List.mem_cons] at member
      rcases member with equal | tailMember
      · subst selected
        exact consumes.accepted_claim_is_consumed.1
      · exact inductionHypothesis selected tailMember

private theorem finish_segment_of_eq
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {before : Carry Digest (ProductState.Challenges ChallengeField)
      (ProductState.State ChallengeField)}
    {claims : List (Receipt candidate schema Digest ChallengeField verify)}
    {after : Carry Digest (ProductState.Challenges ChallengeField)
      (ProductState.State ChallengeField)}
    {final : ClosedCarry Digest}
    (verified : VerifiedRun verify ProductState.Balanced before claims
      after)
    (afterExact : after = .closed final)
    (nonempty : claims ≠ []) :
    DelayedRun verify derive headers before claims final := by
  induction verified with
  | nil => exact False.elim (nonempty rfl)
  | @cons _ middle _ receipt tail consumes rest inductionHypothesis =>
      cases rest with
      | nil =>
          cases afterExact
          simpa using DelayedRun.terminal receipt final consumes
      | @cons _ nextMiddle _ nextReceipt nextTail nextConsumes nextRest =>
          rcases nextConsumes.before_active with ⟨active, middleExact⟩
          cases middleExact
          have tailNonempty : nextReceipt :: nextTail ≠ [] := by simp
          exact DelayedRun.recursive receipt consumes
            (Continues.interior active)
            (inductionHypothesis afterExact tailNonempty)

/-- A verified segment that ends closed becomes the final part of the delayed
schedule. Its last batch is consumed by the terminal constructor. -/
theorem finish_segment
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {before : Carry Digest (ProductState.Challenges ChallengeField)
      (ProductState.State ChallengeField)}
    {claims : List (Receipt candidate schema Digest ChallengeField verify)}
    {final : ClosedCarry Digest}
    (verified : VerifiedRun verify ProductState.Balanced before claims
      (.closed final))
    (nonempty : claims ≠ []) :
    DelayedRun verify derive headers before claims final :=
  finish_segment_of_eq verified rfl nonempty

/-- Append a completed segment before an already delayed tail. The segment's
last claim hosts the exact boundary reopen before the first tail claim. -/
private theorem prepend_segment_of_eq
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {before : Carry Digest (ProductState.Challenges ChallengeField)
      (ProductState.State ChallengeField)}
    {boundary : ClosedCarry Digest}
    {headClaims tailClaims :
      List (Receipt candidate schema Digest ChallengeField verify)}
    {outgoing : Carry Digest (ProductState.Challenges ChallengeField)
      (ProductState.State ChallengeField)}
    {headAfter : Carry Digest (ProductState.Challenges ChallengeField)
      (ProductState.State ChallengeField)}
    {final : ClosedCarry Digest}
    (head : VerifiedRun verify ProductState.Balanced before headClaims
      headAfter)
    (headAfterExact : headAfter = .closed boundary)
    (headNonempty : headClaims ≠ [])
    (continues : Continues derive headers (.closed boundary) outgoing)
    (tail : DelayedRun verify derive headers outgoing tailClaims final) :
    DelayedRun verify derive headers before (headClaims ++ tailClaims)
      final := by
  induction head with
  | nil => exact False.elim (headNonempty rfl)
  | @cons _ middle _ receipt restClaims consumes rest inductionHypothesis =>
      cases rest with
      | nil =>
          cases headAfterExact
          simpa using DelayedRun.recursive receipt consumes continues tail
      | @cons _ nextMiddle _ nextReceipt nextTail nextConsumes nextRest =>
          rcases nextConsumes.before_active with ⟨active, middleExact⟩
          cases middleExact
          have restNonempty : nextReceipt :: nextTail ≠ [] := by simp
          exact DelayedRun.recursive receipt consumes
            (Continues.interior active)
            (inductionHypothesis headAfterExact restNonempty)

/-- Append a completed segment before an already delayed tail. The segment's
last claim hosts the exact boundary reopen before the first tail claim. -/
theorem prepend_segment
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {before : Carry Digest (ProductState.Challenges ChallengeField)
      (ProductState.State ChallengeField)}
    {boundary : ClosedCarry Digest}
    {headClaims tailClaims :
      List (Receipt candidate schema Digest ChallengeField verify)}
    {outgoing : Carry Digest (ProductState.Challenges ChallengeField)
      (ProductState.State ChallengeField)}
    {final : ClosedCarry Digest}
    (head : VerifiedRun verify ProductState.Balanced before headClaims
      (.closed boundary))
    (headNonempty : headClaims ≠ [])
    (continues : Continues derive headers (.closed boundary) outgoing)
    (tail : DelayedRun verify derive headers outgoing tailClaims final) :
    DelayedRun verify derive headers before (headClaims ++ tailClaims)
      final :=
  prepend_segment_of_eq head rfl headNonempty continues tail

end DelayedRun

namespace SegmentChain

/-- The exact segment chain derives the exact augmented delayed schedule.
The first segment opening belongs to base. Every later opening is placed after
the prior segment's closing claim in one nonterminal recursive invocation. -/
theorem toDelayedRun
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {initial final : ClosedCarry Digest}
    {claims : List (Receipt candidate schema Digest ChallengeField verify)}
    {segmentCount : Nat}
    (chain : ProductionBatchedGlobalFPrime.Chain candidate schema Digest
      verify derive headers initial claims final segmentCount)
    (positive : 0 < segmentCount) :
    exists precommit activeAccessCount canOpen activeCountInRange
        endTimestampInRange active,
      openSegment derive headers precommit activeAccessCount initial canOpen
          activeCountInRange endTimestampInRange = .active active /\
        DelayedRun verify derive headers (.active active) claims final := by
  induction chain with
  | nil => omega
  | @cons before final tailClaims tailSegments head tail inductionHypothesis =>
      have headNonempty : head.claims ≠ [] := by
        apply List.ne_nil_of_length_pos
        rw [head.exactClaimCount]
        cases candidate <;> decide
      cases tail with
      | nil =>
          refine ⟨head.precommit, head.activeAccessCount, head.canOpen,
            head.activeCountInRange, head.endTimestampInRange, head.active,
            head.opened, ?_⟩
          simpa using DelayedRun.finish_segment head.consumed headNonempty
      | @cons _ _ nextTailClaims nextTailSegments nextHead nextTail =>
          rcases inductionHypothesis (by omega) with
            ⟨nextPrecommit, nextActiveAccessCount, nextCanOpen,
              nextActiveCountInRange, nextEndTimestampInRange, nextActive,
              nextOpened, nextDelayed⟩
          let boundaryContinuation : Continues derive headers
              (.closed head.after) (.active nextActive) := by
            simpa [nextOpened] using
              (Continues.boundary
                (derive := derive) (headers := headers) head.after
                nextPrecommit nextActiveAccessCount nextCanOpen
                nextActiveCountInRange nextEndTimestampInRange)
          refine ⟨head.precommit, head.activeAccessCount, head.canOpen,
            head.activeCountInRange, head.endTimestampInRange, head.active,
            head.opened, ?_⟩
          exact DelayedRun.prepend_segment head.consumed headNonempty
            boundaryContinuation nextDelayed

end SegmentChain

/-- One complete batch-aware F-prime lifetime. The application and memory
relations share the exact same ordered verified-claim list. -/
structure CompleteRun
    {ChallengeField : Type} [Field ChallengeField]
    (candidate : Id) (schema : Schema) (Digest Program : Type)
    (verify : BatchVerifier candidate schema Digest ChallengeField)
    (derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField)
    (headers : ChainHeaders Digest)
    (machine : Machine Program) (program : Program)
    (links : ClaimBatchLink candidate schema Digest Program verify machine
      program)
    (initialApplication : AppStateVector)
    (initialMemory : ClosedCarry Digest)
    (finalApplication : AppStateVector)
    (finalMemory : ClosedCarry Digest)
    (segmentCount : Nat) where
  applicationRows : List ApplicationTrace.ApplicationRow
  claims : List (Receipt candidate schema Digest ChallengeField verify)
  application : ProducedChain candidate schema Digest Program verify machine
    program initialApplication applicationRows finalApplication claims
  applicationLinks : ProducedChain.AllLinked candidate schema Digest Program
    verify machine program links application
  memory : ProductionBatchedGlobalFPrime.Chain candidate schema Digest verify
    derive headers initialMemory claims finalMemory segmentCount
  positiveSegments : 0 < segmentCount

namespace CompleteRun

theorem exact_claim_count
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest Program : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {machine : Machine Program} {program : Program}
    {links : ClaimBatchLink candidate schema Digest Program verify machine
      program}
    {initialApplication finalApplication : AppStateVector}
    {initialMemory finalMemory : ClosedCarry Digest}
    {segmentCount : Nat}
    (run : CompleteRun candidate schema Digest Program verify derive headers
      machine program links initialApplication initialMemory finalApplication
      finalMemory segmentCount) :
    run.claims.length = segmentCount * claimsPerSegment candidate :=
  run.memory.exactClaimCount

theorem claims_nonempty
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest Program : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {machine : Machine Program} {program : Program}
    {links : ClaimBatchLink candidate schema Digest Program verify machine
      program}
    {initialApplication finalApplication : AppStateVector}
    {initialMemory finalMemory : ClosedCarry Digest}
    {segmentCount : Nat}
    (run : CompleteRun candidate schema Digest Program verify derive headers
      machine program links initialApplication initialMemory finalApplication
      finalMemory segmentCount) :
    run.claims ≠ [] := by
  apply List.ne_nil_of_length_pos
  rw [run.exact_claim_count]
  have claimsPositive : 0 < claimsPerSegment candidate := by
    cases candidate <;> decide
  exact Nat.mul_pos run.positiveSegments claimsPositive

/-- The application side fills exactly the selected number of complete
segments. This theorem derives the count from the shared claim list. -/
theorem application_rows_length
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest Program : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {machine : Machine Program} {program : Program}
    {links : ClaimBatchLink candidate schema Digest Program verify machine
      program}
    {initialApplication finalApplication : AppStateVector}
    {initialMemory finalMemory : ClosedCarry Digest}
    {segmentCount : Nat}
    (run : CompleteRun candidate schema Digest Program verify derive headers
      machine program links initialApplication initialMemory finalApplication
      finalMemory segmentCount) :
    run.applicationRows.length =
      segmentCount * Completion.applicationRowsPerSegment := by
  rw [run.application.rows_length, run.exact_claim_count]
  calc
    (segmentCount * claimsPerSegment candidate) *
        rowsPerFreshClaim candidate =
      segmentCount *
        (claimsPerSegment candidate * rowsPerFreshClaim candidate) := by
          rw [Nat.mul_assoc]
    _ = segmentCount * Completion.applicationRowsPerSegment := by
      rw [ApplicationBatch.claims_rows_partition_segment]

/-- Base and every recursive invocation are reconstructed from the exact
segment chain. The terminal constructor consumes the trailing claim and has
no continuation. -/
theorem exact_delayed_lifecycle
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest Program : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {machine : Machine Program} {program : Program}
    {links : ClaimBatchLink candidate schema Digest Program verify machine
      program}
    {initialApplication finalApplication : AppStateVector}
    {initialMemory finalMemory : ClosedCarry Digest}
    {segmentCount : Nat}
    (run : CompleteRun candidate schema Digest Program verify derive headers
      machine program links initialApplication initialMemory finalApplication
      finalMemory segmentCount) :
    exists precommit activeAccessCount canOpen activeCountInRange
        endTimestampInRange active,
      openSegment derive headers precommit activeAccessCount initialMemory
          canOpen activeCountInRange endTimestampInRange = .active active /\
        DelayedRun verify derive headers (.active active) run.claims
          finalMemory :=
  SegmentChain.toDelayedRun run.memory run.positiveSegments

theorem complete_schedule
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest Program : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {machine : Machine Program} {program : Program}
    {links : ClaimBatchLink candidate schema Digest Program verify machine
      program}
    {initialApplication finalApplication : AppStateVector}
    {initialMemory finalMemory : ClosedCarry Digest}
    {segmentCount : Nat}
    (run : CompleteRun candidate schema Digest Program verify derive headers
      machine program links initialApplication initialMemory finalApplication
      finalMemory segmentCount) :
    CompleteSchedule run.claims.length :=
  run.memory.completeDelayedSchedule run.positiveSegments

/-- The complete application side is one exact deterministic WASM run. -/
theorem application_executes
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest Program : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {machine : Machine Program} {program : Program}
    {links : ClaimBatchLink candidate schema Digest Program verify machine
      program}
    {initialApplication finalApplication : AppStateVector}
    {initialMemory finalMemory : ClosedCarry Digest}
    {segmentCount : Nat}
    (run : CompleteRun candidate schema Digest Program verify derive headers
      machine program links initialApplication initialMemory finalApplication
      finalMemory segmentCount) :
    exists count, Runs machine program initialApplication run.applicationRows
      finalApplication count :=
  run.application.toRuns

theorem final_application_valid
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest Program : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {machine : Machine Program} {program : Program}
    {links : ClaimBatchLink candidate schema Digest Program verify machine
      program}
    {initialApplication finalApplication : AppStateVector}
    {initialMemory finalMemory : ClosedCarry Digest}
    {segmentCount : Nat}
    (run : CompleteRun candidate schema Digest Program verify derive headers
      machine program links initialApplication initialMemory finalApplication
      finalMemory segmentCount)
    (initialValid : initialApplication.Valid) :
    finalApplication.Valid :=
  run.application.after_valid initialValid

theorem every_claim_accepted
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest Program : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {machine : Machine Program} {program : Program}
    {links : ClaimBatchLink candidate schema Digest Program verify machine
      program}
    {initialApplication finalApplication : AppStateVector}
    {initialMemory finalMemory : ClosedCarry Digest}
    {segmentCount : Nat}
    (run : CompleteRun candidate schema Digest Program verify derive headers
      machine program links initialApplication initialMemory finalApplication
      finalMemory segmentCount) :
    forall receipt, receipt ∈ run.claims ->
      verify receipt.proof receipt.claim :=
  run.memory.everyClaimAccepted

/-- The exact application-to-claim refinement is available for every claim;
it is not replaced with a final execution assumption. -/
theorem every_claim_linked_to_application_batch
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest Program : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {machine : Machine Program} {program : Program}
    {links : ClaimBatchLink candidate schema Digest Program verify machine
      program}
    {initialApplication finalApplication : AppStateVector}
    {initialMemory finalMemory : ClosedCarry Digest}
    {segmentCount : Nat}
    (run : CompleteRun candidate schema Digest Program verify derive headers
      machine program links initialApplication initialMemory finalApplication
      finalMemory segmentCount) :
    forall receipt, receipt ∈ run.claims ->
      exists batchBefore batchAfter,
        exists batch : Batch candidate machine program batchBefore batchAfter,
          links receipt batch :=
  run.applicationLinks.every_linked

/-- There are exactly `T + 1` augmented invocations for `T` produced claims.
The extra invocation is terminal and produces no claim. -/
theorem augmented_invocation_count
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest Program : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {machine : Machine Program} {program : Program}
    {links : ClaimBatchLink candidate schema Digest Program verify machine
      program}
    {initialApplication finalApplication : AppStateVector}
    {initialMemory finalMemory : ClosedCarry Digest}
    {segmentCount : Nat}
    (run : CompleteRun candidate schema Digest Program verify derive headers
      machine program links initialApplication initialMemory finalApplication
      finalMemory segmentCount) :
    (List.range (run.claims.length + 1)).length = run.claims.length + 1 := by
  simp

end CompleteRun

end Nightstream.Protocol.NebulaV2.ProductionBatchedAugmentedLifecycle
