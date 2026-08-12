import Nightstream.Implementation.NebulaV2.Production.FPrime.Lifetime.FPrimeNodesFor
import Nightstream.Protocol.NebulaV2.ExactDelayedSchedule

/-!
Contract: exact claim-level Nebula delayed-consumption lifetime at one
generated relation exponent.

Base produces claim zero and has no prior claim or NIFS proof. Each recursive
node verifies and consumes the exact preceding complete claim before it
produces one next claim. Terminal verifies and consumes the trailing claim and
has no successor or produced claim. The theorem proves equality of the ordered
consumed receipt claims and ordered produced complete claims. It also proves
equal produced and consumed counts and exactly one more augmented invocation.

This lifecycle is a specialized adaptation of HyperNova Construction 2.
Construction 2 checks the trailing fresh relation at its outer terminal and
does not perform another NIFS fold. V2 adds one terminal consumer because the
memory transition is delayed by one claim. Its soundness is proved here as a
separate schedule; it is not inherited unchanged from Construction 2.

This module does not infer state continuity from digest equality. It also does
not own memory-chain extraction, completed execution, cryptographic bounds,
recursive-size closure, generated artifact extraction, Rust, or the compact
terminal backend.

Assurance tier: exponent-indexed claim-lifecycle model.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor

open Nightstream.Protocol.NebulaV2

structure RecursiveEvent
    {Program : Type} (context : Context Program)
    (previous next : context.Claim) where
  node : RecursiveNode context previous
  nextExact : next = node.nextClaim

structure TerminalEvent
    {Program : Type} (context : Context Program)
    (previous : context.Claim) where
  node : TerminalNode context previous

noncomputable def scheduleInterface
    {Program : Type} (context : Context Program) :
    ExactDelayedSchedule.Interface context.Claim context.ProtocolClaim
      context.Proof context.Receipt where
  toProtocolClaim := fun claim => claim.toProtocolClaim
  receiptClaim := fun receipt => receipt.claim
  receiptProof := fun receipt => receipt.proof
  verifier := context.Verifier
  RecursiveEvent := RecursiveEvent context
  TerminalEvent := TerminalEvent context
  recursiveReceipt := fun event => event.node.recursive.verified
  recursiveProof := fun event => event.node.proof
  recursiveClaimExact := fun event => event.node.consumes_previous
  recursiveProofExact := fun event => event.node.proof_is_exact
  recursiveAccepted := fun event => event.node.accepted
  terminalReceipt := fun event => event.node.recursive.verified
  terminalProof := fun event => event.node.proof
  terminalClaimExact := fun event => event.node.consumes_trailing
  terminalProofExact := fun event => event.node.proof_is_exact
  terminalAccepted := fun event => event.node.accepted

abbrev Schedule
    {Program : Type} (context : Context Program) (previous : context.Claim) :=
  ExactDelayedSchedule.Schedule (scheduleInterface context) previous

namespace Schedule

def ProductionExact
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (schedule : Schedule context previous) : Prop :=
  schedule.AllRecursive
    (fun {previous next : context.Claim}
      (event : RecursiveEvent context previous next) =>
        next = event.node.nextClaim)

theorem production_exact
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (schedule : Schedule context previous) : schedule.ProductionExact := by
  unfold ProductionExact
  induction schedule with
  | terminal event => trivial
  | recursive event rest inductionHypothesis =>
      exact ⟨event.nextExact, inductionHypothesis⟩

/-- Every nonterminal event uses the verifier-owned recursive branch of the
single fixed F-prime relation. This predicate records row-derived branch
selection; it does not accept a caller-supplied branch label. -/
def FixedRecursiveBranches
    {Program : Type} {context : Context Program} {previous : context.Claim} :
    Schedule context previous -> Prop
  | .terminal _ => True
  | .recursive event rest =>
      0 < event.node.assignment
          context.relationAuthority.fPrimeProgram.layout.iterationZero.iterationColumn /\
        R1CS.Satisfies context.recursiveProgram.rows event.node.assignment /\
        FixedRecursiveBranches rest

/-- Extract the mandatory trailing terminal node. An open recursive prefix has
no inhabitant of `Schedule`, so this function has no open-tail case. -/
noncomputable def terminalNode
    {Program : Type} {context : Context Program} {previous : context.Claim} :
    (schedule : Schedule context previous) ->
      Sigma fun trailing => TerminalNode context trailing
  | .terminal event => ⟨previous, event.node⟩
  | .recursive _ rest => terminalNode rest

/-- Invocation indexes read from the exact prior-state value consumed by each
post-base node. The final entry is the terminal consumer. This list is tied to
the retained claim schedule; it is not reconstructed from a claim count. -/
noncomputable def consumerInvocationIndices
    {Program : Type} {context : Context Program} {previous : context.Claim} :
    Schedule context previous -> List Nat
  | .terminal event =>
      [event.node.recursive.priorState.augmentedInvocationIndex]
  | .recursive event rest =>
      event.node.recursive.priorState.augmentedInvocationIndex ::
        consumerInvocationIndices rest

@[simp] theorem consumerInvocationIndices_terminal
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (event : TerminalEvent context previous) :
    consumerInvocationIndices
        (ExactDelayedSchedule.Schedule.terminal event) =
      [event.node.recursive.priorState.augmentedInvocationIndex] := by
  rfl

@[simp] theorem consumerInvocationIndices_recursive
    {Program : Type} {context : Context Program}
    {previous next : context.Claim}
    (event : RecursiveEvent context previous next)
    (rest : Schedule context next) :
    consumerInvocationIndices
        (ExactDelayedSchedule.Schedule.recursive event rest) =
      event.node.recursive.priorState.augmentedInvocationIndex ::
        consumerInvocationIndices rest := by
  rfl

/-- The retained schedule has exactly one parsed consumer index for each
verified receipt, including the trailing terminal receipt. -/
theorem consumerInvocationIndices_length
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (schedule : Schedule context previous) :
    schedule.consumerInvocationIndices.length =
      (ExactDelayedSchedule.Schedule.receipts schedule).length := by
  induction schedule with
  | terminal event => rfl
  | recursive event rest inductionHypothesis =>
      simp only [consumerInvocationIndices_recursive,
        ExactDelayedSchedule.Schedule.receipts, List.length_cons,
        inductionHypothesis]

/-- Every consumer reads the complete successor state produced immediately
before it. This is stronger than claim order, receipt count, or invocation
index order: it also binds the application state, running claims, memory
carry, immutable initial state, and every other authenticated state field. -/
noncomputable def FullStateContinuous
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (producer : context.Successor) : Schedule context previous -> Prop
  | .terminal event =>
      producer = event.node.recursive.priorState
  | .recursive event rest =>
      producer = event.node.recursive.priorState /\
        FullStateContinuous event.node.supplement.successor rest

@[simp] theorem fullStateContinuous_terminal
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (producer : context.Successor) (event : TerminalEvent context previous) :
    FullStateContinuous producer
        (ExactDelayedSchedule.Schedule.terminal event) =
      (producer = event.node.recursive.priorState) := by
  rfl

@[simp] theorem fullStateContinuous_recursive
    {Program : Type} {context : Context Program}
    {previous next : context.Claim}
    (producer : context.Successor)
    (event : RecursiveEvent context previous next)
    (rest : Schedule context next) :
    FullStateContinuous producer
        (ExactDelayedSchedule.Schedule.recursive event rest) =
      (producer = event.node.recursive.priorState /\
        FullStateContinuous event.node.supplement.successor rest) := by
  rfl

end Schedule

noncomputable def RecursiveNode.prepend
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous)
    (rest : Schedule context node.nextClaim) : Schedule context previous :=
  .recursive { node := node, nextExact := rfl } rest

noncomputable def TerminalNode.finish
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) : Schedule context previous :=
  .terminal { node := node }

structure Lifetime
    {Program : Type} (context : Context Program) where
  base : BaseNode context
  firstClaim : context.Claim
  firstClaimExact : firstClaim = base.claim
  schedule : Schedule context firstClaim

namespace Lifetime

/-- Construct one global lifetime from an exact base node and an inductive
tail built with `RecursiveNode.prepend` and `TerminalNode.finish`. The tail
type admits no unterminated case. Its dependent claim index prevents a
recursive node from redirecting the next claim. -/
noncomputable def construct
    {Program : Type} {context : Context Program}
    (base : BaseNode context) (schedule : Schedule context base.claim) :
    Lifetime context where
  base := base
  firstClaim := base.claim
  firstClaimExact := rfl
  schedule := schedule

noncomputable def producedClaims
    {Program : Type} {context : Context Program} (lifetime : Lifetime context) :
    List context.Claim :=
  ExactDelayedSchedule.Schedule.claims lifetime.schedule

noncomputable def consumedReceipts
    {Program : Type} {context : Context Program} (lifetime : Lifetime context) :
    List context.Receipt :=
  ExactDelayedSchedule.Schedule.receipts lifetime.schedule

noncomputable def augmentedInvocationCount
    {Program : Type} {context : Context Program} (lifetime : Lifetime context) :
    Nat := 1 + ExactDelayedSchedule.Schedule.invocationCount lifetime.schedule

theorem base_produces_first
    {Program : Type} {context : Context Program} (lifetime : Lifetime context) :
    lifetime.producedClaims.head? = some lifetime.base.claim := by
  calc
    lifetime.producedClaims.head? = some lifetime.firstClaim :=
      ExactDelayedSchedule.Schedule.claims_head lifetime.schedule
    _ = some lifetime.base.claim := congrArg some lifetime.firstClaimExact

theorem consumed_claims_are_exactly_produced_claims
    {Program : Type} {context : Context Program} (lifetime : Lifetime context) :
    lifetime.consumedReceipts.map (fun receipt => receipt.claim) =
      lifetime.producedClaims.map (fun claim => claim.toProtocolClaim) := by
  simpa [producedClaims, consumedReceipts, scheduleInterface] using
    ExactDelayedSchedule.Schedule.receipt_claims_exact lifetime.schedule

theorem consumed_proofs_are_exact
    {Program : Type} {context : Context Program} (lifetime : Lifetime context) :
    lifetime.consumedReceipts.map (fun receipt => receipt.proof) =
      ExactDelayedSchedule.Schedule.proofs lifetime.schedule := by
  simpa [consumedReceipts, scheduleInterface] using
    ExactDelayedSchedule.Schedule.receipt_proofs_exact lifetime.schedule

theorem every_consumed_receipt_accepted
    {Program : Type} {context : Context Program} (lifetime : Lifetime context) :
    forall receipt, receipt ∈ lifetime.consumedReceipts ->
      context.Verifier receipt.proof receipt.claim := by
  simpa [consumedReceipts, scheduleInterface] using
    ExactDelayedSchedule.Schedule.every_receipt_accepted lifetime.schedule

theorem terminal_consumes_trailing
    {Program : Type} {context : Context Program} (lifetime : Lifetime context) :
    (ExactDelayedSchedule.Schedule.lastReceipt lifetime.schedule).claim =
      (ExactDelayedSchedule.Schedule.lastClaim
        lifetime.schedule).toProtocolClaim := by
  simpa [scheduleInterface] using
    ExactDelayedSchedule.Schedule.trailing_receipt_exact lifetime.schedule

theorem produced_count_eq_consumed_count
    {Program : Type} {context : Context Program} (lifetime : Lifetime context) :
    lifetime.producedClaims.length = lifetime.consumedReceipts.length := by
  simpa [producedClaims, consumedReceipts] using
    ExactDelayedSchedule.Schedule.claims_length_eq_receipts lifetime.schedule

theorem augmented_count_eq_claim_count_add_one
    {Program : Type} {context : Context Program} (lifetime : Lifetime context) :
    lifetime.augmentedInvocationCount =
      lifetime.producedClaims.length + 1 := by
  unfold augmentedInvocationCount producedClaims
  rw [ExactDelayedSchedule.Schedule.claims_length_eq_invocations
    lifetime.schedule]
  omega

theorem complete_index_schedule
    {Program : Type} {context : Context Program} (lifetime : Lifetime context) :
    Lifecycle.CompleteSchedule lifetime.producedClaims.length := by
  simpa [producedClaims] using
    ExactDelayedSchedule.Schedule.complete_index_schedule lifetime.schedule

/-! The combined result states the facts that rule out all F-prime index
shifts. None is an assumption of this theorem. -/
structure ExactSchedule
    {Program : Type} {context : Context Program}
    (lifetime : Lifetime context) : Prop where
  first : lifetime.producedClaims.head? = some lifetime.base.claim
  consumedClaims :
    lifetime.consumedReceipts.map (fun receipt => receipt.claim) =
      lifetime.producedClaims.map (fun claim => claim.toProtocolClaim)
  consumedProofs :
    lifetime.consumedReceipts.map (fun receipt => receipt.proof) =
      ExactDelayedSchedule.Schedule.proofs lifetime.schedule
  accepted : forall receipt, receipt ∈ lifetime.consumedReceipts ->
    context.Verifier receipt.proof receipt.claim
  recursiveProduction : Schedule.ProductionExact lifetime.schedule
  trailing :
    (ExactDelayedSchedule.Schedule.lastReceipt lifetime.schedule).claim =
      (ExactDelayedSchedule.Schedule.lastClaim
        lifetime.schedule).toProtocolClaim
  equalClaimCounts : lifetime.producedClaims.length =
    lifetime.consumedReceipts.length
  oneExtraInvocation : lifetime.augmentedInvocationCount =
    lifetime.producedClaims.length + 1
  indexSchedule : Lifecycle.CompleteSchedule lifetime.producedClaims.length

/-- The base and all recursive producers use the two fixed arms of the one
verifier-owned F-prime relation. The base arm is selected only at iteration
zero. Every nonterminal event selects the recursive arm at a positive
iteration. -/
structure FixedBranchSchedule
    {Program : Type} {context : Context Program}
    (lifetime : Lifetime context) : Prop where
  baseIterationZero :
    lifetime.base.baseRows.assignment
        context.relationAuthority.fPrimeProgram.layout.iterationZero.iterationColumn =
      0
  baseRows : R1CS.Satisfies context.baseArtifact.programRows
    lifetime.base.baseRows.assignment
  recursiveRows : lifetime.schedule.FixedRecursiveBranches
  terminalRows : R1CS.Satisfies context.terminalProgram.rows
    lifetime.schedule.terminalNode.2.assignment
  terminalTypedRows : context.terminalTypedProgram.RowsSatisfied
    context.artifact.system
    lifetime.schedule.terminalNode.2.rowAccepted.rows.assignment

theorem exact_schedule
    {Program : Type} {context : Context Program} (lifetime : Lifetime context) :
    ExactSchedule lifetime :=
  { first := lifetime.base_produces_first
    consumedClaims := lifetime.consumed_claims_are_exactly_produced_claims
    consumedProofs := lifetime.consumed_proofs_are_exact
    accepted := lifetime.every_consumed_receipt_accepted
    recursiveProduction := Schedule.production_exact lifetime.schedule
    trailing := lifetime.terminal_consumes_trailing
    equalClaimCounts := lifetime.produced_count_eq_consumed_count
    oneExtraInvocation := lifetime.augmented_count_eq_claim_count_add_one
    indexSchedule := lifetime.complete_index_schedule }

/-- Constructive completeness of the global F-prime scheduler relative to
local exact base, recursive, and terminal invocation witnesses. The exact
schedule is a derived result, not a proposition supplied by the caller. -/
theorem construct_exact_schedule
    {Program : Type} {context : Context Program}
    (base : BaseNode context) (schedule : Schedule context base.claim) :
    ExactSchedule (construct base schedule) :=
  exact_schedule (construct base schedule)

/-- Every constructed complete lifetime contains an exact terminal node. -/
theorem construct_has_terminal
    {Program : Type} {context : Context Program}
    (base : BaseNode context) (schedule : Schedule context base.claim) :
    Nonempty (Sigma fun trailing => TerminalNode context trailing) :=
  ⟨schedule.terminalNode⟩

end Lifetime

end Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor
