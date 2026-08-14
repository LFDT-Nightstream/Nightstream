import Nightstream.Protocol.Nebula.ExactDelayedSchedule
import Nightstream.HyperNova.Construction2.Paper

/-!
Contract: concrete countermodels for conditions enforced by the exact delayed
F-prime schedule.

Each countermodel removes one condition and gives a finite trace that the
weakened rule accepts. The corresponding exact rule rejects that trace. These
examples establish necessity only. They are not a soundness proof.

Assurance tier: model-level adversarial checks.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels

/-! ## Missing trailing terminal consumer -/

/-- Prefix equality is what remains if every existing receipt is checked but
the trailing claim need not have a receipt. -/
def ConsumedPrefixExact (produced consumed : List Nat) : Prop :=
  consumed = produced.take consumed.length

def unterminatedProduced : List Nat := [0]

def unterminatedConsumed : List Nat := []

/-- Local receipt checks are vacuous for the one unconsumed trailing claim. -/
theorem missing_terminal_passes_prefix_check :
    ConsumedPrefixExact unterminatedProduced unterminatedConsumed := by
  rfl

/-- Exact delayed consumption rejects the same open trace. -/
theorem missing_terminal_breaks_equal_claim_counts :
    unterminatedProduced.length ≠ unterminatedConsumed.length := by
  decide

/-! ## Consuming the current claim instead of the prior claim -/

structure WeakRecursiveEvent where
  previous : Nat
  next : Nat
  consumed : Nat

def ConsumesCurrent (event : WeakRecursiveEvent) : Prop :=
  event.consumed = event.next

def ConsumesPrior (event : WeakRecursiveEvent) : Prop :=
  event.consumed = event.previous

def currentClaimEvent : WeakRecursiveEvent :=
  { previous := 0, next := 1, consumed := 1 }

/-- A current-claim rule accepts an event that does not consume its indexed
prior claim. -/
theorem current_claim_rule_accepts_wrong_index :
    ConsumesCurrent currentClaimEvent /\ ¬ ConsumesPrior currentClaimEvent := by
  simp [ConsumesCurrent, ConsumesPrior, currentClaimEvent]

/-! ## A terminal invocation that produces another claim -/

structure WeakTerminalEvent where
  previous : Nat
  consumed : Nat
  produced : Option Nat

def ConsumesTrailing (event : WeakTerminalEvent) : Prop :=
  event.consumed = event.previous

def ProducesNothing (event : WeakTerminalEvent) : Prop :=
  event.produced = none

def producingTerminal : WeakTerminalEvent :=
  { previous := 7, consumed := 7, produced := some 8 }

/-- Exact trailing consumption alone does not stop a malformed terminal from
creating an unconsumed successor. -/
theorem terminal_consumption_does_not_forbid_successor :
    ConsumesTrailing producingTerminal /\
      ¬ ProducesNothing producingTerminal := by
  simp [ConsumesTrailing, ProducesNothing, producingTerminal]

/-! ## A base invocation that consumes a claim -/

structure WeakBaseEvent where
  produced : Nat
  consumed : Option Nat

def ProducesClaimZero (event : WeakBaseEvent) : Prop :=
  event.produced = 0

def ConsumesNothing (event : WeakBaseEvent) : Prop :=
  event.consumed = none

def consumingBase : WeakBaseEvent :=
  { produced := 0, consumed := some 99 }

/-- Producing claim zero does not imply that the base consumed no prior
claim. The exact base node excludes a consumed-claim field by construction. -/
theorem base_production_does_not_forbid_consumption :
    ProducesClaimZero consumingBase /\ ¬ ConsumesNothing consumingBase := by
  simp [ProducesClaimZero, ConsumesNothing, consumingBase]

/-! ## Erasing the generated relation exponent -/

structure RelationIdentity where
  version : Nat
  profile : Nat
  rowVariables : Nat
deriving DecidableEq

def ErasedIdentityEqual (left right : RelationIdentity) : Prop :=
  left.version = right.version /\ left.profile = right.profile

def FullIdentityEqual (left right : RelationIdentity) : Prop :=
  left = right

def relationAt26 : RelationIdentity :=
  { version := 3, profile := 1, rowVariables := 26 }

def relationAt27 : RelationIdentity :=
  { version := 3, profile := 1, rowVariables := 27 }

/-- An identity that omits the row exponent aliases two different generated
relations. Full verifier-key identity rejects the alias. -/
theorem erased_exponent_aliases_distinct_relations :
    ErasedIdentityEqual relationAt26 relationAt27 /\
      ¬ FullIdentityEqual relationAt26 relationAt27 := by
  simp [ErasedIdentityEqual, FullIdentityEqual, relationAt26, relationAt27]

/-! ## Scheduling a claim other than the exact produced successor -/

structure WeakProducerLink where
  previous : Nat
  consumed : Nat
  produced : Nat
  scheduledNext : Nat

def ConsumesExactPrevious (event : WeakProducerLink) : Prop :=
  event.consumed = event.previous

def SchedulesExactProduced (event : WeakProducerLink) : Prop :=
  event.scheduledNext = event.produced

def redirectedSuccessor : WeakProducerLink :=
  { previous := 3, consumed := 3, produced := 4, scheduledNext := 9 }

/-- Exact prior-claim consumption does not bind the successor. The dependent
recursive schedule index separately requires the next schedule to start at the
claim emitted by the recursive node. -/
theorem prior_consumption_does_not_bind_produced_successor :
    ConsumesExactPrevious redirectedSuccessor /\
      ¬ SchedulesExactProduced redirectedSuccessor := by
  simp [ConsumesExactPrevious, SchedulesExactProduced, redirectedSuccessor]

/-! ## Verifier acceptance without exact proof forwarding -/

structure WeakReceipt where
  expectedClaim : Nat
  receiptClaim : Nat
  expectedProof : Nat
  receiptProof : Nat

def ClaimForwarded (receipt : WeakReceipt) : Prop :=
  receipt.receiptClaim = receipt.expectedClaim

def WeakVerifier (_proof _claim : Nat) : Prop := True

def ProofForwarded (receipt : WeakReceipt) : Prop :=
  receipt.receiptProof = receipt.expectedProof

def substitutedProof : WeakReceipt :=
  { expectedClaim := 5
    receiptClaim := 5
    expectedProof := 11
    receiptProof := 12 }

/-- A permissive verifier can accept a receipt for the right claim while the
receipt carries a different proof. Exact proof forwarding is an independent
schedule condition. -/
theorem acceptance_and_claim_equality_do_not_forward_the_exact_proof :
    ClaimForwarded substitutedProof /\
      WeakVerifier substitutedProof.receiptProof substitutedProof.receiptClaim /\
      ¬ ProofForwarded substitutedProof := by
  simp [ClaimForwarded, WeakVerifier, ProofForwarded, substitutedProof]

/-! ## Terminal consumption without closed state -/

structure WeakTerminalState where
  previous : Nat
  consumed : Nat
  closed : Bool

def TerminalStateConsumesTrailing (event : WeakTerminalState) : Prop :=
  event.consumed = event.previous

def TerminalStateIsClosed (event : WeakTerminalState) : Prop :=
  event.closed = true

def openTerminal : WeakTerminalState :=
  { previous := 8, consumed := 8, closed := false }

/-- Consuming the trailing claim does not prove that the terminal relation
opened a closed memory carry. The concrete terminal node must enforce both. -/
theorem trailing_consumption_does_not_imply_closed_state :
    TerminalStateConsumesTrailing openTerminal /\
      ¬ TerminalStateIsClosed openTerminal := by
  simp [TerminalStateConsumesTrailing, TerminalStateIsClosed, openTerminal]

/-! ## Base with no prior claim but the wrong invocation index -/

structure WeakBaseIndex where
  consumed : Option Nat
  invocationIndex : Nat

def BaseHasNoPrior (event : WeakBaseIndex) : Prop :=
  event.consumed = none

def BaseHasCanonicalIndex (event : WeakBaseIndex) : Prop :=
  event.invocationIndex = 1

def misindexedBase : WeakBaseIndex :=
  { consumed := none, invocationIndex := 2 }

/-- The absence of a prior claim does not fix the base invocation index. The
base row theorem must derive index one separately. -/
theorem no_prior_claim_does_not_fix_base_index :
    BaseHasNoPrior misindexedBase /\
      ¬ BaseHasCanonicalIndex misindexedBase := by
  simp [BaseHasNoPrior, BaseHasCanonicalIndex, misindexedBase]

/-! ## Counts do not fix the indexes read from consumer states -/

structure WeakIndexedLifetime where
  producedClaimCount : Nat
  consumerInvocationIndices : List Nat
deriving DecidableEq, Repr

def ClaimCountExact (lifetime : WeakIndexedLifetime) : Prop :=
  lifetime.consumerInvocationIndices.length = lifetime.producedClaimCount

def ConsumerIndexesExact (lifetime : WeakIndexedLifetime) : Prop :=
  lifetime.consumerInvocationIndices =
    List.range' 1 lifetime.producedClaimCount

def repeatedConsumerIndex : WeakIndexedLifetime :=
  { producedClaimCount := 2
    consumerInvocationIndices := [1, 1] }

/-- Equal produced and consumed counts do not prove that the recursive and
terminal rows read prior-state indexes `1, ..., T`. -/
theorem claim_count_does_not_fix_consumer_indexes :
    ClaimCountExact repeatedConsumerIndex /\
      ¬ ConsumerIndexesExact repeatedConsumerIndex := by
  simp [ClaimCountExact, ConsumerIndexesExact, repeatedConsumerIndex,
    List.range']

/-! ## Exact indexes do not fix the rest of the authenticated state -/

structure TinyPriorState where
  invocationIndex : Nat
  applicationState : Bool
deriving DecidableEq, Repr

def SameInvocationIndex (producer consumer : TinyPriorState) : Prop :=
  producer.invocationIndex = consumer.invocationIndex

def FullStateContinuous (producer consumer : TinyPriorState) : Prop :=
  producer = consumer

def producerState : TinyPriorState :=
  { invocationIndex := 1, applicationState := false }

def substitutedConsumerState : TinyPriorState :=
  { invocationIndex := 1, applicationState := true }

/-- Even the exact invocation index permits application-state substitution
unless the complete predecessor and consumer states are equal. -/
theorem exact_index_does_not_fix_full_state :
    SameInvocationIndex producerState substitutedConsumerState /\
      ¬ FullStateContinuous producerState substitutedConsumerState := by
  simp [SameInvocationIndex, FullStateContinuous, producerState,
    substitutedConsumerState]

/-! ## The paper terminal is not the delayed-memory terminal consumer -/

namespace PaperTerminalGap

open Nightstream.HyperNova.NonInteractiveMultiFold
open Nightstream.HyperNova.Construction2

/-- A one-slot Construction-2 setup whose NIFS verifier rejects every fold.
This is valid setup data because the paper model does not assume completeness
inside the setup record. -/
def setup : Paper.Setup Unit Unit Bool Unit 1 where
  verifierKeys := fun _ => ()
  nifs := { verify := fun _ _ _ _ => none }
  defaultRunning := ()

/-- A concrete paper machine with a trivial public link. -/
def machine : Paper.Machine Unit Unit Unit Unit Unit Bool Unit 1 where
  control := fun _ _ => ⟨0, by decide⟩
  step := fun _ _ _ => ()
  freshPublic := fun _ => ()
  encodeInstance := fun _ => ()
  hash := fun _ => ()

/-- In this model, `false` is a valid fresh relation instance, but it still
represents an open delayed-memory state. -/
def relations : Paper.TerminalRelations Unit Unit Unit Bool Unit 1 where
  runningHolds := fun _ _ _ _ => True
  freshHolds := fun _ _ fresh _ => fresh = false

def statement : Paper.TerminalStatement Unit where
  iteration := 1
  z0 := ()
  zi := ()

def payload : Paper.TerminalProof Unit Unit Bool Unit 1 where
  running := fun _ => ()
  runningWitness := fun _ => ()
  fresh := false
  freshWitness := ()
  pc := 1

/-- The added Nebula terminal condition: the delayed memory state is closed.
The Boolean is only a finite countermodel for the missing condition. -/
def DelayedMemoryClosed (fresh : Bool) : Prop :=
  fresh = true

/-- The exact paper terminal relation accepts a relation-valid trailing fresh
claim whose delayed-memory state is still open. -/
theorem paper_terminal_accepts_open_delayed_memory :
    Paper.RecursiveTerminalTransition setup machine relations statement
        payload /\
      ¬ DelayedMemoryClosed payload.fresh := by
  constructor
  · refine ⟨⟨by decide, by decide⟩, by decide, rfl, ?_, rfl⟩
    intro _
    trivial
  · simp [DelayedMemoryClosed, payload]

/-- The same paper terminal acceptance is possible although `NIFS.V` rejects
every possible trailing fold. Thus a final NIFS fold is not part of the paper
terminal theorem. -/
theorem paper_terminal_acceptance_does_not_imply_a_trailing_fold :
    Paper.RecursiveTerminalTransition setup machine relations statement
        payload /\
      (forall output : Unit,
        ¬ Accepts setup.nifs () () payload.fresh () output) := by
  refine ⟨paper_terminal_accepts_open_delayed_memory.1, ?_⟩
  intro output
  simp [Accepts, setup]

/-- V2 deliberately strengthens the paper terminal with delayed-memory
closure. This is a specialized protocol rule, not unchanged Construction 2.
-/
def V2TerminalTransition
    (proof : Paper.TerminalProof Unit Unit Bool Unit 1) : Prop :=
  Paper.RecursiveTerminalTransition setup machine relations statement proof /\
    DelayedMemoryClosed proof.fresh

/-- The added V2 terminal condition is a strict strengthening of the paper
terminal relation in this finite model. The forward implication is structural:
V2 retains every paper terminal check. -/
theorem v2_terminal_implies_paper_terminal
    {proof : Paper.TerminalProof Unit Unit Bool Unit 1}
    (accepted : V2TerminalTransition proof) :
    Paper.RecursiveTerminalTransition setup machine relations statement proof :=
  accepted.1

/-- The V2 strengthening rejects the concrete open terminal accepted by the
paper relation alone. -/
theorem v2_terminal_rejects_open_delayed_memory :
    ¬ V2TerminalTransition payload := by
  simp [V2TerminalTransition, DelayedMemoryClosed, payload]

/-- The implication above is not reversible. Therefore an implementation may
use the paper terminal relation as one V2 terminal condition, but it cannot
use that relation as the complete V2 terminal verifier. -/
theorem paper_terminal_is_strictly_weaker :
    (forall proof : Paper.TerminalProof Unit Unit Bool Unit 1,
      V2TerminalTransition proof ->
        Paper.RecursiveTerminalTransition setup machine relations statement
          proof) /\
      exists proof : Paper.TerminalProof Unit Unit Bool Unit 1,
        Paper.RecursiveTerminalTransition setup machine relations statement
            proof /\
          ¬ V2TerminalTransition proof := by
  constructor
  · intro proof accepted
    exact v2_terminal_implies_paper_terminal accepted
  · exact ⟨payload, paper_terminal_accepts_open_delayed_memory.1,
      v2_terminal_rejects_open_delayed_memory⟩

end PaperTerminalGap

end Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels
