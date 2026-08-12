import Nightstream.Protocol.NebulaV2.Lifecycle

/-!
Contract: generic certificate for the exact delayed F-prime schedule.

Base supplies the first claim outside this certificate. Each recursive event
consumes its indexed claim and advances to its indexed next claim. The terminal
event consumes the trailing claim and cannot produce a successor. The event
families own the evidence that makes each local transition authoritative.

Theorems derive ordered claim consumption, exact proof forwarding, verifier
acceptance, equal produced and consumed counts, and the extra terminal
invocation. They do not assume any of those conclusions.

Assurance tier: model-level lifecycle theorem.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.ExactDelayedSchedule

structure Interface
    (Claim ProtocolClaim Proof Receipt : Type) where
  toProtocolClaim : Claim -> ProtocolClaim
  receiptClaim : Receipt -> ProtocolClaim
  receiptProof : Receipt -> Proof
  verifier : Proof -> ProtocolClaim -> Prop
  RecursiveEvent : Claim -> Claim -> Type
  TerminalEvent : Claim -> Type
  recursiveReceipt : {previous next : Claim} ->
    RecursiveEvent previous next -> Receipt
  recursiveProof : {previous next : Claim} ->
    RecursiveEvent previous next -> Proof
  recursiveClaimExact : {previous next : Claim} ->
    (event : RecursiveEvent previous next) ->
      receiptClaim (recursiveReceipt event) = toProtocolClaim previous
  recursiveProofExact : {previous next : Claim} ->
    (event : RecursiveEvent previous next) ->
      receiptProof (recursiveReceipt event) = recursiveProof event
  recursiveAccepted : {previous next : Claim} ->
    (event : RecursiveEvent previous next) ->
      verifier (receiptProof (recursiveReceipt event))
        (receiptClaim (recursiveReceipt event))
  terminalReceipt : {previous : Claim} -> TerminalEvent previous -> Receipt
  terminalProof : {previous : Claim} -> TerminalEvent previous -> Proof
  terminalClaimExact : {previous : Claim} ->
    (event : TerminalEvent previous) ->
      receiptClaim (terminalReceipt event) = toProtocolClaim previous
  terminalProofExact : {previous : Claim} ->
    (event : TerminalEvent previous) ->
      receiptProof (terminalReceipt event) = terminalProof event
  terminalAccepted : {previous : Claim} ->
    (event : TerminalEvent previous) ->
      verifier (receiptProof (terminalReceipt event))
        (receiptClaim (terminalReceipt event))

inductive Schedule
    {Claim ProtocolClaim Proof Receipt : Type}
    (interface : Interface Claim ProtocolClaim Proof Receipt) : Claim -> Type
  | terminal {previous : Claim}
      (event : interface.TerminalEvent previous) : Schedule interface previous
  | recursive {previous next : Claim}
      (event : interface.RecursiveEvent previous next)
      (rest : Schedule interface next) : Schedule interface previous

namespace Schedule

def claims
    {Claim ProtocolClaim Proof Receipt : Type}
    {interface : Interface Claim ProtocolClaim Proof Receipt}
    {previous : Claim} : Schedule interface previous -> List Claim
  | .terminal _ => [previous]
  | .recursive _ rest => previous :: rest.claims

def receipts
    {Claim ProtocolClaim Proof Receipt : Type}
    {interface : Interface Claim ProtocolClaim Proof Receipt}
    {previous : Claim} : Schedule interface previous -> List Receipt
  | .terminal event => [interface.terminalReceipt event]
  | .recursive event rest => interface.recursiveReceipt event :: rest.receipts

def proofs
    {Claim ProtocolClaim Proof Receipt : Type}
    {interface : Interface Claim ProtocolClaim Proof Receipt}
    {previous : Claim} : Schedule interface previous -> List Proof
  | .terminal event => [interface.terminalProof event]
  | .recursive event rest => interface.recursiveProof event :: rest.proofs

def invocationCount
    {Claim ProtocolClaim Proof Receipt : Type}
    {interface : Interface Claim ProtocolClaim Proof Receipt}
    {previous : Claim} : Schedule interface previous -> Nat
  | .terminal _ => 1
  | .recursive _ rest => 1 + rest.invocationCount

def lastClaim
    {Claim ProtocolClaim Proof Receipt : Type}
    {interface : Interface Claim ProtocolClaim Proof Receipt}
    {previous : Claim} : Schedule interface previous -> Claim
  | .terminal _ => previous
  | .recursive _ rest => rest.lastClaim

def lastReceipt
    {Claim ProtocolClaim Proof Receipt : Type}
    {interface : Interface Claim ProtocolClaim Proof Receipt}
    {previous : Claim} : Schedule interface previous -> Receipt
  | .terminal event => interface.terminalReceipt event
  | .recursive _ rest => rest.lastReceipt

def AllRecursive
    {Claim ProtocolClaim Proof Receipt : Type}
    {interface : Interface Claim ProtocolClaim Proof Receipt}
    (property : {previous next : Claim} ->
      interface.RecursiveEvent previous next -> Prop)
    {previous : Claim} : Schedule interface previous -> Prop
  | .terminal _ => True
  | .recursive event rest => property event ∧ rest.AllRecursive property

theorem claims_nonempty
    {Claim ProtocolClaim Proof Receipt : Type}
    {interface : Interface Claim ProtocolClaim Proof Receipt}
    {previous : Claim} (schedule : Schedule interface previous) :
    schedule.claims ≠ [] := by
  cases schedule <;> simp [claims]

theorem claims_head
    {Claim ProtocolClaim Proof Receipt : Type}
    {interface : Interface Claim ProtocolClaim Proof Receipt}
    {previous : Claim} (schedule : Schedule interface previous) :
    schedule.claims.head? = some previous := by
  cases schedule <;> rfl

theorem receipt_claims_exact
    {Claim ProtocolClaim Proof Receipt : Type}
    {interface : Interface Claim ProtocolClaim Proof Receipt}
    {previous : Claim} (schedule : Schedule interface previous) :
    schedule.receipts.map interface.receiptClaim =
      schedule.claims.map interface.toProtocolClaim := by
  induction schedule with
  | terminal event =>
      simp [receipts, claims, interface.terminalClaimExact event]
  | recursive event rest inductionHypothesis =>
      simp [receipts, claims, interface.recursiveClaimExact event,
        inductionHypothesis]

theorem receipt_proofs_exact
    {Claim ProtocolClaim Proof Receipt : Type}
    {interface : Interface Claim ProtocolClaim Proof Receipt}
    {previous : Claim} (schedule : Schedule interface previous) :
    schedule.receipts.map interface.receiptProof = schedule.proofs := by
  induction schedule with
  | terminal event =>
      simp [receipts, proofs, interface.terminalProofExact event]
  | recursive event rest inductionHypothesis =>
      simp [receipts, proofs, interface.recursiveProofExact event,
        inductionHypothesis]

theorem every_receipt_accepted
    {Claim ProtocolClaim Proof Receipt : Type}
    {interface : Interface Claim ProtocolClaim Proof Receipt}
    {previous : Claim} (schedule : Schedule interface previous) :
    forall receipt, receipt ∈ schedule.receipts ->
      interface.verifier (interface.receiptProof receipt)
        (interface.receiptClaim receipt) := by
  induction schedule with
  | terminal event =>
      intro receipt member
      simp only [receipts, List.mem_singleton] at member
      subst receipt
      exact interface.terminalAccepted event
  | recursive event rest inductionHypothesis =>
      intro receipt member
      simp only [receipts, List.mem_cons] at member
      rcases member with rfl | member
      · exact interface.recursiveAccepted event
      · exact inductionHypothesis receipt member

theorem claims_length_eq_receipts
    {Claim ProtocolClaim Proof Receipt : Type}
    {interface : Interface Claim ProtocolClaim Proof Receipt}
    {previous : Claim} (schedule : Schedule interface previous) :
    schedule.claims.length = schedule.receipts.length := by
  induction schedule with
  | terminal event => rfl
  | recursive event rest inductionHypothesis =>
      simp [claims, receipts, inductionHypothesis]

theorem claims_length_eq_invocations
    {Claim ProtocolClaim Proof Receipt : Type}
    {interface : Interface Claim ProtocolClaim Proof Receipt}
    {previous : Claim} (schedule : Schedule interface previous) :
    schedule.claims.length = schedule.invocationCount := by
  induction schedule with
  | terminal event => rfl
  | recursive event rest inductionHypothesis =>
      simp only [claims, invocationCount, List.length_cons]
      omega

theorem trailing_receipt_exact
    {Claim ProtocolClaim Proof Receipt : Type}
    {interface : Interface Claim ProtocolClaim Proof Receipt}
    {previous : Claim} (schedule : Schedule interface previous) :
    interface.receiptClaim schedule.lastReceipt =
      interface.toProtocolClaim schedule.lastClaim := by
  induction schedule with
  | terminal event => exact interface.terminalClaimExact event
  | recursive event rest inductionHypothesis => exact inductionHypothesis

theorem complete_index_schedule
    {Claim ProtocolClaim Proof Receipt : Type}
    {interface : Interface Claim ProtocolClaim Proof Receipt}
    {previous : Claim} (schedule : Schedule interface previous) :
    Lifecycle.CompleteSchedule schedule.claims.length := by
  apply Lifecycle.completeSchedule
  have nonzero : schedule.claims.length ≠ 0 := by
    intro zero
    apply schedule.claims_nonempty
    exact List.length_eq_zero_iff.mp zero
  omega

end Schedule

end Nightstream.Protocol.NebulaV2.ExactDelayedSchedule
