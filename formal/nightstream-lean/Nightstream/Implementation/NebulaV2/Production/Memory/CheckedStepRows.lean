import Nightstream.Implementation.NebulaV2.Memory.Product.BalanceRows
import Nightstream.Implementation.NebulaV2.Memory.Segment.SourceRows
import Nightstream.Implementation.NebulaV2.Memory.Transition.TransitionSound
import Nightstream.Implementation.NebulaV2.Production.Memory.CarryRows
import Nightstream.Implementation.NebulaV2.Production.Memory.ClaimRows

/-!
Contract: exact field-native row program for one production checked step.

The block decodes one mixed memory suffix, derives all 63 operation and 128
snapshot records, updates all eight fingerprint products, applies the exact
F-prime carry transition, and enforces both balance equations only if the
output carry closes.

The input and output carries are decoded by the enclosing batch and are not
duplicated in this row block. No claim, source records, product update,
balance result, or semantic transition is a placement premise.

Does not own application-port routing, batch boundary links, compact lane
chains, state hashing, NIFS verification, or absolute generated columns.

Emits constraints: yes.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductionMemoryCheckedStepRows

open Nightstream.Implementation.NebulaV2.MemoryClaimProductUpdate
open Nightstream.Implementation.NebulaV2.MemoryProductBalanceRows
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.SuperNeo.Concrete

structure Layout where
  before : ProductionMemoryCarryRows.Layout
  claim : ProductionMemoryClaimRows.Layout
  after : ProductionMemoryCarryRows.Layout
  source : MemorySourceRows.Layout
  transition : MemoryTransitionRows.Layout
  balance : MemoryProductBalanceRows.Layout

structure Layout.Valid (layout : Layout) : Prop where
  sourceUsesClaim :
    layout.source.product.claim = layout.claim.reference
  transitionUsesBefore :
    layout.transition.before = layout.before.reference
  transitionUsesClaim :
    layout.transition.claim = layout.claim.reference
  transitionUsesAfter :
    layout.transition.after = layout.after.reference
  balanceUsesClaim :
    layout.balance.claim = layout.claim.reference
  balanceUsesAfterPhase :
    layout.balance.closePhaseColumn =
      layout.after.carry.fieldColumn .phase

def rows (layout : Layout) : List Row :=
  ProductionMemoryClaimRows.rows layout.claim ++
    MemorySourceRows.checkedRows layout.source ++
    MemoryTransitionRows.rows layout.transition ++
    MemoryProductBalanceRows.rows layout.balance

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 27113 := by
  simp [rows, ProductionMemoryClaimRows.rows_length_exact,
    MemorySourceRows.checkedRows_length_exact,
    MemoryTransitionRows.rows_length_exact,
    MemoryProductBalanceRows.rows_length_exact]

private theorem claimRows_hold
    {layout : Layout} {assignment : Nat -> Nat}
    (satisfied : Satisfies (rows layout) assignment) :
    Satisfies (ProductionMemoryClaimRows.rows layout.claim) assignment := by
  intro row member
  exact satisfied row (by simp [rows, member])

private theorem sourceRows_hold
    {layout : Layout} {assignment : Nat -> Nat}
    (satisfied : Satisfies (rows layout) assignment) :
    Satisfies (MemorySourceRows.checkedRows layout.source) assignment := by
  intro row member
  exact satisfied row (by simp [rows, member])

private theorem sourceOnlyRows_hold
    {layout : Layout} {assignment : Nat -> Nat}
    (satisfied : Satisfies (rows layout) assignment) :
    Satisfies (MemorySourceRows.rows layout.source) assignment := by
  have checked := sourceRows_hold satisfied
  intro row member
  exact checked row (List.mem_append_left _ member)

private theorem transitionRows_hold
    {layout : Layout} {assignment : Nat -> Nat}
    (satisfied : Satisfies (rows layout) assignment) :
    Satisfies (MemoryTransitionRows.rows layout.transition) assignment := by
  intro row member
  exact satisfied row (by simp [rows, member])

private theorem balanceRows_hold
    {layout : Layout} {assignment : Nat -> Nat}
    (satisfied : Satisfies (rows layout) assignment) :
    Satisfies (MemoryProductBalanceRows.rows layout.balance) assignment := by
  intro row member
  exact satisfied row (by simp [rows, member])

/-- Proof-independent semantic result for one checked step. -/
structure Result
    (layout : Layout) (assignment : Nat -> Nat)
    (headers : ChainHeaders Digest.Value)
    (before after : MemoryCarryCodec.Value)
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch
      layout.before.reference assignment headers before)
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch
      layout.after.reference assignment headers after) where
  claim : MemoryClaimCodec.Claim
  claimParsed : MemoryClaimRows.ParsedColumnsMatch
    layout.claim.reference assignment claim
  counterWord : forall counter,
    (layout.claim.counters.word counter).digits assignment =
      WasmStateCodec.encodeWord counter.width
        (counter.claimValue claim)
  source : MemorySourceRows.Sound layout.source assignment claim
  productUpdate :
    mapState claim.productsAfter =
      ProductState.update
        Nightstream.Implementation.NebulaV2.ConcreteField.encode
        (mapChallenges claim.challenge)
        (mapState claim.productsBefore) source.records.chunk
  consumes : Consumes MemoryProductBalanceRows.ConcreteBalanced
    (MemoryCarryParser.semanticCarry before
      beforeParsed.parserCanonical.stepIndex)
    claim
    (MemoryCarryParser.semanticCarry after
      afterParsed.parserCanonical.stepIndex)

/-- Derive all semantic results from one satisfying field-native step block. -/
def derive
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat -> Nat}
    (headers : ChainHeaders Digest.Value)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment)
    (before : MemoryCarryCodec.Value)
    (after : MemoryCarryCodec.Value)
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch
      layout.before.reference assignment headers before)
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch
      layout.after.reference assignment headers after) :
    Result layout assignment headers before after beforeParsed afterParsed := by
  let claimResult := ProductionMemoryClaimRows.derive canonical one
    (claimRows_hold satisfied)
  let claim := claimResult.claim
  have sourceParsed : MemoryClaimRows.ParsedColumnsMatch
      layout.source.product.claim assignment claim := by
    rw [valid.sourceUsesClaim]
    exact claimResult.parsed
  let source := MemorySourceRows.sound canonical one sourceParsed
    (sourceOnlyRows_hold satisfied)
  have productUpdate := MemorySourceRows.product_update canonical one
    sourceParsed (sourceRows_hold satisfied) source
  have transitionBefore : MemoryCarryPublicRows.ParsedColumnsMatch
      layout.transition.before assignment headers before := by
    rw [valid.transitionUsesBefore]
    exact beforeParsed
  have transitionClaim : MemoryClaimRows.ParsedColumnsMatch
      layout.transition.claim assignment claim := by
    rw [valid.transitionUsesClaim]
    exact claimResult.parsed
  have transitionAfter : MemoryCarryPublicRows.ParsedColumnsMatch
      layout.transition.after assignment headers after := by
    rw [valid.transitionUsesAfter]
    exact afterParsed
  have balancedOnClose : after.phase = .closed ->
      MemoryProductBalanceRows.ConcreteBalanced claim.productsAfter := by
    intro phaseClosed
    have phaseColumnClosed :
        assignment layout.balance.closePhaseColumn = 0 := by
      rw [valid.balanceUsesAfterPhase]
      have placedPhase := afterParsed.placed .phase
      change assignment (layout.after.carry.fieldColumn .phase) =
        after.fieldValue .phase at placedPhase
      rw [placedPhase]
      simp [MemoryCarryCodec.Value.fieldValue, MemoryCarryCodec.phaseValue,
        phaseClosed]
    have balanceParsed : MemoryClaimRows.ParsedColumnsMatch
        layout.balance.claim assignment claim := by
      rw [valid.balanceUsesClaim]
      exact claimResult.parsed
    exact MemoryProductBalanceRows.parsed_claim_balanced_of_rows one
      phaseColumnClosed balanceParsed (balanceRows_hold satisfied)
  have consumes := MemoryTransitionSound.consumes_of_rows canonical one
    transitionBefore transitionClaim transitionAfter
    (transitionRows_hold satisfied) balancedOnClose
  exact
    { claim := claim
      claimParsed := claimResult.parsed
      counterWord := by
        intro counter
        exact ProductionMemoryClaimRows.counter_digits_eq_decoded
          canonical one (claimRows_hold satisfied) counter
      source := source
      productUpdate := productUpdate
      consumes := consumes }

end Nightstream.Implementation.NebulaV2.ProductionMemoryCheckedStepRows
