import Nightstream.Implementation.NebulaV2.MemoryProductBalanceBridge
import Nightstream.Implementation.NebulaV2.ProductionMemoryStepSemantics

/-!
Contract: compose every row-derived product update in one production memory
step run.

The result starts from the exact active carry products, follows each checked
step in list order, and derives balance only from the final checked close.
Neither aggregate products nor fingerprint acceptance are premises.

Does not own complete snapshot coverage, challenge probability, root binding,
application-row alignment, or deployed-verifier extraction.

Assurance tier: implementation-to-protocol bridge.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductionMemoryProductAccumulation

open Nightstream.Implementation.NebulaV2.ConcreteField
open Nightstream.Implementation.NebulaV2.MemoryClaimProductUpdate
open Nightstream.Implementation.NebulaV2.MemoryProductBalanceBridge
open Nightstream.Implementation.NebulaV2.MemoryProductBalanceRows
open Nightstream.Implementation.NebulaV2.ProductionMemoryStepSemantics
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.SuperNeo.Concrete

namespace Run

private theorem empty_eq
    {before after : ConcreteCarry}
    (run : ProductionMemoryStepSemantics.Run before [] after) :
    before = after := by
  cases run
  rfl

private theorem activeWellFormedOfCons
    {before after : ConcreteCarry}
    {active : ActiveCarry Digest.Value (Challenges K) (State K)}
    {head : Step} {tail : List Step}
    (run : ProductionMemoryStepSemantics.Run before (head :: tail) after)
    (beforeExact : before = .active active) :
    active.WellFormed := by
  cases run with
  | cons _ _ =>
      have consumption := head.consumes
      rw [beforeExact] at consumption
      exact consumption.activeWellFormed

/-- Reaching a close from an active carry proves the complete opening-carry
well-formedness predicate from the first row-derived claim. -/
theorem activeWellFormed
    {active : ActiveCarry Digest.Value (Challenges K) (State K)}
    {closed : ClosedCarry Digest.Value} {checked : List Step}
    (run : ProductionMemoryStepSemantics.Run (.active active) checked
      (.closed closed)) :
    active.WellFormed := by
  have nonempty : checked ≠ [] := by
    intro empty
    subst checked
    have impossible :
        (Carry.active active : ConcreteCarry) = .closed closed :=
      empty_eq run
    cases impossible
  obtain ⟨head, tail, rfl⟩ := List.exists_cons_of_ne_nil nonempty
  exact activeWellFormedOfCons run rfl

private theorem consumeActiveCases
    {active : ActiveCarry Digest.Value (Challenges K) (State K)}
    {claim : ClaimSuffix Digest.Value (Challenges K) (State K)}
    {after : ConcreteCarry}
    (consumption : Consumes ConcreteBalanced (.active active) claim after) :
    (∃ (_ : MatchesActive active claim)
        (notLast : active.stepIndex.val + 1 < Lifecycle.claimsPerSegment),
        after = .active (interiorCarry active claim notLast)) ∨
      (∃ (_ : MatchesActive active claim)
        (_ : active.stepIndex.val + 1 = Lifecycle.claimsPerSegment)
        (_ : CloseChecks ConcreteBalanced active claim),
        after = .closed (closedCarryAfter active claim)) := by
  cases consumption with
  | interior agreement notLast =>
      exact Or.inl ⟨agreement, notLast, rfl⟩
  | close agreement last checks =>
      exact Or.inr ⟨agreement, last, checks, rfl⟩

private theorem accumulatedProductsBalancedAux
    {before after : ConcreteCarry} {checked : List Step}
    (run : ProductionMemoryStepSemantics.Run before checked after)
    (beforeActive : ∃ active, before = .active active)
    (afterClosed : ∃ closed, after = .closed closed) :
    ∀ active, before = .active active →
      ProductState.Balanced
        (ProductState.accumulate encode (mapChallenges active.challenge)
          (mapState active.products)
          (ProductionMemoryStepSemantics.Run.chunks checked)) := by
  induction run with
  | nil =>
      intro active beforeExact
      rcases afterClosed with ⟨closed, afterExact⟩
      rw [beforeExact] at afterExact
      cases afterExact
  | @cons tail final head rest inductionHypothesis =>
      intro requestedActive beforeExact
      have update := head.productUpdate
      have consumption : Consumes ConcreteBalanced
          (.active requestedActive) head.claim head.after := by
        rw [← beforeExact]
        exact head.consumes
      rcases consumeActiveCases consumption with
        ⟨agreement, notLast, afterExact⟩ |
        ⟨agreement, last, checks, afterExact⟩
      · have tailBalanced := inductionHypothesis
          ⟨_, afterExact⟩ afterClosed _ afterExact
        simpa [ProductionMemoryStepSemantics.Run.chunks,
          ProductState.accumulate, interiorCarry, agreement.challenge,
          agreement.products, update] using tailBalanced
      · have closedRest : ProductionMemoryStepSemantics.Run
            (.closed (closedCarryAfter requestedActive head.claim)) tail
            final := by
          rw [← afterExact]
          exact rest
        have tailEmpty := closedRest.fromClosedIsEmpty
        have mappedBalanced :=
          (concreteBalanced_iff_mapped _).mp checks.productsBalanced
        change ProductState.Balanced
          (ProductState.accumulate encode
            (mapChallenges requestedActive.challenge)
            (mapState requestedActive.products)
            (head.records.chunk ::
              ProductionMemoryStepSemantics.Run.chunks tail))
        rw [ProductState.accumulate]
        have headExact :
            ProductState.update encode
                (mapChallenges requestedActive.challenge)
                (mapState requestedActive.products) head.records.chunk =
              mapState head.claim.productsAfter := by
          rw [← agreement.challenge, ← agreement.products]
          exact update.symm
        rw [headExact, tailEmpty.1]
        simpa [ProductionMemoryStepSemantics.Run.chunks,
          ProductState.accumulate] using mappedBalanced

/-- All row-derived product updates form one balanced aggregate when the run
starts active and reaches the checked close. -/
theorem accumulatedProductsBalanced
    {active : ActiveCarry Digest.Value (Challenges K) (State K)}
    {closed : ClosedCarry Digest.Value} {checked : List Step}
    (run : ProductionMemoryStepSemantics.Run (.active active) checked
      (.closed closed)) :
    ProductState.Balanced
      (ProductState.accumulate encode (mapChallenges active.challenge)
        (mapState active.products)
        (ProductionMemoryStepSemantics.Run.chunks checked)) :=
  accumulatedProductsBalancedAux run ⟨active, rfl⟩ ⟨closed, rfl⟩
    active rfl

/-- The concrete all-one opening makes the aggregate start from the exact
mathematical identity state. -/
theorem accumulatedFromConcreteOneBalanced
    {active : ActiveCarry Digest.Value (Challenges K) (State K)}
    {closed : ClosedCarry Digest.Value} {checked : List Step}
    (run : ProductionMemoryStepSemantics.Run (.active active) checked
      (.closed closed))
    (openingProducts : active.products = MemoryCarryCodec.oneProductsK) :
    ProductState.Balanced
      (ProductState.accumulate encode (mapChallenges active.challenge)
        ProductState.one
        (ProductionMemoryStepSemantics.Run.chunks checked)) := by
  have balanced := accumulatedProductsBalanced run
  rw [openingProducts, mapState_oneProductsK] at balanced
  exact balanced

end Run

end Nightstream.Implementation.NebulaV2.ProductionMemoryProductAccumulation
