import DirectCcsFPrime.Audit.Counterexamples.AggregateChildTableNecessity
import DirectCcsFPrime.ProofSystem.Production.Security.DirectParentOnlyProductionChildMembership
import Mathlib.Tactic

/-!
Second-fold child-swap red-team checks.

These checks model the attack where the first fold is accepted with one private
`CE(b)^14` child table, then the second fold tries to recompute around a
different private child row while keeping only weak aggregate evidence.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionSecondFoldRedTeam

open DecDigitUniqueness

/-- A row-level change in one hidden `CE(b)^14` child table. -/
def RowChanged {n : Nat}
    (row : Fin 14)
    (original mutated : ColumnDigits n) : Prop :=
  ∃ col,
    List.getD (mutated col) row.val 0 ≠
      List.getD (original col) row.val 0

private def firstFoldChildColumn : List Nat :=
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

private def secondFoldMutatedChildColumn : List Nat :=
  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

/-- Child table accepted by the first fold in the weak aggregate model. -/
def firstFoldChildren : ColumnDigits 1 :=
  fun _ => firstFoldChildColumn

/-- Mutated child table used by the adversarial second fold. -/
def secondFoldMutatedChildren : ColumnDigits 1 :=
  fun _ => secondFoldMutatedChildColumn

private def oneSummary : Fin 1 → Nat :=
  fun _ => 1

private theorem firstFoldChildren_binary :
    binaryColumnDigits firstFoldChildren := by
  intro j d hd
  fin_cases j
  simp [firstFoldChildren, firstFoldChildColumn] at hd
  omega

private theorem secondFoldMutatedChildren_binary :
    binaryColumnDigits secondFoldMutatedChildren := by
  intro j d hd
  fin_cases j
  simp [secondFoldMutatedChildren, secondFoldMutatedChildColumn] at hd
  omega

private theorem firstFoldChildren_length14 :
    BinaryChildTableAuthorization.fixedColumnLength 14 firstFoldChildren := by
  intro j
  fin_cases j
  rfl

private theorem secondFoldMutatedChildren_length14 :
    BinaryChildTableAuthorization.fixedColumnLength
      14
      secondFoldMutatedChildren := by
  intro j
  fin_cases j
  rfl

private theorem firstFoldChildren_aggregate :
    AggregateChildTableNecessity.aggregateDigitSum firstFoldChildren =
      oneSummary := by
  funext j
  fin_cases j
  rfl

private theorem secondFoldMutatedChildren_aggregate :
    AggregateChildTableNecessity.aggregateDigitSum
        secondFoldMutatedChildren =
      oneSummary := by
  funext j
  fin_cases j
  rfl

private theorem firstFoldChildren_weakValid :
    AggregateChildTableNecessity.AggregateOnlyChildValidation
      14
      oneSummary
      firstFoldChildren :=
  ⟨firstFoldChildren_binary,
    firstFoldChildren_length14,
    firstFoldChildren_aggregate⟩

private theorem secondFoldMutatedChildren_weakValid :
    AggregateChildTableNecessity.AggregateOnlyChildValidation
      14
      oneSummary
      secondFoldMutatedChildren :=
  ⟨secondFoldMutatedChildren_binary,
    secondFoldMutatedChildren_length14,
    secondFoldMutatedChildren_aggregate⟩

/--
Weak aggregate-only authorization for the toy second-fold model.

This deliberately forgets the pointwise base-2 recomposition and CE membership
facts, so it is not a sound replacement for production private `Pi_DEC`.
-/
def WeakAggregateAuthorized
    (_source : Nat)
    (children : ColumnDigits 1) : Prop :=
  AggregateChildTableNecessity.AggregateOnlyChildValidation
    14
    oneSummary
    children

/-- Toy parent source recomputed from the hidden child table. -/
def weakParentSource (children : ColumnDigits 1) : Nat :=
  recomposeNatDigits (children ⟨0, by decide⟩)

/-- Self-consistent toy stage that recomputes the next parent from children. -/
def WeakParentSourceStep
    (_i : Nat)
    (_prior : ParentOnlyAccumulatorStep.AccumulatorHandle Nat)
    (children : ColumnDigits 1)
    (source : Nat) : Prop :=
  source = weakParentSource children

/-- The concrete second fold changes child row `0` from the first fold table. -/
theorem concreteSecondFoldChangesAChildRow :
    RowChanged
      ⟨0, by decide⟩
      firstFoldChildren
      secondFoldMutatedChildren := by
  refine ⟨⟨0, by decide⟩, ?_⟩
  native_decide

/--
Red-team: aggregate-only validation admits a self-consistent second-fold child
swap.

Both folds satisfy the weak aggregate predicate and each next parent source is
recomputed from the supplied table. The next parent sources differ because the
second fold moved the hot child row from slot `0` to slot `1`.
-/
theorem weakAggregateAllowsSelfConsistentSecondFoldSwap :
    ∃ prior nextA nextB :
      ParentOnlyAccumulatorStep.AccumulatorHandle Nat,
      ParentOnlyAccumulatorStep.Step
          WeakAggregateAuthorized
          WeakParentSourceStep
          1
          prior
          nextA ∧
        ParentOnlyAccumulatorStep.Step
          WeakAggregateAuthorized
          WeakParentSourceStep
          2
          prior
          nextB ∧
        nextA.parentSource = weakParentSource firstFoldChildren ∧
        nextB.parentSource = weakParentSource secondFoldMutatedChildren ∧
        nextA.parentSource ≠ nextB.parentSource ∧
        RowChanged
          ⟨0, by decide⟩
          firstFoldChildren
          secondFoldMutatedChildren := by
  let prior : ParentOnlyAccumulatorStep.AccumulatorHandle Nat :=
    { parentSource := 0 }
  let nextA : ParentOnlyAccumulatorStep.AccumulatorHandle Nat :=
    { parentSource := weakParentSource firstFoldChildren }
  let nextB : ParentOnlyAccumulatorStep.AccumulatorHandle Nat :=
    { parentSource := weakParentSource secondFoldMutatedChildren }
  refine ⟨prior, nextA, nextB, ?_, ?_, rfl, rfl, ?_, concreteSecondFoldChangesAChildRow⟩
  · refine ⟨firstFoldChildren, firstFoldChildren_weakValid, ?_⟩
    rfl
  · refine ⟨secondFoldMutatedChildren, secondFoldMutatedChildren_weakValid, ?_⟩
    rfl
  · native_decide

private theorem table_ne_of_rowChanged
    {n : Nat}
    {row : Fin 14}
    {original mutated : ColumnDigits n}
    (hChanged : RowChanged row original mutated) :
    mutated ≠ original := by
  intro hEq
  rcases hChanged with ⟨col, hDiff⟩
  subst mutated
  exact hDiff rfl

/-- Production pointwise private-DEC requirement for the parent-only context. -/
abbrev PointwiseReq
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params)
    (source : DigestParentBinding.Source Digest)
    (inputs : ColumnDigits n) : Prop :=
  ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
    (n := n)
    (hashEncoded := ctx.parentHash.hashEncoded)
    (params := params)
    (ce := ctx.data.ce)
    (StatementEncodes :=
      ParentOpeningAuthorization.StatementEncodesByCommitment
        ctx.commitmentOfParent)
    source
    inputs

/--
Production red-team guard: after one accepted terminal child audit fixes the
private `CE(b)^14` table, no different second-fold table can also satisfy the
full pointwise private-DEC requirement for the same parent source.
-/
theorem terminalAuditRejectsSecondFoldSwap
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params}
    {priorSteps : Nat}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hAudit :
      DirectParentOnlyProductionChildMembership.TerminalChildAuditTrail
        ctx
        priorSteps
        priorImage
        nextImage
        altNext) :
    ∃ priorInputs,
      PointwiseReq
        ctx
        priorImage.accumulator.parentSource
        priorInputs ∧
      ∀ mutatedInputs,
        mutatedInputs ≠ priorInputs →
          ¬ PointwiseReq
              ctx
              priorImage.accumulator.parentSource
              mutatedInputs := by
  rcases hAudit with
    ⟨priorInputs, hPointwise, _hChildAudit, _hNext, _hAlt, hUnique⟩
  refine ⟨priorInputs, hPointwise, ?_⟩
  intro mutatedInputs hDifferent hMutated
  exact hDifferent (hUnique mutatedInputs hMutated)

/--
Production red-team guard in the exact attack shape: a second fold that changes
even one private child row cannot pass the full pointwise private-DEC
requirement for the same parent source.
-/
theorem terminalAuditRejectsChangedChildRow
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ctx : DirectParentOnlyProductionSoundness.Context Digest Boundary n params}
    {priorSteps : Nat}
    {priorImage nextImage altNext :
      DirectParentOnlyProductionSoundness.PublicImage Digest Boundary}
    (hAudit :
      DirectParentOnlyProductionChildMembership.TerminalChildAuditTrail
        ctx
        priorSteps
        priorImage
        nextImage
        altNext) :
    ∃ priorInputs,
      PointwiseReq
        ctx
        priorImage.accumulator.parentSource
        priorInputs ∧
      ∀ mutatedInputs row,
        RowChanged row priorInputs mutatedInputs →
          ¬ PointwiseReq
              ctx
              priorImage.accumulator.parentSource
              mutatedInputs := by
  rcases terminalAuditRejectsSecondFoldSwap hAudit with
    ⟨priorInputs, hPointwise, hRejectDifferent⟩
  refine ⟨priorInputs, hPointwise, ?_⟩
  intro mutatedInputs row hChanged hMutated
  exact
    hRejectDifferent
      mutatedInputs
      (table_ne_of_rowChanged hChanged)
      hMutated

end DirectParentOnlyProductionSecondFoldRedTeam

end DirectCcsFPrime
