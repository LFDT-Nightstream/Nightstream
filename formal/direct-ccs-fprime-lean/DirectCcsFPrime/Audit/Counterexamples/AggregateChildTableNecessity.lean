import DirectCcsFPrime.ProofSystem.PrivatePiDec.Security.BinaryChildTableAuthorization
import Mathlib.Tactic

/-!
Necessity of pointwise private-child validation.

This module records a concrete counterexample to aggregate-only validation of
private post-DEC child tables. Binary fixed-length child columns can preserve a
summary such as "sum of child digits" while changing the actual child table and
the per-column base-2 recomposition consumed by the next `Pi_CCS` stage.
-/

namespace DirectCcsFPrime

namespace AggregateChildTableNecessity

open DecDigitUniqueness
open BinaryChildTableAuthorization

private def firstChildHot : List Nat :=
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

private def secondChildHot : List Nat :=
  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

private def firstTable : ColumnDigits 1 :=
  fun _ => firstChildHot

private def secondTable : ColumnDigits 1 :=
  fun _ => secondChildHot

private def oneSummary : Fin 1 → Nat :=
  fun _ => 1

private def firstNorms : Fin 14 → Nat :=
  fun i => if i = ⟨0, by decide⟩ then 1 else 0

private def secondNorms : Fin 14 → Nat :=
  fun i => if i = ⟨1, by decide⟩ then 1 else 0

/--
Aggregate digit summary for a child table.

This is intentionally weaker than DEC recomposition: it forgets which child row
contains each bit and therefore cannot authorize the next `Pi_CCS` inputs.
-/
def aggregateDigitSum {n : Nat} (children : ColumnDigits n) : Fin n → Nat :=
  fun j => (children j).sum

/--
Aggregate norm summary for a fixed child vector.

This models the tempting but unsound validation shape "the child norms add up
to the expected total". It intentionally forgets which child carried which
norm and therefore cannot authorize the private post-DEC child identities.
-/
def aggregateNormSum {k : Nat} (norms : Fin k → Nat) : Nat :=
  (List.ofFn norms).sum

/--
An intentionally weak aggregate-only validation predicate.

It checks binary digits, exact column length, and an aggregate per-column sum.
It does not check per-column base-2 recomposition and therefore is not a sound
replacement for `PointwisePrivateDecRequirements`.
-/
def AggregateOnlyChildValidation
    {n : Nat}
    (k : Nat)
    (summary : Fin n → Nat)
    (children : ColumnDigits n) : Prop :=
  binaryColumnDigits children ∧
    fixedColumnLength k children ∧
    aggregateDigitSum children = summary

/-- Accepted aggregate-only validation wired into a next `Pi_CCS` input table. -/
structure AcceptedAggregateOnlyChildTable
    {n : Nat}
    (k : Nat)
    (summary : Fin n → Nat)
    (children nextInputs : ColumnDigits n) : Prop where
  proofVerified :
    AggregateOnlyChildValidation k summary children
  wireIdentity : nextInputs = children

private theorem firstTable_binary :
    binaryColumnDigits firstTable := by
  intro j d hd
  fin_cases j
  simp [firstTable, firstChildHot] at hd
  omega

private theorem secondTable_binary :
    binaryColumnDigits secondTable := by
  intro j d hd
  fin_cases j
  simp [secondTable, secondChildHot] at hd
  omega

private theorem firstTable_fixedLength :
    fixedColumnLength 14 firstTable := by
  intro j
  fin_cases j
  rfl

private theorem secondTable_fixedLength :
    fixedColumnLength 14 secondTable := by
  intro j
  fin_cases j
  rfl

private theorem firstTable_aggregate :
    aggregateDigitSum firstTable = oneSummary := by
  funext j
  fin_cases j
  rfl

private theorem secondTable_aggregate :
    aggregateDigitSum secondTable = oneSummary := by
  funext j
  fin_cases j
  rfl

private theorem firstTable_valid :
    AggregateOnlyChildValidation 14 oneSummary firstTable :=
  ⟨firstTable_binary, firstTable_fixedLength, firstTable_aggregate⟩

private theorem secondTable_valid :
    AggregateOnlyChildValidation 14 oneSummary secondTable :=
  ⟨secondTable_binary, secondTable_fixedLength, secondTable_aggregate⟩

private theorem firstTable_ne_secondTable :
    firstTable ≠ secondTable := by
  intro h
  have hCol := congrFun h ⟨0, by decide⟩
  simp [firstTable, secondTable, firstChildHot, secondChildHot] at hCol

private theorem firstTable_recompose_ne_secondTable :
    recomposeColumns firstTable ≠ recomposeColumns secondTable := by
  intro h
  have hCol := congrFun h ⟨0, by decide⟩
  norm_num
    [recomposeColumns, firstTable, secondTable, firstChildHot,
      secondChildHot, recomposeNatDigits] at hCol

private theorem firstNorms_sum_eq_secondNorms :
    aggregateNormSum firstNorms = aggregateNormSum secondNorms := by
  native_decide

private theorem firstNorms_ne_secondNorms :
    firstNorms ≠ secondNorms := by
  intro h
  have h0 := congrFun h ⟨0, by decide⟩
  norm_num [firstNorms, secondNorms] at h0

/--
Aggregate digit sums are not functional, even with binary fixed-length
columns.

The two tables below both have one `1` bit in their only column and length 14,
but the `1` appears in a different child row.
-/
theorem aggregate_digit_sum_not_functional_for_binary_fixed_length :
    ¬ (∀ a b : ColumnDigits 1,
      binaryColumnDigits a →
      binaryColumnDigits b →
      fixedColumnLength 14 a →
      fixedColumnLength 14 b →
      aggregateDigitSum a = aggregateDigitSum b →
        a = b) := by
  intro hFunctional
  exact firstTable_ne_secondTable
    (hFunctional
      firstTable
      secondTable
      firstTable_binary
      secondTable_binary
      firstTable_fixedLength
      secondTable_fixedLength
      (firstTable_aggregate.trans secondTable_aggregate.symm))

/--
Aggregate child-norm sums are not functional for the length-14 DEC child
vector.

The two norm vectors below both have total norm `1`, but the nonzero norm sits
on a different child. A verifier that only checks the aggregate norm total
therefore does not bind the child identities.
-/
theorem aggregate_norm_sum_not_functional_for_fixed_child_count :
    ¬ (∀ a b : Fin 14 → Nat,
      aggregateNormSum a = aggregateNormSum b →
        a = b) := by
  intro hFunctional
  exact firstNorms_ne_secondNorms
    (hFunctional firstNorms secondNorms firstNorms_sum_eq_secondNorms)

/--
Aggregate-only validation can accept two different next `Pi_CCS` child inputs.

This is the concrete adversarial shape the production theorem avoids by
requiring pointwise DEC recomposition, child CE membership, and wire identity.
-/
theorem aggregate_only_validation_can_feed_different_next_inputs :
    AcceptedAggregateOnlyChildTable
        14
        oneSummary
        firstTable
        firstTable ∧
      AcceptedAggregateOnlyChildTable
        14
        oneSummary
        secondTable
        secondTable ∧
      aggregateDigitSum firstTable = aggregateDigitSum secondTable ∧
      firstTable ≠ secondTable ∧
      recomposeColumns firstTable ≠ recomposeColumns secondTable := by
  constructor
  · exact
      { proofVerified := firstTable_valid
        wireIdentity := rfl }
  constructor
  · exact
      { proofVerified := secondTable_valid
        wireIdentity := rfl }
  exact
    ⟨firstTable_aggregate.trans secondTable_aggregate.symm,
      firstTable_ne_secondTable,
      firstTable_recompose_ne_secondTable⟩

end AggregateChildTableNecessity

end DirectCcsFPrime
