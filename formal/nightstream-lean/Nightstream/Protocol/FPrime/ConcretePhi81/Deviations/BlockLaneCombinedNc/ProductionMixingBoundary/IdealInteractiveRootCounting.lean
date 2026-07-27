import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CoefficientRootCounting
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MultilinearRootCounting
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Reindex
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Rejection

/-!
Finite-root probability adapters for the production ideal-interactive carrier.

Assurance tier: model-level mathematical infrastructure.

Owns: conversion of the existing exact root-count theorems into rational
probability bounds over one explicit support, and a Cartesian component
averaging lemma used to retain the production challenge order.

Does not own: a protocol event, Split-NC semantics, Fiat--Shamir, Poseidon2,
field certificates, Rust/R1CS, artifacts, costs, or rows.

Emits constraints: no.

| Boundary | Owned equation |
| --- | --- |
| Roots | Finite root counts become exact rational support bounds |
| Products | Component averaging preserves Cartesian event probability |
| Ordering | Reindexing changes enumeration only, not the sampled law |
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveRootCounting

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.SumCheck.Finite

universe uField uPrefix uInner

/-- Every ratio used below is nonnegative when its sampled denominator is a
support cardinality. -/
theorem ratio_nonneg
    (numerator : Nat)
    {denominator : Nat}
    (positive : 0 < denominator) :
    0 <= ratio numerator denominator := by
  unfold ratio
  rw [Rat.div_def]
  exact Rat.mul_nonneg Rat.natCast_nonneg
    (Rat.le_of_lt (Rat.inv_pos.mpr (Rat.natCast_pos.mpr positive)))

/-- Uniform Cartesian products may be transposed for probability arithmetic.
This changes enumeration order only; the event receives the correspondingly
swapped pair. -/
theorem product_swap_probabilityBool
    {Prefix : Type uPrefix}
    {Inner : Type uInner}
    (left : Support Prefix)
    (right : Support Inner)
    (event : Prefix -> Inner -> Bool) :
    ((left.product right).uniform).probabilityBool
        (fun seed => event seed.1 seed.2) =
      ((right.product left).uniform).probabilityBool
        (fun seed => event seed.2 seed.1) := by
  let forward : Prefix × Inner -> Inner × Prefix :=
    fun seed => (seed.2, seed.1)
  have supportPermutation :
      ((left.product right).values.map forward).Perm
        (right.product left).values := by
    apply Support.map_values_perm_of_inverse
      (left.product right) (right.product left)
      forward (fun seed => (seed.2, seed.1))
    · intro seed
      rfl
    · intro seed
      rfl
    · intro seed
      rw [Support.mem_product_iff, Support.mem_product_iff]
      exact and_comm
  exact Experiment.probabilityBool_eq_of_reindex
    (left.product right).uniform (right.product left).uniform
    forward supportPermutation
    (fun seed => event seed.1 seed.2)
    (fun seed => event seed.2 seed.1)
    (by intro seed _member; rfl)

/-- A uniform component bound survives the lexicographic Cartesian product.
The inner event may depend on the complete outer prefix. -/
theorem product_probabilityBool_le_of_components
    {Prefix : Type uPrefix}
    {Inner : Type uInner}
    (prefixes : Support Prefix)
    (inner : Support Inner)
    (event : Prefix -> Inner -> Bool)
    (bound : Rat)
    (componentBound : forall outer,
      outer ∈ prefixes.values ->
        inner.uniform.probabilityBool (event outer) <= bound) :
    ((prefixes.product inner).uniform).probabilityBool
        (fun seed => event seed.1 seed.2) <= bound := by
  let mixture : Mixture Prefix (Prefix × Inner) := {
    prefixes := prefixes
    component := fun outer => {
      Seed := Inner
      support := inner
      outcome := fun innerSeed => (outer, innerSeed)
    }
  }
  have averaged :
      mixture.probability
          (fun seed => event seed.1 seed.2 = true) <= bound := by
    apply Mixture.probability_le_of_components
    intro outer member
    rw [(mixture.component outer).probability_bool_event]
    exact componentBound outer member
  calc
    ((prefixes.product inner).uniform).probabilityBool
          (fun seed => event seed.1 seed.2) =
        mixture.probabilityBool
          (fun seed => event seed.1 seed.2) := by
            symm
            exact Mixture.sharedSupport_probabilityBool_eq_product
              prefixes inner (fun outer innerSeed => (outer, innerSeed))
                (fun seed => event seed.1 seed.2)
    _ = mixture.probability
          (fun seed => event seed.1 seed.2 = true) :=
      (mixture.probability_bool_event _).symm
    _ <= bound := averaged

/-- Direct finite probability form of the canonical multilinear root bound. -/
theorem multilinearZero_probability_le
    {Field : Type uField}
    [DecidableEq Field]
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    {variables : Nat}
    (table : BooleanTable Field variables)
    (alphabet : Support Field)
    (nonzero : Not (table.AllEntriesZero ops)) :
    ((Support.challengeVectors alphabet variables).uniform).probabilityBool
        (fun word =>
          decide
            (table.evaluateCoordinates ops (List.ofFn word) = ops.zero)) <=
      ratio variables alphabet.cardinality := by
  let words := Support.challengeVectors alphabet variables
  have countBound :=
    MultilinearRootCounting.zeros_count_le ops laws noZeroDivisors
      table alphabet.values alphabet.nodup nonzero
  have denominatorPos :
      0 < ((words.cardinality : Nat) : Rat) :=
    Rat.natCast_pos.mpr words.cardinality_pos
  unfold Experiment.probabilityBool Experiment.countBool
  apply (div_le_iff_of_pos denominatorPos).2
  have castCountBound :
      ((words.values.countP (fun word =>
          decide
            (table.evaluateCoordinates ops (List.ofFn word) = ops.zero)) :
          Nat) : Rat) <=
        ((variables * alphabet.cardinality ^ variables.pred : Nat) : Rat) := by
    exact Rat.natCast_le_natCast.mpr (by
      simpa [words, Support.challengeVectors_values, Support.cardinality]
        using countBound)
  refine Rat.le_trans castCountBound ?_
  have alphabetNonzero : (alphabet.cardinality : Rat) ≠ 0 :=
    Rat.ne_of_gt (Rat.natCast_pos.mpr alphabet.cardinality_pos)
  change
    ((variables * alphabet.cardinality ^ variables.pred : Nat) : Rat) <=
      ratio variables alphabet.cardinality * (words.cardinality : Rat)
  rw [Support.challengeVectors_cardinality]
  cases variables with
  | zero =>
      simpa [ratio, Rat.div_def] using (Rat.le_refl (0 : Rat))
  | succ prior =>
      have exactValue :
          (((Nat.succ prior *
              alphabet.cardinality ^ prior : Nat) : Rat)) =
            ratio (Nat.succ prior) alphabet.cardinality *
              ((alphabet.cardinality ^ Nat.succ prior : Nat) : Rat) := by
        unfold ratio
        simp only [Nat.pow_succ, Rat.natCast_mul, Rat.natCast_pow]
        calc
          ((Nat.succ prior : Rat) *
              (alphabet.cardinality : Rat) ^ prior) =
            (((Nat.succ prior : Rat) /
                (alphabet.cardinality : Rat)) *
              (alphabet.cardinality : Rat)) *
                (alphabet.cardinality : Rat) ^ prior := by
                  rw [Rat.div_mul_cancel alphabetNonzero]
          _ = ((Nat.succ prior : Rat) /
                (alphabet.cardinality : Rat)) *
              ((alphabet.cardinality : Rat) ^ prior *
                (alphabet.cardinality : Rat)) := by
                  rw [Rat.mul_assoc]
                  rw [Rat.mul_comm
                    (alphabet.cardinality : Rat)
                    ((alphabet.cardinality : Rat) ^ prior)]
      calc
        (((prior + 1) *
            alphabet.cardinality ^ (prior + 1).pred : Nat) : Rat) =
          ratio (prior + 1) alphabet.cardinality *
            ((alphabet.cardinality ^ (prior + 1) : Nat) : Rat) := by
              simpa only [Nat.pred_succ] using exactValue
        _ <= _ := Rat.le_refl

/-- Direct finite probability form of root counting for one constant-first
coefficient list. -/
theorem coefficientZero_probability_le
    {Field : Type uField}
    [DecidableEq Field]
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (degree : Nat)
    (coefficients : List Field)
    (coefficientCount : coefficients.length = degree + 1)
    (alphabet : Support Field)
    (nonzero : Not (CoefficientRootCounting.AllZero ops coefficients)) :
    alphabet.uniform.probabilityBool
        (fun point =>
          decide
            (Message.evaluateCoefficients ops.toOps point coefficients =
              ops.zero)) <=
      ratio degree alphabet.cardinality := by
  have countBound :=
    CoefficientRootCounting.roots_count_le_degree ops laws noZeroDivisors
      degree coefficients coefficientCount alphabet.values alphabet.nodup
      nonzero
  unfold Experiment.probabilityBool Experiment.countBool ratio
  exact div_le_div_of_le
    (Rat.natCast_le_natCast.mpr countBound)
    (Rat.natCast_pos.mpr alphabet.cardinality_pos)

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveRootCounting
