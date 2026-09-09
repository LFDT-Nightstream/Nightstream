import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NumericBooleanDomain
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
import Mathlib.Data.List.FinRange

/-!
Owns a numeric sum loop and its equality with the canonical Boolean sum.
The loop allocates no list of indices. A zero suffix may be omitted only
under the explicit range and zero conditions in the prefix theorem.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NumericCompletionSum

universe uField uIndex

variable {Field : Type uField}

/-- Sum an ascending numeric prefix without an index-list allocation. -/
def numericSum (ops : InterpolationOps Field)
    (count : Nat) (term : Nat → Field) : Field :=
  Nat.fold count (fun index _ accumulated =>
    ops.add accumulated (term index)) ops.zero

private theorem numericSum_succ (ops : InterpolationOps Field)
    (count : Nat) (term : Nat → Field) :
    numericSum ops (count + 1) term =
      ops.add (numericSum ops count term) (term count) := by
  simp only [numericSum, Nat.fold_succ]

private theorem foldl_eq_add_sumMap
    {Index : Type uIndex}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (indices : List Index) (term : Index → Field) (initial : Field) :
    indices.foldl (fun accumulated index =>
        ops.add accumulated (term index)) initial =
      ops.add initial (FiniteSumAlgebra.sumMap ops indices term) := by
  induction indices generalizing initial with
  | nil => exact (laws.add_zero initial).symm
  | cons index indices inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis]
      exact laws.add_assoc initial (term index) _

private theorem numericVertices_perm (arity : Nat) :
    ((List.finRange (2 ^ arity)).map (NumericBooleanDomain.vertex arity)).Perm
      (BooleanVertex.all arity) := by
  have injective : Function.Injective (NumericBooleanDomain.vertex arity) := by
    intro left right equal
    apply Fin.ext
    have indexed := congrArg NumericBooleanDomain.index equal
    simpa only [NumericBooleanDomain.index_vertex] using indexed
  apply (List.perm_ext_iff_of_nodup
    ((List.nodup_finRange (2 ^ arity)).map injective)
    (BooleanVertex.all_nodup arity)).2
  intro point
  constructor
  · intro _
    exact BooleanVertex.mem_all point
  · intro _
    exact List.mem_map.mpr
      ⟨⟨NumericBooleanDomain.index point,
          NumericBooleanDomain.index_lt_twoPow point⟩,
        List.mem_finRange _, NumericBooleanDomain.vertex_index point⟩

/-- The full numeric sum visits every canonical Boolean vertex exactly once.
The term is arbitrary; its values need not be zero or satisfy a relation. -/
theorem numericSum_eq_vertexSum (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (arity : Nat) (term : Nat → Field) :
    numericSum ops (2 ^ arity) term =
      FiniteSumAlgebra.sumMap ops (BooleanVertex.all arity)
        (fun vertex => term (NumericBooleanDomain.index vertex)) := by
  unfold numericSum
  rw [Nat.fold_eq_finRange_foldl]
  calc
    _ = ((List.finRange (2 ^ arity)).map (NumericBooleanDomain.vertex arity)).foldl
        (fun accumulated vertex =>
          ops.add accumulated (term (NumericBooleanDomain.index vertex)))
        ops.zero := by
      rw [List.foldl_map]
      simp only [NumericBooleanDomain.index_vertex]
    _ = (BooleanVertex.all arity).foldl
        (fun accumulated vertex =>
          ops.add accumulated (term (NumericBooleanDomain.index vertex)))
        ops.zero := by
      apply (numericVertices_perm arity).foldl_eq'
      intro left _ right _ accumulated
      change ops.add (ops.add accumulated (term (NumericBooleanDomain.index left)))
          (term (NumericBooleanDomain.index right)) =
        ops.add (ops.add accumulated (term (NumericBooleanDomain.index right)))
          (term (NumericBooleanDomain.index left))
      rw [laws.add_assoc, laws.add_assoc,
        laws.add_comm (term (NumericBooleanDomain.index left))
          (term (NumericBooleanDomain.index right))]
    _ = _ := by
      rw [foldl_eq_add_sumMap ops laws, laws.zero_add]

private theorem numericSum_eq_of_zero_suffix (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (prefixCount fullCount : Nat) (term : Nat → Field)
    (fits : prefixCount ≤ fullCount)
    (outsideZero : ∀ index, prefixCount ≤ index → index < fullCount →
      term index = ops.zero) :
    numericSum ops prefixCount term = numericSum ops fullCount term := by
  induction fullCount generalizing prefixCount with
  | zero =>
      have prefixZero : prefixCount = 0 := by omega
      subst prefixCount
      rfl
  | succ fullCount inductionHypothesis =>
      by_cases equal : prefixCount = fullCount + 1
      · rw [equal]
      · have prefixFits : prefixCount ≤ fullCount := by omega
        rw [numericSum_succ, outsideZero fullCount prefixFits (by omega), laws.add_zero]
        exact inductionHypothesis prefixCount prefixFits
          (fun index lower upper => outsideZero index lower (by omega))

/-- A numeric prefix equals the full Boolean sum when every omitted in-domain
term is zero. No condition is imposed beyond the Boolean domain. -/
theorem numericSum_prefix_eq_vertexSum (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (arity prefixCount : Nat) (term : Nat → Field)
    (fits : prefixCount ≤ 2 ^ arity)
    (outsideZero : ∀ index, prefixCount ≤ index → index < 2 ^ arity →
      term index = ops.zero) :
    numericSum ops prefixCount term =
      FiniteSumAlgebra.sumMap ops (BooleanVertex.all arity)
        (fun vertex => term (NumericBooleanDomain.index vertex)) := by
  exact (numericSum_eq_of_zero_suffix ops laws prefixCount (2 ^ arity)
    term fits outsideZero).trans (numericSum_eq_vertexSum ops laws arity term)

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NumericCompletionSum
