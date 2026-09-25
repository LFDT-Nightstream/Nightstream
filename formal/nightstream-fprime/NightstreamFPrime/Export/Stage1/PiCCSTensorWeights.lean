import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NumericBooleanDomain

/-! Cache the reference tensor weights in two coordinate tables. The split
is half the coordinate count. For 27 coordinates the tables have 8192 and
16384 entries. Lookup retains the reference fallback for missing entries. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSTensorWeights

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NumericBooleanDomain

universe uField
variable {Field : Type uField}

/-- Concatenation keeps the reference's least-significant-coordinate order.
Only the existing multiplicative weight laws are used. -/
theorem tensorWeightCoordinates_append (ops : InterpolationOps Field)
    (laws : WeightProductLaws ops) (left right : List Field) (index : Nat) :
    tensorWeightCoordinates ops (left ++ right) index =
      ops.mul (tensorWeightCoordinates ops left index)
        (tensorWeightCoordinates ops right (index / 2 ^ left.length)) := by
  induction left generalizing index with
  | nil =>
      simp only [List.nil_append, List.length_nil, Nat.pow_zero, Nat.div_one,
        tensorWeightCoordinates, laws.one_mul]
  | cons coordinate left inductionHypothesis =>
      simp only [List.cons_append, List.length_cons, tensorWeightCoordinates,
        inductionHypothesis, Nat.div_div_eq_div_mul, Nat.pow_succ']
      exact (laws.mul_assoc _ _ _).symm

/-- A finite coordinate list reads only its corresponding low index bits.
This identity holds for every index and requires no algebraic laws. -/
theorem tensorWeightCoordinates_mod (ops : InterpolationOps Field)
    (coordinates : List Field) (index : Nat) :
    tensorWeightCoordinates ops coordinates (index % 2 ^ coordinates.length) =
      tensorWeightCoordinates ops coordinates index := by
  induction coordinates generalizing index with
  | nil => rfl
  | cons coordinate coordinates inductionHypothesis =>
      change
        ops.mul
          (if (index % 2 ^ (coordinates.length + 1)) % 2 == 1 then
            coordinate else ops.sub ops.one coordinate)
          (tensorWeightCoordinates ops coordinates
            ((index % 2 ^ (coordinates.length + 1)) / 2)) =
        ops.mul (if index % 2 == 1 then coordinate else ops.sub ops.one coordinate)
          (tensorWeightCoordinates ops coordinates (index / 2))
      rw [Nat.pow_succ']
      have parity : (index % (2 * 2 ^ coordinates.length)) % 2 = index % 2 :=
        Nat.mod_mod_of_dvd index
          (show 2 ∣ 2 * 2 ^ coordinates.length from ⟨2 ^ coordinates.length, rfl⟩)
      rw [parity, Nat.mod_mul_right_div_self, inductionHypothesis]

private theorem split_value (ops : InterpolationOps Field)
    (laws : WeightProductLaws ops) (coordinates : List Field) (index : Nat) :
    tensorWeightCoordinates ops coordinates index =
      ops.mul
        (tensorWeightCoordinates ops (coordinates.take (coordinates.length / 2))
          (index % 2 ^ (coordinates.length / 2)))
        (tensorWeightCoordinates ops (coordinates.drop (coordinates.length / 2))
          (index / 2 ^ (coordinates.length / 2))) := by
  have leftLength : (coordinates.take (coordinates.length / 2)).length =
      coordinates.length / 2 := List.length_take_of_le (Nat.div_le_self _ _)
  have lowBits :
      tensorWeightCoordinates ops (coordinates.take (coordinates.length / 2))
          (index % 2 ^ (coordinates.length / 2)) =
        tensorWeightCoordinates ops (coordinates.take (coordinates.length / 2)) index := by
    simpa only [leftLength] using tensorWeightCoordinates_mod ops
      (coordinates.take (coordinates.length / 2)) index
  rw [lowBits]
  simpa only [List.take_append_drop, leftLength] using tensorWeightCoordinates_append ops laws
    (coordinates.take (coordinates.length / 2)) (coordinates.drop (coordinates.length / 2)) index

/-- Materialize only reference weights. Both extents are derived from the
coordinate count; the two coordinate sublists are shared during preparation. -/
def prepare (ops : InterpolationOps Field) (coordinates : List Field) :
    Array Field × Array Field :=
  let split := coordinates.length / 2
  let left := coordinates.take split
  let right := coordinates.drop split
  (Array.ofFn (fun index : Fin (2 ^ split) =>
      tensorWeightCoordinates ops left index.val),
   Array.ofFn (fun index : Fin (2 ^ (coordinates.length - split)) =>
      tensorWeightCoordinates ops right index.val))

/-- Exact arity-derived extents, including empty and odd-length inputs. -/
theorem prepare_sizes (ops : InterpolationOps Field) (coordinates : List Field) :
    (prepare ops coordinates).1.size = 2 ^ (coordinates.length / 2) ∧
      (prepare ops coordinates).2.size = 2 ^ (coordinates.length - coordinates.length / 2) := by
  constructor <;> simp only [prepare, Array.size_ofFn]

/-- Read the low and high index portions. A missing entry calls the original
weight function; the fallback is evaluated only in that branch. -/
def lookup (ops : InterpolationOps Field) (coordinates : List Field)
    (tables : Array Field × Array Field) (index : Nat) : Field :=
  match tables.1[index % tables.1.size]?, tables.2[index / tables.1.size]? with
  | some left, some right => ops.mul left right
  | _, _ => tensorWeightCoordinates ops coordinates index

/-- Every prepared lookup equals the reference, including arbitrary indices
outside the prepared high table. No index bound is a caller premise. -/
theorem lookup_prepare (ops : InterpolationOps Field) (laws : WeightProductLaws ops)
    (coordinates : List Field) (index : Nat) :
    lookup ops coordinates (prepare ops coordinates) index =
      tensorWeightCoordinates ops coordinates index := by
  have low : index % 2 ^ (coordinates.length / 2) < 2 ^ (coordinates.length / 2) :=
    Nat.mod_lt _ (Nat.two_pow_pos _)
  by_cases high : index / 2 ^ (coordinates.length / 2) <
      2 ^ (coordinates.length - coordinates.length / 2)
  · simp only [lookup, prepare, Array.size_ofFn, Array.getElem?_ofFn,
      dif_pos low, dif_pos high]
    exact (split_value ops laws coordinates index).symm
  · simp only [lookup, prepare, Array.size_ofFn, Array.getElem?_ofFn,
      dif_pos low, dif_neg high]

end NightstreamFPrime.Export.Stage1.PiCCSTensorWeights
