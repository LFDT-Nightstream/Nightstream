import Mathlib.Tactic.Ring
import NightstreamFPrime.Gadgets.Sampling.WideReduction

/-!
Owns the exact footprint of the whole-vector sampler constraints (V5):
681 logical rows, and `264 + h` private variables for a witness program with
`h` outputs. The committed witness program allocates the 353 new bits, which
gives 617 private variables.
-/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Range

/-- Logical rows of one scalar: 4 × 67 child rows, 353 Booleanity rows,
54 digit range rows and 6 check rows. -/
def rowCount : Nat := 681

theorem newBits_length (offset : Nat) : (newBits offset).length = newBitCount := by
  simp [newBits, newBitCount, List.length_flatMap, quotientBitCount, digitCount, digitBitCount,
    checkBitCount]
  omega

theorem newBitCount_eq : newBitCount = 353 := rfl

theorem rowCount_eq (interface : Interface) (hints : Nat → List Hint) (offset : Nat) :
    (flatConstraints (operations interface hints offset)).length = rowCount := by
  rw [flatConstraints_length_eq_rowCount]
  have bits := newBits_length offset
  simp only [operations, childOps, rowOps, Circuit.rowCount, List.map_append, List.sum_append,
    List.map_map, Function.comp_def, Op.rowCount, Sequence.childOp, FormalCircuit.asSubcircuit,
    CanonicalU64.circuit, WitnessBatch.hinted, List.map_cons, List.map_nil, List.sum_cons,
    List.sum_nil, List.length_nil, List.map_const', List.sum_replicate, smul_eq_mul,
    List.length_finRange, List.length_range, bits]
  rfl

theorem localLength_eq (interface : Interface) (hints : Nat → List Hint) (offset : Nat) :
    localLength (operations interface hints offset) =
      childWidth * fieldCount + (hints offset).length := by
  simp only [operations, childOps, rowOps, localLength, List.map_append, List.sum_append,
    List.map_map, Function.comp_def, Op.localLength, Sequence.childOp, FormalCircuit.asSubcircuit,
    CanonicalU64.circuit, WitnessBatch.outputLength, WitnessBatch.hinted, List.map_cons,
    List.map_nil, List.sum_cons, List.sum_nil, List.length_nil, List.map_const',
    List.sum_replicate, smul_eq_mul, List.length_finRange, childWidth]
  ring

/-- With a witness program that allocates exactly the new bits, the gadget
has 617 private variables. -/
theorem localLength_eq_privateCount (interface : Interface) (hints : Nat → List Hint)
    (offset : Nat) (allocates : (hints offset).length = newBitCount) :
    localLength (operations interface hints offset) = privateCount := by
  rw [localLength_eq, allocates]
  rfl

theorem privateCount_eq : privateCount = 617 := rfl

end NightstreamFPrime.Gadgets.Sampling.WideReduction
