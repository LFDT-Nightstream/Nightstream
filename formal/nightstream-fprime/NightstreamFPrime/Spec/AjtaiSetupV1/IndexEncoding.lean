import NightstreamFPrime.Spec.AjtaiSetupV1

/-! The SHAKE128 input of a key element determines its seed, row and block,
and each element has one output lane for each coefficient word. The range
premises are essential. No statement about SHAKE128 output is made. -/

namespace NightstreamFPrime.Spec.AjtaiSetupV1

/-- Equal encodings agree on every encoded bit. -/
theorem littleEndianBytes_mod (count left right : Nat)
    (same : littleEndianBytes count left = littleEndianBytes count right) :
    left % 256 ^ count = right % 256 ^ count := by
  induction count generalizing left right with
  | zero => rw [Nat.pow_zero, Nat.mod_one, Nat.mod_one]
  | succ count ih =>
      simp only [littleEndianBytes, List.cons.injEq] at same
      rw [Nat.pow_succ', Nat.mod_mul, Nat.mod_mul, same.1, ih _ _ same.2]

theorem elementInput_length (seed : Seed) (row block : Nat) :
    (elementInput seed.bytes row block).length = 81 := by
  simp [elementInput, seed.length_eq]

/-- Distinct seeds or in-range coordinates never share a SHAKE128 input. -/
theorem elementInput_injective (left right : Seed)
    (leftRow leftBlock rightRow rightBlock : Nat)
    (leftRowBound : leftRow < 2 ^ 32) (rightRowBound : rightRow < 2 ^ 32)
    (leftBlockBound : leftBlock < 2 ^ 64) (rightBlockBound : rightBlock < 2 ^ 64)
    (same : elementInput left.bytes leftRow leftBlock =
      elementInput right.bytes rightRow rightBlock) :
    left = right ∧ leftRow = rightRow ∧ leftBlock = rightBlock := by
  unfold elementInput at same
  obtain ⟨withRow, blockBytes⟩ := List.append_inj same
    (by simp [left.length_eq, right.length_eq])
  obtain ⟨withSeed, rowBytes⟩ := List.append_inj withRow
    (by simp [left.length_eq, right.length_eq])
  have seedBytes := (List.append_inj withSeed rfl).2
  have row := littleEndianBytes_mod 4 _ _ rowBytes
  have block := littleEndianBytes_mod 8 _ _ blockBytes
  rw [Nat.mod_eq_of_lt (Nat.lt_of_lt_of_eq leftRowBound (by decide)),
    Nat.mod_eq_of_lt (Nat.lt_of_lt_of_eq rightRowBound (by decide))] at row
  rw [Nat.mod_eq_of_lt (Nat.lt_of_lt_of_eq leftBlockBound (by decide)),
    Nat.mod_eq_of_lt (Nat.lt_of_lt_of_eq rightBlockBound (by decide))] at block
  refine ⟨?_, row, block⟩
  cases left
  cases right
  cases seedBytes
  rfl

/-- Every coefficient word reads squeezed output, never a default lane. -/
theorem elementLanes_length (seed : List Nat) (row block : Nat) :
    (elementLanes seed row block).length = 4 * ringDegree :=
  Shake128.lanes_length _ _

end NightstreamFPrime.Spec.AjtaiSetupV1
