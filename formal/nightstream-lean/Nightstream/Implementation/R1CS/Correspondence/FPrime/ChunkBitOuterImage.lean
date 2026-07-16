import Nightstream.Implementation.R1CS.Core.BooleanRowDedup

/-!
Contract: model-level outer image for one production 16-bit sampler chunk.

Owns: the two chunk-bit role classes, direct singleton substitution, and the
lane-terminal substitution obtained by solving the canonical-u64
recomposition equation for bit 63.

Does not own: production column discovery, generated Rust conformance, packed
Mod5 or aggregate-acceptance semantics, or authorization to remove rows.

Emits constraints: no. The lane-terminal Boolean row modeled here must remain
in the translated relation.

Authority boundary: column zero is verifier-owned and must be pinned to one.
Neither an encoded lane nor substitution metadata supplies that authority.

Assurance tier: model-level.

| Predicate/theorem | Mathematical obligation | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `productionRole` | chunk-bit ownership | Only chunk 3, local bit 15 uses the lane-terminal image | fixed 4-by-16 chunking | no |
| `terminalDefinitionTerms_eval` | solved recomposition | The bit-63 definition evaluates to the encoded lane's terminal bit | exact Goldilocks inverse | no |
| `directLeafRows_iff` | direct outer image | Every leaf row is transported through a singleton bit image | exact singleton map | no |
| `terminalLeafRows_iff` | terminal outer image | Every leaf row is transported through the solved bit-63 image | exact terminal map | no |
| `directBit_le_one` | direct Boolean row | The translated singleton row forces the direct value into `{0,1}` | Euclid property; canonical residues; constant one | no |
| `terminalBit_le_one` | retained Boolean row | The translated bit-63 row forces the derived value into `{0,1}` | Euclid property; canonical residues; constant one | no |
| `constantOne_necessary` | authority necessity | A forged constant two makes a direct value two satisfy the translated bit row | explicit witness | no |
| `terminalBitRow_necessary` | row necessity | The solved terminal image can be non-Boolean when its translated bit row is absent | explicit witness | no |
-/

namespace Nightstream.Implementation.R1CS.ChunkBitOuterImage

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.BooleanRowDedup
open Nightstream.Implementation.R1CS.LinearSubstitution
open Nightstream.Implementation.R1CS.Program

/-- Production chunk coordinates have one exceptional outer-image role. -/
inductive Role where
  | directSingleton
  | laneTerminal
deriving DecidableEq, Repr

/-- Four 16-bit chunks cover one 64-bit canonical lane. Bit 63 is chunk 3,
local bit 15; every other chunk bit has a direct singleton image. -/
def productionRole (chunk localBit : Nat) : Role :=
  if chunk = 3 && localBit = 15 then .laneTerminal else .directSingleton

@[simp] theorem productionRole_terminal :
    productionRole 3 15 = .laneTerminal := by
  rfl

theorem productionRole_eq_terminal_iff {chunk localBit : Nat} :
    productionRole chunk localBit = .laneTerminal ↔
      chunk = 3 ∧ localBit = 15 := by
  simp [productionRole]

theorem productionRole_direct_of_chunk_lt_three
    {chunk localBit : Nat} (chunkBound : chunk < 3) :
    productionRole chunk localBit = .directSingleton := by
  simp [productionRole]
  omega

theorem productionRole_direct_of_local_lt_fifteen
    {chunk localBit : Nat} (localBound : localBit < 15) :
    productionRole chunk localBit = .directSingleton := by
  simp [productionRole]
  omega

/-- `2^63` in Goldilocks. -/
def twoPow63 : Nat := 9223372036854775808

/-- The canonical inverse of `2^63` in Goldilocks. -/
def twoPow63Inverse : Nat := 18446744060824649729

theorem twoPow63_inverse :
    twoPow63 * twoPow63Inverse % goldilocksP = 1 := by
  native_decide

/-- Canonical 64-bit recomposition in an encoded lane. -/
def canonicalLaneTerms (laneStart : Nat) : List (Nat × Nat) :=
  (List.range 64).map fun index => (laneStart + index, 2 ^ index)

/-- The lower 63 terms in the same canonical recomposition. -/
def canonicalLowerLaneTerms (laneStart : Nat) : List (Nat × Nat) :=
  (List.range 63).map fun index => (laneStart + index, 2 ^ index)

/-- Numerator
`(sum_{i<63} 2^i bit_i + 2^63 bit_63) - sum_{i<63} 2^i bit_i`. -/
def terminalNumeratorTerms (laneStart : Nat) : List (Nat × Nat) :=
  canonicalLaneTerms laneStart ++
    negateTerms (canonicalLowerLaneTerms laneStart)

/-- Solving recomposition for bit 63 multiplies the numerator by
`(2^63)^-1`. -/
def terminalDefinitionTerms (laneStart : Nat) : List (Nat × Nat) :=
  scaleTerms twoPow63Inverse (terminalNumeratorTerms laneStart)

/-- Exact translated LC used for the lane-terminal source bit. This is the
canonical-u64 recomposition solved by sparse linear substitution, not a
caller-provided digest or conclusion. -/
def terminalDerivedTerms (laneStart : Nat) : List (Nat × Nat) :=
  terminalDefinitionTerms laneStart

private theorem canonicalLowerLaneTerms_canonical (laneStart : Nat) :
    CanonicalTerms (canonicalLowerLaneTerms laneStart) := by
  intro term member
  rcases List.mem_map.mp member with ⟨index, indexMember, rfl⟩
  have indexBound : index < 63 := List.mem_range.mp indexMember
  have positive : 0 < 2 ^ index := Nat.pow_pos (by decide)
  have belowTwo63 : 2 ^ index < 2 ^ 63 :=
    Nat.pow_lt_pow_of_lt (by decide) indexBound
  have two63BelowModulus : 2 ^ 63 < goldilocksP := by
    decide
  exact ⟨positive, Nat.lt_trans belowTwo63 two63BelowModulus⟩

private theorem rawLcEval_append (encoded : Nat → Nat)
    (left right : List (Nat × Nat)) :
    rawLcEval encoded (left ++ right) =
      rawLcEval encoded left + rawLcEval encoded right := by
  induction left with
  | nil => simp [rawLcEval]
  | cons head tail inductionHypothesis =>
      simp [rawLcEval, inductionHypothesis, Nat.add_assoc]

private theorem lcEval_append (encoded : Nat → Nat)
    (left right : List (Nat × Nat)) :
    lcEval encoded (left ++ right) =
      (lcEval encoded left + lcEval encoded right) % goldilocksP := by
  rw [lcEval_eq_raw_mod, rawLcEval_append, Nat.add_mod,
    ← lcEval_eq_raw_mod, ← lcEval_eq_raw_mod]

private def uniformExpansion (sourceTerms : List (Nat × Nat)) :
    ColumnExpansion :=
  fun _ => sourceTerms

/-- Scaling evaluation is derived from the generic linear-substitution
theorem; no parallel sparse-LC evaluator is introduced here. -/
private theorem lcEval_scaleTerms (encoded : Nat → Nat)
    (coefficient : Nat) (sourceTerms : List (Nat × Nat)) :
    lcEval encoded (scaleTerms coefficient sourceTerms) =
      coefficient * lcEval encoded sourceTerms % goldilocksP := by
  simpa [terms, uniformExpansion, assignment, lcEval] using
    (lcEval_terms (uniformExpansion sourceTerms) encoded [(1, coefficient)])

private theorem terminalNumeratorTerms_eval
    (encoded : Nat → Nat) (laneStart : Nat) :
    lcEval encoded (terminalNumeratorTerms laneStart) =
      twoPow63 * encoded (laneStart + 63) % goldilocksP := by
  have laneSplit :
      canonicalLaneTerms laneStart =
        canonicalLowerLaneTerms laneStart ++
          [(laneStart + 63, twoPow63)] := by
    simp [canonicalLaneTerms, canonicalLowerLaneTerms, List.range_succ,
      twoPow63]
  have reordered :
      List.Perm
        ((canonicalLowerLaneTerms laneStart ++
            [(laneStart + 63, twoPow63)]) ++
          negateTerms (canonicalLowerLaneTerms laneStart))
        ((canonicalLowerLaneTerms laneStart ++
            negateTerms (canonicalLowerLaneTerms laneStart)) ++
          [(laneStart + 63, twoPow63)]) := by
    simpa only [List.append_assoc] using
      (List.perm_append_comm : List.Perm
        ([(laneStart + 63, twoPow63)] ++
            negateTerms (canonicalLowerLaneTerms laneStart))
        (negateTerms (canonicalLowerLaneTerms laneStart) ++
            [(laneStart + 63, twoPow63)])).append_left
              (canonicalLowerLaneTerms laneStart)
  have cancelled := lcEval_append_negateTerms_eq_zero encoded
    (canonicalLowerLaneTerms laneStart)
    (canonicalLowerLaneTerms_canonical laneStart)
  rw [terminalNumeratorTerms, laneSplit,
    lcEval_eq_of_perm encoded reordered, lcEval_append, cancelled]
  simp [lcEval, twoPow63, goldilocksP]

/-- The solved definition closes the algebraic obligation: after exact
canonical-lane substitution it is extensionally the encoded bit-63 value. -/
theorem terminalDefinitionTerms_eval
    (encoded : Nat → Nat) (laneStart : Nat) :
    lcEval encoded (terminalDefinitionTerms laneStart) =
      encoded (laneStart + 63) % goldilocksP := by
  rw [terminalDefinitionTerms, lcEval_scaleTerms,
    terminalNumeratorTerms_eval]
  simp only [Nat.mul_mod_mod]
  rw [← Nat.mul_assoc]
  rw [← Nat.mod_mul_mod]
  have inverseProduct :
      twoPow63Inverse * twoPow63 % goldilocksP = 1 := by
    rw [Nat.mul_comm]
    exact twoPow63_inverse
  rw [inverseProduct]
  simp

/-- Map one source terminal bit to the solved bit-63 LC. All other source
columns, including the constant-one column, retain their identity image. -/
def terminalBitExpansion (sourceBit laneStart : Nat) : ColumnExpansion :=
  fun column =>
    if column = sourceBit then terminalDerivedTerms laneStart
    else [(column, 1)]

@[simp] theorem terminal_assignment_source
    {sourceBit laneStart : Nat} (encoded : Nat → Nat) :
    assignment (terminalBitExpansion sourceBit laneStart) encoded sourceBit =
      lcEval encoded (terminalDerivedTerms laneStart) := by
  simp [assignment, terminalBitExpansion]

@[simp] theorem terminal_assignment_one
    {sourceBit laneStart : Nat} (sourceNeZero : sourceBit ≠ 0)
    (encoded : Nat → Nat) :
    assignment (terminalBitExpansion sourceBit laneStart) encoded 0 =
      encoded 0 % goldilocksP := by
  have zeroNeSource : 0 ≠ sourceBit := Ne.symm sourceNeZero
  simp [assignment, terminalBitExpansion, zeroNeSource, lcEval]

/-- The verifier-owned constant-one precondition is preserved by the terminal
outer image. -/
theorem terminal_assignment_one_eq_one
    {sourceBit laneStart : Nat} (sourceNeZero : sourceBit ≠ 0)
    (encoded : Nat → Nat) (constantOne : encoded 0 = 1) :
    assignment (terminalBitExpansion sourceBit laneStart) encoded 0 = 1 := by
  simp [terminal_assignment_one sourceNeZero encoded, constantOne,
    goldilocksP]

/-- The verifier-owned constant-one precondition is also preserved by the
direct singleton image. -/
theorem direct_assignment_one_eq_one
    {sourceBit slot : Nat} (sourceNeZero : sourceBit ≠ 0)
    (encoded : Nat → Nat) (constantOne : encoded 0 = 1) :
    assignment (singletonSlotExpansion sourceBit slot) encoded 0 = 1 := by
  simp [assignment_zero sourceNeZero encoded, constantOne, goldilocksP]

/-- Existing generic linear-substitution semantics transports an arbitrary
leaf relation through the direct singleton image. -/
theorem directLeafRows_iff
    {sourceBit slot : Nat} (encoded : Nat → Nat) (leafRows : List Row) :
    Satisfies
        (leafRows.map
          (row (singletonSlotExpansion sourceBit slot))) encoded ↔
      Satisfies leafRows
        (assignment (singletonSlotExpansion sourceBit slot) encoded) := by
  exact satisfies_mapped_iff leafRows
    (singletonSlotExpansion sourceBit slot) encoded

/-- Existing generic linear-substitution semantics transports the same leaf
relation through the solved lane-terminal image. Authority is supplied
separately by `terminal_assignment_one_eq_one`. -/
theorem terminalLeafRows_iff
    {sourceBit laneStart : Nat} (encoded : Nat → Nat)
    (leafRows : List Row) :
    Satisfies
        (leafRows.map (row (terminalBitExpansion sourceBit laneStart)))
          encoded ↔
      Satisfies leafRows
        (assignment (terminalBitExpansion sourceBit laneStart) encoded) := by
  exact satisfies_mapped_iff leafRows
    (terminalBitExpansion sourceBit laneStart) encoded

/-- The translated terminal bit row is retained and transports exactly to the
source bit predicate on the solved assignment. -/
theorem terminalBitRow_iff
    {sourceBit laneStart : Nat} (encoded : Nat → Nat) :
    RowHolds encoded
        (row (terminalBitExpansion sourceBit laneStart) (bitRow sourceBit)) ↔
      RowHolds (assignment (terminalBitExpansion sourceBit laneStart) encoded)
        (bitRow sourceBit) :=
  rowHolds_iff (terminalBitExpansion sourceBit laneStart) encoded
    (bitRow sourceBit)

/-- With the verifier-owned constant and canonical encoded residues, the
translated singleton row makes a direct source value Boolean. -/
theorem directBit_le_one
    (prime : EuclidPrime goldilocksP)
    {sourceBit slot : Nat} (sourceNeZero : sourceBit ≠ 0)
    (encoded : Nat → Nat) (constantOne : encoded 0 = 1)
    (holds : RowHolds encoded
      (row (singletonSlotExpansion sourceBit slot) (bitRow sourceBit))) :
    assignment (singletonSlotExpansion sourceBit slot) encoded sourceBit
        ≤ 1 := by
  apply bitRow_le_one prime
  · unfold assignment
    exact Nat.mod_lt _ (by decide)
  · exact direct_assignment_one_eq_one sourceNeZero encoded constantOne
  · exact (rowHolds_iff (singletonSlotExpansion sourceBit slot) encoded
      (bitRow sourceBit)).mp holds

/-- With the verifier-owned constant and canonical encoded residues, the
retained translated row makes the derived terminal value Boolean. -/
theorem terminalBit_le_one
    (prime : EuclidPrime goldilocksP)
    {sourceBit laneStart : Nat} (sourceNeZero : sourceBit ≠ 0)
    (encoded : Nat → Nat) (constantOne : encoded 0 = 1)
    (holds : RowHolds encoded
      (row (terminalBitExpansion sourceBit laneStart) (bitRow sourceBit))) :
    assignment (terminalBitExpansion sourceBit laneStart) encoded sourceBit
        ≤ 1 := by
  apply bitRow_le_one prime
  · unfold assignment
    exact Nat.mod_lt _ (by decide)
  · exact terminal_assignment_one_eq_one sourceNeZero encoded constantOne
  · exact (terminalBitRow_iff encoded).mp holds

private def forgedConstantAssignment : Nat → Nat
  | 0 => 2
  | 9 => 2
  | _ => 0

/-- Constant-one authority is necessary: with column zero forged to two, a
direct value two satisfies the translated source bit row. -/
theorem constantOne_necessary :
    forgedConstantAssignment 0 ≠ 1 ∧
      assignment (singletonSlotExpansion 4 9)
          forgedConstantAssignment 4 = 2 ∧
      RowHolds forgedConstantAssignment
        (row (singletonSlotExpansion 4 9) (bitRow 4)) := by
  native_decide

private def nonBooleanTerminalAssignment : Nat → Nat
  | 0 => 1
  | 73 => 2
  | _ => 0

/-- The terminal Boolean row is necessary at this boundary: the exact solved
image admits value two before that row is imposed. -/
theorem terminalBitRow_necessary :
    nonBooleanTerminalAssignment 0 = 1 ∧
      assignment (terminalBitExpansion 4 10)
          nonBooleanTerminalAssignment 4 = 2 ∧
      ¬ RowHolds nonBooleanTerminalAssignment
        (row (terminalBitExpansion 4 10) (bitRow 4)) := by
  native_decide

end Nightstream.Implementation.R1CS.ChunkBitOuterImage
